# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""harness-bench adapter: coding tasks on TypeScript repositories, solved by Pi.

Every rollout is one `scripts/rollout.ts` call in a harness-bench checkout, which copies the
task's repository into a fresh workspace, runs Pi inside harness-bench's own sandbox and leaves
the trajectory and the diff in the run directory. The adapter reads only a task's prompt: hidden
tests and reference patches never reach a rollout or the improver, and the run script is meant
to hide them from the whole process as well. Scoring stays in harness-bench.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

from .. import config
from ..core import runmark as RM
from ..core import tree as HT
from ..core.spec import BenchPrompts, BenchSpec, ChannelSpec
from ..substrate.base import Install
from ..substrate.pi import PI_SURFACE, PiSubstrate

DOMAIN = "typescript-repositories"

HB_PROMPTS = BenchPrompts(
    work=(
        "Software engineering on real TypeScript repositories run with Bun: fixing a reported "
        "bug, adding a feature or refactoring, as a maintainer asks in a short request. The "
        "change lives in the repository itself, must make the requested behaviour true without "
        "breaking what already works, and stays as small as the request allows."
    ),
    domain_word="area",
    deliverable_word="change",
    inv_levers=(
        "- locate the code the request is about before editing: search for the named error, "
        "symbol or behaviour, and read its callers and tests\n"
        "- reproduce the reported behaviour or confirm the missing feature with the "
        "repository's own commands before changing it\n"
        "- run the repository's existing tests and type checks after the change, and read the "
        "output rather than assuming it passed\n"
        "- keep the diff to what the request needs: no unrelated refactors, renames, "
        "dependency changes or stray files"
    ),
    dom_levers=(
        "- respect the repository's conventions: its package manager, test runner, module "
        "layout, naming and error-handling style\n"
        "- treat the request's stated inputs, outputs and messages as the contract the change "
        "must satisfy, including the cases it says must keep behaving as before\n"
        "- when a value crosses a boundary (parsing, formatting, serialisation, a CLI), check "
        "both directions of the round trip"
    ),
    yardstick=(
        "Did the run find the code the request is about, make the requested behaviour true, "
        "keep every behaviour the request said must not change, verify it with the "
        "repository's own tests or commands, and leave no change the request did not need?"
    ),
    selection_checks=(
        "Prefer the change that does what the request asks for every case it names, keeps the "
        "cases it says must not change, matches the repository's conventions, and touches no "
        "more code than needed. A change that only adds tests or comments, or edits files the "
        "request did not concern, is worse than a smaller one that fixes the behaviour."
    ),
)

_BASE_LOCK = threading.Lock()


def _bench_dir() -> Path:
    return Path(config.require("HARNESS2_HARNESS_BENCH_DIR", needed_by="harness-bench rollouts"))


class HarnessBenchAdapter:
    def __init__(self, *, split_file: Path | None = None, timeout: int = 900):
        self.split_file = split_file
        self.timeout = timeout
        self.substrate = PiSubstrate()
        channel = ChannelSpec(
            config.env("HARNESS2_IMPROVER_CHANNEL", "claude_cli"),
            config.require("HARNESS2_IMPROVER_MODEL", needed_by="harness editing"),
            effort=config.env("HARNESS2_IMPROVER_EFFORT", "high"),
        )
        self.spec = BenchSpec(
            name="hb",
            runs_root=config.RUNS / "hb",
            install_mode="workspace_config",
            prompts=HB_PROMPTS,
            channels={"proposer": channel, "selector": channel},
            leak_globs=("hidden/**", "reference.patch"),
        )
        self.model = config.require("HARNESS2_SOLVER_MODEL", needed_by="pi rollouts").split("/", 1)[
            -1
        ]
        self.effort = config.env("HARNESS2_SOLVER_EFFORT", "low")
        self._results = config.RUNS / "hb" / "results"

    def tasks(self, split: str = "all") -> list[str]:
        found = {p.parent.name for p in (_bench_dir() / "tasks").glob("*/prompt.md")}
        if self.split_file is None:
            if split != "all":
                raise ValueError(f"--split {split} needs a split file")
            return sorted(found)
        halves = json.loads(Path(self.split_file).read_text(encoding="utf-8"))
        ids = halves["train"] + halves["test"] if split == "all" else halves[split]
        missing = sorted(set(ids) - found)
        if missing:
            raise ValueError(f"tasks absent from {_bench_dir() / 'tasks'}: {missing}")
        return sorted(ids)

    def domain_of(self, task_id: str) -> str:
        return DOMAIN

    def task_prompt(self, task_id: str) -> str:
        return (_bench_dir() / "tasks" / task_id / "prompt.md").read_text(encoding="utf-8")

    def task_files(self, task_id: str) -> Path | None:
        return None

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        out = self._results / run_id
        if RM.is_complete(out, harness_dir=harness_dir):
            return out
        if out.exists():
            shutil.rmtree(out)
        install = (
            None
            if harness_dir is None
            else self.substrate.materialise(Path(harness_dir), self.spec, {})
        )
        try:
            self._rollout(task_id, out, install, int(timeout or self.timeout))
        finally:
            if install is not None:
                shutil.rmtree(install.staging_dir, ignore_errors=True)
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        error = result.get("error") or {}
        if result.get("kind") != "result" and error.get("class") != "time-limit":
            raise RuntimeError(f"pi rollout {run_id} failed: {error.get('message')}")
        RM.mark_complete(out, run_id=run_id, harness_dir=harness_dir)
        return out

    def _rollout(self, task_id: str, out: Path, install: Install | None, timeout: int) -> None:
        cmd = [
            "bun",
            "scripts/rollout.ts",
            "--task",
            task_id,
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--out",
            str(out),
        ]
        if install is not None:
            cmd += ["--harness-dir", str(install.staging_dir)]
        r = subprocess.run(cmd, cwd=_bench_dir(), capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(
                f"rollout.ts exited {r.returncode} for {out.name}: {r.stderr.strip()[-800:]}"
            )

    def delivered(self, run_dir: Path) -> bool:
        diff = Path(run_dir) / "diff.patch"
        return diff.is_file() and bool(diff.read_text(encoding="utf-8").strip())

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        stream = Path(run_dir) / "events.jsonl"
        if not stream.is_file():
            return f"(no trajectory recorded for {Path(run_dir).name})"
        return self.substrate.read_trajectory(stream, obs_trunc=obs_trunc, max_chars=2_000_000)

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        return {"diff.patch": self.deliverable_text(run_dir)} if self.delivered(run_dir) else {}

    def deliverable_text(self, run_dir: Path) -> str:
        diff = Path(run_dir) / "diff.patch"
        if not self.delivered(run_dir):
            return "(this run changed nothing in the repository)"
        return diff.read_text(encoding="utf-8")

    def base_harness(self) -> Path:
        d = self.spec.runs_root / f"_base_harness_{os.getpid()}"
        with _BASE_LOCK:
            if not HT.is_tree(PI_SURFACE, d):
                HT.write_tree(PI_SURFACE, {}, d)
        return d

    def validate_tree(self, tree: dict[str, str]) -> None:
        pass

    def preflight(self) -> None:
        self.substrate.preflight()
        rollout = _bench_dir() / "scripts" / "rollout.ts"
        if not rollout.is_file():
            raise RuntimeError(
                f"{rollout} missing: point HARNESS2_HARNESS_BENCH_DIR at harness-bench"
            )
