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

"""Adapt a Pi harness on harness-bench's training tasks, then freeze one harness for its tests.

`sweep` runs the method over the training split exactly as `harness2` does for the bundled
benchmarks. The method adapts a harness per task, so its output is a library of per-task
harnesses rather than one harness; `freeze` asks the improver to consolidate that library into
the single harness a harness-bench variant can carry, keeping what generalises and dropping the
workspace facts of the tasks it came from, and writes it in harness-bench's layout
(`configs/<name>/`, `variants/<name>.json`) so the held-out tasks can score it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..benchmarks.harness_bench import DOMAIN, HarnessBenchAdapter
from ..core import library as LIB
from ..core import prompts as PR
from ..core import schedule
from ..core import tree as HT
from ..core.proposer import Proposer
from ..substrate.pi import PI_SURFACE
from . import _common as C

FREEZE_MISSION = """\
# CONSOLIDATE ONE HARNESS FOR TASKS YOU HAVE NOT SEEN

`context/library/` holds, for each earlier task in this area, the full harness that was adopted
for it and an EXPERIENCE.md saying why. Read `INDEX.md` first, then every entry.

Write into `context/harness/` the ONE harness that will run, unchanged, on new tasks in this area
that none of these harnesses saw. Keep a component or rule only when the experience notes show
it helped on more than one task, or when it is procedure that holds for essentially every task
here. Drop every workspace fact tied to one task -- a file, symbol, value or mapping that was
true in that repository -- because on a new task it is a wrong anchor. When two entries carry
the same lesson, keep the clearest version once. Prefer skills to always-on text, and keep the
always-on text short.

{surface}

{work}"""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harness2-hb", description=__doc__, formatter_class=C.HelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep", help="adapt over a split of harness-bench tasks")
    sweep.add_argument("--split-file", type=Path, required=True)
    sweep.add_argument("--split", choices=("all", "train", "test"), default="train")
    sweep.add_argument("--k", type=int, default=2)
    sweep.add_argument("--max-tasks", type=int, default=None)
    sweep.add_argument("--dry-run", action="store_true")
    C.add_modes_arg(sweep)
    freeze = sub.add_parser("freeze", help="consolidate a sweep's library into one harness")
    freeze.add_argument("--k", type=int, default=2)
    freeze.add_argument("--mode", default="parallel")
    freeze.add_argument("--name", required=True, help="harness-bench variant id")
    freeze.add_argument("--out", type=Path, required=True)
    return p


def _sweep(args: argparse.Namespace) -> int:
    adapter = HarnessBenchAdapter(split_file=args.split_file)
    modes = C.parse_modes(args.modes)
    stream = schedule.task_stream(adapter, split=args.split, max_tasks=args.max_tasks)
    for domain, tasks in stream.items():
        print(f"{domain}: {', '.join(tasks)}", flush=True)
    if args.dry_run:
        return 0 if stream else 1
    adapter.preflight()
    runs_root = C.runs_root_for(adapter)
    namespace = C.summary_namespace(adapter, args.k, seed=None)
    rep = schedule.run_sweep(
        args.k,
        adapter=adapter,
        prop=Proposer(
            adapter.spec.channels["proposer"], audit_dir=runs_root / "_audit" / namespace
        ),
        runs_root=runs_root,
        library_root=C.library_root_for(adapter, args.k, seed=None),
        split=args.split,
        modes=modes,
        max_tasks=args.max_tasks,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = runs_root / "_sweeps" / f"{namespace}-{stamp}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    done = sum(int(d.get("done", 0)) for d in rep["domains"])
    failed = sum(int(d.get("failed", 0)) for d in rep["domains"])
    print(f"sweep {namespace}: {done} tasks ok, {failed} failed; record: {record}")
    return 0 if rep["domains"] and not failed else 1


def _freeze(args: argparse.Namespace) -> int:
    adapter = HarnessBenchAdapter()
    mode = schedule.mode_token(args.mode)
    library = LIB.Library(
        root=C.library_root_for(adapter, args.k, seed=None) / mode,
        domain=DOMAIN,
        surface=PI_SURFACE,
    )
    if library.context_dir() is None:
        raise ValueError(f"no adopted harness with experience under {library.dir}")
    work = C.runs_root_for(adapter) / "freeze" / args.name
    shutil.rmtree(work, ignore_errors=True)
    context = work / "context"
    shutil.copytree(library.dir, context / "library")
    HT.write_tree(PI_SURFACE, HT.read_tree(PI_SURFACE, adapter.base_harness()), context / "harness")
    mission = FREEZE_MISSION.format(
        surface=PR._st(PI_SURFACE).surface_md, work=f"# THE WORK\n\n{adapter.spec.prompts.work}"
    )
    prop = Proposer(adapter.spec.channels["proposer"], audit_dir=work / "_audit")
    feedback = ""
    for attempt in (1, 2):
        reply = prop.call(
            mission + feedback,
            context,
            f"freeze.{args.name}.{attempt}",
            harvest="context/harness",
        )
        if reply.workspace is None:
            raise RuntimeError("the consolidation session left no harness")
        harness = HT.read_tree(PI_SURFACE, reply.workspace)
        HT.autotrigger(PI_SURFACE, harness)
        try:
            HT.validate(PI_SURFACE, harness, allow_subagents=False)
            break
        except ValueError as exc:
            if attempt == 2:
                raise
            feedback = f"\n\n# YOUR PREVIOUS HARNESS FAILED VALIDATION\n\n{exc}\n"
    tree_dir = HT.write_tree(PI_SURFACE, harness, work / "harness", stubs=False)
    staged = adapter.substrate.materialise(tree_dir, adapter.spec, {}).staging_dir
    target = args.out / "configs" / args.name
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(staged, target)
    shutil.rmtree(staged, ignore_errors=True)
    files = sorted(
        f"../configs/{args.name}/{p.relative_to(target).as_posix()}"
        for p in target.rglob("*")
        if p.is_file() and p.name != "prompt.md"
    )
    variant = {
        "id": args.name,
        "name": f"Pi with a harness consolidated by Harness² from {len(library.entries())} "
        "training tasks",
        "model": adapter.model,
        **(
            {"promptFile": f"../configs/{args.name}/prompt.md"}
            if (target / "prompt.md").is_file()
            else {}
        ),
        "skills": files,
        "infrastructure": {"harness": "pi"},
    }
    (args.out / "variants").mkdir(parents=True, exist_ok=True)
    (args.out / "variants" / f"{args.name}.json").write_text(
        json.dumps(variant, indent=2) + "\n", encoding="utf-8"
    )
    print(f"froze {len(harness)} files from {len(library.entries())} entries into {target}")
    print(f"notes: {reply.text.strip()[:2000]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _sweep(args) if args.command == "sweep" else _freeze(args)
    except (config.ConfigError, RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
