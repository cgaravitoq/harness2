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

"""PiSubstrate: the Pi coding agent as a solver substrate, run through harness-bench.

The four always-on files compose into one prompt appended to Pi's own system prompt, skills
load from `skills/` (Pi advertises each by name and description and the agent reads the body
on demand), and helper scripts travel beside them under `scripts/`. Pi has no subagents and
this deployment registers no extensions, so c5 and c8 are off the surface. The staged layout
is the one harness-bench's rollout and variants read: `prompt.md`, `skills/**`, `scripts/**`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from ..core import runmark as RM
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install, Launcher, RunRequest

PI_SURFACE = SurfaceSpec(
    name="pi",
    always_on=(
        "systemprompt.md",
        "guardrails.md",
        "LongTermMEMORY.md",
        "tool_descriptions/tool_guidance.md",
    ),
    always_on_max=24_000,
    always_on_cap_reason="every token here competes with the agent's own output budget",
    skill_dir="skills",
    skill_name_rule="must_match",
    trigger_hint=(
        "Fix it by adding a trigger line naming `{name}` to systemprompt.md or "
        'guardrails.md, e.g. "When <the situation this skill handles>, read the '
        '`{name}` skill."'
    ),
    autotrigger_file="guardrails.md",
    trigger_line="- When {desc}, read the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=None,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. "
            "Add one by emitting scripts/<name>.py. -->\n",
        ),
        (
            "skills/README.md",
            "<!-- c6: reusable workflows, advertised by name and description and read on "
            "demand. Add one as skills/<name>/SKILL.md with frontmatter `name:` matching the "
            "directory and a `description:`. -->\n",
        ),
    ),
)


def compose_prompt(tree_dir: Path) -> str:
    return "\n\n".join(
        text
        for name in PI_SURFACE.always_on
        if (path := tree_dir / name).is_file()
        and (text := path.read_text(encoding="utf-8", errors="replace").strip())
    )


class PiSubstrate:
    name = "pi"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return PI_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        tree_dir = Path(tree_dir)
        if not HT.is_tree(PI_SURFACE, tree_dir):
            raise ValueError(f"{tree_dir} is not a pi harness tree (no always-on file present)")
        staged = Path(tempfile.mkdtemp(prefix="adapt-pi-"))
        try:
            harness = HT.read_tree(PI_SURFACE, tree_dir)
            prompt = compose_prompt(tree_dir)
            if prompt:
                (staged / "prompt.md").write_text(prompt + "\n", encoding="utf-8")
            for rel, body in harness.items():
                if not rel.startswith(("skills/", "scripts/")):
                    continue
                path = staged / HT.normalize_rel(rel)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return Install(staging_dir=staged, fingerprint=RM.fingerprint(tree_dir))

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError("PiSubstrate was constructed without a launcher body")
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000
    ) -> str:
        return render_events(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        if shutil.which("bun") is None:
            raise RuntimeError("bun not found on PATH: harness-bench runs Pi through bun")


_TRUNC_FMT = "…(+{n} chars - render truncation)"


def _clip(s, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + _TRUNC_FMT.format(n=len(s) - n)


def render_events(path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000) -> str:
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"(failed to read trajectory: {e})"
    lines: list[str] = []
    for ln in raw.splitlines():
        try:
            message = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content") or ""
        if role == "assistant":
            if str(content).strip():
                lines.append(f"[ASSISTANT] {_clip(content, obs_trunc)}")
            for call in message.get("toolCalls") or []:
                args = json.dumps(call.get("arguments"), ensure_ascii=False, default=str)
                lines.append(f"[ACTION] {call.get('name', 'tool')}({_clip(args, args_trunc)})")
        elif role == "tool":
            tag = "OBSERVATION ERROR" if message.get("isError") else "OBSERVATION"
            lines.append(f"[{tag}] {_clip(content, obs_trunc)}")
    rendered = "\n".join(lines).strip()
    if not rendered:
        return "(no parseable events in trajectory)"
    if len(rendered) > max_chars:
        dropped = len(rendered) - max_chars
        rendered = rendered[:max_chars] + "\n" + _TRUNC_FMT.format(n=dropped)
    return rendered
