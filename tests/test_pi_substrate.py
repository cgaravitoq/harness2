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

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from harness2.core import prompts
from harness2.core import tree as HT
from harness2.substrate.pi import PI_SURFACE, PiSubstrate

SKILL = "---\nname: trace-callers\ndescription: Find callers first\n---\nUse grep.\n"


class PiSubstrateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_materialise_stages_the_layout_harness_bench_reads(self):
        tree_dir = HT.write_tree(
            PI_SURFACE,
            {
                "systemprompt.md": "Read the request twice.",
                "guardrails.md": "- When a signature changes, read the `trace-callers` skill.",
                "skills/trace-callers/SKILL.md": SKILL,
                "scripts/list_exports.py": "print('ok')\n",
                "sub_agents/helper/AGENT.md": "---\ndescription: x\n---\n",
                "plugin/guard.ts": "export {}\n",
            },
            self.root / "tree",
        )
        staged = PiSubstrate().materialise(tree_dir, None, {}).staging_dir
        self.addCleanup(shutil.rmtree, staged, True)

        files = sorted(p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file())
        self.assertEqual(
            files, ["prompt.md", "scripts/list_exports.py", "skills/trace-callers/SKILL.md"]
        )
        prompt = (staged / "prompt.md").read_text()
        self.assertIn("Read the request twice.", prompt)
        self.assertIn("trace-callers", prompt)

    def test_an_empty_base_stages_no_prompt(self):
        tree_dir = HT.write_tree(PI_SURFACE, {}, self.root / "tree")
        staged = PiSubstrate().materialise(tree_dir, None, {}).staging_dir
        self.addCleanup(shutil.rmtree, staged, True)

        self.assertFalse((staged / "prompt.md").exists())

    def test_trajectory_renders_actions_and_observations(self):
        stream = self.root / "events.jsonl"
        stream.write_text(
            "\n".join(
                json.dumps(m)
                for m in (
                    {"role": "user", "content": "fix it", "toolCalls": []},
                    {
                        "role": "assistant",
                        "content": "Looking.",
                        "toolCalls": [{"id": "1", "name": "bash", "arguments": {"command": "ls"}}],
                    },
                    {"role": "tool", "content": "src", "isError": False, "toolCalls": []},
                    {"role": "tool", "content": "boom", "isError": True, "toolCalls": []},
                )
            )
        )
        rendered = PiSubstrate().read_trajectory(stream, obs_trunc=1000, max_chars=10_000)

        self.assertEqual(
            rendered.splitlines(),
            [
                "[ASSISTANT] Looking.",
                '[ACTION] bash({"command": "ls"})',
                "[OBSERVATION] src",
                "[OBSERVATION ERROR] boom",
            ],
        )

    def test_the_improver_sees_the_pi_surface(self):
        text = prompts._st(PI_SURFACE).surface_md
        self.assertIn("no subagents and no in-loop hooks", text)
        self.assertNotIn("Jinja", text)


if __name__ == "__main__":
    unittest.main()
