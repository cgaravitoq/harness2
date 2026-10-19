#!/usr/bin/env bash
# Runs harness2-hb inside a bubblewrap sandbox whose harness-bench checkout shows only the
# training tasks' prompts: the improver is a coding agent with a shell, so a hidden test, a
# reference patch or a held-out task it could read would leak into the harness it writes.
# harness-bench's own sandbox still wraps every Pi rollout inside this one.
set -euo pipefail

: "${HARNESS2_HARNESS_BENCH_DIR:?point it at a harness-bench checkout}"
: "${HARNESS2_RUNS_DIR:?point it at the experiment directory}"
: "${HARNESS2_SPLIT_FILE:?point it at the split file}"
: "${ANTHROPIC_API_KEY:?export the Anthropic API key}"

repo=$(cd "$(dirname "$0")/.." && pwd)
hb=$(cd "$HARNESS2_HARNESS_BENCH_DIR" && pwd)
runs=$(mkdir -p "$HARNESS2_RUNS_DIR" && cd "$HARNESS2_RUNS_DIR" && pwd)
cache=${XDG_CACHE_HOME:-$HOME/.cache}
mkdir -p "$runs/tmp"

task_binds=()
while IFS= read -r task; do
  for file in task.json prompt.md; do
    task_binds+=(--ro-bind "$hb/tasks/$task/$file" "$hb/tasks/$task/$file")
  done
done < <("$repo/.venv/bin/python" -I -c '
import json, sys
print("\n".join(json.load(open(sys.argv[1]))["train"]))' "$HARNESS2_SPLIT_FILE")

exec bwrap \
  --ro-bind /usr /usr \
  --symlink usr/lib /lib \
  --symlink usr/lib64 /lib64 \
  --symlink usr/bin /bin \
  --symlink usr/bin /sbin \
  --ro-bind /etc /etc \
  --ro-bind /run/systemd/resolve /run/systemd/resolve \
  --dev /dev \
  --proc /proc \
  --tmpfs /tmp \
  --tmpfs "$HOME" \
  --ro-bind-try /opt/claude-code /opt/claude-code \
  --ro-bind "$HOME/.bun" "$HOME/.bun" \
  --ro-bind "$HOME/.local/share/uv/python" "$HOME/.local/share/uv/python" \
  --ro-bind "$repo" "$repo" \
  --ro-bind "$hb" "$hb" \
  --tmpfs "$hb/.git" \
  --tmpfs "$hb/tasks" \
  "${task_binds[@]}" \
  --ro-bind "$cache/harness-bench" "$cache/harness-bench" \
  --ro-bind "$HARNESS2_SPLIT_FILE" "$HARNESS2_SPLIT_FILE" \
  --bind "$runs" "$runs" \
  --unshare-pid \
  --die-with-parent \
  --clearenv \
  --setenv PATH "$HOME/.bun/bin:/usr/bin" \
  --setenv HOME "$HOME" \
  --setenv LANG C.UTF-8 \
  --setenv TMPDIR "$runs/tmp" \
  --setenv XDG_CACHE_HOME "$cache" \
  --setenv ANTHROPIC_API_KEY "$ANTHROPIC_API_KEY" \
  --setenv DISABLE_AUTOUPDATER 1 \
  --setenv HARNESS2_HARNESS_BENCH_DIR "$hb" \
  --setenv HARNESS2_RUNS_DIR "$runs" \
  --setenv HARNESS2_SOLVER_MODEL "${HARNESS2_SOLVER_MODEL:-anthropic/claude-sonnet-5-5}" \
  --setenv HARNESS2_IMPROVER_MODEL "${HARNESS2_IMPROVER_MODEL:-claude-sonnet-5-5}" \
  --chdir "$repo" \
  "$repo/.venv/bin/harness2-hb" "$@"
