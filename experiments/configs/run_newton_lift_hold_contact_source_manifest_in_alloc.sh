#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
CONFIG="${CONFIG:-$ROOT/experiments/configs/newton_lift_hold_contact_source_manifest_v1.json}"

cd "$ROOT"

sed -n '1,120p' AGENTS.md >/dev/null

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv; configure envs/ locally before compute use." >&2
  exit 4
fi

source "$NEWTON_VENV/bin/activate"

"$NEWTON_VENV/bin/python" "$ROOT/experiments/configs/build_newton_lift_hold_contact_source_manifest.py" \
  --config "$CONFIG" \
  --root "$ROOT"
