#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run project data export on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
STAMP="${STAMP:-20260705_direct_carry_task_episode_table}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/experiments/outputs/rl_interface/${STAMP}/direct_carry_task_episode_table.jsonl}"

cd "${ROOT_DIR}"
mkdir -p "$(dirname "${OUTPUT}")"

"${ISAAC_VENV}/bin/python" scripts/isaac/export_direct_carry_task_episode_table.py \
  "$@" \
  --output "${OUTPUT}"

echo "[INFO] Direct carry task episode table output: ${OUTPUT}"
