#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
STAMP="${STAMP:-20260705_core_world_g1_front_probe_bumper_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/core_world_g1_box_scene/${STAMP}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/core_world_g1_box_scene}"

cd "${ROOT_DIR}"
echo "[SUBMIT] STAMP=${STAMP}"
echo "[SUBMIT] OUTPUT_DIR=${OUTPUT_DIR}"

srun -p gpu \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=60G \
  --time=01:00:00 \
  --job-name="${JOB_NAME:-g1_probe}" \
  bash -lc "cd '${ROOT_DIR}' && rg -n -- '--probe-mode' scripts/isaac/run_core_world_g1_front_probe_bumper_smoke.sh && bash -n scripts/isaac/run_core_world_g1_front_probe_bumper_smoke.sh && STAMP='${STAMP}' OUTPUT_DIR='${OUTPUT_DIR}' LOG_DIR='${LOG_DIR}' bash scripts/isaac/run_core_world_g1_front_probe_bumper_smoke.sh"
