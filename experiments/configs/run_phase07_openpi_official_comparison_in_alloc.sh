#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
METHOD="openpi_pi0" \
ENV_PATH="envs/openpi/.venv" \
STAGE1_FILES=$'experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/openpi_lerobot_stage1/episodes.jsonl\nexperiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/openpi_lerobot_stage1/openpi_phase07_mapping.json' \
CHECKPOINT_GLOBS=$'*openpi*\n*pi0*\n*pi05*' \
BLOCKER_PATH="experiments/reports/phase07_official_checkpoint_blockers_v1_20260627/openpi_pi0_checkpoint_blocker.json" \
  bash "$ROOT/experiments/configs/run_phase07_official_comparison_gate_common.sh"
