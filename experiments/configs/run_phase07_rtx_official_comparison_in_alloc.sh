#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
METHOD="rtx" \
ENV_PATH="envs/rtx/.venv" \
STAGE1_FILES=$'experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/rtx_stage1/episodes.jsonl\nexperiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/rtx_stage1/rtx_phase07_mapping.json' \
CHECKPOINT_GLOBS=$'*rtx*\n*rt_1_x*\n*open_x*\n*openx*' \
BLOCKER_PATH="experiments/reports/phase07_official_checkpoint_blockers_v1_20260627/rtx_checkpoint_blocker.json" \
  bash "$ROOT/experiments/configs/run_phase07_official_comparison_gate_common.sh"
