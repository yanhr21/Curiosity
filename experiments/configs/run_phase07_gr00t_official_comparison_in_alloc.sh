#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
METHOD="gr00t" \
ENV_PATH="envs/gr00t/.venv" \
STAGE1_FILES=$'experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/gr00t_lerobot_v2_stage1/episodes.jsonl\nexperiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/gr00t_lerobot_v2_stage1/meta/modality.json\nexperiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/gr00t_lerobot_v2_stage1/meta/info.json' \
CHECKPOINT_GLOBS=$'*gr00t*\n*groot*\n*n1*' \
BLOCKER_PATH="experiments/reports/phase07_official_checkpoint_blockers_v1_20260627/gr00t_checkpoint_blocker.json" \
  bash "$ROOT/experiments/configs/run_phase07_official_comparison_gate_common.sh"
