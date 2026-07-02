#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
METHOD="diffusion_policy" \
ENV_PATH="envs/diffusion_policy/conda" \
STAGE1_FILES=$'experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/diffusion_policy_stage1/episodes.jsonl\nexperiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/diffusion_policy_stage1/shape_meta.json' \
CHECKPOINT_GLOBS=$'*diffusion*\n*pusht*\n*robomimic*' \
BLOCKER_PATH="experiments/reports/phase07_official_checkpoint_blockers_v1_20260627/diffusion_policy_checkpoint_blocker.json" \
  bash "$ROOT/experiments/configs/run_phase07_official_comparison_gate_common.sh"
