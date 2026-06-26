#!/usr/bin/env bash
set -euo pipefail

# Launch the Phase 02 no-adaptation scripted infant-prior baseline in an
# existing tmux-held allocation. This script submits no new allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-lift_hold_no_adaptation_scripted_baseline_v1_$(date +%Y%m%d_%H%M%S)}"
WINDOW_NAME="${WINDOW_NAME:-lift_hold_noadapt_baseline}"
SCENE="${SCENE:-cube}"
TRACKED_OBJECT="${TRACKED_OBJECT:-official_object}"
CONTROLLER_MODE="${CONTROLLER_MODE:-lift_hold}"
FINAL_HOLD_DURATION="${FINAL_HOLD_DURATION:-2.5}"
NUM_STEPS="${NUM_STEPS:-360}"
SAMPLE_STEPS="${SAMPLE_STEPS:-0,45,90,135,180,225,270,315,359}"

cd "$ROOT"

if [[ "$TRACKED_OBJECT" != "official_object" ]]; then
  echo "ERROR: Phase 02 no-adaptation baseline v1 must use TRACKED_OBJECT=official_object." >&2
  echo "Cup-asset retargeting remains a Phase 01 adaptation issue until stable cup grasp is cleared." >&2
  exit 2
fi

bash -n "$ROOT/experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh"
bash -n "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_in_alloc.sh"

RUN_TAG="$RUN_TAG" \
WINDOW_NAME="$WINDOW_NAME" \
SCENE="$SCENE" \
TRACKED_OBJECT="$TRACKED_OBJECT" \
CONTROLLER_MODE="$CONTROLLER_MODE" \
FINAL_HOLD_DURATION="$FINAL_HOLD_DURATION" \
NUM_STEPS="$NUM_STEPS" \
SAMPLE_STEPS="$SAMPLE_STEPS" \
bash "$ROOT/experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh"
