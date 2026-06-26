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
PHYSICS_VARIANT_LABEL="${PHYSICS_VARIANT_LABEL:-nominal}"
BODY_MASS_SCALE="${BODY_MASS_SCALE:-1.0}"
SHAPE_FRICTION_SCALE="${SHAPE_FRICTION_SCALE:-1.0}"
OBJECT_MASS_KG="${OBJECT_MASS_KG:-}"
OBJECT_FRICTION_MU="${OBJECT_FRICTION_MU:-}"
NUM_STEPS="${NUM_STEPS:-360}"
SAMPLE_STEPS="${SAMPLE_STEPS:-0,45,90,135,180,225,270,315,359}"

cd "$ROOT"

case "$TRACKED_OBJECT" in
  official_object)
    ;;
  existing_cup_asset)
    if [[ "$SCENE" != "cube" ]]; then
      echo "ERROR: TRACKED_OBJECT=existing_cup_asset requires SCENE=cube so the official cup asset is loaded." >&2
      exit 2
    fi
    ;;
  *)
    echo "ERROR: unsupported TRACKED_OBJECT=$TRACKED_OBJECT for Phase 02 no-adaptation baseline v1." >&2
    echo "Allowed values: official_object, existing_cup_asset." >&2
    exit 2
    ;;
esac

bash -n "$ROOT/experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh"
bash -n "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh"

RUN_TAG="$RUN_TAG" \
WINDOW_NAME="$WINDOW_NAME" \
SCENE="$SCENE" \
TRACKED_OBJECT="$TRACKED_OBJECT" \
CONTROLLER_MODE="$CONTROLLER_MODE" \
FINAL_HOLD_DURATION="$FINAL_HOLD_DURATION" \
PHYSICS_VARIANT_LABEL="$PHYSICS_VARIANT_LABEL" \
BODY_MASS_SCALE="$BODY_MASS_SCALE" \
SHAPE_FRICTION_SCALE="$SHAPE_FRICTION_SCALE" \
OBJECT_MASS_KG="$OBJECT_MASS_KG" \
OBJECT_FRICTION_MU="$OBJECT_FRICTION_MU" \
NUM_STEPS="$NUM_STEPS" \
SAMPLE_STEPS="$SAMPLE_STEPS" \
bash "$ROOT/experiments/configs/launch_newton_panda_hydro_camera_export_tmux.sh"
