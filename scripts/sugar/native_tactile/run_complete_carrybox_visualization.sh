#!/usr/bin/env bash
set -euo pipefail

# Reproduce one complete SUGAR CarryBox whole-hand tactile visualization.
# Run this script as a recorded child process inside an existing retained
# Slurm allocation.  It never allocates, cancels, or releases GPU resources.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run inside an existing retained Slurm allocation." >&2
  exit 2
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 OUTPUT_ROOT [successful_grasp|failed_grasp|failed_closure]" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
SCENARIO="${2:-successful_grasp}"
case "$SCENARIO" in
  successful_grasp)
    MAX_STEPS=660
    START_FRAME=230
    MAIN_VIDEO="successful_carrybox_whole_hand_tactile.mp4"
    TITLE="SUGAR CarryBox - complete grasp, carry, placement and release"
    ;;
  failed_grasp)
    MAX_STEPS=420
    START_FRAME=230
    MAIN_VIDEO="failed_carrybox_whole_hand_tactile.mp4"
    TITLE="SUGAR CarryBox - grasp followed by physical release"
    ;;
  failed_closure)
    MAX_STEPS=320
    START_FRAME=200
    MAIN_VIDEO="failed_closure_carrybox_whole_hand_tactile.mp4"
    TITLE="SUGAR CarryBox - physical failed closure"
    ;;
  *)
    echo "Unknown scenario: $SCENARIO" >&2
    exit 2
    ;;
esac

if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"
fi
case "$OUTPUT_ROOT" in
  "$ROOT"/experiments/*) ;;
  *)
    echo "OUTPUT_ROOT must remain below $ROOT/experiments" >&2
    exit 2
    ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing overwrite: $OUTPUT_ROOT" >&2
  exit 2
fi

export PYTHONPATH="$ROOT:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/SUGAR/source/sugar_rl"
cd "$ROOT"

"$PYTHON_BIN" scripts/sugar/native_tactile/collect_sugar_whole_hand_carrybox.py \
  --output-root "$OUTPUT_ROOT" \
  --scenario "$SCENARIO" \
  --max-steps "$MAX_STEPS" \
  --headless \
  --enable_cameras \
  --device cuda:0

END_FRAME="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_frames"])' "$OUTPUT_ROOT/summary.json")"
if (( START_FRAME >= END_FRAME )); then
  echo "Recorded trace ends before the selected review interval." >&2
  exit 1
fi

COMMON_RENDER_ARGS=(
  --run-root "$OUTPUT_ROOT"
  --normal-max 0.5768324136734009
  --shear-max 0.5144117593765258
  --scale-note "fixed scale shared with physical failure controls"
  --start-frame "$START_FRAME"
  --end-frame "$END_FRAME"
  --fps 50
)

"$PYTHON_BIN" scripts/sugar/native_tactile/render_sugar_whole_hand_carrybox.py \
  "${COMMON_RENDER_ARGS[@]}" \
  --output "$OUTPUT_ROOT/$MAIN_VIDEO" \
  --title "$TITLE"

for KIND in left_detail right_detail palm_optical; do
  "$PYTHON_BIN" scripts/sugar/native_tactile/render_sugar_whole_hand_supplement.py \
    "${COMMON_RENDER_ARGS[@]}" \
    --kind "$KIND" \
    --output "$OUTPUT_ROOT/${KIND}.mp4" \
    --title "$TITLE - ${KIND//_/ }"
done

if [[ "$SCENARIO" == "successful_grasp" ]]; then
  "$PYTHON_BIN" scripts/sugar/native_tactile/render_sugar_force_kinematics_friction.py \
    --run-root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/force_kinematics_friction_complete.mp4" \
    --start-frame "$START_FRAME" \
    --end-frame "$END_FRAME" \
    --fps 50
fi

bash scripts/sugar/native_tactile/validate_complete_carrybox_bundle.sh \
  "$OUTPUT_ROOT" \
  "$SCENARIO"

echo "Complete CarryBox tactile bundle: $OUTPUT_ROOT"
