#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this evaluation inside the retained Slurm allocation." >&2
  exit 2
fi
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 tactile|zero|bounded_tactile|bounded_zero|residual_tactile|residual_zero CHECKPOINT OUTPUT_JSON [EVALUATOR_ARGS...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARM="$1"
CHECKPOINT="$2"
OUTPUT="$3"
shift 3
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
TEACHER="$ROOT/experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt"
MOTION="$ROOT/SUGAR/data/CarryBox/data_045"

case "$ARM" in tactile|zero|bounded_tactile|bounded_zero|residual_tactile|residual_zero) ;; *) echo "Unknown arm: $ARM" >&2; exit 2 ;; esac
if [[ "$CHECKPOINT" != /* ]]; then CHECKPOINT="$ROOT/$CHECKPOINT"; fi
if [[ "$OUTPUT" != /* ]]; then OUTPUT="$ROOT/$OUTPUT"; fi
if [[ ! -f "$CHECKPOINT" ]]; then echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; fi
case "$OUTPUT" in "$ROOT"/experiments/*) ;; *) echo "Output must be under experiments" >&2; exit 2 ;; esac
if [[ -e "$OUTPUT" || -e "${OUTPUT%.json}.npz" ]]; then
  echo "Refusing overwrite: $OUTPUT" >&2
  exit 2
fi

export PYTHONPATH="$ROOT:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/SUGAR/source/sugar_rl"
export PYTHONUNBUFFERED=1
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1

cd "$ROOT/SUGAR"
"$PYTHON_BIN" scripts/sugar_rl/evaluate_native_whole_hand_tactile_bcppo.py \
  --arm "$ARM" \
  --checkpoint "$CHECKPOINT" \
  --teacher_checkpoint "$TEACHER" \
  --motion_folder "$MOTION" \
  --output "$OUTPUT" \
  --start_frame 0 \
  --max_steps 660 \
  --seed 13011 \
  --device cuda:0 \
  --headless \
  "$@"
