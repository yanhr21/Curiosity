#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this preflight inside the retained Slurm allocation." >&2
  exit 2
fi
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 tactile|zero|bounded_tactile|bounded_zero|residual_tactile|residual_zero OUTPUT_DIR" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARM="$1"
OUTPUT_DIR="$2"
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
TEACHER="$ROOT/experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt"
MOTION="$ROOT/SUGAR/data/CarryBox/data_045"

case "$ARM" in
  tactile)
    TASK="Sugar-G129dof-CarryBox-NativeWholeHand-ProprioTaskTacSL-BCPPO"
    ;;
  zero)
    TASK="Sugar-G129dof-CarryBox-NativeWholeHand-ProprioTaskZero-BCPPO"
    ;;
  bounded_tactile)
    TASK="Sugar-G129dof-CarryBox-BoundedNativeWholeHand-ProprioTaskTacSL-BCPPO"
    ;;
  bounded_zero)
    TASK="Sugar-G129dof-CarryBox-BoundedNativeWholeHand-ProprioTaskZero-BCPPO"
    ;;
  residual_tactile)
    TASK="Sugar-G129dof-CarryBox-ActionResidualNativeWholeHand-ProprioTaskTacSL-BCPPO"
    ;;
  residual_zero)
    TASK="Sugar-G129dof-CarryBox-ActionResidualNativeWholeHand-ProprioTaskZero-BCPPO"
    ;;
  *)
    echo "Unknown arm: $ARM" >&2
    exit 2
    ;;
esac

if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT/$OUTPUT_DIR"
fi
case "$OUTPUT_DIR" in
  "$ROOT"/experiments/*) ;;
  *)
    echo "OUTPUT_DIR must remain below $ROOT/experiments" >&2
    exit 2
    ;;
esac
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing overwrite: $OUTPUT_DIR" >&2
  exit 2
fi

export PYTHONPATH="$ROOT:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/SUGAR/source/sugar_rl"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export SUGAR_NATIVE_TACTILE_TRAINING_SIGNAL="$OUTPUT_DIR/training_signal.json"
export SUGAR_NATIVE_TACTILE_TRAINING_TRACE="$OUTPUT_DIR/training_tactile_trace.jsonl"
# Official SUGAR resolves descriptions/ assets relative to its repository root.
cd "$ROOT/SUGAR"
"$PYTHON_BIN" scripts/sugar_rl/train_native_whole_hand_tactile_bcppo.py \
  --task "$TASK" \
  --motion_folder "$MOTION" \
  --teacher_ckpt "$TEACHER" \
  --warm_start_checkpoint_path "$TEACHER" \
  --num_envs 2 \
  --max_iterations 1 \
  --seed 13011 \
  --device cuda:0 \
  --headless \
  --log_dir "$OUTPUT_DIR" \
  hydra/job_logging=disabled
