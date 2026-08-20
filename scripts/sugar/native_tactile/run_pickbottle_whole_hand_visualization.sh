#!/usr/bin/env bash
set -euo pipefail

# Collect and render one full IsaacLab SUGAR G1 PickBottle rollout with the
# official bilateral 27-patch whole-hand TacSL configuration.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run inside the retained H200 Slurm allocation." >&2
  exit 2
fi
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 OUTPUT_ROOT [MOTION_ID] [MAX_STEPS]" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
MOTION_ID="${2:-17}"
MAX_STEPS="${3:-269}"
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"
fi
case "$OUTPUT_ROOT" in
  "$ROOT"/experiments/*) ;;
  *) echo "OUTPUT_ROOT must remain below $ROOT/experiments" >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing overwrite: $OUTPUT_ROOT" >&2
  exit 2
fi

export OMNI_KIT_ACCEPT_EULA=Y
export DISPLAY=
export PYTHONPATH="$ROOT:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/SUGAR/source/sugar_rl"
cd "$ROOT"

"$PYTHON_BIN" scripts/sugar/native_tactile/collect_sugar_whole_hand_carrybox.py \
  --output-root "$OUTPUT_ROOT" \
  --object-kind bottle \
  --motion-id "$MOTION_ID" \
  --scenario unmodified_official_policy \
  --max-steps "$MAX_STEPS" \
  --headless \
  --enable_cameras \
  --device cuda:0

END_FRAME="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_frames"])' "$OUTPUT_ROOT/summary.json")"
"$PYTHON_BIN" scripts/sugar/native_tactile/render_sugar_whole_hand_carrybox.py \
  --run-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/pickbottle_bilateral_27patch_native_tactile.mp4" \
  --title "IsaacLab SUGAR G1 PickBottle motion $MOTION_ID - world motion and bilateral whole-hand TacSL" \
  --normal-max 0.1694514982402323 \
  --shear-max 0.09577531702816477 \
  --scale-note "fixed from motion 17 for cross-motion comparison" \
  --start-frame 0 \
  --end-frame "$END_FRAME" \
  --fps 50

FFMPEG="$($PYTHON_BIN -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
"$FFMPEG" -v error -i "$OUTPUT_ROOT/world_bottle.mp4" -f null -
"$FFMPEG" -v error -i "$OUTPUT_ROOT/pickbottle_bilateral_27patch_native_tactile.mp4" -f null -

echo "PickBottle whole-hand tactile visualization: $OUTPUT_ROOT"
