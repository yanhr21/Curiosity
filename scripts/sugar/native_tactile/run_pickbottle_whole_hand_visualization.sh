#!/usr/bin/env bash
set -euo pipefail

# Collect and render one full IsaacLab SUGAR G1 PickBottle rollout with the
# official bilateral 27-patch whole-hand TacSL configuration.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-/public/home/yanhongru/envs/sugar_py311_isaacsim510_clean/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run inside the retained H200 Slurm allocation." >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
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
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONPATH="$ROOT:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/SUGAR/source/sugar_rl"
cd "$ROOT"

"$PYTHON_BIN" scripts/sugar/native_tactile/collect_sugar_whole_hand_carrybox.py \
  --output-root "$OUTPUT_ROOT" \
  --object-kind bottle \
  --motion-id 17 \
  --scenario successful_grasp \
  --max-steps 270 \
  --headless \
  --enable_cameras \
  --device cuda:0

END_FRAME="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_frames"])' "$OUTPUT_ROOT/summary.json")"
"$PYTHON_BIN" scripts/sugar/native_tactile/render_sugar_whole_hand_carrybox.py \
  --run-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/pickbottle_bilateral_27patch_native_tactile.mp4" \
  --title "IsaacLab SUGAR G1 PickBottle - world motion and bilateral whole-hand TacSL" \
  --scale-note "fixed after this success sample for its matched failure" \
  --start-frame 0 \
  --end-frame "$END_FRAME" \
  --fps 50

FFMPEG="$($PYTHON_BIN -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
"$FFMPEG" -v error -i "$OUTPUT_ROOT/world_bottle.mp4" -f null -
"$FFMPEG" -v error -i "$OUTPUT_ROOT/pickbottle_bilateral_27patch_native_tactile.mp4" -f null -

echo "PickBottle whole-hand tactile visualization: $OUTPUT_ROOT"
