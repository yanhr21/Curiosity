#!/usr/bin/env bash
set -euo pipefail

# Reproduce the complete-G1 flat-sided CarryBox geometry sample.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

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
export PYTHONPATH="$ROOT:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/SUGAR/source/sugar_rl"
cd "$ROOT"

"$PYTHON_BIN" scripts/sugar/native_tactile/collect_sugar_whole_hand_carrybox.py \
  --output-root "$OUTPUT_ROOT" \
  --object-kind carrybox \
  --object-scale 1.6 1.0 1.0 \
  --motion-id 45 \
  --start-step 249 \
  --scenario unmodified_official_policy \
  --max-steps 80 \
  --mass-kg 0.5 \
  --disable-optical \
  --physical-stiffness 100 \
  --physical-damping 20 \
  --normal-stiffness 20 \
  --tangential-stiffness 2 \
  --headless \
  --enable_cameras \
  --device cuda:0

mkdir -p "$OUTPUT_ROOT/videos"
"$PYTHON_BIN" scripts/sugar/native_tactile/render_sugar_whole_hand_carrybox.py \
  --run-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/videos/plain_carrybox_world_bilateral_taxels.mp4" \
  --title "IsaacLab native TacSL | complete G1 plain CarryBox local-X 1.6x" \
  --scale-note "per-trace automatic quantile; flat-box geometry sample" \
  --fps 50

FFMPEG="$("$PYTHON_BIN" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
"$FFMPEG" -v error -i "$OUTPUT_ROOT/world_carrybox.mp4" -f null -
"$FFMPEG" -v error -i "$OUTPUT_ROOT/videos/plain_carrybox_world_bilateral_taxels.mp4" -f null -

echo "Complete-G1 flat CarryBox visualization: $OUTPUT_ROOT"
