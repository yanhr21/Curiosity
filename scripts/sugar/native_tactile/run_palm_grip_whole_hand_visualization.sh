#!/usr/bin/env bash
set -euo pipefail

# Reproduce one complete-G1 free palm-grip sample with the official Refiner
# and both anatomical 27-patch TacSL hands.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run inside the retained H200 Slurm allocation." >&2
  exit 2
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 OUTPUT_ROOT MASS_KG [CONTACT_FRICTION]" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
MASS_KG="$2"
CONTACT_FRICTION="${3:-}"
PALM_GRIP_SCENARIO="${PALM_GRIP_SCENARIO:-nominal}"
case "$PALM_GRIP_SCENARIO" in
  nominal)
    SCENARIO_ARGS=(--scenario unmodified_official_policy)
    ;;
  release_failure)
    SCENARIO_ARGS=(--scenario failed_grasp --release-step "${PALM_GRIP_RELEASE_STEP:-30}")
    ;;
  *)
    echo "PALM_GRIP_SCENARIO must be nominal or release_failure" >&2
    exit 2
    ;;
esac
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

FRICTION_ARGS=()
if [[ -n "$CONTACT_FRICTION" ]]; then
  FRICTION_ARGS=(--contact-friction "$CONTACT_FRICTION")
fi

"$PYTHON_BIN" scripts/sugar/native_tactile/collect_sugar_whole_hand_carrybox.py \
  --output-root "$OUTPUT_ROOT" \
  --object-kind palm_grip \
  --motion-id 45 \
  --start-step 249 \
  "${SCENARIO_ARGS[@]}" \
  --max-steps 80 \
  --mass-kg "$MASS_KG" \
  "${FRICTION_ARGS[@]}" \
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
  --output "$OUTPUT_ROOT/videos/palm_grip_world_bilateral_taxels.mp4" \
  --title "IsaacLab native TacSL | complete G1 palm grip, ${MASS_KG} kg, ${PALM_GRIP_SCENARIO}" \
  --normal-max 0.033457712326198805 \
  --shear-max 0.029537918334826826 \
  --scale-note "fixed to the retained 0.5-kg nominal sample" \
  --fps 50

FFMPEG="$($PYTHON_BIN -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
"$FFMPEG" -v error -i "$OUTPUT_ROOT/world_palm_grip.mp4" -f null -
"$FFMPEG" -v error -i "$OUTPUT_ROOT/videos/palm_grip_world_bilateral_taxels.mp4" -f null -

echo "Complete-G1 palm-grip visualization: $OUTPUT_ROOT"
