#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run late-stop Isaac batch on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
STAMP_SUFFIX="${STAMP_SUFFIX:-late_retry2}"
CRADLE_MASS_SCALE_VALUE="${CRADLE_MASS_SCALE_VALUE:-0.15}"
cd "${ROOT_DIR}"

sleep "${COMPUTE_SIDE_STARTUP_SLEEP:-35}"
nl -ba scripts/isaac/run_core_world_g1_box_scene.sh | sed -n '60,72p'
bash -n scripts/isaac/run_core_world_g1_box_scene.sh
python3 -m py_compile \
  scripts/isaac/build_core_world_g1_box_scene.py \
  scripts/isaac/check_core_world_g1_box_scene_summary.py

for spec in 320:diag59 340:diag60 360:diag61 370:diag62; do
  stop="${spec%%:*}"
  diag="${spec##*:}"
  stamp="20260705_core_world_g1_min_cradle_amp016_stop${stop}_${diag}_${STAMP_SUFFIX}"
  echo "[BATCH] Running ${diag} stop=${stop} stamp=${stamp}"
  STAMP="${stamp}" \
  STEPS=420 \
  ATTACH_BOX=none \
  TORSO_CRADLE=front_tray \
  REQUIRE_BOX_NO_DROP=1 \
  BOX_MASS=0.25 \
  BOX_SIZE_X=0.10 \
  BOX_SIZE_Y=0.08 \
  BOX_SIZE_Z=0.06 \
  BOX_POS_X=0.44 \
  BOX_POS_Y=0.0 \
  BOX_POS_Z=0.95 \
  G1_ROOT_Z=0.78 \
  STAND_HIP_PITCH=-0.12 \
  STAND_KNEE=0.30 \
  STAND_ANKLE_PITCH=-0.15 \
  APPLY_ARENA_STAND_GAINS=1 \
  STAND_DRIVE_PRESET=arena \
  STAND_GAIN_SCALE=1.0 \
  GAIT_MODE=open_loop_march \
  GAIT_AMPLITUDE=0.16 \
  GAIT_FREQUENCY_HZ=0.7 \
  GAIT_STOP_STEP="${stop}" \
  CRADLE_DECK_SIZE_X=0.24 \
  CRADLE_DECK_SIZE_Y=0.26 \
  CRADLE_DECK_SIZE_Z=0.025 \
  CRADLE_DECK_LOCAL_POS0_X=0.44 \
  CRADLE_DECK_LOCAL_POS0_Y=0.0 \
  CRADLE_DECK_LOCAL_POS0_Z=0.10 \
  CRADLE_SIDE_RAIL_HEIGHT=0.07 \
  CRADLE_END_STOP_HEIGHT=0.08 \
  CRADLE_RAIL_THICKNESS=0.018 \
  CRADLE_MASS_SCALE="${CRADLE_MASS_SCALE_VALUE}" \
  bash scripts/isaac/run_core_world_g1_box_scene.sh

  python3 scripts/isaac/check_core_world_g1_box_scene_summary.py \
    "experiments/outputs/core_world_g1_box_scene/${stamp}/core_world_g1_box_scene_summary.json" \
    --min-steps 420 \
    --expect-attach-box none \
    --expect-torso-cradle front_tray \
    --expect-carry-box-spawned true \
    --min-cradle-piece-count 5 \
    --min-joint-count 40 \
    --max-fall-events 0 \
    --max-box-drop-events 0 \
    --min-robot-z 0.45 \
    --min-box-z 0.20 \
    --max-tilt 0.85 \
    --max-root-pose-write-count-rollout 0 \
    --max-root-velocity-write-count-rollout 0 \
    --max-box-pose-write-count-rollout 0 \
    --min-final-box-target-directed-travel 0.10 \
    --require-diagnostic-claim || true
done
