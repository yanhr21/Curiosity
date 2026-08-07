#!/usr/bin/env bash
set -euo pipefail

# New-inode successor to base_v22 admitting a bounded 5 mm near-contact
# approach diagnostic and using the final tilted palm normal. It keeps the exact official coupled hand
# command and additionally freezes the official CarryBox material readback and
# the optional real fixed setup pedestal used to place fingers below the box.

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
UNITREE_REPO=/public/home/yanhongru/reference_upstreams/unitree_sim_isaaclab
ASSET_ROOT=/public/home/yanhongru/reference_upstreams/unitree_sim_isaaclab_usds/extracted
OUTPUT_SLUG=${PLAN10_OUTPUT_SLUG:-g1_inspire_sugar_side_clamp_carrybox_v1}
OUTPUT_ROOT="${ROOT}/experiments/sugar_reproduction/articulated_five_finger_soft_tactile/${OUTPUT_SLUG}"
LEFT_TANGENT_M=${PLAN10_LEFT_TANGENT_M:--0.12}
RIGHT_TANGENT_M=${PLAN10_RIGHT_TANGENT_M:--0.12}
LEFT_LOCAL_Z_M=${PLAN10_LEFT_LOCAL_Z_M:--0.04}
RIGHT_LOCAL_Z_M=${PLAN10_RIGHT_LOCAL_Z_M:--0.04}
BOX_AXIS=${PLAN10_BOX_AXIS:-y}
LEFT_NORMAL_M=${PLAN10_LEFT_NORMAL_M:-}
RIGHT_NORMAL_M=${PLAN10_RIGHT_NORMAL_M:-}
LEFT_CONTACT_PCA_M=${PLAN10_LEFT_CONTACT_PCA_M:-}
RIGHT_CONTACT_PCA_M=${PLAN10_RIGHT_CONTACT_PCA_M:-}
LEFT_OUTWARD_PCA=${PLAN10_LEFT_OUTWARD_PCA:-}
RIGHT_OUTWARD_PCA=${PLAN10_RIGHT_OUTWARD_PCA:-}
CONTACT_GEOMETRY_SOURCE=${PLAN10_CONTACT_GEOMETRY_SOURCE:-}
BOX_LOCAL_X_OFFSET_M=${PLAN10_BOX_LOCAL_X_OFFSET_M:-0.0}
BOX_LOCAL_Y_OFFSET_M=${PLAN10_BOX_LOCAL_Y_OFFSET_M:-0.0}
BOX_LOCAL_Z_OFFSET_M=${PLAN10_BOX_LOCAL_Z_OFFSET_M:-0.0}
APPROACH_CLEARANCE_M=${PLAN10_APPROACH_CLEARANCE_M:-0.04}
PALM_INSET_M=${PLAN10_PALM_INSET_M:-0.0015}
LEFT_PALM_INSET_M=${PLAN10_LEFT_PALM_INSET_M:-}
RIGHT_PALM_INSET_M=${PLAN10_RIGHT_PALM_INSET_M:-}
LEFT_TILT_TANGENT_RAD=${PLAN10_LEFT_TILT_TANGENT_RAD:-0.0}
RIGHT_TILT_TANGENT_RAD=${PLAN10_RIGHT_TILT_TANGENT_RAD:-0.0}
LEFT_TILT_HEIGHT_RAD=${PLAN10_LEFT_TILT_HEIGHT_RAD:-0.0}
RIGHT_TILT_HEIGHT_RAD=${PLAN10_RIGHT_TILT_HEIGHT_RAD:-0.0}
CLOSE_FRACTION=${PLAN10_CLOSE_FRACTION:-0.55}
LEFT_LITTLE_CLOSE_FRACTION=${PLAN10_LEFT_LITTLE_CLOSE_FRACTION:-}
RIGHT_INDEX_CLOSE_FRACTION=${PLAN10_RIGHT_INDEX_CLOSE_FRACTION:-}
RIGHT_LITTLE_CLOSE_FRACTION=${PLAN10_RIGHT_LITTLE_CLOSE_FRACTION:-}
CLOSED_THUMB_PITCH_RAD=${PLAN10_CLOSED_THUMB_PITCH_RAD:-0.30}
CLOSED_THUMB_YAW_RAD=${PLAN10_CLOSED_THUMB_YAW_RAD:-0.30}
PREGRASP_THUMB_PITCH_RAD=${PLAN10_PREGRASP_THUMB_PITCH_RAD:-0.4}
PREGRASP_THUMB_YAW_RAD=${PLAN10_PREGRASP_THUMB_YAW_RAD:--0.1}
LEFT_PREGRASP_THUMB_PITCH_RAD=${PLAN10_LEFT_PREGRASP_THUMB_PITCH_RAD:-${PREGRASP_THUMB_PITCH_RAD}}
RIGHT_PREGRASP_THUMB_PITCH_RAD=${PLAN10_RIGHT_PREGRASP_THUMB_PITCH_RAD:-${PREGRASP_THUMB_PITCH_RAD}}
LEFT_PREGRASP_THUMB_YAW_RAD=${PLAN10_LEFT_PREGRASP_THUMB_YAW_RAD:-${PREGRASP_THUMB_YAW_RAD}}
RIGHT_PREGRASP_THUMB_YAW_RAD=${PLAN10_RIGHT_PREGRASP_THUMB_YAW_RAD:-${PREGRASP_THUMB_YAW_RAD}}
LEFT_CLOSED_THUMB_PITCH_RAD=${PLAN10_LEFT_CLOSED_THUMB_PITCH_RAD:-${CLOSED_THUMB_PITCH_RAD}}
RIGHT_CLOSED_THUMB_PITCH_RAD=${PLAN10_RIGHT_CLOSED_THUMB_PITCH_RAD:-${CLOSED_THUMB_PITCH_RAD}}
LEFT_CLOSED_THUMB_YAW_RAD=${PLAN10_LEFT_CLOSED_THUMB_YAW_RAD:-${CLOSED_THUMB_YAW_RAD}}
RIGHT_CLOSED_THUMB_YAW_RAD=${PLAN10_RIGHT_CLOSED_THUMB_YAW_RAD:-${CLOSED_THUMB_YAW_RAD}}
PALM_PRESS_STEPS=${PLAN10_PALM_PRESS_STEPS:-100}
CLOSE_STEPS=${PLAN10_CLOSE_STEPS:-200}
THUMB_CLOSE_STEPS=${PLAN10_THUMB_CLOSE_STEPS:-200}
CONTACT_SETTLE_STEPS=${PLAN10_CONTACT_SETTLE_STEPS:-100}
REQUIRE_SETTLED_ALL_GROUPS_FRAMES=${PLAN10_REQUIRE_SETTLED_ALL_GROUPS_FRAMES:-0}
CONTACT_PRELOAD_HEIGHT_M=${PLAN10_CONTACT_PRELOAD_HEIGHT_M:-0.0}
CONTACT_PRELOAD_STEPS=${PLAN10_CONTACT_PRELOAD_STEPS:-0}
CONTACT_PRELOAD_RAMP_STEPS=${PLAN10_CONTACT_PRELOAD_RAMP_STEPS:-${CONTACT_PRELOAD_STEPS}}
CONTACT_COMPRESSION_M=${PLAN10_CONTACT_COMPRESSION_M:-0.0}
CONTACT_COMPRESSION_STEPS=${PLAN10_CONTACT_COMPRESSION_STEPS:-0}
CONTACT_COMPRESSION_RAMP_STEPS=${PLAN10_CONTACT_COMPRESSION_RAMP_STEPS:-${CONTACT_COMPRESSION_STEPS}}
LIFT_STEPS=${PLAN10_LIFT_STEPS:-300}
LIFT_HEIGHT_M=${PLAN10_LIFT_HEIGHT_M:-0.18}
FIT_BOX_TO_REACHABLE_PALMS=${PLAN10_FIT_BOX_TO_REACHABLE_PALMS:-0}
DIRECT_SETUP_IK=${PLAN10_DIRECT_SETUP_IK:-0}
DIRECT_REFINEMENT_STEPS=${PLAN10_DIRECT_REFINEMENT_STEPS:-200}
MAX_REACHABLE_ORIENTATION_DELTA_RAD=${PLAN10_MAX_REACHABLE_ORIENTATION_DELTA_RAD:-0.10}
SIMULTANEOUS_HAND_CLOSE=${PLAN10_SIMULTANEOUS_HAND_CLOSE:-0}
BILATERAL_SHOULDER_ROLL_OFFSET_RAD=${PLAN10_BILATERAL_SHOULDER_ROLL_OFFSET_RAD:-0.0}
BILATERAL_HIP_ROLL_OUTWARD_OFFSET_RAD=${PLAN10_BILATERAL_HIP_ROLL_OUTWARD_OFFSET_RAD:-0.0}
LEFT_SHOULDER_ROLL_OFFSET_RAD=${PLAN10_LEFT_SHOULDER_ROLL_OFFSET_RAD:-}
RIGHT_SHOULDER_ROLL_OFFSET_RAD=${PLAN10_RIGHT_SHOULDER_ROLL_OFFSET_RAD:-}
HOLD_STEPS=${PLAN10_HOLD_STEPS:-240}
OBJECT_STATIC_FRICTION=${PLAN10_OBJECT_STATIC_FRICTION:-0.6}
OBJECT_DYNAMIC_FRICTION=${PLAN10_OBJECT_DYNAMIC_FRICTION:-0.5}
SUPPORT_HEIGHT_M=${PLAN10_SUPPORT_HEIGHT_M:-0.0}
SUPPORT_SIZE_X_M=${PLAN10_SUPPORT_SIZE_X_M:-0.18}
SUPPORT_SIZE_Y_M=${PLAN10_SUPPORT_SIZE_Y_M:-0.18}
ROBOT_ROOT_Y_OFFSET_M=${PLAN10_ROBOT_ROOT_Y_OFFSET_M:-0.0}
WAIST_PITCH_ABSOLUTE_RAD=${PLAN10_WAIST_PITCH_ABSOLUTE_RAD:-}
LEFT_NORMAL_ROLL_RAD=${PLAN10_LEFT_NORMAL_ROLL_RAD:-0.0}
RIGHT_NORMAL_ROLL_RAD=${PLAN10_RIGHT_NORMAL_ROLL_RAD:-0.0}
TRACK_LIVE_BOX_DURING_GRASP=${PLAN10_TRACK_LIVE_BOX_DURING_GRASP:-1}
TRACK_LIVE_BOX_DURING_LIFT=${PLAN10_TRACK_LIVE_BOX_DURING_LIFT:-0}
LIVE_LIFT_SCHEDULED_WORLD_Z=${PLAN10_LIVE_LIFT_SCHEDULED_WORLD_Z:-0}
LIVE_LIFT_ANCHOR_WORLD_XY=${PLAN10_LIVE_LIFT_ANCHOR_WORLD_XY:-0}
LIVE_HOLD_RELATIVE_TO_BOX=${PLAN10_LIVE_HOLD_RELATIVE_TO_BOX:-0}
LIVE_LIFT_LEAD_M=${PLAN10_LIVE_LIFT_LEAD_M:-}
LIVE_LIFT_LEAD_RAMP_STEPS=${PLAN10_LIVE_LIFT_LEAD_RAMP_STEPS:-0}
LIFT_LEFT_THUMB_YAW_RAD=${PLAN10_LIFT_LEFT_THUMB_YAW_RAD:-}
LIFT_RIGHT_THUMB_YAW_RAD=${PLAN10_LIFT_RIGHT_THUMB_YAW_RAD:-}
LIFT_THUMB_YAW_START_STEP=${PLAN10_LIFT_THUMB_YAW_START_STEP:-0}
LIFT_THUMB_YAW_RAMP_STEPS=${PLAN10_LIFT_THUMB_YAW_RAMP_STEPS:-1}

if [[ $(hostname) == mgmtserver* || $(hostname) == login* ]]; then
  printf 'Refusing Plan-10 SUGAR side-clamp run on login node: %s\n' "$(hostname)" >&2
  exit 2
fi
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  printf 'Plan-10 SUGAR side-clamp run requires the retained allocation\n' >&2
  exit 2
fi
test ! -e "${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/IsaacLab/source/isaaclab:${ROOT}/IsaacLab/source/isaaclab_contrib:${PYTHONPATH:-}"
PORTABLE_ROOT="${TMPDIR:-/tmp}/curiosity_plan10_sugar_side_clamp_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID:-allocation}"
mkdir -p "${PORTABLE_ROOT}"

printf '%s\n' \
  'PLAN10_G1_INSPIRE_SUGAR_SIDE_CLAMP_V1 no-learning physical mechanics' \
  'official SUGAR motion-45 frame-269 root/body/CarryBox source; state written once' \
  'official Inspire palm local-minus-X faces address the declared real CarryBox mesh surfaces' \
  'official IsaacLab DLS IK drives only the two seven-DoF arms; no object replay' \
  'the object is dynamic 0.5 kg on the official floor and lift is open-loop after one frozen anchor'

cd "${ROOT}"
NORMAL_ARGS=()
if [[ -n ${LEFT_NORMAL_M} || -n ${RIGHT_NORMAL_M} ]]; then
  if [[ -z ${LEFT_NORMAL_M} || -z ${RIGHT_NORMAL_M} ]]; then
    printf 'Both PCA normal coordinates must be declared together\n' >&2
    exit 2
  fi
  NORMAL_ARGS=(
    --left-side-clamp-box-normal-m "${LEFT_NORMAL_M}"
    --right-side-clamp-box-normal-m "${RIGHT_NORMAL_M}"
  )
fi
EXACT_CONTACT_ARGS=()
for side in LEFT RIGHT; do
  contact_var="${side}_CONTACT_PCA_M"
  outward_var="${side}_OUTWARD_PCA"
  contact_value=${!contact_var}
  outward_value=${!outward_var}
  if [[ -n ${contact_value} || -n ${outward_value} ]]; then
    if [[ -z ${contact_value} || -z ${outward_value} ]]; then
      printf '%s exact PCA contact and outward vector must be declared together\n' "${side}" >&2
      exit 2
    fi
    read -r -a contact_components <<< "${contact_value}"
    read -r -a outward_components <<< "${outward_value}"
    if [[ ${#contact_components[@]} -ne 3 || ${#outward_components[@]} -ne 3 ]]; then
      printf '%s exact PCA contact and outward vector require three values each\n' "${side}" >&2
      exit 2
    fi
    side_lower=${side,,}
    EXACT_CONTACT_ARGS+=(
      "--${side_lower}-side-clamp-contact-pca-m" "${contact_components[@]}"
      "--${side_lower}-side-clamp-outward-pca" "${outward_components[@]}"
    )
  fi
done
if [[ ${#EXACT_CONTACT_ARGS[@]} -gt 0 ]]; then
  if [[ -z ${CONTACT_GEOMETRY_SOURCE} ]]; then
    printf 'Exact PCA contacts require PLAN10_CONTACT_GEOMETRY_SOURCE\n' >&2
    exit 2
  fi
  EXACT_CONTACT_ARGS+=(
    --side-clamp-contact-geometry-source "${CONTACT_GEOMETRY_SOURCE}"
  )
fi
FIT_ARGS=()
if [[ ${FIT_BOX_TO_REACHABLE_PALMS} == 1 ]]; then
  FIT_ARGS=(--sugar-side-clamp-fit-box-to-reachable-palms)
fi
DIRECT_SETUP_ARGS=()
if [[ ${DIRECT_SETUP_IK} == 1 ]]; then
  DIRECT_SETUP_ARGS=(--sugar-side-clamp-direct-setup-ik)
fi
SIMULTANEOUS_CLOSE_ARGS=()
if [[ ${SIMULTANEOUS_HAND_CLOSE} == 1 ]]; then
  SIMULTANEOUS_CLOSE_ARGS=(--sugar-side-clamp-simultaneous-hand-close)
fi
TRACK_LIVE_BOX_ARGS=()
if [[ ${TRACK_LIVE_BOX_DURING_GRASP} == 1 ]]; then
  TRACK_LIVE_BOX_ARGS=(--ik-track-live-box-during-grasp)
elif [[ ${TRACK_LIVE_BOX_DURING_GRASP} != 0 ]]; then
  printf 'PLAN10_TRACK_LIVE_BOX_DURING_GRASP must be 0 or 1\n' >&2
  exit 2
fi
TRACK_LIVE_LIFT_ARGS=()
if [[ ${TRACK_LIVE_BOX_DURING_LIFT} == 1 ]]; then
  TRACK_LIVE_LIFT_ARGS=(--ik-track-live-box-during-lift)
elif [[ ${TRACK_LIVE_BOX_DURING_LIFT} != 0 ]]; then
  printf 'PLAN10_TRACK_LIVE_BOX_DURING_LIFT must be 0 or 1\n' >&2
  exit 2
fi
if [[ ${LIVE_LIFT_SCHEDULED_WORLD_Z} == 1 ]]; then
  TRACK_LIVE_LIFT_ARGS+=(--ik-live-lift-scheduled-world-z)
elif [[ ${LIVE_LIFT_SCHEDULED_WORLD_Z} != 0 ]]; then
  printf 'PLAN10_LIVE_LIFT_SCHEDULED_WORLD_Z must be 0 or 1\n' >&2
  exit 2
fi
if [[ ${LIVE_LIFT_ANCHOR_WORLD_XY} == 1 ]]; then
  TRACK_LIVE_LIFT_ARGS+=(--ik-live-lift-anchor-world-xy)
elif [[ ${LIVE_LIFT_ANCHOR_WORLD_XY} != 0 ]]; then
  printf 'PLAN10_LIVE_LIFT_ANCHOR_WORLD_XY must be 0 or 1\n' >&2
  exit 2
fi
if [[ ${LIVE_HOLD_RELATIVE_TO_BOX} == 1 ]]; then
  TRACK_LIVE_LIFT_ARGS+=(--ik-live-hold-relative-to-box)
elif [[ ${LIVE_HOLD_RELATIVE_TO_BOX} != 0 ]]; then
  printf 'PLAN10_LIVE_HOLD_RELATIVE_TO_BOX must be 0 or 1\n' >&2
  exit 2
fi
if [[ -n ${LIVE_LIFT_LEAD_M} ]]; then
  TRACK_LIVE_LIFT_ARGS+=(
    --ik-live-lift-lead-m "${LIVE_LIFT_LEAD_M}"
    --ik-live-lift-lead-ramp-steps "${LIVE_LIFT_LEAD_RAMP_STEPS}"
  )
elif [[ ${LIVE_LIFT_LEAD_RAMP_STEPS} != 0 ]]; then
  printf 'PLAN10_LIVE_LIFT_LEAD_RAMP_STEPS requires PLAN10_LIVE_LIFT_LEAD_M\n' >&2
  exit 2
fi
LATE_LIFT_THUMB_ARGS=()
if [[ -n ${LIFT_LEFT_THUMB_YAW_RAD} ]]; then
  LATE_LIFT_THUMB_ARGS+=(
    --sugar-side-clamp-lift-left-thumb-yaw-rad "${LIFT_LEFT_THUMB_YAW_RAD}"
  )
fi
if [[ -n ${LIFT_RIGHT_THUMB_YAW_RAD} ]]; then
  LATE_LIFT_THUMB_ARGS+=(
    --sugar-side-clamp-lift-right-thumb-yaw-rad "${LIFT_RIGHT_THUMB_YAW_RAD}"
  )
fi
if [[ ${#LATE_LIFT_THUMB_ARGS[@]} -gt 0 ]]; then
  LATE_LIFT_THUMB_ARGS+=(
    --sugar-side-clamp-lift-thumb-yaw-start-step "${LIFT_THUMB_YAW_START_STEP}"
    --sugar-side-clamp-lift-thumb-yaw-ramp-steps "${LIFT_THUMB_YAW_RAMP_STEPS}"
  )
fi
FINGER_CLOSE_ARGS=()
if [[ -n ${LEFT_LITTLE_CLOSE_FRACTION} ]]; then
  FINGER_CLOSE_ARGS+=(--left-little-close-fraction "${LEFT_LITTLE_CLOSE_FRACTION}")
fi
if [[ -n ${RIGHT_INDEX_CLOSE_FRACTION} ]]; then
  FINGER_CLOSE_ARGS+=(--right-index-close-fraction "${RIGHT_INDEX_CLOSE_FRACTION}")
fi
if [[ -n ${RIGHT_LITTLE_CLOSE_FRACTION} ]]; then
  FINGER_CLOSE_ARGS+=(--right-little-close-fraction "${RIGHT_LITTLE_CLOSE_FRACTION}")
fi
PALM_INSET_ARGS=()
if [[ -n ${LEFT_PALM_INSET_M} || -n ${RIGHT_PALM_INSET_M} ]]; then
  if [[ -z ${LEFT_PALM_INSET_M} || -z ${RIGHT_PALM_INSET_M} ]]; then
    printf 'Both independent palm insets must be declared together\n' >&2
    exit 2
  fi
  PALM_INSET_ARGS=(
    --left-side-clamp-palm-inset-m "${LEFT_PALM_INSET_M}"
    --right-side-clamp-palm-inset-m "${RIGHT_PALM_INSET_M}"
  )
fi
SHOULDER_ARGS=()
if [[ -n ${LEFT_SHOULDER_ROLL_OFFSET_RAD} || -n ${RIGHT_SHOULDER_ROLL_OFFSET_RAD} ]]; then
  if [[ -z ${LEFT_SHOULDER_ROLL_OFFSET_RAD} || -z ${RIGHT_SHOULDER_ROLL_OFFSET_RAD} ]]; then
    printf 'Both independent shoulder-roll offsets must be declared together\n' >&2
    exit 2
  fi
  SHOULDER_ARGS=(
    --left-shoulder-roll-offset-rad "${LEFT_SHOULDER_ROLL_OFFSET_RAD}"
    --right-shoulder-roll-offset-rad "${RIGHT_SHOULDER_ROLL_OFFSET_RAD}"
  )
fi
WAIST_ARGS=()
if [[ -n ${WAIST_PITCH_ABSOLUTE_RAD} ]]; then
  WAIST_ARGS=(--waist-pitch-absolute-rad "${WAIST_PITCH_ABSOLUTE_RAD}")
fi
printf 'PLAN10_LIVE_TRACK_READBACK grasp=%s lift=%s scheduled_world_z=%s anchor_world_xy=%s hold_relative=%s lead_m=%s lead_ramp_steps=%s lift_args=%s\n' \
  "${TRACK_LIVE_BOX_DURING_GRASP}" \
  "${TRACK_LIVE_BOX_DURING_LIFT}" \
  "${LIVE_LIFT_SCHEDULED_WORLD_Z}" \
  "${LIVE_LIFT_ANCHOR_WORLD_XY}" \
  "${LIVE_HOLD_RELATIVE_TO_BOX}" \
  "${LIVE_LIFT_LEAD_M:-default}" \
  "${LIVE_LIFT_LEAD_RAMP_STEPS}" \
  "${TRACK_LIVE_LIFT_ARGS[*]:-none}"
"${PYTHON_BIN}" scripts/sugar/run_plan10_g1_inspire_carrybox_reachability_v17.py \
  --unitree-repo "${UNITREE_REPO}" \
  --asset-root "${ASSET_ROOT}" \
  --robot-motion "${ROOT}/SUGAR/data/CarryBox/data_045/robot_50hz.npz" \
  --object-motion "${ROOT}/SUGAR/data/CarryBox/data_045/obj_motion_global_50hz.pkl" \
  --box-usd "${ROOT}/SUGAR/descriptions/objects/small_box/obj_aligned.usd" \
  --output-root "${OUTPUT_ROOT}" \
  --body-control-mode sugar_side_clamp \
  --fix-robot-root \
  --robot-root-y-offset-m "${ROBOT_ROOT_Y_OFFSET_M}" \
  "${WAIST_ARGS[@]}" \
  --bilateral-shoulder-roll-offset-rad "${BILATERAL_SHOULDER_ROLL_OFFSET_RAD}" \
  --bilateral-hip-roll-outward-offset-rad "${BILATERAL_HIP_ROLL_OUTWARD_OFFSET_RAD}" \
  "${SHOULDER_ARGS[@]}" \
  --source-start 269 \
  --source-end 350 \
  --close-start 269.25 \
  --close-end 269.75 \
  --side-clamp-box-axis "${BOX_AXIS}" \
  --side-clamp-box-local-tangent-m -0.12 \
  --left-side-clamp-box-local-tangent-m "${LEFT_TANGENT_M}" \
  --right-side-clamp-box-local-tangent-m "${RIGHT_TANGENT_M}" \
  --left-side-clamp-box-local-z-m "${LEFT_LOCAL_Z_M}" \
  --right-side-clamp-box-local-z-m "${RIGHT_LOCAL_Z_M}" \
  --left-side-clamp-tilt-tangent-rad "${LEFT_TILT_TANGENT_RAD}" \
  --right-side-clamp-tilt-tangent-rad "${RIGHT_TILT_TANGENT_RAD}" \
  --left-side-clamp-tilt-height-rad "${LEFT_TILT_HEIGHT_RAD}" \
  --right-side-clamp-tilt-height-rad "${RIGHT_TILT_HEIGHT_RAD}" \
  --left-side-clamp-normal-roll-rad "${LEFT_NORMAL_ROLL_RAD}" \
  --right-side-clamp-normal-roll-rad "${RIGHT_NORMAL_ROLL_RAD}" \
  "${NORMAL_ARGS[@]}" \
  "${EXACT_CONTACT_ARGS[@]}" \
  "${FIT_ARGS[@]}" \
  "${DIRECT_SETUP_ARGS[@]}" \
  "${SIMULTANEOUS_CLOSE_ARGS[@]}" \
  --sugar-side-clamp-direct-refinement-steps "${DIRECT_REFINEMENT_STEPS}" \
  --sugar-side-clamp-max-reachable-orientation-delta-rad "${MAX_REACHABLE_ORIENTATION_DELTA_RAD}" \
  --sugar-side-clamp-box-local-offset-m "${BOX_LOCAL_X_OFFSET_M}" "${BOX_LOCAL_Y_OFFSET_M}" "${BOX_LOCAL_Z_OFFSET_M}" \
  --unitree-demo-side-clamp-box-local-z-m -0.04 \
  --unitree-demo-side-clamp-palm-inset-m "${PALM_INSET_M}" \
  "${PALM_INSET_ARGS[@]}" \
  --unitree-demo-side-clamp-approach-clearance-m "${APPROACH_CLEARANCE_M}" \
  --unitree-demo-approach-steps 800 \
  "${TRACK_LIVE_BOX_ARGS[@]}" \
  "${TRACK_LIVE_LIFT_ARGS[@]}" \
  "${LATE_LIFT_THUMB_ARGS[@]}" \
  --ik-palm-press-steps "${PALM_PRESS_STEPS}" \
  --unitree-demo-close-steps "${CLOSE_STEPS}" \
  --ik-thumb-close-steps "${THUMB_CLOSE_STEPS}" \
  --unitree-demo-settle-steps "${CONTACT_SETTLE_STEPS}" \
  --require-settled-all-groups-frames "${REQUIRE_SETTLED_ALL_GROUPS_FRAMES}" \
  --ik-contact-preload-height-m "${CONTACT_PRELOAD_HEIGHT_M}" \
  --ik-contact-preload-steps "${CONTACT_PRELOAD_STEPS}" \
  --ik-contact-preload-ramp-steps "${CONTACT_PRELOAD_RAMP_STEPS}" \
  --ik-contact-compression-m "${CONTACT_COMPRESSION_M}" \
  --ik-contact-compression-steps "${CONTACT_COMPRESSION_STEPS}" \
  --ik-contact-compression-ramp-steps "${CONTACT_COMPRESSION_RAMP_STEPS}" \
  --ik-lift-height-m "${LIFT_HEIGHT_M}" \
  --ik-lift-steps "${LIFT_STEPS}" \
  --close-fraction "${CLOSE_FRACTION}" \
  "${FINGER_CLOSE_ARGS[@]}" \
  --pregrasp-thumb-pitch-rad "${PREGRASP_THUMB_PITCH_RAD}" \
  --pregrasp-thumb-yaw-rad "${PREGRASP_THUMB_YAW_RAD}" \
  --pregrasp-left-thumb-pitch-rad "${LEFT_PREGRASP_THUMB_PITCH_RAD}" \
  --pregrasp-right-thumb-pitch-rad "${RIGHT_PREGRASP_THUMB_PITCH_RAD}" \
  --pregrasp-left-thumb-yaw-rad "${LEFT_PREGRASP_THUMB_YAW_RAD}" \
  --pregrasp-right-thumb-yaw-rad "${RIGHT_PREGRASP_THUMB_YAW_RAD}" \
  --closed-left-thumb-pitch-rad "${LEFT_CLOSED_THUMB_PITCH_RAD}" \
  --closed-right-thumb-pitch-rad "${RIGHT_CLOSED_THUMB_PITCH_RAD}" \
  --closed-left-thumb-yaw-rad "${LEFT_CLOSED_THUMB_YAW_RAD}" \
  --closed-right-thumb-yaw-rad "${RIGHT_CLOSED_THUMB_YAW_RAD}" \
  --zero-initial-object-velocity \
  --object-static-friction "${OBJECT_STATIC_FRICTION}" \
  --object-dynamic-friction "${OBJECT_DYNAMIC_FRICTION}" \
  --sugar-side-clamp-support-height-m "${SUPPORT_HEIGHT_M}" \
  --sugar-side-clamp-support-size-xy-m "${SUPPORT_SIZE_X_M}" "${SUPPORT_SIZE_Y_M}" \
  --physics-substeps-per-source 4 \
  --solver-position-iterations 16 \
  --solver-velocity-iterations 4 \
  --hold-steps "${HOLD_STEPS}" \
  --contact-threshold-n 0.01 \
  --headless \
  --device cuda:0 \
  --kit_args "--portable-root ${PORTABLE_ROOT} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"
