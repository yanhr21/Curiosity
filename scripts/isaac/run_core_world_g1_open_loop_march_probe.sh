#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

cd /public/home/yanhongru/Curiosity
hostname

STAMP="${STAMP:-20260705_core_world_g1_open_loop_march_probe}"
ROOT_OUTPUT="experiments/outputs/core_world_g1_open_loop_march_probe/${STAMP}"
LOG_DIR="logs/core_world_g1_open_loop_march_probe"
mkdir -p "${ROOT_OUTPUT}" "${LOG_DIR}"

HEIGHTS=(${HEIGHTS:-0.84 0.96})
AMPLITUDES=(${AMPLITUDES:-0.10 0.20 0.30})
ATTACH_BOX_MODE="${ATTACH_BOX_MODE:-none}"
EXPECT_ATTACH_BOX="${EXPECT_ATTACH_BOX:-${ATTACH_BOX_MODE}}"
MAX_BOX_DROP_EVENTS="${MAX_BOX_DROP_EVENTS:-999}"
MIN_BOX_Z="${MIN_BOX_Z:-0.0}"
STEPS="${STEPS:-360}"

for height in "${HEIGHTS[@]}"; do
  for amplitude in "${AMPLITUDES[@]}"; do
    case_name="z_${height/./p}_amp_${amplitude/./p}"
    output_dir="${ROOT_OUTPUT}/${case_name}"
    mkdir -p "${output_dir}"
    echo "[CASE] height=${height} amplitude=${amplitude} attach=${ATTACH_BOX_MODE} output=${output_dir}"
    set +e
    STAMP="${STAMP}_${case_name}" \
    OUTPUT_DIR="${output_dir}" \
    LOG_DIR="${LOG_DIR}" \
    STEPS="${STEPS}" \
    G1_ROOT_Z="${height}" \
    BOX_MASS="${BOX_MASS:-0.5}" \
    BOX_POS_X="${BOX_POS_X:-0.40}" \
    BOX_POS_Y="${BOX_POS_Y:-0.0}" \
    BOX_POS_Z="${BOX_POS_Z:-0.88}" \
    ATTACH_BOX="${ATTACH_BOX_MODE}" \
    ATTACH_BODY_PATH="${ATTACH_BODY_PATH:-/World/G1/torso_link}" \
    ATTACH_LOCAL_POS0_X="${ATTACH_LOCAL_POS0_X:-0.18}" \
    ATTACH_LOCAL_POS0_Y="${ATTACH_LOCAL_POS0_Y:-0.0}" \
    ATTACH_LOCAL_POS0_Z="${ATTACH_LOCAL_POS0_Z:-0.08}" \
    APPLY_ARENA_STAND_GAINS=1 \
    STAND_GAIN_SCALE="${STAND_GAIN_SCALE:-1.0}" \
    GAIT_MODE=open_loop_march \
    GAIT_AMPLITUDE="${amplitude}" \
    GAIT_FREQUENCY_HZ="${GAIT_FREQUENCY_HZ:-0.7}" \
    bash scripts/isaac/run_core_world_g1_box_scene.sh \
      2>&1 | tee "${LOG_DIR}/${STAMP}_${case_name}.log"
    run_status=${PIPESTATUS[0]}
    set -e
    echo "${run_status}" > "${output_dir}/run_status.txt"
    if [[ "${run_status}" -ne 0 ]]; then
      echo "[WARN] G1 march case ${case_name} runner exited ${run_status}; preserving result and continuing." >&2
    fi
    summary="${output_dir}/core_world_g1_box_scene_summary.json"
    if [[ ! -f "${summary}" ]]; then
      echo "[WARN] Missing summary for ${case_name}: ${summary}" >&2
      continue
    fi
    set +e
    python3 scripts/isaac/check_core_world_g1_box_scene_summary.py \
      "${summary}" \
      --min-steps "${STEPS}" \
      --expect-attach-box "${EXPECT_ATTACH_BOX}" \
      --min-joint-count 29 \
      --require-stand-drive-gains \
      --min-stand-drive-gain-count 20 \
      --max-fall-events 0 \
      --max-box-drop-events "${MAX_BOX_DROP_EVENTS}" \
      --min-robot-z "${MIN_ROBOT_Z:-0.45}" \
      --min-box-z "${MIN_BOX_Z}" \
      --max-tilt "${MAX_TILT:-0.85}" \
      --max-root-pose-write-count-rollout 0 \
      --max-root-velocity-write-count-rollout 0 \
      --max-box-pose-write-count-rollout 0 \
      --require-diagnostic-claim \
      > "${output_dir}/strict_check_report.json"
    check_status=$?
    set -e
    echo "${check_status}" > "${output_dir}/check_status.txt"
    if [[ "${check_status}" -ne 0 ]]; then
      echo "[WARN] G1 march case ${case_name} strict checker failed; continuing." >&2
    fi
  done
done

python3 scripts/isaac/summarize_core_world_g1_stand_height_sweep.py \
  --root "${ROOT_OUTPUT}" \
  --output "${ROOT_OUTPUT}/core_world_g1_open_loop_march_probe_summary.json"

echo "[INFO] G1 open-loop march probe output: ${ROOT_OUTPUT}"
