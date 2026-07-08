#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

cd /public/home/yanhongru/Curiosity
hostname

STAMP="${STAMP:-20260705_core_world_g1_payload_sweep}"
ROOT_OUTPUT="experiments/outputs/core_world_g1_payload_sweep/${STAMP}"
LOG_DIR="logs/core_world_g1_payload_sweep"
mkdir -p "${ROOT_OUTPUT}" "${LOG_DIR}"

HEIGHTS=(${HEIGHTS:-0.84 0.96})
MASSES=(${MASSES:-0.25 0.50 1.00 2.00})
ATTACH_XS=(${ATTACH_XS:-0.12 0.18 0.24})
ATTACH_Z="${ATTACH_Z:-0.08}"
STEPS="${STEPS:-240}"
BOX_COLLISION_ENABLED="${BOX_COLLISION_ENABLED:-1}"

for height in "${HEIGHTS[@]}"; do
  for mass in "${MASSES[@]}"; do
    for attach_x in "${ATTACH_XS[@]}"; do
      case_name="z_${height/./p}_m_${mass/./p}_x_${attach_x/./p}"
      output_dir="${ROOT_OUTPUT}/${case_name}"
      box_pos_x="${BOX_POS_X:-${attach_x}}"
      box_pos_y="${BOX_POS_Y:-0.0}"
      box_pos_z="${BOX_POS_Z:-$(awk -v h="${height}" -v z="${ATTACH_Z}" 'BEGIN { printf "%.6f", h + z }')}"
      mkdir -p "${output_dir}"
      cat > "${output_dir}/case_config.json" <<EOF
{
  "height_m": ${height},
  "box_mass_kg": ${mass},
  "attach_local_pos0_x_m": ${attach_x},
  "attach_local_pos0_y_m": 0.0,
  "attach_local_pos0_z_m": ${ATTACH_Z},
  "attach_box": "fixed_torso",
  "box_collision_enabled": ${BOX_COLLISION_ENABLED},
  "box_position_requested_m": [${box_pos_x}, ${box_pos_y}, ${box_pos_z}]
}
EOF
      echo "[CASE] height=${height} mass=${mass} attach_x=${attach_x} output=${output_dir}"
      set +e
      STAMP="${STAMP}_${case_name}" \
      OUTPUT_DIR="${output_dir}" \
      LOG_DIR="${LOG_DIR}" \
      STEPS="${STEPS}" \
      G1_ROOT_Z="${height}" \
      BOX_MASS="${mass}" \
      BOX_POS_X="${box_pos_x}" \
      BOX_POS_Y="${box_pos_y}" \
      BOX_POS_Z="${box_pos_z}" \
      ATTACH_BOX="fixed_torso" \
      ATTACH_BODY_PATH="${ATTACH_BODY_PATH:-/World/G1/torso_link}" \
      ATTACH_LOCAL_POS0_X="${attach_x}" \
      ATTACH_LOCAL_POS0_Y="0.0" \
      ATTACH_LOCAL_POS0_Z="${ATTACH_Z}" \
      BOX_COLLISION_ENABLED="${BOX_COLLISION_ENABLED}" \
      APPLY_ARENA_STAND_GAINS=1 \
      STAND_GAIN_SCALE="${STAND_GAIN_SCALE:-1.0}" \
      bash scripts/isaac/run_core_world_g1_box_scene.sh \
        2>&1 | tee "${LOG_DIR}/${STAMP}_${case_name}.log"
      run_status=${PIPESTATUS[0]}
      set -e
      echo "${run_status}" > "${output_dir}/run_status.txt"
      if [[ "${run_status}" -ne 0 ]]; then
        echo "[WARN] G1 payload case ${case_name} runner exited ${run_status}; preserving result and continuing." >&2
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
        --expect-attach-box fixed_torso \
        --min-joint-count 29 \
        --require-stand-drive-gains \
        --min-stand-drive-gain-count 20 \
        --max-fall-events 0 \
        --max-box-drop-events 0 \
        --min-robot-z "${MIN_ROBOT_Z:-0.45}" \
        --min-box-z "${MIN_BOX_Z:-0.35}" \
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
        echo "[WARN] G1 payload case ${case_name} strict checker failed; continuing." >&2
      fi
    done
  done
done

python3 scripts/isaac/summarize_core_world_g1_payload_sweep.py \
  --root "${ROOT_OUTPUT}" \
  --output "${ROOT_OUTPUT}/core_world_g1_payload_sweep_summary.json"

echo "[INFO] G1 payload sweep output: ${ROOT_OUTPUT}"
