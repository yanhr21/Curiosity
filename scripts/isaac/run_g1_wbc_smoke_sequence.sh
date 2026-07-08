#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${ROOT_DIR}/experiments/outputs/minimal_carry_scene}"
STAND_STEPS="${STAND_STEPS:-240}"
WALK_STEPS="${WALK_STEPS:-400}"
PAYLOAD_STEPS="${PAYLOAD_STEPS:-400}"
DEVICE="${DEVICE:-cuda:0}"
RUN_PAYLOAD="${RUN_PAYLOAD:-1}"
CHECK_SUMMARY="${CHECK_SUMMARY:-1}"
CHECKER_PYTHON="${CHECKER_PYTHON:-/usr/local/python3.12/bin/python3}"

cd "${ROOT_DIR}"

check_summary() {
  local mode="$1"
  local output_dir="$2"
  local min_steps="$3"
  local expected_wbc_mode="walk"
  local expected_attach_box="none"
  if [[ "${CHECK_SUMMARY}" != "1" ]]; then
    return 0
  fi
  if [[ "${mode}" == "stand" ]]; then
    expected_wbc_mode="stand"
  fi
  if [[ "${mode}" == "payload" ]]; then
    expected_wbc_mode="walk"
    expected_attach_box="fixed_torso"
  fi
  "${CHECKER_PYTHON}" scripts/isaac/check_carry_smoke_summary.py \
    "${output_dir}/minimal_carry_scene_summary.json" \
    --mode "${mode}" \
    --min-steps "${min_steps}" \
    --min-joint-count 20 \
    --expect-wbc-mode "${expected_wbc_mode}" \
    --expect-attach-box "${expected_attach_box}" \
    --max-root-pose-writes-rollout 0 \
    --max-root-velocity-writes-rollout 0 \
    --max-box-pose-writes-rollout 0
}

echo "[SEQUENCE] G1 WBC stand smoke"
STAND_OUTPUT_DIR="${BASE_OUTPUT_DIR}/g1_wbc_stand_${STAMP}_${STAND_STEPS}steps"
DEVICE="${DEVICE}" SKIP_ROBOT=0 WBC_MODE=stand RENDER=0 STEPS="${STAND_STEPS}" \
OUTPUT_DIR="${STAND_OUTPUT_DIR}" \
bash scripts/isaac/run_minimal_carry_scene.sh
check_summary stand "${STAND_OUTPUT_DIR}" "${STAND_STEPS}"

echo "[SEQUENCE] G1 WBC walk smoke"
WALK_OUTPUT_DIR="${BASE_OUTPUT_DIR}/g1_wbc_walk_${STAMP}_${WALK_STEPS}steps"
DEVICE="${DEVICE}" SKIP_ROBOT=0 WBC_MODE=walk WALK_COMMAND_X=0.15 WALK_COMMAND_Y=0.0 WALK_COMMAND_YAW=0.0 \
RENDER=0 STEPS="${WALK_STEPS}" \
OUTPUT_DIR="${WALK_OUTPUT_DIR}" \
bash scripts/isaac/run_minimal_carry_scene.sh
check_summary walk "${WALK_OUTPUT_DIR}" "${WALK_STEPS}"

if [[ "${RUN_PAYLOAD}" == "1" ]]; then
  echo "[SEQUENCE] G1 WBC fixed payload balance diagnostic"
  PAYLOAD_OUTPUT_DIR="${BASE_OUTPUT_DIR}/g1_wbc_walk_fixed_payload_${STAMP}_${PAYLOAD_STEPS}steps"
  DEVICE="${DEVICE}" SKIP_ROBOT=0 WBC_MODE=walk WALK_COMMAND_X=0.10 WALK_COMMAND_Y=0.0 WALK_COMMAND_YAW=0.0 \
  ATTACH_BOX=fixed_torso BOX_MASS="${BOX_MASS:-5.0}" RENDER=0 STEPS="${PAYLOAD_STEPS}" \
  OUTPUT_DIR="${PAYLOAD_OUTPUT_DIR}" \
  bash scripts/isaac/run_minimal_carry_scene.sh
  check_summary payload "${PAYLOAD_OUTPUT_DIR}" "${PAYLOAD_STEPS}"
else
  echo "[SEQUENCE] Skipping fixed payload diagnostic because RUN_PAYLOAD=${RUN_PAYLOAD}"
fi
