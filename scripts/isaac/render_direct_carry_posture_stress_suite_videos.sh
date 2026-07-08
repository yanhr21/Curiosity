#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to generate videos on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUITE_STAMP="${SUITE_STAMP:-20260706_direct_carry_posture_stress_suite_64cm_8kg}"
VIS_DIR="${ROOT_DIR}/experiments/visuals/direct_carry_posture_suite/${SUITE_STAMP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/isaac_arena_py312/bin/python}"
mkdir -p "${VIS_DIR}"
cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m py_compile scripts/isaac/render_prismatic_carrier_csv_video.py

render_case() {
  local posture="$1"
  local case_dir="experiments/outputs/direct_carry_task_physical_backend/${SUITE_STAMP}_${posture}"
  local csv="${case_dir}/backend_anchored_cradle/core_world_anchored_footstep_carrier_state.csv"
  local summary="${case_dir}/backend_anchored_cradle/core_world_anchored_footstep_carrier_summary.json"
  local output="${VIS_DIR}/${SUITE_STAMP}_${posture}.mp4"

  if [[ ! -f "${csv}" ]]; then
    echo "[ERROR] Missing CSV for ${posture}: ${csv}" >&2
    exit 4
  fi
  if [[ ! -f "${summary}" ]]; then
    echo "[ERROR] Missing summary for ${posture}: ${summary}" >&2
    exit 5
  fi

  "${PYTHON_BIN}" scripts/isaac/render_prismatic_carrier_csv_video.py \
    --csv "${csv}" \
    --summary "${summary}" \
    --output "${output}" \
    --fps 20 \
    --stride 10 \
    --width 1280 \
    --height 720
  test -s "${output}"
  echo "[VIDEO] ${posture} ${output}"
}

render_case front_mid
render_case low_front
render_case chest_high
render_case front_reach
render_case close_mid

{
  echo "suite_stamp=${SUITE_STAMP}"
  echo "visual_dir=${VIS_DIR}"
  find "${VIS_DIR}" -maxdepth 1 -type f -name '*.mp4' -printf '%f %s bytes\n' | sort
} > "${VIS_DIR}/render_manifest.txt"

cat "${VIS_DIR}/render_manifest.txt"
