#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
OUT_DIR="${ROOT_DIR}/experiments/outputs/phase00/ref_tactile/envprep/availability"
REPORT_DIR="${ROOT_DIR}/experiments/reports/phase00/ref_tactile/envprep"
mkdir -p "${OUT_DIR}" "${REPORT_DIR}"

STATUS_JSON="${OUT_DIR}/reference_env_availability_status.json"
REPORT_MD="${REPORT_DIR}/reference_env_availability.md"

UNIVTAC_CONDA="${ROOT_DIR}/envs/univtac/conda/bin/python"
UNIVTAC_VENV="${ROOT_DIR}/envs/univtac/.venv/bin/python"
TACAUCHY_CONDA="${ROOT_DIR}/envs/tacauchy/conda/bin/python"
TACAUCHY_VENV="${ROOT_DIR}/envs/tacauchy/.venv/bin/python"
UNIVTAC_SELECTED="${UNIVTAC_PYTHON:-${UNIVTAC_CONDA}}"
TACAUCHY_SELECTED="${TACAUCHY_PYTHON:-${TACAUCHY_CONDA}}"

check_executable() {
  local path="$1"
  if [[ -x "${path}" ]]; then
    printf "present"
  else
    printf "missing"
  fi
}

UNIVTAC_SELECTED_STATUS="$(check_executable "${UNIVTAC_SELECTED}")"
UNIVTAC_CONDA_STATUS="$(check_executable "${UNIVTAC_CONDA}")"
UNIVTAC_VENV_STATUS="$(check_executable "${UNIVTAC_VENV}")"
TACAUCHY_SELECTED_STATUS="$(check_executable "${TACAUCHY_SELECTED}")"
TACAUCHY_CONDA_STATUS="$(check_executable "${TACAUCHY_CONDA}")"
TACAUCHY_VENV_STATUS="$(check_executable "${TACAUCHY_VENV}")"
GIT_LFS_STATUS="$(command -v git-lfs >/dev/null 2>&1 && printf present || printf missing)"
CMAKE_STATUS="$(command -v cmake >/dev/null 2>&1 && printf present || printf missing)"
NVCC_STATUS="$(command -v nvcc >/dev/null 2>&1 && printf present || printf missing)"
UNIVTAC_READY="false"
TACAUCHY_READY="false"
if [[ "${UNIVTAC_SELECTED_STATUS}" == "present" || "${UNIVTAC_CONDA_STATUS}" == "present" || "${UNIVTAC_VENV_STATUS}" == "present" ]]; then
  UNIVTAC_READY="true"
fi
if [[ "${TACAUCHY_SELECTED_STATUS}" == "present" || "${TACAUCHY_CONDA_STATUS}" == "present" || "${TACAUCHY_VENV_STATUS}" == "present" ]]; then
  TACAUCHY_READY="true"
fi
GATE_00F_READY="false"
if [[ "${UNIVTAC_READY}" == "true" && "${TACAUCHY_READY}" == "true" ]]; then
  GATE_00F_READY="candidate_envs_present_pending_compute_sanity"
fi

cat > "${STATUS_JSON}" <<JSON
{
  "schema_version": "phase00_reference_env_availability_v1",
  "date": "2026-07-01",
  "scope": "lightweight_file_and_executable_checks_only",
  "login_node_compute_policy": "no_sim_no_render_no_training_no_dependency_install",
  "univtac": {
    "selected_python": "${UNIVTAC_SELECTED}",
    "selected_status": "${UNIVTAC_SELECTED_STATUS}",
    "conda_python": "${UNIVTAC_CONDA}",
    "conda_status": "${UNIVTAC_CONDA_STATUS}",
    "venv_python": "${UNIVTAC_VENV}",
    "venv_status": "${UNIVTAC_VENV_STATUS}",
    "candidate_environment_present": ${UNIVTAC_READY}
  },
  "tacauchy": {
    "selected_python": "${TACAUCHY_SELECTED}",
    "selected_status": "${TACAUCHY_SELECTED_STATUS}",
    "conda_python": "${TACAUCHY_CONDA}",
    "conda_status": "${TACAUCHY_CONDA_STATUS}",
    "venv_python": "${TACAUCHY_VENV}",
    "venv_status": "${TACAUCHY_VENV_STATUS}",
    "candidate_environment_present": ${TACAUCHY_READY}
  },
  "toolchain_on_path": {
    "git_lfs": "${GIT_LFS_STATUS}",
    "cmake": "${CMAKE_STATUS}",
    "nvcc": "${NVCC_STATUS}"
  },
  "gate_00f_ready": "${GATE_00F_READY}",
  "curiosity_training_allowed": false
}
JSON

cat > "${REPORT_MD}" <<MD
# Reference Env Availability

Date: 2026-07-01

Scope: lightweight file and executable checks only. This script does not run
simulation, rendering, training, dependency installation, package import,
official demos, model loading, or dataset conversion.

## Checked Executables

- UniVTAC selected Python: \`${UNIVTAC_SELECTED}\`
  - status: \`${UNIVTAC_SELECTED_STATUS}\`
- UniVTAC conda Python: \`${UNIVTAC_CONDA}\`
  - status: \`${UNIVTAC_CONDA_STATUS}\`
- UniVTAC venv Python: \`${UNIVTAC_VENV}\`
  - status: \`${UNIVTAC_VENV_STATUS}\`
- TaCauchy selected Python: \`${TACAUCHY_SELECTED}\`
  - status: \`${TACAUCHY_SELECTED_STATUS}\`
- TaCauchy conda Python: \`${TACAUCHY_CONDA}\`
  - status: \`${TACAUCHY_CONDA_STATUS}\`
- TaCauchy venv Python: \`${TACAUCHY_VENV}\`
  - status: \`${TACAUCHY_VENV_STATUS}\`

## Toolchain On PATH

- \`git-lfs\`: \`${GIT_LFS_STATUS}\`
- \`cmake\`: \`${CMAKE_STATUS}\`
- \`nvcc\`: \`${NVCC_STATUS}\`

## Interpretation

Gate 00F is not ready unless both official reference Python executables are
present and later pass their compute-side official sanity probes. This
availability check is only a preflight guard; it is not official reference
sanity and not curiosity progress.
MD

printf '%s\n' "${STATUS_JSON}"
printf '%s\n' "${REPORT_MD}"
