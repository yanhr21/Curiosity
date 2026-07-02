#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
OUT_DIR="${ROOT}/experiments/outputs/phase00/ref_tactile/envprep/gate00f_readiness"
REPORT_DIR="${ROOT}/experiments/reports/phase00/ref_tactile/envprep"
STATUS_JSON="${OUT_DIR}/gate00f_readiness_status.json"
REPORT_MD="${REPORT_DIR}/gate00f_readiness.md"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Gate00F readiness check must run on login/lightweight context, not inside Slurm allocation." >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$REPORT_DIR"

host="$(hostname)"
date_str="$(date +%Y-%m-%d)"

univtac_python="${UNIVTAC_PYTHON:-$ROOT/envs/univtac/conda/bin/python}"
tacauchy_python="${TACAUCHY_PYTHON:-$ROOT/envs/tacauchy/conda/bin/python}"
project_conda="$ROOT/envs/taccel/miniforge/bin/conda"
project_nvcc="$ROOT/envs/taccel/cuda-toolkit/bin/nvcc"

tacauchy_asset_root="$ROOT/external/TaCauchy/source/tacex_assets/tacex_assets/data"
univtac_asset_root="$ROOT/external/UniVTAC/third_party/TacEx/source/tacex_assets/tacex_assets/data"
tacauchy_gsmini_sensor="$tacauchy_asset_root/Sensors/GelSight_Mini/Sensor.usd"
univtac_gsmini_sensor="$univtac_asset_root/Sensors/GelSight_Mini/Sensor.usd"

gate_summary="${GATE_SUMMARY:-$ROOT/experiments/outputs/phase00/ref_tactile/gate_review/p00_gate_d58_marker_v1_20260701_071843/phase00_gate_review_summary.json}"

status_exec() {
  local path="$1"
  if [[ -x "$path" ]]; then
    printf present
  else
    printf missing
  fi
}

status_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    printf present
  else
    printf missing
  fi
}

command_status() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    printf present
  else
    printf missing
  fi
}

count_usd() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    find "$dir" -maxdepth 1 -type f -name '*.usd' 2>/dev/null | wc -l | tr -d ' '
  else
    printf 0
  fi
}

size_human() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null | awk '{print $1}'
  else
    printf missing
  fi
}

json_string() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

univtac_status="$(status_exec "$univtac_python")"
tacauchy_status="$(status_exec "$tacauchy_python")"
conda_status="$(status_exec "$project_conda")"
project_nvcc_status="$(status_exec "$project_nvcc")"
git_lfs_status="$(command_status git-lfs)"
cmake_status="$(command_status cmake)"
path_nvcc_status="$(command_status nvcc)"
nvidia_smi_status="$(command_status nvidia-smi)"

tacauchy_gsmini_status="$(status_file "$tacauchy_gsmini_sensor")"
univtac_gsmini_status="$(status_file "$univtac_gsmini_sensor")"
tacauchy_shape_count="$(count_usd "$tacauchy_asset_root/Props/tactile_test_shapes")"
univtac_shape_count="$(count_usd "$univtac_asset_root/Props/tactile_test_shapes")"
tacauchy_asset_size="$(size_human "$tacauchy_asset_root")"
univtac_asset_size="$(size_human "$univtac_asset_root")"

gate_status="missing"
gate_failed_checks="[]"
if [[ -f "$gate_summary" ]] && command -v jq >/dev/null 2>&1; then
  gate_status="$(jq -r '.status // "missing_status"' "$gate_summary")"
  gate_failed_checks="$(jq -c '.failed_checks // []' "$gate_summary")"
fi

env_ready=false
asset_ready=false
official_sanity_ready=false
if [[ "$univtac_status" == present && "$tacauchy_status" == present ]]; then
  env_ready=true
fi
if [[ "$tacauchy_gsmini_status" == present && "$tacauchy_shape_count" -gt 0 ]]; then
  asset_ready=true
fi
effective_failed_checks="$gate_failed_checks"
if command -v jq >/dev/null 2>&1; then
  if [[ "$asset_ready" == true && "$env_ready" == true ]]; then
    effective_failed_checks="$(printf '%s' "$gate_failed_checks" | jq -c 'map(select(. != "reference_asset_availability" and . != "reference_env_availability"))')"
  elif [[ "$asset_ready" == true ]]; then
    effective_failed_checks="$(printf '%s' "$gate_failed_checks" | jq -c 'map(select(. != "reference_asset_availability"))')"
  elif [[ "$env_ready" == true ]]; then
    effective_failed_checks="$(printf '%s' "$gate_failed_checks" | jq -c 'map(select(. != "reference_env_availability"))')"
  fi
fi
if [[ "$gate_status" == "pass_curiosity_ready" || "$gate_status" == "closed_curiosity_ready" ]]; then
  official_sanity_ready=true
fi

gate00f_ready=false
reason="blocked"
if [[ "$env_ready" == true && "$asset_ready" == true && "$official_sanity_ready" == true ]]; then
  gate00f_ready=true
  reason="ready_after_gate_review"
elif [[ "$env_ready" == false ]]; then
  reason="blocked_missing_target_reference_envs"
elif [[ "$asset_ready" == false ]]; then
  reason="blocked_missing_tacauchy_assets"
else
  reason="blocked_official_sanity_or_gate_review_not_passed"
fi

cat >"$STATUS_JSON" <<JSON
{
  "schema_version": "phase00_gate00f_readiness_check_v1",
  "date": "$(json_string "$date_str")",
  "host": "$(json_string "$host")",
  "classification": "lightweight_gate00f_readiness_check_not_training_not_official_sanity",
  "curiosity_training_allowed": false,
  "gate00f_ready": $gate00f_ready,
  "reason": "$(json_string "$reason")",
  "target_envs": {
    "univtac_python": "$(json_string "$univtac_python")",
    "univtac_status": "$(json_string "$univtac_status")",
    "tacauchy_python": "$(json_string "$tacauchy_python")",
    "tacauchy_status": "$(json_string "$tacauchy_status")"
  },
  "toolchain": {
    "project_conda": "$(json_string "$project_conda")",
    "project_conda_status": "$(json_string "$conda_status")",
    "project_nvcc": "$(json_string "$project_nvcc")",
    "project_nvcc_status": "$(json_string "$project_nvcc_status")",
    "git_lfs_on_path": "$(json_string "$git_lfs_status")",
    "cmake_on_path": "$(json_string "$cmake_status")",
    "nvcc_on_path": "$(json_string "$path_nvcc_status")",
    "nvidia_smi_on_path": "$(json_string "$nvidia_smi_status")"
  },
  "assets": {
    "tacauchy_asset_root": "$(json_string "$tacauchy_asset_root")",
    "tacauchy_asset_size": "$(json_string "$tacauchy_asset_size")",
    "tacauchy_gelsight_mini_sensor_usd": "$(json_string "$tacauchy_gsmini_sensor")",
    "tacauchy_gelsight_mini_sensor_status": "$(json_string "$tacauchy_gsmini_status")",
    "tacauchy_tactile_test_shape_usd_count": $tacauchy_shape_count,
    "univtac_asset_root": "$(json_string "$univtac_asset_root")",
    "univtac_asset_size": "$(json_string "$univtac_asset_size")",
    "univtac_gelsight_mini_sensor_usd": "$(json_string "$univtac_gsmini_sensor")",
    "univtac_gelsight_mini_sensor_status": "$(json_string "$univtac_gsmini_status")",
    "univtac_tactile_test_shape_usd_count": $univtac_shape_count
  },
  "latest_gate_review": {
    "summary": "$(json_string "$gate_summary")",
    "status": "$(json_string "$gate_status")",
    "raw_failed_checks": $gate_failed_checks,
    "effective_failed_checks_after_file_presence": $effective_failed_checks
  },
  "not_claims": [
    "not official UniVTAC sanity",
    "not official TaCauchy sanity",
    "not Gate 00F completion",
    "not curiosity training readiness"
  ]
}
JSON

cat >"$REPORT_MD" <<MD
# Gate 00F Readiness

Date: \`$date_str\`

This is a lightweight readiness check only. It does not import packages, install
dependencies, run simulation, run official demos, submit Slurm work, or claim
curiosity readiness.

## Result

- gate00f_ready: \`$gate00f_ready\`
- reason: \`$reason\`
- latest Gate summary: \`$gate_summary\`
- latest Gate status: \`$gate_status\`
- latest Gate raw failed checks: \`$gate_failed_checks\`
- effective failed checks after current file-presence audit: \`$effective_failed_checks\`

## Target Envs

- UniVTAC Python: \`$univtac_python\` -> \`$univtac_status\`
- TaCauchy Python: \`$tacauchy_python\` -> \`$tacauchy_status\`

## Toolchain

- project conda: \`$project_conda\` -> \`$conda_status\`
- project nvcc: \`$project_nvcc\` -> \`$project_nvcc_status\`
- \`git-lfs\` on PATH: \`$git_lfs_status\`
- \`cmake\` on PATH: \`$cmake_status\`
- \`nvcc\` on PATH: \`$path_nvcc_status\`
- \`nvidia-smi\` on PATH: \`$nvidia_smi_status\`

## Assets

- TaCauchy asset root: \`$tacauchy_asset_root\`, size \`$tacauchy_asset_size\`
- TaCauchy GelSight Mini Sensor.usd: \`$tacauchy_gsmini_status\`
- TaCauchy tactile test shape USD count: \`$tacauchy_shape_count\`
- UniVTAC bundled TacEx root: \`$univtac_asset_root\`, size \`$univtac_asset_size\`
- UniVTAC GelSight Mini Sensor.usd: \`$univtac_gsmini_status\`
- UniVTAC tactile test shape USD count: \`$univtac_shape_count\`

## Interpretation

Gate 00F remains closed until target UniVTAC/TaCauchy envs exist, required
TaCauchy assets exist, official reference sanity runs pass inside Curiosity
tmux-held Slurm allocation, and a fresh Gate review consumes that evidence.
MD

printf '%s\n' "$STATUS_JSON"
printf '%s\n' "$REPORT_MD"
