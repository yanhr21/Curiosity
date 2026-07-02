#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_gate00f_runtime_preflight_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/runtime_preflight/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/runtime_preflight/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/runtime_preflight/$RUN_TAG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

summary_json="$OUTPUT_DIR/gate00f_runtime_preflight_summary.json"
report_md="$REPORT_DIR/gate00f_runtime_preflight.md"
run_log="$LOG_DIR/gate00f_runtime_preflight.log"
RUNTIME_REGISTRY="${RUNTIME_REGISTRY:-$ROOT/experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json}"
REGISTRY_VALIDATOR="${REGISTRY_VALIDATOR:-$ROOT/src/newton_tactile_curiosity/gate00f_runtime_registry_validate.py}"
REGISTRY_VALIDATION_SUMMARY="$OUTPUT_DIR/gate00f_runtime_registry_validation_summary.json"

probe_python() {
  local label="$1"
  local py="$2"
  shift 2
  if [[ ! -x "$py" ]]; then
    printf '{"label":"%s","python":"%s","status":"missing_python","version":"","missing_modules":["python_executable"],"present_modules":[]}' "$label" "$py"
    return 0
  fi
  "$py" - "$label" "$py" "$@" <<'PY'
import importlib.util
import json
import subprocess
import sys

label = sys.argv[1]
python_path = sys.argv[2]
modules = sys.argv[3:]
version = subprocess.run([python_path, "--version"], capture_output=True, text=True, check=False)
version_text = (version.stdout or version.stderr).strip()
present = []
missing = []
for name in modules:
    if importlib.util.find_spec(name) is None:
        missing.append(name)
    else:
        present.append(name)
status = "pass_runtime_preflight" if not missing else "fail_missing_modules"
print(json.dumps({
    "label": label,
    "python": python_path,
    "status": status,
    "version": version_text,
    "present_modules": present,
    "missing_modules": missing,
}, sort_keys=True))
PY
}

registry_value() {
  local target="$1"
  local key="$2"
  jq -r --arg target "$target" --arg key "$key" '.targets[$target][$key] // ""' "$RUNTIME_REGISTRY"
}

probe_container() {
  local label="$1"
  local runtime="$2"
  local artifact_path="$3"
  local image_id="$4"
  local image_ref="$5"
  local container_python="$6"
  shift 6

  local image="${image_id:-$image_ref}"
  local runner=""
  local output=""
  local exit_code=0

  if [[ -z "$container_python" ]]; then
    container_python="python3"
  fi

  case "$runtime" in
    docker)
      if ! command -v docker >/dev/null 2>&1; then
        jq -n --arg label "$label" --arg runtime "$runtime" --arg status "missing_container_runner" \
          '{label: $label, kind: "container", container_runtime: $runtime, status: $status, missing_modules: ["docker"], present_modules: []}'
        return 0
      fi
      if [[ -z "$image" ]]; then
        jq -n --arg label "$label" --arg runtime "$runtime" --arg status "missing_container_image_id" \
          '{label: $label, kind: "container", container_runtime: $runtime, status: $status, missing_modules: ["container_image"], present_modules: []}'
        return 0
      fi
      set +e
      output="$(docker run --rm --gpus all -v "$ROOT:$ROOT" -w "$ROOT" "$image" "$container_python" - "$label" "$container_python" "$@" <<'PY'
import importlib.util
import json
import subprocess
import sys

label = sys.argv[1]
python_path = sys.argv[2]
modules = sys.argv[3:]
version = subprocess.run([python_path, "--version"], capture_output=True, text=True, check=False)
version_text = (version.stdout or version.stderr).strip()
present = []
missing = []
for name in modules:
    if importlib.util.find_spec(name) is None:
        missing.append(name)
    else:
        present.append(name)
status = "pass_runtime_preflight" if not missing else "fail_missing_modules"
print(json.dumps({
    "label": label,
    "kind": "container",
    "python": python_path,
    "status": status,
    "version": version_text,
    "present_modules": present,
    "missing_modules": missing,
}, sort_keys=True))
PY
)"
      exit_code=$?
      set -e
      ;;
    singularity|apptainer|sif)
      if [[ "$runtime" == "apptainer" ]] && command -v apptainer >/dev/null 2>&1; then
        runner="apptainer"
      elif [[ "$runtime" == "singularity" ]] && command -v singularity >/dev/null 2>&1; then
        runner="singularity"
      elif command -v apptainer >/dev/null 2>&1; then
        runner="apptainer"
      elif command -v singularity >/dev/null 2>&1; then
        runner="singularity"
      fi
      if [[ -z "$runner" ]]; then
        jq -n --arg label "$label" --arg runtime "$runtime" --arg status "missing_container_runner" \
          '{label: $label, kind: "container", container_runtime: $runtime, status: $status, missing_modules: ["singularity_or_apptainer"], present_modules: []}'
        return 0
      fi
      if [[ -z "$artifact_path" ]]; then
        jq -n --arg label "$label" --arg runtime "$runtime" --arg status "missing_container_artifact_path" \
          '{label: $label, kind: "container", container_runtime: $runtime, status: $status, missing_modules: ["container_artifact"], present_modules: []}'
        return 0
      fi
      set +e
      output="$("$runner" exec --nv --bind "$ROOT:$ROOT" "$artifact_path" "$container_python" - "$label" "$container_python" "$@" <<'PY'
import importlib.util
import json
import subprocess
import sys

label = sys.argv[1]
python_path = sys.argv[2]
modules = sys.argv[3:]
version = subprocess.run([python_path, "--version"], capture_output=True, text=True, check=False)
version_text = (version.stdout or version.stderr).strip()
present = []
missing = []
for name in modules:
    if importlib.util.find_spec(name) is None:
        missing.append(name)
    else:
        present.append(name)
status = "pass_runtime_preflight" if not missing else "fail_missing_modules"
print(json.dumps({
    "label": label,
    "kind": "container",
    "python": python_path,
    "status": status,
    "version": version_text,
    "present_modules": present,
    "missing_modules": missing,
}, sort_keys=True))
PY
)"
      exit_code=$?
      set -e
      ;;
    *)
      jq -n --arg label "$label" --arg runtime "$runtime" --arg status "unsupported_container_runtime_for_preflight" \
        '{label: $label, kind: "container", container_runtime: $runtime, status: $status, missing_modules: ["container_preflight_runner"], present_modules: []}'
      return 0
      ;;
  esac

  if [[ "$exit_code" -ne 0 ]]; then
    jq -n \
      --arg label "$label" \
      --arg runtime "$runtime" \
      --arg status "fail_container_preflight_command" \
      --arg output "$output" \
      '{label: $label, kind: "container", container_runtime: $runtime, status: $status, command_output: $output, missing_modules: ["container_command"], present_modules: []}'
    return 0
  fi
  printf '%s\n' "$output"
}

probe_registered_target() {
  local label="$1"
  shift
  local kind
  kind="$(registry_value "$label" kind)"
  if [[ "$kind" == "python_env" ]]; then
    probe_python "$label" "$(registry_value "$label" path)" "$@"
    return 0
  fi
  if [[ "$kind" == "container" ]]; then
    probe_container \
      "$label" \
      "$(registry_value "$label" container_runtime)" \
      "$(registry_value "$label" artifact_path)" \
      "$(registry_value "$label" image_id)" \
      "$(registry_value "$label" image_ref)" \
      "$(registry_value "$label" container_python)" \
      "$@"
    return 0
  fi
  jq -n --arg label "$label" --arg kind "$kind" \
    '{label: $label, kind: $kind, status: "unsupported_runtime_kind", missing_modules: ["runtime_kind"], present_modules: []}'
}

{
  echo "GATE00F_RUNTIME_PREFLIGHT_START"
  echo "RUN_TAG=$RUN_TAG"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "HOSTNAME=$(hostname)"
  echo "NOTE=module_spec_preflight_not_training_not_simulation"
  echo "RUNTIME_REGISTRY=$RUNTIME_REGISTRY"

  registry_status="missing"
  registry_exit=1
  if [[ ! -f "$RUNTIME_REGISTRY" ]]; then
    registry_status="missing_registry_file"
  elif [[ ! -f "$REGISTRY_VALIDATOR" ]]; then
    registry_status="missing_registry_validator"
  else
    set +e
    python3 "$REGISTRY_VALIDATOR" \
      --registry "$RUNTIME_REGISTRY" \
      --output-json "$REGISTRY_VALIDATION_SUMMARY"
    registry_exit=$?
    set -e
    registry_status="$(jq -r '.status // "missing"' "$REGISTRY_VALIDATION_SUMMARY" 2>/dev/null || echo missing)"
  fi

  if [[ "$registry_exit" -ne 0 || "$registry_status" != "pass_gate00f_runtime_registry" ]]; then
    jq -n \
      --arg run_tag "$RUN_TAG" \
      --arg slurm_job_id "${SLURM_JOB_ID:-}" \
      --arg host "$(hostname)" \
      --arg runtime_registry "$RUNTIME_REGISTRY" \
      --arg registry_validation_summary "$REGISTRY_VALIDATION_SUMMARY" \
      --arg registry_status "$registry_status" \
      '{
        run_tag: $run_tag,
        classification: "gate00f_runtime_preflight_not_training_not_simulation_not_gate_completion",
        status: "fail_gate00f_runtime_preflight_registry_not_accepted",
        slurm_job_id: $slurm_job_id,
        host: $host,
        runtime_registry: $runtime_registry,
        registry_validation_summary: $registry_validation_summary,
        registry_status: $registry_status,
        checks: {},
        gate_effect: "preflight_blocked_until_runtime_registry_passes"
      }' >"$summary_json"

    cat >"$report_md" <<EOF
# Gate 00F Runtime Preflight

- Run tag: \`$RUN_TAG\`
- Slurm job: \`${SLURM_JOB_ID:-}\`
- Host: \`$(hostname)\`
- Status: \`fail_gate00f_runtime_preflight_registry_not_accepted\`
- Runtime registry: \`$RUNTIME_REGISTRY\`
- Registry status: \`$registry_status\`
- Registry validation summary: \`$REGISTRY_VALIDATION_SUMMARY\`

The runtime registry must pass before module preflight can run. This is not
training, simulation, rendering, evaluation, model loading, or Gate 00F
completion.
EOF

    echo "SUMMARY_JSON=$summary_json"
    echo "REPORT_MD=$report_md"
    echo "GATE00F_RUNTIME_PREFLIGHT_END_REGISTRY_NOT_ACCEPTED"
    exit 1
  fi

  univtac_probe="$(probe_registered_target univtac isaacsim isaaclab tacex tacex_uipc)"
  tacauchy_probe="$(probe_registered_target tacauchy isaacsim isaaclab tacex tacex_uipc)"
  tacsl_probe="$(probe_registered_target isaaclab_tacsl isaacsim isaaclab isaaclab_contrib)"

  univtac_status="$(printf '%s\n' "$univtac_probe" | jq -r '.status')"
  tacauchy_status="$(printf '%s\n' "$tacauchy_probe" | jq -r '.status')"
  tacsl_status="$(printf '%s\n' "$tacsl_probe" | jq -r '.status')"

  status="fail_gate00f_runtime_preflight"
  if [[ "$univtac_status" == "pass_runtime_preflight" && "$tacauchy_status" == "pass_runtime_preflight" && "$tacsl_status" == "pass_runtime_preflight" ]]; then
    status="pass_gate00f_runtime_preflight"
  fi

  jq -n \
    --arg run_tag "$RUN_TAG" \
    --arg status "$status" \
    --arg slurm_job_id "${SLURM_JOB_ID:-}" \
    --arg host "$(hostname)" \
    --arg runtime_registry "$RUNTIME_REGISTRY" \
    --arg registry_validation_summary "$REGISTRY_VALIDATION_SUMMARY" \
    --arg registry_status "$registry_status" \
    --argjson univtac "$univtac_probe" \
    --argjson tacauchy "$tacauchy_probe" \
    --argjson tacsl "$tacsl_probe" \
    '{
      run_tag: $run_tag,
      classification: "gate00f_runtime_preflight_not_training_not_simulation_not_gate_completion",
      status: $status,
      slurm_job_id: $slurm_job_id,
      host: $host,
      runtime_registry: $runtime_registry,
      registry_validation_summary: $registry_validation_summary,
      registry_status: $registry_status,
      checks: {
        univtac: $univtac,
        tacauchy: $tacauchy,
        isaaclab_tacsl: $tacsl
      },
      gate_effect: "preflight_only_does_not_clear_gate00f"
    }' >"$summary_json"

  cat >"$report_md" <<EOF
# Gate 00F Runtime Preflight

- Run tag: \`$RUN_TAG\`
- Slurm job: \`${SLURM_JOB_ID:-}\`
- Host: \`$(hostname)\`
- Status: \`$status\`
- Classification: \`gate00f_runtime_preflight_not_training_not_simulation_not_gate_completion\`

## Checks

- UniVTAC: \`$univtac_status\`
- TaCauchy: \`$tacauchy_status\`
- IsaacLab TacSL: \`$tacsl_status\`

This preflight only checks Python executability and module specs. It does not
run simulation, rendering, training, evaluation, model loading, or dependency
installation. Passing this preflight does not clear Gate 00F by itself; the
Gate 00F reference bundle and strict bundle acceptance must still pass.
EOF

  echo "SUMMARY_JSON=$summary_json"
  echo "REPORT_MD=$report_md"
  echo "GATE00F_RUNTIME_PREFLIGHT_END"
} 2>&1 | tee "$run_log"
