#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p00_tacsl_sanity_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/reference_sanity/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/reference_sanity/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/reference_sanity/$RUN_TAG}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-b4c321024792976150ca55fddb26fa34480d974e}"
PYTHON_PATH="${ISAACLAB_TACSL_PYTHON:-}"
RUNTIME_REGISTRY="${RUNTIME_REGISTRY:-$ROOT/experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json}"
COMMON_SH="$ROOT/experiments/configs/phase00/ref_tactile/gate00f_container_runtime_common.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"
source "$COMMON_SH"

repo_path="$ROOT/external/IsaacLab_official"
if [[ -z "$PYTHON_PATH" && -x "$ROOT/envs/isaaclab_tacsl/conda/bin/python" ]]; then
  PYTHON_PATH="$ROOT/envs/isaaclab_tacsl/conda/bin/python"
fi
if [[ -z "$PYTHON_PATH" && -x "$ROOT/envs/isaaclab_tacsl/.venv/bin/python" ]]; then
  PYTHON_PATH="$ROOT/envs/isaaclab_tacsl/.venv/bin/python"
fi

runtime_kind="python_env"
container_runtime=""
container_artifact_path=""
container_image_id=""
container_image_ref=""
container_python="python3"
if [[ -f "$RUNTIME_REGISTRY" ]]; then
  registered_kind="$(gate00f_registry_value "$RUNTIME_REGISTRY" isaaclab_tacsl kind)"
  if [[ "$registered_kind" == "python_env" || "$registered_kind" == "container" ]]; then
    runtime_kind="$registered_kind"
    if [[ "$runtime_kind" == "python_env" ]]; then
      PYTHON_PATH="$(gate00f_registry_value "$RUNTIME_REGISTRY" isaaclab_tacsl path)"
    else
      container_runtime="$(gate00f_registry_value "$RUNTIME_REGISTRY" isaaclab_tacsl container_runtime)"
      container_artifact_path="$(gate00f_registry_value "$RUNTIME_REGISTRY" isaaclab_tacsl artifact_path)"
      container_image_id="$(gate00f_registry_value "$RUNTIME_REGISTRY" isaaclab_tacsl image_id)"
      container_image_ref="$(gate00f_registry_value "$RUNTIME_REGISTRY" isaaclab_tacsl image_ref)"
      container_python="$(gate00f_registry_value "$RUNTIME_REGISTRY" isaaclab_tacsl container_python)"
      if [[ -z "$container_python" ]]; then
        container_python="python3"
      fi
    fi
  fi
fi

run_log="$LOG_DIR/isaaclab_tacsl_official_sanity.log"
summary_json="$OUTPUT_DIR/isaaclab_tacsl_official_sanity_summary.json"
report_md="$REPORT_DIR/isaaclab_tacsl_official_sanity.md"

{
  echo "ISAACLAB_TACSL_SANITY_START"
  echo "RUN_TAG=$RUN_TAG"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "HOSTNAME=$(hostname)"
  echo "REPO_PATH=$repo_path"
  echo "EXPECTED_COMMIT=$EXPECTED_COMMIT"
  echo "RUNTIME_REGISTRY=$RUNTIME_REGISTRY"
  echo "RUNTIME_KIND=$runtime_kind"
  echo "PYTHON_PATH=${PYTHON_PATH:-missing}"
  echo "CONTAINER_RUNTIME=${container_runtime:-}"
  echo "CONTAINER_ARTIFACT_PATH=${container_artifact_path:-}"
  echo "CONTAINER_IMAGE_ID=${container_image_id:-}"
  echo "CONTAINER_IMAGE_REF=${container_image_ref:-}"
  echo "CONTAINER_PYTHON=${container_python:-}"
  echo "NOTE=official_isaaclab_tacsl_sanity_or_blocker_not_training_not_curiosity_success"

  status="unknown"
  blocker=""
  observed_commit=""
  command=""
  command_exit=""

  if [[ ! -d "$repo_path/.git" ]]; then
    status="blocked_missing_official_isaaclab_repo"
    blocker="missing official IsaacLab repository at $repo_path"
  else
    observed_commit="$(git -C "$repo_path" rev-parse HEAD)"
    if [[ "$observed_commit" != "$EXPECTED_COMMIT" ]]; then
      status="blocked_commit_mismatch"
      blocker="expected $EXPECTED_COMMIT but observed $observed_commit"
    elif [[ "$runtime_kind" == "python_env" && ( -z "$PYTHON_PATH" || ! -x "$PYTHON_PATH" ) ]]; then
      status="blocked_missing_dependency_complete_isaaclab_tacsl_environment"
      blocker="no executable approved IsaacLab TacSL python found; set ISAACLAB_TACSL_PYTHON or prepare envs/isaaclab_tacsl/conda or envs/isaaclab_tacsl/.venv before compute sanity"
    else
      echo "PYTHON_VERSION_BEGIN"
      if [[ "$runtime_kind" == "container" ]]; then
        gate00f_container_exec "$ROOT" "$container_runtime" "$container_artifact_path" "$container_image_id" "$container_image_ref" "$container_python" --version
      else
        "$PYTHON_PATH" --version
      fi
      echo "PYTHON_VERSION_END"
      command="${PYTHON_PATH:-$container_python} scripts/demos/sensors/tacsl_sensor.py --headless --enable_cameras --use_tactile_rgb --use_tactile_ff --normal_contact_stiffness 1.0 --tangential_stiffness 0.1 --friction_coefficient 2.0 --contact_object_type nut --num_envs 1 --save_viz --save_viz_dir $OUTPUT_DIR/tactile_record"
      echo "OFFICIAL_TACSL_COMMAND=$command"
      set +e
      if [[ "$runtime_kind" == "container" ]]; then
        gate00f_container_exec "$ROOT" "$container_runtime" "$container_artifact_path" "$container_image_id" "$container_image_ref" \
          bash -lc "cd '$repo_path' && '$container_python' scripts/demos/sensors/tacsl_sensor.py --headless --enable_cameras --use_tactile_rgb --use_tactile_ff --normal_contact_stiffness 1.0 --tangential_stiffness 0.1 --friction_coefficient 2.0 --contact_object_type nut --num_envs 1 --save_viz --save_viz_dir '$OUTPUT_DIR/tactile_record'"
        command_exit=$?
      else
        (
          cd "$repo_path"
          "$PYTHON_PATH" scripts/demos/sensors/tacsl_sensor.py \
            --headless \
            --enable_cameras \
            --use_tactile_rgb \
            --use_tactile_ff \
            --normal_contact_stiffness 1.0 \
            --tangential_stiffness 0.1 \
            --friction_coefficient 2.0 \
            --contact_object_type nut \
            --num_envs 1 \
            --save_viz \
            --save_viz_dir "$OUTPUT_DIR/tactile_record"
        )
        command_exit=$?
      fi
      set -e
      if [[ "$command_exit" -eq 0 ]]; then
        status="pass_official_isaaclab_tacsl_demo_exited_zero"
      else
        status="blocked_official_isaaclab_tacsl_demo_failed"
        blocker="official TacSL demo exited with code $command_exit; preserve --use_tactile_rgb and inspect runtime/assets rather than dropping tactile RGB"
      fi
    fi
  fi

  cat >"$summary_json" <<EOF
{
  "run_tag": "$RUN_TAG",
  "target": "official_isaaclab_tacsl",
  "status": "$status",
  "blocker": "$blocker",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "host": "$(hostname)",
  "repo_path": "$repo_path",
  "expected_commit": "$EXPECTED_COMMIT",
  "observed_commit": "$observed_commit",
  "runtime_registry": "$RUNTIME_REGISTRY",
  "runtime_kind": "$runtime_kind",
  "python_path": "${PYTHON_PATH:-}",
  "container_runtime": "${container_runtime:-}",
  "container_artifact_path": "${container_artifact_path:-}",
  "container_image_id": "${container_image_id:-}",
  "container_image_ref": "${container_image_ref:-}",
  "container_python": "${container_python:-}",
  "official_command": "$command",
  "official_command_exit": "${command_exit:-}",
  "expected_fields": [
    "tactile_rgb_image",
    "tactile_depth_image",
    "penetration_depth",
    "tactile_normal_force",
    "tactile_shear_force"
  ],
  "classification": "official_isaaclab_tacsl_sanity_or_blocker_not_training_not_curiosity_success"
}
EOF

  cat >"$report_md" <<EOF
# Official IsaacLab TacSL Sanity

- Run tag: \`$RUN_TAG\`
- Slurm job: \`${SLURM_JOB_ID:-}\`
- Host: \`$(hostname)\`
- Repository: \`$repo_path\`
- Expected commit: \`$EXPECTED_COMMIT\`
- Observed commit: \`${observed_commit:-}\`
- Runtime registry: \`$RUNTIME_REGISTRY\`
- Runtime kind: \`$runtime_kind\`
- Python path: \`${PYTHON_PATH:-missing}\`
- Container runtime: \`${container_runtime:-none}\`
- Container artifact path: \`${container_artifact_path:-none}\`
- Container image id: \`${container_image_id:-none}\`
- Container image ref: \`${container_image_ref:-none}\`
- Container python: \`${container_python:-none}\`
- Status: \`$status\`
- Blocker: \`${blocker:-none}\`
- Official command: \`${command:-not run}\`
- Official command exit: \`${command_exit:-not run}\`

Required fields: \`tactile_rgb_image\`, \`tactile_depth_image\`,
\`penetration_depth\`, \`tactile_normal_force\`, and
\`tactile_shear_force\`.

Classification: official IsaacLab TacSL sanity or blocker only. This is not
training, not tactile gate completion, and not curiosity success.
EOF

  echo "SUMMARY_JSON=$summary_json"
  echo "REPORT_MD=$report_md"
  echo "ISAACLAB_TACSL_SANITY_END"
} 2>&1 | tee "$run_log"
