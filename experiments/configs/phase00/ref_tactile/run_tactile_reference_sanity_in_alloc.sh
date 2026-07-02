#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
TARGET="${TARGET:-}"
RUN_TAG="${RUN_TAG:-p00_ref_sanity_${TARGET:-unset}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/reference_sanity/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/reference_sanity/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase00/ref_tactile/reference_sanity/$RUN_TAG}"
RUNTIME_REGISTRY="${RUNTIME_REGISTRY:-$ROOT/experiments/configs/phase00/ref_tactile/gate00f_reference_runtime_registry_v1.json}"
COMMON_SH="$ROOT/experiments/configs/phase00/ref_tactile/gate00f_container_runtime_common.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi
if [[ "$TARGET" != "univtac" && "$TARGET" != "tacauchy" ]]; then
  echo "ERROR: TARGET must be one of: univtac, tacauchy" >&2
  exit 3
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"
source "$COMMON_SH"

repo_path=""
expected_commit=""
python_path=""
runtime_kind="python_env"
container_runtime=""
container_artifact_path=""
container_image_id=""
container_image_ref=""
container_python="python3"
case "$TARGET" in
  univtac)
    repo_path="$ROOT/external/UniVTAC"
    expected_commit="05bcd3edb92237107efa40105292a24f1a9fd761"
    python_path="${UNIVTAC_PYTHON:-}"
    if [[ -z "$python_path" && -x "$ROOT/envs/univtac/conda/bin/python" ]]; then
      python_path="$ROOT/envs/univtac/conda/bin/python"
    fi
    if [[ -z "$python_path" && -x "$ROOT/envs/univtac/.venv/bin/python" ]]; then
      python_path="$ROOT/envs/univtac/.venv/bin/python"
    fi
    ;;
  tacauchy)
    repo_path="$ROOT/external/TaCauchy"
    expected_commit="c228cfe9050904cd5d71d64f6eb5104768d4cbda"
    python_path="${TACAUCHY_PYTHON:-}"
    if [[ -z "$python_path" && -x "$ROOT/envs/tacauchy/conda/bin/python" ]]; then
      python_path="$ROOT/envs/tacauchy/conda/bin/python"
    fi
    if [[ -z "$python_path" && -x "$ROOT/envs/tacauchy/.venv/bin/python" ]]; then
      python_path="$ROOT/envs/tacauchy/.venv/bin/python"
    fi
    ;;
esac

if [[ -f "$RUNTIME_REGISTRY" ]]; then
  registered_kind="$(gate00f_registry_value "$RUNTIME_REGISTRY" "$TARGET" kind)"
  if [[ "$registered_kind" == "python_env" || "$registered_kind" == "container" ]]; then
    runtime_kind="$registered_kind"
    if [[ "$runtime_kind" == "python_env" ]]; then
      python_path="$(gate00f_registry_value "$RUNTIME_REGISTRY" "$TARGET" path)"
    else
      container_runtime="$(gate00f_registry_value "$RUNTIME_REGISTRY" "$TARGET" container_runtime)"
      container_artifact_path="$(gate00f_registry_value "$RUNTIME_REGISTRY" "$TARGET" artifact_path)"
      container_image_id="$(gate00f_registry_value "$RUNTIME_REGISTRY" "$TARGET" image_id)"
      container_image_ref="$(gate00f_registry_value "$RUNTIME_REGISTRY" "$TARGET" image_ref)"
      container_python="$(gate00f_registry_value "$RUNTIME_REGISTRY" "$TARGET" container_python)"
      if [[ -z "$container_python" ]]; then
        container_python="python3"
      fi
    fi
  fi
fi

run_log="$LOG_DIR/${TARGET}_official_reference_sanity.log"
summary_json="$OUTPUT_DIR/${TARGET}_official_reference_sanity_summary.json"
report_md="$REPORT_DIR/${TARGET}_official_reference_sanity.md"

{
  echo "TACTILE_REFERENCE_SANITY_START"
  echo "RUN_TAG=$RUN_TAG"
  echo "TARGET=$TARGET"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "HOSTNAME=$(hostname)"
  echo "REPO_PATH=$repo_path"
  echo "EXPECTED_COMMIT=$expected_commit"
  echo "RUNTIME_REGISTRY=$RUNTIME_REGISTRY"
  echo "RUNTIME_KIND=$runtime_kind"
  echo "PYTHON_PATH=${python_path:-missing}"
  echo "CONTAINER_RUNTIME=${container_runtime:-}"
  echo "CONTAINER_ARTIFACT_PATH=${container_artifact_path:-}"
  echo "CONTAINER_IMAGE_ID=${container_image_id:-}"
  echo "CONTAINER_IMAGE_REF=${container_image_ref:-}"
  echo "CONTAINER_PYTHON=${container_python:-}"
  echo "NOTE=official_reference_sanity_or_blocker_not_training_not_curiosity_success"

  status="unknown"
  blocker=""
  observed_commit=""
  probe_exit=""

  if [[ ! -d "$repo_path/.git" ]]; then
    status="blocked_missing_official_repo"
    blocker="missing official repository at $repo_path"
  else
    observed_commit="$(git -C "$repo_path" rev-parse HEAD)"
    if [[ "$observed_commit" != "$expected_commit" ]]; then
      status="blocked_commit_mismatch"
      blocker="expected $expected_commit but observed $observed_commit"
    elif [[ "$runtime_kind" == "python_env" && ( -z "$python_path" || ! -x "$python_path" ) ]]; then
      status="blocked_missing_prebuilt_environment"
      blocker="no executable prebuilt python found; set ${TARGET^^}_PYTHON or create envs/$TARGET/conda or envs/$TARGET/.venv before compute sanity"
    else
      status="registered_runtime_found"
      echo "PYTHON_VERSION_BEGIN"
      if [[ "$runtime_kind" == "container" ]]; then
        gate00f_container_exec "$ROOT" "$container_runtime" "$container_artifact_path" "$container_image_id" "$container_image_ref" "$container_python" --version
      else
        "$python_path" --version
      fi
      echo "PYTHON_VERSION_END"

      if [[ "$TARGET" == "univtac" ]]; then
        echo "UNIVTAC_OFFICIAL_SCHEMA_PROBE_BEGIN"
        set +e
        if [[ "$runtime_kind" == "container" ]]; then
          gate00f_container_exec "$ROOT" "$container_runtime" "$container_artifact_path" "$container_image_id" "$container_image_ref" bash -lc "cd '$repo_path' && '$container_python' -" <<'PY'
import json
from pathlib import Path

root = Path.cwd()
task_settings = json.loads((root / "policy" / "task_settings.json").read_text())
required = [
    root / "collect_data.sh",
    root / "eval_policy.sh",
    root / "envs" / "lift_can.py",
    root / "envs" / "lift_bottle.py",
    root / "policy" / "ACT" / "train_config_tactile_full.yml",
    root / "policy" / "ACT" / "train_config_vision.yml",
]
missing = [str(path.relative_to(root)) for path in required if not path.exists()]
print(json.dumps({
    "task_count": len(task_settings),
    "tasks": sorted(task_settings.keys()),
    "missing_required_files": missing,
}, indent=2, sort_keys=True))
PY
          probe_exit=$?
        else
          (
            cd "$repo_path"
            "$python_path" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
task_settings = json.loads((root / "policy" / "task_settings.json").read_text())
required = [
    root / "collect_data.sh",
    root / "eval_policy.sh",
    root / "envs" / "lift_can.py",
    root / "envs" / "lift_bottle.py",
    root / "policy" / "ACT" / "train_config_tactile_full.yml",
    root / "policy" / "ACT" / "train_config_vision.yml",
]
missing = [str(path.relative_to(root)) for path in required if not path.exists()]
print(json.dumps({
    "task_count": len(task_settings),
    "tasks": sorted(task_settings.keys()),
    "missing_required_files": missing,
}, indent=2, sort_keys=True))
PY
          )
          probe_exit=$?
        fi
        set -e
        echo "UNIVTAC_OFFICIAL_SCHEMA_PROBE_END"
        if [[ "$probe_exit" -eq 0 ]]; then
          status="pass_official_schema_probe"
        else
          status="blocked_official_schema_probe_failed"
          blocker="UniVTAC official schema probe exited with code $probe_exit"
        fi
      else
        echo "TACAUCHY_OFFICIAL_SCHEMA_PROBE_BEGIN"
        set +e
        if [[ "$runtime_kind" == "container" ]]; then
          gate00f_container_exec "$ROOT" "$container_runtime" "$container_artifact_path" "$container_image_id" "$container_image_ref" bash -lc "cd '$repo_path' && '$container_python' -" <<'PY'
import json
from pathlib import Path

root = Path.cwd()
required = [
    root / "README.md",
    root / "REPRODUCTION.md",
    root / "TACTILE_VISUALIZATION_GUIDE.md",
    root / "examples" / "gelpad_mesh_configs.py",
]
missing = [str(path.relative_to(root)) for path in required if not path.exists()]
semantic_terms = [
    "Cauchy stress",
    "normal pressure",
    "tangential traction",
    "adaptive mesh refinement",
    "GelSight Mini",
    "DIGIT",
    "9DTact",
]
text = "\n".join(path.read_text(errors="ignore") for path in [root / "README.md", root / "REPRODUCTION.md", root / "TACTILE_VISUALIZATION_GUIDE.md"] if path.exists())
present_terms = [term for term in semantic_terms if term.lower() in text.lower()]
print(json.dumps({
    "missing_required_files": missing,
    "present_semantic_terms": present_terms,
}, indent=2, sort_keys=True))
PY
          probe_exit=$?
        else
          (
            cd "$repo_path"
            "$python_path" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
required = [
    root / "README.md",
    root / "REPRODUCTION.md",
    root / "TACTILE_VISUALIZATION_GUIDE.md",
    root / "examples" / "gelpad_mesh_configs.py",
]
missing = [str(path.relative_to(root)) for path in required if not path.exists()]
semantic_terms = [
    "Cauchy stress",
    "normal pressure",
    "tangential traction",
    "adaptive mesh refinement",
    "GelSight Mini",
    "DIGIT",
    "9DTact",
]
text = "\n".join(path.read_text(errors="ignore") for path in [root / "README.md", root / "REPRODUCTION.md", root / "TACTILE_VISUALIZATION_GUIDE.md"] if path.exists())
present_terms = [term for term in semantic_terms if term.lower() in text.lower()]
print(json.dumps({
    "missing_required_files": missing,
    "present_semantic_terms": present_terms,
}, indent=2, sort_keys=True))
PY
          )
          probe_exit=$?
        fi
        set -e
        echo "TACAUCHY_OFFICIAL_SCHEMA_PROBE_END"
        if [[ "$probe_exit" -eq 0 ]]; then
          status="pass_official_schema_probe"
        else
          status="blocked_official_schema_probe_failed"
          blocker="TaCauchy official schema probe exited with code $probe_exit"
        fi
      fi
    fi
  fi

  cat >"$summary_json" <<EOF
{
  "run_tag": "$RUN_TAG",
  "target": "$TARGET",
  "status": "$status",
  "blocker": "$blocker",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "host": "$(hostname)",
  "repo_path": "$repo_path",
  "expected_commit": "$expected_commit",
  "observed_commit": "$observed_commit",
  "runtime_registry": "$RUNTIME_REGISTRY",
  "runtime_kind": "$runtime_kind",
  "python_path": "${python_path:-}",
  "container_runtime": "${container_runtime:-}",
  "container_artifact_path": "${container_artifact_path:-}",
  "container_image_id": "${container_image_id:-}",
  "container_image_ref": "${container_image_ref:-}",
  "container_python": "${container_python:-}",
  "schema_probe_exit": "${probe_exit:-}",
  "classification": "official_reference_sanity_or_blocker_not_training_not_curiosity_success"
}
EOF

  cat >"$report_md" <<EOF
# ${TARGET} Official Reference Sanity

- Run tag: \`$RUN_TAG\`
- Target: \`$TARGET\`
- Slurm job: \`${SLURM_JOB_ID:-}\`
- Host: \`$(hostname)\`
- Repository: \`$repo_path\`
- Expected commit: \`$expected_commit\`
- Observed commit: \`${observed_commit:-}\`
- Runtime registry: \`$RUNTIME_REGISTRY\`
- Runtime kind: \`$runtime_kind\`
- Python path: \`${python_path:-missing}\`
- Container runtime: \`${container_runtime:-none}\`
- Container artifact path: \`${container_artifact_path:-none}\`
- Container image id: \`${container_image_id:-none}\`
- Container image ref: \`${container_image_ref:-none}\`
- Container python: \`${container_python:-none}\`
- Schema probe exit: \`${probe_exit:-not run}\`
- Status: \`$status\`
- Blocker: \`${blocker:-none}\`

Classification: official reference sanity or blocker only. This is not
training, not tactile gate completion, and not curiosity success.
EOF

  echo "SUMMARY_JSON=$summary_json"
  echo "REPORT_MD=$report_md"
  echo "TACTILE_REFERENCE_SANITY_END"
} 2>&1 | tee "$run_log"
