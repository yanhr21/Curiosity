#!/usr/bin/env bash

gate00f_registry_value() {
  local registry="$1"
  local target="$2"
  local key="$3"
  jq -r --arg target "$target" --arg key "$key" '.targets[$target][$key] // ""' "$registry"
}

gate00f_container_exec() {
  local root="$1"
  local runtime="$2"
  local artifact_path="$3"
  local image_id="$4"
  local image_ref="$5"
  shift 5

  case "$runtime" in
    docker)
      local image="${image_id:-$image_ref}"
      if [[ -z "$image" ]]; then
        echo "ERROR: docker container runtime requires local image_id." >&2
        return 21
      fi
      if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker is not available on PATH." >&2
        return 22
      fi
      docker run --rm --gpus all -v "$root:$root" -w "$root" "$image" "$@"
      ;;
    singularity|apptainer|sif)
      local runner=""
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
        echo "ERROR: neither singularity nor apptainer is available on PATH." >&2
        return 23
      fi
      if [[ -z "$artifact_path" ]]; then
        echo "ERROR: singularity/apptainer/sif runtime requires artifact_path." >&2
        return 24
      fi
      "$runner" exec --nv --bind "$root:$root" "$artifact_path" "$@"
      ;;
    *)
      echo "ERROR: unsupported container runtime for official sanity: $runtime" >&2
      return 25
      ;;
  esac
}
