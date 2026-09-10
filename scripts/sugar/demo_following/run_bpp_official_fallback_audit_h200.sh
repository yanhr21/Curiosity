#!/usr/bin/env bash
set -euo pipefail

repo_root=/public/home/yanhongru/Curiosity
python_bin=/public/home/yanhongru/envs/bpp_official_py310/bin/python
source_repo="$repo_root/experiments/demo_following/bpp_official_v1/artifacts/behavior_prompting_b5b494e"
goal_checkpoint="$repo_root/experiments/demo_following/bpp_official_v1/artifacts/liberogen_goal_chain_ab2ce394/liberogen_goal_chain_behavior_prompting.ckpt"
goal_sha=f37e769c9460810be5f59b0ed77e1503c9dbbf2ed9780da85fe2760aac0ef7b4
source_commit=b5b494ed05fdd5d79e57b8b27072d3fa109ebfd0
spatial_repo=austinpatel/liberogen_spatial_combination
spatial_file=liberogen_spatial_combination_behavior_prompting.ckpt
run_root="$repo_root/experiments/demo_following/bpp_official_v1/fallback_audit_job${SLURM_JOB_ID:-missing}"
audit_only=${BPP_AUDIT_ONLY:-0}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: this audit must run inside a Slurm allocation" >&2
  exit 2
fi
case "$(hostname)" in
  mgmtserver*|login*)
    echo "ERROR: refusing to run on a login node" >&2
    exit 2
    ;;
esac

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
if ! grep -q 'H200' <<<"$gpu_inventory"; then
  echo "ERROR: allocated GPU is not an H200" >&2
  echo "$gpu_inventory" >&2
  exit 2
fi

mkdir -p "$run_root"
{
  echo "hostname=$(hostname)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "$gpu_inventory"
} | tee "$run_root/COMPUTE_IDENTITY.txt"

cd "$repo_root"
"$python_bin" -m py_compile \
  scripts/sugar/demo_following/audit_bpp_official_checkpoint_flexible.py \
  scripts/sugar/demo_following/audit_bpp_official_checkpoint.py \
  scripts/sugar/demo_following/audit_bpp_official_interface_shape_gate.py \
  scripts/sugar/demo_following/materialize_bpp_official_replay_buffer.py \
  scripts/sugar/demo_following/bpp_official_data_adapter.py
bash -n \
  scripts/sugar/demo_following/run_bpp_official_fallback_audit_h200.sh \
  scripts/sugar/demo_following/run_bpp_five_variant_action_collection.sh
git diff --check -- \
  scripts/sugar/demo_following/audit_bpp_official_checkpoint.py \
  scripts/sugar/demo_following/audit_bpp_official_checkpoint_flexible.py \
  scripts/sugar/demo_following/audit_bpp_official_interface_shape_gate.py \
  scripts/sugar/demo_following/materialize_bpp_official_replay_buffer.py \
  scripts/sugar/demo_following/bpp_official_data_adapter.py \
  scripts/sugar/demo_following/run_bpp_h200_preflight.sh \
  scripts/sugar/demo_following/run_bpp_official_fallback_audit_h200.sh \
  scripts/sugar/demo_following/run_bpp_five_variant_action_collection.sh \
  DOCS/current_status.md

export PYTHONPATH="$source_repo${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_XET=1
export HF_HOME=/public/home/yanhongru/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

git -C "$source_repo" rev-parse HEAD > "$run_root/LOCAL_SOURCE_COMMIT.txt"
set +e
git ls-remote https://github.com/real-stanford/behavior_prompting.git HEAD \
  > "$run_root/OFFICIAL_SOURCE_REMOTE_HEAD.txt" \
  2> "$run_root/OFFICIAL_SOURCE_REMOTE_HEAD.stderr.log"
remote_head_rc=$?
set -e
echo "$remote_head_rc" > "$run_root/OFFICIAL_SOURCE_REMOTE_HEAD.exitcode"

set +e
"$python_bin" scripts/sugar/demo_following/audit_bpp_official_checkpoint_flexible.py \
  --source-repo "$source_repo" \
  --checkpoint "$goal_checkpoint" \
  --output "$run_root/goal_chain_strict_audit.json" \
  --artifact-label goal_chain \
  --expected-source-commit "$source_commit" \
  --expected-checkpoint-sha256 "$goal_sha" \
  --apply-official-libero-policy-dunetp \
  > "$run_root/goal_chain_strict_audit.stdout.log" 2>&1
goal_rc=$?
set -e
echo "$goal_rc" > "$run_root/goal_chain_strict_audit.exitcode"

set +e
"$python_bin" scripts/sugar/demo_following/audit_bpp_official_checkpoint.py \
  --source-repo "$source_repo" \
  --checkpoint "$goal_checkpoint" \
  --interface-contract \
    scripts/sugar/demo_following/config/bpp_official_interface_adapter_v1.json \
  --output-dir "$run_root/goal_chain_formal_audit" \
  > "$run_root/goal_chain_formal_audit.stdout.log" 2>&1
goal_formal_rc=$?
set -e
echo "$goal_formal_rc" > "$run_root/goal_chain_formal_audit.exitcode"

interface_rc=skipped_goal_chain_failed
if [[ "$goal_rc" -eq 0 && "$goal_formal_rc" -eq 0 ]]; then
  set +e
  "$python_bin" \
    scripts/sugar/demo_following/audit_bpp_official_interface_shape_gate.py \
    --source-repo "$source_repo" \
    --checkpoint "$goal_checkpoint" \
    --contract \
      scripts/sugar/demo_following/config/bpp_official_interface_adapter_v1.json \
    --output "$run_root/bpp_official_interface_shape_gate.json" \
    > "$run_root/bpp_official_interface_shape_gate.stdout.log" 2>&1
  interface_rc=$?
  set -e
  echo "$interface_rc" > "$run_root/bpp_official_interface_shape_gate.exitcode"
fi

spatial_rc=skipped_goal_chain_passed
collection_rc=skipped_goal_chain_failed
if [[ "$goal_rc" -eq 0 && "$goal_formal_rc" -eq 0 ]]; then
  echo "Goal Chain strict inheritance passed; starting the frozen full-size candidate collector" \
    | tee "$run_root/GOAL_CHAIN_ADMITTED.txt"
  if [[ "$audit_only" == 1 ]]; then
    collection_rc=skipped_audit_only
  else
    set +e
    (
      unset PYTHONPATH
      bash scripts/sugar/demo_following/run_bpp_five_variant_action_collection.sh
    ) > "$run_root/bpp_candidate_collection.stdout.log" 2>&1
    collection_rc=$?
    set -e
    echo "$collection_rc" > "$run_root/bpp_candidate_collection.exitcode"
  fi
else
  spatial_metadata="$run_root/spatial_checkpoint_metadata.json"
  spatial_path=$(
    "$python_bin" - "$spatial_repo" "$spatial_file" "$spatial_metadata" "$run_root/hf_cache" <<'PY'
import json
import pathlib
import sys

from huggingface_hub import HfApi, hf_hub_download

repo_id, filename, metadata_path, cache_dir = sys.argv[1:]
info = HfApi().model_info(repo_id, files_metadata=True)
sibling = next((x for x in info.siblings if x.rfilename == filename), None)
if sibling is None:
    raise SystemExit(f"official checkpoint file is absent: {repo_id}/{filename}")
metadata = {
    "repo_id": repo_id,
    "revision": info.sha,
    "filename": filename,
    "size": sibling.size,
    "blob_id": getattr(sibling, "blob_id", None),
    "lfs": getattr(sibling, "lfs", None),
}
pathlib.Path(metadata_path).write_text(
    json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
)
downloaded = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    revision=info.sha,
    cache_dir=cache_dir,
)
print(downloaded)
PY
  )

  set +e
  "$python_bin" scripts/sugar/demo_following/audit_bpp_official_checkpoint_flexible.py \
    --source-repo "$source_repo" \
    --checkpoint "$spatial_path" \
    --output "$run_root/spatial_combination_strict_audit.json" \
    --artifact-label spatial_combination \
    --expected-source-commit "$source_commit" \
    --apply-official-libero-policy-dunetp \
    > "$run_root/spatial_combination_strict_audit.stdout.log" 2>&1
  spatial_rc=$?
  set -e
  echo "$spatial_rc" > "$run_root/spatial_combination_strict_audit.exitcode"
fi

{
  echo "goal_chain_audit_exitcode=$goal_rc"
  echo "goal_chain_formal_audit_exitcode=$goal_formal_rc"
  echo "interface_shape_gate_exitcode=$interface_rc"
  echo "spatial_combination_audit_exitcode=$spatial_rc"
  echo "candidate_collection_exitcode=$collection_rc"
  echo "finished_at=$(date --iso-8601=seconds)"
} | tee "$run_root/AUDIT_COMPLETE.txt"

# A passing Goal Chain audit immediately advances to the full frozen collector.
# Any nonzero collection exit is retained as evidence and propagated.
if [[ "$goal_rc" -eq 0 && "$goal_formal_rc" -eq 0 ]]; then
  if [[ "$audit_only" == 1 ]]; then
    exit 0
  fi
  exit "$collection_rc"
fi
if [[ "$spatial_rc" =~ ^[0-9]+$ && "$spatial_rc" -gt 1 ]]; then
  exit "$spatial_rc"
fi
