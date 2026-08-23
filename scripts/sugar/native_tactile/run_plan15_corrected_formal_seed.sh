#!/usr/bin/env bash
# Start or resume exactly one corrected formal Z/P/PS seed. Never auto-chain.

set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "usage: $0 Z|P|PS TRAIN_SEED CORRECTED_SCALE_FILE OUTPUT_ROOT [DEVICE] [RESUME_CHECKPOINT]" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "formal training must run inside a retained Slurm GPU allocation" >&2
    exit 2
fi

branch=$1
seed=$2
scale_file=$(realpath "$3")
output_root=$4
device=${5:-cuda:0}
resume_checkpoint=${6:-}
case "$branch" in Z|P|PS) ;; *) echo "branch must be Z, P or PS" >&2; exit 2 ;; esac
case "$seed" in 151014|151015|151016) ;; *) echo "unexpected formal seed" >&2; exit 2 ;; esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python

# A stable lock prevents concurrent writers even if an invalid output folder
# is moved aside while its original child process is still alive.
mkdir -p "$repo_root/experiments"
exec 9>"$repo_root/experiments/.plan15_training.lock"
if ! flock -n 9; then
    echo "another Plan-15 training process still owns the pipeline lock" >&2
    exit 75
fi

if [[ "$output_root" != /* ]]; then
    output_root="$repo_root/$output_root"
fi
case "$output_root" in
    "$repo_root"/experiments/*) ;;
    *) echo "OUTPUT_ROOT must remain below $repo_root/experiments" >&2; exit 2 ;;
esac
resume_args=()
if [[ -n "$resume_checkpoint" ]]; then
    resume_checkpoint=$(realpath "$resume_checkpoint")
    if [[ ! -s "$resume_checkpoint" ]]; then
        echo "resume checkpoint is missing: $resume_checkpoint" >&2
        exit 2
    fi
    if [[ $(dirname "$resume_checkpoint") != "$output_root" ]]; then
        echo "resume checkpoint must be a direct child of OUTPUT_ROOT" >&2
        exit 2
    fi
    # The cluster revokes H200 allocations at roughly four hours even when a
    # longer walltime is granted. Live TacSL needs identical explicit boundaries
    # below that limit. Never resume whichever checkpoint merely
    # happened to land before a revocation.
    resume_name=$(basename "$resume_checkpoint")
    case "$resume_name" in
        model_1250.pt) expected_checkpoint="$output_root/model_2000.pt" ;;
        model_2000.pt) expected_checkpoint="$output_root/model_2500.pt" ;;
        model_2500.pt) expected_checkpoint="$output_root/model_2999.pt" ;;
        *)
            echo "formal resource-boundary resume requires model_1250.pt, model_2000.pt or model_2500.pt" >&2
            exit 2
            ;;
    esac
    if [[ -e "$output_root/model_2999.pt" ]]; then
        echo "formal endpoint already exists: $output_root/model_2999.pt" >&2
        exit 2
    fi
    resume_args=(--resume_checkpoint_path "$resume_checkpoint")
elif [[ -e "$output_root" ]]; then
    echo "fresh corrected formal output already exists: $output_root" >&2
    exit 2
fi

export PYTHONPATH="$repo_root/IsaacLab/source/isaaclab:$repo_root/IsaacLab/source/isaaclab_assets:$repo_root/IsaacLab/source/isaaclab_contrib:$repo_root/IsaacLab/source/isaaclab_rl:$repo_root/SUGAR/source/sugar_rl:$repo_root/SUGAR/source/sugar_il:$repo_root/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"

env -u VK_ICD_FILENAMES "$python_bin" -u \
    "$repo_root/SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py" \
    --task "Sugar-G129dof-CarryBox-OnlineMass-Patch-${branch}-BCPPO" \
    --patch-scale-file "$scale_file" \
    --seed "$seed" \
    --log_dir "$output_root" \
    "${resume_args[@]}" \
    --headless \
    --device "$device"

if [[ -n "$resume_checkpoint" ]]; then
    if [[ $(basename "$expected_checkpoint") == model_2999.pt ]]; then
        endpoint_label="endpoint"
    else
        endpoint_label="resource boundary"
    fi
else
    expected_checkpoint="$output_root/model_1250.pt"
    endpoint_label="resource boundary"
fi
if [[ ! -s "$expected_checkpoint" ]]; then
    echo "formal seed did not reach expected $endpoint_label: $expected_checkpoint" >&2
    exit 1
fi
echo "corrected formal $endpoint_label: $expected_checkpoint"
