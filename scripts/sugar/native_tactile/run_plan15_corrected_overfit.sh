#!/usr/bin/env bash
# Run the fixed motion45, 3x-mass corrected PS overfit diagnostic.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 CORRECTED_SCALE_FILE OUTPUT_ROOT [DEVICE] [RESUME_CHECKPOINT]" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "overfit must run inside a retained Slurm GPU allocation" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
scale_file=$(realpath "$1")
output_root=$2
device=${3:-cuda:0}
resume_checkpoint=${4:-}
python_bin=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python

# Keep the lock outside the run directory. Moving a bad run to legacy must not
# allow its still-live process to coexist with a replacement writer.
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
    case $(basename "$resume_checkpoint") in
        model_[0-9]*.pt) ;;
        *) echo "unexpected overfit resume checkpoint name" >&2; exit 2 ;;
    esac
    resume_args=(--resume_checkpoint_path "$resume_checkpoint")
elif [[ -e "$output_root" ]]; then
    echo "refusing to overwrite corrected overfit output: $output_root" >&2
    exit 2
fi

export PYTHONPATH="$repo_root/IsaacLab/source/isaaclab:$repo_root/IsaacLab/source/isaaclab_assets:$repo_root/IsaacLab/source/isaaclab_contrib:$repo_root/IsaacLab/source/isaaclab_rl:$repo_root/SUGAR/source/sugar_rl:$repo_root/SUGAR/source/sugar_il:$repo_root/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"

env -u VK_ICD_FILENAMES "$python_bin" -u \
    "$repo_root/SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py" \
    --task Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-Overfit-BCPPO \
    --patch-scale-file "$scale_file" \
    --seed 151014 \
    --num_envs 4 \
    --log_dir "$output_root" \
    "${resume_args[@]}" \
    --headless \
    --device "$device"

if [[ ! -s "$output_root/model_1499.pt" ]]; then
    echo "corrected overfit did not reach model_1499.pt" >&2
    exit 1
fi
echo "corrected overfit endpoint: $output_root/model_1499.pt"
