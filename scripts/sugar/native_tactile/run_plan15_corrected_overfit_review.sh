#!/usr/bin/env bash
# Strict frozen review under the exact fixed 3x overfit condition.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 CHECKPOINT CORRECTED_SCALE_FILE OUTPUT_ROOT [DEVICE]" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "overfit review must run inside a retained Slurm GPU allocation" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
checkpoint=$(realpath "$1")
scale_file=$(realpath "$2")
output_root=$3
device=${4:-cuda:0}
python_bin=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python

if [[ "$output_root" != /* ]]; then
    output_root="$repo_root/$output_root"
fi
case "$output_root" in
    "$repo_root"/experiments/*) ;;
    *) echo "OUTPUT_ROOT must remain below $repo_root/experiments" >&2; exit 2 ;;
esac
if [[ -e "$output_root" ]]; then
    echo "refusing to overwrite overfit review: $output_root" >&2
    exit 2
fi

export PYTHONPATH="$repo_root/IsaacLab/source/isaaclab:$repo_root/IsaacLab/source/isaaclab_assets:$repo_root/IsaacLab/source/isaaclab_contrib:$repo_root/IsaacLab/source/isaaclab_rl:$repo_root/SUGAR/source/sugar_rl:$repo_root/SUGAR/source/sugar_il:$repo_root/SUGAR/scripts/sugar_rl${PYTHONPATH:+:$PYTHONPATH}"

env -u VK_ICD_FILENAMES "$python_bin" -u \
    "$repo_root/SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py" \
    --branch PS \
    --checkpoint "$checkpoint" \
    --patch-scale-file "$scale_file" \
    --output-root "$output_root" \
    --training-seed 151014 \
    --seed 152014 \
    --mass-factor 3.0 \
    --motion-folder "$repo_root/SUGAR/data/CarryBox/data_045" \
    --motion-id 0 \
    --profiles 4 \
    --num-envs 4 \
    --max-steps 450 \
    --post-jump-window 80 \
    --fixed-3x-overfit-gate \
    --headless \
    --device "$device"

test -s "$output_root/summary.json"
test -s "$output_root/frozen_evaluation_trace.npz"
