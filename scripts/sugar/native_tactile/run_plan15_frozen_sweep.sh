#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "usage: $0 Z|P|PS CKPT_151014 CKPT_151015 CKPT_151016 CORRECTED_SCALE_FILE OUTPUT_ROOT [DEVICE]" >&2
    exit 2
fi

branch=$1
shift
case "$branch" in
    Z|P|PS) ;;
    *) echo "branch must be Z, P, or PS" >&2; exit 2 ;;
esac

checkpoints=("$1" "$2" "$3")
scale_file=$4
output_root=$5
device=${6:-cuda:0}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

train_seeds=(151014 151015 151016)
evaluation_seeds=(152014 152015 152016)

mkdir -p "$output_root"
for index in 0 1 2; do
    train_seed=${train_seeds[$index]}
    evaluation_seed=${evaluation_seeds[$index]}
    "$repo_root/scripts/sugar/native_tactile/run_plan15_frozen_seed.sh" \
        "$branch" \
        "${checkpoints[$index]}" \
        "$train_seed" \
        "$evaluation_seed" \
        "$scale_file" \
        "$output_root" \
        "$device"
done
