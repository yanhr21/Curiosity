#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "usage: $0 Z|P|PS CHECKPOINT TRAIN_SEED EVAL_SEED OUTPUT_ROOT [DEVICE]" >&2
    exit 2
fi

branch=$1
checkpoint=$2
train_seed=$3
evaluation_seed=$4
output_root=$5
device=${6:-cuda:0}

case "$branch" in
    Z|P|PS) ;;
    *) echo "branch must be Z, P, or PS" >&2; exit 2 ;;
esac
case "$train_seed:$evaluation_seed" in
    151014:152014|151015:152015|151016:152016) ;;
    *) echo "unexpected Plan-15 checkpoint/evaluation seed pairing" >&2; exit 2 ;;
esac

# Formal PS endpoints are frozen and inspected before their evaluation child is
# admitted.  This also prevents a long serial launcher from silently starting
# the next formal seed immediately after an endpoint.
if [[ "$branch" == PS && "${PLAN15_ALLOW_PS_ENDPOINT_EVALUATION:-0}" != 1 ]]; then
    echo "PS seed $train_seed endpoint is frozen for review; rerun this evaluation explicitly with PLAN15_ALLOW_PS_ENDPOINT_EVALUATION=1" >&2
    exit 75
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

python_bin=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
checkpoint=$(realpath "$checkpoint")
scale_file="$repo_root/experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json"
mass_factors=(1.0 1.5 3.0 6.0 10.0)

mkdir -p "$output_root"
for mass_factor in "${mass_factors[@]}"; do
    factor_tag=${mass_factor/./p}x
    output="$output_root/train_${train_seed}_eval_${evaluation_seed}_${factor_tag}"
    if [[ -s "$output/summary.json" && -s "$output/frozen_evaluation_trace.npz" ]]; then
        echo "[PLAN15 SEED SWEEP] already complete: $output"
        continue
    fi
    echo "[PLAN15 SEED SWEEP] branch=$branch train_seed=$train_seed eval_seed=$evaluation_seed factor=$mass_factor"
    "$python_bin" -u SUGAR/scripts/sugar_rl/evaluate_online_patch_mass_bcppo.py \
        --branch "$branch" \
        --checkpoint "$checkpoint" \
        --patch-scale-file "$scale_file" \
        --output-root "$output" \
        --training-seed "$train_seed" \
        --seed "$evaluation_seed" \
        --mass-factor "$mass_factor" \
        --motion-id 45 \
        --profiles 20 \
        --num-envs 4 \
        --max-steps 450 \
        --post-jump-window 80 \
        --physical-outcome-view \
        --headless \
        --device "$device"
    if [[ ! -s "$output/summary.json" || ! -s "$output/frozen_evaluation_trace.npz" ]]; then
        echo "frozen evaluation did not write its required outputs: $output" >&2
        exit 1
    fi
done
