#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "usage: $0 Z|P|PS CKPT_151014 CKPT_151015 CKPT_151016 OUTPUT_ROOT [DEVICE]" >&2
    exit 2
fi

branch=$1
shift
case "$branch" in
    Z|P|PS) ;;
    *) echo "branch must be Z, P, or PS" >&2; exit 2 ;;
esac

checkpoints=("$1" "$2" "$3")
output_root=$4
device=${5:-cuda:0}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

python_bin=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
scale_file="$repo_root/experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json"
train_seeds=(151014 151015 151016)
evaluation_seeds=(152014 152015 152016)
mass_factors=(1.0 1.5 3.0 6.0 10.0)

mkdir -p "$output_root"
for index in 0 1 2; do
    checkpoint=$(realpath "${checkpoints[$index]}")
    train_seed=${train_seeds[$index]}
    evaluation_seed=${evaluation_seeds[$index]}
    for mass_factor in "${mass_factors[@]}"; do
        factor_tag=${mass_factor/./p}x
        output="$output_root/train_${train_seed}_eval_${evaluation_seed}_${factor_tag}"
        echo "[PLAN15 SWEEP] branch=$branch train_seed=$train_seed eval_seed=$evaluation_seed factor=$mass_factor"
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
            --max-steps 420 \
            --post-jump-window 80 \
            --headless \
            --device "$device"
    done
done
