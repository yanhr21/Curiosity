#!/usr/bin/env bash
# Finish the formal PS branch serially, compare Z/P/PS, then run friction feasibility.

set -euo pipefail

device=${1:-cuda:0}
only_seed=${PLAN15_PS_ONLY_SEED:-}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
scale_file="$root/experiments/online_patch_tactile_mass_adaptation/leakage_sweep_v1/patch_channel_scales.json"
training_root="$root/experiments/online_patch_tactile_mass_adaptation/training_handoff_anchor025"
evaluation_root="$root/experiments/online_patch_tactile_mass_adaptation/frozen_evaluation_handoff"
runtime_root="$root/experiments/online_patch_tactile_mass_adaptation/runtime"

# Multiple retained allocations may become available at the same time.  Keep
# the formal PS train/evaluate chain strictly serial and prevent two runners
# from writing the same seed directory.
mkdir -p "$runtime_root"
exec 9>"$runtime_root/plan15_ps_remaining_pipeline.lock"
echo "[PLAN15 PS PIPELINE] waiting for exclusive pipeline lock"
flock 9
echo "[PLAN15 PS PIPELINE] acquired exclusive pipeline lock"

latest_checkpoint() {
    local seed=$1
    # model_pre_update.pt is a warm-start diagnostic, not a resumable numbered
    # runner checkpoint.  Only model_<integer>.pt participates in latest-step
    # selection.
    find "$training_root/ps_seed${seed}" -maxdepth 1 -type f \
        -regextype posix-extended -regex '.*/model_[0-9]+\.pt' \
        -print 2>/dev/null | sort -V | tail -n 1
}

train_seed() {
    local seed=$1
    local log_dir="$training_root/ps_seed${seed}"
    local endpoint="$log_dir/model_2999.pt"
    if [[ -s "$endpoint" ]]; then
        echo "[PLAN15 PS PIPELINE] seed=$seed endpoint already complete"
        return
    fi
    mkdir -p "$log_dir"
    local resume
    resume=$(latest_checkpoint "$seed")
    local resume_args=()
    if [[ -n "$resume" ]]; then
        resume_args=(--resume_checkpoint_path "$resume")
        echo "[PLAN15 PS PIPELINE] seed=$seed resume=$resume"
    else
        echo "[PLAN15 PS PIPELINE] seed=$seed start from scratch"
    fi
    "$python_bin" -u "$root/SUGAR/scripts/sugar_rl/train_online_patch_mass_bcppo.py" \
        --task Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-BCPPO \
        --patch-scale-file "$scale_file" \
        --seed "$seed" \
        --log_dir "$log_dir" \
        "${resume_args[@]}" \
        --headless \
        --device "$device"
    if [[ ! -s "$endpoint" ]]; then
        echo "PS seed $seed did not produce model_2999.pt" >&2
        exit 1
    fi
}

evaluate_seed() {
    local train_seed=$1
    local eval_seed=$2
    local output="$evaluation_root/ps_anchor025_formal_seed${train_seed}"
    local complete=1
    local factor tag
    for factor in 1.0 1.5 3.0 6.0 10.0; do
        tag=${factor/./p}x
        if [[ ! -s "$output/train_${train_seed}_eval_${eval_seed}_${tag}/summary.json" || \
              ! -s "$output/train_${train_seed}_eval_${eval_seed}_${tag}/frozen_evaluation_trace.npz" ]]; then
            complete=0
        fi
    done
    if [[ "$complete" -eq 1 ]]; then
        echo "[PLAN15 PS PIPELINE] seed=$train_seed frozen evaluation already complete"
        return
    fi
    "$root/scripts/sugar/native_tactile/run_plan15_frozen_seed.sh" \
        PS \
        "$training_root/ps_seed${train_seed}/model_2999.pt" \
        "$train_seed" \
        "$eval_seed" \
        "$output" \
        "$device"
}

pairs=(151014:152014 151015:152015 151016:152016)
if [[ -n "$only_seed" ]]; then
    case "$only_seed" in
        151014) pairs=(151014:152014) ;;
        151015) pairs=(151015:152015) ;;
        151016) pairs=(151016:152016) ;;
        *) echo "PLAN15_PS_ONLY_SEED must be 151014, 151015, or 151016" >&2; exit 2 ;;
    esac
fi

for pair in "${pairs[@]}"; do
    train_seed=${pair%%:*}
    eval_seed=${pair##*:}
    train_seed "$train_seed"
    evaluate_seed "$train_seed" "$eval_seed"
done

if [[ -n "$only_seed" ]]; then
    echo "[PLAN15 PS PIPELINE] single-seed run complete seed=$only_seed"
    exit 0
fi

comparison="$evaluation_root/z_p_ps_formal_comparison_v1.json"
/usr/local/python3.12/bin/python3 \
    "$root/SUGAR/scripts/sugar_rl/compare_online_patch_mass_sweeps.py" \
    --z-root \
        "$evaluation_root/z_anchor025_formal_seed151014" \
        "$evaluation_root/z_anchor025_formal_seed151015" \
        "$evaluation_root/z_anchor025_formal_seed151016" \
    --p-root \
        "$evaluation_root/p_anchor025_formal_seed151014" \
        "$evaluation_root/p_anchor025_formal_seed151015" \
        "$evaluation_root/p_anchor025_formal_seed151016" \
    --ps-root \
        "$evaluation_root/ps_anchor025_formal_seed151014" \
        "$evaluation_root/ps_anchor025_formal_seed151015" \
        "$evaluation_root/ps_anchor025_formal_seed151016" \
    --output "$comparison"

friction_root="$root/experiments/online_patch_tactile_mass_adaptation/friction_feasibility_after_ps"
mkdir -p "$friction_root"
for friction in 0.5 1.0 1.5 2.0; do
    friction_tag=${friction/./p}
    for factor in 6.0 10.0; do
        factor_tag=${factor/./p}
        output="$friction_root/official_refiner_mu${friction_tag}_factor${factor_tag}x"
        if [[ -s "$output/summary.json" && -s "$output/online_mass_jump_trace.npz" ]]; then
            echo "[PLAN15 FRICTION] already complete: $output"
            continue
        fi
        "$python_bin" -u "$root/scripts/sugar/native_tactile/preflight_online_patch_mass_jump.py" \
            --output-root "$output" \
            --motion-id 45 \
            --seed 150814 \
            --mass-factor "$factor" \
            --object-static-friction "$friction" \
            --object-dynamic-friction "$friction" \
            --max-steps 420 \
            --headless \
            --device "$device"
    done
done

jq -s '{
    schema: "plan15_carrybox_friction_feasibility_v1",
    controller: "frozen official Refiner",
    runs: map({
        mass_factor: .target_mass_factor,
        static_friction: .object_static_friction_readback,
        dynamic_friction: .object_dynamic_friction_readback,
        jump_frame: .first_jump_frame,
        height_loss_m: .maximum_post_jump_height_loss_m,
        hold_5cm: .post_jump_hold_5cm,
        drop_15cm: .post_jump_drop_15cm,
        bilateral_contact_frames: .bilateral_contact_frames
    })
}' "$friction_root"/official_refiner_*/summary.json > "$friction_root/aggregate_summary.json"

echo "[PLAN15 PS PIPELINE] complete comparison=$comparison friction=$friction_root/aggregate_summary.json"
