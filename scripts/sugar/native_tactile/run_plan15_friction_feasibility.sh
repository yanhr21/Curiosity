#!/usr/bin/env bash
# Run the independent frozen-Refiner CarryBox friction-feasibility sweep.

set -euo pipefail

device=${1:-cuda:0}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
output_root=${OUTPUT_ROOT:-"$root/experiments/online_patch_tactile_mass_adaptation/friction_feasibility_after_ps"}

mkdir -p "$output_root"
for friction in 0.5 1.0 1.5 2.0; do
    friction_tag=${friction/./p}
    for factor in 6.0 10.0; do
        factor_tag=${factor/./p}
        output="$output_root/official_refiner_mu${friction_tag}_factor${factor_tag}x"
        summary="$output/summary.json"
        trace="$output/online_mass_jump_trace.npz"
        if [[ -s "$summary" && -s "$trace" ]] && jq -e \
            --argjson friction "$friction" \
            --argjson factor "$factor" \
            '(.target_mass_factor == $factor)
             and (.object_static_friction_readback == $friction)
             and (.object_dynamic_friction_readback == $friction)
             and (.outcome_window_complete == true)
             and (.post_jump_frames >= 80)
             and (.bilateral_contact_for_10_frames_before_jump == true)' \
            "$summary" >/dev/null; then
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
            --max-steps 450 \
            --headless \
            --device "$device"
    done
done

jq -s '{
    schema: "plan15_carrybox_friction_feasibility_v2",
    controller: "frozen official Refiner",
    protocol: "independent frozen-Refiner feasibility sweep; no Z/P/PS result dependency",
    runs: map({
        mass_factor: .target_mass_factor,
        static_friction: .object_static_friction_readback,
        dynamic_friction: .object_dynamic_friction_readback,
        mass_readback_kg: .final_mass_readback_kg,
        jump_frame: .first_jump_frame,
        post_jump_frames: .post_jump_frames,
        outcome_window_complete: .outcome_window_complete,
        bilateral_pre_jump_10: .bilateral_contact_for_10_frames_before_jump,
        height_loss_m: .maximum_post_jump_height_loss_m,
        hold_5cm: .post_jump_hold_5cm,
        drop_15cm: .post_jump_drop_15cm
    })
}' "$output_root"/official_refiner_*/summary.json > "$output_root/aggregate_summary.json"

echo "[PLAN15 FRICTION] complete: $output_root/aggregate_summary.json"
