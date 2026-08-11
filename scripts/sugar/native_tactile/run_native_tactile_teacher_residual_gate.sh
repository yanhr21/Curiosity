#!/usr/bin/env bash
# Collect condition-disjoint contact states and run the serious tactile fusion gate.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this gate inside the retained Slurm GPU allocation." >&2
    exit 2
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 OUTPUT_ROOT [INITIAL_CHECKPOINT]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="$1"
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"
fi
case "$OUTPUT_ROOT" in
    "$ROOT"/experiments/*) ;;
    *) echo "OUTPUT_ROOT must remain below $ROOT/experiments" >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing overwrite: $OUTPUT_ROOT" >&2
    exit 2
fi

INITIAL="${2:-$ROOT/experiments/native_tactile_training/action_residual_zero_64u_seed13011_20260811/model_prelearn.pt}"
if [[ "$INITIAL" != /* ]]; then
    INITIAL="$ROOT/$INITIAL"
fi
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
if [[ ! -f "$INITIAL" ]]; then
    echo "Missing exact official warm-start checkpoint: $INITIAL" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT/rollouts" "$OUTPUT_ROOT/datasets"

collect_default_friction() {
    local label="$1"
    local mass_scale="$2"
    bash "$ROOT/scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh" \
        residual_tactile \
        "$INITIAL" \
        "$OUTPUT_ROOT/rollouts/$label.json" \
        --condition_label "$label" \
        --mass_scale "$mass_scale" \
        --actor_tactile_mode zeroed \
        --supervision_output "$OUTPUT_ROOT/datasets/$label.npz"
}

collect_explicit_friction() {
    local label="$1"
    local mass_scale="$2"
    local static_friction="$3"
    local dynamic_friction="$4"
    bash "$ROOT/scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh" \
        residual_tactile \
        "$INITIAL" \
        "$OUTPUT_ROOT/rollouts/$label.json" \
        --condition_label "$label" \
        --mass_scale "$mass_scale" \
        --static_friction "$static_friction" \
        --dynamic_friction "$dynamic_friction" \
        --actor_tactile_mode zeroed \
        --supervision_output "$OUTPUT_ROOT/datasets/$label.npz"
}

# Frozen before outcomes: two training conditions, one selection condition,
# and two untouched held-out conditions. Every rollout starts at motion-45
# frame zero and uses the exact-zero official actor while the live physical
# tactile history is recorded pre-action.
collect_default_friction train_nominal_mass1p0 1.0
collect_default_friction train_heavy_mass1p5 1.5
collect_explicit_friction selection_mass1p25_friction0p40_0p30 1.25 0.40 0.30
collect_default_friction test_heavy_mass2p0 2.0
collect_explicit_friction test_low_friction0p25_0p20 1.0 0.25 0.20

export PYTHONPATH="$ROOT:$ROOT/SUGAR/source/sugar_rl:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab_contrib"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/train_native_tactile_teacher_residual_gate.py" \
    --initial-checkpoint "$INITIAL" \
    --train "$OUTPUT_ROOT/datasets/train_nominal_mass1p0.npz" \
    --train "$OUTPUT_ROOT/datasets/train_heavy_mass1p5.npz" \
    --selection "$OUTPUT_ROOT/datasets/selection_mass1p25_friction0p40_0p30.npz" \
    --test "$OUTPUT_ROOT/datasets/test_heavy_mass2p0.npz" \
    --test "$OUTPUT_ROOT/datasets/test_low_friction0p25_0p20.npz" \
    --output-dir "$OUTPUT_ROOT/training" \
    --device cuda:0 \
    --seed 13011 \
    --steps 400 \
    --batch-size 16 \
    --learning-rate 1.0e-3 \
    --eval-every 25

"$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/audit_native_tactile_teacher_residual_gate.py" \
    --result-root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/independent_audit.json"

printf 'complete=%s\nreport=%s\naudit=%s\n' \
    "$OUTPUT_ROOT" \
    "$OUTPUT_ROOT/training/report.json" \
    "$OUTPUT_ROOT/independent_audit.json"
