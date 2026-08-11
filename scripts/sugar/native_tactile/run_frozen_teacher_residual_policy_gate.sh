#!/usr/bin/env bash
# Compare one pretrained tactile checkpoint with live and exact-zero input.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this gate inside the retained Slurm GPU allocation." >&2
    exit 2
fi
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 CHECKPOINT OUTPUT_ROOT" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CHECKPOINT="$1"
OUTPUT_ROOT="$2"
if [[ "$CHECKPOINT" != /* ]]; then CHECKPOINT="$ROOT/$CHECKPOINT"; fi
if [[ "$OUTPUT_ROOT" != /* ]]; then OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"; fi
case "$OUTPUT_ROOT" in
    "$ROOT"/experiments/*) ;;
    *) echo "OUTPUT_ROOT must remain below $ROOT/experiments" >&2; exit 2 ;;
esac
if [[ ! -f "$CHECKPOINT" ]]; then echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; fi
if [[ -e "$OUTPUT_ROOT" ]]; then echo "Refusing overwrite: $OUTPUT_ROOT" >&2; exit 2; fi
mkdir -p "$OUTPUT_ROOT"

evaluate() {
    local condition="$1"
    local input_mode="$2"
    shift 2
    local arm="residual_tactile"
    if [[ "$input_mode" == "zero" ]]; then arm="residual_zero"; fi
    bash "$ROOT/scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh" \
        "$arm" "$CHECKPOINT" "$OUTPUT_ROOT/${condition}_${input_mode}.json" \
        --condition_label "$condition" "$@"
}

# Frozen before behavior outcomes: the same checkpoint, source state, seed,
# physics, and task are used. Only the observation function changes from the
# live 54-patch history to the exact-zero/no-read control.
evaluate test_heavy_mass2p0 live --mass_scale 2.0
evaluate test_heavy_mass2p0 zero --mass_scale 2.0
evaluate test_low_friction0p25_0p20 live \
    --mass_scale 1.0 --static_friction 0.25 --dynamic_friction 0.20
evaluate test_low_friction0p25_0p20 zero \
    --mass_scale 1.0 --static_friction 0.25 --dynamic_friction 0.20

PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
"$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/summarize_native_tactile_frozen_pair.py" \
    --tactile "$OUTPUT_ROOT/test_heavy_mass2p0_live.json" \
    --zero "$OUTPUT_ROOT/test_heavy_mass2p0_zero.json" \
    --output "$OUTPUT_ROOT/test_heavy_mass2p0_pair.json"
"$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/summarize_native_tactile_common_horizon.py" \
    --tactile "$OUTPUT_ROOT/test_heavy_mass2p0_live.npz" \
    --zero "$OUTPUT_ROOT/test_heavy_mass2p0_zero.npz" \
    --output "$OUTPUT_ROOT/test_heavy_mass2p0_common.json"
"$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/summarize_native_tactile_frozen_pair.py" \
    --tactile "$OUTPUT_ROOT/test_low_friction0p25_0p20_live.json" \
    --zero "$OUTPUT_ROOT/test_low_friction0p25_0p20_zero.json" \
    --output "$OUTPUT_ROOT/test_low_friction0p25_0p20_pair.json"
"$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/summarize_native_tactile_common_horizon.py" \
    --tactile "$OUTPUT_ROOT/test_low_friction0p25_0p20_live.npz" \
    --zero "$OUTPUT_ROOT/test_low_friction0p25_0p20_zero.npz" \
    --output "$OUTPUT_ROOT/test_low_friction0p25_0p20_common.json"
"$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/summarize_native_tactile_teacher_residual_policy_gate.py" \
    --heavy-pair "$OUTPUT_ROOT/test_heavy_mass2p0_pair.json" \
    --heavy-common "$OUTPUT_ROOT/test_heavy_mass2p0_common.json" \
    --low-friction-pair "$OUTPUT_ROOT/test_low_friction0p25_0p20_pair.json" \
    --low-friction-common "$OUTPUT_ROOT/test_low_friction0p25_0p20_common.json" \
    --output "$OUTPUT_ROOT/report.json"
