#!/usr/bin/env bash
# Render live-versus-zero behavior for the two held-out physical conditions.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run inside an existing retained Slurm allocation." >&2
    exit 2
fi
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 CHECKPOINT OUTPUT_ROOT" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
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

evaluate_pair() {
    local label="$1"
    shift
    bash "$ROOT/scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh" \
        residual_tactile "$CHECKPOINT" "$OUTPUT_ROOT/${label}_live.json" \
        --condition_label "$label" \
        --actor_tactile_mode live \
        --record_bundle "$OUTPUT_ROOT/${label}_live_bundle" \
        --enable_cameras "$@"
    bash "$ROOT/scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh" \
        residual_zero "$CHECKPOINT" "$OUTPUT_ROOT/${label}_zero.json" \
        --condition_label "$label" \
        --record_bundle "$OUTPUT_ROOT/${label}_zero_bundle" \
        --enable_cameras "$@"
}

evaluate_pair heldout_heavy_1p0kg --mass_scale 2.0
evaluate_pair heldout_low_friction_0p5kg \
    --mass_scale 1.0 --static_friction 0.25 --dynamic_friction 0.20

SCALE_ARGS=()
for label in heldout_heavy_1p0kg heldout_low_friction_0p5kg; do
    SCALE_ARGS+=(--scale-bundle-root "$OUTPUT_ROOT/${label}_live_bundle")
    SCALE_ARGS+=(--scale-bundle-root "$OUTPUT_ROOT/${label}_zero_bundle")
done

for label in heldout_heavy_1p0kg heldout_low_friction_0p5kg; do
    "$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/render_native_tactile_policy_rollout.py" \
        --bundle-root "$OUTPUT_ROOT/${label}_live_bundle" \
        --output "$OUTPUT_ROOT/${label}_live_world_and_tactile.mp4" \
        --title "${label}: selected adapter with LIVE tactile" \
        --fps 50 "${SCALE_ARGS[@]}"
    "$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/render_native_tactile_policy_rollout.py" \
        --bundle-root "$OUTPUT_ROOT/${label}_zero_bundle" \
        --output "$OUTPUT_ROOT/${label}_zero_world_and_tactile.mp4" \
        --title "${label}: same checkpoint with EXACT-ZERO tactile" \
        --fps 50 "${SCALE_ARGS[@]}"
    "$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/compose_native_tactile_policy_pair.py" \
        --left "$OUTPUT_ROOT/${label}_live_world_and_tactile.mp4" \
        --right "$OUTPUT_ROOT/${label}_zero_world_and_tactile.mp4" \
        --output "$OUTPUT_ROOT/${label}_live_vs_zero.mp4" \
        --fps 50
done

printf 'videos=%s\n' "$OUTPUT_ROOT"
