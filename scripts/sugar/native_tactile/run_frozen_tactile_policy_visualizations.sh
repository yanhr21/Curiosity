#!/usr/bin/env bash
set -euo pipefail

# Reproduce the matched live/zeroed/anatomical-permutation frozen-policy
# CarryBox videos inside an existing retained Slurm allocation.  The three
# IsaacLab rollouts are deliberately serial.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${CURIOSITY_ISAAC_PYTHON:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run inside an existing retained Slurm allocation." >&2
  exit 2
fi
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT OUTPUT_ROOT" >&2
  exit 2
fi

CHECKPOINT="$1"
OUTPUT_ROOT="$2"
if [[ "$CHECKPOINT" != /* ]]; then CHECKPOINT="$ROOT/$CHECKPOINT"; fi
if [[ "$OUTPUT_ROOT" != /* ]]; then OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"; fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi
case "$OUTPUT_ROOT" in
  "$ROOT"/experiments/*) ;;
  *) echo "OUTPUT_ROOT must remain below $ROOT/experiments" >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing overwrite: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

for MODE in live zeroed patch_permuted; do
  bash "$ROOT/scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh" \
    tactile \
    "$CHECKPOINT" \
    "$OUTPUT_ROOT/${MODE}_evaluation.json" \
    --condition_label nominal \
    --actor_tactile_mode "$MODE" \
    --tactile_permutation_seed 13012 \
    --record_bundle "$OUTPUT_ROOT/${MODE}_bundle" \
    --enable_cameras
done

SCALE_ARGS=()
for MODE in live zeroed patch_permuted; do
  SCALE_ARGS+=(--scale-bundle-root "$OUTPUT_ROOT/${MODE}_bundle")
done

for MODE in live zeroed patch_permuted; do
  TITLE="SUGAR CarryBox frozen tactile policy - ${MODE//_/ } input"
  "$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/render_native_tactile_policy_rollout.py" \
    --bundle-root "$OUTPUT_ROOT/${MODE}_bundle" \
    --output "$OUTPUT_ROOT/${MODE}_policy_world_and_bilateral_tactile.mp4" \
    --title "$TITLE" \
    --fps 50 \
    "${SCALE_ARGS[@]}"
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/summarize_native_tactile_dependence.py" \
  --live "$OUTPUT_ROOT/live_evaluation.json" \
  --zeroed "$OUTPUT_ROOT/zeroed_evaluation.json" \
  --patch-permuted "$OUTPUT_ROOT/patch_permuted_evaluation.json" \
  --output "$OUTPUT_ROOT/summary.json"

echo "Complete frozen-policy tactile visualization cohort: $OUTPUT_ROOT"
