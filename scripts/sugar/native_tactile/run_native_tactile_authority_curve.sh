#!/usr/bin/env bash
set -euo pipefail

# Reproduce the no-learning tactile-column authority response curve. Run as a
# recorded child inside an existing retained Slurm allocation.

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

LABELS=(000 025 050 075 100)
SCALES=(0.0 0.25 0.5 0.75 1.0)
for INDEX in "${!SCALES[@]}"; do
  bash "$ROOT/scripts/sugar/native_tactile/run_native_tactile_bcppo_evaluation.sh" \
    tactile \
    "$CHECKPOINT" \
    "$OUTPUT_ROOT/scale${LABELS[$INDEX]}.json" \
    --condition_label nominal \
    --actor_tactile_mode live \
    --tactile_authority_scale "${SCALES[$INDEX]}" \
    --tactile_permutation_seed 13012
done

"$PYTHON_BIN" "$ROOT/scripts/sugar/native_tactile/summarize_native_tactile_authority_curve.py" \
  --scale-zero "$OUTPUT_ROOT/scale000.json" \
  --scale-025 "$OUTPUT_ROOT/scale025.json" \
  --scale-050 "$OUTPUT_ROOT/scale050.json" \
  --scale-075 "$OUTPUT_ROOT/scale075.json" \
  --scale-100 "$OUTPUT_ROOT/scale100.json" \
  --output "$OUTPUT_ROOT/summary.json"

echo "Complete frozen tactile-authority curve: $OUTPUT_ROOT"
