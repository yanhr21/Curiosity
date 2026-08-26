#!/usr/bin/env bash
# Render the full clean SUGAR corpus, train official XIRL/TCC, evaluate, then retain the GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_xirl_tcc_v1}")"
CORPUS="$OUTPUT/corpus"
RUNS="$OUTPUT/pretrain_runs"
EXPERIMENT_NAME="${2:-sugar_carry_kick_tcc_seed271402}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside Slurm GPU compute." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Official XIRL pipeline requires a Slurm GPU job" >&2
    exit 2
fi

corpus_complete() {
    jq -e '.passed == true and (.frame_counts | length == 100 and all(. == 64))' \
        "$CORPUS/RENDER_RESULT_CarryBox_000_099.json" >/dev/null 2>&1 &&
    jq -e '.passed == true and (.frame_counts | length == 99 and all(. == 64))' \
        "$CORPUS/RENDER_RESULT_KickBox_000_098.json" >/dev/null 2>&1
}

if ! corpus_complete; then
    if [[ -d "$CORPUS" ]] && find "$CORPUS" -type f -name '*.png' -print -quit | grep -q .; then
        echo "partial immutable XIRL corpus exists; use a new output root" >&2
        exit 2
    fi
    XIRL_SKIP_HOLD_AFTER_RENDER=1 \
        bash "$ROOT/scripts/sugar/demo_following/run_xirl_full_reference_corpus_then_hold.sh" \
        "$CORPUS"
fi

bash "$ROOT/scripts/sugar/demo_following/run_official_xirl_tcc_pretrain_then_hold.sh" \
    "$CORPUS" "$RUNS" "$EXPERIMENT_NAME"
