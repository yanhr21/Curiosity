#!/usr/bin/env bash
# Causal-composer coverage test: dense train prefixes, interleaved unseen eval.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/causal_action_composition_dense_prefix_seed171646_v1}")"
DEVICE="${2:-cuda:0}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "dense-prefix causal composition requires a retained srun compute step" >&2
    exit 2
fi

export TRAIN_PREFIXES_CSV_OVERRIDE=33,41,49,57,65
export EVAL_PREFIXES_CSV_OVERRIDE=37,45,53,61
export REQUIRE_DISJOINT_PREFIX_SCHEDULES=1
export TRAIN_SEED_OVERRIDE=171646
export EVAL_SEED_OVERRIDE=181662
export VIDEO_SEED_OVERRIDE=181663
export REPLICATION_TRAIN_SEED_OVERRIDE=171647
export REPLICATION_EVAL_SEED_OVERRIDE=181664
export REPLICATION_VIDEO_SEED_OVERRIDE=181665
export REPLICATION_OUTPUT_ROOT="${REPLICATION_OUTPUT_ROOT:-${OUTPUT_ROOT}_seed171647_replication}"
export AGGREGATE_OUTPUT_ROOT="${AGGREGATE_OUTPUT_ROOT:-${OUTPUT_ROOT}_two_seed_aggregate}"

exec bash "$ROOT/scripts/sugar/demo_following/run_causal_action_composition_transition_recovery.sh" \
    "$OUTPUT_ROOT" "$DEVICE"
