#!/usr/bin/env bash
# Fixed multi-context diagnostic for state-dependent exact Carry/Kick composition.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/causal_action_composition_seed171644_v1}"
DEVICE="${2:-cuda:0}"

export POLICY_TOPOLOGY_OVERRIDE=causal_action_composition
export TRAIN_SEED_OVERRIDE="${TRAIN_SEED_OVERRIDE:-171644}"
export EVAL_SEED_OVERRIDE="${EVAL_SEED_OVERRIDE:-181656}"
export VIDEO_SEED_OVERRIDE="${VIDEO_SEED_OVERRIDE:-181657}"

exec bash "$ROOT/scripts/sugar/demo_following/run_multi_context_transition_recovery.sh" \
    "$OUTPUT_ROOT" "$DEVICE"
