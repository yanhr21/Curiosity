#!/usr/bin/env bash
# Same failure-rich curve with one causal current-rollout recovery objective.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${1:-$ROOT/experiments/demo_following/transition_recovery_objective_overfit_seed181630_v1}"
DEVICE="${2:-cuda:0}"

export SUGAR_TRANSITION_RECOVERY_REWARD_OVERRIDE=1
exec bash "$ROOT/scripts/sugar/demo_following/run_frozen_expert_transition_failure_overfit.sh" \
    "$OUTPUT_ROOT" "$DEVICE"
