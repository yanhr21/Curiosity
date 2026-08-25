#!/usr/bin/env bash
# Formal shared-checkpoint run with the admitted causal recovery objective.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${1:?usage: run_shared_transition_recovery_objective.sh OUTPUT_ROOT [DEVICE]}"
DEVICE="${2:-cuda:0}"

export SUGAR_TRANSITION_RECOVERY_REWARD_OVERRIDE=1
exec bash "$ROOT/scripts/sugar/demo_following/run_shared_frozen_expert_transition.sh" \
    "$OUTPUT_ROOT" "$DEVICE"
