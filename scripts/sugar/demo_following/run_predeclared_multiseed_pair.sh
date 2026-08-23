#!/usr/bin/env bash
# Run one predeclared same-teacher correct/unrelated seed pair serially.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
SEED=${1:-}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "run inside a retained Slurm GPU allocation" >&2
    exit 2
fi

case "$SEED" in
    161583) ACTION_SEED=161584 ;;
    161585) ACTION_SEED=161586 ;;
    *) echo "expected remaining predeclared seed 161583 or 161585" >&2; exit 2 ;;
esac

RUN_ROOT="$ROOT/experiments/demo_following/matched_reward_identity_same_teacher_v1/seed${SEED}"
for arm in correct unrelated; do
    proof="$RUN_ROOT/$arm/update_0064/proof.json"
    checkpoint="$RUN_ROOT/$arm/update_0064/policy.pt"
    if [[ -f "$proof" && -f "$checkpoint" ]]; then
        "$PYTHON" -c \
            'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' \
            "$proof"
        continue
    fi
    if [[ -e "$RUN_ROOT/$arm/update_0064" ]]; then
        echo "incomplete endpoint requires inspection: $RUN_ROOT/$arm/update_0064" >&2
        exit 2
    fi
    "$PYTHON" -u "$ROOT/scripts/sugar/demo_following/run_matched_state_predictor.py" \
        --design same_teacher_reward_only \
        --arm "$arm" \
        --seed "$SEED" \
        --action-seed "$ACTION_SEED" \
        --endpoint-updates 64 \
        --stop-after-segment
    "$PYTHON" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' \
        "$proof"
done

bash "$ROOT/scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh" \
    64 same_teacher_reward_only "$RUN_ROOT"

"$PYTHON" "$ROOT/scripts/sugar/demo_following/analyze_behavior_adherence.py" \
    --correct-trace "$RUN_ROOT/evaluation_update0064/correct/TRACE.npz" \
    --unrelated-trace "$RUN_ROOT/evaluation_update0064/unrelated/TRACE.npz" \
    --output-dir "$RUN_ROOT/behavior_adherence_audit_v1"

echo "completed_predeclared_seed=$SEED"
