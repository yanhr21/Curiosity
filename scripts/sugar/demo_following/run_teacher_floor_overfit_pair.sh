#!/usr/bin/env bash
# Run the one-seed fixed-physics teacher-floor learnability pair serially.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON=/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python
RUN_ROOT="$ROOT/experiments/demo_following/teacher_floor_overfit_v1/seed161581"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "run inside a retained Slurm GPU allocation" >&2
    exit 2
fi

for arm in correct unrelated; do
    proof="$RUN_ROOT/$arm/update_0128/proof.json"
    checkpoint="$RUN_ROOT/$arm/update_0128/policy.pt"
    if [[ -f "$proof" && -f "$checkpoint" ]]; then
        "$PYTHON" -c \
            'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' \
            "$proof"
        continue
    fi
    if [[ -e "$RUN_ROOT/$arm/update_0128" ]]; then
        echo "incomplete endpoint requires inspection: $RUN_ROOT/$arm/update_0128" >&2
        exit 2
    fi
    "$PYTHON" -u "$ROOT/scripts/sugar/demo_following/run_matched_state_predictor.py" \
        --design teacher_floor_overfit \
        --arm "$arm" \
        --endpoint-updates 128 \
        --stop-after-segment
    "$PYTHON" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and all(p["checks"].values())' \
        "$proof"
done

bash "$ROOT/scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh" \
    128 teacher_floor_overfit "$RUN_ROOT"

"$PYTHON" "$ROOT/scripts/sugar/demo_following/analyze_behavior_adherence.py" \
    --correct-trace "$RUN_ROOT/evaluation_update0128/correct/TRACE.npz" \
    --unrelated-trace "$RUN_ROOT/evaluation_update0128/unrelated/TRACE.npz" \
    --output-dir "$RUN_ROOT/behavior_adherence_audit_v1"

"$PYTHON" "$ROOT/scripts/sugar/demo_following/assess_teacher_floor_overfit.py" \
    --behavior-result "$RUN_ROOT/behavior_adherence_audit_v1/RESULT.json" \
    --output "$RUN_ROOT/TEACHER_FLOOR_GATE.json"

echo "completed_teacher_floor_overfit_seed=161581"
