#!/usr/bin/env bash
# Train motion-disjoint official task priors and audit actual prefix-41 recovery.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
DEVICE="${1:-cuda:0}"
OUTPUT_ROOT="${2:-$ROOT/experiments/demo_following/taskwide_smp_v1}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing a fresh task-wide pipeline over existing output: $OUTPUT_ROOT" >&2
    exit 2
fi
for trace in \
    "$ROOT/experiments/demo_following/cross_skill_recovery_tinymdm_state_audit_v1/traces/released_baseline/trace.npz" \
    "$ROOT/experiments/demo_following/cross_skill_recovery_tinymdm_state_audit_v1/traces/unconstrained_update64/trace.npz" \
    "$ROOT/experiments/demo_following/cross_skill_recovery_tinymdm_state_audit_v1/traces/safety_update64/trace.npz"; do
    test -s "$trace"
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_taskwide_tinymdm_${SLURM_JOB_ID:-local}"

"$PYTHON_BIN" "$ROOT/scripts/sugar/smp/run_taskwide_tinymdm.py" all \
    --output-root "$OUTPUT_ROOT" --device "$DEVICE"
test -s "$OUTPUT_ROOT/priors/carry/model.pt"
test -s "$OUTPUT_ROOT/priors/kick/model.pt"
test -s "$OUTPUT_ROOT/motion_disjoint_score/RESULT.json"

"$PYTHON_BIN" "$ROOT/scripts/sugar/smp/audit_cross_skill_recovery_tinymdm.py" \
    --arm released_baseline "$ROOT/experiments/demo_following/cross_skill_recovery_tinymdm_state_audit_v1/traces/released_baseline/trace.npz" \
    --arm unconstrained_update64 "$ROOT/experiments/demo_following/cross_skill_recovery_tinymdm_state_audit_v1/traces/unconstrained_update64/trace.npz" \
    --arm safety_update64 "$ROOT/experiments/demo_following/cross_skill_recovery_tinymdm_state_audit_v1/traces/safety_update64/trace.npz" \
    --prior-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/recovery_score" \
    --device "$DEVICE" --chunk-size 256
test -s "$OUTPUT_ROOT/recovery_score/RESULT.json"
