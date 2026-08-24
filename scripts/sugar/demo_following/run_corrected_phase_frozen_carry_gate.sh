#!/usr/bin/env bash
# Scorer-only re-evaluation of the historical phase-misaligned Carry policies.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
SEED_ROOT="$ROOT/experiments/demo_following/matched_phase_event_reward_v1/seed161587"
OUTPUT_ROOT=${OUTPUT_ROOT:-$SEED_ROOT/corrected_phase_frozen_carry_gate_v1}
TRACE_ROOT="$OUTPUT_ROOT/source_evaluations"
AUDIT_ROOT="$OUTPUT_ROOT/scorer_audit"
EVALUATOR="$ROOT/scripts/sugar/demo_following/evaluate_matched_fixed_teacher.py"
ASSESSOR="$ROOT/scripts/sugar/demo_following/assess_corrected_phase_frozen_carry_gate.py"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "corrected frozen Carry gate requires a retained srun compute step" >&2
    exit 2
fi
case "$(hostname)" in
    mgmtserver*|login*) echo "refusing corrected Carry gate on a login host" >&2; exit 2 ;;
esac
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "refusing to overwrite corrected Carry gate: $OUTPUT_ROOT" >&2
    exit 2
fi

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export DISPLAY=
export PYTHONPATH="$ROOT/scripts/sugar/smp:$ROOT/IsaacLab/source/isaaclab_contrib:$ROOT/IsaacLab/source/isaaclab_assets:$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il"

mkdir -p "$TRACE_ROOT"
for arm in correct unrelated; do
    if [[ "$arm" == correct ]]; then
        evaluator_arm=same_teacher_correct_reward
    else
        evaluator_arm=same_teacher_unrelated_reward
    fi
    config="$SEED_ROOT/$arm/update_0064/protocol.json"
    export ISAACLAB_TMP_ROOT="/tmp/Curiosity_corrected_phase_${arm}_${SLURM_JOB_ID}"
    export SUGAR_UNITREE_TMP_ROOT="/tmp/Curiosity_corrected_phase_unitree_${arm}_${SLURM_JOB_ID}"
    (
        cd "$ROOT/SUGAR"
        "$PYTHON_BIN" -u "$EVALUATOR" \
            --config "$config" \
            --arm "$evaluator_arm" \
            --output-dir "$TRACE_ROOT/$arm" \
            --updates 32,64 \
            --steps 400 \
            --seed 171587 \
            --phase-initialization reference-aware \
            --headless \
            --device cuda:0 \
            --kit_args \
            '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1'
    )
done

cd "$ROOT"
"$PYTHON_BIN" -u "$ASSESSOR" \
    --evaluation-root "$TRACE_ROOT" \
    --output-dir "$AUDIT_ROOT"

echo "CORRECTED_PHASE_FROZEN_CARRY_GATE_COMPLETE result=$AUDIT_ROOT/RESULT.json"
