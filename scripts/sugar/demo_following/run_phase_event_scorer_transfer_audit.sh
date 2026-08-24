#!/usr/bin/env bash
# Recollect exact frozen-policy predictor prefixes and run the phase-only audit.

set -euo pipefail

ROOT=/public/home/yanhongru/Curiosity
PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
SEED_ROOT="$ROOT/experiments/demo_following/matched_phase_event_reward_v1/seed161587"
OUTPUT_ROOT=${OUTPUT_ROOT:-$SEED_ROOT}
TRACE_ROOT="$OUTPUT_ROOT/scorer_transfer_source_trace_v1"
AUDIT_ROOT="$OUTPUT_ROOT/scorer_transfer_phase_ablation_v1"
EVALUATOR="$ROOT/scripts/sugar/demo_following/evaluate_matched_fixed_teacher.py"
AUDITOR="$ROOT/scripts/sugar/demo_following/audit_phase_event_scorer_transfer.py"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "scorer transfer recollection requires a retained Slurm allocation" >&2
    exit 2
fi
if [[ -e "$TRACE_ROOT" || -e "$AUDIT_ROOT" ]]; then
    echo "refusing to overwrite scorer transfer evidence" >&2
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
    export ISAACLAB_TMP_ROOT="/tmp/Curiosity_phase_transfer_${arm}_${SLURM_JOB_ID}"
    export SUGAR_UNITREE_TMP_ROOT="/tmp/Curiosity_phase_transfer_unitree_${arm}_${SLURM_JOB_ID}"
    (
        cd "$ROOT/SUGAR"
        "$PYTHON_BIN" -u "$EVALUATOR" \
            --config "$config" \
            --arm "$evaluator_arm" \
            --output-dir "$TRACE_ROOT/$arm" \
            --updates 32,64 \
            --steps 400 \
            --seed 171587 \
            --phase-initialization reset-zero-diagnostic \
            --headless \
            --device cuda:0 \
            --kit_args \
            '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1'
    )
done

cd "$ROOT"
"$PYTHON_BIN" -u "$AUDITOR" \
    --evaluation-root "$TRACE_ROOT" \
    --output-dir "$AUDIT_ROOT" \
    --device cuda:0

echo "SCORER_TRANSFER_AUDIT_COMPLETE result=$AUDIT_ROOT/RESULT.json"
