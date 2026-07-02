#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
SESSION="${SESSION:-curiosity_phase01_h200}"
WINDOW_NAME="${WINDOW_NAME:-p01_base}"
SLURM_JOB_ID="${SLURM_JOB_ID:-}"
RUN_TAG="${RUN_TAG:-p01_base_$(date +%Y%m%d_%H%M%S)}"
ALLOC_CPUS_PER_TASK="${ALLOC_CPUS_PER_TASK:-16}"

if [[ -z "$SLURM_JOB_ID" ]]; then
  echo "ERROR: set SLURM_JOB_ID to an existing Curiosity H200 allocation." >&2
  exit 2
fi
if ! squeue -h -j "$SLURM_JOB_ID" >/dev/null; then
  echo "ERROR: Slurm job $SLURM_JOB_ID is not visible." >&2
  exit 3
fi
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: missing tmux session $SESSION." >&2
  exit 4
fi

cd "$ROOT"
mkdir -p logs/newton/phase01/core/baselines
log_path="$ROOT/logs/newton/phase01/core/baselines/${RUN_TAG}.srun.log"

tmux new-window -t "$SESSION" -n "$WINDOW_NAME" \
  "cd '$ROOT' && RUN_TAG='$RUN_TAG' srun --jobid='$SLURM_JOB_ID' --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task='$ALLOC_CPUS_PER_TASK' --gres=gpu:1 bash '$ROOT/experiments/configs/phase01/run_baselines_in_alloc.sh' 2>&1 | tee '$log_path'; echo PHASE01_BASELINES_SRUN_EXIT=\${PIPESTATUS[0]}"

echo "Launched Phase 01 baseline evaluation in tmux session=$SESSION window=$WINDOW_NAME"
echo "RUN_TAG=$RUN_TAG"
echo "LOG=$log_path"
