#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase01_dense_real}"
WINDOW_NAME="${WINDOW_NAME:-alloc}"
JOB_NAME="${JOB_NAME:-curiosity_p01_dense_real_attempt001_1gpu}"
PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:NVIDIAH200:1}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase01/dense/closed_loop_probe/real_allocation}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).log}"

cd "$ROOT"
mkdir -p "$LOG_DIR"

if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  tmux new-session -d -s "$TMUX_SESSION" -n bootstrap "cd $(printf '%q' "$ROOT") && bash -l"
fi
if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 2
fi

alloc_cmd=$(
  cat <<EOF
cd $(printf '%q' "$ROOT")
echo PHASE01_DENSE_REAL_ATTEMPT_ALLOCATION_REQUEST_START
echo ROOT=$(printf '%q' "$ROOT")
echo JOB_NAME=$(printf '%q' "$JOB_NAME")
echo PARTITION=$(printf '%q' "$PARTITION")
echo GRES=$(printf '%q' "$GRES")
echo TIME_LIMIT=$(printf '%q' "$TIME_LIMIT")
echo LOG_PATH=$(printf '%q' "$LOG_PATH")
srun --partition=$(printf '%q' "$PARTITION") \
  --job-name=$(printf '%q' "$JOB_NAME") \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=$(printf '%q' "$CPUS_PER_TASK") \
  --mem=$(printf '%q' "$MEMORY") \
  --gres=$(printf '%q' "$GRES") \
  --time=$(printf '%q' "$TIME_LIMIT") \
  --pty bash -lc 'echo PHASE01_DENSE_REAL_ATTEMPT_ALLOCATION_GRANTED; echo SLURM_JOB_ID=\$SLURM_JOB_ID; echo HOSTNAME=\$(hostname); nvidia-smi --query-gpu=name,index,memory.total --format=csv; cd /public/home/yanhongru/Curiosity; bash -l'
echo PHASE01_DENSE_REAL_ATTEMPT_ALLOCATION_EXIT=\$?
EOF
)

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" "$alloc_cmd 2>&1 | tee '$LOG_PATH'" C-m

cat <<EOF
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_NAME=$JOB_NAME
PARTITION=$PARTITION
GRES=$GRES
TIME_LIMIT=$TIME_LIMIT
LOG_PATH=$LOG_PATH
NEXT_CHECK=squeue -n "$JOB_NAME"
DOWNSTREAM_USE=held_allocation_for_phase01_dense_real_attempt_001
EOF
