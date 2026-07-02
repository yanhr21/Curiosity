#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase01_local_adv_h200}"
WINDOW_NAME="${WINDOW_NAME:-p01_local_adv}"
RUN_TAG="${RUN_TAG:-p01_local_adv_$(date +%Y%m%d_%H%M%S)}"
SEGMENT_CONFIG="${SEGMENT_CONFIG:-$ROOT/experiments/configs/phase01/local_adv_segments.json}"
LP_SCORE_CONFIG="${LP_SCORE_CONFIG:-$ROOT/experiments/configs/phase01/local_adv_lp_scores.json}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
DEVICE="${DEVICE:-cuda:0}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/phase01/core/local_adv/${RUN_TAG}.srun.log}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_PATH")"

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a running Curiosity tmux-held H200 allocation." >&2
  exit 2
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session not found: $TMUX_SESSION" >&2
  exit 3
fi
job_state="$(squeue -h -j "$JOB_ID" -o '%T' | head -n 1)"
if [[ "$job_state" != "RUNNING" ]]; then
  echo "ERROR: Slurm job $JOB_ID is not running yet." >&2
  squeue -j "$JOB_ID" -o '%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R' >&2 || true
  exit 4
fi
if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 5
fi
bash -n "$ROOT/experiments/configs/phase01/run_local_adv_segments_in_alloc.sh"

env_file="$ROOT/logs/newton/phase01/core/local_adv/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'SEGMENT_CONFIG=%q\n' "$SEGMENT_CONFIG"
  printf 'LP_SCORE_CONFIG=%q\n' "$LP_SCORE_CONFIG"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'TRAINER_VENV=%q\n' "$TRAINER_VENV"
  printf 'DEVICE=%q\n' "$DEVICE"
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/phase01/run_local_adv_segments_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nPHASE01_LOCAL_ADV_SRUN_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
SEGMENT_CONFIG=$SEGMENT_CONFIG
LP_SCORE_CONFIG=$LP_SCORE_CONFIG
DOWNSTREAM_USE=local_advantage_segment_preflight_before_final_allowed_training
EOF
