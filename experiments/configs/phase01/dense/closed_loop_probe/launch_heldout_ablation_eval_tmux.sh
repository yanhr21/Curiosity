#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase01_dense_probe_eval}"
WINDOW_NAME="${WINDOW_NAME:-heldout_ablation}"
RUN_TAG="${RUN_TAG:-p01_dense_clprobe_heldout_ablation_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/phase01/dense/closed_loop_probe/heldout_ablation/${RUN_TAG}.srun.log}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_PATH")"

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a running Curiosity tmux-held allocation." >&2
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

bash -n "$ROOT/experiments/configs/phase01/dense/closed_loop_probe/run_heldout_ablation_eval_in_alloc.sh"

env_file="$ROOT/logs/newton/phase01/dense/closed_loop_probe/heldout_ablation/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  for name in PYTHON_BIN CHECKPOINT OUTPUT_DIR REPORT_DIR NUM_FRAMES MAP_SIZE REPETITIONS SEED SCORE_LIFT_WEIGHT SCORE_HOLD_WEIGHT SCORE_DROP_WEIGHT HOLD_LIFT_THRESHOLD FEATURE_NOISE_STD CELLS; do
    if [[ -n "${!name:-}" ]]; then
      printf '%s=%q\n' "$name" "${!name}"
    fi
  done
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/phase01/dense/closed_loop_probe/run_heldout_ablation_eval_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=4 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; status=\"\${PIPESTATUS[0]}\"; printf '\nPHASE01_DENSE_HELDOUT_ABLATION_EVAL_SRUN_EXIT=%s\n' \"\$status\"; exit \"\$status\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
DOWNSTREAM_USE=phase01_dense_heldout_ablation_eval
EOF
