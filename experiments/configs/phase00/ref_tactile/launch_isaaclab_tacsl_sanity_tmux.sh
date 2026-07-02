#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase00_ref_tactile}"
WINDOW_NAME="${WINDOW_NAME:-p00_tacsl_sanity}"
RUN_TAG="${RUN_TAG:-p00_tacsl_sanity_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/phase00/ref_tactile/reference_sanity/${RUN_TAG}.srun.log}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_PATH")"

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a running Curiosity tmux-held allocation." >&2
  exit 2
fi

target_python="${ISAACLAB_TACSL_PYTHON:-}"
if [[ -z "$target_python" && -x "$ROOT/envs/isaaclab_tacsl/conda/bin/python" ]]; then
  target_python="$ROOT/envs/isaaclab_tacsl/conda/bin/python"
fi
if [[ -z "$target_python" && -x "$ROOT/envs/isaaclab_tacsl/.venv/bin/python" ]]; then
  target_python="$ROOT/envs/isaaclab_tacsl/.venv/bin/python"
fi
if [[ -z "$target_python" && "${ALLOW_MISSING_TACSL_ENV_BLOCKER_RUN:-0}" != "1" ]]; then
  echo "ERROR: missing executable approved IsaacLab TacSL environment." >&2
  echo "Set ISAACLAB_TACSL_PYTHON, prepare envs/isaaclab_tacsl/conda or envs/isaaclab_tacsl/.venv, or set ALLOW_MISSING_TACSL_ENV_BLOCKER_RUN=1 to intentionally record a compute-side blocker." >&2
  exit 7
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session not found: $TMUX_SESSION" >&2
  exit 4
fi
job_state="$(squeue -h -j "$JOB_ID" -o '%T' | head -n 1)"
if [[ "$job_state" != "RUNNING" ]]; then
  echo "ERROR: Slurm job $JOB_ID is not running yet." >&2
  squeue -j "$JOB_ID" -o '%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R' >&2 || true
  exit 5
fi
if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 6
fi

bash -n "$ROOT/experiments/configs/phase00/ref_tactile/run_isaaclab_tacsl_sanity_in_alloc.sh"

env_file="$ROOT/logs/newton/phase00/ref_tactile/reference_sanity/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  if [[ -n "${ISAACLAB_TACSL_PYTHON:-}" ]]; then
    printf 'ISAACLAB_TACSL_PYTHON=%q\n' "$ISAACLAB_TACSL_PYTHON"
  fi
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/phase00/ref_tactile/run_isaaclab_tacsl_sanity_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nISAACLAB_TACSL_SANITY_SRUN_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TARGET=official_isaaclab_tacsl
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
DOWNSTREAM_USE=official_isaaclab_tacsl_sanity_or_blocker_not_training_not_curiosity_success
EOF
