#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase00_ref_tactile}"
WINDOW_NAME="${WINDOW_NAME:-p00_usd}"
RUN_TAG="${RUN_TAG:-p00_panda_usd_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/phase00/ref_tactile/newton_hydro/${RUN_TAG}.srun.log}"
NEWTON_ROOT="${NEWTON_ROOT:-$ROOT/external/newton_v1.3}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
DEVICE="${DEVICE:-cuda:0}"
NUM_FRAMES="${NUM_FRAMES:-240}"
SCENE="${SCENE:-cube}"
WORLD_COUNT="${WORLD_COUNT:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
VIS_DIR="${VIS_DIR:-$ROOT/experiments/visuals/phase00/ref_tactile/newton_hydro/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase00/ref_tactile/newton_hydro/$RUN_TAG}"

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

bash -n "$ROOT/experiments/configs/phase00/ref_tactile/run_newton_hydro_usd_in_alloc.sh"

env_file="$ROOT/logs/newton/phase00/ref_tactile/newton_hydro/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'NEWTON_ROOT=%q\n' "$NEWTON_ROOT"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'DEVICE=%q\n' "$DEVICE"
  printf 'NUM_FRAMES=%q\n' "$NUM_FRAMES"
  printf 'SCENE=%q\n' "$SCENE"
  printf 'WORLD_COUNT=%q\n' "$WORLD_COUNT"
  printf 'OUTPUT_DIR=%q\n' "$OUTPUT_DIR"
  printf 'VIS_DIR=%q\n' "$VIS_DIR"
  printf 'REPORT_DIR=%q\n' "$REPORT_DIR"
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/phase00/ref_tactile/run_newton_hydro_usd_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nPHASE00_NEWTON_HYDRO_USD_SRUN_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
DOWNSTREAM_USE=official_newton_hydro_visual_export_not_training_not_curiosity_success
EOF
