#!/usr/bin/env bash
set -euo pipefail

# Launch Phase 00 MP4 video export inside an existing Curiosity-owned
# tmux-held H200 Slurm allocation. This launcher does not encode videos locally.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase00_video_h200_alloc}"
WINDOW_NAME="${WINDOW_NAME:-phase00_video_mp4_h200}"
RUN_TAG="${RUN_TAG:-phase00_video_mp4_export_h200_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/${RUN_TAG}.srun.log}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
VIDEO_EXPORT_VENV="${VIDEO_EXPORT_VENV:-$ROOT/envs/trex_dataset/.venv}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_ROW_FILES="${VIDEO_ROW_FILES:-$ROOT/experiments/outputs/phase00_core_asset_generation_h200_long_20260629_182052_phase00_cell_rows.jsonl:$ROOT/experiments/outputs/phase00_core_asset_generation_h200_long_repair2_20260629_183216_phase00_cell_rows.jsonl}"

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a currently running Curiosity tmux-held H200 Slurm allocation." >&2
  exit 2
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: required Curiosity tmux session not found: $TMUX_SESSION" >&2
  exit 3
fi
if ! squeue -h -j "$JOB_ID" >/dev/null 2>&1; then
  echo "ERROR: Slurm job $JOB_ID is not visible." >&2
  exit 4
fi
if [[ ! -x "$VIDEO_EXPORT_VENV/bin/python" ]]; then
  echo "ERROR: missing local video export venv at $VIDEO_EXPORT_VENV/bin/python." >&2
  exit 5
fi
if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 6
fi

env_file="$ROOT/logs/newton/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'VIDEO_EXPORT_VENV=%q\n' "$VIDEO_EXPORT_VENV"
  printf 'VIDEO_FPS=%q\n' "$VIDEO_FPS"
  printf 'VIDEO_ROW_FILES=%q\n' "$VIDEO_ROW_FILES"
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/run_phase00_video_mp4_export_h200_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nPHASE00_VIDEO_MP4_EXPORT_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
VIDEO_FPS=$VIDEO_FPS
VIDEO_EXPORT_VENV=$VIDEO_EXPORT_VENV
VIDEO_ROW_FILES=$VIDEO_ROW_FILES
REQUIRES_H200=true
DOWNSTREAM_USE=video_visualization_only_not_training_evidence
EOF
