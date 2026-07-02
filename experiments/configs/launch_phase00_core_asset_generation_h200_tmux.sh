#!/usr/bin/env bash
set -euo pipefail

# Launch Phase 00 core asset generation inside an existing Curiosity-owned
# tmux-held H200 Slurm allocation. This launcher does not run Newton locally.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase00_h200_asset_alloc}"
WINDOW_NAME="${WINDOW_NAME:-phase00_asset_h200}"
RUN_TAG="${RUN_TAG:-phase00_core_asset_generation_h200_$(date +%Y%m%d_%H%M%S)}"
CATALOG="${CATALOG:-$ROOT/experiments/configs/phase00_core_tabletop_asset_catalog_v1.json}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/${RUN_TAG}.srun.log}"
DEVICE="${DEVICE:-cuda:0}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
NEWTON_CACHE_PATH="${NEWTON_CACHE_PATH:-$ROOT/external/newton-assets-cache}"
PHASE00_MIN_NUM_STEPS="${PHASE00_MIN_NUM_STEPS:-1800}"
PHASE00_NUM_STEPS="${PHASE00_NUM_STEPS:-1800}"
PHASE00_PRE_RECORD_WARMUP_STEPS="${PHASE00_PRE_RECORD_WARMUP_STEPS:-60}"
PHASE00_FINAL_HOLD_DURATION="${PHASE00_FINAL_HOLD_DURATION:-12.0}"
PHASE00_HOLD_DURATION_MIN="${PHASE00_HOLD_DURATION_MIN:-8.0}"
PHASE00_VIDEO_FRAME_STRIDE="${PHASE00_VIDEO_FRAME_STRIDE:-3}"
PHASE00_VIDEO_FPS="${PHASE00_VIDEO_FPS:-20}"
PHASE00_CELL_FILTER="${PHASE00_CELL_FILTER:-}"

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/reports experiments/visuals

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
if [[ ! -f "$CATALOG" ]]; then
  echo "ERROR: missing Phase 00 catalog: $CATALOG" >&2
  exit 5
fi
if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv at $NEWTON_VENV/bin/python; prepare envs on shared filesystem before compute use." >&2
  exit 6
fi

if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 7
fi

env_file="$ROOT/logs/newton/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  printf 'CATALOG=%q\n' "$CATALOG"
  printf 'DEVICE=%q\n' "$DEVICE"
  printf 'NEWTON_VENV=%q\n' "$NEWTON_VENV"
  printf 'TRAINER_VENV=%q\n' "$TRAINER_VENV"
  printf 'NEWTON_CACHE_PATH=%q\n' "$NEWTON_CACHE_PATH"
  printf 'PHASE00_MIN_NUM_STEPS=%q\n' "$PHASE00_MIN_NUM_STEPS"
  printf 'PHASE00_NUM_STEPS=%q\n' "$PHASE00_NUM_STEPS"
  printf 'PHASE00_PRE_RECORD_WARMUP_STEPS=%q\n' "$PHASE00_PRE_RECORD_WARMUP_STEPS"
  printf 'PHASE00_FINAL_HOLD_DURATION=%q\n' "$PHASE00_FINAL_HOLD_DURATION"
  printf 'PHASE00_HOLD_DURATION_MIN=%q\n' "$PHASE00_HOLD_DURATION_MIN"
  printf 'PHASE00_VIDEO_FRAME_STRIDE=%q\n' "$PHASE00_VIDEO_FRAME_STRIDE"
  printf 'PHASE00_VIDEO_FPS=%q\n' "$PHASE00_VIDEO_FPS"
  printf 'PHASE00_CELL_FILTER=%q\n' "$PHASE00_CELL_FILTER"
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/run_phase00_core_asset_generation_h200_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nPHASE00_H200_ASSET_GENERATION_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
CATALOG=$CATALOG
DEVICE=$DEVICE
PHASE00_NUM_STEPS=$PHASE00_NUM_STEPS
PHASE00_FINAL_HOLD_DURATION=$PHASE00_FINAL_HOLD_DURATION
PHASE00_HOLD_DURATION_MIN=$PHASE00_HOLD_DURATION_MIN
PHASE00_VIDEO_FRAME_STRIDE=$PHASE00_VIDEO_FRAME_STRIDE
PHASE00_VIDEO_FPS=$PHASE00_VIDEO_FPS
PHASE00_CELL_FILTER=$PHASE00_CELL_FILTER
REQUIRES_H200=true
DOWNSTREAM_USE=blocked_until_h200_visual_contact_validation_and_manual_inspection_pass
EOF
