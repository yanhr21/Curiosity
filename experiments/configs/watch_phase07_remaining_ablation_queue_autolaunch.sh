#!/usr/bin/env bash
set -euo pipefail

# Lightweight login-node watcher. It does not run training or rendering. It
# waits for a tmux-held Slurm allocation to become RUNNING, then launches the
# Phase07 remaining-ablation queue into that allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-155039}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_contact_ablation_alloc_20260627_222339}"
RUN_TAG="${RUN_TAG:-phase07_remaining_ablation_queue_20260627}"
WINDOW_NAME="${WINDOW_NAME:-phase07_remaining_ablation_queue}"
DEVICE="${DEVICE:-cuda:0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
ABLATED_VARIANTS="${ABLATED_VARIANTS:-contact_only shuffled_contact delayed_contact no_learning_progress}"

cd "$ROOT"
mkdir -p logs/newton

echo "PHASE07_ABLATION_QUEUE_AUTOLAUNCH_START"
echo "ROOT=$ROOT"
echo "JOB_ID=$JOB_ID"
echo "TMUX_SESSION=$TMUX_SESSION"
echo "RUN_TAG=$RUN_TAG"
echo "WINDOW_NAME=$WINDOW_NAME"
echo "DEVICE=$DEVICE"
echo "POLL_SECONDS=$POLL_SECONDS"
echo "ABLATED_VARIANTS=$ABLATED_VARIANTS"
echo "NOTE=login_node_watcher_only_no_training_or_rendering"

while true; do
  state="$(squeue -h -j "$JOB_ID" -o '%T' | head -n 1 || true)"
  reason="$(squeue -h -j "$JOB_ID" -o '%R' | head -n 1 || true)"
  timestamp="$(date -Is)"
  echo "$timestamp JOB_ID=$JOB_ID STATE=${state:-missing} REASON=${reason:-missing}"
  if [[ "$state" == "RUNNING" ]]; then
    echo "$timestamp launching remaining ablation queue"
    JOB_ID="$JOB_ID" \
    TMUX_SESSION="$TMUX_SESSION" \
    RUN_TAG="$RUN_TAG" \
    WINDOW_NAME="$WINDOW_NAME" \
    DEVICE="$DEVICE" \
    ABLATED_VARIANTS="$ABLATED_VARIANTS" \
      bash "$ROOT/experiments/configs/launch_phase07_remaining_ablation_queue_tmux.sh"
    echo "$timestamp autolaunch complete"
    exit 0
  fi
  if [[ -z "$state" ]]; then
    echo "$timestamp ERROR: job missing from squeue; stopping autolaunch watcher" >&2
    exit 2
  fi
  sleep "$POLL_SECONDS"
done
