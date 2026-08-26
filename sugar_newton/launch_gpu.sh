#!/usr/bin/env bash
# Launch the Newton validators in the foreground of an existing H200 allocation.
# Run this from a tmux pane; interrupt/restart the foreground command with Ctrl+C.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 H200_JOB_ID" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec srun --overlap --jobid="$1" --nodes=1 --ntasks=1 \
  bash "$repo_root/sugar_newton/gpu_run.sh"
