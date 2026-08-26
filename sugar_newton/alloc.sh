#!/usr/bin/env bash
# Enter an already allocated Slurm job. Run this in a dedicated tmux pane.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 H200_JOB_ID" >&2
  exit 2
fi

exec srun --overlap --jobid="$1" --nodes=1 --ntasks=1 --pty bash -l
