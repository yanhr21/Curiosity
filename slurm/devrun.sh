#!/bin/bash
# Run a command inside the persistent dev-node container (see slurm/devnode.sbatch).
#
#   bash slurm/devrun.sh "nvidia-smi -L"
#   bash slurm/devrun.sh "source env/activate.sh && python -m sugar_newton.validation.g1_carrybox_policy"
#
# The command runs with the repo as its working directory, so `source env/activate.sh` and
# every other path in these docs is relative to the repo root and needs no editing.
#
# Reads the jobid from slurm/.devnode, written by the holder job when it becomes READY. Set
# RB_STATE=slurm/.devnode8 to target the 8-GPU node instead.
set -u
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
STATE=${RB_STATE:-$REPO/slurm/.devnode}
case "$STATE" in /*) ;; *) STATE=$REPO/$STATE;; esac
[ -f "$STATE" ] || { echo "no dev node: $STATE missing -- sbatch slurm/devnode.sbatch first"; exit 1; }
JID=$(sed -n 's/^jobid=//p' "$STATE")
NAME=$(sed -n 's/^container=//p' "$STATE")
squeue -j "$JID" -h -o %T 2>/dev/null | grep -q RUNNING || {
    echo "dev node job $JID is not RUNNING -- sbatch slurm/devnode.sbatch for a new one"; exit 1; }

# NB: pyxis ignores --container-mounts when attaching to a running container; the mounts are
# inherited from the holder job, which is why devnode.sbatch has to get them right.
exec srun --jobid="$JID" --overlap --container-name="${NAME:-rbdev}" \
     bash -lc "cd $REPO && $*"
