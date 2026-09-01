#!/bin/bash
# Attach a command to the 8-GPU dev node (see renders/devnode8.sbatch).
set -u
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
STATE=$REPO/renders/.devnode8
[ -f "$STATE" ] || { echo "no 8-GPU dev node: $STATE missing — sbatch renders/devnode8.sbatch"; exit 1; }
JID=$(sed -n 's/^jobid=//p' "$STATE")
squeue -j "$JID" -h -o %T 2>/dev/null | grep -q RUNNING || { echo "dev node job $JID is not RUNNING"; exit 1; }
exec srun --jobid="$JID" --overlap --container-name=nwdev8 bash -lc "cd $REPO && $*"
