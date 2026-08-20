#!/bin/bash
# Run a command inside the persistent dev-node container (see renders/devnode.sbatch).
#   bash renders/devrun.sh "nvidia-smi -L"
#   bash renders/devrun.sh "source renders/render_env.sh && uv run python example_g1_in_sage.py --help"
# Reads the jobid from renders/.devnode (written by the holder job when it becomes READY).
set -u
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
STATE=$REPO/renders/.devnode
[ -f "$STATE" ] || { echo "no dev node: $STATE missing — sbatch renders/devnode.sbatch first"; exit 1; }
JID=$(sed -n 's/^jobid=//p' "$STATE")
squeue -j "$JID" -h -o %T 2>/dev/null | grep -q RUNNING || { echo "dev node job $JID is not RUNNING"; exit 1; }
# NB: pyxis ignores --container-mounts when attaching to a running container (mounts are inherited).
exec srun --jobid="$JID" --overlap --container-name=nwdev \
     bash -lc "cd $REPO && $*"
