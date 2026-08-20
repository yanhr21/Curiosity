#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p interactive_singlenode
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH --job-name=nw_devnode
#SBATCH -o /lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton/renders/devnode_%j.log

# Persistent GPU dev node for the G1-in-SAGE Newton work.
#
# Holds an interactive_singlenode allocation (non-preemptible, 4 h max) with a NAMED pyxis
# container ("nwdev") kept alive by a sleep. setup_container.sh runs ONCE here; because the
# container is named, every later step attaches to the SAME container root, so the GL/Xvfb
# dpkg installs persist for the whole job instead of being redone per srun.
#
# Attach a command from the login node (see renders/devrun.sh):
#   srun --jobid=<JID> --overlap --container-name=nwdev \
#        --container-mounts=$HOME:/home,/lustre:/lustre bash -lc "<cmd>"
REPO=/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew/robot_baby/Curiosity_newton
IMG=/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh
STATE=$REPO/renders/.devnode

echo "===== DEVNODE_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"
srun --container-name=nwdev --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     bash -lc "set -u; cd $REPO && \
       echo '===== DEVNODE_SETUP =====' && bash renders/setup_container.sh 2>&1 | tail -6 && \
       nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
       printf 'jobid=%s\nnode=%s\ncontainer=nwdev\n' '$SLURM_JOB_ID' \"\$(hostname)\" > $STATE && \
       echo '===== DEVNODE_READY =====' && sleep 4h"
echo "===== DEVNODE_EXIT ====="; date
