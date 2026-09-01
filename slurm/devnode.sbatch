#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p interactive_singlenode
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH --job-name=rb_devnode
#SBATCH -o slurm/devnode_%j.log
#
# Persistent GPU dev node.
#
# Holds an interactive_singlenode allocation (non-preemptible, 4 h max) with a NAMED pyxis
# container kept alive by a sleep. setup_container.sh runs ONCE here; because the container
# is named, every later step attaches to the SAME container root, so the GL/Xvfb dpkg
# installs persist for the whole job instead of being redone per srun.
#
#   cd <repo> && sbatch slurm/devnode.sbatch
#   bash slurm/devrun.sh "nvidia-smi -L"
#
# Submit from the repo root: the -o path above is relative to it, and the repo is located
# via SLURM_SUBMIT_DIR. It cannot be derived from $0 -- Slurm copies the batch script into
# its spool directory, so `dirname $0` is /var/spool/slurmd. RB_REPO overrides.
set -u
REPO=${RB_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO (submit from the repo root, or set RB_REPO)"; exit 2; }
IMG=${RB_IMAGE:-/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh}
NAME=${RB_CONTAINER:-rbdev}
STATE=$REPO/slurm/.devnode

echo "===== DEVNODE_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"
echo "repo=$REPO image=$IMG container=$NAME"

# /lustre must be mounted: the repo, the conda env and the deb cache all live there.
srun --container-name="$NAME" --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     bash -lc "set -u; cd $REPO && \
       echo '===== DEVNODE_SETUP =====' && bash slurm/setup_container.sh 2>&1 | tail -6 && \
       nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
       printf 'jobid=%s\nnode=%s\ncontainer=%s\nrepo=%s\n' \
              '$SLURM_JOB_ID' \"\$(hostname)\" '$NAME' '$REPO' > $STATE && \
       echo '===== DEVNODE_READY =====' && sleep 4h"
echo "===== DEVNODE_EXIT ====="; date
