#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p interactive
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH -t 4:00:00
#SBATCH --job-name=rb_devnode8
#SBATCH -o slurm/devnode8_%j.log
#
# 8-GPU variant of devnode.sbatch, for sugar_newton/par_policy.sh (one configuration per
# GPU). #SBATCH directives cannot be parameterised by environment variables, hence a second
# file rather than a flag on the first.
#
#   cd <repo> && sbatch slurm/devnode8.sbatch
#   RB_STATE=slurm/.devnode8 bash slurm/devrun.sh "bash sugar_newton/par_policy.sh sugar_newton/sweeps/ab_contact.txt"
#
# Partition is `interactive` rather than `interactive_singlenode`: the latter's QOS rejects
# an 8-GPU request here.
set -u
# Via SLURM_SUBMIT_DIR, not $0: Slurm copies the batch script to its spool directory.
REPO=${RB_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO (submit from the repo root, or set RB_REPO)"; exit 2; }
IMG=${RB_IMAGE:-/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh}
NAME=${RB_CONTAINER:-rbdev8}
STATE=$REPO/slurm/.devnode8

echo "===== DEVNODE_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"
echo "repo=$REPO image=$IMG container=$NAME"

srun --container-name="$NAME" --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     bash -lc "set -u; cd $REPO && \
       echo '===== DEVNODE_SETUP =====' && bash slurm/setup_container.sh 2>&1 | tail -6 && \
       nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
       printf 'jobid=%s\nnode=%s\ncontainer=%s\nrepo=%s\n' \
              '$SLURM_JOB_ID' \"\$(hostname)\" '$NAME' '$REPO' > $STATE && \
       echo '===== DEVNODE_READY =====' && sleep 4h"
echo "===== DEVNODE_EXIT ====="; date
