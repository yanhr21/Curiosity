#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p interactive_singlenode
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH -t 0:30:00
#SBATCH --job-name=rb_swapcheck
#SBATCH -o slurm/swap_train_check_%j.log
#
# Preflight for the sugar_swap Refiner launcher, on a GPU. Runs every stage of
# slurm/swap_train_check.py including the one that builds an environment and the rsl_rl
# runner, then exits. It never trains and never writes into a run directory.
#
#   cd <repo> && sbatch slurm/swap_train_check.sbatch
#
# Its own allocation on purpose: attaching to somebody else's dev container would contend for
# the GPU with whatever they are doing there. `interactive_singlenode` because it has ~1200
# nodes against `interactive`'s 10, and a 30 min preflight has no reason to queue behind the
# training chain.
#
# Submit from the repo root: -o above is relative to it, and the repo is found via
# SLURM_SUBMIT_DIR.
set -u
REPO=${RB_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO"; exit 2; }
IMG=${RB_IMAGE:-/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh}
ENVS=${RB_ENVS:-4}

echo "===== SWAP_CHECK_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"

srun --ntasks=1 --gres=gpu:1 \
     --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     bash -lc "set -u; cd $REPO && \
       bash slurm/setup_container.sh 2>&1 | tail -3 && \
       source env/activate.sh && \
       source slurm/render_env_egl.sh && \
       export WANDB_MODE=offline && \
       python -u slurm/swap_train_check.py --with-runner --with-video --envs $ENVS"
status=$?
echo "===== SWAP_CHECK_DONE status=$status ====="; date
exit $status
