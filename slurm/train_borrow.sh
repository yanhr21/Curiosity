#!/bin/bash
# Run a training leg inside an allocation that ALREADY EXISTS, by job id.
#
#   bash slurm/train_borrow.sh 33616572
#   RB_ENVS=256 bash slurm/train_borrow.sh 33616572 > /tmp/borrow.log 2>&1 &
#   RB_LEG=slurm/swap_train_leg.sh bash slurm/train_borrow.sh 33697036   # sugar_swap path
#
# RB_LEG selects which leg script runs inside the container. Everything this file does --
# borrowing the allocation, giving the step its own container root, and the credential
# lookup -- is independent of which trainer runs, so both paths share it.
#
# Why this exists: every GPU partition here is either tiny and drained (interactive,
# batch_singlenode and grizzly are the SAME 9 nodes, 5 of them drained) or capped at one
# running job per user (interactive_singlenode -- 1227 nodes and hundreds of free GPUs you
# cannot have a second job on). When a dev node you already hold is sitting idle, attaching to
# it with `srun --overlap` sidesteps both: it is a STEP inside an existing job, not a new job,
# so no QOS cap applies and there is no queue.
#
# --container-image on an overlapping step gives this step its own container root rather than
# the host filesystem, which matters: setup_container.sh needs to be root to install the GL
# stack and write the EGL ICD, and without hardware EGL the evaluation video falls back to
# software rasterisation at seconds per frame.
#
# The borrowed allocation's clock is not ours, so this can be killed at any moment. That is
# safe -- train_leg.sh checkpoints every RB_SAVE iterations and auto-resumes -- and it is why
# a queued job should be left in the queue to take over. On interactive_singlenode the timing
# is automatic: that partition's one-job cap is released by the very allocation ending, so a
# job queued there starts exactly when this borrowed run dies.
set -u
JOBID=${1:-}
[ -n "$JOBID" ] || { echo "usage: bash slurm/train_borrow.sh <jobid>"; exit 2; }
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IMG=${RB_IMAGE:-/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh}

squeue -j "$JOBID" -h -o %T 2>/dev/null | grep -q RUNNING || {
    echo "job $JOBID is not RUNNING -- nothing to borrow"; exit 1; }

# Same credential lookup as train.sbatch: outside the repo, so no secret is in git.
if [ "${RB_LOGGER:-wandb}" = "wandb" ]; then
    for _we in "${RB_WANDB_ENV:-}" "$HOME/files/wandb.env" \
               /lustre/fsw/portfolios/nvr/users/"$USER"/files/wandb.env; do
        if [ -n "$_we" ] && [ -r "$_we" ]; then
            # shellcheck source=/dev/null
            set +H; . "$_we"; echo "sourced wandb credentials from $_we"; break
        fi
    done
fi
export WANDB_USERNAME=${RB_WANDB_ENTITY:-nvr-amri}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export RB_STAGE=${RB_STAGE:-tracker}
export RB_GPUS=${RB_GPUS:-1}
export RB_RUN=${RB_RUN:-carrybox_bcppo}
export RB_ENVS=${RB_ENVS:-512}
export RB_ITERS=${RB_ITERS:-3000}
export RB_SAVE=${RB_SAVE:-25}
export RB_EVAL_MIN=${RB_EVAL_MIN:-10}
export RB_LOGROOT=${RB_LOGROOT:-logs/newton_bcppo}
export RB_LOGGER=${RB_LOGGER:-wandb}
export RB_EXTRA=${RB_EXTRA:-}
# Physics, so it travels explicitly rather than by inheritance -- same reasoning as the sbatch
# launchers. Defaults match builder.py; RB_CONE=elliptic pins a control run.
export RB_CONE=${RB_CONE:-pyramidal} RB_IMPRATIO=${RB_IMPRATIO:-20.0}
# A borrowed run should not idle on a GPU that is not ours waiting for someone else's leg.
export RB_WAIT_MIN=${RB_WAIT_MIN:-0}

LEG=${RB_LEG:-slurm/train_leg.sh}
[ -r "$REPO/$LEG" ] || { echo "no such leg script: $LEG"; exit 2; }

echo "===== BORROW jobid=$JOBID node=$(squeue -j "$JOBID" -h -o %N 2>/dev/null)"
echo "leg=$LEG run=$RB_RUN envs=$RB_ENVS iters=$RB_ITERS eval_min=$RB_EVAL_MIN"
echo "wandb entity=$WANDB_USERNAME key_len=${#WANDB_API_KEY}"

exec srun --overlap --jobid="$JOBID" --gres=gpu:"$RB_GPUS" \
     --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     --export=ALL \
     bash -lc "cd $REPO && bash $LEG"
