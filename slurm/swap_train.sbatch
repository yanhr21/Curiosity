#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p interactive
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH --job-name=rb_swaptrain
#SBATCH -o slurm/swap_train_%j.log
#
# One 4 h leg of the sugar_swap Refiner run: SUGAR's unmodified task code on Newton, plain
# PPO, one GPU on one node.
#
#   cd <repo> && sbatch slurm/swap_train.sbatch
#   RB_RUN=my_run RB_ENVS=256 sbatch slurm/swap_train.sbatch
#
# This is the single-GPU sibling of slurm/train.sbatch, which drives the hand-written
# sugar_newton env on eight GPUs under torchrun. The names are deliberately distinct
# (swap_train* against train*) because the two write different log roots and resuming one
# from the other's checkpoints would load a policy trained on a different observation.
#
# **Auto-resumes** regardless of why the previous leg stopped: swap_train_leg.sh picks the
# highest-numbered checkpoint in the run directory and continues from it, so preemption, a
# wall-clock kill and a manual resubmission are all the same event. --max-iterations is an
# ABSOLUTE target, so the remainder shrinks leg by leg and the run converges on one endpoint
# however many legs it takes.
#
# ONE GPU. The eight-GPU sibling is slurm/swap_train8.sbatch, and it is the one to use for a
# production run -- single-GPU throughput saturates at ~4,400 env-steps/s, which is 7.8 days
# for SUGAR's 30001 iterations at 4096 envs. This file remains because a one-GPU leg schedules
# where an eight-GPU leg queues, and because it is the baseline every DDP number is quoted
# against. #SBATCH directives cannot be parameterised by environment variables, hence two
# files rather than a flag on one.
#
# RB_ENVS is a TOTAL across ranks (see swap_train_leg.sh); at one GPU total and per-rank are
# the same number, so nothing here changed when DDP arrived.
#
# Partition is `interactive`: 10 nodes, 4 h cap, QOS allows 3 running and 20 submitted jobs
# per user. The submitted cap is what bounds a chain -- 20 legs including any dev node you
# hold. RB_PARTITION on swap_train_chain.sh overrides it.
#
# Submit from the repo root: -o above is relative to it, and the repo is found via
# SLURM_SUBMIT_DIR (Slurm copies the batch script to its spool dir, so $0 is useless).
set -u
REPO=${RB_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO"; exit 2; }
IMG=${RB_IMAGE:-/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh}

RUN=${RB_RUN:-carrybox_refiner_swap}
ENVS=${RB_ENVS:-512}
ITERS=${RB_ITERS:-30001}
# 20 min: an evaluation costs a couple of minutes (rollout, render, tactile panels, encode)
# and nothing else runs while it does, so a tighter cadence buys pictures with wall clock.
EVAL_MIN=${RB_EVAL_MIN:-20}
# Preemption is the only thing that should cost progress, so checkpoint well inside a leg.
# SUGAR's own save_interval is 1000, which a 4 h leg does not reach.
SAVE=${RB_SAVE:-25}
LOGROOT=${RB_LOGROOT:-logs/swap_refiner}
LOGGER=${RB_LOGGER:-wandb}
SEED=${RB_SEED:-}
EXTRA=${RB_EXTRA:-}

# Credentials live OUTSIDE the repo on purpose. RB_WANDB_ENV points at a shell snippet that
# exports WANDB_API_KEY; it is sourced here, on the submitting node, so the key lands in the
# job environment and then in --export below. Inside the container $HOME is /root, so
# ~/.netrc is not the one on Lustre and cannot be relied on. NOTE the leg script sources it
# AGAIN -- see the comment there; that second source is the one that actually wins.
if [ "$LOGGER" = "wandb" ]; then
    for _we in "${RB_WANDB_ENV:-}" "$HOME/files/wandb.env" \
               /lustre/fsw/portfolios/nvr/users/"$USER"/files/wandb.env; do
        if [ -n "$_we" ] && [ -r "$_we" ]; then
            # shellcheck source=/dev/null
            set +H; . "$_we"; echo "sourced wandb credentials from $_we"; break
        fi
    done
    [ -n "${WANDB_API_KEY:-}" ] || echo "WARNING: no WANDB_API_KEY found; the run will abort"
fi
# rsl_rl reads the wandb ENTITY from WANDB_USERNAME, not WANDB_ENTITY
# (rsl_rl/utils/wandb_utils.py:33). Unset it lands the run in the personal entity.
export WANDB_USERNAME=${RB_WANDB_ENTITY:-nvr-amri}

echo "===== SWAP_TRAIN_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"
echo "run=$RUN envs=$ENVS batch=$(( ENVS * 24 )) iters=$ITERS eval_min=$EVAL_MIN" \
     "save=$SAVE logger=$LOGGER logroot=$LOGROOT"
echo "wandb entity=$WANDB_USERNAME key_len=${#WANDB_API_KEY}"

# Everything swap_train_leg.sh reads is exported here and carried by --export=ALL. NB:
# repeated --export flags do NOT merge -- the last one wins -- so there is exactly one, and
# the variables travel as environment rather than as arguments.
export RB_RUN=$RUN RB_ENVS=$ENVS RB_ITERS=$ITERS RB_SAVE=$SAVE
export RB_EVAL_MIN=$EVAL_MIN RB_LOGROOT=$LOGROOT RB_LOGGER=$LOGGER RB_EXTRA=$EXTRA
export RB_SEED=$SEED
# Carried explicitly for the same reason as in swap_train8.sbatch. This is the launcher the
# ELLIPTIC control run uses, so pin it: RB_CONE=elliptic bash slurm/swap_train_chain.sh
export RB_CONE=${RB_CONE:-pyramidal} RB_IMPRATIO=${RB_IMPRATIO:-20.0}
# Forced to 1, not defaulted: this job holds ONE GPU, and an RB_GPUS=8 inherited from a chain
# that also submits swap_train8.sbatch would have the leg start eight ranks on it.
export RB_GPUS=1
export WANDB_API_KEY=${WANDB_API_KEY:-}

srun --ntasks=1 --gres=gpu:1 \
     --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     --export=ALL \
     bash -lc "cd $REPO && bash slurm/swap_train_leg.sh"
echo "===== SWAP_TRAIN_EXIT ====="; date
