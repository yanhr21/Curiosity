#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p polar3,polar4
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH -c 64
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH --job-name=rb_train
#SBATCH -o slurm/train_%j.log
#
# A BCPPO training run on Newton, sized to finish in one job.
#
#   cd <repo> && sbatch slurm/train.sbatch
#   RB_RUN=my_run RB_ENVS=256 sbatch slurm/train.sbatch
#
# **Auto-resumes** regardless of why it stopped: it picks the highest-numbered checkpoint in
# the run's log directory and continues from it. That makes preemption, a wall-clock kill and
# a manual resubmission all the same event, and it is what lets this run on a preemptible
# partition safely.
#
# This asks for a WHOLE NODE and trains data-parallel under torchrun. `bench_scaling`,
# measured 2026-09-01, is why: one collection costs 45.6 s at 512 worlds on a single GPU but
# 13.9 s at 64, because 512 worlds saturate a GPU and 64 do not. Splitting the same
# 12288-sample batch over 8 ranks of 64 therefore cuts the 3000-iteration curriculum from
# ~38 h to ~12 h, and because the batch and the iteration count are both unchanged it is the
# same experiment, not a larger-batch one. The gain is 3.3x rather than 8x, and the
# all-reduce plus a 110 s evaluation every 10 min eats into even that.
#
# Partitions, measured 2026-09-01/02:
#
#   interactive_singlenode  1227 nodes -- but its QOS allows ONE running job per user, so it
#                           queues behind any dev node you hold, however idle the cluster is.
#   batch_singlenode        no per-user cap -- but 6 of 9 nodes DRAINED. Never schedules.
#   backfill_singlenode     1227 nodes and 7 days of wall clock, so the run fits in ONE job,
#                           but a whole-node request there sat at (Priority) without
#                           starting.
#   polar3 / polar4         ~1200 nodes each, ~27 fully idle, and short jobs turn over fast
#                           even with thousands pending. 4 h cap.
#
# So: polar. The 4 h cap means the ~13 h run is CHAINED -- see slurm/train_chain.sh, which
# submits the legs with --dependency=afterany so each starts when the previous ends for any
# reason. The auto-resume above is what makes a leg boundary a non-event. At 8 GPUs this is
# ~5 legs; the same run on one GPU needed 13, which is the other half of why DDP was worth
# doing.
#
# Safe to queue on SEVERAL partitions at once. Which one frees first is unpredictable here, so
# the useful thing is to sit in all of their queues -- but two legs training the same run
# directory would interleave checkpoints and corrupt the run. A 60 s heartbeat file in the run
# directory prevents it: a starting leg that finds a heartbeat under 3 minutes old exits
# immediately. The check is a plain file mtime rather than a squeue lookup because squeue does
# not exist inside the container, and the heartbeat dies with the job whether it exits, is
# killed by the wall clock, or is preempted.
#
# Submit from the repo root: -o above is relative to it, and the repo is found via
# SLURM_SUBMIT_DIR (Slurm copies the batch script to its spool dir, so $0 is useless).
set -u
REPO=${RB_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO"; exit 2; }
IMG=${RB_IMAGE:-/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh}

RUN=${RB_RUN:-carrybox_bcppo}
# Worlds PER RANK. 8 x 64 is the same 12288-sample batch as the old 1 x 512, so the curriculum
# is unchanged; see the RB_GPUS note in the header for why it is faster.
ENVS=${RB_ENVS:-64}
GPUS=${RB_GPUS:-8}
ITERS=${RB_ITERS:-3000}
EVAL_MIN=${RB_EVAL_MIN:-10}
# Preemption is the only thing that should cost progress, so checkpoint well inside a leg:
# at ~14 s/iteration 25 iterations is ~6 min of work at risk.
SAVE=${RB_SAVE:-25}
LOGROOT=${RB_LOGROOT:-logs/newton_bcppo}
LOGGER=${RB_LOGGER:-wandb}
EXTRA=${RB_EXTRA:-}

# Credentials live OUTSIDE the repo on purpose. RB_WANDB_ENV points at a shell snippet that
# exports WANDB_API_KEY; it is sourced here, on the submitting node, so the key lands in the
# job environment and then in --export below. Inside the container $HOME is /root, so
# ~/.netrc is not the one on Lustre and cannot be relied on.
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

echo "===== TRAIN_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"
echo "run=$RUN gpus=$GPUS envs/rank=$ENVS total_worlds=$(( GPUS * ENVS ))" \
     "batch=$(( GPUS * ENVS * 24 )) iters=$ITERS eval_min=$EVAL_MIN logger=$LOGGER"
echo "wandb entity=$WANDB_USERNAME key_len=${#WANDB_API_KEY}"

# Everything train_leg.sh reads is exported here and carried by --export=ALL. NB: repeated
# --export flags do NOT merge -- the last one wins -- so there is exactly one, and the
# variables travel as environment rather than as arguments. WANDB_API_KEY has to make the trip
# too: it is not in the container's environment by default, and $HOME there is /root, so the
# ~/.netrc on Lustre is not visible either.
export RB_RUN=$RUN RB_ENVS=$ENVS RB_ITERS=$ITERS RB_SAVE=$SAVE RB_GPUS=$GPUS
export RB_EVAL_MIN=$EVAL_MIN RB_LOGROOT=$LOGROOT RB_LOGGER=$LOGGER RB_EXTRA=$EXTRA
export WANDB_API_KEY=${WANDB_API_KEY:-}

# ONE task holding all 8 GPUs, not 8 tasks of 1: torchrun spawns and ranks the workers, and
# srun doing it too would give every rank LOCAL_RANK=0 and eight processes fighting over
# cuda:0. --gres is repeated here because a step does not always inherit the job's full
# allocation.
srun --ntasks=1 --gres=gpu:"$GPUS" \
     --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     --export=ALL \
     bash -lc "cd $REPO && bash slurm/train_leg.sh"
echo "===== TRAIN_EXIT ====="; date
