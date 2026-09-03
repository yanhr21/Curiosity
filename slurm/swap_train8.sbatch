#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p interactive
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH -c 64
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH --job-name=rb_swaptrain8
#SBATCH -o slurm/swap_train8_%j.log
#
# One 4 h leg of the sugar_swap Refiner run on EIGHT GPUs of one node, data-parallel under
# torchrun. SUGAR's unmodified task code on Newton, plain PPO, SUGAR's own batch.
#
#   cd <repo> && sbatch slurm/swap_train8.sbatch
#   RB_RUN=my_run RB_ENVS=4096 sbatch slurm/swap_train8.sbatch
#
# The eight-GPU sibling of slurm/swap_train.sbatch. Everything operational is shared with it
# through slurm/swap_train_leg.sh -- the container setup, the heartbeat guard that stops two
# legs sharing a log directory, the checkpoint auto-resume, hardware EGL, and the wandb
# credential re-source that has to happen after the container's profile scripts. Only the
# #SBATCH directives and RB_GPUS differ, because directives cannot be parameterised by an
# environment variable.
#
# WHY: single-GPU throughput saturates at ~4,400 env-steps/s (512/1024/2048/4096 envs measured
# 3,320 / 3,981 / 4,292 / 4,379 on a dedicated A100-80GB, 2026-09-02), so SUGAR's registered
# 4096 envs x 24 steps x 30001 iterations = 2.95e9 env-steps is ~7.8 days on one GPU. Splitting
# the SAME 4096 envs over eight ranks of 512 is what brings that to day-scale, and because both
# the batch and the iteration count are unchanged it is the same experiment rather than a
# larger-batch one.
#
# MEASURED, job 33707941 on an 8x A100-SXM4-80GB node, 2026-09-02
# (slurm/swap_ddp_bench.sbatch; 50 steady-state iterations after 10 of warm-up):
#
#   8 ranks x 512 envs   3.704 s/iteration   26,534 env-steps/s
#   1 rank  x 4096 envs  17.15 s/iteration    5,733 env-steps/s   (same node, same container)
#
# 4.6x, i.e. 58 % of ideal -- and NONE of the shortfall is communication. The PPO update
# including the gradient all-reduce is 0.104 s of a 3.704 s iteration (2.8 %), and a rank's
# 3.704 s at 512 envs is the same as the 3.70 s a single GPU takes at 512 envs. The whole gap
# is that a 512-env A100 is not saturated: 8 x 3,320 = 26,560 predicts the measured 26,534 to
# within 0.1 %. So more throughput at this batch is not available -- it would need more envs
# per rank, which is a different experiment. 30001 iterations is ~31 h of compute, ~34 h with
# the evaluation videos: 9 legs.
#
# RB_ENVS IS A TOTAL ACROSS RANKS. 4096 means 512 per rank at eight GPUs and reproduces
# SUGAR's 98304-sample batch exactly. This is the opposite of slurm/train_leg.sh's per-rank
# RB_ENVS, on purpose: `sugar_swap.train --num-envs` overrides `scene.num_envs=4096`, which is
# SUGAR's whole batch, so under this convention RB_GPUS buys wall clock and never changes the
# experiment. See the header of slurm/swap_train_leg.sh.
#
# Partition is `interactive`, which is the ONLY interactive partition that accepts a multi-GPU
# request -- interactive_singlenode's QOS refuses anything above 1 GPU -- and it allows 3
# concurrent nodes per user, enough to validate 8-GPU single-node here and then extend to
# multi-node. polar3,polar4 have ~1200 nodes each but queue on Reason=Priority behind other
# users for an unbounded time, which is the wrong trade for a run someone is waiting on; keep
# them for long unattended work. Both cap at 4 h, so a long run is CHAINED --
# slurm/swap_train_chain.sh submits the legs with --dependency=afterany and the auto-resume in
# swap_train_leg.sh makes a leg boundary a non-event.
#
# **Auto-resumes** regardless of why the previous leg stopped, and all eight ranks load the
# same checkpoint (they must: rsl_rl derives its loop bounds from the checkpoint's iteration
# label, so a rank that skipped the load would run a different number of iterations and the
# job would hang at an all-reduce). --max-iterations is an ABSOLUTE target, so the remainder
# shrinks leg by leg and the run converges on one endpoint however many legs it takes.
#
# Submit from the repo root: -o above is relative to it, and the repo is found via
# SLURM_SUBMIT_DIR (Slurm copies the batch script to its spool dir, so $0 is useless).
set -u
REPO=${RB_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO"; exit 2; }
IMG=${RB_IMAGE:-/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh}

RUN=${RB_RUN:-carrybox_refiner_swap8}
GPUS=${RB_GPUS:-8}
# TOTAL envs across ranks. SUGAR's own 4096, so the batch is SUGAR's 4096 x 24 = 98304.
ENVS=${RB_ENVS:-4096}
ITERS=${RB_ITERS:-30001}
# 20 min: an evaluation costs a couple of minutes (rollout, render, tactile panels, encode)
# and on eight GPUs the other seven ranks wait in the next all-reduce while rank 0 does it, so
# a tighter cadence buys pictures with eight GPUs' worth of wall clock rather than one's.
EVAL_MIN=${RB_EVAL_MIN:-20}
# Preemption is the only thing that should cost progress, so checkpoint well inside a leg.
# SUGAR's own save_interval is 1000, which a 4 h leg does not reach.
SAVE=${RB_SAVE:-25}
LOGROOT=${RB_LOGROOT:-logs/swap_refiner}
LOGGER=${RB_LOGGER:-wandb}
SEED=${RB_SEED:-}
EXTRA=${RB_EXTRA:-}

if [ $(( ENVS % GPUS )) -ne 0 ]; then
    echo "RB_ENVS=$ENVS is a TOTAL and must divide RB_GPUS=$GPUS evenly"; exit 2
fi

# Credentials live OUTSIDE the repo on purpose. RB_WANDB_ENV points at a shell snippet that
# exports WANDB_API_KEY; it is sourced here, on the submitting node, so the key lands in the
# job environment and then in --export below. Inside the container $HOME is /root, so
# ~/.netrc is not the one on Lustre and cannot be relied on. NOTE the leg script sources it
# AGAIN -- see the comment there; that second source is the one that actually wins, because
# srun runs the step under `bash -l` and the container's /root/.bashrc exports a stale
# WANDB_API_KEY over the good one after --export has delivered it.
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

echo "===== SWAP_TRAIN8_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"
echo "run=$RUN gpus=$GPUS envs_total=$ENVS envs_per_rank=$(( ENVS / GPUS ))" \
     "batch=$(( ENVS * 24 )) iters=$ITERS eval_min=$EVAL_MIN save=$SAVE" \
     "logger=$LOGGER logroot=$LOGROOT"
echo "wandb entity=$WANDB_USERNAME key_len=${#WANDB_API_KEY}"
echo "friction cone=${RB_CONE:-pyramidal} impratio=${RB_IMPRATIO:-20.0}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# Everything swap_train_leg.sh reads is exported here and carried by --export=ALL. NB:
# repeated --export flags do NOT merge -- the last one wins -- so there is exactly one, and
# the variables travel as environment rather than as arguments.
export RB_RUN=$RUN RB_ENVS=$ENVS RB_ITERS=$ITERS RB_SAVE=$SAVE RB_GPUS=$GPUS
export RB_EVAL_MIN=$EVAL_MIN RB_LOGROOT=$LOGROOT RB_LOGGER=$LOGGER RB_EXTRA=$EXTRA
export RB_SEED=$SEED
# The friction cone is a physics choice, so it is carried EXPLICITLY rather than left to
# --export=ALL: this container's /root/.bashrc has already been caught clobbering an inherited
# variable (see the wandb key comment above), and a cone that silently reverts would produce a
# run that is neither the pyramidal experiment nor the elliptic control. Defaults match
# builder.py; set RB_CONE=elliptic to pin a control run.
export RB_CONE=${RB_CONE:-pyramidal} RB_IMPRATIO=${RB_IMPRATIO:-20.0}
export WANDB_API_KEY=${WANDB_API_KEY:-}

# ONE task holding all eight GPUs, not eight tasks of one: torchrun spawns and ranks the
# workers itself, and srun doing it too would give every rank LOCAL_RANK=0 and eight processes
# fighting over cuda:0 -- which presents as a memory blow-up on GPU 0 rather than as an error.
# --gres is repeated because a step does not always inherit the job's full allocation, and
# `srun --overlap` is deliberately absent: it has silently exposed half the GPUs before.
srun --ntasks=1 --gres=gpu:"$GPUS" \
     --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     --export=ALL \
     bash -lc "cd $REPO && bash slurm/swap_train_leg.sh"
echo "===== SWAP_TRAIN8_EXIT ====="; date
