#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p polar3,polar4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH --job-name=rb_swaptrain32
#SBATCH -o slurm/swap_train32_%j.log
#
# One 4 h leg of the sugar_swap Refiner run across FOUR nodes, 32 ranks under torchrun.
#
#   cd <repo> && sbatch slurm/swap_train32.sbatch
#   RB_ENVS=16384 sbatch slurm/swap_train32.sbatch     # weak scaling; see the warning below
#
# The multi-node sibling of slurm/swap_train8.sbatch, sharing slurm/swap_train_leg.sh for
# everything operational. Two structural differences from the 8-GPU file, both required:
#
#   --ntasks-per-node=1, so srun starts ONE leg per node and torchrun spawns the 8 local
#   ranks. One task per GPU would have every rank see LOCAL_RANK=0 and 32 processes pile onto
#   4 GPUs, which presents as a memory blow-up rather than an error.
#
#   RB_NODES=4 makes the leg use STATIC rendezvous instead of --standalone. --standalone binds
#   localhost, so each node would form its own 8-rank world, train happily, and never exchange
#   a gradient with the other three -- four independent quarter-runs wearing one run's name.
#
# Partition is polar3,polar4 because `interactive` caps a user at 3 NODES and this asks for 4.
# polar queues on Reason=Priority for an unpredictable time; that is the trade for the shape.
#
# READ THIS BEFORE TRUSTING THE SPEED-UP. RB_ENVS is a TOTAL across ranks, so the default of
# SUGAR's 4096 puts 128 envs on each of 32 ranks. Measured scaling on this workload is
# "N copies of the per-rank datapoint" -- at 8 GPUs, 8 x the 512-env rate predicted the
# measured aggregate to 0.1%, because the bottleneck is that a rank does not saturate its
# A100, not communication (the all-reduce was 0.10 s of a 3.70 s iteration). A 128-env rank is
# further from saturation still, so 4 nodes at SUGAR's batch is expected to buy well under 4x
# -- roughly 1.5x over one node. The alternative is RB_ENVS=16384, which keeps 512 envs per
# rank and should scale close to linearly, but quadruples SUGAR's batch and is therefore a
# DIFFERENT EXPERIMENT, not a faster route to the same one. The default here stays faithful.
set -u
REPO=${RB_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO"; exit 2; }
IMG=${RB_IMAGE:-/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh}

RUN=${RB_RUN:-carrybox_refiner_swap32}
NODES=${RB_NODES:-4}
GPUS=${RB_GPUS:-32}              # WORLD SIZE, not per node
ENVS=${RB_ENVS:-4096}            # TOTAL across ranks
ITERS=${RB_ITERS:-30001}
EVAL_MIN=${RB_EVAL_MIN:-20}
SAVE=${RB_SAVE:-25}
LOGROOT=${RB_LOGROOT:-logs/swap_refiner}
LOGGER=${RB_LOGGER:-wandb}
SEED=${RB_SEED:-}
EXTRA=${RB_EXTRA:-}

if [ $(( ENVS % GPUS )) -ne 0 ]; then
    echo "RB_ENVS=$ENVS is a TOTAL and must divide RB_GPUS=$GPUS evenly"; exit 2
fi
if [ $(( GPUS % NODES )) -ne 0 ]; then
    echo "RB_GPUS=$GPUS (world size) must divide RB_NODES=$NODES evenly"; exit 2
fi

# Same credential handling as the other launchers: sourced here so the key reaches the job,
# and sourced AGAIN inside the leg because the container's /root/.bashrc overwrites it.
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
export WANDB_USERNAME=${RB_WANDB_ENTITY:-nvr-amri}

echo "===== SWAP_TRAIN32_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NODELIST"
echo "run=$RUN nodes=$NODES world=$GPUS ranks/node=$(( GPUS / NODES ))" \
     "envs_total=$ENVS envs/rank=$(( ENVS / GPUS )) batch=$(( ENVS * 24 ))" \
     "iters=$ITERS eval_min=$EVAL_MIN save=$SAVE logger=$LOGGER"
echo "friction cone=${RB_CONE:-pyramidal} impratio=${RB_IMPRATIO:-20.0}"

# Resolved HERE, on the allocation, because the leg runs inside the container and the
# container has no SLURM client -- `scontrol` there is "command not found", which kills the
# step. SLURM_NODEID does survive, since srun passes it as an environment variable.
# The port is derived from the job id so two chained legs cannot collide on it.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=${MASTER_PORT:-$(( 20000 + SLURM_JOB_ID % 20000 ))}
echo "rendezvous master=$MASTER_ADDR:$MASTER_PORT"

export RB_RUN=$RUN RB_ENVS=$ENVS RB_ITERS=$ITERS RB_SAVE=$SAVE
export RB_GPUS=$GPUS RB_NODES=$NODES
export RB_EVAL_MIN=$EVAL_MIN RB_LOGROOT=$LOGROOT RB_LOGGER=$LOGGER RB_EXTRA=$EXTRA
export RB_SEED=$SEED
export RB_CONE=${RB_CONE:-pyramidal} RB_IMPRATIO=${RB_IMPRATIO:-20.0}
export WANDB_API_KEY=${WANDB_API_KEY:-}

# ONE task per NODE. torchrun does the per-GPU fan-out; srun only has to place one leg on each
# node and give it all 8 GPUs. SLURM_NODEID is what the leg turns into torchrun's --node_rank.
srun --ntasks="$NODES" --ntasks-per-node=1 --gres=gpu:$(( GPUS / NODES )) \
     --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     --export=ALL \
     bash -lc "cd $REPO && bash slurm/swap_train_leg.sh"
echo "===== SWAP_TRAIN32_EXIT status=$? ====="; date
