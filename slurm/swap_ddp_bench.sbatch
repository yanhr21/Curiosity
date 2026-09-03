#!/bin/bash
#SBATCH -A nvr_nxp_visionconferencing
#SBATCH -p interactive
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH -c 64
#SBATCH -t 1:30:00
#SBATCH --job-name=rb_swapddpbench
#SBATCH -o slurm/swap_ddp_bench_%j.log
#
# Verify and benchmark the DDP path in sugar_swap.train. Never trains to convergence and never
# touches a production run directory: everything lands under logs/swap_ddp_bench/.
#
#   cd <repo> && sbatch slurm/swap_ddp_bench.sbatch
#
# Five phases, in the order that makes a failure cheap to localise:
#
#   0  the PRODUCTION launcher -- slurm/swap_train_leg.sh at RB_GPUS=8 -- for 3 iterations, so
#      that the path the real chain takes (heartbeat guard, checkpoint resume, wandb credential
#      re-source, torchrun) is exercised and not just the direct torchrun the other phases use.
#   A  8 ranks x 512 envs = 4096 total, ~60 iterations, no video. Measures env-steps/s and
#      seconds per iteration against the single-GPU baseline, and prints the per-rank rollout
#      and parameter hashes that distinguish data-parallel PPO from 8 independent trainers.
#      nvidia-smi is sampled while it runs, so the process-to-device mapping is on the record.
#   B  resume from A's last checkpoint on all 8 ranks. A rank that skipped the load would run
#      a different iteration count and the job would hang, so reaching phase C is the result.
#   C  1 rank x 4096 envs, same node, same container, ~25 iterations. The baseline, re-measured
#      here rather than quoted, so the speed-up is not comparing two different machines.
#   D  8 ranks, one iteration with the evaluation video forced on, to confirm exactly one rank
#      builds a recorder and exactly one wandb run exists for the eight processes.
#
# WANDB_MODE=offline throughout. The wandb path still runs end to end -- bind_wandb_run mints
# the id, rsl_rl builds its WandbSummaryWriter, wandb.init is called -- but nothing is uploaded,
# so a benchmark cannot leave a junk curve in the shared project. Counting the offline run
# directories afterwards is also the cleanest evidence that seven ranks created nothing.
set -u
REPO=${RB_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO"; exit 2; }
IMG=${RB_IMAGE:-/lustre/fsw/portfolios/nvr/users/shengzew/cuda_docker_ver3.sqsh}

GPUS=${RB_GPUS:-8}
ENVS=${RB_ENVS:-4096}            # TOTAL across ranks
ITERS_A=${RB_ITERS_A:-60}
ITERS_B=${RB_ITERS_B:-70}
ITERS_C=${RB_ITERS_C:-25}
OUT=${RB_BENCH_ROOT:-logs/swap_ddp_bench}

echo "===== DDP_BENCH_ALLOC ====="; hostname; date; echo "jobid=$SLURM_JOB_ID"
echo "gpus=$GPUS envs_total=$ENVS envs_per_rank=$(( ENVS / GPUS )) batch=$(( ENVS * 24 ))"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun --ntasks=1 --gres=gpu:"$GPUS" \
     --container-image="$IMG" \
     --container-mounts="$HOME":/home,/lustre:/lustre \
     --export=ALL \
     bash -lc "cd $REPO && RB_GPUS=$GPUS RB_ENVS=$ENVS RB_ITERS_A=$ITERS_A \
               RB_ITERS_B=$ITERS_B RB_ITERS_C=$ITERS_C RB_BENCH_ROOT=$OUT \
               bash slurm/swap_ddp_bench.sh"
echo "===== DDP_BENCH_EXIT status=$? ====="; date
