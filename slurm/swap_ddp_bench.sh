#!/bin/bash
# The inside-the-container half of slurm/swap_ddp_bench.sbatch. See that file for the plan.
#
# A FILE rather than a string in the sbatch for the reason swap_train_leg.sh is one: it needs
# `&` (for the nvidia-smi sampler) inside a chain, `&` binds looser than `&&`, and `bash -n`
# catches what escaping inside `bash -lc "..."` hides.
set -u

GPUS=${RB_GPUS:-8}
ENVS=${RB_ENVS:-4096}
ITERS_A=${RB_ITERS_A:-60}
ITERS_B=${RB_ITERS_B:-70}
ITERS_C=${RB_ITERS_C:-25}
OUT=${RB_BENCH_ROOT:-logs/swap_ddp_bench}
export PYTHONUNBUFFERED=1

echo "===== BENCH_SETUP ====="
bash slurm/setup_container.sh 2>&1 | tail -4

# shellcheck source=/dev/null
source env/activate.sh || exit 2
# shellcheck source=/dev/null
source slurm/render_env_egl.sh

# ---- wandb credentials, LAST -------------------------------------------------------------
# Same ordering rule as swap_train_leg.sh, and exercising it here is part of the point: srun
# runs this under `bash -l`, so the container's /root/.bashrc has already exported a stale
# WANDB_API_KEY over whatever --export delivered. Sourcing the credential file after every
# profile has had its turn is what actually wins.
for _we in "${RB_WANDB_ENV:-}" "$HOME/files/wandb.env" \
           /lustre/fsw/portfolios/nvr/users/"$USER"/files/wandb.env \
           /lustre/fsw/portfolios/nvr/users/shengzew/files/wandb.env; do
    if [ -n "$_we" ] && [ -r "$_we" ]; then
        set +H
        # shellcheck source=/dev/null
        . "$_we"
        echo "[wandb] key from $_we (len ${#WANDB_API_KEY})"
        break
    fi
done
[ -n "${WANDB_API_KEY:-}" ] || { echo "[wandb] no key found"; exit 2; }
# Offline: the whole wandb path runs, nothing is uploaded, and the offline run directories are
# countable evidence that only rank 0 created one.
export WANDB_MODE=offline
export WANDB_DIR=$OUT
export WANDB_USERNAME=${WANDB_USERNAME:-nvr-amri}
# Rank 0 records the evaluation video while the other seven wait in the next all-reduce, so
# NCCL's watchdog has to tolerate a gap much longer than an iteration.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

mkdir -p "$OUT"
PER_RANK=$(( ENVS / GPUS ))

run_train() {   # run_train <nproc> <log-subdir> <num-envs-total> <max-iters> <extra...>
    local nproc=$1 name=$2 total=$3 iters=$4
    shift 4
    if [ "$nproc" -gt 1 ]; then
        torchrun --standalone --nnodes=1 --nproc_per_node="$nproc" -m sugar_swap.train \
            --num-envs "$total" --max-iterations "$iters" --save-interval 25 \
            --eval-minutes 0 --log-root "$OUT" --run-name "$name" --logger wandb "$@"
    else
        python -u -m sugar_swap.train \
            --num-envs "$total" --max-iterations "$iters" --save-interval 25 \
            --eval-minutes 0 --log-root "$OUT" --run-name "$name" --logger wandb "$@"
    fi
}

# =========================================================================================
# Phase 0 is the PRODUCTION launcher, not a hand-rolled torchrun: swap_train_leg.sh is what
# swap_train8.sbatch invokes, and it carries the heartbeat guard, the highest-checkpoint
# resume and the wandb credential re-source. Phases A-D below call torchrun directly to keep
# the measurement free of that machinery, so without this phase nothing would ever exercise
# the path the real chain takes. Three iterations is enough: what is being tested is that the
# leg script starts eight ranks and hands them the right flags.
#
# The credential file only sets WANDB_API_KEY, so the WANDB_MODE=offline exported above
# survives the leg script's own re-source and this phase stays offline too.
echo "===== PHASE 0: production launcher (swap_train_leg.sh), $GPUS ranks, 3 iterations ====="
RB_RUN=bench_leg RB_LOGROOT=$OUT RB_ENVS=$ENVS RB_GPUS=$GPUS RB_ITERS=3 \
RB_SAVE=25 RB_EVAL_MIN=0 RB_LOGGER=wandb \
    bash slurm/swap_train_leg.sh
echo "===== PHASE 0 status=$? ====="

# =========================================================================================
echo "===== PHASE A: $GPUS ranks x $PER_RANK envs = $ENVS total, $ITERS_A iterations ====="
# Sample the process-to-device mapping while training is actually resident. Started before the
# run and killed after, because a snapshot taken once cannot show whether all eight ranks are
# simultaneously on eight devices -- which is the claim.
( sleep 200
  for _ in 1 2 3 4 5 6; do
      echo "----- nvidia-smi $(date +%T) -----"
      nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv
      nvidia-smi --query-gpu=index,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader
      sleep 40
  done ) &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

run_train "$GPUS" bench8 "$ENVS" "$ITERS_A" --ddp-verify 3
echo "===== PHASE A status=$? ====="
kill $SMI 2>/dev/null; trap - EXIT

echo "----- wandb offline runs created by $GPUS ranks (expect exactly 1) -----"
ls -1d "$OUT"/wandb/offline-run-* 2>/dev/null | wc -l
ls -1d "$OUT"/wandb/*run* 2>/dev/null

# =========================================================================================
LAST=$(ls -1 "$OUT"/bench8/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
echo "===== PHASE B: resume all $GPUS ranks from model_${LAST:-none}.pt to $ITERS_B ====="
if [ -n "$LAST" ]; then
    run_train "$GPUS" bench8 "$ENVS" "$ITERS_B" --ddp-verify 2 \
              --resume "$OUT/bench8/model_$LAST.pt"
    echo "===== PHASE B status=$? ====="
else
    echo "no checkpoint from phase A; skipping"
fi

# =========================================================================================
echo "===== PHASE C: 1 rank x $ENVS envs, $ITERS_C iterations (same-node baseline) ====="
run_train 1 bench1 "$ENVS" "$ITERS_C"
echo "===== PHASE C status=$? ====="

# =========================================================================================
echo "===== PHASE D: $GPUS ranks, video forced on, 1 iteration (rank-0-only check) ====="
# --video-interval 1 rather than a wall-clock cadence, so it fires on the first logged
# iteration instead of 20 minutes in. Eight ranks; exactly one line of [eval] output and one
# videos/ directory is the property being checked.
torchrun --standalone --nnodes=1 --nproc_per_node="$GPUS" -m sugar_swap.train \
    --num-envs "$ENVS" --max-iterations 1 --save-interval 25 \
    --eval-minutes 0 --video-interval 1 --video-frames 60 \
    --log-root "$OUT" --run-name benchvid --logger wandb
echo "===== PHASE D status=$? ====="
echo "----- videos written -----"
find "$OUT" -name '*.mp4' -o -name '*.gif' | sed "s|^|  |"

echo "===== BENCH_DONE ====="
