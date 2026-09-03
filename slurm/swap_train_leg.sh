#!/bin/bash
# One leg of the sugar_swap Refiner run, from inside the container. Invoked by
# slurm/swap_train.sbatch (1 GPU) or slurm/swap_train8.sbatch (8 GPUs under torchrun).
#
#   bash slurm/swap_train_leg.sh          # reads RB_* from the environment
#
# Reads: RB_RUN RB_ENVS RB_GPUS RB_ITERS RB_SAVE RB_EVAL_MIN RB_LOGROOT RB_LOGGER RB_SEED
#        RB_EXTRA
#
# **RB_ENVS is a TOTAL across ranks here, not a per-rank count.** That is deliberately the
# opposite of slurm/train_leg.sh, where RB_ENVS is per-rank, and the difference is worth the
# inconsistency: `sugar_swap.train --num-envs` overrides SUGAR's registered
# `scene.num_envs=4096`, which is SUGAR's whole batch, so keeping RB_ENVS a total means
# RB_GPUS changes only the wall clock and never the experiment. Under train_leg.sh's
# convention, raising RB_GPUS from 1 to 8 silently multiplies the batch by 8.
#
# This is a FILE rather than a string embedded in the sbatch for two reasons. It needs `&`
# (for the heartbeat) inside a chain of `&&`, and `&` binds looser than `&&`, so written
# inline the whole preceding chain gets backgrounded and the variables set in it are never
# visible to what follows -- which is the bug that killed job 33620512 with "HB: unbound
# variable". And a file is testable: `bash -n` catches what escaping inside `bash -lc "..."`
# hides.
set -u

RUN=${RB_RUN:-carrybox_refiner_swap}
ENVS=${RB_ENVS:-512}                    # TOTAL across ranks; see the header
GPUS=${RB_GPUS:-1}
export PYTHONUNBUFFERED=1               # torchrun has no -u; keep both paths line-buffered
ITERS=${RB_ITERS:-30001}
SAVE=${RB_SAVE:-25}
EVAL_MIN=${RB_EVAL_MIN:-20}
LOGROOT=${RB_LOGROOT:-logs/swap_refiner}
LOGGER=${RB_LOGGER:-wandb}
SEED=${RB_SEED:-}
EXTRA=${RB_EXTRA:-}

# Checked before the container setup and the ~2 min environment build, so a bad pairing costs
# a second rather than an allocation. `sugar_swap.train` raises on this too; the point of
# repeating it is where the message appears.
if [ "$GPUS" -gt 1 ] && [ $(( ENVS % GPUS )) -ne 0 ]; then
    echo "RB_ENVS=$ENVS is a TOTAL and must divide RB_GPUS=$GPUS evenly"
    echo "  (unequal ranks would carry unequal weight in the gradient average)"
    exit 2
fi

echo "===== SWAP_TRAIN_SETUP ====="
bash slurm/setup_container.sh 2>&1 | tail -4

# shellcheck source=/dev/null
source env/activate.sh || exit 2
# shellcheck source=/dev/null
source slurm/render_env_egl.sh          # hardware EGL, so the eval video renders on the GPU

# `sugar_swap.train` puts IsaacLab, isaaclab_tasks, isaaclab_rl, sugar_rl and sugar_il on
# sys.path itself, so there is nothing to add here. Isaac Sim is never booted, so this leg
# needs neither the Ubuntu-24.04 image nor the Vulkan ICD workaround experiments/isaac needs.

# ---- wandb credentials, LAST ---------------------------------------------------------
# This has to happen here, not in the sbatch, and it is not belt-and-braces. srun runs the
# step under `bash -l`, so the container's /root/.bashrc is sourced AFTER --export has
# delivered the environment, and it exports a stale WANDB_API_KEY that overwrites the good
# one. The symptom is a 401 "user is not logged in" several minutes into a run that reported
# the right key length at submit time (job 33621324). Sourcing the credential file here,
# after every profile has had its turn, is what actually wins. The same stale key is in the
# container's ~/.netrc, which wandb would otherwise fall back to.
if [ "$LOGGER" = "wandb" ]; then
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
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "[wandb] no key found -- refusing to run unlogged; pass RB_LOGGER=tensorboard"
        exit 2
    fi
    # Pin the mode for the same reason the key is re-sourced: `--export=ALL` copies whatever
    # the submitting shell happened to hold, and an interactive `WANDB_MODE=offline` left over
    # from some unrelated debugging then silently follows a production run onto the cluster.
    # It is a nasty failure because nothing looks wrong -- training proceeds, the run id is
    # minted and recorded, checkpoints land -- but the run never appears on the server; it is
    # only discoverable by noticing the run is missing from the UI, which is how it was found
    # (job 33725508). Losing a run's live curves is worse than the rare offline need, so this
    # defaults ON and RB_WANDB_MODE is the deliberate way to ask for offline.
    export WANDB_MODE=${RB_WANDB_MODE:-online}
    echo "[wandb] mode=$WANDB_MODE"
fi

LOGDIR=$LOGROOT/$RUN
mkdir -p "$LOGDIR"

# ---- one leg at a time --------------------------------------------------------------
# A chain can be queued on several partitions at once, so two legs can start close together;
# both writing $LOGDIR would interleave checkpoints and corrupt the run. A heartbeat file
# settles it. File mtime, not squeue: squeue does not exist in the container, and a heartbeat
# stops on exit, wall-clock kill and preemption alike, so there is no stale lock to clean by
# hand. A leg that finds a live heartbeat WAITS rather than exiting at once, because the
# common case is a handoff from a leg that is finishing, and exiting immediately would throw
# away an allocation that was hours coming. The wait is bounded (RB_WAIT_MIN) so a leg cannot
# sit on a GPU indefinitely doing nothing.
#
# MULTI-NODE: the guard is about two independent LEGS, not the several tasks of one leg. A
# 4-node leg deliberately runs this script once per node, so only node 0 owns the heartbeat;
# the others would otherwise find node 0's file, conclude a rival leg holds the directory, and
# wait 25 minutes while node 0's torchrun blocks at a rendezvous they never join -- a hang, and
# one that looks like a network problem rather than a lock.
HB=$LOGDIR/.heartbeat
WAIT_MIN=${RB_WAIT_MIN:-25}
waited=0
if [ "${SLURM_NODEID:-0}" -ne 0 ]; then
    echo "===== node_rank=${SLURM_NODEID} joins the rendezvous; node 0 owns the heartbeat"
    HB=""
fi
while [ -n "$HB" ] && [ -f "$HB" ]; do
    age=$(( $(date +%s) - $(cat "$HB" 2>/dev/null || echo 0) ))
    if [ "$age" -ge 180 ]; then
        echo "===== stale heartbeat (${age}s old), taking over"
        break
    fi
    if [ "$waited" -ge $(( WAIT_MIN * 60 )) ]; then
        echo "===== ANOTHER LEG STILL LIVE after ${WAIT_MIN} min -- exiting rather than"
        echo "      sharing $LOGDIR; resubmit when it ends"
        exit 0
    fi
    [ "$waited" -eq 0 ] && echo "===== another leg is live (heartbeat ${age}s old); waiting up" \
                                "to ${WAIT_MIN} min for it to finish"
    sleep 30
    waited=$(( waited + 30 ))
done
if [ -n "$HB" ]; then
    while true; do date +%s > "$HB"; sleep 60; done &
    HBPID=$!
    trap 'kill $HBPID 2>/dev/null; rm -f "$HB"' EXIT
fi

# ---- resume ------------------------------------------------------------------------
# Highest saved iteration. --max-iterations is an absolute target, so the resumed leg asks
# `sugar_swap.train` for max_iterations - current_learning_iteration; getting that wrong
# retrains the same prefix on every leg and the run never advances.
LAST=$(ls -1 "$LOGDIR"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
if [ -n "$LAST" ]; then
    echo "===== RESUMING from iteration $LAST"
    set -- --resume "$LOGDIR/model_$LAST.pt"
else
    echo "===== FRESH START"
    set --
fi
[ -n "$SEED" ] && set -- "$@" --seed "$SEED"

echo "===== SWAP_TRAIN_START ====="
echo "===== $ENVS envs TOTAL, batch $(( ENVS * 24 )), target $ITERS iters, save every $SAVE"

# ---- single process or torchrun -----------------------------------------------------
# RB_GPUS is the WORLD SIZE (total ranks), RB_NODES the node count; ranks per node is the
# quotient. Keeping RB_GPUS as the total means the divisibility check above still guards the
# thing that matters -- that the batch splits evenly across ranks -- on one node or four.
#
# Only rank 0 records the evaluation video (~2 min) while the others wait in the next
# all-reduce, so NCCL's watchdog has to tolerate a gap far longer than a training iteration.
# The default 600 s would be a coin flip on a contended node.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}
NODES=${RB_NODES:-1}
if [ "$GPUS" -gt 1 ] && [ $(( GPUS % NODES )) -ne 0 ]; then
    echo "RB_GPUS=$GPUS (world size) must divide RB_NODES=$NODES evenly"; exit 2
fi
PER_NODE=$(( GPUS / NODES ))
if [ "$GPUS" -gt 1 ]; then
    echo "===== DDP: $GPUS ranks x $(( ENVS / GPUS )) envs/rank = $ENVS envs TOTAL," \
         "batch $(( ENVS * 24 )) -- unchanged by RB_GPUS"
    if [ "$NODES" -gt 1 ]; then
        # Static rendezvous rather than --standalone, which binds localhost and would give
        # each node its own separate world of PER_NODE ranks that trains happily and never
        # exchanges a gradient -- a silent wrong answer, not a crash.
        #
        # MASTER_ADDR/PORT must be resolved by the SBATCH, not here: this script runs inside
        # the container, which has no SLURM client, so `scontrol` is "command not found".
        # Failing loudly beats falling back to localhost, which would produce exactly the
        # four-separate-worlds bug this rendezvous exists to prevent.
        if [ -z "${MASTER_ADDR:-}" ]; then
            echo "RB_NODES=$NODES but MASTER_ADDR is unset -- the sbatch must export it" >&2
            exit 2
        fi
        MASTER_PORT=${MASTER_PORT:-$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))}
        echo "===== ${NODES} nodes x ${PER_NODE} ranks, node_rank=${SLURM_NODEID:-0}," \
             "master ${MASTER_ADDR}:${MASTER_PORT}"
        launcher=(torchrun --nnodes="$NODES" --node_rank="${SLURM_NODEID:-0}"
                  --nproc_per_node="$PER_NODE"
                  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
    else
        launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="$PER_NODE")
    fi
else
    launcher=(python -u)
fi

# shellcheck disable=SC2086
"${launcher[@]}" -m sugar_swap.train \
    --num-envs "$ENVS" --max-iterations "$ITERS" --save-interval "$SAVE" \
    --eval-minutes "$EVAL_MIN" --video-frames 400 \
    --run-name "$RUN" --log-root "$LOGROOT" --logger "$LOGGER" \
    "$@" $EXTRA
status=$?
echo "===== SWAP_TRAIN_DONE status=$status ====="
exit $status
