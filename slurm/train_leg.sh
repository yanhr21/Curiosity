#!/bin/bash
# One leg of a training run, from inside the container. Invoked by slurm/train.sbatch.
#
#   bash slurm/train_leg.sh          # reads RB_* from the environment
#
# This is a FILE rather than a string embedded in the sbatch for two reasons. It needs `&`
# (for the heartbeat) inside a chain of `&&`, and `&` binds looser than `&&`, so written
# inline the whole preceding chain gets backgrounded and the variables set in it are never
# visible to what follows -- which is exactly the bug that killed job 33620512 with
# "HB: unbound variable". And a file is testable: `bash -n` catches what escaping inside
# `bash -lc "..."` hides.
#
# Reads: RB_RUN RB_ENVS RB_ITERS RB_SAVE RB_EVAL_MIN RB_LOGROOT RB_LOGGER RB_EXTRA
set -u

RUN=${RB_RUN:-carrybox_bcppo}
ENVS=${RB_ENVS:-512}
ITERS=${RB_ITERS:-3000}
SAVE=${RB_SAVE:-25}
EVAL_MIN=${RB_EVAL_MIN:-10}
LOGROOT=${RB_LOGROOT:-logs/newton_bcppo}
LOGGER=${RB_LOGGER:-wandb}
EXTRA=${RB_EXTRA:-}

echo "===== TRAIN_SETUP ====="
bash slurm/setup_container.sh 2>&1 | tail -4

# shellcheck source=/dev/null
source env/activate.sh || exit 2
# shellcheck source=/dev/null
source slurm/render_env_egl.sh          # hardware EGL, so the eval video renders on the GPU

# ---- wandb credentials, LAST ---------------------------------------------------------
# This has to happen here, not in the sbatch, and it is not belt-and-braces. srun runs the
# step under `bash -l`, so the container's /root/.bashrc is sourced AFTER --export has
# delivered the environment, and it exports a stale WANDB_API_KEY that overwrites the good
# one. The symptom is a 401 "user is not logged in" several minutes into a run that reported
# the right key length at submit time (job 33621324). Sourcing the credential file here, after
# every profile has had its turn, is what actually wins. The same stale key is in the
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
fi

LOGDIR=$LOGROOT/$RUN
mkdir -p "$LOGDIR"

# ---- one leg at a time --------------------------------------------------------------
# The run is queued on several partitions at once (see train.sbatch), so two legs can start
# close together; both writing $LOGDIR would interleave checkpoints. A heartbeat file settles
# it. File mtime, not squeue: squeue does not exist in the container, and a heartbeat stops on
# exit, wall-clock kill and preemption alike, so there is no stale lock to clean up by hand.
# A leg that finds a live heartbeat WAITS rather than exiting at once. The common case is a
# handoff: a borrowed allocation is finishing while this one has just been scheduled, and
# exiting immediately would throw away a queued allocation that was hours coming. The wait is
# bounded (RB_WAIT_MIN) so a leg cannot sit on a GPU indefinitely doing nothing.
HB=$LOGDIR/.heartbeat
WAIT_MIN=${RB_WAIT_MIN:-25}
waited=0
while [ -f "$HB" ]; do
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
while true; do date +%s > "$HB"; sleep 60; done &
HBPID=$!
trap 'kill $HBPID 2>/dev/null; rm -f "$HB"' EXIT

# ---- resume ------------------------------------------------------------------------
# Highest saved iteration, so preemption, a wall-clock kill and a manual resubmission are all
# the same event. --max-iterations is an absolute target, so a resumed leg trains the
# remainder and the run converges on one endpoint however many legs it takes.
LAST=$(ls -1 "$LOGDIR"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
if [ -n "$LAST" ]; then
    echo "===== RESUMING from iteration $LAST"
    set -- --resume "$LOGDIR/model_$LAST.pt"
else
    echo "===== FRESH START"
    set --
fi

echo "===== TRAIN_START ====="
# shellcheck disable=SC2086
python -u -m sugar_newton.rl.train_bcppo \
    --num-envs "$ENVS" --max-iterations "$ITERS" --save-interval "$SAVE" \
    --eval-minutes "$EVAL_MIN" --video-frames 400 \
    --run-name "$RUN" --log-root "$LOGROOT" --logger "$LOGGER" \
    "$@" $EXTRA
status=$?
echo "===== TRAIN_DONE status=$status ====="
exit $status
