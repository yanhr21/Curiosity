#!/bin/bash
# Submit the IsaacLab/PhysX reference run as a chain of dependent 4 h legs.
#
#   bash slurm/isaac_train_chain.sh                 # 3 legs on `interactive`
#   bash slurm/isaac_train_chain.sh 4               # 4 legs
#   RB_PARTITION=polar3,polar4 bash slurm/isaac_train_chain.sh 3     # the hedge
#   RB_SBATCH=experiments/isaac/isaac_video.sbatch bash slurm/isaac_train_chain.sh 3
#                                                   # the evaluation-video loop, same chain
#
# The sibling of slurm/swap_train_chain.sh, for the reference rather than the Newton port, and
# it works the same way for the same reasons. `interactive` caps a job at 4 h, so a ~9.5 h run
# is cut into legs, each submitted with --dependency=afterany on the one before.
#
# `afterany`, NOT `afterok`. A leg killed by the wall clock exits non-zero, and that is the
# NORMAL way a leg of a long run ends, so `afterok` would stall the chain at the first
# boundary -- and an expiry is precisely the event the chain exists to survive.
#
# Nothing is lost at a boundary. experiments/isaac/train_refiner_leg.sh finds the highest
# model_N.pt in the run directory and resumes from it, so a leg ending cleanly, by wall clock,
# by preemption or by a crash are the same event, and RB_SAVE_INTERVAL bounds what is redone
# (120 iterations ~= 7 min). Legs beyond what the run needs are nearly free: SUGAR's own
# train.py reads SUGAR_TOTAL_ITERATION_BUDGET as an absolute target (train.py:597) and asks
# rsl_rl for `budget - current` iterations, so a leg starting past the budget boots and exits.
# Over-provisioning is therefore the safe direction.
#
# PARTITIONS -- read .cursor/rules/slurm-partitions.mdc; this shape has exactly one legal
# interactive home. The job needs 8 GPUs on one node, and `interactive_singlenode`'s QOS
# refuses any request above 1 GPU, so despite its 1243 nodes it cannot run this at all.
# `interactive` can, and caps a user at 3 concurrent nodes. polar3/polar4 have ~1200 nodes each
# and will take the job, but queue on Reason=Priority behind other users for an unbounded time.
# So: submit to `interactive` for a leg someone is waiting on, hedge the same chain onto
# polar3,polar4, and cancel the loser once one lands. Overlap is safe if you are slow to
# cancel -- the heartbeat in the leg script stops two legs sharing the run directory -- but do
# cancel, because a losing leg that waits out RB_WAIT_MIN and exits 0 lets `afterany` start the
# next one, and a chain can eat itself that way in an afternoon.
#
# Defaults reproduce the reference EXACTLY as it has been running (job 33753485): 8 GPUs,
# 4096 envs TOTAL, seed 42, checkpoints every 120 iterations. RB_ENVS is a TOTAL and
# train_refiner.sh divides it by the GPU count, because SUGAR's --num_envs is PER RANK; 4096
# passed through unchanged would be 32,768 envs and eight times the batch, which does not error
# and quietly becomes a different experiment.
#
# Submit from the repo root; the sbatch resolves paths from SLURM_SUBMIT_DIR.
set -u

LEGS=${1:-3}
REPO=${RB_REPO:-$PWD}
[ -d "$REPO/experiments/isaac" ] || { echo "not a repo root: $REPO"; exit 2; }
# RB_SBATCH selects the payload. The chaining, the afterany reasoning and the partition
# argument are identical for the training legs and for the evaluation-video loop, and the loop
# was dying at its own 4 h wall for exactly the same reason, so it reuses this rather than
# getting a near-copy of it. Both sbatches read the same RB_* names.
SBATCH_FILE=${RB_SBATCH:-experiments/isaac/isaac_train.sbatch}
[ -r "$REPO/$SBATCH_FILE" ] || { echo "missing $SBATCH_FILE"; exit 2; }
JOBNAME=$(sed -nE 's/^#SBATCH --job-name=(.*)/\1/p' "$REPO/$SBATCH_FILE" | head -1)

RUN=${RB_RUN:-isaac_refiner_ref2}
GPUS=${RB_GPUS:-8}
ENVS=${RB_ENVS:-4096}                   # TOTAL across ranks
SEED=${RB_SEED:-42}
SAVE_INTERVAL=${RB_SAVE_INTERVAL:-120}
# ~10,000 iterations is SUGAR's refiner budget; at the measured ~3.4 s/iter that is ~9.5 h.
TOTAL_ITERS=${RB_TOTAL_ITERS:-10000}
# Seeds run_meta.json on the first leg only, so the run wandb ALREADY created is adopted
# instead of abandoned. Once run_meta.json exists it is the sole source of truth and this is
# ignored -- otherwise a stale value here could split a run in half at some later leg.
WANDB_ID=${RB_WANDB_ID:-7ftxsusp}
WANDB_ENTITY=${RB_WANDB_ENTITY:-nvr-amri}

if [ $(( ENVS % GPUS )) -ne 0 ]; then
    echo "RB_ENVS=$ENVS (total) must divide RB_GPUS=$GPUS evenly"; exit 2
fi

# Named on the sbatch line rather than trusted to inheritance. `--export=ALL` is the usual
# default, but a site can set SBATCH_EXPORT=NONE and the failure is silent and expensive: a leg
# missing RB_RUN resumes a different run's checkpoints, and one missing RB_ENVS trains a
# different experiment under the reference's name.
#
# WANDB_API_KEY is deliberately absent: train_refiner.sh reads it from a file outside the repo,
# so the secret stays out of both git and what `scontrol show job` prints.
EXPORT="ALL,RB_RUN=$RUN,RB_GPUS=$GPUS,RB_ENVS=$ENVS,RB_SEED=$SEED"
EXPORT="$EXPORT,RB_SAVE_INTERVAL=$SAVE_INTERVAL,RB_TOTAL_ITERS=$TOTAL_ITERS"
EXPORT="$EXPORT,RB_WANDB_ID=$WANDB_ID,RB_WANDB_ENTITY=$WANDB_ENTITY"
EXPORT="$EXPORT,RB_WAIT_MIN=${RB_WAIT_MIN:-25}"
[ -n "${RB_LOG_DIR:-}" ] && EXPORT="$EXPORT,RB_LOG_DIR=$RB_LOG_DIR"
# Payload-specific names, e.g. RB_EVERY / RB_ITERS_MIN / RB_WANDB_RUN for isaac_video.sbatch.
# One --export flag only: repeated flags do NOT merge, the last one silently wins.
[ -n "${RB_EXTRA_EXPORT:-}" ] && EXPORT="$EXPORT,$RB_EXTRA_EXPORT"

# RB_PARTITION overrides the sbatch's `#SBATCH -p interactive` (the CLI wins over the
# directive). This is the knob the hedge above is made of.
PART=${RB_PARTITION:-}
part_flag=""
[ -n "$PART" ] && part_flag="-p $PART"

# RB_BEGIN holds the first leg until a wall-clock time (sbatch --begin syntax), for when the
# run is ALREADY going somewhere else -- typically on a hand-driven node. Gating on that
# allocation's EndTime is better than leaning on the heartbeat, which parks a granted node
# doing nothing while it waits its turn.
BEGIN=${RB_BEGIN:-}
begin_flag=""
[ -n "$BEGIN" ] && begin_flag="--begin=$BEGIN"

echo "submitting $LEGS legs of $SBATCH_FILE for run '$RUN'" \
     "(${PART:-partition from the sbatch})"
case "$SBATCH_FILE" in
    *isaac_train.sbatch)
        echo "  $GPUS GPU(s) x $(( ENVS / GPUS )) envs/rank = $ENVS envs TOTAL (batch $(( ENVS * 24 )))"
        echo "  seed $SEED, save every $SAVE_INTERVAL, absolute budget $TOTAL_ITERS iters"
        echo "  wandb $WANDB_ENTITY/sugar_newton run $WANDB_ID" \
             "(seed value; run_meta.json wins if present)";;
esac
dep="$begin_flag"
note="${BEGIN:+not before $BEGIN}"
ids=()
for i in $(seq 1 "$LEGS"); do
    # shellcheck disable=SC2086
    out=$(sbatch $part_flag --export="$EXPORT" $dep "$SBATCH_FILE" 2>&1 | tail -1)
    id=${out##* }
    case "$id" in
        ''|*[!0-9]*) echo "leg $i FAILED: $out"; exit 1;;
    esac
    ids+=("$id")
    printf "  leg %-2d %s%s\n" "$i" "$id" "${note:+  ($note)}"
    dep="--dependency=afterany:$id"
    note="after $id"
done

echo
echo "chain: ${ids[*]}"
echo "cancel all:  scancel ${ids[*]}"
echo "watch:       squeue -u \$USER -n ${JOBNAME:-isaac_ref_leg}"
echo "log:         $(dirname "$SBATCH_FILE")/$(basename "$SBATCH_FILE" .sbatch)_<jobid>.log"
