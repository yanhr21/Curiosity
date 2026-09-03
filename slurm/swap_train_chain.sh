#!/bin/bash
# Submit the sugar_swap Refiner run as a chain of dependent 4 h legs.
#
#   bash slurm/swap_train_chain.sh                    # 6 legs, defaults from the sbatch
#   RB_RUN=my_run bash slurm/swap_train_chain.sh 12   # 12 legs
#   RB_GPUS=8 RB_ENVS=4096 bash slurm/swap_train_chain.sh 8    # eight GPUs per leg
#
# RB_GPUS selects the sbatch: 1 -> slurm/swap_train.sbatch, anything more ->
# slurm/swap_train8.sbatch (which asks for 8 and is the only multi-GPU shape measured).
# RB_ENVS is a TOTAL across ranks either way, so RB_GPUS changes the wall clock and never the
# batch; see the header of slurm/swap_train_leg.sh for why that convention differs from
# slurm/train_leg.sh's.
#
# `interactive` caps a job at 4 h, so a long run is cut into legs, each submitted with
# --dependency=afterany on the one before. `afterany`, not `afterok`: a leg killed by the
# wall clock exits non-zero, and that is the NORMAL way a leg ends here, so `afterok` would
# stall the chain at the first leg.
#
# Nothing is lost at a boundary. swap_train_leg.sh resumes from the highest checkpoint in the
# run directory, so a leg ending -- cleanly, by wall clock, or by preemption -- is the same
# event, and --save-interval bounds what is redone. Legs beyond what the run needs cost
# nothing: sugar_swap.train exits immediately once current_learning_iteration reaches
# --max-iterations, so over-provisioning the chain is the safe direction.
#
# The QOS on `interactive` allows 20 SUBMITTED jobs per user, dev nodes included, so 20 legs
# is the ceiling and 80 h of wall clock. Check what you already hold before asking for a long
# chain: `squeue -u $USER`.
#
# Submit from the repo root; swap_train.sbatch resolves paths from SLURM_SUBMIT_DIR.
set -u

LEGS=${1:-6}
REPO=${RB_REPO:-$PWD}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO"; exit 2; }

export RB_GPUS=${RB_GPUS:-1}
# The default run name carries the rank count. Without that an RB_GPUS=8 chain submitted with
# no RB_RUN would adopt the single-GPU run's directory and resume from its checkpoints --
# harmless numerically, since the batch and the policy shape are identical, but it silently
# merges two runs into one wandb curve and one checkpoint series, and the reader has no way to
# see where the rank count changed. Pass RB_RUN explicitly for anything real.
RUN=${RB_RUN:-$([ "$RB_GPUS" -gt 1 ] && echo carrybox_refiner_swap8 || echo carrybox_refiner_swap)}
# Exported, not just set, so `sbatch`'s default --export=ALL carries them into the job. A leg
# that inherits the run name but not the log root would resume from the WRONG run's
# checkpoints, which is worse than failing.
export RB_RUN=$RUN
export RB_LOGROOT=${RB_LOGROOT:-logs/swap_refiner}
export RB_ITERS=${RB_ITERS:-30001}
# TOTAL envs across ranks. The default follows RB_GPUS: one GPU keeps the 512 this chain has
# always used, and eight GPUs default to SUGAR's own 4096 -- which is 512 per rank, the same
# per-GPU load that was measured, and the same batch SUGAR registers.
export RB_ENVS=${RB_ENVS:-$([ "$RB_GPUS" -gt 1 ] && echo 4096 || echo 512)}
export RB_EVAL_MIN=${RB_EVAL_MIN:-20}
export RB_SAVE=${RB_SAVE:-25}
export RB_LOGGER=${RB_LOGGER:-wandb}
export RB_SEED=${RB_SEED:-}

# #SBATCH --gres and --nodes cannot be parameterised, so the shape picks the file.
export RB_NODES=${RB_NODES:-1}
if [ "$RB_NODES" -gt 1 ]; then
    SBATCH_FILE=slurm/swap_train32.sbatch
elif [ "$RB_GPUS" -gt 1 ]; then
    SBATCH_FILE=slurm/swap_train8.sbatch
else
    SBATCH_FILE=slurm/swap_train.sbatch
fi
if [ $(( RB_GPUS % RB_NODES )) -ne 0 ]; then
    echo "RB_GPUS=$RB_GPUS (world size) must divide RB_NODES=$RB_NODES evenly"; exit 2
fi
[ -r "$REPO/$SBATCH_FILE" ] || { echo "missing $SBATCH_FILE"; exit 2; }
if [ $(( RB_ENVS % RB_GPUS )) -ne 0 ]; then
    echo "RB_ENVS=$RB_ENVS is a TOTAL and must divide RB_GPUS=$RB_GPUS evenly"; exit 2
fi

# Named on the sbatch line rather than trusting inheritance. `--export=ALL` is the usual
# default, but a site can set SBATCH_EXPORT=NONE, and the failure mode is silent and
# expensive: a leg missing RB_LOGROOT writes a second run directory and the chain forks.
#
# WANDB_API_KEY is deliberately NOT here -- swap_train.sbatch reads it from a file outside the
# repo, so the secret stays out of both git and the job record `scontrol show job` prints.
EXPORT="ALL,RB_RUN=$RUN,RB_LOGROOT=$RB_LOGROOT,RB_ITERS=$RB_ITERS,RB_ENVS=$RB_ENVS"
EXPORT="$EXPORT,RB_EVAL_MIN=$RB_EVAL_MIN,RB_SAVE=$RB_SAVE,RB_LOGGER=$RB_LOGGER"
EXPORT="$EXPORT,RB_SEED=$RB_SEED,RB_GPUS=$RB_GPUS,RB_NODES=$RB_NODES"
EXPORT="$EXPORT,RB_CONE=${RB_CONE:-pyramidal},RB_IMPRATIO=${RB_IMPRATIO:-20.0}"
# Pinned rather than left to builder.py's default so that every leg of a chain trains under
# the value the chain was SUBMITTED with, even if the default later moves: a chain spans days,
# and a leg that silently picked up a different self-collision rule mid-run would be the same
# class of bug this flag exists to fix (the pre-fix env differs from this one by nothing else,
# and that difference was worth the whole box lift). Note the value is not recoverable from
# `scontrol show job` afterwards -- it does not print the export list -- so the run's own log
# is the record; builder.py prints the mode at build time.
EXPORT="$EXPORT,RB_SELF_COLLISION=${RB_SELF_COLLISION:-weld}"

# RB_PARTITION overrides the sbatch's #SBATCH -p (the CLI wins over the directive).
# `interactive` has 10 nodes and is shared with everyone else; polar3/polar4 have ~1200 each
# and no such scarcity, so they are worth trying if `interactive` will not schedule.
PART=${RB_PARTITION:-}
part_flag=""
[ -n "$PART" ] && part_flag="-p $PART"

# RB_BEGIN holds the first leg until a wall-clock time (sbatch --begin syntax), for when the
# run is ALREADY going somewhere else -- typically borrowed on a dev node.
#
# Without it the chain can eat itself: a leg that starts while another holds the run directory
# waits RB_WAIT_MIN and then exits 0, `afterany` promptly starts the next, and the whole chain
# is consumed in a couple of hours without training anything. Gating on the borrowed
# allocation's EndTime avoids the overlap, and is better than a long RB_WAIT_MIN, which leaves
# a node idling while it waits its turn.
BEGIN=${RB_BEGIN:-}
begin_flag=""
[ -n "$BEGIN" ] && begin_flag="--begin=$BEGIN"

echo "submitting $LEGS legs for sugar_swap run '$RUN' (${PART:-partition from sbatch})"
echo "  $SBATCH_FILE, log root $RB_LOGROOT, target $RB_ITERS iters"
echo "  $RB_GPUS GPU(s) x $(( RB_ENVS / RB_GPUS )) envs/rank = $RB_ENVS envs TOTAL" \
     "(batch $(( RB_ENVS * 24 )))"
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
if [ "$RB_NODES" -gt 1 ]; then JOBNAME=rb_swaptrain32
elif [ "$RB_GPUS" -gt 1 ]; then JOBNAME=rb_swaptrain8
else JOBNAME=rb_swaptrain
fi
echo "watch:       squeue -u \$USER -n $JOBNAME"
