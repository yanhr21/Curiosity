#!/bin/bash
# Submit a training run as a chain of dependent legs.
#
#   bash slurm/train_chain.sh                    # 6 legs, defaults from train.sbatch
#   RB_RUN=my_run bash slurm/train_chain.sh 8    # 8 legs
#
# polar3/polar4 cap a job at 4 h but turn over quickly, which is the trade this makes: the
# ~13 h run is cut into legs, each submitted with --dependency=afterany on the one before.
# `afterany`, not `afterok`: a leg that is killed by the wall clock exits non-zero, and that
# is the NORMAL way a leg ends here, so `afterok` would stall the chain at the first leg.
#
# Nothing is lost at a boundary. train_leg.sh resumes from the highest checkpoint in the run
# directory, so a leg ending -- cleanly, by wall clock, or by preemption -- is the same
# event, and --save-interval bounds what is redone. Legs beyond what the run needs cost
# nothing: train_bcppo exits immediately once current_learning_iteration reaches
# --max-iterations, so over-provisioning the chain is the safe direction.
#
# Submit from the repo root; train.sbatch resolves paths from SLURM_SUBMIT_DIR.
set -u

LEGS=${1:-6}
REPO=${RB_REPO:-$PWD}
[ -d "$REPO/slurm" ] || { echo "not a repo root: $REPO"; exit 2; }

RUN=${RB_RUN:-carrybox_bcppo_ddp}
# Exported, not just set, so `sbatch`'s default --export=ALL carries them into the job. A
# leg that inherits the stage but not the log root would resume from the WRONG run's
# checkpoints, which is worse than failing.
export RB_RUN=$RUN
export RB_STAGE=${RB_STAGE:-tracker}
export RB_LOGROOT=${RB_LOGROOT:-logs/newton_bcppo}
export RB_ITERS=${RB_ITERS:-3000}
export RB_ENVS=${RB_ENVS:-64}
export RB_GPUS=${RB_GPUS:-8}
export RB_EVAL_MIN=${RB_EVAL_MIN:-20}
export RB_SAVE=${RB_SAVE:-25}
export RB_LOGGER=${RB_LOGGER:-wandb}

# Named on the sbatch line rather than trusting inheritance. `--export=ALL` is the usual
# default, but a site can set SBATCH_EXPORT=NONE, and the failure mode is silent and
# expensive: a leg missing RB_STAGE runs the TRACKER instead, writing BCPPO checkpoints
# into whatever log root it also failed to inherit. Better to state them.
#
# WANDB_API_KEY is deliberately NOT here -- train.sbatch reads it from a file outside the
# repo, so the secret stays out of both git and the job record `scontrol show job` prints.
EXPORT="ALL,RB_STAGE=$RB_STAGE,RB_RUN=$RUN,RB_LOGROOT=$RB_LOGROOT,RB_ITERS=$RB_ITERS"
EXPORT="$EXPORT,RB_ENVS=$RB_ENVS,RB_GPUS=$RB_GPUS,RB_EVAL_MIN=$RB_EVAL_MIN"
EXPORT="$EXPORT,RB_SAVE=$RB_SAVE,RB_LOGGER=$RB_LOGGER"

# RB_PARTITION overrides train.sbatch's #SBATCH -p (the CLI wins over the directive).
#
# `interactive` is worth knowing the shape of before choosing it: 9 nodes of which 6 are
# DRAINED, a 4 h cap, and a QOS allowing 3 running jobs, 20 SUBMITTED and 24 GPUs per user.
# The submit cap is the binding one for a chain -- 20 legs is 80 h of wall clock and the
# ceiling, dev nodes included -- but the 3 usable nodes are shared with everyone else, so a
# whole-node leg schedules only when one of them empties. polar3/polar4 have ~1200 nodes
# each and no such scarcity; they are the better default and the reason this is a knob
# rather than a change of the directive.
PART=${RB_PARTITION:-}
part_flag=""
[ -n "$PART" ] && part_flag="-p $PART"

# RB_BEGIN holds the first leg until a wall-clock time (sbatch --begin syntax), for when the
# run is ALREADY going somewhere else -- typically borrowed on an interactive node.
#
# Without it the chain eats itself. A leg that starts while another holds the run directory
# waits RB_WAIT_MIN and then exits 0; `afterany` promptly starts the next leg, which does the
# same, so the whole chain can be consumed in a couple of hours without training anything.
# Gating on the borrowed allocation's EndTime avoids the overlap entirely, and is better than
# a long RB_WAIT_MIN, which would leave a whole node idling while it waits its turn.
BEGIN=${RB_BEGIN:-}
begin_flag=""
[ -n "$BEGIN" ] && begin_flag="--begin=$BEGIN"

echo "submitting $LEGS legs for run '$RUN' (stage=$RB_STAGE, ${PART:-partition from sbatch})"
echo "  log root $RB_LOGROOT, target $RB_ITERS iters, $RB_GPUS GPU(s) x $RB_ENVS worlds"
dep="$begin_flag"
note="${BEGIN:+not before $BEGIN}"
ids=()
for i in $(seq 1 "$LEGS"); do
    # shellcheck disable=SC2086
    out=$(sbatch $part_flag --export="$EXPORT" $dep slurm/train.sbatch 2>&1 | tail -1)
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
echo "watch:       squeue -u \$USER -n rb_train"
