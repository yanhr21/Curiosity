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
export RB_RUN=$RUN

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

echo "submitting $LEGS legs for run '$RUN'${BEGIN:+, first leg not before $BEGIN}"
dep="$begin_flag"
note="${BEGIN:+not before $BEGIN}"
ids=()
for i in $(seq 1 "$LEGS"); do
    # shellcheck disable=SC2086
    out=$(sbatch $dep slurm/train.sbatch 2>&1 | tail -1)
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
