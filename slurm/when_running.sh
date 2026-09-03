#!/bin/bash
# Block until a queued job starts, then run a command. For pouncing on a node the moment the
# scheduler hands it over instead of leaving it idle until someone notices.
#
#   bash slurm/when_running.sh 33705894 -- bash slurm/some_step.sh
#   bash slurm/when_running.sh 33705894 --timeout 14400 -- bash slurm/other.sh
#
# Exits 1 if the job leaves the queue without ever running (cancelled, or failed to start), so
# a caller can tell "never got the node" from "ran and failed". Exits 2 on timeout.
#
# WHY this and not just waiting: `interactive` has 10 nodes for the whole cluster and 8-GPU
# requests can sit in Reason=Priority for an unpredictable time. A 4 h allocation that lands
# unattended at 3am and expires unused is the expensive failure mode -- see
# claude_context/operations.md on partition choice.
set -u
JOBID=${1:-}
shift || true
TIMEOUT=86400
POLL=${RB_POLL:-20}
while [ $# -gt 0 ]; do
    case "$1" in
        --timeout) TIMEOUT=$2; shift 2 ;;
        --) shift; break ;;
        *) echo "unexpected argument: $1"; exit 2 ;;
    esac
done
[ -n "$JOBID" ] && [ $# -gt 0 ] || {
    echo "usage: bash slurm/when_running.sh <jobid> [--timeout SEC] -- <command...>"; exit 2; }

echo "===== WAITING for job $JOBID (poll ${POLL}s, timeout ${TIMEOUT}s)"; date
start=$SECONDS
while :; do
    state=$(squeue -j "$JOBID" -h -o %T 2>/dev/null)
    case "$state" in
        RUNNING)
            echo "===== job $JOBID RUNNING on $(squeue -j "$JOBID" -h -o %N 2>/dev/null)" \
                 "after $(( SECONDS - start ))s"; date
            exec "$@"
            ;;
        "")
            # Gone from the queue. sacct distinguishes "ran and finished" from "never started",
            # but either way there is no allocation left to use.
            echo "===== job $JOBID left the queue without us seeing it run"
            sacct -j "$JOBID" --format=JobID,State,Elapsed -n 2>/dev/null | head -3
            exit 1
            ;;
    esac
    if [ $(( SECONDS - start )) -ge "$TIMEOUT" ]; then
        echo "===== timed out after ${TIMEOUT}s with job $JOBID in state ${state:-unknown}"
        exit 2
    fi
    sleep "$POLL"
done
