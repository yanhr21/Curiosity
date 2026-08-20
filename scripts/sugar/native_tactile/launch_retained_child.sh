#!/usr/bin/env bash
# Launch one bounded child in its own process group inside a retained allocation.

set -euo pipefail

record=""
status=""
log=""
tag=""
foreground=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --record) record="$2"; shift 2 ;;
        --status) status="$2"; shift 2 ;;
        --log) log="$2"; shift 2 ;;
        --tag) tag="$2"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        --) shift; break ;;
        *) echo "unknown launcher argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "launch_retained_child.sh must run inside a retained Slurm allocation" >&2
    exit 2
fi
if [[ -z "$record" || -z "$status" || -z "$log" || -z "$tag" || $# -eq 0 ]]; then
    echo "required: --record PATH --status PATH --log PATH --tag TAG -- COMMAND..." >&2
    exit 2
fi
for path in "$record" "$status" "$log"; do
    if [[ -e "$path" ]]; then
        echo "refusing to overwrite: $path" >&2
        exit 2
    fi
    mkdir -p "$(dirname "$path")"
done

run_child() {
setsid bash -c '
    record=$1
    status=$2
    tag=$3
    shift 3
    pid=$$
    pgid=$(ps -o pgid= -p "$pid" | tr -d " ")
    {
        printf "slurm_job_id=%s\n" "$SLURM_JOB_ID"
        printf "host=%s\n" "$(hostname)"
        printf "child_pid=%s\n" "$pid"
        printf "child_pgid=%s\n" "$pgid"
        printf "command_tag=%s\n" "$tag"
        printf "slurm_step_gpus=%s\n" "${SLURM_STEP_GPUS:-}"
        printf "cuda_visible_devices=%s\n" "${CUDA_VISIBLE_DEVICES:-}"
        printf "command="
        printf "%q " "$@"
        printf "\n"
        printf "started_utc=%s\n" "$(date -u +%FT%TZ)"
    } > "$record"
    set +e
    "$@"
    rc=$?
    set -e
    {
        printf "exit_code=%s\n" "$rc"
        printf "finished_utc=%s\n" "$(date -u +%FT%TZ)"
    } > "$status"
    exit "$rc"
' retained-child "$record" "$status" "$tag" "$@" > "$log" 2>&1 < /dev/null
}

if [[ "$foreground" -eq 1 ]]; then
    printf "launching_foreground tag=%s log=%s\n" "$tag" "$log"
    run_child "$@"
else
    run_child "$@" &
    launcher_pid=$!
    printf "launched_pid=%s tag=%s log=%s\n" "$launcher_pid" "$tag" "$log"
fi
