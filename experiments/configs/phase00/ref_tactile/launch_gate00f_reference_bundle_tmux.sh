#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase00_ref_tactile}"
WINDOW_NAME="${WINDOW_NAME:-p00_gate00f_bundle}"
RUN_TAG="${RUN_TAG:-p00_gate00f_bundle_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/phase00/ref_tactile/gate00f_bundle/${RUN_TAG}.srun.log}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_PATH")"

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a running Curiosity tmux-held allocation." >&2
  exit 2
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session not found: $TMUX_SESSION" >&2
  exit 3
fi
job_state="$(squeue -h -j "$JOB_ID" -o '%T' | head -n 1)"
if [[ "$job_state" != "RUNNING" ]]; then
  echo "ERROR: Slurm job $JOB_ID is not running yet." >&2
  squeue -j "$JOB_ID" -o '%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R' >&2 || true
  exit 4
fi
job_workdir="$(squeue -h -j "$JOB_ID" -o '%Z' | head -n 1)"
if [[ "$job_workdir" != "$ROOT"* ]]; then
  echo "ERROR: Slurm job $JOB_ID is not Curiosity-owned by workdir: $job_workdir" >&2
  exit 5
fi
if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 6
fi

bash -n "$ROOT/experiments/configs/phase00/ref_tactile/run_gate00f_reference_bundle_in_alloc.sh"

env_file="$ROOT/logs/newton/phase00/ref_tactile/gate00f_bundle/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  if [[ -n "${ALLOW_BLOCKER_SANITY:-}" ]]; then
    printf 'ALLOW_BLOCKER_SANITY=%q\n' "$ALLOW_BLOCKER_SANITY"
  fi
  if [[ -n "${REQUIRE_RUNTIME_PREFLIGHT:-}" ]]; then
    printf 'REQUIRE_RUNTIME_PREFLIGHT=%q\n' "$REQUIRE_RUNTIME_PREFLIGHT"
  fi
  if [[ -n "${RUNTIME_REGISTRY:-}" ]]; then
    printf 'RUNTIME_REGISTRY=%q\n' "$RUNTIME_REGISTRY"
  fi
  if [[ -n "${REGISTRY_VALIDATOR:-}" ]]; then
    printf 'REGISTRY_VALIDATOR=%q\n' "$REGISTRY_VALIDATOR"
  fi
  if [[ -n "${GATE00F_RUNTIME_PREFLIGHT_SUMMARY:-}" ]]; then
    printf 'GATE00F_RUNTIME_PREFLIGHT_SUMMARY=%q\n' "$GATE00F_RUNTIME_PREFLIGHT_SUMMARY"
  fi
  if [[ -n "${UNIVTAC_PYTHON:-}" ]]; then
    printf 'UNIVTAC_PYTHON=%q\n' "$UNIVTAC_PYTHON"
  fi
  if [[ -n "${TACAUCHY_PYTHON:-}" ]]; then
    printf 'TACAUCHY_PYTHON=%q\n' "$TACAUCHY_PYTHON"
  fi
  if [[ -n "${ISAACLAB_TACSL_PYTHON:-}" ]]; then
    printf 'ISAACLAB_TACSL_PYTHON=%q\n' "$ISAACLAB_TACSL_PYTHON"
  fi
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/phase00/ref_tactile/run_gate00f_reference_bundle_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nGATE00F_REFERENCE_BUNDLE_SRUN_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
DOWNSTREAM_USE=gate00f_reference_bundle_not_training_not_curiosity_success
EOF
