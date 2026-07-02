#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TARGET="${TARGET:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_phase00_ref_tactile}"
WINDOW_NAME="${WINDOW_NAME:-p00_ref_${TARGET:-unset}_sanity}"
RUN_TAG="${RUN_TAG:-p00_ref_sanity_${TARGET:-unset}_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-$ROOT/logs/newton/phase00/ref_tactile/reference_sanity/${RUN_TAG}.srun.log}"

cd "$ROOT"
mkdir -p "$(dirname "$LOG_PATH")"

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a running Curiosity tmux-held allocation." >&2
  exit 2
fi
if [[ "$TARGET" != "univtac" && "$TARGET" != "tacauchy" ]]; then
  echo "ERROR: TARGET must be one of: univtac, tacauchy" >&2
  exit 3
fi

target_python=""
case "$TARGET" in
  univtac)
    target_python="${UNIVTAC_PYTHON:-}"
    if [[ -z "$target_python" && -x "$ROOT/envs/univtac/conda/bin/python" ]]; then
      target_python="$ROOT/envs/univtac/conda/bin/python"
    fi
    if [[ -z "$target_python" && -x "$ROOT/envs/univtac/.venv/bin/python" ]]; then
      target_python="$ROOT/envs/univtac/.venv/bin/python"
    fi
    ;;
  tacauchy)
    target_python="${TACAUCHY_PYTHON:-}"
    if [[ -z "$target_python" && -x "$ROOT/envs/tacauchy/conda/bin/python" ]]; then
      target_python="$ROOT/envs/tacauchy/conda/bin/python"
    fi
    if [[ -z "$target_python" && -x "$ROOT/envs/tacauchy/.venv/bin/python" ]]; then
      target_python="$ROOT/envs/tacauchy/.venv/bin/python"
    fi
    ;;
esac
if [[ -z "$target_python" && "${ALLOW_MISSING_REFERENCE_ENV_BLOCKER_RUN:-0}" != "1" ]]; then
  "$ROOT/experiments/configs/phase00/ref_tactile/envprep/check_reference_env_availability.sh" >/dev/null
  echo "ERROR: missing executable reference environment for TARGET=$TARGET." >&2
  echo "Set ${TARGET^^}_PYTHON, create envs/$TARGET/conda or envs/$TARGET/.venv, or set ALLOW_MISSING_REFERENCE_ENV_BLOCKER_RUN=1 to intentionally record a compute-side blocker." >&2
  echo "Availability report: $ROOT/experiments/reports/phase00/ref_tactile/envprep/reference_env_availability.md" >&2
  exit 7
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session not found: $TMUX_SESSION" >&2
  exit 4
fi
job_state="$(squeue -h -j "$JOB_ID" -o '%T' | head -n 1)"
if [[ "$job_state" != "RUNNING" ]]; then
  echo "ERROR: Slurm job $JOB_ID is not running yet." >&2
  squeue -j "$JOB_ID" -o '%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R' >&2 || true
  exit 5
fi
if tmux list-windows -t "$TMUX_SESSION" -F '#W' | grep -qx "$WINDOW_NAME"; then
  echo "ERROR: tmux window already exists: $TMUX_SESSION:$WINDOW_NAME" >&2
  exit 6
fi

bash -n "$ROOT/experiments/configs/phase00/ref_tactile/run_tactile_reference_sanity_in_alloc.sh"

env_file="$ROOT/logs/newton/phase00/ref_tactile/reference_sanity/${RUN_TAG}_env.sh"
{
  printf 'ROOT=%q\n' "$ROOT"
  printf 'TARGET=%q\n' "$TARGET"
  printf 'RUN_TAG=%q\n' "$RUN_TAG"
  if [[ -n "${UNIVTAC_PYTHON:-}" ]]; then
    printf 'UNIVTAC_PYTHON=%q\n' "$UNIVTAC_PYTHON"
  fi
  if [[ -n "${TACAUCHY_PYTHON:-}" ]]; then
    printf 'TACAUCHY_PYTHON=%q\n' "$TACAUCHY_PYTHON"
  fi
} >"$env_file"

remote_cmd="cd $(printf '%q' "$ROOT") && set -a && source $(printf '%q' "$env_file") && set +a && bash experiments/configs/phase00/ref_tactile/run_tactile_reference_sanity_in_alloc.sh"
srun_cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=1 bash -lc $(printf '%q' "$remote_cmd")"

window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$WINDOW_NAME")"
tmux send-keys -t "$window_id" \
  "$srun_cmd 2>&1 | tee '$LOG_PATH'; printf '\nTACTILE_REFERENCE_SANITY_SRUN_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TARGET=$TARGET
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$WINDOW_NAME
JOB_ID=$JOB_ID
LOG_PATH=$LOG_PATH
ENV_FILE=$env_file
DOWNSTREAM_USE=official_reference_sanity_or_blocker_not_training_not_curiosity_success
EOF
