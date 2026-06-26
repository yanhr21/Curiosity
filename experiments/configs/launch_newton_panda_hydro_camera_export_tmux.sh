#!/usr/bin/env bash
set -euo pipefail

# Launch Newton Panda hydro SensorTiledCamera export in an existing tmux-held allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
JOB_ID="${JOB_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-curiosity_q_sweep_alloc_20260626_145352}"
RUN_TAG="${RUN_TAG:-newton_panda_hydro_camera_export_$(date +%Y%m%d_%H%M%S)}"
WINDOW_NAME="${WINDOW_NAME:-newton_camera_export}"
SCENE="${SCENE:-pen}"
TRACKED_OBJECT="${TRACKED_OBJECT:-official_object}"
CONTROLLER_MODE="${CONTROLLER_MODE:-official_pick_place}"
FINAL_HOLD_DURATION="${FINAL_HOLD_DURATION:-1.0}"
LIFT_HEIGHT_MIN="${LIFT_HEIGHT_MIN:-0.12}"
HOLD_DURATION_MIN="${HOLD_DURATION_MIN:-2.0}"
DROP_HEIGHT_LOSS="${DROP_HEIGHT_LOSS:-0.05}"
PHYSICS_VARIANT_LABEL="${PHYSICS_VARIANT_LABEL:-nominal}"
BODY_MASS_SCALE="${BODY_MASS_SCALE:-1.0}"
SHAPE_FRICTION_SCALE="${SHAPE_FRICTION_SCALE:-1.0}"
OBJECT_MASS_KG="${OBJECT_MASS_KG:-}"
OBJECT_FRICTION_MU="${OBJECT_FRICTION_MU:-}"
FEEDBACK_MIN_CONTACT_COUNT="${FEEDBACK_MIN_CONTACT_COUNT:-20}"
FEEDBACK_ACCEL_THRESHOLD="${FEEDBACK_ACCEL_THRESHOLD:-6.5}"
FEEDBACK_HEIGHT_DROP_THRESHOLD="${FEEDBACK_HEIGHT_DROP_THRESHOLD:-0.015}"
FEEDBACK_INITIAL_LIFT_DURATION_SCALE="${FEEDBACK_INITIAL_LIFT_DURATION_SCALE:-1.35}"
FEEDBACK_LIFT_DURATION_SCALE_MAX="${FEEDBACK_LIFT_DURATION_SCALE_MAX:-2.25}"
FEEDBACK_HOLD_HEIGHT_STEP="${FEEDBACK_HOLD_HEIGHT_STEP:-0.003}"
FEEDBACK_HOLD_HEIGHT_OFFSET_MAX="${FEEDBACK_HOLD_HEIGHT_OFFSET_MAX:-0.03}"
FEEDBACK_STABILIZATION_STEP="${FEEDBACK_STABILIZATION_STEP:-0.25}"
FEEDBACK_STABILIZATION_MAX="${FEEDBACK_STABILIZATION_MAX:-2.0}"
PRE_RECORD_WARMUP_STEPS="${PRE_RECORD_WARMUP_STEPS:-0}"
NUM_STEPS="${NUM_STEPS:-240}"
SAMPLE_STEPS="${SAMPLE_STEPS:-0,60,120,180,239}"
DEVICE="${DEVICE:-cuda:0}"
NEWTON_VENV="${NEWTON_VENV:-$ROOT/envs/newton/.venv}"
TRAINER_VENV="${TRAINER_VENV:-$ROOT/envs/residual_adapter/.venv}"
NEWTON_CACHE_PATH="${NEWTON_CACHE_PATH:-$ROOT/external/newton-assets-cache}"
RESIDUAL_ADAPTER_CHECKPOINT="${RESIDUAL_ADAPTER_CHECKPOINT:-}"
RESIDUAL_ADAPTER_ACTIVE_THRESHOLD="${RESIDUAL_ADAPTER_ACTIVE_THRESHOLD:-0.5}"

cd "$ROOT"
mkdir -p logs/newton experiments/outputs experiments/visuals

if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: JOB_ID must be set to a currently running tmux-held Slurm allocation." >&2
  exit 1
fi
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: required tmux-held session not found: $TMUX_SESSION" >&2
  exit 2
fi
if ! squeue -h -j "$JOB_ID" >/dev/null 2>&1; then
  echo "ERROR: Slurm job $JOB_ID is not visible." >&2
  exit 3
fi
if [[ ! -x "$NEWTON_VENV/bin/python" ]]; then
  echo "ERROR: missing local Newton venv; configure envs/ locally before compute use." >&2
  exit 4
fi
if [[ "$CONTROLLER_MODE" == "lift_hold_learned_residual" ]]; then
  if [[ ! -x "$TRAINER_VENV/bin/python" ]]; then
    echo "ERROR: missing local residual-adapter trainer venv; configure envs/residual_adapter locally before compute use." >&2
    exit 7
  fi
  if [[ -z "$RESIDUAL_ADAPTER_CHECKPOINT" || ! -f "$RESIDUAL_ADAPTER_CHECKPOINT" ]]; then
    echo "ERROR: CONTROLLER_MODE=lift_hold_learned_residual requires RESIDUAL_ADAPTER_CHECKPOINT." >&2
    exit 8
  fi
fi

bash -n "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh"
sed -n '1,120p' AGENTS.md >/dev/null

ps -u "$USER" -o pid,ppid,stat,etime,cmd \
  | awk -v self="$$" '
      $1 == self {next}
      /awk -v self=/ {next}
      /run_newton_panda_hydro_camera_export_in_alloc.sh/ {print; next}
      /run_newton_panda_hydro_camera_export_v2_in_alloc.sh/ {print; next}
      /srun .*newton_panda_hydro_camera_export/ {print; next}
    ' >/tmp/newton_panda_hydro_camera_export_running.$$
if [[ -s /tmp/newton_panda_hydro_camera_export_running.$$ ]]; then
  echo "REFUSE_START: Newton Panda hydro camera export is already running." >&2
  cat /tmp/newton_panda_hydro_camera_export_running.$$ >&2
  rm -f /tmp/newton_panda_hydro_camera_export_running.$$
  exit 6
fi
rm -f /tmp/newton_panda_hydro_camera_export_running.$$

log="$ROOT/logs/newton/${RUN_TAG}.log"
remote_cmd="cd $(printf '%q' "$ROOT") && RUN_TAG=$(printf '%q' "$RUN_TAG") NEWTON_VENV=$(printf '%q' "$NEWTON_VENV") TRAINER_VENV=$(printf '%q' "$TRAINER_VENV") SCENE=$(printf '%q' "$SCENE") TRACKED_OBJECT=$(printf '%q' "$TRACKED_OBJECT") CONTROLLER_MODE=$(printf '%q' "$CONTROLLER_MODE") FINAL_HOLD_DURATION=$(printf '%q' "$FINAL_HOLD_DURATION") LIFT_HEIGHT_MIN=$(printf '%q' "$LIFT_HEIGHT_MIN") HOLD_DURATION_MIN=$(printf '%q' "$HOLD_DURATION_MIN") DROP_HEIGHT_LOSS=$(printf '%q' "$DROP_HEIGHT_LOSS") PHYSICS_VARIANT_LABEL=$(printf '%q' "$PHYSICS_VARIANT_LABEL") BODY_MASS_SCALE=$(printf '%q' "$BODY_MASS_SCALE") SHAPE_FRICTION_SCALE=$(printf '%q' "$SHAPE_FRICTION_SCALE") OBJECT_MASS_KG=$(printf '%q' "$OBJECT_MASS_KG") OBJECT_FRICTION_MU=$(printf '%q' "$OBJECT_FRICTION_MU") FEEDBACK_MIN_CONTACT_COUNT=$(printf '%q' "$FEEDBACK_MIN_CONTACT_COUNT") FEEDBACK_ACCEL_THRESHOLD=$(printf '%q' "$FEEDBACK_ACCEL_THRESHOLD") FEEDBACK_HEIGHT_DROP_THRESHOLD=$(printf '%q' "$FEEDBACK_HEIGHT_DROP_THRESHOLD") FEEDBACK_INITIAL_LIFT_DURATION_SCALE=$(printf '%q' "$FEEDBACK_INITIAL_LIFT_DURATION_SCALE") FEEDBACK_LIFT_DURATION_SCALE_MAX=$(printf '%q' "$FEEDBACK_LIFT_DURATION_SCALE_MAX") FEEDBACK_HOLD_HEIGHT_STEP=$(printf '%q' "$FEEDBACK_HOLD_HEIGHT_STEP") FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=$(printf '%q' "$FEEDBACK_HOLD_HEIGHT_OFFSET_MAX") FEEDBACK_STABILIZATION_STEP=$(printf '%q' "$FEEDBACK_STABILIZATION_STEP") FEEDBACK_STABILIZATION_MAX=$(printf '%q' "$FEEDBACK_STABILIZATION_MAX") PRE_RECORD_WARMUP_STEPS=$(printf '%q' "$PRE_RECORD_WARMUP_STEPS") RESIDUAL_ADAPTER_CHECKPOINT=$(printf '%q' "$RESIDUAL_ADAPTER_CHECKPOINT") RESIDUAL_ADAPTER_ACTIVE_THRESHOLD=$(printf '%q' "$RESIDUAL_ADAPTER_ACTIVE_THRESHOLD") NUM_STEPS=$(printf '%q' "$NUM_STEPS") SAMPLE_STEPS=$(printf '%q' "$SAMPLE_STEPS") DEVICE=$(printf '%q' "$DEVICE") NEWTON_CACHE_PATH=$(printf '%q' "$NEWTON_CACHE_PATH") bash $(printf '%q' "$ROOT/experiments/configs/run_newton_panda_hydro_camera_export_v2_in_alloc.sh")"
cmd="cd $(printf '%q' "$ROOT") && srun --jobid=$(printf '%q' "$JOB_ID") --overlap --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 bash -lc $(printf '%q' "$remote_cmd")"

window="${WINDOW_NAME}_${RUN_TAG##*_}"
window_id="$(tmux new-window -P -F '#{window_id}' -t "$TMUX_SESSION" -n "$window")"
tmux send-keys -t "$window_id" \
  "$cmd 2>&1 | tee '$log'; printf '\nTMUX_NEWTON_PANDA_HYDRO_CAMERA_EXPORT_EXIT=%s\n' \"\${PIPESTATUS[0]}\"" C-m

cat <<EOF
RUN_TAG=$RUN_TAG
TMUX_SESSION=$TMUX_SESSION
WINDOW_NAME=$window
JOB_ID=$JOB_ID
LOG=$log
SCENE=$SCENE
TRACKED_OBJECT=$TRACKED_OBJECT
CONTROLLER_MODE=$CONTROLLER_MODE
FINAL_HOLD_DURATION=$FINAL_HOLD_DURATION
LIFT_HEIGHT_MIN=$LIFT_HEIGHT_MIN
HOLD_DURATION_MIN=$HOLD_DURATION_MIN
DROP_HEIGHT_LOSS=$DROP_HEIGHT_LOSS
PHYSICS_VARIANT_LABEL=$PHYSICS_VARIANT_LABEL
BODY_MASS_SCALE=$BODY_MASS_SCALE
SHAPE_FRICTION_SCALE=$SHAPE_FRICTION_SCALE
OBJECT_MASS_KG=$OBJECT_MASS_KG
OBJECT_FRICTION_MU=$OBJECT_FRICTION_MU
FEEDBACK_MIN_CONTACT_COUNT=$FEEDBACK_MIN_CONTACT_COUNT
FEEDBACK_ACCEL_THRESHOLD=$FEEDBACK_ACCEL_THRESHOLD
FEEDBACK_HEIGHT_DROP_THRESHOLD=$FEEDBACK_HEIGHT_DROP_THRESHOLD
FEEDBACK_INITIAL_LIFT_DURATION_SCALE=$FEEDBACK_INITIAL_LIFT_DURATION_SCALE
FEEDBACK_LIFT_DURATION_SCALE_MAX=$FEEDBACK_LIFT_DURATION_SCALE_MAX
FEEDBACK_HOLD_HEIGHT_STEP=$FEEDBACK_HOLD_HEIGHT_STEP
FEEDBACK_HOLD_HEIGHT_OFFSET_MAX=$FEEDBACK_HOLD_HEIGHT_OFFSET_MAX
FEEDBACK_STABILIZATION_STEP=$FEEDBACK_STABILIZATION_STEP
FEEDBACK_STABILIZATION_MAX=$FEEDBACK_STABILIZATION_MAX
PRE_RECORD_WARMUP_STEPS=$PRE_RECORD_WARMUP_STEPS
RESIDUAL_ADAPTER_CHECKPOINT=$RESIDUAL_ADAPTER_CHECKPOINT
RESIDUAL_ADAPTER_ACTIVE_THRESHOLD=$RESIDUAL_ADAPTER_ACTIVE_THRESHOLD
NUM_STEPS=$NUM_STEPS
SAMPLE_STEPS=$SAMPLE_STEPS
DEVICE=$DEVICE
DOWNSTREAM_USE=blocked_until_manual_visual_inspection_pass
EOF
