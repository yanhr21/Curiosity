#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_dense_clprobe_attempt001_pen_mu002_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/newton/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase01/dense/closed_loop_probe/real_attempts/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase01/dense/closed_loop_probe/real_attempts/$RUN_TAG}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints/phase01/dense/closed_loop_probe/real_attempts/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase01/dense/closed_loop_probe/real_attempts/$RUN_TAG}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/experiments/outputs/phase01/dense/closed_loop_probe/real_attempts/${RUN_TAG}_eval}"
EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-$ROOT/experiments/reports/phase01/dense/closed_loop_probe/real_attempts/${RUN_TAG}_eval}"

SCENE="${SCENE:-pen}"
OVERRIDE_MU="${OVERRIDE_MU:-0.02}"
OVERRIDE_KH="${OVERRIDE_KH:-1.0e12}"
TRAIN_MU_VALUES="${TRAIN_MU_VALUES:-}"
NUM_FRAMES="${NUM_FRAMES:-240}"
MAP_SIZE="${MAP_SIZE:-16}"
POPULATION_SIZE="${POPULATION_SIZE:-2}"
ELITE_COUNT="${ELITE_COUNT:-1}"
GENERATIONS="${GENERATIONS:-1}"
SEED="${SEED:-901}"
TARGET_DURATION_S="${TARGET_DURATION_S:-3600}"
MIN_DURATION_S="${MIN_DURATION_S:-3600}"
INTRINSIC_WEIGHT="${INTRINSIC_WEIGHT:-1.0}"
SAFETY_WEIGHT="${SAFETY_WEIGHT:-1.0}"
SCORE_LIFT_WEIGHT="${SCORE_LIFT_WEIGHT:-4.0}"
SCORE_FINAL_LIFT_WEIGHT="${SCORE_FINAL_LIFT_WEIGHT:-0.0}"
SCORE_HOLD_WEIGHT="${SCORE_HOLD_WEIGHT:-0.01}"
SCORE_TAIL_HOLD_WEIGHT="${SCORE_TAIL_HOLD_WEIGHT:-0.0}"
SCORE_DROP_WEIGHT="${SCORE_DROP_WEIGHT:-2.0}"
HOLD_LIFT_THRESHOLD="${HOLD_LIFT_THRESHOLD:-0.08}"
STABLE_TAIL_FRAMES="${STABLE_TAIL_FRAMES:-60}"
SIGMA_MIN_FRAC="${SIGMA_MIN_FRAC:-0.15}"
SIGMA_DECAY="${SIGMA_DECAY:-0.95}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$EVAL_OUTPUT_DIR" "$EVAL_REPORT_DIR"

for path in \
  "$PYTHON_BIN" \
  "$ROOT/src/newton_tactile_curiosity/phase01_dense_closed_loop_probe.py" \
  "$ROOT/src/newton_tactile_curiosity/phase01_dense_closed_loop_eval.py"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

train_log="$LOG_DIR/real_training.log"
eval_log="$LOG_DIR/validation_eval.log"
gpu_log="$LOG_DIR/gpu_utilization.csv"

echo "PHASE01_DENSE_CLOSED_LOOP_REAL_ATTEMPT_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "EVAL_OUTPUT_DIR=$EVAL_OUTPUT_DIR"
echo "EVAL_REPORT_DIR=$EVAL_REPORT_DIR"
echo "SCENE=$SCENE"
echo "OVERRIDE_MU=$OVERRIDE_MU"
echo "OVERRIDE_KH=$OVERRIDE_KH"
echo "TRAIN_MU_VALUES=$TRAIN_MU_VALUES"
echo "NUM_FRAMES=$NUM_FRAMES"
echo "TARGET_DURATION_S=$TARGET_DURATION_S"
echo "MIN_DURATION_S=$MIN_DURATION_S"
echo "INTRINSIC_WEIGHT=$INTRINSIC_WEIGHT"
echo "SAFETY_WEIGHT=$SAFETY_WEIGHT"
echo "POPULATION_SIZE=$POPULATION_SIZE"
echo "ELITE_COUNT=$ELITE_COUNT"
echo "SCORE_LIFT_WEIGHT=$SCORE_LIFT_WEIGHT"
echo "SCORE_FINAL_LIFT_WEIGHT=$SCORE_FINAL_LIFT_WEIGHT"
echo "SCORE_HOLD_WEIGHT=$SCORE_HOLD_WEIGHT"
echo "SCORE_TAIL_HOLD_WEIGHT=$SCORE_TAIL_HOLD_WEIGHT"
echo "SCORE_DROP_WEIGHT=$SCORE_DROP_WEIGHT"
echo "HOLD_LIFT_THRESHOLD=$HOLD_LIFT_THRESHOLD"
echo "STABLE_TAIL_FRAMES=$STABLE_TAIL_FRAMES"
echo "SIGMA_MIN_FRAC=$SIGMA_MIN_FRAC"
echo "SIGMA_DECAY=$SIGMA_DECAY"
echo "NOTE=counted_real_attempt_candidate_not_success_claim"

train_mu_args=()
if [[ -n "$TRAIN_MU_VALUES" ]]; then
  read -r -a train_mu_values <<<"$TRAIN_MU_VALUES"
  train_mu_args=(--train-mu-values "${train_mu_values[@]}")
fi

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used --format=csv -l 5 >"$gpu_log" &
gpu_monitor_pid="$!"
cleanup() {
  kill "$gpu_monitor_pid" 2>/dev/null || true
  wait "$gpu_monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

(
  cd "$ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$ROOT/src:$ROOT/external/newton_8c501:${PYTHONPATH:-}"
  "$PYTHON_BIN" "$ROOT/src/newton_tactile_curiosity/phase01_dense_closed_loop_probe.py" \
    --root "$ROOT" \
    --run-tag "$RUN_TAG" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$REPORT_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --scene "$SCENE" \
    --override-mu "$OVERRIDE_MU" \
    --override-kh "$OVERRIDE_KH" \
    "${train_mu_args[@]}" \
    --num-frames "$NUM_FRAMES" \
    --map-size "$MAP_SIZE" \
    --generations "$GENERATIONS" \
    --population-size "$POPULATION_SIZE" \
    --elite-count "$ELITE_COUNT" \
    --seed "$SEED" \
    --intrinsic-weight "$INTRINSIC_WEIGHT" \
    --safety-weight "$SAFETY_WEIGHT" \
    --score-lift-weight "$SCORE_LIFT_WEIGHT" \
    --score-final-lift-weight "$SCORE_FINAL_LIFT_WEIGHT" \
    --score-hold-weight "$SCORE_HOLD_WEIGHT" \
    --score-tail-hold-weight "$SCORE_TAIL_HOLD_WEIGHT" \
    --score-drop-weight "$SCORE_DROP_WEIGHT" \
    --hold-lift-threshold "$HOLD_LIFT_THRESHOLD" \
    --stable-tail-frames "$STABLE_TAIL_FRAMES" \
    --sigma-min-frac "$SIGMA_MIN_FRAC" \
    --sigma-decay "$SIGMA_DECAY" \
    --target-duration-s "$TARGET_DURATION_S" \
    --min-duration-s "$MIN_DURATION_S"
) 2>&1 | tee "$train_log"

checkpoint="$CHECKPOINT_DIR/dense_closed_loop_probe_checkpoint.npz"
if [[ ! -s "$checkpoint" ]]; then
  echo "ERROR: missing checkpoint after training: $checkpoint" >&2
  exit 4
fi

(
  cd "$ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$ROOT/src:$ROOT/external/newton_8c501:${PYTHONPATH:-}"
  "$PYTHON_BIN" "$ROOT/src/newton_tactile_curiosity/phase01_dense_closed_loop_eval.py" \
    --root "$ROOT" \
    --run-tag "${RUN_TAG}_validation" \
    --checkpoint "$checkpoint" \
    --output-dir "$EVAL_OUTPUT_DIR" \
    --report-dir "$EVAL_REPORT_DIR" \
    --scene "$SCENE" \
    --override-mu "$OVERRIDE_MU" \
    --override-kh "$OVERRIDE_KH" \
    --num-frames "$NUM_FRAMES" \
    --map-size "$MAP_SIZE" \
    --score-lift-weight "$SCORE_LIFT_WEIGHT" \
    --score-final-lift-weight "$SCORE_FINAL_LIFT_WEIGHT" \
    --score-hold-weight "$SCORE_HOLD_WEIGHT" \
    --score-tail-hold-weight "$SCORE_TAIL_HOLD_WEIGHT" \
    --score-drop-weight "$SCORE_DROP_WEIGHT" \
    --hold-lift-threshold "$HOLD_LIFT_THRESHOLD" \
    --stable-tail-frames "$STABLE_TAIL_FRAMES" \
    --repetitions 1
) 2>&1 | tee "$eval_log"

echo "GPU_UTILIZATION_LOG=$gpu_log"
echo "PHASE01_DENSE_CLOSED_LOOP_REAL_ATTEMPT_END"
