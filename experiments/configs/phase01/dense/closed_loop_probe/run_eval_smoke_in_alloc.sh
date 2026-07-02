#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_dense_clprobe_eval_smoke_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/newton/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/phase01/dense/closed_loop_probe/smoke/p01_dense_clprobe_smoke240_20260701_2101/dense_closed_loop_probe_checkpoint.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase01/dense/closed_loop_probe/eval_smoke/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase01/dense/closed_loop_probe/eval_smoke/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase01/dense/closed_loop_probe/eval_smoke/$RUN_TAG}"
SCENE="${SCENE:-cube}"
NUM_FRAMES="${NUM_FRAMES:-240}"
MAP_SIZE="${MAP_SIZE:-16}"
REPETITIONS="${REPETITIONS:-1}"
SEED="${SEED:-333}"
OVERRIDE_MU="${OVERRIDE_MU:-0.3}"
OVERRIDE_KH="${OVERRIDE_KH:-1.0e12}"
SCORE_LIFT_WEIGHT="${SCORE_LIFT_WEIGHT:-4.0}"
SCORE_HOLD_WEIGHT="${SCORE_HOLD_WEIGHT:-0.01}"
SCORE_DROP_WEIGHT="${SCORE_DROP_WEIGHT:-2.0}"
HOLD_LIFT_THRESHOLD="${HOLD_LIFT_THRESHOLD:-0.08}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$PYTHON_BIN" "$CHECKPOINT" "$ROOT/src/newton_tactile_curiosity/phase01_dense_closed_loop_eval.py"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

run_log="$LOG_DIR/dense_closed_loop_eval_smoke.log"

echo "PHASE01_DENSE_CLOSED_LOOP_EVAL_SMOKE_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "SCENE=$SCENE"
echo "NUM_FRAMES=$NUM_FRAMES"
echo "MAP_SIZE=$MAP_SIZE"
echo "REPETITIONS=$REPETITIONS"
echo "SEED=$SEED"
echo "OVERRIDE_MU=$OVERRIDE_MU"
echo "OVERRIDE_KH=$OVERRIDE_KH"
echo "SCORE_LIFT_WEIGHT=$SCORE_LIFT_WEIGHT"
echo "SCORE_HOLD_WEIGHT=$SCORE_HOLD_WEIGHT"
echo "SCORE_DROP_WEIGHT=$SCORE_DROP_WEIGHT"
echo "HOLD_LIFT_THRESHOLD=$HOLD_LIFT_THRESHOLD"
echo "NOTE=eval_smoke_not_training_not_curiosity_success"

(
  cd "$ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$ROOT/src:$ROOT/external/newton_8c501:${PYTHONPATH:-}"
  "$PYTHON_BIN" "$ROOT/src/newton_tactile_curiosity/phase01_dense_closed_loop_eval.py" \
    --root "$ROOT" \
    --run-tag "$RUN_TAG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$REPORT_DIR" \
    --scene "$SCENE" \
    --num-frames "$NUM_FRAMES" \
    --map-size "$MAP_SIZE" \
    --repetitions "$REPETITIONS" \
    --seed "$SEED" \
    --override-mu "$OVERRIDE_MU" \
    --override-kh "$OVERRIDE_KH" \
    --score-lift-weight "$SCORE_LIFT_WEIGHT" \
    --score-hold-weight "$SCORE_HOLD_WEIGHT" \
    --score-drop-weight "$SCORE_DROP_WEIGHT" \
    --hold-lift-threshold "$HOLD_LIFT_THRESHOLD" \
    --smoke
) 2>&1 | tee "$run_log"

echo "PHASE01_DENSE_CLOSED_LOOP_EVAL_SMOKE_END"
