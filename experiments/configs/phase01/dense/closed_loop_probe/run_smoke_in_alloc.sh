#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_dense_clprobe_smoke_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/newton/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase01/dense/closed_loop_probe/smoke/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase01/dense/closed_loop_probe/smoke/$RUN_TAG}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints/phase01/dense/closed_loop_probe/smoke/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase01/dense/closed_loop_probe/smoke/$RUN_TAG}"
SCENE="${SCENE:-cube}"
OVERRIDE_MU="${OVERRIDE_MU:-0.3}"
OVERRIDE_KH="${OVERRIDE_KH:-1.0e12}"
NUM_FRAMES="${NUM_FRAMES:-90}"
MAP_SIZE="${MAP_SIZE:-16}"
GENERATIONS="${GENERATIONS:-1}"
POPULATION_SIZE="${POPULATION_SIZE:-2}"
ELITE_COUNT="${ELITE_COUNT:-1}"
SEED="${SEED:-123}"
SCORE_LIFT_WEIGHT="${SCORE_LIFT_WEIGHT:-4.0}"
SCORE_HOLD_WEIGHT="${SCORE_HOLD_WEIGHT:-0.01}"
SCORE_DROP_WEIGHT="${SCORE_DROP_WEIGHT:-2.0}"
HOLD_LIFT_THRESHOLD="${HOLD_LIFT_THRESHOLD:-0.08}"
SIGMA_MIN_FRAC="${SIGMA_MIN_FRAC:-0.15}"
SIGMA_DECAY="${SIGMA_DECAY:-0.95}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

for path in "$PYTHON_BIN" "$ROOT/src/newton_tactile_curiosity/phase01_dense_closed_loop_probe.py"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

run_log="$LOG_DIR/dense_closed_loop_probe_smoke.log"

echo "PHASE01_DENSE_CLOSED_LOOP_PROBE_SMOKE_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "SCENE=$SCENE"
echo "OVERRIDE_MU=$OVERRIDE_MU"
echo "OVERRIDE_KH=$OVERRIDE_KH"
echo "NUM_FRAMES=$NUM_FRAMES"
echo "MAP_SIZE=$MAP_SIZE"
echo "GENERATIONS=$GENERATIONS"
echo "POPULATION_SIZE=$POPULATION_SIZE"
echo "ELITE_COUNT=$ELITE_COUNT"
echo "SEED=$SEED"
echo "SCORE_LIFT_WEIGHT=$SCORE_LIFT_WEIGHT"
echo "SCORE_HOLD_WEIGHT=$SCORE_HOLD_WEIGHT"
echo "SCORE_DROP_WEIGHT=$SCORE_DROP_WEIGHT"
echo "HOLD_LIFT_THRESHOLD=$HOLD_LIFT_THRESHOLD"
echo "SIGMA_MIN_FRAC=$SIGMA_MIN_FRAC"
echo "SIGMA_DECAY=$SIGMA_DECAY"
echo "NOTE=smoke_not_real_training_attempt_not_curiosity_success"

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
    --num-frames "$NUM_FRAMES" \
    --map-size "$MAP_SIZE" \
    --generations "$GENERATIONS" \
    --population-size "$POPULATION_SIZE" \
    --elite-count "$ELITE_COUNT" \
    --seed "$SEED" \
    --score-lift-weight "$SCORE_LIFT_WEIGHT" \
    --score-hold-weight "$SCORE_HOLD_WEIGHT" \
    --score-drop-weight "$SCORE_DROP_WEIGHT" \
    --hold-lift-threshold "$HOLD_LIFT_THRESHOLD" \
    --sigma-min-frac "$SIGMA_MIN_FRAC" \
    --sigma-decay "$SIGMA_DECAY" \
    --smoke
) 2>&1 | tee "$run_log"

echo "PHASE01_DENSE_CLOSED_LOOP_PROBE_SMOKE_END"
