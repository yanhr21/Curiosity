#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
RUN_TAG="${RUN_TAG:-p01_dense_clprobe_heldout_ablation_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/newton/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/phase01/dense/closed_loop_probe/real_attempts/p01_dense_clprobe_attempt002_pen_mu005_20260701_2310/dense_closed_loop_probe_checkpoint.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/outputs/phase01/dense/closed_loop_probe/heldout_ablation/$RUN_TAG}"
REPORT_DIR="${REPORT_DIR:-$ROOT/experiments/reports/phase01/dense/closed_loop_probe/heldout_ablation/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/newton/phase01/dense/closed_loop_probe/heldout_ablation/$RUN_TAG}"
NUM_FRAMES="${NUM_FRAMES:-240}"
MAP_SIZE="${MAP_SIZE:-16}"
REPETITIONS="${REPETITIONS:-3}"
SEED="${SEED:-777}"
SCORE_LIFT_WEIGHT="${SCORE_LIFT_WEIGHT:-4.0}"
SCORE_HOLD_WEIGHT="${SCORE_HOLD_WEIGHT:-0.03}"
SCORE_DROP_WEIGHT="${SCORE_DROP_WEIGHT:-6.0}"
HOLD_LIFT_THRESHOLD="${HOLD_LIFT_THRESHOLD:-0.08}"
FEATURE_NOISE_STD="${FEATURE_NOISE_STD:-0.15}"
CELLS="${CELLS:-train_like_pen_mu005:pen:0.05:1.0e12:false heldout_pen_mu004:pen:0.04:1.0e12:true heldout_pen_mu006:pen:0.06:1.0e12:true heldout_pen_mu003:pen:0.03:1.0e12:true}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: must run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

for path in "$PYTHON_BIN" "$CHECKPOINT" "$ROOT/src/newton_tactile_curiosity/phase01_dense_heldout_ablation_eval.py"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 3
  fi
done

run_log="$LOG_DIR/dense_heldout_ablation_eval.log"

echo "PHASE01_DENSE_HELDOUT_ABLATION_EVAL_START"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "NUM_FRAMES=$NUM_FRAMES"
echo "MAP_SIZE=$MAP_SIZE"
echo "REPETITIONS=$REPETITIONS"
echo "SEED=$SEED"
echo "CELLS=$CELLS"
echo "NOTE=evaluation_only_not_training_not_curiosity_success"

read -r -a cell_args <<<"$CELLS"

(
  cd "$ROOT"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export PYTHONPATH="$ROOT/src:$ROOT/external/newton_8c501:${PYTHONPATH:-}"
  "$PYTHON_BIN" "$ROOT/src/newton_tactile_curiosity/phase01_dense_heldout_ablation_eval.py" \
    --root "$ROOT" \
    --run-tag "$RUN_TAG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$REPORT_DIR" \
    --num-frames "$NUM_FRAMES" \
    --map-size "$MAP_SIZE" \
    --repetitions "$REPETITIONS" \
    --seed "$SEED" \
    --score-lift-weight "$SCORE_LIFT_WEIGHT" \
    --score-hold-weight "$SCORE_HOLD_WEIGHT" \
    --score-drop-weight "$SCORE_DROP_WEIGHT" \
    --hold-lift-threshold "$HOLD_LIFT_THRESHOLD" \
    --feature-noise-std "$FEATURE_NOISE_STD" \
    --cells "${cell_args[@]}"
) 2>&1 | tee "$run_log"

echo "PHASE01_DENSE_HELDOUT_ABLATION_EVAL_END"
