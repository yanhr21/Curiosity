#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}
DATASET=${DATASET:-experiments/demo_following/official_transition_risk_v1/dataset}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/demo_following/official_transition_risk_v1}
TRAINER=scripts/sugar/demo_following/train_official_transition_risk_transformer.py

export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" -u "$TRAINER" \
    --dataset-root "$DATASET" --output-dir "$OUTPUT_ROOT/overfit_seed171625" \
    --mode overfit --overfit-steps 500 --seed 171625 --device cuda:0

"$PYTHON_BIN" -u "$TRAINER" \
    --dataset-root "$DATASET" --output-dir "$OUTPUT_ROOT/formal_seed171626" \
    --mode formal --epochs 20 --seed 171626 --device cuda:0

echo "OFFICIAL_TRANSITION_RISK_TRAINING_COMPLETE output_root=$OUTPUT_ROOT"
