#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing Tracker checkpoint eval suite on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
OUTPUT_ABS="${OUTPUT_ABS:-${ROOT_DIR}/experiments/sugar_reproduction/outputs/CarryBox_20260712_official_carrybox_full}"
TRACKER_ITERATION="${TRACKER_ITERATION:-1000}"
SUGAR_DISABLE_RENDERER_MULTIGPU="${SUGAR_DISABLE_RENDERER_MULTIGPU:-0}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${OUTPUT_ABS}/logs/tracker/model_${TRACKER_ITERATION}.pt}"
NAMED_CHECKPOINT="${NAMED_CHECKPOINT:-${OUTPUT_ABS}/ckpts/tracker_model${TRACKER_ITERATION}.pt}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${OUTPUT_ABS}/eval/tracker_model${TRACKER_ITERATION}_rollout_eval_novideo/raw_npz}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_ABS}/visualizations}"

if [[ ! -s "${SOURCE_CHECKPOINT}" ]]; then
  echo "Missing Tracker checkpoint: ${SOURCE_CHECKPOINT}" >&2
  exit 3
fi

cp -p "${SOURCE_CHECKPOINT}" "${NAMED_CHECKPOINT}"
chmod a-w "${NAMED_CHECKPOINT}"
source_sha="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
named_sha="$(sha256sum "${NAMED_CHECKPOINT}" | awk '{print $1}')"
if [[ "${source_sha}" != "${named_sha}" ]]; then
  echo "Tracker checkpoint copy mismatch: source=${source_sha} named=${named_sha}" >&2
  exit 4
fi
echo "[SUGAR-TRACKER-EVAL-SUITE] checkpoint=${NAMED_CHECKPOINT}"
echo "[SUGAR-TRACKER-EVAL-SUITE] sha256=${named_sha}"

CHECKPOINT="${NAMED_CHECKPOINT}" \
EVAL_NAME="tracker_model${TRACKER_ITERATION}_rollout_eval_novideo" \
ROLLOUT_DIR="${ROLLOUT_ROOT}" \
NUM_ENVS=16 ENABLE_VIDEO=0 \
SUGAR_DISABLE_RENDERER_MULTIGPU="${SUGAR_DISABLE_RENDERER_MULTIGPU}" \
STAMP="20260713_sugar_tracker_model${TRACKER_ITERATION}_rollout_eval" \
bash "${ROOT_DIR}/scripts/sugar/run_official_sugar_carrybox_tracker_eval.sh"

CHECKPOINT="${NAMED_CHECKPOINT}" \
EVAL_NAME="tracker_model${TRACKER_ITERATION}_video_eval" \
ROLLOUT_DIR="${OUTPUT_ABS}/eval/tracker_model${TRACKER_ITERATION}_video_eval/raw_npz" \
NUM_ENVS=4 ENABLE_VIDEO=1 VIDEO_LENGTH=200 SKIP_PREFLIGHT=1 \
SUGAR_DISABLE_RENDERER_MULTIGPU="${SUGAR_DISABLE_RENDERER_MULTIGPU}" \
STAMP="20260713_sugar_tracker_model${TRACKER_ITERATION}_video_eval" \
bash "${ROOT_DIR}/scripts/sugar/run_official_sugar_carrybox_tracker_eval.sh"

video_source="$(dirname "${NAMED_CHECKPOINT}")/videos/play/rl-video-step-0.mp4"
if [[ ! -s "${video_source}" ]]; then
  echo "Missing Tracker eval video: ${video_source}" >&2
  exit 5
fi
mkdir -p "${PLOT_DIR}"
cp -p "${video_source}" "${PLOT_DIR}/tracker_model${TRACKER_ITERATION}_rollout_video.mp4"

bash "${ROOT_DIR}/scripts/sugar/render_official_sugar_training_curves.sh"

ROLLOUT_DIR="${ROLLOUT_ROOT}" \
CHECKPOINT_LABEL="model_${TRACKER_ITERATION}" \
POLICY_STAGE=tracker EXPECTED_WINDOWS=16 \
OUTPUT_BASENAME="tracker_model${TRACKER_ITERATION}_rollout_summary" \
bash "${ROOT_DIR}/scripts/sugar/render_official_sugar_refiner5000_rollout_summary.sh"

echo "[SUGAR-TRACKER-EVAL-SUITE] completed tracker model_${TRACKER_ITERATION} eval suite"
