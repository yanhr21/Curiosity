#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
MODELS_DIR="${MODELS_DIR:-/public/home/yanhongru/models/isaaclab_arena/locomanipulation_tutorial}"
DATASET_DIR="${DATASET_DIR:-/public/home/yanhongru/datasets/isaaclab_arena/locomanipulation_tutorial}"
DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-0}"
DOWNLOAD_FULL_TRAINING_STATE="${DOWNLOAD_FULL_TRAINING_STATE:-0}"

mkdir -p "${MODELS_DIR}/checkpoint-20000"

echo "[INFO] Downloading official Arena G1 loco-manipulation GR00T checkpoint."
echo "[INFO] Target: ${MODELS_DIR}/checkpoint-20000"
if [[ "${DOWNLOAD_FULL_TRAINING_STATE}" == "1" ]]; then
  hf download \
    --revision gn1_6 \
    nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
    --local-dir "${MODELS_DIR}/checkpoint-20000"
else
  hf download \
    --revision gn1_6 \
    nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation \
    --local-dir "${MODELS_DIR}/checkpoint-20000" \
    --include "config.json" \
    --include "embodiment_id.json" \
    --include "latest" \
    --include "model-*.safetensors" \
    --include "model.safetensors.index.json" \
    --include "processor_config.json" \
    --include "statistics.json" \
    --include "README.md" \
    --include ".gitattributes" \
    --include "experiment_cfg/*"
fi

if [[ "${DOWNLOAD_DATASET}" == "1" ]]; then
  mkdir -p "${DATASET_DIR}"
  echo "[INFO] Downloading official generated simulation HDF5 dataset."
  echo "[INFO] Target: ${DATASET_DIR}"
  hf download \
    nvidia/Arena-G1-Loco-Manipulation-Task \
    arena_g1_loco_manipulation_dataset_generated.hdf5 \
    --repo-type dataset \
    --revision arena_v0.2_lab_v3.0 \
    --local-dir "${DATASET_DIR}"
else
  echo "[INFO] Skipping 23GB generated dataset. Set DOWNLOAD_DATASET=1 to fetch it."
fi

echo "[INFO] Asset download finished."
