#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/public/home/yanhongru/Curiosity"
BPP_ENV="/public/home/yanhongru/envs/bpp_official_py310"
BPP_ROOT="${PROJECT_ROOT}/experiments/demo_following/bpp_official_v1"
BPP_SOURCE="${BPP_ROOT}/artifacts/behavior_prompting_b5b494e"
BPP_CHECKPOINT="${BPP_ROOT}/artifacts/liberogen_goal_chain_ab2ce394/liberogen_goal_chain_behavior_prompting.ckpt"
INTERFACE_CONTRACT="${PROJECT_ROOT}/scripts/sugar/demo_following/config/bpp_official_interface_adapter_v1.json"
OUT_DIR="${1:-${BPP_ROOT}/official_checkpoint_audit_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: checkpoint audit must run inside a Slurm allocation" >&2
  exit 2
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H200"; then
  echo "ERROR: checkpoint audit requires an H200 compute node" >&2
  exit 3
fi
if [[ ! -x "${BPP_ENV}/bin/python" ]]; then
  echo "ERROR: official BPP environment is unavailable: ${BPP_ENV}" >&2
  exit 4
fi
if [[ ! -s "${BPP_CHECKPOINT}" ]]; then
  echo "ERROR: official BPP checkpoint is unavailable: ${BPP_CHECKPOINT}" >&2
  exit 5
fi

mkdir -p "${OUT_DIR}"
exec > >(tee -a "${OUT_DIR}/audit.log") 2>&1
cd "${PROJECT_ROOT}"

export PYTHONPATH="${BPP_SOURCE}:${BPP_SOURCE}/deps/LIBERO:${BPP_SOURCE}/deps/icrt:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPYCACHEPREFIX="${OUT_DIR}/pycache"

echo "STARTED_AT=$(date --iso-8601=seconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

"${BPP_ENV}/bin/python" \
  scripts/sugar/demo_following/audit_bpp_official_checkpoint.py \
  --source-repo "${BPP_SOURCE}" \
  --checkpoint "${BPP_CHECKPOINT}" \
  --interface-contract "${INTERFACE_CONTRACT}" \
  --output-dir "${OUT_DIR}"

echo "FINISHED_AT=$(date --iso-8601=seconds)"
