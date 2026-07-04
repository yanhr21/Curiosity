#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to run Isaac simulation on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
ARENA_DIR="${ARENA_DIR:-${ROOT_DIR}/external/IsaacLab-Arena}"
GROOT_DIR="${GROOT_DIR:-${ARENA_DIR}/submodules/Isaac-GR00T}"
ISAAC_VENV="${ISAAC_VENV:-/public/home/yanhongru/envs/isaac_arena_py312}"
GROOT_VENV="${GROOT_VENV:-/public/home/yanhongru/envs/gr00t_n16_py310}"
MODELS_DIR="${MODELS_DIR:-/public/home/yanhongru/models/isaaclab_arena/locomanipulation_tutorial}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${MODELS_DIR}/checkpoint-20000}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${ROOT_DIR}/experiments/outputs/isaac_arena_g1_locomanip}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/isaac_arena_g1_locomanip}"
REMOTE_HOST="${REMOTE_HOST:-127.0.0.1}"
REMOTE_PORT="${REMOTE_PORT:-5555}"
NUM_STEPS="${NUM_STEPS:-1500}"
OV_REGISTRY_MIRROR="${OV_REGISTRY_MIRROR:-/public/home/yanhongru/ov_registry_mirror}"
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.registry.nucleus/registries/0/url=${OV_REGISTRY_MIRROR}/kit_prod_default --/exts/omni.kit.registry.nucleus/registries/1/url=${OV_REGISTRY_MIRROR}/kit_prod_sdk}"
ARENA_GALILEO_LOCOMANIP_USD="${ARENA_GALILEO_LOCOMANIP_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/galileo_locomanip.usd}"
ARENA_BROWN_BOX_USD="${ARENA_BROWN_BOX_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/brown_box.usd}"
ARENA_BLUE_SORTING_BIN_USD="${ARENA_BLUE_SORTING_BIN_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/blue_sorting_bin.usd}"
ARENA_G1_29DOF_WITH_HAND_USD="${ARENA_G1_29DOF_WITH_HAND_USD:-/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd}"
export ARENA_GALILEO_LOCOMANIP_USD ARENA_BROWN_BOX_USD ARENA_BLUE_SORTING_BIN_USD ARENA_G1_29DOF_WITH_HAND_USD

mkdir -p "${OUTPUT_BASE_DIR}" "${LOG_DIR}"

if [[ ! -d "${ARENA_DIR}/isaaclab_arena" ]]; then
  echo "Arena source not found: ${ARENA_DIR}" >&2
  exit 3
fi

if [[ ! -f "${GROOT_DIR}/gr00t/eval/run_gr00t_server.py" ]]; then
  echo "Pinned Isaac-GR00T server script not found under: ${GROOT_DIR}" >&2
  exit 4
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Official checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
  echo "Run scripts/isaac/download_arena_g1_official_assets.sh first." >&2
  exit 5
fi

if [[ ! -x "${ISAAC_VENV}/bin/python" ]]; then
  echo "Isaac/Arena Python not found: ${ISAAC_VENV}/bin/python" >&2
  exit 6
fi

if [[ ! -x "${GROOT_VENV}/bin/python" ]]; then
  echo "GR00T Python not found: ${GROOT_VENV}/bin/python" >&2
  exit 7
fi

if [[ ! -f "${CHECKPOINT_DIR}/model-00001-of-00002.safetensors" ]] ||
   [[ ! -f "${CHECKPOINT_DIR}/model-00002-of-00002.safetensors" ]] ||
   [[ ! -f "${CHECKPOINT_DIR}/model.safetensors.index.json" ]]; then
  echo "Official inference checkpoint files are incomplete under: ${CHECKPOINT_DIR}" >&2
  exit 8
fi

if [[ ! -f "${OV_REGISTRY_MIRROR}/kit_prod_default/v2/registry.gz" ]] ||
   [[ ! -f "${OV_REGISTRY_MIRROR}/kit_prod_sdk/v2/registry.gz" ]]; then
  echo "Local Omniverse registry mirror is incomplete under: ${OV_REGISTRY_MIRROR}" >&2
  exit 9
fi

if [[ ! -f "${ARENA_GALILEO_LOCOMANIP_USD}" ]]; then
  echo "Local Galileo loco-manipulation USD not found: ${ARENA_GALILEO_LOCOMANIP_USD}" >&2
  exit 12
fi

if [[ ! -f "${ARENA_BROWN_BOX_USD}" ]]; then
  echo "Local brown box USD not found: ${ARENA_BROWN_BOX_USD}" >&2
  exit 13
fi

if [[ ! -f "${ARENA_BLUE_SORTING_BIN_USD}" ]]; then
  echo "Local blue sorting bin USD not found: ${ARENA_BLUE_SORTING_BIN_USD}" >&2
  exit 14
fi

if [[ ! -f "${ARENA_G1_29DOF_WITH_HAND_USD}" ]]; then
  echo "Local G1 USD not found: ${ARENA_G1_29DOF_WITH_HAND_USD}" >&2
  exit 15
fi

SERVER_LOG="${LOG_DIR}/gr00t_server_$(date +%Y%m%d_%H%M%S).log"
EVAL_LOG="${LOG_DIR}/arena_eval_$(date +%Y%m%d_%H%M%S).log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[INFO] Starting official GR00T policy server."
(
  cd "${GROOT_DIR}"
  "${GROOT_VENV}/bin/python" gr00t/eval/run_gr00t_server.py \
    --modality-config-path "${ARENA_DIR}/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py" \
    --model-path "${CHECKPOINT_DIR}" \
    --embodiment-tag NEW_EMBODIMENT \
    --device cuda \
    --host "${REMOTE_HOST}" \
    --port "${REMOTE_PORT}"
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[INFO] Waiting for GR00T server at ${REMOTE_HOST}:${REMOTE_PORT}."
for _ in $(seq 1 120); do
  if (echo >"/dev/tcp/${REMOTE_HOST}/${REMOTE_PORT}") >/dev/null 2>&1; then
    echo "[INFO] GR00T server is reachable."
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "GR00T server exited early. See ${SERVER_LOG}" >&2
    exit 10
  fi
  sleep 2
done

if ! (echo >"/dev/tcp/${REMOTE_HOST}/${REMOTE_PORT}") >/dev/null 2>&1; then
  echo "Timed out waiting for GR00T server. See ${SERVER_LOG}" >&2
  exit 11
fi

echo "[INFO] Running official Arena G1 loco-manipulation evaluation."
echo "[INFO] Kit args: ${KIT_ARGS}"
echo "[INFO] Galileo USD: ${ARENA_GALILEO_LOCOMANIP_USD}"
echo "[INFO] Brown box USD: ${ARENA_BROWN_BOX_USD}"
echo "[INFO] Blue sorting bin USD: ${ARENA_BLUE_SORTING_BIN_USD}"
echo "[INFO] G1 USD: ${ARENA_G1_29DOF_WITH_HAND_USD}"
cd "${ARENA_DIR}"
PYTHONPATH="${GROOT_DIR}:${ARENA_DIR}:${PYTHONPATH:-}" "${ISAAC_VENV}/bin/python" isaaclab_arena/evaluation/policy_runner.py \
  --headless \
  --enable_cameras \
  --kit_args "${KIT_ARGS}" \
  --record_camera_video \
  --record_viewport_video \
  --output_base_dir "${OUTPUT_BASE_DIR}" \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml \
  --remote_host "${REMOTE_HOST}" \
  --remote_port "${REMOTE_PORT}" \
  --num_steps "${NUM_STEPS}" \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_joint \
  2>&1 | tee "${EVAL_LOG}"

echo "[INFO] Arena evaluation complete."
echo "[INFO] Output base: ${OUTPUT_BASE_DIR}"
echo "[INFO] Server log: ${SERVER_LOG}"
echo "[INFO] Eval log: ${EVAL_LOG}"
