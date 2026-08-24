#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to prepare the SUGAR environment on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${ROOT_DIR}/IsaacLab}"
PYTHON311="${PYTHON311:-/public/home/yanhongru/.local/bin/python3.11}"
VENV_DIR="${VENV_DIR:-/public/home/yanhongru/envs/sugar_py311_isaacsim510}"
SUGAR_HTTP_PROXY="${SUGAR_HTTP_PROXY:-}"
SUGAR_INSTALL_EXTSCACHE="${SUGAR_INSTALL_EXTSCACHE:-1}"
SUGAR_PREINSTALL_TORCH="${SUGAR_PREINSTALL_TORCH:-0}"
SUGAR_SKIP_XFORMERS="${SUGAR_SKIP_XFORMERS:-1}"
SUGAR_SKIP_ISAACLAB_INSTALL="${SUGAR_SKIP_ISAACLAB_INSTALL:-0}"

if [[ ! -x "${PYTHON311}" ]]; then
  echo "Missing Python 3.11 executable: ${PYTHON311}" >&2
  exit 4
fi

if [[ ! -f "${SUGAR_DIR}/CURIOSITY_UPSTREAM_COMMIT" ]]; then
  echo "Missing vendored official SUGAR source: ${SUGAR_DIR}" >&2
  exit 5
fi

if [[ ! -f "${ISAACLAB_DIR}/VERSION" ]]; then
  echo "Missing vendored official IsaacLab source: ${ISAACLAB_DIR}" >&2
  exit 6
fi

isaaclab_version="v$(tr -d '[:space:]' < "${ISAACLAB_DIR}/VERSION")-curiosity-glue"
if [[ "${isaaclab_version}" != v2.3.0* ]]; then
  echo "IsaacLab must be v2.3.0 for official SUGAR; found ${isaaclab_version}" >&2
  exit 7
fi

echo "[SUGAR-ENV] host=$(hostname)"
echo "[SUGAR-ENV] root=${ROOT_DIR}"
echo "[SUGAR-ENV] sugar_dir=${SUGAR_DIR}"
echo "[SUGAR-ENV] isaaclab_dir=${ISAACLAB_DIR}"
echo "[SUGAR-ENV] venv_dir=${VENV_DIR}"
echo "[SUGAR-ENV] python311=${PYTHON311}"
echo "[SUGAR-ENV] install_extscache=${SUGAR_INSTALL_EXTSCACHE}"
echo "[SUGAR-ENV] preinstall_torch=${SUGAR_PREINSTALL_TORCH}"
echo "[SUGAR-ENV] skip_xformers=${SUGAR_SKIP_XFORMERS}"
echo "[SUGAR-ENV] skip_isaaclab_install=${SUGAR_SKIP_ISAACLAB_INSTALL}"

if [[ -n "${SUGAR_HTTP_PROXY}" ]]; then
  export http_proxy="${SUGAR_HTTP_PROXY}"
  export https_proxy="${SUGAR_HTTP_PROXY}"
  echo "[SUGAR-ENV] using SUGAR_HTTP_PROXY=${SUGAR_HTTP_PROXY}"
fi

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"
export PIP_RETRIES="${PIP_RETRIES:-30}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
echo "[SUGAR-ENV] pip_timeout=${PIP_DEFAULT_TIMEOUT}"
echo "[SUGAR-ENV] pip_retries=${PIP_RETRIES}"

if [[ ! -e "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON311}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
if [[ "${SUGAR_PREINSTALL_TORCH}" == "1" ]]; then
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
fi
if [[ "${SUGAR_INSTALL_EXTSCACHE}" == "1" ]]; then
  python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
else
  python -m pip install "isaacsim[all]==5.1.0" --extra-index-url https://pypi.nvidia.com
fi
python -m pip install "setuptools<81"
python -m pip install flatdict==4.0.1 --no-build-isolation
python -m pip install cmake
mkdir -p "${ROOT_DIR}/experiments/sugar_reproduction/logs"
constraint_file="${ROOT_DIR}/experiments/sugar_reproduction/logs/sugar_isaacsim_constraints.txt"
cat > "${constraint_file}" <<'EOF'
torch==2.7.0
torchvision==0.22.0
torchaudio==2.7.0
typing_extensions==4.12.2
click==8.1.7
transformers<5
huggingface-hub<1
pyarrow<21
EOF
python -m pip install -c "${constraint_file}" click==8.1.7 typing_extensions==4.12.2 "transformers<5" "huggingface-hub<1"

if [[ "${SUGAR_SKIP_ISAACLAB_INSTALL}" == "1" ]]; then
  echo "[SUGAR-ENV] skipping IsaacLab install because SUGAR_SKIP_ISAACLAB_INSTALL=1"
else
  cd "${ISAACLAB_DIR}"
  ./isaaclab.sh --install rsl_rl
fi

cd "${SUGAR_DIR}"

python -m pip install -c "${constraint_file}" -e source/sugar_rl
if [[ "${SUGAR_SKIP_XFORMERS}" == "1" ]]; then
  python -m pip install -c "${constraint_file}" \
    accelerate==1.2.1 \
    diffusers==0.32.1 \
    timm==1.0.12 \
    numpy==1.26.0 \
    scipy \
    einops \
    matplotlib \
    numba \
    numcodecs==0.12.1 \
    zarr==2.12.0 \
    tqdm \
    datasets==2.6.1 \
    hydra-core \
    json_repair \
    scikit-image \
    seaborn \
    pydantic==2.11.4
  python -m pip install --no-deps -e source/sugar_il
else
  python -m pip install -c "${constraint_file}" -e source/sugar_il
fi
python -m pip install -c "${constraint_file}" click==8.1.7 typing_extensions==4.12.2 "transformers<5" "huggingface-hub<1"

ROOT_DIR="${ROOT_DIR}" SUGAR_DIR="${SUGAR_DIR}" PYTHON_BIN="${VENV_DIR}/bin/python" \
  bash "${ROOT_DIR}/scripts/sugar/preflight_official_sugar_env.sh"
