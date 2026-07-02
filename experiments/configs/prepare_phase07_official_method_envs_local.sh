#!/usr/bin/env bash
set -euo pipefail

# Prepare official-method environments on the shared filesystem before compute
# use. Default mode is dry-run. Set RUN_ENV_INSTALL=1 to execute commands.
# This script must not run inside a Slurm allocation.

ROOT="${ROOT:-/public/home/yanhongru/Curiosity}"
METHOD="${METHOD:-all}"
RUN_ENV_INSTALL="${RUN_ENV_INSTALL:-0}"
PYTHON_OPENPI="${PYTHON_OPENPI:-3.11}"
PYTHON_GR00T="${PYTHON_GR00T:-3.10}"
PYTHON_RTX="${PYTHON_RTX:-3.10}"
CONDA_BIN="${CONDA_BIN:-$(command -v mamba || command -v conda || true)}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: environment setup must not run inside a Slurm allocation." >&2
  exit 2
fi

cd "$ROOT"
mkdir -p envs logs/newton

run_cmd() {
  echo "+ $*"
  if [[ "$RUN_ENV_INSTALL" == "1" ]]; then
    "$@"
  fi
}

run_shell() {
  echo "+ $*"
  if [[ "$RUN_ENV_INSTALL" == "1" ]]; then
    bash -lc "$*"
  fi
}

need_uv() {
  if [[ -z "$UV_BIN" ]]; then
    if [[ "$RUN_ENV_INSTALL" == "1" ]]; then
      echo "ERROR: uv is required for $1 but was not found. Install uv locally before RUN_ENV_INSTALL=1." >&2
      exit 10
    fi
    UV_BIN="uv"
  fi
}

prepare_openpi() {
  need_uv "OpenPI"
  local repo="$ROOT/external/openpi"
  local env="$ROOT/envs/openpi/.venv"
  [[ -d "$repo" ]] || { echo "ERROR: missing official OpenPI repo at $repo" >&2; exit 11; }
  run_shell "cd '$repo' && UV_PROJECT_ENVIRONMENT='$env' '$UV_BIN' sync --python '$PYTHON_OPENPI'"
  run_cmd "$env/bin/python" -c "import openpi; print('openpi_env_ok')"
}

prepare_gr00t() {
  need_uv "GR00T"
  local repo="$ROOT/external/Isaac-GR00T"
  local env="$ROOT/envs/gr00t/.venv"
  [[ -d "$repo" ]] || { echo "ERROR: missing official GR00T repo at $repo" >&2; exit 12; }
  run_shell "cd '$repo' && UV_PROJECT_ENVIRONMENT='$env' '$UV_BIN' sync --python '$PYTHON_GR00T'"
  run_cmd "$env/bin/python" -c "import gr00t; print('gr00t_env_ok')"
}

prepare_diffusion_policy() {
  local repo="$ROOT/external/diffusion_policy"
  local env="$ROOT/envs/diffusion_policy/conda"
  [[ -d "$repo" ]] || { echo "ERROR: missing official Diffusion Policy repo at $repo" >&2; exit 13; }
  if [[ -z "$CONDA_BIN" ]]; then
    if [[ "$RUN_ENV_INSTALL" == "1" ]]; then
      echo "ERROR: mamba or conda is required for Diffusion Policy official environment." >&2
      exit 14
    fi
    CONDA_BIN="mamba_or_conda"
  fi
  run_shell "'$CONDA_BIN' env create -p '$env' -f '$repo/conda_environment.yaml'"
  run_shell "'$CONDA_BIN' run -p '$env' python -c \"import diffusion_policy; print('diffusion_policy_env_ok')\""
}

prepare_rtx() {
  local repo="$ROOT/external/open_x_embodiment"
  local env="$ROOT/envs/rtx/.venv"
  [[ -d "$repo" ]] || { echo "ERROR: missing official Open X-Embodiment repo at $repo" >&2; exit 15; }
  run_cmd "python$PYTHON_RTX" -m venv "$env"
  run_shell "source '$env/bin/activate' && python -m pip install --upgrade pip -i '$PIP_INDEX_URL'"
  run_shell "source '$env/bin/activate' && python -m pip install -i '$PIP_INDEX_URL' tensorflow tensorflow-datasets flax jax google-cloud-storage"
  run_cmd "$env/bin/python" -c "import tensorflow_datasets, flax, jax; print('rtx_env_ok')"
}

{
  printf 'RUN_TAG=%q\n' "phase07_official_method_env_prepare_local_20260627"
  printf 'ROOT=%q\n' "$ROOT"
  printf 'METHOD=%q\n' "$METHOD"
  printf 'RUN_ENV_INSTALL=%q\n' "$RUN_ENV_INSTALL"
  printf 'PIP_INDEX_URL=%q\n' "$PIP_INDEX_URL"
  printf 'NOTE=%q\n' "default_dry_run_local_shared_filesystem_only_no_compute_node_install"
} >"$ROOT/logs/newton/phase07_official_method_env_prepare_local_20260627_env.sh"

case "$METHOD" in
  all)
    prepare_openpi
    prepare_gr00t
    prepare_diffusion_policy
    prepare_rtx
    ;;
  openpi_pi0|openpi)
    prepare_openpi
    ;;
  gr00t)
    prepare_gr00t
    ;;
  diffusion_policy)
    prepare_diffusion_policy
    ;;
  rtx)
    prepare_rtx
    ;;
  *)
    echo "ERROR: unknown METHOD=$METHOD" >&2
    exit 20
    ;;
esac

if [[ "$RUN_ENV_INSTALL" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "Set RUN_ENV_INSTALL=1 to execute the printed commands on the login/shared-filesystem side, not inside Slurm."
fi
