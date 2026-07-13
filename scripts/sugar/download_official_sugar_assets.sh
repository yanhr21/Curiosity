#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname)" == mgmtserver* ]]; then
  echo "Refusing to download/unzip SUGAR assets on login/management node: $(hostname)" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-/public/home/yanhongru/Curiosity}"
SUGAR_DIR="${SUGAR_DIR:-${ROOT_DIR}/SUGAR}"
GDOWN_BIN="${GDOWN_BIN:-/public/home/yanhongru/.local/bin/gdown}"
SUGAR_HTTP_PROXY="${SUGAR_HTTP_PROXY:-}"

if [[ ! -f "${SUGAR_DIR}/CURIOSITY_UPSTREAM_COMMIT" ]]; then
  echo "Missing official SUGAR clone at ${SUGAR_DIR}" >&2
  exit 3
fi

if [[ ! -x "${GDOWN_BIN}" ]]; then
  if command -v gdown >/dev/null 2>&1; then
    GDOWN_BIN="$(command -v gdown)"
  else
    echo "gdown is not available. Install/use it outside this script, then rerun." >&2
    exit 4
  fi
fi

cd "${SUGAR_DIR}"
mkdir -p downloads

if [[ -n "${SUGAR_HTTP_PROXY}" ]]; then
  export http_proxy="${SUGAR_HTTP_PROXY}"
  export https_proxy="${SUGAR_HTTP_PROXY}"
  echo "[SUGAR-ASSETS] using SUGAR_HTTP_PROXY=${SUGAR_HTTP_PROXY}"
fi

download_and_unzip() {
  local file_id="$1"
  local zip_name="$2"
  local expected_dir="$3"

  if [[ -e "${expected_dir}" ]]; then
    echo "[SUGAR-ASSETS] ${expected_dir} already exists; skipping ${zip_name}"
    return 0
  fi

  echo "[SUGAR-ASSETS] downloading ${zip_name} from official Google Drive id ${file_id}"
  "${GDOWN_BIN}" "${file_id}" -O "downloads/${zip_name}"

  echo "[SUGAR-ASSETS] unzipping ${zip_name}"
  unzip -q "downloads/${zip_name}"
  rm -f "downloads/${zip_name}"

  if [[ ! -e "${expected_dir}" ]]; then
    echo "Expected ${expected_dir} after unzipping ${zip_name}, but it is missing" >&2
    exit 5
  fi
}

download_and_unzip "1AIJWqS5rFGl5u2Qq6jCCTHKdh51SX2Sc" "data.zip" "data"
download_and_unzip "1wXNAjNMrfV0e-d2pQ6m9dm4xrG5lSoyD" "descriptions.zip" "descriptions"
download_and_unzip "1Uc2SPPVvTboEgw4Scyuz3TmzNKDg-dx-" "demo_ckpts.zip" "demo_ckpts"

echo "[SUGAR-ASSETS] official asset directories:"
du -sh data descriptions demo_ckpts
find data descriptions demo_ckpts -maxdepth 2 -type d | sort | sed 's#^#[SUGAR-ASSETS] dir #'
