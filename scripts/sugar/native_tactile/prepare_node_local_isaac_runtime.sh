#!/usr/bin/env bash
# Copy the current Isaac/PyTorch environment off the shared filesystem.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run inside a retained Slurm allocation." >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NODE_LOCAL_ROOT" >&2
  exit 2
fi

SOURCE_ENV="/public/home/yanhongru/envs/sugar_py311_isaacsim510"
SOURCE_PYTHON="/public/home/yanhongru/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu"
TARGET_ROOT="$1"
TARGET_ENV="$TARGET_ROOT/env"
TARGET_PYTHON="$TARGET_ROOT/python-base"

mkdir -p "$TARGET_ENV" "$TARGET_PYTHON"
rsync -a --info=stats2 \
  --exclude='__pycache__/***' --exclude='*.pyc' \
  "$SOURCE_PYTHON/" "$TARGET_PYTHON/"
rsync -a --info=stats2 \
  --exclude='__pycache__/***' --exclude='*.pyc' \
  --exclude='lib/python3.11/site-packages/isaacsim/kit/data/documents/Kit/shared/screenshots/***' \
  "$SOURCE_ENV/" "$TARGET_ENV/"

ln -sfn "$TARGET_PYTHON/bin/python3.11" "$TARGET_ENV/bin/python3.11"
sed -i \
  -e "s#^home = .*#home = $TARGET_PYTHON/bin#" \
  -e "s#^executable = .*#executable = $TARGET_PYTHON/bin/python3.11#" \
  "$TARGET_ENV/pyvenv.cfg"

PYTHONPYCACHEPREFIX="$TARGET_ROOT/pycache" \
  "$TARGET_ENV/bin/python" -u -c \
  'import numpy, sympy, torch; print("NODE_LOCAL_RUNTIME_OK", numpy.__version__, sympy.__version__, torch.__version__, flush=True)'
