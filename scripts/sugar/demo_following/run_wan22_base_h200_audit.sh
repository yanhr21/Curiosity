#!/usr/bin/env bash
# Strict-load and hash-audit the exact public Wan2.2-TI2V-5B base on H200.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/yanhongru/Curiosity}"
SOURCE_ROOT=${1:?official Wan2.2 source root is required}
CHECKPOINT_ROOT=${2:?official Wan2.2-TI2V-5B checkpoint root is required}
RUNTIME_PYTHON=${3:?isolated official-compatible Python is required}
OUTPUT_DIR=${4:?output directory is required}
SOURCE_COMMIT=${SOURCE_COMMIT:-42bf4cfaa384bc21833865abc2f9e6c0e67233dc}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "run inside a retained Slurm allocation" >&2
    exit 2
fi
DEVICE_NAME=$(
    "$RUNTIME_PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
)
if [[ "$DEVICE_NAME" != *H200* ]]; then
    echo "expected H200, found: $DEVICE_NAME" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
"$RUNTIME_PYTHON" "$PROJECT_ROOT/scripts/sugar/demo_following/strict_load_wan22_ti2v_5b.py" \
    --source-root "$SOURCE_ROOT" \
    --source-commit "$SOURCE_COMMIT" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-dir "$OUTPUT_DIR/strict_load"
"$RUNTIME_PYTHON" "$PROJECT_ROOT/scripts/sugar/demo_following/audit_wan22_ti2v_5b_base.py" \
    --source-root "$SOURCE_ROOT" \
    --source-commit "$SOURCE_COMMIT" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --strict-load-result "$OUTPUT_DIR/strict_load/STRICT_LOAD_RESULT.json" \
    --hash-files \
    --output-dir "$OUTPUT_DIR"
jq -e '.passed == true' "$OUTPUT_DIR/WAN22_BASE_AUDIT.json" >/dev/null
