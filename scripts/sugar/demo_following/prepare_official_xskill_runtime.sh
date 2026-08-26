#!/usr/bin/env bash
# Pin the released XSkill source used by the SUGAR prototype-sequence audit.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASSETS="$ROOT/experiments/runtime_assets"
XSKILL="$ASSETS/official_xskill_b748071"
DEPS="$ASSETS/official_xskill_pydeps"
COMMIT="b748071daeb031d6b42a8dcb88c38c52297e20af"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/gr00t_n16_py310/bin/python}"

mkdir -p "$ASSETS" "$DEPS"
if [[ ! -d "$XSKILL/.git" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/real-stanford/xskill.git "$XSKILL"
fi
if [[ "$(git -C "$XSKILL" rev-parse HEAD 2>/dev/null || true)" != "$COMMIT" ]]; then
    git -C "$XSKILL" fetch --quiet --depth=1 origin "$COMMIT"
    git -C "$XSKILL" checkout --quiet --detach "$COMMIT"
fi
[[ "$(git -C "$XSKILL" rev-parse HEAD)" == "$COMMIT" ]]

if [[ ! -d "$DEPS/absl" ]]; then
    "$PYTHON_BIN" -m pip install --no-input --no-deps --target "$DEPS" absl-py==2.0.0
fi

PYTHONPATH="$DEPS:$XSKILL${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import absl
import torch
import torchvision
import wandb

assert torch.cuda.is_available() or not __import__("os").environ.get("SLURM_JOB_ID")
print(
    "OFFICIAL_XSKILL_RUNTIME_READY",
    f"torch={torch.__version__}",
    f"torchvision={torchvision.__version__}",
    flush=True,
)
PY
