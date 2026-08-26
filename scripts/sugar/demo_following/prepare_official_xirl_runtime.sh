#!/usr/bin/env bash
# Pin the released XIRL runtime and narrow Python-3.11 compatibility dependencies.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ASSETS="$ROOT/experiments/runtime_assets"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
XIRL="$ASSETS/official_google_research_xirl"
TORCHKIT="$ASSETS/official_torchkit_v0p0p2"
XMAGICAL="$ASSETS/official_xmagical_v0p0p2"
DEPS="$ASSETS/official_xirl_py311_compat_deps"

clone_pin() {
    local url="$1"
    local destination="$2"
    local commit="$3"
    local sparse_path="${4:-}"
    if [[ ! -d "$destination/.git" ]]; then
        git clone --filter=blob:none --no-checkout "$url" "$destination"
    fi
    if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]]; then
        git -C "$destination" fetch --quiet --depth=1 origin "$commit"
        if [[ -n "$sparse_path" ]]; then
            git -C "$destination" sparse-checkout init --cone
            git -C "$destination" sparse-checkout set "$sparse_path"
        fi
        git -C "$destination" checkout --quiet --detach "$commit"
    fi
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]]
}

mkdir -p "$ASSETS" "$DEPS"
clone_pin https://github.com/google-research/google-research.git "$XIRL" \
    807d4a2f41202059bac2446259d135a89ed3630a xirl
clone_pin https://github.com/kevinzakka/torchkit.git "$TORCHKIT" \
    dd5824445b5c3ec9f5b0973c89ffd489500b9eae
clone_pin https://github.com/kevinzakka/x-magical.git "$XMAGICAL" \
    31fa989e2ecd6fcdbcdc6f9b70057ab28f6184f2

ensure_dependency() {
    local requirement="$1"
    local marker="$2"
    if [[ ! -d "$DEPS/$marker" ]]; then
        "$PYTHON_BIN" -m pip install --no-input --no-deps --target "$DEPS" "$requirement"
    fi
}

ensure_dependency albumentations==0.5.2 albumentations-0.5.2.dist-info
ensure_dependency ml-collections==0.1.0 ml_collections-0.1.0.dist-info
ensure_dependency gym==0.17.3 gym-0.17.3.dist-info
ensure_dependency pymunk==5.6.0 pymunk-5.6.0.dist-info
ensure_dependency pygame==2.6.1 pygame-2.6.1.dist-info

export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONPATH="$ROOT/scripts/sugar/demo_following/xirl_compat:$DEPS:$XMAGICAL:$TORCHKIT:$XIRL/xirl"
"$PYTHON_BIN" - <<'PY'
import gym
import xmagical
import utils
from xirl import common

assert gym.__version__ == "0.17.3"
assert xmagical.__version__ == "0.0.2"
print("OFFICIAL_XIRL_RUNTIME_READY", flush=True)
PY
