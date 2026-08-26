#!/usr/bin/env bash
# Render full Carry45 and Kick21 CHORD-geometry evidence, then retain GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GEOMETRY_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/sugar_demo_chord_geometry_v1}")"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac

render_motion() {
    local task="$1"
    local motion="$2"
    local geometry="$3"
    local output="$GEOMETRY_ROOT/visualizations/$geometry"
    if [[ -f "$output/RENDER_PROOF.json" ]]; then
        echo "CHORD_DEMO_RENDER_REUSE task=$task output=$output"
        return
    fi
    "$PYTHON_BIN" -u "$ROOT/scripts/sugar/demo_following/render_sugar_demo_chord_geometry_offline.py" \
        --task "$task" \
        --motion-dir "$ROOT/$motion" \
        --geometry-dir "$GEOMETRY_ROOT/$geometry" \
        --output-dir "$output"
}

render_motion CarryBox SUGAR/data/CarryBox/data_045 carry45
render_motion KickBox SUGAR/data/KickBox/data_021 kick21

echo "SUGAR_DEMO_CHORD_GEOMETRY_VIDEOS_READY output=$GEOMETRY_ROOT/visualizations"
exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_SUGAR_DEMO_CHORD_RENDER_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
