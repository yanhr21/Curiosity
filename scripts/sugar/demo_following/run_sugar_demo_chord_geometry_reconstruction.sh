#!/usr/bin/env bash
# Reconstruct Carry45 and Kick21 demonstration contact geometry, then retain GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="$(realpath -m "${1:-$ROOT/experiments/demo_following/sugar_demo_chord_geometry_v1}")"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained GPU compute step." >&2; exit 2 ;;
esac

run_motion() {
    local task="$1"
    local motion="$2"
    local name="$3"
    local output="$OUTPUT_ROOT/$name"
    if [[ -f "$output/RESULT.json" ]]; then
        echo "CHORD_DEMO_GEOMETRY_REUSE task=$task output=$output"
        return
    fi
    "$PYTHON_BIN" -u \
        "$ROOT/scripts/sugar/demo_following/reconstruct_sugar_demo_chord_geometry.py" \
        --task "$task" \
        --motion-dir "$ROOT/$motion" \
        --output-dir "$output" \
        --official-chord-root "$ROOT/experiments/runtime_assets/official_chord_5654c50e" \
        --headless --device "$DEVICE"
}

run_motion CarryBox SUGAR/data/CarryBox/data_045 carry45
run_motion KickBox SUGAR/data/KickBox/data_021 kick21

echo "SUGAR_DEMO_CHORD_GEOMETRY_READY output=$OUTPUT_ROOT"
exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_SUGAR_DEMO_CHORD_GEOMETRY_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
