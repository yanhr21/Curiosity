#!/usr/bin/env bash
# Render all official CarryBox/KickBox reference motions for XIRL, then retain the GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_xirl_tcc_v1/corpus}")"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_xirl_full_pycache_${SLURM_JOB_ID:-local}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export SUGAR_DISABLE_RSL_RL_GIT_SNAPSHOT=1
export DISPLAY=""

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside Slurm GPU compute." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "XIRL corpus rendering requires a Slurm GPU job" >&2
    exit 2
fi

render_task() {
    local task="$1"
    local last_motion="$2"
    local motion_ids=()
    local motion_id
    for ((motion_id = 0; motion_id <= last_motion; motion_id++)); do
        motion_ids+=("$motion_id")
    done
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_xirl_reference_corpus.py" \
        --task "$task" --motion-ids "${motion_ids[@]}" --output-root "$OUTPUT" \
        --frames-per-motion 64 --camera-width 320 --camera-height 320 \
        --enable_cameras --headless --device cuda:0 \
        --kit_args="--/renderer/multiGpu/enabled=false"
}

render_task CarryBox 99
render_task KickBox 98

if [[ "${XIRL_SKIP_HOLD_AFTER_RENDER:-0}" == "1" ]]; then
    exit 0
fi

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_XIRL_FULL_REFERENCE_CORPUS_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
