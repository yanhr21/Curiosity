#!/usr/bin/env bash
# Render clean official SUGAR reference frames for XIRL, then retain the GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_xirl_tcc_v1/corpus}")"
TASK="${2:-CarryBox}"
shift 2 || true
MOTION_IDS=("$@")
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
KIT_ARGS="--/renderer/multiGpu/enabled=false"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside retained compute." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "XIRL reference rendering requires a retained srun compute step" >&2
    exit 2
fi
if [[ ${#MOTION_IDS[@]} -eq 0 ]]; then
    echo "at least one motion ID is required" >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_xirl_render_pycache_${SLURM_JOB_ID:-local}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export SUGAR_DISABLE_RSL_RL_GIT_SNAPSHOT=1
export DISPLAY=""

"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_xirl_reference_corpus.py" \
    --task "$TASK" --motion-ids "${MOTION_IDS[@]}" --output-root "$OUTPUT" \
    --frames-per-motion 64 --camera-width 320 --camera-height 320 --write-preview-mp4 \
    --enable_cameras --headless --device cuda:0 --kit_args "$KIT_ARGS"

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_XIRL_REFERENCE_RENDER_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
