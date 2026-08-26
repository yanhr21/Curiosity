#!/usr/bin/env bash
# Render Carry45/Kick21 with the released-XSkill-style sphere embodiment, then retain the GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_xskill_cross_embodiment_v1/sphere_canary}")"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_xskill_sphere_canary_${SLURM_JOB_ID:-local}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export SUGAR_DISABLE_RSL_RL_GIT_SNAPSHOT=1
export DISPLAY=""

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside retained Slurm compute." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "XSkill sphere rendering requires retained Slurm compute" >&2
    exit 2
fi

render_reference() {
    local task="$1"
    local motion_id="$2"
    "$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/render_xirl_reference_corpus.py" \
        --task "$task" --motion-ids "$motion_id" --output-root "$OUTPUT" \
        --embodiment sphere --frames-per-motion 64 \
        --camera-width 320 --camera-height 320 --write-preview-mp4 \
        --enable_cameras --headless --device cuda:0 \
        --kit_args="--/renderer/multiGpu/enabled=false"
    local result="$OUTPUT/RENDER_RESULT_SPHERE_${task}_$(printf '%03d' "$motion_id")_$(printf '%03d' "$motion_id").json"
    jq -e '.passed == true and .embodiment == "sphere" and .frame_counts == [64] and (.preview_videos | length) == 1' \
        "$result" >/dev/null
}

render_reference CarryBox 45
render_reference KickBox 21

[[ -s "$OUTPUT/preview_videos/CarryBox_045.mp4" ]]
[[ -s "$OUTPUT/preview_videos/KickBox_021.mp4" ]]

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_XSKILL_SPHERE_CANARIES_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
