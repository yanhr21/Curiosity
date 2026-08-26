#!/usr/bin/env bash
# Render the complete Carry/Kick sphere corpus for official cross-embodiment XSkill.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_xskill_cross_embodiment_v1/sphere_corpus}")"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="/tmp/Curiosity_xskill_sphere_full_${SLURM_JOB_ID:-local}"
export PYTHONPATH="$ROOT/IsaacLab/source/isaaclab:$ROOT/IsaacLab/source/isaaclab_tasks:$ROOT/IsaacLab/source/isaaclab_rl:$ROOT/IsaacLab/source/isaaclab_mimic:$ROOT/SUGAR/source/sugar_rl:$ROOT/SUGAR/source/sugar_il:${PYTHONPATH:-}"
export ISAACLAB_GROUND_PLANE_USD="$ROOT/SUGAR/descriptions/terrain/sugar_ground_plane.usda"
export ISAACLAB_USE_LOCAL_FRAME_MARKER=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export SUGAR_DISABLE_TRAIN_DEBUG_VIS=1
export SUGAR_DISABLE_RSL_RL_GIT_SNAPSHOT=1
export DISPLAY=""

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside retained Slurm GPU compute." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "XSkill sphere corpus rendering requires a retained Slurm GPU job" >&2
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
        --embodiment sphere --frames-per-motion 64 \
        --camera-width 320 --camera-height 320 \
        --enable_cameras --headless --device cuda:0 \
        --kit_args="--/renderer/multiGpu/enabled=false"
    local result="$OUTPUT/RENDER_RESULT_SPHERE_${task}_000_$(printf '%03d' "$last_motion").json"
    local expected_count=$((last_motion + 1))
    jq -e --argjson count "$expected_count" \
        '.passed == true and .embodiment == "sphere" and (.frame_counts | length) == $count and all(.frame_counts[]; . == 64)' \
        "$result" >/dev/null
}

render_task CarryBox 99
render_task KickBox 98

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_XSKILL_SPHERE_CORPUS_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
