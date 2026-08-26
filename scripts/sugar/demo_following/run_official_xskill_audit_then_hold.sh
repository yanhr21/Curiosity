#!/usr/bin/env bash
# Train/gate the released XSkill prototype sequence, then retain the GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_xskill_sugar_v1}")"
CORPUS="$ROOT/experiments/demo_following/official_xirl_tcc_v1/corpus"
XSKILL="$ROOT/experiments/runtime_assets/official_xskill_b748071"
DEPS="$ROOT/experiments/runtime_assets/official_xskill_pydeps"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/gr00t_n16_py310/bin/python}"
COMMIT="b748071daeb031d6b42a8dcb88c38c52297e20af"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside Slurm GPU compute." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Official XSkill audit requires a Slurm GPU job" >&2
    exit 2
fi

bash "$ROOT/scripts/sugar/demo_following/prepare_official_xskill_runtime.sh"
[[ "$(git -C "$XSKILL" rev-parse HEAD)" == "$COMMIT" ]]
jq -e '.passed == true and (.frame_counts | length == 100 and all(. == 64))' \
    "$CORPUS/RENDER_RESULT_CarryBox_000_099.json" >/dev/null
jq -e '.passed == true and (.frame_counts | length == 99 and all(. == 64))' \
    "$CORPUS/RENDER_RESULT_KickBox_000_098.json" >/dev/null

mkdir -p "$OUTPUT"
PYTHONPATH="$DEPS:$XSKILL${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled \
    "$PYTHON_BIN" -u "$ROOT/scripts/sugar/demo_following/train_evaluate_official_xskill_prototypes.py" \
    --corpus "$CORPUS" \
    --official-repo "$XSKILL" \
    --output "$OUTPUT" \
    --device cuda:0

jq -e 'has("passed") and (.criteria | type == "object")' \
    "$OUTPUT/REPRESENTATION_RESULT.json" >/dev/null
decision="$(jq -r '.passed' "$OUTPUT/REPRESENTATION_RESULT.json")"
echo "XSKILL_AUTOMATIC_DECISION passed=$decision result=$OUTPUT/REPRESENTATION_RESULT.json"

PYTHONPATH="$DEPS:$XSKILL${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -u "$ROOT/scripts/sugar/demo_following/render_official_xskill_prototype_evidence.py" \
    --corpus "$CORPUS" \
    --official-repo "$XSKILL" \
    --experiment "$OUTPUT" \
    --output "$OUTPUT/videos_prototype_evidence" \
    --adapter "$ROOT/scripts/sugar/demo_following/train_evaluate_official_xskill_prototypes.py" \
    --ffmpeg /public/home/yanhongru/envs/sugar_py311_isaacsim510/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 \
    --device cuda:0
jq -e '(.videos | length) == 4' "$OUTPUT/videos_prototype_evidence/VIDEO_RESULT.json" >/dev/null

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_XSKILL_AUDIT_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
