#!/usr/bin/env bash
# Train/gate released XSkill on paired G1/sphere streams, then retain the GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_xskill_cross_embodiment_v1/representation}")"
G1_CORPUS="$ROOT/experiments/demo_following/official_xirl_tcc_v1/corpus"
SPHERE_CORPUS="$ROOT/experiments/demo_following/official_xskill_cross_embodiment_v1/sphere_corpus"
XSKILL="$ROOT/experiments/runtime_assets/official_xskill_b748071"
DEPS="$ROOT/experiments/runtime_assets/official_xskill_pydeps"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/gr00t_n16_py310/bin/python}"
COMMIT="b748071daeb031d6b42a8dcb88c38c52297e20af"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside retained Slurm GPU compute." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Official cross-embodiment XSkill audit requires retained Slurm GPU compute" >&2
    exit 2
fi

bash "$ROOT/scripts/sugar/demo_following/prepare_official_xskill_runtime.sh"
[[ "$(git -C "$XSKILL" rev-parse HEAD)" == "$COMMIT" ]]
jq -e '.passed == true and ((.embodiment // "g1") == "g1") and (.frame_counts | length == 100 and all(. == 64))' \
    "$G1_CORPUS/RENDER_RESULT_CarryBox_000_099.json" >/dev/null
jq -e '.passed == true and ((.embodiment // "g1") == "g1") and (.frame_counts | length == 99 and all(. == 64))' \
    "$G1_CORPUS/RENDER_RESULT_KickBox_000_098.json" >/dev/null
jq -e '.passed == true and .embodiment == "sphere" and (.frame_counts | length == 100 and all(. == 64))' \
    "$SPHERE_CORPUS/RENDER_RESULT_SPHERE_CarryBox_000_099.json" >/dev/null
jq -e '.passed == true and .embodiment == "sphere" and (.frame_counts | length == 99 and all(. == 64))' \
    "$SPHERE_CORPUS/RENDER_RESULT_SPHERE_KickBox_000_098.json" >/dev/null

mkdir -p "$OUTPUT"
PYTHONPATH="$DEPS:$XSKILL${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled \
    "$PYTHON_BIN" -u "$ROOT/scripts/sugar/demo_following/train_evaluate_official_xskill_prototypes.py" \
    --corpus "$G1_CORPUS" \
    --sphere-corpus "$SPHERE_CORPUS" \
    --official-repo "$XSKILL" \
    --output "$OUTPUT" \
    --device cuda:0

jq -e 'has("passed") and (.criteria | type == "object") and (.protocol == "official_xskill_cross_embodiment_sugar_prototype_gate_v1")' \
    "$OUTPUT/REPRESENTATION_RESULT.json" >/dev/null
decision="$(jq -r '.passed' "$OUTPUT/REPRESENTATION_RESULT.json")"
echo "XSKILL_CROSS_EMBODIMENT_AUTOMATIC_DECISION passed=$decision result=$OUTPUT/REPRESENTATION_RESULT.json"

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_XSKILL_CROSS_EMBODIMENT_AUDIT_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
