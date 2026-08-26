#!/usr/bin/env bash
# Run the frozen official RoboCLIP representation gate, then retain the granted GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"
CORPUS="${SUGAR_ROBOCLIP_CORPUS:-$ROOT/experiments/demo_following/official_xirl_tcc_v1/corpus}"
OFFICIAL="${SUGAR_ROBOCLIP_OFFICIAL:-$ROOT/experiments/runtime_assets/official_roboclip_2d3f779}"
OUTPUT="${1:-$ROOT/experiments/demo_following/official_roboclip_v1}"
RESULT="$OUTPUT/REPRESENTATION_RESULT.json"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained Slurm compute step" >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "official RoboCLIP audit requires a retained Slurm compute step" >&2
    exit 2
fi
if [[ -e "$RESULT" ]]; then
    echo "refusing to overwrite completed RoboCLIP result: $RESULT" >&2
    exit 2
fi

[[ "$(git -C "$OFFICIAL" rev-parse HEAD)" == "2d3f779033f1f3adf307a64080742e158caafe67" ]]
[[ "$(git -C "$OFFICIAL/S3D_HowTo100M" rev-parse HEAD)" == "b8cd0bbfd16fe41629d1b15e0cf384d75f56101a" ]]
SOURCE="$OFFICIAL/S3D_HowTo100M/s3dg.py"
WEIGHTS="$OFFICIAL/checkpoints/s3d_howto100m.pth"
DICTIONARY="$OFFICIAL/checkpoints/s3d_dict.npy"
for path in "$SOURCE" "$WEIGHTS" "$DICTIONARY"; do
    [[ -s "$path" ]] || { echo "missing official RoboCLIP asset: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT"
"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/evaluate_official_roboclip_video_reward.py" \
    --corpus "$CORPUS" \
    --official-source "$SOURCE" \
    --weights "$WEIGHTS" \
    --dictionary "$DICTIONARY" \
    --output "$RESULT" \
    --device cuda:0 \
    --batch-size 2

"$PYTHON_BIN" - "$RESULT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["protocol"] == "official_roboclip_sugar_selected_demo_v1"
assert len(result["criteria"]) == 6
assert result["passed"] == all(result["criteria"].values())
print(f"ROBOCLIP_AUTOMATIC_DECISION passed={result['passed']}", flush=True)
PY

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_ROBOCLIP_AUDIT_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
