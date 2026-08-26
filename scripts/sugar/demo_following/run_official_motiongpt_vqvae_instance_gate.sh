#!/usr/bin/env bash
# Run the released MotionGPT VQ-VAE selected-demo gate, then retain the GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT="$(realpath -m "${1:-$ROOT/experiments/demo_following/official_motiongpt_vqvae_instance_gate_v2}")"
DEVICE="${2:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python}"

case "$(hostname)" in
    mgmtserver*|login*) echo "Run inside a retained compute step." >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
    echo "official MotionGPT VQ-VAE gate requires a retained srun compute step" >&2
    exit 2
fi
for required in \
    "$ROOT/experiments/runtime_assets/official_motiongpt_qiqi/checkpoints/pretrained_vqvae/t2m.pth" \
    "$ROOT/experiments/runtime_assets/official_motiongpt_qiqi/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy" \
    "$ROOT/experiments/runtime_assets/official_motiongpt_qiqi/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"; do
    test -f "$required" || { echo "missing official input: $required" >&2; exit 2; }
done

PYTHONPATH="$ROOT/experiments/runtime_assets/official_tmr/runtime_deps:$ROOT/experiments/runtime_assets/official_tmr${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" "$ROOT/scripts/sugar/demo_following/audit_official_motiongpt_vqvae_instance_latent.py" \
    --output "$OUTPUT" --device "$DEVICE"

printf 'gate_complete=1\nresult=%s\n' "$OUTPUT/RESULT.json" > "$OUTPUT/STATUS.env"

exec "$PYTHON_BIN" -u -c '
import torch
a = torch.randn((32768, 32768), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
print("GPU_HOLD_AFTER_OFFICIAL_MOTIONGPT_VQVAE_GATE_READY", flush=True)
while True:
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
'
