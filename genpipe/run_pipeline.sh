#!/bin/bash
# End-to-end: text prompt -> PixelDiT image -> TRELLIS.2 GLB.
# Two conda envs (pixeldit, trellis2) bridged by files in $OUTDIR. See genpipe/README.md.
#
#   bash genpipe/run_pipeline.sh [PROMPT_FILE] [OUTDIR] [GPU]
#
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_FILE="${1:-$REPO/pipeline_out/prompt.txt}"
OUTDIR="${2:-$REPO/pipeline_out}"
GPU="${3:-${CUDA_VISIBLE_DEVICES:-1}}"
mkdir -p "$OUTDIR"

# The hf-xet transfer protocol stalls on this box — force plain HTTPS downloads.
export HF_HUB_DISABLE_XET=1
# TRELLIS.2's DINOv3 image encoder is a GATED repo: export an HF token with access:
#   export HF_TOKEN=hf_...    (must have facebook/dinov3-vitl16-pretrain-lvd1689m access)
[ -n "${HF_TOKEN:-}" ] || echo "WARN: HF_TOKEN not set — TRELLIS.2 DINOv3 download will 401 (gated)."

source ~/miniconda3/etc/profile.d/conda.sh

echo "=================================================================="
echo "[1/2] PixelDiT  text -> image   (env: pixeldit, GPU $GPU)"
echo "=================================================================="
conda activate pixeldit
( cd "$REPO/third_party/PixelDiT/t2i" && CUDA_VISIBLE_DEVICES="$GPU" python inference.py \
    --config configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
    --model_path pixeldit_t2i_v1.pth \
    --txt_file "$PROMPT_FILE" \
    --custom_height 1024 --custom_width 1024 \
    --cfg_scale 2.75 --seed 2025 --step 50 \
    --negative_prompt "low quality, worst quality, over-saturated, blurry, deformed, watermark" \
    --work_dir "$OUTDIR" )
conda deactivate

IMG="$(find "$OUTDIR/vis" -name '*.png' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "$IMG" ] || { echo "ERROR: no image produced under $OUTDIR/vis"; exit 1; }
echo "[bridge] PixelDiT image -> $IMG"

echo "=================================================================="
echo "[2/2] TRELLIS.2  image -> GLB   (env: trellis2, GPU $GPU)"
echo "=================================================================="
conda activate trellis2
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export ATTN_BACKEND="xformers"   # flash-attn not installed; xformers is the drop-in backend
CUDA_VISIBLE_DEVICES="$GPU" python "$REPO/genpipe/trellis_image_to_glb.py" \
    --image "$IMG" \
    --out "$OUTDIR/object.glb" \
    --preview "$OUTDIR/object_preview.png"
conda deactivate

echo "DONE: $OUTDIR/object.glb"
