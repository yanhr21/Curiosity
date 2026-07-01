#!/bin/bash
# Set up the PixelDiT + TRELLIS.2 conda envs on a cluster (no system CUDA needed;
# a conda cuda-toolkit provides nvcc). Run from anywhere; resolves the repo from
# this script's location. GPU is NOT required to build (extensions compile for
# TORCH_ARCH). Usage:
#   bash genpipe/cluster_setup.sh [TORCH_ARCH]     # default arch "8.0;9.0" (A100+H100)
#
# Stages are idempotent-ish (skips an env if it already imports). Heavy pip/compile
# is fine on a login node; model downloads happen later at generation time.
set -uo pipefail
TORCH_ARCH="${1:-8.0;9.0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT=/tmp/trellis_ext_$USER
export HF_HUB_DISABLE_XET=1   # xet transfer stalls on some networks

source "$(conda info --base)/etc/profile.d/conda.sh"
say() { echo; echo "########## $* ##########"; }
have_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }

# ---------------- PixelDiT env (pure pip) ----------------
say "PixelDiT env"
if have_env pixeldit && conda run -n pixeldit python -c "import torch,transformers" 2>/dev/null; then
  echo "[skip] pixeldit already set up"
else
  conda create -n pixeldit python=3.10 -y || exit 1
  conda run --no-capture-output -n pixeldit pip install -r "$REPO/third_party/PixelDiT/requirements.txt" || exit 1
fi
conda run -n pixeldit python -c "import torch;print('[pixeldit] torch',torch.__version__,torch.version.cuda)" || true

# ---------------- TRELLIS.2 env ----------------
say "TRELLIS.2 env — torch 2.6.0 cu124"
if ! have_env trellis2; then
  conda create -n trellis2 python=3.10 -y || exit 1
  conda run --no-capture-output -n trellis2 pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124 || exit 1
fi

say "cuda-toolkit 12.4 into trellis2 (nvcc matching cu124)"
conda run -n trellis2 bash -c 'command -v nvcc >/dev/null && nvcc --version | tail -1' \
  || conda install -n trellis2 -y -c "nvidia/label/cuda-12.4.1" cuda-toolkit || exit 1

# env for the compiled extensions
TRE_PREFIX="$(conda run -n trellis2 python -c 'import sys,os;print(os.path.dirname(os.path.dirname(sys.executable)))')"
export CUDA_HOME="$TRE_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="$TORCH_ARCH"
echo "CUDA_HOME=$CUDA_HOME  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

say "basic deps (skip sudo apt + pillow-simd; stock Pillow is fine)"
conda run --no-capture-output -n trellis2 pip install imageio imageio-ffmpeg tqdm easydict \
  opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips \
  zstandard kornia timm einops || true
conda run --no-capture-output -n trellis2 pip install \
  "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" || true

say "xformers 0.0.29.post3 (attention backend; flash-attn not used)"
conda run --no-capture-output -n trellis2 pip install "xformers==0.0.29.post3" --no-deps || true

say "compiled extensions (CUDA_HOME=$CUDA_HOME, arch=$TORCH_ARCH)"
mkdir -p "$EXT"
build() {  # name repo_url [branch]
  local name="$1" url="$2" br="${3:-}"
  [ -d "$EXT/$name" ] || git clone ${br:+-b "$br"} --recursive "$url" "$EXT/$name" || return 1
  CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST="$TORCH_ARCH" \
    conda run --no-capture-output -n trellis2 pip install "$EXT/$name" --no-build-isolation
}
build nvdiffrast https://github.com/NVlabs/nvdiffrast.git v0.4.0    && echo "[ok] nvdiffrast" || echo "[FAIL] nvdiffrast"
build nvdiffrec  https://github.com/JeffreyXiang/nvdiffrec.git renderutils && echo "[ok] nvdiffrec" || echo "[FAIL] nvdiffrec"
build CuMesh     https://github.com/JeffreyXiang/CuMesh.git      && echo "[ok] CuMesh" || echo "[FAIL] CuMesh"
build FlexGEMM   https://github.com/JeffreyXiang/FlexGEMM.git    && echo "[ok] FlexGEMM" || echo "[FAIL] FlexGEMM"
rm -rf "$EXT/o-voxel" && cp -r "$REPO/third_party/TRELLIS.2/o-voxel" "$EXT/o-voxel"
CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST="$TORCH_ARCH" \
  conda run --no-capture-output -n trellis2 pip install "$EXT/o-voxel" --no-build-isolation \
  && echo "[ok] o-voxel" || echo "[FAIL] o-voxel"

say "import smoke test (CPU — full CUDA check needs a GPU via srun)"
conda run -n trellis2 python - <<'PY'
import importlib
for m in ["torch","xformers","nvdiffrast.torch","cumesh","flex_gemm","o_voxel"]:
    try:
        importlib.import_module(m); print("[import-ok]", m)
    except Exception as e:
        print("[import-FAIL]", m, type(e).__name__, str(e)[:80])
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
echo "CLUSTER_SETUP_DONE"
