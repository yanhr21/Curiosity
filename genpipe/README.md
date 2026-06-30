# genpipe — text → image → 3D asset generation for Newton sims

A two-model generation bridge that produces **rigid-body 3D assets** (GLB meshes)
for use in Newton simulations:

```
text prompt ──[PixelDiT]──▶ image (PNG) ──[TRELLIS.2]──▶ textured mesh (GLB) ──▶ Newton rigid body
```

Both models live as git submodules under `third_party/` and run in their **own
conda envs** (their dependency stacks conflict with each other and with `newton`).
They are bridged by files on disk in `pipeline_out/` (gitignored).

| Stage | Repo (submodule) | conda env | In → Out |
|---|---|---|---|
| Text → image | `third_party/PixelDiT` | `pixeldit` | prompt → 1024² PNG |
| Image → 3D | `third_party/TRELLIS.2` | `trellis2` | PNG → `.glb` (PBR mesh) |

## Submodules

```bash
git submodule update --init --recursive third_party/PixelDiT third_party/TRELLIS.2
```

## Environment setup (conda)

### PixelDiT (`pixeldit`) — pure pip, no compile
```bash
conda create -n pixeldit python=3.10 -y
conda activate pixeldit
pip install -r third_party/PixelDiT/requirements.txt    # torch 2.5.0+cu124
```
Checkpoint `nvidia/PixelDiT-1300M-1024px` and the `Efficient-Large-Model/gemma-2-2b-it`
text encoder auto-download on first run (both public — no HF token needed).

### TRELLIS.2 (`trellis2`) — compiles CUDA extensions
The upstream `setup.sh --basic` runs `sudo apt install libjpeg-dev` + `pillow-simd`,
and assumes a cu124-matching `nvcc`. This box's system `nvcc` is 11.8, so we install a
**cuda-toolkit 12.4 into the env** and skip pillow-simd (stock Pillow is fine). The
staged build script is `/tmp/build_trellis2.sh` (kept out of the repo); essentials:

```bash
conda create -n trellis2 python=3.10 -y && conda activate trellis2
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
conda install -y -c "nvidia/label/cuda-12.4.1" cuda-toolkit   # nvcc matching cu124
export CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH" TORCH_CUDA_ARCH_LIST="8.9"
# basic deps (no pillow-simd/sudo):
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
    transformers gradio==6.0.1 tensorboard pandas lpips zstandard kornia timm
pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
# Attention backend: flash-attn==2.7.3 ships only an sdist whose build can't see torch
# under pip isolation, so it fails to install. TRELLIS.2 supports xformers as a drop-in
# (sparse + dense), selected via ATTN_BACKEND=xformers. Use the torch-2.6.0-matching wheel:
pip install "xformers==0.0.29.post3" --no-deps
# compiled extensions (need CUDA_HOME above): nvdiffrast (v0.4.0), nvdiffrec (renderutils),
# CuMesh, FlexGEMM (import name flex_gemm), o-voxel — clone URLs in third_party/TRELLIS.2/setup.sh.
```
The 4B checkpoint `microsoft/TRELLIS.2-4B` auto-downloads on first run (≥24 GB VRAM).

> **Runtime note:** `trellis2` needs `CUDA_HOME=$CONDA_PREFIX` on the PATH **and**
> `ATTN_BACKEND=xformers` at inference time. `nvdiffrast` JIT-compiles its CUDA ops on
> first render (needs CUDA_HOME). `genpipe/trellis_image_to_glb.py` sets ATTN_BACKEND
> itself; `run_pipeline.sh` exports both.

## Usage

End-to-end (defaults: `pipeline_out/prompt.txt`, output to `pipeline_out/`, GPU 1):
```bash
bash genpipe/run_pipeline.sh [PROMPT_FILE] [OUTDIR] [GPU]
```

Just the 3D step on an existing image:
```bash
conda activate trellis2
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
CUDA_VISIBLE_DEVICES=1 python genpipe/trellis_image_to_glb.py \
    --image pipeline_out/vis/<img>.png --out pipeline_out/object.glb \
    --preview pipeline_out/object_preview.png
```

Outputs (all under `pipeline_out/`, gitignored):
- `vis/.../*.png` — PixelDiT image
- `object.glb` — TRELLIS.2 textured mesh (load as a Newton rigid body)
- `object_preview.png` — one rendered frame of the mesh

## Gotchas (all resolved — handled in the scripts unless noted)

Setting these up surfaced several issues; the fixes are baked into `genpipe/` so a
clean run works, but they're recorded here for reproducing the envs from scratch:

1. **hf-xet stalls** on this box — HF downloads hang at 0 B. Set `HF_HUB_DISABLE_XET=1`
   (run_pipeline.sh exports it).
2. **flash-attn won't install** — its PyPI sdist build can't see torch under pip isolation
   (`No module named 'torch'`). Use **xformers** instead (`ATTN_BACKEND=xformers`), which
   TRELLIS.2 supports for both sparse + dense attention. `trellis_image_to_glb.py` sets it.
3. **nvcc mismatch** — system `nvcc` is 11.8 but torch is cu124; the CUDA extensions won't
   build. Install `cuda-toolkit 12.4` into the env and `export CUDA_HOME=$CONDA_PREFIX`.
4. **DINOv3 image encoder is gated** (`facebook/dinov3-vitl16-pretrain-lvd1689m`) — needs an
   `HF_TOKEN` with access. *(Not auto-handled — you must provide the token.)*
5. **RMBG-2.0 rembg is gated** (`briaai/RMBG-2.0`) — `trellis_image_to_glb.py` redirects it to
   the ungated `ZhengPeng7/BiRefNet` (identical `AutoModelForImageSegmentation` interface).
6. **BiRefNet remote code needs `einops`** — `pip install einops` into the env.
7. **BiRefNet ships fp16 weights**, wrapper feeds fp32 input → conv dtype mismatch; the script
   forces the rembg model to fp32.
8. **transformers ≥5 nests DINOv3 blocks** under `.model` (the encoder); TRELLIS.2 expects the
   flat `.layer`. The script patches `DinoV3FeatureExtractor.extract_features` to find either.
9. **`trellis2` is a source package** (not pip-installed) — the script adds its repo root to
   `sys.path` and `chdir`s into it.

## Tips for sim-ready assets
Prompt for a **single, centered object on a plain background, fully visible** — TRELLIS.2
reconstructs one foreground object per image. The GLB is unit-scaled into the
`[-0.5, 0.5]³` AABB; rescale to physical size when adding it to a Newton `ModelBuilder`.
