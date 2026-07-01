"""Image -> 3D (GLB) via TRELLIS.2, parametrized from the upstream example.py.

Run inside the `trellis2` conda env (see genpipe/README.md). Loads
Trellis2ImageTo3DPipeline, reconstructs a mesh from a single image, exports a
GLB, and (optionally) saves a single preview PNG + a turntable MP4.

Usage:
    CUDA_VISIBLE_DEVICES=1 python genpipe/trellis_image_to_glb.py \
        --image pipeline_out/vis/0.png \
        --out   pipeline_out/object.glb \
        --preview pipeline_out/object_preview.png
"""

import argparse
import os
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# flash-attn isn't installed (its sdist build can't see torch under pip isolation);
# TRELLIS.2 supports xformers as a drop-in attention backend for both sparse + dense.
os.environ.setdefault("ATTN_BACKEND", "xformers")

# trellis2 is a source package (not pip-installed) — run from / importable via its repo root.
_TRELLIS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "TRELLIS.2")
)
if _TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, _TRELLIS_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="TRELLIS.2 image -> GLB")
    ap.add_argument("--image", required=True, help="Input image path (single object, plain background).")
    ap.add_argument("--out", required=True, help="Output .glb path.")
    ap.add_argument("--preview", default=None, help="Optional: save one rendered PNG frame here.")
    ap.add_argument("--video", default=None, help="Optional: save a turntable .mp4 here.")
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B", help="HF pipeline id.")
    ap.add_argument(
        "--rembg-model",
        default="ZhengPeng7/BiRefNet",
        help="Background-removal model. The 4B pipeline.json defaults to the gated "
        "briaai/RMBG-2.0; we redirect to this ungated BiRefNet (same interface).",
    )
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--decimation-target", type=int, default=200000)
    args = ap.parse_args()

    # Resolve all user paths before chdir; cd into the TRELLIS.2 repo so its
    # relative asset loads (configs, hdri) resolve regardless of caller cwd.
    args.image = os.path.abspath(args.image)
    args.out = os.path.abspath(args.out)
    if args.preview:
        args.preview = os.path.abspath(args.preview)
    if args.video:
        args.video = os.path.abspath(args.video)
    os.chdir(_TRELLIS_ROOT)

    import time

    import imageio
    import torch
    from PIL import Image

    _prof = {}

    import o_voxel
    import trellis2.pipelines.rembg as _rembg_mod
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # The 4B pipeline.json sets rembg to the gated briaai/RMBG-2.0; redirect to an
    # ungated BiRefNet (the wrapper's own default, identical interface) so no extra
    # HF gate is needed for background removal.
    _OrigBiRefNet = _rembg_mod.BiRefNet
    _rembg_target = args.rembg_model

    class _BiRefNetUngated(_OrigBiRefNet):
        def __init__(self, model_name="ZhengPeng7/BiRefNet", *a, **k):
            if model_name == "briaai/RMBG-2.0":
                print(f"[trellis] redirecting gated rembg -> {_rembg_target}", flush=True)
                model_name = _rembg_target
            super().__init__(model_name, *a, **k)
            # ungated BiRefNet ships fp16 weights; the wrapper feeds an fp32 input tensor,
            # so keep the model in fp32 to avoid a dtype mismatch in the first conv.
            self.model = self.model.float()

    _rembg_mod.BiRefNet = _BiRefNetUngated

    # transformers >=5 nests DINOv3's transformer blocks under `.model` (the encoder);
    # TRELLIS.2 was written against the flat `.layer`. Patch the feature walk to find
    # the layer list in either layout. Everything else (embeddings/rope/layer forward
    # signatures) is unchanged, so only the layer access needs fixing.
    import torch.nn.functional as _F
    from trellis2.modules import image_feature_extractor as _ife

    def _extract_features_compat(self, image):
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)
        layers = getattr(self.model, "layer", None)
        if layers is None:
            layers = self.model.model.layer
        for layer_module in layers:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
        return _F.layer_norm(hidden_states, hidden_states.shape[-1:])

    _ife.DinoV3FeatureExtractor.extract_features = _extract_features_compat

    print(f"[trellis] loading pipeline {args.model} ...", flush=True)
    _t = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    torch.cuda.synchronize()
    _prof["load_4B_model_s"] = time.perf_counter() - _t

    print(f"[trellis] running on {args.image} ...", flush=True)
    _t = time.perf_counter()
    image = Image.open(args.image).convert("RGB")
    mesh = pipeline.run(image)[0]
    mesh.simplify(16777216)  # nvdiffrast limit
    torch.cuda.synchronize()
    _prof["reconstruct_s"] = time.perf_counter() - _t

    if args.preview or args.video:
        import cv2
        from trellis2.renderers import EnvMap
        from trellis2.utils import render_utils

        # bundled HDRI from the TRELLIS.2 repo
        hdri = os.path.join(_TRELLIS_ROOT, "assets", "hdri", "forest.exr")
        envmap = EnvMap(
            torch.tensor(
                cv2.cvtColor(cv2.imread(hdri, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32,
                device="cuda",
            )
        )
        frames = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        if args.preview:
            os.makedirs(os.path.dirname(os.path.abspath(args.preview)), exist_ok=True)
            imageio.imwrite(args.preview, frames[0])
            print(f"[trellis] preview frame -> {os.path.abspath(args.preview)}", flush=True)
        if args.video:
            os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
            imageio.mimsave(args.video, frames, fps=15)
            print(f"[trellis] turntable -> {os.path.abspath(args.video)}", flush=True)

    print("[trellis] exporting GLB ...", flush=True)
    _t = time.perf_counter()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    glb.export(args.out, extension_webp=True)
    _prof["mesh_texture_export_s"] = time.perf_counter() - _t
    print(f"[trellis] GLB -> {os.path.abspath(args.out)}", flush=True)
    print(
        "[PROFILE trellis] "
        + "  ".join(f"{k}={v:.2f}" for k, v in _prof.items())
        + f"  total={sum(_prof.values()):.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
