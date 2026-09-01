"""How fast is the closed loop: policy -> physics -> render -> visual+tactile -> policy?

Every fps number quoted elsewhere in this repo is physics-only. This measures the whole
loop and attributes it, because the interesting question for a real training setup is which
stage is the wall, not what the total is.

Stages, each fenced with ``wp.synchronize()`` so the attribution is real and not just kernel
launch time:

    policy    the tracker's actor (NumPy, on the CPU) plus observe()
    physics   scene.step(), the number every other benchmark here reports
    render    ViewerGL begin_frame / log_state / end_frame
    readout   get_frame(), a GPU->GPU PBO copy, so a policy could consume it in place
    tactile   solver.update_contacts() + PatchTactile.update(), also GPU-resident

``total`` is then re-measured with the fences removed, which is the number a training loop
would actually see; the fenced sum is slightly higher because synchronising costs something.

One resolution per process: ViewerGL owns an EGL context, and tearing several down inside one
process is a good way to get a driver crash rather than a measurement. Sweep from the shell.

    python -m sugar_newton.validation.bench_loop --width 640 --height 480 --frames 120
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from sugar_newton.validation.g1_carrybox_policy import (
    Actor,
    G1PolicyScene,
    load_clip,
)


class Stage:
    """Accumulates per-stage wall time, synchronising so the number means something."""

    def __init__(self, fenced: bool = True):
        self.t: dict[str, float] = {}
        self.fenced = fenced
        self._name = None

    def __call__(self, name: str):
        self._name = name
        return self

    def __enter__(self):
        if self.fenced:
            wp.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.fenced:
            wp.synchronize()
        self.t[self._name] = self.t.get(self._name, 0.0) + time.perf_counter() - self._t0
        return False


def build_tactile(scene):
    """PatchTactile over the two rubber hands, counting only box contacts."""
    from sugar_newton.tactile.reducer import PatchTactile

    body_of = scene.model.shape_body.numpy()
    labels = [l.split("/")[-1] for l in scene.model.body_label]
    patches, box = [], []
    for s, b in enumerate(body_of):
        if b < 0:
            continue
        if labels[b] in ("left_rubber_hand", "right_rubber_hand"):
            patches.append(s)
        elif labels[b] == "box":
            box.append(s)
    if not patches or not box:
        return None, patches
    return PatchTactile(scene.model, patch_shapes=patches,
                        counterpart_shapes=box), patches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data_000")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--start", type=int, default=180, help="begin in the carry")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--box-tris", type=int, default=2000)
    ap.add_argument("--hand-tris", type=int, default=5000)
    ap.add_argument("--no-render", action="store_true",
                    help="physics + tactile only, for a render-free reference")
    ap.add_argument("--dump", default=None,
                    help="npz of per-frame timings and tactile rows, for the video")
    args = ap.parse_args()

    wp.init()
    clip = load_clip(args.clip)
    dt = 1.0 / clip["fps"]
    scene = G1PolicyScene(clip, box_tris=args.box_tris, hand_tris=args.hand_tris)
    actor = Actor()
    tactile, patches = build_tactile(scene)

    viewer = None
    if not args.no_render:
        import math
        import os

        import pyglet

        if os.environ.get("G1_XVFB") != "1":
            pyglet.options["headless"] = True
        from newton.viewer import ViewerGL

        viewer = ViewerGL(width=args.width, height=args.height, headless=True)
        viewer.set_model(scene.model)

        def aim():
            bq = scene.state_0.body_q.numpy()
            pel, bx = bq[0, :3], bq[scene.box_body, :3]
            mid = 0.5 * (pel + bx) if np.isfinite(bx).all() else pel
            cam = mid + np.array([1.6, -1.6, 0.5])
            d = np.array([mid[0], mid[1], mid[2] + 0.15]) - cam
            d /= max(np.linalg.norm(d), 1e-9)
            viewer.set_camera(
                wp.vec3(*cam.tolist()),
                math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0)))),
                math.degrees(math.atan2(float(d[1]), float(d[0]))))

    frame_buf = None
    rows, times = [], []

    def one_frame(k, st: Stage, keep: bool):
        nonlocal frame_buf
        with st("policy"):
            obs = scene.observe()
            act = actor(obs)
            scene.apply(act)
        with st("physics"):
            scene.step(dt, args.substeps, "step")
        if viewer is not None:
            with st("render"):
                aim()
                viewer.begin_frame(k * dt)
                viewer.log_state(scene.state_0)
                viewer.end_frame()
            with st("readout"):
                frame_buf = viewer.get_frame(frame_buf)
        if tactile is not None:
            with st("tactile"):
                scene.solver.update_contacts(scene.contacts, scene.state_0)
                tactile.update(scene.state_0, scene.contacts)
        if keep:
            rows.append(np.concatenate([
                tactile.normal_load.numpy(), tactile.friction_load_abs.numpy(),
                tactile.contact_area.numpy(), tactile.peak_pressure.numpy()])
                if tactile is not None else np.zeros(0))

    scene.reset(args.start)
    for k in range(args.warmup):
        one_frame(k, Stage(fenced=False), keep=False)

    # Pass 1: fenced, to attribute the loop.
    scene.reset(args.start)
    st = Stage(fenced=True)
    wp.synchronize()
    t0 = time.perf_counter()
    for k in range(args.frames):
        one_frame(k, st, keep=False)
    wp.synchronize()
    fenced_total = time.perf_counter() - t0

    # Pass 2: unfenced, which is what a training loop actually gets.
    scene.reset(args.start)
    free = Stage(fenced=False)
    wp.synchronize()
    t0 = time.perf_counter()
    for k in range(args.frames):
        one_frame(k, free, keep=args.dump is not None)
        times.append(time.perf_counter() - t0)
    wp.synchronize()
    total = time.perf_counter() - t0

    n = args.frames
    res = "no-render" if viewer is None else f"{args.width}x{args.height}"
    print(f"\n=== {res}, {n} frames, box_tris={args.box_tris} "
          f"hand_tris={args.hand_tris} ===")
    print(f"{'stage':>10} {'ms/frame':>10} {'fps alone':>11} {'% of loop':>10}")
    order = [k for k in ("policy", "physics", "render", "readout", "tactile")
             if k in st.t]
    for k in order:
        ms = 1e3 * st.t[k] / n
        print(f"{k:>10} {ms:>10.2f} {1e3 / max(ms, 1e-9):>11.1f} "
              f"{100 * st.t[k] / fenced_total:>10.1f}")
    print(f"{'-' * 44}")
    phys = 1e3 * st.t["physics"] / n
    print(f"{'physics':>10} {phys:>10.2f} {1e3 / max(phys, 1e-9):>11.1f}   <- physics FPS")
    if "render" in st.t:
        r = 1e3 * (st.t["render"] + st.t["readout"]) / n
        print(f"{'render':>10} {r:>10.2f} {1e3 / max(r, 1e-9):>11.1f}   "
              f"<- rendering FPS (draw + readout)")
    tot_ms = 1e3 * total / n
    print(f"{'FULL LOOP':>10} {tot_ms:>10.2f} {1e3 / max(tot_ms, 1e-9):>11.1f}   "
          f"<- full FPS, unfenced")
    print(f"(fenced sum {1e3 * fenced_total / n:.2f} ms/frame; fences cost "
          f"{1e3 * (fenced_total - total) / n:+.2f} ms)")
    if tactile is not None:
        print(f"tactile rows: {tactile.num_patches} patches, "
              f"normal load {np.round(tactile.normal_load.numpy(), 3)}")

    if args.dump:
        np.savez_compressed(args.dump, rows=np.array(rows), times=np.array(times),
                            stage=np.array(list(st.t.keys())),
                            stage_ms=np.array([1e3 * st.t[k] / n for k in st.t]),
                            total_ms=tot_ms, res=np.array(res))
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
