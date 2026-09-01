"""Scene render beside both tactile maps, with measured physics / render / full FPS burned in.

The two hand panels are the G1 rubber-hand collider drawn flat on its palm (see
``hand_atlas``), which is the geometry Newton actually collides against.

Runs the actual loop -- policy, physics, ViewerGL render, GPU frame readout, contact readback
-- and times each stage per frame. The compositing that produces this video is deliberately
OUTSIDE those timers: drawing a matplotlib panel per frame costs more than the simulation
does, and including it would report a frame rate no training run would ever see.

The FPS shown are therefore what the loop sustains without visualisation, which is the number
that matters for a policy -> physics+render -> sensing -> policy setup.

    python -m sugar_newton.validation.make_loop_video --frames 300 --width 960 --height 720
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from sugar_newton.validation.g1_carrybox_policy import (
    Actor,
    G1PolicyScene,
    load_clip,
)
from sugar_newton.validation.hand_atlas import CANVAS_TRIS, HandAtlas, load_hands, palm_sign


def hand_shapes(scene):
    body_of = scene.model.shape_body.numpy()
    labels = [l.split("/")[-1] for l in scene.model.body_label]
    hands, box = {}, set()
    for s, b in enumerate(body_of):
        if b < 0:
            continue
        if labels[b] == "box":
            box.add(s)
        for side in ("left", "right"):
            if labels[b] == f"{side}_rubber_hand":
                hands.setdefault(side, set()).add(s)
    return hands, box


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data_000")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--start", type=int, default=60)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--box-tris", type=int, default=2000)
    ap.add_argument("--hand-tris", type=int, default=5000)
    ap.add_argument("--canvas", type=int, default=CANVAS_TRIS)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default="sugar_newton/_out/loop_render_tactile.gif")
    args = ap.parse_args()

    wp.init()
    clip = load_clip(args.clip)
    dt = 1.0 / clip["fps"]
    scene = G1PolicyScene(clip, box_tris=args.box_tris, hand_tris=args.hand_tris)
    actor = Actor()
    hands, box = hand_shapes(scene)
    sides = [s for s in ("left", "right") if s in hands]

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
        cam = mid + np.array([1.7, -1.7, 0.45])
        d = np.array([mid[0], mid[1], mid[2] + 0.15]) - cam
        d /= max(np.linalg.norm(d), 1e-9)
        viewer.set_camera(wp.vec3(*cam.tolist()),
                          math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0)))),
                          math.degrees(math.atan2(float(d[1]), float(d[0]))))

    canvases = load_hands(args.canvas, tuple(sides))

    scene.reset(args.start)
    for _ in range(8):                      # warm up kernels before timing
        scene.apply(actor(scene.observe()))
        scene.step(dt, args.substeps, "step")
    scene.reset(args.start)

    buf = None
    frames, t_phys, t_rend, t_full = [], [], [], []
    nets, loads = [], []
    raw = {s: [] for s in sides}
    for k in range(args.frames):
        wp.synchronize()
        t0 = time.perf_counter()
        scene.apply(actor(scene.observe()))
        scene.step(dt, args.substeps, "step")
        wp.synchronize()
        t1 = time.perf_counter()
        aim()
        viewer.begin_frame(k * dt)
        viewer.log_state(scene.state_0)
        viewer.end_frame()
        buf = viewer.get_frame(buf)
        wp.synchronize()
        t2 = time.perf_counter()
        scene.solver.update_contacts(scene.contacts, scene.state_0)
        c = scene.contacts
        n = int(c.rigid_contact_count.numpy()[0])
        frame = {s: (np.zeros((0, 3)), np.zeros(0)) for s in sides}
        net, load = 0.0, 0
        if n:
            s0 = c.rigid_contact_shape0.numpy()[:n]
            s1 = c.rigid_contact_shape1.numpy()[:n]
            f = c.force.numpy()[:n, :3]
            p0 = c.rigid_contact_point0.numpy()[:n]
            p1 = c.rigid_contact_point1.numpy()[:n]
            for side in sides:
                patch = hands[side]
                # point0 is local to shape0 and point1 to shape1, so pick whichever of the
                # pair is the hand -- otherwise half the contacts land in the box's frame.
                h0 = np.array([a in patch and b in box for a, b in zip(s0, s1)])
                h1 = np.array([b in patch and a in box for a, b in zip(s0, s1)])
                sel = h0 | h1
                if not sel.any():
                    continue
                mag = np.linalg.norm(f[sel], axis=1)
                frame[side] = (np.where(h0[sel, None], p0[sel], p1[sel]), mag)
                net += float(np.linalg.norm(f[sel].sum(0)))
                load += int((mag > 0.01).sum())
        nets.append(net)
        loads.append(load)
        wp.synchronize()
        t3 = time.perf_counter()

        t_phys.append(t1 - t0)
        t_rend.append(t2 - t1)
        t_full.append(t3 - t0)
        frames.append(buf.numpy().copy())
        for side in sides:
            raw[side].append(frame[side])

    print(f"loop: physics {1e3 * np.mean(t_phys):.1f} ms, render {1e3 * np.mean(t_rend):.1f} ms, "
          f"sensing {1e3 * (np.mean(t_full) - np.mean(t_phys) - np.mean(t_rend)):.1f} ms, "
          f"full {1e3 * np.mean(t_full):.1f} ms ({1 / np.mean(t_full):.1f} fps)")

    # ---- compositing (untimed) ----
    atlases, maps = {}, {}
    for side in sides:
        _, _, cv, ct = canvases[side]
        allp = np.concatenate([p for p, _ in raw[side]]) if raw[side] else np.zeros((0, 3))
        allm = np.concatenate([m for _, m in raw[side]]) if raw[side] else np.zeros(0)
        atlases[side] = HandAtlas(cv, ct, palm_sign(cv, allp, allm))
        maps[side] = np.array([atlases[side].splat(p, m) for p, m in raw[side]])

    stack = np.concatenate([m for m in maps.values()])
    vmax = float(np.percentile(stack[stack > 1e-6], 99.0)) if (stack > 1e-6).any() else 1.0
    norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax or 1e-9)

    fig = plt.figure(figsize=(16.8, 6.6))
    axl = fig.add_axes([0.004, 0.02, 0.545, 0.87])
    axl.set_axis_off()
    im = axl.imshow(frames[0])
    pcs = {}
    for i, side in enumerate(sides):
        ax = fig.add_axes([0.565 + 0.185 * i, 0.04, 0.175, 0.80])
        pcs[side] = atlases[side].draw(ax, maps[side][0], norm)
        atlases[side].label_digits(ax, fontsize=7.5)
        ax.set_title(f"{side} hand", fontsize=12)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="inferno"),
                 cax=fig.add_axes([0.945, 0.12, 0.013, 0.62]),
                 label="contact force this frame [N]")
    fig.text(0.655, 0.90, "G1 rubber-hand collider, palm view, fingertips up",
             ha="center", fontsize=11, color="#444444")
    sup = fig.text(0.5, 0.965, "", ha="center", fontsize=13)
    hud = fig.text(0.012, 0.985, "", ha="left", va="top", fontsize=11,
                   family="monospace",
                   bbox=dict(boxstyle="round,pad=0.35", fc="#ffffffcc", ec="#999999"))

    # Running means, so the HUD reads as a rate rather than jittering per frame.
    def rate(v, k, win=20):
        lo = max(0, k - win + 1)
        return 1.0 / max(float(np.mean(v[lo:k + 1])), 1e-9)

    import imageio.v2 as imageio
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The gif and ffmpeg writers take different rate arguments, and ffmpeg is not always
    # installed in the container, so fall back rather than lose the run.
    try:
        if out.suffix == ".gif":
            raise ValueError("gif")
        writer = imageio.get_writer(str(out), mode="I", fps=args.fps,
                                    macro_block_size=1, quality=8)
    except Exception:
        out = out.with_suffix(".gif")
        writer = imageio.get_writer(str(out), mode="I", duration=1.0 / args.fps, loop=0)
    ntri = len(canvases[sides[0]][3])
    for k in range(args.frames):
        im.set_data(frames[k])
        tot = 0.0
        for side in sides:
            atlases[side].paint(pcs[side], maps[side][k], norm)
            tot += float(maps[side][k].sum())
        sup.set_text(f"G1 carry, clip {args.clip} frame {args.start + k}   |   "
                     f"both hands: {loads[k]} load-bearing contacts, "
                     f"sum|f| {tot:.0f} N, net {nets[k]:.0f} N")
        hud.set_text(f"physics {rate(t_phys, k):6.1f} fps  ({1e3 * t_phys[k]:5.1f} ms)\n"
                     f"render  {rate(t_rend, k):6.1f} fps  ({1e3 * t_rend[k]:5.1f} ms)\n"
                     f"FULL    {rate(t_full, k):6.1f} fps  ({1e3 * t_full[k]:5.1f} ms)\n"
                     f"{args.width}x{args.height}, {ntri}-tri tactile canvas")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(img)
    writer.close()
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
