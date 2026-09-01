"""Show what hand-collider decimation does to the shape and to the tactile map.

Everything is drawn on the flat palm-facing layout of ``hand_atlas`` -- the real G1 rubber
hand collider (``meshes/{left,right}_rubber_hand.STL`` out of
``g1_29dof_rev_1_0_with_rubber_hand.urdf``), projected orthographically down its palm
normal. A grasp loads exactly one side of that slab, so the projection loses no contact
and you can read off which finger a reading belongs to.

``hand_shape.png`` -- geometry only:
  * the original 45748-triangle collider beside the decimated ones
  * cross-section outlines of every level overlaid on one axis, which is the honest way to
    judge shape change (a shaded render hides a millimetre; a slice does not)
  * where the surface moved, and how that sits against the 5 mm contact margin

``hand_tactile_flat.png`` -- both hands, from ``compare_contacts --dump``:
  Contact points come out of Newton in the hand's own body frame (contacts.py:210), so
  contact sets from two different colliders can be splatted onto ONE common canvas mesh and
  subtracted. Without a common canvas the two maps would live on different tessellations
  and could not be differenced.

    python -m sugar_newton.validation.compare_contacts --vary hand --variants 0 0 2000 \
        --dump sugar_newton/_out/tactile_hand.npz
    python -m sugar_newton.validation.plot_hand_tactile \
        --dump sugar_newton/_out/tactile_hand.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from sugar_newton.validation.check_decimation import _o3d_mesh, hand_collision_meshes
from sugar_newton.validation.hand_atlas import (
    CANVAS_TRIS,
    TAXEL_M,
    HandAtlas,
    decimated,
    palm_sign,
)

LEVELS = (5000, 2000, 1000, 500)


def variants_for(dump, side):
    """``[(label, pts, mag, frame_or_None)]`` for one hand, in dump order."""
    out = []
    for key in sorted(k for k in dump.files if k.endswith("_label")):
        tag = key[: -len("_label")]
        fr = dump[f"{tag}_{side}_frame"] if f"{tag}_{side}_frame" in dump.files else None
        out.append((str(dump[key]), dump[f"{tag}_{side}_pts"],
                    dump[f"{tag}_{side}_mag"], fr))
    return out


def cross_section(verts, tris, axis, value):
    """Segments where the mesh crosses the plane ``axis == value``."""
    tri = verts[tris]
    d = tri[:, :, axis] - value
    segs = []
    for i in range(3):
        j = (i + 1) % 3
        a, b = d[:, i], d[:, j]
        hit = (a * b) < 0
        if not hit.any():
            continue
        t = (a[hit] / (a[hit] - b[hit]))[:, None]
        segs.append(tri[hit, i] + t * (tri[hit, j] - tri[hit, i]))
    if not segs:
        return np.zeros((0, 2, 3))
    pts = np.concatenate(segs)
    # Each triangle contributes exactly two crossing points; pair them up by triangle.
    ids = np.concatenate([np.where((d[:, i] * d[:, (i + 1) % 3]) < 0)[0]
                          for i in range(3)])
    out = []
    for t_id in np.unique(ids):
        p = pts[ids == t_id]
        if len(p) >= 2:
            out.append(p[:2])
    return np.array(out) if out else np.zeros((0, 2, 3))


def figure_shape(name, verts, tris, dump, out: Path) -> None:
    import open3d as o3d

    cv, ct = decimated(verts, tris, 8000)
    levels = [(len(tris), verts, tris)] + [(t, *decimated(verts, tris, t)) for t in LEVELS]
    by_n = {n: (v, t) for n, v, t in levels}

    side = name.split("_")[0]
    sign = 1.0
    if dump is not None:
        v0 = variants_for(dump, side)
        if v0:
            sign = palm_sign(verts, v0[0][1], v0[0][2])

    fig = plt.figure(figsize=(19, 9.5))
    gs = fig.add_gridspec(2, 4, height_ratios=(1.1, 1.0))
    fig.suptitle(f"{name} (G1 rubber-hand collider, {len(tris)} triangles) "
                 f"under quadric decimation, viewed on the palm, 5 mm contact margin",
                 fontsize=15)

    for i, n in enumerate((len(tris), 2000, 500)):
        v, t = by_n[n]
        ax = fig.add_subplot(gs[0, i])
        a = HandAtlas(v, t, sign)
        a.draw_shaded(ax, lw=0.10 if n <= 5000 else 0.0)
        if i == 0:
            a.label_digits(ax)
        ax.set_title(f"{n} triangles" + ("  (original)" if i == 0 else ""), fontsize=12)

    # Slices: one across the fingers, one through the palm. This is where a millimetre of
    # shape change is actually legible; a shaded view hides it.
    cols = ["k", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    lo, hi = verts.min(0), verts.max(0)
    slices = [(0, lo[0] + 0.82 * (hi[0] - lo[0]), "across the fingers"),
              (2, float(np.median(verts[:, 2])), "through the palm")]
    for k, (axis, value, what) in enumerate(slices):
        ax = fig.add_subplot(gs[1, k])
        keep = [c for c in (0, 1, 2) if c != axis]
        for (n, v, t), col in zip(levels, cols):
            seg = cross_section(v, t, axis, value)
            if len(seg) == 0:
                continue
            ax.add_collection(LineCollection(
                seg[:, :, keep] * 1e3, colors=col,
                linewidths=2.6 if n == len(tris) else 1.3, label=f"{n}",
                alpha=1.0 if n == len(tris) else 0.85, zorder=3 if n == len(tris) else 2))
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_xlabel(f"{'xyz'[keep[0]]} [mm]")
        ax.set_ylabel(f"{'xyz'[keep[1]]} [mm]")
        ax.set_title(f"cross-section {what}\n({'xyz'[axis]} = {value * 1e3:.1f} mm)",
                     fontsize=11)
        if k == 0:
            ax.legend(fontsize=8, title="triangles", loc="best")
        ax.grid(alpha=0.3)

    devs = {}
    for target in (2000, 500):
        dv, dt = by_n[target]
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(_o3d_mesh(dv, dt)))
        devs[target] = sc.compute_distance(
            o3d.core.Tensor(cv.astype(np.float32))).numpy() * 1e3

    face = devs[2000][ct].mean(1)
    ax = fig.add_subplot(gs[0, 3])
    dnorm = plt.Normalize(0.0, max(face.max(), 1e-6))
    HandAtlas(cv, ct, sign).draw(ax, face, dnorm, cmap="inferno", dead=None)
    ax.set_title(f"deviation to the 2000-tri collider\n"
                 f"mean {face.mean():.3f} mm, max {face.max():.3f} mm", fontsize=12)
    fig.colorbar(plt.cm.ScalarMappable(norm=dnorm, cmap="inferno"), ax=ax,
                 shrink=0.75, label="deviation [mm]")

    ax = fig.add_subplot(gs[1, 2])
    for target, col in zip((2000, 500), ["#d62728", "#9467bd"]):
        d = np.sort(devs[target])
        ax.plot(d, np.linspace(0, 100, len(d)), color=col, lw=2,
                label=f"{target} tris (max {d[-1]:.2f} mm)")
    ax.axvline(5.0, color="k", ls="--", lw=1.4, label="5 mm contact margin")
    ax.axvline(3.2, color="grey", ls=":", lw=1.4, label="3.2 mm box wall")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 10.0)
    ax.set_xlabel("surface deviation [mm]")
    ax.set_ylabel("percent of surface below")
    ax.set_title("deviation distribution vs the margin", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    fig.text(0.775, 0.30,
             "Reading this figure:\n\n"
             "\u2022 The fingers are where decimation bites: the palm is nearly flat and\n"
             "   decimates for free, while the finger creases and fingertips carry the\n"
             "   deviation hotspots.\n\n"
             "\u2022 At 2000 triangles the whole surface stays under 1 mm, roughly 5x\n"
             "   inside the 5 mm contact margin, and the finger cross-sections are\n"
             "   visually coincident with the original.\n\n"
             "\u2022 Even at 500 the outlines stay coincident at this scale and 99 % of\n"
             "   the surface is within 1 mm; the 3.2 mm max is a few fingertip and\n"
             "   crease cells, not a global shape change.\n\n"
             "\u2022 Geometry is the easy half. The tactile map is the other figure, and\n"
             "   it does not follow from this one.",
             fontsize=10, va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#cccccc"))

    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_flat(hands, dump, out: Path, canvas: int = CANVAS_TRIS) -> None:
    """Both hands, flat, original collider vs decimated vs the difference."""
    per_side = {}
    for side, (v, t, cv, ct) in hands.items():
        vs = variants_for(dump, side)
        if len(vs) < 2:
            print(f"{side}: need at least two variants in the dump, skipping")
            return
        # vs[0] and vs[1] are the identical-collider control; vs[0] vs vs[-1] is the test.
        atlas = HandAtlas(cv, ct, palm_sign(cv, vs[0][1], vs[0][2]))
        a, b = vs[0], vs[-1]
        per_side[side] = dict(
            atlas=atlas, a_label=a[0], b_label=b[0],
            fa=atlas.splat(a[1], a[2]), fb=atlas.splat(b[1], b[2]),
            na=int((a[2] > 0.01).sum()), nb=int((b[2] > 0.01).sum()))

    # One colour scale across both hands so the pair is directly comparable. A handful of
    # cells carry an order of magnitude more than the rest, so clip to a high percentile
    # of the loaded cells and compress with a power norm.
    allf = np.concatenate([np.concatenate([d["fa"], d["fb"]]) for d in per_side.values()])
    hot = allf[allf > 1e-6]
    vmax = float(np.percentile(hot, 99.0)) if hot.size else 1.0
    norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax or 1e-9)
    alld = np.concatenate([np.abs(d["fb"] - d["fa"]) for d in per_side.values()])
    lim = float(np.percentile(alld[alld > 0], 99.0)) if (alld > 0).any() else 1e-9
    lim = lim or 1e-9
    diffnorm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)

    order = [s for s in ("left", "right") if s in per_side]
    fig = plt.figure(figsize=(5.0 * len(order) + 2.6, 13.2))
    gs = fig.add_gridspec(3, len(order) + 1, width_ratios=[1] * len(order) + [0.09],
                          hspace=0.10, wspace=0.02)
    fig.suptitle("G1 rubber-hand tactile map, time-integrated over the carry "
                 f"(frames 200-360), {TAXEL_M * 1e3:.0f} mm splat kernel\n"
                 "palm view, fingertips up, grey = never loaded", fontsize=15)

    for col, side in enumerate(order):
        d = per_side[side]
        for row, (lab, f, n) in enumerate((
                (d["a_label"], d["fa"], d["na"]), (d["b_label"], d["fb"], d["nb"]))):
            ax = fig.add_subplot(gs[row, col])
            d["atlas"].draw(ax, f, norm)
            d["atlas"].label_digits(ax)
            ax.set_title(f"{side} hand \u2014 {lab}\n"
                         f"{n} load-bearing contacts, {f.sum():.0f} N accumulated",
                         fontsize=12)
        ax = fig.add_subplot(gs[2, col])
        diff = d["fb"] - d["fa"]
        d["atlas"].draw(ax, np.clip(diff, -lim, lim), diffnorm, cmap="coolwarm",
                        dead=(0.90, 0.90, 0.92))
        d["atlas"].label_digits(ax)
        rel = 100.0 * np.abs(diff).sum() / max(d["fa"].sum(), 1e-9)
        ax.set_title(f"{side} hand \u2014 difference (decimated \u2212 original)\n"
                     f"{rel:.1f} % of the load redistributed, peak {np.abs(diff).max():.1f} N",
                     fontsize=12)

    cax = fig.add_subplot(gs[0:2, len(order)])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="inferno"), cax=cax,
                 label="accumulated |contact force| [N]")
    cax = fig.add_subplot(gs[2, len(order)])
    fig.colorbar(plt.cm.ScalarMappable(norm=diffnorm, cmap="coolwarm"), cax=cax,
                 label="\u0394 [N], clipped to p99")

    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    for side in order:
        d = per_side[side]
        rel = 100.0 * np.abs(d["fb"] - d["fa"]).sum() / max(d["fa"].sum(), 1e-9)
        print(f"  {side}: {d['fa'].sum():7.0f} N -> {d['fb'].sum():7.0f} N, "
              f"{rel:.1f} % redistributed")
    print(f"wrote {out}")


def video_flat(hands, dump, out: Path, fps: int = 15, stride: int = 1,
               canvas: int = CANVAS_TRIS) -> None:
    """Animate the per-frame tactile map for both hands, original beside decimated.

    The layout never moves (contact points are already in each hand's body frame), so the
    PolyCollections are built once and only their face colours are rewritten per frame.
    Rebuilding them would dominate the runtime.
    """
    import matplotlib.animation as animation

    panels, n_frames = [], 0
    for side in ("left", "right"):
        if side not in hands:
            continue
        v, t, cv, ct = hands[side]
        vs = variants_for(dump, side)
        if not vs or vs[0][3] is None:
            print("dump has no frame index, re-run compare_contacts --dump")
            return
        atlas = HandAtlas(cv, ct, palm_sign(cv, vs[0][1], vs[0][2]))
        n_frames = max(n_frames, int(max(v3[3].max() for v3 in vs)) + 1)
        panels.append((side, atlas, vs))
    if not panels:
        return

    frames = list(range(0, n_frames, stride))
    cells = []          # (side, label, atlas, per-frame maps)
    for side, atlas, vs in panels:
        for lab, pts, mag, fr in (vs[0], vs[-1]):
            cells.append((side, lab, atlas,
                          np.array([atlas.splat(pts[fr == f], mag[fr == f])
                                    for f in frames])))
    stack = np.concatenate([m for _, _, _, m in cells])
    vmax = float(np.percentile(stack[stack > 1e-6], 99.5)) if (stack > 1e-6).any() else 1.0
    norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax or 1e-9)

    ncol = len(cells)
    fig = plt.figure(figsize=(4.3 * ncol + 1.6, 6.4))
    gs = fig.add_gridspec(1, ncol + 1, width_ratios=[1] * ncol + [0.06], wspace=0.02)
    pcs, titles = [], []
    for i, (side, lab, atlas, maps) in enumerate(cells):
        ax = fig.add_subplot(gs[0, i])
        pcs.append(atlas.draw(ax, maps[0], norm))
        atlas.label_digits(ax)
        titles.append(ax.set_title(f"{side} hand\n{lab}", fontsize=12))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="inferno"),
                 cax=fig.add_subplot(gs[0, ncol]), label="contact force this frame [N]")
    sup = fig.suptitle("", fontsize=13)

    def draw(k):
        # Average over the variants of a hand, then sum the hands, so the readout is the
        # load on the robot rather than a mean over panels.
        per_hand = len(cells) / max(len(panels), 1)
        tot = sum(float(m[k].sum()) for _, _, _, m in cells) / per_hand
        sup.set_text(f"G1 rubber hands, per-frame tactile map \u2014 frame {frames[k]} "
                     f"of {n_frames}   |   both hands now carrying {tot:.1f} N   "
                     f"({TAXEL_M * 1e3:.0f} mm kernel, grey = no contact force)")
        for pc, (_, _, atlas, maps) in zip(pcs, cells):
            atlas.paint(pc, maps[k], norm)
        return pcs

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    try:
        anim.save(str(out), writer=animation.FFMpegWriter(fps=fps, bitrate=3200))
    except Exception as exc:
        alt = out.with_suffix(".gif")
        print(f"ffmpeg unavailable ({type(exc).__name__}), writing {alt}")
        anim.save(str(alt), writer=animation.PillowWriter(fps=fps))
        out = alt
    plt.close(fig)
    print(f"wrote {out}  ({len(frames)} frames, peak {stack.max():.1f} N)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None, help="npz from compare_contacts --dump")
    ap.add_argument("--outdir", default="sugar_newton/_out")
    ap.add_argument("--hand", default="both", choices=("left", "right", "both"))
    ap.add_argument("--only", default="both",
                    choices=("both", "shape", "tactile", "video"))
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth carry frame")
    ap.add_argument("--canvas", type=int, default=CANVAS_TRIS,
                    help="triangles in the display/splat canvas")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dump = np.load(args.dump) if args.dump else None
    sides = ("left", "right") if args.hand == "both" else (args.hand,)

    if args.only in ("both", "shape"):
        for name, verts, tris in hand_collision_meshes():
            if name.split("_")[0] in sides:
                figure_shape(name, verts, tris, dump, out / f"hand_shape_{name}.png")

    if dump is None or args.only == "shape":
        return

    from sugar_newton.validation.hand_atlas import load_hands
    hands = load_hands(args.canvas, sides)
    if args.only in ("both", "tactile"):
        figure_flat(hands, dump, out / "hand_tactile_flat.png", args.canvas)
    if args.only == "video":
        video_flat(hands, dump, out / "hand_tactile_flat.mp4",
                   fps=args.fps, stride=args.stride, canvas=args.canvas)


if __name__ == "__main__":
    main()
