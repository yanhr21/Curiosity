# SPDX-License-Identifier: BSD-3-Clause
"""The tactile field drawn on the hand itself, in a canonical frame that never moves.

The earlier composite projected contact points onto whatever plane best fitted them and
called that a map. It answers "how much" but not "where on the hand" -- the canvas moved
with the data, so the same pad landed somewhere different from frame to frame.

Here the canvas *is* the hand. Every pad has a fixed, known place: ``PATCH_SPECS`` gives
each of the 27 patches a centre, a size and a tangent angle in the rubber hand's own mesh
frame, which is exactly the frame :class:`~sugar_newton.tactile.ContactField` expresses
its samples in. So a contact-surface face lands inside its own pad rectangle with no
projection and no fitting, and the palm sits in the same place in every frame of every
clip. X runs wrist to fingertip, Z runs little finger to index; the two hands are drawn
side by side.

Three channels, three panels, all on the same canonical hand:

* **pressure** [kPa] -- per-face, scaled to integrate to the solved normal load
* **tangential traction** [kPa] -- magnitude per-face, direction from the patch's own
  measured friction vector, drawn as arrows
* **slip velocity** [mm/s] -- per-face, drawn as arrows: where the box is sliding across
  the skin, and which way

    python -m sugar_newton.validation.compose_hand_canvas --run <dir> --video out.mp4
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from sugar_newton.hand.patches import PATCH_SPECS  # noqa: E402

SURFACE, PANEL = "#12100f", "#1a1a19"
INK, INK_2, INK_MUTED = "#f5f4f0", "#c3c2b7", "#84837a"
PAD_EDGE, PAD_FACE = "#4a4944", "#232320"
ARROW = "#efeee8"

CHANNELS = (
    ("pressure", "contact pressure", "kPa", 1e-3, "#2a78d6",
     ["#12233a", "#184f95", "#2a78d6", "#86b6ef", "#e8f1fd"], None),
    ("traction", "tangential traction", "kPa", 1e-3, "#c25a20",
     ["#2a1206", "#7a3211", "#c25a20", "#eb8c5a", "#fbd9c6"], "traction_vec"),
    ("slip", "slip velocity", "mm/s", 1e3, "#1baf7a",
     ["#0a231b", "#0f5a41", "#1baf7a", "#7fd9b8", "#e3f7ef"], "slip_vec"),
)


def gaussian_matrix(n: int, sigma: float) -> np.ndarray:
    d = np.arange(n)[:, None] - np.arange(n)[None, :]
    k = np.exp(-0.5 * (d / sigma) ** 2)
    return k / k.sum(axis=1, keepdims=True)


def hand_offsets(labels: list[str], gap: float) -> tuple[np.ndarray, list[str]]:
    """Z shift per patch row that puts each hand in its own column of the canvas.

    Both hands' pads are authored at the same coordinates in their own mesh frames -- the
    meshes are mirrored, the specs are not -- so without a shift the two hands would be
    drawn exactly on top of each other.
    """
    sides = []
    for lbl in labels:
        s = str(lbl).split("_")[0]
        if s not in sides:
            sides.append(s)
    shift = {s: i * gap for i, s in enumerate(sides)}
    return np.array([shift[str(lbl).split("_")[0]] for lbl in labels]), sides


def pad_rectangles(ax, sides, gap, zorder=1):
    """Draw the 27 pads of each hand where PATCH_SPECS says they are."""
    for h, side in enumerate(sides):
        for spec in PATCH_SPECS:
            ang = spec.tangent_angle_deg
            # canvas axes are (z -> horizontal, x -> vertical); the spec's tangent angle
            # is about the palm normal, so it rotates the pad within exactly this plane
            r = Rectangle(
                (spec.center_z_m + h * gap - spec.length_m / 2,
                 spec.center_x_m - spec.width_m / 2),
                spec.length_m, spec.width_m,
                angle=-ang, rotation_point="center",
                facecolor=PAD_FACE, edgecolor=PAD_EDGE, linewidth=0.55, zorder=zorder,
            )
            ax.add_patch(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--field", default="g1_carrybox_field.npz")
    ap.add_argument("--trace", default="g1_carrybox.npz")
    ap.add_argument("--video", default=None)
    ap.add_argument("--video-fps", type=float, default=30.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grid", type=int, default=260, help="canvas raster width [cells]")
    ap.add_argument("--sigma", type=float, default=1.8, help="splat width [cells]")
    ap.add_argument("--gap", type=float, default=0.115, help="spacing between hands [m]")
    ap.add_argument("--quiver", type=int, default=34)
    ap.add_argument("--arrow-len", type=float, default=0.06)
    ap.add_argument("--percentile", type=float, default=97.0)
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--dpi", type=float, default=110.0)
    args = ap.parse_args()

    run = Path(args.run)
    f = np.load(run / args.field)
    tac = np.load(run / args.trace) if (run / args.trace).exists() else None
    dt = float(f["dt"])
    off = f["offsets"]
    n = len(off) - 1
    t = np.arange(n) * dt
    labels = [str(x) for x in f["labels"]]
    pos, area, patch = f["pos"], f["area"], f["patch"]
    chan = {k: f[k] for k, *_ in CHANNELS}
    vecs = {vk: f[vk] for *_, vk in CHANNELS if vk and vk in f}

    if len(pos) == 0:
        print("no contact-surface faces were recorded -- the hands never touched the box")
        return 1

    shift, sides = hand_offsets(labels, args.gap)
    # canvas coordinates: horizontal = Z (+ the hand's own shift), vertical = X
    cx = pos[:, 2] + shift[patch]
    cy = pos[:, 0]
    # Vectors are rotated into the canvas the same way -- (z, x), not (x, z), or the
    # arrows would be transposed relative to the field they sit on.
    vec_xy = {k: np.stack([v[:, 2], v[:, 0]], axis=1) for k, v in vecs.items()}

    # extent from the PADS, not from the data: the canvas must not change when the
    # contact does, or it stops being canonical.
    px = [spec.center_x_m for spec in PATCH_SPECS]
    pz = [spec.center_z_m for spec in PATCH_SPECS]
    pw = max(spec.width_m for spec in PATCH_SPECS)
    pl = max(spec.length_m for spec in PATCH_SPECS)
    pad = 0.012
    x0, x1 = min(px) - pw / 2 - pad, max(px) + pw / 2 + pad
    z0 = min(pz) - pl / 2 - pad
    z1 = max(pz) + pl / 2 + pad + (len(sides) - 1) * args.gap
    nx = args.grid
    ny = max(int(round(nx * (x1 - x0) / (z1 - z0))), 8)
    ex = np.linspace(z0, z1, nx + 1)
    ey = np.linspace(x0, x1, ny + 1)
    extent = (z0, z1, x0, x1)

    Gx, Gy = gaussian_matrix(nx, args.sigma), gaussian_matrix(ny, args.sigma)
    support = Gx.diagonal().max() * Gy.diagonal().max() * 0.10

    def rasterise(i: int):
        sl = slice(off[i], off[i + 1])
        if off[i + 1] <= off[i]:
            return {k: None for k, *_ in CHANNELS}, None
        x, y, w = cx[sl], cy[sl], area[sl]
        cnt, _, _ = np.histogram2d(y, x, bins=[ey, ex])
        den, _, _ = np.histogram2d(y, x, bins=[ey, ex], weights=w)
        cnt, den = Gy @ cnt @ Gx.T, Gy @ den @ Gx.T
        good = den > 0
        out = {}
        for key, _ti, _u, mult, _h, _r, _vk in CHANNELS:
            num, _, _ = np.histogram2d(y, x, bins=[ey, ex], weights=w * chan[key][sl] * mult)
            num = Gy @ num @ Gx.T
            out[key] = np.where(good, num / np.where(good, den, 1.0), 0.0).astype(np.float32)
        return out, cnt < support

    nqx = max(int(args.quiver), 0)
    nqy = max(int(round(nqx * (x1 - x0) / (z1 - z0))), 1) if nqx else 0
    if nqx:
        qex, qey = np.linspace(z0, z1, nqx + 1), np.linspace(x0, x1, nqy + 1)
        QX, QY = np.meshgrid(0.5 * (qex[1:] + qex[:-1]), 0.5 * (qey[1:] + qey[:-1]))

    def bin_vectors(i: int, vkey: str, mult: float):
        sl = slice(off[i], off[i + 1])
        if not nqx or off[i + 1] <= off[i]:
            return None, None, None
        x, y, w = cx[sl], cy[sl], area[sl]
        v = vec_xy[vkey][sl] * mult
        wsum, _, _ = np.histogram2d(y, x, bins=[qey, qex], weights=w)
        u, _, _ = np.histogram2d(y, x, bins=[qey, qex], weights=w * v[:, 0])
        vv, _, _ = np.histogram2d(y, x, bins=[qey, qex], weights=w * v[:, 1])
        g = wsum > 0
        return (np.where(g, u / np.where(g, wsum, 1.0), 0.0),
                np.where(g, vv / np.where(g, wsum, 1.0), 0.0), g)

    print("rasterising...", flush=True)
    cache, traces = {}, {k: np.zeros(n) for k, *_ in CHANNELS}
    for i in range(n):
        fields, mask = rasterise(i)
        cache[i] = (fields, mask)
        for key, *_ in CHANNELS:
            if fields[key] is not None and mask is not None and (~mask).any():
                traces[key][i] = float(fields[key][~mask].max())

    s0 = min(int(args.settle / dt), n - 1)
    # Scale off the frames that HAVE contact. A grasp touches down for a fraction of a
    # clip -- 7 frames in 250 here -- so a percentile over all frames is a percentile over
    # zeros, and every colour bar ends up reading 1e-9.
    def _scale(v: np.ndarray) -> float:
        nz = v[s0:][v[s0:] > 0]
        if not nz.size:
            nz = v[v > 0]
        return max(float(np.percentile(nz, args.percentile)) if nz.size else 1.0, 1e-9)

    scales = {k: _scale(traces[k]) for k, *_ in CHANNELS}
    arrow_ref, qcache = {}, {}
    for _k, _ti, _u, mult, _h, _r, vkey in CHANNELS:
        if not vkey or not nqx:
            continue
        mags = []
        for i in range(n):
            u, v, g = bin_vectors(i, vkey, mult)
            qcache[(i, vkey)] = (u, v, g)
            if g is not None and g.any():
                mags.append(float(np.hypot(u[g], v[g]).max()))
        arrow_ref[vkey] = max(float(np.median(mags)) if mags else 1.0, 1e-9)

    cmaps = {k: LinearSegmentedColormap.from_list(k, ramp)
             for k, _ti, _u, _m, _h, ramp, _vk in CHANNELS}
    print(f"{n} frames, {len(pos)} faces on {len(sides)} hand(s), canvas "
          f"{(z1 - z0) * 1e3:.0f} x {(x1 - x0) * 1e3:.0f} mm, scales: "
          + ", ".join(f"{k}<={scales[k]:.3g}" for k in scales), flush=True)

    todo = list(range(0, n if args.limit <= 0 else min(n, args.limit), args.stride))
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1.0], left=0.055, right=0.985,
                          top=0.875, bottom=0.075, wspace=0.20, hspace=0.30)
    panels = []
    for c, (key, title, unit, mult, hue, _ramp, vkey) in enumerate(CHANNELS):
        a = fig.add_subplot(gs[0, c])
        a.set_facecolor(PANEL)
        a.set_xlim(z0, z1); a.set_ylim(x0, x1)
        # box aspect, not data aspect: mm stay square because the axes box is given the
        # canvas's own ratio, and the panel fills its cell instead of letterboxing.
        a.set_box_aspect((x1 - x0) / (z1 - z0))
        pad_rectangles(a, sides, args.gap, zorder=1)
        norm = Normalize(0.0, scales[key])
        im = a.imshow(np.ma.masked_all((ny, nx)), origin="lower", extent=extent,
                      cmap=cmaps[key], norm=norm, interpolation="bilinear", zorder=3)
        qh = ql = None
        if vkey and nqx:
            z = np.zeros(QX.size)
            sc = arrow_ref[vkey] / (args.arrow_len * (z1 - z0))
            qh = a.quiver(QX.ravel(), QY.ravel(), z, z.copy(), angles="xy",
                          scale_units="xy", scale=sc, color=SURFACE, width=0.011,
                          headwidth=2.6, headlength=2.9, headaxislength=2.4, zorder=4)
            ql = a.quiver(QX.ravel(), QY.ravel(), z.copy(), z.copy(), angles="xy",
                          scale_units="xy", scale=sc, color=ARROW, width=0.0055,
                          headwidth=4.0, headlength=4.4, headaxislength=3.8, zorder=5)
            if c == 1:
                a.quiverkey(ql, 0.80, 0.03, arrow_ref[vkey], f"{arrow_ref[vkey]:.3g} {unit}",
                            labelpos="E", color=ARROW, labelcolor=INK_2,
                            fontproperties={"size": 7.0}, coordinates="axes")
        for h, side in enumerate(sides):
            a.text(np.mean([s.center_z_m for s in PATCH_SPECS]) + h * args.gap, x1 - 0.004,
                   side, ha="center", va="top", fontsize=8.5, color=INK_2, zorder=6)
        a.set_xticks([]); a.set_yticks([])
        for sp in ("top", "right", "left", "bottom"):
            a.spines[sp].set_color(INK_MUTED); a.spines[sp].set_linewidth(0.6)
        a.set_title(title, color=INK, fontsize=10.5, loc="left", pad=16)
        sub = f"hand frame · [{unit}]" + (" · arrows: direction + magnitude" if vkey else "")
        a.text(0.0, 1.006, sub, transform=a.transAxes, fontsize=8, color=INK_MUTED,
               va="bottom")
        if c == 0:
            a.plot([z0 + 0.006, z0 + 0.006 + 0.02], [x0 + 0.006] * 2, color=INK_2, lw=1.6,
                   zorder=6)
            a.text(z0 + 0.016, x0 + 0.009, "20 mm", ha="center", va="bottom", fontsize=7.5,
                   color=INK_2, zorder=6)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmaps[key]), ax=a,
                          fraction=0.036, pad=0.015)
        cb.set_ticks(np.linspace(0.0, scales[key], 5))
        cb.ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.3g}")
        cb.ax.tick_params(colors=INK_2, labelsize=7.5, length=2.5, width=0.7)
        cb.outline.set_visible(False)

        b = fig.add_subplot(gs[1, c]); b.set_facecolor(PANEL)
        for sp in ("top", "right"):
            b.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            b.spines[sp].set_color(INK_MUTED); b.spines[sp].set_linewidth(0.7)
        b.tick_params(colors=INK_2, labelsize=7.5, length=2.5)
        b.grid(True, axis="y", color=INK_MUTED, alpha=0.18, linewidth=0.5)
        b.set_axisbelow(True)
        b.plot(t, traces[key], color=hue, lw=1.4)
        b.fill_between(t, 0, traces[key], color=hue, alpha=0.16, lw=0)
        cursor = b.axvline(t[0], color=INK_2, lw=0.9, alpha=0.75)
        top = max(traces[key].max() * 1.08, scales[key] * 1.2, 1e-9)
        b.set_ylim(0, top)
        b.set_xticks(np.linspace(t[0], t[-1], 5))
        b.set_yticks(np.linspace(0, top, 5))
        b.xaxis.set_major_formatter(lambda v, _p: f"{v:.1f}")
        b.yaxis.set_major_formatter(lambda v, _p: f"{v:.3g}")
        b.set_ylabel(f"peak  [{unit}]", color=INK_2, fontsize=8)
        b.set_xlabel("time [s]", color=INK_2, fontsize=8)
        panels.append(dict(key=key, vkey=vkey, im=im, qh=qh, ql=ql, cursor=cursor))

    fig.suptitle("SUGAR CarryBox — the skin, on the hand", color=INK, fontsize=14,
                 x=0.03, ha="left", y=0.965)
    header = fig.text(0.03, 0.918, "", color=INK_MUTED, fontsize=9.5, ha="left",
                      family="monospace")

    proc, vw, vh = None, 0, 0
    if args.video:
        fig.canvas.draw()
        vw, vh = fig.canvas.get_width_height()
        vw, vh = vw - vw % 2, vh - vh % 2
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "rgba",
             "-s", f"{vw}x{vh}", "-r", str(args.video_fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(args.video)],
            stdin=subprocess.PIPE)
    out_dir = run / "hand_canvas"
    if proc is None:
        out_dir.mkdir(exist_ok=True)

    t_start = time.perf_counter()
    for k, i in enumerate(todo):
        fields, mask = cache[i]
        for pan in panels:
            fld = fields[pan["key"]]
            pan["im"].set_data(np.ma.masked_all((ny, nx)) if fld is None
                               else np.ma.masked_where(mask, fld))
            vkey = pan["vkey"]
            if vkey and nqx:
                u, v, g = qcache.get((i, vkey), (None, None, None))
                if g is None:
                    u = v = np.full((nqy, nqx), np.nan)
                else:
                    mag = np.hypot(u, v)
                    keep = g & (mag > 0.02 * arrow_ref[vkey])
                    shrink = np.where(mag > 3.0 * arrow_ref[vkey],
                                      3.0 * arrow_ref[vkey] / np.maximum(mag, 1e-12), 1.0)
                    u = np.where(keep, u * shrink, np.nan)
                    v = np.where(keep, v * shrink, np.nan)
                pan["qh"].set_UVC(u.ravel(), v.ravel())
                pan["ql"].set_UVC(u.ravel(), v.ravel())
            pan["cursor"].set_xdata([t[i], t[i]])
        nf = int(off[i + 1] - off[i])
        live = int(len(np.unique(patch[off[i]:off[i + 1]]))) if nf else 0
        extra = ""
        if tac is not None:
            extra = (f"   ·   normal load {tac['normal_load'][i].sum():6.2f} N"
                     f"   ·   box z {tac['box_q'][i, 2]:5.3f} m")
        header.set_text(f"t = {t[i]:5.2f} s   ·   {nf:5d} contact-surface faces   ·   "
                        f"{live:2d}/{len(labels)} pads loaded{extra}")
        if proc is not None:
            fig.canvas.draw()
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:vh, :vw].tobytes())
        else:
            fig.savefig(out_dir / f"h{i:05d}.png", dpi=args.dpi, facecolor=SURFACE)
        if k and k % 25 == 0:
            print(f"  composed {k}/{len(todo)} "
                  f"({k / (time.perf_counter() - t_start):.1f} frames/s)", flush=True)
    if proc is not None:
        proc.stdin.close(); proc.wait()
    plt.close(fig)
    print(f"wrote {args.video if args.video else out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
