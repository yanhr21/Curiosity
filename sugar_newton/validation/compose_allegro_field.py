# SPDX-License-Identifier: BSD-3-Clause
"""Compose the Allegro tactile video as a continuous field, not a table of numbers.

The previous composite drew one rounded box per link with the channel value printed
inside it. That is a legend for a hand, not a tactile map: eighteen numbers cannot show
where on a pad the load sits, that pressure concentrates on an edge, or that one end of
a fingertip is slipping while the other is stuck.

Here every contact-surface triangle is its own sample (see
:mod:`sugar_newton.tactile.field`), positioned in the palm's frame so the map holds still
while the fingers move, and resolved into a continuous field by area-weighted Gaussian
splatting. Three panels, three quantities, three separate scales -- pressure, tangential
traction and slip velocity are Pa, Pa and m/s, and sharing an axis between them would
bury whichever is smaller.

Each map sits above its own time trace in its own hue, so a panel and its history are
identified by position as well as colour. The dark-surface hue anchors pass the
categorical validator (lightness band, chroma, CVD separation) at
``#2a78d6 / #c25a20 / #1baf7a``; each map's ramp is a single hue, light to dark, because
what it encodes is magnitude.

    python -m sugar_newton.validation.compose_allegro_field --run <dir>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

SURFACE, PANEL = "#12100f", "#1a1a19"
INK, INK_2, INK_MUTED = "#f5f4f0", "#c3c2b7", "#84837a"
OUTLINE_RGB = (0.30, 0.295, 0.275)   # the hand, under the field
CUBE_RGB = (0.52, 0.515, 0.48)       # the cube it is working

# One sequential single-hue ramp per channel, dark end first: on a dark surface the
# ramp has to run away from the background, not toward it.
CHANNELS = (
    ("pressure", "contact pressure", "kPa", 1e-3, "#2a78d6",
     ["#12233a", "#184f95", "#2a78d6", "#86b6ef", "#e8f1fd"], None),
    ("traction", "tangential traction", "kPa", 1e-3, "#c25a20",
     ["#2a1206", "#7a3211", "#c25a20", "#eb8c5a", "#fbd9c6"], "traction_vec"),
    ("slip", "slip velocity", "mm/s", 1e3, "#1baf7a",
     ["#0a231b", "#0f5a41", "#1baf7a", "#7fd9b8", "#e3f7ef"], "slip_vec"),
)
ARROW = "#efeee8"   # direction rides on a neutral ink; colour is still magnitude


def gaussian_matrix(n: int, sigma: float) -> np.ndarray:
    """Row-normalised 1-D Gaussian as a matrix, so a blur is ``G @ A @ G.T``.

    Two dense n x n products per array beat a Python loop over rows, and the
    row-normalisation makes the edge treatment identical for the weighted and the
    weight image -- so the division that follows cancels it exactly.
    """
    d = np.arange(n)[:, None] - np.arange(n)[None, :]
    k = np.exp(-0.5 * (d / sigma) ** 2)
    return k / k.sum(axis=1, keepdims=True)


def project(pts: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return pts @ basis.T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--grid", type=int, default=180, help="field raster resolution")
    ap.add_argument("--sigma", type=float, default=2.2, help="splat width [grid cells]")
    ap.add_argument("--proj", default="pca", choices=("pca", "xy", "xz", "yz"),
                    help="2-D view of the palm frame; pca is the contact set's own plane")
    ap.add_argument("--percentile", type=float, default=97.0,
                    help="colour-scale ceiling over the steady-state per-frame peaks")
    ap.add_argument("--aspect", type=float, default=1.25,
                    help="height/width of each map panel; the view is scaled to match")
    ap.add_argument("--quiver", type=int, default=22,
                    help="arrow grid across the panel width; 0 draws no arrows. Contact on "
                         "this hand is thin strips, so a coarse grid misses it entirely")
    ap.add_argument("--arrow-len", type=float, default=0.075,
                    help="length of a reference-magnitude arrow, as a fraction of the panel")
    ap.add_argument("--scene-center", type=float, nargs=2, default=(0.46, 0.60),
                    help="fractional (x, y) the scene render is cropped around")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds excluded when fixing the colour scales")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--video", default=None,
                    help="write this mp4 directly, piping raw frames to ffmpeg. Skips PNG "
                         "encoding and the disk entirely -- PNG encode alone was 40 %% of "
                         "the compositing time")
    ap.add_argument("--video-fps", type=float, default=30.0)
    ap.add_argument("--dpi", type=float, default=110.0)
    ap.add_argument("--interp", default="bilinear",
                    help="field image interpolation; 'nearest' is cheaper to draw")
    args = ap.parse_args()

    run = Path(args.run)
    f = np.load(run / "allegro_field.npz")
    tac = np.load(run / "allegro_tactile.npz")
    dt = float(f["dt"])
    off = f["offsets"]
    n = len(off) - 1
    t = np.arange(n) * dt
    pos, area = f["pos"], f["area"]
    # NpzFile decompresses on every __getitem__; these are read once per frame per panel.
    chan = {k: f[k] for k, *_ in CHANNELS}
    vecs = {vk: f[vk] for *_, vk in CHANNELS if vk and vk in f}
    outline, out_cube = f["outline"], f["outline_is_cube"]

    if len(pos) == 0:
        print("no contact-surface faces were recorded -- nothing to map")
        return 1

    # ---- fixed 2-D view -------------------------------------------------------
    # Fixed for the whole clip: a per-frame projection would make the map swim even
    # though the palm frame is already stationary.
    centre = pos.mean(axis=0)
    if args.proj == "pca":
        _, _, vt = np.linalg.svd(pos - centre, full_matrices=False)
        basis = vt[:2]
    else:
        axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[args.proj]
        basis = np.eye(3)[list(axes)]
    xy = project(pos - centre, basis) * 1e3          # mm
    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    pad = 0.12 * (hi - lo).max()
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo).max() + pad
    # The view is stretched to the panel's own aspect rather than letterboxed into it,
    # so mm-per-inch stays equal on both axes AND the panel fills: an "equal aspect"
    # square inside a tall cell wastes a third of the figure on empty background.
    ny = int(round(args.grid * args.aspect))
    ex = np.linspace(mid[0] - half, mid[0] + half, args.grid + 1)
    ey = np.linspace(mid[1] - half * args.aspect, mid[1] + half * args.aspect, ny + 1)
    extent = (ex[0], ex[-1], ey[0], ey[-1])
    out_xy = project(outline.reshape(-1, 3) - centre, basis).reshape(outline.shape[0], -1, 2) * 1e3
    # Vectors are rotated by the same basis, not translated: they are directions.
    # What the panel shows is the IN-PLANE part -- a slip straight into the pad has no
    # arrow, which is correct for a map of how the object travels ACROSS the skin.
    vec_xy = {k: project(v, basis) for k, v in vecs.items()}

    # Direct labels for the links that actually carry contact, at the median position
    # their faces occupy over the whole run. Identity should never rest on position alone
    # any more than it should rest on colour alone -- and "which finger is that blob?" is
    # the first question anyone asks of a tactile map.
    def _short(name: str) -> str:
        if name.startswith("palm"):
            return "palm"
        digit, _, link = name.partition("_link_")
        return f"{digit[:3]}{link}" if link else name[:6]

    patch = f["patch"]
    marks = []
    for r, name in enumerate(tac["labels"]):
        m = patch == r
        if int(m.sum()) >= 40:
            marks.append((_short(str(name)), np.median(xy[m], axis=0)))

    span = extent[1] - extent[0]
    bar = min([b for b in (1, 2, 5, 10, 20, 50, 100) if b >= 0.22 * span], default=100)

    Gx = gaussian_matrix(args.grid, args.sigma)
    Gy = gaussian_matrix(ny, args.sigma)
    Gxs = gaussian_matrix(args.grid, args.sigma * 2.2)   # wider, for the silhouette
    Gys = gaussian_matrix(ny, args.sigma * 2.2)
    support = Gx.diagonal().max() * Gy.diagonal().max() * 0.12  # this much of one sample

    def rasterise(i: int) -> tuple[dict, np.ndarray]:
        """Area-weighted Gaussian splat of one frame's faces onto the fixed grid."""
        sl = slice(off[i], off[i + 1])
        if off[i + 1] <= off[i]:
            return {k: None for k, *_ in CHANNELS}, None
        px, py, w = xy[sl, 0], xy[sl, 1], area[sl]
        cnt, _, _ = np.histogram2d(py, px, bins=[ey, ex])
        den, _, _ = np.histogram2d(py, px, bins=[ey, ex], weights=w)
        cnt = Gy @ cnt @ Gx.T
        den = Gy @ den @ Gx.T
        good = den > 0
        fields = {}
        for key, _t, _u, mult, _h, _r, _vk in CHANNELS:
            num, _, _ = np.histogram2d(py, px, bins=[ey, ex], weights=w * chan[key][sl] * mult)
            num = Gy @ num @ Gx.T
            fields[key] = np.where(good, num / np.where(good, den, 1.0), 0.0).astype(np.float32)
        return fields, cnt < support

    nqx = max(int(args.quiver), 0)
    nqy = max(int(round(nqx * args.aspect)), 1) if nqx else 0
    ny_q = nqy
    if nqx:
        qex = np.linspace(extent[0], extent[1], nqx + 1)
        qey = np.linspace(extent[2], extent[3], nqy + 1)
        qx = 0.5 * (qex[1:] + qex[:-1])
        qy = 0.5 * (qey[1:] + qey[:-1])
        QX, QY = np.meshgrid(qx, qy)

    def bin_vectors(i: int, vkey: str, mult: float):
        """Area-weighted mean vector per coarse cell, in display units.

        The MEAN, not the sum: two faces sliding opposite ways cancel, which is what a
        reader should see. A sum would draw a long arrow wherever faces are merely dense.
        """
        sl = slice(off[i], off[i + 1])
        if not nqx or off[i + 1] <= off[i]:
            return None, None, None
        px, py, w = xy[sl, 0], xy[sl, 1], area[sl]
        v = vec_xy[vkey][sl] * mult
        wsum, _, _ = np.histogram2d(py, px, bins=[qey, qex], weights=w)
        u, _, _ = np.histogram2d(py, px, bins=[qey, qex], weights=w * v[:, 0])
        vv, _, _ = np.histogram2d(py, px, bins=[qey, qex], weights=w * v[:, 1])
        good = wsum > 0
        u = np.where(good, u / np.where(good, wsum, 1.0), 0.0)
        vv = np.where(good, vv / np.where(good, wsum, 1.0), 0.0)
        return u, vv, good

    def blur_points(pts: np.ndarray) -> np.ndarray:
        if not len(pts):
            return np.zeros((ny, args.grid))
        h, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[ey, ex])
        return Gys @ h @ Gxs.T

    def _alpha(dens: np.ndarray, gamma: float) -> np.ndarray:
        peak = dens.max()
        return np.clip(dens / peak, 0.0, 1.0) ** gamma if peak > 0 else np.zeros_like(dens)

    def silhouette(hand: np.ndarray, cube: np.ndarray) -> np.ndarray:
        """Hand and cube as ONE uint8 RGBA layer.

        One image instead of two per panel, and uint8 instead of float64: matplotlib
        resamples every image artist on every draw, and the draw is ~90 % of compositing.
        """
        ah, ac = _alpha(hand, 0.85), _alpha(cube, 0.7)
        out = np.zeros(hand.shape + (4,), dtype=np.uint8)
        # cube over hand, straight alpha-over in one pass
        a = ah + ac * (1.0 - ah)
        safe = np.maximum(a, 1e-9)
        for ch in range(3):
            col = (OUTLINE_RGB[ch] * ah + CUBE_RGB[ch] * ac * (1.0 - ah)) / safe
            out[..., ch] = np.clip(col * 255.0, 0, 255).astype(np.uint8)
        out[..., 3] = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        return out

    # ---- one pass to fix the scales, so map and trace cannot disagree ---------
    # The trace is the peak of the field the map actually shows, not a percentile of
    # the raw face values: smoothing lowers peaks, and a trace drawn from raw samples
    # would run off the top of a colour bar keyed to the smoothed field.
    print("rasterising...", flush=True)
    cache, traces = {}, {k: np.zeros(n) for k, *_ in CHANNELS}
    for i in range(n):
        fields, mask = rasterise(i)
        cache[i] = (fields, mask)
        for key, *_ in CHANNELS:
            if fields[key] is not None:
                traces[key][i] = float(fields[key][~mask].max()) if (~mask).any() else 0.0

    s0 = min(int(args.settle / dt), n - 1)
    scales = {k: max(float(np.percentile(traces[k][s0:], args.percentile)), 1e-9)
              for k, *_ in CHANNELS}

    # One arrow length scale per channel, fixed for the clip off the same steady window,
    # so an arrow that grows between two frames grew because the field did.
    arrow_ref, qcache = {}, {}
    cell_w = (extent[1] - extent[0]) / max(nqx, 1)
    for _k, _t, _u, mult, _h, _r, vkey in CHANNELS:
        if not vkey or not nqx:
            continue
        mags = []
        for i in range(n):
            u, v, good = bin_vectors(i, vkey, mult)
            qcache[(i, vkey)] = (u, v, good)
            if good is not None and good.any():
                mags.append(float(np.hypot(u[good], v[good]).max()))
        # The MEDIAN of the per-frame maxima over the steady window, not a high
        # percentile. The colour scale can be keyed to the top of the range because
        # saturating is a legible failure; an arrow scale cannot, because a transient 30x
        # the working value (traction peaks at 1600 kPa against ~50 kPa steady) makes
        # every arrow in the part anyone watches too short to see. Clipped at 3x below,
        # so the transient still cannot cover the panel.
        ref = float(np.median(mags[s0:])) if len(mags) > s0 else (
            float(np.median(mags)) if mags else 1.0)
        arrow_ref[vkey] = max(ref, 1e-9)

    cmaps = {k: LinearSegmentedColormap.from_list(k, ramp)
             for k, _ti, _u, _m, _h, ramp, _vk in CHANNELS}
    frames = sorted((run / "frames").glob("f*.png")) or \
             sorted((run / "frames").glob("f*.jpg"))
    out = run / "field"
    out.mkdir(exist_ok=True)
    dropped = int(f["dropped"].sum())
    print(f"{n} frames, {len(pos)} faces, view={args.proj}, "
          f"scales: " + ", ".join(f"{k}<={scales[k]:.3g}" for k in scales), flush=True)
    if dropped:
        print(f"NOTE: {dropped} faces were dropped at record time (--field-max); "
              f"the map is a subsample of the surface, not an integral over it", flush=True)

    last = n if args.limit <= 0 else min(n, args.limit)
    todo = list(range(0, last, args.stride))

    # ---- build the figure ONCE -----------------------------------------------
    # Rebuilding a 16x9 figure with six axes, three colour bars and six quivers every
    # frame is most of the cost of composing a clip -- and none of it changes. Everything
    # below is created once; the loop only pushes new data into the artists.
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.32, 1, 1, 1], height_ratios=[1.42, 1.0],
                          left=0.012, right=0.985, top=0.875, bottom=0.075,
                          wspace=0.26, hspace=0.22)
    frames = sorted((run / "frames").glob("f*.png")) or \
             sorted((run / "frames").glob("f*.jpg"))
    ax = fig.add_subplot(gs[:, 0]); ax.axis("off"); ax.set_facecolor(SURFACE)
    bb = ax.get_position()
    want = (bb.width * 16.0) / (bb.height * 9.0)

    def crop_scene(path):
        """Crop the 16:9 render to the cell's aspect rather than letterboxing it."""
        img = plt.imread(path)
        h, w = img.shape[:2]
        cw, ch = w, h
        if w / h > want:
            cw = int(round(h * want))
        else:
            ch = int(round(w / want))
        cx = int(np.clip(args.scene_center[0] * w, cw / 2, w - cw / 2))
        cy = int(np.clip(args.scene_center[1] * h, ch / 2, h - ch / 2))
        return img[cy - ch // 2:cy + ch // 2, cx - cw // 2:cx + cw // 2]

    im_scene = ax.imshow(crop_scene(frames[0])) if frames else None

    panels = []
    for c, (key, title, unit, mult, hue, _ramp, vkey) in enumerate(CHANNELS):
        a = fig.add_subplot(gs[0, c + 1])
        a.set_facecolor(PANEL)
        a.set_xlim(extent[0], extent[1]); a.set_ylim(extent[2], extent[3])
        a.set_box_aspect(args.aspect)
        blank = np.zeros((ny, args.grid, 4), dtype=np.uint8)
        im_hand = a.imshow(blank.copy(), origin="lower", extent=extent, zorder=1,
                           interpolation="nearest")
        im_cube = None
        norm = Normalize(0.0, scales[key])
        im_fld = a.imshow(np.ma.masked_all((ny, args.grid)), origin="lower", extent=extent,
                          cmap=cmaps[key], norm=norm, interpolation=args.interp, zorder=3)
        qh = ql = None
        if vkey and nqx:
            zeros = np.zeros(QX.size)
            sc = arrow_ref[vkey] / (args.arrow_len * span)
            # Drawn twice: a dark halo first, the light arrow on top. These ramps run to
            # near-white at the top, and a single light arrow vanishes exactly where the
            # field is most interesting. Fixed positions, zero length where there is no
            # contact -- a zero-length arrow draws nothing, so one artist covers every
            # frame however the contact set changes.
            qh = a.quiver(QX.ravel(), QY.ravel(), zeros, zeros.copy(), angles="xy",
                          scale_units="xy", scale=sc, color=SURFACE, width=0.013,
                          headwidth=2.6, headlength=2.9, headaxislength=2.4, alpha=0.9,
                          zorder=4)
            ql = a.quiver(QX.ravel(), QY.ravel(), zeros.copy(), zeros.copy(), angles="xy",
                          scale_units="xy", scale=sc, color=ARROW, width=0.0065,
                          headwidth=4.0, headlength=4.4, headaxislength=3.8, alpha=0.95,
                          zorder=5)
            if c == 1:
                a.quiverkey(ql, 0.78, 0.045, arrow_ref[vkey],
                            f"{arrow_ref[vkey]:.3g} {unit}", labelpos="E", color=ARROW,
                            labelcolor=INK_2, fontproperties={"size": 7.0},
                            coordinates="axes")
        for name, (mx, my) in marks:
            a.text(mx, my, name, ha="center", va="center", fontsize=6.5,
                   color=INK_MUTED, zorder=6)
        a.set_xticks([]); a.set_yticks([])
        if c == 0:
            # A scale bar, not tick labels: the axes are a projection of a body frame, so
            # the only number worth reading off them is how big the patch is.
            x0 = extent[0] + 0.06 * (extent[1] - extent[0])
            y0 = extent[2] + 0.05 * (extent[3] - extent[2])
            a.plot([x0, x0 + bar], [y0, y0], color=INK_2, lw=1.6, solid_capstyle="butt",
                   zorder=6)
            a.text(x0 + bar / 2, y0 + 0.012 * (extent[3] - extent[2]), f"{bar:g} mm",
                   ha="center", va="bottom", fontsize=7.5, color=INK_2, zorder=6)
        for side in ("top", "right", "left", "bottom"):
            a.spines[side].set_color(INK_MUTED); a.spines[side].set_linewidth(0.6)
        a.set_title(f"{title}", color=INK, fontsize=10.5, loc="left", pad=16)
        sub = "palm frame · [" + unit + "]"
        if vkey and nqx:
            sub += " · arrows: direction + magnitude"
        a.text(0.0, 1.008, sub, transform=a.transAxes, fontsize=8, color=INK_MUTED,
               va="bottom")
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmaps[key]), ax=a,
                          fraction=0.045, pad=0.02)
        cb.set_ticks(np.linspace(0.0, scales[key], 5))
        cb.ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.3g}")
        cb.ax.tick_params(colors=INK_2, labelsize=7.5, length=2.5, width=0.7)
        cb.outline.set_visible(False)

        b = fig.add_subplot(gs[1, c + 1]); b.set_facecolor(PANEL)
        for side in ("top", "right"):
            b.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            b.spines[side].set_color(INK_MUTED); b.spines[side].set_linewidth(0.7)
        b.tick_params(colors=INK_2, labelsize=7.5, length=2.5)
        b.grid(True, axis="y", color=INK_MUTED, alpha=0.18, linewidth=0.5)
        b.set_axisbelow(True)
        b.plot(t, traces[key], color=hue, lw=1.4)
        b.fill_between(t, 0, traces[key], color=hue, alpha=0.16, lw=0)
        cursor = b.axvline(t[0], color=INK_2, lw=0.9, alpha=0.75)
        # Where the map saturates, drawn: the colour scale is fixed off the steady state,
        # so the settling transient genuinely runs off the top of the ramp and the trace
        # is the only place that is visible.
        if traces[key].max() > scales[key]:
            b.axhline(scales[key], color=INK_MUTED, lw=0.8, ls=(0, (4, 3)))
            if c == 0:
                b.annotate("map ceiling", xy=(t[-1], scales[key]), xytext=(-2, 3),
                           textcoords="offset points", ha="right", va="bottom",
                           fontsize=7.5, color=INK_MUTED)
        b.set_ylim(0, max(traces[key].max() * 1.08, scales[key] * 1.2, 1e-9))
        b.set_ylabel(f"peak  [{unit}]", color=INK_2, fontsize=8)
        b.set_xlabel("time [s]", color=INK_2, fontsize=8)
        # Freeze the ticks. The axes never change, but a locator is re-run on every draw
        # and drags text metrics with it -- together the single largest avoidable cost in
        # compositing.
        b.set_xticks(np.linspace(t[0], t[-1], 5))
        b.set_yticks(np.linspace(*b.get_ylim(), 5))
        b.xaxis.set_major_formatter(lambda v, _p: f"{v:.0f}")
        b.yaxis.set_major_formatter(lambda v, _p: f"{v:.3g}")
        panels.append(dict(key=key, vkey=vkey, im_hand=im_hand, im_cube=im_cube,
                           im_fld=im_fld, qh=qh, ql=ql, cursor=cursor))

    fig.suptitle("Newton tactile field — Allegro hand working a cube", color=INK,
                 fontsize=14, x=0.015, ha="left", y=0.965)
    header = fig.text(0.015, 0.918, "", color=INK_MUTED, fontsize=9.5, ha="left",
                      family="monospace")

    proc = None
    if args.video:
        fig.canvas.draw()
        vw, vh = fig.canvas.get_width_height()
        vw, vh = vw - vw % 2, vh - vh % 2      # libx264 with yuv420p needs even sides
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "rgba",
             "-s", f"{vw}x{vh}", "-r", str(args.video_fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(args.video)],
            stdin=subprocess.PIPE,
        )
        print(f"piping {vw}x{vh} frames straight into {args.video}", flush=True)

    t_start = time.perf_counter()
    prof = {"silhouette": 0.0, "update": 0.0, "draw": 0.0, "write": 0.0}
    for k, i in enumerate(todo):
        tA = time.perf_counter()
        if im_scene is not None and i < len(frames):
            im_scene.set_data(crop_scene(frames[i]))
        fields, mask = cache[i]
        hand_s = blur_points(out_xy[i][~out_cube])
        cube_s = blur_points(out_xy[i][out_cube])
        tB = time.perf_counter(); prof["silhouette"] += tB - tA
        sil = silhouette(hand_s, cube_s)
        for pan in panels:
            pan["im_hand"].set_data(sil)
            fld = fields[pan["key"]]
            if fld is None:
                pan["im_fld"].set_data(np.ma.masked_all((ny, args.grid)))
            else:
                pan["im_fld"].set_data(np.ma.masked_where(mask, fld))
            vkey = pan["vkey"]
            if vkey and nqx:
                u, v, good = qcache.get((i, vkey), (None, None, None))
                # NaN, not zero, for "no arrow here": a zero-length quiver arrow still
                # draws its head, which papers the panel with a grid of dots.
                if good is None:
                    u = v = np.full((ny_q, nqx), np.nan)
                else:
                    keep = good & (np.hypot(u, v) > 0.02 * arrow_ref[vkey])
                    # Clip, do not drop: the settling transient is many times the working
                    # magnitude, and an unclipped arrow there spans the whole panel.
                    mag = np.hypot(u, v)
                    shrink = np.where(mag > 3.0 * arrow_ref[vkey],
                                      3.0 * arrow_ref[vkey] / np.maximum(mag, 1e-12), 1.0)
                    u = np.where(keep, u * shrink, np.nan)
                    v = np.where(keep, v * shrink, np.nan)
                pan["qh"].set_UVC(u.ravel(), v.ravel())
                pan["ql"].set_UVC(u.ravel(), v.ravel())
            pan["cursor"].set_xdata([t[i], t[i]])
        nf = int(off[i + 1] - off[i])
        live = int((tac["contact_count"][i] > 0).sum())
        header.set_text(
            f"t = {t[i]:5.2f} s   ·   {nf:5d} contact-surface faces   ·   "
            f"{live}/{len(tac['labels'])} links loaded   ·   "
            f"normal load {tac['normal_load'][i].sum():6.2f} N   ·   "
            f"penetration {tac['peak_depth'][i].max() * 1e3:5.2f} mm")
        tC = time.perf_counter(); prof["update"] += tC - tB
        if proc is not None:
            fig.canvas.draw()
            tD = time.perf_counter(); prof["draw"] += tD - tC
            buf = np.asarray(fig.canvas.buffer_rgba())[:vh, :vw]
            proc.stdin.write(buf.tobytes())
            prof["write"] += time.perf_counter() - tD
        else:
            fig.savefig(out / f"g{i:05d}.png", dpi=args.dpi, facecolor=SURFACE)
            prof["draw"] += time.perf_counter() - tC
        if k and k % 25 == 0:
            rate = k / (time.perf_counter() - t_start)
            print(f"  composed {k}/{len(todo)}  ({rate:.1f} frames/s)", flush=True)
    if proc is not None:
        proc.stdin.close()
        proc.wait()
    plt.close(fig)
    el = max(time.perf_counter() - t_start, 1e-9)
    print("  compose profile: " + "  ".join(
        f"{k}={1e3 * v / max(len(todo), 1):.1f}ms({100 * v / el:.0f}%)"
        for k, v in prof.items()), flush=True)
    print(f"  composed {len(todo)} frames at "
          f"{len(todo) / max(time.perf_counter() - t_start, 1e-9):.1f} frames/s", flush=True)


    if args.video:
        print(f"wrote {args.video}")
    else:
        print(f"wrote {out}")
        print(f"assemble on the login node:\n  ffmpeg -y -framerate 30 -i {out}/g%05d.png "
              f"-c:v libx264 -pix_fmt yuv420p -crf 18 {run}/allegro_field.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
