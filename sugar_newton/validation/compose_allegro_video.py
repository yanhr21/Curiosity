# SPDX-License-Identifier: BSD-3-Clause
"""Compose the Allegro tactile video: scene + a per-link hand map + traces.

Left: the rendered hand. Right: the hand drawn as its own kinematic tree -- palm at the
base, four digits as columns of four links -- with each link coloured by the channel
being shown. Below: the three channels over time with a moving cursor.

Pressure, friction and slip velocity get their own row rather than a shared axis: they
are different quantities in different units, and overlaying them would hide whichever
has the smaller range.

    python -m sugar_newton.validation.compose_allegro_video --run <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

SURFACE, PANEL = "#12100f", "#1a1a19"
INK, INK_2, INK_MUTED = "#f5f4f0", "#c3c2b7", "#84837a"
# one hue per channel, each its own light->dark sequential ramp (magnitude encoding)
RAMPS = {
    "peak_pressure": ["#0d2a4d", "#184f95", "#2a78d6", "#86b6ef", "#e8f1fd"],
    "friction_load": ["#3a1405", "#7a3211", "#c25a20", "#eb6834", "#fbd9c6"],
    "slip_velocity": ["#0a2b20", "#0f5a41", "#1baf7a", "#7fd9b8", "#e3f7ef"],
}
TITLES = {
    "peak_pressure": ("contact pressure", "kPa", 1e-3),
    "friction_load": ("friction load", "N", 1.0),
    "slip_velocity": ("slip velocity", "mm/s", 1e3),
}
DIGITS = ("index", "middle", "ring", "thumb")


def hand_cells(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Anatomical boxes: a digit per column, link_0 (proximal) at the bottom."""
    cells = {}
    for c, digit in enumerate(DIGITS):
        for link in range(4):
            name = f"{digit}_link_{link}"
            if name in labels:
                cells[name] = (c * 1.15, 1.5 + link * 1.12, 1.0, 1.0)
    if "palm_link" in labels:
        cells["palm_link"] = (0.0, 0.15, 1.15 * 3 + 1.0, 1.15)
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--channel", default="peak_pressure", choices=list(RAMPS))
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    run = Path(args.run)
    d = np.load(run / "allegro_tactile.npz")
    labels = [str(x) for x in d["labels"]]
    dt = float(d["dt"])
    n = d["normal_load"].shape[0]
    t = np.arange(n) * dt

    frames = sorted((run / "frames").glob("f*.png"))
    out = run / "composite"
    out.mkdir(exist_ok=True)
    cells = hand_cells(labels)
    idx = {name: labels.index(name) for name in cells}

    # Percentile scaling, not max: the first frames carry a settling transient that is
    # orders of magnitude above the grasp and would flatten the whole clip to one colour.
    scales = {}
    for ch in RAMPS:
        v = d[ch]
        hi = float(np.percentile(v[v > 0], 99)) if (v > 0).any() else 1.0
        scales[ch] = max(hi, 1e-9)

    cmaps = {k: LinearSegmentedColormap.from_list(k, v) for k, v in RAMPS.items()}
    ch = args.channel
    title, unit, mult = TITLES[ch]
    norm = Normalize(0.0, scales[ch] * mult)

    for i in range(0, n, args.stride):
        fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
        gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 0.95, 1.5], hspace=0.55, wspace=0.22,
                              left=0.02, right=0.965, top=0.88, bottom=0.07)

        ax = fig.add_subplot(gs[:, 0]); ax.axis("off"); ax.set_facecolor(SURFACE)
        if i < len(frames):
            ax.imshow(plt.imread(frames[i]))

        # ---- the hand map ----
        hm = fig.add_subplot(gs[:, 1]); hm.set_facecolor(SURFACE); hm.axis("off")
        hm.set_xlim(-0.25, 1.15 * 3 + 1.25); hm.set_ylim(-0.25, 1.5 + 4 * 1.12 + 0.75)
        hm.set_aspect("equal")
        for name, (x, y, w, h) in cells.items():
            val = float(d[ch][i, idx[name]]) * mult
            hm.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        facecolor=cmaps[ch](norm(val)), edgecolor=INK_MUTED, linewidth=0.6))
            live = d["contact_count"][i, idx[name]] > 0
            label_y = y + h * 0.62 if name == "palm_link" else y + h / 2
            hm.text(x + w / 2, label_y, f"{val:.0f}" if live else "·", ha="center", va="center",
                    fontsize=8.5, color=INK if norm(val) < 0.55 else "#0b0b0b")
        for c, digit in enumerate(DIGITS):
            hm.text(c * 1.15 + 0.5, 1.5 + 4 * 1.12 + 0.08, digit, ha="center", color=INK_2, fontsize=9)
        hm.text((1.15 * 3 + 1.0) / 2, 0.15 + 1.15 * 0.25, "palm", ha="center", va="center",
                color=INK_2, fontsize=9)
        hm.set_title(f"{title}  [{unit}]", color=INK, fontsize=11, pad=22)

        # ---- traces ----
        for r, key in enumerate(RAMPS):
            a = fig.add_subplot(gs[r, 2]); a.set_facecolor(PANEL)
            ttl, un, m = TITLES[key]
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                a.spines[side].set_color(INK_MUTED); a.spines[side].set_linewidth(0.7)
            a.tick_params(colors=INK_2, labelsize=7.5, length=2.5)
            a.grid(True, axis="y", color=INK_MUTED, alpha=0.18, linewidth=0.5)
            a.set_axisbelow(True)
            series = d[key] * m
            a.plot(t, series.max(axis=1), color=RAMPS[key][3], lw=1.4)
            a.fill_between(t, 0, series.max(axis=1), color=RAMPS[key][3], alpha=0.16, lw=0)
            a.axvline(t[i], color=INK_2, lw=0.9, alpha=0.75)
            a.set_ylim(0, max(scales[key] * m * 1.15, 1e-9))
            a.set_ylabel(f"{ttl}\n[{un}]", color=INK_2, fontsize=8)
            if r == 2:
                a.set_xlabel("time [s]", color=INK_2, fontsize=8)

        live = int((d["contact_count"][i] > 0).sum())
        fig.suptitle("Newton tactile — Allegro hand regrasping a cube", color=INK, fontsize=14,
                     x=0.02, ha="left", y=0.965)
        fig.text(0.02, 0.925,
                 f"t = {t[i]:5.2f} s   ·   {live}/{len(labels)} links in contact   ·   "
                 f"normal load {d['normal_load'][i].sum():6.2f} N   ·   "
                 f"peak slip {d['slip_velocity'][i].max() * 1e3:6.1f} mm/s",
                 color=INK_MUTED, fontsize=9.5, ha="left", family="monospace")
        fig.savefig(out / f"c{i:05d}.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)
        if i % 50 == 0:
            print(f"  composed {i}/{n}", flush=True)

    print(f"wrote {out}")
    print(f"assemble on the login node:\n  ffmpeg -y -framerate 30 -i {out}/c%05d.png "
          f"-c:v libx264 -pix_fmt yuv420p -crf 18 {run}/allegro_tactile.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
