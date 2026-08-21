# SPDX-License-Identifier: Apache-2.0
"""Plot the tactile fields recorded by tactile_field.py.

Three measures on three different scales get three panels, never a shared axis.
Contact pressure carries one series (the floor's is structurally absent -- Newton
clears HYDROELASTIC on PLANE shapes, so an analytic ground plane produces no
contact surface); normal force and slip velocity carry two.

  uv run python plot_tactile.py renders/tactile_walk.npz --out renders/tactile_walk.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"
FURN, FLOOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated together)
# sequential blue ramp, steps 100 -> 700: magnitude encoding for the pressure map
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
FPS = 50.0


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis="y", color=INK_MUTED, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=0.8)


def _label_end(ax, t, y, color, text):
    """Direct-label the series at its last point, in ink -- identity is never colour alone."""
    if not len(t):
        return
    ax.annotate(
        text,
        xy=(t[-1], y[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
        color=INK_2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="G1 tactile fields — SAGE room, mu 1.4, 1.0 m/s")
    args = ap.parse_args()
    d = np.load(args.npz)
    out = args.out or os.path.splitext(args.npz)[0] + ".png"

    n = len(d["press_peak_furn"])
    t = np.arange(n) / FPS
    has_snap = "snap_points" in d.files

    fig = plt.figure(figsize=(12.4, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.62, 1.0], hspace=0.42, wspace=0.24,
                          left=0.075, right=0.955, top=0.845, bottom=0.098)

    # ---- 1. contact pressure (furniture only) ----
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    y = d["press_peak_furn"] / 1e6
    ax.plot(t, y, color=FURN, linewidth=2.0, solid_capstyle="round")
    ax.fill_between(t, 0, y, color=FURN, alpha=0.13, linewidth=0)
    _label_end(ax, t, y, FURN, "furniture")
    ax.set_ylabel("peak pressure  [MPa]", color=INK_2, fontsize=9)
    ax.set_title("Contact pressure — hydroelastic surface", color=INK, fontsize=11,
                 loc="left", pad=20, fontweight="medium")
    ax.text(0.0, 1.005, "floor omitted: an analytic ground plane carries no hydroelastic surface",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")

    # ---- 2. normal force ----
    ax = fig.add_subplot(gs[1, 0])
    _style(ax)
    ax.plot(t, d["fn_floor"], color=FLOOR, linewidth=2.0, solid_capstyle="round", label="floor")
    ax.plot(t, d["fn_furn"], color=FURN, linewidth=2.0, solid_capstyle="round", label="furniture")
    _label_end(ax, t, d["fn_floor"], FLOOR, "floor")
    _label_end(ax, t, d["fn_furn"], FURN, "furniture")
    ax.set_ylabel("normal force  [N]", color=INK_2, fontsize=9)
    ax.set_title("Normal force — solved contact, summed per surface", color=INK, fontsize=11,
                 loc="left", pad=20, fontweight="medium")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncols=2, handlelength=1.6)
    for txt in leg.get_texts():
        txt.set_color(INK_2)

    # ---- 3. slip velocity ----
    ax = fig.add_subplot(gs[2, 0])
    _style(ax)
    ax.plot(t, d["slip_peak_floor"], color=FLOOR, linewidth=2.0, solid_capstyle="round", label="floor")
    ax.plot(t, d["slip_peak_furn"], color=FURN, linewidth=2.0, solid_capstyle="round", label="furniture")
    _label_end(ax, t, d["slip_peak_floor"], FLOOR, "floor")
    _label_end(ax, t, d["slip_peak_furn"], FURN, "furniture")
    ax.set_ylabel("peak slip speed  [m/s]", color=INK_2, fontsize=9)
    ax.set_xlabel("time  [s]", color=INK_2, fontsize=9)
    ax.set_title("Slip velocity — tangential surface speed at contact", color=INK, fontsize=11,
                 loc="left", pad=20, fontweight="medium")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncols=2, handlelength=1.6)
    for txt in leg.get_texts():
        txt.set_color(INK_2)

    # ---- 4. the contact patch itself, coloured by pressure ----
    ax = fig.add_subplot(gs[0:2, 1])
    ax.set_facecolor(SURFACE)
    if has_snap:
        pts, press = d["snap_points"], d["snap_pressure"] / 1e6
        c = pts - pts.mean(axis=0)
        # view the patch face-on: its own two principal axes
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        xy = c @ vt[:2].T * 1e3  # mm
        order = np.argsort(press)
        sc = ax.scatter(xy[order, 0], xy[order, 1], c=press[order], cmap=SEQ, s=26,
                        linewidths=0.4, edgecolors=SURFACE)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label("pressure  [MPa]", color=INK_2, fontsize=9)
        cb.ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=0.8)
        cb.outline.set_visible(False)
        ax.set_aspect("equal")
        ax.set_xlabel("patch axis 1  [mm]", color=INK_2, fontsize=9)
        ax.set_ylabel("patch axis 2  [mm]", color=INK_2, fontsize=9)
        ax.set_title(f"Contact patch at peak — frame {int(d['snap_frame'])}, {len(press)} faces",
                     color=INK, fontsize=11, loc="left", pad=20, fontweight="medium")
        ax.text(0.0, 1.005, "robot↔furniture, in the patch's principal plane",
                transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=0.8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no furniture contact recorded", ha="center", color=INK_MUTED)

    # ---- 5. contact area: the other half of "pressure" ----
    ax = fig.add_subplot(gs[2, 1])
    _style(ax)
    a = d["area_furn"] * 1e4
    ax.plot(t, a, color=FURN, linewidth=2.0, solid_capstyle="round")
    ax.fill_between(t, 0, a, color=FURN, alpha=0.13, linewidth=0)
    _label_end(ax, t, a, FURN, "furniture")
    ax.set_ylabel("contact area  [cm²]", color=INK_2, fontsize=9)
    ax.set_xlabel("time  [s]", color=INK_2, fontsize=9)
    ax.set_title("Contact area", color=INK, fontsize=11, loc="left", pad=20, fontweight="medium")
    ax.text(0.0, 1.005, "the denominator behind the pressure above",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")

    fig.suptitle(args.title, color=INK, fontsize=13.5, x=0.075, ha="left", y=0.955, fontweight="medium")
    fig.text(0.075, 0.925,
             f"{n} frames @ {FPS:g} fps · peak {d['press_peak_furn'].max() / 1e6:.0f} MPa · "
             f"{int((d['faces_furn'] > 0).sum())} frames in furniture contact",
             color=INK_MUTED, fontsize=9, ha="left")
    fig.text(0.075, 0.012,
             "Pressure magnitudes follow from shape_material_kh (1e12 Pa/m on rigid links): "
             "GPa-scale peaks are that stiffness choice, not a material limit. Compliant links "
             "(feet, kh 5e8) are the realistic tactile setting.",
             color=INK_MUTED, fontsize=8, ha="left")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
