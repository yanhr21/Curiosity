# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Pass 2: composite the render_friction trace into video frames.

Runs on the login node, where matplotlib and ffmpeg live.  Reads the ``.npz``
(and the scene PNGs, if pass 1 captured any) written by
``render_friction.py`` and emits composited PNGs plus the ffmpeg command.

Layout: scene on the left when frames exist, three stacked trace panels on the
right --

    friction utilization vs the analytic tan(theta)/mu
    slip displacement and slip velocity
    normal and friction load

with a moving time cursor and the current mu called out, so the stick-to-slip
transition is legible without reading the axes.

    python -m sugar_newton.validation.compose_friction_video --run <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BG = "#12151c"
FG = "#dfe4ee"
ACCENT = "#4da3ff"
WARN = "#ff8842"
OK = "#5fd38d"


def style(ax, title: str) -> None:
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=10, loc="left", pad=6)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a3040")
    ax.grid(True, color="#222836", linewidth=0.6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="directory written by render_friction.py")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    run = Path(args.run)
    data = np.load(run / "trace.npz")
    mu = data["mu"]
    util = data["utilization_mean"]
    util_max = data["utilization_max"]
    util_exp = data["expected_utilization"]
    slip_d = data["slip_displacement"]
    slip_v = data["slip_velocity"]
    gross = data["gross_slip_fraction"]
    n_load = data["normal_load"]
    f_load = data["friction_load"]
    theta = float(data["theta"])
    n = len(mu)

    scene_frames = sorted((run / "frames").glob("f*.png"))
    have_scene = len(scene_frames) >= n
    if scene_frames and not have_scene:
        print(f"note: {len(scene_frames)} scene PNGs for {n} samples; panels only")

    out = run / "composite"
    out.mkdir(exist_ok=True)
    t = np.arange(n)

    # Where the block first genuinely breaks away.
    slipping = np.where(gross > 0.5)[0]
    breakaway = int(slipping[0]) if len(slipping) else None

    for i in range(0, n, args.stride):
        if have_scene:
            fig = plt.figure(figsize=(16, 9), facecolor=BG)
            gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], hspace=0.45, wspace=0.18)
            ax_scene = fig.add_subplot(gs[:, 0])
            ax_scene.set_facecolor(BG)
            ax_scene.axis("off")
            import PIL.Image

            ax_scene.imshow(np.asarray(PIL.Image.open(scene_frames[i])))
            axes = [fig.add_subplot(gs[r, 1]) for r in range(3)]
        else:
            fig = plt.figure(figsize=(11, 9), facecolor=BG)
            gs = fig.add_gridspec(3, 1, hspace=0.45)
            axes = [fig.add_subplot(gs[r, 0]) for r in range(3)]

        fig.suptitle(
            f"Newton tactile — incline {theta:.0f}°, friction swept down    "
            f"μ = {mu[i]:.3f}",
            color=FG,
            fontsize=13,
            x=0.02,
            ha="left",
        )

        a = axes[0]
        style(a, "friction utilization  ‖f_t‖ / (μ‖f_n‖)")
        a.plot(t, util_exp, color="#7a8699", lw=1.2, ls="--", label="tan(θ)/μ  analytic")
        a.plot(t, util, color=ACCENT, lw=1.6, label="measured (load-weighted)")
        a.plot(t, util_max, color="#2e6da8", lw=0.9, alpha=0.8, label="per-contact max")
        a.axhline(1.0, color=WARN, lw=1.0, ls=":", label="Coulomb limit")
        a.set_ylim(0, min(2.0, max(2.0, np.nanmax(util_exp[: i + 1]) * 1.1)))
        a.legend(facecolor=BG, edgecolor="#2a3040", labelcolor=FG, fontsize=7, loc="upper left")

        b = axes[1]
        style(b, "slip — displacement [m] and velocity [m/s]")
        b.plot(t, slip_d, color=OK, lw=1.5, label="anchor drift (incipient)")
        b.plot(t, slip_v, color="#ffcc55", lw=1.3, label="relative velocity")
        b.plot(t, gross * max(slip_v.max(), 1e-9), color=WARN, lw=1.0, alpha=0.7,
               label="gross-slip fraction (scaled)")
        b.set_yscale("symlog", linthresh=1e-6)
        b.legend(facecolor=BG, edgecolor="#2a3040", labelcolor=FG, fontsize=7, loc="upper left")

        c = axes[2]
        style(c, "contact load [N]")
        c.plot(t, n_load, color=ACCENT, lw=1.5, label="normal")
        c.plot(t, f_load, color="#ff8f6b", lw=1.3, label="friction (tangential)")
        c.set_xlabel("frame", color=FG, fontsize=8)
        c.legend(facecolor=BG, edgecolor="#2a3040", labelcolor=FG, fontsize=7, loc="upper left")

        for ax in axes:
            ax.axvline(i, color=FG, lw=1.0, alpha=0.55)
            if breakaway is not None:
                ax.axvline(breakaway, color=WARN, lw=1.0, alpha=0.5, ls="--")
        if breakaway is not None and i >= breakaway:
            axes[0].annotate(
                "break-away",
                xy=(breakaway, 1.0),
                color=WARN,
                fontsize=8,
                xytext=(6, 6),
                textcoords="offset points",
            )

        fig.savefig(out / f"c{i:05d}.png", dpi=100, facecolor=BG)
        plt.close(fig)

    mp4 = run / "friction_sweep.mp4"
    print(f"composited {len(list(out.glob('c*.png')))} frames -> {out}")
    print()
    print("assemble with:")
    print(
        f"  ffmpeg -y -framerate {args.fps} -pattern_type glob -i '{out}/c*.png' "
        f"-c:v libx264 -pix_fmt yuv420p -crf 20 {mp4}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
