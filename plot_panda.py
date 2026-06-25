# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rows, key, cast=float):
    return [cast(r[key]) for r in rows]


iters = load("results_panda_iters.csv")
substeps = load("results_panda_substeps.csv")
impratio = load("results_panda_impratio.csv")
kh = load("results_panda_kh.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))


def dual(ax, rows, xkey, xlabel, logx=True):
    x = col(rows, xkey)
    fps = col(rows, "fps")
    pen = col(rows, "max_pen_mm")
    ax.plot(x, fps, "o-", color="tab:blue")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FPS", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_ylim(0, max(fps) * 1.15)
    a2 = ax.twinx()
    a2.plot(x, pen, "s--", color="tab:red")
    a2.set_ylabel("max penetration (mm)", color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red")
    a2.set_ylim(0, max(pen) * 1.15)
    return a2


dual(axes[0, 0], kh, "kh", "kh  (hydroelastic stiffness)")
axes[0, 0].set_title("kh: penetration ↓↓ steeply, FPS flat  →  THE lever (≈free)")

dual(axes[0, 1], substeps, "substeps", "substeps  (1/timestep)")
axes[0, 1].set_title("substeps: penetration flat, FPS craters  →  pure cost past ~5")

dual(axes[1, 0], impratio, "impratio", "impratio")
axes[1, 0].set_title("impratio: penetration flat, FPS ↓  →  dead for penetration")

# Pareto: only kh moves horizontally (reduces penetration)
ax = axes[1, 1]
for rows, name, m in (
    (kh, "kh 1e9→1e12", "o-"),
    (substeps, "substeps 2→40", "s-"),
    (impratio, "impratio 1→1e4", "^-"),
    (iters, "iterations 1→60", "d-"),
):
    ax.plot(col(rows, "max_pen_mm"), col(rows, "fps"), m, label=name, alpha=0.85)
ax.set_xscale("log")
ax.set_xlabel("max penetration (mm)  — lower = better")
ax.set_ylabel("FPS")
ax.set_title("Pareto: only kh moves left (less penetration);\nthe rest only move FPS")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("panda_hydro: FPS vs collision quality across four knobs (world_count=4)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("panda_hydro_levers.png", dpi=130)
print("saved panda_hydro_levers.png")
