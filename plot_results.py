# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


xpbd = load("results_fps_collision.csv")
panda = load("results_panda_iters.csv")
panda_ls = load("results_panda_ls.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# Panel A: panda_hydro iterations -> FPS and penetration (twin axis)
ax = axes[0]
it = [int(r["iterations"]) for r in panda]
fps = [float(r["fps"]) for r in panda]
pen = [float(r["max_pen_mm"]) for r in panda]
ax.plot(it, fps, "o-", color="tab:blue", label="FPS")
ax.set_xscale("log", base=2)
ax.set_xlabel("solver iterations")
ax.set_ylabel("FPS", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_ylim(0, max(fps) * 1.15)
ax2 = ax.twinx()
ax2.plot(it, pen, "s--", color="tab:red", label="max penetration")
ax2.set_ylabel("max penetration (mm)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0, max(pen) * 1.15)
ax.axvline(15, color="gray", ls=":", lw=1)
ax.set_title("panda_hydro: iterations sweep\n(penetration saturates ~5 iters; FPS floors ~10)")

# Panel B: panda_hydro ls_iterations (shows no effect)
ax = axes[1]
ls = [int(r["ls_iters"]) for r in panda_ls]
fps = [float(r["fps"]) for r in panda_ls]
pen = [float(r["max_pen_mm"]) for r in panda_ls]
ax.plot(ls, fps, "o-", color="tab:blue", label="FPS")
ax.set_xscale("log")
ax.set_xlabel("ls_iterations")
ax.set_ylabel("FPS", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_ylim(0, max(fps) * 1.3)
ax2 = ax.twinx()
ax2.plot(ls, pen, "s--", color="tab:red")
ax2.set_ylabel("max penetration (mm)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0, max(pen) * 1.3)
ax.set_title("panda_hydro: line-search sweep\n(no effect — solver-cost not the bottleneck)")

# Panel C: Pareto comparison FPS vs penetration (log-log)
ax = axes[2]
for data, name, m in ((xpbd, "XPBD box pyramid", "o-"), (panda, "panda_hydro (MuJoCo+hydro)", "s-")):
    fps = [float(r["fps"]) for r in data]
    pen = [float(r["max_pen_mm"]) for r in data]
    its = [int(r["iterations"]) for r in data]
    ax.plot(pen, fps, m, label=name)
    for x, y, n in zip(pen, fps, its):
        ax.annotate(str(n), (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("max penetration (mm)  — lower = better quality")
ax.set_ylabel("FPS  — higher = faster")
ax.set_title("Pareto: FPS vs collision quality\n(labels = iteration count)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig("fps_vs_collision.png", dpi=130)
print("saved fps_vs_collision.png")
