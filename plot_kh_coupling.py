# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Measured points (panda_hydro, world_count=4, iters=15). (substeps, max_pen_mm, fps)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

kh11 = [(2, 1.079, 243), (5, 0.910, 137), (10, 0.907, 73), (20, 0.919, 46), (40, 0.922, 29)]
kh13 = [(5, 0.309, 147), (10, 0.128, 92), (20, 0.100, 55), (40, 0.101, 32)]
kh14 = [(2, 1.088, 257), (20, 0.053, 56), (40, 0.033, 25)]
series = [("kh=1e11", kh11, "o-"), ("kh=1e13", kh13, "s-"), ("kh=1e14", kh14, "^-")]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: recovered Pareto frontier (FPS vs penetration), labels = substeps
for name, data, m in series:
    pen = [d[1] for d in data]
    fps = [d[2] for d in data]
    axA.plot(pen, fps, m, label=name)
    for ss, p, f in data:
        axA.annotate(f"{ss}", (p, f), fontsize=7, xytext=(3, 3), textcoords="offset points")
axA.set_xscale("log")
axA.set_xlabel("max penetration (mm) — lower = better")
axA.set_ylabel("FPS")
axA.set_title("Recovered trade-off: low penetration needs\nhigh kh AND many substeps (labels = substeps)")
axA.legend(fontsize=9)
axA.grid(True, which="both", alpha=0.3)

# Panel B: penetration vs substeps shows the dt-cap per kh
for name, data, m in series:
    ss = [d[0] for d in data]
    pen = [d[1] for d in data]
    axB.plot(ss, pen, m, label=name)
axB.set_xscale("log")
axB.set_yscale("log")
axB.set_xlabel("substeps (1/timestep)")
axB.set_ylabel("max penetration (mm)")
axB.set_title("kh is inert at large dt (implicit stiffness cap);\nits benefit only appears as dt shrinks")
axB.legend(fontsize=9)
axB.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig("kh_dt_coupling.png", dpi=130)
print("saved kh_dt_coupling.png")
