# SPDX-License-Identifier: Apache-2.0
"""Analytic validator for tactile channels 9-10 — contact_area and peak_pressure.

A 10 cm cube seated flat on a ramp is the whole point of this scene: the contact is
planar and the pressure over it is uniform, so all three quantities are pinned by
statics with nothing to tune.

    normal_load   = m g cos(theta)                       [already asserted by incline.py]
    contact_area ~= the block's own face, 0.1 x 0.1 m
    peak_pressure ~= normal_load / contact_area

The third is the assertion that earns this file, and it already paid for itself. The
first implementation followed Plan 16 section 4 literally, ``kh * max(0, -depth)``, and
this test rejected it two ways: the sign is backwards for this Newton version
(``contact_surface_depth`` is positive where the shapes overlap), and once corrected,
``integral(kh * depth) dA`` came to 328.8 N against a true normal load of 4.886 N.
``kh * depth`` is the hydroelastic model's law, but under ``use_mujoco_contacts=False``
the normal force is MuJoCo's constraint solve; the surface supplies geometry, not force.
The channel now takes its shape from the depth field and its magnitude from the solved
load, so it integrates to that load by construction.

The ratio is read as a band rather than an equality because a block on a slope carries
more load at its downhill edge -- the same reason ``utilization_max`` exceeds
``utilization_mean`` in the sticking cases. A uniform field would read 1.0.

Hydroelastic SDF construction is CUDA-only, so this needs a GPU.

    uv run python -m sugar_newton.validation.pressure
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import warp as wp

from sugar_newton.validation.incline import SLIDER_HALF, SLIDER_MASS, InclineScene

FACE_AREA = (2.0 * SLIDER_HALF) ** 2  # the cube face that rests on the ramp [m^2]
GRAVITY = 9.81


def run_case(theta_deg: float, mu: float, steps: int, dt: float, window: int = 30) -> dict:
    scene = InclineScene(theta_deg, mu=mu, hydroelastic=True)
    for _ in range(steps):
        scene.step(dt)

    load = area = peak = 0.0
    n = 0
    for _ in range(window):
        scene.step(dt)
        ch = scene.tactile.to_numpy()
        if int(ch["contact_count"][0]) == 0 or float(ch["contact_area"][0]) <= 0.0:
            continue
        n += 1
        load += float(ch["normal_load"][0])
        area += float(ch["contact_area"][0])
        peak += float(ch["peak_pressure"][0])
    if n:
        load, area, peak = load / n, area / n, peak / n

    return {
        "theta": theta_deg,
        "normal_load": load,
        "expected_load": SLIDER_MASS * GRAVITY * math.cos(math.radians(theta_deg)),
        "contact_area": area,
        "peak_pressure": peak,
        "mean_pressure": load / area if area > 0 else 0.0,
        "steps_with_surface": n,
        "surface_available": bool(scene.tactile.contact_surface_available),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thetas", type=float, nargs="+", default=[5.0, 12.0, 20.0])
    ap.add_argument("--mu", type=float, default=0.9, help="high enough that every case sticks")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=1.0 / 240.0)
    ap.add_argument("--area-band", type=float, nargs=2, default=[0.5, 1.5])
    ap.add_argument("--pressure-band", type=float, nargs=2, default=[0.8, 2.5])
    args = ap.parse_args()

    wp.init()
    if not wp.get_device().is_cuda:
        print("SKIPPED — hydroelastic SDF construction is CUDA-only; this is a CPU device.")
        print("  Run inside the CUDA container. Channels 9-10 have no meaning without a")
        print("  contact surface, and the surface is what needs the GPU.")
        return 0

    print(f"device={wp.get_device()}  mu={args.mu}  face area={FACE_AREA * 1e4:.2f} cm^2")
    print()
    header = f"{'theta':>7} {'N meas':>9} {'N exp':>9} {'area cm2':>9} {'p_peak Pa':>11} {'p_mean Pa':>11} {'ratio':>7} {'n':>4}"
    print(header)
    print("-" * len(header))

    results = [run_case(t, args.mu, args.steps, args.dt) for t in args.thetas]
    for r in results:
        ratio = r["peak_pressure"] / r["mean_pressure"] if r["mean_pressure"] > 0 else float("nan")
        print(
            f"{r['theta']:7.2f} {r['normal_load']:9.4f} {r['expected_load']:9.4f} "
            f"{r['contact_area'] * 1e4:9.3f} {r['peak_pressure']:11.2f} {r['mean_pressure']:11.2f} "
            f"{ratio:7.3f} {r['steps_with_surface']:4d}"
        )
    print()

    failures: list[str] = []
    for r in results:
        t = r["theta"]
        if not r["surface_available"]:
            failures.append(f"theta={t}: no contact surface reached the reducer")
            continue
        if r["steps_with_surface"] == 0:
            failures.append(f"theta={t}: contact_area stayed zero for every step of the window")
            continue
        if not np.isclose(r["normal_load"], r["expected_load"], rtol=0.05):
            failures.append(f"theta={t}: normal load {r['normal_load']:.4f} vs {r['expected_load']:.4f}")
        lo, hi = args.area_band
        if not (lo * FACE_AREA <= r["contact_area"] <= hi * FACE_AREA):
            failures.append(
                f"theta={t}: contact_area {r['contact_area'] * 1e4:.3f} cm^2 outside "
                f"[{lo}, {hi}] x the {FACE_AREA * 1e4:.2f} cm^2 block face"
            )
        lo, hi = args.pressure_band
        ratio = r["peak_pressure"] / r["mean_pressure"] if r["mean_pressure"] > 0 else float("inf")
        if not (lo <= ratio <= hi):
            failures.append(
                f"theta={t}: peak_pressure / (N/A) = {ratio:.3f} outside [{lo}, {hi}] — "
                f"the kh side of -kh*depth is the first thing to check"
            )

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED — contact_area matches the block face and peak_pressure matches N/A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
