# SPDX-License-Identifier: BSD-3-Clause
"""Is the grasp physics right? Weigh the cube with the tactile sensor.

A tactile channel can look plausible and be wrong by a constant. The check that catches
that is Newton's third law: with the hand held still and the cube resting in the palm,
the total contact force the sensor reports must equal the cube's weight, in direction as
well as magnitude. Nothing else in the scene is pushing it.

    sum over patches of (normal force + friction force)  ==  m * g

Note the sign. The reducer orients every contact force onto the PATCH, so what the
sensor reports is the force the cube exerts on the hand -- which points *down*, along
``m g``, not against it. Comparing against ``-m g`` reads a perfect result as 180 degrees
wrong, which is how this check was first misread.

This is the same check that validated the G1 scene at a ratio of 1.002, re-run here
because contact stiffness, hydroelastic ``kh`` and the solver settings have all changed
since -- and every one of them can move the answer while the picture still looks fine.

Reported, not just asserted: the ratio, the direction error, and how they vary with
``--ke``, so the cost of a soft contact is visible next to its speed.

    uv run python -m sugar_newton.validation.allegro_static --ke 1e3 1e5 1e6
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import warp as wp

from sugar_newton.validation.allegro_tactile import AllegroTactileScene


def weigh(scene, dt: float, substeps: int, settle: int, window: int) -> dict:
    """Settle the cube, then average the reported contact force over a window."""
    scene.reset()
    scene.rock = 0.0          # nothing may move: the only force left is gravity
    scene.amplitude = 0.0
    for _ in range(settle):
        scene.step(dt, substeps=substeps)

    cube_body = int(scene.model.shape_body.numpy()[scene.cube_shapes[0]])
    mass = float(scene.model.body_mass.numpy()[cube_body])
    # Model.gravity is a wp.array[wp.vec3] (one per world), not a tuple.
    g = np.asarray(scene.model.gravity.numpy(), dtype=float).reshape(-1, 3)[0]
    weight = mass * g

    tot, pen, live = [], [], []
    for _ in range(window):
        scene.step(dt, substeps=substeps)
        f = scene.tactile.normal_vec.numpy().sum(axis=0) + \
            scene.tactile.friction_vec.numpy().sum(axis=0)
        tot.append(f)
        pen.append(float(scene.tactile._peak_depth.numpy().max()))
        live.append(int((scene.tactile.count.numpy() > 0).sum()))
    f = np.mean(tot, axis=0)
    wnorm = np.linalg.norm(weight)
    cos = float(f @ weight / (np.linalg.norm(f) * wnorm)) if np.linalg.norm(f) > 0 else 0.0
    return dict(
        mass=mass, weight=wnorm,
        measured=float(np.linalg.norm(f)),
        ratio=float(np.linalg.norm(f) / wnorm) if wnorm > 0 else 0.0,
        angle=float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
        pen_mm=float(np.mean(pen)) * 1e3,
        live=float(np.mean(live)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ke", type=float, nargs="+", default=[1.0e3, 1.0e5, 1.0e6])
    ap.add_argument("--kd", type=float, default=1.0e2)
    ap.add_argument("--kh", type=float, default=1.0e10)
    ap.add_argument("--substeps", type=int, nargs="+", default=[8])
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--settle", type=int, default=200)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--grasp", type=float, default=0.34)
    ap.add_argument("--tol", type=float, default=0.10, help="allowed error in the weight ratio")
    args = ap.parse_args()

    wp.init()
    if not wp.get_device().is_cuda:
        print("ERROR: hydroelastic SDF is CUDA-only.")
        return 2
    dt = 1.0 / args.fps

    print(f"{'ke':>8} {'sub':>4} | {'weight N':>9} {'measured':>9} {'ratio':>7} "
          f"{'dir err':>8} | {'pen mm':>7} {'links':>6}")
    rows = []
    for ke in args.ke:
        scene = AllegroTactileScene(kh=args.kh, ke=ke, kd=args.kd,
                                    iterations=args.iterations)
        scene.grasp = args.grasp
        for sub in args.substeps:
            r = weigh(scene, dt, sub, args.settle, args.window)
            r["ke"], r["sub"] = ke, sub
            rows.append(r)
            print(f"{ke:>8.0e} {sub:>4} | {r['weight']:>9.4f} {r['measured']:>9.4f} "
                  f"{r['ratio']:>7.4f} {r['angle']:>7.2f}d | {r['pen_mm']:>7.3f} "
                  f"{r['live']:>6.1f}", flush=True)

    # A row with no contact is the cube having rolled off the palm, not a failed weighing:
    # with the drive off this grasp is a cradle, and over a few hundred steps it is
    # marginal. Those rows are excluded, and counted, rather than scored as zero.
    dropped = [r for r in rows if r["live"] == 0]
    if dropped:
        print(f"\n{len(dropped)}/{len(rows)} settings lost the cube before the window "
              f"(the drive is off in this test, so nothing holds it but the palm)")
    good = [r for r in rows if r["live"] > 0 and abs(r["ratio"] - 1.0) <= args.tol]
    if not good:
        print(f"\nFAIL: no setting weighed the cube to within {args.tol:.0%}. "
              f"Either the cube is not resting in the hand, or the reported contact force "
              f"does not balance gravity -- which would make every channel suspect.")
        return 1
    best = min(good, key=lambda r: r["pen_mm"])
    print(f"\nPASS: {len(good)}/{len(rows)} settings weigh the cube to within {args.tol:.0%}.")
    print(f"least penetration among them: --ke {best['ke']:.0e} --substeps {best['sub']} "
          f"-> ratio {best['ratio']:.4f}, direction off by {best['angle']:.2f} deg, "
          f"penetration {best['pen_mm']:.3f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
