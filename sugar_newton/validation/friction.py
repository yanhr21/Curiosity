# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Prove the friction channel reads the friction the solver actually used.

TODO 16 section B-open, item 1.  This is the test that closes audit #4.

Plan 15's ``friction_utilization`` divided by the *sensor's* fixed ``mu = 0.5``
-- the same constant TacSL had already used to cap the shear numerator -- while
training randomized the box's real friction over ``U[0.2, 0.8]``.  The channel
was therefore invariant to the quantity it was named after, and no test existed
that could tell.

Reading the coefficient out of the contact buffer is only a repair if it is
really the coefficient the solve used, so this file is written to *fail* under
each way of getting that wrong:

* **Falling back to a constant.**  ``--fallback`` is set to an absurd value
  (7.0 by default).  If the reducer ever falls through to it, every case reads
  the same near-zero utilization and the spread assertion fails.
* **Reading only one shape's material.**  Cases A and B are the same pair with
  the friction values *swapped between the two shapes*.  MuJoCo resolves a
  contact pair by elementwise ``max`` (``kernels.py:165``), so A and B must
  agree exactly.  Code that reads shape0's value, or shape1's, splits them.
* **Mistaking the per-contact scale for mu.**  ``rigid_contact_friction`` is a
  *scale* (default 1.0), not a coefficient: hydroelastic contact reduction
  writes it for moment matching (``contact_reduction_hydroelastic.py:885``) and
  MuJoCo multiplies the resolved material friction by it
  (``kernels.py:460-468``).  Code that returns the scale alone reads ~1.0 for
  every material, so all cases collapse and the spread assertion fails.

The scene is the incline of ``validation/incline.py``, held at a fixed
sub-critical angle so the block sticks and utilization has a closed form::

    utilization = tan(theta) / mu_pair,   mu_pair = max(mu_ramp, mu_block)

Runs on CPU::

    python -m sugar_newton.validation.friction
"""

from __future__ import annotations

import argparse
import math

from sugar_newton.validation.incline import InclineScene


# (label, mu_ramp, mu_block).  A and B are the same pair, swapped.
CASES: tuple[tuple[str, float, float], ...] = (
    ("A  high ramp / low block", 0.8, 0.3),
    ("B  low ramp / high block", 0.3, 0.8),
    ("C  both low            ", 0.3, 0.3),
    ("D  both high           ", 0.9, 0.9),
)


def run_case(
    mu_ramp: float,
    mu_block: float,
    theta_deg: float,
    steps: int,
    dt: float,
    fallback: float,
    hydroelastic: bool = False,
    window: int = 30,
) -> dict:
    scene = InclineScene(
        theta_deg,
        mu=max(mu_ramp, mu_block),
        mu_ramp=mu_ramp,
        mu_block=mu_block,
        fallback_friction=fallback,
        hydroelastic=hydroelastic,
    )
    for _ in range(steps):
        scene.step(dt)

    util = 0.0
    load = 0.0
    n = 0
    for _ in range(window):
        scene.step(dt)
        ch = scene.tactile.to_numpy()
        if int(ch["contact_count"][0]) == 0:
            continue
        n += 1
        util += float(ch["utilization_mean"][0])
        load += float(ch["normal_load"][0])
    if n:
        util /= n
        load /= n

    mu_pair = max(mu_ramp, mu_block)
    return {
        "mu_ramp": mu_ramp,
        "mu_block": mu_block,
        "mu_pair": mu_pair,
        "expected_utilization": math.tan(math.radians(theta_deg)) / mu_pair,
        "utilization": util,
        "normal_load": load,
        "contact_steps": n,
        "per_contact_scale": scene.contacts.rigid_contact_friction is not None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta", type=float, default=12.0, help="sub-critical for every case")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--dt", type=float, default=1.0 / 240.0)
    ap.add_argument(
        "--fallback",
        type=float,
        default=7.0,
        help="deliberately absurd: if the reducer falls back to it, the test fails",
    )
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument(
        "--hydroelastic",
        action="store_true",
        help="enable hydroelastic SDF contacts, which is what allocates and fills\n"
        "the per-contact friction SCALE array",
    )
    args = ap.parse_args()

    import warp as wp

    wp.init()
    if args.hydroelastic and not wp.get_device().is_cuda:
        # Loud, and deliberately NOT a pass.  Hydroelastic SDF builds use
        # wp.Volume.allocate_by_tiles and wp.Texture3D, which are CUDA-only
        # (builder.py finalize()).  The scale half of
        # mu_contact = max(mu_a, mu_b) * scale therefore cannot be exercised on
        # a CPU device at all -- and reporting that as a pass would be exactly
        # the kind of quiet gap the Plan 15 audit kept finding.
        print("SKIPPED — --hydroelastic requires a CUDA device; this is a CPU device.")
        print()
        print("  Hydroelastic SDF construction uses wp.Volume.allocate_by_tiles and")
        print("  wp.Texture3D, both CUDA-only. rigid_contact_friction (the per-contact")
        print("  SCALE) is only allocated when the pipeline has a hydroelastic config")
        print("  (collide.py:896), so the scale path needs a GPU. Run this inside the")
        print("  CUDA container to close TODO 16 section B-open item 1 in full.")
        print()
        print("  The material-mu half DOES run on CPU: drop --hydroelastic.")
        return 2

    print(f"device={wp.get_device()}  theta={args.theta} deg  fallback mu={args.fallback} (absurd on purpose)")
    print()
    header = f"{'case':<26} {'mu_ramp':>8} {'mu_blk':>7} {'mu_pair':>8} {'util':>8} {'util exp':>9} {'N':>8}"
    print(header)
    print("-" * len(header))

    results = []
    for label, mu_r, mu_b in CASES:
        r = run_case(
            mu_r, mu_b, args.theta, args.steps, args.dt, args.fallback, args.hydroelastic
        )
        r["label"] = label
        results.append(r)
        print(
            f"{label:<26} {r['mu_ramp']:8.2f} {r['mu_block']:7.2f} {r['mu_pair']:8.2f} "
            f"{r['utilization']:8.4f} {r['expected_utilization']:9.4f} {r['normal_load']:8.4f}"
        )

    print()
    print(f"per-contact friction scale array allocated : {results[0]['per_contact_scale']}")
    if not results[0]["per_contact_scale"]:
        print(
            "  -> box primitives, no hydroelastic SDF, so rigid_contact_friction is None\n"
            "     and the SCALE path is not exercised here. The material-mu path is."
        )
    print()

    failures: list[str] = []
    for r in results:
        tag = r["label"].strip()
        if r["contact_steps"] == 0:
            failures.append(f"{tag}: no contact")
            continue
        if abs(r["utilization"] - r["expected_utilization"]) > args.tol:
            failures.append(
                f"{tag}: utilization {r['utilization']:.4f} != tan(theta)/max(mu) = "
                f"{r['expected_utilization']:.4f}"
            )

    # The discriminating assertions.
    a, b, c, d = results
    if abs(a["utilization"] - b["utilization"]) > 1.0e-3:
        failures.append(
            f"swapping mu between the two shapes changed the reading "
            f"({a['utilization']:.4f} vs {b['utilization']:.4f}) -- the pair rule is MAX, "
            f"so it must be symmetric. The reducer is reading one shape, not the pair."
        )
    spread = max(r["utilization"] for r in results) - min(r["utilization"] for r in results)
    if spread < 0.1:
        failures.append(
            f"utilization spread across mu = 0.3 to 0.9 is only {spread:.4f} -- the channel "
            f"is not seeing the material at all. This is the Plan 15 defect (audit #4)."
        )
    if c["utilization"] <= d["utilization"]:
        failures.append(
            f"lower friction must give HIGHER utilization: mu=0.3 read "
            f"{c['utilization']:.4f}, mu=0.9 read {d['utilization']:.4f}"
        )

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PASSED — utilization tracks max(mu_a, mu_b); spread {spread:.4f} across mu 0.3-0.9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
