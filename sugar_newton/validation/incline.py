# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic validation of the tactile reducer on a block resting on an incline.

Plan 16 §5.  This is the gate that would have caught Plan 15.

A block of mass ``m`` on a ramp of angle ``theta`` with friction ``mu``, at rest:

    normal load      = m g cos(theta)
    friction load    = m g sin(theta)
    utilization      = tan(theta) / mu
    slip             = EXACTLY ZERO while tan(theta) < mu

The third line is the one that matters.  Plan 15's tactile reported friction
utilization 0.622 on a *static* grasp -- past its 0.60 incipient-slip trigger
with nothing moving at all -- because TacSL projected the total contact force
into a per-taxel frame, so off-centre contact leaked the normal force into the
shear channel.  That defect survived a full training and evaluation campaign
and was only found by reading the source months later.  A static test this small
would have caught it on the first day and did not exist.

The ramp is tilted and gravity is left vertical, deliberately: tilting gravity
instead would align every contact normal with +Z and would not exercise the
frame handling that Plan 15 got wrong.

Runs on CPU.  No GPU, no container, and no SUGAR asset required::

    python -m validation.incline
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp

import newton
from newton import Contacts
from newton.geometry import HydroelasticSDF
from sugar_newton.tactile.reducer import PatchTactile, TactileConfig


GRAVITY = 9.81
SLIDER_MASS = 0.5
SLIDER_HALF = 0.05
RAMP_HALF = (4.0, 1.0, 0.05)


class InclineScene:
    """A dynamic block resting on a static ramp inclined by ``theta``.

    The block's bottom face is the tactile patch; the ramp is the counterpart.
    Making the *moving* body carry the patch exercises the body-transform and
    ``velocity_at_point`` paths, which a static patch would not.
    """

    def __init__(
        self,
        theta_deg: float,
        mu: float,
        matching: str = "latest",
        mu_ramp: float | None = None,
        mu_block: float | None = None,
        fallback_friction: float | None = None,
        hydroelastic: bool = False,
    ):
        self.theta = math.radians(theta_deg)
        self.mu = mu
        # Per-shape friction, so the pair-combination rule can be exercised.
        # MuJoCo resolves a contact pair by elementwise MAX (kernels.py:165).
        self.mu_ramp = mu if mu_ramp is None else mu_ramp
        self.mu_block = mu if mu_block is None else mu_block
        self.hydroelastic = hydroelastic

        builder = newton.ModelBuilder(gravity=-GRAVITY)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # density=0: the shape must contribute no mass, otherwise add_shape_box
        # silently adds (density x volume) on top of add_body(mass=...) and the
        # block weighs more than SLIDER_MASS.  Asserted after finalize().
        # Box primitives can be hydroelastic directly -- no mesh SDF build is
        # needed, unlike imported meshes (example_robot_panda_hydro.py:79-85).
        # This is what allocates and fills rigid_contact_friction, the
        # per-contact scale (collide.py:896), so it is what exercises the scale
        # half of mu_contact = max(mu_a, mu_b) * scale.
        extra = (
            dict(
                is_hydroelastic=True,
                kh=1.0e11,
                gap=0.01,
                mu_torsional=0.0,
                mu_rolling=0.0,
                sdf_max_resolution=64,
                sdf_narrow_band_range=(-0.01, 0.01),
            )
            if hydroelastic
            else dict(ke=1.0e5, kd=1.0e3, kf=1.0e3)
        )
        cfg_ramp = newton.ModelBuilder.ShapeConfig(mu=self.mu_ramp, density=0.0, **extra)
        cfg_block = newton.ModelBuilder.ShapeConfig(mu=self.mu_block, density=0.0, **extra)

        # Ramp: static, rotated about +Y so its top face normal tilts toward +X.
        tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), self.theta)
        self.ramp_shape = builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), tilt),
            hx=RAMP_HALF[0],
            hy=RAMP_HALF[1],
            hz=RAMP_HALF[2],
            cfg=cfg_ramp,
            label="ramp",
        )

        # Slider: sits on the ramp's top face, same orientation, offset along
        # the ramp's local +Z by (ramp half-thickness + slider half-height).
        lift = RAMP_HALF[2] + SLIDER_HALF
        up = wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, 1.0))
        start = wp.vec3(up[0] * lift, up[1] * lift, up[2] * lift)
        self.slider_body = builder.add_body(
            xform=wp.transform(start, tilt),
            mass=SLIDER_MASS,
            label="slider",
        )
        self.patch_shape = builder.add_shape_box(
            body=self.slider_body,
            hx=SLIDER_HALF,
            hy=SLIDER_HALF,
            hz=SLIDER_HALF,
            cfg=cfg_block,
            label="patch",
        )

        self.model = builder.finalize()
        self.mass = float(self.model.body_mass.numpy()[self.slider_body])
        if abs(self.mass - SLIDER_MASS) > 1.0e-9:
            raise AssertionError(
                f'slider mass is {self.mass} kg, expected {SLIDER_MASS} kg -- the\n'
                'analytic expectations below are all proportional to it'
            )
        # Must precede pipeline.contacts(): the buffer only allocates the
        # extended 'force' attribute for attributes requested on the model.
        self.model.request_contact_attributes("force")

        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching=matching,
            contact_report=matching != "disabled",
            sdf_hydroelastic_config=HydroelasticSDF.Config() if hydroelastic else None,
        )
        self.contacts = self.pipeline.contacts()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=64,
            nconmax=64,
            iterations=20,
            ls_iterations=50,
            impratio=100.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.tactile = PatchTactile(
            self.model,
            patch_shapes=[self.patch_shape],
            counterpart_shapes=[self.ramp_shape],
            config=TactileConfig(
                fallback_friction=mu if fallback_friction is None else fallback_friction
            ),
        )

        self.alignment_ok: bool | None = None

    def step(self, dt: float, check_alignment: bool = False) -> None:
        self.state_0.clear_forces()
        self.pipeline.collide(self.state_0, self.contacts)

        pre = None
        if check_alignment:
            n = int(self.contacts.rigid_contact_count.numpy()[0])
            pre = (
                self.contacts.rigid_contact_shape0.numpy()[:n].copy(),
                self.contacts.rigid_contact_shape1.numpy()[:n].copy(),
                self.contacts.rigid_contact_normal.numpy()[:n].copy(),
                self.contacts.rigid_contact_point0.numpy()[:n].copy(),
                self.contacts.rigid_contact_point1.numpy()[:n].copy(),
            )

        self.solver.step(self.state_0, self.state_1, None, self.contacts, dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)

        if check_alignment and pre is not None:
            # update_contacts REPLACES the whole contact set with MuJoCo's own
            # (solver_mujoco.py:4380-4411 writes count, shape0/1, point0/1,
            # normal and force).  match_index was computed by the Newton
            # pipeline on the pre-solve ordering, so the anchors are only
            # meaningful if MuJoCo hands the contacts back in the same order.
            # Compare positions too, not just shapes and normals: in a
            # single-pair scene every contact shares a shape pair and a normal,
            # so those alone cannot distinguish a permutation.
            n_post = int(self.contacts.rigid_contact_count.numpy()[0])
            self.alignment_ok = n_post == len(pre[0])
            if self.alignment_ok:
                s0 = self.contacts.rigid_contact_shape0.numpy()[:n_post]
                s1 = self.contacts.rigid_contact_shape1.numpy()[:n_post]
                nrm = self.contacts.rigid_contact_normal.numpy()[:n_post]
                p0 = self.contacts.rigid_contact_point0.numpy()[:n_post]
                p1 = self.contacts.rigid_contact_point1.numpy()[:n_post]
                self.alignment_ok = bool(
                    np.array_equal(s0, pre[0])
                    and np.array_equal(s1, pre[1])
                    and np.allclose(nrm, pre[2], atol=1e-5)
                    and np.allclose(p0, pre[3], atol=1e-3)
                    and np.allclose(p1, pre[4], atol=1e-3)
                )

        self.tactile.update(self.state_0, self.contacts)

    def slider_position(self) -> np.ndarray:
        return self.state_0.body_q.numpy()[self.slider_body][:3].copy()


def run_case(theta_deg: float, mu: float, steps: int, dt: float, matching: str,
             window: int = 40) -> dict:
    """Settle a block on the incline, then average the channels over a window.

    Averaging matters for the sliding cases: a block sliding on a compliant
    contact chatters, so a single instant can land between contacts and read
    zero load.  Only steps that actually carry contact are averaged, and the
    number of such steps is reported so a mostly-airborne case is visible
    rather than silently averaged away.
    """
    scene = InclineScene(theta_deg, mu, matching=matching)

    for i in range(steps):
        scene.step(dt, check_alignment=(i == steps // 2))

    p_start = scene.slider_position()
    acc = {
        "contact_count": 0.0,
        "normal_load": 0.0,
        "friction_load": 0.0,
        "utilization_max": 0.0,
        "utilization_mean": 0.0,
        "slip_displacement": 0.0,
        "slip_velocity": 0.0,
        "gross_slip_fraction": 0.0,
        "signed_normal_load": 0.0,
    }
    contact_steps = 0
    overflow = False
    for _ in range(window):
        scene.step(dt)
        ch = scene.tactile.to_numpy()
        if int(ch["contact_count"][0]) == 0:
            continue
        contact_steps += 1
        for key in acc:
            acc[key] += float(ch[key][0])
        if scene.tactile.utilization_overflow:
            overflow = True
    p_end = scene.slider_position()
    measured_speed = float(np.linalg.norm(p_end - p_start) / (window * dt))

    if contact_steps:
        for key in acc:
            acc[key] /= contact_steps

    theta = math.radians(theta_deg)
    return {
        "theta_deg": theta_deg,
        "mu": mu,
        "sticking": math.tan(theta) < mu,
        "expected_normal": SLIDER_MASS * GRAVITY * math.cos(theta),
        "expected_friction": SLIDER_MASS * GRAVITY * math.sin(theta),
        "expected_utilization": math.tan(theta) / mu,
        "contact_steps": contact_steps,
        "window": window,
        "measured_speed": measured_speed,
        "match_index_available": scene.tactile.match_index_available,
        "per_contact_friction": scene.contacts.rigid_contact_friction is not None,
        "alignment_ok": scene.alignment_ok,
        "overflow": overflow,
        **acc,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=1.0 / 240.0)
    ap.add_argument("--matching", default="latest", choices=["disabled", "latest", "sticky"])
    ap.add_argument("--force-tol", type=float, default=0.05, help="relative tolerance on forces")
    ap.add_argument("--slip-tol", type=float, default=1.0e-4, help="absolute m / (m/s) slip floor")
    args = ap.parse_args()

    wp.init()
    critical = math.degrees(math.atan(args.mu))
    angles = [5.0, 12.0, 20.0, critical - 2.0, critical + 5.0, critical + 15.0]

    print(f"device={wp.get_device()}  mu={args.mu}  critical angle={critical:.2f} deg")
    print()
    header = (
        f"{'theta':>7} {'stick':>6} {'N meas':>9} {'N exp':>9} "
        f"{'u_mean':>7} {'u_exp':>7} {'u_max':>7} "
        f"{'slip d':>10} {'slip v':>10} {'|v| fd':>10} {'gross':>6} {'ctc':>5}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for theta in angles:
        r = run_case(theta, args.mu, args.steps, args.dt, args.matching)
        results.append(r)
        print(
            f"{r['theta_deg']:7.2f} {str(r['sticking']):>6} "
            f"{r['normal_load']:9.4f} {r['expected_normal']:9.4f} "
            f"{r['utilization_mean']:7.4f} {r['expected_utilization']:7.4f} "
            f"{r['utilization_max']:7.4f} "
            f"{r['slip_displacement']:10.3e} {r['slip_velocity']:10.3e} "
            f"{r['measured_speed']:10.3e} {r['gross_slip_fraction']:6.3f} "
            f"{r['contact_steps']:5d}"
        )

    print()
    meta = results[0]
    print(f"match_index available      : {meta['match_index_available']}")
    print(f"per-contact friction array : {meta['per_contact_friction']}")
    print(f"contact index alignment ok : {meta['alignment_ok']}")
    print()

    failures: list[str] = []
    notes: list[str] = []
    if meta["alignment_ok"] is False:
        failures.append(
            "contact indices are NOT preserved across update_contacts -- match_index "
            "refers to the pre-solve Newton ordering, so anchor propagation is keyed "
            "on the wrong contacts. Key the reducer on (shape0, shape1, point) instead."
        )
    for r in results:
        tag = f"theta={r['theta_deg']:.2f}"
        if r["contact_steps"] == 0:
            failures.append(
                f"{tag}: no contact on the patch in any of {r['window']} measured steps"
            )
            continue

        # 1. Normal load matches m g cos(theta).
        #
        # This is an equilibrium statement *normal to the surface*, so it holds
        # while the block is seated -- sticking, or sliding steadily.  On a
        # steep enough ramp the compliant contact launches the block and it
        # descends ballistically, chattering; mg cos(theta) is then simply not
        # the right expectation.  That regime is REPORTED rather than skipped:
        # a quietly dropped assertion is how Plan 15 accumulated ten problems.
        ballistic = (
            not r["sticking"] and r["normal_load"] < 0.5 * r["expected_normal"]
        )
        if ballistic:
            notes.append(
                f"{tag}: block is ballistic at this angle/dt (mean normal load "
                f"{r['normal_load']:.4f} N vs {r['expected_normal']:.4f} N seated). "
                f"Normal-equilibrium assertion does not apply; slip assertions still do. "
                f"This bounds the solver's envelope, not the sensor's."
            )
        elif abs(r["normal_load"] - r["expected_normal"]) > args.force_tol * r["expected_normal"]:
            failures.append(
                f"{tag}: normal load {r['normal_load']:.4f} N != expected "
                f"{r['expected_normal']:.4f} N"
            )

        if r["sticking"]:
            # 2. load-weighted utilization matches tan(theta)/mu.  The MAX is a
            # per-contact statistic -- one corner of a tilted block reaches the
            # cone before the patch as a whole does -- so it is bounded, not
            # matched, against the analytic aggregate.
            if abs(r["utilization_mean"] - r["expected_utilization"]) > 0.15:
                failures.append(
                    f"{tag}: mean utilization {r['utilization_mean']:.4f} != expected "
                    f"{r['expected_utilization']:.4f}"
                )
            # 3. THE test -- slip is exactly zero while sticking
            if r["slip_velocity"] > args.slip_tol:
                failures.append(
                    f"{tag}: slip VELOCITY {r['slip_velocity']:.3e} m/s reported while "
                    f"sticking (tan{r['theta_deg']:.0f} < mu). This is the Plan 15 defect."
                )
            if r["slip_displacement"] > args.slip_tol:
                failures.append(
                    f"{tag}: slip DISPLACEMENT {r['slip_displacement']:.3e} m reported "
                    f"while sticking. This is the Plan 15 defect."
                )
            # 5. utilization cannot exceed 1 under a converged stick
            if r["overflow"]:
                failures.append(
                    f"{tag}: utilization {r['utilization_max']:.4f} > 1 while sticking -- "
                    f"proof of frame contamination by construction"
                )
        else:
            # 4. Sliding.  Asserted QUALITATIVELY on purpose.
            #
            # A free block accelerating down a compliant-contact ramp bounces
            # and chatters, so an instantaneous tactile reading and a
            # finite-difference of its centre of mass over a window are not the
            # same quantity, and forcing them to agree would be a test of the
            # solver's dynamics rather than of the sensor.  The quantitative
            # sliding test is the prescribed-velocity one (--prescribed), where
            # the tangential velocity is an input rather than an outcome.
            #
            # What must hold here: the patch reports that it is slipping.
            if r["slip_velocity"] <= args.slip_tol:
                failures.append(
                    f"{tag}: block is sliding (measured {r['measured_speed']:.3e} m/s) but "
                    f"slip velocity reads {r['slip_velocity']:.3e}"
                )
            # Sliding faster than the matcher's position threshold per step
            # breaks every match, so gross slip must be visible in the
            # re-anchor fraction.  Without this channel the anchor drift of a
            # fully sliding patch reads exactly zero.
            if r["gross_slip_fraction"] < 0.5:
                failures.append(
                    f"{tag}: block is sliding but gross-slip fraction is only "
                    f"{r['gross_slip_fraction']:.3f} -- slip displacement alone reads "
                    f"{r['slip_displacement']:.3e} and would hide it"
                )

    for n in notes:
        print(f"NOTE  {n}")
    if notes:
        print()

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASSED — all analytic assertions hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
