# SPDX-License-Identifier: BSD-3-Clause
"""Drive the 27-patch hand against a plate and record the tactile map.

Prescribed motion, gravity off: the hand is carried by two position-controlled
prismatic joints, presses the palm into a static plate, holds, then drags along X at a
constant speed. Nothing is left to a controller or to gravity, so every channel has a
known cause:

    press  ->  normal load and pressure rise, contact area grows
    hold   ->  utilization sits below 1, slip stays at zero
    drag   ->  friction load saturates at mu * N, then the patch breaks away and
               slip velocity settles at the commanded drag speed

That last one also closes the TODO's "prescribed-velocity sliding scene for a
*quantitative* slip test": the answer is an input, not an outcome, so slip velocity can
be asserted rather than eyeballed.

Hydroelastic needs CUDA.

    uv run python -m sugar_newton.validation.hand_map --out <dir> --frames 600
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.geometry import HydroelasticSDF
from sugar_newton.hand.patches import PATCH_SPECS, add_hand_patches, load_hand_mesh, patch_names
from sugar_newton.tactile.reducer import PatchTactile

RECORD = (
    "contact_count", "normal_load", "friction_load", "utilization_mean", "utilization_max",
    "slip_displacement", "slip_velocity", "gross_slip_fraction", "contact_area", "peak_pressure",
)


class HandPlateScene:
    """One rubber hand on a 2-DoF carriage, pressed into a static plate."""

    def __init__(self, side="left", mu=0.6, press_depth=0.0015, drag_speed=0.05, kh=1.0e10):
        self.side, self.press_depth, self.drag_speed = side, press_depth, drag_speed

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = mu

        mesh = load_hand_mesh(side)
        verts = np.asarray(mesh.vertices, float)
        self.palm_y = float(verts[:, 1].max())

        # Kinematic free root: Newton's supported prescribed-motion path (the
        # same pattern as example_basic_conveyor.py).  A stiff dynamic load cell
        # below provides the reaction body, so contacts still have solved forces;
        # the hand trajectory itself is exact rather than an under-actuated PD
        # outcome.
        self.hand_body = builder.add_body(
            mass=1.0, is_kinematic=True, label="prescribed_hand"
        )
        hand_joint = next(
            j for j, child in enumerate(builder.joint_child) if child == self.hand_body
        )
        self.hand_q0 = int(builder.joint_q_start[hand_joint])
        self.hand_qd0 = int(builder.joint_qd_start[hand_joint])

        # pads: hydroelastic boxes. Box primitives take hydroelastic directly -- no mesh
        # SDF build -- which is what keeps 27 pads per hand affordable.
        # The hand itself: the real rubber-hand mesh, visual only. The 27 pads stand proud
        # of it and are what collides, so this is the shell the skin sits on -- without it
        # the scene renders as 27 floating rectangles rather than a hand.
        builder.add_shape_mesh(
            body=self.hand_body,
            mesh=newton.Mesh(
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.int32).flatten(),
                compute_inertia=False,
            ),
            cfg=replace(builder.default_shape_cfg, has_shape_collision=False,
                        is_hydroelastic=False, density=0.0),
            label=f"{side}_rubber_hand",
        )

        # SDF resolution is sized to each shape, not shared: a pad is ~24 mm across and a
        # plate ~200 mm, so one resolution would give the plate voxels coarser than the
        # narrow band and it would find no contact surface at all (TODO 16 B-open).
        # No `gap` -- MuJoCo only activates a contact below margin - gap.
        pad_cfg = replace(
            builder.default_shape_cfg, mu=mu, restitution=0.0, kh=kh,
            is_hydroelastic=True, density=0.0, has_shape_collision=True,
            mu_torsional=0.0, mu_rolling=0.0,
            sdf_max_resolution=64, sdf_narrow_band_range=(-0.003, 0.003),
        )
        self.patch_shapes = add_hand_patches(builder, self.hand_body, side, pad_cfg, mesh=mesh)

        # Same-body pads are permanently overlapping neighbours and Newton's pipeline
        # does not exclude same-body pairs. Without this every pad pair is a standing
        # constraint: 27 pads is 351 of them.
        for i, a in enumerate(self.patch_shapes):
            for b in self.patch_shapes[i + 1:]:
                builder.add_shape_collision_filter_pair(a, b)

        # the plate the palm presses into, sitting just above the highest pad
        pad_top = self.palm_y + 0.0049
        plate_cfg = replace(
            builder.default_shape_cfg, mu=mu, restitution=0.0, kh=kh,
            is_hydroelastic=True, density=0.0,
            mu_torsional=0.0, mu_rolling=0.0,
            sdf_max_resolution=256, sdf_narrow_band_range=(-0.006, 0.006),
        )
        seat = 0.02
        plate_half_height = 0.015
        plate_mass = 10.0
        self.plate_body = builder.add_link(
            mass=plate_mass,
            inertia=wp.mat33(
                0.0195, 0.0, 0.0,
                0.0, 0.0521, 0.0,
                0.0, 0.0, 0.0341,
            ),
            lock_inertia=True,
            label="load_cell_plate",
        )
        plate_centre = wp.vec3(0.07, pad_top + seat + plate_half_height, 0.0)
        plate_joint = builder.add_joint_prismatic(
            parent=-1,
            child=self.plate_body,
            parent_xform=wp.transform(plate_centre, wp.quat_identity()),
            axis=wp.vec3(0.0, 1.0, 0.0),
            target_ke=2.0e6,
            # Approximately critical for the 10-kg plate.  Underdamping here
            # leaks normal load-cell ringing into per-face tangential speed.
            target_kd=1.0e4,
            effort_limit=1.0e8,
            velocity_limit=10.0,
            actuator_mode=newton.JointTargetMode.POSITION,
            label="plate_load_cell",
        )
        builder.add_articulation([plate_joint], label="plate_load_cell")
        self.plate_shape = builder.add_shape_box(
            body=self.plate_body,
            # Put the plate underside at pad_top + seat.  The previous centre
            # omitted the 15-mm half-height, so contact began at 5 mm while the
            # command assumed 20 mm and produced a false 16-mm drive error.
            hx=0.10, hy=plate_half_height, hz=0.075,
            cfg=replace(plate_cfg, density=0.0), label="plate",
        )
        self.contact_y = pad_top + seat

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))  # prescribed motion only
        self.model.request_contact_attributes("force")

        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="latest",
            contact_report=True,
            # 27 pads against one plate overflows smaller broad/iso buffers, which
            # silently drops contacts and invalidates the quantitative slip gate.
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                output_contact_surface=True, buffer_fraction=1.0, buffer_mult_iso=4
            ),
        )
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=False, solver="newton",
            # nconmax must not exceed the pipeline's rigid_contact_max, or
            # update_contacts refuses to copy MuJoCo's set back (solver_mujoco.py:4374).
            integrator="implicitfast", cone="elliptic", njmax=2048, nconmax=1000,
            iterations=20, ls_iterations=50, impratio=1000.0,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.tactile = PatchTactile(self.model, self.patch_shapes, [self.plate_shape])
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def command(self, lift: float, drag: float, lift_speed: float, drag_speed: float) -> None:
        q = self.state_0.joint_q.numpy()
        qd = self.state_0.joint_qd.numpy()
        q[self.hand_q0:self.hand_q0 + 7] = np.array(
            [drag, lift, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        qd[self.hand_qd0:self.hand_qd0 + 6] = np.array(
            [drag_speed, lift_speed, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.state_0.joint_q.assign(q)
        self.state_0.joint_qd.assign(qd)
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )

    def step(self, dt: float) -> None:
        self.state_0.clear_forces()
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)
        self.tactile.update(
            self.state_0,
            self.contacts,
            contact_surface=self.pipeline.hydroelastic_sdf.get_contact_surface(),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--dt", type=float, default=1.0 / 240.0)
    ap.add_argument("--side", default="left")
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--press-depth", type=float, default=0.0015, help="palm indentation into the plate [m]")
    ap.add_argument("--drag-speed", type=float, default=0.05, help="commanded tangential speed [m/s]")
    args = ap.parse_args()

    wp.init()
    if not wp.get_device().is_cuda:
        print("ERROR: hydroelastic SDF is CUDA-only; run inside the CUDA container.")
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = HandPlateScene(args.side, args.mu, args.press_depth, args.drag_speed)
    n_patch = len(scene.patch_shapes)
    print(f"hand={args.side}  patches={n_patch}  plate_shape={scene.plate_shape}  "
          f"mu={args.mu}  drag={args.drag_speed} m/s", flush=True)

    # phase boundaries as fractions of the clip
    approach, hold, = int(0.25 * args.frames), int(0.40 * args.frames)
    trace = {k: np.zeros((args.frames, n_patch), dtype=np.float32) for k in RECORD}
    lift_cmd = np.zeros(args.frames, dtype=np.float32)
    drag_cmd = np.zeros(args.frames, dtype=np.float32)
    joint_q = np.zeros((args.frames, 2), dtype=np.float32)
    joint_qd = np.zeros((args.frames, 2), dtype=np.float32)
    seat = 0.02  # lift at which the pads just meet the plate

    drag = 0.0
    prev_lift = 0.0
    for i in range(args.frames):
        if i < approach:
            lift = seat * (i / max(1, approach - 1)) + args.press_depth * (i / max(1, approach - 1))
        else:
            lift = seat + args.press_depth
            if i >= hold:
                drag += args.drag_speed * args.dt
        lift_speed = (lift - prev_lift) / args.dt
        scene.command(lift, drag, lift_speed, args.drag_speed if i >= hold else 0.0)
        scene.step(args.dt)
        prev_lift = lift
        lift_cmd[i], drag_cmd[i] = lift, drag
        hand_q = scene.state_0.body_q.numpy()[scene.hand_body]
        hand_qd = scene.state_0.body_qd.numpy()[scene.hand_body]
        joint_q[i] = (hand_q[1], hand_q[0])
        joint_qd[i] = (hand_qd[1], hand_qd[0])

        ch = scene.tactile.to_numpy()
        for k in RECORD:
            trace[k][i] = ch[k]
        if i % 100 == 0:
            live = int((ch["contact_count"] > 0).sum())
            print(f"  frame {i:4d}  lift={lift:.4f} drag={drag:.4f}  patches in contact={live:2d}/{n_patch}  "
                  f"N={ch['normal_load'].sum():8.2f}  slip_v_max={ch['slip_velocity'].max():.4f}", flush=True)

    np.savez(
        out / "hand_map.npz",
        names=np.array(patch_names((args.side,))),
        centres=np.array([[s.center_x_m, s.center_z_m] for s in PATCH_SPECS]),
        extents=np.array([[s.width_m, s.length_m] for s in PATCH_SPECS]),
        angles=np.array([s.tangent_angle_deg for s in PATCH_SPECS]),
        lift_cmd=lift_cmd, drag_cmd=drag_cmd, joint_q=joint_q, joint_qd=joint_qd,
        dt=args.dt,
        drag_speed=args.drag_speed, mu=args.mu, hold_frame=hold, **trace,
    )
    drag_begin = min(args.frames - 1, hold + max(10, (args.frames - hold) // 5))
    live = trace["normal_load"][drag_begin:] > 1.0e-4
    frame_load = trace["normal_load"][drag_begin:].sum(axis=1)
    live_frames = frame_load > 1.0e-4
    weighted_slip = np.divide(
        (trace["slip_velocity"][drag_begin:] * trace["normal_load"][drag_begin:]).sum(axis=1),
        frame_load,
        out=np.zeros_like(frame_load),
        where=live_frames,
    )
    steady_slip = float(np.median(weighted_slip[live_frames])) if live_frames.any() else 0.0
    steady_gross = (
        float(np.median(trace["gross_slip_fraction"][drag_begin:][live])) if live.any() else 0.0
    )
    max_pos_error = float(np.abs(joint_q[approach:] - np.column_stack(
        (lift_cmd[approach:], drag_cmd[approach:])
    )).max())
    drag_vel = float(np.median(joint_qd[drag_begin:, 1]))
    peak = float(trace["slip_velocity"][hold:].max())
    ever_contact = int((trace["contact_count"] > 0).any(axis=0).sum())
    peak_load = float(trace["normal_load"].sum(axis=1).max())
    post_hold_contact = bool((trace["contact_count"][hold:] > 0).any())

    slip_tol = max(0.01, 0.20 * abs(args.drag_speed))
    failures = []
    if ever_contact == 0:
        failures.append("no tactile patch ever contacted the plate")
    if peak_load <= 1.0e-3:
        failures.append("peak normal load is zero")
    if not post_hold_contact:
        failures.append("contact was not sustained into hold/drag")
    if max_pos_error > 5.0e-4:
        failures.append(f"carriage position error {max_pos_error:.6f} m exceeds 0.0005 m")
    if abs(drag_vel - args.drag_speed) > slip_tol:
        failures.append(
            f"carriage drag velocity {drag_vel:.5f} m/s does not track {args.drag_speed:.5f} m/s"
        )
    if abs(steady_slip - abs(args.drag_speed)) > slip_tol:
        failures.append(
            f"load-weighted slip velocity {steady_slip:.5f} m/s does not match "
            f"{abs(args.drag_speed):.5f} m/s"
        )
    print(f"\nwrote {out / 'hand_map.npz'}")
    print(f"patches ever in contact : {ever_contact}/{n_patch}")
    print(f"peak normal load        : {peak_load:.2f} N")
    print(f"peak pressure           : {trace['peak_pressure'].max():.1f} Pa")
    print(f"peak slip velocity      : {peak:.4f} m/s  (commanded {args.drag_speed})")
    print(f"steady weighted slip    : {steady_slip:.4f} m/s")
    print(f"steady gross-slip frac  : {steady_gross:.4f}")
    print(f"carriage drag velocity  : {drag_vel:.4f} m/s")
    print(f"max carriage pos error  : {max_pos_error:.6f} m")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: prescribed hand-map contact and quantitative slip gates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
