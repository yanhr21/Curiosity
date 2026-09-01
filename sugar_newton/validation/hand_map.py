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

    python -m sugar_newton.validation.hand_map --out <dir> --frames 600
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

        # carriage: world -> (prismatic Y) -> (prismatic X) -> hand. Both position
        # controlled, so the trajectory is an input.
        lift = builder.add_body(mass=1.0, label="lift")
        builder.add_joint_prismatic(parent=-1, child=lift, axis=wp.vec3(0.0, 1.0, 0.0))
        self.hand_body = builder.add_body(mass=1.0, label="hand")
        builder.add_joint_prismatic(parent=lift, child=self.hand_body, axis=wp.vec3(1.0, 0.0, 0.0))

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
        self.plate_shape = builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.07, pad_top + 0.02, 0.0), wp.quat_identity()),
            hx=0.10, hy=0.015, hz=0.075, cfg=plate_cfg, label="plate",
        )
        self.contact_y = pad_top  # pads touch the plate underside when lift == 0.02

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))  # prescribed motion only
        self.model.request_contact_attributes("force")

        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="latest",
            contact_report=True,
            # 27 pads against one plate overflows the default iso buffers
            # ("iso subblock L1 overflow"), which silently drops contacts.
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                output_contact_surface=True, buffer_fraction=0.35
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

    def command(self, lift: float, drag: float) -> None:
        self.control.joint_target_q.assign(np.array([lift, drag], dtype=np.float32))

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
    seat = 0.02  # lift at which the pads just meet the plate

    drag = 0.0
    for i in range(args.frames):
        if i < approach:
            lift = seat * (i / max(1, approach - 1)) + args.press_depth * (i / max(1, approach - 1))
        else:
            lift = seat + args.press_depth
            if i >= hold:
                drag += args.drag_speed * args.dt
        scene.command(lift, drag)
        scene.step(args.dt)
        lift_cmd[i], drag_cmd[i] = lift, drag

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
        lift_cmd=lift_cmd, drag_cmd=drag_cmd, dt=args.dt,
        drag_speed=args.drag_speed, mu=args.mu, hold_frame=hold, **trace,
    )
    peak = trace["slip_velocity"][hold:].max()
    print(f"\nwrote {out / 'hand_map.npz'}")
    print(f"patches ever in contact : {int((trace['contact_count'] > 0).any(axis=0).sum())}/{n_patch}")
    print(f"peak normal load        : {trace['normal_load'].sum(axis=1).max():.2f} N")
    print(f"peak pressure           : {trace['peak_pressure'].max():.1f} Pa")
    print(f"peak slip velocity      : {peak:.4f} m/s  (commanded {args.drag_speed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
