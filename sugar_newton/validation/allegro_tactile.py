# SPDX-License-Identifier: BSD-3-Clause
"""Tactile sensing on Newton's Allegro hand: pressure, friction, slip velocity.

Built on ``newton/examples/robot/example_robot_allegro_hand.py`` -- an Allegro left hand
holding a cube, fingers driven on a sinusoid so the cube is continuously regrasped.
That motion is what makes it a tactile scene rather than a static one: the grasp loads
and unloads, and the cube slips against the pads as it does.

Every one of the hand's 18 collision links becomes a tactile patch, and the cube is the
counterpart. Channels come from :class:`sugar_newton.tactile.PatchTactile` over Newton's
own contacts -- no TacSL, no taxel grids, no threshold-based "slip detector". Slip is a
velocity in m/s, measured two independent ways.

The links are made hydroelastic so a contact *surface* exists; without it there is no
contact area and therefore no pressure, only force.

    uv run python -m sugar_newton.validation.allegro_tactile --out <dir> --frames 400
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.geometry import HydroelasticSDF
from sugar_newton.tactile.reducer import PatchTactile

RECORD = (
    "contact_count", "normal_load", "friction_load", "utilization_mean", "utilization_max",
    "slip_displacement", "slip_velocity", "gross_slip_fraction", "contact_area", "peak_pressure",
)


class AllegroTactileScene:
    def __init__(self, kh=1.0e8, sdf_res=48):
        newton.use_coord_layout_targets = True
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.default_shape_cfg.ke = 1.0e3
        b.default_shape_cfg.kd = 1.0e2
        b.default_shape_cfg.margin = 0.005
        b.default_shape_cfg.gap = 0.015

        asset = newton.utils.download_asset("wonik_allegro")
        b.add_usd(
            str(asset / "usd" / "allegro_left_hand_with_cube.usda"),
            xform=wp.transform(wp.vec3(0, 0, 0.5)),
            enable_self_collisions=False,
            ignore_paths=[".*Dummy", ".*CollisionPlane"],
            hide_collision_shapes=True,
        )

        # finger drive gains, exactly as the stock example
        for i in range(b.joint_dof_count - 6):
            b.joint_target_ke[i] = 150
            b.joint_target_kd[i] = 5
            b.joint_q[i] = b.joint_target_q[i] = 0.3
            if b.joint_label[i][-2:] == "_0":
                b.joint_q[i] = b.joint_target_q[i] = 0.6
            b.joint_target_mode[i] = int(JointTargetMode.POSITION)
            if b.joint_type[i] == newton.JointType.REVOLUTE:
                b.joint_armature[i] = 1e-2
        q = np.array(b.joint_q)
        q[-7:-4] += np.array([0.0, 0.0, 0.05])
        q[-4:] = wp.quat_rpy(0.3, 0.5, 0.1)
        b.joint_q = q.tolist()

        # Pressure and contact area are reductions over the contact SURFACE, so every
        # collider has to be hydroelastic. The USD ships the hand's 18 colliders as
        # GeoType.CONVEX_MESH, and Newton rejects hydroelastic on that type outright
        # (builder.py:5984) -- a convex mesh falls into the primitive branch, which
        # produces no SDF. So each convex collider is replaced by a real mesh collider
        # built from the same vertices with its own SDF, and the convex proxy is retired.
        # The hulls are 40-64 vertices and watertight, so the replacement is exact.
        self.patch_shapes, self.patch_labels, self.cube_shapes = [], [], []
        for s in list(range(b.shape_count)):
            if not b.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                continue
            body = b.shape_body[s]
            label = b.body_label[body].split("/")[-1] if 0 <= body < len(b.body_label) else "?"
            hydro = replace(
                b.default_shape_cfg, is_hydroelastic=True, kh=kh, density=0.0,
                has_shape_collision=True, mu=b.shape_material_mu[s], restitution=0.0,
            )
            if b.shape_type[s] == newton.GeoType.CONVEX_MESH:
                src = b.shape_source[s]
                m = newton.Mesh(
                    np.asarray(src.vertices, dtype=np.float32),
                    np.asarray(src.indices, dtype=np.int32).flatten(),
                    compute_inertia=False,
                )
                m.build_sdf(max_resolution=sdf_res, narrow_band_range=(-0.004, 0.004), margin=0.002)
                new_s = b.add_shape_mesh(
                    body=body, xform=b.shape_transform[s], mesh=m, scale=b.shape_scale[s],
                    cfg=hydro, label=f"{label}_tactile",
                )
                # Invisible: the USD's own visual mesh already draws this link, and two
                # near-identical surfaces render as z-fighting stripes.
                b.shape_flags[new_s] &= ~int(newton.ShapeFlags.VISIBLE)
                b.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)  # retire the hull
                s = new_s
            else:
                # the cube is a BOX; primitives take hydroelastic through cfg.sdf_*
                b.shape_material_kh[s] = kh
                b.shape_flags[s] |= int(newton.ShapeFlags.HYDROELASTIC)
                b.shape_sdf_max_resolution[s] = sdf_res
                b.shape_sdf_narrow_band_range[s] = (-0.004, 0.004)
            if "DexCube" in label:
                self.cube_shapes.append(s)
            else:
                self.patch_shapes.append(s)
                self.patch_labels.append(label)

        b.add_ground_plane()
        self.model = b.finalize()
        self.model.request_contact_attributes("force")

        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="latest",   # "sticky" replays anchors into the solve; that is a
            contact_report=True,         # physics change, and Plan 16 measures without perturbing
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                output_contact_surface=True, buffer_fraction=0.4
            ),
        )
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton", integrator="implicitfast",
            njmax=800, nconmax=min(1000, self.contacts.rigid_contact_max),
            impratio=20.0, cone="elliptic", iterations=100, ls_iterations=50,
            use_mujoco_contacts=False,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.tactile = PatchTactile(self.model, self.patch_shapes, self.cube_shapes)
        self.n_dof = self.model.joint_dof_count - 6
        self.limit_lo = self.model.joint_limit_lower.numpy()
        self.limit_hi = self.model.joint_limit_upper.numpy()
        self.t = 0.0
        self.grasp = 0.3  # finger closing target; raise it to engage more links

    def drive(self, dt: float) -> None:
        """Stock example's sinusoid: each joint leads the last, so the grasp breathes."""
        tgt = self.control.joint_target_q.numpy()
        i = np.arange(self.n_dof)
        tgt[: self.n_dof] = np.clip(
            np.sin(self.t + i * 0.6) * 0.08 + self.grasp,
            self.limit_lo[: self.n_dof], self.limit_hi[: self.n_dof]
        )
        self.control.joint_target_q.assign(tgt)
        self.t += dt

    def step(self, dt: float, substeps: int = 8) -> None:
        self.drive(dt)
        sub = dt / substeps
        self.pipeline.collide(self.state_0, self.contacts)
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)   # fills contacts.force
        self.tactile.update(
            self.state_0, self.contacts,
            contact_surface=self.pipeline.hydroelastic_sdf.get_contact_surface(),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--kh", type=float, default=1.0e8)
    ap.add_argument("--render", action="store_true", help="also write scene PNGs (needs Xvfb)")
    # Framed from the bodies themselves. The USD carries its own offset on top of the
    # spawn transform, so the hand actually sits near z = 1.0, not the 0.5 in the
    # add_usd call -- a hand-written camera aimed at the floor below it.
    ap.add_argument("--cam-offset", type=float, nargs=3, default=(0.34, 0.26, 0.13),
                    help="camera position relative to the hand+cube centroid [m]")
    ap.add_argument("--grasp", type=float, default=0.40,
                    help="finger closing target [rad]; higher engages more links")
    args = ap.parse_args()

    wp.init()
    if not wp.get_device().is_cuda:
        print("ERROR: hydroelastic SDF is CUDA-only.")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scene = AllegroTactileScene(kh=args.kh)
    scene.grasp = args.grasp
    n = len(scene.patch_shapes)
    print(f"patches={n} ({', '.join(scene.patch_labels)})", flush=True)
    print(f"counterpart cube shapes={scene.cube_shapes}", flush=True)

    viewer = None
    if args.render:
        import math
        import os

        import pyglet

        if os.environ.get("G1_XVFB") != "1":
            pyglet.options["headless"] = True
        from newton.viewer import ViewerGL

        viewer = ViewerGL(headless=os.environ.get("G1_XVFB") != "1")
        viewer.set_model(scene.model)
        viewer.show_hydro_contact_surface = True  # off by default; the call is a no-op without it
        newton.eval_fk(scene.model, scene.state_0.joint_q, scene.state_0.joint_qd, scene.state_0)
        p = scene.state_0.body_q.numpy()[:, :3]
        centre = 0.5 * (p.min(axis=0) + p.max(axis=0))
        cam = centre + np.asarray(args.cam_offset, dtype=float)
        d = centre - cam
        d /= np.linalg.norm(d)
        # Z-up convention, straight out of Camera._set_orientation_from_direction:
        # pitch = asin(dz), yaw = atan2(dy, dx). Deriving both from the look direction
        # is what keeps the framing correct when the asset moves.
        viewer.set_camera(
            wp.vec3(*cam.tolist()),
            math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0)))),
            math.degrees(math.atan2(float(d[1]), float(d[0]))),
        )
        print(f"camera {np.round(cam,3)} -> centre {np.round(centre,3)}", flush=True)
        (out / "frames").mkdir(exist_ok=True)

    dt = 1.0 / args.fps
    trace = {k: np.zeros((args.frames, n), dtype=np.float32) for k in RECORD}
    for i in range(args.frames):
        scene.step(dt)
        ch = scene.tactile.to_numpy()
        for k in RECORD:
            trace[k][i] = ch[k]
        if viewer is not None:
            from PIL import Image

            viewer.begin_frame(i * dt)
            viewer.log_state(scene.state_0)
            viewer.log_hydro_contact_surface(
                scene.pipeline.hydroelastic_sdf.get_contact_surface(), penetrating_only=True
            )
            viewer.end_frame()
            Image.fromarray(viewer.get_frame().numpy()).save(out / "frames" / f"f{i:05d}.png")
        if i % 50 == 0:
            print(f"  frame {i:4d}  live={int((ch['contact_count'] > 0).sum()):2d}/{n}  "
                  f"N={ch['normal_load'].sum():7.2f}  p_max={ch['peak_pressure'].max():9.1f} Pa  "
                  f"slip_max={ch['slip_velocity'].max():.4f} m/s", flush=True)

    np.savez(out / "allegro_tactile.npz", labels=np.array(scene.patch_labels), dt=dt, **trace)
    ever = int((trace["contact_count"] > 0).any(axis=0).sum())
    print(f"\nwrote {out / 'allegro_tactile.npz'}")
    print(f"patches ever in contact : {ever}/{n}")
    print(f"peak normal load        : {trace['normal_load'].sum(axis=1).max():.2f} N")
    print(f"peak contact area       : {trace['contact_area'].max() * 1e4:.3f} cm^2")
    print(f"peak pressure           : {trace['peak_pressure'].max():.1f} Pa")
    print(f"peak friction load      : {trace['friction_load'].max():.2f} N")
    print(f"peak slip velocity      : {trace['slip_velocity'].max():.4f} m/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
