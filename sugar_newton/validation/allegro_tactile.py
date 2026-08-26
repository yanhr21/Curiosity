# SPDX-License-Identifier: BSD-3-Clause
"""Tactile sensing on Newton's Allegro hand: pressure, friction, slip velocity.

Built on ``newton/examples/robot/example_robot_allegro_hand.py`` -- an Allegro left hand
with a cube in its palm, wrist rocking and digits driven out of phase so the cube is
continuously worked. That motion is what makes it a tactile scene rather than a static
one: the grasp loads and unloads, and the cube slips against the pads as it does.

Every one of the hand's 18 collision links becomes a tactile patch, and the cube is the
counterpart. Channels come from :class:`sugar_newton.tactile.PatchTactile` over Newton's
own contacts -- no TacSL, no taxel grids, no threshold-based "slip detector". Slip is a
velocity in m/s, measured two independent ways. Alongside the per-link reduction,
:class:`sugar_newton.tactile.ContactField` records the same quantities at their native
per-face resolution, which is what
:mod:`sugar_newton.validation.compose_allegro_field` draws as a continuous map.

The links are made hydroelastic so a contact *surface* exists; without it there is no
contact area and therefore no pressure, only force.

Four settings are load-bearing and were each established by measurement, not taste --
see :meth:`AllegroTactileScene.rock_wrist`, :meth:`drive`, the ``ke``/``kh`` help text,
``allegro_grasp_sweep`` for the drive, and ``allegro_bench`` for the rest:

* the root joint's parent rotation is overwritten so the palm faces up,
* ``ke`` and ``kd`` move together; raising stiffness alone makes the contact
  springy enough to throw the cube (see the comment where they are set),
* ``kh`` is 1e10, not the 1e8 the shapes default to,
* the digits ease into the grasp before they start swinging.

    uv run python -m sugar_newton.validation.allegro_tactile --out <dir> --frames 300 \
        --grasp 0.34 --amplitude 0.08 --rate 1.4 --rock 0.25 --swing-mode sym --render
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode, ModelFlags
from newton.geometry import HydroelasticSDF
from sugar_newton.tactile.field import ContactField
from sugar_newton.tactile.reducer import PatchTactile

RECORD = (
    "contact_count", "normal_load", "friction_load", "utilization_mean", "utilization_max",
    "slip_displacement", "slip_velocity", "gross_slip_fraction", "contact_area", "peak_pressure",
)


def _quat_matrix(q) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion, as numpy -- the outline is host-side."""
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _xform_points(xf, pts: np.ndarray) -> np.ndarray:
    """Apply a (pos, quat_xyzw) transform to an (N, 3) array."""
    xf = np.asarray(list(xf), dtype=np.float32).reshape(7)
    return pts @ _quat_matrix(xf[3:]).T + xf[:3]


def _xform_points_batch(xfs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a per-point transform: ``xfs`` is (N, 7), ``pts`` is (N, 3)."""
    q = xfs[:, 3:]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # rotate by quaternion without materialising N 3x3 matrices
    u = np.stack([x, y, z], axis=1)
    t = 2.0 * np.cross(u, pts)
    return pts + w[:, None] * t + np.cross(u, t) + xfs[:, :3]


def _unxform_points(xf, pts: np.ndarray) -> np.ndarray:
    """Apply the inverse of a (pos, quat_xyzw) transform to an (N, 3) array."""
    xf = np.asarray(list(xf), dtype=np.float32).reshape(7)
    return (pts - xf[:3]) @ _quat_matrix(xf[3:])


def _sample_surface(verts: np.ndarray, tris: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Area-weighted point sample of a triangle mesh's surface.

    The silhouette drawn under the tactile map comes from these. Sampling the SURFACE
    rather than reusing the hull's own vertices matters: a convex hull carries ~50
    vertices, all at corners, which blur into speckle instead of into a hand.
    """
    rng = np.random.default_rng(seed)
    tri = verts[tris]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    total = area.sum()
    if total <= 0:
        return tri[:, 0].astype(np.float32)
    pick = rng.choice(len(tri), size=n, p=area / total)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return (tri[pick, 0] + u[:, None] * e1[pick] + v[:, None] * e2[pick]).astype(np.float32)


def _box_surface_points(hx: float, hy: float, hz: float, n: int, seed: int) -> np.ndarray:
    """A box as 12 triangles, then sampled like any other mesh."""
    c = np.array([[sx * hx, sy * hy, sz * hz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float32)
    tris = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],   # -x, +x
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],   # -y, +y
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],   # -z, +z
    ], dtype=np.int32)
    return _sample_surface(c, tris, n, seed)


class AllegroTactileScene:
    def __init__(self, kh=1.0e10, sdf_res=48, mu=None, cube_drop=0.05,
                 buffer_fraction=1.0, buffer_mult_iso=2, contact_surface=True,
                 want_field=True, iterations=100, ls_iterations=50, cone="elliptic",
                 njmax=800, ke=1.0e4, kd=3.2e2):
        newton.use_coord_layout_targets = True
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        # Contact stiffness and damping. `ke` -- not `kh` -- is what sets how far the cube
        # sinks in: Newton turns (ke, kd) into MuJoCo's solref and the constraint solve
        # produces the normal force from that, while `kh` only shapes the hydroelastic
        # pressure field. The stock example's 1e3 N/m is soft enough to see through a
        # fingertip.
        #
        # `ke` IS BOUNDED BY THE SUBSTEP. The contact's own period is ~sqrt(m/ke); with the
        # 0.216 kg cube that is 4.6 ms at ke = 1e4 and 1.5 ms at 1e5, against a 2.5 ms
        # substep at 50 Hz x 8. Measured over four identical runs each: ke = 1e4 held the
        # cube 4/4, ke = 1e5 held it 3/4 and threw it 26 m on the fourth. Penetration is
        # the same either way, so the stiffer setting buys nothing and risks everything.
        #
        # RAISE THEM TOGETHER. `convert_solref` (mujoco/kernels.py:192, called at :432)
        # gives a damping ratio of `(kd/2) * sqrt(factor/ke)`, so stiffening the contact
        # alone DIVIDES the damping by sqrt(ke): ke 1e3 -> 1e5 at a fixed kd takes the
        # ratio from ~1.5 (dead) to ~0.15 (springy), and a springy contact throws the
        # cube out of the hand -- measured at several metres of travel. kd must go up as
        # sqrt(ke) to hold the ratio.
        b.default_shape_cfg.ke = ke
        b.default_shape_cfg.kd = kd
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

        # THE thing the earlier versions of this scene were missing. The stock example
        # does not leave the hand in the USD's own orientation: it overwrites the root
        # joint's parent rotation with `hand_rotation`, which turns the palm up so the
        # cube is cradled. Without it the hand sits tilted, the cube balances on the edge
        # of the palm out of the thumb's reach, and any finger motion at all flicks it
        # onto the floor -- measured across 20 drive settings, every one of which dropped
        # it. It is not a camera pose; it is the grasp.
        self.root_joint = next(
            (i for i, lbl in enumerate(b.joint_label) if lbl.endswith("root_joint")), 0
        )
        self.hand_rotation = wp.normalize(
            wp.quat(0.21643, 0.706218, -0.648166, 0.185191)
        )
        root_xform = b.joint_X_p[self.root_joint]
        b.joint_X_p[self.root_joint] = wp.transform(root_xform.p, self.hand_rotation)

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
        # How far above its USD rest pose the cube starts. The stock example's 0.05 makes
        # it fall into the hand, and the landing is the single largest number in every
        # channel -- 12 MPa against a 3.7 MPa working peak. Lower it and the grasp is
        # measured instead of the impact.
        q = np.array(b.joint_q)
        q[-7:-4] += np.array([0.0, 0.0, cube_drop])
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
        self.outline_verts, self.outline_body = [], []
        for s in list(range(b.shape_count)):
            if not b.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                continue
            body = b.shape_body[s]
            label = b.body_label[body].split("/")[-1] if 0 <= body < len(b.body_label) else "?"
            hydro = replace(
                b.default_shape_cfg, is_hydroelastic=True, kh=kh, density=0.0,
                has_shape_collision=True, restitution=0.0,
                # The USD's own mu unless overridden. Rubber pads on a plastic cube run
                # 0.8-1.2; the asset ships something much lower, and a grasp that only
                # holds by squeezing is a grasp that squirts the cube out when it closes.
                mu=b.shape_material_mu[s] if mu is None else mu,
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
                if mu is not None:
                    b.shape_material_mu[s] = mu
                b.shape_material_kh[s] = kh
                b.shape_flags[s] |= int(newton.ShapeFlags.HYDROELASTIC)
                b.shape_sdf_max_resolution[s] = sdf_res
                b.shape_sdf_narrow_band_range[s] = (-0.004, 0.004)
            # Keep each collider's own vertices, in its BODY frame, to draw under the
            # field: a heat map floating in empty axes cannot be read as a hand.
            if b.shape_type[s] == newton.GeoType.BOX:
                hx, hy, hz = b.shape_scale[s]
                verts = _box_surface_points(hx, hy, hz, 1200, seed=s)
            else:
                src = b.shape_source[s]
                verts = _sample_surface(
                    np.asarray(src.vertices, dtype=np.float32)
                    * np.asarray(b.shape_scale[s], dtype=np.float32),
                    np.asarray(src.indices, dtype=np.int32).reshape(-1, 3), 400, seed=s,
                )
            self.outline_verts.append(_xform_points(b.shape_transform[s], verts))
            self.outline_body.append(np.full(len(verts), body, dtype=np.int32))

            if "DexCube" in label:
                self.cube_shapes.append(s)
            else:
                self.patch_shapes.append(s)
                self.patch_labels.append(label)

        self.outline_verts = np.concatenate(self.outline_verts).astype(np.float32)
        self.outline_body = np.concatenate(self.outline_body).astype(np.int32)

        # the frame every tactile map is expressed in: fingers move, the palm does not
        self.palm_body = next(
            (i for i, lbl in enumerate(b.body_label) if lbl.split("/")[-1] == "palm_link"), -1
        )

        b.add_ground_plane()
        self.model = b.finalize()
        self.model.request_contact_attributes("force")

        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="latest",   # "sticky" replays anchors into the solve; that is a
            contact_report=True,         # physics change, and Plan 16 measures without perturbing
            # buffer_fraction 0.4 overflows the L1 iso-subblock stage on this scene
            # ("iso subblock L1 overflow: 1789 > 1670"), and an overflow silently DROPS
            # contacts -- holes in the very field this module exists to measure.
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                output_contact_surface=contact_surface,
                buffer_fraction=buffer_fraction,
                buffer_mult_iso=buffer_mult_iso,
            ),
        )
        self.contacts = self.pipeline.contacts()
        self._solver_kwargs = dict(
            solver="newton", integrator="implicitfast",
            njmax=njmax, nconmax=min(1000, self.contacts.rigid_contact_max),
            impratio=20.0, cone=cone, iterations=iterations, ls_iterations=ls_iterations,
            use_mujoco_contacts=False,
        )
        self.solver = newton.solvers.SolverMuJoCo(self.model, **self._solver_kwargs)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.tactile = PatchTactile(self.model, self.patch_shapes, self.cube_shapes)
        self.field = ContactField(self.tactile, frame_body=self.palm_body) if want_field else None
        self.want_surface = contact_surface
        self.profile = False
        self.graph = None
        self.timings: dict[str, float] = {}
        self.n_dof = self.model.joint_dof_count - 6
        self.limit_lo = self.model.joint_limit_lower.numpy()
        self.limit_hi = self.model.joint_limit_upper.numpy()
        self.t = 0.0
        self.grasp = 0.40      # mean flexion target: how hard the digits close
        self.spread = None     # abduction target; None means "same as grasp"
        self.thumb_oppose = None    # thumb _0 target; None means "same as spread"
        self.amplitude = 0.10  # per-digit flexion swing [rad]
        self.rate = 1.4        # rad/s
        self.settle = 1.0      # s before the swing starts
        self.drop = 0.0        # s of doing nothing first, while the cube falls in
        self.close_time = None # s to close over; None means "all the way to settle"
        self.swing_mode = "close"   # "close" = one-sided push; "sym" = about the hold
        self.rock = 0.10       # wrist rock about X [rad]; the stock example's own value
        self.ramp = 0.5        # s to fade the swing in, so nothing is impulsive

        # Only the flexion dofs SWING. The ``_0`` dof of each digit is its spread (and,
        # for the thumb, its opposition); it is still driven to a constant ``spread``
        # target, because leaving those four dofs at the USD's own pose measurably drops
        # the cube -- the fingers splay and it falls straight through. What they must not
        # do is oscillate with the flexion dofs.
        # ``joint_label`` is per JOINT and includes the fixed root, mount and biotac-tip
        # joints, which carry no coordinates at all -- indexing it by dof (as the stock
        # example does) lands on the wrong joint and marks flexion dofs as spread. The
        # coordinate layout is the only correct bridge, so it is used.
        qs = self.model.joint_q_start.numpy()
        self.dof_labels = []
        for j, lbl in enumerate(self.model.joint_label):
            self.dof_labels += [lbl.split("/")[-1]] * int(qs[j + 1] - qs[j])
        self.dof_labels = self.dof_labels[: self.n_dof]

        self.is_flex = np.array([not lbl.endswith("_0") for lbl in self.dof_labels], dtype=bool)
        self.dof_digit_name = [lbl.rsplit("_joint_", 1)[0] for lbl in self.dof_labels]
        names = self.dof_digit_name
        order = list(dict.fromkeys(names))
        self.digit = np.array([order.index(x) for x in names], dtype=np.int32)
        self.n_digits = max(len(order), 1)
        self.q_init = self.model.joint_q.numpy()[: self.n_dof].copy()
        self._is_thumb = np.array([x == "thumb" for x in self.dof_digit_name], dtype=bool)
        self.hold_digits: tuple[str, ...] = ()   # digits that clamp instead of swinging

    def _swing_mask(self) -> np.ndarray:
        """Which dofs actually oscillate: flexion dofs of the non-clamping digits.

        Holding a digit still is how a hand turns an object -- the thumb opposes while
        the fingers walk the cube around against it. Swinging all four digits at once
        means the cube is never held by a stationary surface, and it escapes.
        """
        hold = np.isin(self.dof_digit_name, self.hold_digits)
        return (self.is_flex & ~hold).astype(np.float64)

    def reset(self, rebuild_solver: bool = True) -> None:
        """Back to the built pose, so one process can try several drive settings.

        The solver is rebuilt by default, and that is not paranoia: ``SolverMuJoCo``
        carries warm-start and contact state that no amount of resetting the *states*
        touches. With it reused, a drive setting that survived 300 frames in its own
        process was scored as having dropped the cube when it ran fifth in a sweep.
        Pass ``rebuild_solver=False`` only when timing something where that does not
        matter.
        """
        if rebuild_solver:
            self.solver = newton.solvers.SolverMuJoCo(self.model, **self._solver_kwargs)
            self.graph = None
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.tactile.reset()
        self.t = 0.0

    def drive(self, dt: float) -> None:
        """Settle into the grasp, then work the cube with the digits out of phase.

        The stock example moves every joint on one small sinusoid, which only makes the
        whole grasp breathe in place -- the cube barely moves and the tactile field is
        nearly static. Here each digit gets its own phase, so while one pushes the others
        give: the cube rolls in the hand and the contact patches migrate across the pads,
        which is the regime slip velocity is worth measuring in.

        The swing is fenced by two guards learned the expensive way. It rides on the
        *flexion* dofs only (see ``is_flex``), and it fades in over ``ramp`` seconds after
        a ``settle`` hold. Without either, the hand opens and closes hard enough on the
        first frame to launch the cube -- measured 1.15 m of travel and 178 deg of
        rotation in 3 s, i.e. a throw, not a manipulation.
        """
        # ease the flexion dofs from the built pose to the closing target over `settle`,
        # rather than stepping to it on frame 0: a 0.2 rad step on sixteen dofs at
        # ke = 150 is a snap-close, and a snap-close flicks the cube out of the hand
        # before there is anything to measure.
        span = self.settle - self.drop if self.close_time is None else self.close_time
        close = 0.5 - 0.5 * np.cos(np.pi * float(np.clip(
            (self.t - self.drop) / max(span, 1e-6), 0.0, 1.0)))
        spread = self.grasp if self.spread is None else self.spread
        hold = np.where(self.is_flex, self.grasp, spread)
        if self.thumb_oppose is not None:
            # The thumb's ``_0`` dof is opposition, not abduction: it swings the thumb
            # across the palm. Left at the abduction target the thumb sits out beside the
            # hand and never touches the cube, so a quarter of the skin is dead.
            hold = np.where((~self.is_flex) & self._is_thumb, self.thumb_oppose, hold)
        base = self.q_init + close * (hold - self.q_init)
        gain = float(np.clip((self.t - self.settle) / max(self.ramp, 1e-6), 0.0, 1.0))
        wave = np.sin(self.rate * (self.t - self.settle)
                      + self.digit * (2.0 * np.pi / self.n_digits))
        if self.swing_mode == "close":
            # One-sided: each digit rides between `grasp` and `grasp + amplitude`, never
            # below the hold. A symmetric swing has every digit spending half its cycle
            # OPENING past the secure grip, and with four digits phased 90 deg apart two
            # of them are always retreating -- measured, that ejects the cube for any
            # amplitude above 0.06 rad. One-sided, the grip is never released and the
            # digits merely take turns pushing, which is what rolls the cube.
            wave = 0.5 * (1.0 + wave)
        swing = gain * self.amplitude * self._swing_mask() * wave
        tgt = self.control.joint_target_q.numpy()
        tgt[: self.n_dof] = np.clip(
            base + swing, self.limit_lo[: self.n_dof], self.limit_hi[: self.n_dof]
        )
        self.control.joint_target_q.assign(tgt)
        self.t += dt

    def rock_wrist(self) -> None:
        """Rock the whole hand about X, exactly as the stock example does.

        This is half the manipulation: gravity does the work of rolling the cube in the
        palm while the fingers cage it, which is far gentler than trying to drive the
        cube with the digits alone and is why the cube survives to be measured.
        """
        if self.rock == 0.0:
            return
        xp = self.model.joint_X_p.numpy()
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.sin(self.t) * self.rock))
        qq = q * self.hand_rotation
        xp[self.root_joint, 3:] = np.array([qq[0], qq[1], qq[2], qq[3]], dtype=xp.dtype)
        self.model.joint_X_p.assign(xp)
        self.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

    def _mark(self, key: str, t0: float) -> float:
        """Charge elapsed time to a phase. Only when profiling: it forces a sync."""
        if not self.profile:
            return 0.0
        wp.synchronize_device()
        now = time.perf_counter()
        self.timings[key] = self.timings.get(key, 0.0) + now - t0
        return now

    def _run_substeps(self, sub: float, substeps: int) -> None:
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self, dt: float, substeps: int) -> None:
        """Record the substep loop as one CUDA graph.

        The solve is ~95 % of the step and, at ``iterations=100``, it is hundreds of tiny
        kernel launches -- with an empty scene it still cost 4.3 ms per substep, which is
        launch overhead, not arithmetic. Capturing collapses that to a single launch.

        ``substeps`` must be EVEN: the loop swaps ``state_0``/``state_1`` each pass, and a
        graph replays whichever buffers were current when it was recorded. An odd count
        leaves the pair swapped at the end, so the second replay would read the wrong one.
        """
        if not wp.get_device().is_cuda:
            self.graph = None
            return
        if substeps % 2:
            raise ValueError(f"--graph needs an even --substeps, got {substeps}")
        # MuJoCo-Warp's solver records a CUDA *conditional* node for its convergence
        # early-exit, and conditional nodes need driver 12.4+. This cluster is on 12.2, so
        # capture raises "Conditional graph nodes require CUDA driver 12.4+" unless the
        # flag is off. With it off the solver simply runs its full `iterations` every
        # step -- which is why `--iterations` is worth tuning alongside `--graph`.
        mjw = getattr(self.solver, "mjw_model", None)
        if mjw is not None and hasattr(mjw.opt, "graph_conditional"):
            mjw.opt.graph_conditional = False
        self._run_substeps(dt / substeps, substeps)   # warm the kernels before recording
        with wp.ScopedCapture() as cap:
            self._run_substeps(dt / substeps, substeps)
        self.graph = cap.graph

    def step(self, dt: float, substeps: int = 8) -> None:
        t0 = time.perf_counter() if self.profile else 0.0
        self.rock_wrist()
        self.drive(dt)
        t0 = self._mark("drive", t0) or t0
        sub = dt / substeps
        self.pipeline.collide(self.state_0, self.contacts)
        t0 = self._mark("collide", t0) or t0
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._run_substeps(sub, substeps)
        t0 = self._mark("solve", t0) or t0
        self.solver.update_contacts(self.contacts, self.state_0)   # fills contacts.force
        surface = self.pipeline.hydroelastic_sdf.get_contact_surface() if self.want_surface else None
        self.tactile.update(self.state_0, self.contacts, contact_surface=surface)
        t0 = self._mark("tactile", t0) or t0
        if self.field is not None:
            self.field.update(self.state_0, surface)   # per-face map; needs patch loads
            self._mark("field", t0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--kh", type=float, default=1.0e10,
                    help="hydroelastic stiffness [Pa/m]; 1e8 visibly sinks the fingers into the cube")
    ap.add_argument("--substeps", type=int, default=8,
                    help="8 is the floor for stability: at 4 the 5 ms substep is longer than "
                         "the contact's own period and the cube is flung metres. ~20 fps")
    ap.add_argument("--cube-drop", type=float, default=0.05,
                    help="height the cube starts above its USD rest pose [m]")
    ap.add_argument("--mu", type=float, default=None,
                    help="override friction on every collider; default keeps the USD's")
    ap.add_argument("--graph", action="store_true",
                    help="capture the substep loop as a CUDA graph (needs even --substeps)")
    ap.add_argument("--ke", type=float, default=1.0e4,
                    help="contact stiffness [N/m]. Bounded by STABILITY, not accuracy: the "
                         "substep must be shorter than sqrt(m/ke). Penetration barely moves "
                         "across 1e3..1e6 (1.16-1.28 mm p99), so there is nothing to win by "
                         "going stiffer and a blow-up to lose")
    ap.add_argument("--kd", type=float, default=3.2e2,
                    help="contact damping; scale it as sqrt(ke) or the contact goes springy")
    ap.add_argument("--iterations", type=int, default=100, help="MuJoCo solver iterations")
    ap.add_argument("--ls-iterations", type=int, default=50)
    ap.add_argument("--cone", default="elliptic", choices=("elliptic", "pyramidal"))
    ap.add_argument("--njmax", type=int, default=800)
    ap.add_argument("--field-max", type=int, default=6000,
                    help="max contact-surface faces kept per frame for the continuous map")
    ap.add_argument("--render", action="store_true", help="also write scene frames (needs Xvfb)")
    ap.add_argument("--image-format", default="png", choices=("png", "jpg"),
                    help="jpg encodes several times faster and the composer only shrinks it")
    # Framed from the bodies themselves. The USD carries its own offset on top of the
    # spawn transform, so the hand actually sits near z = 1.0, not the 0.5 in the
    # add_usd call -- a hand-written camera aimed at the floor below it.
    ap.add_argument("--cam-offset", type=float, nargs=3, default=(0.34, 0.26, 0.13),
                    help="camera position relative to the hand+cube centroid [m]")
    ap.add_argument("--amplitude", type=float, default=0.10, help="per-digit flexion swing [rad]")
    ap.add_argument("--rate", type=float, default=1.4, help="manipulation rate [rad/s]")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds before the manipulation swing starts")
    ap.add_argument("--drop", type=float, default=0.0,
                    help="seconds to leave the hand open while the cube falls into it")
    ap.add_argument("--close-time", type=float, default=None,
                    help="seconds to close over; default is drop -> settle")
    ap.add_argument("--thumb-oppose", type=float, default=None,
                    help="thumb opposition target [rad]; default follows --spread")
    ap.add_argument("--rock", type=float, default=0.10,
                    help="wrist rock amplitude about X [rad]; 0 pins the hand")
    ap.add_argument("--hold-digits", default="", help="comma-separated digits that clamp "
                    "instead of swinging, e.g. 'thumb'")
    ap.add_argument("--swing-mode", choices=("close", "sym"), default="close",
                    help="'close' never opens past the secure grip; 'sym' swings both ways")
    ap.add_argument("--spread", type=float, default=None,
                    help="abduction/opposition target [rad]; default follows --grasp")
    ap.add_argument("--grasp", type=float, default=0.40,
                    help="finger closing target [rad]; higher engages more links")
    args = ap.parse_args()

    t_prog = time.perf_counter()
    wp.init()
    if not wp.get_device().is_cuda:
        print("ERROR: hydroelastic SDF is CUDA-only.")
        return 2
    t_init = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scene = AllegroTactileScene(
        kh=args.kh, mu=args.mu, cube_drop=args.cube_drop, iterations=args.iterations,
        ls_iterations=args.ls_iterations, cone=args.cone, njmax=args.njmax,
        ke=args.ke, kd=args.kd,
    )
    scene.grasp = args.grasp
    scene.amplitude = args.amplitude
    scene.rate = args.rate
    scene.settle = args.settle
    scene.drop = args.drop
    scene.close_time = args.close_time
    scene.spread = args.spread
    scene.swing_mode = args.swing_mode
    scene.hold_digits = tuple(x for x in args.hold_digits.split(',') if x)
    scene.rock = args.rock
    scene.thumb_oppose = args.thumb_oppose
    if args.graph:
        scene.capture(1.0 / args.fps, args.substeps)
    t_build = time.perf_counter()
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

    t_viewer = time.perf_counter()

    # The cube's own body, so "is it actually being manipulated?" is a number rather
    # than an impression, and so penetration depth has something to be reported against.
    shape_body = scene.model.shape_body.numpy()
    cube_body = int(shape_body[scene.cube_shapes[0]])

    dt = 1.0 / args.fps
    trace = {k: np.zeros((args.frames, n), dtype=np.float32) for k in RECORD}
    trace["peak_depth"] = np.zeros((args.frames, n), dtype=np.float32)
    cube_q = np.zeros((args.frames, 7), dtype=np.float32)
    # the continuous map: one sample per contact-surface face, in the palm's frame
    fld = {k: [] for k in ("pos", "area", "pressure", "traction", "traction_vec",
                          "slip", "slip_vec", "patch")}
    offsets = np.zeros(args.frames + 1, dtype=np.int64)
    dropped = np.zeros(args.frames, dtype=np.int64)
    outline = np.zeros((args.frames, len(scene.outline_verts), 3), dtype=np.float32)
    stage = {k: 0.0 for k in ("step", "readback", "render", "encode")}
    for i in range(args.frames):
        t0 = time.perf_counter()
        scene.step(dt, substeps=args.substeps)
        t1 = time.perf_counter(); stage["step"] += t1 - t0
        ch = scene.tactile.to_numpy()
        for k in RECORD:
            trace[k][i] = ch[k]
        trace["peak_depth"][i] = scene.tactile._peak_depth.numpy()
        bq = scene.state_0.body_q.numpy()
        cube_q[i] = bq[cube_body]

        f = scene.field.to_numpy(stride_to=args.field_max)
        for k in fld:
            fld[k].append(f[k])
        offsets[i + 1] = offsets[i] + len(f["pressure"])
        dropped[i] = scene.field.total - len(f["pressure"])

        # collider vertices in the palm frame, so the map has a hand under it
        world = _xform_points_batch(bq[scene.outline_body], scene.outline_verts)
        outline[i] = _unxform_points(bq[scene.palm_body], world) if scene.palm_body >= 0 else world
        t2 = time.perf_counter(); stage["readback"] += t2 - t1
        if viewer is not None:
            from PIL import Image

            viewer.begin_frame(i * dt)
            viewer.log_state(scene.state_0)
            viewer.log_hydro_contact_surface(
                scene.pipeline.hydroelastic_sdf.get_contact_surface(), penetrating_only=True
            )
            viewer.end_frame()
            frame = viewer.get_frame().numpy()
            t3 = time.perf_counter(); stage["render"] += t3 - t2
            Image.fromarray(frame).save(out / "frames" / f"f{i:05d}.{args.image_format}",
                                        quality=92)
            stage["encode"] += time.perf_counter() - t3
        if i % 50 == 0:
            print(f"  frame {i:4d}  live={int((ch['contact_count'] > 0).sum()):2d}/{n}  "
                  f"N={ch['normal_load'].sum():7.2f}  p_max={ch['peak_pressure'].max():9.1f} Pa  "
                  f"slip_max={ch['slip_velocity'].max():.4f} m/s", flush=True)

    np.savez(out / "allegro_tactile.npz", labels=np.array(scene.patch_labels), dt=dt,
             cube_q=cube_q, **trace)
    np.savez_compressed(
        out / "allegro_field.npz", dt=dt, labels=np.array(scene.patch_labels),
        offsets=offsets, dropped=dropped, outline=outline,
        outline_is_cube=np.isin(scene.outline_body, [
            int(scene.model.shape_body.numpy()[c]) for c in scene.cube_shapes]),
        **{k: np.concatenate(v) if len(v) else np.zeros((0, 3), np.float32)
           for k, v in fld.items()},
    )
    ever = int((trace["contact_count"] > 0).any(axis=0).sum())
    print(f"\nwrote {out / 'allegro_tactile.npz'}")
    print(f"patches ever in contact : {ever}/{n}")
    print(f"peak normal load        : {trace['normal_load'].sum(axis=1).max():.2f} N")
    print(f"peak contact area       : {trace['contact_area'].max() * 1e4:.3f} cm^2")
    print(f"peak pressure           : {trace['peak_pressure'].max():.1f} Pa")
    print(f"peak friction load      : {trace['friction_load'].max():.2f} N")
    print(f"peak slip velocity      : {trace['slip_velocity'].max():.4f} m/s")

    # (1) penetration and (2) is the cube actually worked -- reported, not eyeballed.
    # Steady state excludes the first 20 % of frames: the grasp settles with a transient
    # that dominates every peak in this scene.
    tail = slice(int(0.2 * args.frames), None)
    dep = trace["peak_depth"]
    print(f"peak penetration        : {dep.max() * 1e3:.3f} mm  "
          f"(steady {dep[tail].max() * 1e3:.3f} mm)")
    dxyz = cube_q[:, :3] - cube_q[0, :3]
    travel = np.linalg.norm(dxyz, axis=1)
    # geodesic angle between each frame's quaternion and the first: 2*acos(|q0.q|)
    dots = np.abs(cube_q[:, 3:] @ cube_q[0, 3:])
    ang = np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))
    print(f"cube travel             : {travel.max() * 1e3:.1f} mm max, "
          f"{travel[-1] * 1e3:.1f} mm net")
    print(f"cube rotation           : {ang.max():.1f} deg max from start")

    print(f"\nwrote {out / 'allegro_field.npz'}")
    total = time.perf_counter() - t_prog
    print(f"\n--- where the wall clock went ({total:.1f} s total) ---")
    print(f"  warp init             : {t_init - t_prog:6.1f} s")
    print(f"  build scene (18 SDFs) : {t_build - t_init:6.1f} s")
    print(f"  viewer setup          : {t_viewer - t_build:6.1f} s")
    for k in ("step", "readback", "render", "encode"):
        print(f"  {k:<22s}: {stage[k]:6.1f} s"
              f"  ({1e3 * stage[k] / max(args.frames, 1):6.1f} ms/frame)")
    print(f"  writing npz + other   : "
          f"{total - (t_viewer - t_prog) - sum(stage.values()):6.1f} s")

    print(f"field faces             : {offsets[-1]} kept over {args.frames} frames, "
          f"{int(np.diff(offsets).max())} peak/frame")
    if dropped.any():
        print(f"field faces DROPPED     : {dropped.sum()} total, {dropped.max()} in one frame "
              f"(--field-max {args.field_max}); the map is a subsample, not an integral")
    if scene.field.overflow_steps:
        print(f"field buffer overflowed : {scene.field.overflow_steps} steps, "
              f"peak {scene.field.max_faces_seen} faces > capacity {scene.field.capacity}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
