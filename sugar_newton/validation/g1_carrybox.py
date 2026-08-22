# SPDX-License-Identifier: BSD-3-Clause
"""SUGAR's CarryBox on Newton: the G1 picks up the box, and the skin feels it.

This is Plan 16 Phase 2's asset half, standing up for the first time: the real
``g1_29dof_rev_1_0_with_rubber_hand`` URDF -- not Newton's stock G1 -- because the
anatomical patch layout is defined in the *rubber hand's* mesh frame, and 27 pads per
hand only land in the right places if that link is the body they hang off.

Motion is SUGAR's own reference clip played open-loop: the 29 joint targets come
straight from ``robot_50hz.npz`` and the pelvis is carried along the reference root
trajectory. No policy yet -- Phase 3 is what puts the teacher in this loop -- but the
grasp is real physics: the box is a dynamic body, the pads are hydroelastic, and whether
it gets lifted is an outcome, not a script.

Two things worth knowing before trusting a number out of this:

* **Joint order and quaternion order were assumptions, and are now measurements.**
  The reference stores 29 joint angles with no names and a root quaternion with no stated
  convention. :func:`check_reference` settles both without any name mapping: drive forward
  kinematics with the reference's own joint angles, then ask how far each reference body
  is from the nearest simulated one. Swept over three orderings and both quaternion
  conventions, exactly one combination collapses the error to zero --
  **breadth-first joints, ``wxyz`` root** at **0.0 mm**, against ~142 mm for depth-first
  or URDF order and ~207 mm for ``xyzw``. That is IsaacLab's convention, which is where
  the clip came from.

    uv run python -m sugar_newton.validation.g1_carrybox --out <dir> --check
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode, ModelFlags
from newton.geometry import HydroelasticSDF
from sugar_newton.hand.patches import (
    PATCH_SPECS,
    palm_normal_sign,
    patch_footprints,
)
from sugar_newton.tactile.field import ContactField
from sugar_newton.tactile.reducer import PatchTactile

SUGAR = Path("/lustre/fs12/portfolios/nvr/projects/nvr_nxp_visionconferencing/users/shengzew"
             "/robot_baby/Curiosity/SUGAR")
URDF = SUGAR / "descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
CLIPS = SUGAR / "data/CarryBox"
BOX_USD = {
    "small": SUGAR / "descriptions/objects/small_box/obj_aligned.usd",
    "big": SUGAR / "descriptions/objects/big_box/obj_aligned.usd",
}

# CarryBox uses SMALLBOX, not the big box: carry_box_refiner_env_cfg.py:29 imports
# SMALLBOX_CFG. (`big_box` is really the kick/push box -- its own config.yaml still
# points at `data/kick_box`.) Half-extents measured off the mesh; mass from the
# IsaacLab cfg, which overrides the 1.0 in the asset's config.yaml.
BOXES = {
    "small": ((0.2000, 0.2730, 0.2681), 0.5),
    "big": ((0.3177, 0.3179, 0.3942), 0.75),
}

RECORD = (
    "contact_count", "normal_load", "friction_load", "utilization_mean", "utilization_max",
    "slip_displacement", "slip_velocity", "gross_slip_fraction", "contact_area", "peak_pressure",
)


def load_box_mesh(which: str) -> tuple[np.ndarray, np.ndarray]:
    """The CarryBox object's OWN mesh, straight out of the asset.

    SUGAR spawns it with a 128-resolution SDF collider (``SMALLBOX_SDF_CFG``,
    ``solid_outer_shell_only=True``), so a bounding box in its place is not the same
    object -- it is a different rigid body that happens to be about the same size.
    """
    # obj_aligned.usd only REFERENCES the geometry; the points live in the instanceable
    # payload beside it, and TraverseAll on the wrapper finds nothing.
    root = BOX_USD[which]
    for candidate in (root, root.parent / "Props" / "instanceable_meshes.usd"):
        verts, tris = _mesh_from_usd(candidate)
        if verts is not None:
            return verts, tris
    raise RuntimeError(f"no mesh in {root} or its Props payload")


def _mesh_from_usd(path):
    from pxr import Usd, UsdGeom

    if not path.exists():
        return None, None
    stage = Usd.Stage.Open(str(path))
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        g = UsdGeom.Mesh(prim)
        pts = g.GetPointsAttr().Get()
        if not pts:
            continue
        verts = np.array([[q[0], q[1], q[2]] for q in pts], dtype=np.float32)
        counts = np.asarray(g.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        idx = np.asarray(g.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        tris, o = [], 0
        for c in counts:
            fv = idx[o:o + c]
            o += c
            for k in range(1, c - 1):          # fan-triangulate whatever the asset uses
                tris.append((fv[0], fv[k], fv[k + 1]))
        return verts, np.asarray(tris, dtype=np.int32)
    return None, None


def load_clip(name: str) -> dict:
    """One CarryBox reference clip: 29 joints, 35 bodies and the box, all at 50 Hz."""
    d = np.load(CLIPS / name / "robot_50hz.npz", allow_pickle=True)
    with open(CLIPS / name / "obj_motion_global_50hz.pkl", "rb") as f:
        obj = pickle.load(f)
    return {
        "fps": float(d["fps"][0]),
        "joint_pos": np.asarray(d["joint_pos"], dtype=np.float64),
        "body_pos_w": np.asarray(d["body_pos_w"], dtype=np.float64),
        "body_quat_w": np.asarray(d["body_quat_w"], dtype=np.float64),
        "obj_trans": np.asarray(obj["obj_trans"], dtype=np.float64),
        "obj_rot": np.asarray(obj["obj_rot"], dtype=np.float64),
        "contact": np.load(CLIPS / name / "contact_labels_50hz.npy"),
    }


def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation to xyzw quaternion (Warp's order), branch-free enough to be safe."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


# --- SUGAR's actuator table, transcribed from assets/robots/unitree.py ------------
_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0


def _gain(armature: float) -> tuple[float, float]:
    """SUGAR derives every gain from an armature and a 10 Hz natural frequency."""
    return (armature * _NATURAL_FREQ ** 2,
            2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ)


_A_5020, _A_7520_14, _A_7520_22, _A_4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
_ACTUATORS = (
    # (substring test, armature, effort limit [N.m], armature scale)
    (("hip_pitch_joint", "hip_yaw_joint"), _A_7520_14, 88.0, 1.0),
    (("hip_roll_joint", "knee_joint"), _A_7520_22, 139.0, 1.0),
    (("ankle_pitch_joint", "ankle_roll_joint"), _A_5020, 50.0, 2.0),
    (("waist_roll_joint", "waist_pitch_joint"), _A_5020, 50.0, 2.0),
    (("waist_yaw_joint",), _A_7520_14, 88.0, 1.0),
    (("wrist_pitch_joint", "wrist_yaw_joint"), _A_4010, 5.0, 1.0),
    (("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
      "elbow_joint", "wrist_roll_joint"), _A_5020, 25.0, 1.0),
)


def _sugar_actuator(label: str):
    """(stiffness, damping, armature, effort limit) for a joint, or None if unactuated."""
    for names, armature, effort, scale in _ACTUATORS:
        if any(label.endswith(n) for n in names):
            k, d = _gain(armature)
            return k * scale, d * scale, armature * scale, effort
    return None


def _dof_labels(builder) -> list[str]:
    """Coordinate-indexed joint labels; `joint_label` is per joint, not per dof."""
    out = []
    for j, lbl in enumerate(builder.joint_label):
        n = builder.joint_q_start[j + 1] - builder.joint_q_start[j] \
            if hasattr(builder, "joint_q_start") and len(builder.joint_q_start) > j + 1 \
            else (7 if builder.joint_type[j] == newton.JointType.FREE else 1)
        out += [lbl.split("/")[-1]] * int(n)
    return out


class G1CarryBoxScene:
    def __init__(self, clip: dict, kh=1.0e10, ke=1.0e4, kd=3.2e2, mu=0.75,
                 hand_sdf_res=128, box_sdf_res=128, quat_wxyz=True, joint_ordering="bfs",
                 box="small", rest_box_on_ground=True, upper_body_only=True,
                 tactile=True):
        self.clip, self.quat_wxyz = clip, quat_wxyz
        self.want_tactile = tactile
        _bbox_half, box_mass = BOXES[box]
        self.box_half = np.asarray(_bbox_half, dtype=float)
        self.rest_box_on_ground = rest_box_on_ground
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.default_shape_cfg.ke = ke
        b.default_shape_cfg.kd = kd
        b.default_shape_cfg.mu = mu
        b.default_shape_cfg.margin = 0.005

        # collapse_fixed_joints=False on purpose: the rubber hand hangs off the wrist by a
        # fixed joint, and collapsing it would merge the hand into the wrist link. The pad
        # coordinates in PATCH_SPECS are in the HAND's frame, so that link has to survive.
        b.add_urdf(
            str(URDF), floating=False, collapse_fixed_joints=False,
            enable_self_collisions=False, joint_ordering=joint_ordering,
            ignore_inertial_definitions=False,
        )
        self.n_bodies_robot = b.body_count

        labels = [lbl.split("/")[-1] for lbl in b.body_label]
        self.hand_body = {}
        for side in ("left", "right"):
            name = f"{side}_rubber_hand"
            if name not in labels:
                raise RuntimeError(f"{name} not in URDF bodies: {labels[-6:]}")
            self.hand_body[side] = labels.index(name)

        # The URDF's own hand collider is retired: the 27 pads stand proud of it, and if
        # the shell also collides the box rests on the shell and the skin reads nothing.
        #
        # Legs and head come off too, unless asked otherwise. The pelvis is carried along
        # the reference trajectory, so leg-ground contact changes nothing in this scene,
        # and the full-resolution leg meshes are what overflow the narrow phase's triangle
        # pair buffer ("Triangle pair buffer overflowed 1029134 > 1000000"). This is a
        # deletion, not a convex-hull approximation: nothing that CAN touch the box is
        # simplified.
        _OFF = ("hip", "knee", "ankle", "foot", "head", "contour")
        self._retire_later = [s for s in range(b.shape_count)
                              if b.shape_body[s] in self.hand_body.values()]
        for s in range(b.shape_count):
            body = b.shape_body[s]
            name = labels[body] if 0 <= body < len(labels) else ""
            if upper_body_only and any(k in name for k in _OFF):
                b.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

        # THE HAND IS THE SENSOR. Its own collider -- the rubber-hand mesh the URDF
        # ships, 45 748 triangles on the left and watertight -- is rebuilt as a
        # hydroelastic mesh so a contact SURFACE exists on it. Nothing is substituted:
        # the vertices and triangles are the asset's own, taken straight off
        # ``shape_source``. The 27 anatomical patches are applied later as a LABELLING of
        # that surface (see ContactField's pad assignment), not as added geometry.
        skin_cfg = replace(
            b.default_shape_cfg, mu=mu, restitution=0.0, kh=kh, is_hydroelastic=True,
            density=0.0, has_shape_collision=True, mu_torsional=0.0, mu_rolling=0.0,
        )
        self.patch_shapes, self.patch_labels, self.patch_frame = [], [], []
        self.pad_offset, self.palm_sign = [], []
        for side in ("left", "right"):
            body = self.hand_body[side]
            src_shapes = [s for s in range(b.shape_count) if b.shape_body[s] == body
                          and b.shape_source[s] is not None]
            if not src_shapes:
                raise RuntimeError(f"{side}_rubber_hand has no mesh collider in the URDF")
            src = b.shape_source[src_shapes[0]]
            nv = len(np.asarray(src.vertices))
            print(f"  {side}_rubber_hand collider: type={b.shape_type[src_shapes[0]]} "
                  f"verts={nv} tris={len(np.asarray(src.indices).reshape(-1, 3))}",
                  flush=True)
            if nv < 1000:
                print(f"    WARNING: the URDF's hand mesh has 22 876 vertices. {nv} means "
                      f"the importer hulled it -- the sensor would be reading a mitten.",
                      flush=True)
            m = newton.Mesh(
                np.asarray(src.vertices, dtype=np.float32),
                np.asarray(src.indices, dtype=np.int32).flatten(),
                compute_inertia=False,
            )
            # Hydroelastic needs the texture SDF, and the builder's per-shape sdf_* fields
            # never reach an imported convex mesh -- the same trap as the Allegro hand.
            m.build_sdf(max_resolution=hand_sdf_res,
                        narrow_band_range=(-0.004, 0.004), margin=0.002)
            shape = b.add_shape_mesh(
                body=body, xform=b.shape_transform[src_shapes[0]],
                mesh=m, scale=b.shape_scale[src_shapes[0]], cfg=skin_cfg,
                label=f"{side}_rubber_hand_skin",
            )
            b.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)  # the URDF already draws it
            self.patch_shapes.append(shape)
            self.patch_labels.append(f"{side}_rubber_hand")
            self.patch_frame.append(body)
            self.pad_offset.append(len(self.pad_offset) * len(PATCH_SPECS))
            self.palm_sign.append(palm_normal_sign(side))
        self.pad_labels = [f"{side}_{spec.name}" for side in ("left", "right")
                           for spec in PATCH_SPECS]

        # The hand's ORIGINAL collider is retired only now, after its replacement with
        # the identical mesh plus an SDF has been added. Same vertices, same triangles.
        for s in self._retire_later:
            b.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

        # the box: a dynamic free body, hydroelastic so there is a contact surface to
        # measure pressure over. A primitive, so it costs no mesh SDF build.
        #
        # ``add_body`` ALREADY makes the body, its free joint and its own articulation
        # (builder.py:3975). Adding a free joint on top gives the body two parents --
        # "Multiple joints lead to body 39" from the topological sort, or, if it slips
        # past that, a "Loop joint ... skipping loop closure" warning and a box MuJoCo
        # never simulates.
        self.box_body = b.add_body(mass=box_mass, label="carry_box")
        box_cfg = replace(
            b.default_shape_cfg, mu=mu, restitution=0.0, kh=kh, is_hydroelastic=True,
            density=0.0, mu_torsional=0.0, mu_rolling=0.0,
        )
        bv, bt = load_box_mesh(box)
        self.box_verts = np.asarray(bv, dtype=float)
        self.box_half = ((bv.max(axis=0) - bv.min(axis=0)) * 0.5).astype(float)
        bm = newton.Mesh(bv, bt.flatten(), compute_inertia=False)
        bm.build_sdf(max_resolution=box_sdf_res, narrow_band_range=(-0.006, 0.006),
                     margin=0.004)
        self.box_shape = b.add_shape_mesh(
            body=self.box_body, mesh=bm, cfg=box_cfg, label="carry_box",
        )

        # SUGAR's OWN actuator parameters, not invented ones
        # (assets/robots/unitree.py:100-230). They are an order of magnitude softer than
        # the 500/10 this file used before -- the arms are 14.25 N.m/rad, not 500 -- and
        # they carry armature and effort limits. A 35x-too-stiff arm does not track the
        # reference more faithfully; it drives into the box like a ram, which is what was
        # launching it.
        applied = 0
        for i, lbl in enumerate(_dof_labels(b)):
            gains = _sugar_actuator(lbl)
            if gains is None:
                continue
            k, d, arm, eff = gains
            b.joint_target_ke[i] = k
            b.joint_target_kd[i] = d
            b.joint_armature[i] = arm
            b.joint_effort_limit[i] = eff
            b.joint_target_mode[i] = int(JointTargetMode.POSITION)
            applied += 1
        self.actuated_dofs = applied

        # The FLOOR moves, not the scene. Lifting robot and box together looked
        # conservative but was not: the box then settles onto the raised floor and drops
        # ~16 cm relative to the robot, which breaks the one thing the reference is for --
        # where the box is relative to the hands. Measured: with the real box mesh the
        # hands then never reach it, 0 contact over the whole 481-frame clip. Dropping the
        # plane to where the clip's own floor must have been leaves every reference pose
        # untouched.
        b.add_ground_plane(height=self._ground_height())
        self.root_joint = next((i for i, p in enumerate(b.joint_parent) if p == -1), 0)
        self.model = b.finalize()
        self.model.request_contact_attributes("force")

        self.pipeline = newton.CollisionPipeline(
            self.model, contact_matching="latest", contact_report=True,
            # The contact SURFACE is only needed to measure pressure and area. For a
            # plain visual pass it is pure cost, so it is switched off with the sensing.
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                output_contact_surface=tactile, buffer_fraction=1.0, buffer_mult_iso=2
            ),
        )
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton", integrator="implicitfast",
            njmax=8192, nconmax=min(4000, self.contacts.rigid_contact_max),
            impratio=20.0, cone="elliptic", iterations=100, ls_iterations=50,
            use_mujoco_contacts=False,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.tactile = PatchTactile(self.model, self.patch_shapes, [self.box_shape])
        # Ask the MODEL which body each pad hangs off, rather than reusing the builder
        # index. `finalize` is free to renumber bodies, and it does here: taking the
        # builder's number put every face in a neighbouring link's frame, which showed up
        # as a constant ~13 mm offset between the contact and the pad it was on.
        self.patch_frame = self.model.shape_body.numpy()[self.patch_shapes].tolist()
        # Each hand's faces are expressed in ITS OWN link frame, which is what makes the
        # map a canonical skin diagram: the pads never move in it.
        self.field = ContactField(
            self.tactile, frame_body=self.patch_frame,
            pad_footprint=patch_footprints(), pad_offset=self.pad_offset,
            palm_sign=self.palm_sign,
        )
        self.n_joint_dofs = min(29, self.model.joint_dof_count)
        self.z_lift = 0.0
        self.z_lift = 0.0  # the floor moves instead; see _ground_height
        self.frame = 0
        self.profile = False
        self.timings: dict[str, float] = {}
        self.graph = None

    # ---- reference playback --------------------------------------------------
    def root_xform(self, i: int) -> wp.transform:
        c = self.clip
        i = min(i, len(c["body_pos_w"]) - 1)
        p = np.array(c["body_pos_w"][i, 0], dtype=float)
        p[2] += self.z_lift
        q = c["body_quat_w"][i, 0]
        q = np.array([q[1], q[2], q[3], q[0]]) if self.quat_wxyz else np.asarray(q)
        return wp.transform(wp.vec3(*p.tolist()), wp.quat(*(q / np.linalg.norm(q)).tolist()))

    def box_xform(self, i: int) -> wp.transform:
        """The box's reference pose, optionally lifted to sit ON the ground.

        The clip's own height buries it: at frame 0 the centre is 0.198 m up while the
        box's half-extent along world Z is ~0.34 m at that orientation. Retargeted
        reference motion routinely interpenetrates the floor -- IsaacLab pushes the object
        out at reset -- and a hydroelastic box 0.15 m inside the ground is a detonation.
        Only the height is corrected; x, y and the orientation are the reference's.
        """
        c = self.clip
        i = min(i, len(c["obj_trans"]) - 1)
        p = np.array(c["obj_trans"][i], dtype=float)
        q = mat_to_quat_xyzw(c["obj_rot"][i])
        p[2] += self.z_lift
        return wp.transform(wp.vec3(*p.tolist()), wp.quat(*q.tolist()))

    def _ground_height(self) -> float:
        """Where the floor has to be for the clip's box pose to be a resting pose.

        Computed from the MESH VERTICES, not from half-extents about the origin. The
        asset's origin is not its geometric centre, and a support computed as
        ``|R| . half`` is then wrong by that offset -- which is what left the box floating
        16 cm above the floor, falling on release, and the hands reaching 16 cm past it
        for the whole clip with zero contact.
        """
        if not self.rest_box_on_ground:
            return 0.0
        R = np.asarray(self.clip["obj_rot"][0], dtype=float)
        low = float((self.box_verts @ R.T)[:, 2].min())     # lowest point in world Z
        return min(0.0, float(self.clip["obj_trans"][0][2]) + low - 0.002)

    def _ground_lift(self) -> float:
        """How far the WHOLE scene rises so the box starts resting on the floor.

        The clip has the box centred 0.198 m up while its half-extent along world Z is
        ~0.34 m at that orientation -- retargeted reference motion routinely leaves the
        object inside the floor, and IsaacLab pushes it out at reset. Lifting only the box
        would break the one thing the reference is for: where the box is *relative to the
        hands*. So robot and box rise together and the floor ends up in the right place.
        """
        if not self.rest_box_on_ground:
            return 0.0
        c = self.clip
        q = mat_to_quat_xyzw(c["obj_rot"][0])
        x, y, z, w = q
        reach = (abs(2.0 * (x * z - y * w)) * self.box_half[0]
                 + abs(2.0 * (y * z + x * w)) * self.box_half[1]
                 + abs(1.0 - 2.0 * (x * x + y * y)) * self.box_half[2])
        return max(0.0, reach + 0.002 - float(c["obj_trans"][0][2]))

    def reset(self) -> None:
        """Pose the robot and the box on the clip's first frame."""
        self.frame = 0
        xp = self.model.joint_X_p.numpy()
        xf = self.root_xform(0)
        xp[self.root_joint] = np.array(
            [xf.p[0], xf.p[1], xf.p[2], xf.q[0], xf.q[1], xf.q[2], xf.q[3]], dtype=xp.dtype)
        self.model.joint_X_p.assign(xp)

        q = self.model.joint_q.numpy()
        q[: self.n_joint_dofs] = self.clip["joint_pos"][0, : self.n_joint_dofs]
        bxf = self.box_xform(0)
        q[-7:] = np.array([bxf.p[0], bxf.p[1], bxf.p[2],
                           bxf.q[0], bxf.q[1], bxf.q[2], bxf.q[3]], dtype=q.dtype)
        self.model.joint_q.assign(q)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.tactile.reset()
        self.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

    def drive(self) -> None:
        """Carry the pelvis along the reference and hand the 29 joints their targets."""
        xp = self.model.joint_X_p.numpy()
        xf = self.root_xform(self.frame)
        xp[self.root_joint] = np.array(
            [xf.p[0], xf.p[1], xf.p[2], xf.q[0], xf.q[1], xf.q[2], xf.q[3]], dtype=xp.dtype)
        self.model.joint_X_p.assign(xp)
        self.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

        i = min(self.frame, len(self.clip["joint_pos"]) - 1)
        tgt = self.control.joint_target_q.numpy()
        tgt[: self.n_joint_dofs] = self.clip["joint_pos"][i, : self.n_joint_dofs]
        self.control.joint_target_q.assign(tgt)

    def _mark(self, key: str, t0: float) -> float:
        if not self.profile:
            return 0.0
        wp.synchronize_device()
        now = time.perf_counter()
        self.timings[key] = self.timings.get(key, 0.0) + now - t0
        return now

    def step(self, dt: float, substeps: int = 8) -> None:
        t0 = time.perf_counter() if self.profile else 0.0
        self.drive()
        t0 = self._mark("drive", t0) or t0
        sub = dt / substeps
        self.pipeline.collide(self.state_0, self.contacts)
        t0 = self._mark("collide", t0) or t0
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub)
            self.state_0, self.state_1 = self.state_1, self.state_0
        t0 = self._mark("solve", t0) or t0
        if self.want_tactile:
            self.solver.update_contacts(self.contacts, self.state_0)
            surface = self.pipeline.hydroelastic_sdf.get_contact_surface()
            self.tactile.update(self.state_0, self.contacts, contact_surface=surface)
            t0 = self._mark("tactile", t0) or t0
            self.field.update(self.state_0, surface)
            self._mark("field", t0)
        self.frame += 1


def check_reference(scene: G1CarryBoxScene, frames=(0, 60, 150, 300)) -> float:
    """Do the reference's joint angles reproduce its own body positions under FK?

    The clip stores 29 joint angles with no names and 35 body poses that do not map
    one-to-one onto the URDF's 40 links, so the ordering cannot be checked by name.
    What CAN be checked without any mapping: drive FK with the reference joints and the
    reference root, then ask how far each simulated body is from the NEAREST reference
    body. Right ordering puts the two point clouds on top of each other; a wrong one
    scatters them by limb lengths.
    """
    worst = 0.0
    for i in frames:
        if i >= len(scene.clip["joint_pos"]):
            continue
        scene.frame = i
        scene.drive()
        q = scene.state_0.joint_q.numpy()
        q[: scene.n_joint_dofs] = scene.clip["joint_pos"][i, : scene.n_joint_dofs]
        scene.state_0.joint_q.assign(q)
        newton.eval_fk(scene.model, scene.state_0.joint_q, scene.state_0.joint_qd, scene.state_0)
        sim = scene.state_0.body_q.numpy()[: scene.n_bodies_robot, :3]
        ref = scene.clip["body_pos_w"][i].copy()
        ref[:, 2] += scene.z_lift
        dm = np.linalg.norm(sim[:, None, :] - ref[None, :, :], axis=2)
        # ref -> sim is the direction that matters: the URDF carries more links than the
        # reference stores, so sim -> ref lets spare links inflate the score, while every
        # reference body must have a counterpart if the mapping is right.
        d = dm.min(axis=0)
        print(f"  frame {i:4d}: reference body to nearest simulated body  "
              f"mean {d.mean() * 1e3:7.1f} mm  max {d.max() * 1e3:7.1f} mm", flush=True)
        worst = max(worst, float(d.mean()))
    return worst


def check_reach(scene, clip: dict) -> None:
    """Does the reference ever bring the two hands closer together than the box is wide?

    A rigid rubber hand cannot wrap anything -- the only way this robot holds the box is
    by squeezing it between two palms. That requires the reference's own hand separation
    to close to less than the box's width at the grasp. If it never does, no amount of
    contact tuning will produce a lift, and the gap has to be closed by the policy.
    """
    lh, rh = scene.hand_body["left"], scene.hand_body["right"]
    n = len(clip["joint_pos"])
    sep = np.zeros(n)
    to_box = np.zeros(n)
    box_c = np.array(clip["obj_trans"][0], dtype=float)
    for i in range(n):
        scene.frame = i
        scene.drive()
        q = scene.state_0.joint_q.numpy()
        q[: scene.n_joint_dofs] = clip["joint_pos"][i, : scene.n_joint_dofs]
        scene.state_0.joint_q.assign(q)
        newton.eval_fk(scene.model, scene.state_0.joint_q, scene.state_0.joint_qd,
                       scene.state_0)
        bq = scene.state_0.body_q.numpy()
        sep[i] = float(np.linalg.norm(bq[lh, :3] - bq[rh, :3]))
        mid = 0.5 * (bq[lh, :3] + bq[rh, :3])
        to_box[i] = float(np.linalg.norm(mid - np.array(clip["obj_trans"][i], dtype=float)))
    width = 2.0 * float(np.min(scene.box_half))
    print(f"box smallest width      : {width * 1e3:6.1f} mm")
    print(f"hand separation         : min {sep.min() * 1e3:6.1f} mm at frame {int(sep.argmin())}, "
          f"median {np.median(sep) * 1e3:6.1f} mm")
    print(f"hand midpoint to box    : min {to_box.min() * 1e3:6.1f} mm at frame "
          f"{int(to_box.argmin())}")
    lab = clip["contact"][:n].astype(bool)
    if lab.any():
        print(f"over the clip's own contact-labelled frames: separation "
              f"{sep[lab].min() * 1e3:6.1f}..{sep[lab].max() * 1e3:6.1f} mm")
    if sep.min() > width:
        print("\nThe hands NEVER come closer than the box is wide. Open-loop playback of "
              "this reference cannot squeeze it, whatever the contact settings are.")
    else:
        print("\nThe hands do close inside the box width -- a squeeze is geometrically "
              "available, so failure to lift is dynamics, not reach.")
    print(f"(box centre at frame 0: {box_c.round(3)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--clip", default="data_000")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--kh", type=float, default=1.0e10)
    ap.add_argument("--ke", type=float, default=1.0e4)
    ap.add_argument("--kd", type=float, default=3.2e2)
    ap.add_argument("--mu", type=float, default=0.75)
    ap.add_argument("--box", default="small", choices=("small", "big"),
                    help="CarryBox uses the SMALL box; big is the kick/push box")
    ap.add_argument("--keep-legs", action="store_true",
                    help="keep leg and head colliders; they overflow the narrow phase")
    ap.add_argument("--box-at-reference-height", action="store_true",
                    help="do not lift the box out of the floor at reset")
    ap.add_argument("--field-max", type=int, default=8000)
    ap.add_argument("--quat-xyzw", action="store_true",
                    help="treat the reference quaternions as xyzw instead of IsaacLab wxyz")
    ap.add_argument("--joint-ordering", default="bfs", choices=("dfs", "bfs", "urdf"),
                    help="which order the reference's 29 joint angles are assumed to be in. "
                         "bfs + wxyz reproduces the reference body positions to 0.0 mm; dfs "
                         "and urdf are both ~142 mm out, and xyzw is ~207 mm")
    ap.add_argument("--check", action="store_true", help="only run the reference cross-check")
    ap.add_argument("--reach", action="store_true",
                    help="only ask whether the reference geometrically closes on the box")
    ap.add_argument("--render", action="store_true", help="write scene frames (headless EGL)")
    ap.add_argument("--no-tactile", action="store_true",
                    help="skip the contact surface, reducer and field -- a visual pass only")
    ap.add_argument("--image-format", default="jpg", choices=("png", "jpg"))
    ap.add_argument("--cam-offset", type=float, nargs=3, default=(1.6, -1.9, 0.8),
                    help="camera position relative to the pelvis [m]")
    ap.add_argument("--cam-height", type=float, default=0.9,
                    help="height above the pelvis the camera aims at [m]")
    args = ap.parse_args()

    wp.init()
    if not wp.get_device().is_cuda:
        print("ERROR: hydroelastic SDF is CUDA-only.")
        return 2

    clip = load_clip(args.clip)
    print(f"clip {args.clip}: {len(clip['joint_pos'])} frames at {clip['fps']:.0f} Hz, "
          f"{clip['body_pos_w'].shape[1]} reference bodies, "
          f"contact labelled on {int(clip['contact'].sum())} frames", flush=True)

    scene = G1CarryBoxScene(
        clip, kh=args.kh, ke=args.ke, kd=args.kd, mu=args.mu,
        quat_wxyz=not args.quat_xyzw,
        joint_ordering=None if args.joint_ordering == "urdf" else args.joint_ordering,
        box=args.box, rest_box_on_ground=not args.box_at_reference_height,
        upper_body_only=not args.keep_legs, tactile=not args.no_tactile,
    )
    print(f"actuated dofs with SUGAR gains: {scene.actuated_dofs}", flush=True)
    print(f"bodies={scene.n_bodies_robot} patches={len(scene.patch_shapes)} "
          f"dofs={scene.model.joint_dof_count} hands={scene.hand_body}", flush=True)
    names = [lbl.split("/")[-1] for lbl in scene.model.joint_label]
    print("driven joints, in the order the reference is assumed to use:")
    print("  " + ", ".join(f"{k}:{v}" for k, v in enumerate(names[: scene.n_joint_dofs])),
          flush=True)
    scene.reset()

    print("reference cross-check (wxyz)" if not args.quat_xyzw else "reference cross-check (xyzw)")
    err = check_reference(scene)
    if args.reach:
        check_reach(scene, clip)
        return 0
    if args.check:
        print(f"\nmean nearest-reference distance: {err * 1e3:.1f} mm")
        print("Under ~30 mm means the joint order and quaternion convention are right;"
              " limb-scale error means one of them is not.")
        return 0

    scene.reset()

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
        viewer.show_hydro_contact_surface = True
        (Path(args.out) / "frames").mkdir(parents=True, exist_ok=True)

        def aim(state):
            """Frame the pelvis AND the box.

            Aiming at the pelvis alone puts the box off the bottom of the frame for the
            whole reach -- which is the only part anyone wants to watch.
            """
            bq = state.body_q.numpy()
            p = bq[0, :3]
            b = bq[scene.box_body, :3]
            if not np.isfinite(b).all():
                b = p
            mid = 0.5 * (p + b)
            cam = mid + np.asarray(args.cam_offset, dtype=float)
            look = np.array([mid[0], mid[1], mid[2] + args.cam_height * 0.2])
            d = look - cam
            d /= max(np.linalg.norm(d), 1e-9)
            viewer.set_camera(
                wp.vec3(*cam.tolist()),
                math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0)))),
                math.degrees(math.atan2(float(d[1]), float(d[0]))),
            )

    dt = 1.0 / clip["fps"]
    n = min(args.frames, len(clip["joint_pos"]))
    npatch = len(scene.patch_shapes)
    trace = {k: np.zeros((n, npatch), dtype=np.float32) for k in RECORD}
    box_q = np.zeros((n, 7), dtype=np.float32)
    fld = {k: [] for k in ("pos", "area", "pressure", "traction", "traction_vec",
                           "slip", "slip_vec", "patch", "pad")}
    offsets = np.zeros(n + 1, dtype=np.int64)
    dropped = np.zeros(n, dtype=np.int64)

    t0 = time.perf_counter()
    for i in range(n):
        scene.step(dt, substeps=args.substeps)
        ch = scene.tactile.to_numpy() if scene.want_tactile else None
        if ch is not None:
            for k in RECORD:
                trace[k][i] = ch[k]
        box_q[i] = scene.state_0.body_q.numpy()[scene.box_body]
        if not np.isfinite(box_q[i]).all():
            print(f"  DIVERGED at frame {i}: the box pose went non-finite", flush=True)
            n = i
            break
        if scene.want_tactile:
            f = scene.field.to_numpy(stride_to=args.field_max)
            for k in fld:
                fld[k].append(f[k])
            offsets[i + 1] = offsets[i] + len(f["pressure"])
            dropped[i] = scene.field.total - len(f["pressure"])
        else:
            offsets[i + 1] = offsets[i]
        if viewer is not None:
            from PIL import Image

            aim(scene.state_0)
            viewer.begin_frame(i * dt)
            viewer.log_state(scene.state_0)
            viewer.log_hydro_contact_surface(
                scene.pipeline.hydroelastic_sdf.get_contact_surface(), penetrating_only=True
            )
            viewer.end_frame()
            Image.fromarray(viewer.get_frame().numpy()).save(
                Path(args.out) / "frames" / f"f{i:05d}.{args.image_format}", quality=92)
        if i % 50 == 0:
            live = int((ch["contact_count"] > 0).sum()) if ch is not None else -1
            load = float(ch["normal_load"].sum()) if ch is not None else float("nan")
            print(f"  frame {i:4d}  live={live:3d}/{npatch}  N={load:8.2f}  "
                  f"box_z={box_q[i, 2]:6.3f}", flush=True)
    el = time.perf_counter() - t0

    box_q = box_q[:n]
    for k in RECORD:
        trace[k] = trace[k][:n]
    # Measured from the box's SETTLED height, not from frame 0: the first frames are it
    # coming to rest, and calling that a lift flatters every run.
    rest = float(np.median(box_q[: max(int(0.15 * n), 5), 2])) if n else float("nan")
    lift = float(box_q[:, 2].max() - rest) if n else float("nan")
    held = float(box_q[-1, 2] - rest) if n else float("nan")
    ever = int((trace["contact_count"] > 0).any(axis=0).sum())
    print(f"\n{n} frames in {el:.1f} s ({n / el:.1f} fps)")
    print(f"patches ever in contact : {ever}/{npatch}")
    print(f"peak normal load        : {trace['normal_load'].sum(axis=1).max():.2f} N")
    print(f"peak pressure           : {trace['peak_pressure'].max():.1f} Pa")
    print(f"box rest height         : {rest:.3f} m")
    print(f"box lift                : {lift * 1e3:.1f} mm peak above rest, "
          f"{held * 1e3:.1f} mm at the end")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "g1_carrybox.npz", labels=np.array(scene.patch_labels), dt=dt,
                 box_q=box_q, **trace)
        np.savez_compressed(
            out / "g1_carrybox_field.npz", dt=dt,
            labels=np.array(scene.patch_labels), offsets=offsets, dropped=dropped,
            pad_labels=np.array(scene.pad_labels),
            pad_footprint=patch_footprints(),
            **{k: np.concatenate(v) if len(v) else np.zeros((0, 3), np.float32)
               for k, v in fld.items()},
        )
        print(f"wrote {out / 'g1_carrybox_field.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
