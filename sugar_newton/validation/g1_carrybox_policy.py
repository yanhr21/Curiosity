# SPDX-License-Identifier: BSD-3-Clause
"""Plan 16 Phase 3: SUGAR's trained tracker in the loop, on Newton, whole body.

What this changes relative to :mod:`g1_carrybox`, and why the change *is* the experiment:

* **Floating base.** ``g1_carrybox`` builds the robot with ``floating=False`` and carries
  the pelvis kinematically along the reference. That robot cannot squat or stand, and in
  the Isaac footage the lift comes almost entirely from the legs. Here the root is a free
  joint and the feet carry the weight.
* **Legs on.** ``upper_body_only`` deleted hip/knee/ankle/foot collision. Nothing is
  deleted here.
* **A policy, not a script.** ``g1_carrybox`` drove the 29 joint targets straight from
  ``robot_50hz.npz``. That reference is retargeted *human* motion -- kinematically
  plausible, not physically feasible -- and closing that gap is the entire job of the
  tracker. Here ``tracker.pt`` sees the state and chooses the targets.

The observation is the 510-D ``TrackerCfg`` group, validated offline against Isaac's own
recorded actions to RMSE 0.088 (action std ~1.3) by ``isaac/verify_obs.py``. Layout::

    ref_joint_pos 29 | ref_root_lin_vel_b 3 | ref_root_ang_vel_b 3 | contact_label 1
    base_ang_vel x5 15 | joint_pos_rel x5 145 | joint_vel_rel x5 145
    last_action x5 145 | project_gravity x5 15 | obj_pos_b 3 | obj_ori_b 6

Ground plane at z = 0: measured, not assumed. Over the clip the ankles sit at z = 0.03 to
0.13 and the box's lowest vertex at the reference pose is -0.0007 m, so the reference
already rests on a floor at the origin. None of ``g1_carrybox``'s floor-shifting applies.

The actor is evaluated in NumPy from ``tracker_actor.npz`` (four Linear layers and ELU,
~0.4 MFLOP/step) so the container needs no torch. That file and ``hand_hulls.npz`` are
gitignored derived artifacts; regenerate both with
``python -m sugar_newton.validation.make_policy_assets`` on a host that has torch and
scipy. The observation half has its own offline unit test in
:mod:`sugar_newton.validation.verify_tracker_obs`.

Result
------
The robot stands on its own feet for all 481 frames, grasps, and lifts the box. Measured
on three clips at mu=1.0, substeps=4, one run each::

    clip        lift    reference   fraction   joint tracking
    data_000    0.430   0.628       68 %       7.8 deg
    data_001    0.458   0.692       66 %       8.0 deg
    data_005    0.460   0.643       72 %       7.7 deg

Contact-rich rollouts here are chaotic: two runs at *identical* parameters have differed
by 0.07 m, so treat anything smaller than that as noise rather than an effect. The 66-72 %
consistency across three clips is well outside that spread.

The box mass bug
----------------
Those numbers used to read 0.21-0.30 m, i.e. a third of the reference rather than two
thirds, and the cause was mass, not contact::

    box mass, as simulated   4.39 kg
    box mass, as SUGAR spawns 0.50 kg      (tactile_objects.py:225)
    ratio                    8.78x

``ShapeConfig.density`` defaults to 1000 kg/m^3 and ``add_shape_mesh`` *adds* the shape's
mass and inertia to the body (builder.py:6125-6126). So ``add_body(mass=0.5)`` followed by
a default-density mesh does not make a 0.5 kg box; it makes 0.5 kg plus 3.89 dm^3 of water.
The robot's own links escaped this because the URDF importer overwrites the accumulated
value with the URDF inertial (import_urdf.py:687-691) -- which is exactly why the symptom
looked like a contact problem: every non-wrist joint agreed with Isaac, and only the joints
holding the box did not. See :mod:`sugar_newton.validation.check_masses`.

This retracts the "Newton demands ~6x more wrist torque than Isaac, cause unknown" finding
that stood here before. It demanded more torque because it was lifting 8.8x the mass.

What the gap is not
-------------------
* **Not the policy drifting.** ~8 deg joint tracking over 481 closed-loop steps.
* **Not grip.** Sweeping mu 0.5 -> 1.0 -> 1.5 saturates, so friction stops helping.
* **Not the hand collider.** Isaac collides this hand as a convex hull
  (``UrdfConverterCfg.collider_type`` defaults to ``"convex_hull"`` and SUGAR never
  overrides it), 2.35x the mesh volume. Giving Newton the same hull (``--hull-hands``)
  *lowers* the lift. Refuted; the flag is kept so the check stays repeatable.
* **Not contact compliance.** ``ke``/``kd`` at 1e4/3.2e2 is already the optimum; 1e3 and
  1e5 are both worse.
* **Not stale contacts.** Isaac collides every physics step and this port collided once per
  control step, reusing 20 ms-old normals. Fixing that (``--contact-refresh substep``, now
  matching Isaac and ``example_g1_in_sage.py:425-429``) changed the lift by -0.4 % on
  data_000 and +11 % on data_001 while costing 3-4x throughput. More correct, but not the
  gap, and not worth paying for by default.

Still open: the remaining ~30 %. The robot ends the clip ~0.65 m from the reference root
position while tracking joints to 8 deg, so it under-travels rather than mis-poses. The
observation carries reference joint angles and reference root *velocities*, never an
absolute reference position, so position error is uncorrectable by construction and the
question is whether it grows faster here than in Isaac.
"""

from __future__ import annotations

import argparse
import collections
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode, ModelFlags

HERE = Path(__file__).resolve().parent
SUGAR = HERE.parents[1] / "SUGAR"
URDF = SUGAR / "descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
CLIPS = SUGAR / "data/CarryBox"
BOX_USD = {
    "small": SUGAR / "descriptions/objects/small_box/obj_aligned.usd",
    "big": SUGAR / "descriptions/objects/big_box/obj_aligned.usd",
}
BOX_MASS = {"small": 0.5, "big": 0.75}
ACTOR_NPZ = HERE / "tracker_actor.npz"
HAND_HULLS = HERE / "hand_hulls.npz"

ANCHOR_LINK = "torso_link"          # base_tracker_env_cfg.py: anchor_body_name
N_DOF = 29

# --- SUGAR's actuator table, assets/robots/unitree.py -----------------------------
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
A_5020, A_7520_14, A_7520_22, A_4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
ACTUATORS = (
    # (suffixes, armature, effort limit [N.m], armature scale)
    (("hip_pitch_joint", "hip_yaw_joint"), A_7520_14, 88.0, 1.0),
    (("hip_roll_joint", "knee_joint"), A_7520_22, 139.0, 1.0),
    (("ankle_pitch_joint", "ankle_roll_joint"), A_5020, 50.0, 2.0),
    (("waist_roll_joint", "waist_pitch_joint"), A_5020, 50.0, 2.0),
    (("waist_yaw_joint",), A_7520_14, 88.0, 1.0),
    (("wrist_pitch_joint", "wrist_yaw_joint"), A_4010, 5.0, 1.0),
    (("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
      "elbow_joint", "wrist_roll_joint"), A_5020, 25.0, 1.0),
)
DEFAULT_POSE = (
    ("hip_pitch_joint", -0.312), ("knee_joint", 0.669), ("ankle_pitch_joint", -0.363),
    ("elbow_joint", 0.6), ("left_shoulder_roll_joint", 0.2),
    ("left_shoulder_pitch_joint", 0.2), ("right_shoulder_roll_joint", -0.2),
    ("right_shoulder_pitch_joint", 0.2),
)


def actuator_for(label: str):
    """(stiffness, damping, armature, effort) for a joint label, or None if unactuated."""
    for suffixes, armature, effort, scale in ACTUATORS:
        if any(label.endswith(s) for s in suffixes):
            k = armature * NATURAL_FREQ ** 2 * scale
            d = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ * scale
            return k, d, armature * scale, effort
    return None


def default_pose(names: list[str]) -> np.ndarray:
    out = np.zeros(len(names))
    for i, n in enumerate(names):
        for pat, val in DEFAULT_POSE:
            if n.endswith(pat):
                out[i] = val
    return out


def action_scale(names: list[str]) -> np.ndarray:
    """UNITREE_G1_29DOF_MIMIC_ACTION_SCALE = 0.25 * effort_limit / stiffness."""
    out = np.zeros(len(names))
    for i, n in enumerate(names):
        g = actuator_for(n)
        if g is None:
            raise KeyError(f"no actuator group for {n}")
        out[i] = 0.25 * g[3] / g[0]
    return out


# --- quaternion helpers, all xyzw to match Newton -------------------------------
def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def mat_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_mat_xyzw(m: np.ndarray) -> np.ndarray:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        q = np.array([(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
                      (m[1, 0] - m[0, 1]) * s, 0.25 / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                      (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                      (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                      0.25 * s, (m[1, 0] - m[0, 1]) / s])
    return q / np.linalg.norm(q)


# --- assets ----------------------------------------------------------------------
def bfs_body_names(urdf: Path) -> list[str]:
    """Bodies breadth-first, dropping the inertialess links the importer merges away.

    The clip's ``body_pos_w`` has 35 entries while the URDF declares 39 links; the four
    without an ``<inertial>`` block (``imu_in_pelvis``, ``imu_in_torso``, ``d435_link``,
    ``mid360_link``) are merged into their parents, exactly as Isaac's own import log
    reports. Dropping those makes the counts agree and puts the ankles near the floor
    instead of above the torso, which is how this ordering was checked.
    """
    root = ET.parse(str(urdf)).getroot()
    joints = [(j.find("parent").get("link"), j.find("child").get("link"))
              for j in root.findall("joint")]
    kids = collections.defaultdict(list)
    for p, c in joints:
        kids[p].append(c)
    links = {l.get("name") for l in root.findall("link")}
    massless = {l.get("name") for l in root.findall("link") if l.find("inertial") is None}
    roots = [l for l in links if l not in {c for _, c in joints}]
    out, queue = [], collections.deque(sorted(roots))
    while queue:
        l = queue.popleft()
        if l not in massless:
            out.append(l)
        queue.extend(kids[l])
    return out


def load_box_mesh(which: str):
    """Vertices and triangles of the box, read out of the instanced USD prototype.

    ``obj_aligned.usd`` references ``Props/instanceable_meshes.usd``, so the geometry
    lives in a prototype and ``Stage.TraverseAll`` does not reach it.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(BOX_USD[which]), load=Usd.Stage.LoadAll)
    for proto in stage.GetPrototypes():
        for prim in Usd.PrimRange(proto):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            g = UsdGeom.Mesh(prim)
            pts = g.GetPointsAttr().Get()
            if not pts:
                continue
            verts = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
            counts = np.asarray(g.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
            idx = np.asarray(g.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            tris, o = [], 0
            for c in counts:
                fv = idx[o:o + c]
                o += c
                tris.extend((fv[0], fv[k], fv[k + 1]) for k in range(1, c - 1))
            return verts, np.asarray(tris, dtype=np.int32)
    raise RuntimeError(f"no mesh in {BOX_USD[which]}")


def decimate(verts: np.ndarray, tris: np.ndarray, target: int):
    """Quadric-decimate a collision mesh to roughly ``target`` triangles.

    Collision is 94.5 % of the step at 16 worlds and its cost tracks triangle pairs, so the
    collider's triangle count is the throughput lever. The box ships as 100k triangles of
    median 7.8 mm^2 tiling 1.2264 m^2 -- within 0.4 % of the exact surface area of an open
    carton with this bounding box -- so that budget is tessellation density, not shape.

    Measured symmetric surface deviation against the original
    (:mod:`sugar_newton.validation.check_decimation`), against a 5 mm contact margin and a
    3.2 mm wall thickness::

        target   mean      p99       max
         5000    0.037 mm  0.205 mm  1.046 mm
         2000    0.076 mm  0.428 mm  1.634 mm
         1000    0.135 mm  0.865 mm  4.353 mm   <- approaches both limits
          200    0.624 mm  2.808 mm  6.077 mm   <- exceeds the margin

    2000 is the RL default: a 50x reduction whose worst-case deviation is half the wall
    thickness and a third of the margin. This is decimation, not hulling or decomposition --
    the mesh stays non-convex and the carton stays open, so the concavity the grasp needs is
    untouched.

    What that surface deviation does NOT buy is an unchanged contact set. Probing both
    colliders at identical states (:mod:`sugar_newton.validation.compare_contacts`, whose
    identical-collider control is exact to 0.1 %) shows retessellation moves the contacts
    even though it barely moves the surface::

        vs original      net load/hand   sum|f|   patch centroid   per-contact corr
        20000 tris         1.4-2.0 %     8-15 %      1.6-2.0 mm       0.65-0.97
         2000 tris         2.8-3.2 %    18-19 %      4.9-10.1 mm      0.48-0.80

    The split is the point. The net wrench each hand puts on the box -- 30.9 N of squeeze --
    is fixed by the physics and survives to ~3 %. How that wrench is distributed over the
    ~80 contacts per hand is underdetermined (a 6-DOF load spread over 80x3 unknowns), so it
    is set by the tessellation and the solver's regularisation, not by the shape; note that
    the per-contact correlation is not even monotone in triangle count. So decimation is
    safe for dynamics and for patch-level tactile readings, and not safe for per-taxel
    pressure patterns. Validation therefore stays at the full 100k mesh by default.
    """
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32)))
    m = m.simplify_quadric_decimation(target_number_of_triangles=target)
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    return (np.asarray(m.vertices, dtype=np.float32),
            np.asarray(m.triangles, dtype=np.int32))


def decimate_hand_colliders(b, target: int) -> tuple[int, int]:
    """Decimate the two rubber-hand collision meshes in place, in the builder.

    The hands are the larger half of the triangle budget, not the smaller one: the URDF
    collides them as 45748 and 43852 triangles, 89.6k together against the box's 100k. (The
    5.9-6.4k figure quoted by ``check_geometry`` is the precomputed convex hull in
    ``HAND_HULLS``, which is only used by the ``--hull-hands`` ablation and is not what the
    scene collides.) Every hand triangle multiplies against every box triangle in the broad
    phase, so this compounds with the box reduction rather than adding to it: measured on
    top of ``box_tris=2000``, 64 worlds go 491.8 -> 345.9 ms/step at 5000 per hand and
    373.1 ms at 2000, i.e. the two budgets are tied and the first 9x reduction banks the win.

    Use 5000. Surface deviation would suggest going further -- 0.13 mm mean / 0.89 mm max at
    2000, six times tighter than the box at the same budget -- but the contact probe
    (:mod:`sugar_newton.validation.compare_contacts`, ``--vary hand``) finds a cliff::

        per hand   net load/hand   sum|f|      per-contact corr
        10 000       1.1-1.3 %     2.3-2.6 %      0.95
         5 000       1.8-2.2 %     3.8-4.6 %      0.92-0.94
         2 000      10.1-10.7 %    12-14 %        0.57-0.59   <- past the cliff

    The hand is the contact patch: this grasp is a fingertip pinch touching only 8-14 % of
    the hand, so local curvature at the fingertips sets the contact area directly, while a
    mesh-averaged deviation is dominated by the untouched palm and cannot see it. Since 2000
    is neither faster nor as accurate, it is dominated by 5000 outright.

    Rewrites ``shape_source`` before finalize (the same swap the builder's own remeshing
    path uses, builder.py:7094) and only for collision-enabled shapes, so the visual meshes
    keep their full detail for rendering.
    """
    hands = {i for i, lbl in enumerate(b.body_label)
             if lbl.split("/")[-1] in ("left_rubber_hand", "right_rubber_hand")}
    before = after = 0
    for sh in range(b.shape_count):
        if b.shape_body[sh] not in hands:
            continue
        if not (b.shape_flags[sh] & int(newton.ShapeFlags.COLLIDE_SHAPES)):
            continue
        src = b.shape_source[sh]
        if src is None or getattr(src, "indices", None) is None:
            continue
        tris = np.asarray(src.indices).reshape(-1, 3)
        if len(tris) <= target:
            before += len(tris)
            after += len(tris)
            continue
        v, t = decimate(np.asarray(src.vertices, dtype=np.float64), tris, target)
        before += len(tris)
        after += len(t)
        b.shape_source[sh] = src.copy(vertices=v, indices=t.flatten())
    return before, after


def box_density(verts: np.ndarray, tris: np.ndarray, mass: float) -> float:
    """Density that makes the box weigh exactly ``mass``.

    ``ShapeConfig.density`` defaults to 1000 kg/m^3 and ``add_shape_mesh`` ADDS the shape's
    computed mass and inertia to the body (builder.py:6125-6126). So passing the asset mass
    to ``add_body`` and then leaving the default density does not give a 0.5 kg box: it gives
    0.5 kg plus 3.89 dm^3 of water, i.e. 4.39 kg, 8.8x the mass SUGAR spawns
    (``tactile_objects.py:225``, ``MassPropertiesCfg(mass=0.5)``).

    Deriving the density instead of setting the mass directly also gets the inertia right,
    because the shape's contribution is then scaled consistently -- which is what Isaac does,
    since the asset authors ``physics:density = 0`` and ``diagonalInertia = (0,0,0)`` and
    leaves PhysX to compute the tensor from geometry and the spawned mass.
    """
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    volume = float(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)
    if volume <= 0.0:
        raise ValueError("box mesh has no enclosed volume")
    return mass / volume


def load_clip(name: str) -> dict:
    d = np.load(CLIPS / name / "robot_50hz.npz", allow_pickle=True)
    with open(CLIPS / name / "obj_motion_global_50hz.pkl", "rb") as f:
        obj = pickle.load(f)
    t = d["joint_pos"].shape[0]           # MotionLoader truncates everything to this
    return {
        "fps": float(d["fps"][0]),
        "n": t,
        "joint_pos": np.asarray(d["joint_pos"], dtype=np.float64),
        "joint_vel": np.asarray(d["joint_vel"], dtype=np.float64),
        "body_pos_w": np.asarray(d["body_pos_w"], dtype=np.float64),
        "body_quat_w": np.asarray(d["body_quat_w"], dtype=np.float64),   # wxyz
        "body_lin_vel_w": np.asarray(d["body_lin_vel_w"], dtype=np.float64),
        "body_ang_vel_w": np.asarray(d["body_ang_vel_w"], dtype=np.float64),
        "obj_trans": np.asarray(obj["obj_trans"], dtype=np.float64)[:t],
        "obj_rot": np.asarray(obj["obj_rot"], dtype=np.float64)[:t],
        "contact": np.load(CLIPS / name / "contact_labels_50hz.npy")[:t],
    }


class Actor:
    """The tracker's actor: 510 -> 512 -> 256 -> 128 -> 29, ELU between."""

    def __init__(self, path: Path = ACTOR_NPZ):
        z = np.load(path)
        self.layers = [(z[f"actor_{i}_weight"].astype(np.float64),
                        z[f"actor_{i}_bias"].astype(np.float64)) for i in (0, 2, 4, 6)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))   # ELU
        return x


class G1PolicyScene:
    def __init__(self, clip: dict, box: str = "small", mu: float = 0.75,
                 ke: float = 1.0e4, kd: float = 3.2e2, self_collision: bool = False,
                 hull_hands: bool = False, collision: str = "mesh",
                 sdf_resolution: int = 64, iterations: int = 100,
                 ls_iterations: int = 50, box_tris: int = 0, hand_tris: int = 0,
                 margin: float = 0.005):
        self.clip = clip
        self.margin = margin
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.default_shape_cfg.ke = ke
        b.default_shape_cfg.kd = kd
        b.default_shape_cfg.mu = mu
        # Surface thickness added to every collider. The solver's separation is
        # ``dot(n, p1 - p0) - (margin0 + margin1)`` (contacts.py:65), so this is not merely a
        # detection radius: it inflates each shape, and a pair closes at this separation
        # rather than at touch. Measured on the carry, that means the loaded fingertips sit
        # ~4.5 mm off the box surface while carrying tens of newtons.
        b.default_shape_cfg.margin = margin

        b.add_urdf(str(URDF), floating=True, collapse_fixed_joints=False,
                   enable_self_collisions=self_collision, joint_ordering="bfs",
                   ignore_inertial_definitions=False)
        self.n_robot_bodies = b.body_count

        # DIAGNOSTIC ABLATION ONLY, off by default. Isaac collides this hand as a convex
        # hull (UrdfConverterCfg.collider_type defaults to "convex_hull" and SUGAR never
        # overrides it), and that hull is 2.35x the mesh volume. Switching Newton to the
        # same hull is how the causal claim about wrist torque is tested; it is not how
        # this scene is meant to be run, because a hull is not the hand.
        self.hull_hands = hull_hands
        self.collision = collision
        self.sdf_resolution = sdf_resolution
        self.hand_tris = hand_tris
        if hand_tris:
            n0, n1 = decimate_hand_colliders(b, hand_tris)
            print(f"hand colliders decimated: {n0} -> {n1} triangles")
        if collision == "hydro":
            self._hydroelastic_hands(b, ke, kd, mu)
        if hull_hands:
            hulls = np.load(HAND_HULLS)
            body_of = {}
            for i, lbl in enumerate(b.body_label):
                name = lbl.split("/")[-1]
                if name in ("left_rubber_hand", "right_rubber_hand"):
                    body_of[name.split("_")[0]] = i
            for sh in range(b.shape_count):
                if b.shape_body[sh] in body_of.values():
                    b.shape_flags[sh] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            for side, body_i in body_of.items():
                b.add_shape_mesh(
                    body=body_i,
                    mesh=newton.Mesh(hulls[f"{side}_verts"], hulls[f"{side}_tris"].flatten()),
                    cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, mu=mu),
                    label=f"{side}_hand_hull")

        # Locate the 29 actuated DOFs by walking the per-DOF layout, not by assuming the
        # free joint occupies 6 slots at the front. joint_q and joint_qd disagree for a
        # free joint (7 coords, 6 dofs), and indexing one with the other's offset shifts
        # every actuated joint by one.
        self.act_dofs: list[int] = []
        self.act_coords: list[int] = []
        self.joint_names: list[str] = []
        for j, lbl in enumerate(b.joint_label):
            name = lbl.split("/")[-1]
            n_lin, n_ang = b.joint_dof_dim[j]
            if n_lin + n_ang != 1 or actuator_for(name) is None:
                continue                              # free joints and anything unactuated
            self.act_dofs.append(int(b.joint_qd_start[j]))
            self.act_coords.append(int(b.joint_q_start[j]))
            self.joint_names.append(name)
        if len(self.act_dofs) != N_DOF:
            raise RuntimeError(f"expected {N_DOF} actuated dofs, found {len(self.act_dofs)}")
        self.q_default = default_pose(self.joint_names)
        self.a_scale = action_scale(self.joint_names)

        for i, name in zip(self.act_dofs, self.joint_names):
            k, d, arm, eff = actuator_for(name)
            b.joint_target_ke[i] = k
            b.joint_target_kd[i] = d
            b.joint_armature[i] = arm
            b.joint_effort_limit[i] = eff
            b.joint_target_mode[i] = int(JointTargetMode.POSITION)

        # the box: the asset's own mesh, free, at its reference pose
        verts, tris = load_box_mesh(box)
        if box_tris:
            # Decimate BEFORE deriving the density, so the mass is exact for the mesh that
            # actually generates the inertia rather than for the one it replaced.
            verts, tris = decimate(verts, tris, box_tris)
        # The mesh must compute its inertia in BOTH collision paths: the shape's mass and
        # inertia are what give the box its dynamics now that the density is derived rather
        # than the mass passed to add_body.
        self.box_verts = verts.astype(np.float64)
        body = b.add_body(label="box")
        _bm = newton.Mesh(verts, tris.flatten())
        _bcfg = newton.ModelBuilder.ShapeConfig(
            ke=ke, kd=kd, mu=mu,
            density=box_density(verts.astype(np.float64), tris, BOX_MASS[box]))
        if collision == "hydro":
            # narrow band: this "box" is an open carton, so there is no interior to fill
            _bm.build_sdf(max_resolution=sdf_resolution,
                          narrow_band_range=(-0.006, 0.006), margin=0.004)
            _bcfg.is_hydroelastic = True
            _bcfg.kh = 1.0e10
        b.add_shape_mesh(body=body, mesh=_bm, cfg=_bcfg)
        # NO add_joint_free here. `add_body` already creates the body, its free joint
        # and its own articulation (builder.py:3975); adding another gives the box two
        # parents, and MuJoCo then reports "Loop joint ... skipping loop closure" and
        # never simulates it -- the box's body_q stays all-zero and the observation goes
        # NaN on the first frame. Same trap documented in g1_carrybox.py:312.
        box_joint = next(j for j in range(len(b.joint_label)) if b.joint_child[j] == body)
        self.box_q0 = int(b.joint_q_start[box_joint])
        self.box_qd0 = int(b.joint_qd_start[box_joint])
        self.box_body_builder = body

        # The robot's own free joint: the one parented to the world with 6 dofs.
        root_joint = next(j for j in range(len(b.joint_label))
                          if b.joint_parent[j] == -1 and sum(b.joint_dof_dim[j]) == 6
                          and j != box_joint)
        self.root_q0 = int(b.joint_q_start[root_joint])
        self.root_qd0 = int(b.joint_qd_start[root_joint])
        self.multi_dof = [(b.joint_label[j].split("/")[-1], tuple(b.joint_dof_dim[j]))
                          for j in range(len(b.joint_label))
                          if sum(b.joint_dof_dim[j]) != 1]

        # Measured, not assumed: over this clip the ankles sit at z = 0.03..0.13 and the
        # box's lowest vertex at the reference pose is -0.7 mm. The floor is the origin.
        b.add_ground_plane(height=0.0)

        self.model = b.finalize()
        self.model.request_contact_attributes("force")
        labels = [l.split("/")[-1] for l in self.model.body_label]
        self.anchor_body = labels.index(ANCHOR_LINK)
        self.box_body = labels.index("box")

        _pipe_kw = {}
        if collision == "hydro":
            from newton.geometry import HydroelasticSDF

            # buffer_mult_iso=2 is not enough at low resolution. The iso buffer is sized
            # as buffer_mult_iso * total_num_tiles (sdf_hydroelastic.py:430), so it shrinks
            # with sdf_resolution while the grip's contact demand does not: at resolution
            # 32 the grasp asked for 1280 L1 subblocks against a budget of 960 and MJWarp
            # dropped the excess, exactly like the nconmax truncation before it. 4 clears
            # the measured peak with headroom at both 32 and 64.
            _pipe_kw["sdf_hydroelastic_config"] = HydroelasticSDF.Config(
                output_contact_surface=False, buffer_fraction=1.0, buffer_mult_iso=4)
        self.pipeline = newton.CollisionPipeline(self.model, contact_matching="latest",
                                                 **_pipe_kw,
                                                 contact_report=True)
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton", integrator="implicitfast",
            njmax=16384, nconmax=min(8000, self.contacts.rigid_contact_max),
            impratio=20.0, cone="elliptic", iterations=iterations,
            ls_iterations=ls_iterations, use_mujoco_contacts=False,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.n_q = self.model.joint_coord_count
        self.n_qd = self.model.joint_dof_count

        self.hist: dict[str, collections.deque] = {}
        self.last_action = np.zeros(N_DOF)
        self.frame = 0
        self.graph = None

    def _hydroelastic_hands(self, b, ke, kd, mu) -> None:
        """Rebuild each rubber-hand collider as a hydroelastic SDF mesh.

        The builder's per-shape ``sdf_*`` fields never reach an imported mesh, so the SDF
        has to be built on a fresh :class:`newton.Mesh` and re-added, with the original
        shape's collision switched off so the hand is not counted twice.
        """
        labels = [l.split("/")[-1] for l in b.body_label]
        hands = {i for i, n in enumerate(labels) if n.endswith("_rubber_hand")}
        for sh in [s for s in range(b.shape_count) if b.shape_body[s] in hands]:
            src = b.shape_source[sh]
            if src is None or not hasattr(src, "vertices"):
                continue
            m = newton.Mesh(np.asarray(src.vertices, dtype=np.float32),
                            np.asarray(src.indices, dtype=np.int32).flatten(),
                            compute_inertia=False)
            m.build_sdf(max_resolution=self.sdf_resolution,
                        narrow_band_range=(-0.004, 0.004), margin=0.002)
            cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, mu=mu)
            cfg.is_hydroelastic = True
            cfg.kh = 1.0e10
            new = b.add_shape_mesh(body=b.shape_body[sh], xform=b.shape_transform[sh],
                                   mesh=m, scale=b.shape_scale[sh], cfg=cfg,
                                   label=f"{labels[b.shape_body[sh]]}_skin")
            b.shape_flags[new] &= ~int(newton.ShapeFlags.VISIBLE)
            b.shape_flags[sh] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

    # ---- initial state -------------------------------------------------------
    def reset(self, t0: int = 0) -> None:
        """Put robot and box on the clip's frame ``t0``, at rest in the reference sense."""
        c = self.clip
        q = np.zeros(self.n_q)
        qd = np.zeros(self.n_qd)

        root_p = c["body_pos_w"][t0, 0]
        rq = c["body_quat_w"][t0, 0]                     # clip stores wxyz
        root_q = np.array([rq[1], rq[2], rq[3], rq[0]])
        root_q /= np.linalg.norm(root_q)
        q[self.root_q0:self.root_q0 + 3] = root_p
        q[self.root_q0 + 3:self.root_q0 + 7] = root_q
        q[self.act_coords] = c["joint_pos"][t0, :N_DOF]

        # Free-joint twist is (linear, angular) in the parent frame -- world here.
        qd[self.root_qd0:self.root_qd0 + 3] = c["body_lin_vel_w"][t0, 0]
        qd[self.root_qd0 + 3:self.root_qd0 + 6] = c["body_ang_vel_w"][t0, 0]
        for k, i in enumerate(self.act_dofs):
            qd[i] = c["joint_vel"][t0, k]

        bq = quat_from_mat_xyzw(c["obj_rot"][t0])
        q[self.box_q0:self.box_q0 + 3] = c["obj_trans"][t0]
        q[self.box_q0 + 3:self.box_q0 + 7] = bq

        # Reuse the State/Control buffers allocated in __init__ instead of replacing them.
        # A captured CUDA graph records kernels against specific arrays, so handing out
        # fresh ones would leave the graph integrating the state we just discarded.
        # ``state_1`` needs no clearing: ``solver.step`` overwrites it in full.
        self.state_0.joint_q.assign(q.astype(np.float32))
        self.state_0.joint_qd.assign(qd.astype(np.float32))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

        self.frame = t0
        self.last_action = np.zeros(N_DOF)
        self.hist = {}

    # ---- observation ---------------------------------------------------------
    def _push(self, key: str, value: np.ndarray, length: int = 5) -> np.ndarray:
        """Append and return the flattened window, oldest first.

        IsaacLab's CircularBuffer repeats the first entry until it fills, so a fresh
        buffer is seeded with the current value rather than zero-padded.
        """
        if key not in self.hist:
            self.hist[key] = collections.deque([value.copy()] * length, maxlen=length)
        else:
            self.hist[key].append(value.copy())
        return np.concatenate(list(self.hist[key]))

    def observe(self) -> np.ndarray:
        c = self.clip
        t = min(self.frame, c["n"] - 1)
        q = self.state_0.joint_q.numpy().astype(np.float64)
        qd = self.state_0.joint_qd.numpy().astype(np.float64)
        body_q = self.state_0.body_q.numpy().astype(np.float64)

        root_quat = q[self.root_q0 + 3:self.root_q0 + 7]
        root_quat = root_quat / np.linalg.norm(root_quat)
        omega_w = qd[self.root_qd0 + 3:self.root_qd0 + 6]
        base_ang_vel_b = quat_apply(quat_inv(root_quat), omega_w)
        proj_g = quat_apply(quat_inv(root_quat), np.array([0.0, 0.0, -1.0]))

        joint_pos = q[self.act_coords]
        joint_vel = qd[self.act_dofs]

        # object pose in the ANCHOR (torso_link) frame -- observations.obj_pos_b/obj_ori_b
        a_p, a_q = body_q[self.anchor_body, :3], body_q[self.anchor_body, 3:7]
        o_p, o_q = body_q[self.box_body, :3], body_q[self.box_body, 3:7]
        a_q = a_q / np.linalg.norm(a_q)
        o_q = o_q / np.linalg.norm(o_q)
        if not np.isfinite(np.concatenate([a_q, o_q])).all():
            raise FloatingPointError("anchor or box quaternion is not finite")
        obj_pos_b = quat_apply(quat_inv(a_q), o_p - a_p)
        obj_quat_b = quat_mul(quat_inv(a_q), o_q)
        obj_ori_b = mat_from_quat_xyzw(obj_quat_b)[:, :2].reshape(-1)

        # reference terms: root is body_names[0] = pelvis, velocities in its own frame
        ref_rq = c["body_quat_w"][t, 0]
        ref_rq = np.array([ref_rq[1], ref_rq[2], ref_rq[3], ref_rq[0]])
        ref_rq /= np.linalg.norm(ref_rq)
        ref_lin_b = quat_apply(quat_inv(ref_rq), c["body_lin_vel_w"][t, 0])
        ref_ang_b = quat_apply(quat_inv(ref_rq), c["body_ang_vel_w"][t, 0])

        return np.concatenate([
            c["joint_pos"][t, :N_DOF],                      # 29
            ref_lin_b, ref_ang_b,                           # 3 + 3
            np.array([float(c["contact"][t])]),             # 1
            self._push("ang", base_ang_vel_b),              # 15
            self._push("jp", joint_pos - self.q_default),   # 145
            self._push("jv", joint_vel),                    # 145
            self._push("act", self.last_action),            # 145
            self._push("grav", proj_g),                     # 15
            obj_pos_b, obj_ori_b,                           # 3 + 6
        ]).astype(np.float64)

    # ---- control -------------------------------------------------------------
    def apply(self, action: np.ndarray) -> None:
        """IsaacLab JointPositionAction: target = action * scale + default_joint_pos."""
        self.last_action = action.copy()
        tgt = self.control.joint_target_q.numpy()
        tgt[self.act_dofs] = (action * self.a_scale + self.q_default).astype(tgt.dtype)
        self.control.joint_target_q.assign(tgt)

    def step(self, dt: float, substeps: int = 4, contact_refresh: str = "step") -> None:
        """Advance one 50 Hz control step as ``substeps`` physics steps.

        ``contact_refresh`` decides how often the contact set is regenerated:

        ``step``
            Once per control step, reusing the same contacts for all four physics steps.
            This is Newton's own example convention (``example_robot_g1.py:109-120``).
        ``substep``
            Once per physics step, which is what Isaac does (``sim.dt = 0.005`` with
            ``decimation = 4``, so PhysX collides at 200 Hz) and what this repo's own
            ``example_g1_in_sage.py:425-429`` does.

        ``step`` is the default on measurement, not on principle. The geometric argument for
        ``substep`` is real -- over a 20 ms control step a hand moving at 0.5 m/s travels
        10 mm, well past the 5 mm contact margin, so normals go stale -- but switching to it
        moved the lift by -0.4 % on data_000 and +11 % on data_001 while costing 3-4x
        throughput, which does not justify paying for it in every rollout. Contact-force and
        tactile work should still pass ``substep``: a pressure field read off 20 ms-old
        geometry is wrong in a way that a scalar lift height cannot detect.
        """
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._advance(dt, substeps, contact_refresh)
        self.frame += 1

    def _advance(self, dt: float, substeps: int, contact_refresh: str) -> None:
        sub = dt / substeps
        if contact_refresh == "step":
            self.pipeline.collide(self.state_0, self.contacts)
        for _ in range(substeps):
            self.state_0.clear_forces()
            if contact_refresh == "substep":
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self, dt: float, substeps: int, contact_refresh: str) -> None:
        """Record one control step as a single CUDA graph.

        At ``iterations=100`` the solve is hundreds of tiny kernel launches, so a
        single-world scene is launch-bound rather than arithmetic-bound. Newton's own G1
        example captures the collide as well (``example_robot_g1.py:101-120``), so the
        contact pipeline is capture-safe.

        ``substeps`` must be even: the loop swaps ``state_0``/``state_1`` on every pass, and
        the graph records kernels against specific buffers, so only an even number of swaps
        leaves the Python references where the recording assumed they were.
        """
        if not wp.get_device().is_cuda:
            return
        if substeps % 2:
            raise ValueError(f"graph capture needs an even substep count, got {substeps}")
        # Conditional graph nodes need driver 12.4+ and this box has 12.2, so capture fails
        # outright until they are switched off. Switching them off is not free: the
        # conditional node is what lets the constraint solver exit early once it converges
        # (mujoco_warp/_src/solver.py:3371), so without it every step pays the full
        # `iterations` count. That is why capture must be measured together with the
        # iteration count rather than on its own -- at iterations=100 it is a pessimisation.
        mjw = getattr(self.solver, "mjw_model", None)
        if mjw is not None:
            mjw.opt.graph_conditional = False
        # Warm up first: the capture cannot allocate, and the first call both loads Warp
        # kernel modules and sizes the contact buffers.
        for _ in range(3):
            self._advance(dt, substeps, contact_refresh)
        with wp.ScopedCapture() as cap:
            self._advance(dt, substeps, contact_refresh)
        self.graph = cap.graph

    # ---- readout -------------------------------------------------------------
    def box_pos(self) -> np.ndarray:
        return self.state_0.body_q.numpy()[self.box_body, :3].astype(np.float64)

    def pelvis_z(self) -> float:
        return float(self.state_0.joint_q.numpy()[self.root_q0 + 2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data_000")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--box", default="small")
    ap.add_argument("--mu", type=float, default=0.75)
    ap.add_argument("--ke", type=float, default=1.0e4)
    ap.add_argument("--kd", type=float, default=3.2e2)
    ap.add_argument("--render", default="", help="directory for scene frames (headless EGL)")
    ap.add_argument("--image-format", default="jpg", choices=("png", "jpg"))
    ap.add_argument("--cam-offset", type=float, nargs=3, default=(2.2, -2.2, 0.9))
    ap.add_argument("--collision", default="mesh", choices=("mesh", "hydro"),
                    help="raw triangle meshes, or hydroelastic SDF (handles concavity "
                         "natively, cost scales with resolution not mesh complexity)")
    ap.add_argument("--sdf-resolution", type=int, default=64)
    ap.add_argument("--box-tris", type=int, default=0,
                    help="decimate the box collider to about this many triangles "
                         "(0 = the asset's 100k); 2000 keeps the surface within 1.6 mm")
    ap.add_argument("--hand-tris", type=int, default=0,
                    help="decimate each rubber-hand collider to about this many triangles "
                         "(0 = the asset's 5.9-6.4k)")
    ap.add_argument("--graph", action="store_true",
                    help="capture the control step as one CUDA graph")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--ls-iterations", type=int, default=50)
    ap.add_argument("--contact-refresh", default="step", choices=("substep", "step"),
                    help="regenerate contacts once per control step (default, 3-4x faster "
                         "with no measured lift penalty) or every physics step, as Isaac "
                         "does -- use substep for contact-force and tactile work")
    ap.add_argument("--hull-hands", action="store_true",
                    help="diagnostic ablation: collide the hands as convex hulls, as Isaac does")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    wp.init()
    clip = load_clip(args.clip)
    scene = G1PolicyScene(clip, box=args.box, mu=args.mu, ke=args.ke, kd=args.kd,
                          hull_hands=args.hull_hands, collision=args.collision,
                          sdf_resolution=args.sdf_resolution,
                          iterations=args.iterations, ls_iterations=args.ls_iterations,
                          box_tris=args.box_tris, hand_tris=args.hand_tris)
    actor = Actor()
    print(f"clip {args.clip}: {clip['n']} frames at {clip['fps']:.0f} Hz")
    print(f"model: {scene.model.body_count} bodies, {scene.n_q} coords, {scene.n_qd} dofs, "
          f"{len(scene.act_dofs)} actuated")
    print(f"anchor body '{ANCHOR_LINK}' = {scene.anchor_body}, box body = {scene.box_body}")
    print(f"root free joint at q[{scene.root_q0}:] qd[{scene.root_qd0}:]; "
          f"box free joint at q[{scene.box_q0}:] qd[{scene.box_qd0}:]")
    print(f"non-1-dof joints: {scene.multi_dof}")

    viewer = None
    if args.render:
        import math
        import os

        import pyglet
        from PIL import Image

        if os.environ.get("G1_XVFB") != "1":
            pyglet.options["headless"] = True
        from newton.viewer import ViewerGL

        viewer = ViewerGL(headless=os.environ.get("G1_XVFB") != "1")
        viewer.set_model(scene.model)
        Path(args.render).mkdir(parents=True, exist_ok=True)

        def aim(state):
            """Frame the pelvis and the box together -- the reach is the part to watch."""
            bq = state.body_q.numpy()
            pel = bq[0, :3]
            bx = bq[scene.box_body, :3]
            if not np.isfinite(bx).all():
                bx = pel
            mid = 0.5 * (pel + bx)
            cam = mid + np.asarray(args.cam_offset, dtype=float)
            look = np.array([mid[0], mid[1], mid[2] + 0.15])
            d = look - cam
            d /= max(np.linalg.norm(d), 1e-9)
            viewer.set_camera(
                wp.vec3(*cam.tolist()),
                math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0)))),
                math.degrees(math.atan2(float(d[1]), float(d[0]))))

    scene.reset(args.start)
    _q = scene.state_0.joint_q.numpy()
    _bq = scene.state_0.body_q.numpy()
    print(f"after reset: joint_q[box {scene.box_q0}:] = {np.round(_q[scene.box_q0:], 4)}")
    print(f"             body_q[box {scene.box_body}]  = {np.round(_bq[scene.box_body], 4)}")
    print(f"             body_q[anchor {scene.anchor_body}] = {np.round(_bq[scene.anchor_body], 4)}")
    print(f"             joint_q[root] = {np.round(_q[scene.root_q0:scene.root_q0+7], 4)}")
    print(f"             finite: q {np.isfinite(_q).all()}  body_q {np.isfinite(_bq).all()}")
    dt = 1.0 / clip["fps"]
    n = min(args.frames, clip["n"] - args.start)

    if args.graph:
        # capture() warms up by advancing the scene, so re-seat it afterwards. reset() now
        # reuses the same buffers, which is what keeps the graph valid across it.
        scene.capture(dt, args.substeps, args.contact_refresh)
        scene.reset(args.start)
        print(f"CUDA graph captured (substeps {args.substeps}, "
              f"contact refresh {args.contact_refresh})")

    box0 = scene.box_pos().copy()
    ref0 = clip["obj_trans"][args.start]
    rec = {"box": [], "ref_box": [], "pelvis": [], "action": [],
           "jp": [], "ref_jp": [], "root": [], "ref_root": []}
    t_start = time.perf_counter()
    for k in range(n):
        obs = scene.observe()
        action = actor(obs)
        scene.apply(action)
        scene.step(dt, args.substeps, args.contact_refresh)
        bp = scene.box_pos()
        rec["box"].append(bp.copy())
        rec["ref_box"].append(clip["obj_trans"][min(args.start + k, clip["n"] - 1)].copy())
        rec["pelvis"].append(scene.pelvis_z())
        rec["action"].append(action.copy())
        ti = min(args.start + k, clip["n"] - 1)
        rec["jp"].append(scene.state_0.joint_q.numpy()[scene.act_coords].astype(np.float64))
        rec["ref_jp"].append(clip["joint_pos"][ti, :N_DOF].copy())
        rec["root"].append(scene.state_0.joint_q.numpy()[
            scene.root_q0:scene.root_q0 + 3].astype(np.float64))
        rec["ref_root"].append(clip["body_pos_w"][ti, 0].copy())
        if viewer is not None:
            aim(scene.state_0)
            viewer.begin_frame(k * dt)
            viewer.log_state(scene.state_0)
            viewer.end_frame()
            Image.fromarray(viewer.get_frame().numpy()).save(
                Path(args.render) / f"f{k:05d}.{args.image_format}", quality=92)
        if k % 50 == 0:
            print(f"  f{k:4d}  box z {bp[2]:+.3f} (ref {rec['ref_box'][-1][2]:+.3f})  "
                  f"pelvis z {scene.pelvis_z():+.3f}  |a| {np.abs(action).max():.2f}")
        if not np.isfinite(bp).all() or abs(bp[2]) > 5.0:
            print(f"  DIVERGED at frame {k}")
            break
    el = time.perf_counter() - t_start

    box = np.asarray(rec["box"])
    ref = np.asarray(rec["ref_box"])
    print(f"\n{len(box)} frames in {el:.1f} s ({len(box) / max(el, 1e-9):.1f} fps)")
    print(f"box start z   {box0[2]:+.4f}   (reference {ref0[2]:+.4f})")
    print(f"box peak z    {box[:, 2].max():+.4f}   (reference peak {ref[:, 2].max():+.4f})")
    print(f"box final z   {box[-1, 2]:+.4f}   (reference final {ref[-1, 2]:+.4f})")
    print(f"box lift      {box[:, 2].max() - box0[2]:+.4f} m   "
          f"(reference {ref[:, 2].max() - ref0[2]:+.4f} m)")
    print(f"box travel    {np.linalg.norm(box[-1] - box0):.4f} m   "
          f"(reference {np.linalg.norm(ref[-1] - ref0):.4f} m)")
    print(f"tracking err  mean {np.linalg.norm(box - ref, axis=1).mean():.4f} m   "
          f"final {np.linalg.norm(box[-1] - ref[-1]):.4f} m")
    print(f"pelvis z      min {min(rec['pelvis']):+.3f}  max {max(rec['pelvis']):+.3f}")
    jp, rjp = np.asarray(rec["jp"]), np.asarray(rec["ref_jp"])
    rt, rrt = np.asarray(rec["root"]), np.asarray(rec["ref_root"])
    jerr = np.abs(jp - rjp)
    print(f"\njoint tracking  mean |dq| {jerr.mean():.4f} rad  "
          f"({np.degrees(jerr.mean()):.2f} deg)   max {jerr.max():.4f} rad")
    print(f"  per-window mean |dq| [deg]: " + "  ".join(
        f"f{a}-{b}:{np.degrees(jerr[a:b].mean()):.1f}"
        for a, b in ((0, 100), (100, 200), (200, 300), (300, 400), (400, len(jerr)))))
    worst = np.argsort(-jerr.mean(0))[:5]
    print(f"  worst joints: {[(scene.joint_names[i], round(float(np.degrees(jerr[:, i].mean())), 1)) for i in worst]}")
    print(f"root pos error  mean {np.linalg.norm(rt - rrt, axis=1).mean():.4f} m   "
          f"final {np.linalg.norm(rt[-1] - rrt[-1]):.4f} m")

    # Actuator saturation, the one measured discrepancy against Isaac. Same estimator as
    # the Isaac-side check: PD demand from position error against each joint's effort
    # limit. Not a measured joint torque, so it is comparable only to itself.
    gains = [actuator_for(n) for n in scene.joint_names]
    kk = np.array([g[0] for g in gains])
    eff = np.array([g[3] for g in gains])
    tau = kk * (np.asarray(rec["action"]) * scene.a_scale + scene.q_default - jp)
    sat = np.abs(tau) > eff
    wr = [i for i, n in enumerate(scene.joint_names) if "wrist" in n]
    ot = [i for i, n in enumerate(scene.joint_names) if "wrist" not in n]
    carry = slice(200, min(350, len(tau)))
    print(f"\nactuator saturation (PD estimate)  wrists {100 * sat[:, wr].mean():.1f}%  "
          f"(carry f200-350 {100 * sat[carry][:, wr].mean():.1f}%)   "
          f"non-wrists {100 * sat[:, ot].mean():.1f}%")
    print(f"  peak |tau| {np.abs(tau).max():.1f} N.m over all joints")
    top = np.argsort(-sat.mean(0))[:4]
    print("  most saturated: " + "  ".join(
        f"{scene.joint_names[i]} {100 * sat[:, i].mean():.1f}% "
        f"(mean {np.abs(tau[:, i]).mean():.2f} / lim {eff[i]:.0f})" for i in top))

    if args.out:
        np.savez(args.out, box=box, ref_box=ref, pelvis=np.asarray(rec["pelvis"]),
                 action=np.asarray(rec["action"]), jp=jp, ref_jp=rjp, root=rt, ref_root=rrt)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
