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
on three clips at mu=1.0, substeps=4::

    clip        lift        reference   joint tracking
    data_000    0.23-0.30   0.63        7.7-8.5 deg
    data_001    0.21        0.69        7.8 deg
    data_005    0.24        0.64        7.7 deg

So roughly a third of the reference lift, reproducibly, with the reference joints tracked
to about 8 degrees. Note the 0.23-0.30 spread on data_000: those are two runs at
*identical* parameters. Contact-rich rollouts here are chaotic, and differences below
~0.1 m between single runs should not be read as effects.

Where the gap is not
--------------------
* **Not the policy drifting.** 8 deg joint tracking over 481 closed-loop steps.
* **Not grip.** Sweeping mu 0.5 -> 1.0 -> 1.5 gives lifts of 0.05 -> 0.30 -> 0.30 m; it
  saturates, so adding friction stops helping.
* **Not the hand collider.** Isaac collides this hand as a convex hull
  (``UrdfConverterCfg.collider_type`` defaults to ``"convex_hull"`` and SUGAR never
  overrides it), and that hull is 2.35x the mesh volume -- 135% more material, filling the
  mitten's concavity. That looked like the answer. It is not: the ``--hull-hands``
  ablation, which gives Newton the same hull, *lowers* the lift from 0.23 m to 0.07 m.
  Hypothesis tested and refuted; the flag is kept so the check is repeatable.
* **Not contact compliance.** Sweeping ``ke``/``kd`` over 1e3/3.2e1, 1e4/3.2e2 and
  1e5/1.0e3 gives lifts of 0.17, 0.30, 0.26 m with joint tracking 10.8, 8.6, 19.4 deg.
  The default is already the optimum; softening and stiffening both hurt.

What is measured, and still unexplained
---------------------------------------
Newton demands far more wrist torque than Isaac. Estimating PD demand as
``k * (action * scale + q_default - q)`` against each joint's effort limit::

    joint group      Newton              Isaac (SUGAR's own rollout dump)
    wrists           11.7% at limit      1.9%  (worst single clip 6.6%)
                     37.4% in the carry
    everything else   0.1%               0.0%
    mean |tau|, left wrist pitch
                     5.60 N.m (lim 5.0)  0.25 - 2.83 N.m

Non-wrist joints agree, so legs, waist and shoulders behave the same in both; the whole
discrepancy sits at the hand-box interface, and the wrists are the weakest actuators on
this robot by a factor of ten (5 N.m against 50-139). Peak demand reaches 30.8 N.m, which
is impulsive rather than static. Contact compliance was the obvious candidate and the
``ke`` sweep above rules it out. The next untested one: Isaac hulls *every* link, not just
the hands, and in the reference the box is carried pressed against the chest -- so the
torso collider, which ``--hull-hands`` leaves alone, may be what forms the shelf the box
rests on. These are PD estimates from position error, not measured joint torques.
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
                 hull_hands: bool = False):
        self.clip = clip
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.default_shape_cfg.ke = ke
        b.default_shape_cfg.kd = kd
        b.default_shape_cfg.mu = mu
        b.default_shape_cfg.margin = 0.005

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
        self.box_verts = verts.astype(np.float64)
        body = b.add_body(mass=BOX_MASS[box], label="box")
        b.add_shape_mesh(body=body, mesh=newton.Mesh(verts, tris.flatten()),
                         cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, mu=mu))
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

        self.pipeline = newton.CollisionPipeline(self.model, contact_matching="latest",
                                                 contact_report=True)
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton", integrator="implicitfast",
            njmax=16384, nconmax=min(8000, self.contacts.rigid_contact_max),
            impratio=20.0, cone="elliptic", iterations=100, ls_iterations=50,
            use_mujoco_contacts=False,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.n_q = self.model.joint_coord_count
        self.n_qd = self.model.joint_dof_count

        self.hist: dict[str, collections.deque] = {}
        self.last_action = np.zeros(N_DOF)
        self.frame = 0

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

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
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

    def step(self, dt: float, substeps: int = 4) -> None:
        sub = dt / substeps
        self.pipeline.collide(self.state_0, self.contacts)
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.frame += 1

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
    ap.add_argument("--hull-hands", action="store_true",
                    help="diagnostic ablation: collide the hands as convex hulls, as Isaac does")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    wp.init()
    clip = load_clip(args.clip)
    scene = G1PolicyScene(clip, box=args.box, mu=args.mu, ke=args.ke, kd=args.kd,
                          hull_hands=args.hull_hands)
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

    box0 = scene.box_pos().copy()
    ref0 = clip["obj_trans"][args.start]
    rec = {"box": [], "ref_box": [], "pelvis": [], "action": [],
           "jp": [], "ref_jp": [], "root": [], "ref_root": []}
    t_start = time.perf_counter()
    for k in range(n):
        obs = scene.observe()
        action = actor(obs)
        scene.apply(action)
        scene.step(dt, args.substeps)
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

    if args.out:
        np.savez(args.out, box=box, ref_box=ref, pelvis=np.asarray(rec["pelvis"]),
                 action=np.asarray(rec["action"]), jp=jp, ref_jp=rjp, root=rt, ref_root=rrt)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
