# SPDX-License-Identifier: BSD-3-Clause
"""Vectorised CarryBox environment on Newton, for retraining the tracker under accurate contact.

Why this exists: :mod:`sugar_newton.validation.g1_carrybox_policy` showed that SUGAR's
official ``tracker.pt`` only partially transfers to Newton -- it lifts the box about a
third of the reference height, and the wrists sit at their effort limit 37% of the carry
against Isaac's 1.9%. Whatever the residual cause, the policy was trained against a
different contact model and needs to be retrained (or fine-tuned) against this one.

Design notes that are not obvious:

* **One model, many worlds.** ``ModelBuilder.replicate(world_count=N)`` puts N copies in a
  single Newton model, so one ``solver.step`` advances every environment. Per-world slices
  of ``joint_q``/``joint_qd``/``body_q`` are contiguous and identically shaped, which is
  what makes the torch views below valid.
* **No numpy in the hot loop.** State is read with ``wp.to_torch``, which is a zero-copy
  view of the Warp array, and every observation and reward term is computed in torch on
  the GPU. The validation scripts pull ``.numpy()`` each step; that is fine for one
  environment and hopeless for a thousand.
* **The observation is the validated one.** Same 510-D ``TrackerCfg`` layout that
  :mod:`sugar_newton.validation.verify_tracker_obs` checks against Isaac's own recorded
  actions to RMSE 0.088. Any change here must be re-checked there.

Reward and termination terms are transcribed from SUGAR's
``train_tracker/base_tracker_env_cfg.py`` and ``mdp/rewards.py``; the weights and stds are
theirs, not invented. See :mod:`sugar_newton.rl.rewards`.
"""

from __future__ import annotations

import collections
import pickle
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import warp as wp

import newton
from newton import JointTargetMode

from sugar_newton.rl import rewards as R

HERE = Path(__file__).resolve().parent
SUGAR = HERE.parents[1] / "SUGAR"
URDF = SUGAR / "descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
CLIPS = SUGAR / "data/CarryBox"
BOX_USD = {"small": SUGAR / "descriptions/objects/small_box/obj_aligned.usd",
           "big": SUGAR / "descriptions/objects/big_box/obj_aligned.usd"}
BOX_MASS = {"small": 0.5, "big": 0.75}

ANCHOR_LINK = "torso_link"
N_DOF = 29
OBS_DIM = 510
HIST = 5

# base_tracker_env_cfg.py: MotionCommandCfg.body_names, in this order. Index 0 is the
# reference root and index 7 the anchor; both matter for the observation.
BODY_NAMES = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
              "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
              "torso_link", "left_shoulder_roll_link", "left_elbow_link",
              "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
              "right_wrist_yaw_link")
EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
             "left_wrist_yaw_link", "right_wrist_yaw_link")

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
A_5020, A_7520_14, A_7520_22, A_4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
ACTUATORS = (
    (("hip_pitch_joint", "hip_yaw_joint"), A_7520_14, 88.0, 1.0),
    (("hip_roll_joint", "knee_joint"), A_7520_22, 139.0, 1.0),
    (("ankle_pitch_joint", "ankle_roll_joint"), A_5020, 50.0, 2.0),
    (("waist_roll_joint", "waist_pitch_joint"), A_5020, 50.0, 2.0),
    (("waist_yaw_joint",), A_7520_14, 88.0, 1.0),
    (("wrist_pitch_joint", "wrist_yaw_joint"), A_4010, 5.0, 1.0),
    (("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
      "elbow_joint", "wrist_roll_joint"), A_5020, 25.0, 1.0),
)
DEFAULT_POSE = (("hip_pitch_joint", -0.312), ("knee_joint", 0.669),
                ("ankle_pitch_joint", -0.363), ("elbow_joint", 0.6),
                ("left_shoulder_roll_joint", 0.2), ("left_shoulder_pitch_joint", 0.2),
                ("right_shoulder_roll_joint", -0.2), ("right_shoulder_pitch_joint", 0.2))

# terminations, base_tracker_env_cfg.py: BaseTerminationsCfg
TERM_ANCHOR_ORI = 0.8
TERM_ANCHOR_POS = 0.3
TERM_EE_POS = 0.3
TERM_OBJ_POS = 0.3
TERM_OBJ_ORI = 0.8


def actuator_for(label: str):
    for suffixes, armature, effort, scale in ACTUATORS:
        if any(label.endswith(s) for s in suffixes):
            return (armature * NATURAL_FREQ ** 2 * scale,
                    2.0 * DAMPING_RATIO * armature * NATURAL_FREQ * scale,
                    armature * scale, effort)
    return None


def bfs_body_names(urdf: Path) -> list[str]:
    """Bodies breadth-first, minus the inertialess links the importer merges away."""
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


def box_density(verts: np.ndarray, tris: np.ndarray, mass: float) -> float:
    """Density that makes the box weigh exactly ``mass``.

    ``ShapeConfig.density`` defaults to 1000 kg/m^3 and ``add_shape_mesh`` ADDS the shape's
    mass and inertia to the body (builder.py:6125-6126), so ``add_body(mass=0.5)`` plus a
    default-density mesh gives 4.39 kg -- 8.8x what SUGAR spawns. Deriving the density also
    scales the inertia consistently, which is what Isaac does: the asset authors
    ``physics:density = 0`` and ``diagonalInertia = (0,0,0)`` and lets PhysX compute the
    tensor from geometry and the spawned mass.
    """
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    volume = float(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)
    if volume <= 0.0:
        raise ValueError("box mesh has no enclosed volume")
    return mass / volume


def load_box_mesh(which: str):
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


def load_clips(names: list[str]) -> dict:
    """Stack the reference clips into padded (M, T, ...) tensors, MotionLoader's way.

    Everything is truncated to the robot clip's own length, which is what
    ``commands.py:MotionLoader`` does -- the object pickle and the contact labels are both
    a few frames longer, from separate acquisition.
    """
    robots, objs, contacts, lengths = [], [], [], []
    for name in names:
        d = np.load(CLIPS / name / "robot_50hz.npz", allow_pickle=True)
        with open(CLIPS / name / "obj_motion_global_50hz.pkl", "rb") as f:
            o = pickle.load(f)
        t = d["joint_pos"].shape[0]
        lengths.append(t)
        robots.append({k: np.asarray(d[k])[:t] for k in
                       ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                        "body_lin_vel_w", "body_ang_vel_w")})
        objs.append((np.asarray(o["obj_trans"])[:t],
                     np.stack([quat_from_mat_xyzw(m) for m in np.asarray(o["obj_rot"])[:t]]),
                     np.asarray(o["obj_lin_vel"])[:t], np.asarray(o["obj_ang_vel"])[:t]))
        contacts.append(np.load(CLIPS / name / "contact_labels_50hz.npy")[:t])

    m, tmax = len(names), max(lengths)

    def pad(seq, shape):
        out = np.zeros((m, tmax, *shape), dtype=np.float32)
        for i, a in enumerate(seq):
            out[i, :len(a)] = a
        return out

    return {
        "length": np.asarray(lengths, dtype=np.int64),
        "fps": 50.0,
        "joint_pos": pad([r["joint_pos"] for r in robots], (N_DOF,)),
        "joint_vel": pad([r["joint_vel"] for r in robots], (N_DOF,)),
        "body_pos_w": pad([r["body_pos_w"] for r in robots], (35, 3)),
        "body_quat_w": pad([r["body_quat_w"] for r in robots], (35, 4)),
        "body_lin_vel_w": pad([r["body_lin_vel_w"] for r in robots], (35, 3)),
        "body_ang_vel_w": pad([r["body_ang_vel_w"] for r in robots], (35, 3)),
        "obj_pos": pad([o[0] for o in objs], (3,)),
        "obj_quat": pad([o[1] for o in objs], (4,)),
        "obj_lin_vel": pad([o[2] for o in objs], (3,)),
        "obj_ang_vel": pad([o[3] for o in objs], (3,)),
        "contact": pad([c.astype(np.float32) for c in contacts], ()),
    }


class CarryBoxEnv:
    """N parallel G1 + box worlds driven by 29-D joint-position actions at 50 Hz."""

    def __init__(self, num_envs: int, clip_names: list[str] | None = None,
                 box: str = "small", mu: float = 1.0, ke: float = 1.0e4,
                 kd: float = 3.2e2, substeps: int = 4, episode_length: int = 300,
                 device: str = "cuda:0", seed: int = 0,
                 njmax: int = 16384, nconmax: int = 8192,
                 collision: str = "mesh", sdf_resolution: int = 64,
                 contact_surface: bool = False,
                 contact_refresh: str = "step", box_tris: int = 2000,
                 hand_tris: int = 0, margin: float = 0.0):
        self.num_envs = num_envs
        self.substeps = substeps
        self.episode_length = episode_length
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        if clip_names is None:
            clip_names = sorted(p.name for p in CLIPS.glob("data_*"))
        self.clip_names = clip_names
        raw = load_clips(clip_names)
        self.ref = {k: (torch.as_tensor(v, device=self.device)
                        if isinstance(v, np.ndarray) else v) for k, v in raw.items()}
        self.num_motions = len(clip_names)
        self.dt = 1.0 / raw["fps"]

        self.collision = collision
        self.sdf_resolution = sdf_resolution
        self.contact_refresh = contact_refresh
        self.box_tris = box_tris
        self.hand_tris = hand_tris
        self.margin = margin
        world = self._build_world(box, mu, ke, kd)
        builder = newton.ModelBuilder()
        # Zero spacing on purpose. Worlds do not collide with each other, and Newton's
        # own guidance on replicate() is that physical separation hurts numerical
        # stability -- keeping every world at the origin also means the reference's
        # absolute world coordinates apply directly, with no per-world offset to add and
        # subtract. Use the viewer's set_world_offsets() if worlds need to be seen apart.
        builder.replicate(world, world_count=num_envs, spacing=(0.0, 0.0, 0.0))
        self.model = builder.finalize(device=device)

        # Hydroelastic handles concavity natively -- no decomposition, and its cost scales
        # with sdf_resolution rather than with mesh complexity, which is what removes the
        # per-object tail variance a convex decomposition introduces across a dataset.
        pipe_kwargs = {}
        if collision == "hydro":
            from newton.geometry import HydroelasticSDF

            # buffer_mult_iso is sized as a multiple of total_num_tiles
            # (sdf_hydroelastic.py:430), so the iso buffer shrinks with sdf_resolution while
            # the grip's contact demand does not. At resolution 32 the grasp asked for 1280
            # L1 subblocks against a budget of 960 and the excess was silently dropped --
            # the same failure mode as the nconmax truncation. 4 clears the measured peak.
            pipe_kwargs["sdf_hydroelastic_config"] = HydroelasticSDF.Config(
                output_contact_surface=contact_surface, buffer_fraction=1.0,
                buffer_mult_iso=4)
        # max_triangle_pairs is a whole-pipeline budget with a fixed 1e6 default, but the
        # candidate pairs scale with the number of worlds: at 256 worlds the mesh path asked
        # for 5.58e6 and dropped the rest ("Triangle pair buffer overflowed") -- the same
        # silent-truncation failure as nconmax and the hydroelastic iso buffer.
        #
        # It cannot simply be raised to meet demand. Deterministic contact packing indexes
        # contacts with 20 bits, so 2**20 is a hard ceiling and asking for more raises
        # ValueError rather than overflowing. Measured demand during the grasp is ~19k pairs
        # per world (1.06-1.31e6 across 64 worlds), so the ceiling bites at ~55 worlds: below
        # that the mesh path is correct, above it contacts are dropped no matter what is
        # requested. The 25k estimate below is deliberately conservative, so the warning
        # fires before the dropping starts rather than after.
        #
        # That ceiling, not the solver, is what caps this scene's world count on the raw
        # triangle-mesh path: the box collides as a ~100k-triangle mesh, and no buffer size
        # makes that scale. Reducing the collision geometry is the fix; see the README.
        _TRI_PAIR_CEILING = 1 << 20
        want = max(1_000_000, 25_000 * num_envs)
        if want > _TRI_PAIR_CEILING:
            warnings.warn(
                f"{num_envs} worlds want ~{want} triangle pairs but deterministic contact "
                f"packing caps them at {_TRI_PAIR_CEILING}; contacts will be dropped. Use "
                f"collision='hydro' or a reduced-triangle box for this world count.",
                stacklevel=2)
        pipe_kwargs["max_triangle_pairs"] = min(want, _TRI_PAIR_CEILING)
        self.pipeline = newton.CollisionPipeline(self.model, contact_matching="latest",
                                                 **pipe_kwargs)
        self.contacts = self.pipeline.contacts()
        # njmax and nconmax are PER WORLD (solver_mujoco.py:3183-3184), and they must be
        # sized for the WORST case, not the initial one. Leaving them None lets Newton
        # size from the initial near-static pose, which yields nconmax=1024 -- and then
        # the hand-box grip generates up to 6524 contacts per world, so MJWarp silently
        # drops everything above the limit ("Number of Newton contacts (6524) exceeded
        # MJWarp limit (1024)", 489 times in one short benchmark) and the simulation is
        # simply wrong wherever it matters most.
        # njmax caps constraint ROWS, not contacts, and an elliptic friction cone costs
        # several rows per contact -- so it has to scale with nconmax. The 2048 here was
        # derived from overflow messages logged while nconmax was still truncating at 1024,
        # which is the same mistake twice: sizing one limit from measurements taken while
        # another was silently clipping. 16384 is what the working playback path uses.
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton", integrator="implicitfast",
            njmax=njmax, nconmax=nconmax,
            impratio=20.0, cone="elliptic", iterations=100, ls_iterations=50,
            use_mujoco_contacts=False)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()

        self.nq = self.model.joint_coord_count // num_envs
        self.nqd = self.model.joint_dof_count // num_envs
        self.nbody = self.model.body_count // num_envs

        # zero-copy torch views; every per-world slice below indexes into these
        self.q = wp.to_torch(self.state_0.joint_q).view(num_envs, self.nq)
        self.qd = wp.to_torch(self.state_0.joint_qd).view(num_envs, self.nqd)
        self.target = wp.to_torch(self.control.joint_target_q).view(num_envs, self.nqd)

        idx = torch.as_tensor(self._act_coords, device=self.device)
        self.act_coords = idx
        self.act_dofs = torch.as_tensor(self._act_dofs, device=self.device)
        self.q_default = torch.as_tensor(self._q_default, device=self.device,
                                         dtype=torch.float32)
        self.a_scale = torch.as_tensor(self._a_scale, device=self.device,
                                       dtype=torch.float32)
        self.k = torch.as_tensor(self._k, device=self.device)
        self.d = torch.as_tensor(self._d, device=self.device)
        self.effort = torch.as_tensor(self._effort, device=self.device)
        self.joint_limit_lo = torch.as_tensor(self._lim_lo, device=self.device)
        self.joint_limit_hi = torch.as_tensor(self._lim_hi, device=self.device)
        self.body_idx = torch.as_tensor(self._body_idx, device=self.device)
        self.ee_idx = torch.as_tensor(self._ee_idx, device=self.device)

        self.motion_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.start = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.last_action = torch.zeros(num_envs, N_DOF, device=self.device)
        self.prev_action = torch.zeros(num_envs, N_DOF, device=self.device)
        self.prev_qd = torch.zeros(num_envs, N_DOF, device=self.device)
        self.hist: dict[str, torch.Tensor] = {}
        self.extras: dict[str, torch.Tensor] = {}
        self.num_diverged = 0
        self.diverged = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.reset()

    # ---- construction -------------------------------------------------------
    def _build_world(self, box: str, mu: float, ke: float, kd: float):
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.default_shape_cfg.ke = ke
        b.default_shape_cfg.kd = kd
        b.default_shape_cfg.mu = mu
        # 0 = Newton's own default, and the collider is then exactly the asset surface.
        # Margin is NOT a detection distance: the solver's separation subtracts
        # (margin0 + margin1) (sim/contacts.py:65), so a nonzero margin inflates each shape
        # and a grasp rests that far off the surface. Detection earliness comes from the
        # separate `gap`, which broad phase uses as (margin + gap) and which defaults to
        # 0.1 m. This used to be 5 mm, inherited from the Allegro scene, which left the
        # fingertips carrying the box across 4.5 mm of air.
        b.default_shape_cfg.margin = self.margin
        b.add_urdf(str(URDF), floating=True, collapse_fixed_joints=False,
                   enable_self_collisions=False, joint_ordering="bfs",
                   ignore_inertial_definitions=False)

        if self.hand_tris:
            from sugar_newton.validation.g1_carrybox_policy import decimate_hand_colliders
            decimate_hand_colliders(b, self.hand_tris)

        self._act_dofs, self._act_coords, names = [], [], []
        for j, lbl in enumerate(b.joint_label):
            name = lbl.split("/")[-1]
            n_lin, n_ang = b.joint_dof_dim[j]
            if n_lin + n_ang != 1 or actuator_for(name) is None:
                continue
            self._act_dofs.append(int(b.joint_qd_start[j]))
            self._act_coords.append(int(b.joint_q_start[j]))
            names.append(name)
        if len(self._act_dofs) != N_DOF:
            raise RuntimeError(f"expected {N_DOF} actuated dofs, got {len(self._act_dofs)}")
        self.joint_names = names
        self._q_default = np.array([next((v for p, v in DEFAULT_POSE if n.endswith(p)), 0.0)
                                    for n in names], dtype=np.float32)
        self._a_scale = np.array([0.25 * actuator_for(n)[3] / actuator_for(n)[0]
                                  for n in names], dtype=np.float32)
        self._k = np.array([actuator_for(n)[0] for n in names], dtype=np.float32)
        self._d = np.array([actuator_for(n)[1] for n in names], dtype=np.float32)
        self._effort = np.array([actuator_for(n)[3] for n in names], dtype=np.float32)
        # soft_joint_pos_limit_factor = 0.9 (UnitreeArticulationCfg), applied to the URDF
        # limits -- the joint_limit reward penalises the last 10% of travel
        lim = {j.get("name"): j.find("limit") for j in
               ET.parse(str(URDF)).getroot().findall("joint")}
        self._lim_lo = np.array([float(lim[n].get("lower")) * 0.9 if lim.get(n) is not None
                                 else -np.inf for n in names], dtype=np.float32)
        self._lim_hi = np.array([float(lim[n].get("upper")) * 0.9 if lim.get(n) is not None
                                 else np.inf for n in names], dtype=np.float32)
        for i, name in zip(self._act_dofs, names):
            k, d, arm, eff = actuator_for(name)
            b.joint_target_ke[i] = k
            b.joint_target_kd[i] = d
            b.joint_armature[i] = arm
            b.joint_effort_limit[i] = eff
            b.joint_target_mode[i] = int(JointTargetMode.POSITION)

        if self.collision == "hydro":
            self._make_hydroelastic_hands(b)
        verts, tris = load_box_mesh(box)
        if self.box_tris:
            # Decimate before deriving the density, so the mass is exact for the mesh that
            # generates the inertia. See validation.g1_carrybox_policy.decimate for the
            # measured deviation: at 2000 triangles the surface moves by at most 1.6 mm,
            # against a 5 mm contact margin and a 3.2 mm carton wall.
            from sugar_newton.validation.g1_carrybox_policy import decimate

            verts, tris = decimate(verts, tris, self.box_tris)
        # add_body already creates the free joint and its own articulation; adding another
        # gives the box two parents and MuJoCo silently drops it (g1_carrybox.py:312)
        body = b.add_body(label="box")
        box_mesh = newton.Mesh(verts, tris.flatten())
        box_cfg = newton.ModelBuilder.ShapeConfig(
            ke=ke, kd=kd, mu=mu,
            density=box_density(verts.astype(np.float64), tris, BOX_MASS[box]))
        if self.collision == "hydro":
            # narrow band, not a full interior field: this "box" is an open carton with no
            # well-defined inside, so only a shell around the surface is meaningful
            box_mesh.build_sdf(max_resolution=self.sdf_resolution,
                               narrow_band_range=(-0.006, 0.006), margin=0.004)
            box_cfg.is_hydroelastic = True
            box_cfg.kh = 1.0e10
        b.add_shape_mesh(body=body, mesh=box_mesh, cfg=box_cfg)
        box_joint = next(j for j in range(len(b.joint_label)) if b.joint_child[j] == body)
        self.box_q0 = int(b.joint_q_start[box_joint])
        self.box_qd0 = int(b.joint_qd_start[box_joint])
        root_joint = next(j for j in range(len(b.joint_label))
                          if b.joint_parent[j] == -1 and sum(b.joint_dof_dim[j]) == 6
                          and j != box_joint)
        self.root_q0 = int(b.joint_q_start[root_joint])
        self.root_qd0 = int(b.joint_qd_start[root_joint])

        # measured, not assumed: ankles sit at z = 0.03..0.13 over the clip and the box's
        # lowest vertex at its reference pose is -0.7 mm, so the reference floor is z = 0
        b.add_ground_plane(height=0.0)

        labels = [l.split("/")[-1] for l in b.body_label]
        self.box_body = labels.index("box")
        self._body_idx = [labels.index(n) for n in BODY_NAMES]
        self._ee_idx = [BODY_NAMES.index(n) for n in EE_BODIES]
        self.anchor_local = BODY_NAMES.index(ANCHOR_LINK)
        # clip body ordering is the same BFS, minus the inertialess links
        clip_bodies = bfs_body_names(URDF)
        self.ref_body_idx = torch.as_tensor([clip_bodies.index(n) for n in BODY_NAMES],
                                            device=self.device)
        return b

    def _make_hydroelastic_hands(self, b) -> None:
        """Rebuild each rubber hand's collider as a hydroelastic SDF mesh.

        The builder's per-shape ``sdf_*`` fields never reach an imported mesh, so the SDF
        has to be built on a fresh :class:`newton.Mesh` and re-added -- the same trap
        documented in ``validation/g1_carrybox.py``. The original shape keeps drawing but
        stops colliding, so the hand is not counted twice.
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
            cfg = newton.ModelBuilder.ShapeConfig(
                ke=b.default_shape_cfg.ke, kd=b.default_shape_cfg.kd,
                mu=b.default_shape_cfg.mu)
            cfg.is_hydroelastic = True
            cfg.kh = 1.0e10
            new = b.add_shape_mesh(body=b.shape_body[sh], xform=b.shape_transform[sh],
                                   mesh=m, scale=b.shape_scale[sh], cfg=cfg,
                                   label=f"{labels[b.shape_body[sh]]}_skin")
            b.shape_flags[new] &= ~int(newton.ShapeFlags.VISIBLE)
            b.shape_flags[sh] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

    # ---- state views --------------------------------------------------------
    def _body_q(self) -> torch.Tensor:
        return wp.to_torch(self.state_0.body_q).view(self.num_envs, self.nbody, 7)

    def _body_qd(self) -> torch.Tensor:
        return wp.to_torch(self.state_0.body_qd).view(self.num_envs, self.nbody, 6)

    def _ref(self, key: str) -> torch.Tensor:
        """Reference term for each env at its current timestep, clamped to clip length."""
        t = torch.minimum(self.t, self.ref["length"][self.motion_id] - 1)
        return self.ref[key][self.motion_id, t]

    # ---- reset --------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        self.motion_id[env_ids] = torch.randint(0, self.num_motions, (n,),
                                                device=self.device, generator=self.gen)
        lengths = self.ref["length"][self.motion_id[env_ids]]
        # start anywhere that leaves a full episode; SUGAR's rollout uses the same idea
        span = torch.clamp(lengths - self.episode_length - 1, min=1)
        self.start[env_ids] = (torch.rand(n, device=self.device, generator=self.gen)
                               * span.float()).long()
        self.t[env_ids] = self.start[env_ids]

        mid, t0 = self.motion_id[env_ids], self.t[env_ids]
        q = self.q[env_ids]
        qd = self.qd[env_ids]
        q.zero_()
        qd.zero_()
        root_p = self.ref["body_pos_w"][mid, t0, 0]
        rq = self.ref["body_quat_w"][mid, t0, 0]                      # clip stores wxyz
        q[:, self.root_q0:self.root_q0 + 3] = root_p
        q[:, self.root_q0 + 3:self.root_q0 + 7] = rq[:, [1, 2, 3, 0]]
        q[:, self.act_coords] = self.ref["joint_pos"][mid, t0]
        qd[:, self.root_qd0:self.root_qd0 + 3] = self.ref["body_lin_vel_w"][mid, t0, 0]
        qd[:, self.root_qd0 + 3:self.root_qd0 + 6] = self.ref["body_ang_vel_w"][mid, t0, 0]
        qd[:, self.act_dofs] = self.ref["joint_vel"][mid, t0]
        q[:, self.box_q0:self.box_q0 + 3] = self.ref["obj_pos"][mid, t0]
        q[:, self.box_q0 + 3:self.box_q0 + 7] = self.ref["obj_quat"][mid, t0]
        self.q[env_ids] = q
        self.qd[env_ids] = qd

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.last_action[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        self.prev_qd[env_ids] = 0.0
        for k in self.hist:                                   # CircularBuffer semantics:
            self.hist[k][env_ids] = 0.0                       # refill on the next observe

    # ---- observation --------------------------------------------------------
    def _push(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if key not in self.hist:
            self.hist[key] = value.unsqueeze(1).repeat(1, HIST, 1)
        else:
            self.hist[key] = torch.cat([self.hist[key][:, 1:], value.unsqueeze(1)], dim=1)
        return self.hist[key].reshape(self.num_envs, -1)

    def observe(self) -> torch.Tensor:
        body_q = self._body_q()
        root_quat = R.normalize(self.q[:, self.root_q0 + 3:self.root_q0 + 7])
        omega_w = self.qd[:, self.root_qd0 + 3:self.root_qd0 + 6]
        base_ang_vel_b = R.quat_apply_inv(root_quat, omega_w)
        grav = torch.zeros(self.num_envs, 3, device=self.device)
        grav[:, 2] = -1.0
        proj_g = R.quat_apply_inv(root_quat, grav)

        joint_pos = self.q[:, self.act_coords]
        joint_vel = self.qd[:, self.act_dofs]

        a_p, a_q = self._anchor_pose(body_q)
        o_p = body_q[:, self.box_body, :3]
        o_q = R.normalize(body_q[:, self.box_body, 3:7])
        obj_pos_b = R.quat_apply_inv(a_q, o_p - a_p)
        obj_quat_b = R.quat_mul(R.quat_conj(a_q), o_q)
        obj_ori_b = R.mat_from_quat(obj_quat_b)[..., :2].reshape(self.num_envs, 6)

        ref_rq = self._ref("body_quat_w")[:, 0]
        ref_rq = R.normalize(ref_rq[:, [1, 2, 3, 0]])
        ref_lin_b = R.quat_apply_inv(ref_rq, self._ref("body_lin_vel_w")[:, 0])
        ref_ang_b = R.quat_apply_inv(ref_rq, self._ref("body_ang_vel_w")[:, 0])

        return torch.cat([
            self._ref("joint_pos"), ref_lin_b, ref_ang_b,
            self._ref("contact").unsqueeze(-1),
            self._push("ang", base_ang_vel_b),
            self._push("jp", joint_pos - self.q_default),
            self._push("jv", joint_vel),
            self._push("act", self.last_action),
            self._push("grav", proj_g),
            obj_pos_b, obj_ori_b,
        ], dim=-1)

    def _anchor_pose(self, body_q: torch.Tensor):
        i = self.body_idx[self.anchor_local]
        return body_q[:, i, :3], R.normalize(body_q[:, i, 3:7])

    # ---- step ---------------------------------------------------------------
    def step(self, action: torch.Tensor):
        self.prev_action = self.last_action
        self.last_action = action
        self.prev_qd = self.qd[:, self.act_dofs].clone()

        # IsaacLab JointPositionAction: target = action * scale + default_joint_pos
        self.target[:, self.act_dofs] = action * self.a_scale + self.q_default

        # contact_refresh="substep" regenerates contacts every physics step, which is what
        # Isaac does (sim.dt=0.005 with decimation=4, so PhysX collides at 200 Hz) and what
        # example_g1_in_sage.py:425-429 does. It is NOT the default: measured on the
        # playback path it moved the lift by -0.4% and +11% on two clips while costing 3-4x
        # throughput. Pass it when reading contact forces or tactile fields, where 20 ms-old
        # normals are wrong in a way a scalar lift height cannot detect.
        sub = self.dt / self.substeps
        if self.contact_refresh == "step":
            self.pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            if self.contact_refresh == "substep":
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self._rebind()
        self.t += 1

        done, timeout = self._done()
        reward, terms = R.compute(self)
        # A diverged env is reset on this same call, so its reward is meaningless rather
        # than merely bad. It is also often huge and FINITE -- joint_acc on exploding
        # velocities reached -1.7e5 and swamped every other env in the batch -- so
        # nan_to_num alone is not enough and the flagged envs are zeroed outright.
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = torch.where(self.diverged, torch.zeros_like(reward), reward)
        self.extras = {"reward_terms": terms, "timeout": timeout}
        obs = self.observe()
        reset_ids = done.nonzero(as_tuple=False).flatten()
        if reset_ids.numel():
            self.reset(reset_ids)
            obs = self.observe()
        return obs, reward, done, self.extras

    def _rebind(self) -> None:
        """Re-take the torch views after the double-buffer swap."""
        self.q = wp.to_torch(self.state_0.joint_q).view(self.num_envs, self.nq)
        self.qd = wp.to_torch(self.state_0.joint_qd).view(self.num_envs, self.nqd)

    def _done(self):
        body_q = self._body_q()
        a_p, a_q = self._anchor_pose(body_q)
        ref_body_p = self._ref("body_pos_w")[:, self.ref_body_idx]
        ref_body_q = self._ref("body_quat_w")[:, self.ref_body_idx][..., [1, 2, 3, 0]]
        rob_body_p = body_q[:, self.body_idx, :3]

        grav = torch.zeros(self.num_envs, 3, device=self.device)
        grav[:, 2] = -1.0
        ref_a_q = R.normalize(ref_body_q[:, self.anchor_local])
        bad_ori = (R.quat_apply_inv(ref_a_q, grav)
                   - R.quat_apply_inv(a_q, grav)).norm(dim=-1) > TERM_ANCHOR_ORI
        bad_anchor = (ref_body_p[:, self.anchor_local] - a_p).norm(dim=-1) > TERM_ANCHOR_POS
        bad_ee = ((ref_body_p[:, self.ee_idx] - rob_body_p[:, self.ee_idx])
                  .norm(dim=-1) > TERM_EE_POS).any(dim=-1)
        obj_p = body_q[:, self.box_body, :3]
        ref_obj_p = self._ref("obj_pos")
        bad_obj = (ref_obj_p - obj_p).norm(dim=-1) > TERM_OBJ_POS
        bad_obj_ori = R.quat_angle(R.normalize(body_q[:, self.box_body, 3:7]),
                                   R.normalize(self._ref("obj_quat"))) > TERM_OBJ_ORI

        # A contact-rich solve occasionally blows up; that env's state (and therefore its
        # reward) goes non-finite. Treat divergence as a termination so the env resets
        # instead of poisoning the rollout, and count it so it cannot hide.
        diverged = ~(torch.isfinite(body_q).all(dim=(1, 2))
                     & torch.isfinite(self.q).all(dim=1)
                     & torch.isfinite(self.qd).all(dim=1))
        self.num_diverged += int(diverged.sum())
        self.diverged = diverged

        timeout = ((self.t - self.start) >= self.episode_length) | \
                  (self.t >= self.ref["length"][self.motion_id] - 1)
        fail = bad_ori | bad_anchor | bad_ee | bad_obj | bad_obj_ori | diverged
        return fail | timeout, timeout
