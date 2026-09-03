"""Build the Newton model from SUGAR's own scene config, and step it.

This is the floor of the swap. Everything above reads the simulator through
`sugar_swap.data`; this module is what puts a simulator there, and it takes its instructions
from SUGAR's `InteractiveSceneCfg` rather than from constants of our own. That distinction is
the point: the hand-written port in `sugar_newton/` derived its gains from a natural-frequency
formula and its box mass from a table, and both had to be reconciled with SUGAR afterwards.
Here the config is the source of truth, so there is nothing to reconcile.

Read from SUGAR's config and applied:

- the URDF path, and `enabled_self_collisions` (True)
- `solver_position_iteration_count` / `solver_velocity_iteration_count` (8 / 4), the fields
  the fidelity report flagged as dropped
- per-actuator `stiffness`, `damping`, `velocity_limit_sim` and `effort_limit_sim`, resolved
  through IsaacLab's own regex matcher so the joint groups are identical
- `init_state.joint_pos` as the default pose, and `soft_joint_pos_limit_factor` (0.9)
- the object's `mass_props.mass` (0.5 kg)

Taken from the Newton port instead, because SUGAR leaves it to Isaac Sim's URDF importer:
joint armature, via the verified `actuator_for` table.

Effort limits are NOT in that category, though they were treated as such and it cost us the
grasp. `effort_limit` is indeed None throughout SUGAR's actuator configs, but `effort_limit_sim`
is set on every group, and that is the field IsaacLab writes to the simulator -- so it wins
over the URDF rather than deferring to it. Preferring the URDF ran both ankles and the waist
(pitch and roll) at 35 N*m against SUGAR's 50, and because the action scale is
`0.25 * effort_limit_sim / stiffness`, the policy commands a displacement those six joints
then cannot reach. The robot stopped bending short of the box and never touched it.

Colliders are convex hulls. That is what makes training tractable -- 42x measured, because
convex hulls skip the triangle midphase entirely -- and it matches PhysX, whose G1 links are
convex hulls anyway. It is *not* tactile-grade: per-taxel contact patterns need the mesh
path, so a policy trained here is for locomotion fidelity, not contact fidelity.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

import newton
from newton import JointTargetMode

from .assets import Articulation, RigidObject
from .scene import InteractiveScene
from .sensors import ContactSensor

_HERE = Path(__file__).resolve().parent
_SUGAR = _HERE.parent / "SUGAR"

# MuJoCo-Warp allocates and *iterates over* declared capacity, not live count: `nconmax`
# sets the launch dimension of both contact-conversion kernels and `njmax` sizes the
# per-world constraint arrays the solver sweeps. Measured at 4096 worlds with convex
# colliders (`experiments/perf/out/`): 218 live contacts and 88 constraint rows per world at
# peak, against the 8192 / 16384 these were before. Sized here to ~4.7x and ~23x the
# measured peak, which is past the knee of the measured cost curve -- 512 / 1024 is only
# 2.6 % faster and gives up most of the margin.
#
# Overflow is loud, not silent: Newton prints "Contact buffer overflowed" and MuJoCo-Warp
# "Number of Newton contacts (N) exceeded MJWarp limit", both per step. Do not raise these
# from a run in which either message appeared -- that is sizing one limit while another
# clips. Re-derive from `swap_perf.py --mode caps`, which reports the peak against the cap.
NCONMAX_PER_WORLD = 1024
NJMAX_PER_WORLD = 2048

# The single most expensive line in this file, and the one least justified by SUGAR's config.
# Neither of these comes from SUGAR: MuJoCo's own defaults are "pyramidal" and 1.0, and they
# were set here to make contact stricter than MuJoCo's default. That is the opposite
# direction from the engine being matched -- PhysX linearises friction into a pyramid (two
# box-constrained tangential directions per contact) and offers no elliptic cone at all, so
# "elliptic" is not PhysX parity, it is a model PhysX cannot express.
#
# It costs 1.48x. Measured at 4096 envs, everything else at the settings below:
# elliptic 5,550 env-steps/s, pyramidal 8,231. An elliptic cone couples the tangential rows
# of each contact, so the solver cannot treat a row independently; pyramidal leaves them
# box-constrained and separable. `impratio` only sharpens that coupling.
#
# Pyramidal is not merely the cheaper choice, it is the more FAITHFUL one, which is why the
# default flipped. Against the IsaacLab reference the worst contact-force disagreement falls
# from 949 N to 67 N, step-1 agreement rises from 605/1170 terms to 798/1170, and the
# `joint_torque` reward scale error drops from 1.81x to 0.987x. Mean penetration also improves
# (1.50 mm -> 1.32 mm). The pyramid IS anisotropic (up to sqrt(2) mu on the diagonal), so it is
# a worse model of Coulomb friction in the abstract -- but it is the same approximation PhysX
# makes, and reproducing SUGAR is the objective, not out-modelling it.
#
# IMPRATIO STAYS AT 20 with the pyramid. MuJoCo's default of 1.0 looks like the matching
# choice and is a trap: measured at 4096 envs, pyramidal/20 runs 8,092 env-steps/s while
# pyramidal/1.0 runs 3,649 -- slower than elliptic. impratio conditions the friction rows
# relative to the normal ones, and slackening it makes the solve harder, not cheaper.
#
# This IS a dynamics change. Adopt it by retraining; do not switch under a policy trained on
# elliptic. RB_CONE / RB_IMPRATIO override, which is how an elliptic control run is pinned
# while the pyramidal run trains beside it.
CONE = os.environ.get("RB_CONE", "pyramidal")
IMPRATIO = float(os.environ.get("RB_IMPRATIO", "20.0"))

# `enabled_self_collisions` still comes from SUGAR's config; this only decides which pairs
# the importer is allowed to collide. Weld-aware parent filtering is the DEFAULT because it
# is what MuJoCo's `filterparent` and PhysX's articulation adjacency rule both do and what
# Newton's importer does not. Running without it left 9 phantom pairs of 657 live, which
# carried 20.6 kN against a 33 kg robot and pinned the hip bow at 25 deg where IsaacLab
# reaches 58, costing the entire box lift (0.057 m vs 0.650 m fixed). Every Newton refiner
# run before this flag is therefore invalid. See `_weld_filter_pairs` and
# experiments/selfcollide. Overrides:
#
#   RB_SELF_COLLISION=raw   Newton's own filtering only, i.e. reinstate the bug. Kept so the
#                           pre-fix runs remain reproducible for A/B.
#   RB_SELF_COLLISION=0     filter every robot-vs-robot pair. A diagnostic only -- SUGAR
#                           trained with self collision on, so a policy evaluated without it
#                           is evaluated off-distribution.
SELF_COLLISION = os.environ.get("RB_SELF_COLLISION", "weld")


def _resolve_asset(path: str | None) -> Path:
    """SUGAR's asset paths are relative to `SUGAR/`, since its scripts run from there."""
    if not path:
        raise ValueError("sugar_swap: scene config has no asset path")
    clean = str(path).lstrip("./")
    for root in (_SUGAR, _HERE.parent):
        candidate = root / clean
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"sugar_swap: cannot find asset {path!r} under {_SUGAR}")


def _expand(spec: Any, names: list[str], default: float | None = None) -> np.ndarray:
    """Expand an IsaacLab actuator field over joints.

    The field is either a scalar applying to the whole group or a dict of regex -> value.
    Matching uses IsaacLab's own `resolve_matching_names_values`, so a pattern selects the
    same joints it would on Isaac Sim; reimplementing the matcher is how an action vector
    silently gets permuted.
    """
    out = np.full(len(names), np.nan if default is None else default, dtype=np.float32)
    if spec is None:
        return out
    if isinstance(spec, dict):
        from isaaclab.utils.string import resolve_matching_names_values

        idx, _, values = resolve_matching_names_values(spec, names, preserve_order=False)
        for i, v in zip(idx, values):
            out[i] = float(v)
    else:
        out[:] = float(spec)
    return out


def _inertialess_links(urdf: Path) -> set[str]:
    """URDF links Isaac Sim's importer merges into their parent, and Newton keeps.

    This is the body-ordering contract, and it is load-bearing. SUGAR resolves its
    `body_names` against `robot.body_names` and then uses those same integers to index the
    *reference clip's* body axis (`MotionLoader._body_indexes`), so the articulation's body
    list has to be the one the clips were recorded against. Isaac Sim drops a link with no
    `<inertial>`; Newton keeps it, which is why the G1 imports as 39 bodies against the
    clip's 35. The four extras are pure sensor frames with no visual or collision geometry,
    so hiding them from the asset changes the index space and nothing physical.
    """
    root = ET.parse(str(urdf)).getroot()
    return {l.get("name") for l in root.findall("link") if l.find("inertial") is None}


def _weld_filter_pairs(b: Any, urdf: Path) -> int:
    """Filter the shape pairs a weld-aware parent rule excludes and Newton's does not.

    Newton's importer filters shapes on the same body and shapes on bodies joined directly
    by a joint, but neither rule looks THROUGH a fixed joint. MuJoCo's `filterparent` and
    PhysX's articulation adjacency filter both do: bodies with zero degrees of freedom
    between them are one rigid unit, and contacts are excluded within such a unit and
    between it and its parent unit.

    On this URDF the gap is load-bearing rather than cosmetic. `pelvis` declares no
    collision geometry; the pelvis collider is on `pelvis_contour_link`, welded to `pelvis`.
    Newton therefore filters `pelvis` against its children -- which filters nothing, since
    `pelvis` has no shape -- and leaves the real pelvis collider colliding with `pelvis`'s
    own children, the two `*_hip_pitch_link`s. Their hulls interfere by up to 1 mm from
    about 10 to 53 degrees of hip flexion, and the resulting constraint carried 18.5 kN
    against a 33 kg robot in the refiner replay, pinning the bow at 24 deg where IsaacLab
    reaches 58. See experiments/selfcollide.

    Returns the number of pairs filtered.
    """
    root = ET.parse(str(urdf)).getroot()
    parent, jtype = {}, {}
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        parent[child] = j.find("parent").get("link")
        jtype[child] = j.get("type")

    def weld_of(link: str) -> str:
        while jtype.get(link) == "fixed":
            link = parent[link]
        return link

    links = [l.get("name") for l in root.findall("link")]
    weld = {name: weld_of(name) for name in links}
    weld_parent = {
        w: weld[parent[w]] for w in set(weld.values()) if w in parent
    }

    labels = [str(l).split("/")[-1] for l in b.body_label]
    shapes_of: dict[int, list[int]] = {}
    for shape, body in enumerate(b.shape_body):
        shapes_of.setdefault(int(body), []).append(shape)

    filtered = 0
    for body_a, shapes_a in shapes_of.items():
        for body_b, shapes_b in shapes_of.items():
            if body_b <= body_a or body_a < 0:
                continue
            wa, wb = weld.get(labels[body_a]), weld.get(labels[body_b])
            if wa is None or wb is None:
                continue
            if wa != wb and weld_parent.get(wa) != wb and weld_parent.get(wb) != wa:
                continue
            for sa in shapes_a:
                for sb in shapes_b:
                    b.add_shape_collision_filter_pair(sa, sb)
                    filtered += 1
    # Most of these Newton had already excluded under its own same-body and directly-jointed
    # rules; re-filtering them is a no-op. On this URDF 176 are requested and only 9 are new,
    # so do not read this count as the number of pairs the fix actually changes -- compare
    # the live pair count before and after for that.
    print(
        f"sugar_swap: RB_SELF_COLLISION=weld -> {filtered} weld-adjacent shape pairs filtered"
        " (most already excluded by Newton's own rules)"
    )
    return filtered


def _urdf_limits(urdf: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position limits and effort from the URDF, which is where PhysX gets them too."""
    limits = {j.get("name"): j.find("limit") for j in ET.parse(str(urdf)).getroot().findall("joint")}
    lo, hi, eff = [], [], []
    for name in names:
        node = limits.get(name)
        lo.append(float(node.get("lower")) if node is not None and node.get("lower") else -np.inf)
        hi.append(float(node.get("upper")) if node is not None and node.get("upper") else np.inf)
        eff.append(float(node.get("effort")) if node is not None and node.get("effort") else 0.0)
    return (
        np.asarray(lo, dtype=np.float32),
        np.asarray(hi, dtype=np.float32),
        np.asarray(eff, dtype=np.float32),
    )


def _build_world(scene_cfg: Any) -> tuple[Any, dict[str, Any]]:
    """Assemble one world: G1 from the URDF, the box, and a ground plane."""
    from sugar_newton.rl.carrybox_env import actuator_for, box_density, load_box_mesh

    robot_cfg = scene_cfg.robot
    spawn = robot_cfg.spawn
    urdf = _resolve_asset(getattr(spawn, "asset_path", None))
    props = getattr(spawn, "articulation_props", None)
    self_collisions = bool(getattr(props, "enabled_self_collisions", True))
    mode = (SELF_COLLISION or "").strip().lower()
    if mode in ("0", "false", "off", "no"):
        self_collisions = False
        print("sugar_swap: RB_SELF_COLLISION -> all robot self-collision filtered")

    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    # margin 0: the solver subtracts (margin0 + margin1) from separation, so any nonzero
    # value leaves the grasp resting that far off the surface. This was a real bug once.
    b.default_shape_cfg.margin = 0.0
    b.add_urdf(
        str(urdf),
        floating=True,
        collapse_fixed_joints=False,
        enable_self_collisions=self_collisions,
        joint_ordering="bfs",
        ignore_inertial_definitions=False,
    )

    # Actuated joints, in Newton's BFS order. Single-DOF joints only, which excludes the
    # floating base and the box's free joint.
    act_dofs, act_coords, names = [], [], []
    for j, label in enumerate(b.joint_label):
        name = label.split("/")[-1]
        n_lin, n_ang = b.joint_dof_dim[j]
        if n_lin + n_ang != 1 or actuator_for(name) is None:
            continue
        act_dofs.append(int(b.joint_qd_start[j]))
        act_coords.append(int(b.joint_q_start[j]))
        names.append(name)

    # Gains come from SUGAR's actuator groups, unioned over the five groups.
    stiffness = np.full(len(names), np.nan, dtype=np.float32)
    damping = np.full(len(names), np.nan, dtype=np.float32)
    vel_limit = np.full(len(names), np.nan, dtype=np.float32)
    cfg_effort = np.full(len(names), np.nan, dtype=np.float32)
    for group in (robot_cfg.actuators or {}).values():
        selected = _expand({p: 1.0 for p in _patterns(group)}, names)
        mask = ~np.isnan(selected)
        for target, field in (
            (stiffness, "stiffness"),
            (damping, "damping"),
            (vel_limit, "velocity_limit_sim"),
            (cfg_effort, "effort_limit_sim"),
        ):
            values = _expand(getattr(group, field, None), names)
            target[mask & ~np.isnan(values)] = values[mask & ~np.isnan(values)]

    lo, hi, urdf_effort = _urdf_limits(urdf, names)
    # `effort_limit_sim` outranks both the URDF and the table: it is what IsaacLab hands the
    # simulator, and it is also the numerator of the action scale the policy was trained
    # against. See the header for what preferring the URDF cost.
    effort = np.array([actuator_for(n)[3] for n in names], dtype=np.float32)
    effort = np.where(urdf_effort > 0.0, urdf_effort, effort)
    effort = np.where(np.isnan(cfg_effort), effort, cfg_effort)
    armature = np.array([actuator_for(n)[2] for n in names], dtype=np.float32)

    for slot, dof in enumerate(act_dofs):
        b.joint_target_ke[dof] = float(stiffness[slot])
        b.joint_target_kd[dof] = float(damping[slot])
        b.joint_armature[dof] = float(armature[slot])
        b.joint_effort_limit[dof] = float(effort[slot])
        if not np.isnan(vel_limit[slot]):
            b.joint_velocity_limit[dof] = float(vel_limit[slot])
        b.joint_target_mode[dof] = int(JointTargetMode.POSITION)

    if mode == "weld":
        _weld_filter_pairs(b, urdf)

    # ---- the carried object -------------------------------------------------
    obj_cfg = getattr(scene_cfg, "obj", None) or getattr(scene_cfg, "object")
    usd = str(getattr(obj_cfg.spawn, "usd_path", ""))
    key = "small" if "small" in usd else "big"
    mass = float(getattr(getattr(obj_cfg.spawn, "mass_props", None), "mass", 0.5) or 0.5)
    verts, tris = load_box_mesh(key)
    # add_body creates the free joint itself; adding a second one gives the box two parents
    # and MuJoCo silently drops it.
    box_body = b.add_body(label="box")
    box_cfg = newton.ModelBuilder.ShapeConfig(
        density=box_density(verts.astype(np.float64), tris, mass)
    )
    b.add_shape_mesh(body=box_body, mesh=newton.Mesh(verts, tris.flatten()), cfg=box_cfg)

    # Hull after every collision shape exists: hulling first would leave later shapes on the
    # original mesh, and the two would not be the same kind of collider.
    b.approximate_meshes(method="convex_hull", raise_on_failure=True)
    b.add_ground_plane(height=0.0)

    labels = [l.split("/")[-1] for l in b.body_label]
    box_joint = next(j for j in range(len(b.joint_label)) if b.joint_child[j] == box_body)
    root_joint = next(
        j
        for j in range(len(b.joint_label))
        if b.joint_parent[j] == -1 and sum(b.joint_dof_dim[j]) == 6 and j != box_joint
    )
    root_body = next(i for i in range(len(b.body_label)) if b.joint_child[root_joint] == i)

    hidden = _inertialess_links(urdf)

    # IsaacLab shrinks the soft limits toward the *centre of the range*, not toward zero:
    # `mid +/- factor * half_range` (Articulation._initialize_impl). Scaling each bound by the
    # factor instead is the same thing only for a range centred on zero, and 14 of the G1's 29
    # joints are asymmetric. Measured on the left knee, `0.9 * bound` gives [-0.079, 2.592]
    # where IsaacLab gives [0.061, 2.731] -- so `joint_limit` penalised a band IsaacLab treats
    # as legal, by up to 50%. Found by the per-term diff in experiments/equiv.
    soft_factor = float(getattr(robot_cfg, "soft_joint_pos_limit_factor", 1.0) or 1.0)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)

    meta = {
        "joint_names": names,
        "act_coords": act_coords,
        "act_dofs": act_dofs,
        "body_labels": labels,
        "robot_body_names": [n for n in labels if n != "box" and n not in hidden],
        "box_body": labels.index("box"),
        "root_body": root_body,
        "box_q0": int(b.joint_q_start[box_joint]),
        "root_q0": int(b.joint_q_start[root_joint]),
        "box_qd0": int(b.joint_qd_start[box_joint]),
        "root_qd0": int(b.joint_qd_start[root_joint]),
        "stiffness": stiffness,
        "damping": damping,
        "effort": effort,
        "soft_lo": mid - soft_factor * half,
        "soft_hi": mid + soft_factor * half,
        "lo": lo,
        "hi": hi,
        "shape_body": list(b.shape_body),
        "shapes_per_world": len(b.shape_body),
        "bodies_per_world": len(b.body_label),
    }
    return b, meta


def _patterns(group: Any) -> list[str]:
    """Joint patterns a group covers, whether declared explicitly or implied by its gains."""
    explicit = getattr(group, "joint_names_expr", None)
    if explicit:
        return list(explicit)
    stiffness = getattr(group, "stiffness", None)
    return list(stiffness) if isinstance(stiffness, dict) else [".*"]


def build_scene(scene_cfg: Any, device: str) -> InteractiveScene:
    """Build the Newton model and wrap it in the objects SUGAR's terms expect."""
    num_envs = int(scene_cfg.num_envs)
    world, meta = _build_world(scene_cfg)

    builder = newton.ModelBuilder()
    # Zero spacing: worlds do not collide, Newton's guidance is that separation hurts
    # numerical stability, and co-locating means the reference clip's absolute world
    # coordinates apply directly with no per-world offset.
    builder.replicate(world, world_count=num_envs, spacing=(0.0, 0.0, 0.0))
    model = builder.finalize(device=device)

    scene = InteractiveScene(scene_cfg, device)
    scene.model = model
    scene.state_0, scene.state_1 = model.state(), model.state()
    scene.control = model.control()
    scene._bodies_per_env = meta["bodies_per_world"]
    scene._shapes_per_env = meta["shapes_per_world"]
    scene.total_bodies = meta["bodies_per_world"] * num_envs
    scene.shape_body = torch.as_tensor(
        _tile_shape_body(meta, num_envs), dtype=torch.long, device=device
    )

    # Free joints span 7 coordinates but 6 degrees of freedom, so the two index spaces
    # diverge and the writers need the mapping.
    scene._coord_to_dof = {meta["root_q0"]: meta["root_qd0"], meta["box_q0"]: meta["box_qd0"]}

    # SUGAR's contact rewards and terminations read forces, so the pipeline must populate
    # them. `force` is an opt-in attribute the model has to be asked for before the
    # pipeline allocates, and it must be requested before `contacts()`.
    model.request_contact_attributes("force")

    # `SolverMuJoCo.update_contacts` refuses to run when the solver's total capacity
    # (nconmax * worlds) exceeds the Newton buffer it writes back into, so both are sized
    # from one number rather than left to Newton's estimator. That estimator budgets 20
    # neighbours per shape at up to 40 contacts each and produced 15,414 slots per world
    # against a measured peak of 218 -- and because `nconmax` used to be read back off it,
    # the over-estimate propagated into the solver, where capacity costs time.
    nconmax = NCONMAX_PER_WORLD
    # `contact_matching` and `contact_report` are left at Newton's defaults (off). Nothing
    # here reads `rigid_contact_match_index` or the new/broken index arrays they fill, and
    # any matching mode forces `deterministic=True`, which adds a radix sort plus a gather
    # over the whole contact buffer to every `collide`. Dropping them costs run-to-run
    # bitwise reproducibility of contact *ordering* -- the contact set is unchanged.
    scene.pipeline = newton.CollisionPipeline(
        model, rigid_contact_max=nconmax * max(num_envs, 1)
    )
    scene.contacts = scene.pipeline.contacts()

    props = getattr(scene_cfg.robot.spawn, "articulation_props", None)
    scene.solver = newton.solvers.SolverMuJoCo(
        model,
        solver="newton",
        integrator="implicitfast",
        # njmax caps constraint ROWS, which an elliptic cone spends several of per contact,
        # so it is kept above nconmax -- but it is the more expensive of the two: at 4096
        # envs dropping njmax alone from 16384 to 2048 is 1.09x, dropping nconmax alone from
        # 8192 to 1024 is 1.04x. The solver's dense per-row work is what scales with it.
        njmax=NJMAX_PER_WORLD,
        nconmax=nconmax,
        impratio=IMPRATIO,
        cone=CONE,
        # `iterations` is the honest counterpart of PhysX's position iteration count: both cap
        # the outer solver loop. `ls_iterations` is NOT the counterpart of PhysX's velocity
        # iteration count -- it bounds MuJoCo's LINE SEARCH inside each outer iteration, a
        # different algorithm with no PhysX equivalent, and MuJoCo's own default is 50 rather
        # than SUGAR's 4. The mapping below reuses the number only because it is the closest
        # thing SUGAR states; it is a convention mismatch, not an equivalence. Measured at 4096
        # envs it costs almost nothing either way (8/4 -> 4,582 env-steps/s, 8/2 -> 4,552,
        # 8/1 -> 4,669), so it is not worth tuning for speed, but do not read it as fidelity
        # to PhysX. The outer count IS a real accuracy knob and a tempting false economy:
        # 4/4 -> 5,264 and 1/4 -> 6,151 env-steps/s, bought by leaving contacts unconverged.
        iterations=int(getattr(props, "solver_position_iteration_count", 8) or 8),
        ls_iterations=int(getattr(props, "solver_velocity_iteration_count", 4) or 4),
        use_mujoco_contacts=False,
    )

    _make_assets(scene, scene_cfg, meta, num_envs, device)
    _make_sensors(scene, scene_cfg, meta, num_envs, device)
    return scene


def _tile_shape_body(meta: dict[str, Any], num_envs: int) -> np.ndarray:
    """Global shape -> global body map, since Newton's arrays are flat over all worlds."""
    per_world = np.asarray(meta["shape_body"], dtype=np.int64)
    bodies = meta["bodies_per_world"]
    return np.concatenate([per_world + env * bodies for env in range(num_envs)])


def _make_assets(scene, scene_cfg, meta, num_envs: int, device: str) -> None:
    robot_names = meta["robot_body_names"]
    robot_local = torch.tensor(
        [meta["body_labels"].index(n) for n in robot_names], dtype=torch.long, device=device
    )
    robot = Articulation(
        scene_cfg.robot,
        scene,
        "robot",
        body_indices=robot_local,
        body_names=robot_names,
        joint_names=meta["joint_names"],
        joint_coords=torch.tensor(meta["act_coords"], dtype=torch.long, device=device),
        joint_dofs=torch.tensor(meta["act_dofs"], dtype=torch.long, device=device),
        root_body_index=meta["root_body"],
        root_joint_coord=meta["root_q0"],
    )
    box = RigidObject(
        getattr(scene_cfg, "obj", None) or scene_cfg.object,
        scene,
        "obj",
        body_index=meta["box_body"],
        joint_coord=meta["box_q0"],
    )
    scene.articulations["robot"] = robot
    scene.rigid_objects["obj"] = box
    # SUGAR's terms reach for both spellings depending on the task family.
    scene.rigid_objects["object"] = box

    _seed_defaults(robot, box, scene_cfg, meta, num_envs, device)
    scene._asset_shapes = {
        "robot": _shapes_of(meta, robot_local, device),
        "obj": _shapes_of(meta, torch.tensor([meta["box_body"]], device=device), device),
        "object": _shapes_of(meta, torch.tensor([meta["box_body"]], device=device), device),
    }
    scene._joint_limits = {
        "robot": torch.stack(
            [
                torch.as_tensor(meta["lo"], device=device),
                torch.as_tensor(meta["hi"], device=device),
            ],
            dim=-1,
        ).repeat(num_envs, 1, 1)
    }


def _shapes_of(meta: dict[str, Any], body_local: torch.Tensor, device: str) -> torch.Tensor:
    """Local shape indices belonging to a set of local body indices."""
    per_world = torch.as_tensor(meta["shape_body"], dtype=torch.long, device=device)
    wanted = torch.isin(per_world, body_local.to(device))
    return wanted.nonzero(as_tuple=False).squeeze(-1)


def _seed_defaults(robot, box, scene_cfg, meta, num_envs: int, device: str) -> None:
    """Populate the `data` fields SUGAR's reset events and rewards read."""
    names = meta["joint_names"]
    init = getattr(scene_cfg.robot, "init_state", None)
    default = _expand(getattr(init, "joint_pos", None) or {}, names, default=0.0)

    d = robot.data
    d.default_joint_pos = torch.as_tensor(default, device=device).repeat(num_envs, 1)
    d.default_joint_vel = torch.zeros(num_envs, len(names), device=device)
    d.joint_stiffness = torch.as_tensor(meta["stiffness"], device=device).repeat(num_envs, 1)
    d.joint_damping = torch.as_tensor(meta["damping"], device=device).repeat(num_envs, 1)
    d.joint_effort_limits = torch.as_tensor(meta["effort"], device=device).repeat(num_envs, 1)
    d.soft_joint_pos_limits = torch.stack(
        [
            torch.as_tensor(meta["soft_lo"], device=device),
            torch.as_tensor(meta["soft_hi"], device=device),
        ],
        dim=-1,
    ).repeat(num_envs, 1, 1)
    d.joint_pos_limits = torch.stack(
        [torch.as_tensor(meta["lo"], device=device), torch.as_tensor(meta["hi"], device=device)],
        dim=-1,
    ).repeat(num_envs, 1, 1)

    for asset, cfg in ((robot, scene_cfg.robot), (box, box.cfg)):
        state = getattr(cfg, "init_state", None)
        pos = torch.tensor(getattr(state, "pos", (0.0, 0.0, 0.0)), device=device)
        rot = torch.tensor(getattr(state, "rot", (1.0, 0.0, 0.0, 0.0)), device=device)
        asset.data.default_root_state = torch.cat(
            [pos, rot, torch.zeros(6, device=device)]
        ).repeat(num_envs, 1)
        # Read back through the view rather than off the model, so the layout is whatever
        # the randomisation term will index -- the two differ between the rigid-body and
        # articulation views.
        asset.data.default_mass = asset.root_physx_view.get_masses().clone()
        asset.data.default_inertia = asset.root_physx_view.get_inertias().clone()


def _make_sensors(scene, scene_cfg, meta, num_envs: int, device: str) -> None:
    """Create a ContactSensor per `ContactSensorCfg`, resolving its prim path to bodies."""
    from .sensors import ContactSensorCfg

    bodies_per = meta["bodies_per_world"]
    offsets = torch.arange(num_envs, device=device).unsqueeze(-1) * bodies_per

    def resolve(expr: str) -> tuple[torch.Tensor, list[str]]:
        """`{ENV_REGEX_NS}/Robot/<body>` -> matching body names. `/Obj` is the box."""
        tail = str(expr).rsplit("/", 1)[-1]
        if tail in ("Obj", "Object"):
            local = [meta["box_body"]]
            return offsets + torch.tensor(local, device=device), ["box"]
        import re

        # The prim path names the asset, so `/Robot/.*` must not reach the box, and must
        # match the same link set Isaac Sim's prim hierarchy would have.
        matched = [n for n in meta["robot_body_names"] if re.fullmatch(tail, n) or n == tail]
        if not matched:
            raise ValueError(f"sugar_swap: contact sensor path {expr!r} matches no body")
        local = [meta["body_labels"].index(n) for n in matched]
        return offsets + torch.tensor(local, device=device), matched

    for name in [k for k in vars(scene_cfg) if not k.startswith("_")]:
        cfg = getattr(scene_cfg, name)
        if not isinstance(cfg, ContactSensorCfg):
            continue
        body_idx, body_names = resolve(cfg.prim_path)
        filters = getattr(cfg, "filter_prim_paths_expr", None)
        filter_idx = resolve(filters[0])[0] if filters else None
        scene.sensors[name] = ContactSensor(
            cfg, scene, name, body_idx, body_names, filter_body_indices=filter_idx
        )


def _substep(scene: InteractiveScene) -> None:
    """One physics substep: collide, solve, swap, read the solved forces back.

    `collide` only fills the contact geometry; the forces stay zero until the solver hands
    its solved set back, which MuJoCo does only in `update_contacts`. Without it every
    contact-force reward and termination reads an all-zero buffer and silently contributes
    nothing. It must follow the swap, since it reads the post-step poses.
    """
    scene.state_0.clear_forces()
    scene.pipeline.collide(scene.state_0, scene.contacts)
    scene.solver.step(
        scene.state_0, scene.state_1, scene.control, scene.contacts, scene.physics_dt
    )
    scene.swap_states()
    scene.solver.update_contacts(scene.contacts, scene.state_0)


def step_physics(scene: InteractiveScene, substeps: int) -> None:
    """Advance Newton by one physics step of `scene.physics_dt`.

    Contacts are regenerated every substep rather than once per policy step, because that
    is what Isaac Sim does: PhysX collides at `1/sim.dt` = 200 Hz. Refreshing only once per
    decision is 3-4x cheaper but leaves normals up to 20 ms stale, which a contact-force
    reward reads as the wrong direction.
    """
    scene.flush_kinematics()
    graphs = getattr(scene, "_physics_graphs", None)
    for _ in range(substeps):
        if graphs is None:
            _substep(scene)
        else:
            # Two graphs, because a substep reads one state buffer and writes the other and
            # a captured graph holds those two pointers fixed. Replaying a single graph
            # would keep writing the buffer the previous replay just made current.
            wp.capture_launch(graphs[scene._graph_parity])
            scene._graph_parity ^= 1
            scene.swap_states()
        scene.invalidate_contacts()


def enable_cuda_graph(scene: InteractiveScene) -> None:
    """Capture the physics substep into CUDA graphs, one per state-buffer parity.

    Two things have to be true first, and both are set here rather than assumed:

    * `mjw_model.opt.graph_conditional` must be off. MuJoCo-Warp's solver loop is a
      `wp.capture_while`, and a conditional graph node needs CUDA 12.4 on both the toolkit
      and the driver. Outside a capture that call degrades to a host-side loop that copies
      the convergence flag back and calls `synchronize_stream` on *every* solver iteration;
      inside a capture on an older driver it raises. Turning it off replaces both with a
      fixed-length unrolled loop, which is what makes the capture possible -- and costs the
      solver's early exit, which this driver could not use anyway.
    * Nothing may allocate during the capture. `SolverMuJoCo` lazily builds its contact
      fast-path buffers on the first step, so two eager substeps run first.
    """
    if scene.model.device.is_cpu:
        raise RuntimeError("sugar_swap: CUDA graph capture needs a CUDA device")
    scene.solver.mjw_model.opt.graph_conditional = False
    for _ in range(2):
        _substep(scene)
        scene.invalidate_contacts()
    wp.synchronize()

    graphs = []
    for _ in range(2):
        with wp.ScopedCapture() as capture:
            _substep(scene)
        graphs.append(capture.graph)
    # `_substep` swaps the state buffers, and the swap is a Python rebinding that happens
    # during capture as well, so two captures leave the layout where it started.
    scene._physics_graphs = graphs
    scene._graph_parity = 0
    scene.invalidate_contacts()


def disable_cuda_graph(scene: InteractiveScene) -> None:
    """Drop the captured graphs and go back to eager launches."""
    scene._physics_graphs = None
    scene._graph_parity = 0
