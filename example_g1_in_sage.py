# SPDX-License-Identifier: Apache-2.0
###########################################################################
# G1 walking in a SAGE-10k scene (Newton, hydroelastic contact)
#
# Drops a Unitree G1 (driven by the shipped IsaacLab RL locomotion policy,
# `newton.examples.robot_policy`) into a room ingested from SAGE-10k
# (`scene_ingest`), with per-part material properties and a hydroelastic
# foot↔floor contact model (SolverMuJoCo + hydroelastic SDF, as in
# example_panda_clock_metal.py).
#
# Headless render → mp4:
#   conda activate newton   # (or the repo venv)
#   python example_g1_in_sage.py \
#       --scene $MY_DATA_HOME/robot_baby_data/_inspect/layout_84b703fb.json \
#       --record g1_in_sage.mp4 --frames 900 --command 1.0,0.0,0.15
#
# Interactive (GL window, keyboard i/j/k/l/u/o, p=reset):
#   python example_g1_in_sage.py --scene <layout.json>
#
# NOTE: reference implementation — first run in the live `newton` env will need
# small fixes (foot link names, SAGE mesh up-axis/scale, hydroelastic stability
# under the loco policy). Env-dependent lines are flagged  # CALIBRATE.
###########################################################################

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace

import numpy as np
import yaml

# ---- material properties (SI) -------------------------------------------------
# Floors / walls: dry indoor surfaces. Objects: from SAGE mass + PBR (per object).
# Robot feet: hard rubber sole → high friction, low restitution, compliant-ish
# hydroelastic pad for a broad, stable contact patch.
FLOOR_MU = 0.9  # tile/wood floor friction
FLOOR_REST = 0.0
WALL_MU = 0.8
RIGID_KH = 1.0e12  # hard hydroelastic stiffness shared by all rigid bodies
FOOT_MU = 1.2  # hard-rubber sole grip
FOOT_REST = 0.0
FOOT_KH = 5.0e8  # compliant foot pad (broad contact patch; << rigid 1e12)
FOOT_LINK_KEYS = ("ankle_roll", "foot", "sole")  # CALIBRATE: G1 foot link substrings
DEFAULT_OBJ_DENSITY = 500.0
# Room-shell / furniture rendering (GLB visual meshes).
FLOOR_VIZ_LIFT = 0.012  # [m] lift the floor visual off the ground plane to avoid coplanar z-fight
MIN_COLLIDER_EXTENT = 0.12  # [m] skip convex-hull colliders for clutter below this max AABB extent

# SDF params for hydroelastic contact (feet + floor + touched objects)
SDF_MAX_RES = 48
SDF_NARROW_BAND = (-0.01, 0.01)


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _ensure_repo_on_path() -> None:
    """Put the repo root on ``sys.path`` so ``scene_ingest`` is importable."""
    if _repo_root() not in sys.path:
        sys.path.insert(0, _repo_root())


def build(args, viewer):
    """Build the model: G1 + policy config + ingested SAGE room + materials.

    Returns a dict with model/solver/state/control/policy tensors — enough for the
    step loop below. Mirrors example_robot_policy.Example.__init__ and the
    hydroelastic material setup of example_panda_clock_metal.py.
    """
    import torch
    import warp as wp

    import newton
    import newton.examples
    import newton.utils
    from newton import JointTargetMode
    from newton.examples.robot.example_robot_policy import (
        ROBOT_CONFIGS,
        load_policy_and_setup_tensors,
    )

    _ensure_repo_on_path()

    # ---- robot config + policy (shipped G1 loco policy) ----
    rc = ROBOT_CONFIGS[args.robot]
    asset_dir = str(newton.utils.download_asset(rc.asset_dir))
    with open(f"{asset_dir}/{rc.yaml_path}", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    policy_path = f"{asset_dir}/{rc.policy_path['mjw']}"
    mjc_to_physx = list(range(config["num_dofs"]))
    physx_to_mjc = list(range(config["num_dofs"]))

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.1, limit_ke=1.0e2, limit_kd=1.0e0)
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    # G1 spawn: a clear floor spot in the SAGE room (metres). CALIBRATE per scene.
    spawn = wp.vec3(float(args.spawn[0]), float(args.spawn[1]), 0.8)
    builder.add_usd(
        newton.examples.get_asset(asset_dir + "/" + rc.asset_path),
        xform=wp.transform(spawn),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        joint_ordering="dfs",
        hide_collision_shapes=True,
    )
    builder.approximate_meshes("convex_hull")
    n_robot_shapes = builder.shape_count

    # init pose + joint gains from the policy config
    builder.joint_q[:3] = [spawn[0], spawn[1], 0.76]
    builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
    builder.joint_q[7:] = config["mjw_joint_pos"]
    for i in range(len(config["mjw_joint_stiffness"])):
        builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i]
        builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i]
        builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
        builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

    # ---- feet: hard-rubber sole + hydroelastic pad ----
    mu_ovr = getattr(args, "mu", None)  # global friction override (slippery/rough versions)
    foot_shapes = []
    for s in range(n_robot_shapes):
        body = builder.shape_body[s]
        label = builder.body_label[body] if 0 <= body < len(builder.body_label) else ""
        if any(k in label.lower() for k in FOOT_LINK_KEYS):
            foot_shapes.append(s)
            builder.shape_material_mu[s] = mu_ovr if mu_ovr is not None else FOOT_MU
            builder.shape_material_restitution[s] = FOOT_REST
            # hydroelastic pad: build an SDF for the foot mesh + set the flag
            src = builder.shape_source[s]
            if src is not None and getattr(src, "sdf", None) is None and builder.shape_type[s] == newton.GeoType.MESH:
                src.build_sdf(max_resolution=SDF_MAX_RES, narrow_band_range=SDF_NARROW_BAND, margin=0.01)
            builder.shape_flags[s] |= newton.ShapeFlags.HYDROELASTIC
    print(f"[g1_in_sage] tagged {len(foot_shapes)} foot shapes hydroelastic (mu={FOOT_MU})")  # CALIBRATE if 0

    # ---- floor (hydroelastic ground) ----
    floor_mu = mu_ovr if mu_ovr is not None else FLOOR_MU
    # collision-only ground plane; its default grey visual would z-fight the GLB floor mesh (both at
    # z=0) and hide the authentic olive texture, so hide it and let the GLB floor be the visual.
    floor_cfg = replace(
        builder.default_shape_cfg,
        mu=floor_mu,
        restitution=FLOOR_REST,
        kh=RIGID_KH,
        is_hydroelastic=True,
        is_visible=False,
    )
    builder.add_ground_plane(cfg=floor_cfg)
    print(f"[g1_in_sage] floor mu={floor_mu}")

    # ---- SAGE room + furniture from the pre-assembled GLB (authoritative baked UVs + textures) ----
    # The dataset ships an assembled GLB (_out/layout_<id>.glb) that its own reference previews are
    # rendered from, so its per-vertex UVs and baked textures are correct. Load those directly instead
    # of hand-parsing PLY texcoords (which mismap) or building floor/wall quads. Meshes are world-placed
    # (glTF Y-up rotated to Newton Z-up in the loader) so the floor sits on the z=0 ground plane.
    # Furniture is static — it matches the reference layout and avoids tumbling; big pieces also get a
    # convex-hull collider so the robot collides with / touches them (tactile).
    import glob

    from scene_ingest.newton_build import load_glb_scene

    scene_dir = os.path.dirname(os.path.abspath(args.scene))
    stem = os.path.splitext(os.path.basename(args.scene))[0]
    glb_path = os.path.join(scene_dir, "_out", stem + ".glb")
    if not os.path.exists(glb_path):
        glb_path = next(iter(glob.glob(os.path.join(scene_dir, "_out", "*.glb"))), None)
    meshes = load_glb_scene(glb_path)
    print(f"[g1_in_sage] loaded {len(meshes)} meshes from {os.path.basename(glb_path)}")

    # drop the ceiling + the wall nearest the camera so we can see into the room
    cpx, cpy = float(args.cam[0]), float(args.cam[1])
    walls = [m for m in meshes if m["category"] == "wall"]
    near_wall = (
        min(
            walls, key=lambda m: math.hypot(float(m["verts"][:, 0].mean()) - cpx, float(m["verts"][:, 1].mean()) - cpy)
        )["name"]
        if walls and getattr(args, "walls", True)
        else None
    )

    viz_cfg = replace(builder.default_shape_cfg, has_shape_collision=False, is_hydroelastic=False, density=0.0)
    obj_shapes: list[int] = []
    n_viz = n_col = 0
    for m in meshes:
        if m["category"] == "ceiling" or (
            m["category"] == "wall" and (not getattr(args, "walls", True) or m["name"] == near_wall)
        ):
            continue
        # visible mesh: authentic baked texture + UVs. White shape color so the shader shows the
        # texture unmodified (albedo = ObjectColor * texture; a palette color would tint it).
        verts = m["verts"]
        if m["category"] == "floor":
            verts = verts + np.array([0.0, 0.0, FLOOR_VIZ_LIFT], dtype=np.float32)  # avoid coplanar z-fight
        if m["uvs"] is not None and m["texture"] is not None:
            viz = newton.Mesh(verts, m["faces"], uvs=m["uvs"], texture=m["texture"], compute_inertia=False)
        else:
            viz = newton.Mesh(verts, m["faces"], compute_inertia=False)
        builder.add_shape_mesh(body=-1, mesh=viz, cfg=viz_cfg, color=(1.0, 1.0, 1.0))
        n_viz += 1
        # static convex-hull collider for furniture big enough for the robot to bump (skip tiny clutter).
        # add_shape_convex_hull hulls the mesh itself into an efficient CONVEX_MESH shape.
        if m["category"] == "furniture" and float(np.max(m["verts"].max(0) - m["verts"].min(0))) >= MIN_COLLIDER_EXTENT:
            obj_mu = mu_ovr if mu_ovr is not None else 0.8
            col_cfg = replace(
                builder.default_shape_cfg,
                is_visible=False,
                has_shape_collision=True,
                is_hydroelastic=False,
                mu=obj_mu,
                restitution=0.0,
                kh=RIGID_KH,
            )
            try:
                hull = newton.Mesh(m["verts"], m["faces"], compute_inertia=False)
                obj_shapes.append(builder.add_shape_convex_hull(body=-1, mesh=hull, cfg=col_cfg))
                n_col += 1
            except Exception:
                pass
    print(
        f"[g1_in_sage] room shell + furniture: {n_viz} visual meshes, {n_col} static colliders "
        f"(dropped ceiling + near wall '{near_wall}')"
    )

    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    # budgets scaled for ~40 dynamic bodies resting/colliding, not just the lone robot
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, solver="newton", nconmax=1024, njmax=2048)

    ex = argparse.Namespace()
    ex.model, ex.solver = model, solver
    ex.state_0, ex.state_1 = model.state(), model.state()
    ex.control = model.control()

    # tactile: net contact force on the robot from furniture (excludes the floor). Must be
    # created before the Contacts buffer so the "force" contact attribute is allocated.
    ex.contact_sensor = None
    try:
        from newton.sensors import SensorContact

        if obj_shapes:
            ex.contact_sensor = SensorContact(
                model,
                sensing_shapes=list(range(n_robot_shapes)),
                counterpart_shapes=obj_shapes,
                measure_total=True,
            )
            print(f"[g1_in_sage] tactile SensorContact: {n_robot_shapes} robot shapes vs {len(obj_shapes)} furniture")
    except Exception as e:  # sensor API mismatch must not kill the render
        print(f"[g1_in_sage] tactile sensor unavailable: {e}")

    ex.contacts = newton.Contacts(solver.get_max_contact_count(), 0)
    ex.torch_device = "cuda" if wp.get_device().is_cuda else "cpu"
    newton.eval_fk(model, ex.state_0.joint_q, ex.state_0.joint_qd, ex.state_0)
    ex.config = config
    ex.n_dof = config["num_dofs"]  # robot actuated joints (policy slice width)
    ex.n_ctrl = ex.control.joint_target_q.shape[0]  # full target vector incl. furniture free joints
    ex.physx_to_mjc_indices = torch.tensor(physx_to_mjc, device=ex.torch_device, dtype=torch.long)
    ex.mjc_to_physx_indices = torch.tensor(mjc_to_physx, device=ex.torch_device, dtype=torch.long)
    ex.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=ex.torch_device).unsqueeze(0)
    ex.command = torch.tensor([[float(c) for c in args.command]], device=ex.torch_device, dtype=torch.float32)
    ex.decimation = 4
    ex.frame_dt = 1.0 / 200.0
    ex.sim_dt = ex.frame_dt
    # policy joint_pos_initial: ONLY the robot's joints (furniture free-joint coords follow them)
    load_policy_and_setup_tensors(ex, policy_path, config["num_dofs"], slice(7, 7 + config["num_dofs"]))
    viewer.set_model(model)
    return ex


def _compute_obs_robot(ex):
    """Loco-policy observation, sliced to the robot's DOFs only.

    Mirrors ``example_robot_policy.compute_obs`` but the scene now also holds dynamic furniture
    whose free-joint coords trail the robot's in ``joint_q``/``joint_qd`` — so the joint slices
    are ``[7 : 7+n_dof]`` (positions) and ``[6 : 6+n_dof]`` (velocities), not open-ended.
    """
    import torch

    from newton.examples.robot.example_robot_policy import quat_rotate_inverse

    n = ex.n_dof
    dev = ex.torch_device
    jq, jqd = ex.state_0.joint_q, ex.state_0.joint_qd
    root_quat_w = torch.tensor(jq[3:7], device=dev, dtype=torch.float32).unsqueeze(0)
    root_lin_vel_w = torch.tensor(jqd[:3], device=dev, dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = torch.tensor(jqd[3:6], device=dev, dtype=torch.float32).unsqueeze(0)
    joint_pos_current = torch.tensor(jq[7 : 7 + n], device=dev, dtype=torch.float32).unsqueeze(0)
    joint_vel_current = torch.tensor(jqd[6 : 6 + n], device=dev, dtype=torch.float32).unsqueeze(0)

    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, ex.gravity_vec)
    joint_pos_rel = joint_pos_current - ex.joint_pos_initial
    rearr_pos = torch.index_select(joint_pos_rel, 1, ex.physx_to_mjc_indices)
    rearr_vel = torch.index_select(joint_vel_current, 1, ex.physx_to_mjc_indices)
    return torch.cat([vel_b, a_vel_b, grav, ex.command, rearr_pos, rearr_vel, ex.act], dim=1)


def policy_step(ex):
    """One control frame: policy → joint targets → `decimation` sim substeps."""
    import torch
    import warp as wp

    obs = _compute_obs_robot(ex)
    with torch.no_grad():
        ex.act = ex.policy(obs)
        ra = torch.index_select(ex.act, 1, ex.mjc_to_physx_indices)
        a = ex.joint_pos_initial + ex.config["action_scale"] * ra
        # write ONLY the robot slice; furniture free joints are unactuated (targets ignored)
        a_full = torch.zeros(ex.n_ctrl, device=ex.torch_device, dtype=torch.float32)
        a_full[6 : 6 + ex.n_dof] = a.squeeze(0)
        wp.copy(ex.control.joint_target_q, wp.from_torch(a_full, dtype=wp.float32, requires_grad=False))
    for _ in range(ex.decimation):
        ex.state_0.clear_forces()
        ex.solver.step(ex.state_0, ex.state_1, ex.control, ex.contacts, ex.sim_dt)
        ex.state_0, ex.state_1 = ex.state_1, ex.state_0
    ex.solver.update_contacts(ex.contacts, ex.state_0)
    if ex.contact_sensor is not None:
        try:
            ex.contact_sensor.update(ex.state_0, ex.contacts)
        except Exception:
            ex.contact_sensor = None


def _make_parser():
    import newton.examples

    p = newton.examples.create_parser()
    p.add_argument("--scene", required=True, help="Path to an extracted SAGE layout_<id>.json")
    p.add_argument("--robot", default="g1_29dof", choices=["g1_29dof", "g1_23dof"])
    p.add_argument(
        "--spawn",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[2.5, 3.5],
        help="G1 spawn x,y [m] (a clear floor spot)",
    )
    p.add_argument(
        "--command",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[1.0, 0.0, 0.0],
        help="velocity command fwd,lat,rot",
    )
    p.add_argument(
        "--cam",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[4.5, 1.0, 1.7],
        help="camera position x,y,z [m] (close, eye-level)",
    )
    p.add_argument("--pitch", type=float, default=-13.0, help="camera pitch (deg)")
    p.add_argument(
        "--mu", type=float, default=None, help="global friction override for feet+floor (slippery ~0.05, rough ~1.4)"
    )
    p.add_argument("--no-walls", dest="walls", action="store_false", help="omit the room walls")
    p.set_defaults(walls=True)
    p.add_argument(
        "--settle",
        type=int,
        default=30,
        help="pre-roll control frames (command=0) to let furniture settle before recording",
    )
    p.add_argument("--record", default=None, help="output mp4 (headless GL); omit for interactive")
    p.add_argument("--frames", type=int, default=300)
    return p


def main():
    # Render backend: EGL headless (context.md §6) OR a windowed GL context on an X
    # display (Xvfb) when G1_XVFB=1 — needed in compute-only containers with no EGL device.
    xvfb = os.environ.get("G1_XVFB") == "1"
    if "--record" in sys.argv and not xvfb:
        import pyglet

        pyglet.options["headless"] = True

    import newton.examples

    parser = _make_parser()
    if "--record" in sys.argv:
        parser.set_defaults(viewer="gl", headless=(not xvfb))
    viewer, args = newton.examples.init(parser)

    ex = build(args, viewer)

    # frame the room: look from --cam toward the G1 spawn
    if hasattr(viewer, "set_camera"):
        import math

        import warp as wp

        cx, cy = float(args.spawn[0]), float(args.spawn[1])
        px, py, pz = args.cam
        yaw = math.degrees(math.atan2(cy - py, cx - px))
        viewer.set_camera(pos=wp.vec3(px, py, pz), pitch=args.pitch, yaw=yaw)

    # settle: let the dynamic furniture drop to rest (robot stands still) before recording,
    # so the video opens on a stable room rather than objects visibly dropping.
    if getattr(args, "settle", 0) > 0:
        import torch

        saved_cmd = ex.command.clone()
        ex.command = torch.zeros_like(ex.command)
        for _ in range(args.settle):
            policy_step(ex)
        ex.command = saved_cmd
        print(f"[g1_in_sage] settled {args.settle} frames", flush=True)

    if args.record:
        import subprocess

        from PIL import Image

        frames_dir = args.record + ".frames"
        os.makedirs(frames_dir, exist_ok=True)
        peak_touch = 0.0  # peak robot↔furniture contact force [N] over the clip (tactile signal)
        for f in range(args.frames):
            policy_step(ex)
            if ex.contact_sensor is not None and getattr(ex.contact_sensor, "total_force", None) is not None:
                try:
                    import numpy as _np

                    fmag = float(_np.linalg.norm(ex.contact_sensor.total_force.numpy(), axis=-1).max())
                    if fmag > peak_touch:
                        peak_touch = fmag
                except Exception:
                    pass
            viewer.begin_frame(f * ex.frame_dt)
            viewer.log_state(ex.state_0)
            try:
                viewer.log_contacts(ex.contacts, ex.state_0)
            except Exception:
                pass
            viewer.end_frame()
            img = viewer.get_frame().numpy()  # (H,W,3) uint8, top-left origin
            Image.fromarray(img).save(f"{frames_dir}/f{f:05d}.png")
            if f % 50 == 0:
                print(f"[g1_in_sage] frame {f}/{args.frames}  peak_touch={peak_touch:.1f}N", flush=True)
        print(f"[g1_in_sage] peak robot↔furniture contact force = {peak_touch:.1f} N", flush=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "50",
                "-i",
                f"{frames_dir}/f%05d.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                args.record,
            ],
            check=True,
        )
        print(f"[g1_in_sage] wrote {os.path.abspath(args.record)}", flush=True)
    else:
        while getattr(viewer, "is_running", lambda: True)():
            policy_step(ex)
            viewer.begin_frame(ex.frame_dt)
            viewer.log_state(ex.state_0)
            viewer.end_frame()


if __name__ == "__main__":
    main()
