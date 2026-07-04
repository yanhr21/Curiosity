#!/usr/bin/env python3
"""Non-tensor USD/PhysX dynamic quadruped payload-carry scene.

This route avoids IsaacLab Articulation/RigidObject tensor APIs, which are
currently failing in this environment.  The robot is built directly from USD
rigid bodies, revolute joints, fixed joints, and USD Physics drive targets.

First milestone: prove that a dynamically simulated robot body with actuated
legs can continue walking while carrying a physical box payload fixed to its
torso.  This is not yet unknown-object grasping or learned control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USD/PhysX dynamic quadruped fixed-payload carry smoke.")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--payload-mass", type=float, default=4.0)
    parser.add_argument("--target-x", type=float, default=1.0)
    parser.add_argument("--gait-frequency", type=float, default=1.2)
    parser.add_argument("--hip-amplitude-deg", type=float, default=18.0)
    parser.add_argument("--knee-amplitude-deg", type=float, default=16.0)
    parser.add_argument(
        "--articulation-root",
        action="store_true",
        help=(
            "Diagnostic only. Reduced-coordinate articulations cannot receive "
            "changing drive targets when the PhysX direct GPU API is enabled."
        ),
    )
    parser.add_argument("--root-assist", type=float, default=0.0, help="Reserved; must remain 0 for unassisted evidence.")
    parser.add_argument(
        "--control-mode",
        choices=("usd_drive_attr", "core_articulation"),
        default="usd_drive_attr",
        help="Runtime joint control path. core_articulation uses Isaac Sim SingleArticulation.apply_action().",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/usd_dynamic_quadruped_carry_scene"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import build_simulation_context  # noqa: E402


TORSO_PATH = "/World/Robot/Torso"
BOX_PATH = "/World/CarryBox"
TARGET_XY = (0.0, 0.0)
LEG_PHASES = {
    "fl": 0.0,
    "fr": math.pi,
    "rl": math.pi,
    "rr": 0.0,
}


def _pose_wxyz(stage: Usd.Stage, prim_path: str) -> list[float]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"USD prim not found: {prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat()
    imag = quat.GetImaginary()
    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(quat.GetReal()),
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
    ]


def _quat_to_roll_pitch(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _spawn_body(
    path: str,
    size: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    translation: tuple[float, float, float],
    friction: float = 1.1,
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.04,
            angular_damping=0.08,
            max_linear_velocity=6.0,
            max_angular_velocity=8.0,
            max_depenetration_velocity=2.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=0.8 * friction),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
    )
    cfg.func(path, cfg, translation=translation)


def _fixed_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
    collision_enabled: bool = False,
) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos1]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(bool(collision_enabled))


def _revolute_drive_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
    *,
    axis: str = "Y",
    lower_deg: float = -45.0,
    upper_deg: float = 45.0,
    stiffness: float = 1200.0,
    damping: float = 85.0,
    max_force: float = 900.0,
) -> UsdPhysics.DriveAPI:
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos1]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLowerLimitAttr().Set(float(lower_deg))
    joint.CreateUpperLimitAttr().Set(float(upper_deg))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(stiffness))
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)
    return drive


def design_scene(stage: Usd.Stage) -> dict[str, UsdPhysics.DriveAPI]:
    floor_cfg = sim_utils.CuboidCfg(
        size=(5.0, 2.6, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.1),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.33, 0.33), roughness=0.9),
    )
    floor_cfg.func("/World/Ground", floor_cfg, translation=(0.0, 0.0, -0.025))
    sim_utils.DomeLightCfg(intensity=2100.0, color=(0.82, 0.82, 0.82)).func("/World/DomeLight", sim_utils.DomeLightCfg())
    sim_utils.DistantLightCfg(intensity=2400.0, color=(0.9, 0.86, 0.78)).func(
        "/World/KeyLight", sim_utils.DistantLightCfg(), translation=(2.4, -2.0, 4.0)
    )
    target_cfg = sim_utils.CuboidCfg(
        size=(0.42, 0.36, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.40, 0.85), opacity=0.38),
    )
    target_cfg.func("/World/CarryTarget", target_cfg, translation=(args_cli.target_x, 0.0, 0.01))

    torso_size = (0.44, 0.24, 0.16)
    thigh_len = 0.23
    shin_len = 0.26
    hip_z = 0.50
    torso_z = 0.58
    knee_z = hip_z - thigh_len
    foot_z = 0.045
    foot_center_z = foot_z
    x_front = 0.18
    x_rear = -0.18
    y_left = 0.16
    y_right = -0.16

    _spawn_body(TORSO_PATH, torso_size, 16.0, (0.14, 0.20, 0.30), (0.0, 0.0, torso_z), friction=1.0)
    if args_cli.articulation_root or args_cli.control_mode == "core_articulation":
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(TORSO_PATH))

    _spawn_body(BOX_PATH, (0.30, 0.22, 0.22), args_cli.payload_mass, (0.56, 0.42, 0.23), (0.26, 0.0, torso_z + 0.03))
    _fixed_joint(
        stage,
        "/World/Robot/Torso/FixedPayloadJoint",
        TORSO_PATH,
        BOX_PATH,
        (0.26, 0.0, 0.03),
        (0.0, 0.0, 0.0),
        collision_enabled=False,
    )

    drives: dict[str, UsdPhysics.DriveAPI] = {}
    leg_specs = {
        "fl": (x_front, y_left, LEG_PHASES["fl"]),
        "fr": (x_front, y_right, LEG_PHASES["fr"]),
        "rl": (x_rear, y_left, LEG_PHASES["rl"]),
        "rr": (x_rear, y_right, LEG_PHASES["rr"]),
    }
    for name, (hip_x, hip_y, phase) in leg_specs.items():
        thigh = f"/World/Robot/{name}_thigh"
        shin = f"/World/Robot/{name}_shin"
        foot = f"/World/Robot/{name}_foot"
        _spawn_body(thigh, (0.065, 0.055, thigh_len), 0.95, (0.10, 0.17, 0.26), (hip_x, hip_y, hip_z - thigh_len / 2.0))
        _spawn_body(shin, (0.055, 0.050, shin_len), 0.75, (0.10, 0.17, 0.26), (hip_x, hip_y, knee_z - shin_len / 2.0))
        _spawn_body(foot, (0.18, 0.075, 0.045), 0.35, (0.06, 0.08, 0.09), (hip_x + 0.03, hip_y, foot_center_z), friction=1.8)
        drives[f"{name}_hip"] = _revolute_drive_joint(
            stage,
            f"/World/Robot/{name}_hip_joint",
            TORSO_PATH,
            thigh,
            (hip_x, hip_y, hip_z - torso_z),
            (0.0, 0.0, thigh_len / 2.0),
            lower_deg=-35.0,
            upper_deg=35.0,
            stiffness=1400.0,
            damping=95.0,
            max_force=900.0,
        )
        drives[f"{name}_knee"] = _revolute_drive_joint(
            stage,
            f"/World/Robot/{name}_knee_joint",
            thigh,
            shin,
            (0.0, 0.0, -thigh_len / 2.0),
            (0.0, 0.0, shin_len / 2.0),
            lower_deg=-58.0,
            upper_deg=18.0,
            stiffness=1100.0,
            damping=80.0,
            max_force=700.0,
        )
        _fixed_joint(
            stage,
            f"/World/Robot/{name}_ankle_fixed_joint",
            shin,
            foot,
            (0.0, 0.0, -shin_len / 2.0),
            (-0.03, 0.0, 0.02),
            collision_enabled=False,
        )
        drives[f"{name}_phase"] = phase  # type: ignore[assignment]
    return drives


def _joint_targets(t: float, *, radians: bool) -> dict[str, float]:
    freq = float(args_cli.gait_frequency)
    targets = {}
    for leg in ("fl", "fr", "rl", "rr"):
        phase = LEG_PHASES[leg]
        s = math.sin(2.0 * math.pi * freq * t + phase)
        c = math.cos(2.0 * math.pi * freq * t + phase)
        hip_deg = float(args_cli.hip_amplitude_deg) * s - 5.0
        knee_deg = -18.0 - float(args_cli.knee_amplitude_deg) * max(0.0, c)
        targets[f"{leg}_hip"] = math.radians(hip_deg) if radians else hip_deg
        targets[f"{leg}_knee"] = math.radians(knee_deg) if radians else knee_deg
    return targets


def _set_drive_targets(drives: dict[str, UsdPhysics.DriveAPI], t: float) -> None:
    targets = _joint_targets(t, radians=False)
    for name, value in targets.items():
        drives[name].GetTargetPositionAttr().Set(value)


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "usd_dynamic_quadruped_carry_state.csv"
    summary_path = args_cli.output_dir / "usd_dynamic_quadruped_carry_summary.json"
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)

    with build_simulation_context(create_new_stage=False, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        sim.set_setting("/physics/updateToUsd", True)
        sim.set_setting("/physics/updateVelocitiesToUsd", True)
        drives = design_scene(sim.stage)
        sim.set_camera_view(eye=[2.1, -1.6, 1.05], target=[0.45, 0.0, 0.38])
        sim.reset()
        dt = sim.get_physics_dt()
        initial_torso = None
        initial_box = None
        summary = {
            "scene_type": "usd_physx_dynamic_quadruped_fixed_payload_carry",
            "success_claim": "dynamic_robot_fixed_payload_diagnostic_not_unknown_box_grasp_or_learned_policy",
            "uses_isaaclab_tensor_api": False,
            "articulation_root_enabled": bool(args_cli.articulation_root),
            "control_mode": args_cli.control_mode,
            "payload_mode": "fixed_joint_to_torso",
            "root_assist": float(args_cli.root_assist),
            "payload_mass_kg": float(args_cli.payload_mass),
            "steps_requested": int(args_cli.steps),
            "completed_steps": 0,
            "fall_events": 0,
            "box_drop_events": 0,
            "max_torso_travel_xy_m": 0.0,
            "max_box_travel_xy_m": 0.0,
            "final_box_target_distance_xy_m": None,
            "min_torso_z_m": None,
            "max_tilt_rad": 0.0,
        }
        core_robot = None
        core_joint_indices: dict[str, int] = {}
        core_action_type = None
        core_np = None
        if args_cli.control_mode == "core_articulation":
            from isaacsim.core.prims import SingleArticulation
            from isaacsim.core.utils.types import ArticulationAction

            core_robot = SingleArticulation(prim_path=TORSO_PATH, name="usd_dynamic_quad")
            core_robot.initialize()
            print(f"[INFO] Core articulation DOFs: {core_robot.num_dof} {core_robot.dof_names}", flush=True)
            summary["core_articulation_dof_names"] = list(core_robot.dof_names)
            for wanted in (f"{leg}_{joint}" for leg in ("fl", "fr", "rl", "rr") for joint in ("hip", "knee")):
                for idx, dof_name in enumerate(core_robot.dof_names):
                    if wanted in dof_name:
                        core_joint_indices[wanted] = idx
                        break
            missing = [name for name in (f"{leg}_{joint}" for leg in ("fl", "fr", "rl", "rr") for joint in ("hip", "knee")) if name not in core_joint_indices]
            if missing:
                raise RuntimeError(f"Core articulation missing expected joints: {missing}; dofs={core_robot.dof_names}")
            summary["core_joint_indices"] = dict(core_joint_indices)
            core_action_type = ArticulationAction
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "time_s",
                    "torso_x",
                    "torso_y",
                    "torso_z",
                    "box_x",
                    "box_y",
                    "box_z",
                    "roll",
                    "pitch",
                    "tilt",
                    "torso_travel_xy_m",
                    "box_travel_xy_m",
                    "box_target_distance_xy_m",
                    "fall",
                    "box_drop",
                ]
            )
            for step in range(args_cli.steps):
                t = step * dt
                if args_cli.control_mode == "core_articulation":
                    targets = _joint_targets(t, radians=True)
                    joint_positions = [None] * core_robot.num_dof
                    for name, value in targets.items():
                        joint_positions[core_joint_indices[name]] = value
                    core_robot.apply_action(core_action_type(joint_positions=joint_positions))
                else:
                    _set_drive_targets(drives, t)
                sim.step(render=args_cli.render)
                if step % 10 == 0 or step == args_cli.steps - 1:
                    torso = _pose_wxyz(sim.stage, TORSO_PATH)
                    box = _pose_wxyz(sim.stage, BOX_PATH)
                    if initial_torso is None:
                        initial_torso = list(torso)
                    if initial_box is None:
                        initial_box = list(box)
                    roll, pitch = _quat_to_roll_pitch(torso[3], torso[4], torso[5], torso[6])
                    tilt = math.hypot(roll, pitch)
                    torso_travel = math.hypot(torso[0] - initial_torso[0], torso[1] - initial_torso[1])
                    box_travel = math.hypot(box[0] - initial_box[0], box[1] - initial_box[1])
                    target_distance = math.hypot(box[0] - args_cli.target_x, box[1])
                    fall = int(torso[2] < 0.35 or tilt > 0.80)
                    box_drop = int(box[2] < 0.20)
                    summary["completed_steps"] = int(step + 1)
                    summary["fall_events"] += fall
                    summary["box_drop_events"] += box_drop
                    summary["max_torso_travel_xy_m"] = max(float(summary["max_torso_travel_xy_m"]), float(torso_travel))
                    summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), float(box_travel))
                    summary["final_box_target_distance_xy_m"] = float(target_distance)
                    summary["min_torso_z_m"] = (
                        float(torso[2]) if summary["min_torso_z_m"] is None else min(float(summary["min_torso_z_m"]), torso[2])
                    )
                    summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), float(tilt))
                    writer.writerow(
                        [
                            step,
                            t,
                            torso[0],
                            torso[1],
                            torso[2],
                            box[0],
                            box[1],
                            box[2],
                            roll,
                            pitch,
                            tilt,
                            torso_travel,
                            box_travel,
                            target_distance,
                            fall,
                            box_drop,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} torso=({torso[0]:.3f},{torso[1]:.3f},{torso[2]:.3f}) "
                        f"box=({box[0]:.3f},{box[1]:.3f},{box[2]:.3f}) "
                        f"travel={torso_travel:.3f} tilt={tilt:.3f} fall={fall} drop={box_drop}",
                        flush=True,
                    )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    print(f"[INFO] Metrics written to: {csv_path}")
    return csv_path


def main() -> None:
    if args_cli.root_assist != 0.0:
        raise RuntimeError("root-assist must remain 0.0 for this non-tensor dynamic carry diagnostic.")
    run_scene()


if __name__ == "__main__":
    main()
    simulation_app.close()
