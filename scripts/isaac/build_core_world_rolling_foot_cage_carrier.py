#!/usr/bin/env python3
"""Rolling-foot cage carrier diagnostic in Isaac Core World.

This is an actuator-driven ground-contact scaffold for the box-carrying scene.
It is not walking, not humanoid locomotion, and not a learned policy. The goal
is narrower: verify that a no-root articulated carrier with wheel joints can
move a physical cage while a free dynamic box remains retained.
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
    parser = argparse.ArgumentParser(description="Rolling-foot cage carrier diagnostic.")
    parser.add_argument("--steps", type=int, default=760)
    parser.add_argument("--settle-steps", type=int, default=180)
    parser.add_argument("--drive-steps", type=int, default=420)
    parser.add_argument("--wheel-control-mode", choices=("velocity", "effort"), default="velocity")
    parser.add_argument("--wheel-velocity", type=float, default=1.2)
    parser.add_argument("--wheel-effort", type=float, default=200.0)
    parser.add_argument("--wheel-radius", type=float, default=0.085)
    parser.add_argument("--wheel-width", type=float, default=0.060)
    parser.add_argument("--wheel-mass", type=float, default=1.6)
    parser.add_argument("--wheel-damping", type=float, default=350.0)
    parser.add_argument("--wheel-max-force", type=float, default=1200.0)
    parser.add_argument("--target-x", type=float, default=0.30)
    parser.add_argument("--torso-mass", type=float, default=38.0)
    parser.add_argument("--torso-z", type=float, default=0.30)
    parser.add_argument("--torso-size", type=float, nargs=3, default=(0.70, 0.50, 0.16))
    parser.add_argument("--stance-half-length", type=float, default=0.30)
    parser.add_argument("--stance-half-width", type=float, default=0.22)
    parser.add_argument("--payload-mass", type=float, default=1.0)
    parser.add_argument("--payload-size", type=float, nargs=3, default=(0.34, 0.24, 0.24))
    parser.add_argument("--cage-local-x", type=float, default=0.04)
    parser.add_argument("--cage-local-z", type=float, default=0.17)
    parser.add_argument("--cage-size", type=float, nargs=3, default=(0.72, 0.56, 0.04))
    parser.add_argument("--cage-wall-height", type=float, default=0.30)
    parser.add_argument("--cage-wall-thickness", type=float, default=0.055)
    parser.add_argument("--cage-lid-clearance", type=float, default=0.015)
    parser.add_argument("--cage-part-mass", type=float, default=0.8)
    parser.add_argument("--static-friction", type=float, default=2.0)
    parser.add_argument("--dynamic-friction", type=float, default=1.8)
    parser.add_argument("--wheel-static-friction", type=float, default=3.0)
    parser.add_argument("--wheel-dynamic-friction", type=float, default=2.5)
    parser.add_argument("--fall-z", type=float, default=0.18)
    parser.add_argument("--drop-z", type=float, default=0.20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/core_world_rolling_foot_cage_carrier"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PROGRESS] AppLauncher started", flush=True)

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


TORSO_PATH = "/World/Robot/Torso"
BOX_PATH = "/World/CarryBox"
WHEEL_NAMES = ("fl", "fr", "rl", "rr")


def _patch_core_api_simulation_manager_compat() -> None:
    if not hasattr(SimulationManager, "_backend"):
        SimulationManager._backend = "numpy"
    if not hasattr(SimulationManager, "get_backend"):
        SimulationManager.get_backend = classmethod(lambda cls: getattr(cls, "_backend", "numpy"))
    if not hasattr(SimulationManager, "_get_backend_utils"):
        def _get_backend_utils(cls):
            import isaacsim.core.utils.numpy as np_utils

            return np_utils

        SimulationManager._get_backend_utils = classmethod(_get_backend_utils)
    if not hasattr(SimulationManager, "get_physics_sim_device"):
        SimulationManager.get_physics_sim_device = classmethod(lambda cls: args_cli.device)
    if not hasattr(SimulationManager, "get_physics_dt"):
        SimulationManager.get_physics_dt = classmethod(lambda cls: 0.005)


def _set_xform(prim: Usd.Prim, translation: tuple[float, float, float], scale: tuple[float, float, float]) -> None:
    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(*[float(v) for v in translation]))
    xform_api.SetScale(Gf.Vec3f(*[float(v) for v in scale]))


def _define_physics_material(stage: Usd.Stage, path: str, static_friction: float, dynamic_friction: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(float(static_friction))
    physics_material.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    physics_material.CreateRestitutionAttr().Set(0.0)
    return material


def _bind_physics_material(prim: Usd.Prim, material: UsdShade.Material | None) -> None:
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _spawn_box_body(
    stage: Usd.Stage,
    path: str,
    size: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    translation: tuple[float, float, float],
    *,
    physics_material: UsdShade.Material | None = None,
    rigid: bool = True,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translation, size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_physics_material(cube.GetPrim(), physics_material)
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(float(mass))


def _spawn_wheel(
    stage: Usd.Stage,
    path: str,
    radius: float,
    width: float,
    mass: float,
    translation: tuple[float, float, float],
    *,
    physics_material: UsdShade.Material | None = None,
) -> None:
    wheel = UsdGeom.Cylinder.Define(stage, path)
    wheel.CreateRadiusAttr(float(radius))
    wheel.CreateHeightAttr(float(width))
    wheel.CreateAxisAttr("Y")
    _set_xform(wheel.GetPrim(), translation, (1.0, 1.0, 1.0))
    wheel.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.04, 0.05)])
    UsdPhysics.CollisionAPI.Apply(wheel.GetPrim())
    _bind_physics_material(wheel.GetPrim(), physics_material)
    UsdPhysics.RigidBodyAPI.Apply(wheel.GetPrim())
    UsdPhysics.MassAPI.Apply(wheel.GetPrim()).CreateMassAttr(float(mass))


def _fixed_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos1]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)


def _wheel_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    *,
    damping: float,
    max_force: float,
) -> None:
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateAxisAttr().Set("Y")
    joint.CreateLowerLimitAttr().Set(-1.0e9)
    joint.CreateUpperLimitAttr().Set(1.0e9)
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)


def _pose_wxyz(prim: SingleArticulation | SingleRigidPrim) -> list[float]:
    pos, quat = prim.get_world_pose()
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def _quat_to_roll_pitch(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _wheel_xy() -> dict[str, tuple[float, float]]:
    return {
        "fl": (float(args_cli.stance_half_length), float(args_cli.stance_half_width)),
        "fr": (float(args_cli.stance_half_length), -float(args_cli.stance_half_width)),
        "rl": (-float(args_cli.stance_half_length), float(args_cli.stance_half_width)),
        "rr": (-float(args_cli.stance_half_length), -float(args_cli.stance_half_width)),
    }


def design_scene(stage: Usd.Stage) -> None:
    UsdGeom.Xform.Define(stage, "/World/Robot")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/Robot"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    ground_material = _define_physics_material(stage, "/World/Looks/ground", args_cli.static_friction, args_cli.dynamic_friction)
    wheel_material = _define_physics_material(
        stage, "/World/Looks/wheel", args_cli.wheel_static_friction, args_cli.wheel_dynamic_friction
    )
    cage_material = _define_physics_material(stage, "/World/Looks/cage", args_cli.static_friction, args_cli.dynamic_friction)

    _spawn_box_body(
        stage,
        "/World/Ground",
        (8.0, 4.0, 0.04),
        0.0,
        (0.35, 0.35, 0.35),
        (0.8, 0.0, -0.02),
        physics_material=ground_material,
        rigid=False,
    )

    torso_size = tuple(float(v) for v in args_cli.torso_size)
    _spawn_box_body(stage, TORSO_PATH, torso_size, float(args_cli.torso_mass), (0.20, 0.25, 0.30), (0.0, 0.0, float(args_cli.torso_z)), physics_material=cage_material)

    wheel_z = float(args_cli.wheel_radius)
    torso_bottom_z = float(args_cli.torso_z) - torso_size[2] * 0.5
    local_wheel_z = wheel_z - float(args_cli.torso_z)
    for name, (x, y) in _wheel_xy().items():
        wheel_path = f"/World/Robot/{name}_wheel"
        _spawn_wheel(stage, wheel_path, float(args_cli.wheel_radius), float(args_cli.wheel_width), float(args_cli.wheel_mass), (x, y, wheel_z), physics_material=wheel_material)
        _wheel_joint(
            stage,
            f"/World/Robot/{name}_wheel_joint",
            TORSO_PATH,
            wheel_path,
            (x, y, local_wheel_z),
            damping=0.0 if str(args_cli.wheel_control_mode) == "effort" else float(args_cli.wheel_damping),
            max_force=float(args_cli.wheel_max_force),
        )

    cage_size = tuple(float(v) for v in args_cli.cage_size)
    box_size = tuple(float(v) for v in args_cli.payload_size)
    cage_x = float(args_cli.cage_local_x)
    cage_z = float(args_cli.torso_z) + float(args_cli.cage_local_z)
    deck_path = "/World/Robot/CageDeck"
    _spawn_box_body(stage, deck_path, cage_size, float(args_cli.cage_part_mass), (0.18, 0.28, 0.20), (cage_x, 0.0, cage_z), physics_material=cage_material)
    _fixed_joint(stage, "/World/Robot/CageDeck_fixed", TORSO_PATH, deck_path, (cage_x, 0.0, float(args_cli.cage_local_z)), (0.0, 0.0, 0.0))

    wall_h = float(args_cli.cage_wall_height)
    wall_t = float(args_cli.cage_wall_thickness)
    wall_z = cage_z + cage_size[2] * 0.5 + wall_h * 0.5
    parts = [
        ("left", (cage_size[0], wall_t, wall_h), (cage_x, cage_size[1] * 0.5 - wall_t * 0.5, wall_z)),
        ("right", (cage_size[0], wall_t, wall_h), (cage_x, -cage_size[1] * 0.5 + wall_t * 0.5, wall_z)),
        ("front", (wall_t, cage_size[1], wall_h), (cage_x + cage_size[0] * 0.5 - wall_t * 0.5, 0.0, wall_z)),
        ("rear", (wall_t, cage_size[1], wall_h), (cage_x - cage_size[0] * 0.5 + wall_t * 0.5, 0.0, wall_z)),
    ]
    for label, size, pos in parts:
        path = f"/World/Robot/Cage_{label}"
        _spawn_box_body(stage, path, size, float(args_cli.cage_part_mass), (0.12, 0.22, 0.16), pos, physics_material=cage_material)
        _fixed_joint(
            stage,
            f"/World/Robot/Cage_{label}_fixed",
            TORSO_PATH,
            path,
            (pos[0], pos[1], pos[2] - float(args_cli.torso_z)),
            (0.0, 0.0, 0.0),
        )

    lid_z = wall_z + wall_h * 0.5 + float(args_cli.cage_lid_clearance) + cage_size[2] * 0.5
    lid_path = "/World/Robot/Cage_lid"
    _spawn_box_body(stage, lid_path, cage_size, float(args_cli.cage_part_mass), (0.10, 0.20, 0.14), (cage_x, 0.0, lid_z), physics_material=cage_material)
    _fixed_joint(stage, "/World/Robot/Cage_lid_fixed", TORSO_PATH, lid_path, (cage_x, 0.0, lid_z - float(args_cli.torso_z)), (0.0, 0.0, 0.0))

    box_z = cage_z + cage_size[2] * 0.5 + box_size[2] * 0.5 + 0.003
    _spawn_box_body(stage, BOX_PATH, box_size, float(args_cli.payload_mass), (0.85, 0.42, 0.18), (cage_x, 0.0, box_z), physics_material=cage_material)


def _find_wheel_joint_indices(dof_names: list[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for name in WHEEL_NAMES:
        for idx, dof_name in enumerate(dof_names):
            if name in dof_name and "wheel_joint" in dof_name:
                indices[name] = idx
                break
    if len(indices) != len(WHEEL_NAMES):
        raise RuntimeError(f"Missing wheel joints: indices={indices}, dof_names={dof_names}")
    return indices


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_rolling_foot_cage_carrier_state.csv"
    summary_path = args_cli.output_dir / "core_world_rolling_foot_cage_carrier_summary.json"

    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    create_new_stage()
    stage = get_current_stage()
    design_scene(stage)

    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    robot = SingleArticulation(prim_path="/World/Robot", name="rolling_foot_carrier")
    torso = SingleRigidPrim(prim_path=TORSO_PATH, name="rolling_foot_torso")
    payload = SingleRigidPrim(prim_path=BOX_PATH, name="rolling_foot_payload")
    world.reset()
    robot.initialize()
    torso.initialize()
    payload.initialize()

    dof_names = list(robot.dof_names)
    wheel_indices = _find_wheel_joint_indices(dof_names)
    initial_joint_positions = np.array(robot.get_joint_positions(), dtype=float)
    initial_robot = _pose_wxyz(torso)
    initial_payload = _pose_wxyz(payload)
    settle_origin_robot_x: float | None = None
    settle_origin_payload_x: float | None = None
    target_direction = 1.0 if float(args_cli.target_x) >= 0.0 else -1.0
    signed_wheel_velocity = -target_direction * abs(float(args_cli.wheel_velocity))
    signed_wheel_effort = -target_direction * abs(float(args_cli.wheel_effort))
    wheel_velocity_targets = [signed_wheel_velocity for _ in WHEEL_NAMES]
    wheel_effort_targets = [signed_wheel_effort for _ in WHEEL_NAMES]
    wheel_joint_indices = [wheel_indices[name] for name in WHEEL_NAMES]

    summary: dict[str, object] = {
        "diagnostic_type": "rolling_foot_cage_carrier",
        "not_walking": True,
        "articulated_carrier_enabled": True,
        "wheel_joint_drive_enabled": True,
        "wheel_control_mode": str(args_cli.wheel_control_mode),
        "articulated_joint_count": int(len(dof_names)),
        "body_root_pose_write_count": 0,
        "body_root_velocity_command_count": 0,
        "box_pose_write_count": 0,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "settle_steps": int(args_cli.settle_steps),
        "drive_steps": int(args_cli.drive_steps),
        "target_x_m": float(args_cli.target_x),
        "wheel_velocity_radps": float(args_cli.wheel_velocity),
        "wheel_effort_nm": float(args_cli.wheel_effort),
        "wheel_radius_m": float(args_cli.wheel_radius),
        "payload_mass_kg": float(args_cli.payload_mass),
        "fall_events": 0,
        "box_drop_events": 0,
        "nonfinite_state_events": 0,
        "max_tilt_rad": 0.0,
        "min_torso_z_m": None,
        "min_payload_z_m": None,
        "max_robot_travel_x_m": 0.0,
        "final_robot_travel_x_m": 0.0,
        "max_payload_travel_x_m": 0.0,
        "final_payload_travel_x_m": 0.0,
        "final_post_settle_payload_travel_x_m": 0.0,
        "final_post_settle_payload_target_distance_x_m": abs(float(args_cli.target_x)),
        "payload_relative_error_m": 0.0,
        "max_payload_relative_offset_error_m": 0.0,
        "wheel_joint_indices": {k: int(v) for k, v in wheel_indices.items()},
        "max_abs_wheel_joint_motion_rad": 0.0,
        "final_abs_wheel_joint_motion_rad": 0.0,
        "dof_names": dof_names,
    }

    initial_rel = np.array(initial_payload[:3], dtype=float) - np.array(initial_robot[:3], dtype=float)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "robot_x",
                "robot_y",
                "robot_z",
                "payload_x",
                "payload_y",
                "payload_z",
                "post_settle_payload_travel_x",
                "payload_relative_error",
                "roll",
                "pitch",
                "fall",
                "drop",
            ],
        )
        writer.writeheader()
        for step in range(int(args_cli.steps)):
            if int(args_cli.settle_steps) <= step < int(args_cli.settle_steps) + int(args_cli.drive_steps):
                if str(args_cli.wheel_control_mode) == "effort":
                    robot.apply_action(
                        ArticulationAction(
                            joint_efforts=wheel_effort_targets,
                            joint_indices=wheel_joint_indices,
                        )
                    )
                else:
                    robot.apply_action(
                        ArticulationAction(
                            joint_velocities=wheel_velocity_targets,
                            joint_indices=wheel_joint_indices,
                        )
                    )
            else:
                if str(args_cli.wheel_control_mode) == "effort":
                    robot.apply_action(
                        ArticulationAction(
                            joint_efforts=[0.0 for _ in WHEEL_NAMES],
                            joint_indices=wheel_joint_indices,
                        )
                    )
                else:
                    robot.apply_action(
                        ArticulationAction(
                            joint_velocities=[0.0 for _ in WHEEL_NAMES],
                            joint_indices=wheel_joint_indices,
                        )
                    )
            world.step(render=bool(args_cli.render))

            robot_pose = _pose_wxyz(torso)
            payload_pose = _pose_wxyz(payload)
            joint_positions = np.array(robot.get_joint_positions(), dtype=float)
            finite = np.all(np.isfinite(np.array(robot_pose + payload_pose, dtype=float)))
            if not finite:
                summary["nonfinite_state_events"] = int(summary["nonfinite_state_events"]) + 1
                continue
            if step == int(args_cli.settle_steps):
                settle_origin_robot_x = float(robot_pose[0])
                settle_origin_payload_x = float(payload_pose[0])
            robot_travel_x = float(robot_pose[0] - initial_robot[0])
            payload_travel_x = float(payload_pose[0] - initial_payload[0])
            if settle_origin_payload_x is None:
                post_settle_payload_travel_x = 0.0
            else:
                post_settle_payload_travel_x = target_direction * float(payload_pose[0] - settle_origin_payload_x)
            rel = np.array(payload_pose[:3], dtype=float) - np.array(robot_pose[:3], dtype=float)
            rel_error = float(np.linalg.norm(rel - initial_rel))
            wheel_motion = 0.0
            if joint_positions.size and initial_joint_positions.size:
                wheel_motion = max(
                    abs(float(joint_positions[idx] - initial_joint_positions[idx]))
                    for idx in wheel_joint_indices
                )
            roll, pitch = _quat_to_roll_pitch(robot_pose[3], robot_pose[4], robot_pose[5], robot_pose[6])
            tilt = max(abs(roll), abs(pitch))
            fall = int(float(robot_pose[2]) < float(args_cli.fall_z) or tilt > 0.85)
            drop = int(float(payload_pose[2]) < float(args_cli.drop_z))
            summary["completed_steps"] = int(step) + 1
            summary["fall_events"] = int(summary["fall_events"]) + fall
            summary["box_drop_events"] = int(summary["box_drop_events"]) + drop
            summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), tilt)
            summary["min_torso_z_m"] = robot_pose[2] if summary["min_torso_z_m"] is None else min(float(summary["min_torso_z_m"]), robot_pose[2])
            summary["min_payload_z_m"] = payload_pose[2] if summary["min_payload_z_m"] is None else min(float(summary["min_payload_z_m"]), payload_pose[2])
            summary["max_robot_travel_x_m"] = max(float(summary["max_robot_travel_x_m"]), target_direction * robot_travel_x)
            summary["final_robot_travel_x_m"] = target_direction * robot_travel_x
            summary["max_payload_travel_x_m"] = max(float(summary["max_payload_travel_x_m"]), target_direction * payload_travel_x)
            summary["final_payload_travel_x_m"] = target_direction * payload_travel_x
            summary["final_post_settle_payload_travel_x_m"] = post_settle_payload_travel_x
            summary["final_post_settle_payload_target_distance_x_m"] = abs(float(args_cli.target_x)) - post_settle_payload_travel_x
            summary["payload_relative_error_m"] = rel_error
            summary["max_payload_relative_offset_error_m"] = max(float(summary["max_payload_relative_offset_error_m"]), rel_error)
            summary["max_abs_wheel_joint_motion_rad"] = max(float(summary["max_abs_wheel_joint_motion_rad"]), wheel_motion)
            summary["final_abs_wheel_joint_motion_rad"] = wheel_motion
            if step % 10 == 0 or step == int(args_cli.steps) - 1:
                print(
                    "[STATE] "
                    f"step={step} robot=({robot_pose[0]:.3f},{robot_pose[1]:.3f},{robot_pose[2]:.3f}) "
                    f"payload_z={payload_pose[2]:.3f} post_travel={post_settle_payload_travel_x:.4f} "
                    f"rel_err={rel_error:.4f} wheel_motion={wheel_motion:.4f} "
                    f"tilt={tilt:.4f} fall={summary['fall_events']} drop={summary['box_drop_events']}",
                    flush=True,
                )
            writer.writerow(
                {
                    "step": step,
                    "robot_x": robot_pose[0],
                    "robot_y": robot_pose[1],
                    "robot_z": robot_pose[2],
                    "payload_x": payload_pose[0],
                    "payload_y": payload_pose[1],
                    "payload_z": payload_pose[2],
                    "post_settle_payload_travel_x": post_settle_payload_travel_x,
                    "payload_relative_error": rel_error,
                    "roll": roll,
                    "pitch": pitch,
                    "fall": fall,
                    "drop": drop,
                }
            )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


try:
    run_scene()
finally:
    simulation_app.close()
