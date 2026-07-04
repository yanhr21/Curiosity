#!/usr/bin/env python3
"""Velocity-controlled dynamic Isaac carry diagnostic.

This is an engineering probe for the current Isaac environment.  It avoids the
IsaacLab Articulation/RigidObject tensor paths that have been invalidated in
prior runs and avoids USD joint-drive actuation that produced zero travel.

The carrier has a dynamic torso rigid body and a dynamic payload box connected
by a fixed joint.  A runtime force or velocity servo is used while visual legs
execute a walking gait.

Passing this smoke would mean only: a dynamic rigid-body carrier can move with a
fixed payload in Isaac without falling or dropping the box through a non-tensor
runtime-control path.  It is not a legged articulation controller, not unknown
free-object grasping, and not a learned policy.
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
    parser = argparse.ArgumentParser(description="Velocity-controlled dynamic Isaac carry smoke.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--payload-mass", type=float, default=5.0)
    parser.add_argument("--target-x", type=float, default=1.4)
    parser.add_argument("--target-speed", type=float, default=0.38)
    parser.add_argument("--target-height", type=float, default=0.58)
    parser.add_argument("--control-mode", choices=("physx_force", "velocity_attr"), default="physx_force")
    parser.add_argument("--step-mode", choices=("physx_direct", "sim_step"), default="physx_direct")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/velocity_controlled_dynamic_carry_scene"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import build_simulation_context  # noqa: E402


TORSO_PATH = "/World/Robot/Torso"
BOX_PATH = "/World/CarryBox"
LEG_PARTS = (
    "left_thigh",
    "right_thigh",
    "left_shin",
    "right_shin",
    "left_foot",
    "right_foot",
)


def _set_xform(prim: Usd.Prim, translation: tuple[float, float, float], scale: tuple[float, float, float]) -> None:
    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(*[float(v) for v in translation]))
    xform_api.SetScale(Gf.Vec3f(*[float(v) for v in scale]))


def _set_translate(stage: Usd.Stage, prim_path: str, xyz: tuple[float, float, float]) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(*[float(v) for v in xyz]))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in xyz]))


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


def _spawn_dynamic_box(
    path: str,
    size: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    translation: tuple[float, float, float],
    *,
    friction: float = 1.0,
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.12,
            angular_damping=0.18,
            max_linear_velocity=5.0,
            max_angular_velocity=5.0,
            max_depenetration_velocity=2.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=float(mass)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=0.8 * friction),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.82),
    )
    cfg.func(path, cfg, translation=translation)


def _spawn_visual_box(stage: Usd.Stage, path: str, size: tuple[float, float, float], color: tuple[float, float, float]) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), (0.0, 0.0, 0.0), size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])


def _fixed_joint(stage: Usd.Stage, joint_path: str, body0: str, body1: str) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.30, 0.0, 0.03))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)


def design_scene(stage: Usd.Stage) -> tuple[UsdPhysics.RigidBodyAPI, UsdPhysics.RigidBodyAPI]:
    floor_cfg = sim_utils.CuboidCfg(
        size=(5.0, 2.5, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.1),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.33, 0.33), roughness=0.9),
    )
    floor_cfg.func("/World/Ground", floor_cfg, translation=(0.0, 0.0, -0.025))
    sim_utils.DomeLightCfg(intensity=2200.0, color=(0.82, 0.82, 0.82)).func("/World/DomeLight", sim_utils.DomeLightCfg())
    target_cfg = sim_utils.CuboidCfg(
        size=(0.5, 0.42, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.40, 0.85), opacity=0.38),
    )
    target_cfg.func("/World/CarryTarget", target_cfg, translation=(float(args_cli.target_x), 0.0, 0.01))

    _spawn_dynamic_box(TORSO_PATH, (0.46, 0.24, 0.18), 18.0, (0.14, 0.20, 0.30), (0.0, 0.0, float(args_cli.target_height)))
    _spawn_dynamic_box(
        BOX_PATH,
        (0.32, 0.24, 0.24),
        float(args_cli.payload_mass),
        (0.56, 0.42, 0.23),
        (0.30, 0.0, float(args_cli.target_height) + 0.03),
    )
    _fixed_joint(stage, "/World/Robot/FixedPayloadJoint", TORSO_PATH, BOX_PATH)

    UsdGeom.Xform.Define(stage, "/World/Robot")
    for part in LEG_PARTS:
        _spawn_visual_box(stage, f"/World/Robot/{part}", (0.08, 0.08, 0.32), (0.08, 0.14, 0.22))

    torso_rb = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(TORSO_PATH))
    box_rb = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(BOX_PATH))
    for rb in (torso_rb, box_rb):
        rb.CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        rb.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    return torso_rb, box_rb


def _update_visual_gait(stage: Usd.Stage, torso_pose: list[float], t: float) -> None:
    torso_x, torso_y, _torso_z = torso_pose[:3]
    phase = math.sin(2.0 * math.pi * 1.4 * t)
    stride = 0.13 * phase
    left_foot = (torso_x + stride, torso_y + 0.12, 0.035)
    right_foot = (torso_x - stride, torso_y - 0.12, 0.035)
    left_hip = (torso_x - 0.05, torso_y + 0.11, 0.50)
    right_hip = (torso_x - 0.05, torso_y - 0.11, 0.50)
    _set_translate(stage, "/World/Robot/left_foot", left_foot)
    _set_translate(stage, "/World/Robot/right_foot", right_foot)
    _set_translate(stage, "/World/Robot/left_thigh", ((left_hip[0] + left_foot[0]) * 0.5, left_hip[1], 0.34))
    _set_translate(stage, "/World/Robot/right_thigh", ((right_hip[0] + right_foot[0]) * 0.5, right_hip[1], 0.34))
    _set_translate(stage, "/World/Robot/left_shin", ((left_hip[0] + left_foot[0]) * 0.5, left_foot[1], 0.18))
    _set_translate(stage, "/World/Robot/right_shin", ((right_hip[0] + right_foot[0]) * 0.5, right_foot[1], 0.18))


def _apply_force_control(
    physx_simulation_interface: object,
    stage_id: int,
    body_id: int,
    torso: list[float],
    previous_torso: list[float] | None,
    dt: float,
) -> tuple[float, float]:
    roll, pitch = _quat_to_roll_pitch(torso[3], torso[4], torso[5], torso[6])
    if previous_torso is None:
        vx = 0.0
        vz = 0.0
    else:
        vx = (torso[0] - previous_torso[0]) / max(1e-6, dt)
        vz = (torso[2] - previous_torso[2]) / max(1e-6, dt)

    total_mass = 18.0 + float(args_cli.payload_mass)
    remaining = float(args_cli.target_x) - (torso[0] + 0.30)
    target_speed = float(args_cli.target_speed) if remaining > 0.04 else 0.0
    force_x = max(-90.0, min(90.0, 220.0 * (target_speed - vx)))
    force_z = total_mass * 9.81 + 1200.0 * (float(args_cli.target_height) - torso[2]) - 90.0 * vz
    force_z = max(-80.0, min(520.0, force_z))

    center = carb.Float3(float(torso[0]), float(torso[1]), float(torso[2]))
    physx_simulation_interface.apply_force_at_pos(stage_id, body_id, carb.Float3(float(force_x), 0.0, float(force_z)), center)

    roll_torque = max(-90.0, min(90.0, -220.0 * roll))
    pitch_torque = max(-90.0, min(90.0, -220.0 * pitch))
    y_lever = 0.13
    x_lever = 0.20
    roll_force = roll_torque / max(1e-6, 2.0 * y_lever)
    pitch_force = pitch_torque / max(1e-6, 2.0 * x_lever)
    for pos, force in (
        (
            carb.Float3(float(torso[0]), float(torso[1] + y_lever), float(torso[2])),
            carb.Float3(0.0, 0.0, float(roll_force)),
        ),
        (
            carb.Float3(float(torso[0]), float(torso[1] - y_lever), float(torso[2])),
            carb.Float3(0.0, 0.0, float(-roll_force)),
        ),
        (
            carb.Float3(float(torso[0] + x_lever), float(torso[1]), float(torso[2])),
            carb.Float3(0.0, 0.0, float(pitch_force)),
        ),
        (
            carb.Float3(float(torso[0] - x_lever), float(torso[1]), float(torso[2])),
            carb.Float3(0.0, 0.0, float(-pitch_force)),
        ),
    ):
        physx_simulation_interface.apply_force_at_pos(stage_id, body_id, force, pos)
    return float(force_x), float(force_z)


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "velocity_controlled_dynamic_carry_state.csv"
    summary_path = args_cli.output_dir / "velocity_controlled_dynamic_carry_summary.json"
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)

    with build_simulation_context(create_new_stage=False, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        sim.set_setting("/physics/updateToUsd", True)
        sim.set_setting("/physics/updateVelocitiesToUsd", True)
        torso_rb, box_rb = design_scene(sim.stage)
        stage_id = UsdUtils.StageCache.Get().GetId(sim.stage).ToLongInt()
        torso_body_id = PhysicsSchemaTools.sdfPathToInt(sim.stage.GetPrimAtPath(TORSO_PATH).GetPath())
        physx_simulation_interface = omni.physx.get_physx_simulation_interface()
        sim.set_camera_view(eye=[2.4, -1.6, 1.05], target=[0.7, 0.0, 0.42])
        sim.reset()
        omni.timeline.get_timeline_interface().play()
        if hasattr(physx_simulation_interface, "flush_changes"):
            physx_simulation_interface.flush_changes()
        dt = float(sim.get_physics_dt())
        initial_torso = None
        initial_box = None
        previous_torso_for_control = None
        summary = {
            "scene_type": "isaac_velocity_controlled_dynamic_base_fixed_payload_carry",
            "success_claim": "diagnostic_only_velocity_servo_dynamic_base_not_legged_articulation_or_learned_policy",
            "uses_isaaclab_tensor_api": False,
            "control_path": str(args_cli.control_mode),
            "step_mode": str(args_cli.step_mode),
            "payload_mode": "fixed_joint_to_dynamic_torso",
            "visual_gait_only": True,
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
            "max_box_torso_separation_m": 0.0,
        }

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
                    "command_or_force_x",
                    "command_or_force_z",
                    "tilt",
                    "torso_travel_xy_m",
                    "box_travel_xy_m",
                    "box_torso_separation_m",
                    "box_target_distance_xy_m",
                    "fall",
                    "box_drop",
                ]
            )
            for step in range(int(args_cli.steps)):
                t = step * dt
                torso = _pose_wxyz(sim.stage, TORSO_PATH)
                box = _pose_wxyz(sim.stage, BOX_PATH)
                roll, pitch = _quat_to_roll_pitch(torso[3], torso[4], torso[5], torso[6])
                if args_cli.control_mode == "velocity_attr":
                    remaining = float(args_cli.target_x) - box[0]
                    desired_vx = min(float(args_cli.target_speed), max(0.0, 2.0 * remaining))
                    desired_vz = 4.0 * (float(args_cli.target_height) - torso[2])
                    angular = Gf.Vec3f(float(-2.2 * roll), float(-2.2 * pitch), 0.0)
                    velocity = Gf.Vec3f(float(desired_vx), 0.0, float(desired_vz))
                    torso_rb.GetVelocityAttr().Set(velocity)
                    torso_rb.GetAngularVelocityAttr().Set(angular)
                    command_x = float(desired_vx)
                    command_z = float(desired_vz)
                else:
                    command_x, command_z = _apply_force_control(
                        physx_simulation_interface,
                        stage_id,
                        torso_body_id,
                        torso,
                        previous_torso_for_control,
                        dt,
                    )
                    previous_torso_for_control = list(torso)
                # The box is attached by a fixed joint.  Its velocity is not
                # commanded separately; separation is logged to detect joint or
                # runtime-control failure.
                _update_visual_gait(sim.stage, torso, t)
                if args_cli.step_mode == "physx_direct":
                    physx_simulation_interface.simulate(dt, 0)
                    physx_simulation_interface.fetch_results()
                    simulation_app.update()
                else:
                    sim.step(render=args_cli.render)

                if step % 10 == 0 or step == int(args_cli.steps) - 1:
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
                    separation = math.sqrt((box[0] - torso[0] - 0.30) ** 2 + (box[1] - torso[1]) ** 2 + (box[2] - torso[2] - 0.03) ** 2)
                    target_distance = math.hypot(box[0] - float(args_cli.target_x), box[1])
                    fall = int(torso[2] < 0.34 or tilt > 0.65)
                    box_drop = int(box[2] < 0.20 or separation > 0.18)
                    summary["completed_steps"] = int(step + 1)
                    summary["fall_events"] += fall
                    summary["box_drop_events"] += box_drop
                    summary["max_torso_travel_xy_m"] = max(float(summary["max_torso_travel_xy_m"]), float(torso_travel))
                    summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), float(box_travel))
                    summary["final_box_target_distance_xy_m"] = float(target_distance)
                    summary["min_torso_z_m"] = (
                        float(torso[2]) if summary["min_torso_z_m"] is None else min(float(summary["min_torso_z_m"]), float(torso[2]))
                    )
                    summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), float(tilt))
                    summary["max_box_torso_separation_m"] = max(float(summary["max_box_torso_separation_m"]), float(separation))
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
                            command_x,
                            command_z,
                            tilt,
                            torso_travel,
                            box_travel,
                            separation,
                            target_distance,
                            fall,
                            box_drop,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} torso=({torso[0]:.3f},{torso[1]:.3f},{torso[2]:.3f}) "
                        f"box=({box[0]:.3f},{box[1]:.3f},{box[2]:.3f}) "
                        f"travel={box_travel:.3f} target={target_distance:.3f} "
                        f"tilt={tilt:.3f} sep={separation:.3f} fall={fall} drop={box_drop}",
                        flush=True,
                    )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    print(f"[INFO] Metrics written to: {csv_path}")
    return summary_path


def main() -> None:
    run_scene()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
