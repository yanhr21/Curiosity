#!/usr/bin/env python3
"""Isaac proxy carrier scene for box-carry task scaffolding.

This is a scene-construction diagnostic, not a humanoid-control success claim.
It avoids IsaacLab articulation tensors while preserving the carry payload,
mass, target, drop, and travel metrics needed by the later G1 scene.
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
    parser = argparse.ArgumentParser(description="Kinematic carrier + dynamic box Isaac carry-scene scaffold.")
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--box-mass", type=float, default=5.0)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.55, 0.35, 0.35), metavar=("X", "Y", "Z"))
    parser.add_argument("--carrier-speed", type=float, default=0.35)
    parser.add_argument("--carrier-height", type=float, default=0.78)
    parser.add_argument("--carry-offset", type=float, nargs=3, default=(0.42, 0.0, 0.06), metavar=("X", "Y", "Z"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/proxy_carry_scene"),
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


TARGET_POSITION_XY = (1.8, 0.0)
BOX_DROP_HEIGHT_M = 0.10


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


def _set_translate(stage: Usd.Stage, prim_path: str, xyz: tuple[float, float, float]) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(*xyz))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))


def _create_fixed_joint(stage: Usd.Stage, body0: str, body1: str, local_pos0: tuple[float, float, float]) -> str:
    joint_path = f"{body1}/FixedJointToCarrier"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(local_pos0[0]), float(local_pos0[1]), float(local_pos0[2])))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint_path


def design_scene(stage: Usd.Stage) -> tuple[str, str, str]:
    floor_cfg = sim_utils.CuboidCfg(
        size=(8.0, 8.0, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.34, 0.34), roughness=0.9),
    )
    floor_cfg.func("/World/Ground", floor_cfg, translation=(0.0, 0.0, -0.025))
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)).func("/World/DomeLight", sim_utils.DomeLightCfg())
    sim_utils.DistantLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.8)).func(
        "/World/KeyLight", sim_utils.DistantLightCfg(), translation=(2.0, -2.0, 4.0)
    )

    target_cfg = sim_utils.CuboidCfg(
        size=(0.65, 0.45, 0.02),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 0.85), opacity=0.35),
    )
    target_cfg.func("/World/CarryTarget", target_cfg, translation=(TARGET_POSITION_XY[0], TARGET_POSITION_XY[1], 0.01))

    carrier_cfg = sim_utils.CuboidCfg(
        size=(0.36, 0.28, 0.58),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            max_linear_velocity=4.0,
            max_angular_velocity=4.0,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=45.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.24, 0.32), roughness=0.65),
    )
    carrier_path = "/World/CarrierTorso"
    carrier_cfg.func(carrier_path, carrier_cfg, translation=(0.0, 0.0, args_cli.carrier_height))

    box_cfg = sim_utils.CuboidCfg(
        size=tuple(args_cli.box_size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.box_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.25), roughness=0.8),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
    )
    box_path = "/World/CarryBox"
    box_pos = (
        args_cli.carry_offset[0],
        args_cli.carry_offset[1],
        args_cli.carrier_height + args_cli.carry_offset[2],
    )
    box_cfg.func(box_path, box_cfg, translation=box_pos)
    joint_path = _create_fixed_joint(stage, carrier_path, box_path, tuple(args_cli.carry_offset))
    return carrier_path, box_path, joint_path


def run_scene(carrier_path: str, box_path: str, joint_path: str) -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "proxy_carry_scene_state.csv"
    summary_path = args_cli.output_dir / "proxy_carry_scene_summary.json"

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    with build_simulation_context(create_new_stage=False, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        sim.set_camera_view(eye=[2.7, -2.2, 1.7], target=[0.8, 0.0, 0.7])
        sim.reset()
        stage = sim.stage
        sim_dt = sim.get_physics_dt()
        initial_box_pose = None
        initial_carrier_pose = None
        summary = {
            "steps_requested": int(args_cli.steps),
            "completed_steps": 0,
            "physics_dt": float(sim_dt),
            "scene_type": "kinematic_proxy_carrier_fixed_payload",
            "success_claim": "diagnostic_only_not_humanoid_control",
            "box_mass_kg": float(args_cli.box_mass),
            "box_size_m": [float(v) for v in args_cli.box_size],
            "carrier_speed_mps": float(args_cli.carrier_speed),
            "joint_path": joint_path,
            "max_carrier_travel_xy_m": 0.0,
            "max_box_travel_xy_m": 0.0,
            "min_box_z_m": None,
            "min_box_target_distance_xy_m": None,
            "box_drop_events": 0,
        }

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "time_s",
                    "carrier_x",
                    "carrier_y",
                    "carrier_z",
                    "box_x",
                    "box_y",
                    "box_z",
                    "box_qw",
                    "box_qx",
                    "box_qy",
                    "box_qz",
                    "box_drop_flag",
                    "carrier_travel_xy_m",
                    "box_travel_xy_m",
                    "box_target_distance_xy_m",
                ]
            )
            for step in range(args_cli.steps):
                t = step * sim_dt
                sway = 0.015 * math.sin(2.0 * math.pi * 1.4 * t)
                bob = 0.015 * math.sin(2.0 * math.pi * 2.8 * t)
                carrier_pos = (args_cli.carrier_speed * t, sway, args_cli.carrier_height + bob)
                box_pos = (
                    carrier_pos[0] + args_cli.carry_offset[0],
                    carrier_pos[1] + args_cli.carry_offset[1],
                    carrier_pos[2] + args_cli.carry_offset[2],
                )
                _set_translate(stage, carrier_path, carrier_pos)
                _set_translate(stage, box_path, box_pos)
                sim.step(render=args_cli.render)

                if step % 10 == 0 or step == args_cli.steps - 1:
                    carrier_pose = _pose_wxyz(stage, carrier_path)
                    box_pose = _pose_wxyz(stage, box_path)
                    if initial_carrier_pose is None:
                        initial_carrier_pose = list(carrier_pose)
                    if initial_box_pose is None:
                        initial_box_pose = list(box_pose)

                    carrier_travel = math.hypot(carrier_pose[0] - initial_carrier_pose[0], carrier_pose[1] - initial_carrier_pose[1])
                    box_travel = math.hypot(box_pose[0] - initial_box_pose[0], box_pose[1] - initial_box_pose[1])
                    target_distance = math.hypot(box_pose[0] - TARGET_POSITION_XY[0], box_pose[1] - TARGET_POSITION_XY[1])
                    drop_flag = int(box_pose[2] < BOX_DROP_HEIGHT_M)
                    summary["completed_steps"] = int(step + 1)
                    summary["max_carrier_travel_xy_m"] = max(float(summary["max_carrier_travel_xy_m"]), float(carrier_travel))
                    summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), float(box_travel))
                    summary["min_box_z_m"] = (
                        float(box_pose[2])
                        if summary["min_box_z_m"] is None
                        else min(float(summary["min_box_z_m"]), float(box_pose[2]))
                    )
                    summary["min_box_target_distance_xy_m"] = (
                        float(target_distance)
                        if summary["min_box_target_distance_xy_m"] is None
                        else min(float(summary["min_box_target_distance_xy_m"]), float(target_distance))
                    )
                    summary["box_drop_events"] += drop_flag
                    writer.writerow(
                        [
                            step,
                            t,
                            carrier_pose[0],
                            carrier_pose[1],
                            carrier_pose[2],
                            *box_pose,
                            drop_flag,
                            carrier_travel,
                            box_travel,
                            target_distance,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} carrier=({carrier_pose[0]:.3f},{carrier_pose[1]:.3f},{carrier_pose[2]:.3f}) "
                        f"box=({box_pose[0]:.3f},{box_pose[1]:.3f},{box_pose[2]:.3f}) "
                        f"box_drop={drop_flag}"
                    )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    print(f"[INFO] Metrics written to: {csv_path}")
    return csv_path


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    with build_simulation_context(create_new_stage=True, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        sim.set_camera_view(eye=[2.7, -2.2, 1.7], target=[0.8, 0.0, 0.7])
        carrier_path, box_path, joint_path = design_scene(sim.stage)
        print("[INFO] Proxy carry scene setup complete.")
        print(f"[INFO] Carrier path: {carrier_path}")
        print(f"[INFO] Box path: {box_path}")
        print(f"[INFO] Payload joint: {joint_path}")
        sim.reset()
        stage = sim.stage
        sim_dt = sim.get_physics_dt()
        initial_box_pose = None
        initial_carrier_pose = None
        args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args_cli.output_dir / "proxy_carry_scene_state.csv"
        summary_path = args_cli.output_dir / "proxy_carry_scene_summary.json"
        summary = {
            "steps_requested": int(args_cli.steps),
            "completed_steps": 0,
            "physics_dt": float(sim_dt),
            "scene_type": "kinematic_proxy_carrier_pose-follow_payload",
            "success_claim": "diagnostic_only_not_humanoid_control_or_grasp",
            "box_mass_kg": float(args_cli.box_mass),
            "box_size_m": [float(v) for v in args_cli.box_size],
            "carrier_speed_mps": float(args_cli.carrier_speed),
            "joint_path": joint_path,
            "max_carrier_travel_xy_m": 0.0,
            "max_box_travel_xy_m": 0.0,
            "min_box_z_m": None,
            "min_box_target_distance_xy_m": None,
            "box_drop_events": 0,
        }

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "time_s",
                    "carrier_x",
                    "carrier_y",
                    "carrier_z",
                    "box_x",
                    "box_y",
                    "box_z",
                    "box_qw",
                    "box_qx",
                    "box_qy",
                    "box_qz",
                    "box_drop_flag",
                    "carrier_travel_xy_m",
                    "box_travel_xy_m",
                    "box_target_distance_xy_m",
                ]
            )
            for step in range(args_cli.steps):
                t = step * sim_dt
                sway = 0.015 * math.sin(2.0 * math.pi * 1.4 * t)
                bob = 0.015 * math.sin(2.0 * math.pi * 2.8 * t)
                carrier_pos = (args_cli.carrier_speed * t, sway, args_cli.carrier_height + bob)
                box_pos = (
                    carrier_pos[0] + args_cli.carry_offset[0],
                    carrier_pos[1] + args_cli.carry_offset[1],
                    carrier_pos[2] + args_cli.carry_offset[2],
                )
                _set_translate(stage, carrier_path, carrier_pos)
                _set_translate(stage, box_path, box_pos)
                sim.step(render=args_cli.render)

                if step % 10 == 0 or step == args_cli.steps - 1:
                    carrier_pose = _pose_wxyz(stage, carrier_path)
                    box_pose = _pose_wxyz(stage, box_path)
                    if initial_carrier_pose is None:
                        initial_carrier_pose = list(carrier_pose)
                    if initial_box_pose is None:
                        initial_box_pose = list(box_pose)

                    carrier_travel = math.hypot(
                        carrier_pose[0] - initial_carrier_pose[0], carrier_pose[1] - initial_carrier_pose[1]
                    )
                    box_travel = math.hypot(box_pose[0] - initial_box_pose[0], box_pose[1] - initial_box_pose[1])
                    target_distance = math.hypot(box_pose[0] - TARGET_POSITION_XY[0], box_pose[1] - TARGET_POSITION_XY[1])
                    drop_flag = int(box_pose[2] < BOX_DROP_HEIGHT_M)
                    summary["completed_steps"] = int(step + 1)
                    summary["max_carrier_travel_xy_m"] = max(float(summary["max_carrier_travel_xy_m"]), float(carrier_travel))
                    summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), float(box_travel))
                    summary["min_box_z_m"] = (
                        float(box_pose[2])
                        if summary["min_box_z_m"] is None
                        else min(float(summary["min_box_z_m"]), float(box_pose[2]))
                    )
                    summary["min_box_target_distance_xy_m"] = (
                        float(target_distance)
                        if summary["min_box_target_distance_xy_m"] is None
                        else min(float(summary["min_box_target_distance_xy_m"]), float(target_distance))
                    )
                    summary["box_drop_events"] += drop_flag
                    writer.writerow(
                        [
                            step,
                            t,
                            carrier_pose[0],
                            carrier_pose[1],
                            carrier_pose[2],
                            *box_pose,
                            drop_flag,
                            carrier_travel,
                            box_travel,
                            target_distance,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} carrier=({carrier_pose[0]:.3f},{carrier_pose[1]:.3f},{carrier_pose[2]:.3f}) "
                        f"box=({box_pose[0]:.3f},{box_pose[1]:.3f},{box_pose[2]:.3f}) "
                        f"box_drop={drop_flag}"
                    )

        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[INFO] Summary written to: {summary_path}")
        print(f"[INFO] Metrics written to: {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
