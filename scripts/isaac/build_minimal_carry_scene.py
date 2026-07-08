#!/usr/bin/env python3
"""Build and smoke-test a minimal Isaac carry scene.

This is a physics-scene scaffold, not a policy success claim. It deliberately
avoids the heavy Arena Galileo scene and GR00T policy server.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Optional

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Isaac scene for G1 box-carry scaffolding.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--box-mass", type=float, default=5.0)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.55, 0.35, 0.35), metavar=("X", "Y", "Z"))
    parser.add_argument("--box-position", type=float, nargs=3, default=(0.85, 0.0, 0.45), metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--skip-robot",
        action="store_true",
        help="Build only the floor, target marker, and physical carry box. This is the fastest Isaac scene smoke test.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render every simulation step. Leave disabled for the first headless CSV smoke test.",
    )
    parser.add_argument(
        "--disable-fabric",
        action="store_true",
        help="Disable Fabric. Keep Fabric enabled for robot tensor-control smoke tests.",
    )
    parser.add_argument(
        "--disable-usd-physics-updates",
        action="store_true",
        help="Diagnostic only: do not force PhysX pose/velocity synchronization back to USD during stepping.",
    )
    parser.add_argument(
        "--skip-explicit-state-reset",
        action="store_true",
        help=(
            "Diagnostic only: skip explicit Articulation/RigidObject state writes after sim.reset(). "
            "Use only to isolate tensor backend reset failures."
        ),
    )
    parser.add_argument(
        "--wbc-mode",
        choices=("none", "stand", "walk"),
        default="none",
        help="Use Arena's official G1 WBC policy for standing or walking. Requires --skip-robot to be false.",
    )
    parser.add_argument(
        "--walk-command",
        type=float,
        nargs=3,
        default=(0.25, 0.0, 0.0),
        metavar=("VX", "VY", "YAW_RATE"),
        help="Base velocity command used when --wbc-mode walk is active.",
    )
    parser.add_argument("--base-height-command", type=float, default=0.75)
    parser.add_argument(
        "--wbc-asset-root",
        type=Path,
        default=Path(
            "/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/wbc_policy"
        ),
        help="Local mirror root containing Arena WBC ONNX models and G1 robot_model assets.",
    )
    parser.add_argument(
        "--attach-box",
        choices=("none", "fixed_torso"),
        default="none",
        help="Attach the box as a physical payload for balance diagnostics. This is not a grasp success claim.",
    )
    parser.add_argument(
        "--attach-body-path",
        default="/World/G1/torso_link",
        help="Robot rigid body prim path used when --attach-box fixed_torso is active.",
    )
    parser.add_argument(
        "--attach-local-pos0",
        type=float,
        nargs=3,
        default=(0.28, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Robot-body local payload joint position for --attach-box fixed_torso.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/minimal_carry_scene"),
    )
    parser.add_argument(
        "--g1-usd",
        type=Path,
        default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/g1_29dof_with_hand_rev_1_0.usd"),
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    return args


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, build_simulation_context
from isaaclab_physx.assets import RigidObject
from isaaclab.utils import configclass


TARGET_POSITION_XY = (1.8, 0.0)
ROBOT_FALL_HEIGHT_M = 0.45
BOX_DROP_HEIGHT_M = 0.10


class G1WBCDriver:
    """Thin adapter around Arena's official G1 WBC policy for direct-scene smoke tests."""

    def __init__(
        self,
        robot: Articulation,
        mode: str,
        walk_command: tuple[float, float, float],
        base_height: float,
        wbc_asset_root: Path,
    ):
        import numpy as np
        import yaml

        from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action import (
            convert_sim_joint_to_wbc_joint,
            postprocess_actions,
            prepare_observations,
        )
        from isaaclab_arena_g1.g1_env.g1_supplemental_info import G1SupplementalInfo
        from isaaclab_arena_g1.g1_env.robot_model import RobotModel
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.config.configs import HomieV2Config
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.wbc_policy_factory import get_wbc_policy

        self.robot = robot
        self.mode = mode
        self._np = np
        self._prepare_observations = prepare_observations
        self._convert_sim_joint_to_wbc_joint = convert_sim_joint_to_wbc_joint
        self._postprocess_actions = postprocess_actions

        wbc_asset_root = wbc_asset_root.expanduser().resolve()
        stand_onnx = wbc_asset_root / "models/homie_v2/stand.onnx"
        walk_onnx = wbc_asset_root / "models/homie_v2/walk.onnx"
        robot_asset_path = wbc_asset_root / "robot_model/g1"
        robot_urdf = robot_asset_path / "g1_29dof_with_hand.urdf"
        required_paths = (stand_onnx, walk_onnx, robot_urdf, robot_asset_path / "meshes")
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing local WBC asset(s): " + ", ".join(missing))

        self.robot_model = RobotModel(
            str(robot_urdf),
            str(robot_asset_path),
            supplemental_info=G1SupplementalInfo(),
        )
        wbc_config = HomieV2Config()
        wbc_config.wbc_model_path = f"{stand_onnx},{walk_onnx}"
        self.wbc_policy = get_wbc_policy("g1", self.robot_model, wbc_config, num_envs=1)

        order_path = (
            Path(__file__).resolve().parents[2]
            / "external/IsaacLab-Arena/isaaclab_arena_g1/g1_env/config/loco_manip_g1_joints_order_43dof.yaml"
        )
        with order_path.open() as f:
            self.wbc_g1_joints_order = yaml.safe_load(f)

        self.device = wp.to_torch(robot.data.default_joint_pos).device
        default_joint_pos = wp.to_torch(robot.data.default_joint_pos).detach().cpu().numpy()
        default_wbc_joint_pos = self._convert_sim_joint_to_wbc_joint(
            default_joint_pos, robot.data.joint_names, self.wbc_g1_joints_order
        )
        upper_ids = self.robot_model.get_joint_group_indices("upper_body")
        self.upper_body_target = default_wbc_joint_pos[:, upper_ids].copy()

        navigate_cmd = (0.0, 0.0, 0.0) if mode == "stand" else tuple(walk_command)
        self.goal = {
            "navigate_cmd": np.array([navigate_cmd], dtype=np.float32),
            "base_height_command": np.array([[base_height]], dtype=np.float32),
            "toggle_policy_action": np.array([[0]], dtype=np.float32),
            "torso_orientation_rpy_cmd": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        }

    def compute_joint_target(self) -> torch.Tensor:
        wbc_obs = self._prepare_observations(1, self.robot.data, self.wbc_g1_joints_order)
        self.wbc_policy.set_goal(self.goal)
        self.wbc_policy.set_observation(wbc_obs)
        wbc_action = self.wbc_policy.get_action(self.upper_body_target)
        return self._postprocess_actions(wbc_action, self.robot.data, self.wbc_g1_joints_order, self.device)


def _make_box_cfg(args: argparse.Namespace) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/CarryBox",
        spawn=sim_utils.CuboidCfg(
            size=tuple(args.box_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args.box_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.25), roughness=0.8),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(args.box_position), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _read_usd_world_pose(sim: SimulationContext, prim_path: str) -> list[float]:
    prim = sim.stage.GetPrimAtPath(prim_path)
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


def _read_first_valid_usd_pose(sim: SimulationContext, prim_paths: tuple[str, ...]) -> list[float] | None:
    for prim_path in prim_paths:
        prim = sim.stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            return _read_usd_world_pose(sim, prim_path)
    return None


def _read_asset_root_pose_values(asset, sim: SimulationContext, prim_path: str) -> list[float]:
    """Read asset root pose as x, y, z, qw, qx, qy, qz.

    IsaacLab tensor poses use xyzw quaternion order. The CSV keeps wxyz order to
    match the existing USD fallback rows.
    """

    try:
        pose = wp.to_torch(asset.data.root_link_pose_w).detach().cpu()[0].tolist()
        return [
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
            float(pose[6]),
            float(pose[3]),
            float(pose[4]),
            float(pose[5]),
        ]
    except Exception as exc:
        print(f"[WARN] Tensor pose read failed for {prim_path}; falling back to USD: {exc}")
        return _read_usd_world_pose(sim, prim_path)


def _xy_distance(a: list[float], b_xy: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b_xy[0]), float(a[1]) - float(b_xy[1]))


def _xy_distance_between(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _create_fixed_payload_joint(
    sim: SimulationContext,
    robot_body_path: str,
    local_pos0: tuple[float, float, float],
    box_path: str = "/World/CarryBox",
) -> str:
    robot_body = sim.stage.GetPrimAtPath(robot_body_path)
    box_body = sim.stage.GetPrimAtPath(box_path)
    if not robot_body.IsValid():
        raise RuntimeError(f"Robot payload body path not found: {robot_body_path}")
    if not box_body.IsValid():
        raise RuntimeError(f"Box body path not found: {box_path}")

    joint_path = f"{box_path}/FixedJointToRobot"
    joint = UsdPhysics.FixedJoint.Define(sim.stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(robot_body_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(box_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(local_pos0[0]), float(local_pos0[1]), float(local_pos0[2])))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint_path


def design_scene(args: argparse.Namespace) -> tuple[InteractiveScene, Optional[Articulation], RigidObject, str, str]:
    ground_cfg = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(8.0, 8.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.34, 0.34), roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
    )
    target_cfg = AssetBaseCfg(
        prim_path="/World/CarryTarget",
        spawn=sim_utils.CuboidCfg(
            size=(0.65, 0.45, 0.02),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 0.85), opacity=0.35),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.8, 0.0, 0.01)),
    )
    dome_light_cfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
    )
    key_light_cfg = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.8)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -2.0, 4.0)),
    )

    box_cfg = _make_box_cfg(args)
    box_cfg.prim_path = "{ENV_REGEX_NS}/CarryBox"

    if not args.skip_robot:
        from isaaclab_arena.embodiments.g1.g1 import G1_CFG

        g1_cfg = G1_CFG.replace(prim_path="/World/G1")
        g1_cfg.prim_path = "{ENV_REGEX_NS}/G1"
        g1_cfg.spawn.usd_path = str(args.g1_usd)
        g1_cfg.init_state.pos = (0.0, 0.0, 0.78)
        g1_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    else:
        g1_cfg = None

    @configclass
    class CarrySceneCfg(InteractiveSceneCfg):
        ground = ground_cfg
        target = target_cfg
        box = box_cfg
        dome_light = dome_light_cfg
        key_light = key_light_cfg
        if g1_cfg is not None:
            robot = g1_cfg

    scene = InteractiveScene(CarrySceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False))
    robot = scene["robot"] if not args.skip_robot else None
    box = scene["box"]
    robot_prim_path = "/World/envs/env_0/G1"
    box_prim_path = "/World/envs/env_0/CarryBox"
    return scene, robot, box, robot_prim_path, box_prim_path


def run_scene(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Optional[Articulation],
    box: RigidObject,
    args: argparse.Namespace,
    wbc_driver: Optional[G1WBCDriver],
    robot_prim_path: str,
    box_prim_path: str,
    payload_joint_path: str | None,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "minimal_carry_scene_state.csv"

    joint_pos = None
    root_pose_write_count_setup = 0
    root_velocity_write_count_setup = 0
    joint_state_write_count_setup = 0
    if robot is not None and not args.skip_explicit_state_reset:
        joint_pos = wp.to_torch(robot.data.default_joint_pos).clone()
        joint_vel = wp.to_torch(robot.data.default_joint_vel).clone()
        root_pose = wp.to_torch(robot.data.default_root_pose).clone()
        root_vel = wp.to_torch(robot.data.default_root_vel).clone()
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
        joint_state_write_count_setup += 2
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
        root_pose_write_count_setup += 1
        root_velocity_write_count_setup += 1
        robot.reset()
    elif robot is not None:
        joint_pos = wp.to_torch(robot.data.default_joint_pos).clone()
        print("[WARN] Skipping explicit robot state reset writes. Diagnostic mode only.")

    if not args.skip_explicit_state_reset:
        box.reset()

    sim_dt = sim.get_physics_dt()
    initial_box_pose = None
    initial_robot_pose = None
    joint_names = list(getattr(robot.data, "joint_names", [])) if robot is not None else []
    summary = {
        "scene_type": "minimal_g1_wbc_carry_smoke",
        "success_claim": "controller_backed_g1_wbc_smoke_not_free_box_carrying_success",
        "steps_requested": int(args.steps),
        "physics_dt": float(sim_dt),
        "robot_enabled": robot is not None,
        "articulated_carrier_enabled": robot is not None,
        "articulated_joint_count": len(joint_names),
        "robot_prim_path": robot_prim_path if robot is not None else None,
        "box_prim_path": box_prim_path,
        "wbc_mode": args.wbc_mode,
        "attach_box": args.attach_box,
        "payload_joint_path": payload_joint_path,
        "payload_joint_created": payload_joint_path is not None,
        "box_mass_kg": float(args.box_mass),
        "box_size_m": [float(value) for value in args.box_size],
        "root_pose_write_count_setup": root_pose_write_count_setup,
        "root_velocity_write_count_setup": root_velocity_write_count_setup,
        "joint_state_write_count_setup": joint_state_write_count_setup,
        "root_pose_write_count_rollout": 0,
        "root_velocity_write_count_rollout": 0,
        "box_pose_write_count_rollout": 0,
        "min_robot_base_z_m": None,
        "min_box_z_m": None,
        "max_robot_travel_xy_m": 0.0,
        "max_box_travel_xy_m": 0.0,
        "min_box_target_distance_xy_m": None,
        "fall_events": 0,
        "box_drop_events": 0,
        "completed_steps": 0,
    }

    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "time_s",
                "box_x",
                "box_y",
                "box_z",
                "box_qw",
                "box_qx",
                "box_qy",
                "box_qz",
                "robot_base_x",
                "robot_base_y",
                "robot_base_z",
                "robot_base_qw",
                "robot_base_qx",
                "robot_base_qy",
                "robot_base_qz",
                "robot_fall_flag",
                "box_drop_flag",
                "robot_travel_xy_m",
                "box_travel_xy_m",
                "box_target_distance_xy_m",
            ]
        )
        for step in range(args.steps):
            if robot is not None:
                if wbc_driver is not None:
                    joint_pos = wbc_driver.compute_joint_target()
                robot.set_joint_position_target_index(target=joint_pos)
            scene.write_data_to_sim()
            sim.step(render=args.render)
            scene.update(sim_dt)

            if step % 10 == 0 or step == args.steps - 1:
                box_pose_values = _read_asset_root_pose_values(box, sim, box_prim_path)
                if robot is not None:
                    robot_pose_values = _read_asset_root_pose_values(robot, sim, robot_prim_path)
                    robot_text = (
                        "robot_base="
                        f"({robot_pose_values[0]:.3f},{robot_pose_values[1]:.3f},{robot_pose_values[2]:.3f})"
                    )
                else:
                    robot_pose_values = [float("nan")] * 7
                    robot_text = "robot_base=(none)"
                if initial_box_pose is None:
                    initial_box_pose = list(box_pose_values)
                if robot is not None and initial_robot_pose is None:
                    initial_robot_pose = list(robot_pose_values)

                robot_fall_flag = int(robot is not None and robot_pose_values[2] < ROBOT_FALL_HEIGHT_M)
                box_drop_flag = int(box_pose_values[2] < BOX_DROP_HEIGHT_M)
                robot_travel_xy = (
                    _xy_distance_between(robot_pose_values, initial_robot_pose)
                    if robot is not None and initial_robot_pose is not None
                    else float("nan")
                )
                box_travel_xy = _xy_distance_between(box_pose_values, initial_box_pose)
                box_target_distance_xy = _xy_distance(box_pose_values, TARGET_POSITION_XY)

                summary["completed_steps"] = int(step + 1)
                summary["min_box_z_m"] = (
                    float(box_pose_values[2])
                    if summary["min_box_z_m"] is None
                    else min(float(summary["min_box_z_m"]), float(box_pose_values[2]))
                )
                summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), float(box_travel_xy))
                summary["min_box_target_distance_xy_m"] = (
                    float(box_target_distance_xy)
                    if summary["min_box_target_distance_xy_m"] is None
                    else min(float(summary["min_box_target_distance_xy_m"]), float(box_target_distance_xy))
                )
                summary["box_drop_events"] += int(box_drop_flag)
                if robot is not None:
                    summary["min_robot_base_z_m"] = (
                        float(robot_pose_values[2])
                        if summary["min_robot_base_z_m"] is None
                        else min(float(summary["min_robot_base_z_m"]), float(robot_pose_values[2]))
                    )
                    summary["max_robot_travel_xy_m"] = max(
                        float(summary["max_robot_travel_xy_m"]), float(robot_travel_xy)
                    )
                    summary["fall_events"] += int(robot_fall_flag)

                writer.writerow(
                    [
                        step,
                        step * sim_dt,
                        *box_pose_values,
                        *robot_pose_values,
                        robot_fall_flag,
                        box_drop_flag,
                        robot_travel_xy,
                        box_travel_xy,
                        box_target_distance_xy,
                    ]
                )
                print(
                    "[STATE] "
                    f"step={step} box_pos=({box_pose_values[0]:.3f},{box_pose_values[1]:.3f},{box_pose_values[2]:.3f}) "
                    f"{robot_text} robot_fall={robot_fall_flag} box_drop={box_drop_flag}"
                )

    summary_path = args.output_dir / "minimal_carry_scene_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    return metrics_path


def main() -> None:
    if not args_cli.skip_robot and not args_cli.g1_usd.is_file():
        raise FileNotFoundError(f"G1 USD not found: {args_cli.g1_usd}")
    if args_cli.skip_robot and args_cli.wbc_mode != "none":
        raise ValueError("--wbc-mode requires a robot; do not combine it with --skip-robot.")

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    with build_simulation_context(create_new_stage=True, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        if not args_cli.disable_usd_physics_updates:
            sim.set_setting("/physics/updateToUsd", True)
            sim.set_setting("/physics/updateVelocitiesToUsd", True)
        sim.set_camera_view(eye=[2.7, -2.2, 1.7], target=[0.8, 0.0, 0.7])

        scene, robot, box, robot_prim_path, box_prim_path = design_scene(args_cli)
        payload_joint_path = None
        if robot is not None and args_cli.attach_box == "fixed_torso":
            attach_body_path = args_cli.attach_body_path
            if attach_body_path.startswith("/World/G1/"):
                attach_body_path = attach_body_path.replace("/World/G1/", f"{robot_prim_path}/", 1)
            payload_joint_path = _create_fixed_payload_joint(
                sim, attach_body_path, tuple(args_cli.attach_local_pos0), box_path=box_prim_path
            )
        sim.reset()
        scene.reset()
        wbc_driver = None
        if robot is not None and args_cli.wbc_mode != "none":
            wbc_driver = G1WBCDriver(
                robot=robot,
                mode=args_cli.wbc_mode,
                walk_command=tuple(args_cli.walk_command),
                base_height=args_cli.base_height_command,
                wbc_asset_root=args_cli.wbc_asset_root,
            )
        print("[INFO] Minimal carry scene setup complete.")
        print(f"[INFO] Robot enabled: {not args_cli.skip_robot}")
        print(f"[INFO] WBC mode: {args_cli.wbc_mode}")
        print(f"[INFO] Payload joint: {payload_joint_path or 'none'}")
        print(f"[INFO] Skip explicit state reset: {args_cli.skip_explicit_state_reset}")
        print(f"[INFO] Box mass: {args_cli.box_mass} kg")
        print(f"[INFO] Box size: {tuple(args_cli.box_size)} m")
        metrics_path = run_scene(
            sim,
            scene,
            robot,
            box,
            args_cli,
            wbc_driver,
            robot_prim_path,
            box_prim_path,
            payload_joint_path,
        )
        print(f"[INFO] Metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
