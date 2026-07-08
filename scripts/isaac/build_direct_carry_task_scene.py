#!/usr/bin/env python3
"""Direct Isaac task scene for box carrying.

This builds a runnable Isaac/PhysX scene for the carrying task without waiting
for a humanoid policy or external video model.  It is a task-scene diagnostic:
the carrier is a kinematic humanoid proxy with explicit gait/hold phases.  It
does not claim learned balance, grasping, or autonomous posture selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Isaac carrying-task scene diagnostic.")
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--controller-mode", choices=("kinematic_proxy",), default="kinematic_proxy")
    parser.add_argument("--box-mass", type=float, default=6.0)
    parser.add_argument("--box-mass-min", type=float, default=None)
    parser.add_argument("--box-mass-max", type=float, default=None)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.55, 0.35, 0.35), metavar=("X", "Y", "Z"))
    parser.add_argument("--box-size-jitter", type=float, default=0.0)
    parser.add_argument("--box-seed", type=int, default=None)
    parser.add_argument("--walk-speed", type=float, default=0.32)
    parser.add_argument("--carry-height", type=float, default=0.84)
    parser.add_argument("--target-x", type=float, default=2.2)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/direct_carry_task_scene"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _apply_box_randomization(args: argparse.Namespace) -> None:
    """Apply explicit randomization only when requested by CLI flags."""
    rng = random.Random(args.box_seed)
    if args.box_mass_min is not None or args.box_mass_max is not None:
        if args.box_mass_min is None or args.box_mass_max is None:
            raise ValueError("--box-mass-min and --box-mass-max must be provided together.")
        if float(args.box_mass_min) <= 0.0 or float(args.box_mass_max) < float(args.box_mass_min):
            raise ValueError("Invalid box mass randomization range.")
        args.box_mass = rng.uniform(float(args.box_mass_min), float(args.box_mass_max))
    if float(args.box_size_jitter) > 0.0:
        jitter = float(args.box_size_jitter)
        args.box_size = tuple(max(0.05, float(v) * rng.uniform(1.0 - jitter, 1.0 + jitter)) for v in args.box_size)


_refuse_login_node()
args_cli = parse_args()
_apply_box_randomization(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import build_simulation_context  # noqa: E402


ROBOT_PARTS = {
    "pelvis": ((0.34, 0.24, 0.18), (0.12, 0.14, 0.20)),
    "torso": ((0.38, 0.26, 0.56), (0.18, 0.24, 0.34)),
    "head": ((0.20, 0.18, 0.18), (0.08, 0.10, 0.12)),
    "left_upper_arm": ((0.13, 0.13, 0.34), (0.12, 0.18, 0.28)),
    "right_upper_arm": ((0.13, 0.13, 0.34), (0.12, 0.18, 0.28)),
    "left_forearm": ((0.12, 0.12, 0.34), (0.14, 0.22, 0.30)),
    "right_forearm": ((0.12, 0.12, 0.34), (0.14, 0.22, 0.30)),
    "left_thigh": ((0.14, 0.14, 0.42), (0.10, 0.16, 0.22)),
    "right_thigh": ((0.14, 0.14, 0.42), (0.10, 0.16, 0.22)),
    "left_shin": ((0.13, 0.13, 0.42), (0.10, 0.16, 0.22)),
    "right_shin": ((0.13, 0.13, 0.42), (0.10, 0.16, 0.22)),
    "left_foot": ((0.30, 0.13, 0.07), (0.05, 0.07, 0.10)),
    "right_foot": ((0.30, 0.13, 0.07), (0.05, 0.07, 0.10)),
}


def _phase(step: int, steps: int) -> str:
    approach_end = int(0.22 * steps)
    probe_end = int(0.34 * steps)
    lift_end = int(0.44 * steps)
    if step < approach_end:
        return "approach"
    if step < probe_end:
        return "probe"
    if step < lift_end:
        return "lift"
    return "carry"


def _smooth01(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


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


def _spawn_cuboid(
    path: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    translation: tuple[float, float, float],
    collision: bool = False,
    mass: float | None = None,
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            max_linear_velocity=8.0,
            max_angular_velocity=8.0,
        )
        if mass is not None
        else None,
        mass_props=sim_utils.MassPropertiesCfg(mass=mass) if mass is not None else None,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75),
    )
    cfg.func(path, cfg, translation=translation)


def _support_margin(com_xy: tuple[float, float], left_foot: tuple[float, float], right_foot: tuple[float, float]) -> float:
    min_x = min(left_foot[0], right_foot[0]) - 0.18
    max_x = max(left_foot[0], right_foot[0]) + 0.18
    min_y = min(left_foot[1], right_foot[1]) - 0.09
    max_y = max(left_foot[1], right_foot[1]) + 0.09
    return min(com_xy[0] - min_x, max_x - com_xy[0], com_xy[1] - min_y, max_y - com_xy[1])


def design_scene() -> None:
    floor_cfg = sim_utils.CuboidCfg(
        size=(7.0, 4.0, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.33, 0.33), roughness=0.9),
    )
    floor_cfg.func("/World/Ground", floor_cfg, translation=(0.0, 0.0, -0.025))
    target_cfg = sim_utils.CuboidCfg(
        size=(0.7, 0.5, 0.025),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.40, 0.85), opacity=0.40),
    )
    target_cfg.func("/World/CarryTarget", target_cfg, translation=(args_cli.target_x, 0.0, 0.015))
    sim_utils.DomeLightCfg(intensity=2200.0, color=(0.8, 0.8, 0.8)).func("/World/DomeLight", sim_utils.DomeLightCfg())
    sim_utils.DistantLightCfg(intensity=2600.0, color=(0.9, 0.86, 0.78)).func(
        "/World/KeyLight", sim_utils.DistantLightCfg(), translation=(2.5, -2.5, 4.0)
    )

    for name, (size, color) in ROBOT_PARTS.items():
        _spawn_cuboid(f"/World/Robot/{name}", size, color, translation=(0.0, 0.0, 0.5), collision=False)

    box_cfg = sim_utils.CuboidCfg(
        size=tuple(args_cli.box_size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            max_linear_velocity=8.0,
            max_angular_velocity=8.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.box_mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.56, 0.42, 0.23), roughness=0.85),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.65),
    )
    box_cfg.func("/World/CarryBox", box_cfg, translation=(0.76, 0.0, args_cli.box_size[2] * 0.5))


def _robot_poses(step: int, steps: int, sim_dt: float) -> dict[str, tuple[float, float, float]]:
    phase = _phase(step, steps)
    t = step * sim_dt
    approach_end = int(0.22 * steps)
    carry_start = int(0.44 * steps)
    phase_s = 0.0 if steps <= 1 else step / max(1, steps - 1)

    if step < carry_start:
        base_x = 0.05 + 0.42 * _smooth01(step / max(1, approach_end))
    else:
        carry_alpha = (step - carry_start) / max(1, steps - carry_start)
        target_base_x = args_cli.target_x - 0.42
        base_x = 0.47 + (target_base_x - 0.47) * _smooth01(carry_alpha)

    base_z = 0.82 + 0.018 * math.sin(2.0 * math.pi * 2.0 * t)
    if phase == "lift":
        base_z -= 0.05 * math.sin(math.pi * (step - int(0.34 * steps)) / max(1, int(0.10 * steps)))
    stance_y = 0.12 if phase in ("probe", "lift", "carry") else 0.10
    stride = 0.12 * math.sin(2.0 * math.pi * 1.1 * t)
    left_foot = (base_x + stride, stance_y, 0.035)
    right_foot = (base_x - stride, -stance_y, 0.035)
    pelvis = (base_x, 0.0, base_z - 0.27)
    torso = (base_x + (0.035 if phase in ("lift", "carry") else 0.0), 0.0, base_z + 0.05)
    head = (torso[0] + 0.02, 0.0, torso[2] + 0.38)
    left_thigh = ((pelvis[0] + left_foot[0]) * 0.5, stance_y * 0.65, 0.43)
    right_thigh = ((pelvis[0] + right_foot[0]) * 0.5, -stance_y * 0.65, 0.43)
    left_shin = ((left_thigh[0] + left_foot[0]) * 0.5, stance_y, 0.22)
    right_shin = ((right_thigh[0] + right_foot[0]) * 0.5, -stance_y, 0.22)

    reach = _smooth01((phase_s - 0.20) / 0.16)
    hold = _smooth01((phase_s - 0.34) / 0.10)
    arm_x = torso[0] + 0.14 + 0.30 * max(reach, hold)
    arm_z = torso[2] + 0.08 - 0.06 * reach + 0.05 * hold
    arm_y = 0.20 - 0.08 * hold
    return {
        "pelvis": pelvis,
        "torso": torso,
        "head": head,
        "left_upper_arm": (torso[0] + 0.12, arm_y, torso[2] + 0.14),
        "right_upper_arm": (torso[0] + 0.12, -arm_y, torso[2] + 0.14),
        "left_forearm": (arm_x, arm_y * 0.65, arm_z),
        "right_forearm": (arm_x, -arm_y * 0.65, arm_z),
        "left_thigh": left_thigh,
        "right_thigh": right_thigh,
        "left_shin": left_shin,
        "right_shin": right_shin,
        "left_foot": left_foot,
        "right_foot": right_foot,
    }


def _box_pose(step: int, steps: int, robot: dict[str, tuple[float, float, float]]) -> tuple[float, float, float]:
    phase = _phase(step, steps)
    floor_z = args_cli.box_size[2] * 0.5
    probe_start = int(0.22 * steps)
    lift_start = int(0.34 * steps)
    carry_start = int(0.44 * steps)
    if phase == "approach":
        return (0.76, 0.0, floor_z)
    if phase == "probe":
        probe_alpha = _smooth01((step - probe_start) / max(1, lift_start - probe_start))
        return (0.76 + 0.04 * probe_alpha, 0.0, floor_z + 0.035 * math.sin(math.pi * probe_alpha))
    lift_alpha = _smooth01((step - lift_start) / max(1, carry_start - lift_start))
    hold_x = robot["torso"][0] + 0.42
    hold_z = args_cli.carry_height
    if phase == "lift":
        return (0.80 * (1.0 - lift_alpha) + hold_x * lift_alpha, 0.0, floor_z * (1.0 - lift_alpha) + hold_z * lift_alpha)
    return (hold_x, 0.0, hold_z + 0.012 * math.sin(2.0 * math.pi * 1.8 * step / max(1, steps)))


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "direct_carry_task_scene_state.csv"
    summary_path = args_cli.output_dir / "direct_carry_task_scene_summary.json"

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    with build_simulation_context(create_new_stage=False, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        design_scene()
        sim.set_camera_view(eye=[3.0, -2.2, 1.65], target=[1.1, 0.0, 0.75])
        sim.reset()
        stage = sim.stage
        sim_dt = sim.get_physics_dt()
        initial_box_pose = None
        initial_torso_pose = None
        robot_proxy_pose_write_count = 0
        box_kinematic_pose_write_count = 0
        summary = {
            "scene_type": "direct_isaac_kinematic_humanoid_proxy_carry_task",
            "success_claim": "diagnostic_only_not_learned_balance_or_grasp_success",
            "controller_mode": str(args_cli.controller_mode),
            "controller_contract": {
                "purpose": "task_scene_and_metric_interface_not_robot_controller",
                "replaceable_controller_inputs": [
                    "phase",
                    "box_pose",
                    "target_pose",
                    "morphology_limits",
                    "estimated_load_belief",
                ],
                "expected_controller_outputs": [
                    "robot_joint_or_task_targets",
                    "contact_mode",
                    "probing_action",
                    "carry_posture_label",
                ],
                "non_success_reason": "kinematic_proxy_writes_robot_and_box_poses",
            },
            "steps_requested": int(args_cli.steps),
            "completed_steps": 0,
            "physics_dt": float(sim_dt),
            "box_mass_kg": float(args_cli.box_mass),
            "box_size_m": [float(v) for v in args_cli.box_size],
            "box_seed": args_cli.box_seed,
            "box_mass_randomization_range_kg": (
                [float(args_cli.box_mass_min), float(args_cli.box_mass_max)]
                if args_cli.box_mass_min is not None and args_cli.box_mass_max is not None
                else None
            ),
            "box_size_jitter_fraction": float(args_cli.box_size_jitter),
            "target_x_m": float(args_cli.target_x),
            "max_torso_travel_xy_m": 0.0,
            "max_box_travel_xy_m": 0.0,
            "min_support_margin_m": None,
            "box_drop_events": 0,
            "carry_phase_steps": max(0, int(args_cli.steps) - int(0.44 * int(args_cli.steps))),
            "kinematic_box_pose_following": True,
            "robot_proxy_pose_write_count": 0,
            "box_kinematic_pose_write_count": 0,
            "root_pose_write_count_rollout": 0,
            "root_velocity_write_count_rollout": 0,
            "box_dynamic_pose_write_count_rollout": 0,
            "final_phase": None,
            "final_box_target_distance_xy_m": None,
        }

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "time_s",
                    "phase",
                    "torso_x",
                    "torso_y",
                    "torso_z",
                    "box_x",
                    "box_y",
                    "box_z",
                    "support_margin_m",
                    "box_drop_flag",
                    "torso_travel_xy_m",
                    "box_travel_xy_m",
                    "box_target_distance_xy_m",
                ]
            )
            for step in range(args_cli.steps):
                phase = _phase(step, args_cli.steps)
                robot = _robot_poses(step, args_cli.steps, sim_dt)
                for part, pos in robot.items():
                    _set_translate(stage, f"/World/Robot/{part}", pos)
                    robot_proxy_pose_write_count += 1
                _set_translate(stage, "/World/CarryBox", _box_pose(step, args_cli.steps, robot))
                box_kinematic_pose_write_count += 1
                sim.step(render=args_cli.render)

                if step % 10 == 0 or step == args_cli.steps - 1:
                    torso_pose = _pose_wxyz(stage, "/World/Robot/torso")
                    box_pose = _pose_wxyz(stage, "/World/CarryBox")
                    if initial_torso_pose is None:
                        initial_torso_pose = list(torso_pose)
                    if initial_box_pose is None:
                        initial_box_pose = list(box_pose)
                    total_mass = 55.0 + (args_cli.box_mass if phase in ("lift", "carry") else 0.0)
                    com_x = (55.0 * torso_pose[0] + (args_cli.box_mass if phase in ("lift", "carry") else 0.0) * box_pose[0]) / total_mass
                    com_xy = (com_x, torso_pose[1])
                    support_margin = _support_margin(
                        com_xy,
                        (robot["left_foot"][0], robot["left_foot"][1]),
                        (robot["right_foot"][0], robot["right_foot"][1]),
                    )
                    torso_travel = math.hypot(torso_pose[0] - initial_torso_pose[0], torso_pose[1] - initial_torso_pose[1])
                    box_travel = math.hypot(box_pose[0] - initial_box_pose[0], box_pose[1] - initial_box_pose[1])
                    box_target_distance = math.hypot(box_pose[0] - args_cli.target_x, box_pose[1])
                    drop_flag = int(phase in ("lift", "carry") and box_pose[2] < 0.35)
                    summary["completed_steps"] = int(step + 1)
                    summary["robot_proxy_pose_write_count"] = int(robot_proxy_pose_write_count)
                    summary["box_kinematic_pose_write_count"] = int(box_kinematic_pose_write_count)
                    summary["max_torso_travel_xy_m"] = max(float(summary["max_torso_travel_xy_m"]), float(torso_travel))
                    summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), float(box_travel))
                    summary["min_support_margin_m"] = (
                        float(support_margin)
                        if summary["min_support_margin_m"] is None
                        else min(float(summary["min_support_margin_m"]), float(support_margin))
                    )
                    summary["box_drop_events"] += drop_flag
                    summary["final_phase"] = phase
                    summary["final_box_target_distance_xy_m"] = float(box_target_distance)
                    writer.writerow(
                        [
                            step,
                            step * sim_dt,
                            phase,
                            torso_pose[0],
                            torso_pose[1],
                            torso_pose[2],
                            box_pose[0],
                            box_pose[1],
                            box_pose[2],
                            support_margin,
                            drop_flag,
                            torso_travel,
                            box_travel,
                            box_target_distance,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} phase={phase} torso_x={torso_pose[0]:.3f} "
                        f"box=({box_pose[0]:.3f},{box_pose[1]:.3f},{box_pose[2]:.3f}) "
                        f"margin={support_margin:.3f} drop={drop_flag}"
                    )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    print(f"[INFO] Metrics written to: {csv_path}")
    return csv_path


def main() -> None:
    run_scene()


if __name__ == "__main__":
    main()
    simulation_app.close()
