#!/usr/bin/env python3
"""Adaptive Isaac carrying scene with active probing and posture selection.

This is the direct scene-building path: no external policy, no retargeting, no
waiting for video models.  It is still a scaffold: the carrier is a kinematic
humanoid proxy and the carried box pose is scripted after a probing decision.
The value of this scene is the task structure, morphology/load decision logic,
and evidence logging that later dynamic policies must satisfy.
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
    parser = argparse.ArgumentParser(description="Isaac scaffold for active-probing adaptive box carrying.")
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--box-mass", type=float, default=8.0)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.58, 0.38, 0.36), metavar=("X", "Y", "Z"))
    parser.add_argument("--box-com-offset", type=float, nargs=3, default=(0.04, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--robot-height", type=float, default=1.45)
    parser.add_argument("--robot-mass", type=float, default=52.0)
    parser.add_argument("--arm-length", type=float, default=0.58)
    parser.add_argument("--max-payload", type=float, default=16.0)
    parser.add_argument("--target-x", type=float, default=2.15)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/adaptive_probe_carry_scene"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import build_simulation_context  # noqa: E402


BOX_START_X = 0.78


def _smooth01(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _phase(step: int, steps: int) -> str:
    if step < int(0.18 * steps):
        return "approach"
    if step < int(0.32 * steps):
        return "probe"
    if step < int(0.44 * steps):
        return "posture_adjust"
    if step < int(0.58 * steps):
        return "lift"
    return "carry"


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
    opacity: float = 1.0,
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
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75, opacity=opacity),
    )
    cfg.func(path, cfg, translation=translation)


def _estimate_load() -> dict[str, float]:
    # Deterministic probing proxy: micro-lift and lateral nudge infer mass and
    # COM bias from effort. This is a placeholder for future force/contact
    # feedback, not privileged policy input.
    mass = float(args_cli.box_mass)
    com_x = float(args_cli.box_com_offset[0])
    size_x, size_y, size_z = (float(v) for v in args_cli.box_size)
    micro_lift_force = mass * 9.81 * (1.0 + 0.015 * size_x)
    lateral_moment = mass * 9.81 * com_x
    width_penalty = max(0.0, size_y - 0.34) * 8.0
    return {
        "estimated_mass_kg": micro_lift_force / 9.81,
        "estimated_com_x_m": lateral_moment / max(1e-6, mass * 9.81),
        "estimated_grip_difficulty": width_penalty + 0.12 * mass + 0.35 * size_z,
        "probe_micro_lift_force_n": micro_lift_force,
        "probe_lateral_moment_nm": lateral_moment,
    }


def _select_strategy(belief: dict[str, float]) -> dict[str, float | str]:
    mass_ratio = belief["estimated_mass_kg"] / max(1e-6, float(args_cli.max_payload))
    arm_reach_ratio = float(args_cli.box_size[0]) / max(1e-6, float(args_cli.arm_length))
    com_bias = abs(belief["estimated_com_x_m"])
    if mass_ratio > 0.55 or arm_reach_ratio > 1.02:
        name = "chest_supported_slow"
        carry_height = 0.53 * args_cli.robot_height
        box_offset = 0.23
        stance_width = 0.19
        walk_speed = 0.18
    elif mass_ratio > 0.34 or com_bias > 0.035:
        name = "low_front_carry"
        carry_height = 0.45 * args_cli.robot_height
        box_offset = 0.34
        stance_width = 0.17
        walk_speed = 0.24
    else:
        name = "front_carry"
        carry_height = 0.58 * args_cli.robot_height
        box_offset = 0.42
        stance_width = 0.14
        walk_speed = 0.32
    return {
        "name": name,
        "carry_height_m": carry_height,
        "box_offset_m": box_offset,
        "stance_width_m": stance_width,
        "walk_speed_mps": walk_speed,
        "mass_ratio": mass_ratio,
        "arm_reach_ratio": arm_reach_ratio,
    }


def _support_margin(
    com_xy: tuple[float, float],
    left_foot: tuple[float, float],
    right_foot: tuple[float, float],
    stance_width: float,
) -> float:
    foot_half_x = 0.16
    foot_half_y = max(0.075, 0.52 * stance_width)
    min_x = min(left_foot[0], right_foot[0]) - foot_half_x
    max_x = max(left_foot[0], right_foot[0]) + foot_half_x
    min_y = min(left_foot[1], right_foot[1]) - foot_half_y
    max_y = max(left_foot[1], right_foot[1]) + foot_half_y
    return min(com_xy[0] - min_x, max_x - com_xy[0], com_xy[1] - min_y, max_y - com_xy[1])


def _robot_sizes(scale: float) -> dict[str, tuple[float, float, float]]:
    return {
        "pelvis": (0.32 * scale, 0.22 * scale, 0.16 * scale),
        "torso": (0.36 * scale, 0.25 * scale, 0.46 * scale),
        "head": (0.18 * scale, 0.16 * scale, 0.17 * scale),
        "left_upper_arm": (0.11 * scale, 0.11 * scale, 0.28 * scale),
        "right_upper_arm": (0.11 * scale, 0.11 * scale, 0.28 * scale),
        "left_forearm": (0.10 * scale, 0.10 * scale, 0.28 * scale),
        "right_forearm": (0.10 * scale, 0.10 * scale, 0.28 * scale),
        "left_thigh": (0.13 * scale, 0.13 * scale, 0.34 * scale),
        "right_thigh": (0.13 * scale, 0.13 * scale, 0.34 * scale),
        "left_shin": (0.12 * scale, 0.12 * scale, 0.36 * scale),
        "right_shin": (0.12 * scale, 0.12 * scale, 0.36 * scale),
        "left_foot": (0.28 * scale, 0.12 * scale, 0.06 * scale),
        "right_foot": (0.28 * scale, 0.12 * scale, 0.06 * scale),
    }


def design_scene() -> None:
    floor_cfg = sim_utils.CuboidCfg(
        size=(6.0, 3.6, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.33, 0.33), roughness=0.9),
    )
    floor_cfg.func("/World/Ground", floor_cfg, translation=(0.0, 0.0, -0.025))
    target_cfg = sim_utils.CuboidCfg(
        size=(0.7, 0.5, 0.025),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.40, 0.85), opacity=0.40),
    )
    target_cfg.func("/World/CarryTarget", target_cfg, translation=(args_cli.target_x, 0.0, 0.015))
    sim_utils.DomeLightCfg(intensity=2200.0, color=(0.8, 0.8, 0.8)).func("/World/DomeLight", sim_utils.DomeLightCfg())

    scale = args_cli.robot_height / 1.45
    colors = {
        "head": (0.16, 0.22, 0.30),
        "pelvis": (0.13, 0.18, 0.25),
        "torso": (0.14, 0.20, 0.30),
    }
    for name, size in _robot_sizes(scale).items():
        color = colors.get(name, (0.10, 0.17, 0.26))
        _spawn_cuboid(f"/World/Robot/{name}", size, color, translation=(0.0, 0.0, 0.5))

    _spawn_cuboid(
        "/World/CarryBox",
        tuple(float(v) for v in args_cli.box_size),
        (0.56, 0.42, 0.23),
        translation=(BOX_START_X, 0.0, float(args_cli.box_size[2]) * 0.5),
        collision=True,
        mass=float(args_cli.box_mass),
    )
    _spawn_cuboid(
        "/World/EstimatedCom",
        (0.05, 0.05, 0.05),
        (0.90, 0.12, 0.10),
        translation=(BOX_START_X + float(args_cli.box_com_offset[0]), 0.0, float(args_cli.box_size[2]) * 0.75),
        opacity=0.8,
    )


def _poses(
    step: int,
    steps: int,
    dt: float,
    strategy: dict[str, float | str],
) -> tuple[dict[str, tuple[float, float, float]], tuple[float, float, float]]:
    phase = _phase(step, steps)
    stance_width = float(strategy["stance_width_m"])
    carry_height = float(strategy["carry_height_m"])
    box_offset = float(strategy["box_offset_m"])
    carry_start = int(0.58 * steps)
    lift_start = int(0.44 * steps)
    adjust_start = int(0.32 * steps)
    approach_end = int(0.18 * steps)

    if step < carry_start:
        base_x = 0.08 + 0.38 * _smooth01(step / max(1, approach_end))
    else:
        carry_alpha = (step - carry_start) / max(1, steps - carry_start)
        base_x = 0.46 + (args_cli.target_x - box_offset - 0.02 - 0.46) * _smooth01(carry_alpha)
    t = step * dt
    gait_amp = 0.09 if phase == "carry" else 0.05
    stride = gait_amp * math.sin(2.0 * math.pi * 1.05 * t)
    base_z = 0.56 * args_cli.robot_height + 0.012 * math.sin(2.0 * math.pi * 2.1 * t)
    if phase == "posture_adjust":
        alpha = _smooth01((step - adjust_start) / max(1, lift_start - adjust_start))
        base_z -= 0.05 * alpha

    left_foot = (base_x + stride, stance_width, 0.035)
    right_foot = (base_x - stride, -stance_width, 0.035)
    pelvis = (base_x, 0.0, base_z - 0.20 * args_cli.robot_height)
    torso_forward = 0.03 if strategy["name"] == "chest_supported_slow" else 0.05
    torso = (base_x + torso_forward, 0.0, base_z + 0.05 * args_cli.robot_height)
    head = (torso[0] + 0.015, 0.0, torso[2] + 0.25 * args_cli.robot_height)

    arm_close = _smooth01((step - int(0.18 * steps)) / max(1, int(0.26 * steps)))
    arm_x = torso[0] + 0.16 + box_offset * arm_close
    arm_y = 0.22 - 0.11 * arm_close
    arm_z = torso[2] + 0.10 - 0.08 * arm_close
    if strategy["name"] == "low_front_carry":
        arm_z -= 0.08
    if strategy["name"] == "chest_supported_slow":
        arm_x -= 0.08
        arm_z += 0.03

    robot = {
        "pelvis": pelvis,
        "torso": torso,
        "head": head,
        "left_upper_arm": (torso[0] + 0.11, arm_y, torso[2] + 0.12),
        "right_upper_arm": (torso[0] + 0.11, -arm_y, torso[2] + 0.12),
        "left_forearm": (arm_x, arm_y * 0.58, arm_z),
        "right_forearm": (arm_x, -arm_y * 0.58, arm_z),
        "left_thigh": ((pelvis[0] + left_foot[0]) * 0.5, stance_width * 0.65, 0.32 * args_cli.robot_height),
        "right_thigh": ((pelvis[0] + right_foot[0]) * 0.5, -stance_width * 0.65, 0.32 * args_cli.robot_height),
        "left_shin": ((pelvis[0] + left_foot[0]) * 0.5, stance_width, 0.15 * args_cli.robot_height),
        "right_shin": ((pelvis[0] + right_foot[0]) * 0.5, -stance_width, 0.15 * args_cli.robot_height),
        "left_foot": left_foot,
        "right_foot": right_foot,
    }

    floor_z = float(args_cli.box_size[2]) * 0.5
    probe_start = int(0.18 * steps)
    if phase == "approach":
        box = (BOX_START_X, 0.0, floor_z)
    elif phase == "probe":
        alpha = _smooth01((step - probe_start) / max(1, int(0.14 * steps)))
        box = (BOX_START_X + 0.025 * alpha, 0.0, floor_z + 0.02 * math.sin(math.pi * alpha))
    elif phase == "posture_adjust":
        box = (BOX_START_X + 0.025, 0.0, floor_z)
    elif phase == "lift":
        alpha = _smooth01((step - lift_start) / max(1, carry_start - lift_start))
        hold = (torso[0] + box_offset, 0.0, carry_height)
        box = (
            (BOX_START_X + 0.025) * (1.0 - alpha) + hold[0] * alpha,
            0.0,
            floor_z * (1.0 - alpha) + hold[2] * alpha,
        )
    else:
        box = (torso[0] + box_offset, 0.0, carry_height + 0.008 * math.sin(2.0 * math.pi * 1.5 * t))
    return robot, box


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "adaptive_probe_carry_scene_state.csv"
    summary_path = args_cli.output_dir / "adaptive_probe_carry_scene_summary.json"
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    belief = _estimate_load()
    strategy = _select_strategy(belief)

    with build_simulation_context(create_new_stage=False, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        design_scene()
        sim.set_camera_view(eye=[3.0, -2.1, 1.55], target=[1.15, 0.0, 0.68])
        sim.reset()
        stage = sim.stage
        dt = sim.get_physics_dt()
        initial_box_pose = None
        initial_torso_pose = None
        summary = {
            "scene_type": "direct_isaac_active_probe_morphology_adaptive_carry_scaffold",
            "success_claim": "diagnostic_only_kinematic_proxy_not_dynamic_robot_or_learned_policy",
            "active_probing_scaffold": True,
            "video_conditioning_used": False,
            "box_pose_following": True,
            "steps_requested": int(args_cli.steps),
            "completed_steps": 0,
            "physics_dt": float(dt),
            "box_mass_kg": float(args_cli.box_mass),
            "box_size_m": [float(v) for v in args_cli.box_size],
            "box_com_offset_m": [float(v) for v in args_cli.box_com_offset],
            "robot_height_m": float(args_cli.robot_height),
            "robot_mass_kg": float(args_cli.robot_mass),
            "arm_length_m": float(args_cli.arm_length),
            "max_payload_kg": float(args_cli.max_payload),
            "belief": belief,
            "selected_strategy": strategy,
            "max_torso_travel_xy_m": 0.0,
            "max_box_travel_xy_m": 0.0,
            "min_support_margin_m": None,
            "max_effort_proxy": 0.0,
            "energy_proxy": 0.0,
            "box_drop_events": 0,
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
                    "strategy",
                    "torso_x",
                    "torso_y",
                    "torso_z",
                    "box_x",
                    "box_y",
                    "box_z",
                    "support_margin_m",
                    "effort_proxy",
                    "energy_proxy",
                    "box_drop_flag",
                    "box_target_distance_xy_m",
                ]
            )
            last_box_x = None
            for step in range(args_cli.steps):
                phase = _phase(step, args_cli.steps)
                robot, box = _poses(step, args_cli.steps, dt, strategy)
                for part, pos in robot.items():
                    _set_translate(stage, f"/World/Robot/{part}", pos)
                _set_translate(stage, "/World/CarryBox", box)
                _set_translate(
                    stage,
                    "/World/EstimatedCom",
                    (box[0] + float(args_cli.box_com_offset[0]), box[1] + float(args_cli.box_com_offset[1]), box[2]),
                )
                sim.step(render=args_cli.render)

                if step % 10 == 0 or step == args_cli.steps - 1:
                    torso_pose = _pose_wxyz(stage, "/World/Robot/torso")
                    box_pose = _pose_wxyz(stage, "/World/CarryBox")
                    if initial_torso_pose is None:
                        initial_torso_pose = list(torso_pose)
                    if initial_box_pose is None:
                        initial_box_pose = list(box_pose)
                    payload_mass = float(args_cli.box_mass) if phase in ("lift", "carry") else 0.0
                    total_mass = float(args_cli.robot_mass) + payload_mass
                    com_x = (float(args_cli.robot_mass) * torso_pose[0] + payload_mass * box_pose[0]) / total_mass
                    support_margin = _support_margin(
                        (com_x, torso_pose[1]),
                        (robot["left_foot"][0], robot["left_foot"][1]),
                        (robot["right_foot"][0], robot["right_foot"][1]),
                        float(strategy["stance_width_m"]),
                    )
                    box_vx_proxy = 0.0 if last_box_x is None else (box_pose[0] - last_box_x) / max(1e-6, 10.0 * dt)
                    last_box_x = box_pose[0]
                    effort_proxy = (
                        float(args_cli.box_mass) * 9.81 * max(0.0, box_pose[2] - float(args_cli.box_size[2]) * 0.5)
                        + 0.35 * float(args_cli.box_mass) * abs(box_vx_proxy)
                        + 18.0 / max(0.01, support_margin + 0.04)
                    )
                    summary["energy_proxy"] = float(summary["energy_proxy"]) + effort_proxy * 10.0 * dt
                    torso_travel = math.hypot(torso_pose[0] - initial_torso_pose[0], torso_pose[1] - initial_torso_pose[1])
                    box_travel = math.hypot(box_pose[0] - initial_box_pose[0], box_pose[1] - initial_box_pose[1])
                    box_target_distance = math.hypot(box_pose[0] - args_cli.target_x, box_pose[1])
                    drop_flag = int(phase == "carry" and box_pose[2] < 0.35)
                    summary["completed_steps"] = int(step + 1)
                    summary["max_torso_travel_xy_m"] = max(float(summary["max_torso_travel_xy_m"]), float(torso_travel))
                    summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), float(box_travel))
                    summary["min_support_margin_m"] = (
                        float(support_margin)
                        if summary["min_support_margin_m"] is None
                        else min(float(summary["min_support_margin_m"]), float(support_margin))
                    )
                    summary["max_effort_proxy"] = max(float(summary["max_effort_proxy"]), float(effort_proxy))
                    summary["box_drop_events"] += drop_flag
                    summary["final_phase"] = phase
                    summary["final_box_target_distance_xy_m"] = float(box_target_distance)
                    writer.writerow(
                        [
                            step,
                            step * dt,
                            phase,
                            strategy["name"],
                            torso_pose[0],
                            torso_pose[1],
                            torso_pose[2],
                            box_pose[0],
                            box_pose[1],
                            box_pose[2],
                            support_margin,
                            effort_proxy,
                            summary["energy_proxy"],
                            drop_flag,
                            box_target_distance,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} phase={phase} strategy={strategy['name']} "
                        f"box=({box_pose[0]:.3f},{box_pose[1]:.3f},{box_pose[2]:.3f}) "
                        f"margin={support_margin:.3f} effort={effort_proxy:.2f} drop={drop_flag}",
                        flush=True,
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
