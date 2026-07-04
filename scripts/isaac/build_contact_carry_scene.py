#!/usr/bin/env python3
"""Low-level Isaac contact-carry scene.

This avoids IsaacLab articulation tensors.  A kinematic robot proxy closes two
palms around a dynamic box, lifts by contact/friction, and walks forward.  The
box pose is not pose-followed.  This is still a diagnostic: the robot proxy is
kinematic, not a learned balancing robot.
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
    parser = argparse.ArgumentParser(description="Contact-based kinematic robot proxy carrying a dynamic box.")
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--box-mass", type=float, default=4.0)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.46, 0.30, 0.30), metavar=("X", "Y", "Z"))
    parser.add_argument("--target-x", type=float, default=1.55)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/contact_carry_scene"),
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


BOX_START_X = 0.62
BOX_DROP_Z = 0.12


def _smooth01(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _phase(step: int) -> str:
    if step < int(0.18 * args_cli.steps):
        return "approach"
    if step < int(0.34 * args_cli.steps):
        return "squeeze"
    if step < int(0.54 * args_cli.steps):
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
    kinematic: bool,
    mass: float | None = None,
    friction: float = 1.0,
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=kinematic,
            disable_gravity=kinematic,
            max_linear_velocity=8.0,
            max_angular_velocity=8.0,
            max_depenetration_velocity=4.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass) if mass is not None else None,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.75),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction * 0.85),
    )
    cfg.func(path, cfg, translation=translation)


def design_scene() -> None:
    floor_cfg = sim_utils.CuboidCfg(
        size=(5.0, 3.0, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.33, 0.33), roughness=0.9),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
    )
    floor_cfg.func("/World/Ground", floor_cfg, translation=(0.0, 0.0, -0.025))
    target_cfg = sim_utils.CuboidCfg(
        size=(0.55, 0.45, 0.02),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.40, 0.85), opacity=0.35),
    )
    target_cfg.func("/World/CarryTarget", target_cfg, translation=(args_cli.target_x, 0.0, 0.01))
    sim_utils.DomeLightCfg(intensity=2200.0, color=(0.8, 0.8, 0.8)).func("/World/DomeLight", sim_utils.DomeLightCfg())

    _spawn_cuboid(
        "/World/Robot/Torso",
        (0.34, 0.22, 0.55),
        (0.16, 0.22, 0.32),
        translation=(0.15, 0.0, 0.72),
        kinematic=True,
        mass=45.0,
        friction=1.4,
    )
    _spawn_cuboid(
        "/World/Robot/LeftPalm",
        (0.30, 0.08, 0.34),
        (0.10, 0.18, 0.28),
        translation=(BOX_START_X, 0.27, 0.33),
        kinematic=True,
        mass=3.0,
        friction=3.0,
    )
    _spawn_cuboid(
        "/World/Robot/RightPalm",
        (0.30, 0.08, 0.34),
        (0.10, 0.18, 0.28),
        translation=(BOX_START_X, -0.27, 0.33),
        kinematic=True,
        mass=3.0,
        friction=3.0,
    )

    box_z = float(args_cli.box_size[2]) * 0.5
    _spawn_cuboid(
        "/World/CarryBox",
        tuple(float(v) for v in args_cli.box_size),
        (0.56, 0.42, 0.23),
        translation=(BOX_START_X, 0.0, box_z),
        kinematic=False,
        mass=float(args_cli.box_mass),
        friction=1.6,
    )


def _control_pose(step: int) -> dict[str, tuple[float, float, float]]:
    phase = _phase(step)
    squeeze_start = int(0.18 * args_cli.steps)
    lift_start = int(0.34 * args_cli.steps)
    carry_start = int(0.54 * args_cli.steps)

    squeeze = _smooth01((step - squeeze_start) / max(1, lift_start - squeeze_start))
    lift = _smooth01((step - lift_start) / max(1, carry_start - lift_start))
    carry = _smooth01((step - carry_start) / max(1, args_cli.steps - carry_start))

    palm_gap = 0.27 * (1.0 - squeeze) + 0.17 * squeeze
    z = 0.33 * (1.0 - lift) + 0.70 * lift
    x = BOX_START_X + (args_cli.target_x - BOX_START_X) * carry
    torso_x = x - 0.34
    torso_z = 0.72 + 0.03 * math.sin(2.0 * math.pi * carry)
    if phase == "approach":
        torso_x = -0.05 + 0.39 * _smooth01(step / max(1, squeeze_start))
    return {
        "torso": (torso_x, 0.0, torso_z),
        "left_palm": (x, palm_gap, z),
        "right_palm": (x, -palm_gap, z),
    }


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "contact_carry_scene_state.csv"
    summary_path = args_cli.output_dir / "contact_carry_scene_summary.json"
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)

    with build_simulation_context(create_new_stage=False, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        sim.set_setting("/physics/updateToUsd", True)
        sim.set_setting("/physics/updateVelocitiesToUsd", True)
        design_scene()
        sim.set_camera_view(eye=[2.4, -1.9, 1.4], target=[0.8, 0.0, 0.45])
        sim.reset()
        stage = sim.stage
        initial_box_pose = None
        summary = {
            "scene_type": "low_level_isaac_contact_carry_kinematic_robot_dynamic_box",
            "success_claim": "diagnostic_only_not_dynamic_robot_balance_or_learned_carrying",
            "box_pose_followed": False,
            "steps_requested": int(args_cli.steps),
            "completed_steps": 0,
            "box_mass_kg": float(args_cli.box_mass),
            "box_size_m": [float(v) for v in args_cli.box_size],
            "max_box_travel_xy_m": 0.0,
            "max_box_lift_m": 0.0,
            "final_box_target_distance_xy_m": None,
            "box_drop_events": 0,
        }

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "time_s",
                    "phase",
                    "box_x",
                    "box_y",
                    "box_z",
                    "left_palm_y",
                    "right_palm_y",
                    "box_travel_xy_m",
                    "box_lift_m",
                    "box_target_distance_xy_m",
                    "box_drop_flag",
                ]
            )
            for step in range(args_cli.steps):
                targets = _control_pose(step)
                _set_translate(stage, "/World/Robot/Torso", targets["torso"])
                _set_translate(stage, "/World/Robot/LeftPalm", targets["left_palm"])
                _set_translate(stage, "/World/Robot/RightPalm", targets["right_palm"])
                sim.step(render=args_cli.render)
                if step % 10 == 0 or step == args_cli.steps - 1:
                    phase = _phase(step)
                    box_pose = _pose_wxyz(stage, "/World/CarryBox")
                    if initial_box_pose is None:
                        initial_box_pose = list(box_pose)
                    travel = math.hypot(box_pose[0] - initial_box_pose[0], box_pose[1] - initial_box_pose[1])
                    lift = box_pose[2] - initial_box_pose[2]
                    target_distance = math.hypot(box_pose[0] - args_cli.target_x, box_pose[1])
                    drop_flag = int(phase in ("lift", "carry") and box_pose[2] < BOX_DROP_Z)
                    summary["completed_steps"] = int(step + 1)
                    summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), float(travel))
                    summary["max_box_lift_m"] = max(float(summary["max_box_lift_m"]), float(lift))
                    summary["final_box_target_distance_xy_m"] = float(target_distance)
                    summary["box_drop_events"] += drop_flag
                    writer.writerow(
                        [
                            step,
                            step * sim.get_physics_dt(),
                            phase,
                            box_pose[0],
                            box_pose[1],
                            box_pose[2],
                            targets["left_palm"][1],
                            targets["right_palm"][1],
                            travel,
                            lift,
                            target_distance,
                            drop_flag,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} phase={phase} box=({box_pose[0]:.3f},{box_pose[1]:.3f},{box_pose[2]:.3f}) "
                        f"travel={travel:.3f} lift={lift:.3f} target_dist={target_distance:.3f} drop={drop_flag}",
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
