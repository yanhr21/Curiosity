#!/usr/bin/env python3
"""Standalone Isaac Sim core-World dynamic quadruped payload smoke.

This is a control-path diagnostic for the direct Isaac carrying route.  It
avoids IsaacLab's SimulationContext and tensor APIs entirely, then uses
Isaac Sim core `World` plus `SingleArticulation.apply_action()` to verify that
custom-authored USD joints can be driven in this environment.

Success here means only: a physical articulated carrier with a fixed payload
can receive joint targets and produce measurable joint motion.  It is not yet
unknown-object grasping, learned control, or stable humanoid carrying.
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
    parser = argparse.ArgumentParser(description="Standalone core-World quadruped payload smoke.")
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--payload-mass", type=float, default=4.0)
    parser.add_argument("--target-x", type=float, default=0.8)
    parser.add_argument("--gait-frequency", type=float, default=1.1)
    parser.add_argument("--hip-amplitude-deg", type=float, default=18.0)
    parser.add_argument("--knee-amplitude-deg", type=float, default=16.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/core_world_dynamic_quadruped_carry_scene"),
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
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

print("[PROGRESS] Isaac core imports loaded", flush=True)


TORSO_PATH = "/World/Robot/Torso"
BOX_PATH = "/World/CarryBox"
LEG_PHASES = {
    "fl": 0.0,
    "fr": math.pi,
    "rl": math.pi,
    "rr": 0.0,
}


def _set_xform(prim: Usd.Prim, translation: tuple[float, float, float], scale: tuple[float, float, float]) -> None:
    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(*[float(v) for v in translation]))
    xform_api.SetScale(Gf.Vec3f(*[float(v) for v in scale]))


def _spawn_box_body(
    stage: Usd.Stage,
    path: str,
    size: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    translation: tuple[float, float, float],
    *,
    rigid: bool = True,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translation, size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
        mass_api.CreateMassAttr(float(mass))


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


def _revolute_joint(
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
    stiffness: float = 1800.0,
    damping: float = 120.0,
    max_force: float = 1100.0,
) -> None:
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos1]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLowerLimitAttr().Set(float(lower_deg))
    joint.CreateUpperLimitAttr().Set(float(upper_deg))
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(stiffness))
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)


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


def design_scene(stage: Usd.Stage) -> None:
    UsdGeom.Xform.Define(stage, "/World/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/Robot"))

    _spawn_box_body(stage, "/World/Ground", (5.0, 2.6, 0.05), 1.0, (0.31, 0.33, 0.33), (0.0, 0.0, -0.025), rigid=False)
    _spawn_box_body(
        stage,
        "/World/CarryTarget",
        (0.42, 0.36, 0.02),
        1.0,
        (0.05, 0.40, 0.85),
        (args_cli.target_x, 0.0, 0.01),
        rigid=False,
    )

    torso_size = (0.44, 0.24, 0.16)
    thigh_len = 0.23
    shin_len = 0.26
    torso_z = 0.62
    hip_z = 0.54
    knee_z = hip_z - thigh_len
    foot_z = 0.06
    x_front = 0.18
    x_rear = -0.18
    y_left = 0.16
    y_right = -0.16

    _spawn_box_body(stage, TORSO_PATH, torso_size, 16.0, (0.14, 0.20, 0.30), (0.0, 0.0, torso_z))
    _spawn_box_body(stage, BOX_PATH, (0.30, 0.22, 0.22), args_cli.payload_mass, (0.56, 0.42, 0.23), (0.26, 0.0, torso_z + 0.03))
    _fixed_joint(stage, "/World/Robot/FixedPayloadJoint", TORSO_PATH, BOX_PATH, (0.26, 0.0, 0.03), (0.0, 0.0, 0.0))

    leg_specs = {
        "fl": (x_front, y_left),
        "fr": (x_front, y_right),
        "rl": (x_rear, y_left),
        "rr": (x_rear, y_right),
    }
    for name, (hip_x, hip_y) in leg_specs.items():
        thigh = f"/World/Robot/{name}_thigh"
        shin = f"/World/Robot/{name}_shin"
        foot = f"/World/Robot/{name}_foot"
        _spawn_box_body(stage, thigh, (0.065, 0.055, thigh_len), 0.95, (0.10, 0.17, 0.26), (hip_x, hip_y, hip_z - thigh_len / 2.0))
        _spawn_box_body(stage, shin, (0.055, 0.050, shin_len), 0.75, (0.10, 0.17, 0.26), (hip_x, hip_y, knee_z - shin_len / 2.0))
        _spawn_box_body(stage, foot, (0.18, 0.075, 0.045), 0.35, (0.06, 0.08, 0.09), (hip_x + 0.03, hip_y, foot_z))
        _revolute_joint(
            stage,
            f"/World/Robot/{name}_hip_joint",
            TORSO_PATH,
            thigh,
            (hip_x, hip_y, hip_z - torso_z),
            (0.0, 0.0, thigh_len / 2.0),
            lower_deg=-38.0,
            upper_deg=38.0,
        )
        _revolute_joint(
            stage,
            f"/World/Robot/{name}_knee_joint",
            thigh,
            shin,
            (0.0, 0.0, -thigh_len / 2.0),
            (0.0, 0.0, shin_len / 2.0),
            lower_deg=-62.0,
            upper_deg=20.0,
            stiffness=1500.0,
            damping=100.0,
            max_force=900.0,
        )
        _fixed_joint(stage, f"/World/Robot/{name}_ankle_fixed_joint", shin, foot, (0.0, 0.0, -shin_len / 2.0), (-0.03, 0.0, 0.02))


def _joint_targets(t: float) -> dict[str, float]:
    targets = {}
    for leg in ("fl", "fr", "rl", "rr"):
        phase = LEG_PHASES[leg]
        s = math.sin(2.0 * math.pi * float(args_cli.gait_frequency) * t + phase)
        c = math.cos(2.0 * math.pi * float(args_cli.gait_frequency) * t + phase)
        hip_deg = float(args_cli.hip_amplitude_deg) * s - 5.0
        knee_deg = -18.0 - float(args_cli.knee_amplitude_deg) * max(0.0, c)
        targets[f"{leg}_hip"] = math.radians(hip_deg)
        targets[f"{leg}_knee"] = math.radians(knee_deg)
    return targets


def _find_joint_indices(dof_names: list[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for wanted in (f"{leg}_{joint}" for leg in ("fl", "fr", "rl", "rr") for joint in ("hip", "knee")):
        for idx, dof_name in enumerate(dof_names):
            if wanted in dof_name:
                indices[wanted] = idx
                break
    missing = [name for name in (f"{leg}_{joint}" for leg in ("fl", "fr", "rl", "rr") for joint in ("hip", "knee")) if name not in indices]
    if missing:
        raise RuntimeError(f"Missing expected joints: {missing}; dof_names={dof_names}")
    return indices


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_dynamic_quadruped_carry_state.csv"
    summary_path = args_cli.output_dir / "core_world_dynamic_quadruped_carry_summary.json"

    print("[PROGRESS] Creating USD stage", flush=True)
    create_new_stage()
    stage = get_current_stage()
    print("[PROGRESS] Designing articulated scene", flush=True)
    design_scene(stage)

    print("[PROGRESS] Creating core World", flush=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    print("[PROGRESS] Creating SingleArticulation wrapper", flush=True)
    robot = SingleArticulation(prim_path="/World/Robot", name="core_world_quad")
    print("[PROGRESS] Resetting World", flush=True)
    world.reset()
    print("[PROGRESS] World reset complete", flush=True)
    print("[PROGRESS] Initializing SingleArticulation", flush=True)
    robot.initialize()
    print("[PROGRESS] SingleArticulation initialize complete", flush=True)

    dof_names = list(robot.dof_names)
    joint_indices = _find_joint_indices(dof_names)
    initial_joint_positions = np.array(robot.get_joint_positions(), dtype=float)
    initial_torso = _pose_wxyz(stage, TORSO_PATH)
    initial_box = _pose_wxyz(stage, BOX_PATH)

    summary = {
        "scene_type": "standalone_isaac_core_world_dynamic_quadruped_fixed_payload_carry",
        "success_claim": "control_path_diagnostic_not_unknown_box_grasp_or_learned_policy",
        "uses_isaaclab_simulation_context": False,
        "payload_mode": "fixed_joint_to_torso",
        "payload_mass_kg": float(args_cli.payload_mass),
        "device": args_cli.device,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "dof_names": dof_names,
        "joint_indices": joint_indices,
        "max_joint_motion_rad": 0.0,
        "fall_events": 0,
        "box_drop_events": 0,
        "max_torso_travel_xy_m": 0.0,
        "max_box_travel_xy_m": 0.0,
        "final_box_target_distance_xy_m": None,
        "min_torso_z_m": float(initial_torso[2]),
        "max_tilt_rad": 0.0,
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
                "max_joint_motion_rad",
                "torso_travel_xy_m",
                "box_travel_xy_m",
                "box_target_distance_xy_m",
                "tilt",
                "fall",
                "box_drop",
            ]
        )
        for step in range(args_cli.steps):
            t = step * 0.005
            joint_positions = [None] * int(robot.num_dof)
            for name, value in _joint_targets(t).items():
                joint_positions[joint_indices[name]] = value
            robot.apply_action(ArticulationAction(joint_positions=joint_positions))
            world.step(render=args_cli.render)

            if step % 10 == 0 or step == args_cli.steps - 1:
                current_joint_positions = np.array(robot.get_joint_positions(), dtype=float)
                joint_motion = float(np.nanmax(np.abs(current_joint_positions - initial_joint_positions)))
                torso = _pose_wxyz(stage, TORSO_PATH)
                box = _pose_wxyz(stage, BOX_PATH)
                roll, pitch = _quat_to_roll_pitch(torso[3], torso[4], torso[5], torso[6])
                tilt = math.hypot(roll, pitch)
                torso_travel = math.hypot(torso[0] - initial_torso[0], torso[1] - initial_torso[1])
                box_travel = math.hypot(box[0] - initial_box[0], box[1] - initial_box[1])
                target_distance = math.hypot(box[0] - args_cli.target_x, box[1])
                fall = int(torso[2] < 0.34 or tilt > 0.85)
                box_drop = int(box[2] < 0.20)
                summary["completed_steps"] = int(step + 1)
                summary["max_joint_motion_rad"] = max(float(summary["max_joint_motion_rad"]), joint_motion)
                summary["fall_events"] += fall
                summary["box_drop_events"] += box_drop
                summary["max_torso_travel_xy_m"] = max(float(summary["max_torso_travel_xy_m"]), float(torso_travel))
                summary["max_box_travel_xy_m"] = max(float(summary["max_box_travel_xy_m"]), float(box_travel))
                summary["final_box_target_distance_xy_m"] = float(target_distance)
                summary["min_torso_z_m"] = min(float(summary["min_torso_z_m"]), float(torso[2]))
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
                        joint_motion,
                        torso_travel,
                        box_travel,
                        target_distance,
                        tilt,
                        fall,
                        box_drop,
                    ]
                )
                print(
                    "[STATE] "
                    f"step={step} joint_motion={joint_motion:.4f} "
                    f"torso=({torso[0]:.3f},{torso[1]:.3f},{torso[2]:.3f}) "
                    f"travel={torso_travel:.3f} tilt={tilt:.3f} fall={fall} drop={box_drop}",
                    flush=True,
                )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


def main() -> None:
    run_scene()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
