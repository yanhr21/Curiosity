#!/usr/bin/env python3
"""Standalone Isaac Sim core-World dynamic quadruped carry diagnostics.

This is a control-path diagnostic for the direct Isaac carrying route.  It
avoids IsaacLab's SimulationContext and tensor APIs entirely, then uses
Isaac Sim core `World` plus `SingleArticulation.apply_action()` to verify that
custom-authored USD joints can be driven in this environment.

Success here means only: a physical articulated carrier task scaffold can
receive joint targets and produce measurable motion.  It is not yet learned
control, stable unassisted locomotion, or final humanoid carrying.
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
    parser.add_argument("--payload-mode", choices=("fixed_joint_to_torso", "staged_free_box"), default="fixed_joint_to_torso")
    parser.add_argument(
        "--staged-attach-mode",
        choices=("pose-lock", "velocity-servo", "contact-proxy", "fixed-joint"),
        default="pose-lock",
    )
    parser.add_argument("--box-x", type=float, default=0.30)
    parser.add_argument("--attach-after-step", type=int, default=90)
    parser.add_argument("--probe-speed", type=float, default=0.035)
    parser.add_argument("--target-x", type=float, default=0.8)
    parser.add_argument("--target-speed", type=float, default=0.24)
    parser.add_argument("--target-hold-radius", type=float, default=0.025)
    parser.add_argument("--target-body-margin", type=float, default=0.03)
    parser.add_argument("--min-hold-torso-travel", type=float, default=0.0)
    parser.add_argument("--carry-local-x", type=float, default=0.26)
    parser.add_argument("--carry-local-z", type=float, default=0.03)
    parser.add_argument("--contact-proxy-gain", type=float, default=14.0)
    parser.add_argument("--contact-proxy-max-speed", type=float, default=0.95)
    parser.add_argument("--base-velocity-assist", action="store_true")
    parser.add_argument("--base-assist-mode", choices=("velocity", "upright_velocity", "pose"), default="velocity")
    parser.add_argument("--base-x-gain", type=float, default=3.0)
    parser.add_argument("--base-max-x-speed", type=float, default=0.8)
    parser.add_argument("--base-x-command-scale", type=float, default=1.0)
    parser.add_argument("--base-lateral-gain", type=float, default=2.0)
    parser.add_argument("--base-height-gain", type=float, default=8.0)
    parser.add_argument("--base-max-z-speed", type=float, default=0.8)
    parser.add_argument("--base-upright-gain", type=float, default=8.0)
    parser.add_argument("--base-max-angular-speed", type=float, default=4.0)
    parser.add_argument("--base-post-step-velocity-assist", action="store_true")
    parser.add_argument("--support-drive", action="store_true")
    parser.add_argument("--support-drive-gain", type=float, default=3.0)
    parser.add_argument("--support-drive-max-speed", type=float, default=0.45)
    parser.add_argument("--support-pad-z", type=float, default=0.018)
    parser.add_argument("--torso-z", type=float, default=0.62)
    parser.add_argument("--stance-half-length", type=float, default=0.18)
    parser.add_argument("--stance-half-width", type=float, default=0.16)
    parser.add_argument("--foot-length", type=float, default=0.18)
    parser.add_argument("--foot-width", type=float, default=0.075)
    parser.add_argument("--foot-height", type=float, default=0.045)
    parser.add_argument("--static-friction", type=float, default=1.0)
    parser.add_argument("--dynamic-friction", type=float, default=0.8)
    parser.add_argument("--hip-stiffness", type=float, default=1800.0)
    parser.add_argument("--hip-damping", type=float, default=120.0)
    parser.add_argument("--hip-max-force", type=float, default=1100.0)
    parser.add_argument("--knee-stiffness", type=float, default=1500.0)
    parser.add_argument("--knee-damping", type=float, default=100.0)
    parser.add_argument("--knee-max-force", type=float, default=900.0)
    parser.add_argument("--gait-frequency", type=float, default=1.1)
    parser.add_argument("--hip-neutral-deg", type=float, default=-5.0)
    parser.add_argument("--knee-neutral-deg", type=float, default=-18.0)
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
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

print("[PROGRESS] Isaac core imports loaded", flush=True)
print(f"[CONFIG] base_x_command_scale={args_cli.base_x_command_scale}", flush=True)


TORSO_PATH = "/World/Robot/Torso"
BOX_PATH = "/World/CarryBox"
BOX_SIZE = (0.30, 0.22, 0.22)
PROXY_SPECS = {
    "left_palm": ("/World/GraspProxy/LeftPalm", (0.055, 0.040, 0.18), 18.0, (0.85, 0.70, 0.18)),
    "right_palm": ("/World/GraspProxy/RightPalm", (0.055, 0.040, 0.18), 18.0, (0.85, 0.70, 0.18)),
    "chest": ("/World/GraspProxy/ChestPad", (0.055, 0.24, 0.18), 24.0, (0.90, 0.42, 0.14)),
    "shelf": ("/World/GraspProxy/ForearmShelf", (0.38, 0.30, 0.035), 28.0, (0.25, 0.58, 0.36)),
    "front_stop": ("/World/GraspProxy/FrontStop", (0.035, 0.30, 0.22), 24.0, (0.55, 0.22, 0.70)),
}
SUPPORT_PAD_SPECS = {
    "fl": ("/World/SupportDrive/FLPad", 0.18, 0.16),
    "fr": ("/World/SupportDrive/FRPad", 0.18, -0.16),
    "rl": ("/World/SupportDrive/RLPad", -0.18, 0.16),
    "rr": ("/World/SupportDrive/RRPad", -0.18, -0.16),
}
LEG_PHASES = {
    "fl": 0.0,
    "fr": math.pi,
    "rl": math.pi,
    "rr": 0.0,
}


def _patch_core_api_simulation_manager_compat() -> None:
    if not hasattr(SimulationManager, "_backend"):
        SimulationManager._backend = "numpy"
    if not hasattr(SimulationManager, "get_backend"):
        SimulationManager.get_backend = classmethod(lambda cls: getattr(cls, "_backend", "numpy"))
    if not hasattr(SimulationManager, "_get_backend_utils"):
        def _get_backend_utils(cls):
            backend = getattr(cls, "_backend", "numpy")
            if backend == "numpy":
                import isaacsim.core.utils.numpy as np_utils

                return np_utils
            if backend == "torch":
                import isaacsim.core.utils.torch as torch_utils

                return torch_utils
            if backend == "warp":
                import isaacsim.core.utils.warp as warp_utils

                return warp_utils
            raise RuntimeError(f"Unsupported backend for compatibility shim: {backend}")

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
    rigid: bool = True,
    kinematic: bool = False,
    physics_material: UsdShade.Material | None = None,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translation, size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_physics_material(cube.GetPrim(), physics_material)
    if rigid:
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        if kinematic:
            rigid_api.CreateKinematicEnabledAttr(True)
        mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
        mass_api.CreateMassAttr(float(mass))


def _fixed_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
) -> UsdPhysics.FixedJoint:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos1]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint


def _disable_fixed_joint(joint: UsdPhysics.FixedJoint) -> None:
    joint.CreateJointEnabledAttr().Set(False)


def _enable_fixed_joint(joint: UsdPhysics.FixedJoint, local_pos0: tuple[float, float, float]) -> None:
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.GetJointEnabledAttr().Set(True)


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


def _runtime_pose_wxyz(prim: SingleArticulation | SingleRigidPrim) -> list[float]:
    pos, quat = prim.get_world_pose()
    return [
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]


def _quat_to_roll_pitch(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def design_scene(stage: Usd.Stage) -> UsdPhysics.FixedJoint | None:
    UsdGeom.Xform.Define(stage, "/World/Robot")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/Robot"))
    contact_material = _define_physics_material(
        stage,
        "/World/Looks/HighFrictionContactMaterial",
        float(args_cli.static_friction),
        float(args_cli.dynamic_friction),
    )

    _spawn_box_body(
        stage,
        "/World/Ground",
        (5.0, 2.6, 0.05),
        1.0,
        (0.31, 0.33, 0.33),
        (0.0, 0.0, -0.025),
        rigid=False,
        physics_material=contact_material,
    )
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
    torso_z = float(args_cli.torso_z)
    hip_z = torso_z - 0.08
    knee_z = hip_z - thigh_len
    foot_z = 0.06
    x_front = float(args_cli.stance_half_length)
    x_rear = -float(args_cli.stance_half_length)
    y_left = float(args_cli.stance_half_width)
    y_right = -float(args_cli.stance_half_width)

    _spawn_box_body(stage, TORSO_PATH, torso_size, 16.0, (0.14, 0.20, 0.30), (0.0, 0.0, torso_z))
    if args_cli.payload_mode == "fixed_joint_to_torso":
        _spawn_box_body(
            stage,
            BOX_PATH,
            BOX_SIZE,
            args_cli.payload_mass,
            (0.56, 0.42, 0.23),
            (0.26, 0.0, torso_z + 0.03),
            physics_material=contact_material,
        )
        attach_joint = _fixed_joint(stage, "/World/Robot/FixedPayloadJoint", TORSO_PATH, BOX_PATH, (0.26, 0.0, 0.03), (0.0, 0.0, 0.0))
    else:
        _spawn_box_body(
            stage,
            BOX_PATH,
            BOX_SIZE,
            args_cli.payload_mass,
            (0.56, 0.42, 0.23),
            (float(args_cli.box_x), 0.0, BOX_SIZE[2] * 0.5),
            physics_material=contact_material,
        )
        attach_joint = None
        if args_cli.staged_attach_mode == "contact-proxy":
            UsdGeom.Xform.Define(stage, "/World/GraspProxy")
            for idx, (_name, (path, size, mass, color)) in enumerate(PROXY_SPECS.items()):
                _spawn_box_body(
                    stage,
                    path,
                    size,
                    mass,
                    color,
                    (-0.85, -0.55 + 0.18 * idx, 0.22),
                    rigid=True,
                    physics_material=contact_material,
                )

    if args_cli.support_drive:
        UsdGeom.Xform.Define(stage, "/World/SupportDrive")
        for name, (path, foot_x, foot_y) in SUPPORT_PAD_SPECS.items():
            color = (0.18, 0.48, 0.88) if name in ("fl", "rr") else (0.16, 0.66, 0.42)
            _spawn_box_body(
                stage,
                path,
                (0.28, 0.115, 0.035),
                250.0,
                color,
                (foot_x + 0.03, foot_y, float(args_cli.support_pad_z)),
                rigid=True,
                kinematic=False,
                physics_material=contact_material,
            )

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
        _spawn_box_body(
            stage,
            foot,
            (float(args_cli.foot_length), float(args_cli.foot_width), float(args_cli.foot_height)),
            0.35,
            (0.06, 0.08, 0.09),
            (hip_x + 0.03, hip_y, foot_z),
            physics_material=contact_material,
        )
        _revolute_joint(
            stage,
            f"/World/Robot/{name}_hip_joint",
            TORSO_PATH,
            thigh,
            (hip_x, hip_y, hip_z - torso_z),
            (0.0, 0.0, thigh_len / 2.0),
            lower_deg=-38.0,
            upper_deg=38.0,
            stiffness=float(args_cli.hip_stiffness),
            damping=float(args_cli.hip_damping),
            max_force=float(args_cli.hip_max_force),
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
            stiffness=float(args_cli.knee_stiffness),
            damping=float(args_cli.knee_damping),
            max_force=float(args_cli.knee_max_force),
        )
        _fixed_joint(stage, f"/World/Robot/{name}_ankle_fixed_joint", shin, foot, (0.0, 0.0, -shin_len / 2.0), (-0.03, 0.0, 0.02))
    return attach_joint


def _joint_targets(t: float) -> dict[str, float]:
    targets = {}
    for leg in ("fl", "fr", "rl", "rr"):
        phase = LEG_PHASES[leg]
        s = math.sin(2.0 * math.pi * float(args_cli.gait_frequency) * t + phase)
        c = math.cos(2.0 * math.pi * float(args_cli.gait_frequency) * t + phase)
        hip_deg = float(args_cli.hip_neutral_deg) + float(args_cli.hip_amplitude_deg) * s
        knee_deg = float(args_cli.knee_neutral_deg) - float(args_cli.knee_amplitude_deg) * max(0.0, c)
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


def _carry_world_pose(torso: list[float]) -> list[float]:
    return [float(torso[0]) + float(args_cli.carry_local_x), float(torso[1]), float(torso[2]) + float(args_cli.carry_local_z)]


def _proxy_targets(box_center: list[float]) -> dict[str, np.ndarray]:
    bx, by, bz = [float(v) for v in box_center[:3]]
    return {
        "left_palm": np.array([bx, by + BOX_SIZE[1] * 0.5 + 0.035, bz], dtype=float),
        "right_palm": np.array([bx, by - BOX_SIZE[1] * 0.5 - 0.035, bz], dtype=float),
        "chest": np.array([bx - BOX_SIZE[0] * 0.5 - 0.035, by, bz], dtype=float),
        "shelf": np.array([bx, by, bz - BOX_SIZE[2] * 0.5 - 0.025], dtype=float),
        "front_stop": np.array([bx + BOX_SIZE[0] * 0.5 + 0.035, by, bz], dtype=float),
    }


def _proxy_standby_targets(torso: list[float]) -> dict[str, np.ndarray]:
    tx = float(torso[0]) - 0.65
    tz = float(torso[2])
    return {
        "left_palm": np.array([tx, 0.45, tz], dtype=float),
        "right_palm": np.array([tx, -0.45, tz], dtype=float),
        "chest": np.array([tx - 0.10, 0.0, tz], dtype=float),
        "shelf": np.array([tx, 0.0, 0.20], dtype=float),
        "front_stop": np.array([tx + 0.18, 0.0, tz], dtype=float),
    }


def _set_velocity_toward(prim: SingleRigidPrim, target: np.ndarray, feedforward_x: float, max_speed: float = 0.85) -> None:
    pos = np.array(_runtime_pose_wxyz(prim)[:3], dtype=float)
    vel = float(args_cli.contact_proxy_gain) * (target - pos) + np.array([float(feedforward_x), 0.0, 0.0], dtype=float)
    speed = float(np.linalg.norm(vel))
    if speed > max_speed:
        vel *= max_speed / speed
    prim.set_linear_velocity(vel)


def _place_contact_proxies(proxy_prims: dict[str, SingleRigidPrim], targets: dict[str, np.ndarray]) -> None:
    for name, prim in proxy_prims.items():
        prim.set_world_pose(position=targets[name], orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
        prim.set_linear_velocity(np.zeros(3, dtype=float))


def _proxy_gap(proxy_prims: dict[str, SingleRigidPrim], targets: dict[str, np.ndarray]) -> float | None:
    if not proxy_prims:
        return None
    gaps = []
    for name, prim in proxy_prims.items():
        pos = np.array(_runtime_pose_wxyz(prim)[:3], dtype=float)
        gaps.append(float(np.linalg.norm(pos - targets[name])))
    return max(gaps) if gaps else None


def _staged_phase(step: int, attached: bool, box_target_distance: float, target_hold_latched: bool = False) -> str:
    if args_cli.payload_mode != "staged_free_box":
        return "fixed_payload_carry"
    if attached and (target_hold_latched or box_target_distance <= float(args_cli.target_hold_radius)):
        return "target_hold"
    if attached:
        return "carry_to_target"
    if step >= int(args_cli.attach_after_step):
        return "staged_attach"
    if step >= max(0, int(args_cli.attach_after_step) - 12):
        return "staged_lift_settle"
    return "probe_free_box"


def _body_ready_for_target_hold(torso: list[float], initial_torso: list[float]) -> bool:
    target_body_x = float(args_cli.target_x) - float(args_cli.carry_local_x)
    torso_travel = math.hypot(float(torso[0]) - float(initial_torso[0]), float(torso[1]) - float(initial_torso[1]))
    return (
        float(torso[0]) >= target_body_x - float(args_cli.target_body_margin)
        and torso_travel >= float(args_cli.min_hold_torso_travel)
    )


def _body_gated_phase(
    step: int,
    attached: bool,
    box_target_distance: float,
    target_hold_latched: bool,
    torso: list[float],
    initial_torso: list[float],
) -> str:
    phase = _staged_phase(step, attached, box_target_distance, target_hold_latched)
    if phase == "target_hold" and not target_hold_latched and not _body_ready_for_target_hold(torso, initial_torso):
        return "carry_to_target"
    return phase


def _clip(value: float, limit: float) -> float:
    limit = abs(float(limit))
    if limit <= 0.0:
        return 0.0
    return max(-limit, min(limit, float(value)))


def _apply_velocity_assist(
    robot: SingleArticulation,
    torso: list[float],
    initial_torso: list[float],
    speed_x: float,
    summary: dict,
) -> None:
    if args_cli.base_assist_mode == "velocity":
        linear = np.array([float(speed_x), 0.0, 0.0], dtype=float)
        angular = None
    elif args_cli.base_assist_mode == "upright_velocity":
        roll, pitch = _quat_to_roll_pitch(float(torso[3]), float(torso[4]), float(torso[5]), float(torso[6]))
        lateral_speed = -float(args_cli.base_lateral_gain) * float(torso[1] - initial_torso[1])
        z_speed = _clip(float(args_cli.base_height_gain) * float(initial_torso[2] - torso[2]), float(args_cli.base_max_z_speed))
        roll_rate = _clip(-float(args_cli.base_upright_gain) * roll, float(args_cli.base_max_angular_speed))
        pitch_rate = _clip(-float(args_cli.base_upright_gain) * pitch, float(args_cli.base_max_angular_speed))
        linear = np.array([float(speed_x), lateral_speed, z_speed], dtype=float)
        angular = np.array([roll_rate, pitch_rate, 0.0], dtype=float)
    else:
        raise RuntimeError(f"Velocity assist called for unsupported base_assist_mode={args_cli.base_assist_mode}")
    linear[0] *= float(args_cli.base_x_command_scale)
    robot.set_linear_velocity(linear)
    if angular is not None:
        robot.set_angular_velocity(angular)
        summary["root_angular_velocity_write_count"] += 1
    summary["root_velocity_write_count"] += 1


def _base_speed_x_command(step: int, attached: bool, target_hold_latched: bool, torso: list[float], initial_torso: list[float]) -> float:
    speed_x = float(args_cli.target_speed)
    if args_cli.payload_mode == "staged_free_box":
        if not attached:
            return 0.0
        elif target_hold_latched:
            desired_body_x = float(args_cli.target_x) - float(args_cli.carry_local_x)
        else:
            elapsed = float(max(0, step + 1 - int(args_cli.attach_after_step))) * 0.005
            raw_x = float(initial_torso[0]) + float(args_cli.target_speed) * elapsed
            desired_body_x = min(raw_x, float(args_cli.target_x) - float(args_cli.carry_local_x))
        speed_x = _clip(
            float(args_cli.base_x_gain) * (desired_body_x - float(torso[0])),
            float(args_cli.base_max_x_speed),
        )
    return speed_x


def _support_drive_speed_x(step: int, attached: bool, target_hold_latched: bool, torso: list[float], initial_torso: list[float]) -> float:
    if not args_cli.support_drive:
        return 0.0
    if args_cli.payload_mode == "staged_free_box" and not attached:
        return 0.0
    if target_hold_latched:
        return 0.0
    if args_cli.payload_mode == "staged_free_box":
        elapsed = float(max(0, step + 1 - int(args_cli.attach_after_step))) * 0.005
        desired_x = min(
            float(initial_torso[0]) + float(args_cli.target_speed) * elapsed,
            float(args_cli.target_x) - float(args_cli.carry_local_x),
        )
    else:
        desired_x = float(initial_torso[0]) + float(args_cli.target_speed) * float(step + 1) * 0.005
    return _clip(float(args_cli.support_drive_gain) * (desired_x - float(torso[0])), float(args_cli.support_drive_max_speed))


def _drive_support_pads(
    support_pad_prims: dict[str, SingleRigidPrim],
    torso: list[float],
    drive_speed_x: float,
    summary: dict,
) -> None:
    if not support_pad_prims:
        return
    if not np.all(np.isfinite(np.array(torso[:3], dtype=float))) or not math.isfinite(float(drive_speed_x)):
        summary["nonfinite_support_drive_events"] += 1
        message = "nonfinite support-drive command skipped"
        if message not in summary["control_errors"]:
            summary["control_errors"].append(message)
        return
    for name, prim in support_pad_prims.items():
        _path, foot_x, foot_y = SUPPORT_PAD_SPECS[name]
        position = np.array([float(torso[0]) + foot_x + 0.03, foot_y, float(args_cli.support_pad_z)], dtype=float)
        prim.set_world_pose(position=position, orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
        prim.set_linear_velocity(np.array([float(drive_speed_x), 0.0, 0.0], dtype=float))
        summary["support_pad_pose_write_count"] += 1
        summary["support_pad_velocity_write_count"] += 1


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_dynamic_quadruped_carry_state.csv"
    summary_path = args_cli.output_dir / "core_world_dynamic_quadruped_carry_summary.json"

    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    print("[PROGRESS] Creating USD stage", flush=True)
    create_new_stage()
    stage = get_current_stage()
    print("[PROGRESS] Designing articulated scene", flush=True)
    attach_joint = design_scene(stage)

    print("[PROGRESS] Creating core World", flush=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    print("[PROGRESS] Creating SingleArticulation wrapper", flush=True)
    robot = SingleArticulation(prim_path="/World/Robot", name="core_world_quad")
    box_runtime = SingleRigidPrim(prim_path=BOX_PATH, name="core_world_carry_box")
    proxy_prims: dict[str, SingleRigidPrim] = {}
    if args_cli.payload_mode == "staged_free_box" and args_cli.staged_attach_mode == "contact-proxy":
        for name, (path, _size, _mass, _color) in PROXY_SPECS.items():
            proxy_prims[name] = SingleRigidPrim(prim_path=path, name=f"grasp_proxy_{name}")
    support_pad_prims: dict[str, SingleRigidPrim] = {}
    if args_cli.support_drive:
        for name, (path, _foot_x, _foot_y) in SUPPORT_PAD_SPECS.items():
            support_pad_prims[name] = SingleRigidPrim(prim_path=path, name=f"support_drive_{name}")
    print("[PROGRESS] Resetting World", flush=True)
    world.reset()
    print("[PROGRESS] World reset complete", flush=True)
    print("[PROGRESS] Initializing SingleArticulation", flush=True)
    robot.initialize()
    box_runtime.initialize()
    for proxy in proxy_prims.values():
        proxy.initialize()
    for pad in support_pad_prims.values():
        pad.initialize()
    print("[PROGRESS] SingleArticulation initialize complete", flush=True)

    dof_names = list(robot.dof_names)
    joint_indices = _find_joint_indices(dof_names)
    initial_joint_positions = np.array(robot.get_joint_positions(), dtype=float)
    initial_torso = _runtime_pose_wxyz(robot)
    initial_box = _runtime_pose_wxyz(box_runtime)
    attached = args_cli.payload_mode == "fixed_joint_to_torso"
    attach_step = 0 if attached else None
    attach_local_pos0 = (0.26, 0.0, 0.03) if attached else None
    probe_start_x = float(initial_box[0])
    target_hold_steps = 0
    target_hold_latched = False
    max_box_relative_error = None
    proxy_grip_gap = None
    max_proxy_grip_gap = None

    summary = {
        "scene_type": "standalone_isaac_core_world_dynamic_quadruped_carry",
        "success_claim": "articulated_task_scaffold_not_unassisted_locomotion_not_learned_policy",
        "uses_isaaclab_simulation_context": False,
        "payload_mode": str(args_cli.payload_mode),
        "staged_attach_mode": str(args_cli.staged_attach_mode) if args_cli.payload_mode == "staged_free_box" else "fixed_joint_to_torso",
        "base_velocity_assist": bool(args_cli.base_velocity_assist),
        "base_assist_mode": str(args_cli.base_assist_mode) if args_cli.base_velocity_assist else "none",
        "root_pose_write_enabled": bool(args_cli.base_velocity_assist and args_cli.base_assist_mode == "pose"),
        "root_velocity_write_enabled": bool(
            args_cli.base_velocity_assist and args_cli.base_assist_mode in ("velocity", "upright_velocity")
        ),
        "root_angular_velocity_write_enabled": bool(
            args_cli.base_velocity_assist and args_cli.base_assist_mode == "upright_velocity"
        ),
        "base_post_step_velocity_assist": bool(args_cli.base_post_step_velocity_assist),
        "support_drive_enabled": bool(args_cli.support_drive),
        "support_drive_claim": "diagnostic_contact_support_scaffold_not_final_locomotion_controller",
        "support_drive_gain": float(args_cli.support_drive_gain),
        "support_drive_max_speed": float(args_cli.support_drive_max_speed),
        "support_pad_z_m": float(args_cli.support_pad_z),
        "support_pad_pose_write_count": 0,
        "support_pad_velocity_write_count": 0,
        "root_pose_write_count": 0,
        "root_velocity_write_count": 0,
        "root_angular_velocity_write_count": 0,
        "base_lateral_gain": float(args_cli.base_lateral_gain),
        "base_x_gain": float(args_cli.base_x_gain),
        "base_max_x_speed": float(args_cli.base_max_x_speed),
        "base_x_command_scale": float(args_cli.base_x_command_scale),
        "base_height_gain": float(args_cli.base_height_gain),
        "base_max_z_speed": float(args_cli.base_max_z_speed),
        "base_upright_gain": float(args_cli.base_upright_gain),
        "base_max_angular_speed": float(args_cli.base_max_angular_speed),
        "target_speed_mps": float(args_cli.target_speed),
        "gait_frequency_hz": float(args_cli.gait_frequency),
        "hip_neutral_deg": float(args_cli.hip_neutral_deg),
        "knee_neutral_deg": float(args_cli.knee_neutral_deg),
        "hip_amplitude_deg": float(args_cli.hip_amplitude_deg),
        "knee_amplitude_deg": float(args_cli.knee_amplitude_deg),
        "target_hold_radius_m": float(args_cli.target_hold_radius),
        "target_body_margin_m": float(args_cli.target_body_margin),
        "min_hold_torso_travel_m": float(args_cli.min_hold_torso_travel),
        "target_body_x_m": float(args_cli.target_x) - float(args_cli.carry_local_x),
        "payload_mass_kg": float(args_cli.payload_mass),
        "torso_z_m": float(args_cli.torso_z),
        "stance_half_length_m": float(args_cli.stance_half_length),
        "stance_half_width_m": float(args_cli.stance_half_width),
        "foot_length_m": float(args_cli.foot_length),
        "foot_width_m": float(args_cli.foot_width),
        "foot_height_m": float(args_cli.foot_height),
        "static_friction": float(args_cli.static_friction),
        "dynamic_friction": float(args_cli.dynamic_friction),
        "hip_stiffness": float(args_cli.hip_stiffness),
        "hip_damping": float(args_cli.hip_damping),
        "hip_max_force": float(args_cli.hip_max_force),
        "knee_stiffness": float(args_cli.knee_stiffness),
        "knee_damping": float(args_cli.knee_damping),
        "knee_max_force": float(args_cli.knee_max_force),
        "carry_local_x_m": float(args_cli.carry_local_x),
        "carry_local_z_m": float(args_cli.carry_local_z),
        "contact_proxy_gain": float(args_cli.contact_proxy_gain),
        "contact_proxy_max_speed": float(args_cli.contact_proxy_max_speed),
        "device": args_cli.device,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "dof_names": dof_names,
        "joint_indices": joint_indices,
        "attached": bool(attached),
        "attach_step": attach_step,
        "attach_local_pos0_m": None if attach_local_pos0 is None else [float(v) for v in attach_local_pos0],
        "probe_displacement_x_m": 0.0,
        "target_hold_steps": 0,
        "target_hold_latched": bool(target_hold_latched),
        "target_hold_body_ready": False,
        "max_joint_motion_rad": 0.0,
        "fall_events": 0,
        "box_drop_events": 0,
        "box_relative_error_m_after_attach": None,
        "max_box_relative_error_m_after_attach": None,
        "contact_proxy_enabled": bool(proxy_prims),
        "contact_proxy_grip_gap_m": None,
        "max_contact_proxy_grip_gap_m": None,
        "max_torso_travel_xy_m": 0.0,
        "max_box_travel_xy_m": 0.0,
        "final_box_target_distance_xy_m": None,
        "min_torso_z_m": float(initial_torso[2]),
        "max_tilt_rad": 0.0,
        "nonfinite_joint_events": 0,
        "nonfinite_support_drive_events": 0,
        "control_errors": [],
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
                "phase",
                "attached",
                "box_relative_error_m_after_attach",
                "contact_proxy_grip_gap_m",
                "support_drive_speed_x",
                "probe_displacement_x_m",
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
            torso_pre = _runtime_pose_wxyz(robot)
            box_pre = _runtime_pose_wxyz(box_runtime)
            pre_target_distance = math.hypot(float(box_pre[0]) - args_cli.target_x, float(box_pre[1]))
            phase = _body_gated_phase(step, attached, pre_target_distance, target_hold_latched, torso_pre, initial_torso)
            if phase == "target_hold":
                target_hold_latched = True
            proxy_feedforward_x = 0.0 if target_hold_latched else float(args_cli.target_speed)
            if proxy_prims:
                if not attached:
                    for name, prim in proxy_prims.items():
                        _set_velocity_toward(prim, _proxy_standby_targets(torso_pre)[name], 0.0)
                else:
                    desired_proxy_box = _carry_world_pose(torso_pre)
                    proxy_targets = _proxy_targets(desired_proxy_box)
                    for name, prim in proxy_prims.items():
                        _set_velocity_toward(prim, proxy_targets[name], proxy_feedforward_x, max_speed=float(args_cli.contact_proxy_max_speed))
            support_drive_speed_x = _support_drive_speed_x(step, attached, target_hold_latched, torso_pre, initial_torso)
            _drive_support_pads(support_pad_prims, torso_pre, support_drive_speed_x, summary)
            if args_cli.payload_mode == "staged_free_box":
                if phase == "probe_free_box":
                    direction = 1.0 if (step // 18) % 2 == 0 else -0.35
                    box_runtime.set_linear_velocity(np.array([direction * float(args_cli.probe_speed), 0.0, 0.0], dtype=float))
                elif phase == "staged_lift_settle":
                    carry_pos = _carry_world_pose(torso_pre)
                    box_runtime.set_world_pose(position=np.array(carry_pos, dtype=float), orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
                    box_runtime.set_linear_velocity(np.zeros(3, dtype=float))
                    if proxy_prims:
                        _place_contact_proxies(proxy_prims, _proxy_targets(carry_pos))
                elif phase == "staged_attach":
                    carry_pos = _carry_world_pose(torso_pre)
                    box_runtime.set_world_pose(position=np.array(carry_pos, dtype=float), orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
                    box_runtime.set_linear_velocity(np.zeros(3, dtype=float))
                    if proxy_prims:
                        _place_contact_proxies(proxy_prims, _proxy_targets(carry_pos))
                    attach_local_pos0 = (
                        float(carry_pos[0] - torso_pre[0]),
                        float(carry_pos[1] - torso_pre[1]),
                        float(carry_pos[2] - torso_pre[2]),
                    )
                    if args_cli.staged_attach_mode == "fixed-joint":
                        if attach_joint is None:
                            attach_joint = _fixed_joint(stage, "/World/Robot/StagedFreeBoxJoint", TORSO_PATH, BOX_PATH, attach_local_pos0, (0.0, 0.0, 0.0))
                        else:
                            _enable_fixed_joint(attach_joint, attach_local_pos0)
                    elif attach_joint is not None:
                        _enable_fixed_joint(attach_joint, attach_local_pos0)
                    attached = True
                    attach_step = step
                    print(f"[EVENT] staged free-box attach step={step} mode={args_cli.staged_attach_mode}", flush=True)
            if args_cli.base_velocity_assist and args_cli.base_assist_mode in ("velocity", "upright_velocity"):
                try:
                    speed_x = _base_speed_x_command(step, attached, target_hold_latched, torso_pre, initial_torso)
                    _apply_velocity_assist(robot, torso_pre, initial_torso, speed_x, summary)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if message not in summary["control_errors"]:
                        summary["control_errors"].append(message)
            world.step(render=args_cli.render)
            if (
                args_cli.base_velocity_assist
                and args_cli.base_assist_mode in ("velocity", "upright_velocity")
                and args_cli.base_post_step_velocity_assist
            ):
                try:
                    torso_after_step = _runtime_pose_wxyz(robot)
                    speed_x = _base_speed_x_command(step, attached, target_hold_latched, torso_after_step, initial_torso)
                    _apply_velocity_assist(robot, torso_after_step, initial_torso, speed_x, summary)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if message not in summary["control_errors"]:
                        summary["control_errors"].append(message)
            if args_cli.base_velocity_assist and args_cli.base_assist_mode == "pose":
                try:
                    if args_cli.payload_mode == "fixed_joint_to_torso":
                        commanded_x = float(initial_torso[0]) + float(args_cli.target_speed) * float(step + 1) * 0.005
                    elif not attached:
                        commanded_x = float(initial_torso[0])
                    elif target_hold_latched:
                        commanded_x = float(torso_pre[0])
                    else:
                        raw_x = float(initial_torso[0]) + float(args_cli.target_speed) * float(max(0, step + 1 - int(args_cli.attach_after_step))) * 0.005
                        commanded_x = min(raw_x, float(args_cli.target_x) - float(args_cli.carry_local_x))
                    robot.set_world_pose(
                        position=np.array([commanded_x, float(initial_torso[1]), float(initial_torso[2])], dtype=float),
                        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                    )
                    robot.set_linear_velocity(np.zeros(3, dtype=float))
                    robot.set_angular_velocity(np.zeros(3, dtype=float))
                    summary["root_pose_write_count"] += 1
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if message not in summary["control_errors"]:
                        summary["control_errors"].append(message)
            if args_cli.payload_mode == "staged_free_box" and attached and args_cli.staged_attach_mode == "pose-lock":
                torso_lock = _runtime_pose_wxyz(robot)
                locked_box = _carry_world_pose(torso_lock)
                box_runtime.set_world_pose(position=np.array(locked_box, dtype=float), orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
                box_runtime.set_linear_velocity(np.zeros(3, dtype=float))
            elif args_cli.payload_mode == "staged_free_box" and attached and args_cli.staged_attach_mode == "velocity-servo":
                torso_servo = _runtime_pose_wxyz(robot)
                box_servo = _runtime_pose_wxyz(box_runtime)
                desired_box = np.array(_carry_world_pose(torso_servo), dtype=float)
                err = desired_box - np.array(box_servo[:3], dtype=float)
                servo_vel = np.array([float(args_cli.target_speed), 0.0, 0.0], dtype=float) + 10.0 * err
                speed = float(np.linalg.norm(servo_vel))
                if speed > 0.65:
                    servo_vel *= 0.65 / speed
                box_runtime.set_linear_velocity(servo_vel)
            elif args_cli.payload_mode == "staged_free_box" and attached and args_cli.staged_attach_mode == "contact-proxy":
                torso_proxy = _runtime_pose_wxyz(robot)
                desired_proxy_box = _carry_world_pose(torso_proxy)
                proxy_targets = _proxy_targets(desired_proxy_box)
                for name, prim in proxy_prims.items():
                    _set_velocity_toward(prim, proxy_targets[name], 0.0 if target_hold_latched else float(args_cli.target_speed), max_speed=float(args_cli.contact_proxy_max_speed))

            if step % 10 == 0 or step == args_cli.steps - 1:
                current_joint_positions = np.array(robot.get_joint_positions(), dtype=float)
                finite_joints = np.isfinite(current_joint_positions)
                if not bool(np.all(finite_joints)):
                    summary["nonfinite_joint_events"] += 1
                if bool(np.any(finite_joints)):
                    joint_delta = np.abs(current_joint_positions[finite_joints] - initial_joint_positions[finite_joints])
                    joint_motion = float(np.max(joint_delta))
                else:
                    joint_motion = float("nan")
                torso = _runtime_pose_wxyz(robot)
                box = _runtime_pose_wxyz(box_runtime)
                target_distance = math.hypot(box[0] - args_cli.target_x, box[1])
                phase = _body_gated_phase(step, attached, target_distance, target_hold_latched, torso, initial_torso)
                if phase == "target_hold":
                    target_hold_latched = True
                    target_hold_steps += 1
                roll, pitch = _quat_to_roll_pitch(torso[3], torso[4], torso[5], torso[6])
                tilt = math.hypot(roll, pitch)
                torso_travel = math.hypot(torso[0] - initial_torso[0], torso[1] - initial_torso[1])
                box_travel = math.hypot(box[0] - initial_box[0], box[1] - initial_box[1])
                rel_err = None
                if attached and attach_local_pos0 is not None:
                    desired_box = [
                        float(torso[0]) + float(attach_local_pos0[0]),
                        float(torso[1]) + float(attach_local_pos0[1]),
                        float(torso[2]) + float(attach_local_pos0[2]),
                    ]
                    rel_err = float(math.dist(desired_box, box[:3]))
                    max_box_relative_error = rel_err if max_box_relative_error is None else max(max_box_relative_error, rel_err)
                if proxy_prims:
                    proxy_targets_now = _proxy_targets(_carry_world_pose(torso))
                    proxy_grip_gap = _proxy_gap(proxy_prims, proxy_targets_now)
                    if attached:
                        max_proxy_grip_gap = proxy_grip_gap if max_proxy_grip_gap is None else max(max_proxy_grip_gap, proxy_grip_gap)
                fall = int(torso[2] < 0.34 or tilt > 0.85)
                box_drop = int(attached and box[2] < 0.20)
                summary["completed_steps"] = int(step + 1)
                summary["attached"] = bool(attached)
                summary["attach_step"] = attach_step
                summary["attach_local_pos0_m"] = None if attach_local_pos0 is None else [float(v) for v in attach_local_pos0]
                summary["probe_displacement_x_m"] = float(box[0] - probe_start_x)
                summary["target_hold_steps"] = int(target_hold_steps)
                summary["target_hold_latched"] = bool(target_hold_latched)
                summary["target_hold_body_ready"] = bool(_body_ready_for_target_hold(torso, initial_torso))
                summary["max_joint_motion_rad"] = max(float(summary["max_joint_motion_rad"]), joint_motion)
                summary["fall_events"] += fall
                summary["box_drop_events"] += box_drop
                summary["box_relative_error_m_after_attach"] = rel_err
                summary["max_box_relative_error_m_after_attach"] = max_box_relative_error
                summary["contact_proxy_grip_gap_m"] = proxy_grip_gap
                summary["max_contact_proxy_grip_gap_m"] = max_proxy_grip_gap
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
                        phase,
                        int(attached),
                        rel_err,
                        proxy_grip_gap,
                        support_drive_speed_x,
                        summary["probe_displacement_x_m"],
                        tilt,
                        fall,
                        box_drop,
                    ]
                )
                print(
                    "[STATE] "
                    f"step={step} phase={phase} attached={int(attached)} joint_motion={joint_motion:.4f} "
                    f"torso=({torso[0]:.3f},{torso[1]:.3f},{torso[2]:.3f}) "
                    f"travel={torso_travel:.3f} box_target={target_distance:.3f} "
                    f"tilt={tilt:.3f} fall={fall} drop={box_drop}",
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
