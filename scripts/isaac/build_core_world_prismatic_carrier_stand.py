#!/usr/bin/env python3
"""No-root articulated prismatic-leg carrier stand diagnostic.

This is the first direct Isaac diagnostic for replacing the velocity-commanded
support-proxy body.  The carrier is a free articulated body with four driven
vertical prismatic leg joints and four physical feet in contact with the
ground.  A physical payload box is fixed to the torso.

Passing this script only means: the articulated carrier can stand with a fixed
payload without body root pose or velocity writes.  It is not walking yet, not
unknown free-box carrying, and not a learned policy.
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
    parser = argparse.ArgumentParser(description="No-root prismatic carrier stand diagnostic.")
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument(
        "--payload-mode",
        choices=("fixed_joint_to_torso", "top_contact_free_box", "tray_contact_free_box", "cradle_free_box"),
        default="fixed_joint_to_torso",
    )
    parser.add_argument("--payload-mass", type=float, default=6.0)
    parser.add_argument("--torso-mass", type=float, default=32.0)
    parser.add_argument("--torso-z", type=float, default=0.62)
    parser.add_argument("--torso-size", type=float, nargs=3, default=(0.56, 0.34, 0.16), metavar=("X", "Y", "Z"))
    parser.add_argument("--payload-size", type=float, nargs=3, default=(0.34, 0.24, 0.24), metavar=("X", "Y", "Z"))
    parser.add_argument("--payload-local-x", type=float, default=0.24)
    parser.add_argument("--payload-local-z", type=float, default=0.04)
    parser.add_argument("--tray-local-x", type=float, default=0.08)
    parser.add_argument("--tray-local-z", type=float, default=0.12)
    parser.add_argument("--tray-size", type=float, nargs=3, default=(0.58, 0.42, 0.035), metavar=("X", "Y", "Z"))
    parser.add_argument("--tray-rail-height", type=float, default=0.11)
    parser.add_argument("--tray-rail-thickness", type=float, default=0.035)
    parser.add_argument("--tray-mass", type=float, default=4.0)
    parser.add_argument("--enable-tray-lid", action="store_true")
    parser.add_argument("--tray-lid-clearance", type=float, default=0.025)
    parser.add_argument("--tray-lid-thickness", type=float, default=0.035)
    parser.add_argument("--tray-lid-mass", type=float, default=2.0)
    parser.add_argument("--cradle-clearance-x", type=float, default=0.025)
    parser.add_argument("--cradle-clearance-y", type=float, default=0.040)
    parser.add_argument("--cradle-wall-height", type=float, default=0.26)
    parser.add_argument("--cradle-wall-thickness", type=float, default=0.030)
    parser.add_argument("--cradle-part-mass", type=float, default=1.0)
    parser.add_argument(
        "--motion-mode",
        choices=(
            "stand",
            "creep",
            "stance_translate",
            "quasistatic_stance_transfer",
            "quasistatic_step_cycle",
            "gated_quasistatic_step_cycle",
            "prelift_quasistatic_step_cycle",
            "guarded_prelift_quasistatic_step_cycle",
            "sync_inchworm",
            "feedback_sync_inchworm",
            "rear_anchor_push",
            "rear_anchor_velocity_push",
            "rear_anchor_effort_push",
        ),
        default="stand",
    )
    parser.add_argument("--x-slide-velocity", type=float, default=0.03)
    parser.add_argument("--x-slide-effort", type=float, default=5000.0)
    parser.add_argument("--enable-horizontal-legs", action="store_true")
    parser.add_argument("--target-x", type=float, default=0.20)
    parser.add_argument(
        "--gait-drive-target-x",
        type=float,
        default=None,
        help=(
            "Optional internal gait drive distance. The reported task target "
            "remains --target-x; this only changes diagnostic step-cycle stride "
            "planning when reset losses require over-driving the scaffold."
        ),
    )
    parser.add_argument("--step-length", type=float, default=0.08)
    parser.add_argument("--step-height", type=float, default=0.09)
    parser.add_argument("--gait-period-steps", type=int, default=160)
    parser.add_argument("--swing-fraction", type=float, default=0.22)
    parser.add_argument("--sync-cycle-pause-fraction", type=float, default=0.0)
    parser.add_argument("--sync-inchworm-min-cycles", type=int, default=0)
    parser.add_argument("--sync-inchworm-stride-override", type=float, default=0.0)
    parser.add_argument("--feedback-tilt-hold-threshold", type=float, default=0.22)
    parser.add_argument("--feedback-payload-error-hold-threshold", type=float, default=0.12)
    parser.add_argument("--gated-step-max-travel-loss", type=float, default=0.015)
    parser.add_argument("--gated-step-recovery-phase", type=float, default=0.43)
    parser.add_argument(
        "--gated-step-loss-rebaseline-steps",
        type=int,
        default=0,
        help=(
            "If >0, accept the current post-settle travel as a new guarded-step "
            "peak after this many consecutive travel-loss recovery steps."
        ),
    )
    parser.add_argument("--prelift-reset-lift-fraction", type=float, default=0.30)
    parser.add_argument("--prelift-reset-lower-fraction", type=float, default=0.30)
    parser.add_argument(
        "--prelift-stance-overdrive",
        type=float,
        default=1.0,
        help=(
            "During prelift reset, multiply not-yet-reset stance-leg x targets "
            "to counter swing-leg return reaction. Diagnostic only."
        ),
    )
    parser.add_argument("--guarded-step-target-tolerance", type=float, default=0.018)
    parser.add_argument(
        "--guarded-stop-target-x",
        type=float,
        default=None,
        help=(
            "Optional target used only by guarded step modes to decide when to "
            "hold. This lets diagnostic gait overdrive differ from the real "
            "task target."
        ),
    )
    parser.add_argument("--enable-active-probe", action="store_true")
    parser.add_argument("--active-probe-steps", type=int, default=0)
    parser.add_argument("--active-probe-lift-amplitude", type=float, default=0.030)
    parser.add_argument("--active-probe-horizontal-amplitude", type=float, default=0.0)
    parser.add_argument("--enable-probe-adaptive-gait", action="store_true")
    parser.add_argument("--probe-adaptive-medium-risk-threshold", type=float, default=0.25)
    parser.add_argument("--probe-adaptive-high-risk-threshold", type=float, default=0.75)
    parser.add_argument("--probe-adaptive-medium-gait-drive-scale", type=float, default=0.95)
    parser.add_argument("--probe-adaptive-high-gait-drive-scale", type=float, default=0.85)
    parser.add_argument("--enable-probe-adaptive-posture", action="store_true")
    parser.add_argument("--probe-adaptive-medium-posture-leg-target-offset", type=float, default=0.012)
    parser.add_argument("--probe-adaptive-high-posture-leg-target-offset", type=float, default=0.024)
    parser.add_argument("--quasistatic-compensate-settle-drift", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--ramp-steps", type=int, default=1)
    parser.add_argument("--stance-half-length", type=float, default=0.30)
    parser.add_argument("--stance-half-width", type=float, default=0.24)
    parser.add_argument("--foot-length", type=float, default=0.34)
    parser.add_argument("--foot-width", type=float, default=0.18)
    parser.add_argument("--foot-height", type=float, default=0.055)
    parser.add_argument("--foot-mass", type=float, default=2.8)
    parser.add_argument("--foot-contact-z-threshold", type=float, default=0.050)
    parser.add_argument(
        "--enable-stance-foot-latch",
        action="store_true",
        help=(
            "Enable idealized world fixed joints for stance feet and disable "
            "them for swing feet. Diagnostic scaffold, not final walking."
        ),
    )
    parser.add_argument("--stance-foot-latch-lift-threshold", type=float, default=0.010)
    parser.add_argument("--leg-target", type=float, default=-0.50)
    parser.add_argument("--leg-lower", type=float, default=-0.75)
    parser.add_argument("--leg-upper", type=float, default=-0.25)
    parser.add_argument("--leg-stiffness", type=float, default=18000.0)
    parser.add_argument("--leg-damping", type=float, default=1800.0)
    parser.add_argument("--leg-max-force", type=float, default=25000.0)
    parser.add_argument("--enable-balance-leg-servo", action="store_true")
    parser.add_argument("--balance-roll-gain", type=float, default=0.0)
    parser.add_argument("--balance-pitch-gain", type=float, default=0.0)
    parser.add_argument("--balance-max-correction", type=float, default=0.05)
    parser.add_argument("--x-slide-limit", type=float, default=0.14)
    parser.add_argument("--x-slide-stiffness", type=float, default=9000.0)
    parser.add_argument("--x-slide-damping", type=float, default=900.0)
    parser.add_argument("--x-slide-max-force", type=float, default=12000.0)
    parser.add_argument(
        "--swing-x-force-scale",
        type=float,
        default=1.0,
        help=(
            "Scale x-slide max force for legs with commanded lift. Values below "
            "1.0 are a diagnostic for lower-reaction swing-foot reset."
        ),
    )
    parser.add_argument("--static-friction", type=float, default=3.0)
    parser.add_argument("--dynamic-friction", type=float, default=2.5)
    parser.add_argument("--front-foot-static-friction", type=float, default=None)
    parser.add_argument("--front-foot-dynamic-friction", type=float, default=None)
    parser.add_argument("--rear-foot-static-friction", type=float, default=None)
    parser.add_argument("--rear-foot-dynamic-friction", type=float, default=None)
    parser.add_argument("--fall-z", type=float, default=0.42)
    parser.add_argument("--drop-z", type=float, default=0.24)
    parser.add_argument("--max-stand-drift", type=float, default=0.08)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/core_world_prismatic_carrier_stand"),
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
LEG_NAMES = ("fl", "fr", "rl", "rr")


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
    rigid: bool = True,
    physics_material: UsdShade.Material | None = None,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translation, size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_physics_material(cube.GetPrim(), physics_material)
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


def _disabled_world_fixed_joint(stage: Usd.Stage, joint_path: str, body1: str) -> UsdPhysics.FixedJoint:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(False)
    return joint


def _set_world_fixed_joint_enabled(joint: UsdPhysics.FixedJoint, enabled: bool, world_pos: tuple[float, float, float]) -> None:
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in world_pos]))
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.GetJointEnabledAttr().Set(bool(enabled))


def _prismatic_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
    *,
    axis: str,
    lower: float,
    upper: float,
    target: float,
    stiffness: float,
    damping: float,
    max_force: float,
) -> None:
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos1]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLowerLimitAttr().Set(float(lower))
    joint.CreateUpperLimitAttr().Set(float(upper))
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(stiffness))
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))
    drive.CreateTargetPositionAttr().Set(float(target))
    drive.CreateTargetVelocityAttr().Set(0.0)


def _pose_wxyz(prim: SingleArticulation | SingleRigidPrim) -> list[float]:
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


def _leg_xy() -> dict[str, tuple[float, float]]:
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
    material = _define_physics_material(
        stage,
        "/World/Looks/HighFrictionStandMaterial",
        float(args_cli.static_friction),
        float(args_cli.dynamic_friction),
    )
    front_foot_material = _define_physics_material(
        stage,
        "/World/Looks/FrontFootMaterial",
        float(args_cli.front_foot_static_friction)
        if args_cli.front_foot_static_friction is not None
        else float(args_cli.static_friction),
        float(args_cli.front_foot_dynamic_friction)
        if args_cli.front_foot_dynamic_friction is not None
        else float(args_cli.dynamic_friction),
    )
    rear_foot_material = _define_physics_material(
        stage,
        "/World/Looks/RearFootMaterial",
        float(args_cli.rear_foot_static_friction)
        if args_cli.rear_foot_static_friction is not None
        else float(args_cli.static_friction),
        float(args_cli.rear_foot_dynamic_friction)
        if args_cli.rear_foot_dynamic_friction is not None
        else float(args_cli.dynamic_friction),
    )
    _spawn_box_body(stage, "/World/Ground", (5.0, 3.0, 0.05), 1.0, (0.31, 0.33, 0.33), (0.0, 0.0, -0.025), rigid=False, physics_material=material)
    torso_size = tuple(float(v) for v in args_cli.torso_size)
    payload_size = tuple(float(v) for v in args_cli.payload_size)
    _spawn_box_body(stage, TORSO_PATH, torso_size, float(args_cli.torso_mass), (0.12, 0.20, 0.30), (0.0, 0.0, float(args_cli.torso_z)), physics_material=material)
    if str(args_cli.payload_mode) in ("top_contact_free_box", "tray_contact_free_box"):
        support_top_z = float(args_cli.torso_z) + torso_size[2] * 0.5
        if str(args_cli.payload_mode) == "tray_contact_free_box":
            tray_size = tuple(float(v) for v in args_cli.tray_size)
            support_top_z = float(args_cli.torso_z) + float(args_cli.tray_local_z) + tray_size[2] * 0.5
        payload_pos = (
            float(args_cli.payload_local_x),
            0.0,
            support_top_z + payload_size[2] * 0.5 + 0.006,
        )
    elif str(args_cli.payload_mode) == "cradle_free_box":
        payload_pos = (
            float(args_cli.payload_local_x),
            0.0,
            float(args_cli.torso_z) + float(args_cli.payload_local_z),
        )
    else:
        payload_pos = (
            float(args_cli.payload_local_x),
            0.0,
            float(args_cli.torso_z) + float(args_cli.payload_local_z),
        )
    _spawn_box_body(stage, BOX_PATH, payload_size, float(args_cli.payload_mass), (0.58, 0.43, 0.24), payload_pos, physics_material=material)
    if str(args_cli.payload_mode) == "fixed_joint_to_torso":
        _fixed_joint(
            stage,
            "/World/Robot/FixedPayloadJoint",
            TORSO_PATH,
            BOX_PATH,
            (float(args_cli.payload_local_x), 0.0, float(args_cli.payload_local_z)),
            (0.0, 0.0, 0.0),
        )
    if str(args_cli.payload_mode) == "tray_contact_free_box":
        tray_size = tuple(float(v) for v in args_cli.tray_size)
        rail_h = float(args_cli.tray_rail_height)
        rail_t = float(args_cli.tray_rail_thickness)
        lid_t = float(args_cli.tray_lid_thickness)
        tray_x = float(args_cli.tray_local_x)
        tray_z = float(args_cli.tray_local_z)
        tray_specs = {
            "deck": ((tray_size[0], tray_size[1], tray_size[2]), (tray_x, 0.0, tray_z), (0.16, 0.34, 0.28), float(args_cli.tray_mass)),
            "left_rail": ((tray_size[0], rail_t, rail_h), (tray_x, 0.5 * tray_size[1] + 0.5 * rail_t, tray_z + 0.5 * rail_h), (0.12, 0.30, 0.46), 1.2),
            "right_rail": ((tray_size[0], rail_t, rail_h), (tray_x, -0.5 * tray_size[1] - 0.5 * rail_t, tray_z + 0.5 * rail_h), (0.12, 0.30, 0.46), 1.2),
            "front_stop": ((rail_t, tray_size[1] + 2.0 * rail_t, rail_h), (tray_x + 0.5 * tray_size[0] + 0.5 * rail_t, 0.0, tray_z + 0.5 * rail_h), (0.20, 0.26, 0.50), 1.2),
            "rear_stop": ((rail_t, tray_size[1] + 2.0 * rail_t, rail_h), (tray_x - 0.5 * tray_size[0] - 0.5 * rail_t, 0.0, tray_z + 0.5 * rail_h), (0.20, 0.26, 0.50), 1.2),
        }
        if bool(args_cli.enable_tray_lid):
            lid_local_z = tray_z + 0.5 * tray_size[2] + payload_size[2] + float(args_cli.tray_lid_clearance) + 0.5 * lid_t
            tray_specs["top_lid"] = (
                (tray_size[0], tray_size[1] + 2.0 * rail_t, lid_t),
                (tray_x, 0.0, lid_local_z),
                (0.22, 0.22, 0.26),
                float(args_cli.tray_lid_mass),
            )
        for name, (size, local_pos, color, mass) in tray_specs.items():
            path = f"/World/Robot/CarryTray_{name}"
            world_pos = (
                local_pos[0],
                local_pos[1],
                float(args_cli.torso_z) + local_pos[2],
            )
            _spawn_box_body(stage, path, size, mass, color, world_pos, physics_material=material)
            _fixed_joint(
                stage,
                f"/World/Robot/CarryTray_{name}_fixed_joint",
                TORSO_PATH,
                path,
                local_pos,
                (0.0, 0.0, 0.0),
            )
    if str(args_cli.payload_mode) == "cradle_free_box":
        gap_x = float(args_cli.cradle_clearance_x)
        gap_y = float(args_cli.cradle_clearance_y)
        wall_t = float(args_cli.cradle_wall_thickness)
        wall_h = float(args_cli.cradle_wall_height)
        deck_t = wall_t
        part_mass = float(args_cli.cradle_part_mass)
        payload_local_x = float(args_cli.payload_local_x)
        payload_local_z = float(args_cli.payload_local_z)
        deck_local_z = payload_local_z - 0.5 * payload_size[2] - 0.001 - 0.5 * deck_t
        wall_local_z = payload_local_z + 0.5 * wall_h - 0.5 * payload_size[2] - 0.001
        cradle_x = payload_size[0] + 2.0 * gap_x + 2.0 * wall_t
        cradle_y = payload_size[1] + 2.0 * gap_y + 2.0 * wall_t
        cradle_specs = {
            "deck": (
                (cradle_x, cradle_y, deck_t),
                (payload_local_x, 0.0, deck_local_z),
                (0.16, 0.28, 0.24),
            ),
            "rear_stop": (
                (wall_t, cradle_y, wall_h),
                (payload_local_x - 0.5 * payload_size[0] - gap_x - 0.5 * wall_t, 0.0, wall_local_z),
                (0.12, 0.30, 0.46),
            ),
            "front_stop": (
                (wall_t, cradle_y, wall_h),
                (payload_local_x + 0.5 * payload_size[0] + gap_x + 0.5 * wall_t, 0.0, wall_local_z),
                (0.20, 0.30, 0.42),
            ),
            "left_rail": (
                (cradle_x, wall_t, wall_h),
                (payload_local_x, 0.5 * payload_size[1] + gap_y + 0.5 * wall_t, wall_local_z),
                (0.18, 0.34, 0.30),
            ),
            "right_rail": (
                (cradle_x, wall_t, wall_h),
                (payload_local_x, -0.5 * payload_size[1] - gap_y - 0.5 * wall_t, wall_local_z),
                (0.18, 0.34, 0.30),
            ),
        }
        for name, (size, local_pos, color) in cradle_specs.items():
            path = f"/World/Robot/Cradle_{name}"
            world_pos = (
                local_pos[0],
                local_pos[1],
                float(args_cli.torso_z) + local_pos[2],
            )
            _spawn_box_body(stage, path, size, part_mass, color, world_pos, physics_material=material)
            scaled_local_pos0 = (
                local_pos[0] / max(torso_size[0], 1e-6),
                local_pos[1] / max(torso_size[1], 1e-6),
                local_pos[2] / max(torso_size[2], 1e-6),
            )
            _fixed_joint(
                stage,
                f"/World/Robot/Cradle_{name}_fixed_joint",
                TORSO_PATH,
                path,
                scaled_local_pos0,
                (0.0, 0.0, 0.0),
            )
    foot_z = float(args_cli.foot_height) * 0.5
    hip_z_local = -torso_size[2] * 0.5
    for name, (x, y) in _leg_xy().items():
        foot_material = front_foot_material if name.startswith("f") else rear_foot_material
        foot_path = f"/World/Robot/{name}_foot"
        parent_path = TORSO_PATH
        vertical_local_pos0 = (x, y, hip_z_local)
        if bool(args_cli.enable_horizontal_legs):
            carriage_path = f"/World/Robot/{name}_hip_slider"
            _spawn_box_body(
                stage,
                carriage_path,
                (0.055, 0.055, 0.055),
                0.45,
                (0.18, 0.22, 0.28),
                (x, y, float(args_cli.torso_z) + hip_z_local),
                physics_material=material,
            )
            _prismatic_joint(
                stage,
                f"/World/Robot/{name}_x_slide",
                TORSO_PATH,
                carriage_path,
                (x, y, hip_z_local),
                (0.0, 0.0, 0.0),
                axis="X",
                lower=-float(args_cli.x_slide_limit),
                upper=float(args_cli.x_slide_limit),
                target=0.0,
                stiffness=(
                    0.0
                    if str(args_cli.motion_mode) in ("rear_anchor_velocity_push", "rear_anchor_effort_push")
                    else float(args_cli.x_slide_stiffness)
                ),
                damping=0.0 if str(args_cli.motion_mode) == "rear_anchor_effort_push" else float(args_cli.x_slide_damping),
                max_force=float(args_cli.x_slide_max_force),
            )
            parent_path = carriage_path
            vertical_local_pos0 = (0.0, 0.0, 0.0)
        joint_path = f"/World/Robot/{name}_vertical_slide"
        _spawn_box_body(
            stage,
            foot_path,
            (float(args_cli.foot_length), float(args_cli.foot_width), float(args_cli.foot_height)),
            float(args_cli.foot_mass),
            (0.06, 0.08, 0.09),
            (x, y, foot_z),
            physics_material=foot_material,
        )
        _prismatic_joint(
            stage,
            joint_path,
            parent_path,
            foot_path,
            vertical_local_pos0,
            (0.0, 0.0, 0.0),
            axis="Z",
            lower=float(args_cli.leg_lower),
            upper=float(args_cli.leg_upper),
            target=float(args_cli.leg_target),
            stiffness=float(args_cli.leg_stiffness),
            damping=float(args_cli.leg_damping),
            max_force=float(args_cli.leg_max_force),
        )
        if bool(args_cli.enable_stance_foot_latch):
            _disabled_world_fixed_joint(stage, f"/World/Robot/{name}_stance_world_latch", foot_path)


def _find_leg_joint_indices(dof_names: list[str]) -> dict[str, dict[str, int]]:
    indices: dict[str, dict[str, int]] = {"z": {}, "x": {}}
    for leg in LEG_NAMES:
        for idx, dof_name in enumerate(dof_names):
            if leg in dof_name and "vertical_slide" in dof_name:
                indices["z"][leg] = idx
                break
        for idx, dof_name in enumerate(dof_names):
            if leg in dof_name and "x_slide" in dof_name:
                indices["x"][leg] = idx
                break
    missing = [leg for leg in LEG_NAMES if leg not in indices["z"]]
    if missing:
        raise RuntimeError(f"Missing prismatic leg joints: {missing}; dof_names={dof_names}")
    if bool(args_cli.enable_horizontal_legs):
        missing_x = [leg for leg in LEG_NAMES if leg not in indices["x"]]
        if missing_x:
            raise RuntimeError(f"Missing horizontal leg joints: {missing_x}; dof_names={dof_names}")
    return indices


def _leg_targets_for_step(
    step: int,
    joint_indices: dict[str, dict[str, int]],
    current_targets: np.ndarray,
    *,
    motion_step_override: int | None = None,
    target_x_override: float | None = None,
) -> np.ndarray:
    targets = np.array(current_targets, dtype=float)
    if str(args_cli.motion_mode) == "stand" or step < int(args_cli.settle_steps):
        for idx in joint_indices["z"].values():
            targets[idx] = float(args_cli.leg_target)
        for idx in joint_indices["x"].values():
            targets[idx] = 0.0
        return targets
    motion_step = (
        max(0, int(motion_step_override))
        if motion_step_override is not None
        else max(0, int(step) - int(args_cli.settle_steps))
    )
    ramp_scale = min(1.0, float(motion_step + 1) / float(max(int(args_cli.ramp_steps), 1)))
    if str(args_cli.motion_mode) == "stance_translate":
        target_x = float(args_cli.target_x) if target_x_override is None else float(target_x_override)
        x_target = max(-float(args_cli.x_slide_limit), min(float(args_cli.x_slide_limit), -target_x * ramp_scale))
        for idx in joint_indices["z"].values():
            targets[idx] = float(args_cli.leg_target)
        for idx in joint_indices["x"].values():
            targets[idx] = x_target
        return targets
    if str(args_cli.motion_mode) == "quasistatic_stance_transfer":
        if not bool(args_cli.enable_horizontal_legs):
            raise RuntimeError("quasistatic_stance_transfer requires --enable-horizontal-legs.")
        target_x = float(args_cli.target_x) if target_x_override is None else float(target_x_override)
        direction = 1.0 if target_x >= 0.0 else -1.0
        drive = min(abs(target_x), abs(float(args_cli.x_slide_limit)) * 0.90)
        x_target = -direction * drive * ramp_scale
        for leg in LEG_NAMES:
            targets[joint_indices["z"][leg]] = float(args_cli.leg_target)
            targets[joint_indices["x"][leg]] = x_target
        return targets
    if str(args_cli.motion_mode) in ("rear_anchor_push", "rear_anchor_velocity_push", "rear_anchor_effort_push"):
        if not bool(args_cli.enable_horizontal_legs):
            raise RuntimeError(f"{args_cli.motion_mode} requires --enable-horizontal-legs.")
        target_x = float(args_cli.target_x) if target_x_override is None else float(target_x_override)
        direction = 1.0 if target_x >= 0.0 else -1.0
        drive = min(abs(target_x), abs(float(args_cli.x_slide_limit)) * 0.90)
        rear_x_target = -direction * drive * ramp_scale
        front_lift = min(abs(float(args_cli.step_height)), max(0.0, float(args_cli.leg_upper) - float(args_cli.leg_target)))
        for leg in LEG_NAMES:
            if leg.startswith("r"):
                targets[joint_indices["z"][leg]] = float(args_cli.leg_target)
                targets[joint_indices["x"][leg]] = rear_x_target
            else:
                targets[joint_indices["z"][leg]] = float(args_cli.leg_target) + front_lift * ramp_scale
                targets[joint_indices["x"][leg]] = 0.0
        return targets
    if str(args_cli.motion_mode) in (
        "quasistatic_step_cycle",
        "gated_quasistatic_step_cycle",
        "prelift_quasistatic_step_cycle",
        "guarded_prelift_quasistatic_step_cycle",
    ):
        if not bool(args_cli.enable_horizontal_legs):
            raise RuntimeError(f"{args_cli.motion_mode} requires --enable-horizontal-legs.")
        target_x = float(args_cli.target_x) if target_x_override is None else float(target_x_override)
        direction = 1.0 if target_x >= 0.0 else -1.0
        target_abs = abs(target_x)
        max_stride = max(1.0e-4, min(abs(float(args_cli.step_length)), abs(float(args_cli.x_slide_limit)) * 0.70))
        cycle_count = max(1, int(math.ceil(target_abs / max_stride))) if target_abs > 0.0 else 1
        cycle_stride = target_abs / float(cycle_count) if target_abs > 0.0 else max_stride
        period = max(int(args_cli.gait_period_steps), 32)
        motion_step_clamped = min(motion_step, cycle_count * period - 1)
        cycle_step = motion_step_clamped % period
        phase = float(cycle_step) / float(period)
        drive_fraction = 0.45

        def _smooth01(value: float) -> float:
            value = max(0.0, min(1.0, value))
            return value * value * (3.0 - 2.0 * value)

        def _clamp_x(value: float) -> float:
            limit = abs(float(args_cli.x_slide_limit)) * 0.95
            return max(-limit, min(limit, value))

        stance_x = -direction * cycle_stride
        if phase < drive_fraction:
            x_target = _clamp_x(stance_x * _smooth01(phase / drive_fraction))
            for leg in LEG_NAMES:
                targets[joint_indices["z"][leg]] = float(args_cli.leg_target)
                targets[joint_indices["x"][leg]] = x_target
            return targets

        reset_order = ("fl", "rr", "fr", "rl")
        reset_phase = (phase - drive_fraction) / max(1.0e-6, 1.0 - drive_fraction)
        reset_window = 1.0 / float(len(reset_order))
        active_reset_leg = None
        for leg in reset_order:
            order_idx = reset_order.index(leg)
            leg_start = order_idx * reset_window
            leg_end = leg_start + reset_window
            if leg_start <= reset_phase < leg_end:
                active_reset_leg = leg
                break
        stance_overdrive = max(0.25, min(2.0, float(args_cli.prelift_stance_overdrive)))
        for leg in LEG_NAMES:
            order_idx = reset_order.index(leg)
            leg_start = order_idx * reset_window
            leg_end = leg_start + reset_window
            if reset_phase < leg_start:
                z_target = float(args_cli.leg_target)
                x_target = stance_x
                if (
                    active_reset_leg is not None
                    and str(args_cli.motion_mode)
                    in ("prelift_quasistatic_step_cycle", "guarded_prelift_quasistatic_step_cycle")
                ):
                    x_target *= stance_overdrive
                x_target = _clamp_x(x_target)
            elif reset_phase >= leg_end:
                z_target = float(args_cli.leg_target)
                x_target = 0.0
            else:
                swing = (reset_phase - leg_start) / reset_window
                if str(args_cli.motion_mode) in (
                    "prelift_quasistatic_step_cycle",
                    "guarded_prelift_quasistatic_step_cycle",
                ):
                    lift_fraction = max(0.05, min(0.80, float(args_cli.prelift_reset_lift_fraction)))
                    lower_fraction = max(0.05, min(0.80, float(args_cli.prelift_reset_lower_fraction)))
                    if lift_fraction + lower_fraction > 0.90:
                        scale = 0.90 / (lift_fraction + lower_fraction)
                        lift_fraction *= scale
                        lower_fraction *= scale
                    lower_start = 1.0 - lower_fraction
                    if swing < lift_fraction:
                        lift_progress = _smooth01(swing / lift_fraction)
                        z_target = float(args_cli.leg_target) + float(args_cli.step_height) * lift_progress
                        x_target = _clamp_x(stance_x)
                    elif swing < lower_start:
                        translate_progress = _smooth01((swing - lift_fraction) / max(1.0e-6, lower_start - lift_fraction))
                        z_target = float(args_cli.leg_target) + float(args_cli.step_height)
                        x_target = _clamp_x(stance_x * (1.0 - translate_progress))
                    else:
                        lower_progress = _smooth01((swing - lower_start) / lower_fraction)
                        z_target = float(args_cli.leg_target) + float(args_cli.step_height) * (1.0 - lower_progress)
                        x_target = 0.0
                else:
                    swing_smooth = _smooth01(swing)
                    z_target = float(args_cli.leg_target) + float(args_cli.step_height) * math.sin(math.pi * swing)
                    x_target = _clamp_x(stance_x * (1.0 - swing_smooth))
            targets[joint_indices["z"][leg]] = z_target
            targets[joint_indices["x"][leg]] = x_target
        return targets
    if str(args_cli.motion_mode) in ("sync_inchworm", "feedback_sync_inchworm"):
        if not bool(args_cli.enable_horizontal_legs):
            raise RuntimeError(f"{args_cli.motion_mode} requires --enable-horizontal-legs.")
        period = max(int(args_cli.gait_period_steps), 16)
        target_abs = abs(float(args_cli.target_x))
        direction = 1.0 if float(args_cli.target_x) >= 0.0 else -1.0
        max_stride = max(1.0e-4, min(abs(float(args_cli.step_length)), abs(float(args_cli.x_slide_limit)) * 0.90))
        min_cycles = max(0, int(args_cli.sync_inchworm_min_cycles))
        cycle_count = max(1, min_cycles, int(math.ceil(target_abs / max_stride))) if target_abs > 0.0 else max(1, min_cycles)
        stride_override = abs(float(args_cli.sync_inchworm_stride_override))
        cycle_stride = (
            min(stride_override, abs(float(args_cli.x_slide_limit)) * 0.90)
            if stride_override > 0.0
            else (target_abs / float(cycle_count) if target_abs > 0.0 else max_stride)
        )
        motion_step_clamped = min(motion_step, cycle_count * period - 1)
        cycle_step = motion_step_clamped % period
        phase = float(cycle_step) / float(period)
        pause_fraction = max(0.0, min(0.45, float(args_cli.sync_cycle_pause_fraction)))
        active_fraction = max(0.10, 1.0 - pause_fraction)
        if phase >= active_fraction:
            for leg in LEG_NAMES:
                targets[joint_indices["z"][leg]] = float(args_cli.leg_target)
                targets[joint_indices["x"][leg]] = 0.0
            return targets
        phase = phase / active_fraction
        propel_fraction = 0.55
        reset_order = ("fl", "rr", "fr", "rl")

        def _smooth01(value: float) -> float:
            value = max(0.0, min(1.0, value))
            return value * value * (3.0 - 2.0 * value)

        if phase < propel_fraction:
            progress = _smooth01(phase / propel_fraction)
            x_target_all = -direction * cycle_stride * progress * ramp_scale
            for leg in LEG_NAMES:
                targets[joint_indices["z"][leg]] = float(args_cli.leg_target)
                targets[joint_indices["x"][leg]] = x_target_all
            return targets

        reset_phase = (phase - propel_fraction) / max(1.0e-6, 1.0 - propel_fraction)
        reset_window = 1.0 / float(len(reset_order))
        for leg in LEG_NAMES:
            order_idx = reset_order.index(leg)
            leg_start = order_idx * reset_window
            leg_end = leg_start + reset_window
            if reset_phase < leg_start:
                x_target = -direction * cycle_stride * ramp_scale
                z_target = float(args_cli.leg_target)
            elif reset_phase >= leg_end:
                x_target = 0.0
                z_target = float(args_cli.leg_target)
            else:
                swing = (reset_phase - leg_start) / reset_window
                swing_smooth = _smooth01(swing)
                x_target = -direction * cycle_stride * (1.0 - swing_smooth) * ramp_scale
                z_target = float(args_cli.leg_target) + ramp_scale * float(args_cli.step_height) * math.sin(math.pi * swing)
            targets[joint_indices["z"][leg]] = z_target
            targets[joint_indices["x"][leg]] = x_target
        return targets
    period = max(int(args_cli.gait_period_steps), 4)
    swing_fraction = max(0.05, min(0.45, float(args_cli.swing_fraction)))
    offsets = {"fl": 0.00, "rr": 0.25, "fr": 0.50, "rl": 0.75}
    for leg in LEG_NAMES:
        phase = ((float(step) / float(period)) + offsets[leg]) % 1.0
        if phase < swing_fraction:
            swing = phase / swing_fraction
            x_target = ramp_scale * float(args_cli.step_length) * (swing - 0.5)
            z_target = float(args_cli.leg_target) + ramp_scale * float(args_cli.step_height) * math.sin(math.pi * swing)
        else:
            stance = (phase - swing_fraction) / (1.0 - swing_fraction)
            x_target = ramp_scale * float(args_cli.step_length) * (0.5 - stance)
            z_target = float(args_cli.leg_target)
        targets[joint_indices["z"][leg]] = z_target
        if leg in joint_indices["x"]:
            targets[joint_indices["x"][leg]] = x_target
    return targets


def _apply_balance_leg_servo(
    targets: np.ndarray,
    joint_indices: dict[str, dict[str, int]],
    *,
    roll: float,
    pitch: float,
) -> np.ndarray:
    if not bool(args_cli.enable_balance_leg_servo):
        return targets
    corrected = np.array(targets, dtype=float)
    max_corr = abs(float(args_cli.balance_max_correction))
    xy = _leg_xy()
    for leg, idx in joint_indices["z"].items():
        x, y = xy[leg]
        x_norm = x / max(abs(float(args_cli.stance_half_length)), 1e-6)
        y_norm = y / max(abs(float(args_cli.stance_half_width)), 1e-6)
        correction = float(args_cli.balance_pitch_gain) * float(pitch) * x_norm
        correction += float(args_cli.balance_roll_gain) * float(roll) * y_norm
        correction = max(-max_corr, min(max_corr, correction))
        corrected[idx] = max(float(args_cli.leg_lower), min(float(args_cli.leg_upper), corrected[idx] + correction))
    return corrected


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_prismatic_carrier_stand_state.csv"
    summary_path = args_cli.output_dir / "core_world_prismatic_carrier_stand_summary.json"

    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    create_new_stage()
    stage = get_current_stage()
    design_scene(stage)

    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    robot = SingleArticulation(prim_path="/World/Robot", name="prismatic_carrier")
    payload = SingleRigidPrim(prim_path=BOX_PATH, name="prismatic_payload")
    feet = {name: SingleRigidPrim(prim_path=f"/World/Robot/{name}_foot", name=f"prismatic_{name}_foot") for name in LEG_NAMES}
    cradle_rear_stop = None
    cradle_front_stop = None
    if str(args_cli.payload_mode) == "cradle_free_box":
        cradle_rear_stop = SingleRigidPrim(prim_path="/World/Robot/Cradle_rear_stop", name="prismatic_cradle_rear_stop")
        cradle_front_stop = SingleRigidPrim(prim_path="/World/Robot/Cradle_front_stop", name="prismatic_cradle_front_stop")
    world.reset()
    robot.initialize()
    payload.initialize()
    for foot in feet.values():
        foot.initialize()
    if cradle_rear_stop is not None and cradle_front_stop is not None:
        cradle_rear_stop.initialize()
        cradle_front_stop.initialize()

    dof_names = list(robot.dof_names)
    joint_indices = _find_leg_joint_indices(dof_names)
    initial_torso = _pose_wxyz(robot)
    initial_payload = _pose_wxyz(payload)
    initial_rear_stop = _pose_wxyz(cradle_rear_stop) if cradle_rear_stop is not None else None
    initial_front_stop = _pose_wxyz(cradle_front_stop) if cradle_front_stop is not None else None
    initial_joint_positions = np.array(robot.get_joint_positions(), dtype=float)
    per_leg_near_ground_steps = {leg: 0 for leg in LEG_NAMES}
    per_leg_min_foot_z = {leg: None for leg in LEG_NAMES}
    per_leg_max_foot_z = {leg: None for leg in LEG_NAMES}
    per_leg_max_commanded_lift = {leg: 0.0 for leg in LEG_NAMES}
    per_leg_max_abs_commanded_x = {leg: 0.0 for leg in LEG_NAMES}
    per_leg_max_actual_lift = {leg: 0.0 for leg in LEG_NAMES}
    per_leg_max_abs_actual_x = {leg: 0.0 for leg in LEG_NAMES}
    per_leg_swing_x_force_scaled_steps = {leg: 0 for leg in LEG_NAMES}
    x_drive_max_force_attrs = {}
    for leg in LEG_NAMES:
        joint_prim = stage.GetPrimAtPath(f"/World/Robot/{leg}_x_slide")
        if joint_prim and joint_prim.IsValid():
            attr = joint_prim.GetAttribute("drive:linear:physics:maxForce")
            if not attr or not attr.IsValid():
                attr = joint_prim.GetAttribute("drive:linear:maxForce")
            if attr and attr.IsValid():
                x_drive_max_force_attrs[leg] = attr
    stance_latch_joints = {}
    if bool(args_cli.enable_stance_foot_latch):
        for leg in LEG_NAMES:
            joint = UsdPhysics.FixedJoint.Get(stage, f"/World/Robot/{leg}_stance_world_latch")
            if joint:
                stance_latch_joints[leg] = joint
    per_leg_stance_latch_enabled_steps = {leg: 0 for leg in LEG_NAMES}
    per_leg_stance_latch_enable_count = {leg: 0 for leg in LEG_NAMES}
    per_leg_stance_latch_disable_count = {leg: 0 for leg in LEG_NAMES}
    per_leg_stance_latch_retarget_count = {leg: 0 for leg in LEG_NAMES}
    stance_latch_enabled = {leg: False for leg in LEG_NAMES}
    post_settle_torso_x: float | None = None
    post_settle_payload_x: float | None = None
    post_settle_payload_relative_xyz: tuple[float, float, float] | None = None
    feedback_motion_step = 0
    feedback_hold_steps = 0
    feedback_release_steps = 0
    feedback_last_safe = True
    feedback_last_block_reason = None
    gated_step_motion_step = 0
    gated_step_hold_steps = 0
    gated_step_release_steps = 0
    gated_step_recovery_steps = 0
    gated_step_last_safe = True
    gated_step_last_block_reason = None
    gated_step_peak_post_settle_payload_travel_x = 0.0
    gated_step_peak_post_settle_payload_progress = 0.0
    gated_step_peak_step = None
    gated_step_consecutive_loss_hold_steps = 0
    gated_step_loss_rebaseline_count = 0
    last_post_settle_payload_travel_x = 0.0
    active_probe_steps_requested = (
        max(0, int(args_cli.active_probe_steps)) if bool(args_cli.enable_active_probe) else 0
    )
    carry_start_step = int(args_cli.settle_steps) + int(active_probe_steps_requested)
    active_probe_baseline_torso_z: float | None = None
    active_probe_baseline_payload_z: float | None = None
    active_probe_baseline_payload_relative_xyz: tuple[float, float, float] | None = None
    active_probe_max_torso_z = None
    active_probe_max_payload_z = None
    active_probe_max_relative_offset_error = 0.0
    active_probe_max_tilt = 0.0
    active_probe_observed_steps = 0
    probe_adaptive_gait_decision_made = False
    probe_adaptive_gait_target_x_override: float | None = None
    probe_adaptive_posture_decision_made = False
    probe_adaptive_posture_leg_target_offset = 0.0
    quasistatic_target_x_override: float | None = (
        float(args_cli.gait_drive_target_x) if args_cli.gait_drive_target_x is not None else None
    )
    leg_targets = _leg_targets_for_step(0, joint_indices, np.array(robot.get_joint_positions(), dtype=float))
    sync_inchworm_max_stride = max(1.0e-4, min(abs(float(args_cli.step_length)), abs(float(args_cli.x_slide_limit)) * 0.90))
    sync_inchworm_cycle_count = (
        max(
            1,
            max(0, int(args_cli.sync_inchworm_min_cycles)),
            int(math.ceil(abs(float(args_cli.target_x)) / sync_inchworm_max_stride)),
        )
        if abs(float(args_cli.target_x)) > 0.0
        else max(1, max(0, int(args_cli.sync_inchworm_min_cycles)))
    )
    sync_inchworm_stride = abs(float(args_cli.target_x)) / float(sync_inchworm_cycle_count) if abs(float(args_cli.target_x)) > 0.0 else sync_inchworm_max_stride
    if float(args_cli.sync_inchworm_stride_override) > 0.0:
        sync_inchworm_stride = min(abs(float(args_cli.sync_inchworm_stride_override)), abs(float(args_cli.x_slide_limit)) * 0.90)

    summary = {
        "scene_type": f"core_world_prismatic_legged_carrier_{args_cli.payload_mode}",
        "success_claim": (
            "no_root_articulated_prismatic_legged_cradle_free_box_diagnostic_not_walking_or_learned_policy"
            if str(args_cli.payload_mode) == "cradle_free_box"
            else
            "no_root_articulated_free_tray_box_creep_diagnostic_not_grasping_or_learned_policy"
            if str(args_cli.payload_mode) == "tray_contact_free_box"
            else "no_root_articulated_free_top_contact_box_creep_diagnostic_not_grasping_or_learned_policy"
            if str(args_cli.payload_mode) == "top_contact_free_box"
            else (
                "no_root_articulated_fixed_payload_stand_diagnostic_only_not_walking_or_free_box_carrying"
                if str(args_cli.motion_mode) == "stand"
                else "no_root_articulated_fixed_payload_creep_diagnostic_not_free_box_carrying_or_learned_policy"
            )
        ),
        "articulated_carrier_enabled": True,
        "articulated_joint_count": int(robot.num_dof),
        "foot_contact_drive_enabled": True,
        "carrier_claim": "free_articulated_prismatic_leg_carrier_with_physical_feet_and_no_body_root_writes",
        "motion_mode": str(args_cli.motion_mode),
        "horizontal_legs_enabled": bool(args_cli.enable_horizontal_legs),
        "target_x_m": float(args_cli.target_x),
        "tray_local_x_m": float(args_cli.tray_local_x),
        "tray_local_z_m": float(args_cli.tray_local_z),
        "tray_size_m": [float(v) for v in args_cli.tray_size],
        "tray_rail_height_m": float(args_cli.tray_rail_height),
        "tray_rail_thickness_m": float(args_cli.tray_rail_thickness),
        "tray_mass_kg": float(args_cli.tray_mass),
        "tray_lid_enabled": bool(args_cli.enable_tray_lid),
        "tray_lid_clearance_m": float(args_cli.tray_lid_clearance),
        "tray_lid_thickness_m": float(args_cli.tray_lid_thickness),
        "tray_lid_mass_kg": float(args_cli.tray_lid_mass),
        "cradle_clearance_x_m": float(args_cli.cradle_clearance_x),
        "cradle_clearance_y_m": float(args_cli.cradle_clearance_y),
        "cradle_wall_height_m": float(args_cli.cradle_wall_height),
        "cradle_wall_thickness_m": float(args_cli.cradle_wall_thickness),
        "cradle_part_mass_kg": float(args_cli.cradle_part_mass),
        "cradle_initial_rear_stop_x_m": float(initial_rear_stop[0]) if initial_rear_stop is not None else None,
        "cradle_initial_front_stop_x_m": float(initial_front_stop[0]) if initial_front_stop is not None else None,
        "cradle_initial_rear_surface_gap_x_m": (
            (float(initial_payload[0]) - 0.5 * float(args_cli.payload_size[0]))
            - (float(initial_rear_stop[0]) + 0.5 * float(args_cli.cradle_wall_thickness))
            if initial_rear_stop is not None
            else None
        ),
        "cradle_initial_front_surface_gap_x_m": (
            (float(initial_front_stop[0]) - 0.5 * float(args_cli.cradle_wall_thickness))
            - (float(initial_payload[0]) + 0.5 * float(args_cli.payload_size[0]))
            if initial_front_stop is not None
            else None
        ),
        "step_length_m": float(args_cli.step_length),
        "step_height_m": float(args_cli.step_height),
        "gait_drive_target_x_m": (
            float(args_cli.gait_drive_target_x) if args_cli.gait_drive_target_x is not None else None
        ),
        "gait_period_steps": int(args_cli.gait_period_steps),
        "swing_fraction": float(args_cli.swing_fraction),
        "sync_cycle_pause_fraction": float(args_cli.sync_cycle_pause_fraction),
        "sync_inchworm_min_cycles": int(args_cli.sync_inchworm_min_cycles),
        "sync_inchworm_stride_override_m": float(args_cli.sync_inchworm_stride_override),
        "static_friction": float(args_cli.static_friction),
        "dynamic_friction": float(args_cli.dynamic_friction),
        "front_foot_static_friction": (
            float(args_cli.front_foot_static_friction)
            if args_cli.front_foot_static_friction is not None
            else float(args_cli.static_friction)
        ),
        "front_foot_dynamic_friction": (
            float(args_cli.front_foot_dynamic_friction)
            if args_cli.front_foot_dynamic_friction is not None
            else float(args_cli.dynamic_friction)
        ),
        "rear_foot_static_friction": (
            float(args_cli.rear_foot_static_friction)
            if args_cli.rear_foot_static_friction is not None
            else float(args_cli.static_friction)
        ),
        "rear_foot_dynamic_friction": (
            float(args_cli.rear_foot_dynamic_friction)
            if args_cli.rear_foot_dynamic_friction is not None
            else float(args_cli.dynamic_friction)
        ),
        "feedback_tilt_hold_threshold_rad": float(args_cli.feedback_tilt_hold_threshold),
        "feedback_payload_error_hold_threshold_m": float(args_cli.feedback_payload_error_hold_threshold),
        "gated_step_max_travel_loss_m": float(args_cli.gated_step_max_travel_loss),
        "gated_step_recovery_phase": float(args_cli.gated_step_recovery_phase),
        "gated_step_loss_rebaseline_steps": int(args_cli.gated_step_loss_rebaseline_steps),
        "gated_step_loss_rebaseline_count": 0,
        "prelift_reset_lift_fraction": float(args_cli.prelift_reset_lift_fraction),
        "prelift_reset_lower_fraction": float(args_cli.prelift_reset_lower_fraction),
        "prelift_stance_overdrive": float(args_cli.prelift_stance_overdrive),
        "guarded_step_target_tolerance_m": float(args_cli.guarded_step_target_tolerance),
        "quasistatic_compensate_settle_drift": bool(args_cli.quasistatic_compensate_settle_drift),
        "quasistatic_effective_target_x_m": None,
        "feedback_motion_step_final": 0,
        "feedback_hold_steps": 0,
        "feedback_release_steps": 0,
        "feedback_last_safe": None,
        "feedback_last_block_reason": None,
        "gated_step_motion_step_final": 0,
        "gated_step_hold_steps": 0,
        "gated_step_release_steps": 0,
        "gated_step_recovery_steps": 0,
        "gated_step_last_safe": None,
        "gated_step_last_block_reason": None,
        "gated_step_peak_post_settle_payload_travel_x_m": 0.0,
        "gated_step_peak_post_settle_payload_progress_m": 0.0,
        "gated_step_peak_step": None,
        "gated_step_travel_loss_after_peak_m": 0.0,
        "gated_step_directional_travel_loss_after_peak_m": 0.0,
        "sync_inchworm_cycle_count": int(sync_inchworm_cycle_count) if str(args_cli.motion_mode) in ("sync_inchworm", "feedback_sync_inchworm") else None,
        "sync_inchworm_stride_m": float(sync_inchworm_stride) if str(args_cli.motion_mode) in ("sync_inchworm", "feedback_sync_inchworm") else None,
        "settle_steps": int(args_cli.settle_steps),
        "ramp_steps": int(args_cli.ramp_steps),
        "payload_mode": str(args_cli.payload_mode),
        "attached": bool(str(args_cli.payload_mode) == "fixed_joint_to_torso"),
        "attach_step": 0 if str(args_cli.payload_mode) == "fixed_joint_to_torso" else None,
        "device": args_cli.device,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "dof_names": dof_names,
        "joint_indices": joint_indices,
        "foot_length_m": float(args_cli.foot_length),
        "foot_width_m": float(args_cli.foot_width),
        "foot_height_m": float(args_cli.foot_height),
        "foot_mass_kg": float(args_cli.foot_mass),
        "stance_half_length_m": float(args_cli.stance_half_length),
        "stance_half_width_m": float(args_cli.stance_half_width),
        "foot_contact_z_threshold_m": float(args_cli.foot_contact_z_threshold),
        "leg_target_m": float(args_cli.leg_target),
        "leg_lower_m": float(args_cli.leg_lower),
        "leg_upper_m": float(args_cli.leg_upper),
        "leg_stiffness": float(args_cli.leg_stiffness),
        "leg_damping": float(args_cli.leg_damping),
        "leg_max_force": float(args_cli.leg_max_force),
        "balance_leg_servo_enabled": bool(args_cli.enable_balance_leg_servo),
        "balance_roll_gain": float(args_cli.balance_roll_gain),
        "balance_pitch_gain": float(args_cli.balance_pitch_gain),
        "balance_max_correction_m": float(args_cli.balance_max_correction),
        "x_slide_limit_m": float(args_cli.x_slide_limit),
        "x_slide_stiffness": float(args_cli.x_slide_stiffness),
        "x_slide_velocity_drive_enabled": str(args_cli.motion_mode) == "rear_anchor_velocity_push",
        "x_slide_effort_drive_enabled": str(args_cli.motion_mode) == "rear_anchor_effort_push",
        "x_slide_damping": float(args_cli.x_slide_damping),
        "x_slide_max_force": float(args_cli.x_slide_max_force),
        "x_slide_velocity_mps": float(args_cli.x_slide_velocity),
        "x_slide_effort_n": float(args_cli.x_slide_effort),
        "swing_x_force_scale": float(args_cli.swing_x_force_scale),
        "guarded_stop_target_x_m": (
            float(args_cli.guarded_stop_target_x) if args_cli.guarded_stop_target_x is not None else None
        ),
        "active_probe_enabled": bool(args_cli.enable_active_probe),
        "active_probe_uses_hidden_ground_truth": False,
        "active_probe_steps_requested": int(active_probe_steps_requested),
        "active_probe_steps_observed": 0,
        "active_probe_start_step": int(args_cli.settle_steps) if active_probe_steps_requested > 0 else None,
        "active_probe_end_step": int(carry_start_step) if active_probe_steps_requested > 0 else None,
        "carry_start_step": int(carry_start_step),
        "active_probe_lift_amplitude_m": float(args_cli.active_probe_lift_amplitude),
        "active_probe_horizontal_amplitude_m": float(args_cli.active_probe_horizontal_amplitude),
        "active_probe_baseline_torso_z_m": None,
        "active_probe_baseline_payload_z_m": None,
        "active_probe_max_torso_z_m": None,
        "active_probe_max_payload_z_m": None,
        "active_probe_torso_lift_response_m": None,
        "active_probe_payload_lift_response_m": None,
        "active_probe_max_relative_offset_error_m": 0.0,
        "active_probe_max_tilt_rad": 0.0,
        "active_probe_observed_risk_score": None,
        "active_probe_observed_load_risk_bucket": None,
        "active_probe_belief_available": False,
        "active_probe_belief_source": None,
        "probe_adaptive_gait_enabled": bool(args_cli.enable_probe_adaptive_gait),
        "probe_adaptive_gait_decision_available": False,
        "probe_adaptive_gait_decision_step": None,
        "probe_adaptive_risk_score": None,
        "probe_adaptive_risk_bucket": None,
        "probe_adaptive_medium_risk_threshold": float(args_cli.probe_adaptive_medium_risk_threshold),
        "probe_adaptive_high_risk_threshold": float(args_cli.probe_adaptive_high_risk_threshold),
        "probe_adaptive_gait_drive_scale": None,
        "probe_adaptive_base_gait_drive_target_x_m": (
            float(args_cli.gait_drive_target_x) if args_cli.gait_drive_target_x is not None else None
        ),
        "probe_adaptive_effective_gait_drive_target_x_m": None,
        "probe_adaptive_keeps_real_task_target": True,
        "probe_adaptive_posture_enabled": bool(args_cli.enable_probe_adaptive_posture),
        "probe_adaptive_posture_decision_available": False,
        "probe_adaptive_posture_decision_step": None,
        "probe_adaptive_posture_strategy": None,
        "probe_adaptive_posture_risk_bucket": None,
        "probe_adaptive_posture_risk_score": None,
        "probe_adaptive_posture_leg_target_offset_m": 0.0,
        "probe_adaptive_posture_effective_leg_target_m": float(args_cli.leg_target),
        "probe_adaptive_posture_changes_body_height": bool(args_cli.enable_probe_adaptive_posture),
        "swing_x_force_scaled_steps": 0,
        "stance_foot_latch_enabled": bool(args_cli.enable_stance_foot_latch),
        "stance_foot_latch_lift_threshold_m": float(args_cli.stance_foot_latch_lift_threshold),
        "stance_foot_latch_is_scaffold": bool(args_cli.enable_stance_foot_latch),
        "stance_foot_latch_enable_count": 0,
        "stance_foot_latch_disable_count": 0,
        "stance_foot_latch_retarget_count": 0,
        "payload_mass_kg": float(args_cli.payload_mass),
        "payload_size_m": [float(v) for v in args_cli.payload_size],
        "payload_local_x_m": float(args_cli.payload_local_x),
        "payload_local_z_m": float(args_cli.payload_local_z),
        "torso_mass_kg": float(args_cli.torso_mass),
        "torso_size_m": [float(v) for v in args_cli.torso_size],
        "torso_z_m": float(args_cli.torso_z),
        "root_pose_write_count": 0,
        "root_velocity_write_count": 0,
        "root_angular_velocity_write_count": 0,
        "body_root_pose_write_count": 0,
        "body_root_velocity_command_count": 0,
        "box_pose_write_count": 0,
        "payload_pose_write_count": 0,
        "fall_events": 0,
        "box_drop_events": 0,
        "nonfinite_state_events": 0,
        "max_tilt_rad": 0.0,
        "max_roll_rad": 0.0,
        "max_pitch_rad": 0.0,
        "min_torso_z_m": float(initial_torso[2]),
        "min_payload_z_m": float(initial_payload[2]),
        "max_torso_drift_xy_m": 0.0,
        "max_payload_drift_xy_m": 0.0,
        "max_torso_travel_x_m": 0.0,
        "max_payload_travel_x_m": 0.0,
        "min_torso_travel_x_m": 0.0,
        "min_payload_travel_x_m": 0.0,
        "max_abs_torso_travel_x_m": 0.0,
        "max_abs_payload_travel_x_m": 0.0,
        "post_settle_baseline_step": None,
        "post_settle_baseline_torso_x_m": None,
        "post_settle_baseline_payload_x_m": None,
        "max_post_settle_torso_travel_x_m": 0.0,
        "min_post_settle_torso_travel_x_m": 0.0,
        "max_abs_post_settle_torso_travel_x_m": 0.0,
        "max_post_settle_payload_travel_x_m": 0.0,
        "min_post_settle_payload_travel_x_m": 0.0,
        "max_abs_post_settle_payload_travel_x_m": 0.0,
        "post_settle_payload_travel_loss_after_peak_m": 0.0,
        "final_post_settle_torso_travel_x_m": None,
        "final_post_settle_payload_travel_x_m": None,
        "final_post_settle_target_distance_x_m": None,
        "final_post_settle_payload_target_distance_x_m": None,
        "final_target_distance_x_m": None,
        "final_payload_target_distance_x_m": None,
        "payload_relative_distance_m": float(math.dist(initial_torso[:3], initial_payload[:3])),
        "payload_relative_error_m": 0.0,
        "max_payload_relative_offset_error_m": 0.0,
        "post_settle_payload_relative_error_m": None,
        "max_post_settle_payload_relative_offset_error_m": None,
        "max_joint_motion_m": 0.0,
        "max_commanded_leg_lift_m": 0.0,
        "max_abs_commanded_x_slide_target_m": 0.0,
        "final_commanded_leg_lift_m": 0.0,
        "final_abs_commanded_x_slide_target_m": 0.0,
        "max_actual_leg_lift_m": 0.0,
        "max_abs_actual_x_slide_m": 0.0,
        "final_actual_leg_lift_m": 0.0,
        "final_abs_actual_x_slide_m": 0.0,
        "min_foot_z_m": None,
        "max_foot_z_m": None,
        "per_leg_near_ground_steps": per_leg_near_ground_steps,
        "per_leg_min_foot_z_m": per_leg_min_foot_z,
        "per_leg_max_foot_z_m": per_leg_max_foot_z,
        "per_leg_max_commanded_lift_m": per_leg_max_commanded_lift,
        "per_leg_max_abs_commanded_x_m": per_leg_max_abs_commanded_x,
        "per_leg_max_actual_lift_m": per_leg_max_actual_lift,
        "per_leg_max_abs_actual_x_m": per_leg_max_abs_actual_x,
        "per_leg_swing_x_force_scaled_steps": per_leg_swing_x_force_scaled_steps,
        "per_leg_stance_latch_enabled_steps": per_leg_stance_latch_enabled_steps,
        "per_leg_stance_latch_enable_count": per_leg_stance_latch_enable_count,
        "per_leg_stance_latch_disable_count": per_leg_stance_latch_disable_count,
        "per_leg_stance_latch_retarget_count": per_leg_stance_latch_retarget_count,
        "control_errors": [],
        "error": None,
    }

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "time_s",
                    "torso_x",
                    "torso_y",
                    "torso_z",
                    "payload_x",
                    "payload_y",
                    "payload_z",
                    "torso_drift_xy",
                    "payload_drift_xy",
                    "torso_travel_x",
                    "payload_travel_x",
                    "post_settle_torso_travel_x",
                    "post_settle_payload_travel_x",
                    "post_settle_target_distance_x",
                    "target_distance_x",
                    "tilt",
                    "max_joint_motion_m",
                    "commanded_leg_lift_m",
                    "abs_commanded_x_slide_target_m",
                    "actual_leg_lift_m",
                    "abs_actual_x_slide_m",
                    "near_ground_foot_count",
                    "commanded_swing_foot_count",
                    "min_foot_z",
                    "max_foot_z",
                    "fall",
                    "drop",
                ]
            )
            last_roll = 0.0
            last_pitch = 0.0
            last_tilt = 0.0
            last_payload_relative_offset_error = 0.0
            last_fall = 0
            last_drop = 0
            for step in range(int(args_cli.steps)):
                motion_step_override = None
                active_probe_active = (
                    bool(args_cli.enable_active_probe)
                    and int(active_probe_steps_requested) > 0
                    and int(args_cli.settle_steps) <= int(step) < int(carry_start_step)
                )
                active_probe_commanded_push = 0.0
                active_probe_commanded_x = 0.0
                current_target_x_override = probe_adaptive_gait_target_x_override
                if current_target_x_override is None:
                    current_target_x_override = quasistatic_target_x_override
                if (
                    bool(args_cli.enable_probe_adaptive_gait)
                    and not probe_adaptive_gait_decision_made
                    and int(step) >= int(carry_start_step)
                    and bool(summary.get("active_probe_belief_available"))
                ):
                    risk_score = float(summary.get("active_probe_observed_risk_score") or 0.0)
                    medium_threshold = float(args_cli.probe_adaptive_medium_risk_threshold)
                    high_threshold = float(args_cli.probe_adaptive_high_risk_threshold)
                    if risk_score >= high_threshold:
                        adaptive_bucket = "high"
                        gait_drive_scale = float(args_cli.probe_adaptive_high_gait_drive_scale)
                    elif risk_score >= medium_threshold:
                        adaptive_bucket = "medium"
                        gait_drive_scale = float(args_cli.probe_adaptive_medium_gait_drive_scale)
                    else:
                        adaptive_bucket = "low"
                        gait_drive_scale = 1.0
                    gait_drive_scale = max(0.10, min(1.50, gait_drive_scale))
                    base_gait_drive_target_x = (
                        float(quasistatic_target_x_override)
                        if quasistatic_target_x_override is not None
                        else float(args_cli.target_x)
                    )
                    probe_adaptive_gait_target_x_override = base_gait_drive_target_x * gait_drive_scale
                    current_target_x_override = probe_adaptive_gait_target_x_override
                    probe_adaptive_gait_decision_made = True
                    summary["probe_adaptive_gait_decision_available"] = True
                    summary["probe_adaptive_gait_decision_step"] = int(step)
                    summary["probe_adaptive_risk_score"] = float(risk_score)
                    summary["probe_adaptive_risk_bucket"] = adaptive_bucket
                    summary["probe_adaptive_gait_drive_scale"] = float(gait_drive_scale)
                    summary["probe_adaptive_base_gait_drive_target_x_m"] = float(base_gait_drive_target_x)
                    summary["probe_adaptive_effective_gait_drive_target_x_m"] = float(
                        probe_adaptive_gait_target_x_override
                    )
                if (
                    bool(args_cli.enable_probe_adaptive_posture)
                    and not probe_adaptive_posture_decision_made
                    and int(step) >= int(carry_start_step)
                    and bool(summary.get("active_probe_belief_available"))
                ):
                    risk_score = float(summary.get("active_probe_observed_risk_score") or 0.0)
                    medium_threshold = float(args_cli.probe_adaptive_medium_risk_threshold)
                    high_threshold = float(args_cli.probe_adaptive_high_risk_threshold)
                    if risk_score >= high_threshold:
                        posture_bucket = "high"
                        posture_strategy = "lower_carry_high"
                        posture_offset = float(args_cli.probe_adaptive_high_posture_leg_target_offset)
                    elif risk_score >= medium_threshold:
                        posture_bucket = "medium"
                        posture_strategy = "lower_carry_medium"
                        posture_offset = float(args_cli.probe_adaptive_medium_posture_leg_target_offset)
                    else:
                        posture_bucket = "low"
                        posture_strategy = "nominal_height"
                        posture_offset = 0.0
                    probe_adaptive_posture_leg_target_offset = max(
                        -0.05,
                        min(0.05, float(posture_offset)),
                    )
                    probe_adaptive_posture_decision_made = True
                    summary["probe_adaptive_posture_decision_available"] = True
                    summary["probe_adaptive_posture_decision_step"] = int(step)
                    summary["probe_adaptive_posture_risk_score"] = float(risk_score)
                    summary["probe_adaptive_posture_risk_bucket"] = posture_bucket
                    summary["probe_adaptive_posture_strategy"] = posture_strategy
                    summary["probe_adaptive_posture_leg_target_offset_m"] = float(
                        probe_adaptive_posture_leg_target_offset
                    )
                    summary["probe_adaptive_posture_effective_leg_target_m"] = float(
                        args_cli.leg_target
                    ) + float(probe_adaptive_posture_leg_target_offset)
                if str(args_cli.motion_mode) == "feedback_sync_inchworm" and step >= int(carry_start_step):
                    feedback_last_safe = (
                        last_fall == 0
                        and last_drop == 0
                        and float(last_tilt) <= float(args_cli.feedback_tilt_hold_threshold)
                        and float(last_payload_relative_offset_error) <= float(args_cli.feedback_payload_error_hold_threshold)
                    )
                    if feedback_last_safe:
                        feedback_release_steps += 1
                        feedback_motion_step += 1
                        feedback_last_block_reason = None
                    else:
                        feedback_hold_steps += 1
                        if last_fall:
                            feedback_last_block_reason = "fall"
                        elif last_drop:
                            feedback_last_block_reason = "drop"
                        elif float(last_tilt) > float(args_cli.feedback_tilt_hold_threshold):
                            feedback_last_block_reason = "tilt"
                        else:
                            feedback_last_block_reason = "payload_relative_error"
                    motion_step_override = feedback_motion_step
                if (
                    str(args_cli.motion_mode)
                    in ("gated_quasistatic_step_cycle", "guarded_prelift_quasistatic_step_cycle")
                    and step >= int(carry_start_step)
                ):
                    guarded_target_x = (
                            float(args_cli.guarded_stop_target_x)
                            if args_cli.guarded_stop_target_x is not None
                            else (
                            float(current_target_x_override)
                            if current_target_x_override is not None
                            else float(args_cli.target_x)
                        )
                    )
                    target_direction = 1.0 if float(guarded_target_x) >= 0.0 else -1.0
                    last_post_settle_payload_progress = target_direction * float(last_post_settle_payload_travel_x)
                    travel_loss = max(
                        0.0,
                        float(gated_step_peak_post_settle_payload_progress) - last_post_settle_payload_progress,
                    )
                    target_tolerance = float(args_cli.guarded_step_target_tolerance)
                    target_reached = abs(guarded_target_x - float(last_post_settle_payload_travel_x)) <= target_tolerance
                    target_reached = target_reached or (
                        target_direction >= 0.0
                        and float(last_post_settle_payload_travel_x) >= float(guarded_target_x)
                    )
                    target_reached = target_reached or (
                        target_direction < 0.0
                        and float(last_post_settle_payload_travel_x) <= float(guarded_target_x)
                    )
                    if str(args_cli.motion_mode) == "guarded_prelift_quasistatic_step_cycle":
                        gated_step_last_safe = (
                            last_fall == 0
                            and last_drop == 0
                            and not target_reached
                            and float(last_tilt) <= float(args_cli.feedback_tilt_hold_threshold)
                            and float(last_payload_relative_offset_error) <= float(args_cli.feedback_payload_error_hold_threshold)
                            and travel_loss <= float(args_cli.gated_step_max_travel_loss)
                        )
                    else:
                        gated_step_last_safe = (
                            last_fall == 0
                            and last_drop == 0
                            and float(last_tilt) <= float(args_cli.feedback_tilt_hold_threshold)
                            and float(last_payload_relative_offset_error) <= float(args_cli.feedback_payload_error_hold_threshold)
                            and travel_loss <= float(args_cli.gated_step_max_travel_loss)
                        )
                    if gated_step_last_safe:
                        gated_step_release_steps += 1
                        gated_step_motion_step += 1
                        gated_step_consecutive_loss_hold_steps = 0
                        gated_step_last_block_reason = None
                        motion_step_override = gated_step_motion_step
                    else:
                        gated_step_hold_steps += 1
                        if last_fall:
                            gated_step_last_block_reason = "fall"
                        elif last_drop:
                            gated_step_last_block_reason = "drop"
                        elif float(last_tilt) > float(args_cli.feedback_tilt_hold_threshold):
                            gated_step_last_block_reason = "tilt"
                        elif float(last_payload_relative_offset_error) > float(args_cli.feedback_payload_error_hold_threshold):
                            gated_step_last_block_reason = "payload_relative_error"
                        elif target_reached and str(args_cli.motion_mode) == "guarded_prelift_quasistatic_step_cycle":
                            gated_step_last_block_reason = "target_reached"
                        else:
                            gated_step_last_block_reason = "post_settle_payload_travel_loss"
                        if gated_step_last_block_reason == "post_settle_payload_travel_loss":
                            gated_step_consecutive_loss_hold_steps += 1
                        else:
                            gated_step_consecutive_loss_hold_steps = 0
                        period = max(int(args_cli.gait_period_steps), 32)
                        recovery_phase = max(0.0, min(0.44, float(args_cli.gated_step_recovery_phase)))
                        cycle_start = (max(0, int(gated_step_motion_step)) // period) * period
                        motion_step_override = cycle_start + int(recovery_phase * float(period))
                        if gated_step_last_block_reason in ("post_settle_payload_travel_loss", "target_reached"):
                            gated_step_recovery_steps += 1
                        if (
                            gated_step_last_block_reason == "post_settle_payload_travel_loss"
                            and int(args_cli.gated_step_loss_rebaseline_steps) > 0
                            and gated_step_consecutive_loss_hold_steps >= int(args_cli.gated_step_loss_rebaseline_steps)
                        ):
                            gated_step_peak_post_settle_payload_travel_x = float(last_post_settle_payload_travel_x)
                            gated_step_peak_post_settle_payload_progress = float(last_post_settle_payload_progress)
                            gated_step_peak_step = int(step)
                            gated_step_loss_rebaseline_count += 1
                            gated_step_consecutive_loss_hold_steps = 0
                if (
                    bool(args_cli.enable_active_probe)
                    and int(step) >= int(carry_start_step)
                    and motion_step_override is None
                ):
                    motion_step_override = max(0, int(step) - int(carry_start_step))
                leg_targets = _leg_targets_for_step(
                    step,
                    joint_indices,
                    leg_targets,
                    motion_step_override=motion_step_override,
                    target_x_override=current_target_x_override,
                )
                if active_probe_active:
                    probe_span = max(1, int(active_probe_steps_requested) - 1)
                    probe_progress = float(int(step) - int(args_cli.settle_steps)) / float(probe_span)
                    probe_progress = max(0.0, min(1.0, probe_progress))
                    pulse = math.sin(math.pi * probe_progress)
                    active_probe_commanded_push = max(0.0, float(args_cli.active_probe_lift_amplitude)) * pulse
                    active_probe_commanded_x = float(args_cli.active_probe_horizontal_amplitude) * math.sin(
                        2.0 * math.pi * probe_progress
                    )
                    for idx in joint_indices["z"].values():
                        leg_targets[idx] = max(
                            float(args_cli.leg_lower),
                            min(float(args_cli.leg_upper), float(args_cli.leg_target) - active_probe_commanded_push),
                        )
                    for idx in joint_indices["x"].values():
                        leg_targets[idx] = max(
                            -float(args_cli.x_slide_limit),
                            min(float(args_cli.x_slide_limit), active_probe_commanded_x),
                        )
                elif (
                    bool(args_cli.enable_probe_adaptive_posture)
                    and probe_adaptive_posture_decision_made
                    and int(step) >= int(carry_start_step)
                    and abs(float(probe_adaptive_posture_leg_target_offset)) > 0.0
                ):
                    for idx in joint_indices["z"].values():
                        leg_targets[idx] = max(
                            float(args_cli.leg_lower),
                            min(
                                float(args_cli.leg_upper),
                                float(leg_targets[idx]) + float(probe_adaptive_posture_leg_target_offset),
                            ),
                        )
                leg_targets = _apply_balance_leg_servo(leg_targets, joint_indices, roll=last_roll, pitch=last_pitch)
                x_command_indices: list[int] | None = None
                x_velocity_targets: list[float] | None = None
                x_effort_targets: list[float] | None = None
                if str(args_cli.motion_mode) in ("rear_anchor_velocity_push", "rear_anchor_effort_push") and step >= int(carry_start_step):
                    target_x = float(args_cli.target_x)
                    direction = 1.0 if target_x >= 0.0 else -1.0
                    commanded_speed = min(
                        abs(float(args_cli.x_slide_velocity)),
                        abs(target_x) / max(float(args_cli.ramp_steps) * 0.005, 1.0e-6),
                    )
                    commanded_effort = abs(float(args_cli.x_slide_effort))
                    current_post_step = max(0, int(step) - int(carry_start_step))
                    x_command_indices = []
                    x_velocity_targets = [] if str(args_cli.motion_mode) == "rear_anchor_velocity_push" else None
                    x_effort_targets = [] if str(args_cli.motion_mode) == "rear_anchor_effort_push" else None
                    for leg in LEG_NAMES:
                        x_idx = joint_indices["x"].get(leg)
                        if x_idx is not None:
                            x_command_indices.append(int(x_idx))
                            if x_velocity_targets is not None:
                                x_velocity_targets.append(0.0)
                            if x_effort_targets is not None:
                                x_effort_targets.append(0.0)
                    if current_post_step <= int(args_cli.ramp_steps):
                        for leg in ("rl", "rr"):
                            x_idx = joint_indices["x"].get(leg)
                            if x_idx is not None and x_command_indices is not None:
                                target_slot = x_command_indices.index(int(x_idx))
                                if x_velocity_targets is not None:
                                    x_velocity_targets[target_slot] = -direction * commanded_speed
                                if x_effort_targets is not None:
                                    x_effort_targets[target_slot] = -direction * commanded_effort
                z_targets = [float(leg_targets[idx]) for idx in joint_indices["z"].values()]
                x_targets = [float(leg_targets[idx]) for idx in joint_indices["x"].values()]
                commanded_leg_lift = max(0.0, max(z_targets) - float(args_cli.leg_target)) if z_targets else 0.0
                abs_commanded_x_slide_target = max((abs(v) for v in x_targets), default=0.0)
                commanded_swing_foot_count = 0
                for leg in LEG_NAMES:
                    z_idx = joint_indices["z"].get(leg)
                    x_idx = joint_indices["x"].get(leg)
                    if z_idx is not None:
                        commanded_lift = max(0.0, float(leg_targets[z_idx]) - float(args_cli.leg_target))
                        per_leg_max_commanded_lift[leg] = max(per_leg_max_commanded_lift[leg], commanded_lift)
                        if commanded_lift > 1.0e-4:
                            commanded_swing_foot_count += 1
                    if x_idx is not None:
                        per_leg_max_abs_commanded_x[leg] = max(
                            per_leg_max_abs_commanded_x[leg],
                            abs(float(leg_targets[x_idx])),
                        )
                swing_force_scaled_steps = 0
                swing_force_scale = max(0.0, min(1.0, float(args_cli.swing_x_force_scale)))
                if swing_force_scale < 0.999 and x_drive_max_force_attrs:
                    for leg, attr in x_drive_max_force_attrs.items():
                        z_idx = joint_indices["z"].get(leg)
                        commanded_lift = (
                            max(0.0, float(leg_targets[z_idx]) - float(args_cli.leg_target))
                            if z_idx is not None
                            else 0.0
                        )
                        if commanded_lift > 1.0e-4:
                            attr.Set(float(args_cli.x_slide_max_force) * swing_force_scale)
                            per_leg_swing_x_force_scaled_steps[leg] += 1
                            swing_force_scaled_steps += 1
                        else:
                            attr.Set(float(args_cli.x_slide_max_force))
                elif x_drive_max_force_attrs:
                    for attr in x_drive_max_force_attrs.values():
                        attr.Set(float(args_cli.x_slide_max_force))
                summary["swing_x_force_scaled_steps"] += int(swing_force_scaled_steps)
                if bool(args_cli.enable_stance_foot_latch) and stance_latch_joints:
                    for leg, joint in stance_latch_joints.items():
                        z_idx = joint_indices["z"].get(leg)
                        commanded_lift = (
                            max(0.0, float(leg_targets[z_idx]) - float(args_cli.leg_target))
                            if z_idx is not None
                            else 0.0
                        )
                        should_latch = (
                            step >= int(carry_start_step)
                            and commanded_lift <= float(args_cli.stance_foot_latch_lift_threshold)
                        )
                        if should_latch and not bool(stance_latch_enabled[leg]):
                            foot_pose_for_latch = _pose_wxyz(feet[leg])
                            _set_world_fixed_joint_enabled(
                                joint,
                                True,
                                (
                                    float(foot_pose_for_latch[0]),
                                    float(foot_pose_for_latch[1]),
                                    float(foot_pose_for_latch[2]),
                                ),
                            )
                            stance_latch_enabled[leg] = True
                            per_leg_stance_latch_enable_count[leg] += 1
                            per_leg_stance_latch_retarget_count[leg] += 1
                            summary["stance_foot_latch_enable_count"] += 1
                            summary["stance_foot_latch_retarget_count"] += 1
                        elif (not should_latch) and bool(stance_latch_enabled[leg]):
                            foot_pose_for_latch = _pose_wxyz(feet[leg])
                            _set_world_fixed_joint_enabled(
                                joint,
                                False,
                                (
                                    float(foot_pose_for_latch[0]),
                                    float(foot_pose_for_latch[1]),
                                    float(foot_pose_for_latch[2]),
                                ),
                            )
                            stance_latch_enabled[leg] = False
                            per_leg_stance_latch_disable_count[leg] += 1
                            summary["stance_foot_latch_disable_count"] += 1
                        if bool(stance_latch_enabled[leg]):
                            per_leg_stance_latch_enabled_steps[leg] += 1
                if x_velocity_targets is None and x_effort_targets is None:
                    robot.apply_action(ArticulationAction(joint_positions=leg_targets.tolist()))
                else:
                    z_indices = [int(idx) for idx in joint_indices["z"].values()]
                    z_position_targets = [float(leg_targets[idx]) for idx in z_indices]
                    robot.apply_action(
                        ArticulationAction(
                            joint_positions=z_position_targets,
                            joint_indices=z_indices,
                        )
                    )
                    if x_velocity_targets is not None:
                        robot.apply_action(
                            ArticulationAction(
                                joint_velocities=x_velocity_targets,
                                joint_indices=x_command_indices,
                            )
                        )
                    if x_effort_targets is not None:
                        robot.apply_action(
                            ArticulationAction(
                                joint_efforts=x_effort_targets,
                                joint_indices=x_command_indices,
                            )
                        )
                world.step(render=False)
                torso = _pose_wxyz(robot)
                box = _pose_wxyz(payload)
                finite = np.all(np.isfinite(np.array(torso + box, dtype=float)))
                foot_zs = []
                near_ground_foot_count = 0
                for leg, foot in feet.items():
                    foot_pose = _pose_wxyz(foot)
                    finite = finite and np.all(np.isfinite(np.array(foot_pose, dtype=float)))
                    foot_z = float(foot_pose[2])
                    foot_zs.append(foot_z)
                    per_leg_min_foot_z[leg] = (
                        foot_z if per_leg_min_foot_z[leg] is None else min(float(per_leg_min_foot_z[leg]), foot_z)
                    )
                    per_leg_max_foot_z[leg] = (
                        foot_z if per_leg_max_foot_z[leg] is None else max(float(per_leg_max_foot_z[leg]), foot_z)
                    )
                    if foot_z <= float(args_cli.foot_contact_z_threshold):
                        per_leg_near_ground_steps[leg] += 1
                        near_ground_foot_count += 1
                if not finite:
                    summary["nonfinite_state_events"] += 1
                roll, pitch = _quat_to_roll_pitch(float(torso[3]), float(torso[4]), float(torso[5]), float(torso[6]))
                last_roll = roll
                last_pitch = pitch
                tilt = float(max(abs(roll), abs(pitch)))
                torso_drift = float(math.hypot(float(torso[0]) - float(initial_torso[0]), float(torso[1]) - float(initial_torso[1])))
                payload_drift = float(math.hypot(float(box[0]) - float(initial_payload[0]), float(box[1]) - float(initial_payload[1])))
                torso_travel_x = float(torso[0]) - float(initial_torso[0])
                payload_travel_x = float(box[0]) - float(initial_payload[0])
                if step >= int(carry_start_step) and post_settle_torso_x is None:
                    post_settle_torso_x = float(torso[0])
                    post_settle_payload_x = float(box[0])
                    post_settle_payload_relative_xyz = (
                        float(box[0]) - float(torso[0]),
                        float(box[1]) - float(torso[1]),
                        float(box[2]) - float(torso[2]),
                    )
                    summary["post_settle_baseline_step"] = int(step)
                    summary["post_settle_baseline_torso_x_m"] = float(post_settle_torso_x)
                    summary["post_settle_baseline_payload_x_m"] = float(post_settle_payload_x)
                    if (
                        str(args_cli.motion_mode)
                        in (
                            "quasistatic_stance_transfer",
                            "quasistatic_step_cycle",
                            "gated_quasistatic_step_cycle",
                            "prelift_quasistatic_step_cycle",
                            "guarded_prelift_quasistatic_step_cycle",
                        )
                        and bool(args_cli.quasistatic_compensate_settle_drift)
                        and args_cli.gait_drive_target_x is None
                    ):
                        settle_torso_travel_x = float(post_settle_torso_x) - float(initial_torso[0])
                        quasistatic_target_x_override = float(args_cli.target_x) - settle_torso_travel_x
                        summary["quasistatic_effective_target_x_m"] = float(quasistatic_target_x_override)
                post_settle_torso_travel_x = 0.0
                post_settle_payload_travel_x = 0.0
                post_settle_target_distance_x = None
                post_settle_payload_target_distance_x = None
                if post_settle_torso_x is not None and post_settle_payload_x is not None:
                    post_settle_torso_travel_x = float(torso[0]) - post_settle_torso_x
                    post_settle_payload_travel_x = float(box[0]) - post_settle_payload_x
                    post_settle_target_distance_x = abs(float(args_cli.target_x) - post_settle_torso_travel_x)
                    post_settle_payload_target_distance_x = abs(float(args_cli.target_x) - post_settle_payload_travel_x)
                target_distance_x = abs(float(args_cli.target_x) - torso_travel_x)
                payload_target_distance_x = abs(float(args_cli.target_x) - payload_travel_x)
                joint_positions = np.array(robot.get_joint_positions(), dtype=float)
                joint_motion = float(np.nanmax(np.abs(joint_positions - initial_joint_positions))) if joint_positions.size else 0.0
                actual_z_positions = [float(joint_positions[idx]) for idx in joint_indices["z"].values()]
                actual_x_positions = [float(joint_positions[idx]) for idx in joint_indices["x"].values()]
                actual_leg_lift = max(0.0, max(actual_z_positions) - float(args_cli.leg_target)) if actual_z_positions else 0.0
                abs_actual_x_slide = max((abs(v) for v in actual_x_positions), default=0.0)
                for leg in LEG_NAMES:
                    z_idx = joint_indices["z"].get(leg)
                    x_idx = joint_indices["x"].get(leg)
                    if z_idx is not None:
                        actual_lift = max(0.0, float(joint_positions[z_idx]) - float(args_cli.leg_target))
                        per_leg_max_actual_lift[leg] = max(per_leg_max_actual_lift[leg], actual_lift)
                    if x_idx is not None:
                        per_leg_max_abs_actual_x[leg] = max(
                            per_leg_max_abs_actual_x[leg],
                            abs(float(joint_positions[x_idx])),
                        )
                fall = int(float(torso[2]) < float(args_cli.fall_z) or tilt > 0.85)
                drop = int(float(box[2]) < float(args_cli.drop_z))
                payload_rel = float(math.dist(torso[:3], box[:3]))
                payload_relative_offset_error = float(
                    math.dist(
                        (
                            float(box[0]) - float(torso[0]),
                            float(box[1]) - float(torso[1]),
                            float(box[2]) - float(torso[2]),
                        ),
                        (
                            float(initial_payload[0]) - float(initial_torso[0]),
                            float(initial_payload[1]) - float(initial_torso[1]),
                            float(initial_payload[2]) - float(initial_torso[2]),
                        ),
                    )
                )
                post_settle_payload_relative_offset_error = None
                if post_settle_payload_relative_xyz is not None:
                    post_settle_payload_relative_offset_error = float(
                        math.dist(
                            (
                                float(box[0]) - float(torso[0]),
                                float(box[1]) - float(torso[1]),
                                float(box[2]) - float(torso[2]),
                            ),
                            post_settle_payload_relative_xyz,
                        )
                    )
                if active_probe_active:
                    if active_probe_baseline_torso_z is None:
                        active_probe_baseline_torso_z = float(torso[2])
                        active_probe_baseline_payload_z = float(box[2])
                        active_probe_baseline_payload_relative_xyz = (
                            float(box[0]) - float(torso[0]),
                            float(box[1]) - float(torso[1]),
                            float(box[2]) - float(torso[2]),
                        )
                        active_probe_max_torso_z = float(torso[2])
                        active_probe_max_payload_z = float(box[2])
                        summary["active_probe_baseline_torso_z_m"] = float(active_probe_baseline_torso_z)
                        summary["active_probe_baseline_payload_z_m"] = float(active_probe_baseline_payload_z)
                    active_probe_observed_steps += 1
                    active_probe_max_torso_z = max(float(active_probe_max_torso_z), float(torso[2]))
                    active_probe_max_payload_z = max(float(active_probe_max_payload_z), float(box[2]))
                    if active_probe_baseline_payload_relative_xyz is not None:
                        active_probe_relative_error = float(
                            math.dist(
                                (
                                    float(box[0]) - float(torso[0]),
                                    float(box[1]) - float(torso[1]),
                                    float(box[2]) - float(torso[2]),
                                ),
                                active_probe_baseline_payload_relative_xyz,
                            )
                        )
                        active_probe_max_relative_offset_error = max(
                            active_probe_max_relative_offset_error,
                            active_probe_relative_error,
                        )
                    active_probe_max_tilt = max(float(active_probe_max_tilt), tilt)
                    payload_lift_response = (
                        max(0.0, float(active_probe_max_payload_z) - float(active_probe_baseline_payload_z))
                        if active_probe_baseline_payload_z is not None and active_probe_max_payload_z is not None
                        else None
                    )
                    torso_lift_response = (
                        max(0.0, float(active_probe_max_torso_z) - float(active_probe_baseline_torso_z))
                        if active_probe_baseline_torso_z is not None and active_probe_max_torso_z is not None
                        else None
                    )
                    if payload_lift_response is not None:
                        expected_response = max(1.0e-6, 0.20 * max(float(args_cli.active_probe_lift_amplitude), 0.0))
                        response_shortfall = max(0.0, expected_response - float(payload_lift_response))
                        risk_score = (
                            100.0 * response_shortfall
                            + 35.0 * float(active_probe_max_relative_offset_error)
                            + 10.0 * float(active_probe_max_tilt)
                        )
                        if risk_score >= 3.5:
                            risk_bucket = "high"
                        elif risk_score >= 1.5:
                            risk_bucket = "medium"
                        else:
                            risk_bucket = "low"
                        summary["active_probe_observed_risk_score"] = float(risk_score)
                        summary["active_probe_observed_load_risk_bucket"] = risk_bucket
                        summary["active_probe_belief_available"] = True
                        summary["active_probe_belief_source"] = "observed_micro_lift_response_not_hidden_ground_truth"
                    summary["active_probe_steps_observed"] = int(active_probe_observed_steps)
                    summary["active_probe_max_torso_z_m"] = float(active_probe_max_torso_z)
                    summary["active_probe_max_payload_z_m"] = float(active_probe_max_payload_z)
                    summary["active_probe_torso_lift_response_m"] = torso_lift_response
                    summary["active_probe_payload_lift_response_m"] = payload_lift_response
                    summary["active_probe_max_relative_offset_error_m"] = float(active_probe_max_relative_offset_error)
                    summary["active_probe_max_tilt_rad"] = float(active_probe_max_tilt)
                min_foot_z = min(foot_zs) if foot_zs else None
                max_foot_z = max(foot_zs) if foot_zs else None
                summary["completed_steps"] = step + 1
                if str(args_cli.motion_mode) == "feedback_sync_inchworm":
                    summary["feedback_motion_step_final"] = int(feedback_motion_step)
                    summary["feedback_hold_steps"] = int(feedback_hold_steps)
                    summary["feedback_release_steps"] = int(feedback_release_steps)
                    summary["feedback_last_safe"] = bool(feedback_last_safe)
                    summary["feedback_last_block_reason"] = feedback_last_block_reason
                if str(args_cli.motion_mode) in ("gated_quasistatic_step_cycle", "guarded_prelift_quasistatic_step_cycle"):
                    summary["gated_step_motion_step_final"] = int(gated_step_motion_step)
                    summary["gated_step_hold_steps"] = int(gated_step_hold_steps)
                    summary["gated_step_release_steps"] = int(gated_step_release_steps)
                    summary["gated_step_recovery_steps"] = int(gated_step_recovery_steps)
                    summary["gated_step_last_safe"] = bool(gated_step_last_safe)
                    summary["gated_step_last_block_reason"] = gated_step_last_block_reason
                    summary["gated_step_loss_rebaseline_count"] = int(gated_step_loss_rebaseline_count)
                summary["fall_events"] += fall
                summary["box_drop_events"] += drop
                summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), tilt)
                summary["max_roll_rad"] = max(float(summary["max_roll_rad"]), abs(float(roll)))
                summary["max_pitch_rad"] = max(float(summary["max_pitch_rad"]), abs(float(pitch)))
                summary["min_torso_z_m"] = min(float(summary["min_torso_z_m"]), float(torso[2]))
                summary["min_payload_z_m"] = min(float(summary["min_payload_z_m"]), float(box[2]))
                summary["max_torso_drift_xy_m"] = max(float(summary["max_torso_drift_xy_m"]), torso_drift)
                summary["max_payload_drift_xy_m"] = max(float(summary["max_payload_drift_xy_m"]), payload_drift)
                summary["max_torso_travel_x_m"] = max(float(summary["max_torso_travel_x_m"]), torso_travel_x)
                summary["max_payload_travel_x_m"] = max(float(summary["max_payload_travel_x_m"]), payload_travel_x)
                summary["min_torso_travel_x_m"] = min(float(summary["min_torso_travel_x_m"]), torso_travel_x)
                summary["min_payload_travel_x_m"] = min(float(summary["min_payload_travel_x_m"]), payload_travel_x)
                summary["max_abs_torso_travel_x_m"] = max(float(summary["max_abs_torso_travel_x_m"]), abs(torso_travel_x))
                summary["max_abs_payload_travel_x_m"] = max(float(summary["max_abs_payload_travel_x_m"]), abs(payload_travel_x))
                summary["max_post_settle_torso_travel_x_m"] = max(
                    float(summary["max_post_settle_torso_travel_x_m"]), post_settle_torso_travel_x
                )
                summary["min_post_settle_torso_travel_x_m"] = min(
                    float(summary["min_post_settle_torso_travel_x_m"]), post_settle_torso_travel_x
                )
                summary["max_abs_post_settle_torso_travel_x_m"] = max(
                    float(summary["max_abs_post_settle_torso_travel_x_m"]), abs(post_settle_torso_travel_x)
                )
                summary["max_post_settle_payload_travel_x_m"] = max(
                    float(summary["max_post_settle_payload_travel_x_m"]), post_settle_payload_travel_x
                )
                guarded_summary_target_x = (
                    float(args_cli.guarded_stop_target_x)
                    if args_cli.guarded_stop_target_x is not None
                    else (
                        float(current_target_x_override)
                        if current_target_x_override is not None
                        else float(args_cli.target_x)
                    )
                )
                guarded_summary_direction = 1.0 if float(guarded_summary_target_x) >= 0.0 else -1.0
                post_settle_payload_progress = guarded_summary_direction * float(post_settle_payload_travel_x)
                if post_settle_payload_progress > float(gated_step_peak_post_settle_payload_progress):
                    gated_step_peak_post_settle_payload_travel_x = float(post_settle_payload_travel_x)
                    gated_step_peak_post_settle_payload_progress = float(post_settle_payload_progress)
                    gated_step_peak_step = int(step)
                post_settle_payload_travel_loss_after_peak = max(
                    0.0,
                    float(gated_step_peak_post_settle_payload_progress) - float(post_settle_payload_progress),
                )
                summary["post_settle_payload_travel_loss_after_peak_m"] = post_settle_payload_travel_loss_after_peak
                if str(args_cli.motion_mode) in ("gated_quasistatic_step_cycle", "guarded_prelift_quasistatic_step_cycle"):
                    summary["gated_step_peak_post_settle_payload_travel_x_m"] = float(
                        gated_step_peak_post_settle_payload_travel_x
                    )
                    summary["gated_step_peak_post_settle_payload_progress_m"] = float(
                        gated_step_peak_post_settle_payload_progress
                    )
                    summary["gated_step_peak_step"] = gated_step_peak_step
                    summary["gated_step_travel_loss_after_peak_m"] = post_settle_payload_travel_loss_after_peak
                    summary["gated_step_directional_travel_loss_after_peak_m"] = (
                        post_settle_payload_travel_loss_after_peak
                    )
                summary["min_post_settle_payload_travel_x_m"] = min(
                    float(summary["min_post_settle_payload_travel_x_m"]), post_settle_payload_travel_x
                )
                summary["max_abs_post_settle_payload_travel_x_m"] = max(
                    float(summary["max_abs_post_settle_payload_travel_x_m"]), abs(post_settle_payload_travel_x)
                )
                summary["final_post_settle_torso_travel_x_m"] = post_settle_torso_travel_x
                summary["final_post_settle_payload_travel_x_m"] = post_settle_payload_travel_x
                summary["final_post_settle_target_distance_x_m"] = post_settle_target_distance_x
                summary["final_post_settle_payload_target_distance_x_m"] = post_settle_payload_target_distance_x
                summary["final_target_distance_x_m"] = target_distance_x
                summary["final_payload_target_distance_x_m"] = payload_target_distance_x
                summary["payload_relative_distance_m"] = payload_rel
                summary["payload_relative_error_m"] = payload_relative_offset_error
                summary["max_payload_relative_offset_error_m"] = max(
                    float(summary["max_payload_relative_offset_error_m"]), payload_relative_offset_error
                )
                summary["post_settle_payload_relative_error_m"] = post_settle_payload_relative_offset_error
                if post_settle_payload_relative_offset_error is not None:
                    previous = summary["max_post_settle_payload_relative_offset_error_m"]
                    summary["max_post_settle_payload_relative_offset_error_m"] = (
                        post_settle_payload_relative_offset_error
                        if previous is None
                        else max(float(previous), post_settle_payload_relative_offset_error)
                    )
                summary["max_joint_motion_m"] = max(float(summary["max_joint_motion_m"]), joint_motion)
                summary["max_commanded_leg_lift_m"] = max(
                    float(summary["max_commanded_leg_lift_m"]), commanded_leg_lift
                )
                summary["max_abs_commanded_x_slide_target_m"] = max(
                    float(summary["max_abs_commanded_x_slide_target_m"]), abs_commanded_x_slide_target
                )
                summary["final_commanded_leg_lift_m"] = commanded_leg_lift
                summary["final_abs_commanded_x_slide_target_m"] = abs_commanded_x_slide_target
                summary["max_actual_leg_lift_m"] = max(float(summary["max_actual_leg_lift_m"]), actual_leg_lift)
                summary["max_abs_actual_x_slide_m"] = max(float(summary["max_abs_actual_x_slide_m"]), abs_actual_x_slide)
                summary["final_actual_leg_lift_m"] = actual_leg_lift
                summary["final_abs_actual_x_slide_m"] = abs_actual_x_slide
                if min_foot_z is not None:
                    summary["min_foot_z_m"] = min_foot_z if summary["min_foot_z_m"] is None else min(float(summary["min_foot_z_m"]), min_foot_z)
                if max_foot_z is not None:
                    summary["max_foot_z_m"] = max_foot_z if summary["max_foot_z_m"] is None else max(float(summary["max_foot_z_m"]), max_foot_z)
                summary["per_leg_near_ground_steps"] = {key: int(value) for key, value in per_leg_near_ground_steps.items()}
                summary["per_leg_min_foot_z_m"] = per_leg_min_foot_z
                summary["per_leg_max_foot_z_m"] = per_leg_max_foot_z
                summary["per_leg_max_commanded_lift_m"] = per_leg_max_commanded_lift
                summary["per_leg_max_abs_commanded_x_m"] = per_leg_max_abs_commanded_x
                summary["per_leg_max_actual_lift_m"] = per_leg_max_actual_lift
                summary["per_leg_max_abs_actual_x_m"] = per_leg_max_abs_actual_x
                summary["per_leg_swing_x_force_scaled_steps"] = {
                    key: int(value) for key, value in per_leg_swing_x_force_scaled_steps.items()
                }
                summary["per_leg_stance_latch_enabled_steps"] = {
                    key: int(value) for key, value in per_leg_stance_latch_enabled_steps.items()
                }
                summary["per_leg_stance_latch_enable_count"] = {
                    key: int(value) for key, value in per_leg_stance_latch_enable_count.items()
                }
                summary["per_leg_stance_latch_disable_count"] = {
                    key: int(value) for key, value in per_leg_stance_latch_disable_count.items()
                }
                summary["per_leg_stance_latch_retarget_count"] = {
                    key: int(value) for key, value in per_leg_stance_latch_retarget_count.items()
                }
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    writer.writerow(
                        [
                            step,
                            step * 0.005,
                            torso[0],
                            torso[1],
                            torso[2],
                            box[0],
                            box[1],
                            box[2],
                            torso_drift,
                            payload_drift,
                            torso_travel_x,
                            payload_travel_x,
                            post_settle_torso_travel_x,
                            post_settle_payload_travel_x,
                            post_settle_target_distance_x,
                            target_distance_x,
                            tilt,
                            joint_motion,
                            commanded_leg_lift,
                            abs_commanded_x_slide_target,
                            actual_leg_lift,
                            abs_actual_x_slide,
                            near_ground_foot_count,
                            commanded_swing_foot_count,
                            min_foot_z,
                            max_foot_z,
                            fall,
                            drop,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} torso=({torso[0]:.3f},{torso[1]:.3f},{torso[2]:.3f}) "
                        f"payload_z={box[2]:.3f} travel_x={torso_travel_x:.4f} drift={torso_drift:.4f} tilt={tilt:.4f} "
                        f"joint_motion={joint_motion:.4f} fall={fall} drop={drop}",
                        flush=True,
                    )
                last_tilt = tilt
                last_payload_relative_offset_error = payload_relative_offset_error
                last_post_settle_payload_travel_x = post_settle_payload_travel_x
                last_fall = fall
                last_drop = drop
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run_scene()
    finally:
        simulation_app.close()
