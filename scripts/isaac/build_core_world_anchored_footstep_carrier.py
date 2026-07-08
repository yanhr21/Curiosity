#!/usr/bin/env python3
"""Core API anchored-footstep carrier diagnostic.

This diagnostic replaces direct torso velocity writes with a physical support
joint drive.  A free torso and payload box are rigid bodies; the current stable
mode uses a world-fixed support frame at torso height and a driven prismatic
joint that pulls the torso relative to that frame.  Four visible feet expose
the intended support footprint.

This is still not final robot carrying: the stance anchor is a simplified
support-foot controller, not a learned humanoid or quadruped policy.  It is
useful only if it proves the Isaac carry scene can move a loaded body without
torso/root velocity commands or payload pose writes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anchored-footstep carrier diagnostic.")
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--target-x", type=float, default=0.24)
    parser.add_argument("--step-length", type=float, default=0.06)
    parser.add_argument("--stance-steps", type=int, default=120)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument(
        "--probe-mode",
        choices=("horizontal_push_pull", "vertical_micro_lift"),
        default="horizontal_push_pull",
    )
    parser.add_argument("--probe-x-amplitude", type=float, default=0.0)
    parser.add_argument("--probe-z-amplitude", type=float, default=0.0)
    parser.add_argument("--belief-compliance-low-threshold", type=float, default=0.08)
    parser.add_argument("--belief-compliance-high-threshold", type=float, default=0.22)
    parser.add_argument(
        "--payload-mode",
        choices=(
            "none",
            "fixed_joint_to_torso",
            "caged_free_box",
            "staged_grasp_constraint",
            "open_tray_free_box",
            "side_clamp_free_box",
            "x_cradle_free_box",
            "cradle_free_box",
        ),
        default="fixed_joint_to_torso",
    )
    parser.add_argument("--payload-mass", type=float, default=4.0)
    parser.add_argument("--box-seed", type=int, default=None)
    parser.add_argument("--randomize-payload", action="store_true")
    parser.add_argument("--payload-mass-range", type=float, nargs=2, default=(4.0, 12.0), metavar=("MIN", "MAX"))
    parser.add_argument("--payload-size-jitter", type=float, default=0.0)
    parser.add_argument(
        "--payload-com-offset-range",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Uniform +/- center-of-mass offset range for the payload body in meters.",
    )
    parser.add_argument("--torso-mass", type=float, default=36.0)
    parser.add_argument("--torso-z", type=float, default=0.55)
    parser.add_argument("--torso-size", type=float, nargs=3, default=(0.48, 0.28, 0.18), metavar=("X", "Y", "Z"))
    parser.add_argument("--payload-size", type=float, nargs=3, default=(0.34, 0.24, 0.22), metavar=("X", "Y", "Z"))
    parser.add_argument("--payload-local-x", type=float, default=0.20)
    parser.add_argument("--payload-local-z", type=float, default=0.04)
    parser.add_argument("--cage-clearance-xy", type=float, default=0.025)
    parser.add_argument("--cage-clearance-z", type=float, default=0.025)
    parser.add_argument("--cage-wall-thickness", type=float, default=0.035)
    parser.add_argument("--cage-deck-mass", type=float, default=1.0)
    parser.add_argument("--cage-wall-mass", type=float, default=0.5)
    parser.add_argument("--cage-lid-mass", type=float, default=0.3)
    parser.add_argument("--grasp-enable-step", type=int, default=30)
    parser.add_argument("--grasp-shelf-clearance", type=float, default=0.003)
    parser.add_argument("--tray-clearance-xy", type=float, default=0.030)
    parser.add_argument("--tray-wall-height", type=float, default=0.090)
    parser.add_argument("--tray-wall-thickness", type=float, default=0.025)
    parser.add_argument("--tray-part-mass", type=float, default=0.4)
    parser.add_argument("--clamp-open-gap", type=float, default=0.060)
    parser.add_argument("--clamp-closed-gap", type=float, default=0.006)
    parser.add_argument("--clamp-pad-thickness", type=float, default=0.035)
    parser.add_argument("--clamp-pad-mass", type=float, default=0.8)
    parser.add_argument("--clamp-close-start-step", type=int, default=40)
    parser.add_argument("--clamp-close-steps", type=int, default=120)
    parser.add_argument("--clamp-drive-stiffness", type=float, default=2500.0)
    parser.add_argument("--clamp-drive-damping", type=float, default=600.0)
    parser.add_argument("--clamp-drive-max-force", type=float, default=6000.0)
    parser.add_argument("--x-cradle-open-gap", type=float, default=0.060)
    parser.add_argument("--x-cradle-closed-gap", type=float, default=0.006)
    parser.add_argument("--anchor-size", type=float, nargs=3, default=(0.22, 0.18, 0.06), metavar=("X", "Y", "Z"))
    parser.add_argument("--foot-length", type=float, default=0.22)
    parser.add_argument("--foot-width", type=float, default=0.10)
    parser.add_argument(
        "--support-foot-mode",
        choices=("static_markers", "fixed_to_anchor", "x_prismatic_to_anchor", "xz_prismatic_to_anchor"),
        default="static_markers",
        help=(
            "static_markers preserves the historical non-physical foot markers. "
            "fixed_to_anchor creates dynamic support feet fixed to the stance "
            "anchor so support comes from ground contact/friction instead of a "
            "world fixed joint. x_prismatic_to_anchor creates actuated X legs "
            "so the feet can push the anchor/torso through ground contact. "
            "xz_prismatic_to_anchor adds vertical swing joints for an "
            "alternating support-foot diagnostic."
        ),
    )
    parser.add_argument("--support-foot-mass", type=float, default=6.0)
    parser.add_argument("--support-foot-x-lower", type=float, default=-0.80)
    parser.add_argument("--support-foot-x-upper", type=float, default=0.20)
    parser.add_argument("--support-foot-drive-stiffness", type=float, default=18000.0)
    parser.add_argument("--support-foot-drive-damping", type=float, default=3000.0)
    parser.add_argument("--support-foot-drive-max-force", type=float, default=80000.0)
    parser.add_argument("--use-support-foot-drive", action="store_true")
    parser.add_argument("--support-foot-z-lower", type=float, default=-0.005)
    parser.add_argument("--support-foot-z-upper", type=float, default=0.120)
    parser.add_argument("--support-foot-z-drive-stiffness", type=float, default=14000.0)
    parser.add_argument("--support-foot-z-drive-damping", type=float, default=1800.0)
    parser.add_argument("--support-foot-z-drive-max-force", type=float, default=70000.0)
    parser.add_argument("--support-foot-step-height", type=float, default=0.070)
    parser.add_argument("--support-foot-stance-x", type=float, default=-0.080)
    parser.add_argument("--support-foot-swing-x", type=float, default=0.080)
    parser.add_argument("--support-foot-contact-z-threshold", type=float, default=0.028)
    parser.add_argument(
        "--enable-support-foot-contact-report",
        action="store_true",
        help=(
            "Enable PhysX contact report tracking for support-foot/ground "
            "contacts. This is contact-state evidence; force calibration is "
            "reported separately when available."
        ),
    )
    parser.add_argument(
        "--support-foot-contact-report-threshold",
        type=float,
        default=0.0,
        help="PhysX contact report threshold applied to support feet and ground.",
    )
    parser.add_argument(
        "--support-foot-effort-contact-threshold",
        type=float,
        default=1.0,
        help=(
            "Measured joint-effort threshold used as a force-like support "
            "evidence proxy for support-foot X/Z joints. This is diagnostic "
            "evidence, not a calibrated contact-force sensor."
        ),
    )
    parser.add_argument(
        "--support-foot-double-support-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction at the start and end of each alternating support-foot "
            "cycle where all feet are commanded to the ground. This is a "
            "diagnostic continuity gate for the scaffold controller, not a "
            "claim of biological gait."
        ),
    )
    parser.add_argument(
        "--support-foot-drive-direction-scale",
        type=float,
        default=-1.0,
        help="Scale applied to target direction for X support-foot joint targets.",
    )
    parser.add_argument(
        "--support-foot-placement-mode",
        choices=("alternating_fixed_x", "alternating_directional_x"),
        default="alternating_fixed_x",
        help=(
            "Foot-placement target convention for xz_prismatic_to_anchor. "
            "alternating_fixed_x preserves the historical fixed swing/stance "
            "X targets. alternating_directional_x mirrors swing and stance "
            "targets with target direction so the same controller can step "
            "forward or backward without changing signs by hand."
        ),
    )
    parser.add_argument(
        "--stance-foot-world-lock",
        action="store_true",
        help=(
            "Diagnostic support replacement for xz_prismatic_to_anchor: lock "
            "the commanded stance feet to the world through runtime-enabled "
            "fixed joints, while swing feet are unlocked. This is still a "
            "scaffold, but it directly audits whether progress can be made "
            "without planted-foot sliding."
        ),
    )
    parser.add_argument(
        "--freeze-locked-stance-foot-targets",
        action="store_true",
        help=(
            "When stance-foot world lock is enabled, freeze locked stance-foot "
            "X/Z drive targets at their measured joint positions instead of "
            "driving the same foot against its fixed-world constraint. This "
            "is a diagnostic for support consistency, not a final controller."
        ),
    )
    parser.add_argument(
        "--freeze-commanded-stance-foot-targets",
        action="store_true",
        help=(
            "Freeze commanded stance-foot X/Z drive targets at their measured "
            "joint positions without creating fixed-world stance locks. This "
            "tests whether contact/friction alone can support body-relative "
            "rail propulsion without dragging near-ground stance feet."
        ),
    )
    parser.add_argument(
        "--planted-stance-rail-propulsion",
        action="store_true",
        help=(
            "Diagnostic propulsion mode for locked/frozen stance feet: keep "
            "the torso rail target active during stance so the body moves "
            "relative to planted contacts instead of zeroing the rail whenever "
            "support-foot drive is enabled."
        ),
    )
    parser.add_argument("--feedback-step-controller", action="store_true")
    parser.add_argument("--feedback-step-x-gain", type=float, default=0.0)
    parser.add_argument("--feedback-step-x-limit", type=float, default=0.030)
    parser.add_argument("--feedback-step-tilt-gain", type=float, default=0.0)
    parser.add_argument("--feedback-step-tilt-limit", type=float, default=0.020)
    parser.add_argument(
        "--enable-online-probe-adaptive-support",
        action="store_true",
        help=(
            "After active probing finishes, choose a support-foot carry "
            "profile from observed probe telemetry inside the same episode. "
            "This changes support controller parameters only; it does not "
            "retarget root/body/box poses or rebuild hold geometry."
        ),
    )
    parser.add_argument("--online-probe-adaptive-medium-threshold", type=float, default=0.58)
    parser.add_argument("--online-probe-adaptive-high-threshold", type=float, default=0.75)
    parser.add_argument("--online-low-support-step-height", type=float, default=0.120)
    parser.add_argument("--online-low-support-double-support-fraction", type=float, default=0.12)
    parser.add_argument("--online-low-support-stance-x", type=float, default=-0.130)
    parser.add_argument("--online-low-support-swing-x", type=float, default=0.130)
    parser.add_argument("--online-medium-support-step-height", type=float, default=0.100)
    parser.add_argument("--online-medium-support-double-support-fraction", type=float, default=0.18)
    parser.add_argument("--online-medium-support-stance-x", type=float, default=-0.115)
    parser.add_argument("--online-medium-support-swing-x", type=float, default=0.115)
    parser.add_argument("--online-high-support-step-height", type=float, default=0.080)
    parser.add_argument("--online-high-support-double-support-fraction", type=float, default=0.24)
    parser.add_argument("--online-high-support-stance-x", type=float, default=-0.100)
    parser.add_argument("--online-high-support-swing-x", type=float, default=0.100)
    parser.add_argument(
        "--enable-online-probe-adaptive-hold",
        action="store_true",
        help=(
            "After active probing finishes, choose an actuated hold/contact "
            "closure profile from observed probe telemetry inside the same "
            "episode. This only changes clamp/cradle joint targets for payload "
            "modes with actuated hold joints; it does not retarget root/body/box "
            "poses or rebuild geometry."
        ),
    )
    parser.add_argument("--online-low-hold-closure-fraction", type=float, default=0.45)
    parser.add_argument("--online-medium-hold-closure-fraction", type=float, default=0.75)
    parser.add_argument("--online-high-hold-closure-fraction", type=float, default=1.0)
    parser.add_argument(
        "--support-foot-continuity-grace-steps",
        type=int,
        default=0,
        help=(
            "Ignore this many steps after drive start when accumulating strict "
            "near-ground support continuity metrics. Use only to exclude the "
            "initial contact-establishment transient from diagnostic gates."
        ),
    )
    parser.add_argument("--stance-half-length", type=float, default=0.28)
    parser.add_argument("--stance-half-width", type=float, default=0.22)
    parser.add_argument("--drive-stiffness", type=float, default=22000.0)
    parser.add_argument("--drive-damping", type=float, default=3500.0)
    parser.add_argument("--drive-max-force", type=float, default=60000.0)
    parser.add_argument("--rail-lower", type=float, default=-0.10)
    parser.add_argument("--rail-upper", type=float, default=0.04)
    parser.add_argument("--rail-joint-count", type=int, default=1)
    parser.add_argument(
        "--rail-target-direction-scale",
        type=float,
        default=1.0,
        help=(
            "Diagnostic sign/scale applied to stance rail targets. Use only "
            "when auditing joint-axis convention; changing this does not "
            "constitute a new locomotion controller."
        ),
    )
    parser.add_argument("--stop-threshold", type=float, default=0.01)
    parser.add_argument("--fix-anchor-to-world", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--anchor-as-articulation-root",
        action="store_true",
        help=(
            "Apply ArticulationRootAPI to the stance anchor instead of the "
            "Robot xform. This is a support-switch diagnostic: it allows "
            "replanting the support root without assigning a transform to a "
            "non-root articulation link."
        ),
    )
    parser.add_argument(
        "--replant-anchor-world-joint",
        action="store_true",
        help=(
            "Keep the stance anchor constrained by a world fixed joint and "
            "replant support by moving the joint's world-frame localPos0 target "
            "at cycle boundaries instead of writing a rigid-body pose."
        ),
    )
    parser.add_argument(
        "--cumulative-cycle-target",
        action="store_true",
        help=(
            "Command multi-cycle rail displacement as stride * (cycle + phase) "
            "instead of resetting the rail target each cycle. This is a "
            "diagnostic for stable multi-cycle transport, not a true support "
            "replant claim."
        ),
    )
    parser.add_argument(
        "--disable-support-reposition",
        action="store_true",
        help=(
            "Do not write or retarget the stance support at cycle boundaries. "
            "Use this with physical support contacts when testing replacement "
            "of fixed world support."
        ),
    )
    parser.add_argument("--static-friction", type=float, default=3.0)
    parser.add_argument("--dynamic-friction", type=float, default=2.5)
    parser.add_argument("--fall-z", type=float, default=0.34)
    parser.add_argument("--drop-z", type=float, default=0.20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/core_world_anchored_footstep_carrier"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _apply_payload_randomization(args: argparse.Namespace) -> None:
    args.payload_mass_requested = float(args.payload_mass)
    args.payload_size_requested = tuple(float(v) for v in args.payload_size)
    args.payload_com_offset_m = (0.0, 0.0, 0.0)
    if not bool(args.randomize_payload):
        return

    seed = int(args.box_seed) if args.box_seed is not None else 0
    rng = random.Random(seed)
    mass_min, mass_max = sorted(float(v) for v in args.payload_mass_range)
    args.payload_mass = float(rng.uniform(mass_min, mass_max))

    jitter = max(0.0, float(args.payload_size_jitter))
    if jitter > 0.0:
        args.payload_size = tuple(
            max(0.02, float(size) * rng.uniform(1.0 - jitter, 1.0 + jitter))
            for size in args.payload_size
        )

    com_range = tuple(max(0.0, float(v)) for v in args.payload_com_offset_range)
    args.payload_com_offset_m = tuple(rng.uniform(-limit, limit) for limit in com_range)


_refuse_login_node()
args_cli = parse_args()
_apply_payload_randomization(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PROGRESS] AppLauncher started", flush=True)

import numpy as np  # noqa: E402
from omni.physics.core import ContactEventType, get_physics_simulation_interface  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

print("[PROGRESS] Core imports complete", flush=True)


ROBOT_PATH = "/World/Robot"
ANCHOR_PATH = "/World/Robot/StanceAnchor"
TORSO_PATH = "/World/Robot/Torso"
BOX_PATH = "/World/CarryBox"
FOOT_NAMES = ("fl", "fr", "rl", "rr")


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
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(Gf.Vec3d(*[float(v) for v in translation]))
    xform.SetScale(Gf.Vec3f(*[float(v) for v in scale]))


def _define_physics_material(stage: Usd.Stage, path: str) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(float(args_cli.static_friction))
    physics_material.CreateDynamicFrictionAttr().Set(float(args_cli.dynamic_friction))
    physics_material.CreateRestitutionAttr().Set(0.0)
    return material


def _bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _box_body(
    stage: Usd.Stage,
    path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    mass: float = 1.0,
    rigid: bool = True,
    kinematic: bool = False,
    disable_gravity: bool = False,
    collision: bool = True,
    material: UsdShade.Material,
    center_of_mass: tuple[float, float, float] | None = None,
) -> UsdPhysics.FixedJoint:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translation, size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        _bind_material(cube.GetPrim(), material)
    if rigid:
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        if kinematic:
            rigid_api.CreateKinematicEnabledAttr().Set(True)
        if disable_gravity and hasattr(rigid_api, "GetDisableGravityAttr"):
            rigid_api.GetDisableGravityAttr().Set(True)
        mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
        mass_api.CreateMassAttr(float(mass))
        if center_of_mass is not None:
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in center_of_mass]))


def _set_collision_enabled(stage: Usd.Stage, path: str, enabled: bool) -> bool:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return False
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    attr = collision_api.GetCollisionEnabledAttr()
    if not attr.IsValid():
        attr = collision_api.CreateCollisionEnabledAttr()
    attr.Set(bool(enabled))
    return True


def _enable_contact_report_api(stage: Usd.Stage, paths: list[str], threshold: float) -> list[str]:
    enabled_paths: list[str] = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        contact_report_api.CreateThresholdAttr().Set(float(threshold))
        enabled_paths.append(path)
    return enabled_paths


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
    return joint


def _disabled_fixed_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
) -> UsdPhysics.FixedJoint:
    joint = _fixed_joint(stage, joint_path, body0, body1, local_pos0, (0.0, 0.0, 0.0))
    joint.CreateJointEnabledAttr().Set(False)
    return joint


def _enable_fixed_joint(joint: UsdPhysics.FixedJoint, local_pos0: tuple[float, float, float]) -> None:
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.GetJointEnabledAttr().Set(True)


def _fixed_joint_to_world(
    stage: Usd.Stage,
    joint_path: str,
    body1: str,
    world_pos: tuple[float, float, float],
) -> UsdPhysics.FixedJoint:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in world_pos]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint


def _disabled_fixed_joint_to_world(
    stage: Usd.Stage,
    joint_path: str,
    body1: str,
    world_pos: tuple[float, float, float],
) -> UsdPhysics.FixedJoint:
    joint = _fixed_joint_to_world(stage, joint_path, body1, world_pos)
    joint.CreateJointEnabledAttr().Set(False)
    return joint


def _set_world_fixed_joint(
    joint: UsdPhysics.FixedJoint,
    world_pos: tuple[float, float, float],
    *,
    enabled: bool,
) -> None:
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in world_pos]))
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateJointEnabledAttr().Set(bool(enabled))


def _rail_joint(stage: Usd.Stage, joint_path: str, body0: str, body1: str) -> None:
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateAxisAttr().Set("X")
    joint.CreateLowerLimitAttr().Set(float(args_cli.rail_lower))
    joint.CreateUpperLimitAttr().Set(float(args_cli.rail_upper))
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(args_cli.drive_stiffness))
    drive.CreateDampingAttr().Set(float(args_cli.drive_damping))
    drive.CreateMaxForceAttr().Set(float(args_cli.drive_max_force))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)


def _support_foot_x_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
) -> None:
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateAxisAttr().Set("X")
    joint.CreateLowerLimitAttr().Set(float(args_cli.support_foot_x_lower))
    joint.CreateUpperLimitAttr().Set(float(args_cli.support_foot_x_upper))
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(args_cli.support_foot_drive_stiffness))
    drive.CreateDampingAttr().Set(float(args_cli.support_foot_drive_damping))
    drive.CreateMaxForceAttr().Set(float(args_cli.support_foot_drive_max_force))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)


def _support_foot_z_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
) -> None:
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateAxisAttr().Set("Z")
    joint.CreateLowerLimitAttr().Set(float(args_cli.support_foot_z_lower))
    joint.CreateUpperLimitAttr().Set(float(args_cli.support_foot_z_upper))
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(args_cli.support_foot_z_drive_stiffness))
    drive.CreateDampingAttr().Set(float(args_cli.support_foot_z_drive_damping))
    drive.CreateMaxForceAttr().Set(float(args_cli.support_foot_z_drive_max_force))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)


def _clamp_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0: str,
    body1: str,
    local_pos0: tuple[float, float, float],
    lower: float,
    upper: float,
) -> None:
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateAxisAttr().Set("Y")
    joint.CreateLowerLimitAttr().Set(float(lower))
    joint.CreateUpperLimitAttr().Set(float(upper))
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(args_cli.clamp_drive_stiffness))
    drive.CreateDampingAttr().Set(float(args_cli.clamp_drive_damping))
    drive.CreateMaxForceAttr().Set(float(args_cli.clamp_drive_max_force))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)


def _pose_wxyz(prim: SingleArticulation | SingleRigidPrim) -> list[float]:
    pos, quat = prim.get_world_pose()
    pos = _as_numpy(pos).reshape(-1)
    quat = _as_numpy(quat).reshape(-1)
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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _probe_belief(summary: dict) -> dict:
    probe_steps = int(summary.get("probe_steps_requested") or 0)
    probe_x_amp = abs(float(summary.get("probe_x_amplitude_m") or 0.0))
    probe_z_amp = abs(float(summary.get("probe_z_amplitude_m") or 0.0))
    probe_amp = max(probe_x_amp, probe_z_amp)
    if probe_steps <= 0 or probe_amp <= 0.0:
        return {
            "probe_belief_available": False,
            "probe_belief_source": "no_active_probe",
            "probe_belief_uses_hidden_ground_truth": False,
            "probe_compliance_proxy": None,
            "probe_lag_proxy": None,
            "probe_support_foot_x_tracking_proxy": None,
            "probe_support_foot_z_tracking_proxy": None,
            "probe_support_foot_x_effort_proxy": None,
            "probe_support_foot_z_effort_proxy": None,
            "probe_risk_score": None,
            "probe_load_risk_bucket": "unknown_no_probe",
            "probe_recommended_carry_adjustment": "none",
            "probe_belief_policy_action_applied": bool(summary.get("probe_belief_policy_action_applied")),
        }

    torso_travel = abs(float(summary.get("max_probe_torso_travel_x_m") or 0.0))
    torso_travel_z = abs(float(summary.get("max_probe_torso_travel_z_m") or 0.0))
    box_travel = abs(float(summary.get("max_probe_payload_travel_x_m") or 0.0))
    box_travel_z = abs(float(summary.get("max_probe_payload_travel_z_m") or 0.0))
    relative_error = abs(float(summary.get("max_probe_payload_relative_error_m") or 0.0))
    final_lag = abs(float(summary.get("final_probe_payload_lag_x_m") or 0.0))
    final_z_lag = abs(float(summary.get("final_probe_payload_lag_z_m") or 0.0))
    support_tracking_error = abs(float(summary.get("max_probe_support_foot_x_tracking_error_m") or 0.0))
    support_z_tracking_error = abs(float(summary.get("max_probe_support_foot_z_tracking_error_m") or 0.0))
    support_effort = abs(float(summary.get("max_probe_support_foot_x_measured_effort") or 0.0))
    support_z_effort = abs(float(summary.get("max_probe_support_foot_z_measured_effort") or 0.0))
    compliance_proxy = relative_error / max(torso_travel, torso_travel_z, probe_amp, 1e-6)
    lag_proxy = max(final_lag, final_z_lag) / max(box_travel, box_travel_z, probe_amp, 1e-6)
    support_tracking_proxy = support_tracking_error / max(probe_x_amp, 1e-6) if probe_x_amp > 1e-6 else 0.0
    support_z_tracking_proxy = support_z_tracking_error / max(probe_z_amp, 1e-6) if probe_z_amp > 1e-6 else 0.0
    support_effort_proxy = (
        support_effort / max(float(args_cli.support_foot_drive_max_force), 1e-6)
        if probe_x_amp > 1e-6
        else 0.0
    )
    support_z_effort_proxy = (
        support_z_effort / max(float(args_cli.support_foot_z_drive_max_force), 1e-6)
        if probe_z_amp > 1e-6
        else 0.0
    )
    low = max(1e-6, float(args_cli.belief_compliance_low_threshold))
    high = max(low + 1e-6, float(args_cli.belief_compliance_high_threshold))
    compliance_score = _clamp01((compliance_proxy - low) / (high - low))
    lag_score = _clamp01(lag_proxy / max(high, 1e-6))
    tracking_score = _clamp01(max(support_tracking_proxy, support_z_tracking_proxy) / max(high, 1e-6))
    if bool(summary.get("probe_joint_effort_available")):
        effort_score = _clamp01(max(support_effort_proxy, support_z_effort_proxy) / 0.25)
        risk_score = _clamp01(0.40 * compliance_score + 0.15 * lag_score + 0.15 * tracking_score + 0.30 * effort_score)
    else:
        risk_score = _clamp01(0.55 * compliance_score + 0.20 * lag_score + 0.25 * tracking_score)
    if risk_score < 0.33:
        bucket = "low_observed_load_response"
        adjustment = "nominal_carry"
    elif risk_score < 0.66:
        bucket = "moderate_observed_load_response"
        adjustment = "slow_gait_or_lower_carry_candidate"
    else:
        bucket = "high_observed_load_or_shift_response"
        adjustment = "slow_gait_low_or_chest_supported_candidate"
    return {
        "probe_belief_available": True,
        "probe_belief_source": "heuristic_from_probe_telemetry_not_calibrated_mass_estimator",
        "probe_belief_uses_hidden_ground_truth": False,
        "probe_compliance_proxy": float(compliance_proxy),
        "probe_lag_proxy": float(lag_proxy),
        "probe_support_foot_x_tracking_proxy": float(support_tracking_proxy),
        "probe_support_foot_z_tracking_proxy": float(support_z_tracking_proxy),
        "probe_support_foot_x_effort_proxy": float(support_effort_proxy),
        "probe_support_foot_z_effort_proxy": float(support_z_effort_proxy),
        "probe_risk_score": float(risk_score),
        "probe_load_risk_bucket": bucket,
        "probe_recommended_carry_adjustment": adjustment,
        "probe_belief_policy_action_applied": bool(summary.get("probe_belief_policy_action_applied")),
    }


def _select_online_support_profile(args: argparse.Namespace, risk: float | None) -> dict[str, Any]:
    medium_threshold = float(args.online_probe_adaptive_medium_threshold)
    high_threshold = float(args.online_probe_adaptive_high_threshold)
    risk_value = 0.0 if risk is None else float(risk)
    if risk_value >= high_threshold:
        bucket = "high"
        profile = "compact_high_double_support"
        step_height = float(args.online_high_support_step_height)
        double_support = float(args.online_high_support_double_support_fraction)
        stance_x = float(args.online_high_support_stance_x)
        swing_x = float(args.online_high_support_swing_x)
    elif risk_value >= medium_threshold:
        bucket = "medium"
        profile = "compact_medium_double_support"
        step_height = float(args.online_medium_support_step_height)
        double_support = float(args.online_medium_support_double_support_fraction)
        stance_x = float(args.online_medium_support_stance_x)
        swing_x = float(args.online_medium_support_swing_x)
    else:
        bucket = "low"
        profile = "nominal_reach_support"
        step_height = float(args.online_low_support_step_height)
        double_support = float(args.online_low_support_double_support_fraction)
        stance_x = float(args.online_low_support_stance_x)
        swing_x = float(args.online_low_support_swing_x)
    return {
        "bucket": bucket,
        "profile": profile,
        "step_height": step_height,
        "double_support_fraction": double_support,
        "stance_x": stance_x,
        "swing_x": swing_x,
    }


def _select_online_hold_profile(args: argparse.Namespace, risk: float | None) -> dict[str, Any]:
    medium_threshold = float(args.online_probe_adaptive_medium_threshold)
    high_threshold = float(args.online_probe_adaptive_high_threshold)
    risk_value = 0.0 if risk is None else float(risk)
    if risk_value >= high_threshold:
        bucket = "high"
        profile = "max_contact_closure"
        closure_fraction = float(args.online_high_hold_closure_fraction)
    elif risk_value >= medium_threshold:
        bucket = "medium"
        profile = "reinforced_contact_closure"
        closure_fraction = float(args.online_medium_hold_closure_fraction)
    else:
        bucket = "low"
        profile = "light_contact_closure"
        closure_fraction = float(args.online_low_hold_closure_fraction)
    return {
        "bucket": bucket,
        "profile": profile,
        "closure_fraction": max(0.0, min(1.0, closure_fraction)),
    }


def _foot_offsets() -> dict[str, tuple[float, float]]:
    return {
        "fl": (float(args_cli.stance_half_length), float(args_cli.stance_half_width)),
        "fr": (float(args_cli.stance_half_length), -float(args_cli.stance_half_width)),
        "rl": (-float(args_cli.stance_half_length), float(args_cli.stance_half_width)),
        "rr": (-float(args_cli.stance_half_length), -float(args_cli.stance_half_width)),
    }


def design_scene(stage: Usd.Stage) -> UsdPhysics.FixedJoint | None:
    UsdGeom.Xform.Define(stage, ROBOT_PATH)
    UsdGeom.Xform.Define(stage, "/World/SupportFeet")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = _define_physics_material(stage, "/World/Looks/HighFrictionMaterial")
    disable_internal_body_collision = str(args_cli.payload_mode) == "cradle_free_box"
    kinematic_support_root = bool(args_cli.anchor_as_articulation_root) and not bool(args_cli.fix_anchor_to_world)
    _box_body(stage, "/World/Ground", (5.0, 3.0, 0.05), (0.0, 0.0, -0.025), (0.31, 0.33, 0.33), rigid=False, material=material)
    _box_body(
        stage,
        ANCHOR_PATH,
        tuple(float(v) for v in args_cli.anchor_size),
        (0.0, 0.0, float(args_cli.torso_z)),
        (0.08, 0.12, 0.18),
        mass=500.0,
        kinematic=kinematic_support_root,
        collision=not disable_internal_body_collision,
        material=material,
    )
    articulation_root_path = ANCHOR_PATH if bool(args_cli.anchor_as_articulation_root) else ROBOT_PATH
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(articulation_root_path))
    _box_body(
        stage,
        TORSO_PATH,
        tuple(float(v) for v in args_cli.torso_size),
        (0.0, 0.0, float(args_cli.torso_z)),
        (0.12, 0.20, 0.30),
        mass=float(args_cli.torso_mass),
        collision=not disable_internal_body_collision,
        material=material,
    )
    payload_size = tuple(float(v) for v in args_cli.payload_size)
    payload_world = (
        float(args_cli.payload_local_x),
        0.0,
        float(args_cli.torso_z) + float(args_cli.payload_local_z),
    )
    payload_active = str(args_cli.payload_mode) != "none"
    if payload_active:
        _box_body(
            stage,
            BOX_PATH,
            payload_size,
            payload_world,
            (0.58, 0.43, 0.24),
            mass=float(args_cli.payload_mass),
            material=material,
            center_of_mass=tuple(float(v) for v in args_cli.payload_com_offset_m),
        )
    staged_grasp_joint: UsdPhysics.FixedJoint | None = None
    if str(args_cli.payload_mode) == "fixed_joint_to_torso":
        _fixed_joint(
            stage,
            "/World/Robot/FixedPayloadJoint",
            TORSO_PATH,
            BOX_PATH,
            (float(args_cli.payload_local_x), 0.0, float(args_cli.payload_local_z)),
            (0.0, 0.0, 0.0),
        )
    elif str(args_cli.payload_mode) == "caged_free_box":
        cage_gap_xy = float(args_cli.cage_clearance_xy)
        cage_gap_z = float(args_cli.cage_clearance_z)
        cage_t = float(args_cli.cage_wall_thickness)
        payload_local_x = float(args_cli.payload_local_x)
        payload_local_z = float(args_cli.payload_local_z)
        inner_x = payload_size[0] + 2.0 * cage_gap_xy
        inner_y = payload_size[1] + 2.0 * cage_gap_xy
        outer_x = inner_x + 2.0 * cage_t
        outer_y = inner_y + 2.0 * cage_t
        wall_h = payload_size[2] + 2.0 * cage_gap_z
        cage_specs = {
            "deck": (
                (outer_x, outer_y, cage_t),
                (payload_local_x, 0.0, payload_local_z - 0.5 * payload_size[2] - cage_gap_z - 0.5 * cage_t),
                (0.16, 0.34, 0.28),
                float(args_cli.cage_deck_mass),
            ),
            "front": (
                (cage_t, outer_y, wall_h),
                (payload_local_x + 0.5 * inner_x + 0.5 * cage_t, 0.0, payload_local_z),
                (0.18, 0.28, 0.46),
                float(args_cli.cage_wall_mass),
            ),
            "rear": (
                (cage_t, outer_y, wall_h),
                (payload_local_x - 0.5 * inner_x - 0.5 * cage_t, 0.0, payload_local_z),
                (0.18, 0.28, 0.46),
                float(args_cli.cage_wall_mass),
            ),
            "left": (
                (outer_x, cage_t, wall_h),
                (payload_local_x, 0.5 * inner_y + 0.5 * cage_t, payload_local_z),
                (0.12, 0.30, 0.46),
                float(args_cli.cage_wall_mass),
            ),
            "right": (
                (outer_x, cage_t, wall_h),
                (payload_local_x, -0.5 * inner_y - 0.5 * cage_t, payload_local_z),
                (0.12, 0.30, 0.46),
                float(args_cli.cage_wall_mass),
            ),
            "lid": (
                (outer_x, outer_y, cage_t),
                (payload_local_x, 0.0, payload_local_z + 0.5 * payload_size[2] + cage_gap_z + 0.5 * cage_t),
                (0.22, 0.22, 0.26),
                float(args_cli.cage_lid_mass),
            ),
        }
        for name, (size, local_pos, color, mass) in cage_specs.items():
            path = f"/World/Robot/CarryCage_{name}"
            world_pos = (
                local_pos[0],
                local_pos[1],
                float(args_cli.torso_z) + local_pos[2],
            )
            _box_body(stage, path, size, world_pos, color, mass=mass, material=material)
            _fixed_joint(stage, f"/World/Robot/CarryCage_{name}_joint", TORSO_PATH, path, local_pos, (0.0, 0.0, 0.0))
    elif str(args_cli.payload_mode) == "open_tray_free_box":
        tray_gap_xy = float(args_cli.tray_clearance_xy)
        tray_t = float(args_cli.tray_wall_thickness)
        tray_h = float(args_cli.tray_wall_height)
        tray_mass = float(args_cli.tray_part_mass)
        payload_local_x = float(args_cli.payload_local_x)
        payload_local_z = float(args_cli.payload_local_z)
        inner_x = payload_size[0] + 2.0 * tray_gap_xy
        inner_y = payload_size[1] + 2.0 * tray_gap_xy
        outer_x = inner_x + 2.0 * tray_t
        outer_y = inner_y + 2.0 * tray_t
        tray_specs = {
            "deck": (
                (outer_x, outer_y, tray_t),
                (payload_local_x, 0.0, payload_local_z - 0.5 * payload_size[2] - 0.5 * tray_t),
                (0.16, 0.32, 0.26),
            ),
            "rear_stop": (
                (tray_t, outer_y, tray_h),
                (payload_local_x - 0.5 * inner_x - 0.5 * tray_t, 0.0, payload_local_z - 0.5 * payload_size[2] + 0.5 * tray_h),
                (0.20, 0.30, 0.42),
            ),
            "front_stop": (
                (tray_t, outer_y, tray_h),
                (payload_local_x + 0.5 * inner_x + 0.5 * tray_t, 0.0, payload_local_z - 0.5 * payload_size[2] + 0.5 * tray_h),
                (0.20, 0.30, 0.42),
            ),
            "left_rail": (
                (outer_x, tray_t, tray_h),
                (payload_local_x, 0.5 * inner_y + 0.5 * tray_t, payload_local_z - 0.5 * payload_size[2] + 0.5 * tray_h),
                (0.12, 0.28, 0.38),
            ),
            "right_rail": (
                (outer_x, tray_t, tray_h),
                (payload_local_x, -0.5 * inner_y - 0.5 * tray_t, payload_local_z - 0.5 * payload_size[2] + 0.5 * tray_h),
                (0.12, 0.28, 0.38),
            ),
        }
        for name, (size, local_pos, color) in tray_specs.items():
            path = f"/World/Robot/OpenTray_{name}"
            world_pos = (
                local_pos[0],
                local_pos[1],
                float(args_cli.torso_z) + local_pos[2],
            )
            _box_body(stage, path, size, world_pos, color, mass=tray_mass, material=material)
            _fixed_joint(stage, f"/World/Robot/OpenTray_{name}_joint", TORSO_PATH, path, local_pos, (0.0, 0.0, 0.0))
    elif str(args_cli.payload_mode) == "cradle_free_box":
        gap_x = float(args_cli.tray_clearance_xy)
        gap_y = float(args_cli.cage_clearance_xy)
        wall_t = float(args_cli.tray_wall_thickness)
        wall_h = float(args_cli.tray_wall_height)
        deck_t = float(args_cli.tray_wall_thickness)
        part_mass = float(args_cli.tray_part_mass)
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
            _box_body(stage, path, size, world_pos, color, mass=part_mass, material=material)
            torso_size = tuple(float(v) for v in args_cli.torso_size)
            scaled_local_pos0 = (
                local_pos[0] / max(torso_size[0], 1e-6),
                local_pos[1] / max(torso_size[1], 1e-6),
                local_pos[2] / max(torso_size[2], 1e-6),
            )
            _fixed_joint(stage, f"/World/Robot/Cradle_{name}_joint", TORSO_PATH, path, scaled_local_pos0, (0.0, 0.0, 0.0))
        adaptive_top_size = (cradle_x, cradle_y, deck_t)
        adaptive_top_local = (
            payload_local_x,
            0.0,
            payload_local_z + 0.5 * payload_size[2] + 0.012 + 0.5 * deck_t,
        )
        adaptive_top_path = "/World/Robot/AdaptiveHold_top_lid"
        _box_body(
            stage,
            adaptive_top_path,
            adaptive_top_size,
            (adaptive_top_local[0], adaptive_top_local[1], float(args_cli.torso_z) + adaptive_top_local[2]),
            (0.58, 0.36, 0.16),
            mass=part_mass,
            material=material,
        )
        _set_collision_enabled(stage, adaptive_top_path, False)
        torso_size = tuple(float(v) for v in args_cli.torso_size)
        adaptive_top_scaled_local = (
            adaptive_top_local[0] / max(torso_size[0], 1e-6),
            adaptive_top_local[1] / max(torso_size[1], 1e-6),
            adaptive_top_local[2] / max(torso_size[2], 1e-6),
        )
        _fixed_joint(
            stage,
            "/World/Robot/AdaptiveHold_top_lid_joint",
            TORSO_PATH,
            adaptive_top_path,
            adaptive_top_scaled_local,
            (0.0, 0.0, 0.0),
        )
    elif str(args_cli.payload_mode) == "side_clamp_free_box":
        tray_t = float(args_cli.tray_wall_thickness)
        payload_local_x = float(args_cli.payload_local_x)
        payload_local_z = float(args_cli.payload_local_z)
        deck_size = (
            payload_size[0] + 0.12,
            payload_size[1] + 2.0 * float(args_cli.clamp_open_gap) + 0.16,
            tray_t,
        )
        deck_local = (payload_local_x, 0.0, payload_local_z - 0.5 * payload_size[2] - 0.5 * tray_t)
        _box_body(
            stage,
            "/World/Robot/ClampDeck",
            deck_size,
            (deck_local[0], deck_local[1], float(args_cli.torso_z) + deck_local[2]),
            (0.16, 0.32, 0.26),
            mass=float(args_cli.tray_part_mass),
            material=material,
        )
        _fixed_joint(stage, "/World/Robot/ClampDeck_joint", TORSO_PATH, "/World/Robot/ClampDeck", deck_local, (0.0, 0.0, 0.0))
        pad_t = float(args_cli.clamp_pad_thickness)
        open_gap = float(args_cli.clamp_open_gap)
        closed_gap = float(args_cli.clamp_closed_gap)
        closure = max(0.0, open_gap - closed_gap)
        pad_size = (payload_size[0] + 0.10, pad_t, payload_size[2] + 0.02)
        open_y = 0.5 * payload_size[1] + open_gap + 0.5 * pad_t
        pad_local_z = payload_local_z
        clamp_pad_size = (0.12, pad_t, 0.46)
        clamp_specs = {
            "LeftClampPad": ((payload_local_x, open_y, pad_local_z), -closure, 0.0, clamp_pad_size),
            "RightClampPad": ((payload_local_x, -open_y, pad_local_z), 0.0, closure, clamp_pad_size),
        }
        for name, (local_pos, lower, upper, color) in clamp_specs.items():
            path = f"/World/Robot/{name}"
            _box_body(
                stage,
                path,
                pad_size,
                (local_pos[0], local_pos[1], float(args_cli.torso_z) + local_pos[2]),
                color,
                mass=float(args_cli.clamp_pad_mass),
                material=material,
            )
            _clamp_joint(stage, f"/World/Robot/{name}_Joint", TORSO_PATH, path, local_pos, lower, upper)
    elif str(args_cli.payload_mode) == "x_cradle_free_box":
        tray_t = float(args_cli.tray_wall_thickness)
        payload_local_x = float(args_cli.payload_local_x)
        payload_local_z = float(args_cli.payload_local_z)
        open_gap = float(args_cli.x_cradle_open_gap)
        closed_gap = float(args_cli.x_cradle_closed_gap)
        closure = max(0.0, open_gap - closed_gap)
        plate_t = float(args_cli.clamp_pad_thickness)
        plate_h = payload_size[2] + 0.02
        plate_y = payload_size[1] + 0.14
        deck_size = (payload_size[0] + 2.0 * open_gap + 0.20, plate_y, tray_t)
        deck_local = (payload_local_x, 0.0, payload_local_z - 0.5 * payload_size[2] - 0.5 * tray_t)
        _box_body(
            stage,
            "/World/Robot/XCradleDeck",
            deck_size,
            (deck_local[0], deck_local[1], float(args_cli.torso_z) + deck_local[2]),
            (0.16, 0.32, 0.26),
            mass=float(args_cli.tray_part_mass),
            material=material,
        )
        _fixed_joint(stage, "/World/Robot/XCradleDeck_joint", TORSO_PATH, "/World/Robot/XCradleDeck", deck_local, (0.0, 0.0, 0.0))
        front_local = (
            payload_local_x + 0.5 * payload_size[0] + open_gap + 0.5 * plate_t,
            0.0,
            payload_local_z,
        )
        _box_body(
            stage,
            "/World/Robot/XCradleFrontStop",
            (plate_t, plate_y, plate_h),
            (front_local[0], front_local[1], float(args_cli.torso_z) + front_local[2]),
            (0.20, 0.30, 0.42),
            mass=float(args_cli.clamp_pad_mass),
            material=material,
        )
        _fixed_joint(stage, "/World/Robot/XCradleFrontStop_joint", TORSO_PATH, "/World/Robot/XCradleFrontStop", front_local, (0.0, 0.0, 0.0))
        rear_local = (
            payload_local_x - 0.5 * payload_size[0] - open_gap - 0.5 * plate_t,
            0.0,
            payload_local_z,
        )
        _box_body(
            stage,
            "/World/Robot/XCradleRearPusher",
            (plate_t, plate_y, plate_h),
            (rear_local[0], rear_local[1], float(args_cli.torso_z) + rear_local[2]),
            (0.12, 0.30, 0.46),
            mass=float(args_cli.clamp_pad_mass),
            material=material,
        )
        _rail_joint(stage, "/World/Robot/XCradleRearPusher_Joint", TORSO_PATH, "/World/Robot/XCradleRearPusher")
        joint = UsdPhysics.PrismaticJoint.Get(stage, "/World/Robot/XCradleRearPusher_Joint")
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in rear_local]))
        joint.GetLowerLimitAttr().Set(0.0)
        joint.GetUpperLimitAttr().Set(float(closure))
        drive = UsdPhysics.DriveAPI.Get(joint.GetPrim(), "linear")
        drive.GetStiffnessAttr().Set(float(args_cli.clamp_drive_stiffness))
        drive.GetDampingAttr().Set(float(args_cli.clamp_drive_damping))
        drive.GetMaxForceAttr().Set(float(args_cli.clamp_drive_max_force))
    else:
        if int(args_cli.grasp_enable_step) > 0:
            shelf_t = 0.04
            shelf_z = (
                float(args_cli.torso_z)
                + float(args_cli.payload_local_z)
                - 0.5 * payload_size[2]
                - float(args_cli.grasp_shelf_clearance)
                - 0.5 * shelf_t
            )
            _box_body(
                stage,
                "/World/GraspPrepShelf",
                (payload_size[0] + 0.10, payload_size[1] + 0.10, shelf_t),
                (float(args_cli.payload_local_x), 0.0, shelf_z),
                (0.24, 0.28, 0.22),
                rigid=False,
                material=material,
            )
    rail_count = max(1, int(args_cli.rail_joint_count))
    previous_body = ANCHOR_PATH
    for rail_idx in range(max(0, rail_count - 1)):
        link_path = f"/World/Robot/RailLink_{rail_idx}"
        _box_body(
            stage,
            link_path,
            (0.05, 0.05, 0.05),
            (0.0, 0.0, float(args_cli.torso_z)),
            (0.10, 0.12, 0.16),
            mass=2.0,
            collision=False,
            material=material,
        )
        _rail_joint(stage, f"/World/Robot/StanceRail_{rail_idx}", previous_body, link_path)
        previous_body = link_path
    final_joint_name = "StanceRail" if rail_count == 1 else f"StanceRail_{rail_count - 1}"
    _rail_joint(stage, f"/World/Robot/{final_joint_name}", previous_body, TORSO_PATH)
    if bool(args_cli.fix_anchor_to_world) or bool(args_cli.replant_anchor_world_joint):
        _fixed_joint_to_world(stage, "/World/Robot/AnchorWorldFixedJoint", ANCHOR_PATH, (0.0, 0.0, float(args_cli.torso_z)))
    support_foot_mode = str(args_cli.support_foot_mode)
    anchor_size = tuple(float(v) for v in args_cli.anchor_size)
    for foot, (x, y) in _foot_offsets().items():
        foot_path = f"/World/SupportFeet/{foot}_foot"
        foot_rigid = support_foot_mode in ("fixed_to_anchor", "x_prismatic_to_anchor", "xz_prismatic_to_anchor")
        _box_body(
            stage,
            foot_path,
            (float(args_cli.foot_length), float(args_cli.foot_width), 0.035),
            (x, y, 0.0175),
            (0.05, 0.11, 0.16),
            mass=float(args_cli.support_foot_mass),
            rigid=foot_rigid,
            material=material,
        )
        if support_foot_mode in ("fixed_to_anchor", "x_prismatic_to_anchor", "xz_prismatic_to_anchor"):
            local_pos0 = (
                x / max(anchor_size[0], 1e-6),
                y / max(anchor_size[1], 1e-6),
                (0.0175 - float(args_cli.torso_z)) / max(anchor_size[2], 1e-6),
            )
            if support_foot_mode == "fixed_to_anchor":
                _fixed_joint(stage, f"/World/Robot/{foot}_support_foot_joint", ANCHOR_PATH, foot_path, local_pos0, (0.0, 0.0, 0.0))
            elif support_foot_mode == "x_prismatic_to_anchor":
                _support_foot_x_joint(stage, f"/World/Robot/{foot}_support_foot_x_joint", ANCHOR_PATH, foot_path, local_pos0)
            else:
                link_path = f"/World/Robot/{foot}_support_foot_x_link"
                _box_body(
                    stage,
                    link_path,
                    (0.035, 0.035, 0.035),
                    (x, y, 0.0175),
                    (0.09, 0.11, 0.15),
                    mass=1.0,
                    collision=False,
                    material=material,
                )
                _support_foot_x_joint(stage, f"/World/Robot/{foot}_support_foot_x_joint", ANCHOR_PATH, link_path, local_pos0)
                _support_foot_z_joint(stage, f"/World/Robot/{foot}_support_foot_z_joint", link_path, foot_path)
            if bool(args_cli.stance_foot_world_lock) and support_foot_mode == "xz_prismatic_to_anchor":
                _disabled_fixed_joint_to_world(
                    stage,
                    f"/World/Robot/{foot}_stance_world_lock_joint",
                    foot_path,
                    (x, y, 0.0175),
                )
    return staged_grasp_joint


def _find_rail_indices(dof_names: list[str]) -> list[int]:
    indices: list[int] = []
    for idx, name in enumerate(dof_names):
        if "StanceRail" in name or "stance" in name.lower() or "rail" in name.lower():
            indices.append(idx)
    if not indices and len(dof_names) == 1:
        indices.append(0)
    if not indices:
        raise RuntimeError(f"Could not find stance rail joints in dofs: {dof_names}")
    return indices


def _find_clamp_indices(dof_names: list[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for idx, name in enumerate(dof_names):
        lowered = name.lower()
        if "leftclamppad" in lowered or ("left" in lowered and "clamp" in lowered):
            indices["left"] = idx
        elif "rightclamppad" in lowered or ("right" in lowered and "clamp" in lowered):
            indices["right"] = idx
    return indices


def _find_cradle_indices(dof_names: list[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for idx, name in enumerate(dof_names):
        lowered = name.lower()
        if "xcradlerearpusher" in lowered or ("cradle" in lowered and "pusher" in lowered):
            indices["rear_pusher"] = idx
    return indices


def _find_support_foot_x_indices(dof_names: list[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for idx, name in enumerate(dof_names):
        lowered = name.lower()
        for foot in FOOT_NAMES:
            if foot in lowered and "support" in lowered and "foot" in lowered and "_x_" in lowered:
                indices[foot] = idx
    return indices


def _find_support_foot_z_indices(dof_names: list[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for idx, name in enumerate(dof_names):
        lowered = name.lower()
        for foot in FOOT_NAMES:
            if foot in lowered and "support" in lowered and "foot" in lowered and "_z_" in lowered:
                indices[foot] = idx
    return indices


def _set_foot_marker_pose(stage: Usd.Stage, foot: str, x: float, y: float) -> None:
    prim = stage.GetPrimAtPath(f"/World/SupportFeet/{foot}_foot")
    _set_xform(
        prim,
        (float(x), float(y), 0.0175),
        (float(args_cli.foot_length), float(args_cli.foot_width), 0.035),
    )


def _smooth01(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def run_scene() -> Path:
    print("[PROGRESS] run_scene entered", flush=True)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_anchored_footstep_carrier_state.csv"
    summary_path = args_cli.output_dir / "core_world_anchored_footstep_carrier_summary.json"
    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    create_new_stage()
    stage = get_current_stage()
    print("[PROGRESS] Fresh stage created", flush=True)
    staged_grasp_joint = design_scene(stage)
    print("[PROGRESS] USD scene designed", flush=True)
    support_foot_paths = {foot: f"/World/SupportFeet/{foot}_foot" for foot in FOOT_NAMES}
    adaptive_hold_collision_paths = (
        ["/World/Robot/AdaptiveHold_top_lid"] if str(args_cli.payload_mode) == "cradle_free_box" else []
    )
    contact_report_state: dict[str, Any] = {
        "available": False,
        "enabled_paths": [],
        "active_feet": set(),
        "event_count": 0,
        "error_count": 0,
        "first_error": None,
    }
    contact_report_subscription = None

    def _on_support_contact_event(contact_headers: Any, contact_data: Any, friction_anchors: Any) -> None:
        del contact_data, friction_anchors
        found_type = getattr(ContactEventType, "CONTACT_FOUND", None)
        lost_type = getattr(ContactEventType, "CONTACT_LOST", None)
        for contact_header in contact_headers:
            try:
                collider0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))
                collider1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1))
                pair = (collider0, collider1)
                is_ground_contact = any(path.startswith("/World/Ground") for path in pair)
                if not is_ground_contact:
                    continue
                for foot, foot_path in support_foot_paths.items():
                    if foot_path not in pair:
                        continue
                    if found_type is not None and contact_header.type == found_type:
                        contact_report_state["active_feet"].add(foot)
                        contact_report_state["event_count"] = int(contact_report_state["event_count"]) + 1
                    elif lost_type is not None and contact_header.type == lost_type:
                        contact_report_state["active_feet"].discard(foot)
                        contact_report_state["event_count"] = int(contact_report_state["event_count"]) + 1
            except Exception as contact_exc:
                contact_report_state["error_count"] = int(contact_report_state["error_count"]) + 1
                if contact_report_state["first_error"] is None:
                    contact_report_state["first_error"] = f"{type(contact_exc).__name__}: {contact_exc}"

    if bool(args_cli.enable_support_foot_contact_report):
        try:
            contact_paths = ["/World/Ground"] + list(support_foot_paths.values())
            contact_report_state["enabled_paths"] = _enable_contact_report_api(
                stage,
                contact_paths,
                float(args_cli.support_foot_contact_report_threshold),
            )
            contact_report_subscription = get_physics_simulation_interface().subscribe_physics_contact_report_events(
                _on_support_contact_event
            )
            contact_report_state["available"] = True
            print(
                "[PROGRESS] Support-foot contact report enabled for "
                f"{contact_report_state['enabled_paths']}",
                flush=True,
            )
        except Exception as contact_setup_exc:
            contact_report_state["error_count"] = int(contact_report_state["error_count"]) + 1
            contact_report_state["first_error"] = f"{type(contact_setup_exc).__name__}: {contact_setup_exc}"
            print(f"[WARN] Support-foot contact report setup failed: {contact_report_state['first_error']}", flush=True)

    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    print("[PROGRESS] Core World created", flush=True)
    payload_active = str(args_cli.payload_mode) != "none"
    robot_prim_path = ANCHOR_PATH if bool(args_cli.anchor_as_articulation_root) else ROBOT_PATH
    robot = SingleArticulation(prim_path=robot_prim_path, name="anchored_footstep_carrier")
    torso_prim = SingleRigidPrim(prim_path=TORSO_PATH, name="anchored_torso")
    payload = SingleRigidPrim(prim_path=BOX_PATH, name="anchored_payload") if payload_active else None
    anchor = SingleRigidPrim(prim_path=ANCHOR_PATH, name="stance_anchor")
    support_feet: dict[str, SingleRigidPrim] = {}
    if str(args_cli.support_foot_mode) in ("fixed_to_anchor", "x_prismatic_to_anchor", "xz_prismatic_to_anchor"):
        support_feet = {
            foot: SingleRigidPrim(prim_path=f"/World/SupportFeet/{foot}_foot", name=f"{foot}_support_foot")
            for foot in FOOT_NAMES
        }
    cradle_rear_stop = None
    cradle_front_stop = None
    if str(args_cli.payload_mode) == "cradle_free_box":
        cradle_rear_stop = SingleRigidPrim(prim_path="/World/Robot/Cradle_rear_stop", name="cradle_rear_stop")
        cradle_front_stop = SingleRigidPrim(prim_path="/World/Robot/Cradle_front_stop", name="cradle_front_stop")
    print("[PROGRESS] Core prim wrappers created", flush=True)
    world.reset()
    print("[PROGRESS] World reset complete", flush=True)
    robot.initialize()
    print("[PROGRESS] Robot articulation initialized", flush=True)
    torso_prim.initialize()
    if payload is not None:
        payload.initialize()
    anchor.initialize()
    for support_foot in support_feet.values():
        support_foot.initialize()
    if cradle_rear_stop is not None and cradle_front_stop is not None:
        cradle_rear_stop.initialize()
        cradle_front_stop.initialize()
    print("[PROGRESS] Rigid prims initialized", flush=True)
    anchor_world_joint = None
    if bool(args_cli.replant_anchor_world_joint):
        anchor_world_joint = UsdPhysics.FixedJoint.Get(stage, "/World/Robot/AnchorWorldFixedJoint")
    stance_world_lock_joints: dict[str, UsdPhysics.FixedJoint] = {}
    if bool(args_cli.stance_foot_world_lock):
        for foot in FOOT_NAMES:
            joint = UsdPhysics.FixedJoint.Get(stage, f"/World/Robot/{foot}_stance_world_lock_joint")
            if joint.GetPrim().IsValid():
                stance_world_lock_joints[foot] = joint

    dof_names = list(robot.dof_names)
    rail_indices = _find_rail_indices(dof_names)
    clamp_indices = _find_clamp_indices(dof_names)
    cradle_indices = _find_cradle_indices(dof_names)
    clamp_drives: dict[str, UsdPhysics.DriveAPI] = {}
    for clamp_name in clamp_indices:
        joint_name = "LeftClampPad_Joint" if clamp_name == "left" else "RightClampPad_Joint"
        joint = UsdPhysics.PrismaticJoint.Get(stage, f"/World/Robot/{joint_name}")
        if joint.GetPrim().IsValid():
            clamp_drives[clamp_name] = UsdPhysics.DriveAPI.Get(joint.GetPrim(), "linear")
    cradle_drives: dict[str, UsdPhysics.DriveAPI] = {}
    if "rear_pusher" in cradle_indices:
        joint = UsdPhysics.PrismaticJoint.Get(stage, "/World/Robot/XCradleRearPusher_Joint")
        if joint.GetPrim().IsValid():
            cradle_drives["rear_pusher"] = UsdPhysics.DriveAPI.Get(joint.GetPrim(), "linear")
    support_foot_x_indices = _find_support_foot_x_indices(dof_names)
    support_foot_z_indices = _find_support_foot_z_indices(dof_names)
    initial_torso = _pose_wxyz(torso_prim)
    initial_payload = _pose_wxyz(payload) if payload is not None else initial_torso
    initial_anchor = _pose_wxyz(anchor)
    initial_support_feet = {foot: _pose_wxyz(support_foot) for foot, support_foot in support_feet.items()}
    initial_rear_stop = _pose_wxyz(cradle_rear_stop) if cradle_rear_stop is not None else None
    initial_front_stop = _pose_wxyz(cradle_front_stop) if cradle_front_stop is not None else None
    target_x = float(args_cli.target_x)
    drive_start_step = int(args_cli.settle_steps) + max(0, int(args_cli.probe_steps))
    support_foot_continuity_start_step = drive_start_step + max(
        0,
        int(args_cli.support_foot_continuity_grace_steps),
    )
    rail_count = max(1, len(rail_indices))
    rail_capacity = abs(float(args_cli.rail_upper)) * float(rail_count)
    step_length = min(abs(float(args_cli.step_length)), rail_capacity)
    cycle_count = max(1, int(math.ceil(abs(target_x) / max(step_length, 1e-5)))) if abs(target_x) > 0.0 else 1
    if bool(args_cli.fix_anchor_to_world) and not bool(args_cli.replant_anchor_world_joint):
        cycle_count = 1
    if bool(args_cli.disable_support_reposition) and str(args_cli.support_foot_mode) != "xz_prismatic_to_anchor":
        cycle_count = 1
    stride = min(abs(target_x) / float(cycle_count), rail_capacity) if abs(target_x) > 0.0 else step_length
    direction = 1.0 if target_x >= 0.0 else -1.0
    rail_direction = direction * float(args_cli.rail_target_direction_scale)
    anchor_pose_write_count = 0
    foot_pose_write_count = 0
    last_cycle = -1
    stop_latched = False
    online_probe_adaptive_support_decided = False
    online_probe_adaptive_hold_decided = False
    effective_support_step_height = float(args_cli.support_foot_step_height)
    effective_support_double_support_fraction = float(args_cli.support_foot_double_support_fraction)
    effective_support_stance_x = float(args_cli.support_foot_stance_x)
    effective_support_swing_x = float(args_cli.support_foot_swing_x)
    effective_hold_closure_fraction = 1.0
    latched_per_joint_target: float | None = None
    latched_support_foot_x_targets: dict[str, float] | None = None
    locked_stance_feet: set[str] = set()
    frozen_locked_support_foot_x_targets: dict[str, float] = {}
    frozen_locked_support_foot_z_targets: dict[str, float] = {}
    frozen_commanded_stance_feet: set[str] = set()
    frozen_commanded_support_foot_x_targets: dict[str, float] = {}
    frozen_commanded_support_foot_z_targets: dict[str, float] = {}
    staged_grasp_attached = str(args_cli.payload_mode) != "staged_grasp_constraint"
    staged_grasp_attach_step: int | None = None
    clamp_travel = max(0.0, float(args_cli.clamp_open_gap) - float(args_cli.clamp_closed_gap))
    cradle_travel = max(0.0, float(args_cli.x_cradle_open_gap) - float(args_cli.x_cradle_closed_gap))
    disable_internal_body_collision = str(args_cli.payload_mode) == "cradle_free_box"
    kinematic_support_root = bool(args_cli.anchor_as_articulation_root) and not bool(args_cli.fix_anchor_to_world)
    support_feet_fixed = str(args_cli.support_foot_mode) == "fixed_to_anchor"
    support_feet_x_prismatic = str(args_cli.support_foot_mode) == "x_prismatic_to_anchor"
    support_feet_xz_prismatic = str(args_cli.support_foot_mode) == "xz_prismatic_to_anchor"
    if bool(args_cli.fix_anchor_to_world):
        carrier_claim = "free_torso_pulled_by_driven_prismatic_joint_to_world_fixed_support_frame"
    elif support_feet_fixed and bool(args_cli.disable_support_reposition):
        carrier_claim = "free_torso_pulled_by_driven_prismatic_joint_to_ground_contact_support_feet"
    elif support_feet_x_prismatic and bool(args_cli.use_support_foot_drive):
        carrier_claim = "free_torso_and_payload_moved_by_x_prismatic_support_feet_against_ground_contact"
    elif support_feet_xz_prismatic and bool(args_cli.use_support_foot_drive):
        carrier_claim = "free_torso_and_payload_moved_by_alternating_xz_prismatic_support_feet_against_ground_contact"
    else:
        carrier_claim = "free_torso_pulled_by_driven_prismatic_joint_to_replanted_dynamic_stance_anchor"

    summary = {
        "scene_type": f"core_world_anchored_footstep_carrier_{args_cli.payload_mode}",
        "success_claim": "anchored_support_footstep_diagnostic_not_full_robot_policy_or_unknown_object_carrying",
        "carrier_claim": carrier_claim,
        "articulated_carrier_enabled": True,
        "articulated_joint_count": int(robot.num_dof),
        "payload_spawned": bool(payload_active),
        "no_box_support_smoke": bool(not payload_active),
        "payload_metric_proxy": "torso_pose_when_payload_mode_none" if not payload_active else None,
        "foot_contact_drive_enabled": True,
        "stance_anchor_kinematic": bool(kinematic_support_root),
        "stance_anchor_dynamic_high_mass": True,
        "stance_anchor_fixed_to_world": bool(args_cli.fix_anchor_to_world),
        "stance_anchor_as_articulation_root": bool(args_cli.anchor_as_articulation_root),
        "support_foot_mode": str(args_cli.support_foot_mode),
        "support_feet_fixed_to_anchor": bool(str(args_cli.support_foot_mode) == "fixed_to_anchor"),
        "support_foot_mass_kg": float(args_cli.support_foot_mass),
        "support_foot_joint_count": int(
            len(FOOT_NAMES) * 2
            if str(args_cli.support_foot_mode) == "xz_prismatic_to_anchor"
            else len(FOOT_NAMES)
            if str(args_cli.support_foot_mode) in ("fixed_to_anchor", "x_prismatic_to_anchor")
            else 0
        ),
        "support_foot_x_joint_indices": {key: int(value) for key, value in support_foot_x_indices.items()},
        "support_foot_x_joint_count": int(len(support_foot_x_indices)),
        "support_foot_z_joint_indices": {key: int(value) for key, value in support_foot_z_indices.items()},
        "support_foot_z_joint_count": int(len(support_foot_z_indices)),
        "support_foot_x_lower_m": float(args_cli.support_foot_x_lower),
        "support_foot_x_upper_m": float(args_cli.support_foot_x_upper),
        "support_foot_z_lower_m": float(args_cli.support_foot_z_lower),
        "support_foot_z_upper_m": float(args_cli.support_foot_z_upper),
        "use_support_foot_drive": bool(args_cli.use_support_foot_drive),
        "support_foot_drive_direction_scale": float(args_cli.support_foot_drive_direction_scale),
        "support_foot_placement_mode": str(args_cli.support_foot_placement_mode),
        "support_foot_placement_controller_enabled": bool(support_feet_xz_prismatic and args_cli.use_support_foot_drive),
        "support_foot_directional_placement": bool(
            support_feet_xz_prismatic
            and args_cli.use_support_foot_drive
            and str(args_cli.support_foot_placement_mode) == "alternating_directional_x"
        ),
        "stance_foot_world_lock_enabled": bool(args_cli.stance_foot_world_lock),
        "stance_foot_world_lock_joint_count": int(len(stance_world_lock_joints)),
        "stance_foot_world_lock_switch_count": 0,
        "stance_foot_world_lock_pose_update_count": 0,
        "stance_foot_world_lock_active_feet": [],
        "freeze_locked_stance_foot_targets_enabled": bool(args_cli.freeze_locked_stance_foot_targets),
        "freeze_locked_stance_foot_target_count": 0,
        "freeze_commanded_stance_foot_targets_enabled": bool(args_cli.freeze_commanded_stance_foot_targets),
        "freeze_commanded_stance_foot_target_count": 0,
        "freeze_commanded_stance_foot_target_switch_count": 0,
        "freeze_commanded_stance_foot_active_feet": [],
        "planted_stance_rail_propulsion_enabled": bool(args_cli.planted_stance_rail_propulsion),
        "planted_stance_rail_propulsion_steps": 0,
        "support_foot_step_height_m": float(args_cli.support_foot_step_height),
        "support_foot_stance_x_m": float(args_cli.support_foot_stance_x),
        "support_foot_swing_x_m": float(args_cli.support_foot_swing_x),
        "support_foot_contact_z_threshold_m": float(args_cli.support_foot_contact_z_threshold),
        "support_foot_contact_report_requested": bool(args_cli.enable_support_foot_contact_report),
        "support_foot_contact_report_available": bool(contact_report_state["available"]),
        "support_foot_contact_report_threshold": float(args_cli.support_foot_contact_report_threshold),
        "support_foot_contact_report_enabled_paths": list(contact_report_state["enabled_paths"]),
        "support_foot_contact_report_event_count": 0,
        "support_foot_contact_report_error_count": 0,
        "support_foot_contact_report_first_error": None,
        "support_foot_effort_contact_threshold": float(args_cli.support_foot_effort_contact_threshold),
        "support_foot_double_support_fraction": float(args_cli.support_foot_double_support_fraction),
        "support_foot_continuity_grace_steps": int(args_cli.support_foot_continuity_grace_steps),
        "support_foot_continuity_start_step": int(support_foot_continuity_start_step),
        "feedback_step_controller_enabled": bool(args_cli.feedback_step_controller),
        "feedback_step_x_gain": float(args_cli.feedback_step_x_gain),
        "feedback_step_x_limit_m": float(args_cli.feedback_step_x_limit),
        "feedback_step_tilt_gain": float(args_cli.feedback_step_tilt_gain),
        "feedback_step_tilt_limit_m": float(args_cli.feedback_step_tilt_limit),
        "feedback_step_applied_steps": 0,
        "max_abs_feedback_step_x_adjustment_m": 0.0,
        "max_abs_feedback_step_tilt_adjustment_m": 0.0,
        "online_probe_adaptive_support_enabled": bool(args_cli.enable_online_probe_adaptive_support),
        "online_probe_adaptive_support_decision_applied": False,
        "online_probe_adaptive_support_decision_step": None,
        "online_probe_adaptive_support_uses_hidden_ground_truth": False,
        "online_probe_adaptive_support_risk_score": None,
        "online_probe_adaptive_support_risk_bucket": None,
        "online_probe_adaptive_support_profile": None,
        "online_probe_adaptive_support_step_height_m": None,
        "online_probe_adaptive_support_double_support_fraction": None,
        "online_probe_adaptive_support_stance_x_m": None,
        "online_probe_adaptive_support_swing_x_m": None,
        "online_probe_adaptive_support_medium_threshold": float(args_cli.online_probe_adaptive_medium_threshold),
        "online_probe_adaptive_support_high_threshold": float(args_cli.online_probe_adaptive_high_threshold),
        "online_probe_adaptive_hold_enabled": bool(args_cli.enable_online_probe_adaptive_hold),
        "online_probe_adaptive_hold_decision_applied": False,
        "online_probe_adaptive_hold_decision_step": None,
        "online_probe_adaptive_hold_uses_hidden_ground_truth": False,
        "online_probe_adaptive_hold_risk_score": None,
        "online_probe_adaptive_hold_risk_bucket": None,
        "online_probe_adaptive_hold_profile": None,
        "online_probe_adaptive_hold_closure_fraction": None,
        "online_probe_adaptive_hold_actuated": bool(str(args_cli.payload_mode) in ("side_clamp_free_box", "x_cradle_free_box")),
        "online_probe_adaptive_hold_collision_available": bool(adaptive_hold_collision_paths),
        "online_probe_adaptive_hold_collision_paths": list(adaptive_hold_collision_paths),
        "online_probe_adaptive_hold_collision_enabled": False,
        "online_probe_adaptive_hold_collision_update_count": 0,
        "online_probe_adaptive_hold_low_closure_fraction": float(args_cli.online_low_hold_closure_fraction),
        "online_probe_adaptive_hold_medium_closure_fraction": float(args_cli.online_medium_hold_closure_fraction),
        "online_probe_adaptive_hold_high_closure_fraction": float(args_cli.online_high_hold_closure_fraction),
        "alternating_support_foot_drive": bool(support_feet_xz_prismatic and args_cli.use_support_foot_drive),
        "disable_support_reposition": bool(args_cli.disable_support_reposition),
        "initial_anchor_x_m": float(initial_anchor[0]),
        "final_anchor_travel_x_m": 0.0,
        "max_abs_anchor_travel_x_m": 0.0,
        "max_anchor_travel_xy_m": 0.0,
        "initial_support_foot_x_m": {foot: float(pose[0]) for foot, pose in initial_support_feet.items()},
        "support_foot_min_z_m": None,
        "support_foot_max_z_m": None,
        "max_abs_support_foot_travel_x_m": 0.0,
        "max_support_foot_travel_xy_m": 0.0,
        "final_support_foot_travel_x_m": {},
        "max_actual_support_foot_lift_m": 0.0,
        "per_foot_max_actual_lift_m": {foot: 0.0 for foot in FOOT_NAMES},
        "per_foot_min_z_m": {},
        "per_foot_max_z_m": {},
        "per_foot_near_ground_steps": {foot: 0 for foot in FOOT_NAMES},
        "per_foot_max_near_ground_xy_slip_m": {foot: 0.0 for foot in FOOT_NAMES},
        "per_foot_max_near_ground_xy_speed_mps": {foot: 0.0 for foot in FOOT_NAMES},
        "min_near_ground_foot_count": None,
        "max_near_ground_foot_count": 0,
        "near_ground_zero_steps": 0,
        "near_ground_lt2_steps": 0,
        "min_drive_near_ground_foot_count": None,
        "drive_near_ground_zero_steps": 0,
        "drive_near_ground_lt2_steps": 0,
        "per_foot_contact_report_steps": {foot: 0 for foot in FOOT_NAMES},
        "min_contact_report_foot_count": None,
        "max_contact_report_foot_count": 0,
        "contact_report_zero_steps": 0,
        "contact_report_lt2_steps": 0,
        "min_drive_contact_report_foot_count": None,
        "drive_contact_report_zero_steps": 0,
        "drive_contact_report_lt2_steps": 0,
        "min_commanded_stance_contact_report_foot_count": None,
        "commanded_stance_contact_report_lt2_steps": 0,
        "support_foot_effort_available": False,
        "support_foot_effort_read_error_count": 0,
        "support_foot_effort_first_error": None,
        "per_foot_max_support_x_measured_effort": {foot: 0.0 for foot in FOOT_NAMES},
        "per_foot_max_support_z_measured_effort": {foot: 0.0 for foot in FOOT_NAMES},
        "per_foot_max_support_measured_effort": {foot: 0.0 for foot in FOOT_NAMES},
        "min_drive_effort_supported_foot_count": None,
        "drive_effort_supported_zero_steps": 0,
        "drive_effort_supported_lt2_steps": 0,
        "min_commanded_stance_effort_supported_foot_count": None,
        "commanded_stance_effort_supported_lt2_steps": 0,
        "min_commanded_stance_near_ground_foot_count": None,
        "commanded_stance_near_ground_lt2_steps": 0,
        "min_support_polygon_margin_x_m": None,
        "min_support_polygon_margin_y_m": None,
        "min_support_polygon_margin_m": None,
        "max_support_foot_x_joint_motion_m": 0.0,
        "max_support_foot_z_joint_motion_m": 0.0,
        "max_commanded_support_foot_lift_m": 0.0,
        "per_foot_max_commanded_x_m": {foot: 0.0 for foot in FOOT_NAMES},
        "per_foot_max_commanded_z_m": {foot: 0.0 for foot in FOOT_NAMES},
        "final_support_foot_x_joint_target_m": 0.0,
        "final_support_foot_x_joint_target_m_by_foot": {},
        "final_support_foot_z_joint_target_m_by_foot": {},
        "replant_anchor_world_joint": bool(args_cli.replant_anchor_world_joint),
        "cumulative_cycle_target": bool(args_cli.cumulative_cycle_target),
        "articulation_root_path": str(robot_prim_path),
        "payload_mode": str(args_cli.payload_mode),
        "box_seed": args_cli.box_seed,
        "payload_randomized": bool(args_cli.randomize_payload),
        "payload_mass_requested_kg": float(args_cli.payload_mass_requested),
        "payload_mass_range_kg": [float(v) for v in args_cli.payload_mass_range],
        "payload_size_requested_m": [float(v) for v in args_cli.payload_size_requested],
        "payload_size_jitter_fraction": float(args_cli.payload_size_jitter),
        "payload_com_offset_range_m": [float(v) for v in args_cli.payload_com_offset_range],
        "payload_com_offset_m": [float(v) for v in args_cli.payload_com_offset_m],
        "attached": bool(staged_grasp_attached),
        "staged_grasp_constraint_enabled": bool(str(args_cli.payload_mode) == "staged_grasp_constraint"),
        "staged_grasp_attach_step": None,
        "grasp_enable_step": int(args_cli.grasp_enable_step),
        "grasp_shelf_clearance_m": float(args_cli.grasp_shelf_clearance),
        "tray_clearance_xy_m": float(args_cli.tray_clearance_xy),
        "tray_wall_height_m": float(args_cli.tray_wall_height),
        "tray_wall_thickness_m": float(args_cli.tray_wall_thickness),
        "tray_part_mass_kg": float(args_cli.tray_part_mass),
        "clamp_open_gap_m": float(args_cli.clamp_open_gap),
        "clamp_closed_gap_m": float(args_cli.clamp_closed_gap),
        "clamp_travel_m": float(clamp_travel),
        "clamp_close_start_step": int(args_cli.clamp_close_start_step),
        "clamp_close_steps": int(args_cli.clamp_close_steps),
        "x_cradle_open_gap_m": float(args_cli.x_cradle_open_gap),
        "x_cradle_closed_gap_m": float(args_cli.x_cradle_closed_gap),
        "x_cradle_travel_m": float(cradle_travel),
        "device": args_cli.device,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "probe_steps_requested": int(args_cli.probe_steps),
        "probe_mode": str(args_cli.probe_mode),
        "probe_x_amplitude_m": float(args_cli.probe_x_amplitude),
        "probe_z_amplitude_m": float(args_cli.probe_z_amplitude),
        "probe_start_step": int(args_cli.settle_steps) if int(args_cli.probe_steps) > 0 else None,
        "probe_end_step": int(drive_start_step) if int(args_cli.probe_steps) > 0 else None,
        "probe_belief_available": False,
        "probe_belief_source": "not_computed",
        "probe_belief_uses_hidden_ground_truth": False,
        "probe_compliance_proxy": None,
        "probe_lag_proxy": None,
        "probe_support_foot_x_tracking_proxy": None,
        "probe_support_foot_z_tracking_proxy": None,
        "probe_support_foot_x_effort_proxy": None,
        "probe_support_foot_z_effort_proxy": None,
        "probe_risk_score": None,
        "probe_load_risk_bucket": "not_computed",
        "probe_recommended_carry_adjustment": None,
        "probe_belief_policy_action_applied": False,
        "belief_compliance_low_threshold": float(args_cli.belief_compliance_low_threshold),
        "belief_compliance_high_threshold": float(args_cli.belief_compliance_high_threshold),
        "max_probe_torso_travel_x_m": 0.0,
        "max_probe_torso_travel_z_m": 0.0,
        "max_probe_payload_travel_x_m": 0.0,
        "max_probe_payload_travel_z_m": 0.0,
        "max_probe_payload_relative_error_m": 0.0,
        "max_probe_support_foot_x_tracking_error_m": 0.0,
        "mean_probe_support_foot_x_tracking_error_m": None,
        "probe_support_foot_x_tracking_error_samples": 0,
        "max_probe_support_foot_z_tracking_error_m": 0.0,
        "mean_probe_support_foot_z_tracking_error_m": None,
        "probe_support_foot_z_tracking_error_samples": 0,
        "probe_joint_effort_available": False,
        "probe_joint_effort_read_error_count": 0,
        "probe_joint_effort_first_error": None,
        "max_probe_support_foot_x_measured_effort": 0.0,
        "mean_probe_support_foot_x_measured_effort": None,
        "probe_support_foot_x_measured_effort_samples": 0,
        "max_probe_support_foot_z_measured_effort": 0.0,
        "mean_probe_support_foot_z_measured_effort": None,
        "probe_support_foot_z_measured_effort_samples": 0,
        "final_probe_payload_lag_x_m": None,
        "final_probe_payload_lag_z_m": None,
        "dof_names": dof_names,
        "rail_joint_indices": [int(idx) for idx in rail_indices],
        "clamp_joint_indices": {key: int(value) for key, value in clamp_indices.items()},
        "cradle_joint_indices": {key: int(value) for key, value in cradle_indices.items()},
        "rail_joint_count": int(rail_count),
        "rail_capacity_m": float(rail_capacity),
        "rail_target_direction_scale": float(args_cli.rail_target_direction_scale),
        "stop_threshold_m": float(args_cli.stop_threshold),
        "stop_latched": False,
        "target_x_m": target_x,
        "step_length_m": float(step_length),
        "cycle_count": int(cycle_count),
        "stride_m": float(stride),
        "payload_mass_kg": float(args_cli.payload_mass),
        "payload_size_m": [float(v) for v in args_cli.payload_size],
        "payload_local_x_m": float(args_cli.payload_local_x),
        "payload_local_z_m": float(args_cli.payload_local_z),
        "torso_mass_kg": float(args_cli.torso_mass),
        "torso_z_m": float(args_cli.torso_z),
        "stance_half_length_m": float(args_cli.stance_half_length),
        "stance_half_width_m": float(args_cli.stance_half_width),
        "cradle_disable_internal_body_collision": bool(disable_internal_body_collision),
        "cradle_initial_rear_stop_x_m": float(initial_rear_stop[0]) if initial_rear_stop is not None else None,
        "cradle_initial_front_stop_x_m": float(initial_front_stop[0]) if initial_front_stop is not None else None,
        "cradle_initial_rear_surface_gap_x_m": (
            (float(initial_payload[0]) - 0.5 * float(args_cli.payload_size[0]))
            - (float(initial_rear_stop[0]) + 0.5 * float(args_cli.tray_wall_thickness))
            if initial_rear_stop is not None
            else None
        ),
        "cradle_initial_front_surface_gap_x_m": (
            (float(initial_front_stop[0]) - 0.5 * float(args_cli.tray_wall_thickness))
            - (float(initial_payload[0]) + 0.5 * float(args_cli.payload_size[0]))
            if initial_front_stop is not None
            else None
        ),
        "cage_clearance_xy_m": float(args_cli.cage_clearance_xy),
        "cage_clearance_z_m": float(args_cli.cage_clearance_z),
        "cage_wall_thickness_m": float(args_cli.cage_wall_thickness),
        "root_pose_write_count": 0,
        "root_velocity_write_count": 0,
        "root_angular_velocity_write_count": 0,
        "body_root_pose_write_count": 0,
        "body_root_velocity_command_count": 0,
        "box_pose_write_count": 0,
        "payload_pose_write_count": 0,
        "stance_anchor_pose_write_count": 0,
        "support_root_pose_write_count": 0,
        "anchor_world_joint_retarget_count": 0,
        "foot_pose_write_count": 0,
        "fall_events": 0,
        "box_drop_events": 0,
        "nonfinite_state_events": 0,
        "max_tilt_rad": 0.0,
        "min_torso_z_m": float(initial_torso[2]),
        "min_payload_z_m": float(initial_payload[2]),
        "max_torso_travel_x_m": 0.0,
        "max_payload_travel_x_m": 0.0,
        "max_abs_torso_travel_x_m": 0.0,
        "max_abs_payload_travel_x_m": 0.0,
        "post_settle_baseline_step": None,
        "post_settle_baseline_torso_x_m": None,
        "post_settle_baseline_payload_x_m": None,
        "max_post_settle_torso_travel_x_m": 0.0,
        "max_post_settle_payload_travel_x_m": 0.0,
        "max_abs_post_settle_torso_travel_x_m": 0.0,
        "max_abs_post_settle_payload_travel_x_m": 0.0,
        "max_target_directed_post_settle_torso_travel_m": 0.0,
        "max_target_directed_post_settle_payload_travel_m": 0.0,
        "max_post_settle_payload_relative_error_m": 0.0,
        "final_post_settle_torso_travel_x_m": None,
        "final_post_settle_payload_travel_x_m": None,
        "final_post_settle_payload_target_distance_x_m": None,
        "final_post_settle_payload_relative_error_m": None,
        "post_settle_payload_travel_loss_after_peak_m": None,
        "final_target_distance_x_m": None,
        "final_payload_target_distance_x_m": None,
        "payload_relative_error_m": 0.0,
        "max_payload_relative_offset_error_m": 0.0,
        "max_joint_motion_m": 0.0,
        "max_rail_joint_motion_m": 0.0,
        "max_clamp_joint_motion_m": 0.0,
        "max_cradle_joint_motion_m": 0.0,
        "max_commanded_clamp_target_m": 0.0,
        "final_commanded_clamp_target_m": 0.0,
        "clamp_drive_target_update_count": 0,
        "max_commanded_cradle_target_m": 0.0,
        "final_commanded_cradle_target_m": 0.0,
        "cradle_drive_target_update_count": 0,
        "error": None,
    }
    post_settle_torso_x: float | None = None
    post_settle_payload_x: float | None = None

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "cycle",
                    "phase",
                    "torso_x",
                    "payload_x",
                    "target_distance_x",
                    "post_settle_torso_x",
                    "post_settle_payload_x",
                    "post_settle_relative_error_x",
                    "rail_target",
                    "joint_motion",
                    "tilt",
                    "fall",
                    "drop",
                ]
            )
            initial_joint_positions = _as_numpy(robot.get_joint_positions())
            current_targets = _as_numpy(robot.get_joint_positions())
            previous_support_foot_poses = dict(initial_support_feet)
            near_ground_reference_xy: dict[str, tuple[float, float] | None] = {foot: None for foot in FOOT_NAMES}
            for step in range(int(args_cli.steps)):
                if (
                    bool(args_cli.enable_online_probe_adaptive_support)
                    and not online_probe_adaptive_support_decided
                    and int(args_cli.probe_steps) > 0
                    and step >= drive_start_step
                ):
                    probe_belief = _probe_belief(summary)
                    summary.update(probe_belief)
                    profile = _select_online_support_profile(args_cli, probe_belief.get("probe_risk_score"))
                    effective_support_step_height = float(profile["step_height"])
                    effective_support_double_support_fraction = float(profile["double_support_fraction"])
                    effective_support_stance_x = float(profile["stance_x"])
                    effective_support_swing_x = float(profile["swing_x"])
                    online_probe_adaptive_support_decided = True
                    summary["probe_belief_policy_action_applied"] = True
                    summary["online_probe_adaptive_support_decision_applied"] = True
                    summary["online_probe_adaptive_support_decision_step"] = int(step)
                    summary["online_probe_adaptive_support_risk_score"] = probe_belief.get("probe_risk_score")
                    summary["online_probe_adaptive_support_risk_bucket"] = str(profile["bucket"])
                    summary["online_probe_adaptive_support_profile"] = str(profile["profile"])
                    summary["online_probe_adaptive_support_step_height_m"] = effective_support_step_height
                    summary["online_probe_adaptive_support_double_support_fraction"] = (
                        effective_support_double_support_fraction
                    )
                    summary["online_probe_adaptive_support_stance_x_m"] = effective_support_stance_x
                    summary["online_probe_adaptive_support_swing_x_m"] = effective_support_swing_x
                    print(
                        "[EVENT] online_probe_adaptive_support "
                        f"step={step} risk={probe_belief.get('probe_risk_score')} "
                        f"bucket={profile['bucket']} profile={profile['profile']} "
                        f"step_height={effective_support_step_height:.4f} "
                        f"double_support={effective_support_double_support_fraction:.4f} "
                        f"stance_x={effective_support_stance_x:.4f} "
                        f"swing_x={effective_support_swing_x:.4f}",
                        flush=True,
                    )
                if (
                    bool(args_cli.enable_online_probe_adaptive_hold)
                    and not online_probe_adaptive_hold_decided
                    and int(args_cli.probe_steps) > 0
                    and step >= drive_start_step
                ):
                    probe_belief = _probe_belief(summary)
                    summary.update(probe_belief)
                    hold_profile = _select_online_hold_profile(args_cli, probe_belief.get("probe_risk_score"))
                    effective_hold_closure_fraction = float(hold_profile["closure_fraction"])
                    online_probe_adaptive_hold_decided = True
                    summary["probe_belief_policy_action_applied"] = True
                    summary["online_probe_adaptive_hold_decision_applied"] = True
                    summary["online_probe_adaptive_hold_decision_step"] = int(step)
                    summary["online_probe_adaptive_hold_risk_score"] = probe_belief.get("probe_risk_score")
                    summary["online_probe_adaptive_hold_risk_bucket"] = str(hold_profile["bucket"])
                    summary["online_probe_adaptive_hold_profile"] = str(hold_profile["profile"])
                    summary["online_probe_adaptive_hold_closure_fraction"] = effective_hold_closure_fraction
                    enable_hold_collision = bool(hold_profile["bucket"] != "low")
                    if adaptive_hold_collision_paths:
                        updated_count = 0
                        for collision_path in adaptive_hold_collision_paths:
                            if _set_collision_enabled(stage, collision_path, enable_hold_collision):
                                updated_count += 1
                        summary["online_probe_adaptive_hold_collision_enabled"] = enable_hold_collision
                        summary["online_probe_adaptive_hold_collision_update_count"] = int(
                            summary["online_probe_adaptive_hold_collision_update_count"]
                        ) + updated_count
                    print(
                        "[EVENT] online_probe_adaptive_hold "
                        f"step={step} risk={probe_belief.get('probe_risk_score')} "
                        f"bucket={hold_profile['bucket']} profile={hold_profile['profile']} "
                        f"closure_fraction={effective_hold_closure_fraction:.4f} "
                        f"actuated={summary['online_probe_adaptive_hold_actuated']} "
                        f"collision_enabled={summary['online_probe_adaptive_hold_collision_enabled']}",
                        flush=True,
                    )
                torso_before = _pose_wxyz(torso_prim)
                torso_before_travel_x = float(torso_before[0]) - float(initial_torso[0])
                if str(args_cli.payload_mode) == "staged_grasp_constraint" and not staged_grasp_attached:
                    cycle = 0
                    phase = 0.0
                    rail_target = 0.0
                    phase_name = "staged_grasp_prepare"
                    if step >= int(args_cli.grasp_enable_step):
                        box_before = _pose_wxyz(payload)
                        grasp_local_pos0 = (
                            float(box_before[0] - torso_before[0]),
                            float(box_before[1] - torso_before[1]),
                            float(box_before[2] - torso_before[2]),
                        )
                        staged_grasp_joint = _fixed_joint(
                            stage,
                            "/World/Robot/StagedGraspJoint",
                            TORSO_PATH,
                            BOX_PATH,
                            grasp_local_pos0,
                            (0.0, 0.0, 0.0),
                        )
                        staged_grasp_attached = True
                        staged_grasp_attach_step = step
                        phase_name = "staged_grasp_attach"
                        print(
                            f"[EVENT] staged grasp attach step={step} local_pos0={grasp_local_pos0}",
                            flush=True,
                        )
                elif step < int(args_cli.settle_steps):
                    cycle = 0
                    phase = 0.0
                    rail_target = 0.0
                    phase_name = "settle"
                elif step < drive_start_step:
                    cycle = 0
                    probe_step = step - int(args_cli.settle_steps)
                    phase = probe_step / float(max(int(args_cli.probe_steps), 1))
                    rail_target = 0.0
                    phase_name = f"active_probe_{args_cli.probe_mode}"
                else:
                    motion_step = step - drive_start_step
                    if bool(args_cli.fix_anchor_to_world):
                        cycle = 0
                        phase = min(1.0, motion_step / float(max(int(args_cli.stance_steps), 1)))
                    else:
                        stance_steps = max(int(args_cli.stance_steps), 1)
                        cycle = min(cycle_count - 1, motion_step // stance_steps)
                        within_cycle_step = motion_step - cycle * stance_steps
                        phase = min(1.0, within_cycle_step / float(stance_steps))
                    if bool(args_cli.cumulative_cycle_target) and not bool(args_cli.fix_anchor_to_world):
                        rail_target = rail_direction * stride * (float(cycle) + _smooth01(phase))
                    else:
                        rail_target = rail_direction * stride * _smooth01(phase)
                    phase_name = "stance_drive"
                support_foot_x_target = 0.0
                support_foot_x_targets = {foot: 0.0 for foot in FOOT_NAMES}
                support_foot_z_targets = {foot: 0.0 for foot in FOOT_NAMES}
                commanded_stance_feet = set(FOOT_NAMES)
                if bool(args_cli.use_support_foot_drive):
                    if phase_name.startswith("active_probe"):
                        if str(args_cli.probe_mode) == "vertical_micro_lift":
                            probe_lift = -abs(float(args_cli.probe_z_amplitude)) * math.sin(math.pi * max(0.0, min(1.0, phase)))
                            support_foot_z_targets = {foot: probe_lift for foot in FOOT_NAMES}
                        else:
                            probe_target = float(args_cli.probe_x_amplitude) * math.sin(2.0 * math.pi * phase)
                            support_foot_x_target = probe_target
                            support_foot_x_targets = {foot: probe_target for foot in FOOT_NAMES}
                    elif str(args_cli.support_foot_mode) == "xz_prismatic_to_anchor":
                        stance_pair = {"fl", "rr"} if int(cycle) % 2 == 0 else {"fr", "rl"}
                        double_support = max(0.0, min(0.49, effective_support_double_support_fraction))
                        double_support_phase = bool(double_support > 0.0 and (phase <= double_support or phase >= 1.0 - double_support))
                        commanded_stance_feet = set(FOOT_NAMES) if double_support_phase else set(stance_pair)
                        if str(args_cli.support_foot_placement_mode) == "alternating_directional_x":
                            swing_x = direction * abs(effective_support_swing_x)
                            stance_x = -direction * abs(effective_support_stance_x)
                        else:
                            swing_x = effective_support_swing_x
                            stance_x = effective_support_stance_x
                        lift = 0.0 if double_support_phase else max(0.0, effective_support_step_height) * math.sin(math.pi * max(0.0, min(1.0, phase)))
                        step_progress = _smooth01(phase)
                        for foot in FOOT_NAMES:
                            if foot in stance_pair:
                                support_foot_x_targets[foot] = swing_x + (stance_x - swing_x) * step_progress
                                support_foot_z_targets[foot] = 0.0
                            else:
                                support_foot_x_targets[foot] = swing_x
                                support_foot_z_targets[foot] = lift
                        support_foot_x_target = float(np.mean(list(support_foot_x_targets.values())))
                    else:
                        support_foot_x_target = (
                            float(args_cli.support_foot_drive_direction_scale)
                            * direction
                            * stride
                            * _smooth01(phase)
                        )
                        support_foot_x_targets = {foot: support_foot_x_target for foot in FOOT_NAMES}
                    if (
                        bool(args_cli.planted_stance_rail_propulsion)
                        and support_feet_xz_prismatic
                        and step >= drive_start_step
                        and not phase_name.startswith("active_probe")
                    ):
                        summary["planted_stance_rail_propulsion_steps"] = int(
                            summary["planted_stance_rail_propulsion_steps"]
                        ) + 1
                    else:
                        rail_target = 0.0
                elif phase_name.startswith("active_probe") and str(args_cli.probe_mode) == "horizontal_push_pull":
                    rail_target = float(args_cli.probe_x_amplitude) * math.sin(2.0 * math.pi * phase)
                if (
                    bool(args_cli.feedback_step_controller)
                    and bool(args_cli.use_support_foot_drive)
                    and str(args_cli.support_foot_mode) == "xz_prismatic_to_anchor"
                    and step >= drive_start_step
                    and not phase_name.startswith("active_probe")
                ):
                    roll_before, pitch_before = _quat_to_roll_pitch(
                        float(torso_before[3]),
                        float(torso_before[4]),
                        float(torso_before[5]),
                        float(torso_before[6]),
                    )
                    x_error = float(target_x - torso_before_travel_x)
                    x_limit = abs(float(args_cli.feedback_step_x_limit))
                    x_adjust = max(-x_limit, min(x_limit, float(args_cli.feedback_step_x_gain) * x_error))
                    tilt_before = max(abs(roll_before), abs(pitch_before))
                    tilt_limit = abs(float(args_cli.feedback_step_tilt_limit))
                    tilt_adjust = max(0.0, min(tilt_limit, float(args_cli.feedback_step_tilt_gain) * tilt_before))
                    if abs(x_adjust) > 0.0 or tilt_adjust > 0.0:
                        for foot in FOOT_NAMES:
                            if foot in commanded_stance_feet:
                                support_foot_x_targets[foot] = support_foot_x_targets.get(foot, 0.0) - x_adjust
                            else:
                                support_foot_x_targets[foot] = support_foot_x_targets.get(foot, 0.0) + x_adjust
                                support_foot_z_targets[foot] = max(
                                    0.0,
                                    support_foot_z_targets.get(foot, 0.0) - tilt_adjust,
                                )
                        support_foot_x_target = float(np.mean(list(support_foot_x_targets.values())))
                        summary["feedback_step_applied_steps"] = int(summary["feedback_step_applied_steps"]) + 1
                        summary["max_abs_feedback_step_x_adjustment_m"] = max(
                            float(summary["max_abs_feedback_step_x_adjustment_m"]),
                            abs(x_adjust),
                        )
                        summary["max_abs_feedback_step_tilt_adjustment_m"] = max(
                            float(summary["max_abs_feedback_step_tilt_adjustment_m"]),
                            abs(tilt_adjust),
                        )
                if (
                    bool(args_cli.freeze_commanded_stance_foot_targets)
                    and support_feet_xz_prismatic
                    and step >= drive_start_step
                    and not phase_name.startswith("active_probe")
                ):
                    desired_frozen_feet = set(commanded_stance_feet)
                    if desired_frozen_feet != frozen_commanded_stance_feet:
                        joint_positions_now = _as_numpy(robot.get_joint_positions())
                        frozen_commanded_support_foot_x_targets = {}
                        frozen_commanded_support_foot_z_targets = {}
                        for foot in desired_frozen_feet:
                            x_idx = support_foot_x_indices.get(foot)
                            z_idx = support_foot_z_indices.get(foot)
                            if x_idx is not None:
                                frozen_commanded_support_foot_x_targets[foot] = float(joint_positions_now[x_idx])
                            if z_idx is not None:
                                frozen_commanded_support_foot_z_targets[foot] = float(joint_positions_now[z_idx])
                        frozen_commanded_stance_feet = set(desired_frozen_feet)
                        summary["freeze_commanded_stance_foot_target_switch_count"] = int(
                            summary["freeze_commanded_stance_foot_target_switch_count"]
                        ) + 1
                    for foot in frozen_commanded_stance_feet:
                        if foot in frozen_commanded_support_foot_x_targets:
                            support_foot_x_targets[foot] = frozen_commanded_support_foot_x_targets[foot]
                        if foot in frozen_commanded_support_foot_z_targets:
                            support_foot_z_targets[foot] = frozen_commanded_support_foot_z_targets[foot]
                    if support_foot_x_targets:
                        support_foot_x_target = float(np.mean(list(support_foot_x_targets.values())))
                    summary["freeze_commanded_stance_foot_target_count"] = int(
                        len(frozen_commanded_support_foot_x_targets)
                        + len(frozen_commanded_support_foot_z_targets)
                    )
                    summary["freeze_commanded_stance_foot_active_feet"] = sorted(frozen_commanded_stance_feet)
                elif frozen_commanded_stance_feet:
                    frozen_commanded_stance_feet = set()
                    frozen_commanded_support_foot_x_targets = {}
                    frozen_commanded_support_foot_z_targets = {}
                    summary["freeze_commanded_stance_foot_target_count"] = 0
                    summary["freeze_commanded_stance_foot_active_feet"] = []
                if step >= drive_start_step and not stop_latched:
                    if abs(target_x - torso_before_travel_x) <= float(args_cli.stop_threshold):
                        joint_positions_now = _as_numpy(robot.get_joint_positions())
                        latched_per_joint_target = float(np.nanmean(joint_positions_now[rail_indices]))
                        if support_foot_x_indices:
                            latched_support_foot_x_targets = {
                                foot: float(joint_positions_now[idx])
                                for foot, idx in support_foot_x_indices.items()
                            }
                        stop_latched = True
                if stop_latched and latched_per_joint_target is not None:
                    if bool(args_cli.use_support_foot_drive):
                        if latched_support_foot_x_targets:
                            support_foot_x_targets = {
                                foot: latched_support_foot_x_targets.get(foot, support_foot_x_targets.get(foot, 0.0))
                                for foot in FOOT_NAMES
                            }
                            support_foot_x_target = float(np.mean(list(support_foot_x_targets.values())))
                        else:
                            support_foot_x_target = float(np.nanmean(_as_numpy(robot.get_joint_positions())[list(support_foot_x_indices.values())]))
                            support_foot_x_targets = {foot: support_foot_x_target for foot in FOOT_NAMES}
                        support_foot_z_targets = {foot: 0.0 for foot in FOOT_NAMES}
                        rail_target = 0.0
                    else:
                        rail_target = latched_per_joint_target * float(max(len(rail_indices), 1))
                    phase_name = "target_hold"
                if stance_world_lock_joints:
                    desired_locked_feet = (
                        set(commanded_stance_feet)
                        if step >= drive_start_step and not phase_name.startswith("active_probe")
                        else set()
                    )
                    if desired_locked_feet != locked_stance_feet:
                        for foot, joint in stance_world_lock_joints.items():
                            foot_pose_for_lock = _pose_wxyz(support_feet[foot])
                            should_lock = foot in desired_locked_feet
                            _set_world_fixed_joint(
                                joint,
                                (
                                    float(foot_pose_for_lock[0]),
                                    float(foot_pose_for_lock[1]),
                                    float(foot_pose_for_lock[2]),
                                ),
                                enabled=should_lock,
                            )
                            summary["stance_foot_world_lock_pose_update_count"] = int(
                                summary["stance_foot_world_lock_pose_update_count"]
                            ) + 1
                            if bool(args_cli.freeze_locked_stance_foot_targets) and should_lock:
                                joint_positions_now = _as_numpy(robot.get_joint_positions())
                                x_idx = support_foot_x_indices.get(foot)
                                z_idx = support_foot_z_indices.get(foot)
                                if x_idx is not None:
                                    frozen_locked_support_foot_x_targets[foot] = float(joint_positions_now[x_idx])
                                if z_idx is not None:
                                    frozen_locked_support_foot_z_targets[foot] = float(joint_positions_now[z_idx])
                            else:
                                frozen_locked_support_foot_x_targets.pop(foot, None)
                                frozen_locked_support_foot_z_targets.pop(foot, None)
                        summary["stance_foot_world_lock_switch_count"] = int(
                            summary["stance_foot_world_lock_switch_count"]
                        ) + 1
                        locked_stance_feet = set(desired_locked_feet)
                    summary["stance_foot_world_lock_active_feet"] = sorted(locked_stance_feet)
                    if bool(args_cli.freeze_locked_stance_foot_targets):
                        for foot in locked_stance_feet:
                            if foot in frozen_locked_support_foot_x_targets:
                                support_foot_x_targets[foot] = frozen_locked_support_foot_x_targets[foot]
                            if foot in frozen_locked_support_foot_z_targets:
                                support_foot_z_targets[foot] = frozen_locked_support_foot_z_targets[foot]
                        if support_foot_x_targets:
                            support_foot_x_target = float(np.mean(list(support_foot_x_targets.values())))
                        summary["freeze_locked_stance_foot_target_count"] = int(
                            len(frozen_locked_support_foot_x_targets)
                            + len(frozen_locked_support_foot_z_targets)
                        )
                if (not bool(args_cli.fix_anchor_to_world)) and cycle != last_cycle and step >= drive_start_step:
                    torso_now = _pose_wxyz(torso_prim)
                    anchor_x = float(torso_now[0])
                    if bool(args_cli.disable_support_reposition):
                        pass
                    elif bool(args_cli.replant_anchor_world_joint):
                        if anchor_world_joint is None:
                            raise RuntimeError("replant_anchor_world_joint requested but AnchorWorldFixedJoint was not created")
                        anchor_world_joint.GetLocalPos0Attr().Set(
                            Gf.Vec3f(float(anchor_x), 0.0, float(args_cli.torso_z))
                        )
                        summary["anchor_world_joint_retarget_count"] = int(summary["anchor_world_joint_retarget_count"]) + 1
                    elif bool(args_cli.anchor_as_articulation_root):
                        robot.set_world_pose(position=np.asarray([anchor_x, 0.0, float(args_cli.torso_z)], dtype=float))
                        summary["support_root_pose_write_count"] = int(summary["support_root_pose_write_count"]) + 1
                    else:
                        anchor.set_world_pose(position=np.asarray([anchor_x, 0.0, float(args_cli.torso_z)], dtype=float))
                    if not bool(args_cli.disable_support_reposition):
                        anchor_pose_write_count += 1
                        for foot, (xoff, yoff) in _foot_offsets().items():
                            _set_foot_marker_pose(stage, foot, anchor_x + xoff, yoff)
                            foot_pose_write_count += 1
                    last_cycle = cycle
                per_joint_target = float(rail_target) / float(max(len(rail_indices), 1))
                for rail_idx in rail_indices:
                    current_targets[rail_idx] = per_joint_target
                for foot, foot_idx in support_foot_x_indices.items():
                    current_targets[foot_idx] = support_foot_x_targets.get(foot, support_foot_x_target)
                for foot, foot_idx in support_foot_z_indices.items():
                    current_targets[foot_idx] = support_foot_z_targets.get(foot, 0.0)
                clamp_target = 0.0
                if str(args_cli.payload_mode) == "side_clamp_free_box" and clamp_indices:
                    close_phase = (
                        (step - int(args_cli.clamp_close_start_step))
                        / float(max(int(args_cli.clamp_close_steps), 1))
                    )
                    clamp_target = clamp_travel * effective_hold_closure_fraction * _smooth01(close_phase)
                    summary["max_commanded_clamp_target_m"] = max(
                        float(summary["max_commanded_clamp_target_m"]),
                        abs(float(clamp_target)),
                    )
                    summary["final_commanded_clamp_target_m"] = float(clamp_target)
                    if "left" in clamp_indices:
                        current_targets[clamp_indices["left"]] = -clamp_target
                        if "left" in clamp_drives:
                            clamp_drives["left"].GetTargetPositionAttr().Set(float(-clamp_target))
                            summary["clamp_drive_target_update_count"] = (
                                int(summary["clamp_drive_target_update_count"]) + 1
                            )
                    if "right" in clamp_indices:
                        current_targets[clamp_indices["right"]] = clamp_target
                        if "right" in clamp_drives:
                            clamp_drives["right"].GetTargetPositionAttr().Set(float(clamp_target))
                            summary["clamp_drive_target_update_count"] = (
                                int(summary["clamp_drive_target_update_count"]) + 1
                            )
                if str(args_cli.payload_mode) == "x_cradle_free_box" and cradle_indices:
                    close_phase = (
                        (step - int(args_cli.clamp_close_start_step))
                        / float(max(int(args_cli.clamp_close_steps), 1))
                    )
                    cradle_target = cradle_travel * effective_hold_closure_fraction * _smooth01(close_phase)
                    summary["max_commanded_cradle_target_m"] = max(
                        float(summary["max_commanded_cradle_target_m"]),
                        abs(float(cradle_target)),
                    )
                    summary["final_commanded_cradle_target_m"] = float(cradle_target)
                    current_targets[cradle_indices["rear_pusher"]] = cradle_target
                    if "rear_pusher" in cradle_drives:
                        cradle_drives["rear_pusher"].GetTargetPositionAttr().Set(float(cradle_target))
                        summary["cradle_drive_target_update_count"] = (
                            int(summary["cradle_drive_target_update_count"]) + 1
                        )
                robot.apply_action(ArticulationAction(joint_positions=current_targets.tolist()))
                world.step(render=bool(args_cli.render))
                torso = _pose_wxyz(torso_prim)
                box = _pose_wxyz(payload) if payload is not None else torso
                anchor_pose = _pose_wxyz(anchor)
                measured_joint_efforts = None
                if support_foot_x_indices or support_foot_z_indices:
                    try:
                        measured_joint_efforts = _as_numpy(robot.get_measured_joint_efforts()).reshape(-1)
                        summary["support_foot_effort_available"] = True
                        if phase_name.startswith("active_probe"):
                            summary["probe_joint_effort_available"] = True
                    except Exception as effort_exc:
                        summary["support_foot_effort_read_error_count"] = (
                            int(summary["support_foot_effort_read_error_count"]) + 1
                        )
                        if summary["support_foot_effort_first_error"] is None:
                            summary["support_foot_effort_first_error"] = f"{type(effort_exc).__name__}: {effort_exc}"
                        if phase_name.startswith("active_probe"):
                            summary["probe_joint_effort_read_error_count"] = (
                                int(summary["probe_joint_effort_read_error_count"]) + 1
                            )
                            if summary["probe_joint_effort_first_error"] is None:
                                summary["probe_joint_effort_first_error"] = f"{type(effort_exc).__name__}: {effort_exc}"
                finite = np.all(np.isfinite(np.asarray(torso + box, dtype=float)))
                finite = finite and np.all(np.isfinite(np.asarray(anchor_pose, dtype=float)))
                support_foot_travel_x: dict[str, float] = {}
                near_ground_xy: dict[str, tuple[float, float]] = {}
                for foot, support_foot in support_feet.items():
                    support_foot_pose = _pose_wxyz(support_foot)
                    finite = finite and np.all(np.isfinite(np.asarray(support_foot_pose, dtype=float)))
                    initial_pose = initial_support_feet[foot]
                    previous_pose = previous_support_foot_poses.get(foot, support_foot_pose)
                    foot_travel_x = float(support_foot_pose[0]) - float(initial_pose[0])
                    foot_travel_y = float(support_foot_pose[1]) - float(initial_pose[1])
                    foot_lift_z = float(support_foot_pose[2]) - float(initial_pose[2])
                    foot_step_dx = float(support_foot_pose[0]) - float(previous_pose[0])
                    foot_step_dy = float(support_foot_pose[1]) - float(previous_pose[1])
                    foot_step_xy = math.hypot(foot_step_dx, foot_step_dy)
                    foot_travel_xy = math.hypot(foot_travel_x, foot_travel_y)
                    support_foot_travel_x[foot] = foot_travel_x
                    summary["per_foot_min_z_m"][foot] = (
                        float(support_foot_pose[2])
                        if foot not in summary["per_foot_min_z_m"]
                        else min(float(summary["per_foot_min_z_m"][foot]), float(support_foot_pose[2]))
                    )
                    summary["per_foot_max_z_m"][foot] = (
                        float(support_foot_pose[2])
                        if foot not in summary["per_foot_max_z_m"]
                        else max(float(summary["per_foot_max_z_m"][foot]), float(support_foot_pose[2]))
                    )
                    summary["per_foot_max_actual_lift_m"][foot] = max(
                        float(summary["per_foot_max_actual_lift_m"].get(foot, 0.0)),
                        foot_lift_z,
                    )
                    summary["max_actual_support_foot_lift_m"] = max(
                        float(summary["max_actual_support_foot_lift_m"]),
                        foot_lift_z,
                    )
                    summary["support_foot_min_z_m"] = (
                        float(support_foot_pose[2])
                        if summary["support_foot_min_z_m"] is None
                        else min(float(summary["support_foot_min_z_m"]), float(support_foot_pose[2]))
                    )
                    summary["support_foot_max_z_m"] = (
                        float(support_foot_pose[2])
                        if summary["support_foot_max_z_m"] is None
                        else max(float(summary["support_foot_max_z_m"]), float(support_foot_pose[2]))
                    )
                    summary["max_abs_support_foot_travel_x_m"] = max(
                        float(summary["max_abs_support_foot_travel_x_m"]), abs(foot_travel_x)
                    )
                    summary["max_support_foot_travel_xy_m"] = max(
                        float(summary["max_support_foot_travel_xy_m"]), foot_travel_xy
                    )
                    near_ground = float(support_foot_pose[2]) <= float(args_cli.support_foot_contact_z_threshold)
                    if near_ground:
                        near_ground_xy[foot] = (float(support_foot_pose[0]), float(support_foot_pose[1]))
                        summary["per_foot_near_ground_steps"][foot] = int(summary["per_foot_near_ground_steps"].get(foot, 0)) + 1
                        if near_ground_reference_xy.get(foot) is None:
                            near_ground_reference_xy[foot] = (float(support_foot_pose[0]), float(support_foot_pose[1]))
                        reference_xy = near_ground_reference_xy[foot]
                        if reference_xy is not None:
                            near_ground_slip = math.hypot(
                                float(support_foot_pose[0]) - reference_xy[0],
                                float(support_foot_pose[1]) - reference_xy[1],
                            )
                            summary["per_foot_max_near_ground_xy_slip_m"][foot] = max(
                                float(summary["per_foot_max_near_ground_xy_slip_m"].get(foot, 0.0)),
                                near_ground_slip,
                            )
                        summary["per_foot_max_near_ground_xy_speed_mps"][foot] = max(
                            float(summary["per_foot_max_near_ground_xy_speed_mps"].get(foot, 0.0)),
                            foot_step_xy / 0.005,
                        )
                    else:
                        near_ground_reference_xy[foot] = None
                    previous_support_foot_poses[foot] = support_foot_pose
                near_ground_count = len(near_ground_xy)
                summary["min_near_ground_foot_count"] = (
                    near_ground_count
                    if summary["min_near_ground_foot_count"] is None
                    else min(int(summary["min_near_ground_foot_count"]), near_ground_count)
                )
                summary["max_near_ground_foot_count"] = max(int(summary["max_near_ground_foot_count"]), near_ground_count)
                if near_ground_count == 0:
                    summary["near_ground_zero_steps"] = int(summary["near_ground_zero_steps"]) + 1
                if near_ground_count < 2:
                    summary["near_ground_lt2_steps"] = int(summary["near_ground_lt2_steps"]) + 1
                if step >= support_foot_continuity_start_step:
                    summary["min_drive_near_ground_foot_count"] = (
                        near_ground_count
                        if summary["min_drive_near_ground_foot_count"] is None
                        else min(int(summary["min_drive_near_ground_foot_count"]), near_ground_count)
                    )
                    if near_ground_count == 0:
                        summary["drive_near_ground_zero_steps"] = int(summary["drive_near_ground_zero_steps"]) + 1
                    if near_ground_count < 2:
                        summary["drive_near_ground_lt2_steps"] = int(summary["drive_near_ground_lt2_steps"]) + 1
                commanded_stance_near_ground_count = sum(1 for foot in commanded_stance_feet if foot in near_ground_xy)
                if step >= support_foot_continuity_start_step:
                    summary["min_commanded_stance_near_ground_foot_count"] = (
                        commanded_stance_near_ground_count
                        if summary["min_commanded_stance_near_ground_foot_count"] is None
                        else min(
                            int(summary["min_commanded_stance_near_ground_foot_count"]),
                            commanded_stance_near_ground_count,
                        )
                    )
                    if commanded_stance_near_ground_count < min(2, len(commanded_stance_feet)):
                        summary["commanded_stance_near_ground_lt2_steps"] = (
                            int(summary["commanded_stance_near_ground_lt2_steps"]) + 1
                        )
                if near_ground_count >= 2:
                    xs = [xy[0] for xy in near_ground_xy.values()]
                    ys = [xy[1] for xy in near_ground_xy.values()]
                    margin_x = min(float(torso[0]) - min(xs), max(xs) - float(torso[0]))
                    margin_y = min(float(torso[1]) - min(ys), max(ys) - float(torso[1]))
                    margin = min(margin_x, margin_y)
                    summary["min_support_polygon_margin_x_m"] = (
                        margin_x
                        if summary["min_support_polygon_margin_x_m"] is None
                        else min(float(summary["min_support_polygon_margin_x_m"]), margin_x)
                    )
                    summary["min_support_polygon_margin_y_m"] = (
                        margin_y
                        if summary["min_support_polygon_margin_y_m"] is None
                        else min(float(summary["min_support_polygon_margin_y_m"]), margin_y)
                    )
                    summary["min_support_polygon_margin_m"] = (
                        margin
                        if summary["min_support_polygon_margin_m"] is None
                        else min(float(summary["min_support_polygon_margin_m"]), margin)
                    )
                contact_report_feet = set(contact_report_state["active_feet"]) if bool(contact_report_state["available"]) else set()
                contact_report_count = len(contact_report_feet)
                summary["support_foot_contact_report_event_count"] = int(contact_report_state["event_count"])
                summary["support_foot_contact_report_error_count"] = int(contact_report_state["error_count"])
                summary["support_foot_contact_report_first_error"] = contact_report_state["first_error"]
                if bool(contact_report_state["available"]):
                    for foot in contact_report_feet:
                        summary["per_foot_contact_report_steps"][foot] = (
                            int(summary["per_foot_contact_report_steps"].get(foot, 0)) + 1
                        )
                    summary["min_contact_report_foot_count"] = (
                        contact_report_count
                        if summary["min_contact_report_foot_count"] is None
                        else min(int(summary["min_contact_report_foot_count"]), contact_report_count)
                    )
                    summary["max_contact_report_foot_count"] = max(
                        int(summary["max_contact_report_foot_count"]),
                        contact_report_count,
                    )
                    if contact_report_count == 0:
                        summary["contact_report_zero_steps"] = int(summary["contact_report_zero_steps"]) + 1
                    if contact_report_count < 2:
                        summary["contact_report_lt2_steps"] = int(summary["contact_report_lt2_steps"]) + 1
                    if step >= support_foot_continuity_start_step:
                        summary["min_drive_contact_report_foot_count"] = (
                            contact_report_count
                            if summary["min_drive_contact_report_foot_count"] is None
                            else min(int(summary["min_drive_contact_report_foot_count"]), contact_report_count)
                        )
                        if contact_report_count == 0:
                            summary["drive_contact_report_zero_steps"] = (
                                int(summary["drive_contact_report_zero_steps"]) + 1
                            )
                        if contact_report_count < 2:
                            summary["drive_contact_report_lt2_steps"] = (
                                int(summary["drive_contact_report_lt2_steps"]) + 1
                            )
                        commanded_stance_contact_count = sum(
                            1 for foot in commanded_stance_feet if foot in contact_report_feet
                        )
                        summary["min_commanded_stance_contact_report_foot_count"] = (
                            commanded_stance_contact_count
                            if summary["min_commanded_stance_contact_report_foot_count"] is None
                            else min(
                                int(summary["min_commanded_stance_contact_report_foot_count"]),
                                commanded_stance_contact_count,
                            )
                        )
                        if commanded_stance_contact_count < min(2, len(commanded_stance_feet)):
                            summary["commanded_stance_contact_report_lt2_steps"] = (
                                int(summary["commanded_stance_contact_report_lt2_steps"]) + 1
                            )
                effort_supported_feet: set[str] = set()
                if measured_joint_efforts is not None and measured_joint_efforts.size > 0:
                    effort_threshold = max(0.0, float(args_cli.support_foot_effort_contact_threshold))
                    for foot in FOOT_NAMES:
                        foot_efforts = []
                        x_idx = support_foot_x_indices.get(foot)
                        if x_idx is not None and int(x_idx) < int(measured_joint_efforts.shape[0]):
                            x_effort = abs(float(measured_joint_efforts[int(x_idx)]))
                            foot_efforts.append(x_effort)
                            summary["per_foot_max_support_x_measured_effort"][foot] = max(
                                float(summary["per_foot_max_support_x_measured_effort"].get(foot, 0.0)),
                                x_effort,
                            )
                        z_idx = support_foot_z_indices.get(foot)
                        if z_idx is not None and int(z_idx) < int(measured_joint_efforts.shape[0]):
                            z_effort = abs(float(measured_joint_efforts[int(z_idx)]))
                            foot_efforts.append(z_effort)
                            summary["per_foot_max_support_z_measured_effort"][foot] = max(
                                float(summary["per_foot_max_support_z_measured_effort"].get(foot, 0.0)),
                                z_effort,
                            )
                        max_foot_effort = max(foot_efforts) if foot_efforts else 0.0
                        summary["per_foot_max_support_measured_effort"][foot] = max(
                            float(summary["per_foot_max_support_measured_effort"].get(foot, 0.0)),
                            max_foot_effort,
                        )
                        if max_foot_effort >= effort_threshold:
                            effort_supported_feet.add(foot)
                effort_supported_count = len(effort_supported_feet)
                commanded_stance_effort_supported_count = sum(
                    1 for foot in commanded_stance_feet if foot in effort_supported_feet
                )
                if step >= support_foot_continuity_start_step and bool(summary["support_foot_effort_available"]):
                    summary["min_drive_effort_supported_foot_count"] = (
                        effort_supported_count
                        if summary["min_drive_effort_supported_foot_count"] is None
                        else min(int(summary["min_drive_effort_supported_foot_count"]), effort_supported_count)
                    )
                    if effort_supported_count == 0:
                        summary["drive_effort_supported_zero_steps"] = (
                            int(summary["drive_effort_supported_zero_steps"]) + 1
                        )
                    if effort_supported_count < 2:
                        summary["drive_effort_supported_lt2_steps"] = (
                            int(summary["drive_effort_supported_lt2_steps"]) + 1
                        )
                    summary["min_commanded_stance_effort_supported_foot_count"] = (
                        commanded_stance_effort_supported_count
                        if summary["min_commanded_stance_effort_supported_foot_count"] is None
                        else min(
                            int(summary["min_commanded_stance_effort_supported_foot_count"]),
                            commanded_stance_effort_supported_count,
                        )
                    )
                    if commanded_stance_effort_supported_count < min(2, len(commanded_stance_feet)):
                        summary["commanded_stance_effort_supported_lt2_steps"] = (
                            int(summary["commanded_stance_effort_supported_lt2_steps"]) + 1
                        )
                if not finite:
                    summary["nonfinite_state_events"] += 1
                roll, pitch = _quat_to_roll_pitch(float(torso[3]), float(torso[4]), float(torso[5]), float(torso[6]))
                tilt = float(max(abs(roll), abs(pitch)))
                torso_travel_x = float(torso[0]) - float(initial_torso[0])
                payload_travel_x = float(box[0]) - float(initial_payload[0])
                anchor_travel_x = float(anchor_pose[0]) - float(initial_anchor[0])
                anchor_travel_y = float(anchor_pose[1]) - float(initial_anchor[1])
                anchor_travel_xy = math.hypot(anchor_travel_x, anchor_travel_y)
                if step >= drive_start_step and post_settle_torso_x is None:
                    post_settle_torso_x = float(torso[0])
                    post_settle_payload_x = float(box[0])
                    summary["post_settle_baseline_step"] = int(step)
                    summary["post_settle_baseline_torso_x_m"] = float(post_settle_torso_x)
                    summary["post_settle_baseline_payload_x_m"] = float(post_settle_payload_x)
                post_torso_travel_x = 0.0
                post_payload_travel_x = 0.0
                post_relative_error = 0.0
                if post_settle_torso_x is not None and post_settle_payload_x is not None:
                    post_torso_travel_x = float(torso[0]) - post_settle_torso_x
                    post_payload_travel_x = float(box[0]) - post_settle_payload_x
                    post_relative_error = abs(post_payload_travel_x - post_torso_travel_x)
                target_distance = abs(target_x - torso_travel_x)
                payload_target_distance = abs(target_x - payload_travel_x)
                joint_positions = _as_numpy(robot.get_joint_positions())
                rail_joint_motion = float(np.nansum(np.abs(joint_positions[rail_indices] - initial_joint_positions[rail_indices])))
                support_foot_x_joint_motion = 0.0
                if support_foot_x_indices:
                    support_joint_ids = list(support_foot_x_indices.values())
                    support_foot_x_joint_motion = float(
                        np.nansum(np.abs(joint_positions[support_joint_ids] - initial_joint_positions[support_joint_ids]))
                    )
                support_foot_z_joint_motion = 0.0
                if support_foot_z_indices:
                    support_z_joint_ids = list(support_foot_z_indices.values())
                    support_foot_z_joint_motion = float(
                        np.nansum(np.abs(joint_positions[support_z_joint_ids] - initial_joint_positions[support_z_joint_ids]))
                    )
                clamp_joint_motion = 0.0
                if clamp_indices:
                    clamp_joint_ids = list(clamp_indices.values())
                    clamp_joint_motion = float(
                        np.nansum(np.abs(joint_positions[clamp_joint_ids] - initial_joint_positions[clamp_joint_ids]))
                    )
                cradle_joint_motion = 0.0
                if cradle_indices:
                    cradle_joint_ids = list(cradle_indices.values())
                    cradle_joint_motion = float(
                        np.nansum(np.abs(joint_positions[cradle_joint_ids] - initial_joint_positions[cradle_joint_ids]))
                    )
                joint_motion = rail_joint_motion
                fall = int(float(torso[2]) < float(args_cli.fall_z) or tilt > 0.85)
                drop = int(payload_active and float(box[2]) < float(args_cli.drop_z))
                if payload_active:
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
                else:
                    payload_relative_offset_error = 0.0
                if phase_name.startswith("active_probe"):
                    tracking_errors = []
                    for foot, foot_idx in support_foot_x_indices.items():
                        actual = float(joint_positions[foot_idx])
                        target = float(support_foot_x_targets.get(foot, support_foot_x_target))
                        tracking_errors.append(abs(target - actual))
                    if tracking_errors:
                        max_tracking_error = max(tracking_errors)
                        mean_tracking_error = float(np.mean(np.asarray(tracking_errors, dtype=float)))
                        previous_count = int(summary["probe_support_foot_x_tracking_error_samples"])
                        previous_mean = float(summary["mean_probe_support_foot_x_tracking_error_m"] or 0.0)
                        new_count = previous_count + 1
                        summary["max_probe_support_foot_x_tracking_error_m"] = max(
                            float(summary["max_probe_support_foot_x_tracking_error_m"]),
                            max_tracking_error,
                        )
                        summary["mean_probe_support_foot_x_tracking_error_m"] = (
                            (previous_mean * previous_count + mean_tracking_error) / float(new_count)
                        )
                        summary["probe_support_foot_x_tracking_error_samples"] = new_count
                    z_tracking_errors = []
                    for foot, foot_idx in support_foot_z_indices.items():
                        actual = float(joint_positions[foot_idx])
                        target = float(support_foot_z_targets.get(foot, 0.0))
                        z_tracking_errors.append(abs(target - actual))
                    if z_tracking_errors:
                        max_z_tracking_error = max(z_tracking_errors)
                        mean_z_tracking_error = float(np.mean(np.asarray(z_tracking_errors, dtype=float)))
                        previous_count = int(summary["probe_support_foot_z_tracking_error_samples"])
                        previous_mean = float(summary["mean_probe_support_foot_z_tracking_error_m"] or 0.0)
                        new_count = previous_count + 1
                        summary["max_probe_support_foot_z_tracking_error_m"] = max(
                            float(summary["max_probe_support_foot_z_tracking_error_m"]),
                            max_z_tracking_error,
                        )
                        summary["mean_probe_support_foot_z_tracking_error_m"] = (
                            (previous_mean * previous_count + mean_z_tracking_error) / float(new_count)
                        )
                        summary["probe_support_foot_z_tracking_error_samples"] = new_count
                    if measured_joint_efforts is not None and measured_joint_efforts.size > 0:
                        effort_values = []
                        for foot_idx in support_foot_x_indices.values():
                            if int(foot_idx) < int(measured_joint_efforts.shape[0]):
                                effort_values.append(abs(float(measured_joint_efforts[int(foot_idx)])))
                        if effort_values:
                            max_effort = max(effort_values)
                            mean_effort = float(np.mean(np.asarray(effort_values, dtype=float)))
                            previous_count = int(summary["probe_support_foot_x_measured_effort_samples"])
                            previous_mean = float(summary["mean_probe_support_foot_x_measured_effort"] or 0.0)
                            new_count = previous_count + 1
                            summary["max_probe_support_foot_x_measured_effort"] = max(
                                float(summary["max_probe_support_foot_x_measured_effort"]),
                                max_effort,
                            )
                            summary["mean_probe_support_foot_x_measured_effort"] = (
                                (previous_mean * previous_count + mean_effort) / float(new_count)
                            )
                            summary["probe_support_foot_x_measured_effort_samples"] = new_count
                        z_effort_values = []
                        for foot_idx in support_foot_z_indices.values():
                            if int(foot_idx) < int(measured_joint_efforts.shape[0]):
                                z_effort_values.append(abs(float(measured_joint_efforts[int(foot_idx)])))
                        if z_effort_values:
                            max_z_effort = max(z_effort_values)
                            mean_z_effort = float(np.mean(np.asarray(z_effort_values, dtype=float)))
                            previous_count = int(summary["probe_support_foot_z_measured_effort_samples"])
                            previous_mean = float(summary["mean_probe_support_foot_z_measured_effort"] or 0.0)
                            new_count = previous_count + 1
                            summary["max_probe_support_foot_z_measured_effort"] = max(
                                float(summary["max_probe_support_foot_z_measured_effort"]),
                                max_z_effort,
                            )
                            summary["mean_probe_support_foot_z_measured_effort"] = (
                                (previous_mean * previous_count + mean_z_effort) / float(new_count)
                            )
                            summary["probe_support_foot_z_measured_effort_samples"] = new_count
                    summary["max_probe_torso_travel_x_m"] = max(
                        float(summary["max_probe_torso_travel_x_m"]),
                        abs(torso_travel_x),
                    )
                    summary["max_probe_torso_travel_z_m"] = max(
                        float(summary["max_probe_torso_travel_z_m"]),
                        abs(float(torso[2]) - float(initial_torso[2])),
                    )
                    summary["max_probe_payload_travel_x_m"] = max(
                        float(summary["max_probe_payload_travel_x_m"]),
                        abs(payload_travel_x),
                    )
                    summary["max_probe_payload_travel_z_m"] = max(
                        float(summary["max_probe_payload_travel_z_m"]),
                        abs(float(box[2]) - float(initial_payload[2])),
                    )
                    summary["max_probe_payload_relative_error_m"] = max(
                        float(summary["max_probe_payload_relative_error_m"]),
                        payload_relative_offset_error,
                    )
                    summary["final_probe_payload_lag_x_m"] = float(payload_travel_x - torso_travel_x)
                    summary["final_probe_payload_lag_z_m"] = float(
                        (float(box[2]) - float(initial_payload[2]))
                        - (float(torso[2]) - float(initial_torso[2]))
                    )
                summary["completed_steps"] = step + 1
                summary["fall_events"] += fall
                summary["box_drop_events"] += drop
                summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), tilt)
                summary["min_torso_z_m"] = min(float(summary["min_torso_z_m"]), float(torso[2]))
                summary["min_payload_z_m"] = min(float(summary["min_payload_z_m"]), float(box[2]))
                summary["max_torso_travel_x_m"] = max(float(summary["max_torso_travel_x_m"]), torso_travel_x)
                summary["max_payload_travel_x_m"] = max(float(summary["max_payload_travel_x_m"]), payload_travel_x)
                summary["max_abs_torso_travel_x_m"] = max(float(summary["max_abs_torso_travel_x_m"]), abs(torso_travel_x))
                summary["max_abs_payload_travel_x_m"] = max(float(summary["max_abs_payload_travel_x_m"]), abs(payload_travel_x))
                summary["final_anchor_travel_x_m"] = anchor_travel_x
                summary["max_abs_anchor_travel_x_m"] = max(float(summary["max_abs_anchor_travel_x_m"]), abs(anchor_travel_x))
                summary["max_anchor_travel_xy_m"] = max(float(summary["max_anchor_travel_xy_m"]), anchor_travel_xy)
                summary["final_support_foot_travel_x_m"] = support_foot_travel_x
                summary["max_post_settle_torso_travel_x_m"] = max(
                    float(summary["max_post_settle_torso_travel_x_m"]), post_torso_travel_x
                )
                summary["max_post_settle_payload_travel_x_m"] = max(
                    float(summary["max_post_settle_payload_travel_x_m"]), post_payload_travel_x
                )
                summary["max_abs_post_settle_torso_travel_x_m"] = max(
                    float(summary["max_abs_post_settle_torso_travel_x_m"]), abs(post_torso_travel_x)
                )
                summary["max_abs_post_settle_payload_travel_x_m"] = max(
                    float(summary["max_abs_post_settle_payload_travel_x_m"]), abs(post_payload_travel_x)
                )
                target_directed_post_torso_travel = direction * post_torso_travel_x
                target_directed_post_payload_travel = direction * post_payload_travel_x
                summary["max_target_directed_post_settle_torso_travel_m"] = max(
                    float(summary["max_target_directed_post_settle_torso_travel_m"]),
                    target_directed_post_torso_travel,
                )
                summary["max_target_directed_post_settle_payload_travel_m"] = max(
                    float(summary["max_target_directed_post_settle_payload_travel_m"]),
                    target_directed_post_payload_travel,
                )
                summary["max_post_settle_payload_relative_error_m"] = max(
                    float(summary["max_post_settle_payload_relative_error_m"]), post_relative_error
                )
                summary["final_target_distance_x_m"] = target_distance
                summary["final_payload_target_distance_x_m"] = payload_target_distance
                summary["final_post_settle_torso_travel_x_m"] = post_torso_travel_x
                summary["final_post_settle_payload_travel_x_m"] = post_payload_travel_x
                summary["final_post_settle_payload_target_distance_x_m"] = abs(target_x - post_payload_travel_x)
                summary["post_settle_payload_travel_loss_after_peak_m"] = max(
                    0.0,
                    float(summary["max_target_directed_post_settle_payload_travel_m"])
                    - target_directed_post_payload_travel,
                )
                summary["final_post_settle_payload_relative_error_m"] = post_relative_error
                summary["payload_relative_error_m"] = payload_relative_offset_error
                summary["max_payload_relative_offset_error_m"] = max(
                    float(summary["max_payload_relative_offset_error_m"]), payload_relative_offset_error
                )
                summary["max_joint_motion_m"] = max(float(summary["max_joint_motion_m"]), joint_motion)
                summary["max_rail_joint_motion_m"] = max(
                    float(summary["max_rail_joint_motion_m"]),
                    rail_joint_motion,
                )
                summary["max_support_foot_x_joint_motion_m"] = max(
                    float(summary["max_support_foot_x_joint_motion_m"]), support_foot_x_joint_motion
                )
                summary["max_support_foot_z_joint_motion_m"] = max(
                    float(summary["max_support_foot_z_joint_motion_m"]), support_foot_z_joint_motion
                )
                summary["max_commanded_support_foot_lift_m"] = max(
                    float(summary["max_commanded_support_foot_lift_m"]),
                    max(abs(float(value)) for value in support_foot_z_targets.values()) if support_foot_z_targets else 0.0,
                )
                for foot in FOOT_NAMES:
                    summary["per_foot_max_commanded_x_m"][foot] = max(
                        float(summary["per_foot_max_commanded_x_m"].get(foot, 0.0)),
                        abs(float(support_foot_x_targets.get(foot, support_foot_x_target))),
                    )
                    summary["per_foot_max_commanded_z_m"][foot] = max(
                        float(summary["per_foot_max_commanded_z_m"].get(foot, 0.0)),
                        abs(float(support_foot_z_targets.get(foot, 0.0))),
                    )
                summary["final_support_foot_x_joint_target_m"] = float(support_foot_x_target)
                summary["final_support_foot_x_joint_target_m_by_foot"] = {
                    foot: float(value) for foot, value in support_foot_x_targets.items()
                }
                summary["final_support_foot_z_joint_target_m_by_foot"] = {
                    foot: float(value) for foot, value in support_foot_z_targets.items()
                }
                summary["max_clamp_joint_motion_m"] = max(float(summary["max_clamp_joint_motion_m"]), clamp_joint_motion)
                summary["max_cradle_joint_motion_m"] = max(float(summary["max_cradle_joint_motion_m"]), cradle_joint_motion)
                summary["stance_anchor_pose_write_count"] = int(anchor_pose_write_count)
                summary["foot_pose_write_count"] = int(foot_pose_write_count)
                summary["stop_latched"] = bool(stop_latched)
                summary["attached"] = bool(staged_grasp_attached)
                summary["staged_grasp_attach_step"] = staged_grasp_attach_step
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    writer.writerow(
                        [
                            step,
                            cycle,
                            phase_name,
                            torso[0],
                            box[0],
                            target_distance,
                            post_torso_travel_x,
                            post_payload_travel_x,
                            post_relative_error,
                            rail_target,
                            joint_motion,
                            tilt,
                            fall,
                            drop,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} cycle={cycle} phase={phase_name} torso_x={torso_travel_x:.4f} "
                        f"payload_x={payload_travel_x:.4f} target_dist={target_distance:.4f} "
                        f"rail_target={rail_target:.4f} tilt={tilt:.4f} fall={fall} drop={drop}",
                        flush=True,
                    )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)

    if contact_report_subscription is not None:
        contact_report_subscription = None
    summary.update(_probe_belief(summary))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run_scene()
    except BaseException as exc:
        try:
            args_cli.output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = args_cli.output_dir / "core_world_anchored_footstep_carrier_startup_failure.json"
            failure_path.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2, sort_keys=True) + "\n")
            print(f"[ERROR] Startup failure written to: {failure_path}", flush=True)
        finally:
            print(f"[ERROR] Unhandled startup failure: {type(exc).__name__}: {exc}", flush=True)
    finally:
        simulation_app.close()
