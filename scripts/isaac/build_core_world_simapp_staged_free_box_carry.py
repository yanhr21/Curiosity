#!/usr/bin/env python3
"""Pure SimulationApp staged free-box carry diagnostic.

This scene keeps the project moving on the direct Isaac path.  The box starts
as a free dynamic rigid body.  A quasi-static walking carrier approaches it,
runs a short probing phase, then activates a selected attach proxy after an
explicit staged lift/hold event and carries the box to a target.

The grasp event is an engineering placeholder, not a contact-grasp success:
it is logged as a staged attach event so later work can replace it with real
hands and learned active probing without changing the task metrics interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Core World staged free-box carry diagnostic.")
    parser.add_argument("--steps", type=int, default=560)
    parser.add_argument("--target-x", type=float, default=0.48)
    parser.add_argument("--box-x", type=float, default=0.28)
    parser.add_argument("--box-mass", type=float, default=8.0)
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.36, 0.24, 0.24), metavar=("X", "Y", "Z"))
    parser.add_argument("--box-com-x", type=float, default=0.04)
    parser.add_argument("--robot-mass", type=float, default=48.0)
    parser.add_argument("--robot-height", type=float, default=1.20)
    parser.add_argument("--arm-length", type=float, default=0.52)
    parser.add_argument("--max-payload", type=float, default=16.0)
    parser.add_argument("--base-speed", type=float, default=0.30)
    parser.add_argument("--gait-frequency", type=float, default=1.15)
    parser.add_argument("--probe-speed", type=float, default=0.045)
    parser.add_argument("--attach-after-step", type=int, default=260)
    parser.add_argument("--carry-geometry-mode", choices=("legacy", "nonpenetrating"), default="legacy")
    parser.add_argument("--carry-clearance", type=float, default=0.03)
    parser.add_argument("--carry-z-offset", type=float, default=0.0)
    parser.add_argument("--contact-proxy-gain", type=float, default=10.0)
    parser.add_argument("--contact-proxy-max-speed", type=float, default=0.95)
    parser.add_argument("--palm-proxy-mass", type=float, default=60.0)
    parser.add_argument("--chest-proxy-mass", type=float, default=80.0)
    parser.add_argument("--shelf-proxy-mass", type=float, default=90.0)
    parser.add_argument("--front-stop-proxy-mass", type=float, default=75.0)
    parser.add_argument("--palm-proxy-thickness", type=float, default=0.055)
    parser.add_argument("--chest-proxy-thickness", type=float, default=0.040)
    parser.add_argument("--front-stop-proxy-thickness", type=float, default=0.035)
    parser.add_argument("--target-hold-radius", type=float, default=0.015)
    parser.add_argument("--target-slow-radius", type=float, default=0.080)
    parser.add_argument("--target-body-margin", type=float, default=0.020)
    parser.add_argument("--body-vertical-mode", choices=("zero", "preserve", "height-servo", "height-lock"), default="zero")
    parser.add_argument("--body-height-gain", type=float, default=18.0)
    parser.add_argument("--body-height-max-z-speed", type=float, default=0.80)
    parser.add_argument("--physical-support-mode", choices=("none", "deck", "runway"), default="none")
    parser.add_argument("--support-deck-gap", type=float, default=0.0)
    parser.add_argument(
        "--attachment-mode",
        choices=(
            "fixed-joint",
            "kinematic-pose-lock",
            "velocity-servo-grasp",
            "contact-proxy-servo",
            "dynamic-contact-proxy",
        ),
        default="fixed-joint",
    )
    parser.add_argument("--carrier-mode", choices=("dynamic-velocity", "kinematic-pose"), default="dynamic-velocity")
    parser.add_argument(
        "--carrier-evidence-mode",
        choices=("support-proxy", "articulated-foot-contact"),
        default="support-proxy",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/core_world_simapp_staged_free_box_carry"),
    )
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()

OV_REGISTRY_MIRROR = os.environ.get("OV_REGISTRY_MIRROR", "/public/home/yanhongru/ov_registry_mirror")
ASSET_ROOT = os.environ.get("ISAACSIM_ASSET_ROOT", "/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0")
kit_args = [
    f"--/persistent/isaac/asset_root/default={ASSET_ROOT}",
    "--/persistent/isaac/asset_root/timeout=1.0",
    f"--/exts/omni.kit.registry.nucleus/registries/0/url={OV_REGISTRY_MIRROR}/kit_prod_default",
    f"--/exts/omni.kit.registry.nucleus/registries/1/url={OV_REGISTRY_MIRROR}/kit_prod_sdk",
]
sys.argv = [sys.argv[0]]

from isaacsim import SimulationApp  # noqa: E402

DEFAULT_EXPERIENCE = "/public/home/yanhongru/Curiosity/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit"
print("[BOOT] Launching pure SimulationApp", flush=True)
simulation_app = SimulationApp(
    {"headless": True, "hide_ui": True, "extra_args": kit_args},
    experience=os.environ.get("ISAAC_SIMAPP_EXPERIENCE", DEFAULT_EXPERIENCE),
)
print("[PROGRESS] Pure SimulationApp started", flush=True)

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage, get_current_stage  # noqa: E402
from pxr import Gf, Sdf, UsdPhysics  # noqa: E402


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


def _xyz(obj: DynamicCuboid | FixedCuboid | VisualCuboid) -> list[float]:
    pos, _quat = obj.get_world_pose()
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _linear_velocity_xyz(obj: DynamicCuboid) -> list[float] | None:
    getter = getattr(obj, "get_linear_velocity", None)
    if getter is None:
        return None
    try:
        vel = getter()
    except Exception:
        return None
    return [float(vel[0]), float(vel[1]), float(vel[2])]


def _bump(summary: dict, key: str, amount: int = 1) -> None:
    summary[key] = int(summary.get(key, 0)) + int(amount)


def _set_body_velocity(body: DynamicCuboid, speed_x: float, summary: dict, target_z: float | None = None) -> None:
    if args_cli.body_vertical_mode == "preserve":
        vel = _linear_velocity_xyz(body)
        if vel is None:
            summary["body_vertical_velocity_preserve_available"] = False
            z_vel = 0.0
        else:
            summary["body_vertical_velocity_preserve_available"] = True
            z_vel = vel[2]
    elif args_cli.body_vertical_mode == "height-servo":
        current_z = _xyz(body)[2]
        z_err = 0.0 if target_z is None else float(target_z) - float(current_z)
        z_vel = float(args_cli.body_height_gain) * z_err
        max_z_speed = abs(float(args_cli.body_height_max_z_speed))
        if max_z_speed > 0.0:
            z_vel = max(-max_z_speed, min(max_z_speed, z_vel))
        summary["body_vertical_velocity_preserve_available"] = False
    else:
        z_vel = 0.0
    body.set_linear_velocity(np.array([float(speed_x), 0.0, float(z_vel)], dtype=float))
    _bump(summary, "body_root_velocity_command_count")


def _select_strategy() -> dict[str, float | str]:
    load_ratio = float(args_cli.box_mass) / max(float(args_cli.max_payload), 1e-6)
    reach_ratio = float(args_cli.arm_length) / max(float(args_cli.robot_height), 1e-6)
    if load_ratio > 0.70 or reach_ratio < 0.40:
        return {"name": "chest_supported_creep", "speed_scale": 0.42, "body_z": 0.48, "stance_w": 0.42, "step_l": 0.15, "carry_x": 0.02}
    if load_ratio > 0.45 or abs(float(args_cli.box_com_x)) > 0.03:
        return {"name": "low_front_creep", "speed_scale": 0.58, "body_z": 0.38, "stance_w": 0.38, "step_l": 0.17, "carry_x": 0.10}
    return {"name": "front_carry_walk", "speed_scale": 0.72, "body_z": 0.44, "stance_w": 0.34, "step_l": 0.20, "carry_x": 0.16}


def _define_disabled_fixed_joint(body0: str, body1: str, initial_local_pos0: tuple[float, float, float]) -> UsdPhysics.FixedJoint:
    stage = get_current_stage()
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/StagedCarryRuntimeFixedJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(float(initial_local_pos0[0]), float(initial_local_pos0[1]), float(initial_local_pos0[2]))
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(False)
    return joint


def _enable_fixed_joint(joint: UsdPhysics.FixedJoint, local_pos0: tuple[float, float, float]) -> None:
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(float(local_pos0[0]), float(local_pos0[1]), float(local_pos0[2])))
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.GetJointEnabledAttr().Set(True)


def _phase(
    step: int,
    attached: bool,
    attach_prepared: bool,
    body_x: float,
    box_x: float,
    target_dist: float,
    approach_body_x: float,
) -> str:
    if attached:
        return "target_hold" if target_dist < float(args_cli.target_hold_radius) else "carry_to_target"
    if attach_prepared:
        return "staged_attach_constraint"
    if body_x < approach_body_x:
        return "approach_free_box"
    if step < int(args_cli.attach_after_step):
        return "probe_free_box"
    if not attach_prepared:
        return "staged_lift_settle"
    return "staged_attach_constraint"


def _foot_targets(step: int, body_x: float, strategy: dict[str, float | str]) -> dict[str, tuple[float, float, float]]:
    cycle = (float(args_cli.gait_frequency) * step * 0.005) % 1.0
    step_l = float(strategy["step_l"])
    stance_w = float(strategy["stance_w"])
    base = {
        "lf": (0.12, 0.5 * stance_w),
        "rf": (0.12, -0.5 * stance_w),
        "lh": (-0.16, 0.5 * stance_w),
        "rh": (-0.16, -0.5 * stance_w),
    }
    order = ("lf", "rh", "rf", "lh")
    targets: dict[str, tuple[float, float, float]] = {}
    for name, (x0, y0) in base.items():
        phase = (cycle - 0.25 * order.index(name)) % 1.0
        if phase < 0.18:
            swing = phase / 0.18
            z = 0.025 + 0.045 * math.sin(math.pi * swing)
            x = body_x + x0 + step_l * (swing - 0.5)
        else:
            stance = (phase - 0.18) / 0.82
            z = 0.025
            x = body_x + x0 + step_l * (0.5 - stance)
        targets[name] = (float(x), float(y0), float(z))
    return targets


def _support_metrics(body_x: float, speed: float, feet: dict[str, tuple[float, float, float]]) -> dict[str, float | str]:
    stance = {name: xyz for name, xyz in feet.items() if xyz[2] < 0.035}
    if len(stance) < 2:
        stance = feet
    xs = [xyz[0] for xyz in stance.values()]
    ys = [xyz[1] for xyz in stance.values()]
    support_min_x = min(xs) - 0.08
    support_max_x = max(xs) + 0.08
    support_min_y = min(ys) - 0.05
    support_max_y = max(ys) + 0.05
    load_shift = float(args_cli.box_com_x) + 0.018 * float(args_cli.box_mass) / max(float(args_cli.robot_mass), 1e-6)
    com_x = body_x + load_shift + 0.02 * math.tanh(speed * 4.0)
    margin_x = min(com_x - support_min_x, support_max_x - com_x)
    margin_y = min(0.0 - support_min_y, support_max_y - 0.0)
    return {
        "stance_count": float(len(stance)),
        "support_margin_min_m": float(min(margin_x, margin_y)),
        "support_state": "+".join(sorted(stance.keys())),
    }


def _write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _contact_proxy_poses(
    box_center: list[float] | tuple[float, float, float],
    box_size: tuple[float, float, float],
    body_x: float,
    body_size: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    box_xyz = np.array(box_center, dtype=float)
    palm_y = 0.5 * float(box_size[1]) + 0.025
    chest_x = body_x + 0.5 * float(body_size[0]) + 0.025
    front_x = box_xyz[0] + 0.5 * float(box_size[0]) + 0.025
    return {
        "left_palm": np.array([box_xyz[0], palm_y, box_xyz[2]], dtype=float),
        "right_palm": np.array([box_xyz[0], -palm_y, box_xyz[2]], dtype=float),
        "chest_pad": np.array([chest_x, 0.0, box_xyz[2]], dtype=float),
        "forearm_shelf": np.array([box_xyz[0], 0.0, box_xyz[2] - 0.5 * float(box_size[2]) - 0.022], dtype=float),
        "front_stop": np.array([front_x, 0.0, box_xyz[2]], dtype=float),
    }


def _contact_proxy_gap(
    box_center: list[float] | tuple[float, float, float],
    box_size: tuple[float, float, float],
    body_x: float,
    body_size: tuple[float, float, float],
    poses: dict[str, np.ndarray],
) -> float:
    box_xyz = np.array(box_center, dtype=float)
    expected_palm_y = 0.5 * float(box_size[1]) + 0.025
    expected_chest_x = body_x + 0.5 * float(body_size[0]) + 0.025
    expected_front_x = box_xyz[0] + 0.5 * float(box_size[0]) + 0.025
    return float(
        max(
            abs(float(poses["left_palm"][1]) - (box_xyz[1] + expected_palm_y)),
            abs(float(poses["right_palm"][1]) - (box_xyz[1] - expected_palm_y)),
            abs(float(poses["left_palm"][0]) - box_xyz[0]),
            abs(float(poses["right_palm"][0]) - box_xyz[0]),
            abs(float(poses["chest_pad"][0]) - expected_chest_x),
            abs(float(poses["chest_pad"][2]) - box_xyz[2]),
            abs(float(poses["forearm_shelf"][0]) - box_xyz[0]),
            abs(float(poses["forearm_shelf"][2]) - (box_xyz[2] - 0.5 * float(box_size[2]) - 0.022)),
            abs(float(poses["front_stop"][0]) - expected_front_x),
            abs(float(poses["front_stop"][2]) - box_xyz[2]),
        )
    )


def _set_velocity_toward(
    obj: DynamicCuboid,
    target_xyz: np.ndarray,
    base_x_speed: float,
    gain: float = 10.0,
    max_speed: float = 0.95,
) -> None:
    pos = np.array(_xyz(obj), dtype=float)
    vel = np.array([base_x_speed, 0.0, 0.0], dtype=float) + gain * (target_xyz - pos)
    speed = float(np.linalg.norm(vel))
    if speed > max_speed:
        vel *= max_speed / speed
    obj.set_linear_velocity(vel)


def run() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_simapp_staged_free_box_carry_state.csv"
    summary_path = args_cli.output_dir / "core_world_simapp_staged_free_box_carry_summary.json"
    strategy = _select_strategy()
    strategy_name = str(strategy["name"])
    box_size = tuple(float(v) for v in args_cli.box_size)
    body_size = (0.46, 0.24, 0.18)
    if args_cli.carry_geometry_mode == "nonpenetrating":
        nonpenetrating_carry_x = body_size[0] + box_size[0] + float(args_cli.carry_clearance)
    else:
        nonpenetrating_carry_x = 0.5 * body_size[0] + 0.5 * box_size[0] + 0.02
    carry_x = max(float(strategy["carry_x"]), nonpenetrating_carry_x)
    approach_body_x = float(args_cli.box_x) - (0.16 if args_cli.carry_geometry_mode == "legacy" else float(carry_x))

    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    create_new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    world.scene.add(FixedCuboid("/World/Ground", "ground", position=np.array([0.0, 0.0, -0.025]), scale=np.array([5.0, 2.0, 0.05]), color=np.array([0.32, 0.34, 0.34])))
    target = world.scene.add(FixedCuboid("/World/CarryTarget", "carry_target", position=np.array([float(args_cli.target_x), 0.0, 0.01]), scale=np.array([0.20, 0.34, 0.02]), color=np.array([0.05, 0.40, 0.85])))
    body_z = float(strategy["body_z"])
    body_is_dynamic = args_cli.attachment_mode in (
        "fixed-joint",
        "velocity-servo-grasp",
        "contact-proxy-servo",
        "dynamic-contact-proxy",
    ) or args_cli.carrier_mode == "dynamic-velocity"
    if body_is_dynamic:
        body = world.scene.add(DynamicCuboid("/World/WalkerBody", "walker_body", position=np.array([0.0, 0.0, body_z]), scale=np.array(body_size), color=np.array([0.12, 0.19, 0.29]), mass=float(args_cli.robot_mass)))
    else:
        body = world.scene.add(FixedCuboid("/World/WalkerBody", "walker_body", position=np.array([0.0, 0.0, body_z]), scale=np.array(body_size), color=np.array([0.12, 0.19, 0.29])))
    support_deck = None
    support_surface_top_z = None
    initial_body_bottom_z = float(body_z - body_size[2])
    if args_cli.physical_support_mode in ("deck", "runway"):
        deck_scale = np.array([0.72, 0.42, 0.05]) if args_cli.physical_support_mode == "deck" else np.array([2.2, 0.48, 0.05])
        # Isaac core cuboid scale behaves like half-extent in these diagnostics.
        deck_z = body_z - body_size[2] - float(deck_scale[2]) - float(args_cli.support_deck_gap)
        support_surface_top_z = float(deck_z + float(deck_scale[2]))
        support_deck = world.scene.add(FixedCuboid("/World/PhysicalSupportDeck", "physical_support_deck", position=np.array([0.0, 0.0, deck_z]), scale=deck_scale, color=np.array([0.18, 0.48, 0.38])))
    box = world.scene.add(DynamicCuboid("/World/FreeCarryBox", "free_carry_box", position=np.array([float(args_cli.box_x), 0.0, box_size[2] * 0.5]), scale=np.array(box_size), color=np.array([0.58, 0.43, 0.24]), mass=float(args_cli.box_mass)))
    initial_joint_local_pos0 = (float(args_cli.box_x), 0.0, box_size[2] * 0.5 - body_z)
    carry_joint = None
    if args_cli.attachment_mode == "fixed-joint":
        carry_joint = _define_disabled_fixed_joint("/World/WalkerBody", "/World/FreeCarryBox", initial_joint_local_pos0)

    feet = {}
    legs = {}
    for name in ("lf", "rf", "lh", "rh"):
        feet[name] = world.scene.add(FixedCuboid(f"/World/Foot_{name}", f"foot_{name}", position=np.array([0.0, 0.0, 0.025]), scale=np.array([0.16, 0.08, 0.03]), color=np.array([0.10, 0.36, 0.72])))
        legs[name] = world.scene.add(VisualCuboid(f"/World/Leg_{name}", f"leg_{name}", position=np.array([0.0, 0.0, 0.2]), scale=np.array([0.035, 0.035, 0.36]), color=np.array([0.18, 0.22, 0.26])))
    hold_marker = world.scene.add(VisualCuboid("/World/HoldZone", "hold_zone", position=np.array([0.0, 0.0, body_z]), scale=np.array([0.08, 0.30, 0.18]), color=np.array([0.84, 0.18, 0.16])))
    proxy_is_dynamic = args_cli.attachment_mode == "dynamic-contact-proxy"
    shelf_size = np.array([box_size[0] + 0.08, box_size[1] + 0.12, 0.035], dtype=float)
    if proxy_is_dynamic:
        left_palm = world.scene.add(DynamicCuboid("/World/LeftPalmProxy", "left_palm_proxy", position=np.array([0.0, 0.0, body_z]), scale=np.array([float(args_cli.palm_proxy_thickness), 0.035, 0.16]), color=np.array([0.85, 0.70, 0.18]), mass=float(args_cli.palm_proxy_mass)))
        right_palm = world.scene.add(DynamicCuboid("/World/RightPalmProxy", "right_palm_proxy", position=np.array([0.0, 0.0, body_z]), scale=np.array([float(args_cli.palm_proxy_thickness), 0.035, 0.16]), color=np.array([0.85, 0.70, 0.18]), mass=float(args_cli.palm_proxy_mass)))
        chest_pad = world.scene.add(DynamicCuboid("/World/ChestSupportProxy", "chest_support_proxy", position=np.array([0.0, 0.0, body_z]), scale=np.array([float(args_cli.chest_proxy_thickness), 0.24, 0.18]), color=np.array([0.90, 0.42, 0.14]), mass=float(args_cli.chest_proxy_mass)))
        forearm_shelf = world.scene.add(DynamicCuboid("/World/ForearmShelfProxy", "forearm_shelf_proxy", position=np.array([0.0, 0.0, body_z]), scale=shelf_size, color=np.array([0.25, 0.58, 0.36]), mass=float(args_cli.shelf_proxy_mass)))
        front_stop = world.scene.add(DynamicCuboid("/World/FrontStopProxy", "front_stop_proxy", position=np.array([0.0, 0.0, body_z]), scale=np.array([float(args_cli.front_stop_proxy_thickness), box_size[1] + 0.08, box_size[2]]), color=np.array([0.55, 0.22, 0.70]), mass=float(args_cli.front_stop_proxy_mass)))
    else:
        left_palm = world.scene.add(VisualCuboid("/World/LeftPalmProxy", "left_palm_proxy", position=np.array([0.0, 0.0, body_z]), scale=np.array([0.055, 0.035, 0.16]), color=np.array([0.85, 0.70, 0.18])))
        right_palm = world.scene.add(VisualCuboid("/World/RightPalmProxy", "right_palm_proxy", position=np.array([0.0, 0.0, body_z]), scale=np.array([0.055, 0.035, 0.16]), color=np.array([0.85, 0.70, 0.18])))
        chest_pad = world.scene.add(VisualCuboid("/World/ChestSupportProxy", "chest_support_proxy", position=np.array([0.0, 0.0, body_z]), scale=np.array([0.04, 0.24, 0.18]), color=np.array([0.90, 0.42, 0.14])))
        forearm_shelf = world.scene.add(VisualCuboid("/World/ForearmShelfProxy", "forearm_shelf_proxy", position=np.array([0.0, 0.0, body_z]), scale=shelf_size, color=np.array([0.25, 0.58, 0.36])))
        front_stop = world.scene.add(VisualCuboid("/World/FrontStopProxy", "front_stop_proxy", position=np.array([0.0, 0.0, body_z]), scale=np.array([0.035, box_size[1] + 0.08, box_size[2]]), color=np.array([0.55, 0.22, 0.70])))

    def drive_proxy(obj: DynamicCuboid, target_xyz: np.ndarray, base_x_speed: float) -> None:
        _set_velocity_toward(
            obj,
            target_xyz,
            base_x_speed,
            gain=float(args_cli.contact_proxy_gain),
            max_speed=float(args_cli.contact_proxy_max_speed),
        )

    print(f"[PROGRESS] Strategy selected: {strategy_name}", flush=True)
    world.reset()
    initial_body = _xyz(body)
    initial_box = _xyz(box)
    target_xyz = _xyz(target)
    attached = False
    attach_prepared = False
    attach_prep_step = None
    attach_step = None
    attach_local_pos0 = None
    probe_start_x = None
    probe_attempts = 0
    min_target_dist = None
    min_margin = None
    max_rel_err = None
    grip_gap = None
    max_grip_gap = None
    target_hold_steps = 0
    carry_phase_steps = 0
    min_stance_count = None
    min_margin_after_attach = None
    max_command_speed = 0.0
    target_hold_latched = False

    articulated_carrier_requested = bool(args_cli.carrier_evidence_mode == "articulated-foot-contact")
    articulated_carrier_enabled = False
    if articulated_carrier_requested:
        carrier_claim = (
            "articulated_foot_contact_requested_but_not_implemented_in_this_scaffold_"
            "dynamic_rigidbody_velocity_commanded_body_still_used"
        )
    elif body_is_dynamic:
        carrier_claim = "dynamic_rigidbody_velocity_commanded_body_with_support_proxy_not_verified_articulated_locomotion"
    else:
        carrier_claim = "kinematic_body_pose_commanded_support_proxy"

    summary = {
        "scene_type": "pure_simapp_core_world_staged_free_box_carry",
        "success_claim": "staged_free_box_diagnostic_not_contact_grasp_not_articulated_locomotion_not_learned_policy",
        "attachment_mode": str(args_cli.attachment_mode),
        "attachment_claim": {
            "fixed-joint": "preauthored_disabled_fixed_joint_enabled_after_explicit_staged_lift_placeholder",
            "kinematic-pose-lock": "kinematic_pose_lock_is_a_task_scaffold_not_physical_grasp",
            "velocity-servo-grasp": "dynamic_box_velocity_servo_grasp_proxy_not_contact_grasp",
            "contact-proxy-servo": "dynamic_box_velocity_servo_with_explicit_hand_chest_contact_proxy_not_contact_grasp",
            "dynamic-contact-proxy": "dynamic_hand_chest_shelf_contact_proxy_box_not_directly_servoed_after_attach_diagnostic",
        }[str(args_cli.attachment_mode)],
        "contact_proxy_enabled": bool(args_cli.attachment_mode in ("contact-proxy-servo", "dynamic-contact-proxy")),
        "dynamic_contact_proxy_enabled": bool(proxy_is_dynamic),
        "body_vertical_mode": str(args_cli.body_vertical_mode),
        "body_height_gain": float(args_cli.body_height_gain),
        "body_height_max_z_speed": float(args_cli.body_height_max_z_speed),
        "physical_support_mode": str(args_cli.physical_support_mode),
        "support_deck_gap_m": float(args_cli.support_deck_gap),
        "support_surface_top_z_m": support_surface_top_z,
        "initial_body_bottom_z_m": initial_body_bottom_z,
        "initial_body_support_clearance_m": None if support_surface_top_z is None else float(initial_body_bottom_z - support_surface_top_z),
        "body_vertical_velocity_preserve_available": None,
        "carrier_mode": str(args_cli.carrier_mode),
        "carrier_evidence_mode": str(args_cli.carrier_evidence_mode),
        "articulated_carrier_requested": articulated_carrier_requested,
        "articulated_carrier_enabled": articulated_carrier_enabled,
        "articulated_joint_count": 0,
        "foot_contact_drive_enabled": False,
        "carrier_claim": carrier_claim,
        "body_root_velocity_command_count": 0,
        "body_root_pose_write_count": 0,
        "box_pose_write_count": 0,
        "box_velocity_command_count": 0,
        "device": args_cli.device,
        "strategy": strategy_name,
        "strategy_requested_carry_x_m": float(strategy["carry_x"]),
        "carry_geometry_mode": str(args_cli.carry_geometry_mode),
        "carry_clearance_m": float(args_cli.carry_clearance),
        "carry_z_offset_m": float(args_cli.carry_z_offset),
        "contact_proxy_gain": float(args_cli.contact_proxy_gain),
        "contact_proxy_max_speed": float(args_cli.contact_proxy_max_speed),
        "palm_proxy_mass_kg": float(args_cli.palm_proxy_mass),
        "chest_proxy_mass_kg": float(args_cli.chest_proxy_mass),
        "shelf_proxy_mass_kg": float(args_cli.shelf_proxy_mass),
        "front_stop_proxy_mass_kg": float(args_cli.front_stop_proxy_mass),
        "palm_proxy_thickness_m": float(args_cli.palm_proxy_thickness),
        "chest_proxy_thickness_m": float(args_cli.chest_proxy_thickness),
        "front_stop_proxy_thickness_m": float(args_cli.front_stop_proxy_thickness),
        "nonpenetrating_carry_x_m": float(nonpenetrating_carry_x),
        "actual_staged_carry_x_m": float(carry_x),
        "approach_body_x_m": float(approach_body_x),
        "steps_requested": int(args_cli.steps),
        "attach_after_step": int(args_cli.attach_after_step),
        "target_hold_radius_m": float(args_cli.target_hold_radius),
        "target_slow_radius_m": float(args_cli.target_slow_radius),
        "target_body_margin_m": float(args_cli.target_body_margin),
        "target_body_x_m": None,
        "target_hold_latched": False,
        "attach_prep_step": None,
        "completed_steps": 0,
        "attach_step": None,
        "attach_local_pos0_m": None,
        "probe_attempts": 0,
        "free_box_probe_displacement_x_m": 0.0,
        "body_travel_x_m": 0.0,
        "box_travel_x_m": 0.0,
        "box_relative_error_m_after_attach": None,
        "max_box_relative_error_m_after_attach": None,
        "contact_proxy_grip_gap_m": None,
        "max_contact_proxy_grip_gap_m": None,
        "final_box_target_distance_xy_m": None,
        "min_box_target_distance_xy_m": None,
        "min_support_margin_m": None,
        "min_support_margin_after_attach_m": None,
        "min_stance_count": None,
        "target_hold_steps": 0,
        "carry_phase_steps": 0,
        "max_command_speed_mps": 0.0,
        "initial_body_z_m": float(initial_body[2]),
        "min_body_z_m": float(initial_body[2]),
        "max_body_z_deviation_m": 0.0,
        "fall_events": 0,
        "box_drop_events": 0,
        "balance_gate_slowdowns": 0,
        "error": None,
    }

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "phase", "strategy", "attached", "command_speed", "body_x", "box_x",
                "box_z", "box_target_distance_xy", "support_state", "support_margin_min",
                "box_relative_error_m_after_attach", "contact_proxy_grip_gap_m", "probe_attempts",
                "free_box_probe_displacement_x_m", "fall", "box_drop",
            ])
            for step in range(int(args_cli.steps)):
                bpos = _xyz(body)
                box_pos = _xyz(box)
                target_dist = float(math.hypot(box_pos[0] - target_xyz[0], box_pos[1] - target_xyz[1]))
                if attached and target_hold_latched:
                    phase = "target_hold"
                else:
                    phase = _phase(step, attached, attach_prepared, bpos[0], box_pos[0], target_dist, approach_body_x)
                    if attached and phase == "target_hold" and attach_local_pos0 is not None:
                        target_body_x = float(target_xyz[0]) - float(attach_local_pos0[0])
                        summary["target_body_x_m"] = target_body_x
                        if float(bpos[0]) < target_body_x - float(args_cli.target_body_margin):
                            phase = "carry_to_target"
                    if attached and phase == "target_hold":
                        target_hold_latched = True
                foot_targets = _foot_targets(step, bpos[0], strategy)
                for name, xyz in foot_targets.items():
                    feet[name].set_world_pose(position=np.array(xyz, dtype=float))
                    leg_mid = np.array([(bpos[0] + xyz[0]) * 0.5, xyz[1] * 0.5, (body_z + xyz[2]) * 0.5], dtype=float)
                    legs[name].set_world_pose(position=leg_mid)
                hold_pos = np.array([bpos[0] + carry_x, 0.0, body_z + float(args_cli.carry_z_offset)], dtype=float)
                hold_marker.set_world_pose(position=hold_pos)
                if support_deck is not None and args_cli.physical_support_mode == "deck":
                    support_deck.set_world_pose(
                        position=np.array(
                            [bpos[0], 0.0, body_z - body_size[2] - 0.05 - float(args_cli.support_deck_gap)],
                            dtype=float,
                        )
                    )
                proxy_box = (
                    [bpos[0] + float(attach_local_pos0[0]), bpos[1] + float(attach_local_pos0[1]), bpos[2] + float(attach_local_pos0[2])]
                    if attached and attach_local_pos0 is not None
                    else hold_pos
                )
                proxy_poses = _contact_proxy_poses(proxy_box, box_size, bpos[0], body_size)
                if proxy_is_dynamic:
                    if phase in ("approach_free_box", "probe_free_box"):
                        standby_x = bpos[0] - 0.65
                        standby = {
                            "left_palm": np.array([standby_x, 0.45, body_z], dtype=float),
                            "right_palm": np.array([standby_x, -0.45, body_z], dtype=float),
                            "chest_pad": np.array([standby_x - 0.10, 0.0, body_z], dtype=float),
                            "forearm_shelf": np.array([standby_x, 0.0, 0.12], dtype=float),
                            "front_stop": np.array([standby_x + 0.18, 0.0, body_z], dtype=float),
                        }
                        drive_proxy(left_palm, standby["left_palm"], 0.0)
                        drive_proxy(right_palm, standby["right_palm"], 0.0)
                        drive_proxy(chest_pad, standby["chest_pad"], 0.0)
                        drive_proxy(forearm_shelf, standby["forearm_shelf"], 0.0)
                        drive_proxy(front_stop, standby["front_stop"], 0.0)
                    else:
                        drive_proxy(left_palm, proxy_poses["left_palm"], 0.0)
                        drive_proxy(right_palm, proxy_poses["right_palm"], 0.0)
                        drive_proxy(chest_pad, proxy_poses["chest_pad"], 0.0)
                        drive_proxy(forearm_shelf, proxy_poses["forearm_shelf"], 0.0)
                        drive_proxy(front_stop, proxy_poses["front_stop"], 0.0)
                else:
                    left_palm.set_world_pose(position=proxy_poses["left_palm"])
                    right_palm.set_world_pose(position=proxy_poses["right_palm"])
                    chest_pad.set_world_pose(position=proxy_poses["chest_pad"])
                    forearm_shelf.set_world_pose(position=proxy_poses["forearm_shelf"])
                    front_stop.set_world_pose(position=proxy_poses["front_stop"])

                speed = 0.0
                if phase == "approach_free_box":
                    speed = min(float(args_cli.base_speed) * 0.45, 0.10)
                elif phase == "probe_free_box":
                    if probe_start_x is None:
                        probe_start_x = box_pos[0]
                    probe_attempts += 1
                    if probe_attempts % 34 < 17:
                        box.set_linear_velocity(np.array([float(args_cli.probe_speed), 0.0, 0.0], dtype=float))
                        _bump(summary, "box_velocity_command_count")
                    else:
                        box.set_linear_velocity(np.array([-float(args_cli.probe_speed) * 0.35, 0.0, 0.0], dtype=float))
                        _bump(summary, "box_velocity_command_count")
                elif phase == "staged_lift_settle":
                    box.set_world_pose(position=hold_pos)
                    box.set_linear_velocity(np.zeros(3, dtype=float))
                    _bump(summary, "box_pose_write_count")
                    _bump(summary, "box_velocity_command_count")
                    if proxy_is_dynamic:
                        lift_proxy_poses = _contact_proxy_poses(hold_pos, box_size, bpos[0], body_size)
                        left_palm.set_world_pose(position=lift_proxy_poses["left_palm"])
                        right_palm.set_world_pose(position=lift_proxy_poses["right_palm"])
                        chest_pad.set_world_pose(position=lift_proxy_poses["chest_pad"])
                        forearm_shelf.set_world_pose(position=lift_proxy_poses["forearm_shelf"])
                        front_stop.set_world_pose(position=lift_proxy_poses["front_stop"])
                        left_palm.set_linear_velocity(np.zeros(3, dtype=float))
                        right_palm.set_linear_velocity(np.zeros(3, dtype=float))
                        chest_pad.set_linear_velocity(np.zeros(3, dtype=float))
                        forearm_shelf.set_linear_velocity(np.zeros(3, dtype=float))
                        front_stop.set_linear_velocity(np.zeros(3, dtype=float))
                    attach_prepared = True
                    attach_prep_step = step
                    speed = 0.0
                    print(f"[EVENT] staged lift/hold settle at step={step}", flush=True)
                elif phase == "staged_attach_constraint":
                    attach_local_pos0 = (
                        float(box_pos[0] - bpos[0]),
                        float(box_pos[1] - bpos[1]),
                        float(box_pos[2] - bpos[2]),
                    )
                    box.set_linear_velocity(np.zeros(3, dtype=float))
                    _bump(summary, "box_velocity_command_count")
                    if args_cli.attachment_mode == "fixed-joint":
                        assert carry_joint is not None
                        _enable_fixed_joint(carry_joint, attach_local_pos0)
                    attached = True
                    attach_step = step
                    speed = 0.0
                    print(f"[EVENT] staged attach at step={step} mode={args_cli.attachment_mode} local_pos0={attach_local_pos0}", flush=True)
                elif phase == "carry_to_target":
                    speed = float(args_cli.base_speed) * float(strategy["speed_scale"])
                    if float(args_cli.target_slow_radius) > 0.0 and target_dist < float(args_cli.target_slow_radius):
                        speed *= max(0.20, target_dist / float(args_cli.target_slow_radius))

                if phase == "target_hold":
                    target_hold_steps += 1
                if phase in ("carry_to_target", "target_hold"):
                    carry_phase_steps += 1

                metrics = _support_metrics(bpos[0], speed, foot_targets)
                if float(metrics["support_margin_min_m"]) < 0.04:
                    speed *= 0.35
                    summary["balance_gate_slowdowns"] += 1
                max_command_speed = max(max_command_speed, abs(float(speed)))
                if args_cli.attachment_mode == "kinematic-pose-lock" and args_cli.carrier_mode == "kinematic-pose":
                    next_body = np.array([bpos[0] + speed * 0.005, 0.0, body_z], dtype=float)
                    body.set_world_pose(position=next_body)
                    _bump(summary, "body_root_pose_write_count")
                    if attached and attach_local_pos0 is not None:
                        next_box = np.array(
                            [
                                next_body[0] + float(attach_local_pos0[0]),
                                next_body[1] + float(attach_local_pos0[1]),
                                next_body[2] + float(attach_local_pos0[2]),
                            ],
                            dtype=float,
                        )
                        box.set_world_pose(position=next_box)
                        box.set_linear_velocity(np.zeros(3, dtype=float))
                        _bump(summary, "box_pose_write_count")
                        _bump(summary, "box_velocity_command_count")
                elif args_cli.attachment_mode == "kinematic-pose-lock" and args_cli.carrier_mode == "dynamic-velocity":
                    _set_body_velocity(body, speed, summary, body_z)
                    if attached and attach_local_pos0 is not None:
                        locked_box = np.array(
                            [
                                bpos[0] + float(attach_local_pos0[0]),
                                bpos[1] + float(attach_local_pos0[1]),
                                bpos[2] + float(attach_local_pos0[2]),
                            ],
                            dtype=float,
                        )
                        box.set_world_pose(position=locked_box)
                        box.set_linear_velocity(np.zeros(3, dtype=float))
                        _bump(summary, "box_pose_write_count")
                        _bump(summary, "box_velocity_command_count")
                elif args_cli.attachment_mode in ("velocity-servo-grasp", "contact-proxy-servo"):
                    _set_body_velocity(body, speed, summary, body_z)
                    if attached and attach_local_pos0 is not None:
                        desired_box = np.array(
                            [
                                bpos[0] + float(attach_local_pos0[0]),
                                bpos[1] + float(attach_local_pos0[1]),
                                bpos[2] + float(attach_local_pos0[2]),
                            ],
                            dtype=float,
                        )
                        err = desired_box - np.array(box_pos, dtype=float)
                        servo_vel = np.array([speed, 0.0, 0.0], dtype=float) + 12.0 * err
                        servo_speed = float(np.linalg.norm(servo_vel))
                        if servo_speed > 0.85:
                            servo_vel *= 0.85 / servo_speed
                        box.set_linear_velocity(servo_vel)
                        _bump(summary, "box_velocity_command_count")
                elif args_cli.attachment_mode == "dynamic-contact-proxy":
                    _set_body_velocity(body, speed, summary, body_z)
                    if attached and attach_local_pos0 is not None:
                        desired_box = np.array(
                            [
                                bpos[0] + float(attach_local_pos0[0]),
                                bpos[1] + float(attach_local_pos0[1]),
                                bpos[2] + float(attach_local_pos0[2]),
                            ],
                            dtype=float,
                        )
                        proxy_targets = _contact_proxy_poses(desired_box, box_size, bpos[0], body_size)
                        drive_proxy(left_palm, proxy_targets["left_palm"], speed)
                        drive_proxy(right_palm, proxy_targets["right_palm"], speed)
                        drive_proxy(chest_pad, proxy_targets["chest_pad"], speed)
                        drive_proxy(forearm_shelf, proxy_targets["forearm_shelf"], speed)
                        drive_proxy(front_stop, proxy_targets["front_stop"], speed)
                else:
                    _set_body_velocity(body, speed, summary, body_z)
                world.step(render=False)
                if args_cli.body_vertical_mode == "height-lock":
                    body_pos_after_step = np.array(_xyz(body), dtype=float)
                    body_pos_after_step[2] = body_z
                    body.set_world_pose(position=body_pos_after_step)
                    body.set_linear_velocity(np.array([float(speed), 0.0, 0.0], dtype=float))
                    _bump(summary, "body_root_pose_write_count")
                    _bump(summary, "body_root_velocity_command_count")

                if args_cli.attachment_mode == "kinematic-pose-lock" and args_cli.carrier_mode == "dynamic-velocity" and attached and attach_local_pos0 is not None:
                    bpos_post_step = _xyz(body)
                    locked_box_post_step = np.array(
                        [
                            bpos_post_step[0] + float(attach_local_pos0[0]),
                            bpos_post_step[1] + float(attach_local_pos0[1]),
                            bpos_post_step[2] + float(attach_local_pos0[2]),
                        ],
                        dtype=float,
                    )
                    box.set_world_pose(position=locked_box_post_step)
                    box.set_linear_velocity(np.zeros(3, dtype=float))
                    _bump(summary, "box_pose_write_count")
                    _bump(summary, "box_velocity_command_count")

                bpos_after = _xyz(body)
                box_after = _xyz(box)
                rel_err_current = None
                if attached and attach_local_pos0 is not None:
                    rel_err_current = float(
                        math.dist(
                            [
                                bpos_after[0] + float(attach_local_pos0[0]),
                                bpos_after[1] + float(attach_local_pos0[1]),
                                bpos_after[2] + float(attach_local_pos0[2]),
                            ],
                            box_after,
                        )
                    )
                    max_rel_err = rel_err_current if max_rel_err is None else max(max_rel_err, rel_err_current)
                    if args_cli.attachment_mode in ("contact-proxy-servo", "dynamic-contact-proxy"):
                        proxy_poses_after = _contact_proxy_poses(
                            [
                                bpos_after[0] + float(attach_local_pos0[0]),
                                bpos_after[1] + float(attach_local_pos0[1]),
                                bpos_after[2] + float(attach_local_pos0[2]),
                            ],
                            box_size,
                            bpos_after[0],
                            body_size,
                        )
                        if proxy_is_dynamic:
                            proxy_poses_after = {
                                "left_palm": np.array(_xyz(left_palm), dtype=float),
                                "right_palm": np.array(_xyz(right_palm), dtype=float),
                                "chest_pad": np.array(_xyz(chest_pad), dtype=float),
                                "forearm_shelf": np.array(_xyz(forearm_shelf), dtype=float),
                                "front_stop": np.array(_xyz(front_stop), dtype=float),
                            }
                        grip_gap = _contact_proxy_gap(box_after, box_size, bpos_after[0], body_size, proxy_poses_after)
                        max_grip_gap = grip_gap if max_grip_gap is None else max(max_grip_gap, grip_gap)
                        if not proxy_is_dynamic:
                            left_palm.set_world_pose(position=proxy_poses_after["left_palm"])
                            right_palm.set_world_pose(position=proxy_poses_after["right_palm"])
                            chest_pad.set_world_pose(position=proxy_poses_after["chest_pad"])
                            forearm_shelf.set_world_pose(position=proxy_poses_after["forearm_shelf"])
                            front_stop.set_world_pose(position=proxy_poses_after["front_stop"])

                if step % 10 == 0 or step == int(args_cli.steps) - 1 or phase in ("staged_lift_settle", "staged_attach_constraint"):
                    target_dist_after = float(math.hypot(box_after[0] - target_xyz[0], box_after[1] - target_xyz[1]))
                    rel_err = rel_err_current
                    fall = int(bpos_after[2] < 0.22)
                    drop = int(attached and box_after[2] < 0.22)
                    probe_disp = 0.0 if probe_start_x is None else float(box_after[0] - probe_start_x)
                    min_target_dist = target_dist_after if min_target_dist is None else min(min_target_dist, target_dist_after)
                    min_margin = float(metrics["support_margin_min_m"]) if min_margin is None else min(min_margin, float(metrics["support_margin_min_m"]))
                    min_stance_count = float(metrics["stance_count"]) if min_stance_count is None else min(min_stance_count, float(metrics["stance_count"]))
                    if attached:
                        min_margin_after_attach = (
                            float(metrics["support_margin_min_m"])
                            if min_margin_after_attach is None
                            else min(min_margin_after_attach, float(metrics["support_margin_min_m"]))
                        )

                    summary["completed_steps"] = step + 1
                    summary["attach_prep_step"] = attach_prep_step
                    summary["attach_step"] = attach_step
                    summary["attach_local_pos0_m"] = None if attach_local_pos0 is None else [float(v) for v in attach_local_pos0]
                    summary["probe_attempts"] = probe_attempts
                    summary["free_box_probe_displacement_x_m"] = probe_disp
                    summary["body_travel_x_m"] = float(bpos_after[0] - initial_body[0])
                    summary["box_travel_x_m"] = float(box_after[0] - initial_box[0])
                    summary["min_body_z_m"] = min(float(summary["min_body_z_m"]), float(bpos_after[2]))
                    summary["max_body_z_deviation_m"] = max(
                        float(summary["max_body_z_deviation_m"]),
                        abs(float(bpos_after[2]) - float(summary["initial_body_z_m"])),
                    )
                    summary["box_relative_error_m_after_attach"] = rel_err
                    summary["max_box_relative_error_m_after_attach"] = max_rel_err
                    summary["contact_proxy_grip_gap_m"] = grip_gap
                    summary["max_contact_proxy_grip_gap_m"] = max_grip_gap
                    summary["final_box_target_distance_xy_m"] = target_dist_after
                    summary["min_box_target_distance_xy_m"] = min_target_dist
                    summary["min_support_margin_m"] = min_margin
                    summary["min_support_margin_after_attach_m"] = min_margin_after_attach
                    summary["min_stance_count"] = min_stance_count
                    summary["target_hold_steps"] = target_hold_steps
                    summary["target_hold_latched"] = target_hold_latched
                    summary["carry_phase_steps"] = carry_phase_steps
                    summary["max_command_speed_mps"] = max_command_speed
                    summary["fall_events"] += fall
                    summary["box_drop_events"] += drop
                    writer.writerow([
                        step, phase, strategy_name, int(attached), speed, bpos_after[0], box_after[0],
                        box_after[2], target_dist_after, metrics["support_state"],
                        metrics["support_margin_min_m"], rel_err, grip_gap, probe_attempts, probe_disp, fall, drop,
                    ])
                    print(
                        f"[STATE] step={step} phase={phase} attached={int(attached)} "
                        f"body_x={bpos_after[0]:.3f} box_x={box_after[0]:.3f} "
                        f"target_dist={target_dist_after:.3f} support={float(metrics['support_margin_min_m']):.3f} "
                        f"fall={fall} drop={drop}",
                        flush=True,
                    )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)

    _write_summary(summary_path, summary)
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
