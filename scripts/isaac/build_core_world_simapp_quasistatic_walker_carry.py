#!/usr/bin/env python3
"""Pure SimulationApp quasi-static walker fixed-payload carry diagnostic.

This is an Isaac core-World bridge between the dynamic fixed-payload carrier
and a later real legged policy.  The carrier and payload are dynamic rigid
bodies connected by a USD fixed joint.  A procedural four-foot walking support
controller updates visible/colliding foot pads and leg struts, gates forward
motion on support-margin proxies, and records gait/balance/carry metrics.

This remains diagnostic-only: the torso translation is still commanded through
core rigid-body velocity control, and the support feet are not a verified
articulated locomotion controller.
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
    parser = argparse.ArgumentParser(description="Core World quasi-static walker carry diagnostic.")
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--target-x", type=float, default=0.38)
    parser.add_argument("--payload-mass", type=float, default=8.0)
    parser.add_argument("--payload-com-x", type=float, default=0.04)
    parser.add_argument("--robot-mass", type=float, default=48.0)
    parser.add_argument("--robot-height", type=float, default=1.20)
    parser.add_argument("--arm-length", type=float, default=0.52)
    parser.add_argument("--max-payload", type=float, default=16.0)
    parser.add_argument("--base-speed", type=float, default=0.30)
    parser.add_argument("--gait-frequency", type=float, default=1.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/core_world_simapp_quasistatic_walker_carry"),
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


def _select_strategy() -> dict[str, float | str]:
    load_ratio = float(args_cli.payload_mass) / max(float(args_cli.max_payload), 1e-6)
    reach_ratio = float(args_cli.arm_length) / max(float(args_cli.robot_height), 1e-6)
    if load_ratio > 0.70 or reach_ratio < 0.40:
        return {"name": "chest_supported_creep", "speed_scale": 0.42, "body_z": 0.48, "stance_w": 0.42, "step_l": 0.15}
    if load_ratio > 0.45 or abs(float(args_cli.payload_com_x)) > 0.03:
        return {"name": "low_front_creep", "speed_scale": 0.58, "body_z": 0.38, "stance_w": 0.38, "step_l": 0.17}
    return {"name": "front_carry_walk", "speed_scale": 0.72, "body_z": 0.44, "stance_w": 0.34, "step_l": 0.20}


def _make_fixed_joint(body0: str, body1: str) -> None:
    stage = get_current_stage()
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/WalkerPayloadFixedJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)


def _phase_for_step(step: int) -> str:
    ratio = step / max(int(args_cli.steps), 1)
    if ratio < 0.18:
        return "probe_and_stance_widen"
    if ratio < 0.34:
        return "payload_settle"
    return "creep_carry"


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


def _support_metrics(body_x: float, speed: float, feet: dict[str, tuple[float, float, float]], strategy: dict[str, float | str]) -> dict[str, float | str]:
    stance = {name: xyz for name, xyz in feet.items() if xyz[2] < 0.035}
    if len(stance) < 2:
        stance = feet
    xs = [xyz[0] for xyz in stance.values()]
    ys = [xyz[1] for xyz in stance.values()]
    support_min_x = min(xs) - 0.08
    support_max_x = max(xs) + 0.08
    support_min_y = min(ys) - 0.05
    support_max_y = max(ys) + 0.05
    load_shift = float(args_cli.payload_com_x) + 0.018 * float(args_cli.payload_mass) / max(float(args_cli.robot_mass), 1e-6)
    com_x = body_x + load_shift + 0.02 * math.tanh(speed * 4.0)
    com_y = 0.0
    margin_x = min(com_x - support_min_x, support_max_x - com_x)
    margin_y = min(com_y - support_min_y, support_max_y - com_y)
    return {
        "stance_count": float(len(stance)),
        "support_margin_x_m": float(margin_x),
        "support_margin_y_m": float(margin_y),
        "support_margin_min_m": float(min(margin_x, margin_y)),
        "support_state": "+".join(sorted(stance.keys())),
    }


def run() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_simapp_quasistatic_walker_carry_state.csv"
    summary_path = args_cli.output_dir / "core_world_simapp_quasistatic_walker_carry_summary.json"
    strategy = _select_strategy()
    strategy_name = str(strategy["name"])

    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    create_new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    world.scene.add(FixedCuboid("/World/Ground", "ground", position=np.array([0.0, 0.0, -0.025]), scale=np.array([5.0, 2.0, 0.05]), color=np.array([0.32, 0.34, 0.34])))
    target = world.scene.add(FixedCuboid("/World/CarryTarget", "carry_target", position=np.array([float(args_cli.target_x), 0.0, 0.01]), scale=np.array([0.20, 0.34, 0.02]), color=np.array([0.05, 0.40, 0.85])))
    body_z = float(strategy["body_z"])
    body = world.scene.add(DynamicCuboid("/World/WalkerBody", "walker_body", position=np.array([0.0, 0.0, body_z]), scale=np.array([0.46, 0.24, 0.18]), color=np.array([0.12, 0.19, 0.29]), mass=float(args_cli.robot_mass)))
    payload = world.scene.add(DynamicCuboid("/World/CarryBox", "carry_box", position=np.array([0.0, 0.0, body_z]), scale=np.array([0.36, 0.24, 0.24]), color=np.array([0.58, 0.43, 0.24]), mass=float(args_cli.payload_mass)))
    _make_fixed_joint("/World/WalkerBody", "/World/CarryBox")

    feet = {}
    legs = {}
    for name in ("lf", "rf", "lh", "rh"):
        feet[name] = world.scene.add(FixedCuboid(f"/World/Foot_{name}", f"foot_{name}", position=np.array([0.0, 0.0, 0.025]), scale=np.array([0.16, 0.08, 0.03]), color=np.array([0.10, 0.36, 0.72])))
        legs[name] = world.scene.add(VisualCuboid(f"/World/Leg_{name}", f"leg_{name}", position=np.array([0.0, 0.0, 0.2]), scale=np.array([0.035, 0.035, 0.36]), color=np.array([0.18, 0.22, 0.26])))

    print(f"[PROGRESS] Strategy selected: {strategy_name}", flush=True)
    world.reset()
    initial_body = _xyz(body)
    initial_payload = _xyz(payload)
    target_xyz = _xyz(target)

    summary = {
        "scene_type": "pure_simapp_core_world_quasistatic_walker_fixed_payload_carry",
        "success_claim": "quasistatic_walking_support_diagnostic_not_verified_articulated_locomotion_or_unknown_free_object_carrying",
        "carrier_claim": "dynamic_rigidbody_velocity_commanded_torso_with_procedural_support_feet_not_articulated_locomotion",
        "articulated_carrier_enabled": False,
        "articulated_joint_count": 0,
        "foot_contact_drive_enabled": False,
        "body_root_velocity_command_count": 0,
        "body_root_pose_write_count": 0,
        "payload_pose_write_count": 0,
        "device": args_cli.device,
        "strategy": strategy_name,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "body_travel_x_m": 0.0,
        "payload_travel_x_m": 0.0,
        "payload_relative_error_m": 0.0,
        "final_payload_target_distance_xy_m": None,
        "min_payload_target_distance_xy_m": None,
        "min_support_margin_m": None,
        "balance_gate_slowdowns": 0,
        "fall_events": 0,
        "payload_drop_events": 0,
        "error": None,
    }

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "phase", "strategy", "command_speed", "body_x", "payload_x", "payload_target_distance_xy", "support_state", "support_margin_min", "stance_count", "fall", "payload_drop"])
            for step in range(int(args_cli.steps)):
                bpos_before = _xyz(body)
                foot_targets = _foot_targets(step, bpos_before[0], strategy)
                for name, xyz in foot_targets.items():
                    feet[name].set_world_pose(position=np.array(xyz, dtype=float))
                    hip_x = bpos_before[0] + (0.12 if name[0] == "l" or name[0] == "r" else 0.0)
                    leg_mid = np.array([(bpos_before[0] + xyz[0]) * 0.5, xyz[1] * 0.5, (body_z + xyz[2]) * 0.5], dtype=float)
                    legs[name].set_world_pose(position=leg_mid)
                phase = _phase_for_step(step)
                base_speed = min(0.08, float(args_cli.base_speed) * 0.25) if phase != "creep_carry" else float(args_cli.base_speed) * float(strategy["speed_scale"])
                ppos_before = _xyz(payload)
                target_dist_before = float(math.hypot(ppos_before[0] - target_xyz[0], ppos_before[1] - target_xyz[1]))
                metrics = _support_metrics(bpos_before[0], base_speed, foot_targets, strategy)
                speed = base_speed
                if target_dist_before < 0.015:
                    speed = 0.0
                if float(metrics["support_margin_min_m"]) < 0.04:
                    speed *= 0.35
                    summary["balance_gate_slowdowns"] += 1
                body.set_linear_velocity(np.array([speed, 0.0, 0.0], dtype=float))
                summary["body_root_velocity_command_count"] += 1
                world.step(render=False)
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    bpos = _xyz(body)
                    ppos = _xyz(payload)
                    target_dist = float(math.hypot(ppos[0] - target_xyz[0], ppos[1] - target_xyz[1]))
                    rel_err = float(math.dist(bpos, ppos))
                    fall = int(bpos[2] < 0.22)
                    drop = int(ppos[2] < 0.22)
                    summary["completed_steps"] = step + 1
                    summary["body_travel_x_m"] = float(bpos[0] - initial_body[0])
                    summary["payload_travel_x_m"] = float(ppos[0] - initial_payload[0])
                    summary["payload_relative_error_m"] = rel_err
                    summary["final_payload_target_distance_xy_m"] = target_dist
                    summary["min_payload_target_distance_xy_m"] = (
                        target_dist
                        if summary["min_payload_target_distance_xy_m"] is None
                        else min(float(summary["min_payload_target_distance_xy_m"]), target_dist)
                    )
                    summary["fall_events"] += fall
                    summary["payload_drop_events"] += drop
                    min_margin = float(metrics["support_margin_min_m"])
                    summary["min_support_margin_m"] = min_margin if summary["min_support_margin_m"] is None else min(float(summary["min_support_margin_m"]), min_margin)
                    writer.writerow([step, phase, strategy_name, speed, bpos[0], ppos[0], target_dist, metrics["support_state"], min_margin, metrics["stance_count"], fall, drop])
                    print(f"[STATE] step={step} phase={phase} strategy={strategy_name} body_x={bpos[0]:.3f} payload_x={ppos[0]:.3f} target_dist={target_dist:.3f} support={min_margin:.3f} fall={fall} drop={drop}", flush=True)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
