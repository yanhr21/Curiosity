#!/usr/bin/env python3
"""Pure SimulationApp dynamic adaptive fixed-payload carry diagnostic.

This extends the verified Isaac Sim core-World dynamic-body path with task
metrics for morphology/load-aware carrying.  The carrier and payload are real
dynamic rigid bodies connected by a USD fixed joint, and the carrier is moved
through Isaac core velocity control.  It also adds a procedural walking-support
proxy, visible foot markers, and gait/balance metrics so the task interface is
closer to the eventual legged controller.  This is still diagnostic-only: it is
not legged walking, contact grasping, unknown free-object lifting, or learned
balance control.
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
    parser = argparse.ArgumentParser(description="Pure SimulationApp adaptive fixed-payload carry diagnostic.")
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--target-x", type=float, default=1.2)
    parser.add_argument("--payload-mass", type=float, default=6.0)
    parser.add_argument("--payload-size-x", type=float, default=0.34)
    parser.add_argument("--payload-size-y", type=float, default=0.24)
    parser.add_argument("--payload-size-z", type=float, default=0.24)
    parser.add_argument("--payload-com-x", type=float, default=0.0)
    parser.add_argument("--robot-height", type=float, default=1.35)
    parser.add_argument("--robot-mass", type=float, default=48.0)
    parser.add_argument("--arm-length", type=float, default=0.55)
    parser.add_argument("--max-payload", type=float, default=16.0)
    parser.add_argument("--base-speed", type=float, default=0.34)
    parser.add_argument("--gait-frequency", type=float, default=1.25)
    parser.add_argument("--foot-clearance", type=float, default=0.035)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset-sweep", choices=("none", "strategy_smoke"), default="none")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/core_world_simapp_adaptive_payload_carry"),
    )
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()

OV_REGISTRY_MIRROR = os.environ.get("OV_REGISTRY_MIRROR", "/public/home/yanhongru/ov_registry_mirror")
ASSET_ROOT = os.environ.get("ISAACSIM_ASSET_ROOT", "/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0")
sys.argv.extend(
    [
        f"--/persistent/isaac/asset_root/default={ASSET_ROOT}",
        "--/persistent/isaac/asset_root/timeout=1.0",
        f"--/exts/omni.kit.registry.nucleus/registries/0/url={OV_REGISTRY_MIRROR}/kit_prod_default",
        f"--/exts/omni.kit.registry.nucleus/registries/1/url={OV_REGISTRY_MIRROR}/kit_prod_sdk",
    ]
)

from isaacsim import SimulationApp  # noqa: E402

DEFAULT_EXPERIENCE = "/public/home/yanhongru/Curiosity/external/IsaacLab-Arena/submodules/IsaacLab/apps/isaaclab.python.headless.kit"
simulation_app = SimulationApp(
    {"headless": True, "hide_ui": True},
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


def _xyz(obj: DynamicCuboid | FixedCuboid | VisualCuboid) -> list[float]:
    pos, _quat = obj.get_world_pose()
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _select_strategy() -> dict[str, float | str]:
    load_ratio = float(args_cli.payload_mass) / max(float(args_cli.max_payload), 1e-6)
    reach_ratio = float(args_cli.arm_length) / max(float(args_cli.robot_height), 1e-6)
    com_penalty = abs(float(args_cli.payload_com_x)) / max(float(args_cli.payload_size_x), 1e-6)
    if load_ratio > 0.70 or reach_ratio < 0.38:
        return {
            "name": "chest_supported_slow",
            "speed_scale": 0.48,
            "support_scale": 0.92,
            "carrier_height": 0.48,
            "stance_width": 0.36,
            "step_length": 0.16,
        }
    if load_ratio > 0.45 or com_penalty > 0.12:
        return {
            "name": "low_front_carry",
            "speed_scale": 0.64,
            "support_scale": 1.08,
            "carrier_height": 0.34,
            "stance_width": 0.34,
            "step_length": 0.18,
        }
    return {
        "name": "front_carry",
        "speed_scale": 0.80,
        "support_scale": 1.0,
        "carrier_height": 0.42,
        "stance_width": 0.30,
        "step_length": 0.22,
    }


def _make_fixed_joint(body0: str, body1: str) -> None:
    stage = get_current_stage()
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/CarrierPayloadFixedJoint")
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
        return "probe_speed_ramp"
    if ratio < 0.32:
        return "posture_settle"
    return "carry_to_target"


def _speed_for_phase(phase: str, strategy_scale: float, support_scale: float) -> float:
    if phase == "probe_speed_ramp":
        return min(0.08, float(args_cli.base_speed) * 0.25)
    if phase == "posture_settle":
        return float(args_cli.base_speed) * 0.40 * support_scale
    return float(args_cli.base_speed) * strategy_scale


def _balance_margin_proxy(payload_x: float, speed: float, support_scale: float) -> float:
    support_half_width = 0.18 * support_scale
    load_ratio = float(args_cli.payload_mass) / max(float(args_cli.robot_mass), 1e-6)
    com_term = abs(float(args_cli.payload_com_x) + payload_x * 0.04)
    speed_term = 0.025 * abs(speed)
    load_term = 0.09 * load_ratio
    return float(support_half_width - com_term - speed_term - load_term)


def _effort_proxy(speed: float) -> float:
    combined_mass = float(args_cli.robot_mass) + float(args_cli.payload_mass)
    return float(combined_mass * (0.10 + abs(speed)) * (1.0 + 0.25 * abs(float(args_cli.payload_com_x))))


def _walking_proxy(step: int, carrier_x: float, speed: float, strategy: dict[str, float | str]) -> dict[str, float | str]:
    phase = 2.0 * math.pi * float(args_cli.gait_frequency) * step * 0.005
    step_length = float(strategy["step_length"])
    stance_width = float(strategy["stance_width"])
    left_phase = math.sin(phase)
    right_phase = math.sin(phase + math.pi)
    left_swing = max(0.0, left_phase)
    right_swing = max(0.0, right_phase)
    left_x = carrier_x - 0.08 + step_length * left_phase
    right_x = carrier_x - 0.08 + step_length * right_phase
    left_z = 0.02 + float(args_cli.foot_clearance) * left_swing
    right_z = 0.02 + float(args_cli.foot_clearance) * right_swing
    support_center = 0.5 * (left_x + right_x)
    support_half_length = 0.12 + 0.5 * step_length
    commanded_com = carrier_x + 0.02 * math.tanh(speed * 3.0)
    support_margin_x = support_half_length - abs(commanded_com - support_center)
    support_state = "left_swing" if left_swing > right_swing else "right_swing"
    if left_swing < 0.05 and right_swing < 0.05:
        support_state = "double_support"
    return {
        "left_foot_x": float(left_x),
        "left_foot_y": float(0.5 * stance_width),
        "left_foot_z": float(left_z),
        "right_foot_x": float(right_x),
        "right_foot_y": float(-0.5 * stance_width),
        "right_foot_z": float(right_z),
        "support_center_x": float(support_center),
        "support_margin_x_proxy_m": float(support_margin_x),
        "support_state": support_state,
    }


def _run_single_case() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_simapp_adaptive_payload_carry_state.csv"
    summary_path = args_cli.output_dir / "core_world_simapp_adaptive_payload_carry_summary.json"

    strategy = _select_strategy()
    strategy_name = str(strategy["name"])
    strategy_speed_scale = float(strategy["speed_scale"])
    support_scale = float(strategy["support_scale"])
    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    print("[PROGRESS] Creating fresh core World stage", flush=True)
    create_new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    print("[PROGRESS] Adding adaptive dynamic carrier scene", flush=True)
    try:
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Ground",
                name="ground",
                position=np.array([0.0, 0.0, -0.025], dtype=float),
                scale=np.array([4.0, 2.0, 0.05], dtype=float),
                color=np.array([0.31, 0.33, 0.33], dtype=float),
            )
        )
        target = world.scene.add(
            FixedCuboid(
                prim_path="/World/CarryTarget",
                name="carry_target",
                position=np.array([float(args_cli.target_x), 0.0, 0.01], dtype=float),
                scale=np.array([0.20, 0.32, 0.02], dtype=float),
                color=np.array([0.05, 0.40, 0.85], dtype=float),
            )
        )
        carrier_height = float(strategy["carrier_height"])
        carrier = world.scene.add(
            DynamicCuboid(
                prim_path="/World/AdaptiveCarrierBody",
                name="adaptive_carrier_body",
                position=np.array([0.0, 0.0, carrier_height], dtype=float),
                scale=np.array([0.46, 0.24, 0.18], dtype=float),
                color=np.array([0.14, 0.20, 0.30], dtype=float),
                mass=float(args_cli.robot_mass),
            )
        )
        payload = world.scene.add(
            DynamicCuboid(
                prim_path="/World/CarryBox",
                name="carry_box",
                position=np.array([0.0, 0.0, carrier_height], dtype=float),
                scale=np.array(
                    [float(args_cli.payload_size_x), float(args_cli.payload_size_y), float(args_cli.payload_size_z)],
                    dtype=float,
                ),
                color=np.array([0.58, 0.43, 0.24], dtype=float),
                mass=float(args_cli.payload_mass),
            )
        )
        _make_fixed_joint("/World/AdaptiveCarrierBody", "/World/CarryBox")
        left_foot = world.scene.add(
            VisualCuboid(
                prim_path="/World/LeftSupportFoot",
                name="left_support_foot",
                position=np.array([-0.08, 0.5 * float(strategy["stance_width"]), 0.02], dtype=float),
                scale=np.array([0.22, 0.09, 0.025], dtype=float),
                color=np.array([0.12, 0.38, 0.70], dtype=float),
            )
        )
        right_foot = world.scene.add(
            VisualCuboid(
                prim_path="/World/RightSupportFoot",
                name="right_support_foot",
                position=np.array([-0.08, -0.5 * float(strategy["stance_width"]), 0.02], dtype=float),
                scale=np.array([0.22, 0.09, 0.025], dtype=float),
                color=np.array([0.12, 0.38, 0.70], dtype=float),
            )
        )
        print(f"[PROGRESS] Strategy selected: {strategy_name}", flush=True)
        print("[PROGRESS] Resetting world", flush=True)
        world.reset()
        print("[PROGRESS] World reset complete", flush=True)
    except BaseException as exc:
        summary_path.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}", "stage": "scene_setup"}, indent=2) + "\n")
        print(f"[ERROR] Scene setup failed: {type(exc).__name__}: {exc}", flush=True)
        return summary_path

    initial_carrier = _xyz(carrier)
    initial_payload = _xyz(payload)
    initial_target = _xyz(target)
    summary = {
        "scene_type": "pure_simapp_core_world_dynamic_adaptive_fixed_payload_carry",
        "success_claim": "dynamic_fixed_payload_with_walking_support_proxy_diagnostic_not_legged_walking_or_unknown_free_object_carrying",
        "device": args_cli.device,
        "strategy": strategy_name,
        "strategy_speed_scale": strategy_speed_scale,
        "support_scale": support_scale,
        "carrier_height_m": float(strategy["carrier_height"]),
        "stance_width_m": float(strategy["stance_width"]),
        "step_length_m": float(strategy["step_length"]),
        "gait_frequency_hz": float(args_cli.gait_frequency),
        "foot_clearance_m": float(args_cli.foot_clearance),
        "payload_mass_kg": float(args_cli.payload_mass),
        "payload_com_x_m": float(args_cli.payload_com_x),
        "robot_height_m": float(args_cli.robot_height),
        "robot_mass_kg": float(args_cli.robot_mass),
        "arm_length_m": float(args_cli.arm_length),
        "max_payload_kg": float(args_cli.max_payload),
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "carrier_travel_x_m": 0.0,
        "payload_travel_x_m": 0.0,
        "payload_relative_error_m": 0.0,
        "final_payload_target_distance_xy_m": None,
        "min_balance_margin_proxy_m": None,
        "min_support_margin_x_proxy_m": None,
        "max_effort_proxy": 0.0,
        "fall_events": 0,
        "payload_drop_events": 0,
        "error": None,
    }

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "time_s",
                    "phase",
                    "strategy",
                    "command_speed",
                    "carrier_x",
                    "carrier_y",
                    "carrier_z",
                    "payload_x",
                    "payload_y",
                    "payload_z",
                    "carrier_travel_x",
                    "payload_travel_x",
                    "relative_error",
                    "payload_target_distance_xy",
                    "balance_margin_proxy",
                    "support_state",
                    "left_foot_x",
                    "left_foot_y",
                    "left_foot_z",
                    "right_foot_x",
                    "right_foot_y",
                    "right_foot_z",
                    "support_center_x",
                    "support_margin_x_proxy",
                    "effort_proxy",
                    "fall",
                    "payload_drop",
                ]
            )
            for step in range(int(args_cli.steps)):
                phase = _phase_for_step(step)
                speed = _speed_for_phase(phase, strategy_speed_scale, support_scale)
                carrier.set_linear_velocity(np.array([speed, 0.0, 0.0], dtype=float))
                cpos_for_feet = _xyz(carrier)
                walk = _walking_proxy(step, cpos_for_feet[0], speed, strategy)
                left_foot.set_world_pose(
                    position=np.array(
                        [float(walk["left_foot_x"]), float(walk["left_foot_y"]), float(walk["left_foot_z"])],
                        dtype=float,
                    )
                )
                right_foot.set_world_pose(
                    position=np.array(
                        [float(walk["right_foot_x"]), float(walk["right_foot_y"]), float(walk["right_foot_z"])],
                        dtype=float,
                    )
                )
                world.step(render=False)
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    cpos = _xyz(carrier)
                    ppos = _xyz(payload)
                    walk = _walking_proxy(step, cpos[0], speed, strategy)
                    carrier_travel = float(cpos[0] - initial_carrier[0])
                    payload_travel = float(ppos[0] - initial_payload[0])
                    relative_error = float(math.dist(cpos, ppos))
                    target_distance = float(math.hypot(ppos[0] - initial_target[0], ppos[1] - initial_target[1]))
                    balance_margin = _balance_margin_proxy(ppos[0] - cpos[0], speed, support_scale)
                    effort = _effort_proxy(speed)
                    fall = int(cpos[2] < 0.22)
                    payload_drop = int(ppos[2] < 0.22)
                    summary["completed_steps"] = step + 1
                    summary["carrier_travel_x_m"] = carrier_travel
                    summary["payload_travel_x_m"] = payload_travel
                    summary["payload_relative_error_m"] = relative_error
                    summary["final_payload_target_distance_xy_m"] = target_distance
                    summary["min_balance_margin_proxy_m"] = (
                        balance_margin
                        if summary["min_balance_margin_proxy_m"] is None
                        else min(float(summary["min_balance_margin_proxy_m"]), balance_margin)
                    )
                    summary["min_support_margin_x_proxy_m"] = (
                        float(walk["support_margin_x_proxy_m"])
                        if summary["min_support_margin_x_proxy_m"] is None
                        else min(float(summary["min_support_margin_x_proxy_m"]), float(walk["support_margin_x_proxy_m"]))
                    )
                    summary["max_effort_proxy"] = max(float(summary["max_effort_proxy"]), effort)
                    summary["fall_events"] += fall
                    summary["payload_drop_events"] += payload_drop
                    writer.writerow(
                        [
                            step,
                            step * 0.005,
                            phase,
                            strategy_name,
                            speed,
                            cpos[0],
                            cpos[1],
                            cpos[2],
                            ppos[0],
                            ppos[1],
                            ppos[2],
                            carrier_travel,
                            payload_travel,
                            relative_error,
                            target_distance,
                            balance_margin,
                            walk["support_state"],
                            walk["left_foot_x"],
                            walk["left_foot_y"],
                            walk["left_foot_z"],
                            walk["right_foot_x"],
                            walk["right_foot_y"],
                            walk["right_foot_z"],
                            walk["support_center_x"],
                            walk["support_margin_x_proxy_m"],
                            effort,
                            fall,
                            payload_drop,
                        ]
                    )
                    print(
                        "[STATE] "
                        f"step={step} phase={phase} strategy={strategy_name} "
                        f"carrier_x={cpos[0]:.3f} payload_x={ppos[0]:.3f} "
                        f"target_dist={target_distance:.3f} balance={balance_margin:.3f} "
                        f"support_x={float(walk['support_margin_x_proxy_m']):.3f} "
                        f"fall={fall} drop={payload_drop}",
                        flush=True,
                    )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


def _strategy_smoke_cases() -> list[dict[str, float | int | str]]:
    return [
        {
            "name": "low_front_target",
            "steps": 360,
            "target_x": 0.30,
            "payload_mass": 8.0,
            "payload_com_x": 0.04,
            "robot_height": 1.35,
            "robot_mass": 48.0,
            "arm_length": 0.52,
            "max_payload": 16.0,
            "base_speed": 0.34,
            "gait_frequency": 1.25,
        },
        {
            "name": "chest_supported_target",
            "steps": 360,
            "target_x": 0.20,
            "payload_mass": 12.0,
            "payload_com_x": 0.02,
            "robot_height": 1.25,
            "robot_mass": 44.0,
            "arm_length": 0.45,
            "max_payload": 15.0,
            "base_speed": 0.34,
            "gait_frequency": 1.05,
        },
    ]


def _apply_case(case: dict[str, float | int | str], root_output_dir: Path) -> None:
    args_cli.output_dir = root_output_dir / str(case["name"])
    args_cli.steps = int(case["steps"])
    args_cli.target_x = float(case["target_x"])
    args_cli.payload_mass = float(case["payload_mass"])
    args_cli.payload_com_x = float(case["payload_com_x"])
    args_cli.robot_height = float(case["robot_height"])
    args_cli.robot_mass = float(case["robot_mass"])
    args_cli.arm_length = float(case["arm_length"])
    args_cli.max_payload = float(case["max_payload"])
    args_cli.base_speed = float(case["base_speed"])
    args_cli.gait_frequency = float(case["gait_frequency"])


def run() -> Path:
    if args_cli.preset_sweep == "none":
        return _run_single_case()

    root_output_dir = args_cli.output_dir
    root_output_dir.mkdir(parents=True, exist_ok=True)
    sweep_summary_path = root_output_dir / "core_world_simapp_adaptive_payload_sweep_summary.json"
    cases = []
    for case in _strategy_smoke_cases():
        World.clear_instance()
        _apply_case(case, root_output_dir)
        print(f"[PROGRESS] Running preset case: {case['name']}", flush=True)
        summary_path = _run_single_case()
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            summary = {"error": f"{type(exc).__name__}: {exc}", "summary_path": str(summary_path)}
        summary["case_name"] = str(case["name"])
        summary["summary_path"] = str(summary_path)
        cases.append(summary)

    strategy_counts: dict[str, int] = {}
    for case in cases:
        strategy = str(case.get("strategy", "missing"))
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    sweep_summary = {
        "scene_type": "pure_simapp_core_world_dynamic_adaptive_fixed_payload_carry_sweep",
        "success_claim": "multi_case_dynamic_fixed_payload_diagnostic_not_legged_walking_or_unknown_free_object_carrying",
        "preset_sweep": args_cli.preset_sweep,
        "case_count": len(cases),
        "strategy_counts": strategy_counts,
        "cases": cases,
    }
    sweep_summary_path.write_text(json.dumps(sweep_summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Sweep summary written to: {sweep_summary_path}", flush=True)
    return sweep_summary_path


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
