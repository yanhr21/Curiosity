#!/usr/bin/env python3
"""Pure SimulationApp core-World dynamic carrier with fixed payload.

This is a dynamic Isaac carry-path diagnostic. A dynamic carrier body receives
velocity commands while a physical payload box is attached by a USD fixed
joint. It is not legged walking, balance control, grasping, or unknown-load
carrying evidence.
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
    parser = argparse.ArgumentParser(description="Pure SimulationApp core World fixed-payload carry diagnostic.")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--target-speed", type=float, default=0.30)
    parser.add_argument("--payload-mass", type=float, default=4.0)
    parser.add_argument("--joint-mode", choices=("center_weld", "front_offset"), default="center_weld")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/core_world_simapp_fixed_payload_carry"))
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
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid  # noqa: E402
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


def _xyz(obj: DynamicCuboid) -> list[float]:
    pos, _quat = obj.get_world_pose()
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _make_fixed_joint(body0: str, body1: str) -> None:
    stage = get_current_stage()
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/CarrierPayloadFixedJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    if args_cli.joint_mode == "front_offset":
        local0 = Gf.Vec3f(0.34, 0.0, 0.08)
        local1 = Gf.Vec3f(0.0, 0.0, 0.0)
    else:
        local0 = Gf.Vec3f(0.0, 0.0, 0.0)
        local1 = Gf.Vec3f(0.0, 0.0, 0.0)
    joint.CreateLocalPos0Attr().Set(local0)
    joint.CreateLocalPos1Attr().Set(local1)
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateCollisionEnabledAttr().Set(False)


def run() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_simapp_fixed_payload_carry_state.csv"
    summary_path = args_cli.output_dir / "core_world_simapp_fixed_payload_carry_summary.json"

    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    print("[PROGRESS] Creating fresh core World stage", flush=True)
    create_new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    print("[PROGRESS] Adding dynamic carrier, payload, and ground", flush=True)
    try:
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Ground",
                name="ground",
                position=np.array([0.0, 0.0, -0.025], dtype=float),
                scale=np.array([5.0, 2.0, 0.05], dtype=float),
                color=np.array([0.31, 0.33, 0.33], dtype=float),
            )
        )
        carrier = world.scene.add(
            DynamicCuboid(
                prim_path="/World/CarrierBody",
                name="carrier_body",
                position=np.array([0.0, 0.0, 0.42], dtype=float),
                scale=np.array([0.46, 0.24, 0.18], dtype=float),
                color=np.array([0.14, 0.20, 0.30], dtype=float),
                mass=14.0,
            )
        )
        payload_position = np.array([0.34, 0.0, 0.50], dtype=float)
        if args_cli.joint_mode == "center_weld":
            payload_position = np.array([0.0, 0.0, 0.42], dtype=float)
        payload = world.scene.add(
            DynamicCuboid(
                prim_path="/World/CarryBox",
                name="carry_box",
                position=payload_position,
                scale=np.array([0.28, 0.22, 0.20], dtype=float),
                color=np.array([0.58, 0.43, 0.24], dtype=float),
                mass=float(args_cli.payload_mass),
            )
        )
        _make_fixed_joint("/World/CarrierBody", "/World/CarryBox")
        print("[PROGRESS] Resetting world", flush=True)
        world.reset()
        print("[PROGRESS] World reset complete", flush=True)
    except BaseException as exc:
        summary_path.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}", "stage": "scene_setup"}, indent=2) + "\n")
        print(f"[ERROR] Scene setup failed: {type(exc).__name__}: {exc}", flush=True)
        return summary_path

    initial_carrier = _xyz(carrier)
    initial_payload = _xyz(payload)
    summary = {
        "scene_type": "pure_simapp_core_world_dynamic_fixed_payload_carry",
        "success_claim": "dynamic_fixed_payload_diagnostic_not_legged_walking_or_unknown_box_carrying",
        "device": args_cli.device,
        "payload_mass_kg": float(args_cli.payload_mass),
        "joint_mode": args_cli.joint_mode,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "carrier_travel_x_m": 0.0,
        "payload_travel_x_m": 0.0,
        "payload_relative_error_m": 0.0,
        "fall_events": 0,
        "payload_drop_events": 0,
        "error": None,
    }

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "carrier_x", "carrier_y", "carrier_z", "payload_x", "payload_y", "payload_z", "carrier_travel_x", "payload_travel_x", "relative_error", "fall", "payload_drop"])
            for step in range(int(args_cli.steps)):
                carrier.set_linear_velocity(np.array([float(args_cli.target_speed), 0.0, 0.0], dtype=float))
                world.step(render=False)
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    cpos = _xyz(carrier)
                    ppos = _xyz(payload)
                    carrier_travel = float(cpos[0] - initial_carrier[0])
                    payload_travel = float(ppos[0] - initial_payload[0])
                    if args_cli.joint_mode == "front_offset":
                        expected_payload = [cpos[0] + 0.34, cpos[1], cpos[2] + 0.08]
                    else:
                        expected_payload = cpos
                    relative_error = float(math.dist(expected_payload, ppos))
                    fall = int(cpos[2] < 0.22)
                    payload_drop = int(ppos[2] < 0.22)
                    summary["completed_steps"] = step + 1
                    summary["carrier_travel_x_m"] = carrier_travel
                    summary["payload_travel_x_m"] = payload_travel
                    summary["payload_relative_error_m"] = relative_error
                    summary["fall_events"] += fall
                    summary["payload_drop_events"] += payload_drop
                    writer.writerow([step, cpos[0], cpos[1], cpos[2], ppos[0], ppos[1], ppos[2], carrier_travel, payload_travel, relative_error, fall, payload_drop])
                    print(f"[STATE] step={step} carrier_x={cpos[0]:.3f} payload_x={ppos[0]:.3f} rel={relative_error:.3f} fall={fall} drop={payload_drop}", flush=True)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
