#!/usr/bin/env python3
"""Pure SimulationApp core-World dynamic cube smoke.

This tests Isaac Sim core `World` dynamics without IsaacLab AppLauncher,
IsaacLab tensor objects, or experimental Articulation. It is a control-path
diagnostic only, not robot carrying.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure SimulationApp core World dynamic cube smoke.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--mode", choices=("gravity", "velocity"), default="velocity")
    parser.add_argument("--velocity-x", type=float, default=0.35)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/core_world_simapp_dynamic_cube_smoke"))
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
    {
        "headless": True,
        "hide_ui": True,
        "active_gpu": 0 if str(args_cli.device).startswith("cuda") else None,
        "physics_gpu": 0 if str(args_cli.device).startswith("cuda") else None,
    },
    experience=os.environ.get("ISAAC_SIMAPP_EXPERIENCE", DEFAULT_EXPERIENCE),
)
print("[PROGRESS] Pure SimulationApp started", flush=True)

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage  # noqa: E402


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


def run() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_simapp_dynamic_cube_state.csv"
    summary_path = args_cli.output_dir / "core_world_simapp_dynamic_cube_summary.json"

    _patch_core_api_simulation_manager_compat()
    SimulationManager._backend = "numpy"
    print("[PROGRESS] Creating fresh core World stage", flush=True)
    create_new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    try:
        print("[PROGRESS] Adding fixed ground", flush=True)
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Ground",
                name="ground",
                position=np.array([0.0, 0.0, -0.025], dtype=float),
                scale=np.array([4.0, 2.0, 0.05], dtype=float),
                color=np.array([0.31, 0.33, 0.33], dtype=float),
            )
        )
        print("[PROGRESS] Fixed ground added", flush=True)
        print("[PROGRESS] Adding dynamic cube", flush=True)
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/TestCube",
                name="test_cube",
                position=np.array([0.0, 0.0, 0.45], dtype=float),
                scale=np.array([0.2, 0.2, 0.2], dtype=float),
                color=np.array([0.62, 0.34, 0.18], dtype=float),
                mass=2.0,
            )
        )
        print("[PROGRESS] Dynamic cube added", flush=True)
        print("[PROGRESS] Resetting world", flush=True)
        world.reset()
        print("[PROGRESS] World reset complete", flush=True)
    except BaseException as exc:
        summary_path.write_text(
            json.dumps(
                {
                    "scene_type": "pure_simapp_core_world_dynamic_cube_smoke",
                    "success_claim": "failed_before_rollout",
                    "mode": args_cli.mode,
                    "device": args_cli.device,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"[ERROR] Scene construction/reset failed: {type(exc).__name__}: {exc}", flush=True)
        return summary_path

    initial = _xyz(cube)
    summary = {
        "scene_type": "pure_simapp_core_world_dynamic_cube_smoke",
        "success_claim": "dynamic_rigidbody_control_path_diagnostic_only",
        "mode": args_cli.mode,
        "device": args_cli.device,
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "initial_pos": initial,
        "final_pos": initial,
        "travel_x_m": 0.0,
        "drop_z_m": 0.0,
        "error": None,
    }

    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y", "z", "travel_x_m", "drop_z_m"])
            for step in range(int(args_cli.steps)):
                if args_cli.mode == "velocity":
                    cube.set_linear_velocity(np.array([float(args_cli.velocity_x), 0.0, 0.0], dtype=float))
                world.step(render=False)
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    pos = _xyz(cube)
                    travel_x = float(pos[0] - initial[0])
                    drop_z = float(initial[2] - pos[2])
                    summary["completed_steps"] = step + 1
                    summary["final_pos"] = pos
                    summary["travel_x_m"] = travel_x
                    summary["drop_z_m"] = drop_z
                    writer.writerow([step, pos[0], pos[1], pos[2], travel_x, drop_z])
                    print(f"[STATE] step={step} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) travel_x={travel_x:.3f} drop_z={drop_z:.3f}", flush=True)
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
