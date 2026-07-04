#!/usr/bin/env python3
"""Isaac Sim core-World dynamic cube smoke.

This diagnostic avoids IsaacLab SimulationContext, Articulation, and
RigidObject tensor APIs.  It checks whether the older Isaac Sim core World API
can produce an observable dynamic cuboid under gravity and a commanded
velocity/force in this cluster environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Core World DynamicCuboid smoke.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--mode", choices=("gravity", "velocity", "force"), default="velocity")
    parser.add_argument("--velocity-x", type=float, default=0.45)
    parser.add_argument("--force-x", type=float, default=40.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/core_world_dynamic_cube_smoke"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage  # noqa: E402


def _xyz(obj: DynamicCuboid) -> list[float]:
    pos, _quat = obj.get_world_pose()
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "core_world_dynamic_cube_state.csv"
    summary_path = args_cli.output_dir / "core_world_dynamic_cube_summary.json"

    create_new_stage()
    print("[PROGRESS] Creating core World", flush=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=1.0 / 60.0, backend="numpy", device=args_cli.device)
    print("[PROGRESS] Adding local fixed ground and dynamic cube", flush=True)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Ground",
            name="local_ground",
            position=np.array([0.0, 0.0, -0.025], dtype=float),
            scale=np.array([4.0, 2.0, 0.05], dtype=float),
            color=np.array([0.31, 0.33, 0.33], dtype=float),
        )
    )
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/TestCube",
            name="test_cube",
            position=np.array([0.0, 0.0, 0.75], dtype=float),
            scale=np.array([0.2, 0.2, 0.2], dtype=float),
            color=np.array([0.62, 0.34, 0.18], dtype=float),
            mass=2.0,
        )
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Target",
            name="target_marker",
            position=np.array([0.7, 0.0, 0.01], dtype=float),
            scale=np.array([0.35, 0.25, 0.02], dtype=float),
            color=np.array([0.05, 0.38, 0.85], dtype=float),
        )
    )
    print("[PROGRESS] Resetting core World", flush=True)
    world.reset()
    print("[PROGRESS] Core World reset complete", flush=True)

    initial = _xyz(cube)
    summary = {
        "scene_type": "isaac_core_world_dynamic_cuboid_smoke",
        "success_claim": "diagnostic_only_not_robot_carrying",
        "uses_isaaclab_simulation_context": False,
        "mode": str(args_cli.mode),
        "device": str(args_cli.device),
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "initial_pose": initial,
        "final_pose": None,
        "max_x_travel_m": 0.0,
        "min_z_m": float(initial[2]),
        "error": None,
    }

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_s", "x", "y", "z", "x_travel", "z_drop"])
        for step in range(int(args_cli.steps)):
            if args_cli.mode == "velocity":
                cube.set_linear_velocity(np.array([float(args_cli.velocity_x), 0.0, 0.0], dtype=float))
            elif args_cli.mode == "force":
                cube.apply_forces(np.array([[float(args_cli.force_x), 0.0, 0.0]], dtype=float))
            world.step(render=args_cli.render)
            if step % 10 == 0 or step == int(args_cli.steps) - 1:
                pos = _xyz(cube)
                x_travel = pos[0] - initial[0]
                z_drop = initial[2] - pos[2]
                summary["completed_steps"] = int(step + 1)
                summary["final_pose"] = pos
                summary["max_x_travel_m"] = max(float(summary["max_x_travel_m"]), float(x_travel))
                summary["min_z_m"] = min(float(summary["min_z_m"]), float(pos[2]))
                writer.writerow([step, step * 0.005, pos[0], pos[1], pos[2], x_travel, z_drop])
                print(
                    "[STATE] "
                    f"step={step} cube=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                    f"x_travel={x_travel:.3f} z_drop={z_drop:.3f}",
                    flush=True,
                )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    print(f"[INFO] Metrics written to: {csv_path}")
    return summary_path


if __name__ == "__main__":
    try:
        run_scene()
    finally:
        simulation_app.close()
