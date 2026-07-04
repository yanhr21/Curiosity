#!/usr/bin/env python3
"""Minimal PhysX force/fall isolation smoke.

This script is intentionally narrower than the carrying task.  It verifies
whether a single dynamic cuboid created in the current Isaac environment moves
under gravity and under `apply_force_at_pos` without using IsaacLab tensor
views.  A failure here means higher-level dynamic carry scenes should not keep
tuning controller gains.
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
    parser = argparse.ArgumentParser(description="Single-cube PhysX force smoke.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--force-x", type=float, default=180.0)
    parser.add_argument("--force-z", type=float, default=0.0)
    parser.add_argument("--step-mode", choices=("sim_step", "physx_direct"), default="sim_step")
    parser.add_argument("--reuse-stage", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/physx_force_cube_smoke"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import build_simulation_context  # noqa: E402


CUBE_PATH = "/World/TestCube"


def _pose(stage: Usd.Stage, prim_path: str) -> list[float]:
    prim = stage.GetPrimAtPath(prim_path)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    return [float(translation[0]), float(translation[1]), float(translation[2])]


def design_scene() -> None:
    floor_cfg = sim_utils.CuboidCfg(
        size=(4.0, 2.0, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.33, 0.33), roughness=0.9),
    )
    floor_cfg.func("/World/Ground", floor_cfg, translation=(0.0, 0.0, -0.025))
    cube_cfg = sim_utils.CuboidCfg(
        size=(0.20, 0.20, 0.20),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=8.0,
            max_angular_velocity=8.0,
            max_depenetration_velocity=2.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.34, 0.18), roughness=0.8),
    )
    cube_cfg.func(CUBE_PATH, cube_cfg, translation=(0.0, 0.0, 0.75))


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "physx_force_cube_state.csv"
    summary_path = args_cli.output_dir / "physx_force_cube_summary.json"
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)

    with build_simulation_context(create_new_stage=not args_cli.reuse_stage, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        sim.set_setting("/physics/updateToUsd", True)
        sim.set_setting("/physics/updateVelocitiesToUsd", True)
        design_scene()
        sim.reset()
        omni.timeline.get_timeline_interface().play()
        physx = omni.physx.get_physx_simulation_interface()
        if hasattr(physx, "flush_changes"):
            physx.flush_changes()
        stage_id = UsdUtils.StageCache.Get().GetId(sim.stage).ToLongInt()
        body_id = PhysicsSchemaTools.sdfPathToInt(sim.stage.GetPrimAtPath(CUBE_PATH).GetPath())
        dt = float(sim.get_physics_dt())
        initial = _pose(sim.stage, CUBE_PATH)
        summary = {
            "scene_type": "single_dynamic_cube_physx_force_isolation",
            "success_claim": "diagnostic_only_not_robot_carrying",
            "step_mode": str(args_cli.step_mode),
            "create_new_stage": not bool(args_cli.reuse_stage),
            "force_x_n": float(args_cli.force_x),
            "force_z_n": float(args_cli.force_z),
            "steps_requested": int(args_cli.steps),
            "completed_steps": 0,
            "initial_pose": initial,
            "final_pose": None,
            "max_x_travel_m": 0.0,
            "min_z_m": initial[2],
        }
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "time_s", "x", "y", "z", "x_travel", "z_drop"])
            for step in range(int(args_cli.steps)):
                p = _pose(sim.stage, CUBE_PATH)
                physx.apply_force_at_pos(
                    stage_id,
                    body_id,
                    carb.Float3(float(args_cli.force_x), 0.0, float(args_cli.force_z)),
                    carb.Float3(float(p[0]), float(p[1]), float(p[2])),
                )
                if args_cli.step_mode == "physx_direct":
                    physx.simulate(dt, 0)
                    physx.fetch_results()
                    simulation_app.update()
                else:
                    sim.step(render=args_cli.render)
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    p = _pose(sim.stage, CUBE_PATH)
                    x_travel = p[0] - initial[0]
                    z_drop = initial[2] - p[2]
                    summary["completed_steps"] = int(step + 1)
                    summary["final_pose"] = p
                    summary["max_x_travel_m"] = max(float(summary["max_x_travel_m"]), float(x_travel))
                    summary["min_z_m"] = min(float(summary["min_z_m"]), float(p[2]))
                    writer.writerow([step, step * dt, p[0], p[1], p[2], x_travel, z_drop])
                    print(
                        "[STATE] "
                        f"step={step} cube=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
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
