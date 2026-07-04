#!/usr/bin/env python3
"""RigidObject-backed PhysX force isolation smoke."""

from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RigidObject cube force smoke.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--force-x", type=float, default=180.0)
    parser.add_argument("--force-z", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--write-scene-data",
        action="store_true",
        help="Diagnostic toggle. By default the passive cube is not written back to sim each step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/physx_force_rigidobject_cube_smoke"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import omni.physx  # noqa: E402
import warp as wp  # noqa: E402
from pxr import PhysicsSchemaTools, UsdGeom, UsdUtils  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import build_simulation_context  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


CUBE_PATH = "/World/envs/env_0/TestCube"


def _pose(stage, prim_path: str) -> list[float]:
    prim = stage.GetPrimAtPath(prim_path)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    translation = matrix.ExtractTranslation()
    return [float(translation[0]), float(translation[1]), float(translation[2])]


def _root_pose(obj) -> list[float]:
    pose = wp.to_torch(obj.data.root_link_pose_w).detach().cpu()[0].tolist()
    return [float(pose[0]), float(pose[1]), float(pose[2])]


def design_scene() -> InteractiveScene:
    ground_cfg = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 2.0, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.33, 0.33), roughness=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
    )
    cube_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestCube",
        spawn=sim_utils.CuboidCfg(
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
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.75), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    @configclass
    class ForceCubeSceneCfg(InteractiveSceneCfg):
        ground = ground_cfg
        cube = cube_cfg

    return InteractiveScene(ForceCubeSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False))


def run_scene() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "physx_force_rigidobject_cube_state.csv"
    summary_path = args_cli.output_dir / "physx_force_rigidobject_cube_summary.json"
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)

    with build_simulation_context(create_new_stage=True, sim_cfg=sim_cfg, add_ground_plane=False) as sim:
        sim.set_setting("/physics/updateToUsd", True)
        sim.set_setting("/physics/updateVelocitiesToUsd", True)
        scene = design_scene()
        sim.reset()
        cube = scene["cube"]
        cube.reset()
        physx = omni.physx.get_physx_simulation_interface()
        if hasattr(physx, "flush_changes"):
            physx.flush_changes()
        stage_id = UsdUtils.StageCache.Get().GetId(sim.stage).ToLongInt()
        body_id = PhysicsSchemaTools.sdfPathToInt(sim.stage.GetPrimAtPath(CUBE_PATH).GetPath())
        dt = float(sim.get_physics_dt())
        initial = [0.0, 0.0, 0.75]
        initial_usd = _pose(sim.stage, CUBE_PATH)
        last_pose = list(initial)
        summary = {
            "scene_type": "rigidobject_dynamic_cube_physx_force_isolation",
            "success_claim": "diagnostic_only_not_robot_carrying",
            "force_x_n": float(args_cli.force_x),
            "force_z_n": float(args_cli.force_z),
            "pose_source": "isaaclab_root_link_pose_w",
            "write_scene_data_each_step": bool(args_cli.write_scene_data),
            "create_new_stage": True,
            "steps_requested": int(args_cli.steps),
            "completed_steps": 0,
            "error": None,
            "initial_pose": initial,
            "initial_usd_pose": initial_usd,
            "final_pose": None,
            "final_usd_pose": None,
            "max_x_travel_m": 0.0,
            "min_z_m": initial[2],
        }
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "time_s",
                    "x",
                    "y",
                    "z",
                    "usd_x",
                    "usd_y",
                    "usd_z",
                    "x_travel",
                    "z_drop",
                ]
            )
            for step in range(int(args_cli.steps)):
                physx.apply_force_at_pos(
                    stage_id,
                    body_id,
                    carb.Float3(float(args_cli.force_x), 0.0, float(args_cli.force_z)),
                    carb.Float3(float(last_pose[0]), float(last_pose[1]), float(last_pose[2])),
                )
                if args_cli.write_scene_data:
                    scene.write_data_to_sim()
                sim.step(render=args_cli.render)
                scene.update(dt)
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    try:
                        p = _root_pose(cube)
                    except Exception as exc:
                        summary["completed_steps"] = int(step + 1)
                        summary["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                        print(f"[ERROR] Root-state read failed at step {step}: {exc}", flush=True)
                        break
                    last_pose = list(p)
                    usd_p = _pose(sim.stage, CUBE_PATH)
                    x_travel = p[0] - initial[0]
                    z_drop = initial[2] - p[2]
                    summary["completed_steps"] = int(step + 1)
                    summary["final_pose"] = p
                    summary["final_usd_pose"] = usd_p
                    summary["max_x_travel_m"] = max(float(summary["max_x_travel_m"]), float(x_travel))
                    summary["min_z_m"] = min(float(summary["min_z_m"]), float(p[2]))
                    writer.writerow([step, step * dt, p[0], p[1], p[2], usd_p[0], usd_p[1], usd_p[2], x_travel, z_drop])
                    print(
                        "[STATE] "
                        f"step={step} cube=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                        f"usd=({usd_p[0]:.3f},{usd_p[1]:.3f},{usd_p[2]:.3f}) "
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
