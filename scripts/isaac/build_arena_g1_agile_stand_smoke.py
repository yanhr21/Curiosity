#!/usr/bin/env python3
"""Arena G1 AGILE standing smoke with local assets.

This is a controller-backed G1 diagnostic that avoids the remote default
ground-plane dependency in the upstream test. It does not use GR00T or any
external policy server.
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
    parser = argparse.ArgumentParser(description="Arena G1 AGILE standing smoke.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--command-start-step", type=int, default=80)
    parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "YAW"))
    parser.add_argument("--base-height-command", type=float, default=0.75)
    parser.add_argument("--min-root-height", type=float, default=0.40)
    parser.add_argument("--max-tilt", type=float, default=0.85)
    parser.add_argument("--min-commanded-travel-x", type=float, default=0.0)
    parser.add_argument("--skip-env-reset", action="store_true")
    parser.add_argument(
        "--ground-usd",
        type=Path,
        default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/Environments/Grid/default_environment.usd"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/arena_g1_agile_stand_smoke"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _quat_xyzw_to_tilt(qx: float, qy: float, qz: float, qw: float) -> float:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return float(max(abs(roll), abs(pitch)))


_refuse_login_node()
args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PROGRESS] AppLauncher started", flush=True)

import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg  # noqa: E402
from isaaclab_arena.assets.registries import AssetRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser  # noqa: E402
from isaaclab_arena.embodiments.g1.g1 import G1WBCAgileJointEmbodiment  # noqa: E402
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment  # noqa: E402
from isaaclab_arena.scene.scene import Scene  # noqa: E402


def _build_env():
    if not args_cli.ground_usd.is_file():
        raise FileNotFoundError(f"Local ground USD not found: {args_cli.ground_usd}")
    ground_plane = AssetRegistry().get_asset_by_name("ground_plane")(
        spawner_cfg=GroundPlaneCfg(usd_path=str(args_cli.ground_usd))
    )
    scene = Scene(assets=[ground_plane])
    embodiment = G1WBCAgileJointEmbodiment(enable_cameras=False)
    arena_env = IsaacLabArenaEnvironment(
        name="curiosity_g1_agile_standing_smoke",
        embodiment=embodiment,
        scene=scene,
    )
    builder_args = get_isaaclab_arena_cli_parser().parse_args([])
    builder_args.num_envs = 1
    builder_args.headless = True
    builder_args.device = args_cli.device
    env_builder = ArenaEnvBuilder(arena_env, builder_args)
    env = env_builder.make_registered()
    if not bool(args_cli.skip_env_reset):
        env.reset()
    return env


def run_smoke() -> Path:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "arena_g1_agile_stand_state.csv"
    summary_path = args_cli.output_dir / "arena_g1_agile_stand_summary.json"
    summary = {
        "scene_type": "arena_g1_agile_stand_smoke",
        "success_claim": "controller_backed_g1_agile_stand_walk_smoke_not_box_carrying",
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "warmup_steps": int(args_cli.warmup_steps),
        "command_start_step": int(args_cli.command_start_step),
        "command_xyz_yaw": [float(v) for v in args_cli.command],
        "ground_usd": str(args_cli.ground_usd),
        "base_height_command": float(args_cli.base_height_command),
        "skip_env_reset": bool(args_cli.skip_env_reset),
        "min_root_height_threshold_m": float(args_cli.min_root_height),
        "max_tilt_threshold_rad": float(args_cli.max_tilt),
        "min_root_z_m": None,
        "max_root_z_m": None,
        "initial_root_x_m": None,
        "initial_root_y_m": None,
        "final_root_x_m": None,
        "final_root_y_m": None,
        "max_forward_travel_x_m": 0.0,
        "final_forward_travel_x_m": 0.0,
        "min_commanded_travel_x_m": float(args_cli.min_commanded_travel_x),
        "max_tilt_rad": 0.0,
        "fall_events": 0,
        "commanded_walk_steps": 0,
        "root_pose_write_count_rollout": 0,
        "root_velocity_write_count_rollout": 0,
        "status": "not_run",
        "failures": [],
        "error": None,
    }
    env = None
    try:
        env = _build_env()
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "phase", "cmd_x", "cmd_y", "cmd_yaw", "root_x", "root_y", "root_z", "forward_travel_x", "tilt", "fall"])
            for step in range(int(args_cli.steps)):
                with torch.inference_mode():
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    phase = "stand"
                    if step >= int(args_cli.command_start_step):
                        phase = "commanded_walk"
                        actions[:, -7:-4] = torch.tensor(args_cli.command, device=env.unwrapped.device)
                        summary["commanded_walk_steps"] = int(summary["commanded_walk_steps"]) + 1
                    actions[:, -4] = float(args_cli.base_height_command)
                    env.step(actions)
                    root_pose = wp.to_torch(env.unwrapped.scene["robot"].data.root_link_pose_w)[0].detach().cpu().tolist()
                root_z = float(root_pose[2])
                if summary["initial_root_x_m"] is None:
                    summary["initial_root_x_m"] = float(root_pose[0])
                    summary["initial_root_y_m"] = float(root_pose[1])
                forward_travel_x = float(root_pose[0]) - float(summary["initial_root_x_m"])
                tilt = _quat_xyzw_to_tilt(float(root_pose[3]), float(root_pose[4]), float(root_pose[5]), float(root_pose[6]))
                fall = int(step >= int(args_cli.warmup_steps) and (root_z < float(args_cli.min_root_height) or tilt > float(args_cli.max_tilt)))
                summary["completed_steps"] = step + 1
                summary["final_root_x_m"] = float(root_pose[0])
                summary["final_root_y_m"] = float(root_pose[1])
                summary["min_root_z_m"] = root_z if summary["min_root_z_m"] is None else min(float(summary["min_root_z_m"]), root_z)
                summary["max_root_z_m"] = root_z if summary["max_root_z_m"] is None else max(float(summary["max_root_z_m"]), root_z)
                summary["max_forward_travel_x_m"] = max(float(summary["max_forward_travel_x_m"]), forward_travel_x)
                summary["final_forward_travel_x_m"] = forward_travel_x
                summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), tilt)
                summary["fall_events"] += fall
                if step % 10 == 0 or step == int(args_cli.steps) - 1:
                    writer.writerow([
                        step,
                        phase,
                        float(args_cli.command[0]) if phase == "commanded_walk" else 0.0,
                        float(args_cli.command[1]) if phase == "commanded_walk" else 0.0,
                        float(args_cli.command[2]) if phase == "commanded_walk" else 0.0,
                        root_pose[0],
                        root_pose[1],
                        root_z,
                        forward_travel_x,
                        tilt,
                        fall,
                    ])
                    print(
                        "[STATE] "
                        f"step={step} phase={phase} cmd=({args_cli.command[0] if phase == 'commanded_walk' else 0.0:.3f},"
                        f"{args_cli.command[1] if phase == 'commanded_walk' else 0.0:.3f},"
                        f"{args_cli.command[2] if phase == 'commanded_walk' else 0.0:.3f}) "
                        f"root=({root_pose[0]:.3f},{root_pose[1]:.3f},{root_z:.3f}) "
                        f"travel_x={forward_travel_x:.3f} "
                        f"tilt={tilt:.4f} fall={fall}",
                        flush=True,
                    )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {summary['error']}", flush=True)
    finally:
        if env is not None:
            env.close()

    failures = []
    if summary["error"] is not None:
        failures.append(str(summary["error"]))
    if int(summary["completed_steps"]) < int(args_cli.steps):
        failures.append(f"completed_steps {summary['completed_steps']} < requested {args_cli.steps}")
    if int(summary["fall_events"]) > 0:
        failures.append(f"fall_events {summary['fall_events']} > 0")
    if float(summary["final_forward_travel_x_m"]) < float(args_cli.min_commanded_travel_x):
        failures.append(
            f"final_forward_travel_x_m {summary['final_forward_travel_x_m']} < "
            f"min_commanded_travel_x {args_cli.min_commanded_travel_x}"
        )
    summary["failures"] = failures
    summary["status"] = "pass" if not failures else "fail"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return summary_path


if __name__ == "__main__":
    try:
        run_smoke()
    finally:
        simulation_app.close()
