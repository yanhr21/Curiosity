#!/usr/bin/env python3
"""Arena G1 AGILE walk smoke using Arena's persistent SimulationApp harness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arena G1 AGILE persistent walk smoke.")
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--command-start-step", type=int, default=80)
    parser.add_argument("--command", type=float, nargs=3, default=(0.25, 0.0, 0.0), metavar=("VX", "VY", "YAW"))
    parser.add_argument("--base-height-command", type=float, default=0.75)
    parser.add_argument("--min-root-height", type=float, default=0.40)
    parser.add_argument("--max-tilt", type=float, default=0.85)
    parser.add_argument("--min-commanded-travel-x", type=float, default=0.05)
    parser.add_argument(
        "--ground-usd",
        type=Path,
        default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0/Isaac/Environments/Grid/default_environment.usd"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/public/home/yanhongru/Curiosity/experiments/outputs/arena_g1_agile_walk_persistent_smoke"),
    )
    return parser.parse_args()


def _quat_xyzw_to_tilt(qx: float, qy: float, qz: float, qw: float) -> float:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return float(max(abs(roll), abs(pitch)))


def _empty_summary(args: argparse.Namespace) -> dict:
    return {
        "scene_type": "arena_g1_agile_walk_persistent_smoke",
        "success_claim": "controller_backed_g1_agile_persistent_walk_smoke_not_box_carrying",
        "launcher_path": "isaaclab_arena.tests.utils.subprocess.run_simulation_app_function",
        "steps_requested": int(args.steps),
        "completed_steps": 0,
        "warmup_steps": int(args.warmup_steps),
        "command_start_step": int(args.command_start_step),
        "command_xyz_yaw": [float(v) for v in args.command],
        "base_height_command": float(args.base_height_command),
        "ground_usd": str(args.ground_usd),
        "device": str(args.device),
        "min_root_height_threshold_m": float(args.min_root_height),
        "max_tilt_threshold_rad": float(args.max_tilt),
        "min_commanded_travel_x_m": float(args.min_commanded_travel_x),
        "min_root_z_m": None,
        "max_root_z_m": None,
        "initial_root_x_m": None,
        "initial_root_y_m": None,
        "final_root_x_m": None,
        "final_root_y_m": None,
        "max_forward_travel_x_m": 0.0,
        "final_forward_travel_x_m": 0.0,
        "max_tilt_rad": 0.0,
        "fall_events": 0,
        "commanded_walk_steps": 0,
        "root_pose_write_count_rollout": 0,
        "root_velocity_write_count_rollout": 0,
        "status": "not_run",
        "failures": [],
        "error": None,
    }


def main() -> int:
    _refuse_login_node()
    args = parse_args()
    if not args.ground_usd.is_file():
        raise FileNotFoundError(f"Local ground USD not found: {args.ground_usd}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "arena_g1_agile_walk_persistent_state.csv"
    summary_path = args.output_dir / "arena_g1_agile_walk_persistent_summary.json"
    summary = _empty_summary(args)

    def _run(_simulation_app) -> bool:
        nonlocal summary
        import torch
        import warp as wp
        from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
        from isaaclab_arena.assets.registries import AssetRegistry
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.embodiments.g1.g1 import G1WBCAgileJointEmbodiment
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        env = None
        try:
            ground_plane = AssetRegistry().get_asset_by_name("ground_plane")(
                spawner_cfg=GroundPlaneCfg(usd_path=str(args.ground_usd))
            )
            scene = Scene(assets=[ground_plane])
            embodiment = G1WBCAgileJointEmbodiment(enable_cameras=False)
            arena_env = IsaacLabArenaEnvironment(
                name="curiosity_g1_agile_persistent_walk_smoke",
                embodiment=embodiment,
                scene=scene,
            )
            builder_args = get_isaaclab_arena_cli_parser().parse_args([])
            builder_args.num_envs = 1
            builder_args.headless = True
            builder_args.device = str(args.device)
            env = ArenaEnvBuilder(arena_env, builder_args).make_registered()
            env.reset()

            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "phase",
                    "cmd_x",
                    "cmd_y",
                    "cmd_yaw",
                    "root_x",
                    "root_y",
                    "root_z",
                    "forward_travel_x",
                    "tilt",
                    "fall",
                ])
                for step in range(int(args.steps)):
                    with torch.inference_mode():
                        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                        phase = "stand"
                        if step >= int(args.command_start_step):
                            phase = "commanded_walk"
                            actions[:, -7:-4] = torch.tensor(args.command, device=env.unwrapped.device)
                            summary["commanded_walk_steps"] = int(summary["commanded_walk_steps"]) + 1
                        actions[:, -4] = float(args.base_height_command)
                        env.step(actions)
                        root_pose = wp.to_torch(env.unwrapped.scene["robot"].data.root_link_pose_w)[0].detach().cpu().tolist()
                    root_z = float(root_pose[2])
                    if summary["initial_root_x_m"] is None:
                        summary["initial_root_x_m"] = float(root_pose[0])
                        summary["initial_root_y_m"] = float(root_pose[1])
                    forward_travel_x = float(root_pose[0]) - float(summary["initial_root_x_m"])
                    tilt = _quat_xyzw_to_tilt(float(root_pose[3]), float(root_pose[4]), float(root_pose[5]), float(root_pose[6]))
                    fall = int(step >= int(args.warmup_steps) and (root_z < float(args.min_root_height) or tilt > float(args.max_tilt)))
                    summary["completed_steps"] = step + 1
                    summary["final_root_x_m"] = float(root_pose[0])
                    summary["final_root_y_m"] = float(root_pose[1])
                    summary["min_root_z_m"] = root_z if summary["min_root_z_m"] is None else min(float(summary["min_root_z_m"]), root_z)
                    summary["max_root_z_m"] = root_z if summary["max_root_z_m"] is None else max(float(summary["max_root_z_m"]), root_z)
                    summary["max_forward_travel_x_m"] = max(float(summary["max_forward_travel_x_m"]), forward_travel_x)
                    summary["final_forward_travel_x_m"] = forward_travel_x
                    summary["max_tilt_rad"] = max(float(summary["max_tilt_rad"]), tilt)
                    summary["fall_events"] += fall
                    if step % 10 == 0 or step == int(args.steps) - 1:
                        writer.writerow([
                            step,
                            phase,
                            float(args.command[0]) if phase == "commanded_walk" else 0.0,
                            float(args.command[1]) if phase == "commanded_walk" else 0.0,
                            float(args.command[2]) if phase == "commanded_walk" else 0.0,
                            root_pose[0],
                            root_pose[1],
                            root_z,
                            forward_travel_x,
                            tilt,
                            fall,
                        ])
                        print(
                            "[STATE] "
                            f"step={step} phase={phase} "
                            f"root=({root_pose[0]:.3f},{root_pose[1]:.3f},{root_z:.3f}) "
                            f"travel_x={forward_travel_x:.3f} tilt={tilt:.4f} fall={fall}",
                            flush=True,
                        )
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {summary['error']}", flush=True)
            return False
        finally:
            if env is not None:
                env.close()
        return True

    from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

    run_simulation_app_function(_run, headless=True, enable_cameras=False)

    failures = []
    if summary["error"] is not None:
        failures.append(str(summary["error"]))
    if int(summary["completed_steps"]) < int(args.steps):
        failures.append(f"completed_steps {summary['completed_steps']} < requested {args.steps}")
    if int(summary["fall_events"]) > 0:
        failures.append(f"fall_events {summary['fall_events']} > 0")
    if float(summary["final_forward_travel_x_m"]) < float(args.min_commanded_travel_x):
        failures.append(
            f"final_forward_travel_x_m {summary['final_forward_travel_x_m']} < "
            f"min_commanded_travel_x {args.min_commanded_travel_x}"
        )
    summary["failures"] = failures
    summary["status"] = "pass" if not failures else "fail"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}", flush=True)
    print(f"[INFO] Metrics written to: {csv_path}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
