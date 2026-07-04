#!/usr/bin/env python3
"""Run an official IsaacLab ANYmal velocity policy with a payload-mass box diagnostic.

This is an intermediate carry milestone. The policy is an official RSL-RL
locomotion policy; the payload is represented physically as added base mass and
visually as a box following the base. It is not a grasp/contact success claim.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac simulation or policy loading on a login/management node.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANYmal payload-carry locomotion diagnostic.")
    parser.add_argument("--task", default="Isaac-Velocity-Flat-Anymal-C-Play-v0")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--use-pretrained-checkpoint", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--payload-mass", type=float, default=5.0)
    parser.add_argument("--payload-com", type=float, nargs=3, default=(0.18, 0.0, 0.06), metavar=("X", "Y", "Z"))
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.55, 0.35, 0.25), metavar=("X", "Y", "Z"))
    parser.add_argument("--command", type=float, nargs=3, default=(0.35, 0.0, 0.0), metavar=("VX", "VY", "YAW"))
    parser.add_argument(
        "--local-asset-root",
        type=Path,
        default=Path("/public/home/yanhongru/isaac_asset_mirror/Assets/Isaac/6.0"),
    )
    parser.add_argument(
        "--use-remote-assets",
        action="store_true",
        help="Do not patch Isaac asset paths to the local mirror. Diagnostic only; may hang if remote asset access is slow.",
    )
    parser.add_argument("--fall-height", type=float, default=0.35)
    parser.add_argument("--max-tilt-xy", type=float, default=0.65)
    parser.add_argument("--disable-fabric", action="store_true")
    parser.add_argument("--physics-backend", choices=("physx", "newton"), default="physx")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/anymal_payload_carry"))
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


_refuse_login_node()
args_cli = parse_args()
print(
    "[BOOT] run_anymal_payload_carry parsed args; "
    f"task={args_cli.task} backend={args_cli.physics_backend} device={args_cli.device}",
    flush=True,
)
print("[BOOT] creating Isaac AppLauncher", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[BOOT] Isaac AppLauncher created", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from packaging import version  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.utils import PresetCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

print("[BOOT] post-AppLauncher imports completed", flush=True)


def _as_torch(value) -> torch.Tensor:
    """Convert IsaacLab/Warp/Torch buffers to a torch tensor for logging."""

    if isinstance(value, torch.Tensor):
        return value
    try:
        return wp.to_torch(value)
    except Exception:
        return torch.as_tensor(value, device=env_device())


def env_device() -> torch.device:
    return torch.device(args_cli.device if str(args_cli.device).startswith(("cuda", "cpu")) else "cpu")


def _select_presets(cfg, preset_name: str, visited: set[int] | None = None):
    if visited is None:
        visited = set()
    cfg_id = id(cfg)
    if cfg_id in visited:
        return cfg
    visited.add(cfg_id)
    if isinstance(cfg, PresetCfg):
        selected = getattr(cfg, preset_name, None)
        if selected is None:
            selected = getattr(cfg, "default", None)
        return _select_presets(selected, preset_name, visited) if selected is not None else cfg
    if isinstance(cfg, dict):
        for key, value in list(cfg.items()):
            cfg[key] = _select_presets(value, preset_name, visited)
        return cfg
    if hasattr(cfg, "__dataclass_fields__"):
        for field_name in cfg.__dataclass_fields__:
            value = getattr(cfg, field_name)
            selected = _select_presets(value, preset_name, visited)
            if selected is not value:
                setattr(cfg, field_name, selected)
        return cfg
    return cfg


def _resolve_checkpoint() -> str:
    if args_cli.checkpoint is not None:
        checkpoint = args_cli.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return str(checkpoint)
    if args_cli.use_pretrained_checkpoint:
        train_task_name = args_cli.task.replace("-Play", "")
        checkpoint = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if checkpoint is None:
            raise FileNotFoundError(f"No published RSL-RL checkpoint available for {train_task_name}")
        return checkpoint
    default_checkpoint = Path(".pretrained_checkpoints/rsl_rl") / args_cli.task.replace("-Play", "") / "checkpoint.pt"
    if default_checkpoint.is_file():
        return str(default_checkpoint.resolve())
    raise FileNotFoundError(
        "No checkpoint provided. Pass --checkpoint or --use-pretrained-checkpoint "
        f"(default cache checked: {default_checkpoint})."
    )


def _configure_env():
    local_asset_root = args_cli.local_asset_root.expanduser().resolve()
    ground_usd = local_asset_root / "Isaac/Environments/Grid/default_environment.usd"
    if ground_usd.is_file() and not args_cli.use_remote_assets:
        sim_utils.GroundPlaneCfg.usd_path = str(ground_usd)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg = _select_presets(env_cfg, "newton" if args_cli.physics_backend == "newton" else "default")
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.scene.num_envs = args_cli.num_envs
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (float(args_cli.command[0]), float(args_cli.command[0]))
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (float(args_cli.command[1]), float(args_cli.command[1]))
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (float(args_cli.command[2]), float(args_cli.command[2]))
        env_cfg.commands.base_velocity.ranges.heading = None
    if ground_usd.is_file() and not args_cli.use_remote_assets and hasattr(env_cfg.scene, "terrain"):
        env_cfg.scene.terrain.terrain_type = "usd"
        env_cfg.scene.terrain.usd_path = str(ground_usd)
        env_cfg.scene.terrain.terrain_generator = None
        if getattr(env_cfg, "curriculum", None) is not None and getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
            env_cfg.curriculum.terrain_levels = None

    anymal_usd = local_asset_root / "Isaac/IsaacLab/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
    actuator_net = local_asset_root / "Isaac/IsaacLab/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt"
    robot_cfg = env_cfg.scene.robot if hasattr(env_cfg.scene, "robot") else getattr(env_cfg, "robot", None)
    if anymal_usd.is_file() and not args_cli.use_remote_assets and robot_cfg is not None:
        robot_cfg.spawn.usd_path = str(anymal_usd)
    if actuator_net.is_file() and not args_cli.use_remote_assets and robot_cfg is not None and "legs" in robot_cfg.actuators:
        robot_cfg.actuators["legs"].network_file = str(actuator_net)

    if anymal_usd.is_file() and not args_cli.use_remote_assets and hasattr(env_cfg.scene, "robot"):
        env_cfg.scene.robot.spawn.usd_path = str(anymal_usd)
    if (
        actuator_net.is_file()
        and not args_cli.use_remote_assets
        and hasattr(env_cfg.scene, "robot")
        and "legs" in env_cfg.scene.robot.actuators
    ):
        env_cfg.scene.robot.actuators["legs"].network_file = str(actuator_net)

    if getattr(env_cfg.events, "add_base_mass", None) is not None:
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (
            float(args_cli.payload_mass),
            float(args_cli.payload_mass),
        )
        env_cfg.events.add_base_mass.params["operation"] = "add"
    if getattr(env_cfg.events, "base_com", None) is not None:
        env_cfg.events.base_com.params["com_range"] = {
            "x": (float(args_cli.payload_com[0]), float(args_cli.payload_com[0])),
            "y": (float(args_cli.payload_com[1]), float(args_cli.payload_com[1])),
            "z": (float(args_cli.payload_com[2]), float(args_cli.payload_com[2])),
        }
    env_cfg.log_dir = str(args_cli.output_dir.resolve())
    return env_cfg


def _make_visual_box(stage, prim_path: str) -> None:
    box_cfg = sim_utils.CuboidCfg(
        size=tuple(float(v) for v in args_cli.box_size),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.25), roughness=0.8),
    )
    box_cfg.func(prim_path, box_cfg, translation=(0.0, 0.0, 1.0))
    prim = stage.GetPrimAtPath(prim_path)
    UsdGeom.Imageable(prim).MakeVisible()


def _set_translate(stage, prim_path: str, xyz: tuple[float, float, float]) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(*xyz))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))


def _force_command(env) -> None:
    if hasattr(env.unwrapped, "command_manager"):
        term = env.unwrapped.command_manager.get_term("base_velocity")
        term.vel_command_b[:, 0] = float(args_cli.command[0])
        term.vel_command_b[:, 1] = float(args_cli.command[1])
        term.vel_command_b[:, 2] = float(args_cli.command[2])
        if hasattr(term, "is_standing_env"):
            term.is_standing_env[:] = False
        if hasattr(term, "is_heading_env"):
            term.is_heading_env[:] = False
    elif hasattr(env.unwrapped, "_commands"):
        env.unwrapped._commands[:, 0] = float(args_cli.command[0])
        env.unwrapped._commands[:, 1] = float(args_cli.command[1])
        env.unwrapped._commands[:, 2] = float(args_cli.command[2])


def main() -> None:
    print("[BOOT] main entered", flush=True)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    print("[BOOT] resolving checkpoint", flush=True)
    checkpoint = _resolve_checkpoint()
    print(f"[BOOT] checkpoint resolved: {checkpoint}", flush=True)
    print("[BOOT] configuring environment", flush=True)
    env_cfg = _configure_env()
    print("[BOOT] loading agent config", flush=True)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.device = args_cli.device

    print("[BOOT] creating gym environment", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    print("[BOOT] wrapping RSL-RL environment", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    _force_command(env)

    print("[BOOT] creating RSL-RL runner", flush=True)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    print("[BOOT] loading policy checkpoint", flush=True)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    stage = env.unwrapped.sim.stage
    box_path = "/World/CarryBoxVisual"
    print("[BOOT] creating visual payload box", flush=True)
    _make_visual_box(stage, box_path)

    robot = env.unwrapped.scene["robot"]
    metrics_path = args_cli.output_dir / "anymal_payload_carry_state.csv"
    summary_path = args_cli.output_dir / "anymal_payload_carry_summary.json"
    dt = float(env.unwrapped.step_dt)
    initial_xy = None
    summary = {
        "task": args_cli.task,
        "checkpoint": checkpoint,
        "claim_level": "payload_mass_locomotion_diagnostic_not_grasp_or_contact_carry_success",
        "steps_requested": int(args_cli.steps),
        "completed_steps": 0,
        "physics_dt": dt,
        "num_envs": int(args_cli.num_envs),
        "payload_mass_kg": float(args_cli.payload_mass),
        "payload_com_m": [float(v) for v in args_cli.payload_com],
        "box_size_m": [float(v) for v in args_cli.box_size],
        "command": [float(v) for v in args_cli.command],
        "max_travel_xy_m": 0.0,
        "min_base_z_m": None,
        "max_tilt_xy": 0.0,
        "fall_events": 0,
        "done_events": 0,
    }

    print("[BOOT] entering rollout", flush=True)
    obs = env.get_observations()
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "time_s",
                "base_x",
                "base_y",
                "base_z",
                "base_qx",
                "base_qy",
                "base_qz",
                "base_qw",
                "lin_vel_b_x",
                "lin_vel_b_y",
                "lin_vel_b_z",
                "projected_gravity_x",
                "projected_gravity_y",
                "projected_gravity_z",
                "tilt_xy",
                "travel_xy_m",
                "fall_flag",
                "done_flag",
                "box_x",
                "box_y",
                "box_z",
            ]
        )
        for step in range(args_cli.steps):
            _force_command(env)
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            _force_command(env)

            root_pose = _as_torch(robot.data.root_link_pose_w)[0].detach().cpu().tolist()
            lin_vel_b = _as_torch(robot.data.root_lin_vel_b)[0].detach().cpu().tolist()
            projected_gravity = _as_torch(robot.data.projected_gravity_b)[0].detach().cpu().tolist()
            if initial_xy is None:
                initial_xy = (float(root_pose[0]), float(root_pose[1]))
            travel_xy = math.hypot(float(root_pose[0]) - initial_xy[0], float(root_pose[1]) - initial_xy[1])
            tilt_xy = math.hypot(float(projected_gravity[0]), float(projected_gravity[1]))
            fall_flag = int(float(root_pose[2]) < args_cli.fall_height or tilt_xy > args_cli.max_tilt_xy)
            done_flag = int(bool(dones[0].item()))
            box_xyz = (
                float(root_pose[0]) + float(args_cli.payload_com[0]),
                float(root_pose[1]) + float(args_cli.payload_com[1]),
                float(root_pose[2]) + float(args_cli.payload_com[2]),
            )
            _set_translate(stage, box_path, box_xyz)

            if step % 10 == 0 or step == args_cli.steps - 1:
                summary["completed_steps"] = int(step + 1)
                summary["max_travel_xy_m"] = max(float(summary["max_travel_xy_m"]), float(travel_xy))
                summary["min_base_z_m"] = (
                    float(root_pose[2])
                    if summary["min_base_z_m"] is None
                    else min(float(summary["min_base_z_m"]), float(root_pose[2]))
                )
                summary["max_tilt_xy"] = max(float(summary["max_tilt_xy"]), float(tilt_xy))
                summary["fall_events"] += fall_flag
                summary["done_events"] += done_flag
                writer.writerow(
                    [
                        step,
                        step * dt,
                        *root_pose,
                        *lin_vel_b,
                        *projected_gravity,
                        tilt_xy,
                        travel_xy,
                        fall_flag,
                        done_flag,
                        *box_xyz,
                    ]
                )
                print(
                    "[STATE] "
                    f"step={step} base=({root_pose[0]:.3f},{root_pose[1]:.3f},{root_pose[2]:.3f}) "
                    f"travel={travel_xy:.3f} tilt_xy={tilt_xy:.3f} fall={fall_flag} done={done_flag}"
                )

            if version.parse(installed_version) >= version.parse("4.0.0") and hasattr(policy, "reset"):
                policy.reset(dones)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Summary written to: {summary_path}")
    print(f"[INFO] Metrics written to: {metrics_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
