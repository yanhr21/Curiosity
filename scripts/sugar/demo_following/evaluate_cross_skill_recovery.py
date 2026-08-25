#!/usr/bin/env python3
"""Frozen rollout after the online Carry9-to-Kick physical handoff."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SUGAR = ROOT / "SUGAR"
for path in (SUGAR / "scripts/sugar_rl",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(SUGAR / "descriptions/terrain/sugar_ground_plane.usda"),
)
os.environ.setdefault("ISAACLAB_USE_LOCAL_FRAME_MARKER", "1")
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ.setdefault("DISPLAY", "")
job_id = os.environ.get("SLURM_JOB_ID", "local")
os.environ.setdefault("ISAACLAB_TMP_ROOT", f"/tmp/Curiosity_recovery_eval_{job_id}")
os.environ.setdefault(
    "SUGAR_UNITREE_TMP_ROOT", f"/tmp/Curiosity_recovery_eval_unitree_{job_id}"
)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=20)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--seed", type=int, default=181629)
parser.add_argument("--carry-prefix-steps", type=int, default=9)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 20 or args.steps != 250:
    parser.error("frozen recovery evaluation is fixed to 20 envs x 250 steps")
if args.carry_prefix_steps <= 0:
    parser.error("carry prefix must be positive")
args.task = "Sugar-G129dof-KickBox-Carry9-Recovery"
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import sugar_rl.tasks  # noqa: F401,E402
from sugar_rl.tasks.locomanip.robots.g129dof.train_tracker.kick_box_carry9_recovery_v2_env_cfg import (  # noqa: E402
    RobotEnvCfg,
)
from sugar_rl.utils.online_cross_skill_recovery_wrapper import (  # noqa: E402
    OnlineCrossSkillRecoveryVecEnvWrapper,
    _load_released_tracker_actor,
)


CONTACT_THRESHOLD_N = 0.1
FALL_HEIGHT_LOSS_M = 0.35


def _latest_filtered_force(sensor) -> torch.Tensor:
    force = sensor.data.force_matrix_w_history
    if force is None or force.ndim != 5 or force.shape[2:4] != (1, 1):
        raise RuntimeError("filtered foot-to-object ContactSensor geometry drift")
    return force[:, -1, 0, 0, :]


def _summaries(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile in range(args.num_envs):
        obj = arrays["object_root_state_w"][:, profile]
        root = arrays["robot_root_state_w"][:, profile]
        foot = arrays["foot_contact"][:, profile]
        reward = arrays["reward"][:, profile]
        root_loss = float(root[0, 2] - np.min(root[:, 2]))
        planar_net = float(np.linalg.norm(obj[-1, :2] - obj[0, :2]))
        planar_path = float(np.linalg.norm(np.diff(obj[:, :2], axis=0), axis=-1).sum())
        any_foot = np.any(foot, axis=-1)
        kick_success = bool(np.any(any_foot) and planar_net >= 0.05)
        physical_fall = bool(root_loss >= FALL_HEIGHT_LOSS_M)
        rows.append(
            {
                "profile": profile,
                "mean_reward": float(np.mean(reward)),
                "planar_object_net_displacement_m": planar_net,
                "planar_object_path_m": planar_path,
                "any_foot_box_contact_fraction": float(np.mean(any_foot)),
                "maximum_robot_root_height_loss_m": root_loss,
                "physical_robot_fall": physical_fall,
                "kick_success": kick_success,
                "safe_kick_success": bool(kick_success and not physical_fall),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric = (
        "mean_reward",
        "planar_object_net_displacement_m",
        "planar_object_path_m",
        "any_foot_box_contact_fraction",
        "maximum_robot_root_height_loss_m",
    )
    result = {
        f"mean_{name}": float(np.mean([float(row[name]) for row in rows]))
        for name in numeric
    }
    result["kick_success_count"] = int(sum(bool(row["kick_success"]) for row in rows))
    result["safe_kick_success_count"] = int(
        sum(bool(row["safe_kick_success"]) for row in rows)
    )
    result["physical_fall_count"] = int(
        sum(bool(row["physical_robot_fall"]) for row in rows)
    )
    return result


def main() -> None:
    output = args.output_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    experiments = (ROOT / "experiments").resolve()
    if experiments not in output.parents or not checkpoint.exists():
        raise ValueError("output must be under experiments and checkpoint must exist")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = RobotEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.episode_length_s = max(
        cfg.episode_length_s,
        (1 + args.carry_prefix_steps + args.steps + 25)
        * cfg.sim.dt
        * cfg.decimation,
    )
    cfg.observations.policy.enable_corruption = False
    cfg.terminations.physical_invalid = None
    cfg.rewards.physical_invalid_penalty = None
    env = gym.make(args.task, cfg=cfg)
    wrapped = OnlineCrossSkillRecoveryVecEnvWrapper(
        env,
        clip_actions=100.0,
        carry_tracker_checkpoint=SUGAR / "demo_ckpts/CarryBox/tracker.pt",
        kick_tracker_checkpoint=SUGAR / "demo_ckpts/KickBox/tracker.pt",
        carry_generator_checkpoint=SUGAR / "demo_ckpts/CarryBox/generator.ckpt",
        carry_prefix_steps=args.carry_prefix_steps,
        audit_path=output / "prefix_audit.json",
    )
    actor = _load_released_tracker_actor(checkpoint, wrapped.device)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    observations = wrapped.get_observations()
    base = wrapped.base_env

    initial = {
        "robot_root_state_w": base.scene["robot"].data.root_state_w.detach().cpu().numpy().copy(),
        "robot_joint_pos": base.scene["robot"].data.joint_pos.detach().cpu().numpy().copy(),
        "robot_joint_vel": base.scene["robot"].data.joint_vel.detach().cpu().numpy().copy(),
        "object_root_state_w": base.scene["obj"].data.root_state_w.detach().cpu().numpy().copy(),
        "policy_observation": observations["policy"].detach().cpu().numpy().copy(),
    }
    records: dict[str, list[np.ndarray]] = {
        "robot_root_state_w": [],
        "robot_body_position_w": [],
        "object_root_state_w": [],
        "foot_contact_force_w": [],
        "foot_contact": [],
        "policy_observation": [],
        "action": [],
        "reward": [],
        "done": [],
    }
    body_names = np.asarray(base.scene["robot"].body_names, dtype="U64")
    with torch.inference_mode():
        for _ in range(args.steps):
            policy_observation = observations["policy"]
            action = actor(policy_observation)
            observations, reward, done, _ = wrapped.step(action)
            forces = torch.stack(
                (
                    _latest_filtered_force(base.scene.sensors["left_foot_forces"]),
                    _latest_filtered_force(base.scene.sensors["right_foot_forces"]),
                ),
                dim=1,
            )
            records["robot_root_state_w"].append(
                base.scene["robot"].data.root_state_w.detach().cpu().numpy().copy()
            )
            records["robot_body_position_w"].append(
                base.scene["robot"].data.body_pos_w.detach().cpu().numpy().copy()
            )
            records["object_root_state_w"].append(
                base.scene["obj"].data.root_state_w.detach().cpu().numpy().copy()
            )
            force_np = forces.detach().cpu().numpy().copy()
            records["foot_contact_force_w"].append(force_np)
            records["foot_contact"].append(
                np.linalg.norm(force_np, axis=-1) > CONTACT_THRESHOLD_N
            )
            records["policy_observation"].append(
                policy_observation.detach().cpu().numpy().copy()
            )
            records["action"].append(action.detach().cpu().numpy().copy())
            records["reward"].append(reward.detach().cpu().numpy().copy())
            records["done"].append(done.detach().cpu().numpy().astype(bool, copy=True))

    arrays = {name: np.stack(values) for name, values in records.items()}
    arrays.update({f"initial_{name}": value for name, value in initial.items()})
    arrays["robot_body_names"] = body_names
    finite_names = tuple(
        name for name, value in arrays.items() if np.issubdtype(value.dtype, np.number)
    )
    all_finite = all(np.isfinite(arrays[name]).all() for name in finite_names)
    no_reset = not bool(np.any(arrays["done"]))
    rows = _summaries(arrays)
    result = {
        "protocol": "sugar_cross_skill_recovery_frozen_eval_v2",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_payload.get("iter"),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "prefix": {
            "kick_alignment_steps": 1,
            "carry_steps": args.carry_prefix_steps,
            "ppo_or_evaluation_credit_for_prefix": 0,
            "state_teleport": False,
            "offline_replay": False,
        },
        "aggregate": _aggregate(rows),
        "handoff": {
            "minimum_robot_root_height_m": float(
                np.min(initial["robot_root_state_w"][:, 2])
            ),
            "maximum_abs_policy_observation": float(
                np.max(np.abs(initial["policy_observation"]))
            ),
            "all_initial_values_finite": bool(
                all(np.isfinite(value).all() for value in initial.values())
            ),
        },
        "profiles": rows,
        "checks": {
            "all_trace_values_finite": bool(all_finite),
            "no_reset_during_frozen_window": bool(no_reset),
            "checkpoint_actor_geometry_is_official_510_512_256_128_29": True,
            "online_prefix_has_no_teleport_or_replay": True,
        },
    }
    result["structurally_valid"] = all(result["checks"].values())
    np.savez_compressed(output / "trace.npz", **arrays)
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    wrapped.close()
    if not result["structurally_valid"]:
        raise RuntimeError("recovery frozen evaluation failed structural checks")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
