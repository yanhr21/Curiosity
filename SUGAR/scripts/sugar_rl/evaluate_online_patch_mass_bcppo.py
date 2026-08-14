#!/usr/bin/env python3
"""Frozen online evaluation for one Plan-15 Z/P/PS checkpoint."""

from __future__ import annotations

import argparse
import builtins
import json
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[3]
BRANCH_TASKS = {
    "Z": "Sugar-G129dof-CarryBox-OnlineMass-Patch-Z-BCPPO",
    "P": "Sugar-G129dof-CarryBox-OnlineMass-Patch-P-BCPPO",
    "PS": "Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-BCPPO",
}
EVALUATION_SEEDS = (152014, 152015, 152016)
TRAINING_SEEDS = (151014, 151015, 151016)
MASS_FACTORS = (1.0, 1.5, 3.0, 6.0, 10.0)
OFFICIAL_REFINER = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/"
    "baseline/ckpts/refiner_model10000.pt"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--branch", choices=tuple(BRANCH_TASKS), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--patch-scale-file", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--seed", type=int, choices=EVALUATION_SEEDS, required=True)
parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS)
parser.add_argument("--mass-factor", type=float, choices=MASS_FACTORS, required=True)
parser.add_argument("--motion-id", type=int, default=45)
parser.add_argument("--profiles", type=int, default=20)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--max-steps", type=int, default=420)
parser.add_argument("--post-jump-window", type=int, default=80)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.profiles < 1 or args.num_envs < 1 or args.profiles % args.num_envs:
    parser.error("profiles must be positive and divisible by num-envs")
if args.max_steps < args.post_jump_window:
    parser.error("max-steps must cover the post-jump window")

checkpoint = args.checkpoint.expanduser().resolve()
scale_file = args.patch_scale_file.expanduser().resolve()
output_root = args.output_root.expanduser().resolve()
for required_path in (checkpoint, scale_file, OFFICIAL_REFINER):
    if not required_path.is_file():
        raise FileNotFoundError(required_path)
if output_root.exists():
    raise FileExistsError(output_root)
scales = json.loads(scale_file.read_text(encoding="utf-8")).get(
    "patch_channel_scales"
)
if not isinstance(scales, list) or len(scales) != 9:
    raise ValueError("patch scale file must contain nine patch_channel_scales")
scales = [float(value) for value in scales]
if any(not math.isfinite(value) or value <= 0.0 for value in scales):
    raise ValueError("patch channel scales must be positive and finite")
os.environ["SUGAR_ONLINE_PATCH_CHANNEL_SCALES"] = json.dumps(scales)
os.environ["SUGAR_INIT_AT_RANDOM_EP_LEN"] = "0"
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
os.environ.setdefault("DISPLAY", "")
os.chdir(ROOT / "SUGAR")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import rsl_rl.algorithms  # noqa: E402
import rsl_rl.runners.on_policy_runner as on_policy_runner_module  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from sugar_rl.tasks.locomanip.online_patch_tactile import (  # noqa: E402
    current_whole_hand_patch_features,
)
from sugar_rl.tasks.locomanip.patch_slip import PatchSlipDetector  # noqa: E402
from sugar_rl.utils.online_patch_tactile_actor_critic import (  # noqa: E402
    OnlinePatchTactileActorCritic,
)
from sugar_rl.utils.rsl_rl_bcppo import BCPPO  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_patch_mass_bcppo_task_registration import (  # noqa: E402
    register_online_patch_mass_bcppo_tasks,
)


def cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def observations(wrapper) -> dict[str, torch.Tensor]:
    value = wrapper.get_observations()
    return value[0] if isinstance(value, tuple) else value


def profile_summary(trace: dict[str, np.ndarray], profile: int) -> dict[str, object]:
    jump_indices = np.flatnonzero(trace["jump_applied"][:, profile])
    jump_frame = int(jump_indices[0]) if len(jump_indices) else None
    termination_indices = np.flatnonzero(trace["termination_any"][:, profile])
    termination_frame = (
        int(termination_indices[0]) if len(termination_indices) else None
    )
    initial_z = float(trace["object_pos_w"][0, profile, 2])
    if jump_frame is None:
        return {
            "profile": profile,
            "jump_frame": None,
            "first_termination_frame": termination_frame,
            "eligible_post_jump_window": False,
            "hold_success": False,
            "drop": False,
            "safe_lower": False,
            "robot_fall": bool(trace["robot_fall"][:, profile].any()),
            "bilateral_patch_contact_frames": int(
                np.count_nonzero(trace["bilateral_patch_contact"][:, profile])
            ),
        }
    stop = min(jump_frame + int(args.post_jump_window), len(trace["object_pos_w"]))
    eligible = stop - jump_frame == int(args.post_jump_window)
    z = trace["object_pos_w"][jump_frame:stop, profile, 2]
    vz = trace["object_lin_vel_w"][jump_frame:stop, profile, 2]
    jump_z = float(trace["object_pos_w"][jump_frame, profile, 2])
    height_loss = float(jump_z - np.min(z))
    robot_fall = bool(trace["robot_fall"][jump_frame:stop, profile].any())
    drop = bool(height_loss >= 0.15 or np.min(z) <= initial_z + 0.03)
    hold_success = bool(eligible and height_loss <= 0.05 and not robot_fall)
    safe_lower = bool(
        eligible
        and not hold_success
        and not drop
        and not robot_fall
        and np.min(z) <= initial_z + 0.08
        and np.min(vz) >= -0.35
        and np.max(trace["reference_orientation_error_rad"][jump_frame:stop, profile]) <= 0.8
    )
    bilateral = trace["bilateral_patch_contact"][jump_frame:stop, profile]
    return {
        "profile": profile,
        "jump_frame": jump_frame,
        "first_termination_frame": termination_frame,
        "eligible_post_jump_window": eligible,
        "hold_success": hold_success,
        "drop": drop,
        "safe_lower": safe_lower,
        "robot_fall": robot_fall,
        "jump_height_m": jump_z,
        "minimum_post_jump_height_m": float(np.min(z)),
        "maximum_height_loss_m": height_loss,
        "maximum_reference_position_error_m": float(
            np.max(trace["reference_position_error_m"][jump_frame:stop, profile])
        ),
        "maximum_reference_orientation_error_rad": float(
            np.max(trace["reference_orientation_error_rad"][jump_frame:stop, profile])
        ),
        "bilateral_patch_contact_fraction": float(np.mean(bilateral)),
        "gross_slip_patch_fraction": float(
            np.mean(trace["slip_state"][jump_frame:stop, profile] == 3)
        ),
        "reward_sum": float(
            np.sum(
                trace["reward"][:, profile][
                    np.isfinite(trace["reward"][:, profile])
                ]
            )
        ),
    }


def main() -> None:
    register_online_patch_mass_bcppo_tasks()
    setattr(builtins, "BCPPO", BCPPO)
    setattr(rsl_rl.algorithms, "BCPPO", BCPPO)
    setattr(builtins, "OnlinePatchTactileActorCritic", OnlinePatchTactileActorCritic)
    setattr(on_policy_runner_module, "OnlinePatchTactileActorCritic", OnlinePatchTactileActorCritic)

    task = BRANCH_TASKS[args.branch]
    task_spec = gym.spec(task)
    env_cfg_type = task_spec.kwargs["play_env_cfg_entry_point"]
    module_name, class_name = env_cfg_type.split(":")
    module = __import__(module_name, fromlist=[class_name])
    env_cfg = getattr(module, class_name)()
    env_cfg.seed = int(args.seed)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = args.device
    env_cfg.commands.motion.motion_folder = "data/CarryBox"
    for term_name in ("reset_mass_jump", "step_mass_jump"):
        params = getattr(env_cfg.events, term_name).params
        params["mass_factors"] = (float(args.mass_factor),)
        params["seed"] = int(args.seed)
    for group_name in ("policy", "online_patch_tactile_history", "critic", "teacher"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None:
            group.enable_corruption = False

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = int(args.seed)
    agent_cfg.device = args.device
    agent_cfg.algorithm.teacher_ckpt = str(OFFICIAL_REFINER)
    gym_env = gym.make(task, cfg=env_cfg, render_mode=None)
    base_env = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=base_env.device)
    obj = base_env.scene["obj"]
    command = base_env.command_manager.get_term("motion")

    def fixed_start(env_ids) -> None:
        ids = torch.as_tensor(
            env_ids, dtype=torch.int64, device=base_env.device
        )
        command.motion_id[ids] = int(args.motion_id)
        command.time_steps[ids] = 0
        command._use_motion_data[ids] = True

    command._sample_init_state = fixed_start
    termination_names = tuple(base_env.termination_manager.active_terms)
    robot_fall_names = tuple(
        name
        for name in termination_names
        if name in {"anchor_ori", "anchor_pos", "ee_body_pos"}
    )

    total_profiles = int(args.profiles)
    traces: dict[str, list[np.ndarray]] = {
        "action": [],
        "reward": [],
        "object_pos_w": [],
        "object_lin_vel_w": [],
        "reference_position_error_m": [],
        "reference_orientation_error_rad": [],
        "patch_features": [],
        "slip_state": [],
        "bilateral_patch_contact": [],
        "jump_applied": [],
        "mass_changed": [],
        "mass_readback_kg": [],
        "robot_fall": [],
        "termination_any": [],
        "reference_frame": [],
    }
    original_reset_idx = base_env._reset_idx
    try:
        for batch in range(total_profiles // int(args.num_envs)):
            base_env._reset_idx = original_reset_idx
            reset_value = env.reset()
            obs = reset_value[0] if isinstance(reset_value, tuple) else reset_value
            if not bool((command.motion_id == int(args.motion_id)).all()):
                raise RuntimeError("frozen evaluation motion id did not survive reset")
            if not bool((command.time_steps == 0).all()):
                raise RuntimeError("frozen evaluation did not start from motion frame 0")

            # Reset writes the requested physical motion state, but the official
            # SUGAR command's relative-body buffers are refreshed only by
            # _update_command().  Synchronize them before the first policy
            # observation, following the repository's frozen evaluator path.
            command.time_steps[:] = -1
            command._update_command()
            if not bool((command.time_steps == 0).all()):
                raise RuntimeError("command-buffer synchronization changed frame 0")
            base_env.obs_buf = base_env.observation_manager.compute(
                update_history=False
            )
            obs = observations(env)
            base_env._reset_idx = lambda env_ids: None
            detector = PatchSlipDetector(base_env.num_envs, device=base_env.device)
            batch_rows = {name: [] for name in traces}
            for step in range(int(args.max_steps)):
                with torch.inference_mode():
                    action = policy(obs)
                    obs, reward, done, _ = env.step(action)
                patch = current_whole_hand_patch_features(base_env)
                timestamp = torch.full(
                    (base_env.num_envs,),
                    (step + 1) * float(base_env.step_dt),
                    dtype=torch.float32,
                    device=base_env.device,
                )
                slip = detector.update(
                    contact=patch[..., 0].bool(),
                    normal_load_n=patch[..., 1],
                    mean_pressure_pa=patch[..., 2],
                    shear_xy_n=patch[..., 3:5],
                    friction_utilization=patch[..., 5],
                    timestamp_s=timestamp,
                    reset_mask=torch.full(
                        (base_env.num_envs,),
                        step == 0,
                        dtype=torch.bool,
                        device=base_env.device,
                    ),
                )
                diagnostics = base_env._online_mass_jump_diagnostics
                ref_pos_error = torch.linalg.vector_norm(
                    command.obj_pos_w - command.obj_ref_pos_w, dim=-1
                )
                dot = torch.sum(command.obj_quat_w * command.obj_ref_quat_w, dim=-1).abs()
                ref_ori_error = 2.0 * torch.acos(torch.clamp(dot, 0.0, 1.0))
                robot_fall = torch.zeros_like(done, dtype=torch.bool)
                for name in robot_fall_names:
                    robot_fall |= base_env.termination_manager.get_term(name)
                batch_rows["action"].append(cpu(action))
                batch_rows["reward"].append(cpu(reward))
                batch_rows["object_pos_w"].append(cpu(obj.data.root_pos_w))
                batch_rows["object_lin_vel_w"].append(cpu(obj.data.root_lin_vel_w))
                batch_rows["reference_position_error_m"].append(cpu(ref_pos_error))
                batch_rows["reference_orientation_error_rad"].append(cpu(ref_ori_error))
                batch_rows["patch_features"].append(cpu(patch))
                batch_rows["slip_state"].append(cpu(slip.state))
                batch_rows["bilateral_patch_contact"].append(
                    cpu((patch[..., 0] > 0.5).any(dim=-1).all(dim=-1))
                )
                batch_rows["jump_applied"].append(cpu(diagnostics["jump_applied"]))
                batch_rows["mass_changed"].append(cpu(diagnostics["mass_changed"]))
                batch_rows["mass_readback_kg"].append(cpu(diagnostics["mass_readback_kg"]))
                batch_rows["robot_fall"].append(cpu(robot_fall))
                batch_rows["termination_any"].append(cpu(done.bool()))
                batch_rows["reference_frame"].append(cpu(command.time_steps))
            for name in traces:
                traces[name].append(np.stack(batch_rows[name], axis=0))
            print(
                f"[PLAN15 EVAL] batch {batch + 1}/{total_profiles // int(args.num_envs)} complete",
                flush=True,
            )
    finally:
        base_env._reset_idx = original_reset_idx
        env.close()

    arrays = {
        name: np.concatenate(values, axis=1)
        for name, values in traces.items()
    }
    if any(value.shape[1] != total_profiles for value in arrays.values()):
        raise RuntimeError("profile count mismatch in frozen evaluation trace")
    output_root.mkdir(parents=True)
    np.savez_compressed(output_root / "frozen_evaluation_trace.npz", **arrays)
    episodes = [profile_summary(arrays, index) for index in range(total_profiles)]
    eligible = [item for item in episodes if item["eligible_post_jump_window"]]
    summary = {
        "schema": "plan15_frozen_online_patch_mass_evaluation_v1",
        "branch": args.branch,
        "checkpoint": str(checkpoint),
        "training_seed": args.training_seed,
        "seed": int(args.seed),
        "mass_factor": float(args.mass_factor),
        "motion_id": int(args.motion_id),
        "start_frame": 0,
        "profiles": total_profiles,
        "max_steps": int(args.max_steps),
        "post_jump_window_frames": int(args.post_jump_window),
        "actor_mass_or_jump_input": False,
        "evaluation_patch_and_slip_labels_feed_actor": False,
        "policy_spatial_unit": "27 physical patches per hand; no taxel policy units",
        "eligible_profiles": len(eligible),
        "hold_success_count": sum(bool(item["hold_success"]) for item in eligible),
        "drop_count": sum(bool(item["drop"]) for item in eligible),
        "safe_lower_count": sum(bool(item["safe_lower"]) for item in eligible),
        "robot_fall_count": sum(bool(item["robot_fall"]) for item in episodes),
        "episodes": episodes,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
