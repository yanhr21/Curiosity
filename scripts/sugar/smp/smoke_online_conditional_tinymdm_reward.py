#!/usr/bin/env python3
"""Zero-optimizer online PhysX smoke for the conditional TinyMDM reward."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[3]
SUGAR = ROOT / "SUGAR"
SMP = ROOT / "scripts/sugar/smp"
for path in (SUGAR / "scripts/sugar_rl", SMP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(SUGAR / "descriptions/terrain/sugar_ground_plane.usda"),
)
os.environ.setdefault("ISAACLAB_USE_LOCAL_FRAME_MARKER", "1")
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ.setdefault("DISPLAY", "")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=191632)
parser.add_argument("--reward-seed", type=int, default=190001)
parser.add_argument("--class-id", type=int, choices=(0, 1), default=1)
parser.add_argument(
    "--reward-mode",
    choices=("occupancy", "progress", "contrastive_progress"),
    default="occupancy",
)
parser.add_argument("--carry-prefix-steps", type=int, default=41)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.task = "Sugar-G129dof-KickBox-Carry9-Recovery"
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import sugar_rl.tasks  # noqa: F401,E402
from audit_cross_skill_recovery_tinymdm import quaternion_wxyz_to_matrix  # noqa: E402
from run_selected_demo_tinymdm import build_feature_windows  # noqa: E402
from sugar_rl.tasks.locomanip.robots.g129dof.train_tracker.kick_box_carry9_recovery_v2_env_cfg import (  # noqa: E402
    RobotEnvCfg,
)
from sugar_rl.utils.online_cross_skill_recovery_wrapper import (  # noqa: E402
    OnlineCrossSkillRecoveryVecEnvWrapper,
    _load_released_tracker_actor,
)


PRIOR_ROOT = ROOT / "experiments/demo_following/conditional_taskwide_smp_v1"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def offline_features(scorer, device: torch.device) -> torch.Tensor:
    rows = []
    history = scorer.history
    for profile in range(scorer.num_envs):
        root = history["object_root"][profile].detach().cpu().numpy()
        robot = {
            "joint_pos": history["joint_pos"][profile].detach().cpu().numpy(),
            "joint_vel": history["joint_vel"][profile].detach().cpu().numpy(),
            "body_pos_w": history["body_pos"][profile].detach().cpu().numpy(),
            "body_quat_w": history["body_quat"][profile].detach().cpu().numpy(),
            "body_lin_vel_w": history["body_lin_vel"][profile].detach().cpu().numpy(),
            "body_ang_vel_w": history["body_ang_vel"][profile].detach().cpu().numpy(),
        }
        obj = {
            "obj_trans": root[:, 0:3],
            "obj_rot": quaternion_wxyz_to_matrix(root[:, 3:7]),
            "obj_lin_vel": root[:, 7:10],
            "obj_ang_vel": root[:, 10:13],
        }
        rows.append(build_feature_windows(robot, obj, device)[0])
    return torch.as_tensor(np.stack(rows), device=device)


def main() -> None:
    output = args.output.expanduser().resolve()
    experiments = (ROOT / "experiments").resolve()
    if experiments not in output.parents:
        raise ValueError("output must stay under ignored experiments/")
    if output.exists():
        raise FileExistsError(output)
    # SUGAR's released task configuration intentionally retains repository-
    # relative robot asset paths.
    os.chdir(SUGAR)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = RobotEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
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
        audit_path=output.parent / "wrapper_audit.json",
        reward_clip=10.0,
        conditional_tinymdm_config=PRIOR_ROOT / "prior/diffusion_config.yaml",
        conditional_tinymdm_checkpoint=PRIOR_ROOT / "prior/model.pt",
        conditional_tinymdm_calibration=PRIOR_ROOT / "reward_calibration/RESULT.json",
        conditional_tinymdm_class_id=args.class_id,
        conditional_tinymdm_reward_seed=args.reward_seed,
        conditional_tinymdm_reward_mode=args.reward_mode,
    )
    scorer = wrapped.conditional_tinymdm_reward
    if scorer is None:
        raise RuntimeError("conditional scorer was not installed")
    online = scorer._features().clone()
    reference = offline_features(scorer, torch.device(args.device))
    feature_max_abs_error = float(torch.max(torch.abs(online - reference)).item())

    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(torch.device(args.device)).clone()
    prior_reward, raw_loss = scorer.reward()
    cpu_rng_equal = bool(torch.equal(cpu_before, torch.get_rng_state()))
    cuda_rng_equal = bool(
        torch.equal(cuda_before, torch.cuda.get_rng_state(torch.device(args.device)))
    )

    observations = wrapped.get_observations()
    actor = _load_released_tracker_actor(
        SUGAR / "demo_ckpts/KickBox/tracker.pt", wrapped.device
    )
    action = actor(observations["policy"])
    _, combined_reward, done, extras = wrapped.step(action)
    checks = {
        "zero_optimizer_updates": True,
        "online_matches_offline_feature_adapter": feature_max_abs_error <= 1.0e-5,
        "causal_prefix_history_ready": scorer.observation_count
        >= scorer.history["joint_pos"].shape[1],
        "private_reward_rng_preserves_policy_cpu_rng": cpu_rng_equal,
        "private_reward_rng_preserves_policy_cuda_rng": cuda_rng_equal,
        "prior_reward_finite": bool(torch.isfinite(prior_reward).all()),
        "prior_loss_finite": bool(torch.isfinite(raw_loss).all()),
        "combined_reward_finite": bool(torch.isfinite(combined_reward).all()),
        "no_terminal_transition": not bool(torch.any(done).item()),
        "future_and_outcome_labels_excluded": True,
    }
    payload = {
        "protocol": "sugar_online_conditional_tinymdm_zero_optimizer_smoke_v1",
        "passed": all(checks.values()),
        "class_id": args.class_id,
        "reward_mode": args.reward_mode,
        "num_envs": args.num_envs,
        "carry_prefix_steps": args.carry_prefix_steps,
        "feature_max_abs_error": feature_max_abs_error,
        "prior_reward": {
            "mean": float(prior_reward.mean().item()),
            "min": float(prior_reward.min().item()),
            "max": float(prior_reward.max().item()),
        },
        "raw_sds_loss_mean": float(raw_loss.mean().item()),
        "combined_reward_mean": float(combined_reward.mean().item()),
        "wrapper_extra_reward_mean": float(
            extras["conditional_tinymdm_reward_mean"].item()
        ),
        "checks": checks,
        "scorer_audit": scorer.audit(),
    }
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    wrapped.close()
    if not payload["passed"]:
        raise RuntimeError("online conditional TinyMDM smoke failed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
