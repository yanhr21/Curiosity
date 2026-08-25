#!/usr/bin/env python3
"""Frozen rollout after the online Carry9-to-Kick physical handoff."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[3]
SUGAR = ROOT / "SUGAR"
for path in (SUGAR / "scripts/sugar_rl",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.chdir(SUGAR)

os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(SUGAR / "descriptions/terrain/sugar_ground_plane.usda"),
)
os.environ.setdefault("ISAACLAB_USE_LOCAL_FRAME_MARKER", "1")
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ.setdefault(
    "VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json"
)
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
parser.add_argument(
    "--transition-selected-skill-id", type=int, choices=(0, 1), default=None
)
parser.add_argument(
    "--policy-topology",
    choices=("selected_expert_residual", "causal_action_composition"),
    default="selected_expert_residual",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 20 or args.steps != 250:
    parser.error("frozen recovery evaluation is fixed to 20 envs x 250 steps")
if args.carry_prefix_steps <= 0:
    parser.error("carry prefix must be positive")
if (
    args.policy_topology == "causal_action_composition"
    and args.transition_selected_skill_id is None
):
    parser.error("causal action composition requires a selected skill")
args.task = "Sugar-G129dof-KickBox-Carry9-Recovery"
args.enable_cameras = False

# Reject path/provenance errors before AppLauncher starts Kit.  Isaac Sim can
# swallow a Python exception during shutdown, so this check must not live in
# main() after graphics/physics initialization.
preflight_output = args.output_dir.expanduser().resolve()
preflight_checkpoint = args.checkpoint.expanduser().resolve()
preflight_experiments = (ROOT / "experiments").resolve()
if (
    preflight_experiments not in preflight_output.parents
    or not preflight_checkpoint.exists()
):
    parser.error("output must be under experiments and checkpoint must exist")
if preflight_output.exists():
    parser.error(f"output already exists: {preflight_output}")

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
from sugar_rl.utils.frozen_expert_transition_actor_critic import (  # noqa: E402
    FrozenExpertCausalActionComposerActorCritic,
    FrozenExpertTransitionActorCritic,
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
    output = preflight_output
    checkpoint = preflight_checkpoint
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
        transition_selected_skill_id=args.transition_selected_skill_id,
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    observations = wrapped.get_observations()
    if args.transition_selected_skill_id is None:
        actor = _load_released_tracker_actor(checkpoint, wrapped.device)
        transition_policy = None
    else:
        if args.policy_topology == "causal_action_composition":
            policy_class = FrozenExpertCausalActionComposerActorCritic
            policy_observation_groups = [
                "policy",
                "carry_skill_command",
                "kick_skill_command",
                "selected_skill_id",
            ]
        else:
            policy_class = FrozenExpertTransitionActorCritic
            policy_observation_groups = [
                "policy",
                "selected_skill_command",
                "selected_skill_id",
            ]
        transition_policy = policy_class(
            observations,
            {
                "policy": policy_observation_groups,
                "critic": ["critic", *policy_observation_groups[1:]],
                "teacher": ["teacher"],
            },
            29,
            carry_tracker_checkpoint=str(
                SUGAR / "demo_ckpts/CarryBox/tracker.pt"
            ),
            kick_tracker_checkpoint=str(SUGAR / "demo_ckpts/KickBox/tracker.pt"),
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=0.05,
        ).to(wrapped.device)
        transition_policy.load_state_dict(
            checkpoint_payload["model_state_dict"], strict=True
        )
        transition_policy.eval().requires_grad_(False)
        actor = None
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
        "robot_body_quaternion_w": [],
        "robot_body_linear_velocity_w": [],
        "robot_body_angular_velocity_w": [],
        "robot_joint_position": [],
        "robot_joint_velocity": [],
        "object_root_state_w": [],
        "foot_contact_force_w": [],
        "foot_contact": [],
        "policy_observation": [],
        "action": [],
        "reward": [],
        "done": [],
    }
    if args.policy_topology == "causal_action_composition":
        for name in (
            "kick_composition_weight",
            "selected_endpoint_action",
            "mixed_endpoint_action",
            "bounded_residual_action",
            "composed_action",
        ):
            records[name] = []
    body_names = np.asarray(base.scene["robot"].body_names, dtype="U64")
    with torch.inference_mode():
        for _ in range(args.steps):
            policy_observation = observations["policy"]
            action = (
                actor(policy_observation)
                if transition_policy is None
                else transition_policy.act_inference(observations)
            )
            if isinstance(
                transition_policy, FrozenExpertCausalActionComposerActorCritic
            ):
                terms = transition_policy.composition_audit_terms(observations)
                for source_name, record_name in (
                    ("kick_weight", "kick_composition_weight"),
                    ("selected_endpoint_action", "selected_endpoint_action"),
                    ("mixed_endpoint_action", "mixed_endpoint_action"),
                    ("bounded_residual_action", "bounded_residual_action"),
                    ("composed_action", "composed_action"),
                ):
                    records[record_name].append(
                        terms[source_name].detach().cpu().numpy().copy()
                    )
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
            records["robot_body_quaternion_w"].append(
                base.scene["robot"].data.body_quat_w.detach().cpu().numpy().copy()
            )
            records["robot_body_linear_velocity_w"].append(
                base.scene["robot"].data.body_lin_vel_w.detach().cpu().numpy().copy()
            )
            records["robot_body_angular_velocity_w"].append(
                base.scene["robot"].data.body_ang_vel_w.detach().cpu().numpy().copy()
            )
            records["robot_joint_position"].append(
                base.scene["robot"].data.joint_pos.detach().cpu().numpy().copy()
            )
            records["robot_joint_velocity"].append(
                base.scene["robot"].data.joint_vel.detach().cpu().numpy().copy()
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
    arrays["robot_joint_names"] = np.asarray(
        base.scene["robot"].joint_names, dtype="U64"
    )
    finite_names = tuple(
        name for name, value in arrays.items() if np.issubdtype(value.dtype, np.number)
    )
    all_finite = all(np.isfinite(arrays[name]).all() for name in finite_names)
    no_reset = not bool(np.any(arrays["done"]))
    rows = _summaries(arrays)
    result = {
        "protocol": "sugar_cross_skill_recovery_frozen_eval_v3",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_payload.get("iter"),
        "transition_selected_skill_id": args.transition_selected_skill_id,
        "policy_topology": args.policy_topology,
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
        "action_composition": (
            {
                "minimum_kick_weight": float(
                    np.min(arrays["kick_composition_weight"])
                ),
                "maximum_kick_weight": float(
                    np.max(arrays["kick_composition_weight"])
                ),
                "mean_kick_weight": float(
                    np.mean(arrays["kick_composition_weight"])
                ),
                "mean_abs_deviation_from_selected_endpoint": float(
                    np.mean(
                        np.abs(
                            arrays["kick_composition_weight"]
                            - float(args.transition_selected_skill_id)
                        )
                    )
                ),
                "mean_abs_mixed_minus_selected_endpoint_action": float(
                    np.mean(
                        np.abs(
                            arrays["mixed_endpoint_action"]
                            - arrays["selected_endpoint_action"]
                        )
                    )
                ),
                "mean_abs_bounded_residual_action": float(
                    np.mean(np.abs(arrays["bounded_residual_action"]))
                ),
                "mean_abs_composed_minus_selected_endpoint_action": float(
                    np.mean(
                        np.abs(
                            arrays["composed_action"]
                            - arrays["selected_endpoint_action"]
                        )
                    )
                ),
                "maximum_abs_deployed_minus_composed_action": float(
                    np.max(np.abs(arrays["action"] - arrays["composed_action"]))
                ),
                "future_or_outcome_labels_used": False,
            }
            if args.policy_topology == "causal_action_composition"
            else None
        ),
        "checks": {
            "all_trace_values_finite": bool(all_finite),
            "no_reset_during_frozen_window": bool(no_reset),
            "checkpoint_actor_contract_valid": True,
            "online_prefix_has_no_teleport_or_replay": True,
            "composition_weight_in_unit_interval": bool(
                args.policy_topology != "causal_action_composition"
                or (
                    np.min(arrays["kick_composition_weight"]) >= 0.0
                    and np.max(arrays["kick_composition_weight"]) <= 1.0
                )
            ),
            "composition_terms_match_deployed_action": bool(
                args.policy_topology != "causal_action_composition"
                or np.array_equal(arrays["action"], arrays["composed_action"])
            ),
            "feature_complete_for_official_tinymdm": all(
                name in arrays
                for name in (
                    "robot_body_position_w",
                    "robot_body_quaternion_w",
                    "robot_body_linear_velocity_w",
                    "robot_body_angular_velocity_w",
                    "robot_joint_position",
                    "robot_joint_velocity",
                    "object_root_state_w",
                    "robot_body_names",
                    "robot_joint_names",
                )
            ),
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
    except BaseException:  # preserve the Python cause before Kit shutdown
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
