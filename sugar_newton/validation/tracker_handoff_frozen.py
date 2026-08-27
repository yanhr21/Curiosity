# SPDX-License-Identifier: BSD-3-Clause
"""Matched frozen evaluation for a Newton Tracker after a real Refiner handoff.

The baseline and learned Tracker actors share one Newton environment, the exact admitted
acting Refiner, fixed processed motion profiles, frame-zero resets and deterministic action
means.  Every profile is reset through Newton's official solver reset before each arm.  The
only paired change is the 510-D Tracker actor checkpoint used after the 5 cm / 10-frame
handoff.  No optimizer, future outcome, privileged critic signal or manual selection enters
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl.train_bcppo import (
    activate_rsl_rl,
    require_admitted_teacher,
)


DEFAULT_CLIPS = (
    "data_000_000_t0",
    "data_001_001_t0",
    "data_005_005_t0",
    "data_010_010_t0",
    "data_015_015_t0",
    "data_020_020_t0",
    "data_025_025_t0",
    "data_030_030_t0",
    "data_035_035_t0",
    "data_045_045_t0",
    "data_050_050_t0",
    "data_055_055_t0",
    "data_065_065_t0",
    "data_070_070_t0",
    "data_075_075_t0",
    "data_080_080_t0",
    "data_085_085_t0",
    "data_090_090_t0",
    "data_095_095_t0",
    "data_099_099_t0",
)
TERMINATION_KEYS = (
    "anchor_ori",
    "anchor_pos",
    "ee_pos",
    "obj_pos",
    "obj_ori",
    "diverged",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_tracker_actor(checkpoint: Path, device: torch.device):
    """Strictly rebuild the released 510-D official Tracker actor topology."""
    from rsl_rl.networks.mlp import MLP

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"Tracker checkpoint has no model_state_dict: {checkpoint}")
    weight_keys = sorted(
        (
            key
            for key in state
            if key.startswith("actor.") and key.endswith("weight")
        ),
        key=lambda key: int(key.split(".")[1]),
    )
    if not weight_keys:
        raise KeyError(f"Tracker checkpoint has no actor weights: {checkpoint}")
    input_dim = int(state[weight_keys[0]].shape[1])
    output_dim = int(state[weight_keys[-1]].shape[0])
    hidden_dims = [int(state[key].shape[0]) for key in weight_keys[:-1]]
    if (input_dim, hidden_dims, output_dim) != (
        510,
        [512, 256, 128],
        29,
    ):
        raise ValueError(
            "Tracker actor geometry drifted: "
            f"{input_dim}->{hidden_dims}->{output_dim}"
        )
    actor = MLP(input_dim, output_dim, hidden_dims, "elu").to(device)
    actor.load_state_dict(
        {
            key.removeprefix("actor."): value
            for key, value in state.items()
            if key.startswith("actor.")
        },
        strict=True,
    )
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


@torch.inference_mode()
def evaluate_arm(env, actor, arm: str) -> list[dict]:
    records: list[dict] = []
    env_id = torch.zeros(1, dtype=torch.long, device=env.device)
    for motion_id, clip in enumerate(env.env.clip_names):
        env.env.reset(env_id, motion_ids=motion_id, start_frames=0)
        env.episode_length_buf.zero_()
        env._reset_handoff(env_id)
        observation = env.get_observations()
        initial_observation = {
            key: observation[key].clone()
            for key in ("policy", "critic", "teacher")
        }

        initial_q = env.env.q.clone()
        initial_qd = env.env.qd.clone()
        initial_box_z = env.env._body_q()[0, env.env.box_body, 2].clone()
        initial_root_z = env.env._body_q()[0, env.env.body_idx[0], 2].clone()
        peak_box_z = initial_box_z.clone()
        min_root_z = initial_root_z.clone()
        reward_sum = 0.0
        bilateral_steps = 0
        teacher_steps = 0
        policy_steps = 0
        handoff_step = -1
        finished = False
        timeout_at_finish = False
        all_finite = True
        max_abs_policy_action = 0.0
        max_abs_executed_action = 0.0
        strict_reasons: set[str] = set()
        handoffs_before = int(env.cumulative_handoffs[0])
        evaluated_steps = 0

        source_frames = int(env.env.ref["length"][motion_id])
        for step in range(source_frames + 2):
            policy_action = actor(observation["policy"])
            all_finite &= bool(torch.isfinite(policy_action).all())
            max_abs_policy_action = max(
                max_abs_policy_action, float(policy_action.abs().max())
            )
            observation, reward, done, _ = env.step(policy_action)
            evaluated_steps += 1
            teacher_control = bool(env.last_teacher_control_mask[0])
            teacher_steps += int(teacher_control)
            policy_steps += int(not teacher_control)
            max_abs_executed_action = max(
                max_abs_executed_action,
                float(env.last_executed_action[0].abs().max()),
            )
            if (
                handoff_step < 0
                and int(env.cumulative_handoffs[0]) > handoffs_before
            ):
                handoff_step = step + 1

            body_q = env.env._body_q()
            state_finite = bool(
                torch.isfinite(body_q).all()
                and torch.isfinite(env.env.q).all()
                and torch.isfinite(env.env.qd).all()
            )
            all_finite &= state_finite
            peak_box_z = torch.maximum(
                peak_box_z, body_q[0, env.env.box_body, 2]
            )
            min_root_z = torch.minimum(
                min_root_z, body_q[0, env.env.body_idx[0], 2]
            )
            reward_sum += float(reward[0])
            hand_force = env.env.contact_history.box[
                0, :, env.env.contact_history.hand_idx
            ]
            bilateral_steps += int(
                bool((hand_force.norm(dim=-1).amax(dim=0) > 0.1).all())
            )

            if bool(done[0]):
                finished = True
                extras = env.env.extras
                timeout_at_finish = bool(extras["timeout"][0])
                for key in TERMINATION_KEYS:
                    if bool(extras["termination_terms"][key][0]):
                        strict_reasons.add(key)
                break

        peak_lift = float(peak_box_z - initial_box_z)
        records.append(
            {
                "arm": arm,
                "clip": clip,
                "start_frame": 0,
                "source_frames": source_frames,
                "evaluated_steps": evaluated_steps,
                "initial_state_sha256": tensor_state_sha256(
                    initial_q,
                    initial_qd,
                    initial_observation["policy"],
                    initial_observation["critic"],
                    initial_observation["teacher"],
                ),
                "finished": finished,
                "trajectory_timeout": timeout_at_finish,
                "strict_failure": bool(strict_reasons),
                "strict_failure_reasons": sorted(strict_reasons),
                "all_finite": all_finite,
                "handoff_occurred": handoff_step >= 0,
                "handoff_step": handoff_step,
                "teacher_control_steps": teacher_steps,
                "policy_control_steps": policy_steps,
                "box_peak_lift_m": peak_lift,
                "root_height_loss_m": float(initial_root_z - min_root_z),
                "root_below_0_65": bool(min_root_z < 0.65),
                "bilateral_contact_fraction": float(
                    bilateral_steps / max(evaluated_steps, 1)
                ),
                "mean_reward_per_step": float(
                    reward_sum / max(evaluated_steps, 1)
                ),
                "max_abs_policy_action": max_abs_policy_action,
                "max_abs_executed_action": max_abs_executed_action,
                "safe_complete_after_handoff": bool(
                    handoff_step >= 0
                    and timeout_at_finish
                    and not strict_reasons
                    and all_finite
                ),
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    return {
        "num_profiles": len(records),
        "finished_count": sum(record["finished"] for record in records),
        "finite_count": sum(record["all_finite"] for record in records),
        "handoff_count": sum(record["handoff_occurred"] for record in records),
        "safe_complete_after_handoff_count": sum(
            record["safe_complete_after_handoff"] for record in records
        ),
        "strict_failure_count": sum(
            record["strict_failure"] for record in records
        ),
        "divergence_count": sum(
            "diverged" in record["strict_failure_reasons"]
            for record in records
        ),
        "root_below_0_65_count": sum(
            record["root_below_0_65"] for record in records
        ),
        "mean_peak_lift_m": statistics.fmean(
            record["box_peak_lift_m"] for record in records
        ),
        "mean_bilateral_contact_fraction": statistics.fmean(
            record["bilateral_contact_fraction"] for record in records
        ),
        "mean_reward_per_step": statistics.fmean(
            record["mean_reward_per_step"] for record in records
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--learned-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-gate-result", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument(
        "--teacher-motion-root", type=Path, default=Path("SUGAR/data/CarryBox")
    )
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rsl-rl-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--mu", type=float, default=1.0)
    args = parser.parse_args()

    for label, path in (
        ("baseline Tracker", args.baseline_checkpoint),
        ("learned Tracker", args.learned_checkpoint),
        ("acting Refiner", args.teacher_checkpoint),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} checkpoint not found: {path}")
    if not args.clips:
        raise SystemExit("frozen Tracker evaluation requires fixed profiles")
    missing = [clip for clip in args.clips if not (args.motion_root / clip).is_dir()]
    if missing:
        raise SystemExit(f"missing processed Tracker profiles: {missing}")

    teacher_gate = require_admitted_teacher(
        args.teacher_gate_result, args.teacher_checkpoint
    )
    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("Tracker frozen evaluation must run on a Slurm CUDA node")
    activate_rsl_rl(args.rsl_rl_root)

    from sugar_newton.rl.vec_env import make_acting_teacher_handoff

    device = torch.device(args.device)
    baseline_actor = load_tracker_actor(args.baseline_checkpoint, device)
    learned_actor = load_tracker_actor(args.learned_checkpoint, device)
    env = make_acting_teacher_handoff(
        1,
        teacher_checkpoint=args.teacher_checkpoint,
        clip_names=list(args.clips),
        episode_length=10**9,
        motion_root=args.motion_root,
        teacher_motion_root=args.teacher_motion_root,
        substeps=args.substeps,
        mu=args.mu,
        device=args.device,
        seed=0,
        auto_reset=False,
        sync_divergence_reset=False,
    )
    baseline_records = evaluate_arm(env, baseline_actor, "official_tracker")
    learned_records = evaluate_arm(env, learned_actor, "learned_tracker")
    baseline = summarize(baseline_records)
    learned = summarize(learned_records)

    baseline_by_clip = {record["clip"]: record for record in baseline_records}
    learned_by_clip = {record["clip"]: record for record in learned_records}
    initial_state_match = all(
        baseline_by_clip[clip]["initial_state_sha256"]
        == learned_by_clip[clip]["initial_state_sha256"]
        for clip in args.clips
    )
    paired = {
        clip: {
            "safe_complete_delta": int(
                learned_by_clip[clip]["safe_complete_after_handoff"]
            )
            - int(baseline_by_clip[clip]["safe_complete_after_handoff"]),
            "peak_lift_delta_m": (
                learned_by_clip[clip]["box_peak_lift_m"]
                - baseline_by_clip[clip]["box_peak_lift_m"]
            ),
            "bilateral_contact_fraction_delta": (
                learned_by_clip[clip]["bilateral_contact_fraction"]
                - baseline_by_clip[clip]["bilateral_contact_fraction"]
            ),
            "reward_per_step_delta": (
                learned_by_clip[clip]["mean_reward_per_step"]
                - baseline_by_clip[clip]["mean_reward_per_step"]
            ),
        }
        for clip in args.clips
    }
    checks = {
        "teacher_gate_passed_for_exact_checkpoint": True,
        "baseline_actor_exact_official_510_to_29_mlp": True,
        "learned_actor_exact_official_510_to_29_mlp": True,
        "all_initial_states_elementwise_identical_by_sha256": initial_state_match,
        "all_baseline_profiles_finished": (
            baseline["finished_count"] == len(args.clips)
        ),
        "all_learned_profiles_finished": (
            learned["finished_count"] == len(args.clips)
        ),
        "all_baseline_profiles_finite": (
            baseline["finite_count"] == len(args.clips)
        ),
        "all_learned_profiles_finite": (
            learned["finite_count"] == len(args.clips)
        ),
    }
    physical_advantage = {
        "safe_complete_count_not_lower": (
            learned["safe_complete_after_handoff_count"]
            >= baseline["safe_complete_after_handoff_count"]
        ),
        "divergence_count_not_higher": (
            learned["divergence_count"] <= baseline["divergence_count"]
        ),
        "root_below_0_65_count_not_higher": (
            learned["root_below_0_65_count"]
            <= baseline["root_below_0_65_count"]
        ),
        "mean_reward_per_step_not_lower": (
            learned["mean_reward_per_step"] >= baseline["mean_reward_per_step"]
        ),
    }
    result = {
        "protocol": "sugar_newton_tracker_matched_handoff_frozen_v1",
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": file_sha256(args.baseline_checkpoint),
        "learned_checkpoint": str(args.learned_checkpoint.resolve()),
        "learned_checkpoint_sha256": file_sha256(args.learned_checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_gate": teacher_gate,
        "motion_root": str(args.motion_root.resolve()),
        "teacher_motion_root": str(args.teacher_motion_root.resolve()),
        "clips": list(args.clips),
        "checks": checks,
        "evaluation_valid": all(checks.values()),
        "physical_advantage_checks": physical_advantage,
        "physical_advantage_passed": all(physical_advantage.values()),
        "baseline": baseline,
        "learned": learned,
        "paired": paired,
        "profiles": baseline_records + learned_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "profiles"}, indent=2))
    return 0 if result["evaluation_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
