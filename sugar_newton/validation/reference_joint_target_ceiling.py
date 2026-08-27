# SPDX-License-Identifier: BSD-3-Clause
"""Strict Newton feasibility gate for the supplied CarryBox joint reference.

This is deliberately parameter-free.  At every causal control step it inverts
the exact deployed Isaac joint-position action map,

    action = (reference_joint_position - default_joint_position) / action_scale

and executes that action in the unchanged Newton environment.  It neither loads
nor trains a policy, and it does not clip the resulting action.  The fixed twenty
profiles and strict terminations are shared with the official Refiner/Tracker
gates, so late post-failure motion cannot masquerade as a feasible reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl.carrybox_env import CarryBoxEnv
from sugar_newton.rl.train_bcppo import activate_rsl_rl
from sugar_newton.validation.refiner_open_loop import DEFAULT_CLIPS, evaluate_batch


class ExactReferenceJointTarget:
    """Causal exact inverse of ``CarryBoxEnv.step``'s joint action map."""

    def __init__(self, env: CarryBoxEnv):
        self.env = env
        self.calls = 0
        self.maximum_reconstruction_delta = 0.0
        self.all_actions_finite = True

    def __call__(self, _observation: torch.Tensor) -> torch.Tensor:
        reference = self.env._ref("joint_pos")
        action = (reference - self.env.q_default) / self.env.a_scale
        reconstructed = action * self.env.a_scale + self.env.q_default
        self.maximum_reconstruction_delta = max(
            self.maximum_reconstruction_delta,
            float((reconstructed - reference).abs().max().item()),
        )
        self.all_actions_finite &= bool(torch.isfinite(action).all())
        self.calls += int(action.shape[0])
        return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, default=Path("SUGAR/data/CarryBox"))
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--minimum-lift", type=float, default=0.05)
    parser.add_argument("--minimum-pass-fraction", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rsl-rl-root", required=True)
    args = parser.parse_args()

    if not 1 <= args.num_envs <= 8:
        raise SystemExit("--num-envs must be in the validated range 1..8")
    if not args.clips:
        raise SystemExit("the reference ceiling requires at least one clip")
    missing = [clip for clip in args.clips if not (args.motion_root / clip).is_dir()]
    if missing:
        raise SystemExit(f"missing source clips: {missing}")

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("reference ceiling must run on a Slurm CUDA compute node")
    activate_rsl_rl(args.rsl_rl_root)
    env = CarryBoxEnv(
        num_envs=args.num_envs,
        clip_names=list(args.clips),
        motion_root=args.motion_root,
        teacher_motion_root=args.motion_root,
        episode_length=10**9,
        substeps=args.substeps,
        mu=args.mu,
        device=args.device,
        seed=0,
        auto_reset=False,
    )
    actor = ExactReferenceJointTarget(env)
    records: list[dict] = []
    for begin in range(0, len(args.clips), args.num_envs):
        motion_ids = list(range(begin, min(begin + args.num_envs, len(args.clips))))
        records.extend(evaluate_batch(env, actor, motion_ids))

    lifted = [record["box_peak_lift_m"] >= args.minimum_lift for record in records]
    strict = [
        record["trajectory_timeout"] and not record["strict_failure"]
        for record in records
    ]
    finite = [record["all_finite"] for record in records]
    needed = math.ceil(args.minimum_pass_fraction * len(records))
    checks = {
        "action_is_exact_inverse_of_deployed_joint_target_map": (
            actor.maximum_reconstruction_delta <= 1.0e-6
        ),
        "actions_are_unclipped_and_finite": actor.all_actions_finite,
        "all_profiles_finished": all(record["finished"] for record in records),
        "no_active_profile_diverged": all(finite)
        and not any(
            "diverged" in record["strict_failure_reasons"] for record in records
        ),
        "physical_lift_fraction_passes": sum(lifted) >= needed,
        "strict_completion_fraction_passes": sum(strict) >= needed,
    }
    result = {
        "protocol": "sugar_newton_exact_reference_joint_target_ceiling_v1",
        "controller": "parameter_free_exact_current_reference_joint_target",
        "actor_observation_augmented": False,
        "future_or_outcome_actor_input": False,
        "action_clipping": False,
        "motion_root": str(args.motion_root.resolve()),
        "num_profiles": len(records),
        "minimum_lift_m": args.minimum_lift,
        "minimum_pass_fraction": args.minimum_pass_fraction,
        "required_profile_count": needed,
        "action_calls_including_padded_worlds": actor.calls,
        "maximum_joint_target_reconstruction_delta": (
            actor.maximum_reconstruction_delta
        ),
        "lifted_profile_count": sum(lifted),
        "strict_complete_profile_count": sum(strict),
        "finite_profile_count": sum(finite),
        "mean_peak_lift_m": sum(record["box_peak_lift_m"] for record in records)
        / len(records),
        "mean_bilateral_contact_fraction": sum(
            record["bilateral_contact_fraction"] for record in records
        )
        / len(records),
        "maximum_abs_action": max(record["max_abs_action"] for record in records),
        "checks": checks,
        "passed": all(checks.values()),
        "profiles": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "profiles"}, indent=2))
    print(f"wrote {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
