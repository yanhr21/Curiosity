# SPDX-License-Identifier: BSD-3-Clause
"""Frozen physical gate for the causal official-Refiner action-chunk controller."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl import obs_890
from sugar_newton.rl.carrybox_env import CarryBoxEnv
from sugar_newton.rl.train_bcppo import activate_rsl_rl
from sugar_newton.validation.refiner_open_loop import (
    DEFAULT_CLIPS,
    evaluate_batch,
    load_official_teacher,
    sha256,
)


class FrozenActionChunkRefiner(torch.nn.Module):
    """Checkpoint plan actor composed with the exact released Refiner."""

    KNOT_COUNT = 7
    STEPS_PER_KNOT = 5
    PREFIX_STEPS = 200
    CHUNK_STEPS = KNOT_COUNT * STEPS_PER_KNOT
    ACTION_DIM = KNOT_COUNT * 29

    def __init__(
        self,
        env: CarryBoxEnv,
        checkpoint: Path,
        official_checkpoint: Path,
        device: torch.device,
    ) -> None:
        super().__init__()
        from sugar_rl.utils.frozen_expert_transition_actor_critic import (
            RefinerActionChunkCausalTemporalActor,
        )

        self.env = env
        self.plan_actor = RefinerActionChunkCausalTemporalActor().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state = payload["model_state_dict"]
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in state.items()
            if key.startswith("actor.")
        }
        self.plan_actor.load_state_dict(actor_state, strict=True)
        self.official_refiner, hidden_dims = load_official_teacher(
            official_checkpoint, device
        )
        if hidden_dims != [512, 256, 128]:
            raise RuntimeError(f"official Refiner geometry drifted: {hidden_dims}")
        self.eval().requires_grad_(False)
        self.latched_knots = torch.zeros(
            env.num_envs, self.KNOT_COUNT, 29, device=device
        )
        self.plan_latched = torch.zeros(
            env.num_envs, dtype=torch.bool, device=device
        )
        self.plan_latches = 0
        self.action_calls = 0
        self.chunk_composition_calls = 0
        self.nonzero_pre_handoff_correction_calls = 0
        self.nonzero_post_chunk_correction_calls = 0
        self.maximum_abs_correction = 0.0
        self.maximum_raw_plan = 0.0

    def composition_terms(self, actor_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if tuple(actor_input.shape) != (self.env.num_envs, 9790):
            raise ValueError(
                f"action-chunk actor input is {tuple(actor_input.shape)}, expected "
                f"{(self.env.num_envs, 9790)}"
            )
        raw_plan = self.plan_actor(actor_input)
        if tuple(raw_plan.shape) != (self.env.num_envs, self.ACTION_DIM):
            raise RuntimeError(f"action-chunk plan shape drifted: {tuple(raw_plan.shape)}")
        if not torch.isfinite(raw_plan).all():
            raise RuntimeError("frozen action-chunk plan is non-finite")

        physical_t = self.env.t.detach().clone()
        reset = physical_t == 0
        self.latched_knots[reset] = 0.0
        self.plan_latched[reset] = False

        plan_decision = (physical_t == self.PREFIX_STEPS) & ~self.plan_latched
        if bool(plan_decision.any()):
            knots = torch.tanh(raw_plan).reshape(
                self.env.num_envs, self.KNOT_COUNT, 29
            )
            self.latched_knots[plan_decision] = knots[plan_decision]
            self.plan_latched[plan_decision] = True
            self.plan_latches += int(plan_decision.sum().item())

        chunk_active = (
            (physical_t >= self.PREFIX_STEPS)
            & (physical_t < self.PREFIX_STEPS + self.CHUNK_STEPS)
        )
        if bool((chunk_active & ~self.plan_latched).any()):
            raise RuntimeError("frozen action-chunk execution has no latched plan")
        knot_index = torch.div(
            (physical_t - self.PREFIX_STEPS).clamp(min=0),
            self.STEPS_PER_KNOT,
            rounding_mode="floor",
        ).clamp(max=self.KNOT_COUNT - 1)
        correction = torch.zeros(self.env.num_envs, 29, device=actor_input.device)
        world = torch.arange(self.env.num_envs, device=actor_input.device)
        correction[chunk_active] = self.latched_knots[
            world[chunk_active], knot_index[chunk_active]
        ]

        physical_obs = actor_input[:, :obs_890.OBS_DIM_890]
        official_action = self.official_refiner(physical_obs)
        composed_action = official_action + correction
        if not torch.isfinite(composed_action).all():
            raise RuntimeError("frozen action-chunk composed action is non-finite")

        nonzero = correction.abs().amax(dim=1) > 0
        self.nonzero_pre_handoff_correction_calls += int(
            (nonzero & (physical_t < self.PREFIX_STEPS)).sum().item()
        )
        self.nonzero_post_chunk_correction_calls += int(
            (nonzero & (physical_t >= self.PREFIX_STEPS + self.CHUNK_STEPS)).sum().item()
        )
        self.action_calls += int(physical_t.numel())
        self.chunk_composition_calls += int(chunk_active.sum().item())
        self.maximum_abs_correction = max(
            self.maximum_abs_correction, float(correction.abs().max().item())
        )
        self.maximum_raw_plan = max(
            self.maximum_raw_plan, float(raw_plan.abs().max().item())
        )
        return {
            "expert_retention": torch.ones_like(raw_plan[:, :1]),
            "selected_endpoint_action": official_action,
            "bounded_residual_action": correction,
            "composed_action": composed_action,
            "raw_plan_action": raw_plan,
            "chunk_active": chunk_active[:, None],
        }

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        return self.composition_terms(actor_input)["composed_action"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-refiner", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, default=Path("SUGAR/data/CarryBox"))
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=181748)
    parser.add_argument("--minimum-lift", type=float, default=0.05)
    parser.add_argument("--minimum-pass-fraction", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rsl-rl-root", required=True)
    args = parser.parse_args()

    if not 1 <= args.num_envs <= 8 or not args.clips:
        raise SystemExit("frozen action-chunk gate requires 1..8 worlds and at least one clip")
    missing = [clip for clip in args.clips if not (args.motion_root / clip).is_dir()]
    if missing:
        raise SystemExit(f"missing source clips: {missing}")
    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("frozen action-chunk gate must run on a Slurm CUDA compute node")
    activate_rsl_rl(args.rsl_rl_root)
    device = torch.device(args.device)
    env = CarryBoxEnv(
        num_envs=args.num_envs,
        clip_names=list(args.clips),
        motion_root=args.motion_root,
        teacher_motion_root=args.motion_root,
        episode_length=10**9,
        substeps=4,
        mu=1.0,
        device=args.device,
        seed=args.seed,
        auto_reset=False,
    )
    actor = FrozenActionChunkRefiner(
        env, args.checkpoint, args.official_refiner, device
    )
    records: list[dict] = []
    batch_count = 0
    for begin in range(0, len(args.clips), args.num_envs):
        motion_ids = list(range(begin, min(begin + args.num_envs, len(args.clips))))
        records.extend(
            evaluate_batch(env, actor, motion_ids, temporal_history_steps=10)
        )
        batch_count += 1

    lifted = [record["box_peak_lift_m"] >= args.minimum_lift for record in records]
    strict = [
        record["trajectory_timeout"] and not record["strict_failure"]
        for record in records
    ]
    finite = [record["all_finite"] for record in records]
    needed = math.ceil(args.minimum_pass_fraction * len(records))
    expected_latches = batch_count * args.num_envs
    checks = {
        "checkpoint_actor_is_frozen": not any(
            parameter.requires_grad for parameter in actor.parameters()
        ),
        "one_plan_latched_per_world_per_batch": actor.plan_latches == expected_latches,
        "chunk_was_executed": actor.chunk_composition_calls > 0,
        "official_refiner_is_exact_before_handoff": (
            actor.nonzero_pre_handoff_correction_calls == 0
        ),
        "official_refiner_is_exact_after_chunk": (
            actor.nonzero_post_chunk_correction_calls == 0
        ),
        "correction_respects_tanh_bound": actor.maximum_abs_correction <= 1.0,
        "all_profiles_finished": all(record["finished"] for record in records),
        "no_active_profile_diverged": all(finite)
        and not any("diverged" in record["strict_failure_reasons"] for record in records),
        "physical_lift_fraction_passes": sum(lifted) >= needed,
        "strict_completion_fraction_passes": sum(strict) >= needed,
    }
    result = {
        "protocol": "sugar_newton_refiner_action_chunk_frozen_v1",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "official_refiner": str(args.official_refiner.resolve()),
        "official_refiner_sha256": sha256(args.official_refiner),
        "num_profiles": len(records),
        "required_profile_count": needed,
        "lifted_profile_count": sum(lifted),
        "strict_complete_profile_count": sum(strict),
        "finite_profile_count": sum(finite),
        "mean_peak_lift_m": sum(record["box_peak_lift_m"] for record in records)
        / len(records),
        "mean_bilateral_contact_fraction": sum(
            record["bilateral_contact_fraction"] for record in records
        )
        / len(records),
        "action_chunk_knot_count": actor.KNOT_COUNT,
        "action_chunk_steps_per_knot": actor.STEPS_PER_KNOT,
        "action_chunk_horizon": actor.CHUNK_STEPS,
        "action_chunk_plan_latches": actor.plan_latches,
        "expected_plan_latches_including_padding": expected_latches,
        "chunk_composition_calls_including_audit_replay": actor.chunk_composition_calls,
        "total_action_calls_including_padding_and_audit_replay": actor.action_calls,
        "maximum_abs_correction": actor.maximum_abs_correction,
        "maximum_abs_raw_plan": actor.maximum_raw_plan,
        "future_or_outcome_actor_input": False,
        "physical_clock_modified": False,
        "checks": checks,
        "passed": all(checks.values()),
        "profiles": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "profiles"},
            indent=2,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
