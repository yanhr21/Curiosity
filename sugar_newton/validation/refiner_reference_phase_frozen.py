# SPDX-License-Identifier: BSD-3-Clause
"""Frozen physical gate for the causal official-Refiner phase controller."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl import obs_890
from sugar_newton.rl.carrybox_env import CarryBoxEnv
from sugar_newton.rl.reference_phase import bounded_phase_offset, policy_reference_phase
from sugar_newton.rl.train_bcppo import activate_rsl_rl
from sugar_newton.validation.refiner_open_loop import (
    DEFAULT_CLIPS,
    evaluate_batch,
    load_official_teacher,
    sha256,
)


class FrozenReferencePhaseRefiner(torch.nn.Module):
    """Checkpoint phase actor composed with the exact released Refiner."""

    def __init__(
        self,
        env: CarryBoxEnv,
        checkpoint: Path,
        official_checkpoint: Path,
        device: torch.device,
    ) -> None:
        super().__init__()
        from sugar_rl.utils.frozen_expert_transition_actor_critic import (
            RefinerReferencePhaseCausalTemporalActor,
        )

        self.env = env
        self.phase_actor = RefinerReferencePhaseCausalTemporalActor().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state = payload["model_state_dict"]
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in state.items()
            if key.startswith("actor.")
        }
        self.phase_actor.load_state_dict(actor_state, strict=True)
        self.official_refiner, hidden_dims = load_official_teacher(
            official_checkpoint, device
        )
        if hidden_dims != [512, 256, 128]:
            raise RuntimeError(f"official Refiner geometry drifted: {hidden_dims}")
        self.eval().requires_grad_(False)
        self.maximum_abs_offset = 0
        self.nonzero_prefix_offsets = 0
        self.retimed_calls = 0
        self.action_calls = 0
        self.maximum_phase_action_delta = 0.0

    def composition_terms(self, actor_input: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_phase = self.phase_actor(actor_input)
        physical_t = self.env.t.detach().clone()
        reference_t = policy_reference_phase(raw_phase, physical_t)
        offset = reference_t - physical_t
        retimed_obs = obs_890.build(
            self.env, teacher=False, reference_t=reference_t
        )
        physical_obs = actor_input[:, :obs_890.OBS_DIM_890]
        retimed_action = self.official_refiner(retimed_obs)
        physical_action = self.official_refiner(physical_obs)
        delta = retimed_action - physical_action
        self.maximum_abs_offset = max(
            self.maximum_abs_offset, int(offset.abs().max().item())
        )
        self.nonzero_prefix_offsets += int(
            ((physical_t <= 200) & (offset != 0)).sum().item()
        )
        self.retimed_calls += int((offset != 0).sum().item())
        self.action_calls += int(offset.numel())
        self.maximum_phase_action_delta = max(
            self.maximum_phase_action_delta, float(delta.abs().max().item())
        )
        return {
            "expert_retention": torch.ones_like(raw_phase),
            "selected_endpoint_action": physical_action,
            "bounded_residual_action": delta,
            "composed_action": retimed_action,
            "raw_phase_action": raw_phase,
            "reference_phase_offset": offset,
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
    parser.add_argument("--seed", type=int, default=181746)
    parser.add_argument("--minimum-lift", type=float, default=0.05)
    parser.add_argument("--minimum-pass-fraction", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rsl-rl-root", required=True)
    args = parser.parse_args()

    if not 1 <= args.num_envs <= 8 or not args.clips:
        raise SystemExit("frozen phase gate requires 1..8 worlds and at least one clip")
    missing = [clip for clip in args.clips if not (args.motion_root / clip).is_dir()]
    if missing:
        raise SystemExit(f"missing source clips: {missing}")
    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("frozen phase gate must run on a Slurm CUDA compute node")
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
    actor = FrozenReferencePhaseRefiner(
        env, args.checkpoint, args.official_refiner, device
    )
    records: list[dict] = []
    for begin in range(0, len(args.clips), args.num_envs):
        motion_ids = list(range(begin, min(begin + args.num_envs, len(args.clips))))
        records.extend(
            evaluate_batch(env, actor, motion_ids, temporal_history_steps=10)
        )

    lifted = [record["box_peak_lift_m"] >= args.minimum_lift for record in records]
    strict = [
        record["trajectory_timeout"] and not record["strict_failure"]
        for record in records
    ]
    finite = [record["all_finite"] for record in records]
    needed = math.ceil(args.minimum_pass_fraction * len(records))
    endpoint_offset = bounded_phase_offset(
        torch.tensor([-100.0, 100.0], device=device),
        torch.tensor([200, 235], dtype=env.t.dtype, device=device),
    )
    checks = {
        "checkpoint_actor_is_frozen": not any(
            parameter.requires_grad for parameter in actor.parameters()
        ),
        "phase_offset_is_zero_through_prefix": actor.nonzero_prefix_offsets == 0,
        "phase_offset_is_zero_at_fixed_endpoints": bool((endpoint_offset == 0).all()),
        "phase_offset_respects_fixed_bound": actor.maximum_abs_offset <= 7,
        "all_profiles_finished": all(record["finished"] for record in records),
        "no_active_profile_diverged": all(finite)
        and not any("diverged" in record["strict_failure_reasons"] for record in records),
        "physical_lift_fraction_passes": sum(lifted) >= needed,
        "strict_completion_fraction_passes": sum(strict) >= needed,
    }
    result = {
        "protocol": "sugar_newton_refiner_reference_phase_frozen_v1",
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
        "mean_peak_lift_m": sum(record["box_peak_lift_m"] for record in records) / len(records),
        "mean_bilateral_contact_fraction": sum(
            record["bilateral_contact_fraction"] for record in records
        ) / len(records),
        "maximum_abs_phase_offset": actor.maximum_abs_offset,
        "retimed_action_calls": actor.retimed_calls,
        "total_action_calls_including_padding": actor.action_calls,
        "maximum_phase_induced_action_delta": actor.maximum_phase_action_delta,
        "future_or_outcome_actor_input": False,
        "physical_clock_modified": False,
        "checks": checks,
        "passed": all(checks.values()),
        "profiles": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "profiles"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
