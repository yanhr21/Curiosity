"""Zero-optimizer audit of causal reference-phase retiming semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl import obs_890
from sugar_newton.rl.carrybox_env import CarryBoxEnv
from sugar_newton.rl.reference_phase import (
    bounded_phase_offset,
    policy_reference_phase,
)
from sugar_newton.rl.train_bcppo import activate_rsl_rl
from sugar_newton.validation.refiner_open_loop import (
    load_official_teacher,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFINER = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts"
    / "refiner_model10000.pt"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=171744)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--motion-root", type=Path, default=ROOT / "SUGAR/data/CarryBox"
    )
    parser.add_argument("--refiner-checkpoint", type=Path, default=DEFAULT_REFINER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rsl-rl-root", type=Path, required=True)
    args = parser.parse_args()

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("reference-phase audit must run on a Slurm CUDA node")
    activate_rsl_rl(args.rsl_rl_root)
    device = torch.device(args.device)
    teacher, hidden_dims = load_official_teacher(args.refiner_checkpoint, device)
    if hidden_dims != [512, 256, 128]:
        raise RuntimeError(f"official Refiner geometry drifted: {hidden_dims}")
    teacher_before = {
        key: value.detach().clone() for key, value in teacher.state_dict().items()
    }

    num_envs = 8
    env = CarryBoxEnv(
        num_envs=num_envs,
        clip_names=["data_000"],
        motion_root=args.motion_root,
        teacher_motion_root=args.motion_root,
        episode_length=10**9,
        substeps=4,
        mu=1.0,
        device=args.device,
        seed=args.seed,
        auto_reset=False,
    )
    ids = torch.arange(num_envs, device=device)
    env.reset(
        ids,
        motion_ids=torch.zeros(num_envs, device=device),
        start_frames=0,
    )
    env.t.fill_(217)

    physical_before = {
        "q": env.q.detach().clone(),
        "qd": env.qd.detach().clone(),
        "body_q": env._body_q().detach().clone(),
        "body_qd": env._body_qd().detach().clone(),
        "t": env.t.detach().clone(),
        "reference_obj": env._ref("obj_pos").detach().clone(),
    }
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device).clone()

    default_obs = obs_890.build(env, teacher=True)
    explicit_zero_obs = obs_890.build(env, teacher=True, reference_t=env.t)
    with torch.inference_mode():
        default_action = teacher(default_obs)
        explicit_zero_action = teacher(explicit_zero_obs)

    raw_phase = torch.linspace(-3.0, 3.0, num_envs, device=device)
    offset = bounded_phase_offset(raw_phase, env.t)
    reference_t = policy_reference_phase(raw_phase, env.t)
    retimed_obs = obs_890.build(env, teacher=True, reference_t=reference_t)
    with torch.inference_mode():
        retimed_action = teacher(retimed_obs)

    reference_dim = sum(dim for _, dim in obs_890.LAYOUT[:7])
    prefix_delta = float(
        (retimed_obs[:, :reference_dim] - default_obs[:, :reference_dim])
        .abs()
        .max()
        .item()
    )
    physical_suffix_delta = float(
        (retimed_obs[:, reference_dim:] - default_obs[:, reference_dim:])
        .abs()
        .max()
        .item()
    )
    physical_delta = max(
        float((env.q - physical_before["q"]).abs().max().item()),
        float((env.qd - physical_before["qd"]).abs().max().item()),
        float((env._body_q() - physical_before["body_q"]).abs().max().item()),
        float((env._body_qd() - physical_before["body_qd"]).abs().max().item()),
        float((env.t - physical_before["t"]).abs().max().item()),
        float(
            (env._ref("obj_pos") - physical_before["reference_obj"])
            .abs()
            .max()
            .item()
        ),
    )
    endpoint_clock = torch.tensor(
        [200, 235], dtype=env.t.dtype, device=device
    )
    endpoint_offset = bounded_phase_offset(
        torch.tensor([-10.0, 10.0], device=device), endpoint_clock
    )
    teacher_delta = max(
        float((value - teacher_before[key]).abs().max().item())
        for key, value in teacher.state_dict().items()
    )
    cpu_rng_delta = int(
        (torch.get_rng_state() != cpu_rng_before).sum().item()
    )
    cuda_rng_delta = int(
        (torch.cuda.get_rng_state(device) != cuda_rng_before).sum().item()
    )
    checks = {
        "official_refiner_parameter_exact": teacher_delta == 0.0,
        "default_and_explicit_physical_phase_observation_are_bitwise_equal": torch.equal(
            default_obs, explicit_zero_obs
        ),
        "default_and_explicit_physical_phase_action_are_bitwise_equal": torch.equal(
            default_action, explicit_zero_action
        ),
        "nonzero_phase_offset_exists": bool((offset != 0).any()),
        "retiming_changes_only_reference_features": (
            prefix_delta > 0.0 and physical_suffix_delta == 0.0
        ),
        "retiming_changes_frozen_refiner_action": not torch.equal(
            default_action, retimed_action
        ),
        "physical_state_and_original_clock_reference_are_exact": physical_delta == 0.0,
        "phase_offset_is_exactly_zero_at_fixed_endpoints": bool(
            (endpoint_offset == 0).all()
        ),
        "cpu_rng_is_unchanged": cpu_rng_delta == 0,
        "cuda_rng_is_unchanged": cuda_rng_delta == 0,
    }
    result = {
        "protocol": "sugar_newton_refiner_reference_phase_component_v1",
        "seed": args.seed,
        "refiner_checkpoint": str(args.refiner_checkpoint.resolve()),
        "refiner_sha256": sha256(args.refiner_checkpoint),
        "reference_feature_dim": reference_dim,
        "physical_feature_dim": obs_890.OBS_DIM_890 - reference_dim,
        "maximum_offset_frames": 7,
        "handoff_step": 200,
        "endpoint_step": 235,
        "observed_offsets": offset.detach().cpu().tolist(),
        "maximum_reference_feature_delta": prefix_delta,
        "maximum_physical_feature_delta": physical_suffix_delta,
        "maximum_physical_state_delta": physical_delta,
        "maximum_action_delta": float(
            (retimed_action - default_action).abs().max().item()
        ),
        "teacher_parameter_max_delta": teacher_delta,
        "cpu_rng_byte_delta_count": cpu_rng_delta,
        "cuda_rng_byte_delta_count": cuda_rng_delta,
        "future_rollout_or_outcome_labels_used": False,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
