# SPDX-License-Identifier: BSD-3-Clause
"""Adapt the exact official SUGAR CarryBox Refiner in Newton with official PPO.

This is the declared response to a failed frozen-Refiner transfer gate.  It preserves the
released serious method: the official 890-D privileged observation, 29-D joint-position
action, ``512/256/128`` ActorCritic and SUGAR PPO implementation.  Newton transfer uses
the repository's already-admitted recovery stabilization (`1e-5` learning rate, `0.05`
action standard deviation, reward clipping at 10 and synchronized divergence reset)
instead of silently continuing the unstable Isaac exploration dynamics.
The accepted checkpoint is loaded strictly as a weight initialization, while optimizer
state and the learning-iteration counter start fresh.  It is therefore a Newton transfer
training run, not a resume and not a replacement model.

The script requires CUDA and is intended only for the foreground ``srun`` shell in the
allocated H200 tmux window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl.train_bcppo import activate_rsl_rl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INITIAL = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts"
    / "refiner_model10000.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runner_cfg(args: argparse.Namespace) -> dict:
    """Official PPO geometry with explicit Newton-transfer stabilization."""

    return {
        "num_steps_per_env": 24,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "experiment_name": args.run_name,
        "empirical_normalization": False,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": args.learning_rate,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "logger": "tensorboard",
    }


def load_initial_weights(runner, checkpoint: Path) -> dict:
    """Strictly load official weights without restoring optimizer/iteration state."""

    payload = torch.load(checkpoint, map_location=runner.device, weights_only=False)
    state = payload["model_state_dict"]
    runner.alg.policy.load_state_dict(state, strict=True)
    loaded = runner.alg.policy.state_dict()
    changed = [key for key, value in state.items() if not torch.equal(loaded[key], value)]
    if changed:
        raise RuntimeError(f"strict Refiner initialization changed keys: {changed[:8]}")
    actor_weights = [
        value for key, value in state.items()
        if key.startswith("actor.") and key.endswith("weight")
    ]
    geometry = [(int(value.shape[1]), int(value.shape[0])) for value in actor_weights]
    expected = [(890, 512), (512, 256), (256, 128), (128, 29)]
    if geometry != expected:
        raise RuntimeError(f"official Refiner actor geometry drift: {geometry} != {expected}")
    return {
        "source_checkpoint": str(checkpoint.resolve()),
        "source_sha256": sha256(checkpoint),
        "source_iteration": int(payload.get("iter", -1)),
        "strict_model_state_load": True,
        "actor_geometry": geometry,
        "optimizer_state_loaded": False,
        "learning_iteration_loaded": False,
        "newton_learning_iteration": runner.current_learning_iteration,
        "parameter_count": sum(parameter.numel() for parameter in runner.alg.policy.parameters()),
    }


def run_zero_optimizer_diagnostic(runner, env, *, horizons: int, log_dir: Path) -> dict:
    """Exercise the exact stochastic actor without taking an optimizer step."""

    if horizons < 1:
        raise ValueError("diagnostic horizons must be positive")
    policy = runner.alg.policy
    frozen = {
        key: value.detach().clone()
        for key, value in policy.state_dict().items()
        if key != "std"
    }
    obs, _ = env.reset()
    divergence_by_motion = torch.zeros(
        env.env.num_motions, dtype=torch.long, device=env.device
    )
    divergence_total = 0
    done_total = 0
    finite = True
    for _ in range(horizons * 24):
        motion_before = env.env.motion_id.clone()
        with torch.no_grad():
            actions = policy.act(obs)
        obs, reward, done, _ = env.step(actions)
        diverged = env.env.termination_terms["diverged"]
        if bool(diverged.any()):
            ids = motion_before[diverged]
            divergence_by_motion.scatter_add_(
                0, ids, torch.ones_like(ids, dtype=torch.long)
            )
        divergence_total += int(diverged.sum().item())
        done_total += int(done.sum().item())
        finite &= bool(
            torch.isfinite(actions).all()
            and torch.isfinite(reward).all()
            and torch.isfinite(obs["policy"]).all()
        )
    current = policy.state_dict()
    parameter_max_delta = max(
        (current[key] - value).abs().max().item() for key, value in frozen.items()
    )
    nonzero = torch.nonzero(divergence_by_motion, as_tuple=False).flatten()
    result = {
        "protocol": "sugar_newton_refiner_zero_optimizer_diagnostic_v1",
        "optimizer_steps": 0,
        "horizons": horizons,
        "steps_per_horizon": 24,
        "num_envs": env.num_envs,
        "transitions": horizons * 24 * env.num_envs,
        "all_returned_tensors_finite": finite,
        "done_total": done_total,
        "divergence_total": divergence_total,
        "divergence_by_motion": {
            env.env.clip_names[index]: int(divergence_by_motion[index].item())
            for index in nonzero.tolist()
        },
        "actor_critic_parameter_max_delta": parameter_max_delta,
        "pass": finite and divergence_total == 0 and parameter_max_delta == 0.0,
    }
    (log_dir / "ZERO_OPTIMIZER_AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, default=ROOT / "SUGAR/data/CarryBox")
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=64)
    parser.add_argument("--save-interval", type=int, default=32)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--action-std", type=float, default=0.05)
    parser.add_argument("--reward-clip", type=float, default=10.0)
    parser.add_argument("--frame-zero-env-count", type=int, default=0)
    parser.add_argument("--zero-optimizer-diagnostic-horizons", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=171701)
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--run-name", default="carrybox_refiner_newton_seed171701")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=ROOT / "experiments/sugar_reproduction/outputs/newton_refiner_ppo_20260827",
    )
    parser.add_argument("--rsl-rl-root", required=True)
    args = parser.parse_args()

    if not args.initial_checkpoint.is_file():
        raise SystemExit(f"initial checkpoint not found: {args.initial_checkpoint}")
    if not args.motion_root.is_dir() or not any(args.motion_root.glob("data_*")):
        raise SystemExit(f"raw CarryBox motion root is invalid: {args.motion_root}")
    if not 1 <= args.num_envs <= 8:
        raise SystemExit("--num-envs must remain in the validated Newton range 1..8")
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be positive")
    if not 0.0 < args.learning_rate <= 1.0e-3:
        raise SystemExit("--learning-rate must be in (0, 1e-3]")
    if not 0.0 < args.action_std <= 1.0:
        raise SystemExit("--action-std must be in (0, 1]")
    if args.reward_clip <= 0.0:
        raise SystemExit("--reward-clip must be positive")
    if not 0 <= args.frame_zero_env_count <= args.num_envs:
        raise SystemExit("--frame-zero-env-count must be in [0, num-envs]")

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("Newton Refiner training must run on a Slurm CUDA compute node")
    activate_rsl_rl(args.rsl_rl_root)
    from rsl_rl.runners import OnPolicyRunner

    from sugar_newton.rl.vec_env import make_refiner

    torch.manual_seed(args.seed)
    env = make_refiner(
        args.num_envs,
        clip_names=args.clips,
        motion_root=args.motion_root,
        teacher_motion_root=args.motion_root,
        episode_length=args.episode_length,
        substeps=args.substeps,
        mu=args.mu,
        reward_clip=args.reward_clip,
        frame_zero_env_count=args.frame_zero_env_count,
        sync_divergence_reset=True,
        device=args.device,
        seed=args.seed,
    )
    log_dir = args.log_root / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(
        env,
        runner_cfg(args),
        log_dir=str(log_dir),
        device=args.device,
    )
    audit = load_initial_weights(runner, args.initial_checkpoint)
    source_action_std = float(runner.alg.policy.std.detach().mean().item())
    audit.update(
        {
            "protocol": "sugar_newton_official_refiner_ppo_transfer_v2",
            "seed": args.seed,
            "num_envs": args.num_envs,
            "num_motions": len(env.env.clip_names),
            "observation_dim": 890,
            "action_dim": 29,
            "max_iterations": args.max_iterations,
            "fresh_optimizer": True,
            "source_action_std_mean": source_action_std,
            "training_action_std": args.action_std,
            "learning_rate": args.learning_rate,
            "reward_clip": args.reward_clip,
            "sync_divergence_reset": True,
            "frame_zero_env_count": env.env.frame_zero_env_count,
            "solver_njmax_per_world": env.env.njmax,
            "solver_nconmax_per_world": env.env.nconmax,
        }
    )
    (log_dir / "INITIALIZATION_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    # ``OnPolicyRunner.save`` accesses ``logger_type``, which rsl_rl initializes only
    # inside ``learn``.  Save the exact same payload directly for this pre-update
    # endpoint so creating the frozen baseline cannot mutate or prematurely initialize
    # the logging/training state.
    torch.save(
        {
            "model_state_dict": runner.alg.policy.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "iter": 0,
            "infos": {"initialization_audit": audit},
        },
        log_dir / "model_pre_update.pt",
    )
    # Preserve the exact source distribution in model_pre_update.pt, then make the
    # declared training-only exploration change.  Actor and critic weights remain
    # bitwise equal to the official source at the first Newton transition.
    with torch.no_grad():
        runner.alg.policy.std.fill_(args.action_std)
    pre_training_state = {
        key: value.detach().clone()
        for key, value in runner.alg.policy.state_dict().items()
        if key != "std"
    }
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.zero_optimizer_diagnostic_horizons:
        run_zero_optimizer_diagnostic(
            runner,
            env,
            horizons=args.zero_optimizer_diagnostic_horizons,
            log_dir=log_dir,
        )
        return
    print(f"[train] official Refiner Newton transfer: {args.max_iterations} fresh PPO updates")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    final_state = runner.alg.policy.state_dict()
    parameter_max_delta = max(
        (final_state[key] - value).abs().max().item()
        for key, value in pre_training_state.items()
    )
    all_policy_parameters_finite = all(
        bool(torch.isfinite(value).all()) for value in final_state.values()
    )
    transitions = args.max_iterations * 24 * args.num_envs
    divergence_total = int(env.env.num_diverged)
    divergence_rate = divergence_total / transitions
    result = {
        "protocol": "sugar_newton_refiner_ppo_training_gate_v1",
        "optimizer_updates": args.max_iterations,
        "transitions": transitions,
        "divergence_total": divergence_total,
        "divergence_rate": divergence_rate,
        "maximum_admitted_training_divergence_rate": 0.005,
        "all_policy_parameters_finite": all_policy_parameters_finite,
        "actor_critic_parameter_max_delta": parameter_max_delta,
        "final_action_std_mean": float(runner.alg.policy.std.detach().mean().item()),
        "pass": all_policy_parameters_finite and divergence_rate <= 0.005,
        "frozen_evaluation_required": True,
    }
    (log_dir / "TRAINING_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
