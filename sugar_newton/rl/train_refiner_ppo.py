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
import builtins
import hashlib
import json
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl.train_bcppo import activate_rsl_rl, sugar_bcppo


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INITIAL = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts"
    / "refiner_model10000.pt"
)
DEFAULT_TRACKER_TEACHER = ROOT / "SUGAR/demo_ckpts/CarryBox/tracker.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runner_cfg(args: argparse.Namespace) -> dict:
    """Official PPO geometry with explicit Newton-transfer stabilization."""

    algorithm = {
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
    }
    temporal_mode = (
        args.frozen_expert_temporal_composer
        or args.frozen_expert_temporal_additive_residual
        or args.frozen_expert_temporal_command_additive_residual
    )
    temporal_command_mode = (
        args.frozen_expert_temporal_command_additive_residual
    )
    reference_phase_mode = args.reference_phase_retiming
    action_chunk_mode = args.action_chunk_recovery
    embedded_expert = (
        args.frozen_expert_residual
        or temporal_mode
        or args.released_tracker_action_supervision
        or reference_phase_mode
        or action_chunk_mode
    )
    anchor_enabled = args.official_action_anchor or embedded_expert
    if anchor_enabled:
        free_recovery = args.newton_native_free_recovery
        algorithm.update(
            {
                "class_name": "BCPPO",
                  **(
                      {}
                    if embedded_expert
                      else {"teacher_ckpt": str(args.initial_checkpoint)}
                  ),
                "stage3_distill_weight_floor": 0.0 if free_recovery else 1.0,
                "training_mask_obs_group": None,
                "distill_mask_start_step": (
                    0
                    if free_recovery
                    else (
                        args.max_iterations + 1
                        if args.failure_frontier_prefix_steps
                        else 0
                    )
                ),
                "bc_only_steps": (
                    args.max_iterations + 1
                    if args.pure_distill
                    else 0
                ),
                "critic_warmup_steps": (
                    args.max_iterations + 2
                    if args.pure_distill
                    else 0
                ),
                "full_ppo_warmup_steps": (
                    args.max_iterations + 3
                    if args.pure_distill
                    else 1
                ),
                "teacher_mean_only": True,
                "minimum_action_std": args.action_std,
                # The student and teacher means are identical at initialization.
                # Adaptive KL therefore interprets the first near-zero KL batches as
                # permission to raise the learning rate as high as 1e-2 before the
                # live Newton distribution has moved.  Keep this transfer at the
                # declared 1e-5 rate; BCPPO's schedule flag is changed internally,
                # while ``desired_kl=None`` disables the adaptive update itself.
                "schedule": "fixed",
                "desired_kl": None,
            }
        )
        if args.failure_frontier_prefix_steps:
            algorithm["training_mask_obs_group"] = "training_handoff_mask"
    return {
        "num_steps_per_env": 24,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "experiment_name": args.run_name,
        "empirical_normalization": False,
          "obs_groups": {
            "policy": ["policy"],
            "critic": ["critic"],
              **({"teacher": ["teacher"]} if anchor_enabled else {}),
          },
          "policy": {
              "class_name": (
                    "RefinerActionChunkCausalTemporalActorCritic"
                    if action_chunk_mode
                    else "RefinerReferencePhaseCausalTemporalActorCritic"
                    if reference_phase_mode
                    else "FrozenOfficialRefinerTrackerSupervisedCausalTemporalComposerActorCritic"
                    if (
                        temporal_mode
                        and args.released_tracker_action_supervision
                    )
                    else "FrozenOfficialRefinerCausalTemporalComposerActorCritic"
                    if temporal_mode
                    else "FrozenOfficialRefinerTrackerSupervisedResidualActorCritic"
                    if args.released_tracker_action_supervision
                    else "FrozenOfficialRefinerResidualActorCritic"
                    if args.frozen_expert_residual
                    else "ActorCritic"
                ),
              "init_noise_std": 1.0,
              "actor_hidden_dims": [512, 256, 128],
              "critic_hidden_dims": [512, 256, 128],
              "activation": "elu",
              **(
                  {
                      "official_refiner_checkpoint": str(args.initial_checkpoint),
                      "transition_residual_limit": 1.0,
                      **(
                          {
                              "temporal_additive_residual": (
                                  args.frozen_expert_temporal_additive_residual
                                  or temporal_command_mode
                              ),
                              "temporal_command_conditioned": (
                                  temporal_command_mode
                              ),
                          }
                          if temporal_mode
                          else {}
                      ),
                      **(
                          {
                              "released_tracker_checkpoint": str(
                                  args.tracker_teacher_checkpoint
                              )
                          }
                          if args.released_tracker_action_supervision
                          else {}
                      ),
                  }
                    if embedded_expert and not (reference_phase_mode or action_chunk_mode)
                    else {}
                ),
        },
        "algorithm": algorithm,
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


def audit_official_action_anchor(runner, enabled: bool) -> dict:
    """Prove that BCPPO's behavior target is an exact frozen official Refiner."""

    if not enabled:
        return {"official_action_anchor": False}
    algorithm = runner.alg
    teacher = getattr(algorithm, "teacher_model", None)
    if teacher is None:
        raise RuntimeError("official-action anchor has no checkpoint teacher")
    student_actor = algorithm.policy.actor.state_dict()
    teacher_actor = teacher.state_dict()
    if student_actor.keys() != teacher_actor.keys():
        raise RuntimeError("student/teacher Refiner actor keys differ")
    maximum_delta = max(
        float((student_actor[key] - teacher_actor[key]).abs().max().item())
        for key in student_actor
    )
    teacher_frozen = all(not parameter.requires_grad for parameter in teacher.parameters())
    fixed_learning_rate = algorithm.desired_kl is None
    if maximum_delta != 0.0 or not teacher_frozen or not fixed_learning_rate:
        raise RuntimeError(
            "official action anchor drift: "
            f"delta={maximum_delta}, frozen={teacher_frozen}, "
            f"fixed_learning_rate={fixed_learning_rate}"
        )
    return {
        "official_action_anchor": True,
        "anchor_actor_parameter_max_delta_at_initialization": maximum_delta,
        "anchor_teacher_parameters_frozen": teacher_frozen,
        "anchor_fixed_learning_rate": fixed_learning_rate,
        "anchor_teacher_mean_only": bool(algorithm.teacher_mean_only),
        "anchor_stage3_distill_weight_floor": float(
            algorithm.stage3_distill_weight_floor
        ),
        "anchor_bc_only_steps": int(algorithm.bc_only_steps),
        "anchor_critic_warmup_steps": int(algorithm.critic_warmup_steps),
        "anchor_full_ppo_warmup_steps": int(algorithm.full_ppo_warmup_steps),
    }


def audit_frozen_expert_residual(runner, checkpoint: Path, enabled: bool) -> dict:
    """Prove exact frozen endpoint and zero-start official-scale adapter."""

    if not enabled:
        return {"frozen_expert_residual": False}
    policy = runner.alg.policy
    actor = policy.actor
    source = torch.load(checkpoint, map_location=runner.device, weights_only=False)[
        "model_state_dict"
    ]
    source_actor = {
        key.removeprefix("actor."): value
        for key, value in source.items()
        if key.startswith("actor.")
    }
    expert = actor.expert.state_dict()
    if expert.keys() != source_actor.keys():
        raise RuntimeError("frozen Refiner expert geometry differs from official source")
    maximum_delta = max(
        float((expert[key] - source_actor[key]).abs().max().item()) for key in expert
    )
    final = actor.residual[-1]
    zero_final = float(final.weight.abs().max().item()) == 0.0 and float(
        final.bias.abs().max().item()
    ) == 0.0
    expert_frozen = all(not parameter.requires_grad for parameter in actor.expert.parameters())
    if maximum_delta != 0.0 or not zero_final or not expert_frozen:
        raise RuntimeError(
            "frozen Refiner residual initialization drift: "
            f"expert_delta={maximum_delta}, zero_final={zero_final}, "
            f"expert_frozen={expert_frozen}"
        )
    return {
        "frozen_expert_residual": True,
        "frozen_expert_parameter_max_delta": maximum_delta,
        "frozen_expert_parameters_frozen": expert_frozen,
        "residual_output_layer_exact_zero": zero_final,
        "residual_limit": float(actor.residual_limit),
        "residual_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in actor.residual.parameters()
            if parameter.requires_grad
        ),
    }


def audit_released_tracker_action_supervision(
    runner, env, checkpoint: Path, enabled: bool
) -> dict:
    """Prove the current-state released Tracker teacher and frozen experts."""

    if not enabled:
        return {"released_tracker_action_supervision": False}
    policy = runner.alg.policy
    tracker = policy.tracker_teacher
    source = torch.load(checkpoint, map_location=runner.device, weights_only=False)[
        "model_state_dict"
    ]
    source_actor = {
        key.removeprefix("actor."): value
        for key, value in source.items()
        if key.startswith("actor.")
    }
    tracker_state = tracker.state_dict()
    if tracker_state.keys() != source_actor.keys():
        raise RuntimeError("released Tracker teacher geometry differs from source")
    weight_delta = max(
        float((tracker_state[key] - source_actor[key]).abs().max().item())
        for key in tracker_state
    )
    source_std = source.get("std")
    if source_std is None:
        log_std = source.get("log_std")
        if log_std is None:
            raise KeyError("released Tracker checkpoint is missing std/log_std")
        source_std = log_std.exp()
    std_delta = float(
        (policy.tracker_teacher_std - source_std.to(runner.device)).abs().max().item()
    )
    teacher_frozen = all(not parameter.requires_grad for parameter in tracker.parameters())
    q_before = env.env.q.clone()
    qd_before = env.env.qd.clone()
    live_obs = env.get_observations()
    state_read_delta = max(
        float((env.env.q - q_before).abs().max().item()),
        float((env.env.qd - qd_before).abs().max().item()),
    )
    with torch.no_grad():
        teacher_action, teacher_std = policy.distillation_teacher(live_obs)
        terms = policy.composition_audit_terms(live_obs)
    endpoint_delta = float(
        (terms["composed_action"] - terms["selected_endpoint_action"])
        .abs()
        .max()
        .item()
    )
    if (
        weight_delta != 0.0
        or std_delta != 0.0
        or not teacher_frozen
        or state_read_delta != 0.0
        or endpoint_delta != 0.0
    ):
        raise RuntimeError(
            "released Tracker supervision initialization drift: "
            f"weight={weight_delta}, std={std_delta}, frozen={teacher_frozen}, "
            f"state_read={state_read_delta}, endpoint={endpoint_delta}"
        )
    return {
        "released_tracker_action_supervision": True,
        "tracker_teacher_checkpoint": str(checkpoint.resolve()),
        "tracker_teacher_sha256": sha256(checkpoint),
        "tracker_teacher_weight_max_delta": weight_delta,
        "tracker_teacher_std_max_delta": std_delta,
        "tracker_teacher_parameters_frozen": teacher_frozen,
        "teacher_observation_dim": int(live_obs["teacher"].shape[-1]),
        "student_observation_dim": int(live_obs["policy"].shape[-1]),
        "same_state_observation_read_max_delta": state_read_delta,
        "initial_composed_endpoint_max_delta": endpoint_delta,
        "teacher_action_all_finite": bool(torch.isfinite(teacher_action).all()),
        "teacher_std_all_finite": bool(torch.isfinite(teacher_std).all()),
        "observation_epoch": int(env._observation_epoch),
    }


def audit_frozen_expert_temporal_composer(
    runner, env, checkpoint: Path, enabled: bool
) -> dict:
    """Prove exact endpoint and zero-start serious causal temporal composer."""

    if not enabled:
        return {"frozen_expert_temporal_composer": False}
    actor = runner.alg.policy.actor
    source = torch.load(checkpoint, map_location=runner.device, weights_only=False)[
        "model_state_dict"
    ]
    source_actor = {
        key.removeprefix("actor."): value
        for key, value in source.items()
        if key.startswith("actor.")
    }
    expert = actor.expert.state_dict()
    if expert.keys() != source_actor.keys():
        raise RuntimeError("temporal Refiner expert geometry differs from official source")
    weight_delta = max(
        float((expert[key] - source_actor[key]).abs().max().item()) for key in expert
    )
    if "std" in source:
        source_std = source["std"]
    elif "log_std" in source:
        source_std = source["log_std"].exp()
    else:
        raise KeyError("official Refiner checkpoint is missing std/log_std")
    std_delta = float((actor.expert_std - source_std).abs().max().item())
    final = actor.temporal_composer.output[-1]
    zero_final = float(final.weight.abs().max().item()) == 0.0 and float(
        final.bias.abs().max().item()
    ) == 0.0
    expert_frozen = all(not parameter.requires_grad for parameter in actor.expert.parameters())
    live_obs = env.get_observations()
    with torch.no_grad():
        terms = runner.alg.policy.composition_audit_terms(live_obs)
    current = live_obs["policy"][:, :890]
    history_last_delta = float(
        (terms["temporal_history_last_frame"] - current).abs().max().item()
    )
    endpoint_delta = float(
        (terms["composed_action"] - terms["selected_endpoint_action"])
        .abs()
        .max()
        .item()
    )
    retention_delta = float(
        (terms["expert_retention"] - 1.0).abs().max().item()
    )
    residual_max = float(terms["bounded_residual_action"].abs().max().item())
    command_conditioned = bool(actor.command_conditioned)
    command_dim = int(actor.temporal_composer.condition_dim)
    command_tracker_delta = 0.0
    command_term_delta = 0.0
    command_token_span = 0.0
    if command_conditioned:
        command = live_obs["policy"][:, -command_dim:]
        tracker_command = live_obs["teacher"][:, :command_dim]
        command_tracker_delta = float(
            (command - tracker_command).abs().max().item()
        )
        command_term_delta = float(
            (terms["current_reference_command"] - command).abs().max().item()
        )
        projected = actor.temporal_composer.condition_projection(command)
        command_token_span = float(
            (projected - projected[:1]).abs().max().item()
        )
    if weight_delta != 0.0 or std_delta != 0.0 or not zero_final or not expert_frozen:
        raise RuntimeError(
            "frozen Refiner temporal initialization drift: "
            f"weight_delta={weight_delta}, std_delta={std_delta}, "
            f"zero_final={zero_final}, expert_frozen={expert_frozen}"
        )
    if any(value != 0.0 for value in (
        history_last_delta,
        endpoint_delta,
        retention_delta,
        residual_max,
        command_tracker_delta,
        command_term_delta,
    )):
        raise RuntimeError(
            "live temporal Refiner exact-endpoint audit drift: "
            f"history={history_last_delta}, endpoint={endpoint_delta}, "
            f"retention={retention_delta}, residual={residual_max}"
        )
    if command_conditioned and (
        command_dim != 36
        or actor.temporal_composer.condition_projection is None
        or actor.temporal_composer.position_embedding.shape[1] != 12
    ):
        raise RuntimeError("current-command temporal token geometry drift")
    return {
        "frozen_expert_temporal_composer": True,
        "frozen_expert_temporal_additive_residual": bool(
            actor.additive_residual
        ),
        "frozen_expert_temporal_command_conditioned": command_conditioned,
        "frozen_expert_weight_max_delta": weight_delta,
        "frozen_expert_std_max_delta": std_delta,
        "frozen_expert_parameters_frozen": expert_frozen,
        "temporal_output_layer_exact_zero": zero_final,
        "temporal_history_steps": 10,
        "temporal_model_dim": 384,
        "temporal_transformer_layers": 6,
        "live_policy_observation_dim": int(live_obs["policy"].shape[-1]),
        "live_history_last_frame_max_delta": history_last_delta,
        "live_composed_endpoint_max_delta": endpoint_delta,
        "live_expert_retention_delta_from_one": retention_delta,
        "live_bounded_residual_max": residual_max,
        "current_reference_command_dim": command_dim,
        "live_command_tracker_prefix_max_delta": command_tracker_delta,
        "live_command_composition_term_max_delta": command_term_delta,
        "live_command_token_span_across_worlds": command_token_span,
        "residual_limit": float(actor.residual_limit),
        "temporal_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in actor.temporal_composer.parameters()
            if parameter.requires_grad
        ),
    }


def audit_action_chunk_controller(runner, env, enabled: bool) -> dict:
    """Prove exact-zero serious planner initialization and frozen Refiner."""

    if not enabled:
        return {"action_chunk_recovery": False}
    actor = runner.alg.policy.actor
    temporal = actor.temporal_composer
    final = temporal.output[-1]
    if not isinstance(final, torch.nn.Linear):
        raise RuntimeError("action-chunk temporal output layer drift")
    live_obs = env.get_observations()
    with torch.inference_mode():
        raw_plan = actor(runner.alg.policy._actor_input(live_obs))
    zero_output = bool(
        torch.count_nonzero(final.weight) == 0
        and torch.count_nonzero(final.bias) == 0
        and torch.count_nonzero(raw_plan) == 0
    )
    frozen_refiner = not any(
        parameter.requires_grad for parameter in env.acting_teacher.parameters()
    )
    if tuple(raw_plan.shape) != (env.num_envs, 7 * 29):
        raise RuntimeError(f"action-chunk live plan drift: {tuple(raw_plan.shape)}")
    if not zero_output or not frozen_refiner:
        raise RuntimeError(
            f"action-chunk initialization drift: zero={zero_output}, "
            f"frozen_refiner={frozen_refiner}"
        )
    return {
        "action_chunk_recovery": True,
        "action_chunk_knot_count": 7,
        "action_chunk_steps_per_knot": 5,
        "action_chunk_horizon": 35,
        "action_chunk_output_dim": int(raw_plan.shape[-1]),
        "action_chunk_output_layer_exact_zero": zero_output,
        "action_chunk_live_raw_plan_max_abs": float(raw_plan.abs().max().item()),
        "action_chunk_frozen_refiner_parameters": frozen_refiner,
        "action_chunk_future_or_outcome_actor_input": False,
        "action_chunk_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in temporal.parameters()
            if parameter.requires_grad
        ),
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
    transitions = horizons * 24 * env.num_envs
    recovery_contract_pass = (
        not env.physical_recovery_objective
        or (
            env.physical_recovery_calls == transitions
            and env.physical_recovery_max_abs <= 1.0
        )
    )
    frontier_contract_pass = (
        not env.cfg.get("failure_frontier_prefix_steps", 0)
        or (
            env.cumulative_teacher_control_steps > 0
            and env.cumulative_policy_control_steps > 0
            and env.maximum_teacher_execution_delta == 0.0
            and env.maximum_policy_execution_delta == 0.0
        )
    )
    result = {
        "protocol": "sugar_newton_refiner_zero_optimizer_diagnostic_v2",
        "optimizer_steps": 0,
        "horizons": horizons,
        "steps_per_horizon": 24,
        "num_envs": env.num_envs,
        "transitions": transitions,
        "all_returned_tensors_finite": finite,
        "done_total": done_total,
        "divergence_total": divergence_total,
        "divergence_by_motion": {
            env.env.clip_names[index]: int(divergence_by_motion[index].item())
            for index in nonzero.tolist()
        },
        "actor_critic_parameter_max_delta": parameter_max_delta,
        "physical_recovery_objective": env.physical_recovery_objective,
        "physical_recovery_actor_observation_augmented": False,
        "physical_recovery_calls": int(env.physical_recovery_calls),
        "physical_recovery_max_abs_reward": float(
            env.physical_recovery_max_abs
        ),
        "physical_recovery_contract_pass": recovery_contract_pass,
        "failure_frontier_prefix_steps": int(
            env.cfg.get("failure_frontier_prefix_steps", 0)
        ),
        "failure_frontier_teacher_control_steps": int(
            getattr(env, "cumulative_teacher_control_steps", 0)
        ),
        "failure_frontier_policy_control_steps": int(
            getattr(env, "cumulative_policy_control_steps", 0)
        ),
        "failure_frontier_teacher_execution_max_delta": float(
            getattr(env, "maximum_teacher_execution_delta", 0.0)
        ),
        "failure_frontier_policy_execution_max_delta": float(
            getattr(env, "maximum_policy_execution_delta", 0.0)
        ),
        "failure_frontier_contract_pass": frontier_contract_pass,
        "action_chunk_recovery": bool(env.cfg.get("action_chunk_recovery", False)),
        "action_chunk_plan_latches": int(
            getattr(env, "cumulative_plan_latches", 0)
        ),
        "action_chunk_maximum_correction": float(
            getattr(env, "maximum_correction_abs", 0.0)
        ),
        "pass": (
            finite
            and divergence_total == 0
            and parameter_max_delta == 0.0
            and recovery_contract_pass
            and frontier_contract_pass
        ),
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
    parser.add_argument("--official-action-anchor", action="store_true")
    parser.add_argument("--frozen-expert-residual", action="store_true")
    parser.add_argument("--frozen-expert-temporal-composer", action="store_true")
    parser.add_argument(
        "--frozen-expert-temporal-additive-residual",
        action="store_true",
        help=(
            "retain the exact Refiner additively and learn only a bounded "
            "29-D correction with the admitted causal temporal Transformer"
        ),
    )
    parser.add_argument(
        "--frozen-expert-temporal-command-additive-residual",
        action="store_true",
        help=(
            "retain the exact Refiner and condition the serious additive "
            "causal Transformer on the exact current 36-D Tracker command"
        ),
    )
    parser.add_argument("--released-tracker-action-supervision", action="store_true")
    parser.add_argument(
        "--physical-recovery-objective",
        action="store_true",
        help=(
            "blend the normalized official reward equally with current-rollout "
            "object/hand/contact/lift gate margins; reward labels never enter the actor"
        ),
    )
    parser.add_argument(
        "--failure-frontier-prefix-steps",
        type=int,
        default=0,
        help=(
            "execute the exact Refiner for this fixed physical prefix and mask "
            "PPO/value/entropy credit until the temporal student actually acts"
        ),
    )
    parser.add_argument(
        "--newton-native-free-recovery",
        action="store_true",
        help=(
            "remove the failed post-handoff Refiner action anchor while "
            "retaining the parameter-exact embedded Refiner and masked "
            "Newton recovery topology"
        ),
    )
    parser.add_argument(
        "--reference-phase-retiming",
        action="store_true",
        help=(
            "learn only a bounded causal reference-phase offset around the "
            "parameter-exact frozen official Refiner"
        ),
    )
    parser.add_argument(
        "--action-chunk-recovery",
        action="store_true",
        help=(
            "sample one causal seven-knot correction plan at step200 and "
            "execute its fixed 35-step chunk around the frozen Refiner"
        ),
    )
    parser.add_argument(
        "--pure-distill",
        action="store_true",
        help=(
            "keep every declared update in official BCPPO Stage 1 pure "
            "distillation; valid only with released Tracker supervision"
        ),
    )
    parser.add_argument(
        "--tracker-teacher-checkpoint",
        type=Path,
        default=DEFAULT_TRACKER_TEACHER,
    )
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
    if (
        args.released_tracker_action_supervision
        and not args.tracker_teacher_checkpoint.is_file()
    ):
        raise SystemExit(
            f"released Tracker teacher not found: {args.tracker_teacher_checkpoint}"
        )
    if args.pure_distill and not args.released_tracker_action_supervision:
        raise SystemExit(
            "--pure-distill requires --released-tracker-action-supervision"
        )
    if args.physical_recovery_objective and not (
        (
            args.frozen_expert_temporal_additive_residual
            or args.reference_phase_retiming
            or args.action_chunk_recovery
        )
        and not args.released_tracker_action_supervision
        and not args.pure_distill
    ):
        raise SystemExit(
            "--physical-recovery-objective requires the exact additive causal "
            "Refiner and forbids released-Tracker supervision/pure distillation"
        )
    if args.failure_frontier_prefix_steps not in (0, 200):
        raise SystemExit("--failure-frontier-prefix-steps is fixed at 0 or 200")
    if args.failure_frontier_prefix_steps and not (
        args.physical_recovery_objective
        and (
            args.frozen_expert_temporal_additive_residual
            or args.reference_phase_retiming
            or args.action_chunk_recovery
        )
        and args.clips == ["data_000"]
        and args.frame_zero_env_count == args.num_envs
    ):
        raise SystemExit(
            "failure-frontier training requires the physical-recovery additive "
            "temporal Refiner on only frame-zero data_000"
        )
    if args.newton_native_free_recovery and not (
        args.failure_frontier_prefix_steps == 200
        and args.physical_recovery_objective
        and (
            args.frozen_expert_temporal_additive_residual
            or args.reference_phase_retiming
            or args.action_chunk_recovery
        )
    ):
        raise SystemExit(
            "--newton-native-free-recovery is restricted to the fixed "
            "prefix200 physical-recovery additive Refiner diagnostic"
        )
    if (
        args.frozen_expert_temporal_command_additive_residual
        and not args.released_tracker_action_supervision
    ):
        raise SystemExit(
            "--frozen-expert-temporal-command-additive-residual requires "
            "--released-tracker-action-supervision so the current 36-D "
            "command is audited against the exact Tracker prefix"
        )
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
    temporal_mode = (
        args.frozen_expert_temporal_composer
        or args.frozen_expert_temporal_additive_residual
        or args.frozen_expert_temporal_command_additive_residual
    )
    temporal_command_mode = (
        args.frozen_expert_temporal_command_additive_residual
    )
    if sum(
        int(value)
        for value in (
            args.frozen_expert_temporal_composer,
            args.frozen_expert_temporal_additive_residual,
            args.frozen_expert_temporal_command_additive_residual,
            args.reference_phase_retiming,
            args.action_chunk_recovery,
        )
    ) > 1:
        raise SystemExit("choose exactly one causal temporal composition rule")
    selected_transfer_modes = sum(
        int(value)
        for value in (
            args.official_action_anchor,
            args.frozen_expert_residual,
            temporal_mode,
            args.released_tracker_action_supervision,
            args.reference_phase_retiming,
            args.action_chunk_recovery,
        )
    )
    temporal_tracker_supervision = (
        temporal_mode
        and args.released_tracker_action_supervision
    )
    if selected_transfer_modes > 1 and not (
        temporal_tracker_supervision and selected_transfer_modes == 2
    ):
        raise SystemExit(
            "choose one transfer mode; temporal composer plus released-Tracker "
            "supervision is the only admitted combination"
        )

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("Newton Refiner training must run on a Slurm CUDA compute node")
    activate_rsl_rl(args.rsl_rl_root)
    embedded_expert = (
        args.frozen_expert_residual
        or temporal_mode
        or args.released_tracker_action_supervision
        or args.reference_phase_retiming
        or args.action_chunk_recovery
    )
    if args.official_action_anchor or embedded_expert:
        sugar_bcppo()
    if embedded_expert:
        import rsl_rl.modules
        from sugar_rl.utils.frozen_expert_transition_actor_critic import (
            FrozenOfficialRefinerCausalTemporalComposerActorCritic,
            FrozenOfficialRefinerResidualActorCritic,
            FrozenOfficialRefinerTrackerSupervisedCausalTemporalComposerActorCritic,
            FrozenOfficialRefinerTrackerSupervisedResidualActorCritic,
            RefinerReferencePhaseCausalTemporalActorCritic,
            RefinerActionChunkCausalTemporalActorCritic,
        )

        builtins.FrozenOfficialRefinerResidualActorCritic = (
            FrozenOfficialRefinerResidualActorCritic
        )
        rsl_rl.modules.FrozenOfficialRefinerResidualActorCritic = (
            FrozenOfficialRefinerResidualActorCritic
        )
        builtins.FrozenOfficialRefinerCausalTemporalComposerActorCritic = (
            FrozenOfficialRefinerCausalTemporalComposerActorCritic
        )
        rsl_rl.modules.FrozenOfficialRefinerCausalTemporalComposerActorCritic = (
            FrozenOfficialRefinerCausalTemporalComposerActorCritic
        )
        builtins.FrozenOfficialRefinerTrackerSupervisedCausalTemporalComposerActorCritic = (
            FrozenOfficialRefinerTrackerSupervisedCausalTemporalComposerActorCritic
        )
        rsl_rl.modules.FrozenOfficialRefinerTrackerSupervisedCausalTemporalComposerActorCritic = (
            FrozenOfficialRefinerTrackerSupervisedCausalTemporalComposerActorCritic
        )
        builtins.FrozenOfficialRefinerTrackerSupervisedResidualActorCritic = (
            FrozenOfficialRefinerTrackerSupervisedResidualActorCritic
        )
        rsl_rl.modules.FrozenOfficialRefinerTrackerSupervisedResidualActorCritic = (
            FrozenOfficialRefinerTrackerSupervisedResidualActorCritic
        )
        builtins.RefinerReferencePhaseCausalTemporalActorCritic = (
            RefinerReferencePhaseCausalTemporalActorCritic
        )
        rsl_rl.modules.RefinerReferencePhaseCausalTemporalActorCritic = (
            RefinerReferencePhaseCausalTemporalActorCritic
        )
        builtins.RefinerActionChunkCausalTemporalActorCritic = (
            RefinerActionChunkCausalTemporalActorCritic
        )
        rsl_rl.modules.RefinerActionChunkCausalTemporalActorCritic = (
            RefinerActionChunkCausalTemporalActorCritic
        )
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
        policy_history_steps=(
            10
            if (
                temporal_mode
                or args.reference_phase_retiming
                or args.action_chunk_recovery
            )
            else 0
        ),
        policy_command_dim=(36 if temporal_command_mode else 0),
        tracker_teacher=args.released_tracker_action_supervision,
        physical_recovery_objective=args.physical_recovery_objective,
        failure_frontier_prefix_steps=args.failure_frontier_prefix_steps,
        failure_frontier_teacher_checkpoint=(
            args.initial_checkpoint if args.failure_frontier_prefix_steps else None
        ),
        reference_phase_retiming=args.reference_phase_retiming,
        action_chunk_recovery=args.action_chunk_recovery,
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
    audit = (
        {
            "source_checkpoint": str(args.initial_checkpoint.resolve()),
            "source_sha256": sha256(args.initial_checkpoint),
            "strict_model_state_load": False,
            "optimizer_state_loaded": False,
            "learning_iteration_loaded": False,
            "newton_learning_iteration": runner.current_learning_iteration,
            "parameter_count": sum(
                parameter.numel() for parameter in runner.alg.policy.parameters()
            ),
        }
        if embedded_expert
        else load_initial_weights(runner, args.initial_checkpoint)
    )
    audit.update(audit_official_action_anchor(runner, args.official_action_anchor))
    audit.update(
        audit_frozen_expert_residual(
            runner,
            args.initial_checkpoint,
            args.frozen_expert_residual
            or (
                args.released_tracker_action_supervision
                and not temporal_mode
            ),
        )
    )
    audit.update(
        audit_released_tracker_action_supervision(
            runner,
            env,
            args.tracker_teacher_checkpoint,
            args.released_tracker_action_supervision,
        )
    )
    audit.update(
        audit_frozen_expert_temporal_composer(
            runner,
            env,
            args.initial_checkpoint,
            temporal_mode,
        )
    )
    audit.update(
        audit_action_chunk_controller(
            runner,
            env,
            args.action_chunk_recovery,
        )
    )
    source_action_std = float(runner.alg.policy.std.detach().mean().item())
    audit.update(
        {
            "protocol": "sugar_newton_official_refiner_ppo_transfer_v5",
            "seed": args.seed,
            "num_envs": args.num_envs,
            "num_motions": len(env.env.clip_names),
            "policy_observation_dim": (
                890 * 11 + (36 if temporal_command_mode else 0)
                if (
                    temporal_mode
                    or args.reference_phase_retiming
                    or args.action_chunk_recovery
                )
                else 890
            ),
            "critic_observation_dim": 890,
            "teacher_observation_dim": (
                510 if args.released_tracker_action_supervision else int(
                    env.get_observations()["teacher"].shape[-1]
                )
            ),
            "action_dim": (
                203
                if args.action_chunk_recovery
                else 1
                if args.reference_phase_retiming
                else 29
            ),
            "max_iterations": args.max_iterations,
            "fresh_optimizer": True,
            "source_action_std_mean": source_action_std,
            "training_action_std": args.action_std,
            "learning_rate": args.learning_rate,
            "fixed_learning_rate": bool(
                (args.official_action_anchor or embedded_expert)
                and runner.alg.desired_kl is None
            ),
            "pure_distill": args.pure_distill,
            "bc_only_steps": int(getattr(runner.alg, "bc_only_steps", 0)),
            "critic_warmup_steps": int(
                getattr(runner.alg, "critic_warmup_steps", 0)
            ),
            "full_ppo_warmup_steps": int(
                getattr(runner.alg, "full_ppo_warmup_steps", 0)
            ),
            "reward_clip": args.reward_clip,
            "physical_recovery_objective": args.physical_recovery_objective,
            "physical_recovery_actor_observation_augmented": False,
            "physical_recovery_official_weight": (
                0.5 if args.physical_recovery_objective else None
            ),
            "physical_recovery_physics_weight": (
                0.5 if args.physical_recovery_objective else None
            ),
            "physical_recovery_official_scale": (
                5.125 if args.physical_recovery_objective else None
            ),
            "failure_frontier_prefix_steps": args.failure_frontier_prefix_steps,
            "failure_frontier_training_mask_actor_input": False,
            "newton_native_free_recovery": args.newton_native_free_recovery,
            "reference_phase_retiming": args.reference_phase_retiming,
            "action_chunk_recovery": args.action_chunk_recovery,
            "post_handoff_refiner_action_anchor": (
                not args.newton_native_free_recovery
            ),
            "failure_frontier_teacher_checkpoint": (
                str(args.initial_checkpoint.resolve())
                if args.failure_frontier_prefix_steps
                else None
            ),
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
    method = (
        "BCPPO Stage-1 pure released-Tracker distillation into frozen-Refiner causal temporal composer"
        if (
            temporal_mode
            and args.released_tracker_action_supervision
            and args.pure_distill
        )
        else "BCPPO released-Tracker-supervised frozen-Refiner causal temporal composer"
        if (
            temporal_mode
            and args.released_tracker_action_supervision
        )
        else "BCPPO Stage-1 pure released-Tracker distillation into frozen-Refiner residual"
        if args.released_tracker_action_supervision and args.pure_distill
        else "BCPPO released-Tracker-supervised frozen-Refiner residual"
        if args.released_tracker_action_supervision
        else "BCPPO frozen-official-Refiner causal temporal composer"
        if temporal_mode
        else "BCPPO frozen-official-Refiner residual"
        if args.frozen_expert_residual
        else "BCPPO official-action-anchor"
        if args.official_action_anchor
        else "PPO"
    )
    print(
        f"[train] official Refiner Newton transfer: {args.max_iterations} "
        f"fresh {method} updates"
    )
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    final_state = runner.alg.policy.state_dict()
    parameter_max_delta = max(
        (final_state[key] - value).abs().max().item()
        for key, value in pre_training_state.items()
    )
    actor_parameter_max_delta = max(
        (final_state[key] - value).abs().max().item()
        for key, value in pre_training_state.items()
        if key.startswith("actor.")
    )
    critic_parameter_max_delta = max(
        (final_state[key] - value).abs().max().item()
        for key, value in pre_training_state.items()
        if key.startswith("critic.")
    )
    all_policy_parameters_finite = all(
        bool(torch.isfinite(value).all()) for value in final_state.values()
    )
    transitions = args.max_iterations * 24 * args.num_envs
    divergence_total = int(env.env.num_diverged)
    divergence_rate = divergence_total / transitions
    final_action_std_mean = float(runner.alg.policy.std.detach().mean().item())
    final_action_std_delta_from_config = abs(
        final_action_std_mean - args.action_std
    )
    pure_distill_contract_pass = (
        not args.pure_distill
        or (
            actor_parameter_max_delta > 0.0
            and critic_parameter_max_delta == 0.0
            and final_action_std_delta_from_config <= 1.0e-7
        )
    )
    pure_distill_short_divergence_pass = (
        not (args.pure_distill and args.max_iterations == 2)
        or divergence_total == 0
    )
    frontier_contract_pass = (
        not args.failure_frontier_prefix_steps
        or (
            args.reference_phase_retiming
            and env.cumulative_masked_steps > 0
            and env.maximum_abs_phase_offset <= 7
            and env.teacher_checkpoint_sha256 == sha256(args.initial_checkpoint)
        )
        or (
            not args.reference_phase_retiming
            and
            env.cumulative_teacher_control_steps > 0
            and env.cumulative_policy_control_steps > 0
            and env.maximum_teacher_execution_delta == 0.0
            and env.maximum_policy_execution_delta == 0.0
        )
    )
    frontier_learning_pass = (
        not args.failure_frontier_prefix_steps
        or (
            actor_parameter_max_delta > 0.0
            and critic_parameter_max_delta > 0.0
            and final_action_std_delta_from_config > 0.0
        )
    )
    result = {
        "protocol": "sugar_newton_refiner_ppo_training_gate_v1",
        "optimizer_updates": args.max_iterations,
        "transitions": transitions,
        "divergence_total": divergence_total,
        "divergence_rate": divergence_rate,
        "maximum_admitted_training_divergence_rate": 0.005,
        "all_policy_parameters_finite": all_policy_parameters_finite,
        "actor_critic_parameter_max_delta": parameter_max_delta,
        "actor_parameter_max_delta": actor_parameter_max_delta,
        "critic_parameter_max_delta": critic_parameter_max_delta,
        "final_action_std_mean": final_action_std_mean,
        "final_action_std_delta_from_config": final_action_std_delta_from_config,
        "final_learning_rate": float(runner.alg.learning_rate),
        "official_action_anchor": args.official_action_anchor,
        "frozen_expert_residual": args.frozen_expert_residual,
        "frozen_expert_temporal_composer": temporal_mode,
        "frozen_expert_temporal_additive_residual": (
            args.frozen_expert_temporal_additive_residual
            or temporal_command_mode
        ),
        "frozen_expert_temporal_command_conditioned": temporal_command_mode,
        "released_tracker_action_supervision": args.released_tracker_action_supervision,
        "temporal_tracker_action_supervision": temporal_tracker_supervision,
        "pure_distill": args.pure_distill,
        "physical_recovery_objective": args.physical_recovery_objective,
        "physical_recovery_actor_observation_augmented": False,
        "physical_recovery_calls": int(env.physical_recovery_calls),
        "physical_recovery_max_abs_reward": float(
            env.physical_recovery_max_abs
        ),
        "failure_frontier_prefix_steps": args.failure_frontier_prefix_steps,
        "failure_frontier_training_mask_actor_input": False,
        "newton_native_free_recovery": args.newton_native_free_recovery,
        "reference_phase_retiming": args.reference_phase_retiming,
        "action_chunk_recovery": args.action_chunk_recovery,
        "action_chunk_plan_latches": int(
            getattr(env, "cumulative_plan_latches", 0)
        ),
        "action_chunk_maximum_correction": float(
            getattr(env, "maximum_correction_abs", 0.0)
        ),
        "reference_phase_maximum_abs_offset": int(
            getattr(env, "maximum_abs_phase_offset", 0)
        ),
        "reference_phase_retimed_steps": int(
            getattr(env, "cumulative_retimed_steps", 0)
        ),
        "post_handoff_refiner_action_anchor": (
            not args.newton_native_free_recovery
        ),
        "failure_frontier_teacher_control_steps": int(
            getattr(env, "cumulative_teacher_control_steps", 0)
        ),
        "failure_frontier_policy_control_steps": int(
            getattr(env, "cumulative_policy_control_steps", 0)
        ),
        "failure_frontier_teacher_execution_max_delta": float(
            getattr(env, "maximum_teacher_execution_delta", 0.0)
        ),
        "failure_frontier_policy_execution_max_delta": float(
            getattr(env, "maximum_policy_execution_delta", 0.0)
        ),
        "failure_frontier_contract_pass": frontier_contract_pass,
        "failure_frontier_learning_pass": frontier_learning_pass,
        "pure_distill_contract_pass": pure_distill_contract_pass,
        "pure_distill_short_divergence_pass": (
            pure_distill_short_divergence_pass
        ),
        "pass": (
            all_policy_parameters_finite
            and divergence_rate <= 0.005
            and pure_distill_contract_pass
            and pure_distill_short_divergence_pass
            and frontier_contract_pass
            and frontier_learning_pass
            and (
                not args.physical_recovery_objective
                or (
                    env.physical_recovery_calls == transitions
                    and env.physical_recovery_max_abs <= 1.0
                )
            )
        ),
        "frozen_evaluation_required": True,
    }
    (log_dir / "TRAINING_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
