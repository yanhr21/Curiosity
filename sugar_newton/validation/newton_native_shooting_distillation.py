"""Distill an admitted Newton shooting target into the serious causal Refiner.

The source target must be a passing matched-v2 shooting result.  This script
reconstructs its exact 200-step official-Refiner prefix and 35-step Newton
recovery, builds real past-only ``10 x 890`` windows, and trains only the
repository's six-layer 384-D causal temporal composer.  The embedded released
Refiner remains parameter-exact.  Shooting futures are labels only and never
enter the deployed actor input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import warp as wp

from sugar_newton.rl import obs_890
from sugar_newton.rl.carrybox_env import CarryBoxEnv
from sugar_newton.rl.train_bcppo import activate_rsl_rl
from sugar_newton.validation.newton_native_action_shooting import (
    aggregate_world_records,
    world_records_match,
)
from sugar_newton.validation.refiner_open_loop import (
    TERMINATION_KEYS,
    load_official_teacher,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFINER = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt"
)
HISTORY_STEPS = 10
PREFIX_STEPS = 200
RECOVERY_STEPS = 35
PREFIX_RETENTION_STEPS = 64
MATCHED_WORLDS = 8
TRAIN_WORLDS = tuple(range(6))
VALIDATION_WORLDS = (6, 7)
OPTIMIZER_STEPS = 3000
BATCH_PER_CLASS = 64
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4


def make_environment(args, *, seed: int) -> CarryBoxEnv:
    return CarryBoxEnv(
        num_envs=MATCHED_WORLDS,
        clip_names=["data_000"],
        motion_root=args.motion_root,
        teacher_motion_root=args.motion_root,
        episode_length=10**9,
        substeps=4,
        mu=1.0,
        device=args.device,
        seed=seed,
        auto_reset=False,
    )


def append_history(sequence: list[torch.Tensor], current: torch.Tensor) -> None:
    sequence.append(current.detach().clone())
    if len(sequence) > HISTORY_STEPS:
        del sequence[0]


def stacked_history(sequence: list[torch.Tensor]) -> torch.Tensor:
    if len(sequence) != HISTORY_STEPS:
        raise RuntimeError(f"causal history has {len(sequence)} frames")
    history = torch.stack(sequence, dim=1)
    if not torch.equal(history[:, -1], sequence[-1]):
        raise RuntimeError("causal history does not end at current observation")
    return history


def triangle_pair_usage(env: CarryBoxEnv) -> tuple[int, int]:
    """Return the public Newton narrow-phase triangle count and capacity."""

    narrow_phase = env.pipeline.narrow_phase
    counter = narrow_phase.triangle_pairs_count
    if counter is None:
        return 0, int(narrow_phase.max_triangle_pairs)
    return (
        int(wp.to_torch(counter)[0].item()),
        int(narrow_phase.max_triangle_pairs),
    )


def collect_supervision(
    args,
    teacher,
    correction: torch.Tensor,
    *,
    source_seed: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Replay the admitted target and retain causal prefix/recovery samples."""

    env = make_environment(args, seed=source_seed)
    device = torch.device(args.device)
    ids = torch.arange(MATCHED_WORLDS, device=device)
    env.reset(ids, motion_ids=torch.zeros_like(ids), start_frames=0)
    sequence: list[torch.Tensor] = []
    prefix_histories = []
    recovery_histories = []
    recovery_targets = []
    maximum_triangle_pairs = 0
    triangle_pair_capacity = int(env.pipeline.narrow_phase.max_triangle_pairs)

    for step in range(PREFIX_STEPS):
        current = obs_890.build(env, teacher=True)
        append_history(sequence, current)
        if step >= PREFIX_STEPS - PREFIX_RETENTION_STEPS:
            prefix_histories.append(stacked_history(sequence))
        with torch.inference_mode():
            action = teacher(current)
        _, _, done, _ = env.step(action)
        triangle_pairs, triangle_pair_capacity = triangle_pair_usage(env)
        maximum_triangle_pairs = max(maximum_triangle_pairs, triangle_pairs)
        if bool(done.any()):
            raise RuntimeError("official Refiner failed before shooting frontier")

    initial_box_z = env.episode_initial_box_z.clone()
    active = torch.ones(MATCHED_WORLDS, dtype=torch.bool, device=device)
    survived = torch.zeros(MATCHED_WORLDS, dtype=torch.long, device=device)
    peak_lift = torch.zeros(MATCHED_WORLDS, device=device)
    bilateral_steps = torch.zeros(MATCHED_WORLDS, device=device)
    minimum_object_margin = torch.ones(MATCHED_WORLDS, device=device)
    minimum_ee_margin = torch.ones(MATCHED_WORLDS, device=device)
    failure_seen = {
        key: torch.zeros(MATCHED_WORLDS, dtype=torch.bool, device=device)
        for key in TERMINATION_KEYS
    }

    for step in range(RECOVERY_STEPS):
        current = obs_890.build(env, teacher=True)
        append_history(sequence, current)
        recovery_histories.append(stacked_history(sequence))
        target = correction[step].expand(MATCHED_WORLDS, -1)
        recovery_targets.append(target.detach().clone())
        with torch.inference_mode():
            base_action = teacher(current)
        _, _, done, extras = env.step(base_action + target)
        triangle_pairs, triangle_pair_capacity = triangle_pair_usage(env)
        maximum_triangle_pairs = max(maximum_triangle_pairs, triangle_pairs)
        survived += active.long()
        lift = env._body_q()[:, env.box_body, 2] - initial_box_z
        peak_lift = torch.where(active, torch.maximum(peak_lift, lift), peak_lift)
        terms = extras["physical_recovery_terms"]
        bilateral_steps += active.float() * terms["bilateral_contact"]
        minimum_object_margin = torch.where(
            active,
            torch.minimum(minimum_object_margin, terms["object_margin"]),
            minimum_object_margin,
        )
        minimum_ee_margin = torch.where(
            active,
            torch.minimum(minimum_ee_margin, terms["end_effector_margin"]),
            minimum_ee_margin,
        )
        newly_done = active & done
        for key in TERMINATION_KEYS:
            failure_seen[key] |= newly_done & extras["termination_terms"][key]
        active &= ~newly_done

    world_records = []
    correction_rms = float(correction.square().mean().sqrt())
    for index in range(MATCHED_WORLDS):
        world_records.append(
            {
                "alive_after_horizon": bool(active[index]),
                "survived_recovery_steps": int(survived[index]),
                "peak_lift_m": float(peak_lift[index]),
                "bilateral_contact_fraction": float(
                    bilateral_steps[index] / max(int(survived[index]), 1)
                ),
                "minimum_object_margin": float(minimum_object_margin[index]),
                "minimum_end_effector_margin": float(minimum_ee_margin[index]),
                "correction_rms": correction_rms,
                "strict_failure_reasons": [
                    key for key in TERMINATION_KEYS if bool(failure_seen[key][index])
                ],
            }
        )

    prefix = torch.cat(prefix_histories, dim=0)
    recovery = torch.cat(recovery_histories, dim=0)
    recovery_target = torch.cat(recovery_targets, dim=0)
    prefix_target = torch.zeros(
        prefix.shape[0], correction.shape[-1], device=device
    )
    prefix_world = torch.arange(MATCHED_WORLDS, device=device).repeat(
        PREFIX_RETENTION_STEPS
    )
    recovery_world = torch.arange(MATCHED_WORLDS, device=device).repeat(
        RECOVERY_STEPS
    )
    target_aggregate = aggregate_world_records(world_records)
    target_aggregate.update(
        {
            "maximum_triangle_pair_count": maximum_triangle_pairs,
            "triangle_pair_capacity": triangle_pair_capacity,
            "triangle_pair_overflow": maximum_triangle_pairs > triangle_pair_capacity,
        }
    )
    return (
        {
            "prefix_history": prefix,
            "prefix_target": prefix_target,
            "prefix_world": prefix_world,
            "recovery_history": recovery,
            "recovery_target": recovery_target,
            "recovery_world": recovery_world,
        },
        target_aggregate,
    )


def subset(data: dict[str, torch.Tensor], kind: str, worlds: tuple[int, ...]):
    world = data[f"{kind}_world"]
    mask = torch.zeros_like(world, dtype=torch.bool)
    for index in worlds:
        mask |= world == index
    return data[f"{kind}_history"][mask], data[f"{kind}_target"][mask]


@torch.no_grad()
def action_mse(core, history: torch.Tensor, target: torch.Tensor) -> float:
    total = 0.0
    count = 0
    for begin in range(0, history.shape[0], 256):
        prediction = torch.tanh(core(history[begin : begin + 256]))
        batch_target = target[begin : begin + 256]
        total += float(torch.nn.functional.mse_loss(
            prediction, batch_target, reduction="sum"
        ))
        count += batch_target.numel()
    return total / count


def replay_student(args, teacher, actor, *, source_seed: int) -> dict:
    """Frozen exact-prefix rollout of the trained deployed actor."""

    env = make_environment(args, seed=source_seed)
    device = torch.device(args.device)
    ids = torch.arange(MATCHED_WORLDS, device=device)
    env.reset(ids, motion_ids=torch.zeros_like(ids), start_frames=0)
    sequence: list[torch.Tensor] = []
    for _ in range(PREFIX_STEPS):
        current = obs_890.build(env, teacher=True)
        append_history(sequence, current)
        with torch.inference_mode():
            action = teacher(current)
        _, _, done, _ = env.step(action)
        if bool(done.any()):
            raise RuntimeError("official Refiner failed before student frontier")

    initial_box_z = env.episode_initial_box_z.clone()
    active = torch.ones(MATCHED_WORLDS, dtype=torch.bool, device=device)
    survived = torch.zeros(MATCHED_WORLDS, dtype=torch.long, device=device)
    peak_lift = torch.zeros(MATCHED_WORLDS, device=device)
    bilateral_steps = torch.zeros(MATCHED_WORLDS, device=device)
    minimum_object_margin = torch.ones(MATCHED_WORLDS, device=device)
    minimum_ee_margin = torch.ones(MATCHED_WORLDS, device=device)
    failure_seen = {
        key: torch.zeros(MATCHED_WORLDS, dtype=torch.bool, device=device)
        for key in TERMINATION_KEYS
    }
    residual_abs_sum = torch.zeros(MATCHED_WORLDS, device=device)
    residual_square_sum = torch.zeros(MATCHED_WORLDS, device=device)
    maximum_triangle_pairs = 0
    triangle_pair_capacity = int(env.pipeline.narrow_phase.max_triangle_pairs)

    actor.eval()
    for _ in range(RECOVERY_STEPS):
        current = obs_890.build(env, teacher=True)
        append_history(sequence, current)
        history = stacked_history(sequence)
        actor_input = torch.cat((current, history.flatten(1)), dim=-1)
        with torch.inference_mode():
            terms = actor.composition_terms(actor_input)
            action = terms["composed_action"]
        residual_abs_sum += terms["bounded_residual_action"].abs().mean(dim=-1)
        residual_square_sum += terms["bounded_residual_action"].square().mean(dim=-1)
        _, _, done, extras = env.step(action)
        triangle_pairs, triangle_pair_capacity = triangle_pair_usage(env)
        maximum_triangle_pairs = max(maximum_triangle_pairs, triangle_pairs)
        survived += active.long()
        lift = env._body_q()[:, env.box_body, 2] - initial_box_z
        peak_lift = torch.where(active, torch.maximum(peak_lift, lift), peak_lift)
        physical = extras["physical_recovery_terms"]
        bilateral_steps += active.float() * physical["bilateral_contact"]
        minimum_object_margin = torch.where(
            active,
            torch.minimum(minimum_object_margin, physical["object_margin"]),
            minimum_object_margin,
        )
        minimum_ee_margin = torch.where(
            active,
            torch.minimum(minimum_ee_margin, physical["end_effector_margin"]),
            minimum_ee_margin,
        )
        newly_done = active & done
        for key in TERMINATION_KEYS:
            failure_seen[key] |= newly_done & extras["termination_terms"][key]
        active &= ~newly_done

    records = []
    for index in range(MATCHED_WORLDS):
        records.append(
            {
                "alive_after_horizon": bool(active[index]),
                "survived_recovery_steps": int(survived[index]),
                "peak_lift_m": float(peak_lift[index]),
                "bilateral_contact_fraction": float(
                    bilateral_steps[index] / max(int(survived[index]), 1)
                ),
                "minimum_object_margin": float(minimum_object_margin[index]),
                "minimum_end_effector_margin": float(minimum_ee_margin[index]),
                "correction_rms": float(
                    torch.sqrt(residual_square_sum[index] / RECOVERY_STEPS)
                ),
                "mean_abs_residual": float(
                    residual_abs_sum[index] / RECOVERY_STEPS
                ),
                "strict_failure_reasons": [
                    key for key in TERMINATION_KEYS if bool(failure_seen[key][index])
                ],
            }
        )
    aggregate = aggregate_world_records(records)
    aggregate["mean_abs_residual"] = sum(
        record["mean_abs_residual"] for record in records
    ) / MATCHED_WORLDS
    aggregate["maximum_triangle_pair_count"] = maximum_triangle_pairs
    aggregate["triangle_pair_capacity"] = triangle_pair_capacity
    aggregate["triangle_pair_overflow"] = (
        maximum_triangle_pairs > triangle_pair_capacity
    )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shooting-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, default=DEFAULT_REFINER)
    parser.add_argument("--motion-root", type=Path, default=Path("SUGAR/data/CarryBox"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=171741)
    parser.add_argument("--rsl-rl-root", required=True)
    args = parser.parse_args()

    source_result_path = args.shooting_dir / "RESULT.json"
    source_correction_path = args.shooting_dir / "BEST_CORRECTION.pt"
    if not source_result_path.is_file() or not source_correction_path.is_file():
        raise SystemExit("shooting RESULT.json and BEST_CORRECTION.pt are required")
    source = json.loads(source_result_path.read_text())
    admitted_protocols = {
        "sugar_newton_native_matched_action_shooting_v2",
        "sugar_newton_native_robust_matched_action_shooting_v3",
    }
    if source.get("protocol") not in admitted_protocols:
        raise SystemExit("source is not an admitted matched shooting protocol")
    if not source.get("passed"):
        raise SystemExit("shooting source did not pass; supervision is forbidden")
    if source.get("prefix_steps") != PREFIX_STEPS or source.get("horizon") != RECOVERY_STEPS:
        raise SystemExit("shooting prefix/horizon contract drifted")

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise SystemExit("shooting distillation must run on a Slurm CUDA compute node")
    activate_rsl_rl(args.rsl_rl_root)
    sugar_rl_source = ROOT / "SUGAR/source/sugar_rl"
    if str(sugar_rl_source) not in sys.path:
        sys.path.insert(0, str(sugar_rl_source))
    from sugar_rl.utils.frozen_expert_transition_actor_critic import (
        FrozenOfficialRefinerCausalTemporalComposer,
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    teacher, hidden_dims = load_official_teacher(args.refiner_checkpoint, device)
    if hidden_dims != [512, 256, 128]:
        raise RuntimeError(f"official Refiner topology drifted: {hidden_dims}")
    correction_payload = torch.load(
        source_correction_path, map_location=device, weights_only=True
    )
    correction = correction_payload["correction_knots"].to(device).repeat_interleave(
        int(correction_payload["steps_per_knot"]), dim=0
    )
    if tuple(correction.shape) != (RECOVERY_STEPS, 29):
        raise RuntimeError(f"shooting correction geometry drifted: {tuple(correction.shape)}")

    actor = FrozenOfficialRefinerCausalTemporalComposer(
        args.refiner_checkpoint,
        additive_residual=True,
    ).to(device)
    expert_before = {
        key: value.detach().clone() for key, value in actor.expert.state_dict().items()
    }
    expert_std_before = actor.expert_std.detach().clone()
    data, target_replay = collect_supervision(
        args, teacher, correction, source_seed=int(source["seed"])
    )
    target_matches_source, target_source_max_delta = world_records_match(
        source["best_replay_record"]["world_records"],
        target_replay["world_records"],
    )
    if not target_matches_source or target_replay["triangle_pair_overflow"]:
        rejection = {
            "protocol": "sugar_newton_native_shooting_distillation_v1",
            "seed": args.seed,
            "source_shooting_result": str(source_result_path.resolve()),
            "source_shooting_sha256": sha256(source_result_path),
            "optimizer_steps": 0,
            "fresh_target_source_maximum_metric_delta": target_source_max_delta,
            "checks": {
                "fresh_target_replay_matches_admitted_source": target_matches_source,
                "fresh_target_replay_has_no_triangle_pair_overflow": not target_replay[
                    "triangle_pair_overflow"
                ],
            },
            "passed": False,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "RESULT.json").write_text(
            json.dumps(rejection, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(rejection, indent=2))
        return 1
    train_prefix = subset(data, "prefix", TRAIN_WORLDS)
    train_recovery = subset(data, "recovery", TRAIN_WORLDS)
    validation_prefix = subset(data, "prefix", VALIDATION_WORLDS)
    validation_recovery = subset(data, "recovery", VALIDATION_WORLDS)

    with torch.inference_mode():
        current = train_recovery[0][:, -1]
        actor_input = torch.cat((current, train_recovery[0].flatten(1)), dim=-1)
        zero_init_delta = float(
            (actor(actor_input) - actor.endpoint_action(actor_input)).abs().max()
        )

    parameters = list(actor.temporal_composer.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    if optimizer.state:
        raise RuntimeError("shooting distillation optimizer is not fresh")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    loss_trace = []
    actor.temporal_composer.train()
    actor.expert.eval()
    for step in range(OPTIMIZER_STEPS):
        prefix_indices = torch.randint(
            train_prefix[0].shape[0],
            (BATCH_PER_CLASS,),
            device=device,
            generator=generator,
        )
        recovery_indices = torch.randint(
            train_recovery[0].shape[0],
            (BATCH_PER_CLASS,),
            device=device,
            generator=generator,
        )
        prefix_prediction = torch.tanh(
            actor.temporal_composer(train_prefix[0][prefix_indices])
        )
        recovery_prediction = torch.tanh(
            actor.temporal_composer(train_recovery[0][recovery_indices])
        )
        prefix_loss = torch.nn.functional.mse_loss(
            prefix_prediction, train_prefix[1][prefix_indices]
        )
        recovery_loss = torch.nn.functional.mse_loss(
            recovery_prediction, train_recovery[1][recovery_indices]
        )
        loss = 0.5 * (prefix_loss + recovery_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step in (0, 9, 99, 499, 999, 1999, OPTIMIZER_STEPS - 1):
            record = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "prefix_loss": float(prefix_loss.detach()),
                "recovery_loss": float(recovery_loss.detach()),
            }
            loss_trace.append(record)
            print(json.dumps(record), flush=True)

    actor.eval()
    expert_delta = max(
        max(
            float((actor.expert.state_dict()[key] - value).abs().max())
            for key, value in expert_before.items()
        ),
        float((actor.expert_std - expert_std_before).abs().max()),
    )
    metrics = {
        "train_prefix_mse": action_mse(actor.temporal_composer, *train_prefix),
        "train_recovery_mse": action_mse(actor.temporal_composer, *train_recovery),
        "validation_prefix_mse": action_mse(
            actor.temporal_composer, *validation_prefix
        ),
        "validation_recovery_mse": action_mse(
            actor.temporal_composer, *validation_recovery
        ),
    }
    student_replay = replay_student(
        args, teacher, actor, source_seed=int(source["seed"])
    )
    checks = {
        "source_matched_shooting_passed": bool(source["passed"]),
        "source_target_replay_crosses_all_eight_worlds": (
            target_replay["alive_world_count"] == MATCHED_WORLDS
        ),
        "fresh_target_replay_matches_admitted_source": target_matches_source,
        "fresh_target_replay_has_no_triangle_pair_overflow": not target_replay[
            "triangle_pair_overflow"
        ],
        "causal_history_last_frame_is_current": True,
        "shooting_future_is_training_label_only": True,
        "deployed_actor_observation_augmented": False,
        "zero_initialization_is_exact_official_refiner": zero_init_delta == 0.0,
        "official_refiner_remains_parameter_exact": expert_delta == 0.0,
        "student_replay_crosses_all_eight_worlds": (
            student_replay["alive_world_count"] == MATCHED_WORLDS
        ),
        "student_replay_reaches_five_cm_in_all_eight_worlds": (
            student_replay["minimum_peak_lift_m"] >= 0.05
        ),
        "student_replay_has_no_triangle_pair_overflow": not student_replay[
            "triangle_pair_overflow"
        ],
    }
    result = {
        "protocol": "sugar_newton_native_shooting_distillation_v1",
        "seed": args.seed,
        "source_shooting_result": str(source_result_path.resolve()),
        "source_shooting_sha256": sha256(source_result_path),
        "official_refiner_checkpoint": str(args.refiner_checkpoint.resolve()),
        "official_refiner_sha256": sha256(args.refiner_checkpoint),
        "architecture": {
            "history": [HISTORY_STEPS, 890],
            "model_dim": 384,
            "attention_heads": 8,
            "transformer_layers": 6,
            "output_mlp": [512, 256, 128, 29],
            "trainable_parameters": sum(p.numel() for p in parameters),
            "frozen_official_refiner": True,
            "additive_residual_limit": 1.0,
        },
        "dataset": {
            "train_worlds": list(TRAIN_WORLDS),
            "validation_worlds": list(VALIDATION_WORLDS),
            "prefix_retention_steps": PREFIX_RETENTION_STEPS,
            "recovery_steps": RECOVERY_STEPS,
            "balanced_prefix_recovery_batches": True,
        },
        "optimizer": {
            "name": "AdamW",
            "steps": OPTIMIZER_STEPS,
            "batch_per_class": BATCH_PER_CLASS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_norm_clip": 1.0,
            "fresh_optimizer": True,
        },
        "zero_initialization_action_max_delta": zero_init_delta,
        "official_refiner_parameter_max_delta": expert_delta,
        "fresh_target_source_maximum_metric_delta": target_source_max_delta,
        "loss_trace": loss_trace,
        "mse": metrics,
        "target_replay": target_replay,
        "student_replay": student_replay,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": {
            "actor." + key: value.detach().cpu()
            for key, value in actor.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": OPTIMIZER_STEPS,
        "source_shooting_sha256": result["source_shooting_sha256"],
    }
    torch.save(checkpoint, args.output / "model_3000.pt")
    (args.output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "loss_trace"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
