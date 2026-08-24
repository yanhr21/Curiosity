#!/usr/bin/env python3
"""Distill official Carry45/Kick21 actions into one demo-conditioned SUGAR actor.

This is a fixed serious overfit diagnostic, not a replacement model.  It keeps
the existing shared SUGAR 512/256/128 policy and the frozen 11.386M causal
predictor.  For each recorded physical state trajectory, the same actor input
is paired once with Carry45 conditioning and a zero residual, and once with
Kick21 conditioning and the released ``Kick Tracker - Carry Tracker`` action
residual.  The unchanged frozen Carry Refiner remains the common execution
baseline.  Future actions are training labels only; deployment receives only
causal state and selected-demo conditioning.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
for source in (
    ROOT / "SUGAR/source/sugar_rl",
    ROOT / "IsaacLab/source/isaaclab",
    ROOT / "IsaacLab/source/isaaclab_rl",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from sugar_rl.utils.demo_event_reward_runtime import (  # noqa: E402
    ACTIONABLE_DEMO_CONDITIONING_DIM,
    FrozenPhaseAwareDemoEventScorer,
    FrozenPhaseAwareDemoEventScorerCfg,
)
from sugar_rl.utils.sugar_native_curiosity_ppo import (  # noqa: E402
    SugarNativeZeroPreservingTactileActorCritic,
)


CORPUS = ROOT / (
    "experiments/demo_following/contact_event_reward_redesign_v1/"
    "deployable_goal_core_corpus_v1"
)
RUNTIME_CONFIG = ROOT / (
    "experiments/demo_following/contact_event_reward_redesign_v1/"
    "phase_aware_dense_feedback_scale_audit_v1/RUNTIME_CONFIG.json"
)
SOURCE_CHECKPOINT = ROOT / (
    "experiments/demo_following/shared_actionable_demo_conditioning_v1/"
    "seed161591/update_0064/policy.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/demo_following/shared_topology_distillation_v1/seed161593/"
    "step_3000"
)
TASKS = ("CarryBox", "KickBox")
SELECTED_OPTIONS = ("correct", "unrelated")
SOURCE_MOTION_IDS = {"CarryBox": 45, "KickBox": 21}
POLICY_DIM = 175
GOAL_CORE_DIM = 121
TACTILE_DIM = 2 * 12 * 20 * 25
ACTION_DIM = 29
PHASE_HORIZON = 650
TRAINING_FRAMES = 649


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=161593)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _relative(path: Path) -> str:
    return str(path.expanduser().absolute().relative_to(ROOT))


def _validated_trace(task: str, motion_id: int) -> tuple[Path, int]:
    prefix = task.lower()
    matches: list[tuple[Path, int]] = []
    for path in sorted(CORPUS.glob(f"{prefix}_shard*/TRACE.npz")):
        result = json.loads(path.with_name("RESULT.json").read_text(encoding="utf-8"))
        if (
            result.get("protocol")
            != "sugar_official_tracker_actual_contact_event_canary_v1"
            or result.get("passed") is not True
            or result.get("task_family") != task
        ):
            raise RuntimeError(f"unadmitted official Tracker trace: {path}")
        with np.load(path, allow_pickle=False) as archive:
            hits = np.flatnonzero(
                np.asarray(archive["source_motion_id_by_local_motion"], dtype=np.int64)
                == motion_id
            )
        if hits.size:
            if hits.size != 1:
                raise RuntimeError(f"motion {motion_id} is duplicated within {path}")
            matches.append((path, int(hits[0])))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {task} motion {motion_id}, found {matches}")
    return matches[0]


def load_sequences() -> dict[str, dict[str, np.ndarray]]:
    sequences: dict[str, dict[str, np.ndarray]] = {}
    for task in TASKS:
        path, index = _validated_trace(task, SOURCE_MOTION_IDS[task])
        with np.load(path, allow_pickle=False) as archive:
            core = np.asarray(
                archive["goal_policy_core_observation"][:, index], dtype=np.float32
            )
            action = np.asarray(archive["action"][:, index], dtype=np.float32)
            frame = np.asarray(archive["motion_frame"][:, index], dtype=np.int64)
            done = np.asarray(archive["done"][:, index], dtype=bool)
        if (
            core.shape != (700, GOAL_CORE_DIM)
            or action.shape != (700, ACTION_DIM)
            or not np.array_equal(frame, np.arange(1, 701, dtype=np.int64))
            or np.any(done)
            or not np.isfinite(core).all()
            or not np.isfinite(action).all()
        ):
            raise RuntimeError(f"official {task} sequence contract drift")
        sequences[task] = {
            "core": core,
            "action": action,
            "frame": frame,
            "trace_path": np.asarray(str(path)),
        }
    return sequences


def _policy_observation(core: torch.Tensor) -> torch.Tensor:
    if core.ndim != 2 or core.shape[1] != GOAL_CORE_DIM:
        raise ValueError("goal core geometry drift")
    return torch.cat(
        (
            core,
            torch.zeros(
                core.shape[0], POLICY_DIM - GOAL_CORE_DIM,
                device=core.device, dtype=core.dtype,
            ),
        ),
        dim=-1,
    )


@torch.no_grad()
def causal_conditions(
    core: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    scorer = FrozenPhaseAwareDemoEventScorer(
        num_envs=2,
        device=device,
        cfg=FrozenPhaseAwareDemoEventScorerCfg(
            runtime_config_path=str(RUNTIME_CONFIG),
            selected_option="correct",
            selected_options_by_env=SELECTED_OPTIONS,
            phase_horizon_steps=PHASE_HORIZON,
        ),
    )
    core_t = torch.from_numpy(core[:TRAINING_FRAMES]).to(device)
    observations = {
        "policy": _policy_observation(core_t[0:1].expand(2, -1).clone())
    }
    audit = scorer.begin(
        observations,
        initial_episode_steps=torch.ones(2, device=device, dtype=torch.long),
    )
    values = [scorer.actionable_conditioning().detach().clone()]
    done = torch.zeros(2, dtype=torch.bool, device=device)
    for index in range(1, TRAINING_FRAMES):
        observations = {
            "policy": _policy_observation(
                core_t[index : index + 1].expand(2, -1).clone()
            )
        }
        values.append(
            scorer.process_step(observations, done).actionable_conditioning.detach().clone()
        )
    conditioning = torch.stack(values, dim=0)
    if conditioning.shape != (
        TRAINING_FRAMES,
        2,
        ACTIONABLE_DEMO_CONDITIONING_DIM,
    ) or not torch.isfinite(conditioning).all():
        raise RuntimeError("causal conditioning geometry or finiteness drift")
    return conditioning, audit


def construct_policy(device: torch.device) -> SugarNativeZeroPreservingTactileActorCritic:
    dummy = {
        "policy": torch.zeros(2, POLICY_DIM, device=device),
        "critic": torch.zeros(2, POLICY_DIM, device=device),
        "demo_conditioning": torch.zeros(
            2, ACTIONABLE_DEMO_CONDITIONING_DIM, device=device
        ),
        "tactile_history": torch.zeros(2, TACTILE_DIM, device=device),
    }
    obs_groups = {
        "policy": ["policy", "demo_conditioning", "tactile_history"],
        "critic": ["critic", "demo_conditioning", "tactile_history"],
    }
    policy = SugarNativeZeroPreservingTactileActorCritic(
        dummy,
        obs_groups,
        ACTION_DIM,
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        tactile_obs_group="tactile_history",
        tactile_grid_shape=(20, 25),
        tactile_num_hands=2,
        tactile_channels_per_hand=12,
        tactile_encoder_channels=[32, 64, 64],
        tactile_embedding_dim=128,
    ).to(device)
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=True)
    if (
        checkpoint.get("protocol")
        != "sugar_stage_i_official_refiner_residual_multistep_checkpoint_v1"
        or int(checkpoint.get("iteration", -1)) != 64
    ):
        raise RuntimeError("shared actionable source checkpoint is not admitted update 64")
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    return policy


def build_dataset(
    sequences: dict[str, dict[str, np.ndarray]], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    policy_rows: list[torch.Tensor] = []
    condition_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    base_task_rows: list[torch.Tensor] = []
    selected_rows: list[torch.Tensor] = []
    condition_audits: dict[str, Any] = {}
    carry_actions = torch.from_numpy(
        sequences["CarryBox"]["action"][1 : TRAINING_FRAMES + 1]
    ).to(device)
    kick_actions = torch.from_numpy(
        sequences["KickBox"]["action"][1 : TRAINING_FRAMES + 1]
    ).to(device)
    target_actions = {
        "correct": torch.zeros_like(carry_actions),
        "unrelated": kick_actions - carry_actions,
    }
    for base_index, task in enumerate(TASKS):
        core = sequences[task]["core"][:TRAINING_FRAMES]
        condition, audit = causal_conditions(core, device)
        condition_audits[task] = audit
        base_policy = _policy_observation(torch.from_numpy(core).to(device))
        for selected_index, selected in enumerate(SELECTED_OPTIONS):
            policy_rows.append(base_policy)
            condition_rows.append(condition[:, selected_index])
            target_rows.append(target_actions[selected])
            base_task_rows.append(
                torch.full(
                    (TRAINING_FRAMES,), base_index, dtype=torch.long, device=device
                )
            )
            selected_rows.append(
                torch.full(
                    (TRAINING_FRAMES,), selected_index, dtype=torch.long, device=device
                )
            )
    dataset = {
        "policy": torch.cat(policy_rows, dim=0),
        "demo_conditioning": torch.cat(condition_rows, dim=0),
        "target_action": torch.cat(target_rows, dim=0),
        "base_task": torch.cat(base_task_rows, dim=0),
        "selected_option": torch.cat(selected_rows, dim=0),
    }
    expected = 2 * 2 * TRAINING_FRAMES
    if (
        dataset["policy"].shape != (expected, POLICY_DIM)
        or dataset["demo_conditioning"].shape
        != (expected, ACTIONABLE_DEMO_CONDITIONING_DIM)
        or dataset["target_action"].shape != (expected, ACTION_DIM)
    ):
        raise RuntimeError("topology distillation dataset geometry drift")
    return dataset, condition_audits


@torch.no_grad()
def evaluate_actor(
    policy: SugarNativeZeroPreservingTactileActorCritic,
    dataset: dict[str, torch.Tensor],
) -> dict[str, Any]:
    count = dataset["policy"].shape[0]
    tactile = torch.zeros(count, TACTILE_DIM, device=dataset["policy"].device)
    obs = {
        "policy": dataset["policy"],
        "critic": dataset["policy"],
        "demo_conditioning": dataset["demo_conditioning"],
        "tactile_history": tactile,
    }
    prediction = policy.act_inference(obs)
    error = prediction - dataset["target_action"]
    groups: dict[str, Any] = {}
    for base_index, base_task in enumerate(TASKS):
        for selected_index, selected in enumerate(SELECTED_OPTIONS):
            mask = (
                (dataset["base_task"] == base_index)
                & (dataset["selected_option"] == selected_index)
            )
            group_error = error[mask]
            groups[f"{base_task}__{selected}"] = {
                "mse": float(torch.square(group_error).mean()),
                "mae": float(torch.abs(group_error).mean()),
                "max_abs": float(torch.abs(group_error).max()),
            }
    per_base_condition_delta: dict[str, Any] = {}
    for base_index, base_task in enumerate(TASKS):
        correct = prediction[
            (dataset["base_task"] == base_index)
            & (dataset["selected_option"] == 0)
        ]
        unrelated = prediction[
            (dataset["base_task"] == base_index)
            & (dataset["selected_option"] == 1)
        ]
        difference = unrelated - correct
        per_base_condition_delta[base_task] = {
            "mean_abs": float(torch.abs(difference).mean()),
            "max_abs": float(torch.abs(difference).max()),
        }
    return {
        "mse": float(torch.square(error).mean()),
        "mae": float(torch.abs(error).mean()),
        "max_abs": float(torch.abs(error).max()),
        "groups": groups,
        "same_state_condition_action_delta": per_base_condition_delta,
        "all_finite": bool(torch.isfinite(prediction).all()),
    }


def main() -> None:
    args = parse_args()
    if args.steps != 3000:
        raise ValueError("the fixed topology overfit diagnostic is exactly 3000 steps")
    if args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("batch size and learning rate must be positive")
    output = args.output_dir.expanduser().resolve()
    experiments = (ROOT / "experiments").resolve()
    if experiments not in output.parents:
        raise ValueError("output must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    required = (CORPUS, RUNTIME_CONFIG, SOURCE_CHECKPOINT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    protocol = {
        "protocol": "sugar_shared_topology_distillation_v1",
        "question": (
            "Can the existing shared demo-conditioned SUGAR residual actor generate "
            "a new Kick contact topology when supplied official action-direction supervision?"
        ),
        "source_checkpoint": _relative(SOURCE_CHECKPOINT),
        "frozen_predictor_runtime": _relative(RUNTIME_CONFIG),
        "official_tracker_corpus": _relative(CORPUS),
        "base_state_tasks": list(TASKS),
        "selected_demo_options": list(SELECTED_OPTIONS),
        "source_motion_ids": SOURCE_MOTION_IDS,
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "actor_architecture": "existing_SUGAR_512_256_128_shared_checkpoint",
        "execution_action": "frozen_Carry_Refiner_action_plus_distilled_residual",
        "correct_residual_target": "exact_zero",
        "unrelated_residual_target": "official_Kick21_Tracker_minus_official_Carry45_Tracker",
        "future_action_enters_deployed_actor": False,
        "ground_truth_events_enter_deployed_actor": False,
        "diagnostic_scope": "fixed_action_direction_overfit_not_formal_policy_result",
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return
    if args.device.startswith("cuda") and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA training requires a retained compute allocation")
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    sequences = load_sequences()
    dataset, condition_audits = build_dataset(sequences, device)
    policy = construct_policy(device)
    before_state = {
        name: value.detach().clone() for name, value in policy.state_dict().items()
    }
    before = evaluate_actor(policy, dataset)
    parameters = tuple(policy.actor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    losses: list[float] = []
    count = dataset["policy"].shape[0]
    tactile = torch.zeros(args.batch_size, TACTILE_DIM, device=device)
    # Keep the existing encoder/normalizer state in evaluation mode while
    # optimizing the actor MLP; gradients remain enabled in eval mode.
    policy.eval()
    for step in range(args.steps):
        indices = torch.randint(
            0, count, (args.batch_size,), generator=generator, device="cpu"
        ).to(device)
        obs = {
            "policy": dataset["policy"][indices],
            "critic": dataset["policy"][indices],
            "demo_conditioning": dataset["demo_conditioning"][indices],
            "tactile_history": tactile,
        }
        prediction = policy.act_inference(obs)
        loss = torch.square(prediction - dataset["target_action"][indices]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 250 == 0:
            print(
                f"TOPOLOGY_DISTILL_STEP={step + 1}/{args.steps} "
                f"loss={losses[-1]:.8f}",
                flush=True,
            )
    policy.eval()
    after = evaluate_actor(policy, dataset)
    actor_delta = max(
        float(torch.abs(policy.state_dict()[name] - before_state[name]).max())
        for name in before_state
        if name.startswith("actor.")
    )
    critic_delta = max(
        float(torch.abs(policy.state_dict()[name] - before_state[name]).max())
        for name in before_state
        if name.startswith("critic.")
    )
    tactile_delta = max(
        float(torch.abs(policy.state_dict()[name] - before_state[name]).max())
        for name in before_state
        if "tactile_encoder" in name
    )
    checks = {
        "existing_shared_sugar_checkpoint_loaded": True,
        "official_carry_and_kick_tracker_actions_used": True,
        "same_states_paired_with_both_demo_conditions": True,
        "frozen_predictor_supplies_causal_conditioning": all(
            audit["model_frozen"]
            and audit["future_actual_events_used"] is False
            for audit in condition_audits.values()
        ),
        "all_training_values_finite": bool(
            np.isfinite(losses).all() and before["all_finite"] and after["all_finite"]
        ),
        "actor_changed": actor_delta > 0.0,
        "critic_unchanged": critic_delta == 0.0,
        "zero_tactile_encoder_unchanged": tactile_delta == 0.0,
        "distillation_mse_reduced_90_percent": after["mse"] <= 0.10 * before["mse"],
        "same_state_demo_swap_changes_action": all(
            value["mean_abs"] > 0.05
            for value in after["same_state_condition_action_delta"].values()
        ),
        "exactly_3000_optimizer_steps": len(losses) == 3000,
    }
    proof = {
        **protocol,
        "passed": all(checks.values()),
        "checks": checks,
        "dataset": {
            "samples": count,
            "policy_dim": POLICY_DIM,
            "actionable_conditioning_dim": ACTIONABLE_DEMO_CONDITIONING_DIM,
            "action_dim": ACTION_DIM,
            "training_frames_per_base_condition": TRAINING_FRAMES,
            "trace_paths": {
                task: str(sequences[task]["trace_path"].item()) for task in TASKS
            },
        },
        "before": before,
        "after": after,
        "optimization": {
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "minimum_loss": min(losses),
            "actor_parameter_max_abs_delta": actor_delta,
            "critic_parameter_max_abs_delta": critic_delta,
            "tactile_encoder_parameter_max_abs_delta": tactile_delta,
        },
        "condition_audits": condition_audits,
        "checkpoint": "policy.pt",
    }
    checkpoint = {
        "protocol": "sugar_shared_topology_distillation_checkpoint_v1",
        "iteration": args.steps,
        "policy_state_dict": policy.state_dict(),
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "protocol_config": protocol,
    }
    torch.save(checkpoint, output / "policy.pt")
    (output / "proof.json").write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "passed": proof["passed"],
        "before_mse": before["mse"],
        "after_mse": after["mse"],
        "checks": checks,
    }, indent=2, sort_keys=True))
    if not proof["passed"]:
        raise RuntimeError("shared topology distillation did not pass")


if __name__ == "__main__":
    main()
