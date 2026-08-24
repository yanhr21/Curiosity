#!/usr/bin/env python3
"""Train one demo-conditioned actor on the exact official 510-D Tracker input.

This faithful adaptation preserves the official SUGAR 512/256/128 actor and
29-D ActionManager command.  Its state input is the exact released Tracker
observation: generated command, five-frame proprioceptive/action histories,
projected gravity and object pose.  The only added input is the frozen causal
selected-demo condition.  Same-state Carry/Kick pairs force that condition to
select the released Carry45 or Kick21 Tracker action.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from train_shared_topology_distillation import (
    ACTIONABLE_DEMO_CONDITIONING_DIM,
    ACTION_DIM,
    CORPUS,
    ROOT,
    RUNTIME_CONFIG,
    SELECTED_OPTIONS,
    SOURCE_CHECKPOINT,
    SOURCE_MOTION_IDS,
    TACTILE_DIM,
    TASKS,
    TRAINING_FRAMES,
    _relative,
    _validated_trace,
    causal_conditions,
)
from sugar_rl.utils.sugar_native_curiosity_ppo import (
    SugarNativeZeroPreservingTactileActorCritic,
)


TRACKER_POLICY_DIM = 510
DEFAULT_OUTPUT = ROOT / (
    "experiments/demo_following/shared_full_tracker_v2/seed161601/step_3000"
)
OFFICIAL_TRACKERS = {
    task: ROOT / f"SUGAR/demo_ckpts/{task}/tracker.pt" for task in TASKS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=161601)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sequences() -> dict[str, dict[str, np.ndarray]]:
    sequences: dict[str, dict[str, np.ndarray]] = {}
    for task in TASKS:
        path, index = _validated_trace(task, SOURCE_MOTION_IDS[task])
        with np.load(path, allow_pickle=False) as archive:
            tracker = np.asarray(archive["policy_observation"][:, index], dtype=np.float32)
            core = np.asarray(
                archive["goal_policy_core_observation"][:, index], dtype=np.float32
            )
            action = np.asarray(archive["action"][:, index], dtype=np.float32)
            frame = np.asarray(archive["motion_frame"][:, index], dtype=np.int64)
            done = np.asarray(archive["done"][:, index], dtype=bool)
        if (
            tracker.shape != (700, TRACKER_POLICY_DIM)
            or core.shape != (700, 121)
            or action.shape != (700, ACTION_DIM)
            or not np.array_equal(frame, np.arange(1, 701, dtype=np.int64))
            or np.any(done)
            or not np.isfinite(tracker).all()
            or not np.isfinite(core).all()
            or not np.isfinite(action).all()
        ):
            raise RuntimeError(f"official {task} full-Tracker sequence contract drift")
        sequences[task] = {
            "tracker": tracker,
            "core": core,
            "action": action,
            "trace_path": np.asarray(str(path)),
        }
    return sequences


def construct_full_policy(
    device: torch.device,
) -> SugarNativeZeroPreservingTactileActorCritic:
    dummy = {
        "policy": torch.zeros(2, TRACKER_POLICY_DIM, device=device),
        "critic": torch.zeros(2, TRACKER_POLICY_DIM, device=device),
        "demo_conditioning": torch.zeros(
            2, ACTIONABLE_DEMO_CONDITIONING_DIM, device=device
        ),
        "tactile_history": torch.zeros(2, TACTILE_DIM, device=device),
    }
    groups = {
        "policy": ["policy", "demo_conditioning", "tactile_history"],
        "critic": ["critic", "demo_conditioning", "tactile_history"],
    }
    policy = SugarNativeZeroPreservingTactileActorCritic(
        dummy,
        groups,
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
    state = policy.state_dict()
    shared = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=True)[
        "policy_state_dict"
    ]
    # Preserve the admitted zero-tactile encoder and stochastic-state tensors
    # whenever their geometry is unchanged.
    for name, value in shared.items():
        if name in state and state[name].shape == value.shape and not name.startswith("actor."):
            state[name] = value.detach().clone()
    official = {
        task: torch.load(path, map_location=device, weights_only=True)["model_state_dict"]
        for task, path in OFFICIAL_TRACKERS.items()
    }
    for layer in (0, 2, 4, 6):
        weight_name = f"actor.{layer}.weight"
        bias_name = f"actor.{layer}.bias"
        carry_weight = official["CarryBox"][weight_name]
        kick_weight = official["KickBox"][weight_name]
        carry_bias = official["CarryBox"][bias_name]
        kick_bias = official["KickBox"][bias_name]
        if layer == 0:
            state[weight_name].zero_()
            state[weight_name][:, :TRACKER_POLICY_DIM] = 0.5 * (
                carry_weight + kick_weight
            )
            old_condition_start = 175
            old_condition_stop = old_condition_start + ACTIONABLE_DEMO_CONDITIONING_DIM
            new_condition_start = TRACKER_POLICY_DIM
            new_condition_stop = new_condition_start + ACTIONABLE_DEMO_CONDITIONING_DIM
            state[weight_name][:, new_condition_start:new_condition_stop] = shared[
                weight_name
            ][:, old_condition_start:old_condition_stop]
            state[weight_name][:, new_condition_stop:] = shared[weight_name][
                :, old_condition_stop:
            ]
        else:
            state[weight_name] = 0.5 * (carry_weight + kick_weight)
        state[bias_name] = 0.5 * (carry_bias + kick_bias)
    policy.load_state_dict(state, strict=True)
    return policy


def build_dataset(
    sequences: dict[str, dict[str, np.ndarray]], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    state_rows: list[torch.Tensor] = []
    condition_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    base_rows: list[torch.Tensor] = []
    selected_rows: list[torch.Tensor] = []
    audits: dict[str, object] = {}
    targets = {
        "correct": torch.from_numpy(
            sequences["CarryBox"]["action"][1 : TRAINING_FRAMES + 1]
        ).to(device),
        "unrelated": torch.from_numpy(
            sequences["KickBox"]["action"][1 : TRAINING_FRAMES + 1]
        ).to(device),
    }
    for base_index, task in enumerate(TASKS):
        state = torch.from_numpy(sequences[task]["tracker"][:TRAINING_FRAMES]).to(device)
        condition, audit = causal_conditions(
            sequences[task]["core"][:TRAINING_FRAMES], device
        )
        audits[task] = audit
        for selected_index, selected in enumerate(SELECTED_OPTIONS):
            state_rows.append(state)
            condition_rows.append(condition[:, selected_index])
            target_rows.append(targets[selected])
            base_rows.append(
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
        "policy": torch.cat(state_rows),
        "demo_conditioning": torch.cat(condition_rows),
        "target_action": torch.cat(target_rows),
        "base_task": torch.cat(base_rows),
        "selected_option": torch.cat(selected_rows),
    }
    expected = len(TASKS) * len(SELECTED_OPTIONS) * TRAINING_FRAMES
    if dataset["policy"].shape != (expected, TRACKER_POLICY_DIM):
        raise RuntimeError("full Tracker dataset geometry drift")
    for base_index in range(len(TASKS)):
        start = base_index * len(SELECTED_OPTIONS) * TRAINING_FRAMES
        if not torch.equal(
            dataset["policy"][start : start + TRAINING_FRAMES],
            dataset["policy"][start + TRAINING_FRAMES : start + 2 * TRAINING_FRAMES],
        ):
            raise RuntimeError("same-state demo pair drift")
    return dataset, audits


@torch.no_grad()
def evaluate(policy, dataset: dict[str, torch.Tensor]) -> dict[str, object]:
    count = dataset["policy"].shape[0]
    state = dataset["policy"]
    obs = {
        "policy": state,
        "critic": state,
        "demo_conditioning": dataset["demo_conditioning"],
        "tactile_history": torch.zeros(count, TACTILE_DIM, device=state.device),
    }
    prediction = policy.act_inference(obs)
    error = prediction - dataset["target_action"]
    groups: dict[str, object] = {}
    deltas: dict[str, object] = {}
    for base_index, task in enumerate(TASKS):
        predictions = []
        for selected_index, selected in enumerate(SELECTED_OPTIONS):
            mask = (dataset["base_task"] == base_index) & (
                dataset["selected_option"] == selected_index
            )
            group_error = error[mask]
            groups[f"{task}__{selected}"] = {
                "mse": float(torch.square(group_error).mean()),
                "mae": float(torch.abs(group_error).mean()),
                "max_abs": float(torch.abs(group_error).max()),
            }
            predictions.append(prediction[mask])
        difference = predictions[1] - predictions[0]
        deltas[task] = {
            "mean_abs": float(torch.abs(difference).mean()),
            "max_abs": float(torch.abs(difference).max()),
        }
    return {
        "mse": float(torch.square(error).mean()),
        "mae": float(torch.abs(error).mean()),
        "max_abs": float(torch.abs(error).max()),
        "groups": groups,
        "same_state_condition_action_delta": deltas,
        "all_finite": bool(torch.isfinite(prediction).all()),
    }


def main() -> None:
    args = parse_args()
    if args.steps != 3000 or args.batch_size != 512:
        raise ValueError("full Tracker fit is fixed to 3000 steps and batch 512")
    output = args.output_dir.expanduser().resolve()
    if (ROOT / "experiments").resolve() not in output.parents:
        raise ValueError("output must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    for path in (CORPUS, RUNTIME_CONFIG, SOURCE_CHECKPOINT, *OFFICIAL_TRACKERS.values()):
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = {
        "protocol": "sugar_shared_full_tracker_distillation_v1",
        "state_observation_contract": "exact_official_510D_Tracker_policy_observation",
        "actor_architecture": "official_SUGAR_512_256_128_plus_causal_demo_condition",
        "action_semantics": "absolute_29D_ActionManager_command",
        "official_tracker_checkpoints": {
            task: _relative(path) for task, path in OFFICIAL_TRACKERS.items()
        },
        "source_motion_ids": SOURCE_MOTION_IDS,
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "same_state_paired_with_both_conditions": True,
        "future_action_enters_deployed_actor": False,
        "ground_truth_events_enter_deployed_actor": False,
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
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    sequences = load_sequences()
    dataset, condition_audits = build_dataset(sequences, device)
    policy = construct_full_policy(device)
    before_state = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    before = evaluate(policy, dataset)
    parameters = tuple(policy.actor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    tactile = torch.zeros(args.batch_size, TACTILE_DIM, device=device)
    losses: list[float] = []
    count = dataset["policy"].shape[0]
    policy.eval()
    for step in range(args.steps):
        indices = torch.randint(
            0, count, (args.batch_size,), generator=generator, device="cpu"
        ).to(device)
        state = dataset["policy"][indices]
        obs = {
            "policy": state,
            "critic": state,
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
                f"FULL_TRACKER_STEP={step + 1}/{args.steps} loss={losses[-1]:.8f}",
                flush=True,
            )
    policy.eval()
    after = evaluate(policy, dataset)
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
        "official_carry_and_kick_tracker_weights_loaded": True,
        "exact_510d_tracker_observation_used": True,
        "same_states_paired_with_both_conditions": True,
        "frozen_predictor_conditioning_is_causal": all(
            audit["model_frozen"] and audit["future_actual_events_used"] is False
            for audit in condition_audits.values()
        ),
        "actor_changed": actor_delta > 0.0,
        "critic_unchanged": critic_delta == 0.0,
        "tactile_encoder_unchanged": tactile_delta == 0.0,
        "mse_reduced_90_percent": after["mse"] <= 0.10 * before["mse"],
        "same_state_demo_swap_changes_action": all(
            value["mean_abs"] > 0.05
            for value in after["same_state_condition_action_delta"].values()
        ),
        "all_values_finite": bool(
            np.isfinite(losses).all() and before["all_finite"] and after["all_finite"]
        ),
        "exactly_3000_optimizer_steps": len(losses) == 3000,
    }
    proof = {
        **protocol,
        "passed": all(checks.values()),
        "checks": checks,
        "before": before,
        "after": after,
        "dataset": {
            "samples": count,
            "tracker_policy_dim": TRACKER_POLICY_DIM,
            "actionable_conditioning_dim": ACTIONABLE_DEMO_CONDITIONING_DIM,
            "action_dim": ACTION_DIM,
            "trace_paths": {
                task: str(sequences[task]["trace_path"].item()) for task in TASKS
            },
        },
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
        "protocol": "sugar_shared_full_tracker_checkpoint_v1",
        "iteration": 3000,
        "policy_state_dict": policy.state_dict(),
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
        raise RuntimeError("shared full Tracker distillation failed")


if __name__ == "__main__":
    main()
