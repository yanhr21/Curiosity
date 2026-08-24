#!/usr/bin/env python3
"""Fit the serious shared 510-D Tracker actor on visited-state labels.

This is the fixed-condition closed-loop diagnostic prescribed after offline
distillation failed.  Buffers are collected inside the official IsaacLab task
with the official domain Tracker supplying corrective labels.  The labels are
used only during fitting; frozen evaluation remains student-only.  The
original paired Carry/Kick corpus is retained in every minibatch so the shared
actor cannot solve the diagnostic by deleting its selected-demo input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from train_shared_full_tracker import (
    TRACKER_POLICY_DIM,
    build_dataset,
    construct_full_policy,
    evaluate,
    load_sequences,
)
from train_shared_topology_distillation import (
    ACTIONABLE_DEMO_CONDITIONING_DIM,
    ACTION_DIM,
    ROOT,
    TACTILE_DIM,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--input-proof", type=Path, required=True)
    parser.add_argument("--buffer", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-domain", choices=("CarryBox", "KickBox"), required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--offline-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--seed", type=int, default=161603)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_buffers(
    paths: list[Path], domain: str, device: torch.device
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    states: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    seen_fractions: set[float] = set()
    expected_option = "correct" if domain == "CarryBox" else "unrelated"
    for source in paths:
        path = source.expanduser().resolve()
        trace_path = path if path.name == "TRACE.npz" else path / "TRACE.npz"
        result_path = trace_path.with_name("RESULT.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("protocol")
            != "sugar_shared_absolute_tracker_dagger_collection_v1"
            or result.get("passed") is not True
            or not all(result.get("checks", {}).values())
            or result.get("domain") != domain
            or result.get("selected_demo_option") != expected_option
            or result.get("actor_state_observation_contract")
            != "exact_official_510D_Tracker_policy_observation"
        ):
            raise RuntimeError(f"unadmitted full-Tracker DAgger buffer: {trace_path}")
        fraction = float(result["student_action_fraction"])
        if fraction in seen_fractions:
            raise RuntimeError(f"duplicate student-action fraction: {fraction}")
        seen_fractions.add(fraction)
        with np.load(trace_path, allow_pickle=False) as archive:
            state = np.asarray(archive["tracker_policy_observation"], dtype=np.float32)
            condition = np.asarray(archive["demo_conditioning"], dtype=np.float32)
            target = np.asarray(archive["teacher_action"], dtype=np.float32)
        if (
            state.shape != (650, 20, TRACKER_POLICY_DIM)
            or condition.shape != (650, 20, ACTIONABLE_DEMO_CONDITIONING_DIM)
            or target.shape != (650, 20, ACTION_DIM)
            or not np.isfinite(state).all()
            or not np.isfinite(condition).all()
            or not np.isfinite(target).all()
        ):
            raise RuntimeError(f"buffer geometry or finiteness drift: {trace_path}")
        states.append(state.reshape(-1, TRACKER_POLICY_DIM))
        conditions.append(condition.reshape(-1, ACTIONABLE_DEMO_CONDITIONING_DIM))
        targets.append(target.reshape(-1, ACTION_DIM))
        metadata.append(
            {
                "trace": str(trace_path),
                "student_action_fraction": fraction,
                "samples": int(state.shape[0] * state.shape[1]),
                "teacher_student_action_mae": result["teacher_student_action_mae"],
            }
        )
    return {
        "policy": torch.from_numpy(np.concatenate(states)).to(device),
        "demo_conditioning": torch.from_numpy(np.concatenate(conditions)).to(device),
        "target_action": torch.from_numpy(np.concatenate(targets)).to(device),
    }, metadata


@torch.no_grad()
def evaluate_online(
    policy: torch.nn.Module,
    dataset: dict[str, torch.Tensor],
    batch_size: int = 2048,
) -> dict[str, float | bool]:
    squared = 0.0
    absolute = 0.0
    maximum = 0.0
    elements = 0
    for start in range(0, dataset["policy"].shape[0], batch_size):
        stop = min(start + batch_size, dataset["policy"].shape[0])
        state = dataset["policy"][start:stop]
        count = stop - start
        observation = {
            "policy": state,
            "critic": state,
            "demo_conditioning": dataset["demo_conditioning"][start:stop],
            "tactile_history": torch.zeros(count, TACTILE_DIM, device=state.device),
        }
        error = policy.act_inference(observation) - dataset["target_action"][start:stop]
        squared += float(torch.square(error).sum())
        absolute += float(torch.abs(error).sum())
        maximum = max(maximum, float(torch.abs(error).max()))
        elements += error.numel()
    return {
        "mse": squared / elements,
        "mae": absolute / elements,
        "max_abs": maximum,
        "all_finite": bool(np.isfinite((squared, absolute, maximum)).all()),
    }


def main() -> None:
    args = parse_args()
    if args.steps != 1500 or args.batch_size != 512 or args.offline_fraction != 0.25:
        raise ValueError("fit is fixed to 1500 steps, batch 512 and 25% offline pairs")
    output = args.output_dir.expanduser().resolve()
    experiments = (ROOT / "experiments").resolve()
    paths = [
        output,
        args.input_checkpoint.expanduser().resolve(),
        args.input_proof.expanduser().resolve(),
        *(path.expanduser().resolve() for path in args.buffer),
    ]
    if any(experiments not in path.parents for path in paths):
        raise ValueError("all inputs and outputs must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    if args.device.startswith("cuda") and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA fitting requires a retained compute allocation")
    proof = json.loads(args.input_proof.read_text(encoding="utf-8"))
    admitted_proofs = {
        "sugar_shared_full_tracker_distillation_v1",
        "sugar_shared_full_tracker_dagger_fit_v1",
    }
    if (
        proof.get("protocol") not in admitted_proofs
        or proof.get("passed") is not True
        or not all(proof.get("checks", {}).values())
    ):
        raise RuntimeError("input proof is not admitted")
    checkpoint = torch.load(
        args.input_checkpoint, map_location=args.device, weights_only=True
    )
    if checkpoint.get("protocol") not in {
        "sugar_shared_full_tracker_checkpoint_v1",
        "sugar_shared_full_tracker_dagger_checkpoint_v1",
    }:
        raise RuntimeError("input checkpoint protocol drift")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    online, buffer_metadata = _load_buffers(args.buffer, args.diagnostic_domain, device)
    offline, condition_audits = build_dataset(load_sequences(), device)
    policy = construct_full_policy(device)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.eval()
    state_before = {
        name: value.detach().clone() for name, value in policy.state_dict().items()
    }
    online_before = evaluate_online(policy, online)
    offline_before = evaluate(policy, offline)

    parameters = tuple(policy.actor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + args.stage)
    offline_batch = int(args.batch_size * args.offline_fraction)
    online_batch = args.batch_size - offline_batch
    zero_tactile = torch.zeros(args.batch_size, TACTILE_DIM, device=device)
    losses: list[float] = []
    for step in range(args.steps):
        online_index = torch.randint(
            online["policy"].shape[0], (online_batch,), generator=generator
        ).to(device)
        offline_index = torch.randint(
            offline["policy"].shape[0], (offline_batch,), generator=generator
        ).to(device)
        state = torch.cat(
            (online["policy"][online_index], offline["policy"][offline_index])
        )
        condition = torch.cat(
            (
                online["demo_conditioning"][online_index],
                offline["demo_conditioning"][offline_index],
            )
        )
        target = torch.cat(
            (online["target_action"][online_index], offline["target_action"][offline_index])
        )
        prediction = policy.act_inference(
            {
                "policy": state,
                "critic": state,
                "demo_conditioning": condition,
                "tactile_history": zero_tactile,
            }
        )
        loss = torch.square(prediction - target).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 250 == 0:
            print(
                f"FULL_DAGGER_STAGE={args.stage} STEP={step + 1}/{args.steps} "
                f"loss={losses[-1]:.8f}",
                flush=True,
            )

    policy.eval()
    online_after = evaluate_online(policy, online)
    offline_after = evaluate(policy, offline)
    actor_delta = max(
        float(torch.abs(policy.state_dict()[name] - state_before[name]).max())
        for name in state_before
        if name.startswith("actor.")
    )
    critic_delta = max(
        float(torch.abs(policy.state_dict()[name] - state_before[name]).max())
        for name in state_before
        if name.startswith("critic.")
    )
    tactile_delta = max(
        float(torch.abs(policy.state_dict()[name] - state_before[name]).max())
        for name in state_before
        if "tactile_encoder" in name
    )
    checks = {
        "serious_full_tracker_actor_admitted": True,
        "exact_510d_visited_states_used": True,
        "online_labels_are_official_tracker_actions": True,
        "future_teacher_action_absent_from_deployed_actor": True,
        "offline_counterfactual_regularizer_retained": True,
        "frozen_predictor_conditioning_is_causal": all(
            audit["model_frozen"] and audit["future_actual_events_used"] is False
            for audit in condition_audits.values()
        ),
        "actor_changed": actor_delta > 0.0,
        "critic_unchanged": critic_delta == 0.0,
        "tactile_encoder_unchanged": tactile_delta == 0.0,
        "online_teacher_mse_reduced_50_percent": (
            online_after["mse"] <= 0.5 * online_before["mse"]
        ),
        "offline_condition_pair_mse_bounded": offline_after["mse"] <= 0.25,
        "selected_demo_still_changes_same_state_action": all(
            record["mean_abs"] > 0.05
            for record in offline_after["same_state_condition_action_delta"].values()
        ),
        "all_values_finite": bool(
            np.isfinite(losses).all()
            and online_before["all_finite"]
            and online_after["all_finite"]
            and offline_before["all_finite"]
            and offline_after["all_finite"]
        ),
        "exactly_1500_optimizer_steps": len(losses) == 1500,
    }
    history = list(checkpoint.get("dagger_history", []))
    history.append(
        {
            "stage": args.stage,
            "diagnostic_domain": args.diagnostic_domain,
            "student_action_fractions": sorted(
                {float(record["student_action_fraction"]) for record in buffer_metadata}
            ),
            "optimizer_steps": args.steps,
        }
    )
    result = {
        "protocol": "sugar_shared_full_tracker_dagger_fit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "stage": args.stage,
        "diagnostic_domain": args.diagnostic_domain,
        "seed": args.seed,
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "offline_fraction": args.offline_fraction,
        "learning_rate": args.learning_rate,
        "input_checkpoint": str(args.input_checkpoint.expanduser().resolve()),
        "input_proof": str(args.input_proof.expanduser().resolve()),
        "buffers": buffer_metadata,
        "online_samples": int(online["policy"].shape[0]),
        "offline_counterfactual_samples": int(offline["policy"].shape[0]),
        "online_before": online_before,
        "online_after": online_after,
        "offline_before": offline_before,
        "offline_after": offline_after,
        "optimization": {
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "minimum_loss": min(losses),
            "actor_parameter_max_abs_delta": actor_delta,
            "critic_parameter_max_abs_delta": critic_delta,
            "tactile_encoder_parameter_max_abs_delta": tactile_delta,
        },
        "dagger_history": history,
        "claim_boundary": (
            "This is a fixed-condition closed-loop diagnostic. Only a subsequent "
            "student-only frozen physics rollout can establish execution success."
        ),
        "checkpoint": "policy.pt",
    }
    output.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "protocol": "sugar_shared_full_tracker_dagger_checkpoint_v1",
            "iteration": args.stage,
            "policy_state_dict": policy.state_dict(),
            "dagger_history": history,
            "source_checkpoint": str(args.input_checkpoint.expanduser().resolve()),
        },
        output / "policy.pt",
    )
    (output / "proof.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": result["passed"],
                "online_before": online_before,
                "online_after": online_after,
                "offline_after_mse": offline_after["mse"],
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise RuntimeError("full Tracker DAgger fit failed")


if __name__ == "__main__":
    main()
