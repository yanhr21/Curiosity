#!/usr/bin/env python3
"""Measure the current serious tactile fusion on the canonical CarryBox trace.

This uses the exact Plan-13 spatial encoder and official Refiner warm start.
It checks the causal four-frame serialization and reports how real recorded
TacSL fields enter the appended actor columns.  It is a numerical fusion
diagnostic, not a training-benefit or closed-loop behavior result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sugar_rl.utils.reference_only_tactile_actor_critic import (
    ReferenceOnlyTactileActorCritic,
)


POLICY_WIDTH = 890
TACTILE_WIDTH = 324000
ACTION_WIDTH = 29
HISTORY = 4
NORMAL_SCALE_N = 0.5768324136734009
SHEAR_SCALE_N = 0.5144117593765258


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"minimum": 0.0, "median": 0.0, "q95": 0.0, "maximum": 0.0}
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def make_policy(device: torch.device) -> ReferenceOnlyTactileActorCritic:
    observations = {
        "policy": torch.zeros(1, POLICY_WIDTH, device=device),
        "native_whole_hand_tactile_history": torch.zeros(
            1, TACTILE_WIDTH, device=device
        ),
        "critic": torch.zeros(1, POLICY_WIDTH, device=device),
        "teacher": torch.zeros(1, POLICY_WIDTH, device=device),
    }
    return ReferenceOnlyTactileActorCritic(
        obs=observations,
        obs_groups={
            "policy": ["policy", "native_whole_hand_tactile_history"],
            "critic": ["critic"],
            "teacher": ["teacher"],
        },
        num_actions=ACTION_WIDTH,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=0.5,
        tactile_obs_group="native_whole_hand_tactile_history",
        tactile_grid_shape=(20, 25),
        tactile_num_hands=2,
        tactile_channels_per_hand=324,
        tactile_encoder_channels=(32, 64, 64),
        tactile_embedding_dim=128,
    ).to(device)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    trace_path = args.trace.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (checkpoint_path, trace_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(trace_path, allow_pickle=False) as trace:
        normal = np.asarray(trace["normal_force"], dtype=np.float32)
        shear = np.asarray(trace["signed_shear"], dtype=np.float32)
    expected_normal = (660, 2, 27, 20, 25)
    expected_shear = (*expected_normal, 2)
    if normal.shape != expected_normal or shear.shape != expected_shear:
        raise RuntimeError(
            f"canonical trace shape mismatch: normal={normal.shape}, shear={shear.shape}"
        )

    # Current field: [frame, hand, patch, channel, row, column].
    fields = np.stack(
        (
            normal / NORMAL_SCALE_N,
            shear[..., 0] / SHEAR_SCALE_N,
            shear[..., 1] / SHEAR_SCALE_N,
        ),
        axis=3,
    ).astype(np.float32, copy=False)
    if not np.isfinite(fields).all():
        raise RuntimeError("canonical tactile field contains non-finite values")

    frame_count = fields.shape[0]
    frame_ids = np.arange(frame_count, dtype=np.int64)
    history_ids = np.stack(
        [np.maximum(frame_ids - offset, 0) for offset in (3, 2, 1, 0)],
        axis=1,
    )
    current_contact = np.any(fields != 0.0, axis=(1, 2, 3, 4, 5))
    history_contact = np.any(
        fields[history_ids] != 0.0, axis=(1, 2, 3, 4, 5, 6)
    )
    active_by_hand = np.count_nonzero(np.abs(normal) > 1.0e-6, axis=(2, 3, 4))

    device = torch.device(args.device)
    torch.manual_seed(13011)
    policy = make_policy(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    warm_start = policy.load_sugar_warm_start(checkpoint["model_state_dict"])
    policy.eval()

    first_layer = policy.actor._modules["0"]
    tactile_columns = first_layer.weight[:, POLICY_WIDTH:]
    feature_l2: list[np.ndarray] = []
    preactivation_l2: list[np.ndarray] = []
    preactivation_abs_max: list[np.ndarray] = []
    action_delta_l2: list[np.ndarray] = []
    action_delta_abs_max: list[np.ndarray] = []
    normalized_rms: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, frame_count, args.batch_size):
            end = min(start + args.batch_size, frame_count)
            # [batch, history, hand, patch, channel, row, column]
            history = fields[history_ids[start:end]]
            actor_layout = np.ascontiguousarray(
                history.transpose(0, 2, 1, 3, 4, 5, 6).reshape(end - start, -1)
            )
            if actor_layout.shape[1] != TACTILE_WIDTH:
                raise RuntimeError(
                    f"serialized tactile width is {actor_layout.shape[1]}"
                )
            tactile = torch.from_numpy(actor_layout).to(device=device)
            features = policy.actor_tactile_encoder(tactile)
            preactivation = torch.nn.functional.linear(
                features, tactile_columns, bias=None
            )
            base = torch.zeros(end - start, POLICY_WIDTH, device=device)
            zero_features = torch.zeros_like(features)
            action_with_tactile = policy.actor(torch.cat((base, features), dim=-1))
            action_at_zero = policy.actor(
                torch.cat((base, zero_features), dim=-1)
            )
            action_delta = action_with_tactile - action_at_zero

            feature_l2.append(features.norm(dim=-1).cpu().numpy())
            preactivation_l2.append(preactivation.norm(dim=-1).cpu().numpy())
            preactivation_abs_max.append(
                preactivation.abs().amax(dim=-1).cpu().numpy()
            )
            action_delta_l2.append(action_delta.norm(dim=-1).cpu().numpy())
            action_delta_abs_max.append(
                action_delta.abs().amax(dim=-1).cpu().numpy()
            )
            normalized_rms.append(
                tactile.square().mean(dim=-1).sqrt().cpu().numpy()
            )

    metrics = {
        "normalized_input_rms": np.concatenate(normalized_rms),
        "feature_l2": np.concatenate(feature_l2),
        "first_layer_tactile_preactivation_l2": np.concatenate(
            preactivation_l2
        ),
        "first_layer_tactile_preactivation_abs_max": np.concatenate(
            preactivation_abs_max
        ),
        "zero_base_action_delta_l2": np.concatenate(action_delta_l2),
        "zero_base_action_delta_abs_max": np.concatenate(action_delta_abs_max),
    }
    zero_history_mask = ~history_contact
    first_contact = (
        int(np.flatnonzero(current_contact)[0]) if current_contact.any() else None
    )

    # Use actual recorded contact histories for the same trainability check
    # that the live PPO gate will later repeat online.
    contact_probe_ids = np.flatnonzero(history_contact)[:8]
    contact_history = fields[history_ids[contact_probe_ids]]
    contact_actor_layout = np.ascontiguousarray(
        contact_history.transpose(0, 2, 1, 3, 4, 5, 6).reshape(
            len(contact_probe_ids), -1
        )
    )
    finetune = policy.configure_tactile_actor_finetune()
    policy.zero_grad(set_to_none=True)
    contact_tactile = torch.from_numpy(contact_actor_layout).to(device=device)
    contact_base = torch.zeros(len(contact_probe_ids), POLICY_WIDTH, device=device)
    contact_obs = {
        "policy": contact_base,
        "native_whole_hand_tactile_history": contact_tactile,
        "critic": contact_base,
        "teacher": contact_base,
    }
    policy.act_inference(contact_obs).square().mean().backward()
    first_gradient = policy.actor._modules["0"].weight.grad
    if first_gradient is None:
        raise RuntimeError("canonical contact probe produced no actor.0 gradient")
    base_gradient_abs_max = float(
        first_gradient[:, :POLICY_WIDTH].abs().max().item()
    )
    tactile_gradient_l2 = float(
        first_gradient[:, POLICY_WIDTH:].norm().item()
    )
    encoder_gradient_l2 = float(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in policy.actor_tactile_encoder.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ).sqrt().item()
    )
    checks = {
        "canonical_shapes_exact": normal.shape == expected_normal
        and shear.shape == expected_shear,
        "causal_history_starts_with_four_copies": bool(
            np.all(history_ids[0] == np.array([0, 0, 0, 0]))
        ),
        "causal_history_frame_three_is_0_1_2_3": bool(
            np.all(history_ids[3] == np.array([0, 1, 2, 3]))
        ),
        "actor_serialization_width_is_324000": TACTILE_WIDTH
        == 2 * 4 * 27 * 3 * 20 * 25,
        "trace_contains_physical_contact": bool(current_contact.any()),
        "zero_frames_keep_zero_encoder_and_action_delta": bool(
            np.all(metrics["feature_l2"][zero_history_mask] == 0.0)
            and np.all(
                metrics["zero_base_action_delta_abs_max"][zero_history_mask]
                == 0.0
            )
        ),
        "contact_reaches_spatial_encoder": bool(
            np.all(metrics["feature_l2"][history_contact] > 0.0)
        ),
        "all_fusion_metrics_are_finite": all(
            np.isfinite(values).all() for values in metrics.values()
        ),
        "official_zero_tactile_actor_is_exact": warm_start[
            "actor_zero_tactile_max_abs_error"
        ]
        <= 1.0e-6,
        "canonical_contact_keeps_base_column_gradient_zero": (
            base_gradient_abs_max == 0.0
        ),
        "canonical_contact_reaches_tactile_actor_columns": (
            tactile_gradient_l2 > 0.0
        ),
        "canonical_contact_reaches_spatial_encoder_gradient": (
            encoder_gradient_l2 > 0.0
        ),
    }

    result = {
        "semantics": (
            "serious Plan-13 late-fusion numerical diagnostic on the canonical "
            "physical CarryBox taxel trace; not closed-loop training benefit"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": checkpoint.get("iter"),
        "trace": str(trace_path),
        "frames": frame_count,
        "current_contact_frames": int(np.count_nonzero(current_contact)),
        "four_frame_history_contact_frames": int(
            np.count_nonzero(history_contact)
        ),
        "first_contact_frame": first_contact,
        "maximum_active_taxels_by_hand": active_by_hand.max(axis=0).tolist(),
        "normalization": {
            "normal_scale_n": NORMAL_SCALE_N,
            "signed_xy_shear_scale_n": SHEAR_SCALE_N,
            "clipping": False,
        },
        "fusion": {
            "raw_actor_width": TACTILE_WIDTH,
            "layout": [
                "hand",
                "history",
                "patch",
                "channel",
                "row",
                "column",
            ],
            "embedding_width": 256,
            "warm_start_tactile_gain": warm_start["tactile_gain"],
            "method": "per-hand shared spatial encoder, then late concatenate before actor.0",
        },
        "all_frames": {
            name: quantiles(values) for name, values in metrics.items()
        },
        "current_contact_frames_only": {
            name: quantiles(values[current_contact])
            for name, values in metrics.items()
        },
        "nonzero_history_frames_only": {
            name: quantiles(values[history_contact])
            for name, values in metrics.items()
        },
        "warm_start": warm_start,
        "finetune": finetune,
        "canonical_contact_gradient_probe": {
            "source_frames": contact_probe_ids.tolist(),
            "actor_base_columns_abs_max": base_gradient_abs_max,
            "actor_tactile_columns_l2": tactile_gradient_l2,
            "spatial_encoder_l2": encoder_gradient_l2,
            "loss": "mean squared deterministic actor output at zero base observation",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation_boundary": (
            "zero-base action delta measures adapter scale under a standardized "
            "base input; it is not the action difference of the recorded rollout"
        ),
    }
    if not result["passed"]:
        raise RuntimeError(json.dumps(result, indent=2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_trace_path = output_path.with_suffix(".npz")
    np.savez_compressed(
        metric_trace_path,
        frame=np.arange(frame_count, dtype=np.int32),
        current_contact=current_contact,
        history_contact=history_contact,
        active_taxels_by_hand=active_by_hand,
        **metrics,
    )
    result["metric_trace"] = str(metric_trace_path)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
