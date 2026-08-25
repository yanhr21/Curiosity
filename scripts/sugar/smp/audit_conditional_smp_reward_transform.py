#!/usr/bin/env python3
"""Audit a training-data-anchored official SMP reward on recovery traces.

The released MimicKit transform is ``exp(-normalized_sds * scale)``.  Its
default scale 6 saturates for SUGAR recovery states, so this audit determines
one scale from prior-training data only: the calibration median maps to 0.5.
No recovery outcome is used to choose the transform or its scale.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from audit_cross_skill_recovery_tinymdm import (  # noqa: E402
    load_feature_complete_trace,
    outcome_labels,
)
from run_conditional_taskwide_tinymdm import (  # noqa: E402
    CLASS_IDS,
    OUTPUT_ROOT,
    load_shared_prior,
)
from run_selected_demo_tinymdm import (  # noqa: E402
    DIFFUSION_STEPS,
    SCORE_SEEDS,
    atomic_json,
    require_compute_gpu,
)
from sugar_g1_box_schema import FEATURE_DIM, WINDOW_SIZE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def stable_sds_losses(
    model: Any,
    windows: np.ndarray,
    class_id: int,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, windows.shape[0], chunk_size):
        batch = torch.as_tensor(windows[start : start + chunk_size], device=device)
        normalized = model.normalize(batch).reshape(batch.shape[0], -1)
        labels = torch.full(
            (batch.shape[0],), class_id, dtype=torch.long, device=device
        )
        repeats = []
        for seed in SCORE_SEEDS:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            repeats.append(
                model.ESM_SDS_loss(
                    normalized,
                    t_lst=list(DIFFUSION_STEPS),
                    class_labels=labels,
                )
                .cpu()
                .numpy()
            )
        chunks.append(np.mean(np.stack(repeats), axis=0).astype(np.float32))
    output = np.concatenate(chunks)
    if output.shape != (windows.shape[0], len(DIFFUSION_STEPS)):
        raise RuntimeError(f"SDS loss geometry {output.shape}")
    return output


def reward_from_losses(
    losses: np.ndarray, mean_abs: np.ndarray, scale: float
) -> np.ndarray:
    normalized = np.mean(losses / mean_abs[None, :], axis=-1)
    return np.exp(-normalized * scale).astype(np.float32)


def causal_reward_audit(
    reward: np.ndarray,
    raw_labels: dict[str, np.ndarray],
    outcomes: dict[str, np.ndarray],
) -> dict[str, Any]:
    safe = outcomes["safe_kick"]
    fall = outcomes["physical_fall"]
    early = reward[:50].mean(axis=0)
    early_safe_higher = float(np.mean(early[safe, None] > early[fall][None, :]))
    root = raw_labels["robot_root_state_w"]
    root_loss = root[0, :, 2][None, :] - root[:, :, 2]
    prefall = []
    for profile in np.flatnonzero(fall):
        onset = int(np.flatnonzero(root_loss[:, profile] >= 0.35)[0])
        last_window = onset - WINDOW_SIZE
        first_window = max(0, last_window - 19)
        if last_window < 0:
            prefall.append(
                {"profile": int(profile), "fall_onset_step": onset, "available": False}
            )
            continue
        fall_value = float(reward[first_window : last_window + 1, profile].mean())
        safe_values = reward[first_window : last_window + 1, safe].mean(axis=0)
        prefall.append(
            {
                "profile": int(profile),
                "fall_onset_step": onset,
                "window_start": first_window,
                "window_end": last_window,
                "fall_reward": fall_value,
                "safe_reward_mean_same_clock": float(safe_values.mean()),
                "safe_higher_probability": float(np.mean(safe_values > fall_value)),
                "available": True,
            }
        )
    available = [row for row in prefall if row["available"]]
    return {
        "safe_reward": {
            "mean": float(reward[:, safe].mean()),
            "median": float(np.median(reward[:, safe])),
        },
        "fall_reward": {
            "mean": float(reward[:, fall].mean()),
            "median": float(np.median(reward[:, fall])),
        },
        "early_first_50_pairwise_probability_safe_reward_higher": early_safe_higher,
        "prefall_last_20_windows": prefall,
        "all_prefall_events_safe_higher_probability_at_least_65pct": bool(available)
        and all(float(row["safe_higher_probability"]) >= 0.65 for row in available),
        "non_saturated_safe_reward": bool(
            np.quantile(reward[:, safe], 0.95) > 1.0e-4
        ),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output = output_root / "reward_transform_audit"
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    require_compute_gpu(device)
    calibration = json.loads(
        (output_root / "reward_calibration/RESULT.json").read_text(encoding="utf-8")
    )
    default_scale = float(calibration["official_sds_loss_scale"])
    default_median_reward = float(calibration["calibration_reward"]["median"])
    calibration_normalized_median = -math.log(default_median_reward) / default_scale
    anchored_scale = math.log(2.0) / calibration_normalized_median
    mean_abs = np.asarray(
        calibration["diff_normalizer_mean_abs"], dtype=np.float32
    )
    model = load_shared_prior(output_root, device)
    trace_root = (
        ROOT
        / "experiments/demo_following/cross_skill_recovery_tinymdm_state_audit_v1/traces"
    )
    trace_paths = {
        "released_baseline": trace_root / "released_baseline/trace.npz",
        "unconstrained_update64": trace_root / "unconstrained_update64/trace.npz",
        "safety_update64": trace_root / "safety_update64/trace.npz",
    }
    arms: dict[str, Any] = {}
    gates: list[bool] = []
    output.mkdir(parents=True)
    for arm, trace in trace_paths.items():
        windows, _, raw_labels = load_feature_complete_trace(trace, device)
        flat = windows.transpose(1, 0, 2, 3).reshape(-1, WINDOW_SIZE, FEATURE_DIM)
        rewards: dict[str, np.ndarray] = {}
        for condition, class_id in CLASS_IDS.items():
            losses = stable_sds_losses(
                model, flat, class_id, device, args.chunk_size
            )
            rewards[condition] = reward_from_losses(
                losses, mean_abs, anchored_scale
            ).reshape(windows.shape[1], windows.shape[0]).T
            np.save(
                output / f"{arm}_{condition}_reward.npy",
                rewards[condition],
                allow_pickle=False,
            )
        outcomes = outcome_labels(raw_labels)
        causal = causal_reward_audit(rewards["kick"], raw_labels, outcomes)
        gate = bool(
            causal["non_saturated_safe_reward"]
            and causal["all_prefall_events_safe_higher_probability_at_least_65pct"]
        )
        gates.append(gate)
        arms[arm] = {
            "outcomes": {
                "safe_kick_count": int(outcomes["safe_kick"].sum()),
                "physical_fall_count": int(outcomes["physical_fall"].sum()),
            },
            "kick_condition_causal_reward": causal,
            "condition_difference_kick_minus_carry": {
                "mean": float((rewards["kick"] - rewards["carry"]).mean()),
                "kick_reward_higher_window_fraction": float(
                    np.mean(rewards["kick"] > rewards["carry"])
                ),
            },
        }
    result = {
        "protocol": "sugar_conditional_tinymdm_anchored_smp_reward_audit_v1",
        "training_data_anchor": {
            "calibration_normalized_loss_median": calibration_normalized_median,
            "anchored_sds_loss_scale": anchored_scale,
            "definition": "training-distribution reward median equals 0.5",
        },
        "arms": arms,
        "checks": {
            "scale_uses_training_data_only": True,
            "shared_checkpoint_normalizer_and_diffnormalizer": True,
            "official_exponential_reward_family": True,
            "non_saturated_and_prefall_ranked_all_arms": all(gates),
        },
        "online_reward_diagnostic_supported": all(gates),
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
