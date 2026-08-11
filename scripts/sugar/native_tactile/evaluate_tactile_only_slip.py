#!/usr/bin/env python3
"""Evaluate tactile-only slip states against held-out simulator velocity.

The detector under test never reads contact/object velocity. This script reads
the archived velocity only after collection to quantify physical agreement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NAMES = ("no_contact", "stick", "incipient", "gross")


def oracle_state(
    penetration_m: np.ndarray,
    relative_tangential_velocity_w_m_s: np.ndarray,
    *,
    incipient_speed_m_s: float,
    gross_speed_m_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-patch velocity state and maximum active-taxel speed."""
    active = penetration_m > 0.0
    speed = np.linalg.norm(relative_tangential_velocity_w_m_s, axis=-1)
    active_speed = np.where(active, speed, 0.0)
    maximum = active_speed.max(axis=(-2, -1))
    contact = active.any(axis=(-2, -1))
    state = np.full(contact.shape, 1, dtype=np.int8)
    state[maximum >= incipient_speed_m_s] = 2
    state[maximum >= gross_speed_m_s] = 3
    state[~contact] = 0
    return state, maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--incipient-speed", type=float, default=0.005)
    parser.add_argument("--gross-speed", type=float, default=0.02)
    args = parser.parse_args()

    trace = args.run_root.resolve() / "whole_hand_trace.npz"
    with np.load(trace, allow_pickle=False) as source:
        predicted = np.asarray(source["tactile_only_slip_state"], np.int8)
        penetration = np.asarray(source["penetration"], np.float32)
        relative_velocity = np.asarray(
            source["tactile_relative_tangential_velocity_w"], np.float32
        )
    oracle, maximum_speed = oracle_state(
        penetration,
        relative_velocity,
        incipient_speed_m_s=args.incipient_speed,
        gross_speed_m_s=args.gross_speed,
    )
    if predicted.shape != oracle.shape:
        raise RuntimeError(
            f"Predicted/oracle shape mismatch: {predicted.shape} vs {oracle.shape}"
        )

    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (oracle.reshape(-1), predicted.reshape(-1)), 1)
    contact = oracle > 0
    exact_contact_accuracy = float(np.mean(predicted[contact] == oracle[contact]))
    oracle_slip = oracle >= 2
    predicted_slip = predicted >= 2
    true_positive = int(np.count_nonzero(oracle_slip & predicted_slip))
    false_positive = int(np.count_nonzero(~oracle_slip & predicted_slip))
    false_negative = int(np.count_nonzero(oracle_slip & ~predicted_slip))
    true_negative = int(np.count_nonzero(~oracle_slip & ~predicted_slip))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)

    speed_by_prediction = {}
    for state, name in enumerate(NAMES):
        values = maximum_speed[predicted == state]
        speed_by_prediction[name] = (
            np.quantile(values, (0.05, 0.5, 0.95)).tolist()
            if len(values)
            else []
        )

    report = {
        "schema": "tactile_only_slip_heldout_velocity_evaluation_v1",
        "detector_inputs": [
            "signed_local_z_force",
            "signed_local_xy_shear",
            "penetration",
            "tactile_source_time",
        ],
        "heldout_oracle_only": "simulator contact-point relative tangential velocity",
        "oracle_thresholds_m_s": {
            "incipient": args.incipient_speed,
            "gross": args.gross_speed,
        },
        "state_order": list(NAMES),
        "confusion_rows_oracle_columns_prediction": confusion.tolist(),
        "exact_state_accuracy_on_contact": exact_contact_accuracy,
        "binary_incipient_or_gross": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": precision,
            "recall": recall,
        },
        "maximum_oracle_speed_q05_median_q95_by_predicted_state_m_s": speed_by_prediction,
        "claim_boundary": (
            "The oracle is evaluation-only. Agreement measures simulated slip "
            "detection; it is not real-hardware validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
