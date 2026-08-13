#!/usr/bin/env python3
"""Evaluate Plan-15 causal patch slip against held-out simulator velocity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NO_CONTACT = 0
STICK = 1
INCIPIENT = 2
GROSS = 3


def oracle_state(
    contact: np.ndarray,
    speed_m_s: np.ndarray,
    *,
    incipient_speed_m_s: float,
    gross_speed_m_s: float,
) -> np.ndarray:
    state = np.full(contact.shape, STICK, dtype=np.int8)
    state[speed_m_s >= incipient_speed_m_s] = INCIPIENT
    state[speed_m_s >= gross_speed_m_s] = GROSS
    state[~contact] = NO_CONTACT
    return state


def onset_delays(
    oracle_slip: np.ndarray,
    predicted_slip: np.ndarray,
) -> tuple[list[int], int]:
    delays: list[int] = []
    missed = 0
    frames, hands, patches = oracle_slip.shape
    for hand in range(hands):
        for patch in range(patches):
            truth = oracle_slip[:, hand, patch]
            prediction = predicted_slip[:, hand, patch]
            onsets = np.flatnonzero(truth & ~np.r_[False, truth[:-1]])
            for onset in onsets:
                endings = np.flatnonzero(~truth[onset:])
                end = frames if not len(endings) else onset + int(endings[0])
                detections = np.flatnonzero(prediction[onset:end])
                if len(detections):
                    delays.append(int(detections[0]))
                else:
                    missed += 1
    return delays, missed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--incipient-speed", type=float, default=0.005)
    parser.add_argument("--gross-speed", type=float, default=0.02)
    parser.add_argument("--control-rate-hz", type=float, default=50.0)
    args = parser.parse_args()
    if not 0.0 < args.incipient_speed < args.gross_speed:
        raise ValueError("slip oracle speed thresholds are invalid")
    if args.control_rate_hz <= 0.0:
        raise ValueError("control rate must be positive")

    confusion = np.zeros((4, 4), dtype=np.int64)
    delays: list[int] = []
    missed_onsets = 0
    frames_total = 0
    source_traces = []
    for value in args.trace:
        path = value.expanduser().resolve()
        with np.load(path, allow_pickle=False) as source:
            patch = np.asarray(source["patch_features"], dtype=np.float32)
            predicted = np.asarray(source["slip_state"], dtype=np.int8)
            speed = np.asarray(
                source["oracle_patch_tangential_speed_m_s"], dtype=np.float32
            )
        if patch.shape[1:] != (2, 27, 6):
            raise ValueError(f"invalid patch trace {patch.shape} in {path}")
        if predicted.shape != patch.shape[:3] or speed.shape != patch.shape[:3]:
            raise ValueError(f"slip/oracle shape mismatch in {path}")
        contact = patch[..., 0] > 0.5
        oracle = oracle_state(
            contact,
            speed,
            incipient_speed_m_s=float(args.incipient_speed),
            gross_speed_m_s=float(args.gross_speed),
        )
        np.add.at(confusion, (oracle.reshape(-1), predicted.reshape(-1)), 1)
        trace_delays, trace_missed = onset_delays(
            oracle >= INCIPIENT,
            predicted >= INCIPIENT,
        )
        delays.extend(trace_delays)
        missed_onsets += trace_missed
        frames_total += len(patch)
        source_traces.append(str(path))

    oracle_slip_count = int(confusion[INCIPIENT:, :].sum())
    predicted_slip_count = int(confusion[:, INCIPIENT:].sum())
    true_positive = int(confusion[INCIPIENT:, INCIPIENT:].sum())
    false_positive = predicted_slip_count - true_positive
    false_negative = oracle_slip_count - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    delay_array = np.asarray(delays, dtype=np.float64)
    result = {
        "schema": "plan15_online_patch_slip_velocity_evaluation_v1",
        "detector_inputs": (
            "causal current/past patch contact, pressure, signed shear, "
            "friction utilization and timestamps"
        ),
        "heldout_oracle_only": (
            "maximum active-taxel simulator relative tangential speed per patch"
        ),
        "oracle_thresholds_m_s": {
            "incipient": float(args.incipient_speed),
            "gross": float(args.gross_speed),
        },
        "source_traces": source_traces,
        "frames": frames_total,
        "confusion_rows_oracle_columns_prediction": confusion.tolist(),
        "binary_slip": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
        },
        "onset_detection": {
            "detected": len(delays),
            "missed": missed_onsets,
            "median_delay_frames": (
                None if not len(delays) else float(np.median(delay_array))
            ),
            "p95_delay_frames": (
                None if not len(delays) else float(np.quantile(delay_array, 0.95))
            ),
            "median_delay_s": (
                None
                if not len(delays)
                else float(np.median(delay_array) / args.control_rate_hz)
            ),
        },
        "claim_boundary": (
            "The velocity is an evaluation label only and never enters the "
            "detector, actor, reward, or mass scheduler."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
