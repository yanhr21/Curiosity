#!/usr/bin/env python3
"""Compare time-resolved mass leakage in paired fixed-action Plan-15 traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FACTORS = (1.0, 1.5, 3.0, 6.0, 10.0)
GROUPS = (
    "object_state",
    "proprio_only",
    "patch_tactile",
    "patch_tactile_plus_slip",
)


def parse_trace(value: str) -> tuple[float, Path]:
    factor_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("trace must be FACTOR=PATH")
    factor = float(factor_text)
    if factor not in FACTORS:
        raise argparse.ArgumentTypeError(f"factor must be one of {FACTORS}")
    return factor, Path(path_text).expanduser().resolve()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--trace",
    action="append",
    type=parse_trace,
    required=True,
    help="Paired trace as FACTOR=PATH; repeat each factor in the same seed order.",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--pre-frames", type=int, default=25)
parser.add_argument("--post-frames", type=int, default=50)
parser.add_argument("--probe-c", type=float, default=10.0)
args = parser.parse_args()


def first_consecutive(mask: np.ndarray, count: int = 3) -> int | None:
    run = 0
    for index, active in enumerate(mask):
        run = run + 1 if bool(active) else 0
        if run >= count:
            return index - count + 1
    return None


def feature_groups(trace: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    patch = trace["patch_features"].reshape(len(trace["patch_features"]), -1)
    slip = trace["slip_features"].reshape(len(trace["slip_features"]), -1)
    return {
        "object_state": np.concatenate(
            (
                trace["object_pos_w"],
                trace["object_quat_w"],
                trace["object_lin_vel_w"],
                trace["object_ang_vel_w"],
            ),
            axis=-1,
        ),
        "proprio_only": trace["actor_policy_observation"],
        "patch_tactile": patch,
        "patch_tactile_plus_slip": np.concatenate((patch, slip), axis=-1),
    }


def linear_probe_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    probe_c: float,
) -> np.ndarray:
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=probe_c,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=15,
        ),
    )
    probe.fit(train_x, train_y)
    return probe.predict(test_x)


def main() -> None:
    if args.pre_frames < 1 or args.post_frames < 1 or args.probe_c <= 0.0:
        raise ValueError("window lengths and probe-c must be positive")
    paths: dict[float, list[Path]] = {factor: [] for factor in FACTORS}
    for factor, path in args.trace:
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[factor].append(path)
    repeat_counts = {factor: len(values) for factor, values in paths.items()}
    if len(set(repeat_counts.values())) != 1 or not next(iter(repeat_counts.values())):
        raise ValueError(f"each factor needs the same nonzero repeat count: {repeat_counts}")
    repeats = next(iter(repeat_counts.values()))

    traces: dict[float, list[dict[str, np.ndarray]]] = {
        factor: [] for factor in FACTORS
    }
    jump_frames: dict[float, list[int]] = {factor: [] for factor in FACTORS}
    required = {
        "patch_features",
        "slip_features",
        "actor_policy_observation",
        "object_pos_w",
        "object_quat_w",
        "object_lin_vel_w",
        "object_ang_vel_w",
        "applied_action",
        "jump_applied",
    }
    for factor in FACTORS:
        for path in paths[factor]:
            with np.load(path, allow_pickle=False) as payload:
                missing = required.difference(payload.files)
                if missing:
                    raise KeyError(f"{path} missing {sorted(missing)}")
                trace = {name: np.asarray(payload[name]) for name in required}
            if trace["actor_policy_observation"].shape[1:] != (504,):
                raise ValueError(f"{path} does not contain the 504-D actor contract")
            if trace["patch_features"].shape[1:] != (2, 27, 6):
                raise ValueError(f"{path} does not contain bilateral 27x6 patches")
            if trace["slip_features"].shape[1:] != (2, 27, 3):
                raise ValueError(f"{path} does not contain bilateral 27x3 slip")
            indices = np.flatnonzero(trace["jump_applied"])
            if not len(indices):
                raise ValueError(f"{path} has no applied mass event")
            jump = int(indices[0])
            if jump < args.pre_frames or jump + args.post_frames >= len(trace["jump_applied"]):
                raise ValueError(f"{path} does not cover the requested jump window")
            traces[factor].append(trace)
            jump_frames[factor].append(jump)

    max_action_error = 0.0
    max_jump_frame_error = 0
    for repeat in range(repeats):
        control_action = traces[1.0][repeat]["applied_action"]
        control_jump = jump_frames[1.0][repeat]
        for factor in FACTORS[1:]:
            action = traces[factor][repeat]["applied_action"]
            if action.shape != control_action.shape:
                raise ValueError("paired action trace lengths differ")
            max_action_error = max(
                max_action_error, float(np.max(np.abs(action - control_action)))
            )
            max_jump_frame_error = max(
                max_jump_frame_error,
                abs(jump_frames[factor][repeat] - control_jump),
            )
    if max_action_error != 0.0 or max_jump_frame_error != 0:
        raise RuntimeError(
            "leakage traces are not exactly paired: "
            f"action_error={max_action_error}, jump_frame_error={max_jump_frame_error}"
        )

    offsets = np.arange(-args.pre_frames, args.post_frames + 1)
    windows: dict[str, np.ndarray] = {}
    for group in GROUPS:
        factor_windows = []
        for factor in FACTORS:
            repeat_windows = []
            for repeat, trace in enumerate(traces[factor]):
                features = feature_groups(trace)[group]
                jump = jump_frames[factor][repeat]
                repeat_windows.append(features[jump + offsets])
            factor_windows.append(np.stack(repeat_windows))
        # [factor, repeat, relative_time, feature]
        windows[group] = np.stack(factor_windows).astype(np.float64)

    raw_change: dict[str, dict[str, object]] = {}
    for group, values in windows.items():
        group_change: dict[str, object] = {}
        control = values[0]
        for factor_index, factor in enumerate(FACTORS[1:], start=1):
            curve = np.mean(np.abs(values[factor_index] - control), axis=(0, 2))
            pre = curve[offsets < 0]
            threshold = float(
                np.quantile(pre, 0.99)
                + max(1.0e-8, 5.0 * np.median(np.abs(pre - np.median(pre))))
            )
            post_mask = (curve[offsets >= 0] > threshold)
            onset_index = first_consecutive(post_mask)
            group_change[str(factor)] = {
                "mean_abs_delta_by_offset": curve.tolist(),
                "onset_threshold": threshold,
                "onset_offset_frames": (
                    None if onset_index is None else int(offsets[offsets >= 0][onset_index])
                ),
            }
        raw_change[group] = group_change

    probe: dict[str, object] = {}
    for group, values in windows.items():
        if repeats < 2:
            probe[group] = {
                "balanced_accuracy_by_offset": None,
                "first_reliable_offset_frames": None,
                "reason": "at least two paired repeats are required",
            }
            continue
        accuracy = []
        for time_index in range(len(offsets)):
            correct = []
            for held_out in range(repeats):
                train_x = []
                train_y = []
                test_x = []
                test_y = []
                for factor_index in range(len(FACTORS)):
                    for repeat in range(repeats):
                        sample = values[factor_index, repeat, time_index]
                        if repeat == held_out:
                            test_x.append(sample)
                            test_y.append(factor_index)
                        else:
                            train_x.append(sample)
                            train_y.append(factor_index)
                prediction = linear_probe_predictions(
                    np.asarray(train_x),
                    np.asarray(train_y),
                    np.asarray(test_x),
                    args.probe_c,
                )
                correct.extend(prediction == np.asarray(test_y))
            accuracy.append(float(np.mean(correct)))
        accuracy_array = np.asarray(accuracy)
        reliable = (offsets >= 0) & (accuracy_array >= 0.6)
        reliable_index = first_consecutive(reliable)
        probe[group] = {
            "balanced_accuracy_by_offset": accuracy,
            "first_reliable_offset_frames": (
                None if reliable_index is None else int(offsets[reliable_index])
            ),
            "chance_accuracy": 0.2,
            "reliable_accuracy_threshold": 0.6,
            "probe": "standardized multinomial logistic regression",
            "probe_c": args.probe_c,
        }

    result = {
        "schema": "plan15_online_mass_leakage_audit_v1",
        "mass_factors": list(FACTORS),
        "paired_repeats": repeats,
        "relative_offsets_frames": offsets.tolist(),
        "control_rate_hz": 50,
        "fixed_action_max_abs_error": max_action_error,
        "paired_jump_frame_max_error": max_jump_frame_error,
        "actor_observation_width": 504,
        "actor_contains_measured_object_state": False,
        "object_state_is_evaluation_only": True,
        "raw_change": raw_change,
        "linear_probe": probe,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
