#!/usr/bin/env python3
"""Measure whether live patch signals precede physical failure in frozen Z runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FACTOR_LABELS = {
    1.0: "1p0x",
    1.5: "1p5x",
    3.0: "3p0x",
    6.0: "6p0x",
    10.0: "10p0x",
}
CONTINUOUS_CHANNELS = {
    1: "normal_load",
    2: "pressure",
    3: "shear_x",
    4: "shear_y",
    5: "friction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-root",
        type=Path,
        action="append",
        required=True,
        help="Frozen-evaluation root containing one train_*_eval_* directory per factor.",
    )
    parser.add_argument("--scale-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-frames", type=int, default=10)
    parser.add_argument("--post-frames", type=int, default=80)
    parser.add_argument("--consecutive-frames", type=int, default=2)
    parser.add_argument("--sag-threshold-m", type=float, default=0.02)
    parser.add_argument("--drop-threshold-m", type=float, default=0.15)
    return parser.parse_args()


def first_consecutive(mask: np.ndarray, count: int) -> int | None:
    run = 0
    for index, active in enumerate(mask):
        run = run + 1 if bool(active) else 0
        if run >= count:
            return index - count + 1
    return None


def paired_onset(
    post_values: np.ndarray,
    pre_values: np.ndarray,
    consecutive: int,
    absolute_epsilon: float,
) -> tuple[int | None, float]:
    threshold = float(np.max(pre_values) + absolute_epsilon)
    return first_consecutive(post_values > threshold, consecutive), threshold


def first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if not len(indices) else int(indices[0])


def median_or_none(values: list[int | float]) -> float | None:
    return None if not values else float(np.median(values))


def range_or_none(values: list[int | float]) -> list[float] | None:
    return None if not values else [float(np.min(values)), float(np.max(values))]


def summarize_profiles(records: list[dict[str, object]]) -> dict[str, object]:
    def onset_values(name: str) -> list[int]:
        return [int(record[name]) for record in records if record[name] is not None]

    result: dict[str, object] = {"profiles": len(records)}
    for event in ("sag", "drop", "contact_loss"):
        event_values = onset_values(f"{event}_onset_frames")
        result[f"{event}_onset_available"] = len(event_values)
        result[f"{event}_onset_median_frames"] = median_or_none(event_values)
        result[f"{event}_onset_range_frames"] = range_or_none(event_values)
    for signal in (
        "continuous_patch",
        *CONTINUOUS_CHANNELS.values(),
        "contact_binary",
        "z_action",
        "slip",
    ):
        onset_name = f"{signal}_onset_frames"
        onsets = onset_values(onset_name)
        result[f"{signal}_onset_available"] = len(onsets)
        result[f"{signal}_onset_median_frames"] = median_or_none(onsets)
        result[f"{signal}_onset_range_frames"] = range_or_none(onsets)
        for event in ("sag", "drop", "contact_loss"):
            event_name = f"{event}_onset_frames"
            paired = [
                (int(record[onset_name]), int(record[event_name]))
                for record in records
                if record[onset_name] is not None and record[event_name] is not None
            ]
            leads = [event_frame - onset for onset, event_frame in paired]
            result[f"{signal}_before_{event}_count"] = sum(lead >= 0 for lead in leads)
            result[f"{signal}_to_{event}_paired_count"] = len(leads)
            result[f"{signal}_lead_to_{event}_median_frames"] = median_or_none(leads)
            result[f"{signal}_lead_to_{event}_range_frames"] = range_or_none(leads)
    return result


def find_factor_directory(root: Path, label: str) -> Path:
    matches = sorted(path for path in root.glob(f"train_*_eval_*_{label}") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} directory under {root}, found {matches}")
    return matches[0]


def load_run(directory: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    summary_path = directory / "summary.json"
    trace_path = directory / "frozen_evaluation_trace.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as payload:
        trace = {name: np.asarray(payload[name]) for name in payload.files}
    required = {
        "policy_action",
        "object_pos_w",
        "patch_features",
        "slip_state",
        "bilateral_patch_contact",
    }
    missing = required.difference(trace)
    if missing:
        raise KeyError(f"{trace_path} is missing {sorted(missing)}")
    return summary, trace


def main() -> None:
    args = parse_args()
    if args.pre_frames < 1 or args.post_frames < 1 or args.consecutive_frames < 1:
        raise ValueError("frame windows and consecutive-frames must be positive")
    if not 0.0 < args.sag_threshold_m < args.drop_threshold_m:
        raise ValueError("require 0 < sag-threshold < drop-threshold")

    scale_payload = json.loads(args.scale_file.read_text(encoding="utf-8"))
    patch_scales = np.asarray(scale_payload["patch_channel_scales"][:6], dtype=np.float64)
    if patch_scales.shape != (6,) or np.any(patch_scales <= 0.0):
        raise ValueError("scale file must provide six positive patch-channel scales")

    profile_records: list[dict[str, object]] = []
    excluded_no_event: list[dict[str, int | float]] = []
    seed_metadata: list[dict[str, object]] = []
    offsets = np.arange(-args.pre_frames, args.post_frames + 1)

    for raw_root in args.seed_root:
        root = raw_root.expanduser().resolve()
        nominal_summary, nominal_trace = load_run(find_factor_directory(root, "1p0x"))
        nominal_episodes = {
            int(episode["profile"]): episode for episode in nominal_summary["episodes"]
        }
        seed_metadata.append(
            {
                "root": str(root),
                "training_seed": int(nominal_summary["training_seed"]),
                "evaluation_seed": int(nominal_summary["seed"]),
            }
        )

        for factor in (1.5, 3.0, 6.0, 10.0):
            summary, trace = load_run(find_factor_directory(root, FACTOR_LABELS[factor]))
            training_seed = int(summary["training_seed"])
            evaluation_seed = int(summary["seed"])
            for episode in summary["episodes"]:
                profile = int(episode["profile"])
                jump = episode["jump_frame"]
                nominal_jump = nominal_episodes[profile]["jump_frame"]
                if jump is None or nominal_jump is None:
                    excluded_no_event.append(
                        {
                            "training_seed": training_seed,
                            "evaluation_seed": evaluation_seed,
                            "mass_factor": factor,
                            "profile": profile,
                        }
                    )
                    continue
                jump = int(jump)
                nominal_jump = int(nominal_jump)
                if (
                    jump < args.pre_frames
                    or nominal_jump < args.pre_frames
                    or jump + args.post_frames >= len(trace["policy_action"])
                    or nominal_jump + args.post_frames >= len(nominal_trace["policy_action"])
                ):
                    raise RuntimeError(f"profile {profile} does not cover the analysis window")

                continuous_patch_delta = np.mean(
                    np.abs(
                        trace["patch_features"][jump + offsets, profile, :, :, 1:]
                        / patch_scales[1:]
                        - nominal_trace["patch_features"][
                            nominal_jump + offsets, profile, :, :, 1:
                        ]
                        / patch_scales[1:]
                    ),
                    axis=(1, 2, 3),
                )
                continuous_channel_delta = {
                    name: np.mean(
                        np.abs(
                            trace["patch_features"][jump + offsets, profile, :, :, index]
                            / patch_scales[index]
                            - nominal_trace["patch_features"][
                                nominal_jump + offsets, profile, :, :, index
                            ]
                            / patch_scales[index]
                        ),
                        axis=(1, 2),
                    )
                    for index, name in CONTINUOUS_CHANNELS.items()
                }
                contact_binary_delta = np.mean(
                    trace["patch_features"][jump + offsets, profile, :, :, 0]
                    != nominal_trace["patch_features"][
                        nominal_jump + offsets, profile, :, :, 0
                    ],
                    axis=(1, 2),
                )
                action_delta = np.mean(
                    np.abs(
                        trace["policy_action"][jump + offsets, profile]
                        - nominal_trace["policy_action"][nominal_jump + offsets, profile]
                    ),
                    axis=1,
                )
                slip_delta = np.mean(
                    trace["slip_state"][jump + offsets, profile]
                    != nominal_trace["slip_state"][nominal_jump + offsets, profile],
                    axis=(1, 2),
                )
                pre = slice(0, args.pre_frames)
                post = slice(args.pre_frames, None)
                continuous_patch_onset, continuous_patch_threshold = paired_onset(
                    continuous_patch_delta[post],
                    continuous_patch_delta[pre],
                    args.consecutive_frames,
                    1.0e-6,
                )
                contact_binary_onset, contact_binary_threshold = paired_onset(
                    contact_binary_delta[post],
                    contact_binary_delta[pre],
                    args.consecutive_frames,
                    1.0e-12,
                )
                continuous_channel_audit: dict[str, int | float | None] = {}
                for name, delta in continuous_channel_delta.items():
                    onset, threshold = paired_onset(
                        delta[post], delta[pre], args.consecutive_frames, 1.0e-6
                    )
                    continuous_channel_audit[f"{name}_onset_frames"] = onset
                    continuous_channel_audit[f"{name}_pre_delta_max"] = float(
                        np.max(delta[pre])
                    )
                    continuous_channel_audit[f"{name}_delta_threshold"] = threshold
                action_onset, action_threshold = paired_onset(
                    action_delta[post], action_delta[pre], args.consecutive_frames, 1.0e-6
                )
                slip_onset, slip_threshold = paired_onset(
                    slip_delta[post],
                    slip_delta[pre],
                    args.consecutive_frames,
                    1.0 / 54.0,
                )

                post_height = trace["object_pos_w"][
                    jump : jump + args.post_frames + 1, profile, 2
                ]
                height_loss = float(post_height[0]) - post_height
                sag_onset = first_true(height_loss >= args.sag_threshold_m)
                drop_onset = first_true(height_loss >= args.drop_threshold_m)
                contact_loss_onset = first_true(
                    ~trace["bilateral_patch_contact"][
                        jump : jump + args.post_frames + 1, profile
                    ]
                )
                record: dict[str, object] = {
                    "training_seed": training_seed,
                    "evaluation_seed": evaluation_seed,
                    "mass_factor": factor,
                    "profile": profile,
                    "hold_success": bool(episode["hold_success"]),
                    "drop": bool(episode["drop"]),
                    "robot_fall": bool(episode["robot_fall"]),
                    "jump_frame": jump,
                    "nominal_jump_frame": nominal_jump,
                    "continuous_patch_onset_frames": continuous_patch_onset,
                    "contact_binary_onset_frames": contact_binary_onset,
                    "z_action_onset_frames": action_onset,
                    "slip_onset_frames": slip_onset,
                    "sag_onset_frames": sag_onset,
                    "drop_onset_frames": drop_onset,
                    "contact_loss_onset_frames": contact_loss_onset,
                    "continuous_patch_pre_delta_max": float(
                        np.max(continuous_patch_delta[pre])
                    ),
                    "continuous_patch_delta_threshold": continuous_patch_threshold,
                    "contact_binary_pre_delta_max": float(
                        np.max(contact_binary_delta[pre])
                    ),
                    "contact_binary_delta_threshold": contact_binary_threshold,
                    "z_action_pre_delta_max": float(np.max(action_delta[pre])),
                    "z_action_delta_threshold": action_threshold,
                    "slip_pre_delta_max": float(np.max(slip_delta[pre])),
                    "slip_delta_threshold": slip_threshold,
                }
                record.update(continuous_channel_audit)
                profile_records.append(record)

    factor_summary: dict[str, object] = {}
    for factor in (1.5, 3.0, 6.0, 10.0):
        factor_records = [record for record in profile_records if record["mass_factor"] == factor]
        factor_summary[str(factor)] = {
            "eligible_profiles": len(factor_records),
            "hold_profiles": sum(bool(record["hold_success"]) for record in factor_records),
            "all_profiles": summarize_profiles(factor_records),
            "sag_profiles": summarize_profiles(
                [record for record in factor_records if record["sag_onset_frames"] is not None]
            ),
            "drop_profiles": summarize_profiles(
                [record for record in factor_records if record["drop_onset_frames"] is not None]
            ),
        }

    sag_records = [
        record for record in profile_records if record["sag_onset_frames"] is not None
    ]
    result = {
        "schema": "plan15_frozen_z_mass_reaction_window_v1",
        "description": (
            "Event-aligned paired differences between each mass condition and the same "
            "profile's frozen 1x rollout. This is an audit of sensing opportunity, not "
            "evidence that a tactile policy improves behavior."
        ),
        "seed_pairs": seed_metadata,
        "control_rate_hz": 50,
        "pre_event_calibration_frames": args.pre_frames,
        "post_event_window_frames": args.post_frames,
        "required_consecutive_frames": args.consecutive_frames,
        "sag_threshold_m": args.sag_threshold_m,
        "drop_threshold_m": args.drop_threshold_m,
        "continuous_patch_delta": (
            "mean absolute paired delta over normalized normal-load, pressure, signed-XY "
            "shear and friction-utilization channels; contact binary is excluded"
        ),
        "continuous_patch_onset_rule": (
            "first consecutive post-jump samples above pre-jump maximum + 1e-6"
        ),
        "continuous_channel_names": list(CONTINUOUS_CHANNELS.values()),
        "contact_binary_delta": "fraction of 54 patch contact bits different from paired 1x",
        "contact_binary_onset_rule": (
            "first consecutive post-jump samples above pre-jump maximum + 1e-12"
        ),
        "z_action_delta": (
            "mean absolute paired Z-policy action delta; downstream evidence of actor-visible "
            "closed-loop dynamics, not a direct proprioception measurement"
        ),
        "z_action_onset_rule": (
            "first consecutive post-jump samples above pre-jump maximum + 1e-6"
        ),
        "slip_delta": "fraction of 54 patch states different from the paired 1x rollout",
        "slip_onset_rule": (
            "first consecutive post-jump samples above pre-jump maximum + 1/54"
        ),
        "excluded_no_event": excluded_no_event,
        "factor_summary": factor_summary,
        "all_sag_profiles": summarize_profiles(sag_records),
        "profiles": profile_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "factor_summary", "all_sag_profiles")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
