#!/usr/bin/env python3
"""Audit a motion-disjoint corpus of actual SUGAR rollout contact events."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = (
    ROOT
    / "experiments/demo_following/contact_event_reward_redesign_v1/actual_tracker_corpus"
)
EXPECTED_MOTIONS = {"CarryBox": 100, "KickBox": 99}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def motion_split(source_motion_id: int) -> str:
    """Stable split that never puts one source motion in multiple partitions."""
    remainder = int(source_motion_id) % 10
    if remainder == 8:
        return "validation"
    if remainder == 9:
        return "test"
    return "train"


def _longest_true_run(values: np.ndarray) -> int:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return max(
        (int(stop - start) for start, stop in zip(edges[::2], edges[1::2])),
        default=0,
    )


def load_shard(directory: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    result = json.loads((directory / "RESULT.json").read_text(encoding="utf-8"))
    threshold = float(result["contact_threshold_n"])
    task = str(result["task_family"])
    records: list[dict[str, object]] = []
    with np.load(directory / "TRACE.npz", allow_pickle=False) as trace:
        force = trace["contact_force_w"]
        contact = trace["contact"]
        source = trace["source_motion_id"]
        lift = trace["lift_height_m"]
        regime = trace["motion_regime"]
        reset = trace["reset_before_frame"]
        dt = float(trace["control_dt_s"][0])
        exact = np.linalg.norm(force, axis=-1) > threshold
        if not np.array_equal(contact, exact):
            raise RuntimeError(f"{directory}: contact is not the exact force threshold")
        if source.shape != reset.shape or contact.shape[:2] != source.shape:
            raise RuntimeError(f"{directory}: control-clock shape mismatch")
        for env in range(source.shape[1]):
            ids = np.unique(source[:, env])
            if ids.size != 1:
                raise RuntimeError(f"{directory}: env {env} changed source motion")
            source_id = int(ids[0])
            role_fraction = contact[:, env].mean(axis=0)
            any_hand = contact[:, env, 0] | contact[:, env, 1]
            bilateral_hand = contact[:, env, 0] & contact[:, env, 1]
            any_foot = contact[:, env, 2] | contact[:, env, 3]
            records.append(
                {
                    "task": task,
                    "source_motion_id": source_id,
                    "split": motion_split(source_id),
                    "frames": int(source.shape[0]),
                    "reset_count": int(np.count_nonzero(reset[:, env])),
                    "left_hand_contact_fraction": float(role_fraction[0]),
                    "right_hand_contact_fraction": float(role_fraction[1]),
                    "left_foot_contact_fraction": float(role_fraction[2]),
                    "right_foot_contact_fraction": float(role_fraction[3]),
                    "any_hand_contact_fraction": float(np.mean(any_hand)),
                    "bilateral_hand_contact_fraction": float(np.mean(bilateral_hand)),
                    "any_foot_contact_fraction": float(np.mean(any_foot)),
                    "maximum_lift_m": float(np.max(lift[:, env])),
                    "ground_moving_fraction": float(np.mean(regime[:, env] == 1)),
                    "lifted_moving_fraction": float(np.mean(regime[:, env] == 3)),
                    "longest_hand_event_s": float(_longest_true_run(any_hand) * dt),
                    "longest_foot_event_s": float(_longest_true_run(any_foot) * dt),
                    "peak_hand_force_n": float(
                        np.max(np.linalg.norm(force[:, env, :2], axis=-1))
                    ),
                    "peak_foot_force_n": float(
                        np.max(np.linalg.norm(force[:, env, 2:], axis=-1))
                    ),
                    "trace_directory": str(directory),
                }
            )
    return result, records


def _metric_summary(rows: list[dict[str, object]], name: str) -> dict[str, float]:
    values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def audit(corpus_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    shard_dirs = sorted(path.parent for path in corpus_root.glob("*/TRACE.npz"))
    if not shard_dirs:
        raise FileNotFoundError(f"no TRACE.npz shards under {corpus_root}")
    shard_results: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for directory in shard_dirs:
        shard_result, shard_records = load_shard(directory)
        shard_results.append(shard_result)
        records.extend(shard_records)

    keys = [(str(row["task"]), int(row["source_motion_id"])) for row in records]
    duplicate_count = len(keys) - len(set(keys))
    grouped = {
        task: [row for row in records if row["task"] == task]
        for task in EXPECTED_MOTIONS
    }
    metrics = (
        "any_hand_contact_fraction",
        "bilateral_hand_contact_fraction",
        "any_foot_contact_fraction",
        "maximum_lift_m",
        "ground_moving_fraction",
        "lifted_moving_fraction",
        "longest_hand_event_s",
        "longest_foot_event_s",
        "peak_hand_force_n",
        "peak_foot_force_n",
    )
    summaries = {
        task: {
            "motion_count": len(rows),
            "split_counts": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "test")
            },
            "metrics": {name: _metric_summary(rows, name) for name in metrics}
            if rows
            else {},
        }
        for task, rows in grouped.items()
    }
    carry = summaries["CarryBox"]
    kick = summaries["KickBox"]
    semantic_ready = bool(grouped["CarryBox"]) and bool(grouped["KickBox"])
    checks = {
        "complete_motion_coverage": all(
            summaries[task]["motion_count"] == expected
            for task, expected in EXPECTED_MOTIONS.items()
        ),
        "each_motion_recorded_once": duplicate_count == 0,
        "every_shard_passed_collection_contract": all(
            bool(result["passed"]) for result in shard_results
        ),
        "no_episode_reset_inside_trace": all(
            int(row["reset_count"]) == 0 for row in records
        ),
        "all_motion_disjoint_splits_nonempty": all(
            all(count > 0 for count in summary["split_counts"].values())
            for summary in summaries.values()
        ),
        "carry_has_more_bilateral_hand_contact_than_kick": (
            semantic_ready
            and carry["metrics"]["bilateral_hand_contact_fraction"]["median"]
            > kick["metrics"]["bilateral_hand_contact_fraction"]["median"] + 0.05
        ),
        "kick_has_more_foot_contact_than_carry": (
            semantic_ready
            and kick["metrics"]["any_foot_contact_fraction"]["median"]
            > carry["metrics"]["any_foot_contact_fraction"]["median"] + 0.01
        ),
        "carry_median_lift_exceeds_five_centimeters": (
            semantic_ready
            and carry["metrics"]["maximum_lift_m"]["median"] >= 0.05
        ),
    }
    result = {
        "protocol": "sugar_actual_contact_event_corpus_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "expected_motions": EXPECTED_MOTIONS,
        "duplicate_motion_count": duplicate_count,
        "shard_count": len(shard_dirs),
        "tasks": summaries,
        "split_rule": "source_motion_id mod 10: 8 validation, 9 test, otherwise train",
        "target_contract": {
            "actual_contact": "filtered IsaacLab body-to-box force thresholded at 0.1 N",
            "event_duration": "reset-bounded duration on the actual rollout clock",
            "motion_regime": "episode-relative object lift and velocity",
            "reference_binary_proxy_used_as_actual_target": False,
        },
        "claim_boundary": (
            "Passing establishes a motion-disjoint actual-rollout target corpus and "
            "task-level event separability. It does not establish predictor "
            "generalization or a policy-level demo-following benefit."
        ),
        "automatic_next_branch": (
            "train_and_gate_serious_causal_contact_event_predictor"
            if all(checks.values())
            else "inspect_failed_corpus_contract_before_model_training"
        ),
    }
    return result, records


def main() -> None:
    args = parse_args()
    result, records = audit(args.corpus_root.expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "motions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
