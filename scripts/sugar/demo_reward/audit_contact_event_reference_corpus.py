#!/usr/bin/env python3
"""Audit whether official SUGAR Carry/Kick demos support contact-event targets.

The source ``contact_labels_50hz.npy`` arrays are binary reference annotations,
not tactile force.  This script combines them with named G1 body kinematics and
object motion only to define selected-demo event semantics for the next reward
predictor dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "SUGAR/data"
DEFAULT_OUTPUT = (
    ROOT / "experiments/demo_following/contact_event_reward_redesign_v1/reference_corpus_audit"
)
CONTROL_DT_S = 0.02
LIFT_THRESHOLD_M = 0.05
MOVE_THRESHOLD_MPS = 0.05
BODY_COUNT = 35
EFFECTORS = (
    ("left_hand", 24, "hand"),
    ("right_hand", 32, "hand"),
    ("left_foot", 6, "foot"),
    ("right_foot", 12, "foot"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [
        (int(start), int(stop - 1))
        for start, stop in zip(edges[::2], edges[1::2])
    ]


def load_motion(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / "robot_50hz.npz", allow_pickle=False) as archive:
        body_position = np.asarray(archive["body_pos_w"], dtype=np.float64)
    with (directory / "obj_motion_global_50hz.pkl").open("rb") as stream:
        obj = pickle.load(stream)
    contact = np.load(directory / "contact_labels_50hz.npy", allow_pickle=False)
    object_position = np.asarray(obj["obj_trans"], dtype=np.float64)
    object_velocity = np.asarray(obj["obj_lin_vel"], dtype=np.float64)
    length = min(
        body_position.shape[0],
        object_position.shape[0],
        object_velocity.shape[0],
        contact.shape[0],
    )
    if body_position.shape[1] != BODY_COUNT:
        raise RuntimeError(f"{directory}: expected {BODY_COUNT} G1 bodies")
    if length < 100:
        raise RuntimeError(f"{directory}: reference is too short ({length} frames)")
    return {
        "body_position": body_position[:length],
        "object_position": object_position[:length],
        "object_velocity": object_velocity[:length],
        "contact": np.asarray(contact[:length], dtype=bool),
    }


def motion_record(task: str, directory: Path) -> dict[str, object]:
    motion = load_motion(directory)
    contact = motion["contact"]
    object_position = motion["object_position"]
    object_speed = np.linalg.norm(motion["object_velocity"], axis=-1)
    baseline_z = float(np.median(object_position[:25, 2]))
    lift = object_position[:, 2] - baseline_z
    lifted = lift >= LIFT_THRESHOLD_M
    moving = object_speed >= MOVE_THRESHOLD_MPS

    effector_position = np.stack(
        [motion["body_position"][:, body_index] for _, body_index, _ in EFFECTORS],
        axis=1,
    )
    distance = np.linalg.norm(effector_position - object_position[:, None], axis=-1)
    closest = np.argmin(distance, axis=1)
    closest_role_is_hand = closest < 2
    runs = true_runs(contact)
    return {
        "task": task,
        "motion": directory.name,
        "frames": int(contact.size),
        "contact_fraction": float(np.mean(contact)),
        "contact_run_count": len(runs),
        "longest_contact_s": float(
            max((end - start + 1 for start, end in runs), default=0) * CONTROL_DT_S
        ),
        "contact_hand_closest_fraction": float(
            np.mean(closest_role_is_hand[contact]) if np.any(contact) else 0.0
        ),
        "contact_foot_closest_fraction": float(
            np.mean(~closest_role_is_hand[contact]) if np.any(contact) else 0.0
        ),
        "maximum_lift_m": float(np.max(lift)),
        "lifted_fraction": float(np.mean(lifted)),
        "ground_moving_fraction": float(np.mean(moving & ~lifted)),
        "lifted_moving_fraction": float(np.mean(moving & lifted)),
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    numeric = [
        key
        for key, value in records[0].items()
        if isinstance(value, (int, float)) and key != "frames"
    ]
    return {
        "motion_count": len(records),
        "motions_with_contact": sum(float(row["contact_fraction"]) > 0 for row in records),
        "metrics": {
            key: {
                "mean": float(np.mean([float(row[key]) for row in records])),
                "median": float(np.median([float(row[key]) for row in records])),
                "min": float(np.min([float(row[key]) for row in records])),
                "max": float(np.max([float(row[key]) for row in records])),
            }
            for key in numeric
        },
    }


def audit(data_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    for task in ("CarryBox", "KickBox"):
        for directory in sorted((data_root / task).glob("data_*")):
            records.append(motion_record(task, directory))
    grouped = {
        task: [row for row in records if row["task"] == task]
        for task in ("CarryBox", "KickBox")
    }
    missing = [task for task, rows in grouped.items() if not rows]
    if missing:
        raise FileNotFoundError(
            f"official SUGAR reference motions are missing for: {missing}"
        )
    summaries = {task: summarize(rows) for task, rows in grouped.items()}
    carry = summaries["CarryBox"]
    kick = summaries["KickBox"]
    checks = {
        "at_least_95_complete_motions_per_task": (
            carry["motion_count"] >= 95 and kick["motion_count"] >= 95
        ),
        "every_motion_has_binary_contact_annotation": (
            carry["motions_with_contact"] == carry["motion_count"]
            and kick["motions_with_contact"] == kick["motion_count"]
        ),
        "carry_contact_role_is_predominantly_hand": (
            carry["metrics"]["contact_hand_closest_fraction"]["median"] >= 0.80
        ),
        "kick_contact_role_is_predominantly_foot": (
            kick["metrics"]["contact_foot_closest_fraction"]["median"] >= 0.80
        ),
        "carry_contains_sustained_lifted_motion": (
            carry["metrics"]["lifted_moving_fraction"]["median"] >= 0.25
        ),
        "kick_stays_below_five_centimeter_lift": (
            kick["metrics"]["maximum_lift_m"]["max"] < LIFT_THRESHOLD_M
        ),
        "kick_contains_ground_level_object_motion": (
            kick["metrics"]["ground_moving_fraction"]["median"] >= 0.20
        ),
    }
    passed = all(checks.values())
    result = {
        "protocol": "sugar_contact_event_reference_corpus_audit_v1",
        "passed": passed,
        "checks": checks,
        "thresholds": {
            "lift_m": LIFT_THRESHOLD_M,
            "moving_mps": MOVE_THRESHOLD_MPS,
            "minimum_motions_per_task": 95,
            "minimum_role_fraction": 0.80,
        },
        "source_contract": {
            "contact_labels_are_binary_reference_proxy": True,
            "contact_labels_are_not_tactile_force": True,
            "role_is_assigned_by_nearest_named_hand_or_foot_center": True,
            "physical_contact_force_target_present": False,
        },
        "tasks": summaries,
        "automatic_next_branch": (
            "collect_actual_rollout_contact_event_targets_then_train_multitask_predictor"
            if passed
            else "repair_reference_event_labels_before_model_training"
        ),
        "claim_boundary": (
            "Passing proves that the selected demos contain separable contact-role and "
            "object-motion event labels. It does not prove physical contact-force "
            "accuracy, predictor generalization, reward usefulness, or policy following."
        ),
    }
    return result, records


def main() -> None:
    args = parse_args()
    result, records = audit(args.data_root.expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with (output / "motions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
