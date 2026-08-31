#!/usr/bin/env python3
"""Freeze the full-data exposure floor for bounded official Zero-WAM training.

This script does not implement or launch a model.  It schedules complete SUGAR
training trajectories so the future official loader can pack contiguous atomic
video/action intervals according to its released schema without dropping,
reordering or cherry-picking data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL = "sugar_zero_wam_bounded_posttraining_schedule_v1"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "24cc2b99b26e3136acb1508e4c1d8a6193702d25a43005978e9920179fc366c8"
)
SCHEDULE_SEED = 271_500
MINIMUM_FULL_EPOCHS = 10
EXPECTED_TASK_TRAJECTORIES = {"CarryBox": 80, "KickBox": 80}
EXPECTED_TRAJECTORIES = 160
EXPECTED_INTERVALS_PER_TRAJECTORY = 140
EXPECTED_ACTIONS_PER_TRAJECTORY = 700
EXPECTED_INTERVALS_PER_EPOCH = 22_400
EXPECTED_ACTIONS_PER_EPOCH = 112_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return bytes_sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def order_key(epoch_index: int, trajectory_id: str) -> str:
    return bytes_sha256(
        f"zero-wam-sugar-bounded-v1/{SCHEDULE_SEED}/{epoch_index}/{trajectory_id}".encode()
    )


def maximum_task_prefix_difference(
    trajectory_order: list[str], task_by_trajectory: dict[str, str]
) -> int:
    counts: Counter[str] = Counter()
    maximum = 0
    for trajectory_id in trajectory_order:
        counts[task_by_trajectory[trajectory_id]] += 1
        maximum = max(maximum, abs(counts["CarryBox"] - counts["KickBox"]))
    return maximum


def main() -> None:
    args = parse_args()
    source_bytes = args.source_manifest.read_bytes()
    source_manifest_sha256 = bytes_sha256(source_bytes)
    source_rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines()]
    train_rows = [
        (line_index, row)
        for line_index, row in enumerate(source_rows)
        if row.get("split") == "train"
    ]

    task_counts: Counter[str] = Counter()
    trajectory_records: list[dict[str, Any]] = []
    chronology_exact = True
    deployed_inputs_exact = True
    stream_lengths_exact = True
    for line_index, row in sorted(
        train_rows,
        key=lambda item: (item[1]["task"], int(item[1]["source_motion_id"])),
    ):
        task = str(row["task"])
        motion_id = int(row["source_motion_id"])
        trajectory_id = f"train/{task}/{motion_id}"
        ranges = [list(map(int, pair)) for pair in row["robot_target"]["action_ranges_50hz"]]
        nonempty = [pair for pair in ranges if pair[1] > pair[0]]
        expected_ranges = [[5 * index, 5 * (index + 1)] for index in range(140)]
        chronology_exact = chronology_exact and nonempty == expected_ranges
        chronology_exact = chronology_exact and ranges[-1] == [700, 700]
        stream_lengths_exact = (
            stream_lengths_exact
            and len(row["prompt"]["frame_paths"]) == 64
            and len(row["robot_target"]["frame_paths"]) == 141
            and row["action_target"]["transition_count"] == 700
        )
        allowlist = row["deployed_input_allowlist"]
        deployed_inputs_exact = (
            deployed_inputs_exact
            and allowlist["future_target"] is False
            and allowlist["contact_or_success_label"] is False
            and allowlist["language"] is False
            and allowlist["selected_demo_id_scalar"] is False
            and allowlist["prompt_rgb"] is True
            and allowlist["causal_robot_rgb_history"] is True
            and allowlist["causal_executed_action_history_29d"] is True
            and allowlist["causal_generator_command_history_36d"] is True
            and allowlist["causal_tracker_observation_history_510d"] is True
        )
        task_counts[task] += 1
        trajectory_records.append(
            {
                "trajectory_id": trajectory_id,
                "task": task,
                "source_motion_id": motion_id,
                "source_manifest_line_index": line_index,
                "source_manifest_row_sha256": canonical_sha256(row),
                "prompt_sequence_sha256": row["prompt"]["sequence_sha256"],
                "robot_sequence_sha256": row["robot_target"][
                    "sequence_sha256_at_64_normalized_indices"
                ],
                "action_trace_path": row["action_target"]["trace_path"],
                "action_trace_environment_index": row["action_target"][
                    "environment_index"
                ],
                "atomic_action_ranges_50hz": nonempty,
                "atomic_interval_count": len(nonempty),
                "action_count": row["action_target"]["transition_count"],
                "packing_contract": (
                    "official loader may pack only contiguous chronological atomic intervals "
                    "within this trajectory; no interval may be dropped or reordered"
                ),
            }
        )

    trajectory_ids = [record["trajectory_id"] for record in trajectory_records]
    task_by_trajectory = {record["trajectory_id"]: record["task"] for record in trajectory_records}
    ids_by_task = {
        task: [record["trajectory_id"] for record in trajectory_records if record["task"] == task]
        for task in EXPECTED_TASK_TRAJECTORIES
    }
    epochs: list[dict[str, Any]] = []
    for epoch_index in range(MINIMUM_FULL_EPOCHS):
        carry_order = sorted(
            ids_by_task["CarryBox"], key=lambda value: order_key(epoch_index, value)
        )
        kick_order = sorted(
            ids_by_task["KickBox"], key=lambda value: order_key(epoch_index, value)
        )
        trajectory_order: list[str] = []
        first_task = "CarryBox" if epoch_index % 2 == 0 else "KickBox"
        for carry_id, kick_id in zip(carry_order, kick_order, strict=True):
            pair = [carry_id, kick_id] if first_task == "CarryBox" else [kick_id, carry_id]
            trajectory_order.extend(pair)
        epochs.append(
            {
                "epoch_index": epoch_index,
                "trajectory_order": trajectory_order,
                "trajectory_order_sha256": bytes_sha256(
                    ("\n".join(trajectory_order) + "\n").encode("utf-8")
                ),
                "task_trajectory_counts": dict(Counter(task_by_trajectory[value] for value in trajectory_order)),
                "maximum_task_prefix_count_difference": maximum_task_prefix_difference(
                    trajectory_order, task_by_trajectory
                ),
                "atomic_interval_exposures": EXPECTED_INTERVALS_PER_EPOCH,
                "action_exposures": EXPECTED_ACTIONS_PER_EPOCH,
            }
        )

    schedule = {
        "protocol": PROTOCOL,
        "source_manifest_sha256": source_manifest_sha256,
        "schedule_seed": SCHEDULE_SEED,
        "minimum_full_epochs": MINIMUM_FULL_EPOCHS,
        "trajectory_records": trajectory_records,
        "epochs": epochs,
        "extension_rule": (
            "if the official release requires more than ten epochs, generate each additional "
            "complete epoch with the same SHA256 ordering rule and never stop from validation/test"
        ),
        "budget_rule": (
            "formal training uses max(10 complete SUGAR epochs, official released post-training "
            "budget converted to complete epochs); no partial-coverage early stop"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = args.output_dir / "BOUNDED_POSTTRAINING_SCHEDULE.json"
    schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    schedule_sha256 = file_sha256(schedule_path)

    epoch_hashes = [epoch["trajectory_order_sha256"] for epoch in epochs]
    epoch_orders_valid = all(
        len(epoch["trajectory_order"]) == EXPECTED_TRAJECTORIES
        and len(set(epoch["trajectory_order"])) == EXPECTED_TRAJECTORIES
        and set(epoch["trajectory_order"]) == set(trajectory_ids)
        and epoch["task_trajectory_counts"] == EXPECTED_TASK_TRAJECTORIES
        and epoch["maximum_task_prefix_count_difference"] <= 1
        and epoch["atomic_interval_exposures"] == EXPECTED_INTERVALS_PER_EPOCH
        and epoch["action_exposures"] == EXPECTED_ACTIONS_PER_EPOCH
        for epoch in epochs
    )
    checks = {
        "bound_to_exact_immutable_source_manifest": (
            source_manifest_sha256 == EXPECTED_SOURCE_MANIFEST_SHA256
        ),
        "exact_160_unique_training_trajectories": len(trajectory_ids)
        == len(set(trajectory_ids))
        == EXPECTED_TRAJECTORIES,
        "exact_80_80_task_balance": dict(task_counts) == EXPECTED_TASK_TRAJECTORIES,
        "no_validation_or_test_trajectory": len(train_rows) == EXPECTED_TRAJECTORIES
        and all(record["trajectory_id"].startswith("train/") for record in trajectory_records),
        "all_prompt_robot_action_stream_lengths_exact": stream_lengths_exact,
        "all_140_atomic_intervals_are_contiguous_and_chronological": chronology_exact,
        "future_outcome_language_and_scalar_demo_id_excluded": deployed_inputs_exact,
        "exact_10_complete_epochs": len(epochs) == MINIMUM_FULL_EPOCHS,
        "every_epoch_contains_every_train_trajectory_once": epoch_orders_valid,
        "task_prefix_imbalance_never_exceeds_one": all(
            epoch["maximum_task_prefix_count_difference"] <= 1 for epoch in epochs
        ),
        "all_epoch_permutations_are_distinct": len(epoch_hashes)
        == len(set(epoch_hashes))
        == MINIMUM_FULL_EPOCHS,
        "exact_224000_atomic_interval_exposure_floor": (
            MINIMUM_FULL_EPOCHS * EXPECTED_INTERVALS_PER_EPOCH == 224_000
        ),
        "exact_1120000_action_exposure_floor": (
            MINIMUM_FULL_EPOCHS * EXPECTED_ACTIONS_PER_EPOCH == 1_120_000
        ),
        "official_loader_packing_preserves_full_trajectory_chronology": all(
            "no interval may be dropped or reordered" in record["packing_contract"]
            for record in trajectory_records
        ),
    }
    result = {
        "protocol": PROTOCOL,
        "passed": all(checks.values()),
        "source_manifest_sha256": source_manifest_sha256,
        "schedule_path": str(schedule_path),
        "schedule_sha256": schedule_sha256,
        "schedule_seed": SCHEDULE_SEED,
        "training_trajectory_count": EXPECTED_TRAJECTORIES,
        "task_trajectory_counts": dict(task_counts),
        "minimum_full_epochs": MINIMUM_FULL_EPOCHS,
        "atomic_intervals_per_epoch": EXPECTED_INTERVALS_PER_EPOCH,
        "actions_per_epoch": EXPECTED_ACTIONS_PER_EPOCH,
        "minimum_atomic_interval_exposures": 224_000,
        "minimum_action_exposures": 1_120_000,
        "epoch_order_sha256": epoch_hashes,
        "checks": checks,
        "claim_boundary": (
            "Passing freezes balanced full-trajectory coverage and a ten-epoch minimum exposure "
            "floor for future bounded post-training from an admitted official Zero-WAM checkpoint. "
            "It does not train a model or prove that ten epochs are sufficient for physical success."
        ),
    }
    result_path = args.output_dir / "BOUNDED_POSTTRAINING_SCHEDULE_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
