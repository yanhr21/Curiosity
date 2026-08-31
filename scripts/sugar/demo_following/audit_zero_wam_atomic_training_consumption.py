#!/usr/bin/env python3
"""Prove full atomic-interval consumption by official Zero-WAM training.

This script audits loader/forward evidence only. It contains no model, adapter,
optimizer, surrogate loss or training implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


PROTOCOL = "official_zero_wam_sugar_atomic_training_consumption_v1"
SCHEDULE_PROTOCOL = "sugar_zero_wam_bounded_posttraining_schedule_v1"
EXPECTED_SCHEDULE_SHA256 = (
    "0f30252c0315855a1154d2c9f68d78b283cf35d2eee772a16692f5b0f1882ec1"
)
MINIMUM_EPOCHS = 10
TRAJECTORIES_PER_EPOCH = 160
INTERVALS_PER_TRAJECTORY = 140
ACTIONS_PER_INTERVAL = 5
INTERVALS_PER_EPOCH = 22_400
MINIMUM_INTERVAL_EXPOSURES = 224_000
MINIMUM_ACTION_EXPOSURES = 1_120_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-schedule", type=Path)
    parser.add_argument("--consumption-log", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def evaluate(
    schedule: dict[str, Any],
    schedule_sha256: str,
    rows: list[dict[str, Any]],
    consumption_log_sha256: str,
) -> dict[str, Any]:
    trajectory_records = schedule.get("trajectory_records", [])
    epochs = schedule.get("epochs", [])
    records_by_id = {
        record.get("trajectory_id"): record
        for record in trajectory_records
        if isinstance(record, dict) and isinstance(record.get("trajectory_id"), str)
    }
    schedule_checks = {
        "schedule_protocol_exact": schedule.get("protocol") == SCHEDULE_PROTOCOL,
        "exact_frozen_schedule_hash": schedule_sha256 == EXPECTED_SCHEDULE_SHA256,
        "exact_160_schedule_trajectories": (
            len(trajectory_records) == len(records_by_id) == TRAJECTORIES_PER_EPOCH
        ),
        "at_least_10_frozen_schedule_epochs": len(epochs) >= MINIMUM_EPOCHS,
        "consumption_log_hash_recorded": valid_sha256(consumption_log_sha256),
    }

    consumption_indices = [row.get("consumption_index") for row in rows]
    index_sequence_exact = (
        all(isinstance(index, int) for index in consumption_indices)
        and consumption_indices == list(range(len(rows)))
    )
    rows_by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    valid_epoch_indices = True
    for row in rows:
        epoch_index = row.get("epoch_index")
        if not isinstance(epoch_index, int) or epoch_index < 0:
            valid_epoch_indices = False
            continue
        rows_by_epoch[epoch_index].append(row)
    completed_epochs = sorted(rows_by_epoch)
    contiguous_epochs = completed_epochs == list(range(len(completed_epochs)))

    identity_exact = True
    interval_metadata_exact = True
    all_train_only = True
    all_forward_consumed = True
    all_joint_targets_present = True
    no_forbidden_input = True
    all_forward_fingerprints_recorded = True
    forward_fingerprints_by_epoch: dict[int, set[str]] = defaultdict(set)
    batch_indices: list[int] = []
    optimizer_steps: list[int] = []
    observed_orders: list[list[str]] = []
    every_epoch_complete = valid_epoch_indices and bool(completed_epochs)
    packed_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)

    for epoch_index in completed_epochs:
        epoch_rows = rows_by_epoch[epoch_index]
        by_position: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in epoch_rows:
            position = row.get("epoch_trajectory_position")
            if isinstance(position, int):
                by_position[position].append(row)
            else:
                every_epoch_complete = False
            sample_id = row.get("packed_sample_id")
            if isinstance(sample_id, str) and sample_id:
                packed_groups[(epoch_index, sample_id)].append(row)
            else:
                interval_metadata_exact = False
            batch_index = row.get("batch_index")
            optimizer_step = row.get("optimizer_step")
            if isinstance(batch_index, int) and batch_index >= 0:
                batch_indices.append(batch_index)
            else:
                interval_metadata_exact = False
            if isinstance(optimizer_step, int) and optimizer_step >= 0:
                optimizer_steps.append(optimizer_step)
            else:
                interval_metadata_exact = False

        every_epoch_complete = every_epoch_complete and (
            len(epoch_rows) == INTERVALS_PER_EPOCH
            and sorted(by_position) == list(range(TRAJECTORIES_PER_EPOCH))
            and all(len(by_position[position]) == INTERVALS_PER_TRAJECTORY for position in by_position)
        )
        observed_order: list[str] = []
        for position in range(TRAJECTORIES_PER_EPOCH):
            position_rows = sorted(
                by_position.get(position, []), key=lambda row: row.get("atomic_interval_index", -1)
            )
            if len(position_rows) != INTERVALS_PER_TRAJECTORY:
                continue
            trajectory_ids = {row.get("trajectory_id") for row in position_rows}
            trajectory_id = next(iter(trajectory_ids)) if len(trajectory_ids) == 1 else None
            observed_order.append(trajectory_id if isinstance(trajectory_id, str) else "")
            record = records_by_id.get(trajectory_id, {})
            expected_ranges = record.get("atomic_action_ranges_50hz", [])
            atomic_indices = [row.get("atomic_interval_index") for row in position_rows]
            every_epoch_complete = every_epoch_complete and (
                atomic_indices == list(range(INTERVALS_PER_TRAJECTORY))
                and len(trajectory_ids) == 1
            )
            for row in position_rows:
                interval_index = row.get("atomic_interval_index")
                expected_range = (
                    expected_ranges[interval_index]
                    if isinstance(interval_index, int)
                    and 0 <= interval_index < len(expected_ranges)
                    else None
                )
                identity_exact = identity_exact and (
                    row.get("source_manifest_row_sha256")
                    == record.get("source_manifest_row_sha256")
                    and row.get("prompt_sequence_sha256")
                    == record.get("prompt_sequence_sha256")
                    and row.get("robot_sequence_sha256")
                    == record.get("robot_sequence_sha256")
                    and row.get("action_trace_path") == record.get("action_trace_path")
                    and row.get("action_trace_environment_index")
                    == record.get("action_trace_environment_index")
                )
                interval_metadata_exact = interval_metadata_exact and (
                    isinstance(expected_range, list)
                    and isinstance(row.get("action_start"), int)
                    and isinstance(row.get("action_end"), int)
                    and row.get("action_start") == expected_range[0]
                    and row.get("action_end") == expected_range[1]
                    and row.get("action_end") - row.get("action_start") == ACTIONS_PER_INTERVAL
                    and isinstance(row.get("packed_interval_start"), int)
                    and isinstance(row.get("packed_interval_end_exclusive"), int)
                    and isinstance(interval_index, int)
                    and row.get("packed_interval_start") <= interval_index
                    and row.get("packed_interval_end_exclusive") > interval_index
                )
                all_train_only = all_train_only and row.get("split") == "train"
                all_forward_consumed = (
                    all_forward_consumed and row.get("consumed_by_official_forward") is True
                )
                all_joint_targets_present = all_joint_targets_present and all(
                    row.get(name) is True
                    for name in (
                        "video_flow_target_present",
                        "action_flow_target_present",
                        "ifp_target_present",
                    )
                )
                no_forbidden_input = (
                    no_forbidden_input
                    and row.get("future_target_used_as_input") is False
                    and row.get("outcome_label_used_as_input") is False
                )
                all_forward_fingerprints_recorded = (
                    all_forward_fingerprints_recorded
                    and valid_sha256(row.get("official_forward_input_sha256"))
                )
                if valid_sha256(row.get("official_forward_input_sha256")):
                    forward_fingerprints_by_epoch[epoch_index].add(
                        row["official_forward_input_sha256"]
                    )
        observed_orders.append(observed_order)

    schedule_orders_match = (
        len(observed_orders) >= MINIMUM_EPOCHS
        and len(observed_orders) <= len(epochs)
        and all(
            observed_orders[index] == epochs[index].get("trajectory_order")
            for index in range(len(observed_orders))
        )
    )
    packing_exact = bool(packed_groups)
    for group_rows in packed_groups.values():
        trajectory_ids = {row.get("trajectory_id") for row in group_rows}
        epoch_indices = {row.get("epoch_index") for row in group_rows}
        starts = {row.get("packed_interval_start") for row in group_rows}
        ends = {row.get("packed_interval_end_exclusive") for row in group_rows}
        raw_interval_indices = [row.get("atomic_interval_index") for row in group_rows]
        interval_indices = (
            sorted(raw_interval_indices)
            if all(isinstance(value, int) for value in raw_interval_indices)
            else []
        )
        packing_exact = packing_exact and (
            len(trajectory_ids) == len(epoch_indices) == len(starts) == len(ends) == 1
            and bool(interval_indices)
        )
        if len(starts) == 1 and len(ends) == 1:
            start = next(iter(starts))
            end = next(iter(ends))
            packing_exact = packing_exact and (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= INTERVALS_PER_TRAJECTORY
                and interval_indices == list(range(start, end))
            )

    total_actions = len(rows) * ACTIONS_PER_INTERVAL
    unique_batch_indices = sorted(set(batch_indices))
    unique_optimizer_steps = sorted(set(optimizer_steps))
    checks = {
        **schedule_checks,
        "global_consumption_index_exact": index_sequence_exact,
        "at_least_10_contiguous_epochs_consumed": (
            contiguous_epochs and len(completed_epochs) >= MINIMUM_EPOCHS
        ),
        "every_epoch_consumes_160x140_intervals_exactly_once": every_epoch_complete,
        "trajectory_and_interval_order_matches_frozen_schedule": schedule_orders_match,
        "all_raw_source_identities_match_schedule": identity_exact,
        "all_action_ranges_are_exact_five_action_intervals": interval_metadata_exact,
        "packed_samples_are_contiguous_and_never_cross_trajectory": packing_exact,
        "all_rows_are_train_split_only": all_train_only,
        "every_interval_reaches_official_forward": all_forward_consumed,
        "every_interval_uses_joint_video_action_ifp_targets": all_joint_targets_present,
        "future_and_outcome_labels_never_enter_inputs": no_forbidden_input,
        "every_forward_input_fingerprint_recorded": all_forward_fingerprints_recorded,
        "every_epoch_has_22400_distinct_forward_inputs": (
            len(forward_fingerprints_by_epoch) == len(completed_epochs)
            and all(
                len(forward_fingerprints_by_epoch[epoch_index]) == INTERVALS_PER_EPOCH
                for epoch_index in completed_epochs
            )
        ),
        "batch_and_optimizer_indices_monotonic": (
            bool(batch_indices)
            and bool(optimizer_steps)
            and batch_indices == sorted(batch_indices)
            and optimizer_steps == sorted(optimizer_steps)
        ),
        "batch_indices_are_contiguous_from_zero": (
            bool(unique_batch_indices)
            and unique_batch_indices == list(range(unique_batch_indices[-1] + 1))
        ),
        "optimizer_steps_are_contiguous_from_zero": (
            bool(unique_optimizer_steps)
            and unique_optimizer_steps == list(range(unique_optimizer_steps[-1] + 1))
        ),
        "at_least_224000_atomic_interval_exposures": len(rows) >= MINIMUM_INTERVAL_EXPOSURES,
        "at_least_1120000_action_exposures": total_actions >= MINIMUM_ACTION_EXPOSURES,
    }
    passed = all(checks.values())
    return {
        "protocol": PROTOCOL,
        "passed": passed,
        "schedule_sha256": schedule_sha256,
        "consumption_log_sha256": consumption_log_sha256,
        "completed_full_epochs": len(completed_epochs),
        "atomic_interval_exposures": len(rows),
        "action_exposures": total_actions,
        "packed_sample_count": len(packed_groups),
        "batch_index_min": unique_batch_indices[0] if unique_batch_indices else None,
        "batch_index_max": unique_batch_indices[-1] if unique_batch_indices else None,
        "distinct_batch_count": len(unique_batch_indices),
        "optimizer_step_min": unique_optimizer_steps[0] if unique_optimizer_steps else None,
        "optimizer_step_max": unique_optimizer_steps[-1] if unique_optimizer_steps else None,
        "distinct_optimizer_step_count": len(unique_optimizer_steps),
        "checks": checks,
        "automatic_next_branch": (
            "continue_official_training_completion_audit"
            if passed
            else "reject_incomplete_or_nonconformant_training_data_consumption"
        ),
        "claim_boundary": (
            "Passing proves exact full-data loader/forward consumption, not optimization quality, "
            "checkpoint quality, selected-demo following or physical success."
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def make_consumption_rows(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    records_by_id = {
        record["trajectory_id"]: record for record in schedule["trajectory_records"]
    }
    rows: list[dict[str, Any]] = []
    consumption_index = 0
    batch_index = 0
    for epoch in schedule["epochs"]:
        epoch_index = epoch["epoch_index"]
        for position, trajectory_id in enumerate(epoch["trajectory_order"]):
            record = records_by_id[trajectory_id]
            for pack_start in range(0, INTERVALS_PER_TRAJECTORY, 4):
                pack_end = min(pack_start + 4, INTERVALS_PER_TRAJECTORY)
                sample_id = f"epoch{epoch_index}/position{position}/sample{pack_start // 4}"
                for interval_index in range(pack_start, pack_end):
                    action_start, action_end = record["atomic_action_ranges_50hz"][interval_index]
                    rows.append(
                        {
                            "consumption_index": consumption_index,
                            "epoch_index": epoch_index,
                            "epoch_trajectory_position": position,
                            "trajectory_id": trajectory_id,
                            "atomic_interval_index": interval_index,
                            "action_start": action_start,
                            "action_end": action_end,
                            "split": "train",
                            "source_manifest_row_sha256": record["source_manifest_row_sha256"],
                            "prompt_sequence_sha256": record["prompt_sequence_sha256"],
                            "robot_sequence_sha256": record["robot_sequence_sha256"],
                            "action_trace_path": record["action_trace_path"],
                            "action_trace_environment_index": record["action_trace_environment_index"],
                            "packed_sample_id": sample_id,
                            "packed_interval_start": pack_start,
                            "packed_interval_end_exclusive": pack_end,
                            "batch_index": batch_index,
                            "optimizer_step": batch_index // 8,
                            "consumed_by_official_forward": True,
                            "video_flow_target_present": True,
                            "action_flow_target_present": True,
                            "ifp_target_present": True,
                            "future_target_used_as_input": False,
                            "outcome_label_used_as_input": False,
                            "official_forward_input_sha256": hashlib.sha256(
                                f"forward/{consumption_index}".encode()
                            ).hexdigest(),
                        }
                    )
                    consumption_index += 1
                batch_index += 1
    return rows


def make_fixture() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trajectory_records = []
    trajectory_ids = []
    for index in range(TRAJECTORIES_PER_EPOCH):
        task = "CarryBox" if index % 2 == 0 else "KickBox"
        trajectory_id = f"train/{task}/{index}"
        trajectory_ids.append(trajectory_id)
        trajectory_records.append(
            {
                "trajectory_id": trajectory_id,
                "source_manifest_row_sha256": hashlib.sha256(f"row/{index}".encode()).hexdigest(),
                "prompt_sequence_sha256": hashlib.sha256(f"prompt/{index}".encode()).hexdigest(),
                "robot_sequence_sha256": hashlib.sha256(f"robot/{index}".encode()).hexdigest(),
                "action_trace_path": f"trace/{index}.npz",
                "action_trace_environment_index": index,
                "atomic_action_ranges_50hz": [
                    [5 * interval, 5 * (interval + 1)]
                    for interval in range(INTERVALS_PER_TRAJECTORY)
                ],
            }
        )

    def order_key(epoch_index: int, trajectory_id: str) -> str:
        payload = f"fixture/{epoch_index}/{trajectory_id}"
        return hashlib.sha256(payload.encode()).hexdigest()

    epochs = []
    for epoch_index in range(MINIMUM_EPOCHS):
        carry = sorted(
            (value for value in trajectory_ids if "/CarryBox/" in value),
            key=lambda value: order_key(epoch_index, value),
        )
        kick = sorted(
            (value for value in trajectory_ids if "/KickBox/" in value),
            key=lambda value: order_key(epoch_index, value),
        )
        order = []
        for carry_id, kick_id in zip(carry, kick, strict=True):
            order.extend([carry_id, kick_id])
        epochs.append({"epoch_index": epoch_index, "trajectory_order": order})
    schedule = {
        "protocol": SCHEDULE_PROTOCOL,
        "trajectory_records": trajectory_records,
        "epochs": epochs,
    }
    return schedule, make_consumption_rows(schedule)


def run_self_test(training_schedule: Path | None = None) -> None:
    global EXPECTED_SCHEDULE_SHA256
    if training_schedule is None:
        schedule, rows = make_fixture()
        production_schedule = False
    else:
        schedule = read_json(training_schedule)
        rows = make_consumption_rows(schedule)
        production_schedule = True
    with tempfile.TemporaryDirectory(prefix="zero_wam_consumption_audit_") as temp:
        root = Path(temp)
        schedule_path = root / "schedule.json"
        log_path = root / "consumption.jsonl"
        if production_schedule:
            assert training_schedule is not None
            schedule_path.write_bytes(training_schedule.read_bytes())
            assert file_sha256(schedule_path) == EXPECTED_SCHEDULE_SHA256
        else:
            write_json(schedule_path, schedule)
            EXPECTED_SCHEDULE_SHA256 = file_sha256(schedule_path)
        write_jsonl(log_path, rows)
        log_hash = file_sha256(log_path)
        positive = evaluate(schedule, EXPECTED_SCHEDULE_SHA256, rows, log_hash)
        assert positive["passed"] is True, positive

        undertrained = evaluate(
            schedule,
            EXPECTED_SCHEDULE_SHA256,
            rows[:-INTERVALS_PER_EPOCH],
            log_hash,
        )
        assert undertrained["checks"]["at_least_10_contiguous_epochs_consumed"] is False

        original_interval = rows[1]["atomic_interval_index"]
        rows[1]["atomic_interval_index"] = 0
        duplicated_result = evaluate(schedule, EXPECTED_SCHEDULE_SHA256, rows, log_hash)
        assert duplicated_result["checks"]["every_epoch_consumes_160x140_intervals_exactly_once"] is False
        rows[1]["atomic_interval_index"] = original_interval

        first_trajectory = INTERVALS_PER_TRAJECTORY
        for index in range(INTERVALS_PER_TRAJECTORY):
            rows[index]["epoch_trajectory_position"] = 1
            rows[first_trajectory + index]["epoch_trajectory_position"] = 0
        wrong_order_result = evaluate(schedule, EXPECTED_SCHEDULE_SHA256, rows, log_hash)
        assert wrong_order_result["checks"]["trajectory_and_interval_order_matches_frozen_schedule"] is False
        for index in range(INTERVALS_PER_TRAJECTORY):
            rows[index]["epoch_trajectory_position"] = 0
            rows[first_trajectory + index]["epoch_trajectory_position"] = 1

        original_sample = rows[INTERVALS_PER_TRAJECTORY]["packed_sample_id"]
        rows[INTERVALS_PER_TRAJECTORY]["packed_sample_id"] = rows[0]["packed_sample_id"]
        crossed_result = evaluate(schedule, EXPECTED_SCHEDULE_SHA256, rows, log_hash)
        assert crossed_result["checks"]["packed_samples_are_contiguous_and_never_cross_trajectory"] is False
        rows[INTERVALS_PER_TRAJECTORY]["packed_sample_id"] = original_sample

        rows[0]["split"] = "validation"
        leaked_result = evaluate(schedule, EXPECTED_SCHEDULE_SHA256, rows, log_hash)
        assert leaked_result["checks"]["all_rows_are_train_split_only"] is False
        rows[0]["split"] = "train"

        rows[0]["video_flow_target_present"] = False
        action_only_result = evaluate(schedule, EXPECTED_SCHEDULE_SHA256, rows, log_hash)
        assert action_only_result["checks"]["every_interval_uses_joint_video_action_ifp_targets"] is False
        rows[0]["video_flow_target_present"] = True

        original_forward_hash = rows[1]["official_forward_input_sha256"]
        rows[1]["official_forward_input_sha256"] = rows[0]["official_forward_input_sha256"]
        collapsed_result = evaluate(schedule, EXPECTED_SCHEDULE_SHA256, rows, log_hash)
        assert collapsed_result["checks"]["every_epoch_has_22400_distinct_forward_inputs"] is False
        rows[1]["official_forward_input_sha256"] = original_forward_hash

        changed_optimizer_rows = [row for row in rows if row["optimizer_step"] == 1]
        for row in changed_optimizer_rows:
            row["optimizer_step"] = 2
        optimizer_gap_result = evaluate(schedule, EXPECTED_SCHEDULE_SHA256, rows, log_hash)
        assert optimizer_gap_result["checks"]["optimizer_steps_are_contiguous_from_zero"] is False
        for row in changed_optimizer_rows:
            row["optimizer_step"] = 1
        print(
            json.dumps(
                {
                    "self_test_passed": True,
                    "positive_full_scale_fixture": {
                        "production_schedule": production_schedule,
                        "epochs": MINIMUM_EPOCHS,
                        "atomic_intervals": len(rows),
                        "actions": len(rows) * ACTIONS_PER_INTERVAL,
                    },
                    "rejected": [
                        "nine_epoch_undertraining",
                        "duplicated_interval",
                        "trajectory_reordering",
                        "cross_trajectory_packing",
                        "heldout_leakage",
                        "action_only_target",
                        "collapsed_forward_input",
                        "optimizer_step_gap",
                    ],
                    "fixture_claim_boundary": "synthetic contract test only; not model evidence",
                },
                sort_keys=True,
            )
        )


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args.training_schedule)
        return
    if args.training_schedule is None or args.consumption_log is None or args.output_dir is None:
        raise SystemExit("--training-schedule, --consumption-log and --output-dir are required")
    schedule = read_json(args.training_schedule)
    rows = read_jsonl(args.consumption_log)
    result = evaluate(
        schedule,
        file_sha256(args.training_schedule),
        rows,
        file_sha256(args.consumption_log),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "ATOMIC_TRAINING_CONSUMPTION_AUDIT.json"
    write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
