#!/usr/bin/env python3
"""Audit effective, non-overlapping diversity of the SUGAR Zero-WAM corpus.

This goes beyond counting manifest rows.  It verifies that the 22,400 training
intervals are balanced non-overlapping chunks from 160 distinct trajectories,
then measures exact chunk/sequence duplication and 29-DoF action-space rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SPLITS = {"train": 160, "validation": 20, "test": 19}
EXPECTED_TRAIN_TASK_TRAJECTORIES = {"CarryBox": 80, "KickBox": 80}
EXPECTED_TRAIN_TASK_CHUNKS = {"CarryBox": 11_200, "KickBox": 11_200}
EXPECTED_TRAIN_TRAJECTORIES = 160
EXPECTED_CHUNKS_PER_TRAJECTORY = 140
EXPECTED_ACTIONS_PER_CHUNK = 5
EXPECTED_ACTION_DIM = 29


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def overlap_count(left: set[str], right: set[str]) -> int:
    return len(left.intersection(right))


def effective_rank(eigenvalues: np.ndarray) -> float:
    positive = np.clip(eigenvalues.astype(np.float64), 0.0, None)
    total = float(positive.sum())
    if total <= 0.0:
        return 0.0
    probabilities = positive[positive > 0.0] / total
    return float(math.exp(float(-(probabilities * np.log(probabilities)).sum())))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("manifest must contain JSON objects")

    split_trajectory_counts: Counter[str] = Counter()
    train_task_trajectory_counts: Counter[str] = Counter()
    train_task_chunk_counts: Counter[str] = Counter()
    chunks_per_trajectory: dict[str, int] = {}
    phase_bin_counts: dict[str, Counter[int]] = defaultdict(Counter)
    prompt_hashes: dict[str, set[str]] = defaultdict(set)
    robot_hashes: dict[str, set[str]] = defaultdict(set)
    action_sequence_hashes: dict[str, set[str]] = defaultdict(set)
    action_chunk_hashes: dict[str, set[str]] = defaultdict(set)
    observation_chunk_hashes: dict[str, set[str]] = defaultdict(set)
    train_task_action_chunk_hashes: dict[str, set[str]] = defaultdict(set)
    train_action_chunks_total = 0
    train_observation_chunks_total = 0
    train_task_action_chunks_total: Counter[str] = Counter()
    train_actions_by_task: dict[str, list[np.ndarray]] = defaultdict(list)
    all_finite = True
    range_integrity = True
    terminal_range_integrity = True

    rows_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_trace[str(row["action_target"]["trace_path"])].append(row)

    for relative_trace, trace_rows in sorted(rows_by_trace.items()):
        trace_path = project_root / relative_trace
        with np.load(trace_path, allow_pickle=False) as trace:
            executed_all = trace["executed_action"]
            observation_all = trace["policy_observation_before"]
            for row in trace_rows:
                split = str(row["split"])
                task = str(row["task"])
                motion_id = int(row["source_motion_id"])
                key = f"{split}/{task}/{motion_id}"
                env_index = int(row["action_target"]["environment_index"])
                actions = np.asarray(executed_all[:, env_index], dtype=np.float32)
                observations = np.asarray(observation_all[:, env_index], dtype=np.float32)
                ranges = row["robot_target"]["action_ranges_50hz"]
                nonempty_ranges = [pair for pair in ranges if int(pair[1]) > int(pair[0])]

                split_trajectory_counts[split] += 1
                prompt_hashes[split].add(str(row["prompt"]["sequence_sha256"]))
                robot_hashes[split].add(
                    str(row["robot_target"]["sequence_sha256_at_64_normalized_indices"])
                )
                action_sequence_hashes[split].add(array_sha256(actions))
                chunks_per_trajectory[key] = len(nonempty_ranges)
                all_finite = all_finite and bool(np.isfinite(actions).all())
                all_finite = all_finite and bool(np.isfinite(observations).all())

                expected_start = 0
                for chunk_index, pair in enumerate(nonempty_ranges):
                    start, stop = map(int, pair)
                    range_integrity = range_integrity and start == expected_start
                    range_integrity = range_integrity and stop - start == EXPECTED_ACTIONS_PER_CHUNK
                    expected_start = stop
                    action_chunk = actions[start:stop]
                    observation_chunk = observations[start:stop]
                    action_hash = array_sha256(action_chunk)
                    observation_hash = array_sha256(observation_chunk)
                    action_chunk_hashes[split].add(action_hash)
                    observation_chunk_hashes[split].add(observation_hash)
                    if split == "train":
                        train_action_chunks_total += 1
                        train_observation_chunks_total += 1
                        train_task_chunk_counts[task] += 1
                        train_task_action_chunks_total[task] += 1
                        train_task_action_chunk_hashes[task].add(action_hash)
                        phase_bin = min(9, (10 * chunk_index) // EXPECTED_CHUNKS_PER_TRAJECTORY)
                        phase_bin_counts[task][phase_bin] += 1
                range_integrity = range_integrity and expected_start == 700
                terminal_range_integrity = terminal_range_integrity and ranges[-1] == [700, 700]

                if split == "train":
                    train_task_trajectory_counts[task] += 1
                    train_actions_by_task[task].append(actions)

    train_actions = np.concatenate(
        [np.concatenate(train_actions_by_task[task], axis=0) for task in sorted(train_actions_by_task)],
        axis=0,
    )
    action_std = train_actions.astype(np.float64).std(axis=0)
    action_covariance = np.cov(train_actions.astype(np.float64), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(action_covariance)
    max_eigenvalue = float(max(0.0, eigenvalues.max(initial=0.0)))
    numerical_rank = int((eigenvalues > max_eigenvalue * 1e-6).sum()) if max_eigenvalue > 0 else 0
    variable_action_dimensions = int((action_std > 1e-4).sum())
    action_effective_rank = effective_rank(eigenvalues)

    task_action_means = {
        task: np.concatenate(values, axis=0).astype(np.float64).mean(axis=0)
        for task, values in train_actions_by_task.items()
    }
    task_mean_l2 = float(
        np.linalg.norm(task_action_means["CarryBox"] - task_action_means["KickBox"])
    )

    heldout_splits = action_chunk_hashes["validation"].union(action_chunk_hashes["test"])
    heldout_observations = observation_chunk_hashes["validation"].union(
        observation_chunk_hashes["test"]
    )
    train_action_unique_fraction = len(action_chunk_hashes["train"]) / train_action_chunks_total
    train_observation_unique_fraction = (
        len(observation_chunk_hashes["train"]) / train_observation_chunks_total
    )
    task_action_unique_fractions = {
        task: len(train_task_action_chunk_hashes[task]) / train_task_action_chunks_total[task]
        for task in EXPECTED_TRAIN_TASK_TRAJECTORIES
    }
    train_action_heldout_overlap = overlap_count(action_chunk_hashes["train"], heldout_splits)
    train_observation_heldout_overlap = overlap_count(
        observation_chunk_hashes["train"], heldout_observations
    )

    trajectory_chunk_values = list(chunks_per_trajectory.values())
    train_trajectory_chunk_values = [
        count for key, count in chunks_per_trajectory.items() if key.startswith("train/")
    ]
    maximum_train_trajectory_fraction = (
        max(train_trajectory_chunk_values) / sum(train_trajectory_chunk_values)
    )
    phase_counts_serialized = {
        task: {str(index): int(phase_bin_counts[task][index]) for index in range(10)}
        for task in EXPECTED_TRAIN_TASK_TRAJECTORIES
    }

    checks = {
        "manifest_has_199_distinct_trajectory_rows": len(rows) == 199
        and len(chunks_per_trajectory) == 199,
        "split_trajectory_counts_exact": dict(split_trajectory_counts) == EXPECTED_SPLITS,
        "train_task_trajectory_counts_exactly_balanced": (
            dict(train_task_trajectory_counts) == EXPECTED_TRAIN_TASK_TRAJECTORIES
        ),
        "train_task_chunk_counts_exactly_balanced": (
            dict(train_task_chunk_counts) == EXPECTED_TRAIN_TASK_CHUNKS
        ),
        "exact_22400_nonoverlapping_training_chunks": train_action_chunks_total == 22_400,
        "all_nonterminal_chunks_are_contiguous_five_action_ranges": range_integrity,
        "all_terminal_ranges_are_exact_empty_post_states": terminal_range_integrity,
        "every_trajectory_has_140_nonempty_chunks": set(trajectory_chunk_values) == {140},
        "no_single_trajectory_dominates_training": maximum_train_trajectory_fraction <= 0.00625,
        "ten_phase_bins_balanced_per_task": all(
            set(phase_bin_counts[task].values()) == {1_120}
            for task in EXPECTED_TRAIN_TASK_TRAJECTORIES
        ),
        "all_action_and_observation_values_finite": all_finite,
        "all_train_prompt_sequences_unique": len(prompt_hashes["train"]) == 160,
        "all_train_robot_target_sequences_unique": len(robot_hashes["train"]) == 160,
        "at_least_95_percent_train_action_sequences_unique": (
            len(action_sequence_hashes["train"]) >= 152
        ),
        "at_least_90_percent_train_action_chunks_unique": train_action_unique_fraction >= 0.90,
        "at_least_85_percent_action_chunks_unique_per_task": all(
            fraction >= 0.85 for fraction in task_action_unique_fractions.values()
        ),
        "at_least_99_percent_train_observation_chunks_unique": (
            train_observation_unique_fraction >= 0.99
        ),
        "at_least_24_of_29_action_dimensions_vary": variable_action_dimensions >= 24,
        "action_covariance_numerical_rank_at_least_24": numerical_rank >= 24,
        "no_complete_prompt_sequence_cross_split_reuse": (
            not prompt_hashes["train"].intersection(
                prompt_hashes["validation"].union(prompt_hashes["test"])
            )
        ),
        "no_complete_robot_sequence_cross_split_reuse": (
            not robot_hashes["train"].intersection(
                robot_hashes["validation"].union(robot_hashes["test"])
            )
        ),
        "no_complete_action_sequence_cross_split_reuse": (
            not action_sequence_hashes["train"].intersection(
                action_sequence_hashes["validation"].union(action_sequence_hashes["test"])
            )
        ),
        "less_than_2_percent_exact_action_chunk_cross_split_overlap": (
            train_action_heldout_overlap / max(1, len(action_chunk_hashes["train"])) < 0.02
        ),
        "less_than_1_percent_exact_observation_chunk_cross_split_overlap": (
            train_observation_heldout_overlap
            / max(1, len(observation_chunk_hashes["train"]))
            < 0.01
        ),
    }
    result = {
        "protocol": "sugar_zero_wam_training_data_diversity_v1",
        "manifest_sha256": manifest_sha256,
        "passed": all(checks.values()),
        "checks": checks,
        "split_trajectory_counts": dict(split_trajectory_counts),
        "train_task_trajectory_counts": dict(train_task_trajectory_counts),
        "train_task_chunk_counts": dict(train_task_chunk_counts),
        "phase_bin_counts": phase_counts_serialized,
        "train_action_frame_count": int(train_actions.shape[0]),
        "train_action_chunk_count": train_action_chunks_total,
        "train_unique_action_chunk_count": len(action_chunk_hashes["train"]),
        "train_action_chunk_unique_fraction": train_action_unique_fraction,
        "train_unique_observation_chunk_count": len(observation_chunk_hashes["train"]),
        "train_observation_chunk_unique_fraction": train_observation_unique_fraction,
        "train_task_action_chunk_unique_fractions": task_action_unique_fractions,
        "train_unique_action_sequence_count": len(action_sequence_hashes["train"]),
        "train_unique_prompt_sequence_count": len(prompt_hashes["train"]),
        "train_unique_robot_target_sequence_count": len(robot_hashes["train"]),
        "train_action_heldout_exact_chunk_overlap_count": train_action_heldout_overlap,
        "train_observation_heldout_exact_chunk_overlap_count": train_observation_heldout_overlap,
        "action_dimension_std": action_std.tolist(),
        "variable_action_dimension_count_at_1e_4": variable_action_dimensions,
        "action_covariance_eigenvalues": eigenvalues.tolist(),
        "action_covariance_numerical_rank_at_relative_1e_6": numerical_rank,
        "action_covariance_effective_rank": action_effective_rank,
        "carry_kick_action_mean_l2": task_mean_l2,
        "maximum_single_train_trajectory_interval_fraction": maximum_train_trajectory_fraction,
        "automatic_next_branch": (
            "retain_bounded_posttraining_data_admission"
            if all(checks.values())
            else "close_training_and_repair_effective_data_diversity"
        ),
        "claim_boundary": (
            "Passing proves balanced, non-overlapping and measurably diverse data for the fixed "
            "two-task SUGAR post-training audit only. It does not make 199 trajectories equivalent "
            "to Zero-WAM foundation-scale pre-training data or prove model learnability."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "SUGAR_TRAINING_DATA_DIVERSITY.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
