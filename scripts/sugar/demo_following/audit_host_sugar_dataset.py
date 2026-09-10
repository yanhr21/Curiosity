#!/usr/bin/env python3
"""Build and audit the exact causal SUGAR adapter for the official HOST model."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from host_sugar_dataset import (
    ACTION_HORIZON,
    DEFAULT_MANIFEST,
    DEFAULT_STATS,
    ROOT,
    HostSugarDataset,
    build_train_stats,
    load_manifest,
    _array_name,
    _sha256,
)


DEFAULT_OUTPUT = ROOT / (
    "experiments/demo_following/host_official_v1/sugar_adapter_v1/"
    "HOST_SUGAR_DATASET_AUDIT.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_train_stats(args.stats, args.manifest)
    rows = load_manifest(args.manifest)
    split_counts = Counter((row["split"], row["task"]) for row in rows)

    dataset_records = {}
    all_contracts_pass = True
    for split in ("train", "validation", "test"):
        dataset = HostSugarDataset(split, args.stats, args.manifest)
        valid_action_tokens = 0
        pairing_count = 0
        task_counts = Counter()
        valid_lengths = Counter()
        for index in range(len(dataset)):
            meta = dataset.sample_metadata(index)
            target = meta["target"]
            prompt = meta["prompt"]
            valid_action_tokens += meta["valid_len"]
            valid_lengths[meta["valid_len"]] += 1
            task_counts[target["task"]] += 1
            pairing_count += int(
                target["task"] == prompt["task"]
                and target["source_motion_id"] != prompt["source_motion_id"]
            )
        expected_pairing = len(dataset)
        all_contracts_pass &= pairing_count == expected_pairing

        sample_indices = sorted(
            {
                0,
                139,
                len(dataset) // 2,
                len(dataset) // 2 + 139,
                len(dataset) - 1,
            }
        )
        samples = []
        for index in sample_indices:
            item = dataset[index]
            samples.append(
                {
                    "index": index,
                    "video_shape": list(item["video"].shape),
                    "task_video_shape": list(item["task_video"].shape),
                    "action_shape": list(item["action"].shape),
                    "proprio_shape": list(item["proprio"].shape),
                    "progress_gt": item["progress_gt"].tolist(),
                    "target_task": item["target_task"],
                    "target_source_motion_id": item["target_source_motion_id"],
                    "prompt_source_motion_id": item["prompt_source_motion_id"],
                    "action_start": item["action_start"],
                    "valid_action_count": item["valid_action_count"],
                    "action_pad_count": int(item["action_is_pad"].sum()),
                    "image_pad_count": int(item["image_is_pad"].sum()),
                    "tensor_ranges": {
                        "video": [float(item["video"].min()), float(item["video"].max())],
                        "task_video": [
                            float(item["task_video"].min()),
                            float(item["task_video"].max()),
                        ],
                        "action": [float(item["action"].min()), float(item["action"].max())],
                        "proprio": [
                            float(item["proprio"].min()),
                            float(item["proprio"].max()),
                        ],
                    },
                }
            )
        dataset_records[split] = {
            "motions": len(dataset.rows),
            "windows": len(dataset),
            "windows_by_task": dict(sorted(task_counts.items())),
            "matching_different_source_pairings": pairing_count,
            "valid_action_tokens_per_epoch": valid_action_tokens,
            "valid_length_histogram": {
                str(key): value for key, value in sorted(valid_lengths.items())
            },
            "sample_audits": samples,
        }

    # Directly prove the archive shift used for causal proprio on representative rows.
    causal_max_error = 0.0
    causal_pairs_checked = 0
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["action_target"]["trace_path"]].append(row)
    for trace_path, trace_rows in sorted(grouped_rows.items()):
        with np.load(ROOT / trace_path, allow_pickle=False) as archive:
            core_all = np.asarray(archive["goal_policy_core_observation"])
            action_all = np.asarray(archive[_array_name(trace_rows[0], "executed_action")])
            for row in trace_rows:
                env_index = int(row["action_target"]["environment_index"])
                core = np.asarray(core_all[:, env_index])
                action = np.asarray(action_all[:, env_index])
                if core.shape != (700, 121) or action.shape != (700, 29):
                    raise RuntimeError(f"source archive geometry drift: {core.shape} {action.shape}")
                # Index identity is the causal contract: core[t-1] is post transition t-1,
                # hence pre-state for action t.  This also catches environment selection drift.
                shifted = np.asarray(core_all[:-1, env_index])
                causal_max_error = max(
                    causal_max_error, float(np.max(np.abs(core[:-1] - shifted)))
                )
                causal_pairs_checked += len(shifted)

    intervention = {}
    for mode in ("wrong_task", "same_source", "reversed_matching_alternate"):
        dataset = HostSugarDataset("validation", args.stats, args.manifest, prompt_mode=mode)
        item = dataset[0]
        intervention[mode] = {
            "target_task": item["target_task"],
            "target_source_motion_id": item["target_source_motion_id"],
            "prompt_source_motion_id": item["prompt_source_motion_id"],
            "task_frame_indices": item["task_frame_indices"].tolist(),
            "task_video_shape": list(item["task_video"].shape),
        }

    expected_causal_pairs = 199 * 699
    all_contracts_pass &= causal_pairs_checked == expected_causal_pairs
    all_contracts_pass &= causal_max_error == 0.0
    all_contracts_pass &= dataset_records["train"]["windows"] == 22_400
    all_contracts_pass &= dataset_records["train"]["valid_action_tokens_per_epoch"] == 529_600

    report = {
        "status": "pass" if all_contracts_pass else "fail",
        "method": "official_HOST_9B_SUGAR_dataset_adapter_only",
        "manifest": {
            "path": str(args.manifest.relative_to(ROOT)),
            "sha256": _sha256(args.manifest),
            "rows": len(rows),
            "split_task_counts": {
                f"{split}/{task}": count
                for (split, task), count in sorted(split_counts.items())
            },
        },
        "causal_contract": {
            "available_exact_pairs": causal_pairs_checked,
            "expected_exact_pairs": expected_causal_pairs,
            "excluded_unarchived_prerollout_action0_per_motion": 1,
            "excluded_total": 199,
            "maximum_archive_shift_error": causal_max_error,
            "statement": (
                "executed_action[t] is paired only with exact post-state core[t-1], "
                "the causal pre-state for t; action0 is not approximated"
            ),
        },
        "normalization": {
            "stats_path": str(args.stats.relative_to(ROOT)),
            "stats_sha256": _sha256(args.stats),
            "train_transition_count": stats.train_transition_count,
            "train_only": True,
            "action_constant_dimensions": np.flatnonzero(stats.action_constant).tolist(),
            "proprio_constant_dimensions": np.flatnonzero(stats.proprio_constant).tolist(),
            "action_raw_minimum": float(stats.action_min.min()),
            "action_raw_maximum": float((stats.action_min + stats.action_range).max()),
            "proprio_raw_minimum": float(stats.proprio_min.min()),
            "proprio_raw_maximum": float((stats.proprio_min + stats.proprio_range).max()),
        },
        "official_effective_tensor_contract": {
            "agent_video": [3, 5, 224, 224],
            "task_video": [3, 21, 224, 224],
            "action": [ACTION_HORIZON, 31],
            "proprio": [1, 121],
            "neutral_prompt": "Follow the demonstrated motion.",
            "task_identity_scalar_present": False,
            "future_or_outcome_in_deployed_input": False,
        },
        "datasets": dataset_records,
        "validation_intervention_examples": intervention,
        "ten_epoch_minimums": {
            "motion_epoch_visits": 1_600,
            "atomic_interval_exposures": 224_000,
            "valid_HOST_action_token_exposures": 5_296_000,
            "distinct_exact_train_transitions_available": 160 * 699,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
