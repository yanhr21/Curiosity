#!/usr/bin/env python3
"""Freeze model-agnostic held-out cases for the official Zero-WAM prompt gate.

The cases never execute or approximate Zero-WAM.  They bind the future official
runner to the immutable SUGAR manifest, source-motion-disjoint splits, fixed
causal phase anchors and matched diffusion-noise prompt interventions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL = "sugar_zero_wam_frozen_prompt_gate_cases_v1"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "24cc2b99b26e3136acb1508e4c1d8a6193702d25a43005978e9920179fc366c8"
)
EXPECTED_SPLIT_TRAJECTORIES = {"validation": 20, "test": 19}
EXPECTED_SPLIT_TASK_TRAJECTORIES = {
    "validation": {"CarryBox": 10, "KickBox": 10},
    "test": {"CarryBox": 10, "KickBox": 9},
}
ANCHOR_CHUNK_INDICES = tuple(7 + 14 * index for index in range(10))
CONDITION_NAMES = (
    "matched",
    "wrong_task",
    "reversed",
    "same_task_alternate",
    "masked_prompt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_noise_seed(group_id: str) -> int:
    digest = hashlib.sha256(f"zero-wam-sugar-prompt-gate-v1/{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def prompt_reference(
    row: dict[str, Any],
    *,
    frame_order: list[int],
    intervention: str,
) -> dict[str, Any]:
    return {
        "split": row["split"],
        "task": row["task"],
        "source_motion_id": row["source_motion_id"],
        "source_sequence_sha256": row["prompt"]["sequence_sha256"],
        "frame_order": frame_order,
        "intervention": intervention,
    }


def main() -> None:
    args = parse_args()
    source_manifest_sha256 = file_sha256(args.source_manifest)
    rows = [
        json.loads(line)
        for line in args.source_manifest.read_text(encoding="utf-8").splitlines()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("source manifest must contain JSON objects")

    by_key = {
        (str(row["split"]), str(row["task"]), int(row["source_motion_id"])): row
        for row in rows
    }
    heldout_rows = [row for row in rows if row["split"] in EXPECTED_SPLIT_TRAJECTORIES]
    split_trajectories: Counter[str] = Counter(str(row["split"]) for row in heldout_rows)
    split_task_trajectories: dict[str, Counter[str]] = {
        split: Counter(str(row["task"]) for row in heldout_rows if row["split"] == split)
        for split in EXPECTED_SPLIT_TRAJECTORIES
    }

    cases: list[dict[str, Any]] = []
    reference_integrity = True
    target_integrity = True
    language_disabled = True
    for row in sorted(
        heldout_rows,
        key=lambda value: (value["split"], value["task"], value["source_motion_id"]),
    ):
        split = str(row["split"])
        task = str(row["task"])
        motion_id = int(row["source_motion_id"])
        counterfactual = row["fixed_counterfactual_prompts"]

        wrong_spec = counterfactual["wrong_task"]
        alternate_spec = counterfactual["same_task_alternate"]
        wrong_key = (split, str(wrong_spec["task"]), int(wrong_spec["source_motion_id"]))
        alternate_key = (
            split,
            str(alternate_spec["task"]),
            int(alternate_spec["source_motion_id"]),
        )
        wrong_row = by_key.get(wrong_key)
        alternate_row = by_key.get(alternate_key)
        if wrong_row is None or alternate_row is None:
            raise ValueError(f"unresolved counterfactual for {split}/{task}/{motion_id}")

        normal_order = list(range(64))
        reversed_order = list(counterfactual["reversed"]["frame_order"])
        reference_integrity = reference_integrity and len(row["prompt"]["frame_paths"]) == 64
        reference_integrity = reference_integrity and normal_order == list(range(64))
        reference_integrity = reference_integrity and reversed_order == list(reversed(normal_order))
        reference_integrity = reference_integrity and wrong_row["task"] != task
        reference_integrity = reference_integrity and alternate_row["task"] == task
        reference_integrity = (
            reference_integrity and int(alternate_row["source_motion_id"]) != motion_id
        )
        reference_integrity = reference_integrity and wrong_row["split"] == split
        reference_integrity = reference_integrity and alternate_row["split"] == split
        language_disabled = language_disabled and row["language_condition"] is None
        language_disabled = (
            language_disabled and row["deployed_input_allowlist"]["language"] is False
        )

        robot_frames = row["robot_target"]["frame_paths"]
        action_ranges = row["robot_target"]["action_ranges_50hz"]
        target_integrity = target_integrity and len(robot_frames) == 141
        target_integrity = target_integrity and len(action_ranges) == 141

        for anchor_chunk_index in ANCHOR_CHUNK_INDICES:
            group_id = f"{split}/{task}/{motion_id}/chunk{anchor_chunk_index:03d}"
            first_future_frame = anchor_chunk_index + 1
            second_future_frame = anchor_chunk_index + 2
            target_integrity = target_integrity and second_future_frame < len(robot_frames)
            target_integrity = target_integrity and action_ranges[anchor_chunk_index] == [
                5 * anchor_chunk_index,
                5 * (anchor_chunk_index + 1),
            ]
            target_integrity = target_integrity and action_ranges[anchor_chunk_index + 1] == [
                5 * (anchor_chunk_index + 1),
                5 * (anchor_chunk_index + 2),
            ]

            conditions = {
                "matched": prompt_reference(
                    row, frame_order=normal_order, intervention="identity"
                ),
                "wrong_task": prompt_reference(
                    wrong_row, frame_order=normal_order, intervention="wrong_task_swap"
                ),
                "reversed": prompt_reference(
                    row, frame_order=reversed_order, intervention="reverse_exact_same_frames"
                ),
                "same_task_alternate": prompt_reference(
                    alternate_row,
                    frame_order=normal_order,
                    intervention="same_task_source_motion_swap",
                ),
                "masked_prompt": prompt_reference(
                    row,
                    frame_order=normal_order,
                    intervention="zero_cached_prompt_latents_after_official_preprocessing",
                ),
            }
            cases.append(
                {
                    "protocol": PROTOCOL,
                    "group_id": group_id,
                    "split": split,
                    "target": {
                        "task": task,
                        "source_motion_id": motion_id,
                        "source_manifest_key": [split, task, motion_id],
                        "causal_robot_frame_range": [0, anchor_chunk_index + 1],
                        "predicted_robot_frame_indices": [
                            first_future_frame,
                            second_future_frame,
                        ],
                        "action_ranges_50hz": [
                            action_ranges[anchor_chunk_index],
                            action_ranges[anchor_chunk_index + 1],
                        ],
                    },
                    "anchor_chunk_index": anchor_chunk_index,
                    "phase_bin": anchor_chunk_index // 14,
                    "matched_noise_seed": group_noise_seed(group_id),
                    "language_condition": None,
                    "conditions": conditions,
                    "aggregation_contract": {
                        "anchor_unit": "paired_condition_loss",
                        "primary_statistical_unit": "source_motion",
                        "anchors_per_source_motion": 10,
                        "anchor_losses_reduced_by": "arithmetic_mean_before_statistics",
                    },
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_manifest = args.output_dir / "FROZEN_PROMPT_GATE_CASES.jsonl"
    case_manifest.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    case_manifest_sha256 = file_sha256(case_manifest)

    split_group_counts = Counter(str(case["split"]) for case in cases)
    target_keys = {
        (case["split"], case["target"]["task"], case["target"]["source_motion_id"])
        for case in cases
    }
    seeds = [int(case["matched_noise_seed"]) for case in cases]
    checks = {
        "bound_to_exact_immutable_source_manifest": (
            source_manifest_sha256 == EXPECTED_SOURCE_MANIFEST_SHA256
        ),
        "exact_39_heldout_source_motions": len(target_keys) == 39,
        "heldout_split_trajectory_counts_exact": (
            dict(split_trajectories) == EXPECTED_SPLIT_TRAJECTORIES
        ),
        "heldout_split_task_counts_exact": all(
            dict(split_task_trajectories[split]) == expected
            for split, expected in EXPECTED_SPLIT_TASK_TRAJECTORIES.items()
        ),
        "exact_10_fixed_phase_anchors_per_motion": len(cases) == 390
        and all(
            {
                int(candidate["anchor_chunk_index"])
                for candidate in cases
                if candidate["target"]["source_motion_id"] == motion_id
                and candidate["target"]["task"] == task
                and candidate["split"] == split
            }
            == set(ANCHOR_CHUNK_INDICES)
            for split, task, motion_id in target_keys
        ),
        "anchor_indices_exact": {
            int(case["anchor_chunk_index"]) for case in cases
        }
        == set(ANCHOR_CHUNK_INDICES),
        "validation_and_test_group_counts_exact": dict(split_group_counts)
        == {"validation": 200, "test": 190},
        "all_five_conditions_present": all(
            set(case["conditions"].keys()) == set(CONDITION_NAMES) for case in cases
        ),
        "counterfactual_references_resolve_within_split": reference_integrity,
        "causal_two_frame_targets_and_action_ranges_exact": target_integrity,
        "language_disabled": language_disabled,
        "all_390_matched_noise_seeds_unique": len(seeds) == len(set(seeds)) == 390,
        "source_motion_is_primary_statistical_unit": all(
            case["aggregation_contract"]["primary_statistical_unit"] == "source_motion"
            for case in cases
        ),
        "no_training_target_enters_frozen_gate": all(case["split"] != "train" for case in cases),
    }
    result = {
        "protocol": PROTOCOL,
        "passed": all(checks.values()),
        "source_manifest_sha256": source_manifest_sha256,
        "case_manifest_path": str(case_manifest),
        "case_manifest_sha256": case_manifest_sha256,
        "target_source_motion_count": len(target_keys),
        "case_group_count": len(cases),
        "condition_instance_count": len(cases) * len(CONDITION_NAMES),
        "anchor_chunk_indices": list(ANCHOR_CHUNK_INDICES),
        "split_group_counts": dict(split_group_counts),
        "checks": checks,
        "claim_boundary": (
            "Passing freezes source-disjoint prompt interventions, matched-noise seeds and "
            "trajectory-level aggregation for evaluating an official Zero-WAM checkpoint. "
            "It does not execute a model or constitute prompt-following evidence."
        ),
    }
    result_path = args.output_dir / "FROZEN_PROMPT_GATE_CASES_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
