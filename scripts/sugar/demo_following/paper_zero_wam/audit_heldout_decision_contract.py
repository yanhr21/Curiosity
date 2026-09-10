#!/usr/bin/env python3
"""Fail-closed self-test for the production motion-disjoint score reducer."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .config import PROJECT_ROOT, PaperZeroWAMConfig
from .evaluate_heldout import CONDITIONS, INTERVENTIONS, LATENT_ANCHORS, aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/demo_following/paper_zero_wam_v1/schedule/"
        "HELDOUT_DECISION_CONTRACT_AUDIT.json",
    )
    return parser.parse_args()


def clean_records() -> list[dict[str, object]]:
    noise_seed = PaperZeroWAMConfig().noise_seed
    sources = (
        [("validation", "CarryBox", index) for index in range(10)]
        + [("validation", "KickBox", index) for index in range(10)]
        + [("test", "CarryBox", index) for index in range(10)]
        + [("test", "KickBox", index) for index in range(9)]
    )
    records: list[dict[str, object]] = []
    for source_ordinal, (split, task, source_motion_id) in enumerate(sources):
        for anchor_ordinal, anchor in enumerate(LATENT_ANCHORS):
            losses = {
                "matched": {"video_loss": 1.0, "action_loss": 1.0},
                **{
                    condition: {
                        "video_loss": 2.0 + 0.01 * index,
                        "action_loss": 2.1 + 0.01 * index,
                    }
                    for index, condition in enumerate(INTERVENTIONS)
                },
            }
            if set(losses) != set(CONDITIONS):
                raise AssertionError("synthetic condition set differs from production")
            records.append(
                {
                    "group_index": source_ordinal * 10 + anchor_ordinal,
                    "split": split,
                    "task": task,
                    "source_motion_id": source_motion_id,
                    "latent_anchor": anchor,
                    "matched_noise_seed": noise_seed + 10_000_019 + source_ordinal * 10 + anchor_ordinal,
                    "losses": losses,
                    "connectivity_vs_matched": {
                        condition: {
                            "predicted_future_mse_vs_matched": 0.01,
                            "predicted_action_velocity_mse_vs_matched": 0.01,
                        }
                        for condition in INTERVENTIONS
                    },
                }
            )
    return records


def main() -> None:
    args = parse_args()
    records = clean_records()
    clean = aggregate(records)

    zero_connectivity_records = copy.deepcopy(records)
    zero_connectivity_records[0]["connectivity_vs_matched"]["reversed"][
        "predicted_future_mse_vs_matched"
    ] = 0.0
    zero_connectivity = aggregate(zero_connectivity_records)

    flat_action_records = copy.deepcopy(records)
    for record in flat_action_records:
        if record["split"] == "validation":
            record["losses"]["reversed"]["action_loss"] = record["losses"][
                "matched"
            ]["action_loss"]
    flat_action = aggregate(flat_action_records)

    duplicate_index_records = copy.deepcopy(records)
    duplicate_index_records[1]["group_index"] = 0
    try:
        aggregate(duplicate_index_records)
        duplicate_index_rejected = False
    except ValueError:
        duplicate_index_rejected = True

    duplicate_anchor_records = copy.deepcopy(records)
    duplicate_anchor_records[1]["latent_anchor"] = LATENT_ANCHORS[0]
    try:
        aggregate(duplicate_anchor_records)
        duplicate_anchor_rejected = False
    except ValueError:
        duplicate_anchor_rejected = True

    wrong_noise_seed_records = copy.deepcopy(records)
    wrong_noise_seed_records[0]["matched_noise_seed"] += 1
    try:
        aggregate(wrong_noise_seed_records)
        wrong_noise_seed_rejected = False
    except ValueError:
        wrong_noise_seed_rejected = True

    passed = (
        clean["passed"] is True
        and zero_connectivity["passed"] is False
        and zero_connectivity["checks"]
        ["every_prompt_swap_changes_predicted_future_and_action"]
        is False
        and flat_action["passed"] is False
        and flat_action["checks"]["all_directional_motion_counts_pass"] is False
        and flat_action["checks"]["all_16_holm_tests_pass"] is False
        and duplicate_index_rejected
        and duplicate_anchor_rejected
        and wrong_noise_seed_rejected
    )
    result = {
        "protocol": "paper_zero_wam_heldout_decision_contract_audit_v1",
        "passed": passed,
        "model_constructed": False,
        "optimizer_updates": 0,
        "matched_noise_groups": len(records),
        "score_instances": len(records) * len(CONDITIONS),
        "clean_decision_passed": clean["passed"],
        "clean_checks": clean["checks"],
        "mutation_rejections": {
            "zero_predicted_future_connectivity": not zero_connectivity["passed"],
            "flat_validation_reversed_action_margin": not flat_action["passed"],
            "duplicate_group_index": duplicate_index_rejected,
            "duplicate_anchor_within_motion": duplicate_anchor_rejected,
            "wrong_matched_noise_seed": wrong_noise_seed_rejected,
        },
        "hash_checks": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("held-out decision contract audit failed")


if __name__ == "__main__":
    main()
