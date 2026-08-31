#!/usr/bin/env python3
"""Evaluate official Zero-WAM prompt dependence at the source-motion level.

The model runner is intentionally not implemented here because the official
Zero-WAM release schema is not public yet.  This evaluator accepts only scores
from that future strict-loaded release and applies the predeclared matched-noise,
motion-level and Holm-corrected gates without fitting or threshold selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any


CASE_PROTOCOL = "sugar_zero_wam_frozen_prompt_gate_cases_v1"
SCORE_PROTOCOL = "official_zero_wam_frozen_sugar_prompt_scores_v1"
OUTPUT_PROTOCOL = "official_zero_wam_frozen_sugar_prompt_gate_v1"
EXPECTED_CASE_MANIFEST_SHA256 = (
    "035e554a94ecd506e4e4d287f32cf90d21f78f8a5211277f34daedf4516e37f5"
)
CONDITION_NAMES = (
    "matched",
    "wrong_task",
    "reversed",
    "same_task_alternate",
    "masked_prompt",
)
ALTERNATIVE_CONDITIONS = CONDITION_NAMES[1:]
SPLITS = ("validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--score-jsonl", type=Path)
    parser.add_argument("--expected-model-commit")
    parser.add_argument("--expected-checkpoint-sha256")
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


def exact_one_sided_sign_p(wins: int, losses: int) -> float:
    """P[X >= wins] for the predeclared matched-is-better direction."""
    non_ties = wins + losses
    if non_ties == 0:
        return 1.0
    return float(sum(comb(non_ties, value) for value in range(wins, non_ties + 1))) / float(
        2**non_ties
    )


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, float(count - rank) * float(value))
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def run_self_test() -> None:
    assert exact_one_sided_sign_p(20, 0) == 1.0 / 2**20
    assert exact_one_sided_sign_p(0, 20) == 1.0
    assert exact_one_sided_sign_p(0, 0) == 1.0
    adjusted = holm_adjust({"a": 0.001, "b": 0.01, "c": 0.04})
    assert math.isclose(adjusted["a"], 0.003)
    assert math.isclose(adjusted["b"], 0.02)
    assert math.isclose(adjusted["c"], 0.04)


def require_runtime_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "case_manifest",
            "score_jsonl",
            "expected_model_commit",
            "expected_checkpoint_sha256",
            "output_dir",
        )
        if getattr(args, name) in (None, "")
    ]
    if missing:
        raise ValueError(f"missing required runtime arguments: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        if args.case_manifest is None:
            return
    require_runtime_args(args)
    assert args.case_manifest is not None
    assert args.score_jsonl is not None
    assert args.output_dir is not None
    assert args.expected_model_commit is not None
    assert args.expected_checkpoint_sha256 is not None

    case_manifest_sha256 = file_sha256(args.case_manifest)
    cases = [json.loads(line) for line in args.case_manifest.read_text().splitlines()]
    scores = [json.loads(line) for line in args.score_jsonl.read_text().splitlines()]
    case_by_group = {str(case["group_id"]): case for case in cases}
    score_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_score_keys = False
    for score in scores:
        key = (str(score.get("group_id")), str(score.get("condition")))
        duplicate_score_keys = duplicate_score_keys or key in score_by_key
        score_by_key[key] = score

    expected_keys = {
        (group_id, condition)
        for group_id in case_by_group
        for condition in CONDITION_NAMES
    }
    actual_keys = set(score_by_key)
    complete_score_matrix = (
        len(cases) == len(case_by_group) == 390
        and len(scores) == len(score_by_key) == 1_950
        and not duplicate_score_keys
        and actual_keys == expected_keys
    )

    full_commit = bool(re.fullmatch(r"[0-9a-f]{40}", args.expected_model_commit))
    checkpoint_hash_valid = valid_sha256(args.expected_checkpoint_sha256)
    official_provenance = True
    score_schema_valid = True
    losses_finite_nonnegative = True
    matched_noise_valid = True
    language_disabled = True
    before_action_decoder = True
    for group_id, case in case_by_group.items():
        fingerprints: set[str] = set()
        for condition in CONDITION_NAMES:
            score = score_by_key.get((group_id, condition), {})
            official_provenance = official_provenance and (
                score.get("protocol") == SCORE_PROTOCOL
                and score.get("provenance") == "official_zero_wam_release"
                and score.get("model_commit") == args.expected_model_commit
                and score.get("checkpoint_sha256") == args.expected_checkpoint_sha256
                and score.get("case_manifest_sha256") == case_manifest_sha256
            )
            score_schema_valid = score_schema_valid and (
                score.get("group_id") == group_id
                and score.get("condition") == condition
                and valid_sha256(score.get("predicted_robot_future_sha256"))
            )
            try:
                loss = float(score["next_video_flow_loss"])
            except (KeyError, TypeError, ValueError):
                loss = float("nan")
            losses_finite_nonnegative = (
                losses_finite_nonnegative and math.isfinite(loss) and loss >= 0.0
            )
            matched_noise_valid = matched_noise_valid and (
                score.get("noise_seed") == case["matched_noise_seed"]
                and valid_sha256(score.get("matched_noise_fingerprint"))
            )
            if valid_sha256(score.get("matched_noise_fingerprint")):
                fingerprints.add(str(score["matched_noise_fingerprint"]))
            language_disabled = language_disabled and score.get("language_condition") is None
            before_action_decoder = before_action_decoder and (
                score.get("prediction_stage") == "official_video_future_before_action_decoder"
            )
        matched_noise_valid = matched_noise_valid and len(fingerprints) == 1

    case_schema_valid = (
        case_manifest_sha256 == EXPECTED_CASE_MANIFEST_SHA256
        and all(case.get("protocol") == CASE_PROTOCOL for case in cases)
        and all(set(case.get("conditions", {}).keys()) == set(CONDITION_NAMES) for case in cases)
        and all(case.get("split") in SPLITS for case in cases)
    )

    raw_p_values: dict[str, float] = {}
    comparison_metrics: dict[str, dict[str, Any]] = {}
    future_changed_everywhere = True
    trajectory_margin_cache: dict[str, list[float]] = {}
    if complete_score_matrix:
        for split in SPLITS:
            for alternative in ALTERNATIVE_CONDITIONS:
                comparison_key = f"{alternative}_{split}"
                anchor_margins: dict[tuple[str, int], list[float]] = defaultdict(list)
                future_changed_count = 0
                group_count = 0
                for group_id, case in case_by_group.items():
                    if case["split"] != split:
                        continue
                    matched_score = score_by_key[(group_id, "matched")]
                    alternative_score = score_by_key[(group_id, alternative)]
                    matched_loss = float(matched_score["next_video_flow_loss"])
                    alternative_loss = float(alternative_score["next_video_flow_loss"])
                    target_key = (
                        str(case["target"]["task"]),
                        int(case["target"]["source_motion_id"]),
                    )
                    anchor_margins[target_key].append(alternative_loss - matched_loss)
                    future_changed_count += int(
                        matched_score["predicted_robot_future_sha256"]
                        != alternative_score["predicted_robot_future_sha256"]
                    )
                    group_count += 1

                ten_anchors_each = bool(anchor_margins) and all(
                    len(values) == 10 for values in anchor_margins.values()
                )
                trajectory_margins = {
                    key: sum(values) / len(values) for key, values in anchor_margins.items()
                }
                margin_values = list(trajectory_margins.values())
                trajectory_margin_cache[comparison_key] = margin_values
                wins = sum(value > 0.0 for value in margin_values)
                losses = sum(value < 0.0 for value in margin_values)
                ties = len(margin_values) - wins - losses
                mean_margin = sum(margin_values) / max(1, len(margin_values))
                win_rate = wins / max(1, len(margin_values))
                raw_p = exact_one_sided_sign_p(wins, losses)
                raw_p_values[comparison_key] = raw_p
                task_metrics: dict[str, dict[str, float | int]] = {}
                for task in ("CarryBox", "KickBox"):
                    values = [
                        margin
                        for (target_task, _), margin in trajectory_margins.items()
                        if target_task == task
                    ]
                    task_wins = sum(value > 0.0 for value in values)
                    task_metrics[task] = {
                        "trajectory_count": len(values),
                        "mean_margin": sum(values) / max(1, len(values)),
                        "win_rate": task_wins / max(1, len(values)),
                    }
                future_fraction = future_changed_count / max(1, group_count)
                future_changed_everywhere = future_changed_everywhere and future_fraction == 1.0
                comparison_metrics[comparison_key] = {
                    "alternative_condition": alternative,
                    "split": split,
                    "trajectory_count": len(margin_values),
                    "anchors_per_trajectory_exactly_10": ten_anchors_each,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "mean_alternative_minus_matched_loss": mean_margin,
                    "trajectory_win_rate": win_rate,
                    "raw_one_sided_exact_sign_p": raw_p,
                    "predicted_future_changed_fraction": future_fraction,
                    "task_metrics": task_metrics,
                }

    adjusted_p_values = holm_adjust(raw_p_values)
    comparison_passes: dict[str, bool] = {}
    for comparison_key, metrics in comparison_metrics.items():
        adjusted = adjusted_p_values[comparison_key]
        metrics["holm_adjusted_p"] = adjusted
        both_tasks_positive = all(
            task_metrics["mean_margin"] > 0.0 and task_metrics["win_rate"] > 0.5
            for task_metrics in metrics["task_metrics"].values()
        )
        metrics["both_tasks_positive_with_win_rate_above_half"] = both_tasks_positive
        comparison_passes[comparison_key] = bool(
            metrics["anchors_per_trajectory_exactly_10"]
            and metrics["mean_alternative_minus_matched_loss"] > 0.0
            and metrics["trajectory_win_rate"] > 0.5
            and both_tasks_positive
            and adjusted < 0.05
        )

    checks: dict[str, bool] = {
        "exact_frozen_case_manifest": case_schema_valid,
        "complete_1950_condition_score_matrix": complete_score_matrix,
        "official_release_model_and_checkpoint_provenance": (
            full_commit and checkpoint_hash_valid and official_provenance
        ),
        "official_next_video_flow_loss": score_schema_valid and losses_finite_nonnegative,
        "matched_noise": matched_noise_valid,
        "language_disabled": language_disabled,
        "source_motion_level_statistics_after_ten_anchor_reduction": (
            len(trajectory_margin_cache) == 8
            and all(len(values) in (19, 20) for values in trajectory_margin_cache.values())
        ),
        "matched_beats_wrong_task_validation": comparison_passes.get(
            "wrong_task_validation", False
        ),
        "matched_beats_wrong_task_test": comparison_passes.get("wrong_task_test", False),
        "matched_beats_reversed_validation": comparison_passes.get(
            "reversed_validation", False
        ),
        "matched_beats_reversed_test": comparison_passes.get("reversed_test", False),
        "matched_beats_same_task_alternate_validation": comparison_passes.get(
            "same_task_alternate_validation", False
        ),
        "matched_beats_same_task_alternate_test": comparison_passes.get(
            "same_task_alternate_test", False
        ),
        "matched_beats_masked_prompt_validation": comparison_passes.get(
            "masked_prompt_validation", False
        ),
        "matched_beats_masked_prompt_test": comparison_passes.get(
            "masked_prompt_test", False
        ),
        "holm_corrected_sign_tests_pass": len(adjusted_p_values) == 8
        and all(value < 0.05 for value in adjusted_p_values.values()),
        "predicted_future_changes_before_action_decoder": (
            before_action_decoder and future_changed_everywhere
        ),
    }
    passed = all(checks.values())
    result = {
        "protocol": OUTPUT_PROTOCOL,
        "passed": passed,
        "provenance": "official_zero_wam_release",
        "model_commit": args.expected_model_commit,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "case_manifest_sha256": case_manifest_sha256,
        "score_jsonl_sha256": file_sha256(args.score_jsonl),
        "multiple_testing_family": (
            "8 directional source-motion-level comparisons: 4 prompt interventions x 2 splits"
        ),
        "sign_test_alternative": "alternative_prompt_loss_greater_than_matched_prompt_loss",
        "comparison_metrics": comparison_metrics,
        "checks": checks,
        "automatic_next_branch": (
            "audit_official_sugar_29dof_adapter"
            if passed
            else "close_sugar_adaptation_for_this_official_checkpoint"
        ),
        "claim_boundary": (
            "Passing proves frozen teacher-forced prompt dependence for task, temporal order, "
            "selected-motion identity and prompt presence on held-out SUGAR motions. It does not "
            "prove 29-DoF action compatibility, closed-loop control or arbitrary-video following."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "FROZEN_PROMPT_GATE_RESULT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
