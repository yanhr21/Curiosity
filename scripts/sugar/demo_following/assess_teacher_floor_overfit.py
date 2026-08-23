#!/usr/bin/env python3
"""Apply the predeclared behavior gate to the teacher-floor overfit pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / "experiments/demo_following/teacher_floor_overfit_v1/seed161581"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--behavior-result",
        type=Path,
        default=DEFAULT_RUN / "behavior_adherence_audit_v1/RESULT.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN / "TEACHER_FLOOR_GATE.json",
    )
    return parser.parse_args()


def assess(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("protocol") != "same_teacher_predictor_independent_behavior_audit_v1":
        raise ValueError("unexpected behavior-audit protocol")
    evidence = payload["evidence_contract"]
    if evidence["uses_predictor_output"] or evidence["uses_demo_reward"]:
        raise ValueError("teacher-floor gate requires predictor-independent evidence")
    if payload.get("required_next_trace_fields"):
        raise ValueError("teacher-floor gate requires complete contact-role traces")

    summary = payload["actual_arm_summary"]
    correct = summary["correct"]
    unrelated = summary["unrelated"]
    directions = payload["predeclared_semantic_directions"]
    correct_bilateral = float(correct["bilateral_contact_fraction"]["mean"])
    unrelated_bilateral = float(unrelated["bilateral_contact_fraction"]["mean"])
    correct_lifted = float(correct["lifted_fraction"]["mean"])
    correct_transport = float(correct["lifted_transport_fraction"]["mean"])
    correct_foot = float(correct["any_foot_box_contact_fraction"]["mean"])
    unrelated_foot = float(unrelated["any_foot_box_contact_fraction"]["mean"])

    checks = {
        "correct_preserves_bilateral_hand_contact": correct_bilateral >= 0.50,
        "correct_preserves_lifted_behavior": correct_lifted >= 0.50,
        "correct_preserves_lifted_transport": correct_transport >= 0.80,
        "all_four_kick_directions_observed": all(
            bool(record["direction_observed"])
            for record in directions.values()
        ),
        "unrelated_has_measurable_foot_box_contact": unrelated_foot >= 0.01,
        "unrelated_foot_contact_exceeds_correct": (
            unrelated_foot - correct_foot >= 0.005
        ),
        "unrelated_reduces_bilateral_hand_role": (
            correct_bilateral - unrelated_bilateral >= 0.05
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": "sugar_teacher_floor_overfit_behavior_gate_v1",
        "passed": passed,
        "evidence_source": "predictor-independent frozen behavior trace",
        "checks": checks,
        "measurements": {
            "correct_bilateral_contact_fraction": correct_bilateral,
            "unrelated_bilateral_contact_fraction": unrelated_bilateral,
            "correct_lifted_fraction": correct_lifted,
            "correct_lifted_transport_fraction": correct_transport,
            "correct_any_foot_box_contact_fraction": correct_foot,
            "unrelated_any_foot_box_contact_fraction": unrelated_foot,
            "semantic_directions_observed": int(
                payload["semantic_directions_observed"]
            ),
            "semantic_directions_total": int(
                payload["semantic_directions_total"]
            ),
        },
        "automatic_next_branch": (
            "repeat_teacher_floor_schedule_across_independent_training_seeds"
            if passed
            else "redesign_internal_reward_contact_event_semantics"
        ),
        "claim_boundary": (
            "Task failure alone never passes this gate; the unrelated arm must "
            "show the declared Kick contact and object-motion structure."
        ),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.behavior_result.read_text(encoding="utf-8"))
    result = assess(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
