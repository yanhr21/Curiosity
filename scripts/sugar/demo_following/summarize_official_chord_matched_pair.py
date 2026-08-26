#!/usr/bin/env python3
"""Summarize the one-variable official-CHORD OFF/ON frozen evaluation pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-root", type=Path, required=True)
    parser.add_argument("--on-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(root: Path, prefixes: list[int], endpoint: str) -> dict[str, float | int]:
    rows = [
        _read(root / f"evaluation/prefix{prefix}/{endpoint}_kick/RESULT.json")[
            "aggregate"
        ]
        for prefix in prefixes
    ]
    return {
        "safe_kick_success_count": sum(row["safe_kick_success_count"] for row in rows),
        "physical_fall_count": sum(row["physical_fall_count"] for row in rows),
        "mean_contact_coupled_planar_path_m": sum(
            row["mean_contact_coupled_planar_path_m"] for row in rows
        )
        / len(rows),
        "mean_planar_object_net_displacement_m": sum(
            row["mean_planar_object_net_displacement_m"] for row in rows
        )
        / len(rows),
        "mean_any_foot_box_contact_fraction": sum(
            row["mean_any_foot_box_contact_fraction"] for row in rows
        )
        / len(rows),
    }


def main() -> None:
    args = _args()
    off_result = _read(args.off_root / "RESULT.json")
    on_result = _read(args.on_root / "RESULT.json")
    off_audit = _read(args.off_root / "train/prefix_audit.json")
    on_audit = _read(args.on_root / "train/prefix_audit.json")
    prefixes = [int(value) for value in off_result["evaluation_prefix_schedule"]]
    if prefixes != [int(value) for value in on_result["evaluation_prefix_schedule"]]:
        raise RuntimeError("CHORD OFF/ON evaluation prefix schedules differ")
    common_fields = (
        "training_seed",
        "evaluation_seed",
        "training_prefix_schedule",
        "evaluation_prefix_schedule",
        "policy_topology",
        "profiles_per_endpoint",
    )
    matched = all(off_result[field] == on_result[field] for field in common_fields)
    off_chord = off_audit.get("official_chord_runtime_reward", {"enabled": False})
    on_chord = on_audit.get("official_chord_runtime_reward", {"enabled": False})
    off_pre = _aggregate(args.off_root, prefixes, "pre_update")
    on_pre = _aggregate(args.on_root, prefixes, "pre_update")
    pre_identical = off_pre == on_pre
    off = _aggregate(args.off_root, prefixes, "learned")
    on = _aggregate(args.on_root, prefixes, "learned")
    delta = {
        key: on[key] - off[key]
        for key in (
            "safe_kick_success_count",
            "physical_fall_count",
            "mean_contact_coupled_planar_path_m",
            "mean_planar_object_net_displacement_m",
            "mean_any_foot_box_contact_fraction",
        )
    }
    runtime_valid = bool(
        on_chord.get("enabled")
        and on_chord.get("reward_calls", 0) > 0
        and on_chord.get("maximum_abs_reward", 0.0) > 0.0
        and not off_chord.get("enabled", False)
    )
    positive = bool(
        delta["safe_kick_success_count"] > 0
        or (
            delta["safe_kick_success_count"] == 0
            and delta["physical_fall_count"] < 0
        )
    )
    payload = {
        "protocol": "sugar_official_chord_causal_matched_pair_v1",
        "structurally_valid": bool(matched and pre_identical and runtime_valid),
        "one_variable_difference": "official CHORD online contact-wrench reward",
        "matched_fields": {field: off_result[field] for field in common_fields},
        "pre_update_aggregate_identical": pre_identical,
        "chord_off_runtime": off_chord,
        "chord_on_runtime": on_chord,
        "frozen_evaluation": {"off": off, "on": on, "on_minus_off": delta},
        "visualizations": [
            str(
                args.output.parent
                / f"videos_off_vs_on/chord_off_vs_on_prefix{prefix}.mp4"
            )
            for prefix in prefixes
        ],
        "positive_physical_result": positive,
        "conclusion": (
            "official_chord_improves_matched_frozen_kick_safety"
            if positive
            else "official_chord_does_not_improve_matched_frozen_kick_safety"
        ),
        "claim_boundary": (
            "One matched training seed and one disjoint frozen evaluation seed. "
            "A reward signal or action difference alone is not a physical benefit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["structurally_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
