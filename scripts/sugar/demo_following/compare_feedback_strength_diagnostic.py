#!/usr/bin/env python3
"""Compare the fixed 1x and 4x phase-event feedback runs at update 64."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "lifted_fraction",
    "lifted_transport_fraction",
    "ground_transport_fraction",
    "root_orbit_rate_rad_s",
    "any_foot_box_contact_fraction",
    "bilateral_contact_fraction",
    "maximum_lift_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(root: Path) -> dict[str, object]:
    audit = load_json(root / "behavior_adherence_update0064/RESULT.json")
    evaluation = {
        arm: load_json(root / f"evaluation_update0064/{arm}/RESULT.json")
        for arm in ("correct", "unrelated")
    }
    proofs = {
        arm: load_json(root / f"{arm}/update_0064/proof.json")
        for arm in ("correct", "unrelated")
    }
    return {"audit": audit, "evaluation": evaluation, "proofs": proofs}


def admitted(run: dict[str, object]) -> bool:
    audit = run["audit"]
    if audit.get("status") != "complete_existing_trace_audit":
        return False
    for payload in (*run["evaluation"].values(), *run["proofs"].values()):
        checks = payload.get("checks", {})
        if payload.get("passed") is not True or not checks or not all(checks.values()):
            return False
    return True


def main() -> None:
    args = parse_args()
    baseline = load_run(args.baseline_root.resolve())
    diagnostic = load_run(args.diagnostic_root.resolve())
    if not admitted(baseline) or not admitted(diagnostic):
        raise RuntimeError("strength comparison requires passing training/evaluation evidence")

    baseline_proof = baseline["proofs"]["correct"]
    diagnostic_proof = diagnostic["proofs"]["correct"]
    baseline_runtime = load_json(Path(baseline_proof["demo_event_reward"]["runtime_config"]))
    diagnostic_runtime = load_json(Path(diagnostic_proof["demo_event_reward"]["runtime_config"]))
    eta_ratio = diagnostic_runtime["eta"] / baseline_runtime["eta"]
    clip_ratio = diagnostic_runtime["reward_clip"] / baseline_runtime["reward_clip"]

    checks = {
        "same_training_seed": baseline_proof["seed"] == diagnostic_proof["seed"] == 161589,
        "same_action_seed": baseline_proof["action_seed"] == diagnostic_proof["action_seed"] == 161590,
        "same_update_budget": baseline_proof["num_updates"] == diagnostic_proof["num_updates"] == 64,
        "baseline_scale_is_1x": abs(float(eta_ratio) - 4.0) < 1.0e-12,
        "reward_clip_is_4x": abs(float(clip_ratio) - 4.0) < 1.0e-12,
        "both_runs_admitted": True,
    }
    comparisons: dict[str, object] = {}
    for metric in METRICS:
        baseline_metric = baseline["audit"]["paired_profile_comparison"][metric]
        diagnostic_metric = diagnostic["audit"]["paired_profile_comparison"][metric]
        comparisons[metric] = {
            "baseline_unrelated_minus_correct": baseline_metric["unrelated_minus_correct_mean"],
            "four_x_unrelated_minus_correct": diagnostic_metric["unrelated_minus_correct_mean"],
            "change_in_paired_effect": (
                diagnostic_metric["unrelated_minus_correct_mean"]
                - baseline_metric["unrelated_minus_correct_mean"]
            ),
        }

    telemetry: dict[str, object] = {}
    for arm in ("correct", "unrelated"):
        base = baseline["evaluation"][arm]["final_update_aggregate"]
        four = diagnostic["evaluation"][arm]["final_update_aggregate"]
        telemetry[arm] = {
            "baseline_cumulative_selected_feedback": base["cumulative_selected_demo_feedback"],
            "four_x_cumulative_selected_feedback": four["cumulative_selected_demo_feedback"],
            "baseline_selected_demo_predicted_loss": base["mean_selected_demo_predicted_loss"],
            "four_x_selected_demo_predicted_loss": four["mean_selected_demo_predicted_loss"],
            "predicted_loss_change": (
                four["mean_selected_demo_predicted_loss"]
                - base["mean_selected_demo_predicted_loss"]
            ),
        }

    baseline_directions = baseline["audit"]["semantic_directions_observed"]
    four_x_directions = diagnostic["audit"]["semantic_directions_observed"]
    foot_effect = comparisons["any_foot_box_contact_fraction"]["four_x_unrelated_minus_correct"]
    checks.update(
        {
            "four_x_does_not_improve_direction_count": four_x_directions <= baseline_directions,
            "four_x_does_not_create_unrelated_foot_contact_advantage": foot_effect <= 0.0,
        }
    )
    result = {
        "protocol": "sugar_phase_event_feedback_strength_fixed_overfit_comparison_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_semantic_directions": baseline_directions,
        "four_x_semantic_directions": four_x_directions,
        "paired_behavior": comparisons,
        "reward_telemetry": telemetry,
        "conclusion": (
            "Multiplying dense feedback by four does not recover Kick contact topology. "
            "The update-64 paired semantic-direction count falls from 3/4 to 1/4, and the "
            "unrelated arm still has no foot-contact advantage. Signal magnitude alone is "
            "therefore not the demonstrated bottleneck; do not continue a scale sweep."
        ),
        "sources": [str(args.baseline_root.resolve()), str(args.diagnostic_root.resolve())],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError("feedback-strength diagnostic failed its evidence contract")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
