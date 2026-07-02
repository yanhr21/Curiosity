#!/usr/bin/env python3
"""Strict acceptance check for a Gate 00F reference bundle summary.

This reads JSON summaries only. It does not run simulators, renderers, model
loads, dataset conversion, training, or evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "univtac_status": "pass_official_schema_probe",
    "tacauchy_status": "pass_official_schema_probe",
    "isaaclab_tacsl_status": "pass_official_isaaclab_tacsl_demo_exited_zero",
    "gate00f_status": "pass_official_semantic_reference_sanity",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def status_from_file(path: Path, key: str = "status") -> str:
    if not path.exists():
        return "missing"
    data = load_json(path)
    return str(data.get(key, "missing"))


def validate(bundle_summary: Path) -> dict[str, Any]:
    bundle = load_json(bundle_summary)
    failures: list[str] = []

    observed = {key: str(bundle.get(key, "missing")) for key in EXPECTED}
    for key, expected in EXPECTED.items():
        if observed[key] != expected:
            failures.append(f"{key}: expected {expected}, observed {observed[key]}")

    child_checks = {
        "univtac_summary": ("univtac_status", "status"),
        "tacauchy_summary": ("tacauchy_status", "status"),
        "isaaclab_tacsl_summary": ("isaaclab_tacsl_status", "status"),
        "gate_review_summary": ("gate00f_status", "gate_00f_official_semantic_validation_status"),
    }
    child_observed: dict[str, str] = {}
    for path_key, (status_key, json_key) in child_checks.items():
        raw_path = bundle.get(path_key)
        if not raw_path:
            failures.append(f"{path_key}: missing path")
            child_observed[path_key] = "missing"
            continue
        child_path = Path(str(raw_path))
        child_status = status_from_file(child_path, json_key)
        child_observed[path_key] = child_status
        if child_status != EXPECTED[status_key]:
            failures.append(
                f"{path_key}: child {json_key} expected {EXPECTED[status_key]}, observed {child_status}"
            )

    allow_blocker_sanity = str(bundle.get("allow_blocker_sanity", "0"))
    if allow_blocker_sanity not in {"0", "false", "False", ""}:
        failures.append(f"allow_blocker_sanity must be disabled for acceptance, observed {allow_blocker_sanity}")

    gate_summary_path = bundle.get("gate_review_summary")
    gate_failed_checks: list[str] = []
    gate_hard_blockers: list[str] = []
    gate_curiosity_allowed: Any = None
    if gate_summary_path and Path(str(gate_summary_path)).exists():
        gate_summary = load_json(Path(str(gate_summary_path)))
        gate_failed_checks = list(gate_summary.get("failed_checks", []))
        gate_hard_blockers = list(gate_summary.get("hard_blockers", []))
        gate_curiosity_allowed = gate_summary.get("curiosity_training_allowed")
        if gate_failed_checks:
            failures.append(f"gate_review_summary.failed_checks is nonempty: {gate_failed_checks}")
        if gate_hard_blockers:
            failures.append(f"gate_review_summary.hard_blockers is nonempty: {gate_hard_blockers}")
        if gate_curiosity_allowed is not False:
            failures.append(
                "gate review should not enable curiosity directly; "
                f"observed curiosity_training_allowed={gate_curiosity_allowed}"
            )

    status = "pass_gate00f_bundle_acceptance" if not failures else "fail_gate00f_bundle_acceptance"
    return {
        "classification": "gate00f_bundle_acceptance_not_training_not_curiosity_success",
        "bundle_summary": str(bundle_summary),
        "status": status,
        "expected": EXPECTED,
        "observed": observed,
        "child_observed": child_observed,
        "allow_blocker_sanity": allow_blocker_sanity,
        "gate_failed_checks": gate_failed_checks,
        "gate_hard_blockers": gate_hard_blockers,
        "gate_curiosity_training_allowed": gate_curiosity_allowed,
        "failures": failures,
        "not_training": True,
        "not_curiosity_success": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = validate(args.bundle_summary)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["status"].startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
