"""Compare a Phase 01 curiosity residual held-out eval against baselines.

This script reads completed metrics JSON files and writes an auditable
comparison report. It does not train, render, or load a model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _metric_row(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    rows = payload.get("rows") or []
    if not rows:
        return {}
    row = dict(rows[0])
    row["metrics_json"] = str(path)
    return row


def _cell(row: dict[str, Any]) -> str:
    return str(row.get("pose_seed") or row.get("cell") or "")


def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _baseline_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    patterns = [
        root / "experiments/outputs/phase01/core/baselines",
        root / "experiments/outputs/phase01/core/resid/base_eval",
    ]
    for directory in patterns:
        for path in sorted(directory.glob("*_metrics.json")):
            row = _metric_row(path)
            cell = _cell(row)
            if not cell:
                continue
            row["source_metrics_json"] = _rel(path, root)
            rows_by_cell.setdefault(cell, []).append(row)
    return rows_by_cell


def compare(root: Path, candidate_summary_path: Path, output_json: Path, output_report: Path) -> dict[str, Any]:
    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")

    candidate_summary = _load_json(candidate_summary_path)
    candidate_rows = list(candidate_summary.get("rows") or [])
    baseline_by_cell = _baseline_rows(root)
    per_cell = []
    safety_regression_count = 0
    useful_improvement_count = 0

    for candidate in candidate_rows:
        cell = _cell(candidate)
        baselines = baseline_by_cell.get(cell, [])
        if not baselines:
            failures.append(f"missing_baselines_for_cell:{cell}")
            continue
        best_lift = max(_num(row, "lift_height_m") for row in baselines)
        best_hold = max(_num(row, "hold_duration_s") for row in baselines)
        best_slip = min(_num(row, "max_slip_m") for row in baselines)
        best_contact_loss = min(_num(row, "contact_loss_frames") for row in baselines)
        best_accel = min(_num(row, "max_object_accel_m_s2") for row in baselines)

        regressions = []
        improvements = []
        if str(candidate.get("status")) != "success":
            regressions.append("candidate_not_success")
        if _num(candidate, "max_slip_m") > best_slip + 1.0e-6:
            regressions.append("slip_worse_than_best_baseline")
        elif _num(candidate, "max_slip_m") < best_slip - 1.0e-6:
            improvements.append("slip_better_than_best_baseline")
        if _num(candidate, "contact_loss_frames") > best_contact_loss:
            regressions.append("contact_loss_worse_than_best_baseline")
        elif _num(candidate, "contact_loss_frames") < best_contact_loss:
            improvements.append("contact_loss_better_than_best_baseline")
        if _num(candidate, "max_object_accel_m_s2") > best_accel + 1.0e-6:
            regressions.append("accel_worse_than_best_baseline")
        elif _num(candidate, "max_object_accel_m_s2") < best_accel - 1.0e-6:
            improvements.append("accel_better_than_best_baseline")
        if _num(candidate, "lift_height_m") + 0.002 < best_lift:
            regressions.append("lift_below_best_baseline_tolerance")
        elif _num(candidate, "lift_height_m") > best_lift + 0.002:
            improvements.append("lift_better_than_best_baseline")
        if _num(candidate, "hold_duration_s") + 0.1 < best_hold:
            regressions.append("hold_below_best_baseline_tolerance")
        elif _num(candidate, "hold_duration_s") > best_hold + 0.1:
            improvements.append("hold_better_than_best_baseline")

        safety_regression_count += int(bool(regressions))
        useful_improvement_count += len(improvements)
        per_cell.append(
            {
                "cell": cell,
                "candidate": candidate,
                "baseline_count": len(baselines),
                "best_baseline": {
                    "lift_height_m": best_lift,
                    "hold_duration_s": best_hold,
                    "max_slip_m": best_slip,
                    "contact_loss_frames": best_contact_loss,
                    "max_object_accel_m_s2": best_accel,
                },
                "regressions": regressions,
                "improvements": improvements,
                "cell_pass": not regressions and bool(improvements),
            }
        )

    all_cells_present = len(candidate_rows) == 4 and len(per_cell) == 4
    positive = all_cells_present and safety_regression_count == 0 and useful_improvement_count > 0
    result_classification = "positive_candidate" if positive else "negative_or_incomplete_candidate"
    if not all_cells_present:
        result_classification = "incomplete_candidate"

    payload = {
        "classification": "phase01_curiosity_weighted_residual_strongest_baseline_comparison_v1",
        "status": "pass" if not failures else "fail",
        "candidate_summary": _rel(candidate_summary_path, root),
        "result_classification": result_classification,
        "positive_curiosity_result": positive,
        "candidate_cell_count": len(candidate_rows),
        "safety_regression_cell_count": safety_regression_count,
        "useful_improvement_count": useful_improvement_count,
        "per_cell": per_cell,
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "failures": failures,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Phase 01 Curiosity Residual Strongest-Baseline Comparison",
        "",
        f"- status: `{payload['status']}`",
        f"- result classification: `{result_classification}`",
        f"- positive curiosity result: `{positive}`",
        f"- candidate cells: `{len(candidate_rows)}`",
        f"- safety-regression cells: `{safety_regression_count}`",
        f"- useful improvements: `{useful_improvement_count}`",
        "",
        "This comparison is strict. A checkpoint or successful rollout is not a curiosity success unless this comparison is positive without safety regression.",
        "",
        "## Cells",
        "",
    ]
    for item in per_cell:
        lines.append(
            f"- `{item['cell']}` pass `{item['cell_pass']}` "
            f"regressions `{', '.join(item['regressions']) or 'none'}` "
            f"improvements `{', '.join(item['improvements']) or 'none'}`"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in failures)
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.root, args.candidate_summary, args.output_json, args.output_report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
