"""Build Phase07 held-out comparison report.

This aggregates existing held-out evaluation metrics and visual evidence. It
does not train, run inference, render videos, or claim success. Missing methods
remain explicit so negative/incomplete evidence cannot be hidden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HELD_OUT_CELLS = [
    "empty_high_misleading",
    "full_low_hidden",
    "three_quarter_low_misleading",
]

METHODS = {
    "no_adaptation": {
        "role": "baseline",
        "run_tags": {
            "empty_high_misleading": "phase07_eval_empty_high_misleading_no_adaptation_rerun_20260627",
            "full_low_hidden": "phase07_eval_full_low_hidden_no_adaptation_20260627",
            "three_quarter_low_misleading": "phase07_eval_three_quarter_low_misleading_no_adaptation_20260627",
        },
    },
    "scripted_feedback": {
        "role": "baseline",
        "run_tags": {cell: f"phase07_eval_{cell}_scripted_feedback_20260627" for cell in HELD_OUT_CELLS},
    },
    "residual_baseline": {
        "role": "baseline",
        "run_tags": {cell: f"phase07_eval_{cell}_residual_baseline_20260627" for cell in HELD_OUT_CELLS},
    },
    "curiosity_weighted": {
        "role": "candidate_curiosity",
        "run_tags": {cell: f"phase07_eval_{cell}_curiosity_weighted_20260627" for cell in HELD_OUT_CELLS},
    },
    "random_intrinsic": {
        "role": "ablation",
        "run_tags": {
            "empty_high_misleading": "phase07_eval_empty_high_misleading_random_intrinsic_rerun_20260627",
            "full_low_hidden": "phase07_eval_full_low_hidden_random_intrinsic_20260627",
            "three_quarter_low_misleading": "phase07_eval_three_quarter_low_misleading_random_intrinsic_20260627",
        },
    },
    "object_only": {
        "role": "ablation",
        "run_tags": {cell: f"phase07_eval_{cell}_object_only_20260627" for cell in HELD_OUT_CELLS},
    },
    "contact_only": {
        "role": "ablation_pending",
        "run_tags": {cell: f"phase07_eval_{cell}_contact_only_20260627" for cell in HELD_OUT_CELLS},
    },
    "shuffled_contact": {
        "role": "ablation_pending",
        "run_tags": {cell: f"phase07_eval_{cell}_shuffled_contact_20260627" for cell in HELD_OUT_CELLS},
    },
    "delayed_contact": {
        "role": "ablation_pending",
        "run_tags": {cell: f"phase07_eval_{cell}_delayed_contact_20260627" for cell in HELD_OUT_CELLS},
    },
    "no_learning_progress": {
        "role": "ablation_pending",
        "run_tags": {cell: f"phase07_eval_{cell}_no_learning_progress_20260627" for cell in HELD_OUT_CELLS},
    },
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _metric_row(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics or metrics.get("status") != "pass":
        return None
    rows = metrics.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0]
    return row if isinstance(row, dict) else None


def _score(row: dict[str, Any] | None) -> tuple[float, float, float, float, float, float]:
    if row is None:
        return (-1.0, -1.0, -1.0, -1e9, -1e9, -1e9)
    def value(key: str, default: float = 0.0) -> float:
        raw = row.get(key)
        return float(default if raw is None else raw)

    success = 1.0 if row.get("status") == "success" and row.get("object_not_dropped") is True else 0.0
    return (
        success,
        value("hold_duration_s"),
        value("lift_height_m"),
        -value("max_slip_m", 1e9),
        -value("contact_loss_frames", 1e9),
        -value("max_object_accel_m_s2", 1e9),
    )


def _summarize_run(root: Path, run_tag: str, method: str, cell: str, role: str) -> dict[str, Any]:
    metrics_path = root / "experiments" / "outputs" / f"{run_tag}_metrics.json"
    summary_path = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
    manual_path = root / "experiments" / "outputs" / f"{run_tag}_manual_visual_inspection.json"
    visual_path = root / "experiments" / "outputs" / f"{run_tag}_visual_validation.json"
    accel_path = root / "experiments" / "outputs" / f"{run_tag}_accel_peak_analysis.json"
    metrics = _load_json(metrics_path)
    summary = _load_json(summary_path)
    manual = _load_json(manual_path)
    visual = _load_json(visual_path)
    accel = _load_json(accel_path)
    row = _metric_row(metrics)
    video_export = summary.get("video_export") if isinstance(summary, dict) else None
    video_path = video_export.get("path") if isinstance(video_export, dict) else None
    failures: list[str] = []
    if row is None:
        failures.append("missing_or_failed_metrics")
    if not summary or summary.get("status") != "pass":
        failures.append("missing_or_failed_summary")
    if not isinstance(video_export, dict) or video_export.get("status") != "pass":
        failures.append("missing_or_failed_video_export")
    if not manual or not str(manual.get("status", "")).startswith("pass"):
        failures.append("missing_or_failed_manual_visual_inspection")
    if not visual or visual.get("status") != "pass":
        failures.append("missing_or_failed_visual_validation")
    if accel is None:
        failures.append("missing_accel_peak_analysis")
    return {
        "method": method,
        "role": role,
        "cell": cell,
        "run_tag": run_tag,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "metrics": _rel(root, metrics_path),
        "summary": _rel(root, summary_path),
        "manual_visual_inspection": _rel(root, manual_path),
        "visual_validation": _rel(root, visual_path),
        "accel_peak_analysis": _rel(root, accel_path),
        "rollout_video": video_path,
        "metric_row": row,
        "score_tuple": list(_score(row)),
    }


def _compare_cell(cell_summary: dict[str, Any]) -> dict[str, Any]:
    baselines = {
        method: item
        for method, item in cell_summary.items()
        if item["role"] == "baseline" and item["metric_row"] is not None
    }
    curiosity = cell_summary.get("curiosity_weighted")
    if not baselines:
        return {"status": "fail", "reason": "missing_baselines"}
    strongest_name, strongest = max(baselines.items(), key=lambda item: _score(item[1]["metric_row"]))
    curiosity_row = curiosity.get("metric_row") if curiosity else None
    strongest_row = strongest["metric_row"]
    safety_regressions: list[str] = []
    if curiosity_row is None:
        return {
            "status": "fail",
            "strongest_baseline": strongest_name,
            "reason": "missing_curiosity_metrics",
        }
    for key in ["max_slip_m", "contact_loss_frames", "drop_height_loss_m", "max_object_accel_m_s2"]:
        if float(curiosity_row.get(key) or 0.0) > float(strongest_row.get(key) or 0.0):
            safety_regressions.append(f"{key}_regression")
    curiosity_beats = _score(curiosity_row) > _score(strongest_row)
    safety_ok = not safety_regressions
    return {
        "status": "pass" if curiosity_beats and safety_ok else "fail",
        "strongest_baseline": strongest_name,
        "strongest_baseline_score_tuple": list(_score(strongest_row)),
        "curiosity_score_tuple": list(_score(curiosity_row)),
        "curiosity_beats_strongest_baseline": curiosity_beats,
        "safety_ok": safety_ok,
        "safety_regressions": safety_regressions,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase07 Held-Out Comparison Report V1",
        "",
        "Date: 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This report aggregates existing evidence only. It does not train, render, run inference, or claim success.",
        "",
        "## Summary",
        "",
        f"- final curiosity comparison passed: `{payload['curiosity_beats_all_strongest_baselines_without_safety_regression']}`",
        f"- missing or failed method/cell entries: `{payload['missing_or_failed_entry_count']}`",
        "",
        "## Per Cell",
        "",
    ]
    for cell, comparison in payload["cell_comparisons"].items():
        lines.append(
            f"- `{cell}`: status=`{comparison['status']}`, "
            f"strongest_baseline=`{comparison.get('strongest_baseline')}`, "
            f"curiosity_beats=`{comparison.get('curiosity_beats_strongest_baseline')}`, "
            f"safety_ok=`{comparison.get('safety_ok')}`"
        )
    lines.extend(["", "## Missing / Failed Entries", ""])
    for item in payload["missing_or_failed_entries"]:
        lines.append(f"- `{item['method']}` / `{item['cell']}`: {item['failures']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/phase07_heldout_comparison_report_v1_20260627.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/reports/2026-06-27_phase07_heldout_comparison_report_v1.md"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    cells: dict[str, dict[str, Any]] = {}
    missing_or_failed: list[dict[str, Any]] = []
    for cell in HELD_OUT_CELLS:
        cells[cell] = {}
        for method, spec in METHODS.items():
            run_tag = spec["run_tags"][cell]
            item = _summarize_run(root, run_tag, method, cell, spec["role"])
            cells[cell][method] = item
            if item["status"] != "pass":
                missing_or_failed.append({"method": method, "cell": cell, "run_tag": run_tag, "failures": item["failures"]})
    comparisons = {cell: _compare_cell(cell_summary) for cell, cell_summary in cells.items()}
    comparison_pass = all(item["status"] == "pass" for item in comparisons.values())
    complete_entries = not missing_or_failed
    payload = {
        "classification": "phase07_heldout_comparison_report_v1",
        "status": "pass" if comparison_pass and complete_entries else "open_not_satisfied",
        "held_out_cells": HELD_OUT_CELLS,
        "method_roles": {method: spec["role"] for method, spec in METHODS.items()},
        "cells": cells,
        "cell_comparisons": comparisons,
        "curiosity_beats_all_strongest_baselines_without_safety_regression": comparison_pass,
        "missing_or_failed_entries": missing_or_failed,
        "missing_or_failed_entry_count": len(missing_or_failed),
        "not_training": True,
        "not_rendering": True,
        "not_inference": True,
        "not_success_claim": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
