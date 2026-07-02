"""Build a strict train-only source repair preflight for Phase 01.

This is a preflight/data gate, not training. It refuses to prepare final-attempt
training CSVs if the available train-only corrective sources improve one metric
by trading away lift/hold behavior.
"""

from __future__ import annotations

import argparse
import csv
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


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _delta(source: dict[str, Any], key: str) -> float:
    return float((source.get("paired_metric_deltas") or {}).get(key, 0.0))


def _strict_source_failures(source: dict[str, Any], gate: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    lift_delta = _delta(source, "lift_delta_fb_minus_no")
    hold_delta = _delta(source, "hold_delta_fb_minus_no")
    slip_delta = _delta(source, "slip_delta_fb_minus_no")
    accel_delta = _delta(source, "accel_delta_fb_minus_no")
    contact_loss_delta = _delta(source, "contact_loss_delta_fb_minus_no")
    success_delta = _delta(source, "success_delta_fb_minus_no")

    if success_delta < 0.0:
        failures.append("success_regression")
    if lift_delta < -float(gate["max_lift_regression_m"]):
        failures.append(f"lift_regression:{lift_delta}")
    if hold_delta < -float(gate["max_hold_regression_s"]):
        failures.append(f"hold_regression:{hold_delta}")
    if contact_loss_delta > float(gate["max_contact_loss_regression_frames"]):
        failures.append(f"contact_loss_regression:{contact_loss_delta}")
    if slip_delta > float(gate["max_slip_regression_m"]):
        failures.append(f"slip_regression:{slip_delta}")
    if accel_delta > float(gate["max_accel_regression_m_s2"]):
        failures.append(f"accel_regression:{accel_delta}")

    improvements = []
    if -slip_delta >= float(gate["min_slip_improvement_m"]):
        improvements.append("slip_improvement")
    if -accel_delta >= float(gate["min_accel_improvement_m_s2"]):
        improvements.append("accel_improvement")
    if not improvements:
        failures.append("no_strict_slip_or_accel_improvement")
    return failures


def build(root: Path, config_path: Path, run_tag: str) -> dict[str, Any]:
    config = _load_json(config_path)
    source_manifest_path = root / config["source_manifest"]
    source_manifest = _load_json(source_manifest_path)
    output_dir = root / config["output_dir"]
    report_path = root / config["report"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if source_manifest.get("status") != "pass":
        failures.append(f"source_manifest_status:{source_manifest.get('status')}")

    held_out = set(config["held_out_cells_reserved_for_evaluation"])
    admitted = []
    rejected = []
    gate = dict(config["strict_gate"])
    for source in source_manifest.get("admitted_sources", []):
        cell = str(source.get("cell", ""))
        source_failures = []
        if cell in held_out or cell.startswith("heldout_"):
            source_failures.append(f"held_out_cell_forbidden:{cell}")
        source_failures.extend(_strict_source_failures(source, gate))
        record = {
            "cell": cell,
            "feedback_run_tag": source.get("feedback_run_tag"),
            "feedback_npz": source.get("feedback_npz"),
            "feedback_metrics": source.get("feedback_metrics"),
            "feedback_active_frames": source.get("feedback_active_frames"),
            "feedback_trigger_count": source.get("feedback_trigger_count"),
            "paired_metric_deltas": source.get("paired_metric_deltas"),
        }
        if source_failures:
            record["status"] = "rejected"
            record["failures"] = source_failures
            rejected.append(record)
        else:
            record["status"] = "admitted"
            admitted.append(record)

    if len(admitted) < int(gate["min_admitted_sources"]):
        failures.append(f"strict_admitted_sources_below_min:{len(admitted)}")

    train_csv = output_dir / "train.csv"
    validation_csv = output_dir / "validation.csv"
    source_train_csv = root / str(source_manifest.get("train_csv", ""))
    source_validation_csv = root / str(source_manifest.get("validation_csv", ""))
    train_fieldnames: list[str] = []
    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    if admitted and source_train_csv.exists() and source_validation_csv.exists():
        admitted_cells = {item["cell"] for item in admitted}
        validation_count = min(int(gate["validation_admitted_sources"]), max(0, len(admitted_cells) - 1))
        validation_cells = {item["cell"] for item in admitted[-validation_count:]} if validation_count else set()
        fieldnames_train, rows_train = _read_csv(source_train_csv)
        fieldnames_validation, rows_validation = _read_csv(source_validation_csv)
        train_fieldnames = fieldnames_train or fieldnames_validation
        for row in [*rows_train, *rows_validation]:
            cell = row.get("cell", "")
            if cell not in admitted_cells:
                continue
            updated = dict(row)
            updated["split"] = "validation" if cell in validation_cells else "train"
            if updated["split"] == "validation":
                validation_rows.append(updated)
            else:
                train_rows.append(updated)

    if admitted:
        if not train_rows:
            failures.append("no_train_rows_after_strict_source_filter")
        if not validation_rows:
            failures.append("no_validation_rows_after_strict_source_filter")

    _write_csv(train_csv, train_fieldnames, train_rows)
    _write_csv(validation_csv, train_fieldnames, validation_rows)

    status = "pass" if not failures else "blocked"
    manifest = {
        "classification": "phase01_strict_train_only_source_repair_preflight_v1",
        "phase": "phase01",
        "status": status,
        "run_tag": run_tag,
        "not_training_result": True,
        "no_curiosity_success_claim": True,
        "config": _rel(config_path, root),
        "source_manifest": _rel(source_manifest_path, root),
        "train_csv": _rel(train_csv, root),
        "validation_csv": _rel(validation_csv, root),
        "train_row_count": len(train_rows),
        "validation_row_count": len(validation_rows),
        "admitted_count": len(admitted),
        "rejected_count": len(rejected),
        "admitted_sources": admitted,
        "rejected_sources": rejected,
        "held_out_excluded_from_training": sorted(held_out),
        "strict_gate": gate,
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "forbidden_claims": config["forbidden_claims"],
        "failures": failures,
        "final_one_hour_attempt_allowed_from_this_preflight": status == "pass",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Phase 01 Strict Source Repair Preflight",
        "",
        f"- status: `{status}`",
        f"- run tag: `{run_tag}`",
        f"- admitted strict sources: `{len(admitted)}`",
        f"- rejected strict sources: `{len(rejected)}`",
        f"- train rows: `{len(train_rows)}`",
        f"- validation rows: `{len(validation_rows)}`",
        f"- manifest: `{_rel(manifest_path, root)}`",
        "",
        "This is a train-only data/objective preflight. It is not training and not a curiosity success claim.",
        "",
        "## Gate",
        "",
        f"- max lift regression m: `{gate['max_lift_regression_m']}`",
        f"- max hold regression s: `{gate['max_hold_regression_s']}`",
        f"- min slip improvement m: `{gate['min_slip_improvement_m']}`",
        f"- min accel improvement m/s^2: `{gate['min_accel_improvement_m_s2']}`",
        "",
    ]
    if admitted:
        lines.extend(["## Admitted", ""])
        for item in admitted:
            lines.append(f"- `{item['cell']}`")
        lines.append("")
    if rejected:
        lines.extend(["## Rejected", ""])
        for item in rejected:
            reason = ", ".join(item.get("failures", []))
            lines.append(f"- `{item['cell']}`: {reason}")
        lines.append("")
    if failures:
        lines.extend(["## Blockers", ""])
        for failure in failures:
            lines.append(f"- `{failure}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    manifest = build(args.root, args.config, args.run_tag)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    raise SystemExit(0 if manifest["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
