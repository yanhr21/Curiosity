#!/usr/bin/env python3
"""Build Phase08 residual data from strict baseline-failure repairs.

The old non-regression gate rejects some useful repairs because a failed
baseline can throw the object upward before dropping it, creating a misleading
peak-lift advantage. This builder keeps the old strict gate when the baseline
already succeeds, but uses an absolute repair gate when the baseline fails.
It remains a preflight only: no training, no held-out data, and no success
claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _first_scalar(data: np.ndarray, index: int) -> float:
    arr = np.asarray(data)
    item = np.asarray(arr[0] if arr.shape[0] == 1 else arr[index])
    return float(item.reshape(-1)[0]) if item.size else 0.0


def _first_int(data: np.ndarray, index: int) -> int:
    return int(round(_first_scalar(data, index)))


def _object_z(data: np.ndarray, index: int) -> float:
    arr = np.asarray(data[index]).reshape(-1)
    return float(arr[2]) if arr.size >= 3 else 0.0


def _feature_value(data: Any, column: str, index: int) -> float | int:
    aliases = {
        "newton.contact.rigid_contact_count": "newton.panda.rigid_contact_count",
    }
    if column == "newton.object.body_q.z":
        return _object_z(data["newton.panda.object_body_q"], index)
    key = aliases.get(column, column)
    if key not in data:
        raise KeyError(f"missing feature column {column} mapped to {key}")
    if "phase_index" in column or "contact_count" in column:
        return _first_int(data[key], index)
    return _first_scalar(data[key], index)


def _metric_row(root: Path, run_tag: str) -> dict[str, Any]:
    metrics_path = root / "experiments" / "outputs" / f"{run_tag}_metrics.json"
    if metrics_path.is_file():
        payload = _load_json(metrics_path)
        rows = payload.get("rows") or []
        if rows:
            return rows[0]
    summary = _load_json(root / "experiments" / "outputs" / f"{run_tag}_summary.json")
    world = summary.get("task_metrics", {}).get("per_world", [{}])[0]
    return {
        "status": "success" if world.get("success") is True else "fail",
        "object_not_dropped": bool(world.get("success")),
        "hold_duration_s": world.get("longest_hold_s"),
        "lift_height_m": world.get("max_lift"),
        "max_slip_m": world.get("max_xy_drift", 0.0),
        "contact_loss_frames": 0,
        "drop_height_loss_m": 0.0 if world.get("success") is True else None,
        "max_object_accel_m_s2": 0.0,
    }


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return default if value is None else float(value)


def _metric_score(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        1.0 if row.get("status") == "success" else 0.0,
        1.0 if row.get("object_not_dropped") is True else 0.0,
        _f(row, "hold_duration_s"),
        _f(row, "lift_height_m"),
        -_f(row, "drop_height_loss_m", 999.0),
        -_f(row, "max_slip_m", 999.0),
        -_f(row, "contact_loss_frames", 999.0),
        -_f(row, "max_object_accel_m_s2", 999.0),
    )


def _baseline_succeeded(row: dict[str, Any]) -> bool:
    return row.get("status") == "success" and row.get("object_not_dropped") is True


def _strict_success_gate(
    intervention: dict[str, Any],
    baseline: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    hold_gain = _f(intervention, "hold_duration_s") - _f(baseline, "hold_duration_s")
    lift_gain = _f(intervention, "lift_height_m") - _f(baseline, "lift_height_m")
    if intervention.get("status") != "success":
        reasons.append("intervention_not_success")
    if intervention.get("object_not_dropped") is not True:
        reasons.append("intervention_object_dropped")
    if _metric_score(intervention) <= _metric_score(baseline):
        reasons.append("ordered_score_not_better")
    if hold_gain < float(gate["min_hold_gain_s"]):
        reasons.append("hold_gain_below_min")
    if lift_gain < float(gate["min_lift_gain_m"]):
        reasons.append("lift_gain_below_min")
    if _f(intervention, "max_slip_m") > _f(baseline, "max_slip_m") + float(gate["slip_tolerance_m"]):
        reasons.append("slip_regression")
    if _f(intervention, "max_object_accel_m_s2") > _f(baseline, "max_object_accel_m_s2") + float(
        gate["accel_tolerance_m_s2"]
    ):
        reasons.append("accel_regression")
    return not reasons, reasons


def _failure_repair_gate(
    intervention: dict[str, Any],
    baseline: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    hold_gain = _f(intervention, "hold_duration_s") - _f(baseline, "hold_duration_s")
    if intervention.get("status") != "success":
        reasons.append("intervention_not_success")
    if intervention.get("object_not_dropped") is not True:
        reasons.append("intervention_object_dropped")
    if hold_gain < float(gate["min_hold_gain_s"]):
        reasons.append("hold_gain_below_min")
    if _f(intervention, "hold_duration_s") < float(gate["min_absolute_hold_s"]):
        reasons.append("absolute_hold_below_min")
    if _f(intervention, "lift_height_m") < float(gate["min_absolute_lift_m"]):
        reasons.append("absolute_lift_below_min")
    if _f(intervention, "drop_height_loss_m") > float(gate["max_absolute_drop_height_loss_m"]):
        reasons.append("absolute_drop_loss_above_max")
    if _f(intervention, "max_slip_m") > float(gate["max_absolute_slip_m"]):
        reasons.append("absolute_slip_above_max")
    if _f(intervention, "max_object_accel_m_s2") > float(gate["max_absolute_accel_m_s2"]):
        reasons.append("absolute_accel_above_max")
    if _f(intervention, "contact_loss_frames") > float(gate["max_contact_loss_frames"]):
        reasons.append("contact_loss_above_max")
    if _f(intervention, "drop_height_loss_m") > _f(baseline, "drop_height_loss_m", 999.0) + float(
        gate["drop_tolerance_m"]
    ):
        reasons.append("drop_regression")
    if _f(intervention, "max_slip_m") > _f(baseline, "max_slip_m", 999.0) + float(gate["slip_tolerance_m"]):
        reasons.append("slip_regression")
    if _f(intervention, "max_object_accel_m_s2") > _f(baseline, "max_object_accel_m_s2", 999.0) + float(
        gate["accel_tolerance_m_s2"]
    ):
        reasons.append("accel_regression")
    return not reasons, reasons


def _accept_pair(
    intervention: dict[str, Any],
    baseline: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, str, list[str]]:
    if _baseline_succeeded(baseline):
        accepted, reasons = _strict_success_gate(intervention, baseline, config["strict_success_gate"])
        return accepted, "strict_success_non_regression", reasons
    accepted, reasons = _failure_repair_gate(intervention, baseline, config["failure_repair_gate"])
    return accepted, "failed_baseline_repair", reasons


def build(config_path: Path, root: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    fresh_sanity = _load_json(fresh_sanity_json)
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train_failure_repair_records.csv"
    validation_csv = output_dir / "validation_failure_repair_records.csv"
    manifest_path = output_dir / "manifest.json"

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if fresh_sanity.get("status") != "pass":
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")

    feature_columns = list(config["feature_columns"])
    target_columns = list(config["target_columns"])
    columns = [
        "run_tag",
        "source_name",
        "split",
        "cell",
        "held_out_generalization_cell",
        "timestep_index",
        *feature_columns,
        *target_columns,
        "candidate.advantage.baseline_run_tag",
        "candidate.advantage.intervention_run_tag",
        "candidate.advantage.hold_gain_s",
        "candidate.advantage.lift_gain_m",
        "candidate.failure_repair.acceptance_mode",
        "candidate.failure_repair.accepted_cell",
    ]
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    comparisons: list[dict[str, Any]] = []
    accepted_cells = 0
    accepted_active_frames = 0
    accepted_by_split = {"train": 0, "validation": 0}

    for pair in config["paired_rollouts"]:
        split = pair["split"]
        cell = pair["cell"]
        baseline_tag = pair["baseline_run_tag"]
        intervention_tag = pair["intervention_run_tag"]
        if split not in rows_by_split:
            failures.append(f"invalid_split:{cell}:{split}")
            continue
        if pair.get("held_out_generalization_cell") is True:
            failures.append(f"held_out_pair_forbidden:{cell}")
            continue

        baseline_row = _metric_row(root, baseline_tag)
        intervention_row = _metric_row(root, intervention_tag)
        hold_gain = _f(intervention_row, "hold_duration_s") - _f(baseline_row, "hold_duration_s")
        lift_gain = _f(intervention_row, "lift_height_m") - _f(baseline_row, "lift_height_m")
        accepted, mode, reject_reasons = _accept_pair(intervention_row, baseline_row, config)
        comparisons.append(
            {
                "cell": cell,
                "split": split,
                "baseline_run_tag": baseline_tag,
                "intervention_run_tag": intervention_tag,
                "accepted": accepted,
                "acceptance_mode": mode,
                "reject_reasons": reject_reasons,
                "hold_gain_s": hold_gain,
                "lift_gain_m": lift_gain,
                "baseline_metric_row": baseline_row,
                "intervention_metric_row": intervention_row,
            }
        )
        if not accepted:
            continue

        accepted_cells += 1
        accepted_by_split[split] += 1
        summary = _load_json(root / "experiments" / "outputs" / f"{intervention_tag}_summary.json")
        npz_path = Path(summary["npz"])
        if not npz_path.is_absolute():
            npz_path = root / npz_path
        with np.load(npz_path, allow_pickle=False) as data:
            timesteps = int(np.asarray(data["newton.panda.sim_time"]).shape[0])
            accepted_active_frames += int(np.count_nonzero(np.asarray(data[target_columns[0]])))
            for idx in range(timesteps):
                row: dict[str, Any] = {
                    "run_tag": intervention_tag,
                    "source_name": pair.get("name", intervention_tag),
                    "split": split,
                    "cell": cell,
                    "held_out_generalization_cell": "false",
                    "timestep_index": idx,
                    target_columns[0]: _first_int(data[target_columns[0]], idx),
                    target_columns[1]: _first_scalar(data[target_columns[1]], idx),
                    target_columns[2]: _first_scalar(data[target_columns[2]], idx),
                    target_columns[3]: _first_scalar(data[target_columns[3]], idx),
                    "candidate.advantage.baseline_run_tag": baseline_tag,
                    "candidate.advantage.intervention_run_tag": intervention_tag,
                    "candidate.advantage.hold_gain_s": hold_gain,
                    "candidate.advantage.lift_gain_m": lift_gain,
                    "candidate.failure_repair.acceptance_mode": mode,
                    "candidate.failure_repair.accepted_cell": "true",
                }
                for column in feature_columns:
                    row[column] = _feature_value(data, column, idx)
                rows_by_split[split].append(row)

    for path, rows in ((train_csv, rows_by_split["train"]), (validation_csv, rows_by_split["validation"])):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    if accepted_cells < int(config["minimums"]["min_accepted_cells"]):
        failures.append(f"accepted_cells_below_min:{accepted_cells}")
    if accepted_by_split["train"] < int(config["minimums"]["min_train_accepted_cells"]):
        failures.append(f"accepted_train_cells_below_min:{accepted_by_split['train']}")
    if accepted_by_split["validation"] < int(config["minimums"]["min_validation_accepted_cells"]):
        failures.append(f"accepted_validation_cells_below_min:{accepted_by_split['validation']}")
    if accepted_active_frames <= 0:
        failures.append("accepted_active_frames_zero")
    if not rows_by_split["train"]:
        failures.append("no_train_rows_after_failure_repair_gate")
    if not rows_by_split["validation"]:
        failures.append("no_validation_rows_after_failure_repair_gate")

    manifest = {
        "classification": "phase08_guarded_overlay_failure_repair_preflight_v1",
        "status": "pass" if not failures else "fail",
        "purpose": "admit guarded-overlay labels only when they pass strict success or failed-baseline repair gates",
        "config": _rel(config_path, root),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "train_csv": _rel(train_csv, root),
        "validation_csv": _rel(validation_csv, root),
        "train_record_count": len(rows_by_split["train"]),
        "validation_record_count": len(rows_by_split["validation"]),
        "accepted_cell_count": accepted_cells,
        "accepted_by_split": accepted_by_split,
        "accepted_active_frames": accepted_active_frames,
        "comparisons": comparisons,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "training_started": False,
        "policy_updated": False,
        "not_training": True,
        "not_success_claim": True,
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--fresh-sanity-json", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.config, args.root, args.fresh_sanity_json)
    print(json.dumps({"status": manifest["status"], "manifest": manifest}, indent=2, sort_keys=True))
    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
