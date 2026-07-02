"""Build residual training data only from cells where intervention beats baseline.

This preflight is intentionally strict. It compares paired no-adaptation and
intervention rollouts on train/validation cells, then only admits intervention
NPZ labels for cells with a positive ordered advantage. If no cell beats the
baseline, the preflight fails instead of producing dense over-intervention
labels.
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


def _metric_score(row: dict[str, Any]) -> tuple:
    status_ok = 1 if row.get("status") == "success" else 0
    object_ok = 1 if row.get("object_not_dropped") is True else 0
    return (
        status_ok,
        object_ok,
        float(row.get("hold_duration_s") or 0.0),
        float(row.get("lift_height_m") or 0.0),
        -float(row.get("max_slip_m") or 0.0),
        -float(row.get("contact_loss_frames") or 0.0),
        -float(row.get("max_object_accel_m_s2") or 0.0),
    )


def _beats(intervention: dict[str, Any], baseline: dict[str, Any], min_hold_gain: float, min_lift_gain: float) -> bool:
    if intervention.get("status") != "success" or intervention.get("object_not_dropped") is not True:
        return False
    if _metric_score(intervention) <= _metric_score(baseline):
        return False
    hold_gain = float(intervention.get("hold_duration_s") or 0.0) - float(baseline.get("hold_duration_s") or 0.0)
    lift_gain = float(intervention.get("lift_height_m") or 0.0) - float(baseline.get("lift_height_m") or 0.0)
    accel_regression = float(intervention.get("max_object_accel_m_s2") or 0.0) > float(
        baseline.get("max_object_accel_m_s2") or 0.0
    ) + 1e-6
    return hold_gain >= min_hold_gain and lift_gain >= min_lift_gain and not accel_regression


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
        "max_object_accel_m_s2": 0.0,
    }


def build(config_path: Path, root: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    fresh_sanity = _load_json(fresh_sanity_json)
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train_advantage_gated_records.csv"
    validation_csv = output_dir / "validation_advantage_gated_records.csv"
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
        "candidate.advantage.accepted_cell",
    ]
    rows_by_split = {"train": [], "validation": []}
    comparisons = []
    accepted_cells = 0
    accepted_active_frames = 0

    for pair in config["paired_rollouts"]:
        split = pair["split"]
        cell = pair["cell"]
        baseline_tag = pair["baseline_run_tag"]
        intervention_tag = pair["intervention_run_tag"]
        if split not in rows_by_split:
            failures.append(f"invalid_split:{cell}:{split}")
            continue
        baseline_row = _metric_row(root, baseline_tag)
        intervention_row = _metric_row(root, intervention_tag)
        hold_gain = float(intervention_row.get("hold_duration_s") or 0.0) - float(
            baseline_row.get("hold_duration_s") or 0.0
        )
        lift_gain = float(intervention_row.get("lift_height_m") or 0.0) - float(
            baseline_row.get("lift_height_m") or 0.0
        )
        accepted = _beats(
            intervention_row,
            baseline_row,
            float(config["advantage_gate"]["min_hold_gain_s"]),
            float(config["advantage_gate"]["min_lift_gain_m"]),
        )
        comparisons.append(
            {
                "cell": cell,
                "split": split,
                "baseline_run_tag": baseline_tag,
                "intervention_run_tag": intervention_tag,
                "accepted": accepted,
                "hold_gain_s": hold_gain,
                "lift_gain_m": lift_gain,
                "baseline_metric_row": baseline_row,
                "intervention_metric_row": intervention_row,
            }
        )
        if not accepted:
            continue
        accepted_cells += 1
        summary = _load_json(root / "experiments" / "outputs" / f"{intervention_tag}_summary.json")
        npz_path = Path(summary["npz"])
        if not npz_path.is_absolute():
            npz_path = root / npz_path
        with np.load(npz_path, allow_pickle=False) as data:
            timesteps = int(np.asarray(data["newton.panda.sim_time"]).shape[0])
            accepted_active_frames += int(np.count_nonzero(np.asarray(data[target_columns[0]])))
            for idx in range(timesteps):
                row = {
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
                    "candidate.advantage.accepted_cell": "true",
                }
                for column in feature_columns:
                    row[column] = _feature_value(data, column, idx)
                rows_by_split[split].append(row)

    for path, rows in ((train_csv, rows_by_split["train"]), (validation_csv, rows_by_split["validation"])):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    if accepted_cells < int(config["advantage_gate"]["min_accepted_cells"]):
        failures.append(f"accepted_cells_below_min:{accepted_cells}")
    if accepted_active_frames <= 0:
        failures.append("accepted_active_frames_zero")
    if not rows_by_split["train"]:
        failures.append("no_train_rows_after_advantage_gate")
    if not rows_by_split["validation"] and config["advantage_gate"].get("require_validation_rows", True):
        failures.append("no_validation_rows_after_advantage_gate")

    manifest = {
        "classification": "advantage_gated_residual_preflight_v1",
        "status": "pass" if not failures else "fail",
        "purpose": "admit residual labels only when intervention beats no-adaptation on paired evidence",
        "config": _rel(config_path, root),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "train_csv": _rel(train_csv, root),
        "validation_csv": _rel(validation_csv, root),
        "train_record_count": len(rows_by_split["train"]),
        "validation_record_count": len(rows_by_split["validation"]),
        "accepted_cell_count": accepted_cells,
        "accepted_active_frames": accepted_active_frames,
        "comparisons": comparisons,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "training_started": False,
        "policy_updated": False,
        "not_success_claim": True,
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
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
