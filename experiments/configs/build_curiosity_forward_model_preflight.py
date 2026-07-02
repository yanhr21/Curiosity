"""Build preflight records for Newton-native curiosity forward-model training.

This prepares transition targets from validated residual-label source runs. It
does not train a model, create a checkpoint, or promote fields into a T-Rex
schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


FORBIDDEN_EXACT_FIELDS = {"action", "action_abs", "observation.state", "observation.tactile_f6"}
FORBIDDEN_PREFIXES = ("observation.images.", "observation.tactile_deform.")


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


def _as_float(value: str) -> float:
    return 0.0 if value == "" else float(value)


def _as_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _forbidden_fields(fields: list[str]) -> list[str]:
    failures: list[str] = []
    for field in fields:
        if field in FORBIDDEN_EXACT_FIELDS or any(field.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            failures.append(f"forbidden_field:{field}")
    return failures


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _column_summary(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for column in columns:
        values = [float(row[column]) for row in rows]
        if not values:
            summary[column] = {"count": 0, "nonzero": 0, "min": None, "max": None, "mean": None}
            continue
        summary[column] = {
            "count": len(values),
            "nonzero": sum(1 for value in values if value != 0.0),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "distinct_rounded_6": len({round(value, 6) for value in values}),
        }
    return summary


def _build_transition_rows(
    source_rows: list[dict[str, str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    thresholds = config["risk_thresholds"]
    contact_loss_count = float(thresholds["contact_loss_count"])
    slip_drop_z = float(thresholds["slip_drop_z_m"])
    rows_by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        rows_by_run[row["run_tag"]].append(row)

    transition_rows: list[dict[str, Any]] = []
    for run_tag, run_rows in sorted(rows_by_run.items()):
        ordered = sorted(run_rows, key=lambda row: int(row["timestep_index"]))
        final_success = 1.0
        for current, nxt in zip(ordered, ordered[1:]):
            z_now = _as_float(current["newton.object.body_q.z"])
            z_next = _as_float(nxt["newton.object.body_q.z"])
            t_now = _as_float(current["newton.panda.sim_time"])
            t_next = _as_float(nxt["newton.panda.sim_time"])
            dt = max(t_next - t_now, 1.0e-6)
            contact_now = _as_float(current["newton.contact.rigid_contact_count"])
            contact_next = _as_float(nxt["newton.contact.rigid_contact_count"])
            commanded_lift = _as_float(current["candidate.controller.commanded_lift_target"])
            delta_z = z_next - z_now
            row: dict[str, Any] = {
                "run_tag": run_tag,
                "source_name": current["source_name"],
                "cell": current["cell"],
                "held_out_generalization_cell": str(_as_bool(current["held_out_generalization_cell"])).lower(),
                "timestep_index": int(current["timestep_index"]),
                "next_timestep_index": int(nxt["timestep_index"]),
            }
            for column in config["feature_columns"]:
                row[column] = _as_float(current[column])
            row.update(
                {
                    "curiosity.object.delta_z_next": delta_z,
                    "curiosity.object.velocity_z_next": delta_z / dt,
                    "curiosity.contact.rigid_contact_count_next": contact_next,
                    "curiosity.contact.delta_count_next": contact_next - contact_now,
                    "curiosity.contact_loss_risk_next": 1.0 if contact_next <= contact_loss_count else 0.0,
                    "curiosity.slip_risk_next": 1.0 if delta_z < -abs(slip_drop_z) else 0.0,
                    "curiosity.lift_response_residual_next": z_next - commanded_lift,
                    "curiosity.task.success_final": final_success,
                }
            )
            transition_rows.append(row)
    return transition_rows


def build(config_path: Path, root: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    source_runner_manifest_path = root / config["source_runner_manifest"]
    residual_preflight_path = root / config["residual_adapter_preflight_manifest"]
    fresh_sanity = _load_json(fresh_sanity_json)
    source_runner = _load_json(source_runner_manifest_path)
    residual_preflight = _load_json(residual_preflight_path)

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if fresh_sanity.get("status") != "pass":
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")
    if source_runner.get("status") != "pass":
        failures.append(f"source_runner_status:{source_runner.get('status')}")
    if residual_preflight.get("status") != "pass":
        failures.append(f"residual_preflight_status:{residual_preflight.get('status')}")
    for name, payload in (("source_runner", source_runner), ("residual_preflight", residual_preflight)):
        if payload.get("generated_trex_fields") != []:
            failures.append(f"{name}_generated_trex_fields_not_empty")
        if payload.get("schema_promotion") != "blocked":
            failures.append(f"{name}_schema_promotion:{payload.get('schema_promotion')}")

    records_csv = root / str(source_runner.get("records_csv", ""))
    if not records_csv.is_file():
        failures.append(f"missing_source_records_csv:{_rel(records_csv, root)}")
        fieldnames: list[str] = []
        source_rows: list[dict[str, str]] = []
    else:
        fieldnames, source_rows = _read_csv(records_csv)
    failures.extend(_forbidden_fields(fieldnames))

    required_columns = [
        "run_tag",
        "source_name",
        "cell",
        "held_out_generalization_cell",
        "timestep_index",
        *config["feature_columns"],
    ]
    missing_columns = [column for column in required_columns if column not in fieldnames]
    failures.extend(f"missing_required_column:{column}" for column in missing_columns)

    held_out = set(config["held_out_cells_reserved_for_evaluation"])
    train_cells = set(config["train_cells"])
    validation_cells = set(config["validation_cells"])
    if (train_cells | validation_cells) & held_out:
        failures.append(f"held_out_cell_in_training_split:{sorted((train_cells | validation_cells) & held_out)}")

    source_rows_by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        cell = row.get("cell", "")
        source_rows_by_cell[cell].append(row)
        if cell in held_out or _as_bool(row.get("held_out_generalization_cell", "")):
            failures.append(f"held_out_row_present:{cell}:{row.get('run_tag')}")

    if not missing_columns:
        transition_rows = _build_transition_rows(source_rows, config)
    else:
        transition_rows = []

    train_rows = [row for row in transition_rows if row["cell"] in train_cells]
    validation_rows = [row for row in transition_rows if row["cell"] in validation_cells]
    if not train_rows:
        failures.append("no_train_transition_rows")
    if not validation_rows:
        failures.append("no_validation_transition_rows")
    for cell in sorted(train_cells):
        if not any(row["cell"] == cell for row in train_rows):
            failures.append(f"missing_train_cell:{cell}")
    for cell in sorted(validation_cells):
        if not any(row["cell"] == cell for row in validation_rows):
            failures.append(f"missing_validation_cell:{cell}")

    split_rows = {"train": train_rows, "validation": validation_rows}
    column_summaries = {
        split_name: _column_summary(rows, [*config["feature_columns"], *config["target_columns"]])
        for split_name, rows in split_rows.items()
    }
    for split_name, rows in split_rows.items():
        if not rows:
            continue
        for column in config["required_nonzero_target_columns"]:
            if column_summaries[split_name][column]["nonzero"] <= 0:
                failures.append(f"{split_name}_target_all_zero:{column}")
        for column in config["required_nonconstant_target_columns"]:
            if column_summaries[split_name][column]["distinct_rounded_6"] <= 1:
                failures.append(f"{split_name}_target_constant:{column}")

    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "curiosity_forward_model_train_transitions.csv"
    validation_csv = output_dir / "curiosity_forward_model_validation_transitions.csv"
    manifest_path = output_dir / "manifest.json"
    output_fieldnames = [
        "run_tag",
        "source_name",
        "cell",
        "held_out_generalization_cell",
        "timestep_index",
        "next_timestep_index",
        *config["feature_columns"],
        *config["target_columns"],
    ]
    _write_csv(train_csv, output_fieldnames, train_rows)
    _write_csv(validation_csv, output_fieldnames, validation_rows)

    manifest = {
        "classification": "curiosity_forward_model_preflight_manifest_v1_not_training",
        "status": "pass" if not failures else "fail",
        "phase": "03_curiosity_reward",
        "config": _rel(config_path, root),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "source_runner_manifest": _rel(source_runner_manifest_path, root),
        "residual_adapter_preflight_manifest": _rel(residual_preflight_path, root),
        "source_records_csv": _rel(records_csv, root),
        "train_csv": _rel(train_csv, root),
        "validation_csv": _rel(validation_csv, root),
        "source_record_count": len(source_rows),
        "transition_record_count": len(transition_rows),
        "split_counts": {
            "train": {
                "record_count": len(train_rows),
                "cells": sorted(train_cells),
                "run_tags": sorted({str(row["run_tag"]) for row in train_rows}),
            },
            "validation": {
                "record_count": len(validation_rows),
                "cells": sorted(validation_cells),
                "run_tags": sorted({str(row["run_tag"]) for row in validation_rows}),
            },
        },
        "held_out_cells_reserved_for_evaluation": sorted(held_out),
        "feature_columns": list(config["feature_columns"]),
        "target_columns": list(config["target_columns"]),
        "column_summaries": column_summaries,
        "generated_trex_fields": [],
        "schema_promotion": "blocked",
        "training_started": False,
        "no_model_created": True,
        "no_placeholder_model": True,
        "valid_use": config["valid_use"],
        "not_valid_use": config["not_valid_use"],
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
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "transition_record_count": manifest["transition_record_count"],
                "split_counts": manifest["split_counts"],
                "manifest": str(Path(manifest["train_csv"]).parent / "manifest.json"),
                "schema_promotion": manifest["schema_promotion"],
                "generated_trex_fields": manifest["generated_trex_fields"],
                "training_started": manifest["training_started"],
                "no_model_created": manifest["no_model_created"],
                "failures": manifest["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if manifest["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
