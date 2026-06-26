"""Build a preflight manifest for residual-adapter training inputs.

This validates the five-source residual-label runner output and writes a
train/validation split for a future reviewed adapter trainer. It does not train
a model, create a checkpoint, or promote any field into a T-Rex schema key.
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


def _status_is_pass(payload: dict[str, Any]) -> bool:
    return str(payload.get("status", "")) == "pass"


def _forbidden_fields(fields: list[str]) -> list[str]:
    failures: list[str] = []
    for field in fields:
        if field in FORBIDDEN_EXACT_FIELDS or any(field.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            failures.append(f"forbidden_field:{field}")
    return failures


def _as_float(value: str) -> float:
    if value == "":
        return 0.0
    return float(value)


def _column_summary(rows: list[dict[str, str]], columns: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for column in columns:
        values = [_as_float(row[column]) for row in rows]
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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build(config_path: Path, root: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    contract_path = root / config["contract"]
    source_runner_manifest_path = root / config["source_runner_manifest"]
    source_manifest_path = root / config["source_manifest"]

    contract = _load_json(contract_path)
    source_runner_manifest = _load_json(source_runner_manifest_path)
    source_manifest = _load_json(source_manifest_path)
    fresh_sanity = _load_json(fresh_sanity_json)

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if not _status_is_pass(fresh_sanity):
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")
    if source_runner_manifest.get("status") != config["source_runner_status_required"]:
        failures.append(f"source_runner_status:{source_runner_manifest.get('status')}")
    if source_manifest.get("status") != config["source_manifest_status_required"]:
        failures.append(f"source_manifest_status:{source_manifest.get('status')}")
    for payload_name, payload in (
        ("contract", contract),
        ("source_runner_manifest", source_runner_manifest),
        ("source_manifest", source_manifest),
    ):
        if payload.get("generated_trex_fields") != []:
            failures.append(f"{payload_name}_generated_trex_fields_not_empty")
        if payload.get("schema_promotion") != "blocked":
            failures.append(f"{payload_name}_schema_promotion:{payload.get('schema_promotion')}")
        if payload.get("training_started") is not False:
            failures.append(f"{payload_name}_training_started:{payload.get('training_started')}")

    records_csv = root / str(source_runner_manifest.get("records_csv", ""))
    if not records_csv.is_file():
        failures.append(f"missing_source_records_csv:{_rel(records_csv, root)}")
        rows: list[dict[str, str]] = []
        fieldnames: list[str] = []
    else:
        with records_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

    feature_columns = list(config["feature_columns"])
    teacher_context_columns = list(config["teacher_context_columns"])
    target_columns = list(config["target_columns"])
    required_columns = [
        "run_tag",
        "source_name",
        "cell",
        "held_out_generalization_cell",
        "timestep_index",
        *feature_columns,
        *teacher_context_columns,
        *target_columns,
    ]
    missing_columns = [column for column in required_columns if column not in fieldnames]
    failures.extend(f"missing_required_column:{column}" for column in missing_columns)
    failures.extend(_forbidden_fields(fieldnames))

    held_out = set(config["held_out_cells_reserved_for_evaluation"])
    train_cells = set(config["train_cells"])
    validation_cells = set(config["validation_cells"])
    if train_cells & validation_cells:
        failures.append(f"train_validation_cell_overlap:{sorted(train_cells & validation_cells)}")
    if (train_cells | validation_cells) & held_out:
        failures.append(f"held_out_cell_in_training_split:{sorted((train_cells | validation_cells) & held_out)}")

    rows_by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cell = row.get("cell", "")
        rows_by_cell[cell].append(row)
        if cell in held_out:
            failures.append(f"held_out_row_present:{cell}:{row.get('run_tag')}")

    missing_train_cells = sorted(cell for cell in train_cells if not rows_by_cell.get(cell))
    missing_validation_cells = sorted(cell for cell in validation_cells if not rows_by_cell.get(cell))
    failures.extend(f"missing_train_cell:{cell}" for cell in missing_train_cells)
    failures.extend(f"missing_validation_cell:{cell}" for cell in missing_validation_cells)

    train_rows = [row for row in rows if row.get("cell") in train_cells]
    validation_rows = [row for row in rows if row.get("cell") in validation_cells]
    unused_source_cells = sorted(set(rows_by_cell) - train_cells - validation_cells)
    if not train_rows:
        failures.append("no_train_rows")
    if not validation_rows:
        failures.append("no_validation_rows")

    split_rows = {"train": train_rows, "validation": validation_rows}
    column_summaries = {
        split_name: _column_summary(split, [*feature_columns, *teacher_context_columns, *target_columns])
        for split_name, split in split_rows.items()
    }
    for split_name, split in split_rows.items():
        for column in config["required_nonzero_target_columns"]:
            if column_summaries[split_name][column]["nonzero"] <= 0:
                failures.append(f"{split_name}_target_all_zero:{column}")
        for column in config["required_nonconstant_target_columns"]:
            if column_summaries[split_name][column]["distinct_rounded_6"] <= 1:
                failures.append(f"{split_name}_target_constant:{column}")

    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "residual_adapter_train_records.csv"
    validation_csv = output_dir / "residual_adapter_validation_records.csv"
    manifest_path = output_dir / "manifest.json"
    output_fieldnames = required_columns
    _write_csv(train_csv, output_fieldnames, train_rows)
    _write_csv(validation_csv, output_fieldnames, validation_rows)

    split_counts = {
        "train": {
            "record_count": len(train_rows),
            "cells": sorted(train_cells),
            "run_tags": sorted({row.get("run_tag", "") for row in train_rows}),
        },
        "validation": {
            "record_count": len(validation_rows),
            "cells": sorted(validation_cells),
            "run_tags": sorted({row.get("run_tag", "") for row in validation_rows}),
        },
    }
    manifest = {
        "classification": "residual_adapter_training_preflight_manifest_v1_not_training",
        "status": "pass" if not failures else "fail",
        "phase": "04_closed_loop_adaptation",
        "config": _rel(config_path, root),
        "contract": _rel(contract_path, root),
        "source_runner_manifest": _rel(source_runner_manifest_path, root),
        "source_manifest": _rel(source_manifest_path, root),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "source_records_csv": _rel(records_csv, root),
        "train_csv": _rel(train_csv, root),
        "validation_csv": _rel(validation_csv, root),
        "source_record_count": len(rows),
        "split_counts": split_counts,
        "unused_source_cells": unused_source_cells,
        "held_out_cells_reserved_for_evaluation": sorted(held_out),
        "feature_columns": feature_columns,
        "teacher_context_columns": teacher_context_columns,
        "target_columns": target_columns,
        "column_summaries": column_summaries,
        "source_runner_status": source_runner_manifest.get("status"),
        "source_runner_record_count": source_runner_manifest.get("record_count"),
        "source_runner_source_run_count": source_runner_manifest.get("source_run_count"),
        "generated_trex_fields": [],
        "schema_promotion": "blocked",
        "training_started": False,
        "no_model_created": True,
        "no_placeholder_model": True,
        "valid_use": config["valid_use"],
        "not_valid_use": config["not_valid_use"],
        "failures": failures,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
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
                "source_record_count": manifest["source_record_count"],
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
