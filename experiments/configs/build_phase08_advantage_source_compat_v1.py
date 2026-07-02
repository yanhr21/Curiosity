#!/usr/bin/env python3
"""Build a source-runner-compatible manifest from Phase08 advantage-gated data.

This is glue for the existing Newton-native curiosity forward-model preflight.
It does not create a model, train a checkpoint, or relabel failed held-out
rollouts as training data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


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


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build(config_path: Path, root: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    preflight_path = root / config["advantage_preflight_manifest"]
    preflight = _load_json(preflight_path)
    fresh_sanity = _load_json(fresh_sanity_json)
    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if fresh_sanity.get("status") != "pass":
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")
    if preflight.get("status") != "pass":
        failures.append(f"advantage_preflight_status:{preflight.get('status')}")
    if preflight.get("schema_promotion") != "blocked":
        failures.append(f"advantage_preflight_schema_promotion:{preflight.get('schema_promotion')}")
    if preflight.get("generated_trex_fields") != []:
        failures.append("advantage_preflight_generated_trex_fields_not_empty")

    train_csv = root / preflight.get("train_csv", "")
    validation_csv = root / preflight.get("validation_csv", "")
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    split_counts: dict[str, int] = {}
    for split_name, csv_path in (("train", train_csv), ("validation", validation_csv)):
        if not csv_path.is_file():
            failures.append(f"missing_{split_name}_csv:{_rel(csv_path, root)}")
            continue
        split_fields, split_rows = _read_csv(csv_path)
        if not fieldnames:
            fieldnames = split_fields
        elif split_fields != fieldnames:
            failures.append(f"{split_name}_fieldnames_mismatch")
        for row in split_rows:
            if row.get("held_out_generalization_cell", "").lower() in {"1", "true", "yes"}:
                failures.append(f"held_out_row_present:{row.get('cell')}:{row.get('run_tag')}")
            rows.append(row)
        split_counts[split_name] = len(split_rows)

    required_columns = [
        "run_tag",
        "source_name",
        "split",
        "cell",
        "held_out_generalization_cell",
        "timestep_index",
        "newton.panda.sim_time",
        "newton.contact.rigid_contact_count",
        "newton.object.body_q.z",
        "candidate.controller.commanded_lift_target",
    ]
    for column in required_columns:
        if column not in fieldnames:
            failures.append(f"missing_required_column:{column}")
    if not rows:
        failures.append("no_rows")

    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    records_csv = output_dir / "phase08_advantage_source_records.csv"
    manifest_path = output_dir / "manifest.json"
    if fieldnames:
        _write_csv(records_csv, fieldnames, rows)

    manifest = {
        "classification": "phase08_advantage_source_compat_manifest_v1_not_training",
        "status": "pass" if not failures else "fail",
        "config": _rel(config_path, root),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "advantage_preflight_manifest": _rel(preflight_path, root),
        "records_csv": _rel(records_csv, root),
        "record_count": len(rows),
        "source_run_count": len({row.get("run_tag", "") for row in rows}),
        "cells": sorted({row.get("cell", "") for row in rows}),
        "split_counts": split_counts,
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "training_started": False,
        "not_success_claim": True,
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--fresh-sanity-json", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.config, args.root, args.fresh_sanity_json)
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
