"""Build source-matched Phase 01 learning-progress transition CSVs.

This compute-node step reads the already admitted train-only corrective source
rollouts and writes transition rows whose run_tag/timestep keys match the
residual-controller source manifest. It does not train a model and does not
use held-out cells.
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _scalar(array: Any, index: int, default: float = 0.0) -> float:
    try:
        value = array[index]
    except Exception:
        return default
    try:
        return float(value.reshape(-1)[0])
    except Exception:
        try:
            return float(value)
        except Exception:
            return default


def _object_z(array: Any, index: int) -> float:
    try:
        return float(array[index].reshape(-1)[2])
    except Exception:
        return 0.0


def _split_index(root: Path, source_manifest: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split_name, key in (("train", "train_csv"), ("validation", "validation_csv")):
        for row in _read_csv(root / source_manifest[key]):
            run_tag = row["run_tag"]
            previous = mapping.get(run_tag)
            if previous is not None and previous != split_name:
                raise ValueError(f"run_tag appears in multiple splits: {run_tag}")
            mapping[run_tag] = split_name
    return mapping


def _rows_from_npz(
    run_tag: str,
    cell: str,
    split: str,
    npz_path: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np

    thresholds = config["risk_thresholds"]
    contact_loss_count = float(thresholds["contact_loss_count"])
    slip_drop_z = abs(float(thresholds["slip_drop_z_m"]))
    rows: list[dict[str, Any]] = []
    with np.load(npz_path, allow_pickle=False) as data:
        required = [
            "newton.panda.sim_time",
            "newton.panda.rigid_contact_count",
            "newton.panda.object_body_q",
            "candidate.controller.phase_index",
            "candidate.controller.commanded_gripper_target",
            "candidate.controller.commanded_lift_target",
            "candidate.task.object_mass_kg",
            "candidate.task.object_friction_mu",
            "candidate.modality.vision_available_mask",
            "candidate.modality.contact_available_mask",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{run_tag}:missing_npz_keys:{','.join(missing)}")
        sim_time = data["newton.panda.sim_time"]
        contact = data["newton.panda.rigid_contact_count"]
        body_q = data["newton.panda.object_body_q"]
        length = min(len(sim_time), len(contact), len(body_q)) - 1
        for index in range(max(0, length)):
            z_now = _object_z(body_q, index)
            z_next = _object_z(body_q, index + 1)
            t_now = _scalar(sim_time, index)
            t_next = _scalar(sim_time, index + 1, t_now)
            dt = max(t_next - t_now, 1.0e-6)
            contact_now = _scalar(contact, index)
            contact_next = _scalar(contact, index + 1)
            delta_z = z_next - z_now
            rows.append(
                {
                    "split": split,
                    "cell": cell,
                    "run_tag": run_tag,
                    "source_run_tag": run_tag,
                    "timestep_index": index,
                    "next_timestep_index": index + 1,
                    "newton.panda.sim_time": t_now,
                    "newton.panda.rigid_contact_count": contact_now,
                    "newton.object.body_q.z": z_now,
                    "candidate.controller.phase_index": _scalar(data["candidate.controller.phase_index"], index),
                    "candidate.controller.commanded_gripper_target": _scalar(
                        data["candidate.controller.commanded_gripper_target"], index
                    ),
                    "candidate.controller.commanded_lift_target": _scalar(
                        data["candidate.controller.commanded_lift_target"], index
                    ),
                    "candidate.task.object_mass_kg": _scalar(data["candidate.task.object_mass_kg"], 0),
                    "candidate.task.object_friction_mu": _scalar(data["candidate.task.object_friction_mu"], 0),
                    "candidate.modality.vision_available_mask": _scalar(
                        data["candidate.modality.vision_available_mask"], index, 1.0
                    ),
                    "candidate.modality.contact_available_mask": _scalar(
                        data["candidate.modality.contact_available_mask"], index, 1.0
                    ),
                    "target.object.delta_z_next": delta_z,
                    "target.object.velocity_z_next": delta_z / dt,
                    "target.contact.count_next": contact_next,
                    "target.contact.delta_count_next": contact_next - contact_now,
                    "target.contact_loss_risk_next": 1.0 if contact_next <= contact_loss_count else 0.0,
                    "target.slip_risk_next": 1.0 if delta_z < -slip_drop_z else 0.0,
                }
            )
    return rows


def build(root: Path, config_path: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    source_manifest_path = root / config["source_manifest"]
    source_manifest = _load_json(source_manifest_path)
    fresh_sanity = _load_json(fresh_sanity_json)
    output_dir = root / config["output_dir"]
    manifest_path = root / config["manifest"]
    report_path = root / config["report"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if fresh_sanity.get("status") != "pass":
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")
    if source_manifest.get("status") != "pass":
        failures.append(f"source_manifest_status:{source_manifest.get('status')}")
    if source_manifest.get("generated_trex_fields") != []:
        failures.append("source_manifest_generated_trex_fields_not_empty")
    if source_manifest.get("schema_promotion") != "blocked":
        failures.append(f"source_manifest_schema_promotion:{source_manifest.get('schema_promotion')}")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    source_rows_by_split: dict[str, int] = {"train": int(source_manifest.get("train_row_count", 0)), "validation": int(source_manifest.get("validation_row_count", 0))}
    source_cells: dict[str, list[str]] = {"train": [], "validation": []}
    if not failures:
        try:
            split_by_run_tag = _split_index(root, source_manifest)
        except Exception as exc:
            split_by_run_tag = {}
            failures.append(f"split_index_failed:{type(exc).__name__}:{exc}")
        for item in source_manifest.get("admitted_sources", []):
            run_tag = str(item.get("feedback_run_tag", ""))
            cell = str(item.get("cell", ""))
            split = split_by_run_tag.get(run_tag)
            if split not in rows_by_split:
                failures.append(f"admitted_source_missing_from_source_split:{run_tag}")
                continue
            if cell.startswith("heldout_"):
                failures.append(f"held_out_cell_in_source_lp:{cell}")
                continue
            npz_path = root / str(item.get("feedback_npz", ""))
            if not npz_path.exists():
                failures.append(f"missing_feedback_npz:{run_tag}:{_rel(npz_path, root)}")
                continue
            try:
                rows = _rows_from_npz(run_tag, cell, split, npz_path, config)
            except Exception as exc:
                failures.append(f"source_transition_rows_failed:{run_tag}:{type(exc).__name__}:{exc}")
                continue
            rows_by_split[split].extend(rows)
            if cell not in source_cells[split]:
                source_cells[split].append(cell)

    if not rows_by_split["train"]:
        failures.append("no_train_source_transitions")
    if not rows_by_split["validation"]:
        failures.append("no_validation_source_transitions")

    fieldnames = [
        "split",
        "cell",
        "run_tag",
        "source_run_tag",
        "timestep_index",
        "next_timestep_index",
        *config["feature_columns"],
        *config["target_columns"],
    ]
    train_csv = output_dir / "train.csv"
    validation_csv = output_dir / "validation.csv"
    _write_csv(train_csv, fieldnames, rows_by_split["train"])
    _write_csv(validation_csv, fieldnames, rows_by_split["validation"])

    train_coverage = len(rows_by_split["train"]) / max(1, source_rows_by_split["train"])
    validation_coverage = len(rows_by_split["validation"]) / max(1, source_rows_by_split["validation"])
    manifest = {
        "classification": "phase01_source_matched_learning_progress_manifest_v1",
        "phase": "phase01",
        "status": "pass" if not failures else "fail",
        "not_training_result": True,
        "no_curiosity_success_claim": True,
        "config": _rel(config_path, root),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "source_manifest": _rel(source_manifest_path, root),
        "train_csv": _rel(train_csv, root),
        "validation_csv": _rel(validation_csv, root),
        "train_transition_count": len(rows_by_split["train"]),
        "validation_transition_count": len(rows_by_split["validation"]),
        "source_train_row_count": source_rows_by_split["train"],
        "source_validation_row_count": source_rows_by_split["validation"],
        "expected_train_score_coverage_for_residual_rows": train_coverage,
        "expected_validation_score_coverage_for_residual_rows": validation_coverage,
        "source_cells": source_cells,
        "held_out_excluded_from_training": source_manifest.get("held_out_excluded_from_training", []),
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "forbidden_claims": config["forbidden_claims"],
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Phase 01 Source-Matched Learning-Progress Manifest",
        "",
        f"- status: `{manifest['status']}`",
        f"- source manifest: `{manifest['source_manifest']}`",
        f"- train transitions: `{manifest['train_transition_count']}`",
        f"- validation transitions: `{manifest['validation_transition_count']}`",
        f"- expected train score coverage: `{manifest['expected_train_score_coverage_for_residual_rows']:.6f}`",
        f"- expected validation score coverage: `{manifest['expected_validation_score_coverage_for_residual_rows']:.6f}`",
        f"- train csv: `{manifest['train_csv']}`",
        f"- validation csv: `{manifest['validation_csv']}`",
        "",
        "This is source-matched transition preparation only. It is not policy training and not curiosity success evidence.",
    ]
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in failures)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fresh-sanity-json", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.root, args.config, args.fresh_sanity_json)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    raise SystemExit(0 if manifest["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
