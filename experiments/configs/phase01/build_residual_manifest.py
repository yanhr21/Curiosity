"""Build Phase 01 residual-controller CSVs from accepted Phase 00 NPZ files.

This is a compute-node data preparation step. It keeps the original Newton
feedback fields and does not fabricate residual labels.
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


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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


def _vision_mask(mask_id: float, index: int) -> float:
    if int(mask_id) == 1:
        return 0.0
    if int(mask_id) == 3:
        return 0.0 if index % 2 == 1 else 1.0
    return 1.0


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build(root: Path, config_path: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    import numpy as np

    config = _load_json(config_path)
    contract = _load_json(root / config["contract"])
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

    rows_by_cell: dict[str, dict[str, Any]] = {}
    for row_file_name in contract["phase00_sources"]["row_files"]:
        row_file = root / row_file_name
        if not row_file.exists():
            failures.append(f"missing_row_file:{row_file_name}")
            continue
        for row in _read_rows(row_file):
            if row.get("status") == "generated_pending_manual_review":
                rows_by_cell[str(row["cell"])] = row

    train_cells = set(contract["splits"]["train"])
    validation_cells = set(contract["splits"]["validation"])
    held_out_cells = set(contract["splits"]["held_out"])
    if (train_cells | validation_cells) & held_out_cells:
        failures.append("held_out_overlap_with_training_or_validation")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    source_cells: dict[str, list[str]] = {"train": [], "validation": [], "held_out_seen_but_excluded": []}
    active_counts: dict[str, int] = {"train": 0, "validation": 0}
    required = list(config["required_npz_keys"])

    for split_name, split_cells in (("train", train_cells), ("validation", validation_cells)):
        for cell in sorted(split_cells):
            row = rows_by_cell.get(cell)
            if row is None:
                failures.append(f"missing_generated_cell:{split_name}:{cell}")
                continue
            npz_path = root / "experiments" / "outputs" / f"{row['run_tag']}.npz"
            if not npz_path.exists():
                failures.append(f"missing_npz:{cell}:{_rel(npz_path, root)}")
                continue
            source_cells[split_name].append(cell)
            with np.load(npz_path, allow_pickle=False) as data:
                missing = [key for key in required if key not in data.files]
                if missing:
                    failures.append(f"missing_npz_keys:{cell}:{','.join(missing)}")
                    continue
                sim_time = data["newton.panda.sim_time"]
                contact = data["newton.panda.rigid_contact_count"]
                body_q = data["newton.panda.object_body_q"]
                phase = data["candidate.controller.phase_index"]
                grip = data["candidate.controller.commanded_gripper_target"]
                lift = data["candidate.controller.commanded_lift_target"]
                mass = data["candidate.task.object_mass_kg"]
                friction = data["candidate.task.object_friction_mu"]
                mask_id = _scalar(data["candidate.modality.mask_mode_id"], 0, 0.0)
                contact_mask = data["candidate.modality.contact_available_mask"]
                feedback_active = data["candidate.controller.feedback_active"]
                feedback_lift = data["candidate.controller.feedback_lift_velocity_scale"]
                feedback_hold = data["candidate.controller.feedback_hold_height_offset_m"]
                feedback_stabilize = data["candidate.controller.feedback_stabilization_extension_s"]
                length = min(
                    len(sim_time),
                    len(contact),
                    len(body_q),
                    len(feedback_active),
                    len(feedback_lift),
                    len(feedback_hold),
                    len(feedback_stabilize),
                )
                for index in range(max(0, length)):
                    active = _scalar(feedback_active, index)
                    if active > 0.5:
                        active_counts[split_name] += 1
                    rows_by_split[split_name].append(
                        {
                            "split": split_name,
                            "cell": cell,
                            "run_tag": row["run_tag"],
                            "source_run_tag": row["run_tag"],
                            "timestep_index": index,
                            "newton.panda.sim_time": _scalar(sim_time, index),
                            "newton.panda.rigid_contact_count": _scalar(contact, index),
                            "newton.object.body_q.z": float(body_q[index].reshape(-1)[2]),
                            "candidate.controller.phase_index": _scalar(phase, index),
                            "candidate.controller.commanded_gripper_target": _scalar(grip, index),
                            "candidate.controller.commanded_lift_target": _scalar(lift, index),
                            "candidate.task.object_mass_kg": _scalar(mass, 0),
                            "candidate.task.object_friction_mu": _scalar(friction, 0),
                            "candidate.modality.vision_available_mask": _vision_mask(mask_id, index),
                            "candidate.modality.contact_available_mask": _scalar(contact_mask, index, 1.0),
                            "candidate.controller.feedback_active": active,
                            "candidate.controller.feedback_lift_velocity_scale": _scalar(feedback_lift, index, 1.0),
                            "candidate.controller.feedback_hold_height_offset_m": _scalar(feedback_hold, index, 0.0),
                            "candidate.controller.feedback_stabilization_extension_s": _scalar(
                                feedback_stabilize, index, 0.0
                            ),
                        }
                    )

    for cell in sorted(held_out_cells):
        if cell in rows_by_cell:
            source_cells["held_out_seen_but_excluded"].append(cell)

    fieldnames = [
        "split",
        "cell",
        "run_tag",
        "source_run_tag",
        "timestep_index",
        *config["feature_columns"],
        *config["target_columns"],
    ]
    train_csv = output_dir / "train.csv"
    validation_csv = output_dir / "validation.csv"
    _write_csv(train_csv, fieldnames, rows_by_split["train"])
    _write_csv(validation_csv, fieldnames, rows_by_split["validation"])

    if not rows_by_split["train"]:
        failures.append("no_train_rows")
    if not rows_by_split["validation"]:
        failures.append("no_validation_rows")
    if active_counts["train"] == 0:
        failures.append("no_active_feedback_labels_in_train")
    if active_counts["validation"] == 0:
        failures.append("no_active_feedback_labels_in_validation")

    manifest = {
        "classification": "phase01_residual_manifest",
        "phase": "phase01",
        "status": "pass" if not failures else "fail",
        "not_training_result": True,
        "no_curiosity_success_claim": True,
        "config": _rel(config_path, root),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "train_csv": _rel(train_csv, root),
        "validation_csv": _rel(validation_csv, root),
        "train_row_count": len(rows_by_split["train"]),
        "validation_row_count": len(rows_by_split["validation"]),
        "train_active_feedback_count": active_counts["train"],
        "validation_active_feedback_count": active_counts["validation"],
        "source_cells": source_cells,
        "held_out_excluded_from_training": sorted(held_out_cells),
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "forbidden_claims": config["forbidden_claims"],
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase 01 Residual Manifest",
        "",
        f"- status: `{manifest['status']}`",
        f"- train rows: `{manifest['train_row_count']}`",
        f"- validation rows: `{manifest['validation_row_count']}`",
        f"- train active feedback labels: `{manifest['train_active_feedback_count']}`",
        f"- validation active feedback labels: `{manifest['validation_active_feedback_count']}`",
        f"- train csv: `{manifest['train_csv']}`",
        f"- validation csv: `{manifest['validation_csv']}`",
        f"- held-out excluded: `{', '.join(manifest['held_out_excluded_from_training'])}`",
        "",
        "This is data preparation only. It is not residual training and not curiosity success evidence.",
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
