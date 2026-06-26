"""Build a namespace-preserving Newton contact source manifest.

This converts real Newton lift-hold rollout artifacts into manifest/CSV records
under explicit `newton.*` and `candidate.*` namespaces. It does not create
T-Rex `observation.*`, `action`, tactile F6, tactile deformation, policy, or
model fields.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FORBIDDEN_PREFIXES = (
    "observation.",
    "action",
    "action_abs",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _first_scalar(value: np.ndarray, index: int) -> float:
    arr = np.asarray(value[index])
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def _last_axis_z(value: np.ndarray, index: int) -> float:
    arr = np.asarray(value[index])
    flat = arr.reshape(-1)
    if flat.size < 3:
        return 0.0
    return float(flat[2])


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value)
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        summary.update(
            {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "nonzero": int(np.count_nonzero(arr)),
            }
        )
    return summary


def _collect_cells(source_config: dict[str, Any]) -> list[dict[str, Any]]:
    plan = source_config["initial_run_plan"]
    cells: list[dict[str, Any]] = []

    nominal = dict(plan["nominal_cup"])
    nominal.update(
        {
            "cell": "nominal_cup",
            "split": "nominal",
            "mass_label": "nominal",
            "friction_label": "nominal",
            "held_out_generalization_cell": False,
        }
    )
    cells.append(nominal)

    for cell, entry in sorted(plan.get("completed_ordinary_grid_cells", {}).items()):
        item = dict(entry)
        mass_label, friction_label = cell.split("_", 1)
        item.update(
            {
                "cell": cell,
                "split": "ordinary",
                "mass_label": mass_label,
                "friction_label": friction_label,
                "held_out_generalization_cell": False,
            }
        )
        cells.append(item)

    for cell, entry in sorted(plan.get("completed_held_out_evaluation_cells", {}).items()):
        item = dict(entry)
        mass_label, friction_label = cell.split("_", 1)
        item.update(
            {
                "cell": cell,
                "split": "held_out",
                "mass_label": mass_label,
                "friction_label": friction_label,
                "held_out_generalization_cell": True,
            }
        )
        cells.append(item)

    return cells


def _check_no_forbidden_fields(fields: list[str]) -> list[str]:
    failures: list[str] = []
    for field in fields:
        if field == "action" or field == "action_abs" or field.startswith("observation."):
            failures.append(f"forbidden_field:{field}")
    return failures


def build_manifest(config_path: Path, root: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    source_config_path = root / config["source_config"]
    source_config = _load_json(source_config_path)

    failures: list[str] = []
    required_status = config.get("source_status_required")
    if source_config.get("status") != required_status:
        failures.append(f"source_status:{source_config.get('status')}")

    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "contact_source_records.csv"
    manifest_path = output_dir / "manifest.json"

    rows: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    all_fields: set[str] = set()
    contact_min: int | None = None
    contact_max: int | None = None
    total_feedback_triggers = 0

    for cell in _collect_cells(source_config):
        run_tag = cell["run_tag"]
        npz_path = root / cell["npz"]
        summary_path = root / cell["summary_json"]
        visual_validation_path = root / cell["visual_validation"]
        manual_path = root / cell["manual_visual_inspection"]
        metrics_path = root / cell["metrics_json"]
        sanity_path = root / cell["fresh_newton_sensor_contact_sanity"]

        for label, path in [
            ("npz", npz_path),
            ("summary_json", summary_path),
            ("visual_validation", visual_validation_path),
            ("manual_visual_inspection", manual_path),
            ("metrics_json", metrics_path),
            ("fresh_newton_sensor_contact_sanity", sanity_path),
        ]:
            if not path.is_file():
                failures.append(f"missing_{label}:{run_tag}:{_rel(path, root)}")

        summary = _load_json(summary_path)
        visual_validation = _load_json(visual_validation_path)
        manual = _load_json(manual_path)
        metrics = _load_json(metrics_path)
        sanity = _load_json(sanity_path)

        if visual_validation.get("status") != "pass":
            failures.append(f"visual_validation_not_pass:{run_tag}")
        if manual.get("status") != "pass":
            failures.append(f"manual_visual_not_pass:{run_tag}")
        if sanity.get("status") != "pass":
            failures.append(f"fresh_sanity_not_pass:{run_tag}")

        with np.load(npz_path, allow_pickle=False) as data:
            fields = sorted(data.files)
            all_fields.update(fields)
            failures.extend(_check_no_forbidden_fields(fields))
            required_arrays = [
                "newton.panda.step",
                "newton.panda.sim_time",
                "newton.panda.rigid_contact_count",
                "newton.panda.object_body_q",
                "candidate.controller.feedback_trigger_count",
                "candidate.controller.phase_index",
            ]
            missing_arrays = [field for field in required_arrays if field not in data]
            for field in missing_arrays:
                failures.append(f"missing_required_array:{run_tag}:{field}")
            if missing_arrays:
                continue

            steps = np.asarray(data["newton.panda.step"])
            sim_time = np.asarray(data["newton.panda.sim_time"])
            contact = np.asarray(data["newton.panda.rigid_contact_count"])
            object_q = np.asarray(data["newton.panda.object_body_q"])
            trigger = np.asarray(data["candidate.controller.feedback_trigger_count"])
            phase = np.asarray(data["candidate.controller.phase_index"])
            grip = np.asarray(data.get("candidate.controller.commanded_gripper_target", np.zeros_like(steps)))
            lift = np.asarray(data.get("candidate.controller.commanded_lift_target", np.zeros_like(steps)))

            timesteps = int(steps.shape[0])
            if contact.shape[0] != timesteps or object_q.shape[0] != timesteps:
                failures.append(f"array_length_mismatch:{run_tag}")
                continue

            contact_values = np.asarray(contact).reshape(timesteps, -1)
            run_contact_min = int(np.min(contact_values))
            run_contact_max = int(np.max(contact_values))
            contact_min = run_contact_min if contact_min is None else min(contact_min, run_contact_min)
            contact_max = run_contact_max if contact_max is None else max(contact_max, run_contact_max)
            run_feedback_triggers = int(np.max(trigger)) if trigger.size else 0
            total_feedback_triggers += run_feedback_triggers

            source_runs.append(
                {
                    "run_tag": run_tag,
                    "cell": cell["cell"],
                    "split": cell["split"],
                    "held_out_generalization_cell": bool(cell["held_out_generalization_cell"]),
                    "npz": _rel(npz_path, root),
                    "summary_json": _rel(summary_path, root),
                    "visual_validation": _rel(visual_validation_path, root),
                    "manual_visual_inspection": _rel(manual_path, root),
                    "metrics_json": _rel(metrics_path, root),
                    "fresh_newton_sensor_contact_sanity": _rel(sanity_path, root),
                    "contact_sheet": summary.get("contact_sheet", cell.get("contact_sheet", "")),
                    "frame_browser": summary.get("frame_browser", cell.get("frame_browser", "")),
                    "status": metrics.get("status"),
                    "metrics_row": (metrics.get("rows") or [{}])[0],
                    "array_summaries": {field: _array_summary(np.asarray(data[field])) for field in required_arrays},
                    "contact_count_min": run_contact_min,
                    "contact_count_max": run_contact_max,
                    "feedback_trigger_count": run_feedback_triggers,
                }
            )

            for idx in range(timesteps):
                rows.append(
                    {
                        "run_tag": run_tag,
                        "cell": cell["cell"],
                        "split": cell["split"],
                        "mass_label": cell["mass_label"],
                        "friction_label": cell["friction_label"],
                        "held_out_generalization_cell": str(bool(cell["held_out_generalization_cell"])),
                        "timestep_index": idx,
                        "newton.panda.step": int(np.asarray(steps[idx]).reshape(-1)[0]),
                        "newton.panda.sim_time": float(np.asarray(sim_time[idx]).reshape(-1)[0]),
                        "newton.contact.rigid_contact_count": int(_first_scalar(contact, idx)),
                        "newton.object.body_q.z": _last_axis_z(object_q, idx),
                        "candidate.controller.phase_index": int(_first_scalar(phase, idx)),
                        "candidate.controller.feedback_trigger_count": int(_first_scalar(trigger, idx)),
                        "candidate.controller.commanded_gripper_target": float(_first_scalar(grip, idx)),
                        "candidate.controller.commanded_lift_target": float(_first_scalar(lift, idx)),
                    }
                )

    dataset_fields = [
        "newton.panda.step",
        "newton.panda.sim_time",
        "newton.contact.rigid_contact_count",
        "newton.object.body_q.z",
        "candidate.controller.phase_index",
        "candidate.controller.feedback_trigger_count",
        "candidate.controller.commanded_gripper_target",
        "candidate.controller.commanded_lift_target",
    ]
    failures.extend(_check_no_forbidden_fields(dataset_fields))

    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run_tag",
            "cell",
            "split",
            "mass_label",
            "friction_label",
            "held_out_generalization_cell",
            "timestep_index",
            *dataset_fields,
        ])
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "classification": "newton_lift_hold_contact_source_manifest_v1_not_trex_schema",
        "status": "pass" if not failures else "fail",
        "phase": "05_tactile_contact_sources",
        "source_config": _rel(source_config_path, root),
        "source_config_status": source_config.get("status"),
        "config": _rel(config_path, root),
        "record_count": len(rows),
        "source_run_count": len(source_runs),
        "records_csv": _rel(records_path, root),
        "source_runs": source_runs,
        "dataset_fields": dataset_fields,
        "source_npz_fields": sorted(all_fields),
        "field_mapping": config["field_mapping"],
        "allowed_namespaces": config["allowed_namespaces"],
        "generated_trex_fields": [],
        "schema_promotion": "blocked",
        "forbidden_generated_fields": config["forbidden_generated_fields"],
        "contact_count_min": contact_min,
        "contact_count_max": contact_max,
        "total_feedback_trigger_count": total_feedback_triggers,
        "valid_use": config["valid_use"],
        "not_valid_use": config["not_valid_use"],
        "failures": failures,
        "note": "Namespace-preserving conversion only. Dataset/schema mismatch is not treated as a stop gate; exact T-Rex schema promotion remains blocked.",
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    args = parser.parse_args()
    manifest = build_manifest(args.config, args.root)
    print(json.dumps({
        "status": manifest["status"],
        "record_count": manifest["record_count"],
        "source_run_count": manifest["source_run_count"],
        "records_csv": manifest["records_csv"],
        "manifest": str(Path(manifest["records_csv"]).parent / "manifest.json"),
        "schema_promotion": manifest["schema_promotion"],
        "generated_trex_fields": manifest["generated_trex_fields"],
        "failures": manifest["failures"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
