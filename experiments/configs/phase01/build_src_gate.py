"""Build the Phase 01 train-only corrective source manifest.

This compute-node gate admits only real scripted-feedback source rollouts that
show active feedback and paired advantage over their no-adaptation rollout.
It writes residual-controller CSVs without using held-out cells.
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


def _metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    rows = metrics.get("rows") or []
    if rows and key in rows[0]:
        return float(rows[0][key])
    per_world = metrics.get("per_world") or []
    if per_world and key in per_world[0]:
        return float(per_world[0][key])
    return default


def _status(metrics: dict[str, Any]) -> str:
    rows = metrics.get("rows") or []
    if rows:
        return str(rows[0].get("status", ""))
    per_world = metrics.get("per_world") or []
    if per_world:
        return str(per_world[0].get("status", ""))
    return ""


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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _evaluate_pair(no_metrics: dict[str, Any], fb_metrics: dict[str, Any], gate: dict[str, Any]) -> tuple[bool, list[str], dict[str, float]]:
    no = {
        "lift": _metric(no_metrics, "lift_height_m"),
        "hold": _metric(no_metrics, "hold_duration_s"),
        "slip": _metric(no_metrics, "max_slip_m"),
        "contact_loss": _metric(no_metrics, "contact_loss_frames"),
        "accel": _metric(no_metrics, "max_object_accel_m_s2"),
        "success": 1.0 if _status(no_metrics) == "success" else 0.0,
    }
    fb = {
        "lift": _metric(fb_metrics, "lift_height_m"),
        "hold": _metric(fb_metrics, "hold_duration_s"),
        "slip": _metric(fb_metrics, "max_slip_m"),
        "contact_loss": _metric(fb_metrics, "contact_loss_frames"),
        "accel": _metric(fb_metrics, "max_object_accel_m_s2"),
        "success": 1.0 if _status(fb_metrics) == "success" else 0.0,
    }
    delta = {f"{key}_delta_fb_minus_no": fb[key] - no[key] for key in no}
    failures: list[str] = []
    if fb["success"] < 1.0:
        failures.append("feedback_rollout_not_success")
    if fb["lift"] + float(gate["lift_regression_tolerance_m"]) < no["lift"]:
        failures.append("lift_regression")
    if fb["hold"] + float(gate["hold_regression_tolerance_s"]) < no["hold"]:
        failures.append("hold_regression")
    if fb["slip"] > no["slip"] + float(gate["slip_regression_tolerance_m"]):
        failures.append("slip_regression")
    if fb["accel"] > no["accel"] + float(gate["accel_regression_tolerance_m_s2"]):
        failures.append("accel_regression")
    if fb["contact_loss"] > no["contact_loss"] + float(gate["contact_loss_regression_tolerance_frames"]):
        failures.append("contact_loss_regression")

    improvements = []
    if no["success"] < 1.0 and fb["success"] >= 1.0:
        improvements.append("success_repair")
    if no["slip"] - fb["slip"] >= float(gate["min_slip_improvement_m"]):
        improvements.append("slip_improvement")
    if no["accel"] - fb["accel"] >= float(gate["min_accel_improvement_m_s2"]):
        improvements.append("accel_improvement")
    if fb["hold"] - no["hold"] >= float(gate["min_hold_improvement_s"]):
        improvements.append("hold_improvement")
    if no["contact_loss"] - fb["contact_loss"] >= float(gate["min_contact_loss_improvement_frames"]):
        improvements.append("contact_loss_improvement")
    if not improvements:
        failures.append("no_paired_advantage")
    delta["improvement_count"] = float(len(improvements))
    return not failures, failures, delta


def _rows_from_npz(root: Path, run_tag: str, cell: str, split: str, npz_path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    import numpy as np

    rows: list[dict[str, Any]] = []
    active_count = 0
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
            "candidate.controller.feedback_active",
            "candidate.controller.feedback_lift_velocity_scale",
            "candidate.controller.feedback_hold_height_offset_m",
            "candidate.controller.feedback_stabilization_extension_s",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{run_tag}:missing_npz_keys:{','.join(missing)}")
        length = min(len(data["newton.panda.sim_time"]), len(data["candidate.controller.feedback_active"]))
        for index in range(length):
            active = _scalar(data["candidate.controller.feedback_active"], index)
            if active > 0.5:
                active_count += 1
            rows.append(
                {
                    "split": split,
                    "cell": cell,
                    "run_tag": run_tag,
                    "source_run_tag": run_tag,
                    "timestep_index": index,
                    "newton.panda.sim_time": _scalar(data["newton.panda.sim_time"], index),
                    "newton.panda.rigid_contact_count": _scalar(data["newton.panda.rigid_contact_count"], index),
                    "newton.object.body_q.z": _object_z(data["newton.panda.object_body_q"], index),
                    "candidate.controller.phase_index": _scalar(data["candidate.controller.phase_index"], index),
                    "candidate.controller.commanded_gripper_target": _scalar(data["candidate.controller.commanded_gripper_target"], index),
                    "candidate.controller.commanded_lift_target": _scalar(data["candidate.controller.commanded_lift_target"], index),
                    "candidate.task.object_mass_kg": _scalar(data["candidate.task.object_mass_kg"], 0),
                    "candidate.task.object_friction_mu": _scalar(data["candidate.task.object_friction_mu"], 0),
                    "candidate.modality.vision_available_mask": _scalar(data["candidate.modality.vision_available_mask"], index, 1.0),
                    "candidate.modality.contact_available_mask": _scalar(data["candidate.modality.contact_available_mask"], index, 1.0),
                    "candidate.controller.feedback_active": active,
                    "candidate.controller.feedback_lift_velocity_scale": _scalar(data["candidate.controller.feedback_lift_velocity_scale"], index, 1.0),
                    "candidate.controller.feedback_hold_height_offset_m": _scalar(data["candidate.controller.feedback_hold_height_offset_m"], index, 0.0),
                    "candidate.controller.feedback_stabilization_extension_s": _scalar(data["candidate.controller.feedback_stabilization_extension_s"], index, 0.0),
                }
            )
    return rows, active_count


def build(root: Path, config_path: Path, run_tag: str) -> dict[str, Any]:
    import numpy as np

    config = _load_json(config_path)
    output_root = root / "experiments" / "outputs" / config["output_subdir"]
    processed_dir = root / config["processed_dir"]
    manifest_path = root / config["manifest"]
    report_path = root / config["report"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")

    held_out = set(config["held_out_cells_reserved_for_evaluation"])
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    gate = config["advantage_gate"]

    for cell_spec in config["train_cells"]:
        cell = str(cell_spec["cell"])
        if cell in held_out or cell.startswith("heldout_"):
            failures.append(f"held_out_cell_in_source_collection:{cell}")
            continue
        no_tag = f"{run_tag}_no_{cell}"
        fb_tag = f"{run_tag}_fb_{cell}"
        no_metrics_path = output_root / f"{no_tag}_metrics.json"
        fb_metrics_path = output_root / f"{fb_tag}_metrics.json"
        fb_npz_path = output_root / f"{fb_tag}.npz"
        no_npz_path = output_root / f"{no_tag}.npz"
        missing = [path for path in [no_metrics_path, fb_metrics_path, fb_npz_path, no_npz_path] if not path.exists()]
        if missing:
            rejected.append({"cell": cell, "status": "missing_artifacts", "missing": [_rel(path, root) for path in missing]})
            continue
        no_metrics = _load_json(no_metrics_path)
        fb_metrics = _load_json(fb_metrics_path)
        pair_pass, pair_failures, deltas = _evaluate_pair(no_metrics, fb_metrics, gate)
        with np.load(fb_npz_path, allow_pickle=False) as data:
            active = np.asarray(data["candidate.controller.feedback_active"])
            trigger = np.asarray(data["candidate.controller.feedback_trigger_count"])
            active_frames = int(np.count_nonzero(active))
            trigger_count = int(np.max(trigger)) if trigger.size else 0
        if active_frames < int(gate["min_active_feedback_frames"]):
            pair_failures.append(f"active_feedback_frames_below_min:{active_frames}")
        if trigger_count < int(gate["min_feedback_trigger_count"]):
            pair_failures.append(f"feedback_trigger_count_below_min:{trigger_count}")
        record = {
            "cell": cell,
            "no_adaptation_run_tag": no_tag,
            "feedback_run_tag": fb_tag,
            "feedback_npz": _rel(fb_npz_path, root),
            "no_adaptation_metrics": _rel(no_metrics_path, root),
            "feedback_metrics": _rel(fb_metrics_path, root),
            "feedback_active_frames": active_frames,
            "feedback_trigger_count": trigger_count,
            "paired_metric_deltas": deltas,
        }
        if pair_pass and not pair_failures:
            record["status"] = "admitted"
            admitted.append(record)
        else:
            record["status"] = "rejected"
            record["failures"] = pair_failures
            rejected.append(record)

    if len(admitted) < int(gate["min_admitted_cells"]):
        failures.append(f"admitted_cells_below_min:{len(admitted)}")

    validation_count = min(int(gate["validation_admitted_cells"]), max(0, len(admitted) - 1))
    validation_cells = {item["cell"] for item in admitted[-validation_count:]} if validation_count else set()
    rows_by_split = {"train": [], "validation": []}
    active_by_split = {"train": 0, "validation": 0}
    row_failures: list[str] = []
    for item in admitted:
        split = "validation" if item["cell"] in validation_cells else "train"
        try:
            rows, active_count = _rows_from_npz(root, item["feedback_run_tag"], item["cell"], split, root / item["feedback_npz"], config)
        except Exception as exc:
            row_failures.append(f"{item['cell']}:{type(exc).__name__}:{exc}")
            continue
        rows_by_split[split].extend(rows)
        active_by_split[split] += active_count

    failures.extend(row_failures)
    if admitted and not rows_by_split["train"]:
        failures.append("no_train_rows_from_admitted_sources")
    if admitted and not rows_by_split["validation"]:
        failures.append("no_validation_rows_from_admitted_sources")
    if rows_by_split["train"] and active_by_split["train"] == 0:
        failures.append("train_active_feedback_count_zero_after_gate")
    if rows_by_split["validation"] and active_by_split["validation"] == 0:
        failures.append("validation_active_feedback_count_zero_after_gate")

    fieldnames = ["split", "cell", "run_tag", "source_run_tag", "timestep_index", *config["feature_columns"], *config["target_columns"]]
    train_csv = processed_dir / "train.csv"
    validation_csv = processed_dir / "validation.csv"
    _write_csv(train_csv, fieldnames, rows_by_split["train"])
    _write_csv(validation_csv, fieldnames, rows_by_split["validation"])

    manifest = {
        "classification": "phase01_train_only_corrective_source_gate_v1",
        "phase": "phase01",
        "status": "pass" if not failures else "fail",
        "run_tag": run_tag,
        "not_training_result": True,
        "no_curiosity_success_claim": True,
        "config": _rel(config_path, root),
        "train_csv": _rel(train_csv, root),
        "validation_csv": _rel(validation_csv, root),
        "train_row_count": len(rows_by_split["train"]),
        "validation_row_count": len(rows_by_split["validation"]),
        "train_active_feedback_count": active_by_split["train"],
        "validation_active_feedback_count": active_by_split["validation"],
        "admitted_count": len(admitted),
        "rejected_count": len(rejected),
        "admitted_sources": admitted,
        "rejected_sources": rejected,
        "held_out_excluded_from_training": sorted(held_out),
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "forbidden_claims": config["forbidden_claims"],
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Phase 01 Train-Only Corrective Source Gate",
        "",
        f"- status: `{manifest['status']}`",
        f"- run tag: `{run_tag}`",
        f"- admitted sources: `{manifest['admitted_count']}`",
        f"- rejected sources: `{manifest['rejected_count']}`",
        f"- train rows: `{manifest['train_row_count']}`",
        f"- validation rows: `{manifest['validation_row_count']}`",
        f"- train active feedback labels: `{manifest['train_active_feedback_count']}`",
        f"- validation active feedback labels: `{manifest['validation_active_feedback_count']}`",
        f"- train csv: `{manifest['train_csv']}`",
        f"- validation csv: `{manifest['validation_csv']}`",
        "",
        "This is corrective source data preparation only. It is not residual training and not curiosity success evidence.",
    ]
    if admitted:
        lines.extend(["", "## Admitted", ""])
        lines.extend(f"- `{item['cell']}` active `{item['feedback_active_frames']}` triggers `{item['feedback_trigger_count']}`" for item in admitted)
    if rejected:
        lines.extend(["", "## Rejected", ""])
        for item in rejected:
            reason = ", ".join(item.get("failures", item.get("missing", [])))
            lines.append(f"- `{item['cell']}`: {reason}")
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in failures)
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
