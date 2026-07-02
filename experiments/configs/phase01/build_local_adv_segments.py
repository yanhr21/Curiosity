"""Build Phase 01 local-advantage residual segments.

This is a train-only data/objective repair after source-level gates failed.
It keeps held-out cells excluded and admits fixed-length local segments only
from source rollouts whose paired metrics do not show safety/lift/hold
regression. It is not training and not a curiosity success claim.
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


def _source_failures_allowed(record: dict[str, Any], config: dict[str, Any]) -> bool:
    allowed_prefixes = tuple(config["eligibility"]["allowed_failure_prefixes"])
    for failure in record.get("failures", []):
        if not any(str(failure).startswith(prefix) for prefix in allowed_prefixes):
            return False
    return True


def _paired_deltas_safe(record: dict[str, Any], config: dict[str, Any]) -> tuple[bool, list[str]]:
    deltas = dict(record.get("paired_metric_deltas", {}))
    tol = config["eligibility"]["paired_delta_tolerances"]
    failures: list[str] = []
    if float(deltas.get("lift_delta_fb_minus_no", 0.0)) < -float(tol["lift_regression_tolerance_m"]):
        failures.append("lift_regression")
    if float(deltas.get("hold_delta_fb_minus_no", 0.0)) < -float(tol["hold_regression_tolerance_s"]):
        failures.append("hold_regression")
    if float(deltas.get("slip_delta_fb_minus_no", 0.0)) > float(tol["slip_regression_tolerance_m"]):
        failures.append("slip_regression")
    if float(deltas.get("accel_delta_fb_minus_no", 0.0)) > float(tol["accel_regression_tolerance_m_s2"]):
        failures.append("accel_regression")
    if float(deltas.get("contact_loss_delta_fb_minus_no", 0.0)) > float(
        tol["contact_loss_regression_tolerance_frames"]
    ):
        failures.append("contact_loss_regression")
    improvement_count = float(deltas.get("improvement_count", 0.0))
    if improvement_count < float(config["eligibility"]["min_improvement_count"]):
        failures.append("no_paired_advantage")
    return not failures, failures


def _segment_rows(
    root: Path,
    run_tag: str,
    cell: str,
    split: str,
    npz_path: Path,
    segment_id: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    segment_length = int(config["segment"]["length"])
    min_active = int(config["segment"]["min_active_frames"])
    thresholds = config["risk_thresholds"]
    contact_loss_count = float(thresholds["contact_loss_count"])
    slip_drop_z = abs(float(thresholds["slip_drop_z_m"]))
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
        sim_time = data["newton.panda.sim_time"]
        contact = data["newton.panda.rigid_contact_count"]
        body_q = data["newton.panda.object_body_q"]
        active = np.asarray(data["candidate.controller.feedback_active"]).reshape(-1)
        max_start_exclusive = min(len(sim_time), len(contact), len(body_q), len(active)) - 1
        active_indices = np.flatnonzero(active[:max_start_exclusive] > 0.5)
        if active_indices.size < min_active:
            return [], {
                "segment_id": segment_id,
                "status": "rejected",
                "failure": f"segment_active_frames_below_min:{int(active_indices.size)}",
            }
        center = int(np.median(active_indices))
        start = max(0, min(center - segment_length // 2, max_start_exclusive - segment_length))
        end = start + segment_length
        if start < 0 or end > max_start_exclusive:
            return [], {
                "segment_id": segment_id,
                "status": "rejected",
                "failure": "segment_window_out_of_bounds",
            }
        segment_active = int(np.count_nonzero(active[start:end] > 0.5))
        if segment_active < min_active:
            return [], {
                "segment_id": segment_id,
                "status": "rejected",
                "failure": f"segment_window_active_frames_below_min:{segment_active}",
            }
        rows: list[dict[str, Any]] = []
        for local_index, source_index in enumerate(range(start, end)):
            z_now = _object_z(body_q, source_index)
            z_next = _object_z(body_q, source_index + 1)
            t_now = _scalar(sim_time, source_index)
            t_next = _scalar(sim_time, source_index + 1, t_now)
            dt = max(t_next - t_now, 1.0e-6)
            contact_now = _scalar(contact, source_index)
            contact_next = _scalar(contact, source_index + 1)
            delta_z = z_next - z_now
            rows.append(
                {
                    "split": split,
                    "cell": cell,
                    "run_tag": segment_id,
                    "source_run_tag": run_tag,
                    "source_timestep_index": source_index,
                    "timestep_index": local_index,
                    "next_source_timestep_index": source_index + 1,
                    "newton.panda.sim_time": t_now,
                    "newton.panda.rigid_contact_count": contact_now,
                    "newton.object.body_q.z": z_now,
                    "candidate.controller.phase_index": _scalar(data["candidate.controller.phase_index"], source_index),
                    "candidate.controller.commanded_gripper_target": _scalar(
                        data["candidate.controller.commanded_gripper_target"], source_index
                    ),
                    "candidate.controller.commanded_lift_target": _scalar(
                        data["candidate.controller.commanded_lift_target"], source_index
                    ),
                    "candidate.task.object_mass_kg": _scalar(data["candidate.task.object_mass_kg"], 0),
                    "candidate.task.object_friction_mu": _scalar(data["candidate.task.object_friction_mu"], 0),
                    "candidate.modality.vision_available_mask": _scalar(
                        data["candidate.modality.vision_available_mask"], source_index, 1.0
                    ),
                    "candidate.modality.contact_available_mask": _scalar(
                        data["candidate.modality.contact_available_mask"], source_index, 1.0
                    ),
                    "candidate.controller.feedback_active": _scalar(
                        data["candidate.controller.feedback_active"], source_index
                    ),
                    "candidate.controller.feedback_lift_velocity_scale": _scalar(
                        data["candidate.controller.feedback_lift_velocity_scale"], source_index, 1.0
                    ),
                    "candidate.controller.feedback_hold_height_offset_m": _scalar(
                        data["candidate.controller.feedback_hold_height_offset_m"], source_index, 0.0
                    ),
                    "candidate.controller.feedback_stabilization_extension_s": _scalar(
                        data["candidate.controller.feedback_stabilization_extension_s"], source_index, 0.0
                    ),
                    "target.object.delta_z_next": delta_z,
                    "target.object.velocity_z_next": delta_z / dt,
                    "target.contact.count_next": contact_next,
                    "target.contact.delta_count_next": contact_next - contact_now,
                    "target.contact_loss_risk_next": 1.0 if contact_next <= contact_loss_count else 0.0,
                    "target.slip_risk_next": 1.0 if delta_z < -slip_drop_z else 0.0,
                }
            )
    return rows, {
        "segment_id": segment_id,
        "status": "admitted",
        "source_start_timestep": start,
        "source_end_timestep_exclusive": end,
        "segment_length": segment_length,
        "segment_active_frames": segment_active,
    }


def build(root: Path, config_path: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    fresh_sanity = _load_json(fresh_sanity_json)
    processed_dir = root / config["processed_dir"]
    manifest_path = root / config["manifest"]
    report_path = root / config["report"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if fresh_sanity.get("status") != "pass":
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")

    held_out = set(config["held_out_cells_reserved_for_evaluation"])
    validation_cells = set(config["segment"]["validation_cells"])
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    admitted_segments: list[dict[str, Any]] = []
    rejected_segments: list[dict[str, Any]] = []
    selected_source_keys: set[tuple[str, str]] = set()

    for source_manifest_name in config["source_manifests"]:
        source_manifest_path = root / source_manifest_name
        source_manifest = _load_json(source_manifest_path)
        if source_manifest.get("generated_trex_fields") != []:
            failures.append(f"source_generated_trex_fields_not_empty:{source_manifest_name}")
        if source_manifest.get("schema_promotion") != "blocked":
            failures.append(f"source_schema_promotion:{source_manifest_name}:{source_manifest.get('schema_promotion')}")
        for source_list_name in ("admitted_sources", "rejected_sources"):
            for record in source_manifest.get(source_list_name, []):
                cell = str(record.get("cell", ""))
                run_tag = str(record.get("feedback_run_tag", ""))
                source_key = (source_manifest_name, run_tag)
                if source_key in selected_source_keys:
                    continue
                if cell in held_out or cell.startswith("heldout_"):
                    rejected_segments.append({"cell": cell, "source_run_tag": run_tag, "failure": "held_out_cell"})
                    continue
                if not run_tag:
                    rejected_segments.append({"cell": cell, "source_run_tag": run_tag, "failure": "missing_feedback_run_tag"})
                    continue
                if source_list_name == "rejected_sources" and not _source_failures_allowed(record, config):
                    rejected_segments.append(
                        {
                            "cell": cell,
                            "source_run_tag": run_tag,
                            "failure": "source_failures_not_allowed",
                            "source_failures": record.get("failures", []),
                        }
                    )
                    continue
                safe, safety_failures = _paired_deltas_safe(record, config)
                if not safe:
                    rejected_segments.append(
                        {
                            "cell": cell,
                            "source_run_tag": run_tag,
                            "failure": "paired_deltas_not_safe",
                            "paired_delta_failures": safety_failures,
                            "paired_metric_deltas": record.get("paired_metric_deltas", {}),
                        }
                    )
                    continue
                npz_path = root / str(record.get("feedback_npz", ""))
                if not npz_path.exists():
                    rejected_segments.append(
                        {"cell": cell, "source_run_tag": run_tag, "failure": f"missing_feedback_npz:{_rel(npz_path, root)}"}
                    )
                    continue
                split = "validation" if cell in validation_cells else "train"
                segment_id = f"{config['segment']['run_tag_prefix']}_{cell}_{len(admitted_segments):02d}"
                try:
                    rows, segment_record = _segment_rows(root, run_tag, cell, split, npz_path, segment_id, config)
                except Exception as exc:
                    rows = []
                    segment_record = {
                        "segment_id": segment_id,
                        "status": "rejected",
                        "failure": f"segment_build_failed:{type(exc).__name__}:{exc}",
                    }
                segment_record.update(
                    {
                        "cell": cell,
                        "split": split,
                        "source_manifest": _rel(source_manifest_path, root),
                        "source_run_tag": run_tag,
                        "feedback_npz": _rel(npz_path, root),
                        "source_status": source_list_name,
                        "source_failures": record.get("failures", []),
                        "paired_metric_deltas": record.get("paired_metric_deltas", {}),
                    }
                )
                if rows:
                    selected_source_keys.add(source_key)
                    rows_by_split[split].extend(rows)
                    admitted_segments.append(segment_record)
                else:
                    rejected_segments.append(segment_record)

    train_segment_count = sum(1 for item in admitted_segments if item["split"] == "train")
    validation_segment_count = sum(1 for item in admitted_segments if item["split"] == "validation")
    train_active_count = sum(1 for row in rows_by_split["train"] if float(row["candidate.controller.feedback_active"]) > 0.5)
    validation_active_count = sum(
        1 for row in rows_by_split["validation"] if float(row["candidate.controller.feedback_active"]) > 0.5
    )
    if train_segment_count < int(config["gate"]["min_train_segments"]):
        failures.append(f"train_segments_below_min:{train_segment_count}")
    if validation_segment_count < int(config["gate"]["min_validation_segments"]):
        failures.append(f"validation_segments_below_min:{validation_segment_count}")
    if not rows_by_split["train"]:
        failures.append("no_train_rows")
    if not rows_by_split["validation"]:
        failures.append("no_validation_rows")
    if train_active_count < int(config["gate"]["min_train_active_frames"]):
        failures.append(f"train_active_frames_below_min:{train_active_count}")
    if validation_active_count < int(config["gate"]["min_validation_active_frames"]):
        failures.append(f"validation_active_frames_below_min:{validation_active_count}")

    fieldnames = [
        "split",
        "cell",
        "run_tag",
        "source_run_tag",
        "source_timestep_index",
        "timestep_index",
        "next_source_timestep_index",
        *config["feature_columns"],
        *config["residual_target_columns"],
        *config["lp_target_columns"],
    ]
    train_csv = processed_dir / "train.csv"
    validation_csv = processed_dir / "validation.csv"
    _write_csv(train_csv, fieldnames, rows_by_split["train"])
    _write_csv(validation_csv, fieldnames, rows_by_split["validation"])

    manifest = {
        "classification": "phase01_local_advantage_segment_preflight_v1",
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
        "train_active_feedback_count": train_active_count,
        "validation_active_feedback_count": validation_active_count,
        "train_segment_count": train_segment_count,
        "validation_segment_count": validation_segment_count,
        "admitted_segments": admitted_segments,
        "rejected_segments": rejected_segments,
        "held_out_excluded_from_training": sorted(held_out),
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "forbidden_claims": config["forbidden_claims"],
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Phase 01 Local-Advantage Segment Preflight",
        "",
        f"- status: `{manifest['status']}`",
        f"- train segments: `{train_segment_count}`",
        f"- validation segments: `{validation_segment_count}`",
        f"- train rows: `{manifest['train_row_count']}`",
        f"- validation rows: `{manifest['validation_row_count']}`",
        f"- train active feedback labels: `{train_active_count}`",
        f"- validation active feedback labels: `{validation_active_count}`",
        f"- train csv: `{manifest['train_csv']}`",
        f"- validation csv: `{manifest['validation_csv']}`",
        "",
        "This is a local source/objective repair preflight only. It is not training and not curiosity success evidence.",
    ]
    if admitted_segments:
        lines.extend(["", "## Admitted Segments", ""])
        lines.extend(
            f"- `{item['segment_id']}` from `{item['source_run_tag']}` cell `{item['cell']}` split `{item['split']}` active `{item['segment_active_frames']}`"
            for item in admitted_segments
        )
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
