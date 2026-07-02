"""Build a compute-gated residual-label source dataset manifest.

This validates promoted Newton scripted-feedback source candidates and writes
namespace-preserving CSV/JSON artifacts. It does not train a model and does not
promote data into T-Rex schema keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


FORBIDDEN_EXACT_FIELDS = {"action", "action_abs"}
FORBIDDEN_PREFIXES = ("observation.",)


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


def _status_is_pass(payload: dict[str, Any], allowed: set[str] | None = None) -> bool:
    status = str(payload.get("status", ""))
    return status in (allowed or {"pass"})


def _first_scalar(data: np.ndarray, index: int) -> float:
    arr = np.asarray(data[index])
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def _first_int(data: np.ndarray, index: int) -> int:
    return int(round(_first_scalar(data, index)))


def _object_z(data: np.ndarray, index: int) -> float:
    arr = np.asarray(data[index]).reshape(-1)
    if arr.size < 3:
        return 0.0
    return float(arr[2])


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value)
    summary: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
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


def _forbidden_fields(fields: list[str]) -> list[str]:
    failures: list[str] = []
    for field in fields:
        if field in FORBIDDEN_EXACT_FIELDS or any(field.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            failures.append(f"forbidden_field:{field}")
    return failures


def _path_from_source(root: Path, source: dict[str, Any], key: str) -> Path:
    outputs = source.get("outputs", {})
    value = outputs.get(key)
    if not isinstance(value, str) or not value:
        raise KeyError(f"missing source output key {key}")
    return root / value


def _nested_get(payload: dict[str, Any], dotted_key: str, default: Any = "") -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def build(config_path: Path, root: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    source_manifest_path = root / config["source_manifest"]
    source_manifest = _load_json(source_manifest_path)
    fresh_sanity = _load_json(fresh_sanity_json)

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if source_manifest.get("status") != config["source_status_required"]:
        failures.append(f"source_manifest_status:{source_manifest.get('status')}")
    if source_manifest.get("generated_trex_fields") != []:
        failures.append("source_manifest_generated_trex_fields_not_empty")
    if source_manifest.get("schema_promotion") != "blocked":
        failures.append(f"source_manifest_schema_promotion:{source_manifest.get('schema_promotion')}")
    if not _status_is_pass(fresh_sanity):
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")

    held_out = set(config["held_out_cells_reserved_for_evaluation"])
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "residual_label_records.csv"
    manifest_path = output_dir / "manifest.json"

    rows: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    all_fields: set[str] = set()
    total_feedback_triggers = 0
    total_feedback_active_frames = 0
    contact_min: int | None = None
    contact_max: int | None = None

    for source in source_manifest.get("source_candidates", []):
        run_tag = source.get("run_tag", "unknown")
        cell = str(source.get("cell", ""))
        source_static_values: dict[str, Any] = {}
        for field in config.get("source_static_fields", []):
            output_name = field["name"]
            source_key = field["source_key"]
            source_static_values[output_name] = _nested_get(source, source_key, field.get("default", ""))
        if cell in held_out or source.get("held_out_generalization_cell"):
            failures.append(f"held_out_source_candidate_used:{run_tag}:{cell}")
        if source.get("status") != config["required_source_status"]:
            failures.append(f"source_status_not_promoted:{run_tag}:{source.get('status')}")
        allowed_decisions = set(config.get("allowed_candidate_decisions", []))
        if not allowed_decisions and "required_candidate_decision" in config:
            allowed_decisions.add(config["required_candidate_decision"])
        if source.get("promotion_decision") not in allowed_decisions:
            failures.append(f"source_promotion_decision_mismatch:{run_tag}:{source.get('promotion_decision')}")

        paths: dict[str, Path] = {}
        for key in config["required_source_gates"]:
            try:
                paths[key] = _path_from_source(root, source, key)
            except KeyError as exc:
                failures.append(f"{run_tag}:{exc}")
                continue
            if not paths[key].is_file():
                failures.append(f"missing_source_gate:{run_tag}:{key}:{_rel(paths[key], root)}")

        if any(not path.is_file() for path in paths.values()):
            continue

        sanity = _load_json(paths["fresh_official_newton_sanity"])
        visual = _load_json(paths["visual_validation"])
        manual = _load_json(paths["manual_visual_inspection"])
        metrics = _load_json(paths["metrics_json"])
        accel_peak = _load_json(paths["accel_peak_analysis"])

        if not _status_is_pass(sanity):
            failures.append(f"source_fresh_sanity_not_pass:{run_tag}:{sanity.get('status')}")
        if not _status_is_pass(visual):
            failures.append(f"source_visual_validation_not_pass:{run_tag}:{visual.get('status')}")
        if not _status_is_pass(manual, {"pass", "pass_nonblank_success_with_feedback"}):
            failures.append(f"source_manual_visual_not_pass:{run_tag}:{manual.get('status')}")
        if not _status_is_pass(metrics):
            failures.append(f"source_metrics_not_pass:{run_tag}:{metrics.get('status')}")
        if not _status_is_pass(accel_peak):
            failures.append(f"source_accel_peak_not_pass:{run_tag}:{accel_peak.get('status')}")
        if metrics.get("generated_trex_fields") != []:
            failures.append(f"source_metrics_generated_trex_fields_not_empty:{run_tag}")
        if metrics.get("schema_promotion") != "blocked":
            failures.append(f"source_metrics_schema_promotion:{run_tag}:{metrics.get('schema_promotion')}")

        with np.load(paths["npz"], allow_pickle=False) as data:
            fields = sorted(data.files)
            all_fields.update(fields)
            failures.extend(f"{run_tag}:{failure}" for failure in _forbidden_fields(fields))
            missing = [field for field in config["required_npz_fields"] if field not in data]
            for field in missing:
                failures.append(f"missing_required_npz_field:{run_tag}:{field}")
            if missing:
                continue

            steps = np.asarray(data["newton.panda.step"])
            timesteps = int(steps.shape[0])
            contact = np.asarray(data["newton.panda.rigid_contact_count"])
            if contact.shape[0] != timesteps:
                failures.append(f"contact_length_mismatch:{run_tag}")
                continue
            contact_values = contact.reshape(timesteps, -1)
            run_contact_min = int(np.min(contact_values))
            run_contact_max = int(np.max(contact_values))
            contact_min = run_contact_min if contact_min is None else min(contact_min, run_contact_min)
            contact_max = run_contact_max if contact_max is None else max(contact_max, run_contact_max)

            feedback_trigger = np.asarray(data["candidate.controller.feedback_trigger_count"])
            feedback_active = np.asarray(data["candidate.controller.feedback_active"])
            run_feedback_triggers = int(np.max(feedback_trigger)) if feedback_trigger.size else 0
            run_active_frames = int(np.count_nonzero(feedback_active))
            if run_feedback_triggers <= 0:
                failures.append(f"feedback_trigger_count_zero:{run_tag}")
            if run_active_frames <= 0:
                failures.append(f"feedback_active_frames_zero:{run_tag}")
            total_feedback_triggers += run_feedback_triggers
            total_feedback_active_frames += run_active_frames

            arrays = {
                "sim_time": np.asarray(data["newton.panda.sim_time"]),
                "object_q": np.asarray(data["newton.panda.object_body_q"]),
                "phase": np.asarray(data["candidate.controller.phase_index"]),
                "feedback_active": feedback_active,
                "feedback_trigger": feedback_trigger,
                "feedback_lift_velocity_scale": np.asarray(data["candidate.controller.feedback_lift_velocity_scale"]),
                "feedback_hold_height_offset_m": np.asarray(data["candidate.controller.feedback_hold_height_offset_m"]),
                "feedback_stabilization_extension_s": np.asarray(
                    data["candidate.controller.feedback_stabilization_extension_s"]
                ),
                "commanded_gripper_target": np.asarray(data["candidate.controller.commanded_gripper_target"]),
                "commanded_lift_target": np.asarray(data["candidate.controller.commanded_lift_target"]),
            }

            for idx in range(timesteps):
                rows.append(
                    {
                        "run_tag": run_tag,
                        "source_name": source.get("name", ""),
                        "cell": cell,
                        "held_out_generalization_cell": "false",
                        "timestep_index": idx,
                        "newton.panda.step": _first_int(steps, idx),
                        "newton.panda.sim_time": _first_scalar(arrays["sim_time"], idx),
                        "newton.contact.rigid_contact_count": _first_int(contact, idx),
                        "newton.object.body_q.z": _object_z(arrays["object_q"], idx),
                        "candidate.controller.phase_index": _first_int(arrays["phase"], idx),
                        "candidate.controller.feedback_active": _first_int(arrays["feedback_active"], idx),
                        "candidate.controller.feedback_trigger_count": _first_int(arrays["feedback_trigger"], idx),
                        "candidate.controller.feedback_lift_velocity_scale": _first_scalar(
                            arrays["feedback_lift_velocity_scale"], idx
                        ),
                        "candidate.controller.feedback_hold_height_offset_m": _first_scalar(
                            arrays["feedback_hold_height_offset_m"], idx
                        ),
                        "candidate.controller.feedback_stabilization_extension_s": _first_scalar(
                            arrays["feedback_stabilization_extension_s"], idx
                        ),
                        "candidate.controller.commanded_gripper_target": _first_scalar(
                            arrays["commanded_gripper_target"], idx
                        ),
                        "candidate.controller.commanded_lift_target": _first_scalar(
                            arrays["commanded_lift_target"], idx
                        ),
                        **source_static_values,
                    }
                )

            source_runs.append(
                {
                    "run_tag": run_tag,
                    "source_name": source.get("name", ""),
                    "cell": cell,
                    "held_out_generalization_cell": False,
                    "pre_record_warmup_steps": source.get("pre_record_warmup_steps"),
                    "status": "pass" if run_feedback_triggers > 0 and run_active_frames > 0 else "fail",
                    "npz": _rel(paths["npz"], root),
                    "fresh_official_newton_sanity": _rel(paths["fresh_official_newton_sanity"], root),
                    "visual_validation": _rel(paths["visual_validation"], root),
                    "manual_visual_inspection": _rel(paths["manual_visual_inspection"], root),
                    "metrics_json": _rel(paths["metrics_json"], root),
                    "accel_peak_analysis": _rel(paths["accel_peak_analysis"], root),
                    "contact_sheet": source.get("outputs", {}).get("contact_sheet", ""),
                    "frame_browser": source.get("outputs", {}).get("frame_browser", ""),
                    "metrics_observed": metrics,
                    "observed": source.get("observed", {}),
                    "source_static_values": source_static_values,
                    "array_summaries": {
                        field: _array_summary(np.asarray(data[field])) for field in config["required_npz_fields"]
                    },
                    "record_count": timesteps,
                    "contact_count_min": run_contact_min,
                    "contact_count_max": run_contact_max,
                    "feedback_trigger_count": run_feedback_triggers,
                    "feedback_active_frames": run_active_frames,
                }
            )

    dataset_fields = list(config["dataset_fields"])
    static_dataset_fields = [field["name"] for field in config.get("source_static_fields", [])]
    dataset_fields.extend(static_dataset_fields)
    failures.extend(_forbidden_fields(dataset_fields))
    if not rows:
        failures.append("no_records_written")
    if total_feedback_triggers <= 0:
        failures.append("total_feedback_trigger_count_zero")

    fieldnames = ["run_tag", "source_name", "cell", "held_out_generalization_cell", "timestep_index", *dataset_fields]
    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "classification": "residual_label_source_runner_manifest_v1_not_trex_schema",
        "status": "pass" if not failures else "fail",
        "phase": "04_closed_loop_adaptation",
        "config": _rel(config_path, root),
        "source_manifest": _rel(source_manifest_path, root),
        "source_manifest_status": source_manifest.get("status"),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "record_count": len(rows),
        "source_run_count": len(source_runs),
        "records_csv": _rel(records_path, root),
        "source_runs": source_runs,
        "dataset_fields": dataset_fields,
        "source_npz_fields": sorted(all_fields),
        "allowed_namespaces": config["allowed_namespaces"],
        "forbidden_generated_fields": config["forbidden_generated_fields"],
        "generated_trex_fields": [],
        "schema_promotion": "blocked",
        "training_started": False,
        "no_model_created": True,
        "no_placeholder_model": True,
        "held_out_cells_reserved_for_evaluation": list(held_out),
        "contact_count_min": contact_min,
        "contact_count_max": contact_max,
        "total_feedback_trigger_count": total_feedback_triggers,
        "total_feedback_active_frames": total_feedback_active_frames,
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
                "record_count": manifest["record_count"],
                "source_run_count": manifest["source_run_count"],
                "records_csv": manifest["records_csv"],
                "manifest": str(Path(manifest["records_csv"]).parent / "manifest.json"),
                "schema_promotion": manifest["schema_promotion"],
                "generated_trex_fields": manifest["generated_trex_fields"],
                "training_started": manifest["training_started"],
                "failures": manifest["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if manifest["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
