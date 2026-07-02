"""Build a closed-loop teacher-label preflight from on-policy Newton rollouts.

The source rollouts must be controlled by the current learned residual policy
while recording scripted corrective teacher labels under `candidate.teacher.*`.
This is DAgger-style data aggregation on the policy-induced state distribution,
not an offline replay success claim and not a T-Rex schema promotion.
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
    if arr.shape[0] == 1:
        item = np.asarray(arr[0])
    else:
        item = np.asarray(arr[index])
    if item.size == 0:
        return 0.0
    return float(item.reshape(-1)[0])


def _first_int(data: np.ndarray, index: int) -> int:
    return int(round(_first_scalar(data, index)))


def _object_z(data: np.ndarray, index: int) -> float:
    arr = np.asarray(data[index]).reshape(-1)
    return float(arr[2]) if arr.size >= 3 else 0.0


def build(config_path: Path, root: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    fresh_sanity = _load_json(fresh_sanity_json)
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train_closed_loop_teacher_records.csv"
    validation_csv = output_dir / "validation_closed_loop_teacher_records.csv"
    manifest_path = output_dir / "manifest.json"

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if fresh_sanity.get("status") != "pass":
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")

    held_out = set(config["held_out_cells_reserved_for_evaluation"])
    feature_columns = list(config["feature_columns"])
    target_columns = list(config["target_columns"])
    all_columns = [
        "run_tag",
        "source_name",
        "split",
        "cell",
        "held_out_generalization_cell",
        "timestep_index",
        *feature_columns,
        *target_columns,
        "candidate.controller.feedback_active",
        "candidate.controller.feedback_active_probability",
        "candidate.teacher.feedback_trigger_count",
    ]
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    source_summaries: list[dict[str, Any]] = []
    total_teacher_active = 0

    required_npz_fields = {
        "newton.panda.sim_time",
        "newton.panda.rigid_contact_count",
        "newton.panda.object_body_q",
        "candidate.controller.phase_index",
        "candidate.controller.commanded_gripper_target",
        "candidate.controller.commanded_lift_target",
        "candidate.controller.feedback_active",
        "candidate.controller.feedback_active_probability",
        *target_columns,
    }

    for source in config["source_rollouts"]:
        split = source["split"]
        cell = source["cell"]
        run_tag = source["run_tag"]
        if split not in rows_by_split:
            failures.append(f"invalid_split:{run_tag}:{split}")
            continue
        if cell in held_out or source.get("held_out_generalization_cell"):
            failures.append(f"held_out_source_candidate_used:{run_tag}:{cell}")
            continue
        summary_path = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
        run_status_path = root / "experiments" / "outputs" / f"{run_tag}_run_status.json"
        visual_validation_path = root / "experiments" / "outputs" / f"{run_tag}_visual_validation.json"
        if not summary_path.is_file():
            failures.append(f"missing_summary:{run_tag}")
            continue
        summary = _load_json(summary_path)
        run_status = _load_json(run_status_path) if run_status_path.is_file() else {}
        visual_validation = _load_json(visual_validation_path) if visual_validation_path.is_file() else {}
        if summary.get("status") != "pass":
            failures.append(f"summary_not_pass:{run_tag}:{summary.get('status')}")
        if run_status and run_status.get("status") != "pass_downstream_blocked":
            failures.append(f"run_status_not_pass_downstream_blocked:{run_tag}:{run_status.get('status')}")
        if visual_validation and visual_validation.get("status") != "pass":
            failures.append(f"visual_validation_not_pass:{run_tag}:{visual_validation.get('status')}")
        if summary.get("controller_mode") != "lift_hold_learned_residual":
            failures.append(f"not_learned_residual_controller:{run_tag}:{summary.get('controller_mode')}")
        if summary.get("scripted_teacher_labels", {}).get("enabled") is not True:
            failures.append(f"scripted_teacher_labels_not_enabled:{run_tag}")
        if summary.get("scripted_teacher_labels", {}).get("applied_to_controller") is not False:
            failures.append(f"scripted_teacher_labels_were_applied:{run_tag}")
        npz_path = Path(summary.get("npz", ""))
        if not npz_path.is_absolute():
            npz_path = root / npz_path
        if not npz_path.is_file():
            failures.append(f"missing_npz:{run_tag}:{_rel(npz_path, root)}")
            continue

        with np.load(npz_path, allow_pickle=False) as data:
            missing = sorted(field for field in required_npz_fields if field not in data.files)
            if missing:
                failures.append(f"missing_npz_fields:{run_tag}:{','.join(missing)}")
                continue
            timesteps = int(np.asarray(data["newton.panda.sim_time"]).shape[0])
            teacher_active = int(np.count_nonzero(np.asarray(data["candidate.teacher.feedback_active"])))
            total_teacher_active += teacher_active
            source_summaries.append(
                {
                    "run_tag": run_tag,
                    "split": split,
                    "cell": cell,
                    "summary": _rel(summary_path, root),
                    "run_status": _rel(run_status_path, root),
                    "visual_validation": _rel(visual_validation_path, root),
                    "npz": _rel(npz_path, root),
                    "teacher_active_frames": teacher_active,
                    "teacher_final_trigger_count": summary.get("scripted_teacher_labels", {}).get(
                        "final_trigger_count"
                    ),
                    "controller_final_trigger_count": summary.get("scripted_feedback", {}).get(
                        "final_trigger_count"
                    ),
                }
            )
            if teacher_active <= 0:
                failures.append(f"teacher_active_frames_zero:{run_tag}")
            for idx in range(timesteps):
                row = {
                    "run_tag": run_tag,
                    "source_name": source.get("name", run_tag),
                    "split": split,
                    "cell": cell,
                    "held_out_generalization_cell": "false",
                    "timestep_index": idx,
                    "newton.panda.sim_time": _first_scalar(data["newton.panda.sim_time"], idx),
                    "newton.contact.rigid_contact_count": _first_int(
                        data["newton.panda.rigid_contact_count"], idx
                    ),
                    "newton.object.body_q.z": _object_z(data["newton.panda.object_body_q"], idx),
                    "candidate.controller.phase_index": _first_int(data["candidate.controller.phase_index"], idx),
                    "candidate.controller.commanded_gripper_target": _first_scalar(
                        data["candidate.controller.commanded_gripper_target"], idx
                    ),
                    "candidate.controller.commanded_lift_target": _first_scalar(
                        data["candidate.controller.commanded_lift_target"], idx
                    ),
                    "candidate.task.object_mass_kg": _first_scalar(data["candidate.task.object_mass_kg"], idx),
                    "candidate.task.object_friction_mu": _first_scalar(data["candidate.task.object_friction_mu"], idx),
                    "candidate.task.nominal_visual_fill": _first_scalar(
                        data["candidate.task.nominal_visual_fill"], idx
                    ),
                    "candidate.teacher.feedback_active": _first_int(
                        data["candidate.teacher.feedback_active"], idx
                    ),
                    "candidate.teacher.feedback_lift_velocity_scale": _first_scalar(
                        data["candidate.teacher.feedback_lift_velocity_scale"], idx
                    ),
                    "candidate.teacher.feedback_hold_height_offset_m": _first_scalar(
                        data["candidate.teacher.feedback_hold_height_offset_m"], idx
                    ),
                    "candidate.teacher.feedback_stabilization_extension_s": _first_scalar(
                        data["candidate.teacher.feedback_stabilization_extension_s"], idx
                    ),
                    "candidate.controller.feedback_active": _first_int(
                        data["candidate.controller.feedback_active"], idx
                    ),
                    "candidate.controller.feedback_active_probability": _first_scalar(
                        data["candidate.controller.feedback_active_probability"], idx
                    ),
                    "candidate.teacher.feedback_trigger_count": _first_int(
                        data["candidate.teacher.feedback_trigger_count"], idx
                    ),
                }
                rows_by_split[split].append(row)

    for split, path in (("train", train_csv), ("validation", validation_csv)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_columns)
            writer.writeheader()
            writer.writerows(rows_by_split[split])

    if not rows_by_split["train"]:
        failures.append("no_train_rows")
    if not rows_by_split["validation"]:
        failures.append("no_validation_rows")
    if total_teacher_active <= 0:
        failures.append("total_teacher_active_frames_zero")

    manifest = {
        "classification": "phase07_closed_loop_teacher_preflight_v1",
        "status": "pass" if not failures else "fail",
        "purpose": "closed_loop_dagger_style_teacher_labels_on_policy_induced_newton_states",
        "config": _rel(config_path, root),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "train_csv": _rel(train_csv, root),
        "validation_csv": _rel(validation_csv, root),
        "train_record_count": len(rows_by_split["train"]),
        "validation_record_count": len(rows_by_split["validation"]),
        "source_run_count": len(source_summaries),
        "total_teacher_active_frames": total_teacher_active,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "held_out_cells_reserved_for_evaluation": sorted(held_out),
        "source_summaries": source_summaries,
        "real_closed_loop_source_distribution": True,
        "policy_updated": False,
        "training_started": False,
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "not_official_trex_method": True,
        "not_success_claim": True,
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
    print(json.dumps({"status": manifest["status"], "manifest": manifest}, indent=2, sort_keys=True))
    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
