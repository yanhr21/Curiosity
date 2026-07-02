"""Audit the Phase07 hard-training evidence gate.

This is a lightweight evidence classifier. It reads existing summaries,
metrics, visual evidence, action-bridge fields, and mainstream-comparison audit
artifacts. It does not train, preprocess datasets, render videos, run
inference, download checkpoints, or claim success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


HELD_OUT_CELLS = [
    "empty_high_misleading",
    "full_low_hidden",
    "three_quarter_low_misleading",
]

REQUIRED_METHODS = {
    "no_adaptation": {
        "role": "baseline",
        "run_tags": {
            "empty_high_misleading": "phase07_eval_empty_high_misleading_no_adaptation_rerun_20260627",
            "full_low_hidden": "phase07_eval_full_low_hidden_no_adaptation_20260627",
            "three_quarter_low_misleading": "phase07_eval_three_quarter_low_misleading_no_adaptation_20260627",
        },
    },
    "scripted_feedback": {
        "role": "baseline",
        "run_tags": {
            cell: f"phase07_eval_{cell}_scripted_feedback_20260627" for cell in HELD_OUT_CELLS
        },
    },
    "residual_baseline": {
        "role": "baseline",
        "run_tags": {
            cell: f"phase07_eval_{cell}_residual_baseline_20260627" for cell in HELD_OUT_CELLS
        },
    },
    "curiosity_weighted": {
        "role": "candidate_curiosity",
        "run_tags": {
            cell: f"phase07_eval_{cell}_curiosity_weighted_20260627" for cell in HELD_OUT_CELLS
        },
    },
    "random_intrinsic": {
        "role": "ablation",
        "run_tags": {
            "empty_high_misleading": "phase07_eval_empty_high_misleading_random_intrinsic_rerun_20260627",
            "full_low_hidden": "phase07_eval_full_low_hidden_random_intrinsic_20260627",
            "three_quarter_low_misleading": "phase07_eval_three_quarter_low_misleading_random_intrinsic_20260627",
        },
    },
    "object_only": {
        "role": "ablation",
        "run_tags": {
            cell: f"phase07_eval_{cell}_object_only_20260627" for cell in HELD_OUT_CELLS
        },
    },
    "contact_only": {
        "role": "ablation_pending_queue",
        "run_tags": {
            cell: f"phase07_eval_{cell}_contact_only_20260627" for cell in HELD_OUT_CELLS
        },
    },
    "shuffled_contact": {
        "role": "ablation_pending_queue",
        "run_tags": {
            cell: f"phase07_eval_{cell}_shuffled_contact_20260627" for cell in HELD_OUT_CELLS
        },
    },
    "delayed_contact": {
        "role": "ablation_pending_queue",
        "run_tags": {
            cell: f"phase07_eval_{cell}_delayed_contact_20260627" for cell in HELD_OUT_CELLS
        },
    },
    "no_learning_progress": {
        "role": "ablation_pending_queue",
        "run_tags": {
            cell: f"phase07_eval_{cell}_no_learning_progress_20260627" for cell in HELD_OUT_CELLS
        },
    },
}

TRAINING_SUMMARIES = {
    "residual_baseline": "experiments/outputs/phase07_residual_adapter_trainer_v1_20260627/phase07_residual_adapter_v1_train_20260627_summary.json",
    "curiosity_forward_model": "experiments/outputs/phase07_curiosity_forward_model_v1_20260627/phase07_curiosity_forward_model_v1_train_20260627_summary.json",
    "curiosity_weighted": "experiments/outputs/phase07_curiosity_weighted_residual_adapter_trainer_v1_20260627/phase07_curiosity_weighted_residual_adapter_v1_train_20260627_summary.json",
    "random_intrinsic": "experiments/outputs/phase07_random_intrinsic_residual_adapter_trainer_v1_20260627/phase07_random_intrinsic_residual_adapter_v1_train_20260627_summary.json",
    "object_only": "experiments/outputs/phase07_object_only_residual_adapter_trainer_v1_20260627/phase07_object_only_residual_adapter_v1_train_20260627_summary.json",
    "contact_only": "experiments/outputs/phase07_contact_only_residual_adapter_trainer_v1_20260627/phase07_contact_only_residual_adapter_v1_train_retry_20260627_summary.json",
    "shuffled_contact": "experiments/outputs/phase07_shuffled_contact_residual_adapter_trainer_v1_20260627/phase07_shuffled_contact_residual_adapter_v1_train_20260627_summary.json",
    "delayed_contact": "experiments/outputs/phase07_delayed_contact_residual_adapter_trainer_v1_20260627/phase07_delayed_contact_residual_adapter_v1_train_20260627_summary.json",
    "no_learning_progress": "experiments/outputs/phase07_no_learning_progress_residual_adapter_trainer_v1_20260627/phase07_no_learning_progress_residual_adapter_v1_train_20260627_summary.json",
}

ACTION_FIELDS = [
    "candidate.action.eef_delta_x",
    "candidate.action.eef_delta_y",
    "candidate.action.eef_delta_z",
    "candidate.action.eef_delta_roll",
    "candidate.action.eef_delta_pitch",
    "candidate.action.eef_delta_yaw",
    "candidate.action.gripper",
    "candidate.action.eef_delta_xyzrpy_gripper",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _truthy_pass(status: Any) -> bool:
    if not isinstance(status, str):
        return False
    return status == "pass" or status.startswith("pass_")


def _metric_row(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics or metrics.get("status") != "pass":
        return None
    rows = metrics.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0]
    return row if isinstance(row, dict) else None


def _score_tuple(row: dict[str, Any] | None) -> tuple[float, float, float, float, float, float]:
    if row is None:
        return (-1.0, -1.0, -1.0, -1e9, -1e9, -1e9)
    success = 1.0 if row.get("status") == "success" and row.get("object_not_dropped") is True else 0.0
    return (
        success,
        float(row.get("hold_duration_s") or 0.0),
        float(row.get("lift_height_m") or 0.0),
        -float(row.get("max_slip_m") or 1e9),
        -float(row.get("contact_loss_frames") or 1e9),
        -float(row.get("max_object_accel_m_s2") or 1e9),
    )


def _summarize_training(root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, rel_path in TRAINING_SUMMARIES.items():
        path = root / rel_path
        summary = _load_json(path)
        failures: list[str] = []
        if summary is None:
            failures.append("missing_summary")
        else:
            if summary.get("status") != "pass":
                failures.append(f"status={summary.get('status')}")
            if summary.get("run_mode") != "train":
                failures.append(f"run_mode={summary.get('run_mode')}")
            if summary.get("real_training_result") is not True:
                failures.append("real_training_result_not_true")
            if summary.get("checkpoint_written") is not True:
                failures.append("checkpoint_not_written")
            checkpoint = summary.get("checkpoint_path")
            if checkpoint and not (root / str(checkpoint)).is_file():
                failures.append(f"missing_checkpoint={checkpoint}")
        results[name] = {
            "summary": rel_path,
            "exists": path.is_file(),
            "status": "pass" if not failures else "fail",
            "failures": failures,
        }
    return results


def _backfill_bridge_by_run_tag(root: Path) -> dict[str, dict[str, Any]]:
    manifest = _load_json(root / "experiments/outputs/phase07_action_bridge_backfill_v1_20260627/manifest.json")
    if not manifest or not isinstance(manifest.get("results"), list):
        return {}
    by_tag: dict[str, dict[str, Any]] = {}
    for item in manifest["results"]:
        if not isinstance(item, dict):
            continue
        run_tag = item.get("run_tag")
        output_npz = item.get("output_npz")
        if not isinstance(run_tag, str) or not isinstance(output_npz, str):
            continue
        output_path = root / output_npz
        by_tag[run_tag] = {
            "status": item.get("status"),
            "output_npz": output_npz,
            "exists": output_path.is_file(),
            "missing_action_fields": item.get("missing_action_fields", ACTION_FIELDS),
        }
    return by_tag


def _summarize_eval(root: Path, min_video_frames: int) -> dict[str, Any]:
    backfill_bridge = _backfill_bridge_by_run_tag(root)
    result: dict[str, Any] = {}
    for method, spec in REQUIRED_METHODS.items():
        method_cells: dict[str, Any] = {}
        for cell, run_tag in spec["run_tags"].items():
            metrics_path = root / "experiments" / "outputs" / f"{run_tag}_metrics.json"
            summary_path = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
            visual_path = root / "experiments" / "outputs" / f"{run_tag}_visual_validation.json"
            manual_path = root / "experiments" / "outputs" / f"{run_tag}_manual_visual_inspection.json"
            npz_path = root / "experiments" / "outputs" / f"{run_tag}.npz"
            metrics = _load_json(metrics_path)
            summary = _load_json(summary_path)
            visual = _load_json(visual_path)
            manual = _load_json(manual_path)
            row = _metric_row(metrics)
            failures: list[str] = []
            if row is None:
                failures.append("missing_or_failed_metrics")
            video_export = summary.get("video_export") if isinstance(summary, dict) else None
            video_path = None
            video_frame_count = None
            if isinstance(video_export, dict):
                video_path = video_export.get("path")
                video_frame_count = video_export.get("frame_count")
            if not summary or summary.get("status") != "pass":
                failures.append("missing_or_failed_summary")
            if not isinstance(video_export, dict) or video_export.get("status") != "pass":
                failures.append("missing_or_failed_full_video_export")
            elif int(video_frame_count or 0) < min_video_frames:
                failures.append(f"video_frame_count_lt_{min_video_frames}")
            elif video_path and not Path(str(video_path)).is_file():
                failures.append(f"missing_video_file={video_path}")
            if not visual or visual.get("status") != "pass":
                failures.append("missing_or_failed_visual_validation")
            if not manual or not _truthy_pass(manual.get("status")):
                failures.append("missing_or_failed_manual_visual_inspection")
            missing_action: list[str] = []
            action_bridge_source = "original_npz"
            if npz_path.is_file():
                try:
                    data = np.load(npz_path)
                    missing_action = [field for field in ACTION_FIELDS if field not in data.files]
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"npz_action_bridge_read_error={exc}")
                    missing_action = ACTION_FIELDS.copy()
            else:
                missing_action = ACTION_FIELDS.copy()
                failures.append("missing_npz")
            if missing_action:
                backfilled = backfill_bridge.get(run_tag)
                if (
                    backfilled
                    and backfilled.get("status") == "pass"
                    and backfilled.get("exists") is True
                    and not backfilled.get("missing_action_fields")
                ):
                    missing_action = []
                    action_bridge_source = "backfilled_npz"
                else:
                    failures.append("missing_candidate_action_bridge_fields")
            method_cells[cell] = {
                "run_tag": run_tag,
                "role": spec["role"],
                "metrics": _rel(root, metrics_path),
                "summary": _rel(root, summary_path),
                "visual_validation": _rel(root, visual_path),
                "manual_visual_inspection": _rel(root, manual_path),
                "npz": _rel(root, npz_path),
                "status": "pass" if not failures else "fail",
                "failures": failures,
                "metric_row": row,
                "score_tuple": list(_score_tuple(row)),
                "video_path": video_path,
                "video_frame_count": video_frame_count,
                "action_bridge_source": action_bridge_source,
                "backfilled_action_bridge_npz": backfill_bridge.get(run_tag, {}).get("output_npz"),
                "missing_action_bridge_fields": missing_action,
            }
        result[method] = {
            "role": spec["role"],
            "status": "pass" if all(cell["status"] == "pass" for cell in method_cells.values()) else "fail",
            "cells": method_cells,
        }
    return result


def _compare_curiosity(eval_summary: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    per_cell_pass: list[bool] = []
    for cell in HELD_OUT_CELLS:
        baseline_rows = []
        for method in ["no_adaptation", "scripted_feedback", "residual_baseline"]:
            row = eval_summary[method]["cells"][cell].get("metric_row")
            baseline_rows.append((method, row, _score_tuple(row)))
        strongest_method, strongest_row, strongest_score = max(baseline_rows, key=lambda item: item[2])
        curiosity_row = eval_summary["curiosity_weighted"]["cells"][cell].get("metric_row")
        curiosity_score = _score_tuple(curiosity_row)
        safety_regressions: list[str] = []
        if curiosity_row is None or strongest_row is None:
            beats = False
            safety_ok = False
            safety_regressions.append("missing_metrics_for_comparison")
        else:
            if float(curiosity_row.get("max_slip_m") or 0.0) > float(strongest_row.get("max_slip_m") or 0.0):
                safety_regressions.append("max_slip_m_regression")
            if float(curiosity_row.get("contact_loss_frames") or 0.0) > float(strongest_row.get("contact_loss_frames") or 0.0):
                safety_regressions.append("contact_loss_frames_regression")
            if float(curiosity_row.get("drop_height_loss_m") or 0.0) > float(strongest_row.get("drop_height_loss_m") or 0.0):
                safety_regressions.append("drop_height_loss_m_regression")
            if float(curiosity_row.get("max_object_accel_m_s2") or 0.0) > float(strongest_row.get("max_object_accel_m_s2") or 0.0):
                safety_regressions.append("max_object_accel_m_s2_regression")
            beats = curiosity_score > strongest_score
            safety_ok = not safety_regressions
        per_cell_pass.append(beats and safety_ok)
        comparisons[cell] = {
            "strongest_baseline": strongest_method,
            "strongest_baseline_score_tuple": list(strongest_score),
            "curiosity_score_tuple": list(curiosity_score),
            "curiosity_beats_strongest_baseline": beats,
            "safety_ok_against_strongest_baseline": safety_ok,
            "safety_regressions": safety_regressions,
        }
    return {
        "status": "pass" if all(per_cell_pass) else "fail",
        "all_cells_curiosity_beats_strongest_baseline_without_safety_regression": all(per_cell_pass),
        "cells": comparisons,
    }


def _mainstream_status(root: Path) -> dict[str, Any]:
    audit = _load_json(root / "experiments/outputs/phase07_mainstream_comparison_audit_v1_20260627.json")
    bridge_preflight = _load_json(
        root / "experiments/outputs/phase07_mainstream_adapter_conversion_preflight_v1_20260627/manifest.json"
    )
    backfill_manifest = _load_json(root / "experiments/outputs/phase07_action_bridge_backfill_v1_20260627/manifest.json")
    stage1_index = _load_json(
        root / "experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627/manifest.json"
    )
    stage1_leakage = _load_json(root / "experiments/outputs/phase07_stage1_no_heldout_leakage_v1_20260627.json")
    readiness = _load_json(root / "experiments/outputs/phase07_official_method_readiness_v1_20260627.json")
    heldout_comparison = _load_json(root / "experiments/outputs/phase07_heldout_comparison_report_v1_20260627.json")
    return {
        "comparison_audit_status": audit.get("status") if audit else "missing",
        "comparison_gate_satisfied": bool(audit and audit.get("completion_impact", {}).get("phase07_mainstream_gate_satisfied")),
        "action_bridge_backfill_status": backfill_manifest.get("status") if backfill_manifest else "missing",
        "adapter_conversion_preflight_status": bridge_preflight.get("status") if bridge_preflight else "missing",
        "stage1_dataset_index_status": stage1_index.get("status") if stage1_index else "missing",
        "stage1_dataset_index_episode_count": stage1_index.get("episode_count") if stage1_index else None,
        "stage1_no_heldout_leakage_status": stage1_leakage.get("status") if stage1_leakage else "missing",
        "stage1_no_heldout_leakage_proven": stage1_leakage.get("no_held_out_leakage_proven") if stage1_leakage else False,
        "official_method_readiness_status": readiness.get("status") if readiness else "missing",
        "official_method_comparison_ready": readiness.get("official_method_comparison_ready") if readiness else False,
        "heldout_comparison_status": heldout_comparison.get("status") if heldout_comparison else "missing",
        "heldout_curiosity_beats_strongest_baselines": heldout_comparison.get(
            "curiosity_beats_all_strongest_baselines_without_safety_regression"
        )
        if heldout_comparison
        else False,
        "status": "pass"
        if audit
        and audit.get("completion_impact", {}).get("phase07_mainstream_gate_satisfied") is True
        and stage1_leakage
        and stage1_leakage.get("no_held_out_leakage_proven") is True
        and readiness
        and readiness.get("official_method_comparison_ready") is True
        else "open_not_satisfied",
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase07 Hard-Training Evidence Gate V1",
        "",
        "Date: 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This audit does not train, preprocess datasets, render, run inference, download checkpoints, or claim success.",
        "",
        "## Gate Result",
        "",
        f"- final curiosity success allowed: `{payload['final_curiosity_success_allowed']}`",
        f"- result classification: `{payload['result_classification']}`",
        f"- training evidence: `{payload['training_evidence_status']}`",
        f"- evaluation evidence: `{payload['evaluation_evidence_status']}`",
        f"- curiosity vs strongest baseline: `{payload['curiosity_comparison']['status']}`",
        f"- mainstream comparison: `{payload['mainstream_status']['status']}`",
        f"- mainstream stage-1 dataset index: `{payload['mainstream_status']['stage1_dataset_index_status']}`",
        f"- stage-1 no held-out leakage: `{payload['mainstream_status']['stage1_no_heldout_leakage_status']}`",
        f"- official method readiness: `{payload['mainstream_status']['official_method_readiness_status']}`",
        f"- held-out comparison report: `{payload['mainstream_status']['heldout_comparison_status']}`",
        "",
        "## Blocking / Open Items",
        "",
    ]
    for item in payload["open_items"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Required Next Action", "", payload["next_required_action"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/phase07_hard_training_evidence_gate_v1_20260627.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/reports/2026-06-27_phase07_hard_training_evidence_gate_v1.md"),
    )
    parser.add_argument("--min-video-frames", type=int, default=360)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    training = _summarize_training(root)
    eval_summary = _summarize_eval(root, args.min_video_frames)
    comparison = _compare_curiosity(eval_summary)
    mainstream = _mainstream_status(root)

    training_ok = all(item["status"] == "pass" for item in training.values())
    evaluation_ok = all(item["status"] == "pass" for item in eval_summary.values())
    mainstream_ok = mainstream["status"] == "pass"
    comparison_ok = comparison["status"] == "pass"
    final_allowed = training_ok and evaluation_ok and mainstream_ok and comparison_ok

    open_items: list[str] = []
    if not training_ok:
        failed = [name for name, item in training.items() if item["status"] != "pass"]
        open_items.append(f"missing_or_failed_training_summaries={failed}")
    if not evaluation_ok:
        failed_methods = [name for name, item in eval_summary.items() if item["status"] != "pass"]
        open_items.append(f"missing_or_failed_evaluation_evidence={failed_methods}")
    if not comparison_ok:
        open_items.append("curiosity_weighted_does_not_beat_strongest_declared_baseline_without_safety_regression")
    if mainstream.get("heldout_curiosity_beats_strongest_baselines") is not True:
        open_items.append("heldout_comparison_report_not_passing")
    if not mainstream_ok:
        open_items.append("serious_mainstream_or_official_checkpoint_comparison_gate_open")
    if mainstream.get("stage1_no_heldout_leakage_proven") is not True:
        open_items.append("stage1_no_heldout_leakage_gate_open")
    if mainstream.get("official_method_comparison_ready") is not True:
        open_items.append("official_method_readiness_gate_open")

    payload = {
        "classification": "phase07_hard_training_evidence_gate_v1",
        "status": "pass" if final_allowed else "open_not_satisfied",
        "result_classification": "complete_success" if final_allowed else "incomplete_or_negative_evidence",
        "final_curiosity_success_allowed": final_allowed,
        "not_training": True,
        "not_data_preprocessing": True,
        "not_rendering": True,
        "not_inference": True,
        "not_success_claim": True,
        "held_out_cells": HELD_OUT_CELLS,
        "training_evidence_status": "pass" if training_ok else "fail",
        "training_evidence": training,
        "evaluation_evidence_status": "pass" if evaluation_ok else "fail",
        "evaluation_evidence": eval_summary,
        "curiosity_comparison": comparison,
        "mainstream_status": mainstream,
        "open_items": open_items,
        "next_required_action": (
            "Continue with the queued Phase07 remaining ablations and faithful mainstream comparison work; "
            "do not call the curiosity objective complete until this gate passes."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
