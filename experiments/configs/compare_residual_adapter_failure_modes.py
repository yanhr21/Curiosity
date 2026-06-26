#!/usr/bin/env python3
"""Compare validated lift-hold rollout timing and failure modes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_1d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    while arr.ndim > 1:
        arr = arr[:, 0]
    return arr


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[0])


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_world(metrics: dict[str, Any]) -> dict[str, Any]:
    worlds = metrics.get("per_world") or []
    if worlds:
        return dict(worlds[0])
    return {}


def _extract_series(npz: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    if "newton.panda.sim_time" in npz:
        series["time"] = _as_1d(npz["newton.panda.sim_time"]).astype(float)
    elif "newton.panda.step" in npz:
        steps = _as_1d(npz["newton.panda.step"]).astype(float)
        series["time"] = steps / 60.0

    if "newton.panda.step" in npz:
        series["step"] = _as_1d(npz["newton.panda.step"]).astype(float)

    if "newton.camera.object_z" in npz:
        series["object_z"] = _as_1d(npz["newton.camera.object_z"]).astype(float)
    elif "newton.panda.object_body_q" in npz:
        body_q = np.asarray(npz["newton.panda.object_body_q"])
        if body_q.ndim >= 3:
            series["object_z"] = body_q[:, 0, 2].astype(float)

    for key in (
        "candidate.controller.feedback_active",
        "candidate.controller.feedback_active_probability",
        "candidate.controller.feedback_trigger_count",
        "candidate.controller.feedback_observed_object_accel_m_s2",
        "newton.panda.rigid_contact_count",
    ):
        if key in npz:
            series[key] = _as_1d(npz[key]).astype(float)
    return series


def _time_at(series: dict[str, np.ndarray], index: int | None) -> float | None:
    if index is None:
        return None
    time = series.get("time")
    if time is None or index >= len(time):
        return None
    return float(time[index])


def _step_at(series: dict[str, np.ndarray], index: int | None) -> int | None:
    if index is None:
        return None
    step = series.get("step")
    if step is not None and index < len(step):
        return int(step[index])
    return int(index)


def _analyze_run(root: Path, cfg_run: dict[str, Any], warmup_steps: int, lift_threshold_m: float) -> dict[str, Any]:
    run_tag = cfg_run["run_tag"]
    npz_path = root / "experiments" / "outputs" / f"{run_tag}.npz"
    metrics_path = root / "experiments" / "outputs" / f"{run_tag}_metrics.json"
    summary_path = root / "experiments" / "outputs" / f"{run_tag}_summary.json"

    metrics = _load_json(metrics_path)
    summary = _load_json(summary_path)
    metric_world = _metric_world(metrics)

    with np.load(npz_path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        series = _extract_series(npz)

    object_z = series.get("object_z")
    time = series.get("time")
    dt = None
    if time is not None and len(time) > 1:
        dt = float(np.median(np.diff(time)))
    if dt is None or dt <= 0:
        dt = _safe_float(metrics.get("dt_s")) or 1.0 / 60.0

    initial_z = float(object_z[0]) if object_z is not None and len(object_z) else _safe_float(metric_world.get("initial_z"))
    lift_index = None
    if object_z is not None and initial_z is not None:
        lift_index = _first_index(object_z - float(initial_z) >= lift_threshold_m)

    active = series.get("candidate.controller.feedback_active")
    trigger = series.get("candidate.controller.feedback_trigger_count")
    active_prob = series.get("candidate.controller.feedback_active_probability")
    first_active_index = None
    if active is not None:
        first_active_index = _first_index(active > 0.5)
    if first_active_index is None and trigger is not None:
        first_active_index = _first_index(trigger > 0)
    if first_active_index is None and active_prob is not None:
        first_active_index = _first_index(active_prob > 0.5)

    max_trigger_count = int(np.max(trigger)) if trigger is not None and trigger.size else 0
    max_active_prob = float(np.max(active_prob)) if active_prob is not None and active_prob.size else None

    accel_index = None
    max_accel_est = None
    if object_z is not None and len(object_z) >= 3:
        vel = np.gradient(object_z, dt)
        accel = np.gradient(vel, dt)
        abs_accel = np.abs(accel)
        start = min(max(warmup_steps, 0), len(abs_accel) - 1)
        local = int(np.argmax(abs_accel[start:]))
        accel_index = start + local
        max_accel_est = float(abs_accel[accel_index])

    contact = series.get("newton.panda.rigid_contact_count")
    contact_integral = float(np.sum(contact) * dt) if contact is not None else None

    row = {
        "cell": cfg_run["cell"],
        "split": cfg_run["split"],
        "policy": cfg_run["policy"],
        "run_tag": run_tag,
        "status": metrics.get("status"),
        "controller_mode": metrics.get("controller_mode") or summary.get("controller_mode"),
        "model_evaluation": bool(metrics.get("model_evaluation", False)),
        "no_model_or_training": bool(metrics.get("no_model_or_training", False)),
        "schema_promotion": metrics.get("schema_promotion"),
        "generated_trex_fields": metrics.get("generated_trex_fields", []),
        "failure_reasons": metric_world.get("failure_reasons", []),
        "lift_height_m": _safe_float(metric_world.get("lift_height_m")),
        "hold_duration_s": _safe_float(metric_world.get("hold_duration_s")),
        "max_slip_m": _safe_float(metric_world.get("max_slip_m")),
        "contact_loss_frames": metric_world.get("contact_loss_frames"),
        "max_contact_proxy": metric_world.get("max_contact_proxy"),
        "metric_max_object_accel_m_s2": _safe_float(metric_world.get("max_object_accel_m_s2")),
        "estimated_post_warmup_max_object_accel_m_s2": max_accel_est,
        "estimated_peak_accel_step": _step_at(series, accel_index),
        "estimated_peak_accel_time_s": _time_at(series, accel_index),
        "first_lift_threshold_step": _step_at(series, lift_index),
        "first_lift_threshold_time_s": _time_at(series, lift_index),
        "first_feedback_active_step": _step_at(series, first_active_index),
        "first_feedback_active_time_s": _time_at(series, first_active_index),
        "final_feedback_trigger_count": max_trigger_count,
        "max_feedback_active_probability": max_active_prob,
        "contact_proxy_integral_est": contact_integral,
        "source_npz": str(npz_path),
        "source_metrics": str(metrics_path),
        "source_summary": str(summary_path),
        "npz_key_count": len(keys),
    }
    return row


def _cell_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault(row["cell"], {})[row["policy"]] = row
    comparisons = []
    for cell, policies in sorted(by_cell.items()):
        learned = policies.get("learned_residual")
        noadapt = policies.get("no_adaptation")
        scripted = policies.get("scripted_feedback")
        if not learned:
            continue
        item: dict[str, Any] = {
            "cell": cell,
            "split": learned.get("split"),
            "learned_status": learned.get("status"),
            "learned_first_feedback_active_time_s": learned.get("first_feedback_active_time_s"),
            "learned_final_feedback_trigger_count": learned.get("final_feedback_trigger_count"),
            "learned_first_lift_threshold_time_s": learned.get("first_lift_threshold_time_s"),
            "learned_metric_max_object_accel_m_s2": learned.get("metric_max_object_accel_m_s2"),
        }
        for label, baseline in (("no_adaptation", noadapt), ("scripted_feedback", scripted)):
            if baseline:
                item[f"{label}_status"] = baseline.get("status")
                item[f"{label}_failure_reasons"] = baseline.get("failure_reasons")
                item[f"{label}_metric_max_object_accel_m_s2"] = baseline.get("metric_max_object_accel_m_s2")
                b_accel = baseline.get("metric_max_object_accel_m_s2")
                l_accel = learned.get("metric_max_object_accel_m_s2")
                if b_accel is not None and l_accel is not None:
                    item[f"learned_accel_reduction_vs_{label}_m_s2"] = float(b_accel) - float(l_accel)
        comparisons.append(item)
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = _load_json(config_path)

    rows = [
        _analyze_run(
            root=root,
            cfg_run=run,
            warmup_steps=int(config.get("warmup_steps_for_accel", 15)),
            lift_threshold_m=float(config.get("lift_threshold_m", 0.12)),
        )
        for run in config["runs"]
    ]
    comparisons = _cell_comparisons(rows)

    output_json = root / config["outputs"]["json"]
    output_csv = root / config["outputs"]["csv"]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classification": "residual_adapter_failure_mode_comparison_v1",
        "status": "pass",
        "config": str(config_path),
        "not_training": True,
        "not_model_creation": True,
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "row_count": len(rows),
        "comparison_count": len(comparisons),
        "rows": rows,
        "cell_comparisons": comparisons,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"status": "pass", "json": str(output_json), "csv": str(output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
