#!/usr/bin/env python3
"""Extract Newton-native lift-hold metrics from a namespaced rollout NPZ."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _max_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _max_consecutive(mask: np.ndarray) -> int:
    return _max_true_run(mask)


def _finite_difference_accel(xyz: np.ndarray, sim_time: np.ndarray) -> np.ndarray:
    if xyz.shape[0] < 3:
        return np.zeros((0, xyz.shape[1]), dtype=np.float32)
    dt = np.diff(sim_time).astype(np.float64)
    dt = np.where(dt <= 0, np.nan, dt)
    vel = np.diff(xyz, axis=0) / dt[:, None, None]
    accel_dt = dt[1:]
    accel = np.diff(vel, axis=0) / accel_dt[:, None, None]
    return np.linalg.norm(accel, axis=2)


def _step_dt_s(sim_time: np.ndarray) -> float:
    diffs = np.diff(sim_time.astype(np.float64))
    valid = diffs[diffs > 0]
    if valid.size:
        return float(np.median(valid))
    return 1.0 / 60.0


def _contact_proxy_by_step(contact: np.ndarray, timesteps: int) -> np.ndarray:
    arr = np.asarray(contact)
    if arr.shape[0] != timesteps:
        raise ValueError(f"contact first dimension {arr.shape[0]} does not match timesteps {timesteps}")
    flat = arr.reshape(timesteps, -1)
    return flat.sum(axis=1).astype(np.float64)


def _phase_indices(data: np.lib.npyio.NpzFile, timesteps: int) -> np.ndarray:
    key = "candidate.controller.phase_index"
    if key not in data:
        return np.zeros(timesteps, dtype=np.int32)
    phases = np.asarray(data[key]).reshape(timesteps, -1)
    return phases[:, 0].astype(np.int32)


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[0])


def _extract_metrics(
    data: np.lib.npyio.NpzFile,
    schema: dict[str, Any],
    summary: dict[str, Any],
    run_tag: str,
    baseline_name: str,
    mass_label: str,
    friction_label: str,
    pose_seed: str,
    manual_visual_inspection: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = schema["required_episode_fields"]
    missing = [field for field in required if field not in data.files]
    if missing:
        raise ValueError(f"NPZ missing required metrics fields: {missing}")

    object_q = np.asarray(data["newton.panda.object_body_q"], dtype=np.float64)
    if object_q.ndim != 3 or object_q.shape[2] < 3:
        raise ValueError("newton.panda.object_body_q must have shape [T, world, >=3]")
    xyz = object_q[:, :, :3]
    timesteps, world_count, _ = xyz.shape
    sim_time = np.asarray(data["newton.panda.sim_time"], dtype=np.float64).reshape(timesteps)
    dt_s = _step_dt_s(sim_time)
    contact_proxy = _contact_proxy_by_step(np.asarray(data["newton.panda.rigid_contact_count"]), timesteps)
    phase = _phase_indices(data, timesteps)

    success_cfg = schema["metrics"]["success"]
    failure_cfg = schema["metrics"]["failure"]
    lift_threshold = float(success_cfg["lift_height_m"]["threshold_min"])
    hold_threshold = float(success_cfg["hold_duration_s"]["threshold_min"])
    slip_threshold = float(success_cfg["max_slip_m"]["threshold_max"])
    drop_threshold = float(failure_cfg["drop_height_loss_m"]["threshold"])
    contact_loss_threshold = int(failure_cfg["under_grip_contact_loss_frames"]["threshold"])
    contact_proxy_threshold = float(failure_cfg["over_force_contact_proxy_threshold"]["threshold"])
    accel_threshold = float(failure_cfg["unstable_object_accel_m_s2"]["threshold"])

    accel_norm = _finite_difference_accel(xyz, sim_time)
    initial_xyz = xyz[0]
    z = xyz[:, :, 2]
    xy = xyz[:, :, :2]
    rows = []
    per_world = []

    for world_idx in range(world_count):
        initial_z = float(z[0, world_idx])
        threshold_z = initial_z + lift_threshold
        lifted = z[:, world_idx] >= threshold_z
        first_lift_idx = _first_index(lifted)
        max_z = float(np.max(z[:, world_idx]))
        final_z = float(z[-1, world_idx])
        lift_height = max_z - initial_z
        final_lift = final_z - initial_z

        if first_lift_idx is None:
            hold_duration_s = 0.0
            max_slip_m = float(np.max(np.linalg.norm(xy[:, world_idx] - initial_xyz[world_idx, :2], axis=1)))
            drop_height_loss_m = 0.0
            post_lift_contact_loss_frames = 0
        else:
            hold_duration_s = _max_true_run(lifted[first_lift_idx:]) * dt_s
            hold_xy = xy[first_lift_idx:, world_idx]
            max_slip_m = float(np.max(np.linalg.norm(hold_xy - hold_xy[0], axis=1)))
            post_lift_z = z[first_lift_idx:, world_idx]
            drop_height_loss_m = float(np.max(np.maximum.accumulate(post_lift_z) - post_lift_z))
            post_lift_contact_loss_frames = _max_consecutive(contact_proxy[first_lift_idx:] <= 0)

        phase_after_close = phase >= 2
        if np.any(phase_after_close):
            contact_loss_frames = _max_consecutive(contact_proxy[phase_after_close] <= 0)
        else:
            contact_loss_frames = post_lift_contact_loss_frames

        max_contact_proxy = float(np.max(contact_proxy)) if contact_proxy.size else 0.0
        contact_proxy_integral = float(np.sum(contact_proxy) * dt_s)
        max_accel = float(np.nanmax(accel_norm[:, world_idx])) if accel_norm.size else 0.0

        object_not_dropped = drop_height_loss_m <= drop_threshold
        success = (
            lift_height >= lift_threshold
            and hold_duration_s >= hold_threshold
            and max_slip_m <= slip_threshold
            and object_not_dropped
            and contact_loss_frames <= contact_loss_threshold
            and max_contact_proxy <= contact_proxy_threshold
            and max_accel <= accel_threshold
        )
        failure_reasons = []
        if lift_height < lift_threshold:
            failure_reasons.append("lift_height_below_threshold")
        if hold_duration_s < hold_threshold:
            failure_reasons.append("hold_duration_below_threshold")
        if max_slip_m > slip_threshold:
            failure_reasons.append("slip_above_threshold")
        if not object_not_dropped:
            failure_reasons.append("drop_height_loss_above_threshold")
        if contact_loss_frames > contact_loss_threshold:
            failure_reasons.append("contact_loss_frames_above_threshold")
        if max_contact_proxy > contact_proxy_threshold:
            failure_reasons.append("contact_proxy_above_threshold")
        if max_accel > accel_threshold:
            failure_reasons.append("object_accel_above_threshold")

        success_per_contact_proxy_integral = float(success) / max(contact_proxy_integral, 1e-6)
        row = {
            "run_tag": run_tag,
            "baseline_name": baseline_name,
            "scene": summary.get("scene", ""),
            "tracked_object": summary.get("tracked_object", ""),
            "mass_label": mass_label,
            "friction_label": friction_label,
            "pose_seed": pose_seed,
            "status": "success" if success else "fail",
            "lift_height_m": lift_height,
            "hold_duration_s": hold_duration_s,
            "max_slip_m": max_slip_m,
            "object_not_dropped": object_not_dropped,
            "drop_height_loss_m": drop_height_loss_m,
            "contact_loss_frames": int(contact_loss_frames),
            "max_contact_proxy": max_contact_proxy,
            "max_object_accel_m_s2": max_accel,
            "success_per_contact_proxy_integral": success_per_contact_proxy_integral,
            "manual_visual_inspection": manual_visual_inspection,
            "contact_sheet": summary.get("contact_sheet", ""),
            "frame_browser": summary.get("frame_browser", ""),
            "summary_json": str(summary.get("_summary_path", "")),
            "npz": str(summary.get("npz", "")),
        }
        rows.append(row)
        per_world.append(
            {
                "world_idx": world_idx,
                "status": row["status"],
                "failure_reasons": failure_reasons,
                "initial_z": initial_z,
                "max_z": max_z,
                "final_z": final_z,
                "lift_height_m": lift_height,
                "final_lift_m": final_lift,
                "hold_duration_s": hold_duration_s,
                "max_slip_m": max_slip_m,
                "object_not_dropped": object_not_dropped,
                "drop_height_loss_m": drop_height_loss_m,
                "contact_loss_frames": int(contact_loss_frames),
                "post_lift_contact_loss_frames": int(post_lift_contact_loss_frames),
                "max_contact_proxy": max_contact_proxy,
                "contact_proxy_integral": contact_proxy_integral,
                "max_object_accel_m_s2": max_accel,
                "success_per_contact_proxy_integral": success_per_contact_proxy_integral,
            }
        )

    model_evaluation = summary.get("controller_mode") == "lift_hold_learned_residual"
    payload = {
        "classification": "newton_native_lift_hold_metrics_v1",
        "status": "pass" if rows and all(row["status"] == "success" for row in rows) else "fail",
        "run_tag": run_tag,
        "baseline_name": baseline_name,
        "controller_mode": summary.get("controller_mode"),
        "controller_type": summary.get("controller_type"),
        "schema": schema.get("classification"),
        "dt_s": dt_s,
        "timesteps": int(timesteps),
        "world_count": int(world_count),
        "thresholds": {
            "lift_height_m": lift_threshold,
            "hold_duration_s": hold_threshold,
            "max_slip_m": slip_threshold,
            "drop_height_loss_m": drop_threshold,
            "contact_loss_frames": contact_loss_threshold,
            "max_contact_proxy": contact_proxy_threshold,
            "max_object_accel_m_s2": accel_threshold,
        },
        "source_fields": sorted(data.files),
        "per_world": per_world,
        "rows": rows,
        "generated_trex_fields": [],
        "schema_promotion": "blocked",
        "no_model_or_training": not model_evaluation,
        "model_evaluation": model_evaluation,
        "residual_adapter_checkpoint": summary.get("scripted_feedback", {}).get("residual_adapter_checkpoint"),
    }
    return payload, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--baseline-name", default="no_adaptation_scripted_grasp_lift")
    parser.add_argument("--mass-label", default="nominal")
    parser.add_argument("--friction-label", default="nominal")
    parser.add_argument("--pose-seed", default="nominal")
    parser.add_argument("--manual-visual-inspection", default="not_checked")
    args = parser.parse_args()

    schema = _load_json(args.schema)
    summary = _load_json(args.summary)
    summary["_summary_path"] = str(args.summary)
    with np.load(args.npz) as data:
        payload, rows = _extract_metrics(
            data,
            schema,
            summary,
            run_tag=args.run_tag,
            baseline_name=args.baseline_name,
            mass_label=args.mass_label,
            friction_label=args.friction_label,
            pose_seed=args.pose_seed,
            manual_visual_inspection=args.manual_visual_inspection,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    columns = schema["report_columns"]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps({"status": payload["status"], "metrics": str(args.output_json), "csv": str(args.output_csv)}))


if __name__ == "__main__":
    main()
