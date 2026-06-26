#!/usr/bin/env python3
"""Locate object-acceleration peaks in a Newton lift-hold rollout NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_difference_accel_vec(xyz: np.ndarray, sim_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] < 3:
        return np.zeros((0, xyz.shape[1], 3), dtype=np.float64), np.zeros((0, xyz.shape[1]), dtype=np.float64)
    dt = np.diff(sim_time).astype(np.float64)
    dt = np.where(dt <= 0, np.nan, dt)
    vel = np.diff(xyz, axis=0) / dt[:, None, None]
    accel_dt = dt[1:]
    accel_vec = np.diff(vel, axis=0) / accel_dt[:, None, None]
    accel_norm = np.linalg.norm(accel_vec, axis=2)
    return accel_vec, accel_norm


def _field_at(data: np.lib.npyio.NpzFile, key: str, step: int, default: Any = None) -> Any:
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.shape[0] <= step:
        return default
    value = arr[step]
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _top_events(data: np.lib.npyio.NpzFile, top_k: int) -> list[dict[str, Any]]:
    object_q = np.asarray(data["newton.panda.object_body_q"], dtype=np.float64)
    sim_time = np.asarray(data["newton.panda.sim_time"], dtype=np.float64).reshape(object_q.shape[0])
    xyz = object_q[:, :, :3]
    accel_vec, accel_norm = _finite_difference_accel_vec(xyz, sim_time)
    if accel_norm.size == 0:
        return []

    flat_order = np.argsort(accel_norm.reshape(-1))[::-1]
    events: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for flat_idx in flat_order:
        accel_idx, world_idx = np.unravel_index(int(flat_idx), accel_norm.shape)
        step = int(accel_idx + 2)
        key = (step, int(world_idx))
        if key in seen:
            continue
        seen.add(key)
        prev_step = max(0, step - 1)
        next_step = min(xyz.shape[0] - 1, step + 1)
        event = {
            "rank": len(events) + 1,
            "step": step,
            "accel_index": int(accel_idx),
            "world_idx": int(world_idx),
            "sim_time_s": float(sim_time[step]),
            "accel_norm_m_s2": float(accel_norm[accel_idx, world_idx]),
            "accel_vector_m_s2": [float(x) for x in accel_vec[accel_idx, world_idx]],
            "object_z_m": float(xyz[step, world_idx, 2]),
            "object_z_prev_m": float(xyz[prev_step, world_idx, 2]),
            "object_z_next_m": float(xyz[next_step, world_idx, 2]),
            "phase_index": _field_at(data, "candidate.controller.phase_index", step),
            "commanded_lift_target": _field_at(data, "candidate.controller.commanded_lift_target", step),
            "commanded_gripper_target": _field_at(data, "candidate.controller.commanded_gripper_target", step),
            "contact_proxy": _field_at(data, "newton.panda.rigid_contact_count", step),
            "feedback_active": _field_at(data, "candidate.controller.feedback_active", step),
            "feedback_reason_id": _field_at(data, "candidate.controller.feedback_reason_id", step),
            "feedback_trigger_count": _field_at(data, "candidate.controller.feedback_trigger_count", step),
            "feedback_lift_velocity_scale": _field_at(
                data, "candidate.controller.feedback_lift_velocity_scale", step
            ),
            "feedback_hold_height_offset_m": _field_at(
                data, "candidate.controller.feedback_hold_height_offset_m", step
            ),
            "feedback_stabilization_extension_s": _field_at(
                data, "candidate.controller.feedback_stabilization_extension_s", step
            ),
            "feedback_observed_object_accel_m_s2": _field_at(
                data, "candidate.controller.feedback_observed_object_accel_m_s2", step
            ),
        }
        events.append(event)
        if len(events) >= top_k:
            break
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    data = np.load(args.npz)
    summary = _load_json(args.summary)
    events = _top_events(data, args.top_k)
    payload = {
        "classification": "newton_lift_hold_accel_peak_analysis_v1",
        "status": "pass" if events else "fail",
        "run_tag": args.run_tag,
        "npz": str(args.npz),
        "summary": str(args.summary),
        "controller_mode": summary.get("controller_mode"),
        "scripted_feedback": summary.get("scripted_feedback"),
        "task_metrics": summary.get("task_metrics"),
        "events": events,
        "generated_trex_fields": [],
        "schema_promotion": "blocked",
        "no_model_or_training": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "events": len(events)}))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
