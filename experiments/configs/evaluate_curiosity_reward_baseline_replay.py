#!/usr/bin/env python3
"""Evaluate Phase 03 curiosity reward components on validated baseline rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def _gate_status(value: dict[str, Any]) -> str:
    status = value.get("status")
    if isinstance(status, str):
        return status.lower()
    return ""


def _assert_gates(root: Path, run_tag: str) -> dict[str, str]:
    outputs = root / "experiments" / "outputs"
    gate_paths = {
        "fresh_newton_sensor_contact_sanity": outputs / f"{run_tag}_fresh_newton_sensor_contact_sanity.json",
        "summary": outputs / f"{run_tag}_summary.json",
        "visual_validation": outputs / f"{run_tag}_visual_validation.json",
        "manual_visual_inspection": outputs / f"{run_tag}_manual_visual_inspection.json",
        "lift_hold_metrics": outputs / f"{run_tag}_metrics.json",
    }
    for label, path in gate_paths.items():
        _require_file(path, label)

    sanity = _load_json(gate_paths["fresh_newton_sensor_contact_sanity"])
    visual = _load_json(gate_paths["visual_validation"])
    manual = _load_json(gate_paths["manual_visual_inspection"])
    metrics = _load_json(gate_paths["lift_hold_metrics"])

    if _gate_status(sanity) not in {"pass", "success"}:
        raise ValueError(f"fresh Newton sanity gate failed for {run_tag}: {gate_paths['fresh_newton_sensor_contact_sanity']}")
    if _gate_status(visual) not in {"pass", "success"}:
        raise ValueError(f"visual validation gate failed for {run_tag}: {gate_paths['visual_validation']}")
    if _gate_status(manual) not in {"pass", "success"}:
        raise ValueError(f"manual visual inspection gate failed for {run_tag}: {gate_paths['manual_visual_inspection']}")
    if metrics.get("classification") != "newton_native_lift_hold_metrics_v1":
        raise ValueError(f"unexpected metrics classification for {run_tag}: {gate_paths['lift_hold_metrics']}")

    return {label: str(path) for label, path in gate_paths.items()}


def _step_dt_s(sim_time: np.ndarray) -> float:
    diffs = np.diff(sim_time.astype(np.float64))
    valid = diffs[diffs > 0]
    if valid.size:
        return float(np.median(valid))
    return 1.0 / 60.0


def _contact_proxy(contact: np.ndarray, timesteps: int) -> np.ndarray:
    if contact.shape[0] != timesteps:
        raise ValueError(f"contact first dimension {contact.shape[0]} != timesteps {timesteps}")
    return contact.reshape(timesteps, -1).sum(axis=1).astype(np.float64)


def _phase(data: np.lib.npyio.NpzFile, timesteps: int) -> np.ndarray:
    key = "candidate.controller.phase_index"
    if key not in data:
        return np.zeros(timesteps, dtype=np.int32)
    return np.asarray(data[key]).reshape(timesteps, -1)[:, 0].astype(np.int32)


def _command(data: np.lib.npyio.NpzFile, key: str, timesteps: int) -> np.ndarray:
    if key not in data:
        return np.zeros(timesteps, dtype=np.float64)
    return np.asarray(data[key], dtype=np.float64).reshape(timesteps, -1)[:, 0]


def _mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values.astype(np.float64)))


def _max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmax(values.astype(np.float64)))


def _window_learning_progress(error: np.ndarray) -> float:
    if error.size < 8:
        return 0.0
    window = max(4, error.size // 4)
    start = _mean(error[:window])
    end = _mean(error[-window:])
    return float(start - end)


def _evaluate_one(
    root: Path,
    cfg: dict[str, Any],
    rollout_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_tag = rollout_cfg["run_tag"]
    gates = _assert_gates(root, run_tag)
    outputs = root / "experiments" / "outputs"
    npz_path = outputs / f"{run_tag}.npz"
    _require_file(npz_path, "rollout_npz")

    thresholds = cfg["thresholds"]
    weights = cfg["weights"]

    with np.load(npz_path) as data:
        required = [
            "newton.panda.sim_time",
            "newton.panda.object_body_q",
            "newton.panda.rigid_contact_count",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{run_tag} missing required fields: {missing}")

        object_q = np.asarray(data["newton.panda.object_body_q"], dtype=np.float64)
        if object_q.ndim != 3 or object_q.shape[2] < 3:
            raise ValueError(f"{run_tag} object_body_q must have shape [T, W, >=3]")
        xyz = object_q[:, :, :3]
        timesteps, world_count, _ = xyz.shape
        sim_time = np.asarray(data["newton.panda.sim_time"], dtype=np.float64).reshape(timesteps)
        dt_s = _step_dt_s(sim_time)
        contact = _contact_proxy(np.asarray(data["newton.panda.rigid_contact_count"]), timesteps)
        phase = _phase(data, timesteps)
        lift_cmd = _command(data, "candidate.controller.commanded_lift_target", timesteps)
        gripper_cmd = _command(data, "candidate.controller.commanded_gripper_target", timesteps)

    if timesteps < 4:
        raise ValueError(f"{run_tag} has too few timesteps for replay reward evaluation")

    z = xyz[:, :, 2]
    xy = xyz[:, :, :2]
    active = phase >= 2
    contact_present = contact > 0

    delta_xyz = np.diff(xyz, axis=0)
    pred_delta = np.zeros_like(delta_xyz)
    pred_delta[1:] = delta_xyz[:-1]
    object_error = np.linalg.norm(delta_xyz - pred_delta, axis=2).mean(axis=1)
    object_error_clipped = np.clip(object_error / float(thresholds["object_motion_error_clip_m"]), 0.0, 1.0)

    contact_actual = contact[1:]
    contact_pred = contact[:-1]
    contact_error = np.abs(contact_actual - contact_pred)
    contact_error_clipped = np.clip(contact_error / float(thresholds["contact_error_clip"]), 0.0, 1.0)

    delayed_contact = np.concatenate([[contact[0]], contact[:-1]])
    delayed_error = np.abs(contact - delayed_contact)
    delayed_error_clipped = np.clip(delayed_error[1:] / float(thresholds["contact_error_clip"]), 0.0, 1.0)

    rng = np.random.default_rng(17)
    shuffled_contact = contact.copy()
    rng.shuffle(shuffled_contact)
    shuffled_error = np.abs(contact - shuffled_contact)
    shuffled_error_clipped = np.clip(shuffled_error[1:] / float(thresholds["contact_error_clip"]), 0.0, 1.0)

    dz = np.diff(z, axis=0).mean(axis=1)
    active_next = active[1:]
    contact_next = contact_present[1:]
    useful_lift = np.where(active_next & contact_next, np.maximum(dz, 0.0), 0.0)
    bounded_useful_change = np.clip(useful_lift / float(thresholds["useful_lift_clip_m"]), 0.0, 1.0)

    lift_cmd_delta = np.diff(lift_cmd)
    cmd_scale = max(float(np.nanmax(np.abs(lift_cmd_delta))), float(thresholds["useful_lift_clip_m"]))
    controllable_disagreement = np.clip(np.abs((lift_cmd_delta / cmd_scale) - (dz / float(thresholds["useful_lift_clip_m"]))), 0.0, 1.0)

    dt = np.diff(sim_time)
    dt = np.where(dt <= 0, np.nan, dt)
    vel = np.diff(xyz, axis=0) / dt[:, None, None]
    accel = np.diff(vel, axis=0) / dt[1:, None, None]
    accel_norm = np.linalg.norm(accel, axis=2).mean(axis=1)
    accel_excess = np.clip(accel_norm / float(thresholds["accel_threshold_m_s2"]) - 1.0, 0.0, 1.0)

    running_max_z = np.maximum.accumulate(z, axis=0)
    drop = (running_max_z - z).mean(axis=1)
    drop_penalty = np.clip(drop[2:] / float(thresholds["drop_threshold_m"]), 0.0, 1.0)
    force_excess_all = np.clip(contact / float(thresholds["contact_proxy_force_threshold"]) - 1.0, 0.0, 1.0)
    force_excess = force_excess_all[2:]
    safety_penalty = np.maximum.reduce([accel_excess, drop_penalty, force_excess])

    motion_mag = np.linalg.norm(delta_xyz, axis=2).mean(axis=1)
    command_active = (np.abs(lift_cmd_delta) > 0) | (np.abs(np.diff(gripper_cmd)) > 0) | active_next
    no_op = command_active & (~contact_next) & (motion_mag < float(thresholds["no_op_motion_threshold_m"]))
    no_op_penalty = no_op.astype(np.float64)

    aligned = min(object_error_clipped.size, contact_error_clipped.size, bounded_useful_change.size, controllable_disagreement.size)
    reward_learning_progress = _window_learning_progress(0.5 * object_error_clipped[:aligned] + 0.5 * contact_error_clipped[:aligned])
    lp_series = np.full(aligned, max(reward_learning_progress, 0.0), dtype=np.float64)
    safety_aligned = safety_penalty[: max(0, aligned - 1)]
    if safety_aligned.size < aligned:
        safety_aligned = np.pad(safety_aligned, (0, aligned - safety_aligned.size))
    no_op_aligned = no_op_penalty[:aligned]

    total_reward = (
        weights["learning_progress"] * lp_series
        + weights["controllable_disagreement"] * controllable_disagreement[:aligned]
        + weights["bounded_useful_change"] * bounded_useful_change[:aligned]
        - weights["safety_penalty"] * safety_aligned
        - weights["no_op_penalty"] * no_op_aligned
        - weights["excessive_force_penalty"] * force_excess_all[1 : aligned + 1]
    )

    ablations = {
        "no_curiosity": 0.0,
        "random_intrinsic": _mean(rng.random(aligned)),
        "object_motion_only": _mean(object_error_clipped[:aligned]),
        "contact_only": _mean(contact_error_clipped[:aligned]),
        "tactile_only": _mean(contact_error_clipped[:aligned]),
        "vision_tactile": _mean(0.5 * object_error_clipped[:aligned] + 0.5 * contact_error_clipped[:aligned]),
        "shuffled_tactile": _mean(shuffled_error_clipped[:aligned]),
        "delayed_tactile": _mean(delayed_error_clipped[:aligned]),
    }

    row = {
        "run_tag": run_tag,
        "mass_label": rollout_cfg.get("mass_label", ""),
        "friction_label": rollout_cfg.get("friction_label", ""),
        "held_out": bool(rollout_cfg.get("held_out", False)),
        "timesteps": int(timesteps),
        "world_count": int(world_count),
        "dt_s": dt_s,
        "object_motion_prediction_error_mean": _mean(object_error),
        "object_motion_prediction_error_max": _max(object_error),
        "contact_prediction_error_mean": _mean(contact_error),
        "contact_prediction_error_max": _max(contact_error),
        "tactile_proxy_prediction_error_mean": _mean(contact_error),
        "bounded_useful_change_mean": _mean(bounded_useful_change),
        "controllable_disagreement_mean": _mean(controllable_disagreement),
        "safety_penalty_mean": _mean(safety_penalty),
        "excessive_force_penalty_mean": _mean(force_excess_all),
        "no_op_penalty_mean": _mean(no_op_penalty),
        "learning_progress_proxy": reward_learning_progress,
        "intrinsic_reward_mean": _mean(total_reward),
        "intrinsic_reward_min": float(np.min(total_reward)) if total_reward.size else 0.0,
        "intrinsic_reward_max": float(np.max(total_reward)) if total_reward.size else 0.0,
        "schema_promotion": "blocked",
        "tactile_source": cfg["tactile_source"],
    }
    row.update({f"ablation_{name}_mean": value for name, value in ablations.items()})

    detail = {
        "run_tag": run_tag,
        "gates": gates,
        "npz": str(npz_path),
        "row": row,
        "component_notes": {
            "diagnostic_predictors_only": True,
            "no_model_or_training": True,
            "learning_progress_proxy": "first replay window prediction error minus final replay window prediction error",
            "tactile_exact_schema": "unavailable; using Newton contact proxy only",
        },
    }
    return row, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    args = parser.parse_args()

    cfg = _load_json(args.config)
    rows = []
    details = []
    for rollout_cfg in cfg["rollouts"]:
        row, detail = _evaluate_one(args.root, cfg, rollout_cfg)
        rows.append(row)
        details.append(detail)

    aggregate = {
        "classification": cfg["classification"],
        "phase": cfg["phase"],
        "status": "pass",
        "config": str(args.config),
        "rollout_count": len(rows),
        "no_model_or_training": True,
        "schema_promotion": "blocked",
        "tactile_source": cfg["tactile_source"],
        "diagnostic_predictors": cfg["diagnostic_predictors"],
        "thresholds": cfg["thresholds"],
        "weights": cfg["weights"],
        "rows": rows,
        "details": details,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    columns = list(rows[0].keys()) if rows else []
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps({"status": "pass", "output_json": str(args.output_json), "output_csv": str(args.output_csv)}))


if __name__ == "__main__":
    main()
