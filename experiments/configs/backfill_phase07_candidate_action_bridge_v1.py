"""Backfill Phase07 rollout NPZ files with candidate.action.* bridge fields.

This preserves the source NPZ and writes a new NPZ with explicit provenance.
It is intended to run inside a held allocation because it is data processing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


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

DEFAULT_RUN_TAGS = [
    "phase07_source_quarter_low_truthful_scripted_feedback_rerun_20260627",
    "phase07_source_quarter_medium_hidden_scripted_feedback_20260627",
    "phase07_source_half_low_hidden_scripted_feedback_20260627",
    "phase07_source_half_medium_truthful_scripted_feedback_20260627",
    "phase07_source_three_quarter_medium_misleading_scripted_feedback_20260627",
    "phase07_source_three_quarter_high_truthful_scripted_feedback_20260627",
    "phase07_source_empty_medium_hidden_scripted_feedback_20260627",
    "phase07_source_full_medium_misleading_scripted_feedback_20260627",
    "phase07_eval_empty_high_misleading_no_adaptation_rerun_20260627",
    "phase07_eval_full_low_hidden_no_adaptation_20260627",
    "phase07_eval_three_quarter_low_misleading_no_adaptation_20260627",
    "phase07_eval_empty_high_misleading_scripted_feedback_20260627",
    "phase07_eval_full_low_hidden_scripted_feedback_20260627",
    "phase07_eval_three_quarter_low_misleading_scripted_feedback_20260627",
    "phase07_eval_empty_high_misleading_residual_baseline_20260627",
    "phase07_eval_full_low_hidden_residual_baseline_20260627",
    "phase07_eval_three_quarter_low_misleading_residual_baseline_20260627",
    "phase07_eval_empty_high_misleading_curiosity_weighted_20260627",
    "phase07_eval_full_low_hidden_curiosity_weighted_20260627",
    "phase07_eval_three_quarter_low_misleading_curiosity_weighted_20260627",
    "phase07_eval_empty_high_misleading_random_intrinsic_rerun_20260627",
    "phase07_eval_full_low_hidden_random_intrinsic_20260627",
    "phase07_eval_three_quarter_low_misleading_random_intrinsic_20260627",
    "phase07_eval_empty_high_misleading_object_only_20260627",
    "phase07_eval_full_low_hidden_object_only_20260627",
    "phase07_eval_three_quarter_low_misleading_object_only_20260627",
]


def _relative_eef_action_bridge(ee_body_q: np.ndarray, gripper_target: np.ndarray) -> dict[str, np.ndarray]:
    if ee_body_q.ndim != 3 or ee_body_q.shape[-1] != 7:
        raise ValueError(f"expected ee_body_q with shape (T, W, 7), got {ee_body_q.shape}")
    gripper = np.asarray(gripper_target, dtype=np.float32)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    if gripper.shape[0] != ee_body_q.shape[0]:
        raise ValueError("gripper target time dimension does not match ee_body_q")

    action = np.zeros((ee_body_q.shape[0], ee_body_q.shape[1], 7), dtype=np.float32)
    if ee_body_q.shape[0] > 1:
        action[:-1, :, :3] = ee_body_q[1:, :, :3] - ee_body_q[:-1, :, :3]
        current_quat = ee_body_q[:-1, :, 3:7].reshape(-1, 4)
        next_quat = ee_body_q[1:, :, 3:7].reshape(-1, 4)
        delta_rot = Rotation.from_quat(next_quat) * Rotation.from_quat(current_quat).inv()
        action[:-1, :, 3:6] = delta_rot.as_euler("xyz", degrees=False).reshape(
            ee_body_q.shape[0] - 1, ee_body_q.shape[1], 3
        )
    action[:, :, 6] = gripper if gripper.shape[1] == ee_body_q.shape[1] else gripper[:, :1]
    return {
        "candidate.action.eef_delta_x": action[:, :, 0],
        "candidate.action.eef_delta_y": action[:, :, 1],
        "candidate.action.eef_delta_z": action[:, :, 2],
        "candidate.action.eef_delta_roll": action[:, :, 3],
        "candidate.action.eef_delta_pitch": action[:, :, 4],
        "candidate.action.eef_delta_yaw": action[:, :, 5],
        "candidate.action.gripper": action[:, :, 6],
        "candidate.action.eef_delta_xyzrpy_gripper": action,
    }


def _backfill_one(root: Path, run_tag: str, output_dir: Path) -> dict[str, Any]:
    source_npz = root / "experiments" / "outputs" / f"{run_tag}.npz"
    source_summary = root / "experiments" / "outputs" / f"{run_tag}_summary.json"
    if not source_npz.is_file():
        raise FileNotFoundError(source_npz)
    data = np.load(source_npz)
    keys = set(data.files)
    required_source = ["newton.panda.ee_body_q", "candidate.controller.commanded_gripper_target"]
    missing_source = [key for key in required_source if key not in keys]
    if missing_source:
        raise KeyError(f"{run_tag} missing source fields: {missing_source}")
    arrays = {key: data[key] for key in data.files}
    arrays.update(
        _relative_eef_action_bridge(
            np.asarray(data["newton.panda.ee_body_q"], dtype=np.float32),
            np.asarray(data["candidate.controller.commanded_gripper_target"], dtype=np.float32),
        )
    )
    output_npz = output_dir / f"{run_tag}_with_candidate_action_bridge.npz"
    np.savez_compressed(output_npz, **arrays)
    check = np.load(output_npz)
    missing_action = [key for key in ACTION_FIELDS if key not in check.files]
    payload = {
        "classification": "phase07_candidate_action_bridge_backfill_v1",
        "status": "pass" if not missing_action else "fail",
        "run_tag": run_tag,
        "source_npz": str(source_npz.relative_to(root)),
        "source_summary": str(source_summary.relative_to(root)) if source_summary.is_file() else None,
        "output_npz": str(output_npz.relative_to(root)),
        "source_fields": required_source,
        "action_fields": ACTION_FIELDS,
        "missing_action_fields": missing_action,
        "source_preserved": True,
        "not_training": True,
        "not_mainstream_success_claim": True,
        "provenance": {
            "source": "newton.panda.ee_body_q finite difference plus candidate.controller.commanded_gripper_target",
            "pose_convention": "xyz + quaternion xyzw",
            "action_convention": "current_to_next_relative_eef_delta_xyz_euler_rpy_radians_plus_absolute_gripper_target",
            "final_timestep_delta": "zero_xyzrpy_delta_with_current_gripper_target",
        },
    }
    validation_path = output_dir / f"{run_tag}_candidate_action_bridge_backfill_validation.json"
    validation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if missing_action:
        raise RuntimeError(f"{run_tag} bridge backfill missing fields: {missing_action}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs/phase07_action_bridge_backfill_v1_20260627"))
    parser.add_argument("--run-tags", nargs="*", default=DEFAULT_RUN_TAGS)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/outputs/phase07_action_bridge_backfill_v1_20260627/manifest.json"))
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [_backfill_one(root, run_tag, output_dir) for run_tag in args.run_tags]
    manifest = {
        "classification": "phase07_candidate_action_bridge_backfill_manifest_v1",
        "status": "pass" if all(item["status"] == "pass" for item in results) else "fail",
        "output_dir": str(output_dir.relative_to(root)),
        "result_count": len(results),
        "results": results,
        "not_training": True,
        "not_mainstream_success_claim": True,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
