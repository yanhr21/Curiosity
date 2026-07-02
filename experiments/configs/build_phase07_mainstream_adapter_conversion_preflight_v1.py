"""Build a Phase07 mainstream-adapter conversion preflight manifest.

This validates bridge-bearing NPZ artifacts and records the exact shapes and
field mappings required for later official OpenPI, GR00T, Diffusion Policy, and
RT-X conversion. It does not write converted datasets, download checkpoints,
train, run inference, or claim success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


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

STATE_FIELDS = [
    "newton.panda.ee_body_q",
    "newton.panda.object_body_q",
    "newton.panda.rigid_contact_count",
    "candidate.controller.phase_index",
    "candidate.controller.commanded_gripper_target",
    "candidate.controller.commanded_lift_target",
    "candidate.task.object_mass_kg",
    "candidate.task.object_friction_mu",
    "candidate.task.nominal_visual_fill",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _shape(data: np.lib.npyio.NpzFile, key: str) -> list[int] | None:
    if key not in data.files:
        return None
    return [int(v) for v in data[key].shape]


def _summarize_npz(root: Path, rel_path: str) -> dict[str, Any]:
    path = root / rel_path
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path)
    missing_action = [field for field in ACTION_FIELDS if field not in data.files]
    missing_state = [field for field in STATE_FIELDS if field not in data.files]
    action_shape = _shape(data, "candidate.action.eef_delta_xyzrpy_gripper")
    camera_shape = _shape(data, "newton.camera.color_rgba")
    return {
        "npz": rel_path,
        "status": "pass" if not missing_action and not missing_state else "fail",
        "missing_action_fields": missing_action,
        "missing_state_fields": missing_state,
        "action_shape_T_W_7": action_shape,
        "camera_color_rgba_shape": camera_shape,
        "ee_body_q_shape": _shape(data, "newton.panda.ee_body_q"),
        "object_body_q_shape": _shape(data, "newton.panda.object_body_q"),
        "world_count": action_shape[1] if action_shape and len(action_shape) >= 2 else None,
        "time_steps": action_shape[0] if action_shape else None,
        "camera_sample_frames": camera_shape[0] if camera_shape else None,
    }


def _method_specs() -> dict[str, Any]:
    common_action = [
        "candidate.action.eef_delta_x",
        "candidate.action.eef_delta_y",
        "candidate.action.eef_delta_z",
        "candidate.action.eef_delta_roll",
        "candidate.action.eef_delta_pitch",
        "candidate.action.eef_delta_yaw",
        "candidate.action.gripper",
    ]
    common_state = [
        "newton.panda.ee_body_q",
        "newton.panda.object_body_q",
        "newton.panda.rigid_contact_count",
        "candidate.controller.phase_index",
        "candidate.controller.commanded_gripper_target",
        "candidate.controller.commanded_lift_target",
        "candidate.task.object_mass_kg",
        "candidate.task.object_friction_mu",
        "candidate.task.nominal_visual_fill",
    ]
    return {
        "openpi_pi0": {
            "target": "LeRobot dataset plus custom Inputs/Outputs transforms in official OpenPI",
            "observation_mapping": {
                "image.base_0_rgb": "newton.camera.color_rgba workspace camera frames from rollout artifacts",
                "state": common_state,
                "prompt": "grasp and lift the water cup while adapting to hidden or misleading fill cues",
            },
            "action_mapping": {"actions": common_action},
            "not_ready_until": ["actual LeRobot dataset conversion is run", "official checkpoint is downloaded or blocked"],
        },
        "gr00t": {
            "target": "GR00T-flavored LeRobot v2 plus meta/modality.json and NEW_EMBODIMENT",
            "modality_draft": {
                "video.front": "newton.camera.color_rgba",
                "state.single_arm": common_state,
                "action.single_arm": common_action[:6],
                "action.gripper": ["candidate.action.gripper"],
                "language": "annotation.human.task_description",
            },
            "not_ready_until": ["GR00T modality config is written", "GR00T checkpoint is downloaded or blocked"],
        },
        "diffusion_policy": {
            "target": "Official Diffusion Policy Dataset/EnvRunner/task config/shape_meta",
            "shape_meta_draft": {
                "obs.lowdim": common_state,
                "obs.image": "newton.camera.color_rgba if using ImagePolicy",
                "action": common_action,
            },
            "not_ready_until": ["Dataset and EnvRunner are implemented using official interfaces"],
        },
        "rtx": {
            "target": "RT-1-X RGB workspace image plus task string plus 7D gripper-frame action",
            "observation_mapping": {
                "image": "newton.camera.color_rgba single workspace view sampled near 3 Hz",
                "task_string": "grasp and lift the water cup while adapting to fill and friction",
            },
            "action_mapping": {"gripper_frame_7d": common_action},
            "not_ready_until": ["RT-1-X checkpoint is downloaded", "7D action is confirmed compatible with Newton execution"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument(
        "--backfill-manifest",
        type=Path,
        default=Path("experiments/outputs/phase07_action_bridge_backfill_v1_20260627/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/phase07_mainstream_adapter_conversion_preflight_v1_20260627/manifest.json"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    backfill_manifest = args.backfill_manifest if args.backfill_manifest.is_absolute() else root / args.backfill_manifest
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = _load_json(backfill_manifest)
    if manifest.get("status") != "pass":
        raise SystemExit(f"backfill manifest is not pass: {manifest.get('status')}")
    summaries = [_summarize_npz(root, item["output_npz"]) for item in manifest.get("results", [])]
    payload = {
        "classification": "phase07_mainstream_adapter_conversion_preflight_v1",
        "status": "pass" if summaries and all(item["status"] == "pass" for item in summaries) else "fail",
        "backfill_manifest": str(backfill_manifest.relative_to(root)),
        "bridge_npz_count": len(summaries),
        "bridge_npz_summaries": summaries,
        "method_specs": _method_specs(),
        "not_dataset_conversion": True,
        "not_training": True,
        "not_inference": True,
        "not_mainstream_success_claim": True,
        "next_required_steps": [
            "Create method-specific dataset conversion scripts under allocation-only runners.",
            "Prepare official method environments under envs/ before compute use.",
            "Download official checkpoints only as part of a documented faithful comparison or blocker audit.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
