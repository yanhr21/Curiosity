"""Build Phase07 stage-1 dataset indices for official mainstream adapters.

This script validates bridge-bearing Phase07 NPZ artifacts and writes
method-specific index/config files for later official OpenPI, GR00T, Diffusion
Policy, and RT-X conversion. It intentionally stops before writing full
LeRobot parquet/video datasets, zarr replay buffers, downloading checkpoints,
training, or inference.

Because it reads rollout NPZ files and prepares dataset indices, it must run
inside a held Slurm allocation.
"""

from __future__ import annotations

import argparse
import json
import os
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

TRAIN_CELLS = {
    "quarter_low_truthful",
    "quarter_medium_hidden",
    "half_low_hidden",
    "half_medium_truthful",
    "three_quarter_medium_misleading",
    "three_quarter_high_truthful",
}
VALIDATION_CELLS = {
    "empty_medium_hidden",
    "full_medium_misleading",
}
HELD_OUT_CELLS = {
    "empty_high_misleading",
    "full_low_hidden",
    "three_quarter_low_misleading",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_split_indices(out_dir: Path, episodes: list[dict[str, Any]]) -> dict[str, str]:
    split_dir = out_dir / "splits"
    split_paths: dict[str, str] = {}
    for split in ["train", "validation", "held_out_eval_only"]:
        rows = [episode for episode in episodes if episode["split"] == split]
        path = split_dir / f"{split}.jsonl"
        _jsonl_write(path, rows)
        split_paths[split] = str(path.relative_to(out_dir))
    return split_paths


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _shape(data: np.lib.npyio.NpzFile, key: str) -> list[int] | None:
    if key not in data.files:
        return None
    return [int(v) for v in data[key].shape]


def _cell_from_run_tag(run_tag: str) -> str:
    for cell in sorted(TRAIN_CELLS | VALIDATION_CELLS | HELD_OUT_CELLS, key=len, reverse=True):
        if cell in run_tag:
            return cell
    return "unknown"


def _split_from_cell(cell: str) -> str:
    if cell in TRAIN_CELLS:
        return "train"
    if cell in VALIDATION_CELLS:
        return "validation"
    if cell in HELD_OUT_CELLS:
        return "held_out_eval_only"
    return "unknown"


def _policy_from_run_tag(run_tag: str) -> str:
    if "_source_" in run_tag:
        return "scripted_feedback_source"
    if "no_adaptation" in run_tag:
        return "no_adaptation"
    if "scripted_feedback" in run_tag:
        return "scripted_feedback"
    if "residual_baseline" in run_tag:
        return "residual_baseline"
    if "curiosity_weighted" in run_tag:
        return "curiosity_weighted"
    if "random_intrinsic" in run_tag:
        return "random_intrinsic"
    if "object_only" in run_tag:
        return "object_only"
    return "unknown"


def _episode_summary(root: Path, item: dict[str, Any], episode_index: int) -> dict[str, Any]:
    run_tag = item["run_tag"]
    npz_path = root / item["output_npz"]
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path)
    missing_action = [field for field in ACTION_FIELDS if field not in data.files]
    missing_state = [field for field in STATE_FIELDS if field not in data.files]
    if missing_action or missing_state:
        raise RuntimeError(f"{run_tag} missing action={missing_action} state={missing_state}")
    action_shape = _shape(data, "candidate.action.eef_delta_xyzrpy_gripper")
    image_shape = _shape(data, "newton.camera.color_rgba")
    if not action_shape or len(action_shape) != 3 or action_shape[-1] != 7:
        raise RuntimeError(f"{run_tag} invalid 7D action shape: {action_shape}")
    cell = _cell_from_run_tag(run_tag)
    split = _split_from_cell(cell)
    source_summary = item.get("source_summary")
    summary = _load_json(root / source_summary) if isinstance(source_summary, str) and (root / source_summary).is_file() else {}
    video_export = summary.get("video_export") if isinstance(summary, dict) else None
    video_path = video_export.get("path") if isinstance(video_export, dict) else None
    if isinstance(video_path, str) and Path(video_path).is_absolute():
        video_rel = _rel(root, Path(video_path))
    elif isinstance(video_path, str):
        video_rel = video_path
    else:
        video_rel = None
    return {
        "episode_index": episode_index,
        "run_tag": run_tag,
        "cell": cell,
        "split": split,
        "policy_source": _policy_from_run_tag(run_tag),
        "npz": item["output_npz"],
        "source_npz": item.get("source_npz"),
        "source_summary": source_summary,
        "rollout_video": video_rel,
        "task": "grasp and lift the water cup while adapting to fill mass and friction without dropping it",
        "action": {
            "keys": ACTION_FIELDS,
            "shape_T_W_7": action_shape,
            "convention": "relative EEF delta xyz+rpy radians plus absolute gripper target",
            "source": "candidate.action.eef_delta_xyzrpy_gripper",
        },
        "state": {
            "keys": STATE_FIELDS,
            "ee_body_q_shape": _shape(data, "newton.panda.ee_body_q"),
            "object_body_q_shape": _shape(data, "newton.panda.object_body_q"),
        },
        "image": {
            "source": "newton.camera.color_rgba",
            "shape": image_shape,
            "camera_policy": "workspace RGB; downstream method adapters may sample or encode frames according to official API",
        },
        "must_not_use_for_training": split == "held_out_eval_only",
    }


def _write_openpi_index(out_dir: Path, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "episode_index": ep["episode_index"],
            "split": ep["split"],
            "npz": ep["npz"],
            "image.base_0_rgb": ep["image"]["source"],
            "state_keys": STATE_FIELDS,
            "action_keys": ACTION_FIELDS,
            "prompt": ep["task"],
            "held_out_training_forbidden": ep["must_not_use_for_training"],
        }
        for ep in episodes
    ]
    path = out_dir / "openpi_lerobot_stage1" / "episodes.jsonl"
    _jsonl_write(path, rows)
    config = {
        "classification": "phase07_openpi_lerobot_stage1_index_v1",
        "official_code": "external/openpi",
        "official_basis": "LeRobot dataset conversion plus OpenPI Inputs/Outputs transforms",
        "status": "stage1_index_ready_not_lerobot_dataset",
        "episodes_jsonl": str(path.relative_to(out_dir)),
        "not_training": True,
        "not_official_openpi_run": True,
        "next_required_steps": [
            "Create a real LeRobot dataset with official OpenPI-compatible features.",
            "Prepare OpenPI env under envs/ and official checkpoint/cache path.",
            "Run official policy inference/fine-tune comparison only after no held-out leakage audit.",
        ],
    }
    config_path = path.parent / "openpi_phase07_mapping.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"episodes_jsonl": str(path.relative_to(out_dir)), "config": str(config_path.relative_to(out_dir))}


def _write_gr00t_index(out_dir: Path, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    base = out_dir / "gr00t_lerobot_v2_stage1"
    meta = base / "meta"
    modality = {
        "video": {
            "front": {
                "original_key": "newton.camera.color_rgba",
                "type": "video",
                "shape": [200, 576, 3],
            }
        },
        "state": {
            "single_arm": {
                "start": 0,
                "end": len(STATE_FIELDS),
                "dtype": "float32",
                "original_keys": STATE_FIELDS,
            }
        },
        "action": {
            "single_arm": {
                "start": 0,
                "end": 6,
                "dtype": "float32",
                "original_keys": ACTION_FIELDS[:6],
                "representation": "relative_eef_delta_xyzrpy",
            },
            "gripper": {
                "start": 6,
                "end": 7,
                "dtype": "float32",
                "original_keys": ["candidate.action.gripper"],
                "representation": "absolute_gripper_target",
            },
        },
        "annotation": {
            "human": {
                "task_description": {
                    "type": "language",
                    "value": "grasp and lift the water cup while adapting to hidden or misleading fill cues",
                }
            }
        },
    }
    base.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "modality.json").write_text(json.dumps(modality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _jsonl_write(base / "episodes.jsonl", episodes)
    _jsonl_write(
        base / "tasks.jsonl",
        [{"task_index": 0, "task": "grasp and lift the water cup while adapting to hidden or misleading fill cues"}],
    )
    info = {
        "classification": "phase07_gr00t_lerobot_v2_stage1_index_v1",
        "official_code": "external/Isaac-GR00T",
        "official_basis": "GR00T-flavored LeRobot v2 with meta/modality.json",
        "status": "stage1_index_ready_not_full_lerobot_v2_dataset",
        "not_training": True,
        "not_official_gr00t_run": True,
        "held_out_training_forbidden": [ep["episode_index"] for ep in episodes if ep["must_not_use_for_training"]],
    }
    (base / "meta" / "info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "episodes_jsonl": "gr00t_lerobot_v2_stage1/episodes.jsonl",
        "tasks_jsonl": "gr00t_lerobot_v2_stage1/tasks.jsonl",
        "modality_json": "gr00t_lerobot_v2_stage1/meta/modality.json",
        "info_json": "gr00t_lerobot_v2_stage1/meta/info.json",
    }


def _write_diffusion_policy_index(out_dir: Path, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    base = out_dir / "diffusion_policy_stage1"
    base.mkdir(parents=True, exist_ok=True)
    _jsonl_write(base / "episodes.jsonl", episodes)
    shape_meta = {
        "classification": "phase07_diffusion_policy_shape_meta_stage1_v1",
        "official_code": "external/diffusion_policy",
        "status": "shape_meta_ready_dataset_and_envrunner_not_implemented",
        "shape_meta": {
            "obs": {
                "lowdim": {"shape": [len(STATE_FIELDS)], "type": "low_dim", "source_keys": STATE_FIELDS},
                "image": {"shape": ["H", "W", 3], "type": "rgb", "source_key": "newton.camera.color_rgba"},
            },
            "action": {"shape": [7], "source_key": "candidate.action.eef_delta_xyzrpy_gripper"},
        },
        "not_training": True,
        "not_official_diffusion_policy_run": True,
        "next_required_steps": [
            "Implement official BaseDataset-compatible loader.",
            "Implement Newton EnvRunner for held-out rollouts and videos.",
            "Prepare official Diffusion Policy env under envs/ before training.",
        ],
    }
    path = base / "shape_meta.json"
    path.write_text(json.dumps(shape_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"episodes_jsonl": "diffusion_policy_stage1/episodes.jsonl", "shape_meta": str(path.relative_to(out_dir))}


def _write_rtx_index(out_dir: Path, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    base = out_dir / "rtx_stage1"
    base.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "episode_index": ep["episode_index"],
            "split": ep["split"],
            "npz": ep["npz"],
            "image": "newton.camera.color_rgba",
            "task_string": ep["task"],
            "action_7d": "candidate.action.eef_delta_xyzrpy_gripper",
            "sample_rate_hz_target": 3,
            "held_out_training_forbidden": ep["must_not_use_for_training"],
        }
        for ep in episodes
    ]
    _jsonl_write(base / "episodes.jsonl", rows)
    config = {
        "classification": "phase07_rtx_stage1_index_v1",
        "official_code": "external/open_x_embodiment",
        "official_basis": "RGB workspace image plus task string plus 7D gripper-frame action",
        "status": "stage1_index_ready_not_official_rtx_dataset",
        "not_training": True,
        "not_official_rtx_run": True,
        "target_checkpoint": "gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax",
    }
    path = base / "rtx_phase07_mapping.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"episodes_jsonl": "rtx_stage1/episodes.jsonl", "config": str(path.relative_to(out_dir))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument(
        "--backfill-manifest",
        type=Path,
        default=Path("experiments/outputs/phase07_action_bridge_backfill_v1_20260627/manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/phase07_mainstream_stage1_dataset_index_v1_20260627"),
    )
    parser.add_argument("--allow-outside-slurm", action="store_true")
    args = parser.parse_args()

    if not args.allow_outside_slurm and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("ERROR: this dataset-index build must run inside a held Slurm allocation")

    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest_path = args.backfill_manifest if args.backfill_manifest.is_absolute() else root / args.backfill_manifest
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "pass":
        raise SystemExit(f"backfill manifest is not pass: {manifest.get('status')}")
    raw_results = manifest.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise SystemExit("backfill manifest has no results")
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = [
        _episode_summary(root, item, episode_index)
        for episode_index, item in enumerate(raw_results)
        if isinstance(item, dict)
    ]
    split_counts: dict[str, int] = {}
    for episode in episodes:
        split_counts[episode["split"]] = split_counts.get(episode["split"], 0) + 1
    if split_counts.get("train", 0) == 0:
        raise SystemExit("stage1 dataset index has no train episodes")
    if split_counts.get("validation", 0) == 0:
        raise SystemExit("stage1 dataset index has no validation episodes")
    if split_counts.get("held_out_eval_only", 0) == 0:
        raise SystemExit("stage1 dataset index has no held-out evaluation episodes")

    _jsonl_write(output_dir / "episodes.jsonl", episodes)
    split_indices = _write_split_indices(output_dir, episodes)
    method_outputs = {
        "openpi_pi0": _write_openpi_index(output_dir, episodes),
        "gr00t": _write_gr00t_index(output_dir, episodes),
        "diffusion_policy": _write_diffusion_policy_index(output_dir, episodes),
        "rtx": _write_rtx_index(output_dir, episodes),
    }
    payload = {
        "classification": "phase07_mainstream_stage1_dataset_index_v1",
        "status": "pass",
        "backfill_manifest": str(manifest_path.relative_to(root)),
        "output_dir": str(output_dir.relative_to(root)),
        "episode_count": len(episodes),
        "split_counts": split_counts,
        "episodes_jsonl": "episodes.jsonl",
        "split_indices": split_indices,
        "method_outputs": method_outputs,
        "not_training": True,
        "not_full_lerobot_dataset": True,
        "not_zarr_replay_buffer": True,
        "not_inference": True,
        "not_success_claim": True,
        "held_out_policy": "held_out_eval_only episodes must not be used for training, normalization fitting, threshold tuning, or adapter design",
        "next_required_steps": [
            "Run this stage1 index build inside allocation after action-bridge backfill.",
            "Implement official method-specific final dataset materialization where required.",
            "Prepare official environments under envs/ and official checkpoints before inference or fine-tuning.",
            "Run closed-loop Phase07 held-out evaluation with full videos for any official method comparison.",
        ],
    }
    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
