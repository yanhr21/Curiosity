#!/usr/bin/env python3
"""Freeze the admitted SUGAR robot RGB/proprio/action corpus for official BPP.

This audit binds both official BPP RGB modalities to one independently executed
admitted Generator+Tracker trace and records the exact 1 Hz prompt chunks plus
causal 10 Hz current-observation / full-rate 50 Hz action-target geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "sugar_bpp_sensorimotor_corpus_v2"
ACTION_PROTOCOL = "sugar_bpp_five_variant_action_admission_v1"
EXPECTED_TRAJECTORIES = 995
EXPECTED_FRAMES = 141
EXPECTED_FRAMES_PER_CAMERA = EXPECTED_TRAJECTORIES * EXPECTED_FRAMES
CAMERA_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
EXPECTED_FRAME_TOTAL = EXPECTED_FRAMES_PER_CAMERA * len(CAMERA_KEYS)
EXPECTED_ACTIONS = 700
ACTION_HORIZON = 16
CURRENT_IMAGE_HORIZON = 2
PROMPT_ACTION_CHUNK = 50
PROMPT_OBSERVATION_INDICES_50HZ = tuple(range(0, EXPECTED_ACTIONS + 1, 50))
SAMPLE_ANCHORS_50HZ = tuple(range(EXPECTED_ACTIONS))

# Exact term order in collect_official_tracker_contact_events.py::_goal_policy_core_observation.
# Three schema-aligned groups preserve Goal Chain's three low-dimensional tokens.
PROPRIO_TERMS = (
    ("projected_gravity", 0, 3),
    ("base_height", 3, 4),
    ("base_linear_velocity", 4, 7),
    ("base_angular_velocity", 7, 10),
    ("joint_position_relative", 10, 39),
    ("joint_velocity_relative", 39, 68),
    ("previous_executed_action", 68, 97),
    ("box_position_body", 97, 100),
    ("box_orientation_tangent_normal_body", 100, 106),
    ("box_linear_velocity_body", 106, 109),
    ("box_angular_velocity_body", 109, 112),
    ("goal_position_body", 112, 115),
    ("goal_orientation_tangent_normal_body", 115, 121),
)
PROPRIO_GROUPS = (
    ("robot_kinematics", 0, 68),
    ("previous_action", 68, 97),
    ("task_geometry", 97, 121),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--action-admission", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def expected_action_ranges() -> list[list[int]]:
    return [
        [min(index * 5, EXPECTED_ACTIONS), min((index + 1) * 5, EXPECTED_ACTIONS)]
        for index in range(EXPECTED_FRAMES)
    ]


def target_clock_rows() -> list[dict[str, Any]]:
    rows = []
    for anchor in SAMPLE_ANCHORS_50HZ:
        latest_rgb = anchor // 5
        action_stop = min(anchor + ACTION_HORIZON, EXPECTED_ACTIONS)
        rows.append(
            {
                "target_anchor_action_index_50hz": anchor,
                "causal_proprioception_index_50hz": anchor,
                "current_rgb_frame_indices_10hz": [
                    max(0, latest_rgb - 1),
                    latest_rgb,
                ],
                "action_slice_50hz": [anchor, action_stop],
                "repeat_final_action_pad_count": (
                    ACTION_HORIZON - (action_stop - anchor)
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    action_admission = args.action_admission.resolve()
    rgb_root = args.rgb_root.resolve()
    camera_contract_path = args.camera_contract.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit: {output}")
    contract = read_json(contract_path)
    if (
        contract.get("exact_admitted_totals", {}).get(
            "train_target_samples_per_epoch"
        )
        != 560000
        or contract.get("exact_admitted_totals", {}).get(
            "train_target_sample_exposures_at_goal_chain_7_epochs"
        )
        != 3920000
        or contract.get("render_contract", {}).get("total_two_camera_frames")
        != EXPECTED_FRAME_TOTAL
    ):
        raise ValueError("full-rate target or dual-camera corpus contract drift")
    camera_contract = read_json(camera_contract_path)
    if (
        camera_contract.get("protocol") != "sugar_bpp_dual_camera_render_contract_v1"
        or camera_contract.get("passed_visibility_audit") is not True
        or tuple(camera_contract.get("camera_keys", ())) != CAMERA_KEYS
        or camera_contract.get("visibility_evidence", {})
        .get("candidate_contract", {})
        .get("sha256")
        != contract.get("render_contract", {})
        .get("camera_candidate_contract", {})
        .get("sha256")
    ):
        raise ValueError("dual-camera visibility contract is absent or failed")
    action_result_path = action_admission / "RESULT.json"
    action_result = read_json(action_result_path)
    if (
        action_result.get("protocol") != ACTION_PROTOCOL
        or action_result.get("passed") is not True
        or action_result.get("admitted_trajectory_count") != EXPECTED_TRAJECTORIES
        or action_result.get("train_five_action_interval_count") != 112000
        or action_result.get("contract", {}).get("sha256") != file_sha256(contract_path)
    ):
        raise ValueError("action admission or frozen contract drift")
    admitted_path = action_admission / str(
        action_result["artifacts"]["admitted_variants"]
    )
    admitted_rows = read_jsonl(admitted_path)
    if len(admitted_rows) != EXPECTED_TRAJECTORIES:
        raise ValueError("admitted action manifest row count drift")

    render_results: dict[tuple[int, str], dict[str, Any]] = {}
    render_result_hashes: dict[tuple[int, str], dict[str, str]] = {}
    for result_path in sorted(rgb_root.glob("candidate_variant_*/RENDER_RESULT_*.json")):
        value = read_json(result_path)
        variant_id = int(value.get("bpp_candidate_variant_id", -1))
        source_trace = str(value.get("source_trace", ""))
        key = (variant_id, source_trace)
        if key in render_results:
            raise ValueError(f"duplicate render result identity: {key}")
        if (
            value.get("passed") is not True
            or value.get("resolution") != [320, 320]
            or value.get("rtx_render_resolution") != [640, 640]
            or value.get("environment_spacing_m") != 30.0
            or value.get("camera_far_clip_m") != 20.0
            or value.get("frame_count_per_trajectory") != EXPECTED_FRAMES
            or value.get("camera_keys") != list(CAMERA_KEYS)
            or value.get("total_rgb_frames")
            != int(value.get("trajectory_count", -1)) * EXPECTED_FRAMES * 2
            or value.get("camera_contract", {}).get("sha256")
            != file_sha256(camera_contract_path)
            or value.get("neighbor_geometry_gate", {}).get(
                "neighbor_center_is_beyond_far_clip"
            ) is not True
        ):
            raise ValueError(f"render contract failed: {result_path}")
        render_results[key] = value
        render_result_hashes[key] = {
            "path": relative(result_path),
            "sha256": file_sha256(result_path),
        }

    trace_geometry_cache: dict[Path, dict[str, tuple[int, ...]]] = {}
    rows: list[dict[str, Any]] = []
    global_frame_paths: set[str] = set()
    stream_hashes_by_motion_camera: dict[
        tuple[str, int, str], list[str]
    ] = defaultdict(list)
    paired_stream_hashes_distinct = True
    all_proprio_reconstructions_bitwise_exact = True
    all_images_nonconstant = True
    all_frame_maps_exact = True
    for admitted in admitted_rows:
        task = str(admitted["task"])
        source_id = int(admitted["source_motion_id"])
        split = str(admitted["split"])
        variant_id = int(admitted["candidate_variant_id"])
        trace_path = ROOT / str(admitted["trace_path"])
        env_index = int(admitted["environment_index"])
        render_key = (variant_id, relative(trace_path))
        render_result = render_results.get(render_key)
        if render_result is None:
            raise ValueError(f"missing render result for {render_key}")
        rendered_sources = {
            (str(record["task"]), int(record["source_motion_id"])): record
            for record in render_result["motion_records"]
        }
        motion_record = rendered_sources.get((task, source_id))
        if motion_record is None:
            raise ValueError(f"render result omitted admitted motion {task}:{source_id}")
        if int(motion_record["trace_environment_index"]) != env_index:
            raise ValueError("render/action trace environment index mismatch")

        directory = ROOT / str(motion_record["frame_directory"])
        frame_map_path = directory / "FRAME_MAP.json"
        frame_map = read_json(frame_map_path)
        frame_map_exact = (
            frame_map.get("task") == task
            and frame_map.get("source_motion_id") == source_id
            and frame_map.get("split") == split
            and frame_map.get("frame_count") == EXPECTED_FRAMES
            and frame_map.get("camera_keys") == list(CAMERA_KEYS)
            and frame_map.get("action_ranges_50hz") == expected_action_ranges()
        )
        all_frame_maps_exact &= frame_map_exact
        if not frame_map_exact:
            raise ValueError(f"frame/action map drift: {frame_map_path}")
        camera_records: dict[str, dict[str, Any]] = {}
        per_camera_hashes: dict[str, str] = {}
        for camera_key in CAMERA_KEYS:
            camera_directory = ROOT / str(
                motion_record["camera_frame_directories"][camera_key]
            )
            frame_paths = sorted(
                camera_directory.glob("*.png"), key=lambda path: int(path.stem)
            )
            if [int(path.stem) for path in frame_paths] != list(
                range(EXPECTED_FRAMES)
            ):
                raise ValueError(f"camera frame index drift: {camera_directory}")
            for frame_path in frame_paths:
                frame_relative = relative(frame_path)
                if frame_relative in global_frame_paths:
                    raise ValueError(f"RGB frame path reused: {frame_relative}")
                global_frame_paths.add(frame_relative)
                image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if image is None or image.shape != (320, 320, 3):
                    raise ValueError(f"invalid RGB frame: {frame_path}")
                all_images_nonconstant &= bool(int(image.max()) > int(image.min()))
            stream_hash = stream_sha256(frame_paths)
            per_camera_hashes[camera_key] = stream_hash
            stream_hashes_by_motion_camera[(task, source_id, camera_key)].append(
                stream_hash
            )
            camera_records[camera_key] = {
                "frame_paths": [relative(path) for path in frame_paths],
                "frame_count": EXPECTED_FRAMES,
                "resolution": [320, 320],
                "fps": 10,
                "stream_sha256": stream_hash,
            }
        paired_stream_hashes_distinct &= (
            per_camera_hashes[CAMERA_KEYS[0]] != per_camera_hashes[CAMERA_KEYS[1]]
        )

        if trace_path not in trace_geometry_cache:
            with np.load(trace_path, allow_pickle=False) as archive:
                expected = {
                    "goal_policy_core_observation_before": (EXPECTED_ACTIONS, 121),
                    "goal_policy_core_observation": (EXPECTED_ACTIONS, 121),
                    "executed_action": (EXPECTED_ACTIONS, 29),
                }
                geometry: dict[str, tuple[int, ...]] = {}
                for name, per_env_shape in expected.items():
                    value = np.asarray(archive[name])
                    if value.shape[0] != EXPECTED_ACTIONS or value.shape[2:] != per_env_shape[1:]:
                        raise ValueError(f"BPP source tensor geometry drift {name}: {value.shape}")
                    if not 0 <= env_index < value.shape[1] or not np.isfinite(value).all():
                        raise ValueError(f"invalid BPP source tensor {name}: {trace_path}")
                    geometry[name] = tuple(map(int, value.shape))
                trace_geometry_cache[trace_path] = geometry
                core = np.asarray(archive["goal_policy_core_observation_before"])
                for source_env in range(core.shape[1]):
                    source_core = core[:, source_env]
                    reconstructed = np.concatenate(
                        [source_core[:, start:stop] for _, start, stop in PROPRIO_GROUPS],
                        axis=-1,
                    )
                    all_proprio_reconstructions_bitwise_exact &= np.array_equal(
                        source_core, reconstructed
                    )

        prompt_chunks = []
        for prompt_index, action_start in enumerate(PROMPT_OBSERVATION_INDICES_50HZ):
            action_stop = min(action_start + PROMPT_ACTION_CHUNK, EXPECTED_ACTIONS)
            prompt_chunks.append({
                "prompt_index": prompt_index,
                "rgb_frame_index_10hz": prompt_index * 10,
                "observation_action_index_50hz": action_start,
                "proprioception_source": (
                    {
                        "array": "goal_policy_core_observation_before",
                        "index": action_start,
                    }
                    if action_start < EXPECTED_ACTIONS
                    else {
                        "array": "goal_policy_core_observation",
                        "index": EXPECTED_ACTIONS - 1,
                        "semantics": "exact post-state after final executed action",
                    }
                ),
                "action_source": (
                    {
                        "array": "executed_action",
                        "slice": [action_start, action_stop],
                        "zero_pad_to_actions": PROMPT_ACTION_CHUNK,
                    }
                    if action_start < EXPECTED_ACTIONS
                    else {
                        "array": "zeros",
                        "shape": [PROMPT_ACTION_CHUNK, 29],
                        "semantics": "official pad_end_prompt_actions=zeros contract",
                    }
                ),
            })
        rows.append({
            "task_family": task,
            "behavior_identity": {
                "source_motion_id": source_id,
                "training_task_id": f"{task}:{source_id}",
                "carry_kick_is_reporting_stratum_only": True,
            },
            "split": split,
            "candidate_variant_id": variant_id,
            "admitted_variant_rank": int(admitted["admitted_variant_rank"]),
            "seed": int(admitted["seed"]),
            "startup_profile_sha256": admitted["startup_profile_sha256"],
            "full_trace_sha256": admitted["full_trace_sha256"],
            "executed_action_sha256": admitted["executed_action_sha256"],
            "safe_physics": admitted["physics"],
            "trace": {
                "path": relative(trace_path),
                "file_sha256": admitted["trace_file_sha256"],
                "environment_index": env_index,
            },
            "robot_rgb": {
                "modalities": camera_records,
                "frame_map_path": relative(frame_map_path),
                "frame_map_sha256": file_sha256(frame_map_path),
            },
            "bpp_prompt": {
                "observation_hz": 1,
                "rgb_modalities": list(CAMERA_KEYS),
                "action_hz": 50,
                "action_chunk_size": PROMPT_ACTION_CHUNK,
                "chunks": prompt_chunks,
            },
            "bpp_target_samples": {
                "sample_anchor_action_indices_50hz": list(SAMPLE_ANCHORS_50HZ),
                "sample_count": len(SAMPLE_ANCHORS_50HZ),
                "current_rgb_modalities": list(CAMERA_KEYS),
                "current_rgb_horizon": CURRENT_IMAGE_HORIZON,
                "current_rgb_hz": 10,
                "causal_proprioception_array": "goal_policy_core_observation_before",
                "causal_proprioception_dimension": 121,
                "causal_proprioception_groups": [
                    {"name": name, "slice": [start, stop], "dimension": stop - start}
                    for name, start, stop in PROPRIO_GROUPS
                ],
                "action_array": "executed_action",
                "action_dimension": 29,
                "action_horizon": ACTION_HORIZON,
                "pad_end_actions": True,
            },
        })

    split_counts = Counter(str(row["split"]) for row in rows)
    per_motion_counts = Counter(
        (
            str(row["task_family"]),
            int(row["behavior_identity"]["source_motion_id"]),
        )
        for row in rows
    )
    checks = {
        "action_admission_passed_and_hash_bound": True,
        "trajectory_count_exact_995": len(rows) == EXPECTED_TRAJECTORIES,
        "two_camera_frame_count_exact_280590": (
            len(global_frame_paths) == EXPECTED_FRAME_TOTAL
        ),
        "every_frame_map_exact": all_frame_maps_exact,
        "all_images_nonconstant_320x320": all_images_nonconstant,
        "every_motion_has_five_rendered_variants": all(
            count == 5 for count in per_motion_counts.values()
        ) and len(per_motion_counts) == 199,
        "five_rgb_streams_unique_within_every_motion_and_camera": all(
            len(values) == 5 and len(set(values)) == 5
            for values in stream_hashes_by_motion_camera.values()
        ) and len(stream_hashes_by_motion_camera) == 199 * 2,
        "agentview_and_eye_in_hand_are_never_identical": paired_stream_hashes_distinct,
        "split_counts_exact": dict(split_counts)
        == {"train": 800, "validation": 100, "test": 95},
        "prompt_geometry_exact_1hz_obs_50hz_actions": (
            len(PROMPT_OBSERVATION_INDICES_50HZ) == 15
            and PROMPT_OBSERVATION_INDICES_50HZ[-1] == 700
        ),
        "target_geometry_exact_700_samples_per_trajectory": (
            len(SAMPLE_ANCHORS_50HZ) == 700
            and SAMPLE_ANCHORS_50HZ[-1] == 699
        ),
        "three_proprio_groups_cover_121_dimensions_once": (
            [start for _, start, _ in PROPRIO_GROUPS] == [0, 68, 97]
            and [stop for _, _, stop in PROPRIO_GROUPS] == [68, 97, 121]
            and all_proprio_reconstructions_bitwise_exact
        ),
        "training_pairing_requires_same_identity_distinct_variant": contract.get(
            "pairing_contract", {}
        ).get("same_source_motion_id") is True
        and contract.get("pairing_contract", {}).get(
            "distinct_rollout_variant_id"
        ) is True,
    }
    passed = all(checks.values())
    rows.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["task_family"]),
            int(row["behavior_identity"]["source_motion_id"]),
            int(row["admitted_variant_rank"]),
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "BPP_SENSORIMOTOR_MANIFEST.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    proprioception_partition = {
        "protocol": "sugar_bpp_three_token_proprioception_partition_v1",
        "source_function": (
            "scripts/sugar/demo_reward/collect_official_tracker_contact_events.py::"
            "_goal_policy_core_observation"
        ),
        "source_dimension": 121,
        "token_count": 3,
        "terms": [
            {"name": name, "slice": [start, stop], "dimension": stop - start}
            for name, start, stop in PROPRIO_TERMS
        ],
        "groups": [
            {"name": name, "slice": [start, stop], "dimension": stop - start}
            for name, start, stop in PROPRIO_GROUPS
        ],
        "all_admitted_rows_reconstruct_bitwise": (
            all_proprio_reconstructions_bitwise_exact
        ),
        "claim_boundary": (
            "This is a lossless schema partition that preserves the official "
            "Goal Chain three-lowdim-token topology; it is not a learned module."
        ),
    }
    partition_path = output / "BPP_PROPRIOCEPTION_PARTITION.json"
    partition_path.write_text(
        json.dumps(proprioception_partition, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    target_clock = {
        "protocol": "sugar_bpp_full_rate_target_clock_v1",
        "target_hz": 50,
        "rgb_hz": 10,
        "rgb_modalities": list(CAMERA_KEYS),
        "current_rgb_horizon": CURRENT_IMAGE_HORIZON,
        "action_horizon": ACTION_HORIZON,
        "rows": target_clock_rows(),
        "claim_boundary": (
            "Exact official lower-rate image mapping and 50 Hz action-target "
            "geometry; repeated image references are not new RGB frames."
        ),
    }
    target_clock_path = output / "BPP_TARGET_CLOCK.json"
    target_clock_path.write_text(
        json.dumps(target_clock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "protocol": PROTOCOL,
        "passed": passed,
        "checks": checks,
        "trajectory_count": len(rows),
        "rgb_frame_count": len(global_frame_paths),
        "transition_count": len(rows) * EXPECTED_ACTIONS,
        "five_action_interval_count": len(rows) * (EXPECTED_ACTIONS // 5),
        "train_five_action_interval_count": split_counts["train"]
        * (EXPECTED_ACTIONS // 5),
        "target_sample_count": len(rows) * len(SAMPLE_ANCHORS_50HZ),
        "train_target_samples_per_epoch": split_counts["train"]
        * len(SAMPLE_ANCHORS_50HZ),
        "split_counts": dict(sorted(split_counts.items())),
        "contract": {"path": relative(contract_path), "sha256": file_sha256(contract_path)},
        "camera_contract": {
            "path": relative(camera_contract_path),
            "sha256": file_sha256(camera_contract_path),
        },
        "action_admission": {
            "result_path": relative(action_result_path),
            "result_sha256": file_sha256(action_result_path),
            "manifest_path": relative(admitted_path),
            "manifest_sha256": file_sha256(admitted_path),
        },
        "render_results": [
            {"variant_id": key[0], "source_trace": key[1], **record}
            for key, record in sorted(render_result_hashes.items())
        ],
        "artifacts": {
            "sensorimotor_manifest": manifest_path.name,
            "proprioception_partition": partition_path.name,
            "target_clock": target_clock_path.name,
        },
        "proprioception_partition": {
            "path": partition_path.name,
            "sha256": file_sha256(partition_path),
            "all_admitted_rows_reconstruct_bitwise": (
                all_proprio_reconstructions_bitwise_exact
            ),
        },
        "target_clock": {
            "path": target_clock_path.name,
            "sha256": file_sha256(target_clock_path),
            "rows": len(target_clock["rows"]),
        },
        "claim_boundary": (
            "This is an action-grounded same-embodiment BPP dataset with 160 train "
            "motion identities and five physical variants per identity. It is not "
            "model training, held-out prompt dependence, or arbitrary human prompting."
        ),
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
