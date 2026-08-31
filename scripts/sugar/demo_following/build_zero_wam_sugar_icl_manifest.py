#!/usr/bin/env python3
"""Freeze the complete SUGAR video-action manifest for official Zero-WAM.

Only the explicitly allowlisted causal fields from the action audit are exposed.
Contact, lift, done, success and other outcome arrays remain in the source trace
but are intentionally absent from this manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PROMPT_FRAMES = 64
ROBOT_FRAMES = 141
CONTROL_STEPS = 700
VIDEO_INTERVAL_ACTIONS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-audit-dir", type=Path, required=True)
    parser.add_argument("--robot-rgb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def numeric_pngs(directory: Path, expected: int) -> list[Path]:
    paths = sorted(directory.glob("*.png"), key=lambda path: int(path.stem))
    expected_stems = list(range(expected))
    observed_stems = [int(path.stem) for path in paths]
    if observed_stems != expected_stems:
        raise RuntimeError(
            f"frame sequence contract failed for {directory}: "
            f"expected {expected}, observed {len(paths)}"
        )
    return paths


def main() -> None:
    args = parse_args()
    action_audit = args.action_audit_dir.resolve()
    robot_root = args.robot_rgb_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    action_result = json.loads(
        (action_audit / "RESULT.json").read_text(encoding="utf-8")
    )
    if (
        action_result.get("protocol")
        != "sugar_zero_wam_action_grounded_manifest_audit_v1"
        or action_result.get("passed") is not True
    ):
        raise RuntimeError("action-grounded manifest audit is not admitted")
    action_manifest_path = action_audit / "ACTION_MANIFEST.jsonl"
    action_rows = [
        json.loads(line)
        for line in action_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(action_rows) != 199:
        raise RuntimeError(f"expected 199 action rows, found {len(action_rows)}")

    prompt_roots = {
        (ROOT / row["prompt"]["frame_paths"][0]).resolve().parents[3]
        for row in action_rows
    }
    if len(prompt_roots) != 1:
        raise RuntimeError(f"prompt root drift: {prompt_roots}")
    prompt_root = next(iter(prompt_roots))
    prompt_render_results = []
    for path in sorted(prompt_root.glob("RENDER_RESULT_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("protocol") != "sugar_clean_xirl_reference_render_v1"
            or record.get("passed") is not True
            or record.get("neighbor_center_is_beyond_far_clip") is not True
            or float(record.get("environment_spacing_m", 0.0)) != 30.0
        ):
            raise RuntimeError(f"unadmitted isolated prompt shard: {path}")
        prompt_render_results.append(record)
    if len(prompt_render_results) != 8:
        raise RuntimeError(
            f"expected 8 isolated prompt shard results, found {len(prompt_render_results)}"
        )

    render_results = []
    for path in sorted(robot_root.glob("RENDER_RESULT_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("protocol") != "sugar_zero_wam_exact_state_robot_rgb_v1"
            or record.get("passed") is not True
            or record.get("neighbor_geometry_gate", {}).get(
                "neighbor_center_is_beyond_far_clip"
            ) is not True
            or float(record.get("environment_spacing_m", 0.0)) != 30.0
        ):
            raise RuntimeError(f"unadmitted robot RGB shard: {path}")
        render_results.append(record)
    if len(render_results) != 8:
        raise RuntimeError(f"expected 8 robot RGB shard results, found {len(render_results)}")

    full_rows: list[dict[str, object]] = []
    identical_streams = 0
    identical_compared_frames = 0
    total_compared_frames = 0
    seen_keys: set[tuple[str, int]] = set()
    for action_row in action_rows:
        task = str(action_row["task"])
        source_motion_id = int(action_row["source_motion_id"])
        split = str(action_row["split"])
        key = (task, source_motion_id)
        if key in seen_keys:
            raise RuntimeError(f"duplicate action row: {key}")
        seen_keys.add(key)

        directory = robot_root / split / task / str(source_motion_id)
        robot_frames = numeric_pngs(directory, ROBOT_FRAMES)
        frame_map = json.loads(
            (directory / "FRAME_MAP.json").read_text(encoding="utf-8")
        )
        if (
            frame_map["task"] != task
            or int(frame_map["source_motion_id"]) != source_motion_id
            or frame_map["split"] != split
            or int(frame_map["trace_environment_index"])
            != int(action_row["action_target"]["environment_index"])
        ):
            raise RuntimeError(f"RGB/action identity mismatch for {key}")
        timestamps = np.asarray(frame_map["frame_timestamps_s"], dtype=np.float64)
        exact_timestamps = np.arange(ROBOT_FRAMES, dtype=np.float64) / 10.0
        if not np.array_equal(timestamps, exact_timestamps):
            raise RuntimeError(f"robot video timestamp drift for {key}")
        action_ranges = frame_map["action_ranges_50hz"]
        expected_ranges = [
            [
                min(frame * VIDEO_INTERVAL_ACTIONS, CONTROL_STEPS),
                min((frame + 1) * VIDEO_INTERVAL_ACTIONS, CONTROL_STEPS),
            ]
            for frame in range(ROBOT_FRAMES)
        ]
        if action_ranges != expected_ranges:
            raise RuntimeError(f"robot video/action interval drift for {key}")

        prompt_paths = [ROOT / value for value in action_row["prompt"]["frame_paths"]]
        if len(prompt_paths) != PROMPT_FRAMES or not all(
            path.is_file() for path in prompt_paths
        ):
            raise RuntimeError(f"prompt path contract failed for {key}")
        comparison_indices = np.rint(
            np.linspace(0, ROBOT_FRAMES - 1, PROMPT_FRAMES)
        ).astype(np.int64)
        selected_robot_paths = [robot_frames[int(index)] for index in comparison_indices]
        prompt_hashes = [file_sha256(path) for path in prompt_paths]
        robot_hashes = [file_sha256(path) for path in selected_robot_paths]
        same_count = sum(
            prompt_hash == robot_hash
            for prompt_hash, robot_hash in zip(prompt_hashes, robot_hashes)
        )
        identical_compared_frames += same_count
        total_compared_frames += PROMPT_FRAMES
        prompt_digest = sequence_digest(prompt_paths)
        robot_digest = sequence_digest(selected_robot_paths)
        if prompt_digest == robot_digest:
            identical_streams += 1

        full_rows.append(
            {
                "task": task,
                "source_motion_id": source_motion_id,
                "split": split,
                "language_condition": None,
                "prompt": {
                    **action_row["prompt"],
                    "sequence_sha256": prompt_digest,
                    "stream_semantics": "clean kinematic selected demonstration",
                    "cached_once_at_inference": True,
                },
                "robot_target": {
                    "frame_paths": [relative(path) for path in robot_frames],
                    "frame_timestamps_s": timestamps.tolist(),
                    "action_ranges_50hz": action_ranges,
                    "sequence_sha256_at_64_normalized_indices": robot_digest,
                    "stream_semantics": (
                        "clean exact-state render of official Generator+Tracker "
                        "physical rollout"
                    ),
                },
                "action_target": action_row["action_target"],
                "expert": action_row["expert"],
                "fixed_counterfactual_prompts": action_row[
                    "fixed_counterfactual_prompts"
                ],
                "deployed_input_allowlist": {
                    "prompt_rgb": True,
                    "causal_robot_rgb_history": True,
                    "causal_generator_command_history_36d": True,
                    "causal_tracker_observation_history_510d": True,
                    "causal_executed_action_history_29d": True,
                    "language": False,
                    "future_target": False,
                    "contact_or_success_label": False,
                    "selected_demo_id_scalar": False,
                },
            }
        )

    split_counts = Counter(str(row["split"]) for row in full_rows)
    task_counts = Counter(str(row["task"]) for row in full_rows)
    expected_keys = {
        (task, motion_id)
        for task, count in (("CarryBox", 100), ("KickBox", 99))
        for motion_id in range(count)
    }
    counterfactual_valid = True
    row_by_key = {
        (str(row["task"]), int(row["source_motion_id"])): row
        for row in full_rows
    }
    for row in full_rows:
        task = str(row["task"])
        source_id = int(row["source_motion_id"])
        split = str(row["split"])
        counterfactuals = row["fixed_counterfactual_prompts"]
        wrong_key = (
            str(counterfactuals["wrong_task"]["task"]),
            int(counterfactuals["wrong_task"]["source_motion_id"]),
        )
        alternate_key = (
            str(counterfactuals["same_task_alternate"]["task"]),
            int(counterfactuals["same_task_alternate"]["source_motion_id"]),
        )
        wrong = row_by_key.get(wrong_key)
        alternate = row_by_key.get(alternate_key)
        counterfactual_valid &= bool(
            wrong is not None
            and alternate is not None
            and wrong["task"] != task
            and wrong["split"] == split
            and alternate["task"] == task
            and alternate["split"] == split
            and int(alternate["source_motion_id"]) != source_id
            and counterfactuals["reversed"]["frame_order"]
            == list(reversed(range(PROMPT_FRAMES)))
        )

    checks = {
        "action_grounded_audit_passed": True,
        "eight_isolated_prompt_rgb_shards_passed": len(prompt_render_results) == 8,
        "eight_exact_state_rgb_shards_passed": len(render_results) == 8,
        "complete_199_motion_pairing": seen_keys == expected_keys,
        "source_id_disjoint_splits": split_counts
        == Counter({"train": 160, "validation": 20, "test": 19}),
        "each_prompt_has_64_frames": all(
            len(row["prompt"]["frame_paths"]) == PROMPT_FRAMES for row in full_rows
        ),
        "each_robot_target_has_141_frames": all(
            len(row["robot_target"]["frame_paths"]) == ROBOT_FRAMES
            for row in full_rows
        ),
        "all_700_actions_covered_by_10hz_intervals": all(
            row["robot_target"]["action_ranges_50hz"][-2:] == [[695, 700], [700, 700]]
            for row in full_rows
        ),
        "prompt_and_robot_are_not_identical_pixel_streams": identical_streams == 0,
        "fixed_counterfactuals_are_split_matched": counterfactual_valid,
        "language_disabled_for_primary_icl": all(
            row["language_condition"] is None for row in full_rows
        ),
        "outcome_and_future_fields_excluded_from_deployed_inputs": all(
            row["deployed_input_allowlist"]["future_target"] is False
            and row["deployed_input_allowlist"]["contact_or_success_label"] is False
            for row in full_rows
        ),
        "at_least_20k_train_video_action_intervals": split_counts["train"] * 140
        >= 20_000,
    }
    passed = all(checks.values())
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "ICL_MANIFEST.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in full_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "protocol": "sugar_zero_wam_immutable_video_action_icl_manifest_v1",
        "passed": passed,
        "checks": checks,
        "trajectory_count": len(full_rows),
        "transition_count": len(full_rows) * CONTROL_STEPS,
        "video_action_interval_count": len(full_rows) * 140,
        "train_video_action_interval_count": split_counts["train"] * 140,
        "task_counts": dict(sorted(task_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "identical_normalized_frame_pairs": identical_compared_frames,
        "compared_normalized_frame_pairs": total_compared_frames,
        "identical_complete_streams": identical_streams,
        "action_manifest_sha256": file_sha256(action_manifest_path),
        "artifacts": {"icl_manifest": "ICL_MANIFEST.jsonl"},
        "data_scale_boundary": (
            "This is sufficient for the predeclared 160-train-motion, 22,400-interval "
            "two-task SUGAR adaptation/audit. It is not sufficient to reproduce "
            "Zero-WAM's foundation pre-training scale or open-ended claims."
        ),
        "claim_boundary": (
            "Passing freezes paired data only. Official Zero-WAM strict loading "
            "and frozen prompt-dependence gates remain mandatory before adaptation."
        ),
        "automatic_next_branch": (
            "wait_for_official_zero_wam_release_then_run_strict_load_and_frozen_prompt_gate"
            if passed
            else "repair_video_action_manifest_contract"
        ),
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
