#!/usr/bin/env python3
"""Freeze the action-grounded half of the SUGAR Zero-WAM ICL manifest.

This is an adapter/data audit, not a local Zero-WAM implementation.  It accepts
only rollouts produced by the released SUGAR Generator+Tracker collector and
proves causal pre-step state/command/action alignment before robot RGB is
rendered from the recorded physical states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TASK_COUNTS = {"CarryBox": 100, "KickBox": 99}
EXPECTED_STEPS = 700
CONTROL_DT_S = 0.02
PROMPT_FRAMES = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-corpus", type=Path, required=True)
    parser.add_argument("--prompt-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def split_for_motion(motion_id: int) -> str:
    remainder = int(motion_id) % 10
    if remainder == 8:
        return "validation"
    if remainder == 9:
        return "test"
    return "train"


def prompt_split(split: str) -> str:
    return "valid" if split == "validation" else split


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_shape(
    archive: np.lib.npyio.NpzFile,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in archive.files:
        raise RuntimeError(f"missing required action-grounded array: {name}")
    value = np.asarray(archive[name])
    if value.shape != shape:
        raise RuntimeError(
            f"{name} shape drift: expected {shape}, observed {value.shape}"
        )
    return value


def exact_equal(left: np.ndarray, right: np.ndarray, description: str) -> None:
    if not np.array_equal(left, right):
        maximum_error = float(
            np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
        )
        raise RuntimeError(f"{description}; maximum error={maximum_error}")


def prompt_record(
    prompt_root: Path,
    task: str,
    source_motion_id: int,
    source_reference_steps: int,
) -> dict[str, object]:
    split = split_for_motion(source_motion_id)
    directory = (
        prompt_root
        / prompt_split(split)
        / task
        / str(source_motion_id)
    )
    frames = sorted(directory.glob("*.png"), key=lambda path: int(path.stem))
    if [int(path.stem) for path in frames] != list(range(PROMPT_FRAMES)):
        raise RuntimeError(
            f"prompt frame contract failed for {task}:{source_motion_id}"
        )
    source_indices = np.rint(
        np.linspace(0, source_reference_steps - 1, PROMPT_FRAMES)
    ).astype(np.int64)
    if len(np.unique(source_indices)) != PROMPT_FRAMES:
        raise RuntimeError(
            f"prompt timestamps contain duplicates for {task}:{source_motion_id}"
        )
    return {
        "frame_paths": [relative(path) for path in frames],
        "source_frame_indices_50hz": source_indices.tolist(),
        "timestamps_s": (source_indices.astype(np.float64) * CONTROL_DT_S).tolist(),
    }


def main() -> None:
    args = parse_args()
    action_root = args.action_corpus.resolve()
    prompt_root = args.prompt_corpus.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit: {output}")
    if not action_root.is_dir() or not prompt_root.is_dir():
        raise FileNotFoundError("action and prompt corpus roots must exist")

    shard_dirs = sorted(
        path.parent for path in action_root.glob("*/TRACE.npz")
    )
    if len(shard_dirs) != 8:
        raise RuntimeError(f"expected 8 action shards, found {len(shard_dirs)}")

    rows: list[dict[str, object]] = []
    checkpoint_paths: dict[str, set[Path]] = defaultdict(set)
    shard_records: list[dict[str, object]] = []
    all_source_keys: list[tuple[str, int]] = []
    total_transitions = 0
    all_finite = True
    all_pre_post_continuous = True
    maximum_motion_clock_delta = 0

    for shard_dir in shard_dirs:
        result_path = shard_dir / "RESULT.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("passed") is not True:
            raise RuntimeError(f"collector gate did not pass: {result_path}")
        task = str(result["task_family"])
        if task not in TASK_COUNTS:
            raise RuntimeError(f"unexpected task family: {task}")
        tracker = Path(result["tracker_checkpoint"]).resolve()
        generator = Path(result["generator_checkpoint"]).resolve()
        checkpoint_paths[f"{task}:tracker"].add(tracker)
        checkpoint_paths[f"{task}:generator"].add(generator)

        trace_path = shard_dir / "TRACE.npz"
        with np.load(trace_path, allow_pickle=False) as archive:
            steps = int(np.asarray(archive["executed_action"]).shape[0])
            envs = int(np.asarray(archive["executed_action"]).shape[1])
            if steps != EXPECTED_STEPS:
                raise RuntimeError(f"{trace_path} has {steps} steps")
            expected = (steps, envs)
            root_before = require_shape(
                archive, "robot_root_state_before_w", expected + (13,)
            )
            joint_before = require_shape(
                archive, "robot_joint_pos_before", expected + (29,)
            )
            joint_vel_before = require_shape(
                archive, "robot_joint_vel_before", expected + (29,)
            )
            object_before = require_shape(
                archive, "object_root_state_before_w", expected + (13,)
            )
            policy_before = require_shape(
                archive, "policy_observation_before", expected + (510,)
            )
            command_before = require_shape(
                archive, "generator_command_before", expected + (36,)
            )
            motion_before = require_shape(
                archive, "motion_frame_before", expected
            ).astype(np.int64)
            local_before = require_shape(
                archive, "local_motion_id_before", expected
            ).astype(np.int64)
            source_before = require_shape(
                archive, "source_motion_id_before", expected
            ).astype(np.int64)
            requested_action = require_shape(
                archive, "requested_action", expected + (29,)
            )
            executed_action = require_shape(
                archive, "executed_action", expected + (29,)
            )
            legacy_action = require_shape(archive, "action", expected + (29,))
            root_after = require_shape(
                archive, "robot_root_state_w", expected + (13,)
            )
            joint_after = require_shape(
                archive, "robot_joint_pos", expected + (29,)
            )
            joint_vel_after = require_shape(
                archive, "robot_joint_vel", expected + (29,)
            )
            object_after = require_shape(
                archive, "object_root_state_w", expected + (13,)
            )
            motion_after = require_shape(
                archive, "motion_frame", expected
            ).astype(np.int64)
            source_after = require_shape(
                archive, "source_motion_id", expected
            ).astype(np.int64)
            done = require_shape(archive, "done", expected).astype(bool)
            origins = require_shape(archive, "environment_origin_w", (envs, 3))
            transition_index = require_shape(
                archive, "transition_index", (steps,)
            )
            transition_time = require_shape(
                archive, "transition_time_s", (steps,)
            )
            source_ids = require_shape(
                archive, "source_motion_id_by_local_motion", (envs,)
            ).astype(np.int64)
            reference_steps = require_shape(
                archive, "source_reference_steps_by_local_motion", (envs,)
            ).astype(np.int64)

            exact_equal(command_before, policy_before[:, :, :36],
                        "Generator command/Tracker prefix mismatch")
            exact_equal(requested_action, executed_action,
                        "requested action was not executed exactly")
            exact_equal(executed_action, legacy_action,
                        "legacy action field drifted from executed action")
            exact_equal(source_before, source_after,
                        "source motion changed across a transition")
            if np.any(done):
                raise RuntimeError(f"episode reset inside action trace: {trace_path}")
            if not np.array_equal(transition_index, np.arange(steps)):
                raise RuntimeError(f"transition index drift: {trace_path}")
            exact_equal(
                transition_time,
                np.arange(steps, dtype=np.float64) * CONTROL_DT_S,
                "control timestamp drift",
            )
            clock_delta = motion_after - motion_before
            if np.any(clock_delta < 0) or np.any(clock_delta > 1):
                raise RuntimeError(f"reference clock discontinuity: {trace_path}")
            maximum_motion_clock_delta = max(
                maximum_motion_clock_delta, int(np.max(clock_delta))
            )

            continuity = (
                np.array_equal(root_after[:-1], root_before[1:])
                and np.array_equal(joint_after[:-1], joint_before[1:])
                and np.array_equal(joint_vel_after[:-1], joint_vel_before[1:])
                and np.array_equal(object_after[:-1], object_before[1:])
            )
            all_pre_post_continuous &= continuity
            finite_arrays = (
                root_before, joint_before, joint_vel_before, object_before,
                policy_before, command_before, requested_action, executed_action,
                root_after, joint_after, joint_vel_after, object_after, origins,
                transition_time,
            )
            shard_finite = all(np.isfinite(value).all() for value in finite_arrays)
            all_finite &= shard_finite

            for env_index, source_motion_id in enumerate(source_ids.tolist()):
                if not np.all(source_before[:, env_index] == source_motion_id):
                    raise RuntimeError(
                        f"source ID drift for {task}:{source_motion_id}"
                    )
                local_id = int(local_before[0, env_index])
                if not np.all(local_before[:, env_index] == local_id):
                    raise RuntimeError(
                        f"local motion ID drift for {task}:{source_motion_id}"
                    )
                split = split_for_motion(source_motion_id)
                prompt = prompt_record(
                    prompt_root, task, source_motion_id,
                    int(reference_steps[env_index]),
                )
                rows.append(
                    {
                        "task": task,
                        "source_motion_id": source_motion_id,
                        "split": split,
                        "prompt": prompt,
                        "action_target": {
                            "trace_path": relative(trace_path),
                            "environment_index": env_index,
                            "transition_count": steps,
                            "control_hz": 50,
                            "control_timestamps_s": {
                                "array": "transition_time_s",
                                "slice": [0, steps],
                            },
                            "generator_command": {
                                "array": "generator_command_before",
                                "shape": [steps, 36],
                            },
                            "tracker_observation": {
                                "array": "policy_observation_before",
                                "shape": [steps, 510],
                            },
                            "executed_action": {
                                "array": "executed_action",
                                "shape": [steps, 29],
                            },
                            "pre_state_arrays": [
                                "robot_root_state_before_w",
                                "robot_joint_pos_before",
                                "robot_joint_vel_before",
                                "object_root_state_before_w",
                            ],
                            "post_state_arrays": [
                                "robot_root_state_w",
                                "robot_joint_pos",
                                "robot_joint_vel",
                                "object_root_state_w",
                            ],
                        },
                        "expert": {
                            "generator_checkpoint": relative(generator),
                            "tracker_checkpoint": relative(tracker),
                        },
                    }
                )
                all_source_keys.append((task, source_motion_id))
                total_transitions += steps

            shard_records.append(
                {
                    "path": relative(shard_dir),
                    "task": task,
                    "trajectory_count": envs,
                    "transition_count": steps * envs,
                    "finite": shard_finite,
                    "pre_post_state_continuous": continuity,
                }
            )

    expected_keys = {
        (task, motion_id)
        for task, count in TASK_COUNTS.items()
        for motion_id in range(count)
    }
    observed_counter = Counter(all_source_keys)
    rows.sort(key=lambda row: (str(row["task"]), int(row["source_motion_id"])))
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["split"]))].append(row)
    for row in rows:
        task = str(row["task"])
        split = str(row["split"])
        source_id = int(row["source_motion_id"])
        same = grouped[(task, split)]
        same_index = next(
            index for index, candidate in enumerate(same)
            if int(candidate["source_motion_id"]) == source_id
        )
        alternate = same[(same_index + 1) % len(same)]
        other_task = "KickBox" if task == "CarryBox" else "CarryBox"
        wrong_candidates = grouped[(other_task, split)]
        wrong = wrong_candidates[same_index % len(wrong_candidates)]
        row["fixed_counterfactual_prompts"] = {
            "wrong_task": {
                "task": other_task,
                "source_motion_id": int(wrong["source_motion_id"]),
            },
            "reversed": {
                "task": task,
                "source_motion_id": source_id,
                "frame_order": list(reversed(range(PROMPT_FRAMES))),
            },
            "same_task_alternate": {
                "task": task,
                "source_motion_id": int(alternate["source_motion_id"]),
            },
        }

    checkpoint_records = {}
    for identity, paths in sorted(checkpoint_paths.items()):
        if len(paths) != 1:
            raise RuntimeError(f"checkpoint identity drift for {identity}: {paths}")
        path = next(iter(paths))
        checkpoint_records[identity] = {
            "path": relative(path),
            "sha256": sha256(path),
        }

    split_counts = Counter(str(row["split"]) for row in rows)
    task_counts = Counter(str(row["task"]) for row in rows)
    checks = {
        "eight_official_collection_shards": len(shard_records) == 8,
        "complete_199_motion_coverage": set(observed_counter) == expected_keys,
        "each_motion_recorded_once": all(
            observed_counter[key] == 1 for key in expected_keys
        ),
        "all_action_and_state_arrays_finite": all_finite,
        "exact_generator_command_is_tracker_prefix": True,
        "exact_tracker_output_was_executed": True,
        "causal_pre_post_state_continuity": all_pre_post_continuous,
        "no_episode_reset_inside_trace": True,
        "source_id_disjoint_train_validation_test": len(observed_counter) == len(rows),
        "all_prompts_have_64_clean_frames": len(rows) == 199,
        "at_least_100k_action_transitions": total_transitions >= 100_000,
        "at_least_150_train_trajectories": split_counts["train"] >= 150,
    }
    passed = all(checks.values())
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "ACTION_MANIFEST.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "protocol": "sugar_zero_wam_action_grounded_manifest_audit_v1",
        "passed": passed,
        "checks": checks,
        "trajectory_count": len(rows),
        "transition_count": total_transitions,
        "task_counts": dict(sorted(task_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "maximum_reference_clock_delta_per_control_step": maximum_motion_clock_delta,
        "shards": shard_records,
        "released_experts": checkpoint_records,
        "artifacts": {"action_manifest": "ACTION_MANIFEST.jsonl"},
        "robot_rgb_status": (
            "not_collected_by_this_gate; render only from the exact audited "
            "pre-transition physical states before freezing the full ICL manifest"
        ),
        "data_scale_boundary": (
            "199 trajectories and 139,300 transitions are admitted for the fixed "
            "two-task same-embodiment SUGAR audit. They are not a replacement for "
            "Zero-WAM's 6K+/8.6K-task pre-training data."
        ),
        "claim_boundary": (
            "This proves an action-grounded official SUGAR data interface only. "
            "It is not Zero-WAM training, prompt dependence or demo following."
        ),
        "automatic_next_branch": (
            "render_exact_state_aligned_robot_rgb"
            if passed
            else "repair_action_grounded_data_contract"
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
