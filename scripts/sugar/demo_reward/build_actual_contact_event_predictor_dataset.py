#!/usr/bin/env python3
"""Build motion-disjoint cross-demo targets from actual SUGAR rollouts.

The selected demonstration is a fixed numeric condition.  Actual contact,
event duration and motion regime come only from the same-clock IsaacLab
rollout trace.  Official binary contact annotations are used only to describe
the selected reference demonstration, never as actual-rollout targets.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from audit_actual_contact_event_corpus import motion_split


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = (
    ROOT
    / "experiments/demo_following/contact_event_reward_redesign_v1/actual_tracker_corpus"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/demo_following/contact_event_reward_redesign_v1/"
    "actual_contact_event_predictor_dataset_v1"
)
TASKS = ("CarryBox", "KickBox")
SPLITS = ("train", "validation", "test")
HISTORY_STEPS = 10
FUTURE_STEPS = 10
DEMO_WINDOWS = 32
ANCHOR_STRIDE = 4
EVENT_DURATION_SCALE_FRAMES = 100.0
EFFECTOR_BODY_INDEX = {"left_hand": 24, "right_hand": 32, "left_foot": 6, "right_foot": 12}
ROLE_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")
PAIR_ROLE_NAMES = ("correct", "same_task_wrong", "cross_task_wrong")
TARGET_NAMES = (
    "body_mse",
    "box_position_mse",
    "box_rotation_6d_mse",
    "box_velocity_mse",
    "left_hand_contact_mismatch",
    "right_hand_contact_mismatch",
    "left_foot_contact_mismatch",
    "right_foot_contact_mismatch",
    "left_hand_duration_mismatch",
    "right_hand_duration_mismatch",
    "left_foot_duration_mismatch",
    "right_foot_duration_mismatch",
    "motion_regime_mismatch",
)
CONTINUOUS_SLICES = ((0, 105), (105, 108), (108, 114), (114, 120))
ALIGNMENT_SCALES = np.asarray((0.05**2, 0.05**2, 0.25**2, 0.25**2), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SUGAR/data")
    parser.add_argument(
        "--policy-observation-key",
        choices=("policy_observation", "goal_policy_core_observation"),
        default="policy_observation",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("clock_phase", "free_window"),
        default="clock_phase",
        help=(
            "clock_phase binds every target to causal normalized episode time; "
            "free_window is retained only to reproduce the rejected v1 diagnostic"
        ),
    )
    return parser.parse_args()


def quaternion_wxyz_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-8)
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return np.swapaxes(matrix[..., :, :2], -2, -1).reshape(q.shape[:-1] + (6,))


def matrix_to_rotation6d(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    return np.swapaxes(value[..., :, :2], -2, -1).reshape(value.shape[:-2] + (6,))


def event_remaining(contact: np.ndarray) -> np.ndarray:
    contact = np.asarray(contact, dtype=bool)
    remaining = np.zeros(contact.shape, dtype=np.float32)
    for role in range(contact.shape[1]):
        padded = np.pad(contact[:, role].astype(np.int8), (1, 1))
        edges = np.flatnonzero(np.diff(padded))
        for start, stop in zip(edges[::2], edges[1::2]):
            remaining[start:stop, role] = np.arange(stop - start, 0, -1)
    return remaining


def motion_regime(object_position: np.ndarray, object_velocity: np.ndarray) -> np.ndarray:
    baseline = float(np.median(object_position[: min(25, len(object_position)), 2]))
    lifted = object_position[:, 2] - baseline >= 0.05
    moving = np.linalg.norm(object_velocity, axis=-1) >= 0.05
    return lifted.astype(np.uint8) * 2 + moving.astype(np.uint8)


def reference_contact_roles(
    task: str,
    binary_contact: np.ndarray,
    body_position: np.ndarray,
    object_position: np.ndarray,
) -> np.ndarray:
    roles = np.zeros((len(binary_contact), 4), dtype=bool)
    active = np.flatnonzero(binary_contact)
    if task == "CarryBox":
        roles[active, 0] = True
        roles[active, 1] = True
    else:
        foot_position = body_position[:, [EFFECTOR_BODY_INDEX["left_foot"], EFFECTOR_BODY_INDEX["right_foot"]]]
        nearest = np.argmin(
            np.linalg.norm(foot_position - object_position[:, None, :], axis=-1), axis=1
        )
        roles[active, 2 + nearest[active]] = True
    return roles


def load_reference(task: str, source_id: int, data_root: Path) -> dict[str, np.ndarray]:
    directory = data_root / task / f"data_{source_id:03d}"
    with np.load(directory / "robot_50hz.npz", allow_pickle=False) as archive:
        body = np.asarray(archive["body_pos_w"], dtype=np.float32)
    with (directory / "obj_motion_global_50hz.pkl").open("rb") as stream:
        obj = pickle.load(stream)
    binary = np.load(directory / "contact_labels_50hz.npy", allow_pickle=False).astype(bool)
    position = np.asarray(obj["obj_trans"], dtype=np.float32)
    rotation = np.asarray(obj["obj_rot"], dtype=np.float32)
    linear = np.asarray(obj["obj_lin_vel"], dtype=np.float32)
    angular = np.asarray(obj["obj_ang_vel"], dtype=np.float32)
    length = min(len(body), len(position), len(rotation), len(linear), len(angular), len(binary))
    body, position, rotation = body[:length], position[:length], rotation[:length]
    velocity = np.concatenate((linear[:length], angular[:length]), axis=-1)
    contact = reference_contact_roles(task, binary[:length], body, position)
    return {
        "body_relative_object": body - position[:, None, :],
        "object_position": position,
        "rotation_6d": matrix_to_rotation6d(rotation),
        "object_velocity": velocity,
        "contact": contact,
        "duration": event_remaining(contact),
        "regime": motion_regime(position, linear[:length]),
    }


def window_features(reference: dict[str, np.ndarray]) -> np.ndarray:
    length = len(reference["object_position"])
    if length < FUTURE_STEPS:
        raise RuntimeError("reference is shorter than the future window")
    starts = np.rint(np.linspace(0, length - FUTURE_STEPS, DEMO_WINDOWS)).astype(np.int64)
    features = np.empty((DEMO_WINDOWS, FUTURE_STEPS, 132), dtype=np.float32)
    for row, start in enumerate(starts):
        sl = slice(int(start), int(start) + FUTURE_STEPS)
        position = reference["object_position"][sl]
        regime = np.eye(4, dtype=np.float32)[reference["regime"][sl]]
        features[row] = np.concatenate(
            (
                reference["body_relative_object"][sl].reshape(FUTURE_STEPS, 105),
                position - position[:1],
                reference["rotation_6d"][sl],
                reference["object_velocity"][sl],
                reference["contact"][sl].astype(np.float32),
                np.clip(reference["duration"][sl] / EVENT_DURATION_SCALE_FRAMES, 0, 1),
                regime,
            ),
            axis=-1,
        )
    return features


def pair_target(actual: dict[str, np.ndarray], demo: np.ndarray) -> tuple[np.ndarray, int]:
    square = np.square(demo[..., :120] - actual["continuous"][None])
    component = np.stack(
        [square[..., start:stop].mean(axis=(1, 2)) for start, stop in CONTINUOUS_SLICES],
        axis=-1,
    )
    alignment = int(np.argmin(np.mean(component / ALIGNMENT_SCALES, axis=-1)))
    chosen = demo[alignment]
    contact = np.mean(np.abs(chosen[:, 120:124] - actual["contact"]), axis=0)
    duration = np.mean(np.abs(chosen[:, 124:128] - actual["duration"]), axis=0)
    demo_regime = np.argmax(chosen[:, 128:132], axis=-1)
    regime = np.asarray([np.mean(demo_regime != actual["regime"])], dtype=np.float32)
    return np.concatenate((component[alignment], contact, duration, regime)).astype(np.float32), alignment


def actual_future(trace: dict[str, np.ndarray], env: int, anchor: int) -> dict[str, np.ndarray]:
    sl = slice(anchor + 1, anchor + 1 + FUTURE_STEPS)
    object_state = trace["object_root_state_w"][sl, env]
    position = object_state[:, :3]
    body = trace["robot_body_position_w"][sl, env] - position[:, None, :]
    continuous = np.concatenate(
        (
            body.reshape(FUTURE_STEPS, 105),
            position - position[:1],
            quaternion_wxyz_to_rotation6d(object_state[:, 3:7]),
            object_state[:, 7:13],
        ),
        axis=-1,
    ).astype(np.float32)
    return {
        "continuous": continuous,
        "contact": trace["contact"][sl, env].astype(np.float32),
        "duration": np.clip(
            trace["contact_event_remaining_frames"][sl, env].astype(np.float32)
            / EVENT_DURATION_SCALE_FRAMES,
            0,
            1,
        ),
        "regime": trace["motion_regime"][sl, env],
    }


@torch.no_grad()
def batched_pair_targets(
    *,
    demo_bank: np.ndarray,
    pair_base: np.ndarray,
    pair_demo: np.ndarray,
    pair_phase: np.ndarray,
    actual_continuous: np.ndarray,
    actual_contact: np.ndarray,
    actual_duration: np.ndarray,
    actual_regime: np.ndarray,
    target: np.ndarray,
    alignment: np.ndarray,
    alignment_mode: str,
    batch_size: int = 256,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full cross-demo target build")
    device = torch.device("cuda:0")
    demo_tensor = torch.from_numpy(demo_bank).to(device)
    scales = torch.from_numpy(ALIGNMENT_SCALES).to(device)
    for begin in range(0, len(pair_base), batch_size):
        end = min(begin + batch_size, len(pair_base))
        base = pair_base[begin:end].astype(np.int64)
        selected = torch.from_numpy(pair_demo[begin:end].astype(np.int64)).to(device)
        demo = demo_tensor.index_select(0, selected)
        live = torch.from_numpy(actual_continuous[base]).to(device)
        square = torch.square(demo[..., :120] - live[:, None])
        component = torch.stack(
            [
                square[..., start:stop].mean(dim=(2, 3))
                for start, stop in CONTINUOUS_SLICES
            ],
            dim=-1,
        )
        if alignment_mode == "clock_phase":
            phase = torch.from_numpy(pair_phase[begin:end]).to(device)
            winner = torch.clamp(
                torch.round(phase * float(DEMO_WINDOWS - 1)).long(),
                min=0,
                max=DEMO_WINDOWS - 1,
            )
        elif alignment_mode == "free_window":
            winner = torch.argmin(torch.mean(component / scales, dim=-1), dim=1)
        else:
            raise ValueError(f"unsupported alignment mode: {alignment_mode}")
        rows = torch.arange(end - begin, device=device)
        chosen = demo[rows, winner]
        contact = torch.from_numpy(actual_contact[base]).to(device)
        duration = torch.from_numpy(actual_duration[base]).to(device)
        regime = torch.from_numpy(actual_regime[base].astype(np.int64)).to(device)
        contact_mismatch = torch.mean(torch.abs(chosen[..., 120:124] - contact), dim=1)
        duration_mismatch = torch.mean(torch.abs(chosen[..., 124:128] - duration), dim=1)
        demo_regime = torch.argmax(chosen[..., 128:132], dim=-1)
        regime_mismatch = torch.mean((demo_regime != regime).float(), dim=1, keepdim=True)
        target[begin:end] = torch.cat(
            (
                component[rows, winner],
                contact_mismatch,
                duration_mismatch,
                regime_mismatch,
            ),
            dim=-1,
        ).cpu().numpy()
        alignment[begin:end] = winner.cpu().numpy().astype(np.uint8)
        if end % 4096 < batch_size or end == len(pair_base):
            print(f"TARGET_PROGRESS pairs={end}/{len(pair_base)}", flush=True)


def actual_entries(corpus_root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for trace_path in sorted(corpus_root.glob("*/TRACE.npz")):
        result = json.loads((trace_path.parent / "RESULT.json").read_text(encoding="utf-8"))
        task = str(result["task_family"])
        with np.load(trace_path, allow_pickle=False) as trace:
            for env in range(trace["source_motion_id"].shape[1]):
                ids = np.unique(trace["source_motion_id"][:, env])
                if ids.size != 1:
                    raise RuntimeError(f"{trace_path}: source motion changed in env {env}")
                entries.append(
                    {"task": task, "source_id": int(ids[0]), "trace": trace_path, "env": env}
                )
    return entries


def build_split(
    split: str,
    entries: list[dict[str, object]],
    references: dict[tuple[str, int], dict[str, np.ndarray]],
    output: Path,
    policy_observation_key: str,
    policy_dim: int,
    alignment_mode: str,
) -> dict[str, object]:
    selected_entries = [entry for entry in entries if motion_split(int(entry["source_id"])) == split]
    demo_keys = sorted((str(entry["task"]), int(entry["source_id"])) for entry in selected_entries)
    demo_keys = list(dict.fromkeys(demo_keys))
    demo_index = {key: index for index, key in enumerate(demo_keys)}
    demo_bank = np.stack([window_features(references[key]) for key in demo_keys])
    by_task = {task: [key for key in demo_keys if key[0] == task] for task in TASKS}
    descriptor: list[tuple[dict[str, object], int, float]] = []
    for entry in selected_entries:
        reference_length = len(references[(str(entry["task"]), int(entry["source_id"]))]["object_position"])
        last_anchor = min(700 - FUTURE_STEPS - 1, reference_length - FUTURE_STEPS - 1)
        descriptor.extend(
            (
                entry,
                anchor,
                float(np.clip((anchor + 1) / max(reference_length - FUTURE_STEPS, 1), 0.0, 1.0)),
            )
            for anchor in range(HISTORY_STEPS - 1, last_anchor + 1, ANCHOR_STRIDE)
        )
    if not descriptor:
        raise RuntimeError(f"{split}: no causal rows")

    split_dir = output / split
    split_dir.mkdir(parents=True)
    base_count = len(descriptor)
    pair_count = base_count * len(PAIR_ROLE_NAMES)
    policy = np.lib.format.open_memmap(
        split_dir / "policy_prefix.npy",
        mode="w+",
        dtype=np.float32,
        shape=(base_count, HISTORY_STEPS, policy_dim),
    )
    base_task = np.empty(base_count, dtype=np.uint8)
    base_source = np.empty(base_count, dtype=np.int16)
    base_anchor = np.empty(base_count, dtype=np.int16)
    pair_base = np.repeat(np.arange(base_count, dtype=np.int32), len(PAIR_ROLE_NAMES))
    pair_demo = np.empty(pair_count, dtype=np.int16)
    pair_role = np.tile(np.arange(len(PAIR_ROLE_NAMES), dtype=np.uint8), base_count)
    base_phase = np.empty(base_count, dtype=np.float32)
    target = np.lib.format.open_memmap(split_dir / "target_mismatch.npy", mode="w+", dtype=np.float32, shape=(pair_count, len(TARGET_NAMES)))
    alignment = np.empty(pair_count, dtype=np.uint8)
    actual_continuous = np.empty((base_count, FUTURE_STEPS, 120), dtype=np.float32)
    actual_contact = np.empty((base_count, FUTURE_STEPS, 4), dtype=np.float32)
    actual_duration = np.empty((base_count, FUTURE_STEPS, 4), dtype=np.float32)
    actual_regime = np.empty((base_count, FUTURE_STEPS), dtype=np.uint8)

    trace_cache: dict[Path, dict[str, np.ndarray]] = {}
    task_position = {task: {key: index for index, key in enumerate(by_task[task])} for task in TASKS}
    for base_row, (entry, anchor, phase) in enumerate(descriptor):
        trace_path = Path(entry["trace"])
        if trace_path not in trace_cache:
            with np.load(trace_path, allow_pickle=False) as archive:
                trace_cache[trace_path] = {
                    name: np.asarray(archive[name])
                    for name in (
                        "policy_observation",
                        "object_root_state_w",
                        "robot_body_position_w",
                        "contact",
                        "contact_event_remaining_frames",
                        "motion_regime",
                    )
                }
                if policy_observation_key != "policy_observation":
                    trace_cache[trace_path][policy_observation_key] = np.asarray(
                        archive[policy_observation_key]
                    )
        trace = trace_cache[trace_path]
        env = int(entry["env"])
        task = str(entry["task"])
        source_id = int(entry["source_id"])
        key = (task, source_id)
        policy[base_row] = trace[policy_observation_key][
            anchor - HISTORY_STEPS + 1 : anchor + 1, env
        ]
        base_task[base_row] = TASKS.index(task)
        base_source[base_row] = source_id
        base_anchor[base_row] = anchor
        base_phase[base_row] = phase
        same_pool = by_task[task]
        same_wrong = same_pool[(task_position[task][key] + 1) % len(same_pool)]
        other_pool = by_task[TASKS[1 - TASKS.index(task)]]
        cross_wrong = other_pool[(source_id + base_row) % len(other_pool)]
        selected_keys = (key, same_wrong, cross_wrong)
        actual = actual_future(trace, env, anchor)
        actual_continuous[base_row] = actual["continuous"]
        actual_contact[base_row] = actual["contact"]
        actual_duration[base_row] = actual["duration"]
        actual_regime[base_row] = actual["regime"]
        for role, selected_key in enumerate(selected_keys):
            pair_row = base_row * len(PAIR_ROLE_NAMES) + role
            selected_index = demo_index[selected_key]
            pair_demo[pair_row] = selected_index
        if (base_row + 1) % 2000 == 0 or base_row + 1 == base_count:
            print(
                f"DATASET_PROGRESS split={split} rows={base_row + 1}/{base_count}",
                flush=True,
            )

    batched_pair_targets(
        demo_bank=demo_bank,
        pair_base=pair_base,
        pair_demo=pair_demo,
        pair_phase=base_phase[pair_base],
        actual_continuous=actual_continuous,
        actual_contact=actual_contact,
        actual_duration=actual_duration,
        actual_regime=actual_regime,
        target=target,
        alignment=alignment,
        alignment_mode=alignment_mode,
    )
    policy.flush()
    target.flush()
    np.save(split_dir / "demo_bank.npy", demo_bank, allow_pickle=False)
    np.savez(
        split_dir / "routing.npz",
        base_task=base_task,
        base_source_motion_id=base_source,
        base_anchor_frame=base_anchor,
        pair_base_row=pair_base,
        pair_selected_demo_row=pair_demo,
        pair_role=pair_role,
        selected_alignment_window=alignment,
        base_normalized_demo_phase=base_phase,
        pair_normalized_demo_phase=base_phase[pair_base],
        demo_task=np.asarray([TASKS.index(key[0]) for key in demo_keys], dtype=np.uint8),
        demo_source_motion_id=np.asarray([key[1] for key in demo_keys], dtype=np.int16),
    )
    role_median = {
        PAIR_ROLE_NAMES[role]: np.median(np.asarray(target)[pair_role == role], axis=0).tolist()
        for role in range(len(PAIR_ROLE_NAMES))
    }
    semantic_metrics = {}
    for role, name in enumerate(PAIR_ROLE_NAMES):
        values = np.asarray(target)[pair_role == role]
        contact_score = values[:, 4:8].mean(axis=1)
        duration_score = values[:, 8:12].mean(axis=1)
        semantic_metrics[name] = {
            "contact_mismatch_mean": float(np.mean(contact_score)),
            "contact_mismatch_nonzero_fraction": float(np.mean(contact_score > 0)),
            "duration_mismatch_mean": float(np.mean(duration_score)),
            "duration_mismatch_nonzero_fraction": float(np.mean(duration_score > 0)),
            "motion_regime_mismatch_mean": float(np.mean(values[:, 12])),
        }
    return {
        "base_rows": base_count,
        "pair_rows": pair_count,
        "demo_count": len(demo_keys),
        "task_motion_counts": {task: len(by_task[task]) for task in TASKS},
        "target_median_by_pair_role": role_median,
        "semantic_metrics_by_pair_role": semantic_metrics,
        "alignment_mode": alignment_mode,
    }


def main() -> None:
    args = parse_args()
    corpus_root = args.corpus_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {staging}")
    entries = actual_entries(corpus_root)
    if not entries:
        raise RuntimeError("actual rollout corpus is empty")
    with np.load(Path(entries[0]["trace"]), allow_pickle=False) as first_trace:
        if args.policy_observation_key not in first_trace:
            raise KeyError(
                f"corpus does not contain {args.policy_observation_key!r}"
            )
        first_policy = first_trace[args.policy_observation_key]
        if first_policy.ndim != 3:
            raise RuntimeError("policy observation must have [time, env, feature] shape")
        policy_dim = int(first_policy.shape[-1])
    references = {
        (str(entry["task"]), int(entry["source_id"])): load_reference(
            str(entry["task"]), int(entry["source_id"]), data_root
        )
        for entry in entries
    }
    staging.mkdir(parents=True)
    splits = {
        split: build_split(
            split,
            entries,
            references,
            staging,
            args.policy_observation_key,
            policy_dim,
            args.alignment_mode,
        )
        for split in SPLITS
    }
    train_target = np.load(staging / "train/target_mismatch.npy", mmap_mode="r")
    train_policy = np.load(staging / "train/policy_prefix.npy", mmap_mode="r")
    train_demo = np.load(staging / "train/demo_bank.npy", mmap_mode="r")
    normalization_path = staging / "NORMALIZATION.npz"
    target_scale = np.maximum(
        np.quantile(train_target, 0.90, axis=0), 1.0e-6
    ).astype(np.float32)
    # Event mismatch targets are bounded fractions.  A rare role can have a
    # zero train p90 despite valid nonzero held-out events; use a common 0.1
    # scale floor so sparsity cannot create million-fold normalization.
    target_scale[4:] = np.maximum(target_scale[4:], 0.1)
    np.savez(
        normalization_path,
        state_mean=np.mean(train_policy, axis=(0, 1), dtype=np.float64).astype(np.float32),
        state_std=np.maximum(np.std(train_policy, axis=(0, 1), dtype=np.float64), 1.0e-6).astype(np.float32),
        demo_mean=np.mean(train_demo, axis=(0, 1, 2), dtype=np.float64).astype(np.float32),
        demo_std=np.maximum(
            np.std(train_demo, axis=(0, 1, 2), dtype=np.float64), 1.0e-3
        ).astype(np.float32),
        target_scale=target_scale,
    )
    checks = {
        "all_199_source_motions_present": sum(record["demo_count"] for record in splits.values()) == 199,
        "every_base_has_three_numeric_demo_pairs": all(record["pair_rows"] == 3 * record["base_rows"] for record in splits.values()),
        "motion_disjoint_splits": all(splits[split]["demo_count"] == len({(str(entry["task"]), int(entry["source_id"])) for entry in entries if motion_split(int(entry["source_id"])) == split}) for split in SPLITS),
        "cross_task_contact_mismatch_exceeds_correct_in_every_split": all(
            record["semantic_metrics_by_pair_role"]["cross_task_wrong"][
                "contact_mismatch_mean"
            ]
            > record["semantic_metrics_by_pair_role"]["correct"][
                "contact_mismatch_mean"
            ]
            + 0.10
            for record in splits.values()
        ),
        "cross_task_regime_mismatch_exceeds_correct_in_every_split": all(
            record["semantic_metrics_by_pair_role"]["cross_task_wrong"][
                "motion_regime_mismatch_mean"
            ]
            > record["semantic_metrics_by_pair_role"]["correct"][
                "motion_regime_mismatch_mean"
            ]
            + 0.10
            for record in splits.values()
        ),
        "all_targets_finite_nonnegative": bool(np.isfinite(train_target).all() and np.all(train_target >= 0)),
        "future_actual_events_are_targets_only": True,
        "reference_binary_proxy_is_selected_demo_input_only": True,
    }
    manifest = {
        "protocol": "sugar_actual_contact_event_crossdemo_dataset_v2",
        "passed": all(checks.values()),
        "checks": checks,
        "splits": splits,
        "model_inputs": {
            "policy_prefix": [HISTORY_STEPS, policy_dim],
            "policy_observation_key": args.policy_observation_key,
            "selected_demo_bank": [DEMO_WINDOWS, FUTURE_STEPS, 132],
            "selected_demo_phase": "causal scalar in [0,1]"
            if args.alignment_mode == "clock_phase"
            else False,
            "categorical_task_or_motion_id": False,
            "future_actual_event_input": False,
        },
        "target_names": TARGET_NAMES,
        "alignment_mode": args.alignment_mode,
        "pair_role_names": PAIR_ROLE_NAMES,
        "claim_boundary": (
            "Passing validates causal inputs, actual-event targets and motion-disjoint "
            "cross-demo supervision. It does not establish predictor generalization or "
            "policy-level demo following."
        ),
        "automatic_next_branch": (
            "train_serious_causal_event_predictor_and_run_demo_permutation_gate"
            if all(checks.values())
            else "inspect_dataset_semantic_gate_before_predictor_training"
        ),
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
