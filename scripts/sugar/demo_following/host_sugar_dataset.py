#!/usr/bin/env python3
"""Faithful SUGAR-to-HOST dataset glue for the official 9B HOST model.

This module does not implement or replace any HOST component.  It converts the
immutable Plan-17 SUGAR corpus into the tensor contract consumed by the released
HOST dataset/model path.  In particular, the released loader's effective 24 Hz
action horizon is preserved: 5 agent-video frames, 21 task-video frames, 24
31-channel action tokens, and one 121-D causal proprio state.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / (
    "experiments/demo_following/zero_wam_official_v1/icl_manifest_v2/"
    "ICL_MANIFEST.jsonl"
)
DEFAULT_STATS = ROOT / (
    "experiments/demo_following/host_official_v1/sugar_adapter_v1/"
    "HOST_SUGAR_TRAIN_STATS.npz"
)

ACTION_DIM = 29
ACTION_TOKEN_DIM = 31
PROPRIO_DIM = 121
ACTION_HORIZON = 24
ATOMIC_STRIDE = 5
WINDOWS_PER_MOTION = 140
AGENT_VIDEO_FRAMES = 5
TASK_VIDEO_FRAMES = 21
IMAGE_SIZE = 224
NEUTRAL_PROMPT = "Follow the demonstrated motion."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 199:
        raise RuntimeError(f"immutable HOST/SUGAR manifest count drift: {len(rows)}")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["split"], row["task"])] += 1
    expected = {
        ("train", "CarryBox"): 80,
        ("train", "KickBox"): 80,
        ("validation", "CarryBox"): 10,
        ("validation", "KickBox"): 10,
        ("test", "CarryBox"): 10,
        ("test", "KickBox"): 9,
    }
    if dict(counts) != expected:
        raise RuntimeError(f"immutable motion-disjoint split drift: {dict(counts)}")
    return rows


def _array_name(row: dict[str, Any], field: str) -> str:
    value = row["action_target"][field]
    return value["array"] if isinstance(value, dict) else value


def _selected_array(
    archive: np.lib.npyio.NpzFile, row: dict[str, Any], field: str
) -> np.ndarray:
    name = _array_name(row, field)
    index = int(row["action_target"]["environment_index"])
    value = np.asarray(archive[name])
    if value.ndim < 2 or index >= value.shape[1]:
        raise RuntimeError(f"archive field geometry drift: {name} {value.shape} env={index}")
    return np.asarray(value[:, index])


@dataclass(frozen=True)
class HostSugarStats:
    action_min: np.ndarray
    action_range: np.ndarray
    proprio_min: np.ndarray
    proprio_range: np.ndarray
    action_constant: np.ndarray
    proprio_constant: np.ndarray
    train_transition_count: int
    manifest_sha256: str

    @classmethod
    def load(cls, path: Path = DEFAULT_STATS) -> "HostSugarStats":
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                action_min=np.asarray(archive["action_min"], dtype=np.float32),
                action_range=np.asarray(archive["action_range"], dtype=np.float32),
                proprio_min=np.asarray(archive["proprio_min"], dtype=np.float32),
                proprio_range=np.asarray(archive["proprio_range"], dtype=np.float32),
                action_constant=np.asarray(archive["action_constant"], dtype=bool),
                proprio_constant=np.asarray(archive["proprio_constant"], dtype=bool),
                train_transition_count=int(archive["train_transition_count"].item()),
                manifest_sha256=str(archive["manifest_sha256"].item()),
            )

    def normalize_action(self, value: np.ndarray) -> np.ndarray:
        result = 2.0 * (value.astype(np.float32) - self.action_min) / self.action_range - 1.0
        result[..., self.action_constant] = 0.0
        return np.clip(result, -1.0, 1.0)

    def normalize_proprio(self, value: np.ndarray) -> np.ndarray:
        result = 2.0 * (value.astype(np.float32) - self.proprio_min) / self.proprio_range - 1.0
        result[..., self.proprio_constant] = 0.0
        return np.clip(result, -1.0, 1.0)


def build_train_stats(
    output_path: Path = DEFAULT_STATS,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> HostSugarStats:
    """Compute official-style per-dimension min/range using train motions only."""

    manifest_path = Path(manifest_path)
    rows = [row for row in load_manifest(manifest_path) if row["split"] == "train"]
    action_min = np.full(ACTION_DIM, np.inf, dtype=np.float64)
    action_max = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    proprio_min = np.full(PROPRIO_DIM, np.inf, dtype=np.float64)
    proprio_max = np.full(PROPRIO_DIM, -np.inf, dtype=np.float64)
    transition_count = 0

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["action_target"]["trace_path"]].append(row)
    for trace_path, trace_rows in sorted(grouped.items()):
        with np.load(ROOT / trace_path, allow_pickle=False) as archive:
            action_name = _array_name(trace_rows[0], "executed_action")
            action_all = np.asarray(archive[action_name])
            proprio_all = np.asarray(archive["goal_policy_core_observation"])
            for row in trace_rows:
                # Action t is causally paired with the archived post-state core t-1.
                # The one pre-rollout state for action 0 was not archived and is excluded.
                index = int(row["action_target"]["environment_index"])
                action = np.asarray(action_all[1:, index])
                proprio = np.asarray(proprio_all[:-1, index])
                if action.shape != (699, ACTION_DIM) or proprio.shape != (699, PROPRIO_DIM):
                    raise RuntimeError(
                        f"causal train transition geometry drift: {action.shape} {proprio.shape}"
                    )
                if not np.isfinite(action).all() or not np.isfinite(proprio).all():
                    raise RuntimeError("non-finite immutable training transition")
                action_min = np.minimum(action_min, action.min(axis=0))
                action_max = np.maximum(action_max, action.max(axis=0))
                proprio_min = np.minimum(proprio_min, proprio.min(axis=0))
                proprio_max = np.maximum(proprio_max, proprio.max(axis=0))
                transition_count += len(action)

    if transition_count != 160 * 699:
        raise RuntimeError(f"exact causal train transition count drift: {transition_count}")
    action_span = action_max - action_min
    proprio_span = proprio_max - proprio_min
    action_constant = action_span < 1.0e-8
    proprio_constant = proprio_span < 1.0e-8
    action_span[action_constant] = 1.0
    proprio_span[proprio_constant] = 1.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        action_min=action_min.astype(np.float32),
        action_range=action_span.astype(np.float32),
        proprio_min=proprio_min.astype(np.float32),
        proprio_range=proprio_span.astype(np.float32),
        action_constant=action_constant,
        proprio_constant=proprio_constant,
        train_transition_count=np.asarray(transition_count, dtype=np.int64),
        manifest_sha256=np.asarray(_sha256(manifest_path)),
        normalization_contract=np.asarray(
            "official_absolute_minmax_2x_minus_min_over_range_minus_1_clipped"
        ),
        causal_pairing_contract=np.asarray(
            "executed_action[t]_with_goal_policy_core_observation[t-1]_for_t_1_through_699"
        ),
    )
    return HostSugarStats.load(output_path)


class _ArchiveCache:
    def __init__(self, capacity: int = 2) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def get(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        path = ROOT / row["action_target"]["trace_path"]
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        with np.load(path, allow_pickle=False) as archive:
            value = {
                "executed_action": np.asarray(archive[_array_name(row, "executed_action")]),
                "core": np.asarray(archive["goal_policy_core_observation"]),
                "time": np.asarray(archive["transition_time_s"]),
            }
        self._cache[path] = value
        self._cache.move_to_end(path)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return value


def _load_image(path: str) -> torch.Tensor:
    with Image.open(ROOT / path) as image:
        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC
        )
        value = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(value).permute(2, 0, 1).contiguous()


def _nearest_indices(timestamps: Iterable[float], query: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(list(timestamps), dtype=np.float64)
    right = np.searchsorted(timestamps, query, side="left").clip(0, len(timestamps) - 1)
    left = (right - 1).clip(0, len(timestamps) - 1)
    choose_left = np.abs(query - timestamps[left]) <= np.abs(timestamps[right] - query)
    return np.where(choose_left, left, right).astype(np.int64)


class HostSugarDataset(Dataset[dict[str, Any]]):
    """Official HOST tensor contract over all fixed five-action SUGAR intervals."""

    def __init__(
        self,
        split: str,
        stats_path: Path = DEFAULT_STATS,
        manifest_path: Path = DEFAULT_MANIFEST,
        prompt_mode: str = "matching_alternate",
        epoch: int = 0,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {split}")
        if prompt_mode not in {
            "matching_alternate",
            "wrong_task",
            "same_source",
            "reversed_matching_alternate",
        }:
            raise ValueError(f"unsupported prompt intervention: {prompt_mode}")
        self.manifest_path = Path(manifest_path)
        self.rows = [row for row in load_manifest(self.manifest_path) if row["split"] == split]
        self.stats = HostSugarStats.load(stats_path)
        if self.stats.manifest_sha256 != _sha256(self.manifest_path):
            raise RuntimeError("normalization statistics belong to a different manifest")
        self.split = split
        self.prompt_mode = prompt_mode
        self.epoch = int(epoch)
        self.cache = _ArchiveCache()
        self.task_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.task_indices[row["task"]].append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows) * WINDOWS_PER_MOTION

    def _prompt_row(self, target_index: int) -> dict[str, Any]:
        target = self.rows[target_index]
        task = target["task"]
        if self.prompt_mode == "same_source":
            return target
        prompt_task = task
        if self.prompt_mode == "wrong_task":
            prompt_task = "KickBox" if task == "CarryBox" else "CarryBox"
        candidates = self.task_indices[prompt_task]
        position = candidates.index(target_index) if prompt_task == task else target_index
        # Epoch-varying cyclic pairing covers many prompt identities without random leakage.
        offset = 1 + (self.epoch * 17) % max(len(candidates) - 1, 1)
        selected = candidates[(position + offset) % len(candidates)]
        if prompt_task == task and self.rows[selected]["source_motion_id"] == target["source_motion_id"]:
            selected = candidates[(candidates.index(selected) + 1) % len(candidates)]
        return self.rows[selected]

    def sample_metadata(self, index: int) -> dict[str, Any]:
        target_index, interval = divmod(int(index), WINDOWS_PER_MOTION)
        target = self.rows[target_index]
        prompt = self._prompt_row(target_index)
        action_start = 1 + interval * ATOMIC_STRIDE
        action_stop = min(action_start + ACTION_HORIZON, 700)
        valid_len = action_stop - action_start
        if valid_len <= 0:
            raise RuntimeError(f"empty HOST action window: {action_start}:{action_stop}")
        return {
            "target_index": target_index,
            "interval": interval,
            "target": target,
            "prompt": prompt,
            "action_start": action_start,
            "action_stop": action_stop,
            "valid_len": valid_len,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        meta = self.sample_metadata(index)
        target = meta["target"]
        prompt = meta["prompt"]
        action_start = meta["action_start"]
        action_stop = meta["action_stop"]
        valid_len = meta["valid_len"]
        archive = self.cache.get(target)
        env_index = int(target["action_target"]["environment_index"])

        raw_action = archive["executed_action"][action_start:action_stop, env_index]
        if raw_action.shape != (valid_len, ACTION_DIM):
            raise RuntimeError(f"action window geometry drift: {raw_action.shape}")
        action_physical = np.empty((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
        action_physical[:valid_len] = self.stats.normalize_action(raw_action)
        action_physical[valid_len:] = action_physical[valid_len - 1]

        # Exact causal pre-state: post-state core at t-1 is the state before action t.
        raw_proprio = archive["core"][action_start - 1, env_index]
        proprio = self.stats.normalize_proprio(raw_proprio)[None]

        p_start = action_start / 699.0
        p_end = min(action_stop - 1, 699) / 699.0
        nominal_delta = (ACTION_HORIZON - 1) / 699.0
        task_width = 5.0 * nominal_delta
        window_start = float(np.clip(p_start - 2.0 * nominal_delta, 0.0, 1.0 - task_width))
        window_end = window_start + task_width
        progress_start = (p_start - window_start) / task_width
        progress_end = (p_end - window_start) / task_width
        action_progress = np.linspace(
            progress_start, progress_end, valid_len, dtype=np.float32
        )
        action_progress = action_progress * 2.0 - 1.0
        progress_padded = np.empty(ACTION_HORIZON, dtype=np.float32)
        progress_padded[:valid_len] = action_progress
        progress_padded[valid_len:] = action_progress[-1]
        action_mask = np.zeros(ACTION_HORIZON, dtype=np.float32)
        action_mask[:valid_len] = 1.0
        action = np.concatenate(
            (action_physical, progress_padded[:, None], action_mask[:, None]), axis=-1
        )
        action_is_pad = np.arange(ACTION_HORIZON) >= valid_len

        # Robot video is sampled by the immutable physical timestamps, not frame number guesses.
        query_times = np.linspace(
            archive["time"][action_start],
            archive["time"][action_stop - 1],
            AGENT_VIDEO_FRAMES,
        )
        robot_indices = _nearest_indices(target["robot_target"]["frame_timestamps_s"], query_times)
        robot_paths = target["robot_target"]["frame_paths"]
        video = torch.stack([_load_image(robot_paths[i]) for i in robot_indices], dim=1)
        nominal_offsets = np.rint(
            np.linspace(0, ACTION_HORIZON - 1, AGENT_VIDEO_FRAMES)
        ).astype(np.int64)
        image_is_pad = action_start + nominal_offsets >= 700

        task_progress = np.linspace(window_start, window_end, TASK_VIDEO_FRAMES)
        prompt_paths = list(prompt["prompt"]["frame_paths"])
        prompt_indices = np.rint(task_progress * (len(prompt_paths) - 1)).astype(np.int64)
        if self.prompt_mode == "reversed_matching_alternate":
            prompt_indices = (len(prompt_paths) - 1) - prompt_indices
        task_video = torch.stack([_load_image(prompt_paths[i]) for i in prompt_indices], dim=1)

        tensors = (video, task_video, action, proprio)
        if not all(np.isfinite(np.asarray(value)).all() for value in (action, proprio)):
            raise RuntimeError("non-finite HOST SUGAR numeric tensor")
        if not torch.isfinite(video).all() or not torch.isfinite(task_video).all():
            raise RuntimeError("non-finite HOST SUGAR video tensor")
        if video.shape != (3, 5, 224, 224) or task_video.shape != (3, 21, 224, 224):
            raise RuntimeError(f"HOST video geometry drift: {video.shape} {task_video.shape}")
        if action.shape != (24, 31) or proprio.shape != (1, 121):
            raise RuntimeError(f"HOST action/proprio geometry drift: {action.shape} {proprio.shape}")

        return {
            "video": video,
            "task_video": task_video,
            "task_video_dropped": False,
            "action": torch.from_numpy(action),
            "proprio": torch.from_numpy(proprio),
            "prompt": NEUTRAL_PROMPT,
            "image_is_pad": torch.from_numpy(image_is_pad),
            "action_is_pad": torch.from_numpy(action_is_pad),
            "progress_gt": torch.tensor([progress_start, progress_end], dtype=torch.float32),
            "task_video_start": int(round(progress_start * (TASK_VIDEO_FRAMES - 1))),
            "task_video_end": int(round(progress_end * (TASK_VIDEO_FRAMES - 1))) + 1,
            "target_task": target["task"],
            "target_source_motion_id": int(target["source_motion_id"]),
            "prompt_source_motion_id": int(prompt["source_motion_id"]),
            "action_start": action_start,
            "valid_action_count": valid_len,
            "robot_frame_indices": torch.from_numpy(robot_indices),
            "task_frame_indices": torch.from_numpy(prompt_indices.copy()),
        }


class HostSugarTrainingDataset(HostSugarDataset):
    """HOST dataset with the one exact cached neutral UMT5 context attached at collate."""

    def __init__(self, *args, context_path: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        payload = torch.load(context_path, map_location="cpu", weights_only=True)
        if payload.get("prompt") != NEUTRAL_PROMPT:
            raise RuntimeError("HOST neutral context prompt drift")
        context = payload.get("context")
        mask = payload.get("mask")
        if not isinstance(context, torch.Tensor) or context.shape != (512, 4096):
            raise RuntimeError("HOST neutral context geometry drift")
        if not isinstance(mask, torch.Tensor) or mask.shape != (512,) or not mask.any():
            raise RuntimeError("HOST neutral context mask drift")
        self.neutral_context = context.to(torch.bfloat16).contiguous()
        self.neutral_context_mask = mask.to(torch.bool).contiguous()

    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = default_collate(samples)
        batch_size = len(samples)
        batch["context"] = self.neutral_context.unsqueeze(0).expand(batch_size, -1, -1)
        batch["context_mask"] = self.neutral_context_mask.unsqueeze(0).expand(
            batch_size, -1
        )
        # The official loss accepts an iterable of per-sample booleans.
        batch["task_video_dropped"] = [False] * batch_size
        return batch
