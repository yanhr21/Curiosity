"""Exact SUGAR index adapter for the released BPP data path.

The classes here only choose target and prompt indices.  All item loading,
lower-rate RGB mapping, action padding, prompt loading/chunking, tensor
conversion, and batch collation remain in the official BPP implementation.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from behavior_prompting.common.replay_buffer import ReplayBuffer
from behavior_prompting.train_network.common.sampler import SequenceSampler
from behavior_prompting.train_network.dataset.libero_replay_image_dataset import (
    LiberoReplayImageDataset,
)


ROOT = Path(__file__).resolve().parents[3]
SENSOR_PROTOCOL = "sugar_bpp_sensorimotor_corpus_v2"
SCHEDULE_PROTOCOL = "sugar_bpp_exact_training_schedule_v1"
REPLAY_PROTOCOL = "sugar_bpp_official_replay_buffer_v1"
EXPECTED_SCHEDULE_CONTRACT_SHA256 = (
    "bfe38f38b660d38305a5caba6c778997bd772bf69b1ab383362d9fdad33ec213"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = base / path
    if local.exists():
        return local.resolve()
    return (ROOT / path).resolve()


def _manifest_order(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["task_family"]),
        int(row["behavior_identity"]["source_motion_id"]),
        int(row["admitted_variant_rank"]),
    )


def _bpp_names(row: dict[str, Any]) -> tuple[str, str]:
    task = str(row["task_family"])
    source = int(row["behavior_identity"]["source_motion_id"])
    variant = int(row["candidate_variant_id"])
    task_name = f"{task}_motion{source:03d}"
    episode_name = f"{task_name}_variant{variant:02d}"
    if ":" in task_name or ":" in episode_name:
        raise ValueError("BPP identifiers may not contain ':'")
    return episode_name, task_name


class ExactDistinctVariantPairSampler(SequenceSampler):
    """Official SequenceSampler with only its pair-index choice replaced."""

    def __init__(
        self,
        *,
        schedule_by_epoch: dict[int, list[dict[str, Any]]],
        trajectory_to_task_idx: dict[tuple[str, int, int], int],
        **kwargs,
    ):
        # The released constructor prepares the ReplayBuffer, clock geometry,
        # prompt chunker, and standard pair-prompt fields.  Its random pair map
        # is immediately replaced before any item can be returned.
        super().__init__(**kwargs)
        self._schedule_by_epoch = schedule_by_epoch
        self._trajectory_to_task_idx = trajectory_to_task_idx
        self._active_epoch = -1
        self._active_microbatches: list[dict[str, Any]] = []
        self._apply_epoch(0)

    def _apply_epoch(self, epoch: int) -> None:
        rows = self._schedule_by_epoch.get(epoch)
        if rows is None or len(rows) != 5600:
            raise ValueError(f"exact schedule lacks epoch {epoch}")
        indices = np.empty((560000, 3), dtype=np.int64)
        prompt_indices = np.empty(560000, dtype=np.int64)
        flat_index = 0
        for expected_microbatch, row in enumerate(rows):
            if int(row["epoch_microbatch_index"]) != expected_microbatch:
                raise ValueError("epoch microbatch index is non-contiguous")
            task = str(row["task_label"])
            source = int(row["source_motion_id"])
            target_variant = int(row["target_variant_id"])
            prompt_variant = int(row["prompt_variant_id"])
            if target_variant == prompt_variant:
                raise ValueError("self-pair in exact schedule")
            target_task_idx = self._trajectory_to_task_idx[
                (task, source, target_variant)
            ]
            prompt_task_idx = self._trajectory_to_task_idx[
                (task, source, prompt_variant)
            ]
            if not bool(self.mask[target_task_idx]) or not bool(
                self.mask[prompt_task_idx]
            ):
                raise ValueError("held-out trajectory entered training schedule")
            anchors = [int(value) for value in row["target_anchor_indices"]]
            if len(anchors) != 100 or len(set(anchors)) != 100:
                raise ValueError("microbatch target-anchor geometry drift")
            if min(anchors) < 0 or max(anchors) >= 700:
                raise ValueError("target anchor outside causal trajectory")
            episode_idx = int(self.task_to_episode_idxs[target_task_idx])
            for anchor in anchors:
                indices[flat_index] = (anchor, episode_idx, target_task_idx)
                prompt_indices[flat_index] = prompt_task_idx
                flat_index += 1
        if flat_index != 560000:
            raise AssertionError("epoch exposure count drift")
        self.indices = indices
        self.prompt_indices = prompt_indices
        self._active_microbatches = rows
        self._active_epoch = epoch

    def shuffle_data_ordering(self, seed: int) -> None:
        # The official workspace calls this with its exact integer epoch.
        self._apply_epoch(int(seed))

    def requires_epoch_shuffle(self) -> bool:
        return True

    def sample_sequence(self, idx: int):
        result = super().sample_sequence(idx)
        microbatch_row = self._active_microbatches[idx // 100]
        anchor = int(microbatch_row["target_anchor_indices"][idx % 100])
        target_task_idx = int(self.indices[idx, 2])
        prompt_task_idx = int(self.prompt_indices[idx])
        result["metadata"].update(
            {
                "bpp_schedule_epoch": self._active_epoch,
                "bpp_epoch_microbatch_index": int(
                    microbatch_row["epoch_microbatch_index"]
                ),
                "bpp_global_microbatch_index": int(
                    microbatch_row["global_microbatch_index"]
                ),
                "bpp_target_anchor_50hz": anchor,
                "bpp_target_variant_id": int(
                    microbatch_row["target_variant_id"]
                ),
                "bpp_prompt_variant_id": int(
                    microbatch_row["prompt_variant_id"]
                ),
                "bpp_source_motion_id": int(
                    microbatch_row["source_motion_id"]
                ),
                "bpp_target_sensorimotor_row_sha256": str(
                    microbatch_row["target_sensorimotor_row_sha256"]
                ),
                "bpp_prompt_sensorimotor_row_sha256": str(
                    microbatch_row["prompt_sensorimotor_row_sha256"]
                ),
            }
        )
        result["prompt"]["metadata"].update(
            {
                "bpp_target_task_idx": target_task_idx,
                "bpp_prompt_task_idx": prompt_task_idx,
            }
        )
        return result


class SugarBppExactDataset(LiberoReplayImageDataset):
    """Released LIBERO dataset path bound to the frozen SUGAR schedule."""

    def __init__(
        self,
        *,
        replay_buffer_path: str,
        sensorimotor_result_path: str,
        schedule_result_path: str,
        **kwargs,
    ):
        replay_path = Path(replay_buffer_path).resolve()
        sensor_result_path = Path(sensorimotor_result_path).resolve()
        schedule_result_path = Path(schedule_result_path).resolve()
        replay_buffer = ReplayBuffer.create_from_path(str(replay_path), mode="r")
        if replay_buffer.root.attrs.get("protocol") != REPLAY_PROTOCOL:
            raise ValueError("official ReplayBuffer protocol drift")

        sensor_result = _read_json(sensor_result_path)
        if (
            sensor_result.get("protocol") != SENSOR_PROTOCOL
            or sensor_result.get("passed") is not True
        ):
            raise ValueError("sensorimotor admission is absent or failed")
        manifest_path = _resolve(
            sensor_result_path.parent,
            str(sensor_result["artifacts"]["sensorimotor_manifest"]),
        )
        manifest_rows = sorted(_read_jsonl(manifest_path), key=_manifest_order)
        if len(manifest_rows) != 995:
            raise ValueError("sensorimotor trajectory count drift")
        manifest_sha256 = _sha256_file(manifest_path)
        if replay_buffer.root.attrs.get(
            "sensorimotor_manifest_sha256"
        ) != manifest_sha256:
            raise ValueError("ReplayBuffer/manifest binding mismatch")

        schedule_result = _read_json(schedule_result_path)
        if (
            schedule_result.get("protocol") != SCHEDULE_PROTOCOL
            or schedule_result.get("passed") is not True
            or schedule_result.get("contract", {}).get("sha256")
            != EXPECTED_SCHEDULE_CONTRACT_SHA256
            or schedule_result.get("sensorimotor_admission", {}).get(
                "manifest_sha256"
            )
            != manifest_sha256
        ):
            raise ValueError("exact schedule admission or binding drift")
        schedule_path = _resolve(
            schedule_result_path.parent,
            str(schedule_result["schedule"]["path"]),
        )
        if _sha256_file(schedule_path) != schedule_result["schedule"]["sha256"]:
            raise ValueError("exact schedule file hash drift")
        schedule_rows = _read_jsonl(schedule_path)
        if len(schedule_rows) != 39200:
            raise ValueError("exact schedule microbatch count drift")
        schedule_by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in schedule_rows:
            schedule_by_epoch[int(row["epoch"])].append(row)
        if set(schedule_by_epoch) != set(range(7)):
            raise ValueError("exact seven-epoch schedule drift")
        for epoch, rows in schedule_by_epoch.items():
            rows.sort(key=lambda row: int(row["epoch_microbatch_index"]))
            if len(rows) != 5600:
                raise ValueError(f"epoch {epoch} microbatch count drift")
        scheduled_target_hashes_epoch0 = {
            (
                str(row["task_label"]),
                int(row["source_motion_id"]),
                int(row["target_variant_id"]),
            ): str(row["target_sensorimotor_row_sha256"])
            for row in schedule_by_epoch[0]
        }
        if len(scheduled_target_hashes_epoch0) != 800:
            raise ValueError("epoch-zero target trajectory coverage drift")

        episode_names = replay_buffer.episode_names[:].tolist()
        task_names = replay_buffer.task_names[:].tolist()
        if len(episode_names) != len(manifest_rows) or len(task_names) != len(
            manifest_rows
        ):
            raise ValueError("ReplayBuffer trajectory metadata length drift")
        training_split_info: dict[str, bool] = {}
        trajectory_to_task_idx: dict[tuple[str, int, int], int] = {}
        for task_idx, row in enumerate(manifest_rows):
            task = str(row["task_family"])
            source = int(row["behavior_identity"]["source_motion_id"])
            variant = int(row["candidate_variant_id"])
            expected_episode_name, expected_task_name = _bpp_names(row)
            if (
                str(task_names[task_idx]) != expected_task_name
                or str(episode_names[task_idx]) != expected_episode_name
            ):
                raise ValueError("ReplayBuffer row order or identity drift")
            key = (task, source, variant)
            if key in trajectory_to_task_idx:
                raise ValueError(f"duplicate trajectory identity: {key}")
            trajectory_to_task_idx[key] = task_idx
            training_split_info[
                f"{expected_episode_name}:{expected_task_name}"
            ] = row["split"] == "train"
            if row["split"] == "train" and scheduled_target_hashes_epoch0.get(
                key
            ) != _canonical_sha256(row):
                raise ValueError("train sensorimotor row hash absent from schedule")

        parsed_training_task_names = {
            key.split(":")[1]
            for key, is_train in training_split_info.items()
            if is_train
        }
        expected_training_task_names = {
            str(task_names[index])
            for index, row in enumerate(manifest_rows)
            if row["split"] == "train"
        }
        if parsed_training_task_names != expected_training_task_names:
            raise ValueError("official BPP training-split task-name parser drift")
        if len(parsed_training_task_names) != 160:
            raise ValueError("training split does not contain 160 source-motion tasks")

        # The released class still owns ReplayBuffer preparation, normalization,
        # item conversion, and every non-index sampling operation.
        kwargs.pop("dataset_path", None)
        kwargs.pop("replay_buffer", None)
        kwargs.pop("training_split_info", None)
        super().__init__(
            replay_buffer=replay_buffer,
            dataset_path=None,
            training_split_info=training_split_info,
            **kwargs,
        )
        exact_sampler = ExactDistinctVariantPairSampler(
            schedule_by_epoch=dict(schedule_by_epoch),
            trajectory_to_task_idx=trajectory_to_task_idx,
            mask=self.train_mask,
            **self.sampler_kwargs,
        )
        self.sampler = exact_sampler
        self.replay_buffer_path = replay_path
        self.sensorimotor_manifest_path = manifest_path
        self.schedule_path = schedule_path
