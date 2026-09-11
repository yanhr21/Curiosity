from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .config import PROJECT_ROOT, PaperZeroWAMConfig
from .schedule import select_overfit_schedule_row
from .prompt_coverage import prompt_coverage_record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


LATENT_PAYLOAD_SHAPES = {
    "prompt_latents": (48, 16, 20, 20),
    "reversed_prompt_latents": (48, 16, 20, 20),
    "robot_latents": (48, 36, 20, 20),
}


def latent_materialization_admission_is_exact(cache_root: Path, *, repaired: bool = False) -> bool:
    """Read the completed corpus-wide audit without rescanning large source arrays.

    Every training rank still validates each latent payload and action selector when
    it is first consumed.  This admission only avoids repeating the already
    completed 199-trajectory global scan before constructing the model.

    The per-field expectations below are properties of the corpus that produced
    them, so they are read from the audit record itself rather than pinned to
    the constants of one historical render.  The invariants that the *model*
    depends on -- protocol, pass flag, tensor geometry, finiteness, reversal
    non-identity, and split balance -- are still enforced exactly.
    """

    try:
        result = json.loads(
            (cache_root / "LATENT_RESULT.json").read_text(encoding="utf-8")
        )
        if not isinstance(result, dict):
            return False
        count = int(result.get("trajectory_count", -1))
        if count <= 0:
            return False
        if repaired and (result.get("prompt_coverage") != prompt_coverage_record()
                         or result.get("prompt_coverage_exact_count") != count):
            return False
        counts = result.get("counts")
        if not isinstance(counts, dict) or sum(counts.values()) != count:
            return False
        return bool(
            result.get("protocol")
            == "paper_zero_wam_wan22_latent_materialization_v1"
            and result.get("passed") is True
            and result.get("manifest_clock_exact") is True
            and int(result.get("prompt_robot_frame_path_overlap_count", -1)) == 0
            and result.get("action_trace_geometry_exact") is True
            and result.get("exact_manifest_path_set") is True
            and int(result.get("metadata_identity_exact_count", -1)) == count
            and result.get("prompt_shape") == [48, 16, 20, 20]
            and result.get("robot_shape") == [48, 36, 20, 20]
            and int(result.get("reversed_prompt_nonidentical_count", -1)) == count
            and int(
                result.get("finite_prompt_reversed_robot_trajectory_count", -1)
            )
            == count
            and result.get("all_latent_tensors_finite") is True
            and result.get("action_statistics_valid") is True
            and result.get("hash_checks") is False
        )
    except (OSError, TypeError, ValueError):
        return False


def load_validated_latent_payload(
    path: Path, *, split: str, task: str, source_motion_id: int
) -> dict[str, Any]:
    """Load one semantic latent identity without any digest-based admission."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"latent cache payload is not a mapping: {path}")
    if any(
        key not in payload or tuple(payload[key].shape) != shape
        for key, shape in LATENT_PAYLOAD_SHAPES.items()
    ):
        raise ValueError(f"latent cache geometry mismatch: {path}")
    if not all(
        bool(torch.isfinite(payload[key]).all()) for key in LATENT_PAYLOAD_SHAPES
    ):
        raise ValueError(f"latent cache contains a non-finite tensor: {path}")
    if torch.equal(payload["prompt_latents"], payload["reversed_prompt_latents"]):
        raise ValueError(f"reversed prompt latent is identical: {path}")
    if (
        payload.get("split") != split
        or payload.get("task") != task
        or int(payload.get("source_motion_id", -1)) != int(source_motion_id)
    ):
        raise ValueError(f"latent cache identity mismatch: {path}")
    return payload


def manifest_clock_is_exact(rows: list[dict[str, Any]]) -> bool:
    """Validate the source-level 64/141 RGB and 700-action causal clock.

    Every structural invariant the model relies on is enforced: per-row frame
    counts, strictly increasing prompt indices, exact 50 Hz / 10 Hz timestamp
    grids, the 5-step action range partition, path uniqueness and complete
    prompt/robot pixel disjointness.  The absolute corpus size and split
    balance are derived from the manifest instead of pinned to one historical
    render, so a smaller or rebuilt corpus is validated on the same terms.
    """

    if not rows:
        return False
    identities: set[tuple[str, str, int]] = set()
    prompt_paths_seen: set[str] = set()
    robot_paths_seen: set[str] = set()
    observed_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        try:
            split = str(row["split"])
            task = str(row["task"])
            source_motion_id = int(row["source_motion_id"])
            prompt_paths = list(row["prompt"]["frame_paths"])
            robot_paths = list(row["robot_target"]["frame_paths"])
            prompt_indices = [
                int(value) for value in row["prompt"]["source_frame_indices_50hz"]
            ]
            prompt_timestamps = [
                float(value) for value in row["prompt"]["timestamps_s"]
            ]
            robot_timestamps = [
                float(value) for value in row["robot_target"]["frame_timestamps_s"]
            ]
            action_ranges = [
                [int(value) for value in pair]
                for pair in row["robot_target"]["action_ranges_50hz"]
            ]
            trace_path = str(row["action_target"]["trace_path"])
            environment_index = int(row["action_target"]["environment_index"])
        except (KeyError, TypeError, ValueError):
            return False
        identity = (split, task, source_motion_id)
        prompt_set = set(prompt_paths)
        robot_set = set(robot_paths)
        # The immutable human-render corpus predates the normalized manifest
        # split spelling and names its validation directory ``valid``.  Freeze
        # that one explicit alias; robot targets use the manifest spelling.
        prompt_split_directory = "valid" if split == "validation" else split
        expected_prompt_parent = (
            prompt_split_directory,
            task,
            str(source_motion_id),
        )
        expected_robot_parent = (split, task, str(source_motion_id))
        if (
            identity in identities
            or len(prompt_paths) != 64
            or len(robot_paths) != 141
            or len(prompt_set) != 64
            or len(robot_set) != 141
            or bool(prompt_set & robot_set)
            or bool(prompt_set & prompt_paths_seen)
            or bool(robot_set & robot_paths_seen)
            or bool(prompt_set & robot_paths_seen)
            or bool(robot_set & prompt_paths_seen)
            or not all(isinstance(value, str) and value for value in prompt_paths)
            or not all(isinstance(value, str) and value for value in robot_paths)
            or any(
                tuple(Path(value).parent.parts[-3:]) != expected_prompt_parent
                for value in prompt_paths
            )
            or any(
                tuple(Path(value).parent.parts[-3:]) != expected_robot_parent
                for value in robot_paths
            )
            or row["prompt"].get("stream_semantics")
            != "clean kinematic selected demonstration"
            or row["prompt"].get("cached_once_at_inference") is not True
            or row["robot_target"].get("stream_semantics")
            != "clean exact-state render of official Generator+Tracker physical rollout"
            or len(prompt_indices) != 64
            or len(prompt_timestamps) != 64
            or any(right <= left for left, right in zip(prompt_indices, prompt_indices[1:]))
            or any(index < 0 for index in prompt_indices)
            or any(
                abs(timestamp - index / 50.0) > 1.0e-9
                for timestamp, index in zip(
                    prompt_timestamps, prompt_indices, strict=True
                )
            )
            or len(robot_timestamps) != 141
            or any(
                abs(timestamp - frame / 10.0) > 1.0e-9
                for frame, timestamp in enumerate(robot_timestamps)
            )
            or action_ranges
            != [
                [frame * 5, min((frame + 1) * 5, 700)]
                for frame in range(141)
            ]
            or int(row["action_target"].get("control_hz", -1)) != 50
            or int(row["action_target"].get("transition_count", -1)) != 700
            or row["action_target"].get("executed_action")
            != {"array": "executed_action", "shape": [700, 29]}
            or not trace_path
            or environment_index < 0
        ):
            return False
        identities.add(identity)
        prompt_paths_seen.update(prompt_set)
        robot_paths_seen.update(robot_set)
        observed_counts[(split, task)] += 1
    return (
        observed_counts.total() == len(rows)
        and len(identities) == len(rows)
        and len(prompt_paths_seen) == len(rows) * 64
        and len(robot_paths_seen) == len(rows) * 141
        and not (prompt_paths_seen & robot_paths_seen)
    )


def schedule_action_sources_are_exact(
    schedule_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]
) -> bool:
    """Bind every scheduled action shard to its train source in the manifest."""

    expected: dict[tuple[str, int], tuple[str, int]] = {}
    try:
        for row in manifest_rows:
            if row["split"] != "train":
                continue
            key = (str(row["task"]), int(row["source_motion_id"]))
            value = (
                str(row["action_target"]["trace_path"]),
                int(row["action_target"]["environment_index"]),
            )
            if key in expected:
                return False
            expected[key] = value
        if not expected:
            return False
        return all(
            expected.get((str(row["task"]), int(row["source_motion_id"])))
            == (str(row["trace_path"]), int(row["environment_index"]))
            for row in schedule_rows
        )
    except (KeyError, TypeError, ValueError):
        return False


def select_executed_actions(
    shard: np.ndarray,
    environment_index: int,
    *,
    expected_steps: int = 700,
    action_dim: int = 29,
) -> np.ndarray:
    """Select one real environment without silently collapsing shard geometry."""

    value = np.asarray(shard)
    index = int(environment_index)
    if value.ndim == 3:
        if (
            value.shape[0] < expected_steps
            or value.shape[2] != action_dim
            or index < 0
            or index >= value.shape[1]
        ):
            raise ValueError(
                "executed-action shard/environment geometry mismatch: "
                f"shape={value.shape} environment_index={index}"
            )
        selected = value[:expected_steps, index, :]
    elif value.ndim == 2:
        if (
            value.shape[0] < expected_steps
            or value.shape[1] != action_dim
            or index != 0
        ):
            raise ValueError(
                "unbatched executed-action geometry requires environment_index=0: "
                f"shape={value.shape} environment_index={index}"
            )
        selected = value[:expected_steps, :]
    else:
        raise ValueError(
            f"executed-action shard must be rank two or three, got {value.shape}"
        )
    selected = np.asarray(selected, dtype=np.float32)
    if selected.shape != (expected_steps, action_dim) or not np.isfinite(selected).all():
        raise ValueError(
            f"selected executed actions are invalid: shape={selected.shape}"
        )
    return selected.copy()


def load_executed_actions(
    trace_path: str | Path,
    environment_index: int,
    *,
    expected_steps: int = 700,
    action_dim: int = 29,
) -> np.ndarray:
    """Load the exact PhysX-executed action stream selected by a manifest row."""

    path = Path(trace_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with np.load(path.resolve(), allow_pickle=False) as trace:
        if "executed_action" not in trace.files:
            raise ValueError(f"trace lacks executed_action: {path}")
        return select_executed_actions(
            trace["executed_action"],
            environment_index,
            expected_steps=expected_steps,
            action_dim=action_dim,
        )


def action_trace_geometry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reopen each trace once and summarize all manifest environment selectors."""

    trace_environments: dict[Path, set[int]] = {}
    seen: set[tuple[Path, int]] = set()
    shape_environment_counts: Counter[str] = Counter()
    try:
        for row in rows:
            path = Path(str(row["action_target"]["trace_path"]))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path = path.resolve()
            environment_index = int(row["action_target"]["environment_index"])
            key = (path, environment_index)
            if key in seen:
                raise ValueError(f"duplicate trace/environment selector: {key}")
            seen.add(key)
            trace_environments.setdefault(path, set()).add(environment_index)
        for path, environment_indices in trace_environments.items():
            with np.load(path, allow_pickle=False) as trace:
                if "executed_action" not in trace.files:
                    raise ValueError(f"trace lacks executed_action: {path}")
                shard = trace["executed_action"]
                shape_label = "x".join(str(value) for value in shard.shape)
                shape_environment_counts[shape_label] += len(environment_indices)
                for environment_index in environment_indices:
                    select_executed_actions(shard, environment_index)
    except (KeyError, OSError, TypeError, ValueError):
        return {
            "passed": False,
            "trace_file_count": len(trace_environments),
            "trace_environment_pair_count": len(seen),
            "shape_environment_counts": dict(shape_environment_counts),
        }
    return {
        "passed": len(rows) == len(seen) and bool(rows),
        "trace_file_count": len(trace_environments),
        "trace_environment_pair_count": len(seen),
        "shape_environment_counts": dict(sorted(shape_environment_counts.items())),
    }


def action_trace_geometry_is_exact(rows: list[dict[str, Any]]) -> bool:
    """Validate all 199 manifest action selectors against their stored arrays."""

    return action_trace_geometry_summary(rows)["passed"] is True


class ActionNormalizer:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != "paper_zero_wam_sugar_action_quantiles_v2":
            raise ValueError(f"action normalization protocol mismatch: {path}")
        self.lower = torch.tensor(payload["q01"], dtype=torch.float32)
        self.upper = torch.tensor(payload["q99"], dtype=torch.float32)
        if (
            tuple(self.lower.shape) != (29,)
            or tuple(self.upper.shape) != (29,)
            or not bool(torch.isfinite(self.lower).all())
            or not bool(torch.isfinite(self.upper).all())
            or not bool((self.upper > self.lower).all())
        ):
            raise ValueError(f"action normalization geometry/range invalid: {path}")

    def normalize(self, actions: torch.Tensor, *, training_clip: bool) -> torch.Tensor:
        lower = self.lower.to(device=actions.device, dtype=actions.dtype)
        upper = self.upper.to(device=actions.device, dtype=actions.dtype)
        value = 2.0 * (actions - lower) / (upper - lower + 1.0e-6) - 1.0
        return value.clamp(-1.5, 1.5) if training_clip else value

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize a corpus action with the released training-time clip."""

        return self.normalize(actions, training_clip=True)

    def deployed_history(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize observed executed actions as in the released VA server."""

        return self.normalize(actions, training_clip=False)

    def invert(self, actions: torch.Tensor) -> torch.Tensor:
        lower = self.lower.to(device=actions.device, dtype=actions.dtype)
        upper = self.upper.to(device=actions.device, dtype=actions.dtype)
        value = (actions + 1.0) * 0.5
        return value * (upper - lower + 1.0e-6) + lower


class ScheduledSamples:
    def __init__(
        self,
        schedule_path: Path,
        cache_root: Path,
        rank: int,
        config: PaperZeroWAMConfig,
        mode: str,
        overfit_steps: int = 32,
    ):
        self.config = config
        self.cache_root = cache_root
        self.rank = rank
        self.mode = mode
        self.normalizer = ActionNormalizer(cache_root / "ACTION_QUANTILES.json")
        # Formal training revisits the same 160 trajectories for 15 complete
        # epochs. Keep immutable decoded CPU tensors process-local so the
        # shared filesystem is not asked to decompress the same PT/NPZ member
        # thousands of times. Samples are still sliced, transferred, and
        # consumed in the exact frozen schedule order.
        self._latent_cpu_cache: dict[Path, dict[str, Any]] = {}
        self._action_shard_cpu_cache: dict[Path, np.ndarray] = {}
        self._action_cpu_cache: dict[tuple[Path, int], torch.Tensor] = {}
        manifest_rows = read_jsonl(config.resolved(config.manifest))
        if not manifest_clock_is_exact(manifest_rows):
            raise ValueError(
                "manifest must contain the exact 199-source 64/141-RGB causal clock"
            )
        if not latent_materialization_admission_is_exact(cache_root, repaired=config.repaired_conditioning):
            raise ValueError(
                "completed 199-trajectory latent/action materialization admission "
                "is missing or invalid"
            )
        self.train_ids_by_task = {
            task: sorted(
                int(row["source_motion_id"])
                for row in manifest_rows
                if row["split"] == "train" and row["task"] == task
            )
            for task in ("CarryBox", "KickBox")
        }
        if mode == "formal":
            self.rows = sorted(
                [row for row in read_jsonl(schedule_path) if int(row["rank"]) == rank],
                key=lambda row: int(row["global_step"]),
            )
            if len(self.rows) != config.optimizer_steps:
                raise ValueError(
                    f"rank {rank} expected {config.optimizer_steps} formal rows, found {len(self.rows)}"
                )
        elif mode == "overfit":
            formal_rows = read_jsonl(schedule_path)
            candidate = select_overfit_schedule_row(formal_rows, rank)
            self.rows = [dict(candidate, global_step=step, epoch=0, epoch_step=step) for step in range(overfit_steps)]
        else:
            raise ValueError(f"unknown training mode: {mode}")
        if not schedule_action_sources_are_exact(self.rows, manifest_rows):
            raise ValueError(
                "scheduled action trace/environment shard does not match its manifest source"
            )
        scheduled_action_keys: set[tuple[Path, int]] = set()
        for row in self.rows:
            path = Path(str(row["trace_path"]))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            scheduled_action_keys.add(
                (path.resolve(), int(row["environment_index"]))
            )
        self._scheduled_action_selector_count = len(scheduled_action_keys)

    def __len__(self) -> int:
        return len(self.rows)

    def _latent_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        path = (
            self.cache_root
            / "train"
            / str(row["task"])
            / f'{int(row["source_motion_id"]):03d}.pt'
        )
        if path not in self._latent_cpu_cache:
            payload = load_validated_latent_payload(
                path,
                split="train",
                task=str(row["task"]),
                source_motion_id=int(row["source_motion_id"]),
            )
            if self.config.repaired_conditioning and payload.get("prompt_coverage") != prompt_coverage_record():
                raise ValueError(f"repaired loader refuses old truncated prompt cache: {path}")
            self._latent_cpu_cache[path] = payload
        return self._latent_cpu_cache[path]

    def _actions(self, row: dict[str, Any]) -> torch.Tensor:
        path = Path(row["trace_path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        environment_index = int(row["environment_index"])
        key = (path, environment_index)
        if key in self._action_cpu_cache:
            return self._action_cpu_cache[key]
        if path not in self._action_shard_cpu_cache:
            with np.load(path, allow_pickle=False) as trace:
                if "executed_action" not in trace.files:
                    raise ValueError(f"trace lacks executed_action: {path}")
                self._action_shard_cpu_cache[path] = np.asarray(
                    trace["executed_action"]
                ).copy()
        shard = select_executed_actions(
            self._action_shard_cpu_cache[path],
            environment_index,
            action_dim=self.config.action_dim,
        )
        actions = torch.from_numpy(shard.copy())
        if tuple(actions.shape) != (700, self.config.action_dim) or not bool(
            torch.isfinite(actions).all()
        ):
            raise ValueError(f"action geometry mismatch: {path} {tuple(actions.shape)}")
        normalized = self.normalizer(actions)
        self._action_cpu_cache[key] = normalized
        # Once every scheduled selector has its compact normalized stream,
        # the larger source shards can be released without causing a reload.
        if len(self._action_cpu_cache) == self._scheduled_action_selector_count:
            self._action_shard_cpu_cache.clear()
        return normalized

    def sample(self, index: int, device: torch.device) -> dict[str, Any]:
        row = self.rows[index]
        payload = self._latent_payload(row)
        prompt = payload["prompt_latents"].to(device=device, dtype=torch.bfloat16)
        robot = payload["robot_latents"].to(device=device, dtype=torch.bfloat16)
        actions = self._actions(row).to(device=device, dtype=torch.bfloat16)
        latent_start = int(row["latent_start"])
        latent_stop = int(row["latent_stop"])
        chunk_size = int(row["chunk_size"])
        action_start = int(row["action_start"])
        action_stop = int(row["action_stop"])
        environment_index = int(row["environment_index"])
        if latent_stop - latent_start != chunk_size or action_stop - action_start != chunk_size * 20:
            raise ValueError("schedule/data temporal geometry mismatch")

        ifp_targets: list[torch.Tensor] = []
        for future_start, valid in zip(
            row["ifp_latent_starts"], row["ifp_valid"], strict=True
        ):
            if bool(valid):
                target = robot[:, int(future_start) + 1 : int(future_start) + chunk_size + 1]
            else:
                target = torch.zeros_like(robot[:, 1 : chunk_size + 1])
            ifp_targets.append(target.unsqueeze(0))

        prompt_enabled = self.mode == "overfit" or bool(row["prompt_enabled"])
        return {
            "prompt_latents": prompt.unsqueeze(0),
            "robot_history_latents": robot[:, : latent_start + 1].unsqueeze(0),
            "video_target_latents": robot[:, latent_start + 1 : latent_stop + 1].unsqueeze(0),
            "action_history": actions[:action_start].unsqueeze(0),
            "action_target": actions[action_start:action_stop].unsqueeze(0),
            "ifp_target_latents": ifp_targets,
            "ifp_valid": [bool(value) for value in row["ifp_valid"]],
            "ifp_latent_starts": [int(value) for value in row["ifp_latent_starts"]],
            "prompt_enabled": prompt_enabled,
            "meta": {
                "global_step": int(row["global_step"]),
                "epoch": int(row["epoch"]),
                "epoch_step": int(row["epoch_step"]),
                "rank": int(row["rank"]),
                "task": str(row["task"]),
                "source_motion_id": int(row["source_motion_id"]),
                "trace_path": str(row["trace_path"]),
                "environment_index": environment_index,
                "trajectory_pattern": str(row["trajectory_pattern"]),
                "trajectory_chunk_index": int(row["trajectory_chunk_index"]),
                "latent_start": latent_start,
                "latent_stop": latent_stop,
                "chunk_size": chunk_size,
                "action_start": action_start,
                "action_stop": action_stop,
                "ifp_latent_starts": [
                    int(value) for value in row["ifp_latent_starts"]
                ],
                "ifp_valid": [bool(value) for value in row["ifp_valid"]],
                "latent_transition_exposures": chunk_size,
                "rgb_interval_exposures": chunk_size * 4,
                "action_exposures": chunk_size * 20,
                "main_sequence_tokens_per_pass": int(
                    row["main_sequence_tokens_per_pass"]
                ),
                "main_pass_count": int(row["main_pass_count"]),
                "ifp_sequence_tokens": [
                    int(value) for value in row["ifp_sequence_tokens"]
                ],
                "ifp_executed_sequence_tokens": [
                    int(value) for value in row["ifp_executed_sequence_tokens"]
                ],
                "supervised_transformer_token_exposures": int(
                    row["supervised_transformer_token_exposures"]
                ),
                "transformer_token_exposures": int(
                    row["transformer_token_exposures"]
                ),
                "prompt_enabled": prompt_enabled,
            },
        }

    def conditioned_sample(
        self, index: int, device: torch.device, condition: str
    ) -> dict[str, Any]:
        batch = self.sample(index, device)
        row = self.rows[index]
        if condition == "matched":
            return batch
        if condition == "reversed":
            payload = self._latent_payload(row)
            batch["prompt_latents"] = payload["reversed_prompt_latents"].to(
                device=device, dtype=torch.bfloat16
            ).unsqueeze(0)
            return batch
        if condition == "wrong_task":
            wrong_task = "KickBox" if row["task"] == "CarryBox" else "CarryBox"
            ids = self.train_ids_by_task[wrong_task]
            source_motion_id = int(row["source_motion_id"])
            wrong_id = ids[source_motion_id % len(ids)]
            payload = self._latent_payload(
                {"task": wrong_task, "source_motion_id": wrong_id}
            )
            batch["prompt_latents"] = payload["prompt_latents"].to(
                device=device, dtype=torch.bfloat16
            ).unsqueeze(0)
            return batch
        if condition == "same_task_alternate":
            ids = self.train_ids_by_task[str(row["task"])]
            position = ids.index(int(row["source_motion_id"]))
            alternate_id = ids[(position + 1) % len(ids)]
            payload = self._latent_payload(
                {"task": str(row["task"]), "source_motion_id": alternate_id}
            )
            batch["prompt_latents"] = payload["prompt_latents"].to(
                device=device, dtype=torch.bfloat16
            ).unsqueeze(0)
            return batch
        raise ValueError(f"unknown prompt condition: {condition}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        device = torch.device("cuda", int(torch.cuda.current_device()))
        for index in range(len(self.rows)):
            yield self.sample(index, device)
