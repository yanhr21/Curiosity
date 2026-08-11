#!/usr/bin/env python3
"""Train the existing serious tactile adapter on contact-state teacher actions.

This is an offline fusion diagnostic, not a replacement policy.  The official
SUGAR base actor remains frozen; only the already-declared spatial tactile
encoder and its appended first-layer columns are optimized.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch

from sugar_rl.utils.reference_only_tactile_actor_critic import (
    ReferenceOnlyTactileActorCritic,
)


TACTILE_WIDTH = 2 * 4 * 27 * 3 * 20 * 25
BASE_WIDTH = 890
ACTION_WIDTH = 29
ZERO_RECONSTRUCTION_TOLERANCE = 2.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--test", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13011)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        parser.error("steps, batch-size, and eval-every must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("learning-rate must be positive and finite")
    return args


@dataclass
class SparseArchive:
    path: Path
    metadata: dict[str, object]
    base: np.ndarray
    teacher: np.ndarray
    zero: np.ndarray
    reference_frame: np.ndarray
    current_nonzero: np.ndarray
    current_normal_by_hand: np.ndarray
    row_ptr: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    width: int

    @classmethod
    def load(cls, path: Path) -> "SparseArchive":
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata_path = path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as data:
            required = {
                "policy_base_obs",
                "teacher_action",
                "zero_tactile_action",
                "reference_frame",
                "current_tactile_nonzero_values",
                "current_active_normal_taxels",
                "tactile_row_ptr",
                "tactile_indices",
                "tactile_values",
                "tactile_width",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise RuntimeError(f"{path} is missing arrays: {missing}")
            archive = cls(
                path=path,
                metadata=metadata,
                base=np.asarray(data["policy_base_obs"], dtype=np.float32),
                teacher=np.asarray(data["teacher_action"], dtype=np.float32),
                zero=np.asarray(data["zero_tactile_action"], dtype=np.float32),
                reference_frame=np.asarray(data["reference_frame"], dtype=np.int64),
                current_nonzero=np.asarray(
                    data["current_tactile_nonzero_values"], dtype=np.int64
                ),
                current_normal_by_hand=np.asarray(
                    data["current_active_normal_taxels"], dtype=np.int64
                ),
                row_ptr=np.asarray(data["tactile_row_ptr"], dtype=np.int64),
                indices=np.asarray(data["tactile_indices"], dtype=np.int64),
                values=np.asarray(data["tactile_values"], dtype=np.float32),
                width=int(np.asarray(data["tactile_width"]).item()),
            )
        archive.validate()
        return archive

    @property
    def rows(self) -> int:
        return int(self.base.shape[0])

    def validate(self) -> None:
        if self.rows < 1:
            raise RuntimeError(f"{self.path} contains no contact rows")
        if self.metadata.get("schema") != "native_tactile_teacher_residual_sparse_dataset_v1":
            raise RuntimeError(f"{self.path} has the wrong dataset schema")
        if int(self.metadata.get("rows", -1)) != self.rows:
            raise RuntimeError(f"{self.path} metadata row count is inconsistent")
        if self.metadata.get("actor_tactile_mode") != "zeroed":
            raise RuntimeError(f"{self.path} was not collected with the exact-zero actor")
        expected_shapes = {
            "base": (self.rows, BASE_WIDTH),
            "teacher": (self.rows, ACTION_WIDTH),
            "zero": (self.rows, ACTION_WIDTH),
            "reference_frame": (self.rows,),
            "current_nonzero": (self.rows,),
            "current_normal_by_hand": (self.rows, 2),
            "row_ptr": (self.rows + 1,),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise RuntimeError(
                    f"{self.path} {name} shape {getattr(self, name).shape} != {shape}"
                )
        if self.width != TACTILE_WIDTH:
            raise RuntimeError(f"{self.path} tactile width {self.width} != {TACTILE_WIDTH}")
        if self.row_ptr[0] != 0 or self.row_ptr[-1] != self.values.size:
            raise RuntimeError(f"{self.path} sparse row pointers are inconsistent")
        if self.indices.shape != self.values.shape:
            raise RuntimeError(f"{self.path} sparse index/value lengths differ")
        if np.any(np.diff(self.row_ptr) <= 0):
            raise RuntimeError(f"{self.path} contains an empty sparse contact row")
        if np.any(self.indices < 0) or np.any(self.indices >= self.width):
            raise RuntimeError(f"{self.path} contains an out-of-range tactile index")
        if np.any(self.current_nonzero <= 0):
            raise RuntimeError(f"{self.path} contains a row without current contact")
        for name in ("base", "teacher", "zero", "values"):
            if not np.isfinite(getattr(self, name)).all():
                raise RuntimeError(f"{self.path} {name} contains non-finite values")

    def dense_rows(self, rows: np.ndarray) -> np.ndarray:
        dense = np.zeros((rows.size, self.width), dtype=np.float32)
        for output_row, source_row in enumerate(rows.tolist()):
            start = int(self.row_ptr[source_row])
            stop = int(self.row_ptr[source_row + 1])
            dense[output_row, self.indices[start:stop]] = self.values[start:stop]
        return dense


class ArchiveGroup:
    def __init__(self, paths: Sequence[Path]):
        self.archives = [SparseArchive.load(path) for path in paths]
        self.archive_index = np.concatenate(
            [np.full(archive.rows, index, dtype=np.int64) for index, archive in enumerate(self.archives)]
        )
        self.row_index = np.concatenate(
            [np.arange(archive.rows, dtype=np.int64) for archive in self.archives]
        )

    def __len__(self) -> int:
        return int(self.row_index.size)

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base = np.empty((indices.size, BASE_WIDTH), dtype=np.float32)
        teacher = np.empty((indices.size, ACTION_WIDTH), dtype=np.float32)
        zero = np.empty((indices.size, ACTION_WIDTH), dtype=np.float32)
        tactile = np.zeros((indices.size, TACTILE_WIDTH), dtype=np.float32)
        for archive_id in np.unique(self.archive_index[indices]):
            destination = np.flatnonzero(self.archive_index[indices] == archive_id)
            source_rows = self.row_index[indices[destination]]
            archive = self.archives[int(archive_id)]
            base[destination] = archive.base[source_rows]
            teacher[destination] = archive.teacher[source_rows]
            zero[destination] = archive.zero[source_rows]
            tactile[destination] = archive.dense_rows(source_rows)
        return base, teacher, zero, tactile

    def summary(self) -> list[dict[str, object]]:
        return [
            {
                "path": str(archive.path),
                "condition": archive.metadata["condition"],
                "rollout_actor_tactile_mode": archive.metadata["actor_tactile_mode"],
                "rows": archive.rows,
                "reference_frame_min": int(archive.reference_frame.min()),
                "reference_frame_max": int(archive.reference_frame.max()),
                "bilateral_current_normal_rows": int(
                    np.count_nonzero(np.all(archive.current_normal_by_hand > 0, axis=-1))
                ),
                "current_tactile_nonzero_median": float(
                    np.median(archive.current_nonzero)
                ),
            }
            for archive in self.archives
        ]


def build_model(checkpoint_path: Path, device: torch.device) -> tuple[ReferenceOnlyTactileActorCritic, dict[str, object]]:
    tactile_group = "native_whole_hand_tactile_history"
    dummy = {
        "policy": torch.zeros(1, BASE_WIDTH),
        tactile_group: torch.zeros(1, TACTILE_WIDTH),
        "critic": torch.zeros(1, BASE_WIDTH),
        "teacher": torch.zeros(1, BASE_WIDTH),
    }
    obs_groups = {
        "policy": ["policy", tactile_group],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    model = ReferenceOnlyTactileActorCritic(
        obs=dummy,
        obs_groups=obs_groups,
        num_actions=ACTION_WIDTH,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        tactile_obs_group=tactile_group,
        tactile_grid_shape=(20, 25),
        tactile_num_hands=2,
        tactile_channels_per_hand=324,
        tactile_encoder_channels=(32, 64, 64),
        tactile_embedding_dim=128,
        tactile_preactivation_cap=0.15,
        tactile_action_residual_cap=0.1,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        init_noise_std=0.5,
    )
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(f"{checkpoint_path} has no model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    finetune = model.configure_tactile_actor_finetune()
    trainable = set(finetune["trainable_actor_parameters"])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable)
    if not trainable or not all(name.startswith(("actor.0", "actor_tactile_encoder.")) for name in trainable):
        raise RuntimeError(f"unexpected trainable parameter set: {sorted(trainable)}")
    return model, finetune


def patch_permute(tactile: torch.Tensor, permutations: torch.Tensor) -> torch.Tensor:
    maps = tactile.reshape(tactile.shape[0], 2, 4, 27, 3, 20, 25)
    result = torch.empty_like(maps)
    for hand in range(2):
        result[:, hand] = maps[:, hand].index_select(2, permutations[hand])
    return result.reshape_as(tactile)


def evaluate(
    model: ReferenceOnlyTactileActorCritic,
    group: ArchiveGroup,
    device: torch.device,
    permutations: torch.Tensor,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    live_abs_sum = 0.0
    live_sq_sum = 0.0
    zero_abs_sum = 0.0
    zero_sq_sum = 0.0
    permuted_abs_sum = 0.0
    model_zero_abs_max = 0.0
    model_zero_bitwise = True
    count = 0
    row_live_mae: list[np.ndarray] = []
    row_zero_mae: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(group), batch_size):
            ids = np.arange(start, min(start + batch_size, len(group)), dtype=np.int64)
            base_np, teacher_np, stored_zero_np, tactile_np = group.batch(ids)
            base = torch.from_numpy(base_np).to(device)
            teacher = torch.from_numpy(teacher_np).to(device)
            stored_zero = torch.from_numpy(stored_zero_np).to(device)
            tactile = torch.from_numpy(tactile_np).to(device)
            live_features = model.actor_tactile_encoder(tactile)
            live = model._actor_forward(torch.cat((base, live_features), dim=-1))
            zero_features = model.actor_tactile_encoder(torch.zeros_like(tactile))
            model_zero = model._actor_forward(torch.cat((base, zero_features), dim=-1))
            permuted_tactile = patch_permute(tactile, permutations)
            permuted_features = model.actor_tactile_encoder(permuted_tactile)
            permuted = model._actor_forward(torch.cat((base, permuted_features), dim=-1))

            live_error = live - teacher
            zero_error = stored_zero - teacher
            permuted_error = permuted - teacher
            live_abs_sum += float(live_error.abs().sum().item())
            live_sq_sum += float(live_error.square().sum().item())
            zero_abs_sum += float(zero_error.abs().sum().item())
            zero_sq_sum += float(zero_error.square().sum().item())
            permuted_abs_sum += float(permuted_error.abs().sum().item())
            count += int(live_error.numel())
            row_live_mae.append(live_error.abs().mean(dim=-1).cpu().numpy())
            row_zero_mae.append(zero_error.abs().mean(dim=-1).cpu().numpy())
            model_zero_abs_max = max(
                model_zero_abs_max,
                float((model_zero - stored_zero).abs().max().item()),
            )
            model_zero_bitwise = model_zero_bitwise and bool(
                torch.equal(model_zero, stored_zero)
            )
    live_mae = live_abs_sum / count
    zero_mae = zero_abs_sum / count
    per_live = np.concatenate(row_live_mae)
    per_zero = np.concatenate(row_zero_mae)
    return {
        "rows": len(group),
        "values": count,
        "live_mae": live_mae,
        "live_mse": live_sq_sum / count,
        "zero_baseline_mae": zero_mae,
        "zero_baseline_mse": zero_sq_sum / count,
        "patch_permuted_mae": permuted_abs_sum / count,
        "live_minus_zero_mae": live_mae - zero_mae,
        "relative_mae_reduction": (zero_mae - live_mae) / zero_mae,
        "rows_live_better_than_zero": int(np.count_nonzero(per_live < per_zero)),
        "rows_live_equal_to_zero": int(np.count_nonzero(per_live == per_zero)),
        "model_zero_matches_stored_bitwise": model_zero_bitwise,
        "model_zero_matches_stored_abs_max": model_zero_abs_max,
        "model_zero_matches_stored_within_tolerance": (
            model_zero_abs_max <= ZERO_RECONSTRUCTION_TOLERANCE
        ),
    }


def save_predictions(
    model: ReferenceOnlyTactileActorCritic,
    group: ArchiveGroup,
    device: torch.device,
    permutations: torch.Tensor,
    batch_size: int,
    output: Path,
) -> None:
    rows: dict[str, list[np.ndarray]] = {
        "teacher_action": [],
        "zero_action": [],
        "live_action": [],
        "patch_permuted_action": [],
    }
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(group), batch_size):
            ids = np.arange(start, min(start + batch_size, len(group)), dtype=np.int64)
            base_np, teacher_np, zero_np, tactile_np = group.batch(ids)
            base = torch.from_numpy(base_np).to(device)
            tactile = torch.from_numpy(tactile_np).to(device)
            live = model._actor_forward(
                torch.cat((base, model.actor_tactile_encoder(tactile)), dim=-1)
            )
            permuted_tactile = patch_permute(tactile, permutations)
            permuted = model._actor_forward(
                torch.cat(
                    (base, model.actor_tactile_encoder(permuted_tactile)), dim=-1
                )
            )
            rows["teacher_action"].append(teacher_np.copy())
            rows["zero_action"].append(zero_np.copy())
            rows["live_action"].append(live.cpu().numpy().astype(np.float32))
            rows["patch_permuted_action"].append(
                permuted.cpu().numpy().astype(np.float32)
            )
    arrays = {name: np.concatenate(values, axis=0) for name, values in rows.items()}
    arrays["reference_frame"] = np.concatenate(
        [archive.reference_frame for archive in group.archives]
    )
    arrays["current_active_normal_taxels"] = np.concatenate(
        [archive.current_normal_by_hand for archive in group.archives], axis=0
    )
    arrays["live_row_mae"] = np.mean(
        np.abs(arrays["live_action"] - arrays["teacher_action"]), axis=-1
    )
    arrays["zero_row_mae"] = np.mean(
        np.abs(arrays["zero_action"] - arrays["teacher_action"]), axis=-1
    )
    arrays["patch_permuted_row_mae"] = np.mean(
        np.abs(arrays["patch_permuted_action"] - arrays["teacher_action"]),
        axis=-1,
    )
    np.savez_compressed(output, **arrays)


def state_on_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    if output_dir.parent.name == "experiments":
        raise RuntimeError("output-dir must be a package below experiments, not experiments itself")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    train_group = ArchiveGroup(args.train)
    selection_group = ArchiveGroup([args.selection])
    test_groups = [ArchiveGroup([path]) for path in args.test]
    model, finetune = build_model(args.initial_checkpoint, device)
    initial_state = state_on_cpu(model)
    trainable_names = set(finetune["trainable_actor_parameters"])
    trainable_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name in trainable_names
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    permutation_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    permutations = torch.stack(
        [torch.randperm(27, generator=permutation_generator) for _ in range(2)]
    ).to(device)
    sampler = np.random.default_rng(args.seed)

    trace: list[dict[str, object]] = []
    initial_selection = evaluate(
        model, selection_group, device, permutations, args.batch_size
    )
    trace.append({"step": 0, "training_mse": None, "selection": initial_selection})
    best_step = 0
    best_mae = float(initial_selection["live_mae"])
    best_state = state_on_cpu(model)

    model.train()
    recent_losses: list[float] = []
    for step in range(1, args.steps + 1):
        ids = sampler.integers(0, len(train_group), size=args.batch_size, dtype=np.int64)
        base_np, teacher_np, _, tactile_np = train_group.batch(ids)
        base = torch.from_numpy(base_np).to(device)
        teacher = torch.from_numpy(teacher_np).to(device)
        tactile = torch.from_numpy(tactile_np).to(device)
        features = model.actor_tactile_encoder(tactile)
        prediction = model._actor_forward(torch.cat((base, features), dim=-1))
        loss = torch.mean((prediction - teacher).square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        optimizer.step()
        recent_losses.append(float(loss.detach().item()))

        if step % args.eval_every == 0 or step == args.steps:
            selection = evaluate(
                model, selection_group, device, permutations, args.batch_size
            )
            record = {
                "step": step,
                "training_mse": float(np.mean(recent_losses)),
                "selection": selection,
            }
            trace.append(record)
            recent_losses.clear()
            selection_mae = float(selection["live_mae"])
            if selection_mae < best_mae:
                best_mae = selection_mae
                best_step = step
                best_state = state_on_cpu(model)
            model.train()

    model.load_state_dict(best_state, strict=True)
    final_metrics = {
        "train": evaluate(model, train_group, device, permutations, args.batch_size),
        "selection": evaluate(
            model, selection_group, device, permutations, args.batch_size
        ),
        "tests": [
            evaluate(model, group, device, permutations, args.batch_size)
            for group in test_groups
        ],
    }
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir()
    save_predictions(
        model,
        selection_group,
        device,
        permutations,
        args.batch_size,
        predictions_dir / "selection.npz",
    )
    for index, group in enumerate(test_groups):
        save_predictions(
            model,
            group,
            device,
            permutations,
            args.batch_size,
            predictions_dir / f"test_{index}.npz",
        )
    final_state = state_on_cpu(model)
    changed_keys = sorted(
        name
        for name in initial_state
        if not torch.equal(initial_state[name], final_state[name])
    )
    unexpected_changes = sorted(set(changed_keys).difference(trainable_names))
    actor_base_columns_exact = torch.equal(
        initial_state["actor.0.weight"][:, :BASE_WIDTH],
        final_state["actor.0.weight"][:, :BASE_WIDTH],
    )
    frozen_encoder_biases_zero = all(
        bool(torch.count_nonzero(value).item() == 0)
        for name, value in final_state.items()
        if name.startswith("actor_tactile_encoder.") and name.endswith("bias")
    )

    tests_live_better = [
        float(metric["live_mae"]) < float(metric["zero_baseline_mae"])
        for metric in final_metrics["tests"]
    ]
    test_values = sum(int(metric["values"]) for metric in final_metrics["tests"])
    aggregate_live = sum(
        float(metric["live_mae"]) * int(metric["values"])
        for metric in final_metrics["tests"]
    ) / test_values
    aggregate_zero = sum(
        float(metric["zero_baseline_mae"]) * int(metric["values"])
        for metric in final_metrics["tests"]
    ) / test_values
    all_zero_reconstructions_valid = all(
        bool(metric["model_zero_matches_stored_within_tolerance"])
        for metric in [
            final_metrics["train"],
            final_metrics["selection"],
            *final_metrics["tests"],
        ]
    )
    checks = {
        "trained_checkpoint_selected_after_step_zero": best_step > 0,
        "selection_live_beats_exact_zero": (
            float(final_metrics["selection"]["live_mae"])
            < float(final_metrics["selection"]["zero_baseline_mae"])
        ),
        "each_heldout_condition_live_beats_exact_zero": all(tests_live_better),
        "aggregate_heldout_live_beats_exact_zero": aggregate_live < aggregate_zero,
        "stored_zero_baseline_reconstructs_within_2e_minus_6": all_zero_reconstructions_valid,
        "official_actor_base_columns_remain_bitwise_exact": actor_base_columns_exact,
        "no_undeclared_model_tensor_changed": not unexpected_changes,
        "zero_preserving_encoder_biases_remain_exact_zero": frozen_encoder_biases_zero,
    }
    gate_passed = all(checks.values())
    report = {
        "schema": "native_tactile_heldout_teacher_residual_gate_v1",
        "question": (
            "Can the existing whole-hand spatial tactile adapter predict the "
            "official privileged teacher action better than the exact-zero "
            "official actor on unseen physical contact conditions?"
        ),
        "input_contract": {
            "actor_base_width": BASE_WIDTH,
            "raw_tactile_width": TACTILE_WIDTH,
            "layout": ["hand", "history", "patch", "channel", "row", "column"],
            "shape": [2, 4, 27, 3, 20, 25],
            "rgb": False,
            "measured_object_state_in_actor": False,
            "target": "official privileged Refiner teacher action",
            "zero_baseline": "same official base actor with exact-zero tactile",
            "cross_process_zero_reconstruction_tolerance": (
                ZERO_RECONSTRUCTION_TOLERANCE
            ),
        },
        "model": {
            "class": type(model).__name__,
            "spatial_encoder_channels": [32, 64, 64],
            "embedding_per_hand": 128,
            "official_actor_hidden": [512, 256, 128],
            "hidden_tactile_preactivation_cap": 0.15,
            "normalized_action_residual_cap": 0.1,
            "initial_checkpoint": str(args.initial_checkpoint.expanduser().resolve()),
            "finetune": finetune,
        },
        "split": {
            "train": train_group.summary(),
            "selection": selection_group.summary(),
            "test": [group.summary()[0] for group in test_groups],
            "policy": (
                "condition-disjoint; model selection uses only the declared "
                "selection condition and never the two test conditions"
            ),
        },
        "optimization": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "Adam",
            "loss": "mean squared error to official teacher action",
            "selection_metric": "lowest contact-row action MAE",
            "best_step": best_step,
        },
        "metrics": final_metrics,
        "prediction_records": {
            "selection": str(predictions_dir / "selection.npz"),
            "tests": [
                str(predictions_dir / f"test_{index}.npz")
                for index in range(len(test_groups))
            ],
        },
        "heldout_aggregate": {
            "live_mae": aggregate_live,
            "zero_baseline_mae": aggregate_zero,
            "live_minus_zero_mae": aggregate_live - aggregate_zero,
            "relative_mae_reduction": (aggregate_zero - aggregate_live) / aggregate_zero,
        },
        "model_change_audit": {
            "changed_keys": changed_keys,
            "unexpected_changed_keys": unexpected_changes,
            "actor_base_columns_bitwise_exact": actor_base_columns_exact,
        },
        "checks": checks,
        "gate_passed": gate_passed,
        "decision_rule": (
            "Authorize another policy experiment only when the selected model "
            "beats exact zero on selection, on each untouched test condition, "
            "and on their aggregate while every frozen-base invariant holds."
        ),
        "claim_boundary": (
            "A pass establishes held-out contact-state predictability for this "
            "fusion target, not closed-loop task improvement. A fail closes this "
            "teacher-residual target before another PPO run."
        ),
    }
    checkpoint_output = {
        "model_state_dict": best_state,
        "source_checkpoint": str(args.initial_checkpoint.expanduser().resolve()),
        "best_step": best_step,
        "gate_passed": gate_passed,
        # Keep the artifact directly loadable by the released RSL-RL runner
        # for the following no-learning closed-loop gate.
        "iter": 0,
        "infos": {
            "source": "native_tactile_heldout_teacher_residual_gate_v1",
            "best_step": best_step,
            "gate_passed": gate_passed,
        },
    }
    torch.save(checkpoint_output, output_dir / "model_best.pt")
    (output_dir / "training_trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in trace), encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
