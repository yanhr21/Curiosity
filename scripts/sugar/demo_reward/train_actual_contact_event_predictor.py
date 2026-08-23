#!/usr/bin/env python3
"""Train and gate the serious causal contact-event mismatch predictor."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "SUGAR/source/sugar_rl"))
sys.path.insert(0, str(ROOT / "scripts/sugar/demo_reward"))

from demo_conditioned_causal_predictor_v1 import (  # noqa: E402
    EVENT_TARGET_NAMES,
    DemoConditionedCausalEventPredictorV2,
    DemoConditionedCausalEventPredictorV3,
    count_trainable_parameters,
)


DEFAULT_DATASET = (
    ROOT
    / "experiments/demo_following/contact_event_reward_redesign_v1/"
    "actual_contact_event_predictor_dataset_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=271301)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


class PairDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        directory = root / split
        self.policy = np.load(directory / "policy_prefix.npy", mmap_mode="r")
        self.target = np.load(directory / "target_mismatch.npy", mmap_mode="r")
        self.demo_array = np.load(directory / "demo_bank.npy", mmap_mode="r")
        with np.load(directory / "routing.npz", allow_pickle=False) as routing:
            self.base_row = np.asarray(routing["pair_base_row"], dtype=np.int64)
            self.selected_demo = np.asarray(
                routing["pair_selected_demo_row"], dtype=np.int64
            )
            self.pair_role = np.asarray(routing["pair_role"], dtype=np.int64)
            self.demo_task = np.asarray(routing["demo_task"], dtype=np.int64)
            self.selected_demo_phase = (
                np.asarray(routing["pair_normalized_demo_phase"], dtype=np.float32)
                if "pair_normalized_demo_phase" in routing
                else np.zeros(len(self.base_row), dtype=np.float32)
            )
        if len(self.base_row) != len(self.target):
            raise RuntimeError(f"{split}: pair routing and targets differ")

    def __len__(self) -> int:
        return len(self.base_row)

    def demo_bank(self, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(np.array(self.demo_array, dtype=np.float32, copy=True)).to(
            device
        )

    def demo_permutation(self, device: torch.device) -> torch.Tensor:
        permutation = np.arange(len(self.demo_task), dtype=np.int64)
        for task in np.unique(self.demo_task):
            rows = np.flatnonzero(self.demo_task == task)
            permutation[rows] = np.roll(rows, 1)
        return torch.from_numpy(permutation).to(device)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base = int(self.base_row[index])
        return {
            "policy_prefix": torch.from_numpy(
                np.array(self.policy[base], dtype=np.float32, copy=True)
            ),
            "selected_demo": torch.tensor(
                int(self.selected_demo[index]), dtype=torch.int64
            ),
            "target": torch.from_numpy(
                np.array(self.target[index], dtype=np.float32, copy=True)
            ),
            "pair_role": torch.tensor(int(self.pair_role[index]), dtype=torch.int64),
            "selected_demo_phase": torch.tensor(
                float(self.selected_demo_phase[index]), dtype=torch.float32
            ),
        }


def make_loader(
    dataset: PairDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
        drop_last=shuffle,
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ar, br = rankdata(a), rankdata(b)
    if np.std(ar) == 0 or np.std(br) == 0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def model_from_normalization(path: Path, device: torch.device) -> DemoConditionedCausalEventPredictorV2:
    with np.load(path, allow_pickle=False) as statistics:
        tensors = {
            name: torch.from_numpy(np.asarray(statistics[name], dtype=np.float32))
            for name in statistics.files
        }
    policy_dim = int(tensors["state_mean"].numel())
    manifest_path = path.parent / "MANIFEST.json"
    phase_aware = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        phase_aware = manifest.get("alignment_mode") == "clock_phase"
    model_class = (
        DemoConditionedCausalEventPredictorV3
        if phase_aware
        else DemoConditionedCausalEventPredictorV2
    )
    model = model_class(
        policy_dim=policy_dim,
        policy_history_steps=10,
        demo_windows=32,
        demo_window_steps=10,
        demo_feature_dim=132,
        d_model=384,
        nhead=8,
        num_layers=6,
        dim_feedforward=1536,
        dropout=0.1,
        state_mean=tensors["state_mean"],
        state_std=tensors["state_std"],
        demo_mean=tensors["demo_mean"],
        demo_std=tensors["demo_std"],
        target_scale=tensors["target_scale"],
    )
    return model.to(device)


def forward_model(
    model: DemoConditionedCausalEventPredictorV2,
    *,
    policy_prefix: torch.Tensor,
    selected_demo_condition: torch.Tensor,
    selected_demo_phase: torch.Tensor,
    zero_demo: bool = False,
):
    kwargs = {
        "policy_prefix": policy_prefix,
        "selected_demo_condition": selected_demo_condition,
        "zero_demo": zero_demo,
    }
    if getattr(model, "phase_aware", False):
        kwargs["selected_demo_phase"] = selected_demo_phase
    return model(**kwargs)


@torch.no_grad()
def evaluate(
    model: DemoConditionedCausalEventPredictorV2,
    loader: DataLoader,
    demo_bank: torch.Tensor,
    permutation: torch.Tensor,
    device: torch.device,
    mode: str,
) -> dict[str, object]:
    model.eval()
    targets, predictions, roles = [], [], []
    for batch in loader:
        policy = batch["policy_prefix"].to(device, non_blocking=True)
        selected = batch["selected_demo"].to(device, non_blocking=True)
        if mode == "permuted_demo":
            selected = permutation.index_select(0, selected)
        demo = demo_bank.index_select(0, selected)
        phase = batch["selected_demo_phase"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = forward_model(
                model,
                policy_prefix=policy,
                selected_demo_condition=demo,
                selected_demo_phase=phase,
                zero_demo=mode == "zero_demo",
            )
        prediction = model.decode_mean(output["mean_log1p_scaled"].float())
        targets.append(batch["target"].numpy())
        predictions.append(prediction.cpu().numpy())
        roles.append(batch["pair_role"].numpy())
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    role = np.concatenate(roles)
    scale = model.target_scale.cpu().numpy()
    per_target_mae = np.mean(np.abs(prediction - target), axis=0)
    normalized_mae = per_target_mae / scale
    correlations = [spearman(target[:, i], prediction[:, i]) for i in range(target.shape[1])]
    semantic_prediction = {
        name: {
            "contact_mismatch_mean": float(np.mean(prediction[role == index, 4:8])),
            "duration_mismatch_mean": float(np.mean(prediction[role == index, 8:12])),
            "motion_regime_mismatch_mean": float(np.mean(prediction[role == index, 12])),
        }
        for index, name in enumerate(("correct", "same_task_wrong", "cross_task_wrong"))
    }
    return {
        "mode": mode,
        "row_count": len(target),
        "mean_normalized_mae": float(np.mean(normalized_mae)),
        "per_target_normalized_mae": {
            name: float(value)
            for name, value in zip(EVENT_TARGET_NAMES, normalized_mae)
        },
        "per_target_spearman": {
            name: float(value) for name, value in zip(EVENT_TARGET_NAMES, correlations)
        },
        "median_target_spearman": float(np.median(correlations)),
        "semantic_prediction_by_pair_role": semantic_prediction,
    }


def save_checkpoint(
    path: Path,
    model: DemoConditionedCausalEventPredictorV2,
    epoch: int,
    validation_mae: float,
    parameter_count: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "protocol": (
                "sugar_phase_aware_causal_contact_event_predictor_v3"
                if getattr(model, "phase_aware", False)
                else "sugar_causal_contact_event_predictor_v2"
            ),
            "epoch": epoch,
            "validation_mean_normalized_mae": validation_mae,
            "target_names": EVENT_TARGET_NAMES,
            "trainable_parameter_count": parameter_count,
            "model_state_dict": model.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if socket.gethostname().startswith(("mgmtserver", "login")):
        raise RuntimeError("predictor training must run inside a compute allocation")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("retained CUDA Slurm allocation required")
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    datasets = {split: PairDataset(dataset_root, split) for split in ("train", "validation", "test")}
    loaders = {
        "train": make_loader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            workers=args.num_workers,
        ),
        "validation": make_loader(
            datasets["validation"],
            batch_size=args.batch_size,
            shuffle=False,
            workers=args.num_workers,
        ),
        "test": make_loader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            workers=args.num_workers,
        ),
    }
    demo_banks = {
        split: datasets[split].demo_bank(device)
        for split in ("train", "validation", "test")
    }
    permutations = {
        split: datasets[split].demo_permutation(device)
        for split in ("validation", "test")
    }
    model = model_from_normalization(dataset_root / "NORMALIZATION.npz", device)
    parameter_count = count_trainable_parameters(model)
    if not 10_000_000 <= parameter_count <= 15_000_000:
        raise RuntimeError(f"serious architecture parameter count drift: {parameter_count}")
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(loaders["train"])
    warmup = max(1, int(0.05 * total_steps))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, schedule)
    history = []
    best_mae = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            policy = batch["policy_prefix"].to(device, non_blocking=True)
            selected = batch["selected_demo"].to(device, non_blocking=True)
            demo = demo_banks["train"].index_select(0, selected)
            target = batch["target"].to(device, non_blocking=True)
            phase = batch["selected_demo_phase"].to(device, non_blocking=True)
            transformed = model.encode_targets(target)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_output = forward_model(
                    model,
                    policy_prefix=policy,
                    selected_demo_condition=demo,
                    selected_demo_phase=phase,
                )
                loss = model.gaussian_nll(
                    model_output["mean_log1p_scaled"],
                    model_output["log_variance_log1p_scaled"],
                    transformed,
                ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(
            model,
            loaders["validation"],
            demo_banks["validation"],
            permutations["validation"],
            device,
            "full",
        )
        row = {
            "epoch": epoch,
            "train_nll": float(np.mean(losses)),
            "validation_mean_normalized_mae": validation["mean_normalized_mae"],
        }
        history.append(row)
        print("PREDICTOR_EPOCH " + json.dumps(row, sort_keys=True), flush=True)
        if validation["mean_normalized_mae"] < best_mae:
            best_mae = float(validation["mean_normalized_mae"])
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                output / "best.pt", model, epoch, best_mae, parameter_count
            )
        else:
            stale += 1
        if epoch >= 5 and stale >= args.early_stop_patience:
            break

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluation = {}
    for split in ("validation", "test"):
        evaluation[split] = {
            mode: evaluate(
                model,
                loaders[split],
                demo_banks[split],
                permutations[split],
                device,
                mode,
            )
            for mode in ("full", "zero_demo", "permuted_demo")
        }
    with np.load(dataset_root / "NORMALIZATION.npz") as statistics:
        target_scale = np.asarray(statistics["target_scale"])
    train_target = np.load(dataset_root / "train/target_mismatch.npy", mmap_mode="r")
    constant = np.median(train_target, axis=0)
    baseline = {}
    for split in ("validation", "test"):
        target = np.load(dataset_root / split / "target_mismatch.npy", mmap_mode="r")
        baseline[split] = float(
            np.mean(np.mean(np.abs(target - constant), axis=0) / target_scale)
        )
    checks = {}
    for split in ("validation", "test"):
        full = evaluation[split]["full"]
        zero = evaluation[split]["zero_demo"]
        permuted = evaluation[split]["permuted_demo"]
        checks[f"{split}_improves_constant_by_five_percent"] = (
            full["mean_normalized_mae"] <= 0.95 * baseline[split]
        )
        checks[f"{split}_zero_demo_degrades_two_percent"] = (
            zero["mean_normalized_mae"] >= 1.02 * full["mean_normalized_mae"]
        )
        checks[f"{split}_permuted_demo_degrades_two_percent"] = (
            permuted["mean_normalized_mae"] >= 1.02 * full["mean_normalized_mae"]
        )
        checks[f"{split}_median_spearman_at_least_point_two"] = (
            full["median_target_spearman"] >= 0.20
        )
        semantic = full["semantic_prediction_by_pair_role"]
        checks[f"{split}_predicted_cross_contact_exceeds_correct"] = (
            semantic["cross_task_wrong"]["contact_mismatch_mean"]
            > semantic["correct"]["contact_mismatch_mean"] + 0.05
        )
        checks[f"{split}_predicted_cross_regime_exceeds_correct"] = (
            semantic["cross_task_wrong"]["motion_regime_mismatch_mean"]
            > semantic["correct"]["motion_regime_mismatch_mean"] + 0.10
        )
    result = {
        "protocol": "sugar_causal_contact_event_predictor_training_v2",
        "passed": all(checks.values()),
        "checks": checks,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "trainable_parameter_count": parameter_count,
        "history": history,
        "constant_baseline_mean_normalized_mae": baseline,
        "evaluation": evaluation,
        "claim_boundary": (
            "Passing establishes held-out selected-demo-conditioned mismatch prediction. "
            "It does not establish policy-level demo following; policy training remains blocked."
        ),
        "automatic_next_branch": (
            "freeze_predictor_then_request_policy_experiment_authority"
            if all(checks.values())
            else "inspect_predictor_failure_or_run_fixed_subset_overfit_before_policy_use"
        ),
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": result["passed"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
