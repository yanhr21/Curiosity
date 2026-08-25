#!/usr/bin/env python3
"""Train the serious causal Transformer for released Carry45 transition risk."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import socket

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[3]
FEATURE_DIM = 539
HISTORY_STEPS = 10
AUXILIARY_DIM = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("overfit", "formal"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--overfit-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=171625)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


class TransitionDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        directory = root / split
        self.prefix = np.load(directory / "causal_prefix.npy", mmap_mode="r")
        self.risk = np.load(directory / "risk_target.npy", mmap_mode="r")
        self.auxiliary = np.load(directory / "auxiliary_target.npy", mmap_mode="r")
        with np.load(directory / "routing.npz", allow_pickle=False) as routing:
            self.profile = np.asarray(routing["profile_id"], dtype=np.int64)
            self.anchor = np.asarray(routing["anchor"], dtype=np.int64)
            self.profile_name = np.asarray(routing["profile_name"])
        if self.prefix.shape[1:] != (HISTORY_STEPS, FEATURE_DIM):
            raise RuntimeError(f"{split}: causal prefix geometry drift")
        if not (len(self.prefix) == len(self.risk) == len(self.profile)):
            raise RuntimeError(f"{split}: row count drift")

    def __len__(self) -> int:
        return len(self.risk)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "prefix": torch.from_numpy(
                np.array(self.prefix[index], dtype=np.float32, copy=True)
            ),
            "risk": torch.tensor(float(self.risk[index]), dtype=torch.float32),
            "auxiliary": torch.from_numpy(
                np.array(self.auxiliary[index], dtype=np.float32, copy=True)
            ),
            "profile": torch.tensor(int(self.profile[index]), dtype=torch.int64),
            "anchor": torch.tensor(int(self.anchor[index]), dtype=torch.int64),
        }


class CausalTransitionRiskTransformer(nn.Module):
    """Six-layer 384-D past-only Transformer, matching the serious predictor scale."""

    def __init__(self, state_mean: torch.Tensor, state_std: torch.Tensor) -> None:
        super().__init__()
        if state_mean.shape != (FEATURE_DIM,) or state_std.shape != (FEATURE_DIM,):
            raise ValueError("transition normalization geometry drift")
        self.register_buffer("state_mean", state_mean.float().clone())
        self.register_buffer("state_std", torch.clamp(state_std.float().clone(), min=1e-6))
        d_model = 384
        self.state_projection = nn.Sequential(
            nn.Linear(FEATURE_DIM, d_model), nn.LayerNorm(d_model)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, HISTORY_STEPS + 1, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=6,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.risk_head = nn.Linear(d_model, 1)
        self.auxiliary_head = nn.Linear(d_model, AUXILIARY_DIM)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, causal_prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        if causal_prefix.ndim != 3 or causal_prefix.shape[1:] != (
            HISTORY_STEPS,
            FEATURE_DIM,
        ):
            raise ValueError("causal transition prefix geometry drift")
        normalized = (causal_prefix - self.state_mean) / self.state_std
        state = self.state_projection(normalized)
        cls = self.cls_token.expand(len(causal_prefix), -1, -1)
        encoded = self.transformer(
            torch.cat((cls, state), dim=1) + self.position_embedding
        )
        representation = self.readout(encoded[:, 0])
        return {
            "risk_logit": self.risk_head(representation).squeeze(-1),
            "auxiliary": self.auxiliary_head(representation),
            "representation": representation,
        }


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = score[target == 1]
    negative = score[target == 0]
    if len(positive) == 0 or len(negative) == 0:
        return 0.5
    comparison = positive[:, None] - negative[None]
    return float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))


def _balanced_accuracy(target: np.ndarray, score: np.ndarray, threshold: float) -> float:
    prediction = score >= threshold
    tpr = float(np.mean(prediction[target == 1]))
    tnr = float(np.mean(~prediction[target == 0]))
    return 0.5 * (tpr + tnr)


def _profile_metrics(
    risk: np.ndarray,
    score: np.ndarray,
    profile: np.ndarray,
    anchor: np.ndarray,
    threshold: float | None,
    early_only: bool,
) -> dict[str, object]:
    selected = anchor <= 49 if early_only else np.ones(len(anchor), dtype=bool)
    profiles = np.unique(profile)
    target = np.asarray([risk[(profile == item)][0] for item in profiles], dtype=np.int64)
    probability = np.asarray(
        [np.mean(score[(profile == item) & selected]) for item in profiles],
        dtype=np.float64,
    )
    if threshold is None:
        candidates = np.linspace(0.05, 0.95, 181)
        values = np.asarray(
            [_balanced_accuracy(target, probability, item) for item in candidates]
        )
        threshold = float(candidates[int(np.argmax(values))])
    prevalence = float(np.mean(target))
    result = {
        "profiles": len(profiles),
        "risky_profiles": int(np.sum(target)),
        "threshold": threshold,
        "auroc": _auc(target, probability),
        "balanced_accuracy": _balanced_accuracy(target, probability, threshold),
        "brier": float(np.mean(np.square(probability - target))),
        "prevalence_baseline_brier": float(np.mean(np.square(prevalence - target))),
        "safe_mean_probability": float(np.mean(probability[target == 0])),
        "risky_mean_probability": float(np.mean(probability[target == 1])),
    }
    result["risk_probability_gap"] = (
        result["risky_mean_probability"] - result["safe_mean_probability"]
    )
    return result


@torch.no_grad()
def evaluate_rows(
    model: CausalTransitionRiskTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        name: [] for name in ("risk", "score", "profile", "anchor", "auxiliary", "aux_pred")
    }
    for batch in loader:
        prefix = batch["prefix"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(prefix)
        collected["risk"].append(batch["risk"].numpy())
        collected["score"].append(torch.sigmoid(output["risk_logit"].float()).cpu().numpy())
        collected["profile"].append(batch["profile"].numpy())
        collected["anchor"].append(batch["anchor"].numpy())
        collected["auxiliary"].append(batch["auxiliary"].numpy())
        collected["aux_pred"].append(output["auxiliary"].float().cpu().numpy())
    return {name: np.concatenate(value) for name, value in collected.items()}


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def main() -> None:
    args = parse_args()
    if socket.gethostname().startswith(("mgmtserver", "login")):
        raise RuntimeError("transition-risk training must run on a compute node")
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
    with np.load(dataset_root / "NORMALIZATION.npz", allow_pickle=False) as archive:
        state_mean = torch.from_numpy(np.asarray(archive["state_mean"], dtype=np.float32))
        state_std = torch.from_numpy(np.asarray(archive["state_std"], dtype=np.float32))
        auxiliary_scale = torch.from_numpy(
            np.asarray(archive["auxiliary_scale"], dtype=np.float32)
        ).to(device)
    datasets = {
        split: TransitionDataset(dataset_root, split)
        for split in ("train", "validation", "test")
    }
    training_dataset = datasets["train"]
    if args.mode == "overfit":
        selected_profiles = []
        for label in (0.0, 1.0):
            profile_ids = np.unique(
                training_dataset.profile[training_dataset.risk == label]
            )[:4]
            selected_profiles.extend(int(item) for item in profile_ids)
        indices = np.flatnonzero(np.isin(training_dataset.profile, selected_profiles))
        training_dataset = Subset(training_dataset, indices.tolist())
    loaders = {
        "train": make_loader(
            training_dataset, args.batch_size, True, args.num_workers
        ),
        "validation": make_loader(
            datasets["validation"], args.batch_size, False, args.num_workers
        ),
        "test": make_loader(datasets["test"], args.batch_size, False, args.num_workers),
    }
    model = CausalTransitionRiskTransformer(state_mean, state_std).to(device)
    parameter_count = _parameter_count(model)
    if not 10_000_000 <= parameter_count <= 15_000_000:
        raise RuntimeError(f"serious architecture parameter count drift: {parameter_count}")
    train_risk = np.asarray(
        [training_dataset[index]["risk"].item() for index in range(len(training_dataset))]
    )
    positive_weight = torch.tensor(
        float(np.sum(train_risk == 0) / max(1, np.sum(train_risk == 1))), device=device
    )
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = (
        args.overfit_steps
        if args.mode == "overfit"
        else args.epochs * len(loaders["train"])
    )
    warmup = max(1, int(total_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, schedule)
    history = []
    best_validation_auc = -1.0
    best_epoch = 0
    global_step = 0
    epochs = 10_000 if args.mode == "overfit" else args.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            prefix = batch["prefix"].to(device, non_blocking=True)
            risk = batch["risk"].to(device, non_blocking=True)
            auxiliary = batch["auxiliary"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(prefix)
                risk_loss = nn.functional.binary_cross_entropy_with_logits(
                    prediction["risk_logit"], risk, pos_weight=positive_weight
                )
                auxiliary_loss = nn.functional.smooth_l1_loss(
                    prediction["auxiliary"] / auxiliary_scale,
                    auxiliary / auxiliary_scale,
                )
                loss = risk_loss + 0.25 * auxiliary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            global_step += 1
            if args.mode == "overfit" and global_step >= args.overfit_steps:
                break
        if (
            args.mode == "overfit"
            and global_step < args.overfit_steps
            and epoch % 25 != 0
        ):
            continue
        train_rows = evaluate_rows(model, make_loader(training_dataset, 256, False, 0), device)
        train_metrics = _profile_metrics(
            train_rows["risk"], train_rows["score"], train_rows["profile"],
            train_rows["anchor"], None, False
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "mean_loss": float(np.mean(losses)),
            "train_profile_auroc": train_metrics["auroc"],
            "train_profile_balanced_accuracy": train_metrics["balanced_accuracy"],
        }
        history.append(row)
        print("TRANSITION_RISK_EPOCH " + json.dumps(row, sort_keys=True), flush=True)
        if args.mode == "overfit":
            if global_step >= args.overfit_steps:
                break
            continue
        validation_rows = evaluate_rows(model, loaders["validation"], device)
        validation_metrics = _profile_metrics(
            validation_rows["risk"], validation_rows["score"],
            validation_rows["profile"], validation_rows["anchor"], None, False
        )
        if validation_metrics["auroc"] > best_validation_auc:
            best_validation_auc = float(validation_metrics["auroc"])
            best_epoch = epoch
            torch.save(
                {
                    "protocol": "sugar_causal_transition_risk_transformer_v1",
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "parameter_count": parameter_count,
                },
                output / "best.pt",
            )
    if args.mode == "overfit":
        checks = {
            "same_serious_architecture_used": True,
            "exactly_eight_profiles": len(np.unique(train_rows["profile"])) == 8,
            "train_profile_auroc_at_least_0p99": train_metrics["auroc"] >= 0.99,
            "train_balanced_accuracy_at_least_0p95": (
                train_metrics["balanced_accuracy"] >= 0.95
            ),
            "all_values_finite": bool(np.isfinite(train_rows["score"]).all()),
        }
        result = {
            "protocol": "sugar_transition_risk_serious_overfit_v1",
            "passed": all(checks.values()),
            "checks": checks,
            "parameter_count": parameter_count,
            "optimizer_steps": global_step,
            "metrics": train_metrics,
            "history": history,
            "automatic_next_stage": "formal_training" if all(checks.values()) else "inspect_overfit_failure",
        }
    else:
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        validation_rows = evaluate_rows(model, loaders["validation"], device)
        validation_all = _profile_metrics(
            validation_rows["risk"], validation_rows["score"], validation_rows["profile"],
            validation_rows["anchor"], None, False
        )
        validation_early = _profile_metrics(
            validation_rows["risk"], validation_rows["score"], validation_rows["profile"],
            validation_rows["anchor"], None, True
        )
        threshold = float(validation_early["threshold"])
        metrics = {}
        for split in ("validation", "test"):
            rows = validation_rows if split == "validation" else evaluate_rows(model, loaders[split], device)
            metrics[split] = {
                "all_early_anchors": _profile_metrics(
                    rows["risk"], rows["score"], rows["profile"], rows["anchor"], threshold, False
                ),
                "first_50_frames": _profile_metrics(
                    rows["risk"], rows["score"], rows["profile"], rows["anchor"], threshold, True
                ),
            }
        test_early = metrics["test"]["first_50_frames"]
        checks = {
            "serious_10m_to_15m_architecture": 10_000_000 <= parameter_count <= 15_000_000,
            "validation_profile_auroc_at_least_0p75": validation_early["auroc"] >= 0.75,
            "validation_balanced_accuracy_at_least_0p70": validation_early["balanced_accuracy"] >= 0.70,
            "heldout_test_first50_auroc_at_least_0p70": test_early["auroc"] >= 0.70,
            "heldout_test_first50_balanced_accuracy_at_least_0p65": test_early["balanced_accuracy"] >= 0.65,
            "heldout_test_first50_probability_gap_at_least_0p15": test_early["risk_probability_gap"] >= 0.15,
            "heldout_test_first50_brier_beats_prevalence": test_early["brier"] < test_early["prevalence_baseline_brier"],
            "future_outcome_absent_from_forward_signature": True,
        }
        result = {
            "protocol": "sugar_causal_transition_risk_transformer_training_v1",
            "passed": all(checks.values()),
            "checks": checks,
            "seed": args.seed,
            "best_epoch": int(checkpoint["epoch"]),
            "parameter_count": parameter_count,
            "validation_selected_threshold": threshold,
            "metrics": metrics,
            "history": history,
            "checkpoint": "best.pt",
            "claim_boundary": (
                "A Carry45-only causal transition-risk predictor over early official "
                "Tracker observations and current candidate actions. Future profile outcome "
                "is a training label only. Passing does not yet prove online fallback success."
            ),
            "automatic_next_stage": (
                "frozen_online_fallback_evaluation"
                if all(checks.values())
                else "fixed_subset_or_feature_failure_audit"
            ),
        }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": result["passed"], "checks": checks}, indent=2))
    if not result["passed"]:
        raise RuntimeError("transition-risk Transformer gate failed")


if __name__ == "__main__":
    main()
