"""Train or smoke-test the Newton-native residual controller adapter.

This is not an official T-Rex method and does not consume or emit T-Rex schema
fields. Smoke mode verifies the trainer path only; it is not a real training
result and writes no claimable checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str) -> float:
    if value == "":
        return 0.0
    return float(value)


def _group_sequences(rows: list[dict[str, str]], feature_columns: list[str], target_columns: list[str]):
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["run_tag"]].append(row)
    sequences = []
    for run_tag, run_rows in sorted(groups.items()):
        ordered = sorted(run_rows, key=lambda row: int(row["timestep_index"]))
        features = [[_as_float(row[column]) for column in feature_columns] for row in ordered]
        targets = [[_as_float(row[column]) for column in target_columns] for row in ordered]
        sequences.append((run_tag, features, targets))
    return sequences


def _mean_std(sequences, width: int, value_index: int | None = None) -> tuple[list[float], list[float]]:
    values_by_column = [[] for _ in range(width)]
    for _, _, targets_or_features in sequences:
        for row in targets_or_features:
            for idx in range(width):
                values_by_column[idx].append(float(row[idx]))
    means: list[float] = []
    stds: list[float] = []
    for values in values_by_column:
        mean = sum(values) / max(1, len(values))
        var = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        means.append(mean)
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def _normalize_sequences(sequences, feature_mean, feature_std, continuous_mean, continuous_std):
    normalized = []
    for run_tag, features, targets in sequences:
        norm_features = [
            [(value - feature_mean[idx]) / feature_std[idx] for idx, value in enumerate(row)] for row in features
        ]
        active = [[row[0]] for row in targets]
        continuous = [
            [(value - continuous_mean[idx]) / continuous_std[idx] for idx, value in enumerate(row[1:])]
            for row in targets
        ]
        normalized.append((run_tag, norm_features, active, continuous))
    return normalized


def _torch_stack_sequences(torch, sequences, device):
    features = torch.tensor([item[1] for item in sequences], dtype=torch.float32, device=device)
    active = torch.tensor([item[2] for item in sequences], dtype=torch.float32, device=device)
    continuous = torch.tensor([item[3] for item in sequences], dtype=torch.float32, device=device)
    return features, active, continuous


def run(config_path: Path, root: Path, fresh_sanity_json: Path, run_tag: str, run_mode: str, device_name: str) -> dict[str, Any]:
    config = _load_json(config_path)
    preflight_path = root / config["preflight_manifest"]
    preflight = _load_json(preflight_path)
    fresh_sanity = _load_json(fresh_sanity_json)

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if fresh_sanity.get("status") != "pass":
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")
    if preflight.get("status") != "pass":
        failures.append(f"preflight_status:{preflight.get('status')}")
    if preflight.get("generated_trex_fields") != []:
        failures.append("preflight_generated_trex_fields_not_empty")
    if preflight.get("schema_promotion") != "blocked":
        failures.append(f"preflight_schema_promotion:{preflight.get('schema_promotion')}")
    if run_mode not in {"smoke", "train"}:
        failures.append(f"invalid_run_mode:{run_mode}")
    if run_mode == "train" and int(config["real_training"]["min_train_seconds"]) < 3600:
        failures.append("real_training_min_train_seconds_below_3600")

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - executed in compute env
        torch = None
        nn = None
        failures.append(f"torch_import_failed:{type(exc).__name__}:{exc}")

    if failures:
        torch_available = torch is not None
    else:
        torch_available = True

    output_dir = root / config["output_dir"]
    checkpoint_dir = root / config["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{run_tag}_summary.json"

    if failures:
        summary = {
            "classification": "residual_adapter_trainer_v1_result",
            "status": "fail",
            "run_mode": run_mode,
            "run_tag": run_tag,
            "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
            "preflight_manifest": _rel(preflight_path, root),
            "torch_available": torch_available,
            "schema_promotion": "blocked",
            "generated_trex_fields": [],
            "real_training_result": False,
            "checkpoint_written": False,
            "failures": failures,
        }
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    assert torch is not None and nn is not None
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        failures.append("cuda_requested_but_torch_cuda_unavailable")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name if device_name.startswith("cuda") else "cpu")

    feature_columns = list(config["feature_columns"])
    target_columns = list(config["target_columns"])
    train_rows = _read_csv(root / preflight["train_csv"])
    validation_rows = _read_csv(root / preflight["validation_csv"])
    train_sequences_raw = _group_sequences(train_rows, feature_columns, target_columns)
    validation_sequences_raw = _group_sequences(validation_rows, feature_columns, target_columns)
    if not train_sequences_raw:
        failures.append("no_train_sequences")
    if not validation_sequences_raw:
        failures.append("no_validation_sequences")
    if failures:
        summary = {
            "classification": "residual_adapter_trainer_v1_result",
            "status": "fail",
            "run_mode": run_mode,
            "run_tag": run_tag,
            "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
            "preflight_manifest": _rel(preflight_path, root),
            "schema_promotion": "blocked",
            "generated_trex_fields": [],
            "real_training_result": False,
            "checkpoint_written": False,
            "failures": failures,
        }
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    feature_mean, feature_std = _mean_std([(tag, [], feats) for tag, feats, _ in train_sequences_raw], len(feature_columns))
    continuous_targets = [(tag, [], [row[1:] for row in targets]) for tag, _, targets in train_sequences_raw]
    continuous_mean, continuous_std = _mean_std(continuous_targets, len(target_columns) - 1)
    train_sequences = _normalize_sequences(train_sequences_raw, feature_mean, feature_std, continuous_mean, continuous_std)
    validation_sequences = _normalize_sequences(
        validation_sequences_raw, feature_mean, feature_std, continuous_mean, continuous_std
    )
    train_features, train_active, train_continuous = _torch_stack_sequences(torch, train_sequences, device)
    val_features, val_active, val_continuous = _torch_stack_sequences(torch, validation_sequences, device)
    if run_mode == "train":
        train_repeat = int(config["real_training"].get("train_batch_repeat", 1))
        train_features = train_features.repeat(train_repeat, 1, 1)
        train_active = train_active.repeat(train_repeat, 1, 1)
        train_continuous = train_continuous.repeat(train_repeat, 1, 1)

    class ResidualControllerAdapter(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, gru_layers: int) -> None:
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=gru_layers, batch_first=True)
            self.active_head = nn.Linear(hidden_dim, 1)
            self.continuous_head = nn.Linear(hidden_dim, len(target_columns) - 1)

        def forward(self, features):
            hidden, _ = self.gru(self.input_norm(features))
            return self.active_head(hidden), self.continuous_head(hidden)

    architecture = config["architecture"]
    model = ResidualControllerAdapter(
        input_dim=int(architecture["input_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        gru_layers=int(architecture["gru_layers"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    def evaluate() -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            active_logits, continuous_pred = model(val_features)
            active_loss = bce(active_logits, val_active)
            continuous_loss = mse(continuous_pred, val_continuous)
            active_prob = torch.sigmoid(active_logits)
            active_pred = (active_prob >= 0.5).float()
            active_accuracy = (active_pred == val_active).float().mean()
            total_loss = active_loss + continuous_loss
        return {
            "loss": float(total_loss.detach().cpu()),
            "active_bce": float(active_loss.detach().cpu()),
            "continuous_mse": float(continuous_loss.detach().cpu()),
            "active_accuracy": float(active_accuracy.detach().cpu()),
        }

    start_time = time.time()
    optimizer_steps = 0
    train_losses: list[float] = []
    if run_mode == "smoke":
        max_steps = int(config["smoke"]["max_optimizer_steps"])
        max_epochs = int(config["smoke"]["epochs"])
        target_seconds = 0
    else:
        max_steps = 10_000_000
        max_epochs = 10_000_000
        target_seconds = int(config["real_training"]["min_train_seconds"])

    model.train()
    while optimizer_steps < max_steps:
        for _ in range(max_epochs):
            optimizer.zero_grad(set_to_none=True)
            active_logits, continuous_pred = model(train_features)
            active_loss = bce(active_logits, train_active)
            continuous_loss = mse(continuous_pred, train_continuous)
            loss = (
                float(config["loss"]["feedback_active_bce_weight"]) * active_loss
                + float(config["loss"]["continuous_mse_weight"]) * continuous_loss
            )
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            train_losses.append(float(loss.detach().cpu()))
            if optimizer_steps >= max_steps:
                break
            if run_mode == "train" and time.time() - start_time >= target_seconds:
                break
        if run_mode == "smoke":
            break
        if time.time() - start_time >= target_seconds:
            break

    elapsed = time.time() - start_time
    validation_metrics = evaluate()
    checkpoint_written = False
    checkpoint_path = None
    if run_mode == "train" and elapsed >= target_seconds and config["real_training"]["checkpoint_written"]:
        checkpoint_path = checkpoint_dir / f"{run_tag}.pt"
        torch.save(
            {
                "classification": "newton_native_residual_controller_adapter_v1_checkpoint",
                "model_state_dict": model.state_dict(),
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "config": config,
            },
            checkpoint_path,
        )
        checkpoint_written = True

    summary = {
        "classification": "residual_adapter_trainer_v1_result",
        "status": "pass" if not failures else "fail",
        "run_mode": run_mode,
        "run_tag": run_tag,
        "method_name": config["method_name"],
        "not_official_trex_method": True,
        "not_trex_schema": True,
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "preflight_manifest": _rel(preflight_path, root),
        "train_csv": preflight["train_csv"],
        "validation_csv": preflight["validation_csv"],
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_record_count": len(train_rows),
        "validation_record_count": len(validation_rows),
        "effective_train_batch_size": int(train_features.shape[0]),
        "train_batch_repeat": int(config["real_training"].get("train_batch_repeat", 1)) if run_mode == "train" else 1,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": elapsed,
        "train_loss_first": train_losses[0] if train_losses else None,
        "train_loss_last": train_losses[-1] if train_losses else None,
        "validation_metrics": validation_metrics,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        "real_training_result": run_mode == "train" and elapsed >= target_seconds and checkpoint_written,
        "smoke_diagnostic_only": run_mode == "smoke",
        "checkpoint_written": checkpoint_written,
        "checkpoint_path": _rel(checkpoint_path, root) if checkpoint_path else None,
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "forbidden_claims": config["forbidden_claims"],
        "failures": failures,
    }
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--fresh-sanity-json", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--run-mode", choices=["smoke", "train"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = run(args.config, args.root, args.fresh_sanity_json, args.run_tag, args.run_mode, args.device)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "run_mode": summary["run_mode"],
                "run_tag": summary["run_tag"],
                "optimizer_steps": summary.get("optimizer_steps"),
                "real_training_result": summary.get("real_training_result"),
                "smoke_diagnostic_only": summary.get("smoke_diagnostic_only"),
                "checkpoint_written": summary.get("checkpoint_written"),
                "validation_metrics": summary.get("validation_metrics"),
                "schema_promotion": summary["schema_promotion"],
                "generated_trex_fields": summary["generated_trex_fields"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
