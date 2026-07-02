"""Fine-tune the residual controller adapter with curiosity-derived weights.

This is not an official T-Rex method, not a VQ-VAE/world model, and not an RL
algorithm. It keeps the existing Newton-native residual adapter architecture
and uses bounded learning-progress scores as supervised sample weights.
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


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str) -> float:
    return 0.0 if value == "" else float(value)


def _row_float_first(row: dict[str, str], columns: list[str], default: float = 0.0) -> float:
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return float(value)
    return default


def _mean_std(sequences, width: int, which: str) -> tuple[list[float], list[float]]:
    values_by_column = [[] for _ in range(width)]
    for item in sequences:
        matrix = item[1] if which == "features" else item[2]
        for row in matrix:
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


def _load_scores(score_csv: Path, score_column: str, split: str) -> dict[tuple[str, int], float]:
    rows = _read_csv(score_csv)
    scores: dict[tuple[str, int], float] = {}
    for row in rows:
        if row.get("split") != split:
            continue
        key = (row["run_tag"], int(row["timestep_index"]))
        scores[key] = _as_float(row[score_column])
    return scores


def _group_sequences(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    target_columns: list[str],
    scores: dict[tuple[str, int], float],
    weight_config: dict[str, Any],
):
    positive_scores = [max(0.0, value) for value in scores.values()]
    normalizer = max(positive_scores) if positive_scores else 1.0
    extrinsic_weight = float(weight_config["extrinsic_weight"])
    curiosity_scale = float(weight_config["curiosity_weight_scale"])
    min_weight = float(weight_config["min_sample_weight"])
    max_weight = float(weight_config["max_sample_weight"])
    anchor_config = dict(weight_config.get("baseline_preservation_anchor", {}))
    anchor_enabled = bool(anchor_config.get("enabled", False))
    anchor_strength = float(anchor_config.get("anchor_strength", 0.0))
    anchor_power = float(anchor_config.get("inverse_curiosity_power", 1.0))
    stable_phase_min = int(anchor_config.get("stable_phase_min", 0))
    min_contact_count = float(anchor_config.get("min_contact_count", 0.0))
    anchor_contact_count_columns = list(
        anchor_config.get(
            "contact_count_columns",
            ["newton.contact.rigid_contact_count", "newton.panda.rigid_contact_count"],
        )
    )
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["run_tag"]].append(row)

    matched = 0
    total = 0
    sequences = []
    for run_tag, run_rows in sorted(groups.items()):
        ordered = sorted(run_rows, key=lambda row: int(row["timestep_index"]))
        features = [[_as_float(row[column]) for column in feature_columns] for row in ordered]
        targets = [[_as_float(row[column]) for column in target_columns] for row in ordered]
        weights = []
        anchor_weights = []
        for row in ordered:
            total += 1
            key = (run_tag, int(row["timestep_index"]))
            raw_score = scores.get(key)
            if raw_score is not None:
                matched += 1
            normalized_curiosity = max(0.0, raw_score or 0.0) / normalizer
            weight = extrinsic_weight + curiosity_scale * normalized_curiosity
            weights.append([max(min_weight, min(max_weight, weight))])
            if anchor_enabled and anchor_strength > 0.0:
                phase = _as_float(row.get("candidate.controller.phase_index", "0"))
                contact_count = _row_float_first(row, anchor_contact_count_columns)
                stable_enough = phase >= stable_phase_min and contact_count >= min_contact_count
                inverse_curiosity = max(0.0, 1.0 - min(1.0, normalized_curiosity))
                anchor_weight = anchor_strength * (inverse_curiosity**anchor_power) if stable_enough else 0.0
            else:
                anchor_weight = 0.0
            anchor_weights.append([anchor_weight])
        sequences.append((run_tag, features, targets, weights, anchor_weights))
    return sequences, matched, total


def _normalize_sequences(sequences, feature_mean, feature_std, continuous_mean, continuous_std):
    normalized = []
    for run_tag, features, targets, weights, anchor_weights in sequences:
        norm_features = [
            [(value - feature_mean[idx]) / feature_std[idx] for idx, value in enumerate(row)] for row in features
        ]
        active = [[row[0]] for row in targets]
        continuous = [
            [(value - continuous_mean[idx]) / continuous_std[idx] for idx, value in enumerate(row[1:])]
            for row in targets
        ]
        normalized.append((run_tag, norm_features, active, continuous, weights, anchor_weights))
    return normalized


def _torch_stack_sequences(torch, sequences, device):
    features = torch.tensor([item[1] for item in sequences], dtype=torch.float32, device=device)
    active = torch.tensor([item[2] for item in sequences], dtype=torch.float32, device=device)
    continuous = torch.tensor([item[3] for item in sequences], dtype=torch.float32, device=device)
    weights = torch.tensor([item[4] for item in sequences], dtype=torch.float32, device=device)
    anchor_weights = torch.tensor([item[5] for item in sequences], dtype=torch.float32, device=device)
    return features, active, continuous, weights, anchor_weights


def run(config_path: Path, root: Path, fresh_sanity_json: Path, run_tag: str, run_mode: str, device_name: str) -> dict[str, Any]:
    config = _load_json(config_path)
    preflight_path = root / config["preflight_manifest"]
    learning_progress_path = root / config["learning_progress_summary"]
    base_checkpoint_path = root / config["base_checkpoint"]
    preflight = _load_json(preflight_path)
    learning_progress = _load_json(learning_progress_path) if learning_progress_path.exists() else {}
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
    if learning_progress.get("status") != "pass":
        failures.append(f"learning_progress_status:{learning_progress.get('status')}")
    if learning_progress.get("policy_updated") is not False:
        failures.append("learning_progress_already_claimed_policy_update")
    if learning_progress.get("generated_trex_fields") != []:
        failures.append("learning_progress_generated_trex_fields_not_empty")
    if learning_progress.get("schema_promotion") != "blocked":
        failures.append(f"learning_progress_schema_promotion:{learning_progress.get('schema_promotion')}")
    if not base_checkpoint_path.is_file():
        failures.append(f"missing_base_checkpoint:{_rel(base_checkpoint_path, root)}")
    if run_mode not in {"smoke", "train"}:
        failures.append(f"invalid_run_mode:{run_mode}")
    if run_mode == "train" and int(config["real_training"]["min_train_seconds"]) < 3600:
        failures.append("real_training_min_train_seconds_below_3600")

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        torch = None
        nn = None
        failures.append(f"torch_import_failed:{type(exc).__name__}:{exc}")

    output_dir = root / config["output_dir"]
    checkpoint_dir = root / config["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{run_tag}_summary.json"

    if failures:
        summary = {
            "classification": "curiosity_weighted_residual_adapter_trainer_v1_result",
            "status": "fail",
            "run_mode": run_mode,
            "run_tag": run_tag,
            "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
            "preflight_manifest": _rel(preflight_path, root),
            "learning_progress_summary": _rel(learning_progress_path, root),
            "base_checkpoint": _rel(base_checkpoint_path, root),
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
    weight_config = dict(config["curiosity_weighting"])
    score_csv = root / learning_progress["output_csv"]
    train_scores = _load_scores(score_csv, weight_config["score_column"], weight_config["score_split_for_training"])
    validation_scores = _load_scores(score_csv, weight_config["score_column"], weight_config["score_split_for_validation"])
    train_rows = _read_csv(root / preflight["train_csv"])
    validation_rows = _read_csv(root / preflight["validation_csv"])
    train_sequences_raw, train_matched, train_total = _group_sequences(
        train_rows, feature_columns, target_columns, train_scores, weight_config
    )
    validation_sequences_raw, validation_matched, validation_total = _group_sequences(
        validation_rows, feature_columns, target_columns, validation_scores, weight_config
    )
    train_coverage = train_matched / max(1, train_total)
    if train_coverage < float(weight_config["min_train_score_coverage"]):
        failures.append(f"train_score_coverage_below_min:{train_coverage}")
    if not train_sequences_raw:
        failures.append("no_train_sequences")
    if not validation_sequences_raw:
        failures.append("no_validation_sequences")
    if failures:
        summary = {
            "classification": "curiosity_weighted_residual_adapter_trainer_v1_result",
            "status": "fail",
            "run_mode": run_mode,
            "run_tag": run_tag,
            "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
            "preflight_manifest": _rel(preflight_path, root),
            "learning_progress_summary": _rel(learning_progress_path, root),
            "schema_promotion": "blocked",
            "generated_trex_fields": [],
            "real_training_result": False,
            "checkpoint_written": False,
            "train_score_coverage": train_coverage,
            "failures": failures,
        }
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    feature_mean, feature_std = _mean_std(
        [(tag, feats, targets) for tag, feats, targets, _, _ in train_sequences_raw],
        len(feature_columns),
        "features",
    )
    continuous_targets = [
        (tag, feats, [row[1:] for row in targets]) for tag, feats, targets, _, _ in train_sequences_raw
    ]
    continuous_mean, continuous_std = _mean_std(continuous_targets, len(target_columns) - 1, "targets")
    train_sequences = _normalize_sequences(train_sequences_raw, feature_mean, feature_std, continuous_mean, continuous_std)
    validation_sequences = _normalize_sequences(
        validation_sequences_raw, feature_mean, feature_std, continuous_mean, continuous_std
    )
    train_features, train_active, train_continuous, train_weights, train_anchor_weights = _torch_stack_sequences(
        torch, train_sequences, device
    )
    val_features, val_active, val_continuous, val_weights, val_anchor_weights = _torch_stack_sequences(
        torch, validation_sequences, device
    )
    if run_mode == "train":
        train_repeat = int(config["real_training"].get("train_batch_repeat", 1))
        train_features = train_features.repeat(train_repeat, 1, 1)
        train_active = train_active.repeat(train_repeat, 1, 1)
        train_continuous = train_continuous.repeat(train_repeat, 1, 1)
        train_weights = train_weights.repeat(train_repeat, 1, 1)
        train_anchor_weights = train_anchor_weights.repeat(train_repeat, 1, 1)

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
    base_checkpoint = torch.load(base_checkpoint_path, map_location=device)
    model.load_state_dict(base_checkpoint["model_state_dict"])
    anchor_config = dict(weight_config.get("baseline_preservation_anchor", {}))
    anchor_target_mode = str(anchor_config.get("target_mode", "neutral"))
    reference_model = None
    if anchor_target_mode == "base_policy_distillation":
        reference_model = ResidualControllerAdapter(
            input_dim=int(architecture["input_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            gru_layers=int(architecture["gru_layers"]),
        ).to(device)
        reference_model.load_state_dict(base_checkpoint["model_state_dict"])
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    elif anchor_target_mode != "neutral":
        failures.append(f"unsupported_baseline_preservation_anchor_target_mode:{anchor_target_mode}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")

    neutral_active_target = torch.zeros_like(train_active[:1])
    neutral_continuous_values = list(anchor_config.get("neutral_continuous_targets", [1.0, 0.0, 0.0]))
    if len(neutral_continuous_values) != len(target_columns) - 1:
        failures.append("baseline_preservation_anchor_neutral_target_width_mismatch")
        neutral_continuous_values = [1.0, 0.0, 0.0][: len(target_columns) - 1]
    neutral_continuous = torch.tensor(
        [
            (float(value) - continuous_mean[idx]) / continuous_std[idx]
            for idx, value in enumerate(neutral_continuous_values)
        ],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, -1)
    anchor_active_loss_weight = float(anchor_config.get("active_loss_weight", 1.0))
    anchor_continuous_loss_weights_values = list(
        anchor_config.get("continuous_loss_weights", [1.0] * (len(target_columns) - 1))
    )
    if len(anchor_continuous_loss_weights_values) != len(target_columns) - 1:
        failures.append("baseline_preservation_anchor_continuous_loss_weights_width_mismatch")
        anchor_continuous_loss_weights_values = [1.0] * (len(target_columns) - 1)
    anchor_continuous_loss_weights = torch.tensor(
        [float(value) for value in anchor_continuous_loss_weights_values],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, -1)
    anchor_continuous_loss_weight_sum = anchor_continuous_loss_weights.sum().clamp_min(1.0)

    def weighted_loss(features, active_targets, continuous_targets, weights, anchor_weights):
        active_logits, continuous_pred = model(features)
        active_loss = bce(active_logits, active_targets)
        continuous_loss = ((continuous_pred - continuous_targets) ** 2).mean(dim=2, keepdim=True)
        per_step = (
            float(config["loss"]["feedback_active_bce_weight"]) * active_loss
            + float(config["loss"]["continuous_mse_weight"]) * continuous_loss
        )
        supervised_loss = (per_step * weights).sum() / weights.sum().clamp_min(1.0)
        if anchor_target_mode == "base_policy_distillation":
            assert reference_model is not None
            with torch.no_grad():
                reference_active_logits, reference_continuous = reference_model(features)
            anchor_active_loss = anchor_active_loss_weight * ((active_logits - reference_active_logits) ** 2)
            anchor_continuous_loss = (
                ((continuous_pred - reference_continuous) ** 2) * anchor_continuous_loss_weights
            ).sum(dim=2, keepdim=True) / anchor_continuous_loss_weight_sum
        else:
            neutral_active = neutral_active_target.expand_as(active_targets)
            neutral_cont = neutral_continuous.expand_as(continuous_targets)
            anchor_active_loss = anchor_active_loss_weight * bce(active_logits, neutral_active)
            anchor_continuous_loss = (
                ((continuous_pred - neutral_cont) ** 2) * anchor_continuous_loss_weights
            ).sum(dim=2, keepdim=True) / anchor_continuous_loss_weight_sum
        anchor_step = anchor_active_loss + anchor_continuous_loss
        anchor_loss = (anchor_step * anchor_weights).sum() / anchor_weights.sum().clamp_min(1.0)
        return supervised_loss + anchor_loss, supervised_loss, anchor_loss

    def evaluate() -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            active_logits, continuous_pred = model(val_features)
            active_loss = bce(active_logits, val_active).mean()
            continuous_loss = ((continuous_pred - val_continuous) ** 2).mean()
            weighted_validation_loss, supervised_validation_loss, anchor_validation_loss = weighted_loss(
                val_features, val_active, val_continuous, val_weights, val_anchor_weights
            )
            active_prob = torch.sigmoid(active_logits)
            active_pred = (active_prob >= 0.5).float()
            active_accuracy = (active_pred == val_active).float().mean()
            total_loss = active_loss + continuous_loss
        return {
            "loss": float(total_loss.detach().cpu()),
            "weighted_validation_loss": float(weighted_validation_loss.detach().cpu()),
            "supervised_validation_loss": float(supervised_validation_loss.detach().cpu()),
            "anchor_validation_loss": float(anchor_validation_loss.detach().cpu()),
            "validation_anchor_weight_mean": float(val_anchor_weights.mean().detach().cpu()),
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
            loss, supervised_loss, anchor_loss = weighted_loss(
                train_features, train_active, train_continuous, train_weights, train_anchor_weights
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
    checkpoint_path: Path | None = None
    if run_mode == "train" and elapsed >= target_seconds and config["real_training"]["checkpoint_written"]:
        checkpoint_path = checkpoint_dir / f"{run_tag}.pt"
        torch.save(
            {
                "classification": config.get(
                    "checkpoint_classification",
                    "newton_native_curiosity_weighted_residual_controller_adapter_v1_checkpoint",
                ),
                "model_state_dict": model.state_dict(),
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "continuous_mean": continuous_mean,
                "continuous_std": continuous_std,
                "config": config,
                "ablation_name": config.get("ablation_name"),
                "ablation_not_success_claim": bool(config.get("ablation_name")),
                "base_checkpoint": _rel(base_checkpoint_path, root),
                "learning_progress_summary": _rel(learning_progress_path, root),
            },
            checkpoint_path,
        )
        checkpoint_written = True

    summary = {
        "classification": config.get(
            "result_classification",
            "curiosity_weighted_residual_adapter_trainer_v1_result",
        ),
        "status": "pass" if not failures else "fail",
        "run_mode": run_mode,
        "run_tag": run_tag,
        "method_name": config["method_name"],
        "ablation_name": config.get("ablation_name"),
        "ablation_not_success_claim": bool(config.get("ablation_name")),
        "not_official_trex_method": True,
        "not_trex_schema": True,
        "not_rl_algorithm": True,
        "curiosity_weighted_supervised_finetune": True,
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "preflight_manifest": _rel(preflight_path, root),
        "learning_progress_summary": _rel(learning_progress_path, root),
        "base_checkpoint": _rel(base_checkpoint_path, root),
        "train_csv": preflight["train_csv"],
        "validation_csv": preflight["validation_csv"],
        "train_score_coverage": train_coverage,
        "validation_score_coverage": validation_matched / max(1, validation_total),
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
        "baseline_preservation_anchor": weight_config.get("baseline_preservation_anchor", {"enabled": False}),
        "anchor_target_mode": anchor_target_mode,
        "anchor_contact_count_columns": list(
            weight_config.get("baseline_preservation_anchor", {}).get(
                "contact_count_columns",
                ["newton.contact.rigid_contact_count", "newton.panda.rigid_contact_count"],
            )
        ),
        "train_anchor_weight_mean": float(train_anchor_weights.mean().detach().cpu()),
        "validation_metrics": validation_metrics,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        "real_training_result": run_mode == "train" and elapsed >= target_seconds and checkpoint_written,
        "smoke_diagnostic_only": run_mode == "smoke",
        "checkpoint_written": checkpoint_written,
        "checkpoint_path": _rel(checkpoint_path, root),
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
                "train_score_coverage": summary.get("train_score_coverage"),
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
