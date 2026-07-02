"""Compute learning-progress curiosity scores from two forward-model snapshots.

The score is based on prediction improvement from the initial snapshot to the
trained checkpoint, not raw prediction error alone. It does not update a policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    return 0.0 if value == "" else float(value)


def _row_float_first(row: dict[str, str], columns: list[str]) -> float:
    for column in columns:
        if column in row:
            return _as_float(row[column])
    raise KeyError(f"missing all candidate columns: {columns}")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run(config_path: Path, root: Path, fresh_sanity_json: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    fresh_sanity = _load_json(fresh_sanity_json)
    summary_path = root / config["trainer_summary"]
    preflight_path = root / config["preflight_manifest"]
    trainer_summary = _load_json(summary_path)
    preflight = _load_json(preflight_path)

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if fresh_sanity.get("status") != "pass":
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")
    if trainer_summary.get("status") != "pass" or not trainer_summary.get("real_training_result"):
        failures.append("trainer_summary_not_real_training_pass")
    if not trainer_summary.get("initial_checkpoint_written"):
        failures.append("missing_initial_checkpoint")
    if not trainer_summary.get("checkpoint_written"):
        failures.append("missing_trained_checkpoint")
    if preflight.get("status") != "pass":
        failures.append(f"preflight_status:{preflight.get('status')}")
    if preflight.get("generated_trex_fields") != []:
        failures.append("preflight_generated_trex_fields_not_empty")
    if preflight.get("schema_promotion") != "blocked":
        failures.append(f"preflight_schema_promotion:{preflight.get('schema_promotion')}")

    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        torch = None
        nn = None
        failures.append(f"torch_import_failed:{type(exc).__name__}:{exc}")

    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "curiosity_learning_progress_summary.json"
    output_csv = output_dir / "curiosity_learning_progress_scores.csv"
    score_config = dict(config["score"])
    useful_change_columns = list(
        score_config.get("useful_change_columns", ["curiosity.object.delta_z_next", "target.object.delta_z_next"])
    )
    contact_loss_columns = list(
        score_config.get("contact_loss_columns", ["curiosity.contact_loss_risk_next", "target.contact_loss_risk_next"])
    )
    slip_columns = list(score_config.get("slip_columns", ["curiosity.slip_risk_next", "target.slip_risk_next"]))
    contact_count_columns = list(
        score_config.get("contact_count_columns", ["newton.contact.rigid_contact_count", "newton.panda.rigid_contact_count"])
    )

    if failures:
        payload = {
            "classification": "curiosity_learning_progress_v1",
            "status": "fail",
            "config": _rel(config_path, root),
            "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
            "trainer_summary": _rel(summary_path, root),
            "preflight_manifest": _rel(preflight_path, root),
            "schema_promotion": "blocked",
            "generated_trex_fields": [],
            "policy_updated": False,
            "failures": failures,
        }
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    assert torch is not None and nn is not None
    initial_path = root / trainer_summary["initial_checkpoint_path"]
    trained_path = root / trainer_summary["checkpoint_path"]
    initial = torch.load(initial_path, map_location="cpu")
    trained = torch.load(trained_path, map_location="cpu")
    feature_columns = list(trained["feature_columns"])
    target_columns = list(trained["target_columns"])
    feature_mean = list(trained["feature_mean"])
    feature_std = list(trained["feature_std"])
    target_mean = list(trained["target_mean"])
    target_std = list(trained["target_std"])
    architecture = trained["config"]["architecture"]

    class CuriosityForwardModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, gru_layers: int, output_dim: int) -> None:
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=gru_layers, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

        def forward(self, features):
            hidden, _ = self.gru(self.input_norm(features))
            return self.head(hidden)

    def load_model(payload: dict[str, Any]):
        model = CuriosityForwardModel(
            input_dim=int(architecture["input_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            gru_layers=int(architecture["gru_layers"]),
            output_dim=len(target_columns),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model

    initial_model = load_model(initial)
    trained_model = load_model(trained)
    split_rows = {
        "train": _read_csv(root / preflight["train_csv"]),
        "validation": _read_csv(root / preflight["validation_csv"]),
    }
    if not split_rows["train"]:
        failures.append("no_train_rows")
    if not split_rows["validation"]:
        failures.append("no_validation_rows")

    score_rows: list[dict[str, Any]] = []
    reward_values = []
    split_summaries: dict[str, dict[str, Any]] = {}
    for split_name, rows in split_rows.items():
        features = []
        targets = []
        for row in rows:
            features.append(
                [
                    (_as_float(row[column]) - feature_mean[idx]) / feature_std[idx]
                    for idx, column in enumerate(feature_columns)
                ]
            )
            targets.append(
                [
                    (_as_float(row[column]) - target_mean[idx]) / target_std[idx]
                    for idx, column in enumerate(target_columns)
                ]
            )
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(targets, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            init_pred = initial_model(x)
            trained_pred = trained_model(x)
            init_error = ((init_pred - y) ** 2).mean(dim=2).squeeze(0)
            trained_error = ((trained_pred - y) ** 2).mean(dim=2).squeeze(0)
            progress = torch.clamp(
                init_error - trained_error,
                min=0.0,
                max=float(config["score"]["learning_progress_clip"]),
            )

        split_rewards = []
        for idx, row in enumerate(rows):
            useful_change = abs(_row_float_first(row, useful_change_columns))
            contact_loss = _row_float_first(row, contact_loss_columns)
            slip = _row_float_first(row, slip_columns)
            contact_count = _row_float_first(row, contact_count_columns)
            no_op = 1.0 if useful_change == 0.0 and contact_count <= 0.0 else 0.0
            reward = (
                float(progress[idx])
                + float(score_config["useful_change_weight"]) * useful_change
                - float(score_config["contact_loss_penalty_weight"]) * contact_loss
                - float(score_config["slip_penalty_weight"]) * slip
                - float(score_config["no_op_penalty_weight"]) * no_op
            )
            reward_values.append(reward)
            split_rewards.append(reward)
            score_rows.append(
                {
                    "split": split_name,
                    "run_tag": row["run_tag"],
                    "cell": row["cell"],
                    "timestep_index": row["timestep_index"],
                    "initial_prediction_error": float(init_error[idx]),
                    "trained_prediction_error": float(trained_error[idx]),
                    "learning_progress": float(progress[idx]),
                    "bounded_curiosity_reward": reward,
                    "contact_loss_risk": contact_loss,
                    "slip_risk": slip,
                }
            )
        split_summaries[split_name] = {
            "score_count": len(rows),
            "mean_learning_progress": float(progress.mean().detach().cpu()) if rows else None,
            "mean_bounded_curiosity_reward": sum(split_rewards) / max(1, len(split_rewards)),
        }

    _write_csv(
        output_csv,
        [
            "split",
            "run_tag",
            "cell",
            "timestep_index",
            "initial_prediction_error",
            "trained_prediction_error",
            "learning_progress",
            "bounded_curiosity_reward",
            "contact_loss_risk",
            "slip_risk",
        ],
        score_rows,
    )
    payload = {
        "classification": "curiosity_learning_progress_v1",
        "status": "pass" if not failures else "fail",
        "config": _rel(config_path, root),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "trainer_summary": _rel(summary_path, root),
        "preflight_manifest": _rel(preflight_path, root),
        "output_csv": _rel(output_csv, root),
        "score_count": len(score_rows),
        "split_summaries": split_summaries,
        "mean_learning_progress": sum(float(row["learning_progress"]) for row in score_rows) / max(1, len(score_rows)),
        "mean_bounded_curiosity_reward": sum(reward_values) / max(1, len(reward_values)),
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "policy_updated": False,
        "not_raw_prediction_error_only": True,
        "score_column_mapping": {
            "useful_change_columns": useful_change_columns,
            "contact_loss_columns": contact_loss_columns,
            "slip_columns": slip_columns,
            "contact_count_columns": contact_count_columns,
        },
        "failures": failures,
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--fresh-sanity-json", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.config, args.root, args.fresh_sanity_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
