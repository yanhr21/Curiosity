"""Build Phase07 curiosity ablation score CSVs from real Newton transitions.

This script does not train a policy and does not create a replacement model.
It preserves the existing residual-adapter training path and only generates
controlled intrinsic-score variants for ablation training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_float(value: str | None) -> float:
    return 0.0 if value in {None, ""} else float(value)


def _stable_random(seed: int, run_tag: str, timestep_index: str) -> float:
    digest = hashlib.sha256(f"{seed}:{run_tag}:{timestep_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _normalize(values: list[float]) -> list[float]:
    positive_max = max([abs(value) for value in values] or [1.0])
    if positive_max <= 0:
        return [0.0 for _ in values]
    return [max(0.0, value) / positive_max for value in values]


def _load_base_scores(summary: dict[str, Any], root: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    csv_path = root / summary["output_csv"]
    rows = _read_csv(csv_path)
    return {(row["split"], row["run_tag"], row["timestep_index"]): row for row in rows}


def run(config_path: Path, root: Path, fresh_sanity_json: Path, run_tag: str) -> dict[str, Any]:
    config = _load_json(config_path)
    fresh_sanity = _load_json(fresh_sanity_json)
    preflight = _load_json(root / config["preflight_manifest"])
    base_summary = _load_json(root / config["base_learning_progress_summary"])

    failures: list[str] = []
    if os.environ.get("SLURM_JOB_ID") is None:
        failures.append("not_inside_slurm_allocation")
    if fresh_sanity.get("status") != "pass":
        failures.append(f"fresh_official_newton_sanity_not_pass:{fresh_sanity.get('status')}")
    if preflight.get("status") != "pass":
        failures.append(f"preflight_status:{preflight.get('status')}")
    if preflight.get("schema_promotion") != "blocked":
        failures.append(f"preflight_schema_promotion:{preflight.get('schema_promotion')}")
    if preflight.get("generated_trex_fields") != []:
        failures.append("preflight_generated_trex_fields_not_empty")
    if base_summary.get("status") != "pass":
        failures.append(f"base_learning_progress_status:{base_summary.get('status')}")
    if base_summary.get("policy_updated") is not False:
        failures.append("base_learning_progress_claimed_policy_update")

    output_root = root / config["output_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "curiosity_ablation_score_manifest.json"

    if failures:
        payload = {
            "classification": "curiosity_ablation_score_manifest_v1",
            "status": "fail",
            "run_tag": run_tag,
            "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
            "schema_promotion": "blocked",
            "generated_trex_fields": [],
            "policy_updated": False,
            "failures": failures,
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    split_paths = {
        "train": root / preflight["train_csv"],
        "validation": root / preflight["validation_csv"],
    }
    rows_by_split = {split: _read_csv(path) for split, path in split_paths.items()}
    base_scores = _load_base_scores(base_summary, root)
    seed = int(config.get("seed", 20260627))

    all_rows = []
    for split, rows in rows_by_split.items():
        for row in rows:
            copied = dict(row)
            copied["split"] = split
            all_rows.append(copied)

    object_raw = [
        abs(_as_float(row.get("curiosity.object.delta_z_next")))
        + 0.1 * abs(_as_float(row.get("curiosity.object.velocity_z_next")))
        + 0.1 * abs(_as_float(row.get("curiosity.lift_response_residual_next")))
        for row in all_rows
    ]
    contact_raw = [
        abs(_as_float(row.get("curiosity.contact.delta_count_next")))
        + 2.0 * _as_float(row.get("curiosity.contact_loss_risk_next"))
        + _as_float(row.get("curiosity.slip_risk_next"))
        for row in all_rows
    ]
    object_scores = _normalize(object_raw)
    contact_scores = _normalize(contact_raw)

    shuffled_contact_scores = contact_scores[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled_contact_scores)

    previous_contact_by_run: dict[str, float] = {}
    delayed_contact_scores = []
    for row, contact_score in zip(all_rows, contact_scores, strict=True):
        run = row["run_tag"]
        delayed_contact_scores.append(previous_contact_by_run.get(run, 0.0))
        previous_contact_by_run[run] = contact_score

    variant_values = {
        "random_intrinsic": [
            _stable_random(seed, row["run_tag"], row["timestep_index"]) for row in all_rows
        ],
        "object_only": object_scores,
        "contact_only": contact_scores,
        "shuffled_contact": shuffled_contact_scores,
        "delayed_contact": delayed_contact_scores,
    }
    variant_values["no_learning_progress"] = [
        max(
            0.0,
            float(config["no_learning_progress"]["object_weight"]) * object_score
            + float(config["no_learning_progress"]["contact_weight"]) * contact_score
            - float(config["no_learning_progress"]["contact_loss_penalty_weight"])
            * _as_float(row.get("curiosity.contact_loss_risk_next"))
            - float(config["no_learning_progress"]["slip_penalty_weight"])
            * _as_float(row.get("curiosity.slip_risk_next")),
        )
        for row, object_score, contact_score in zip(all_rows, object_scores, contact_scores, strict=True)
    ]

    generated = {}
    fieldnames = [
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
        "ablation_name",
        "ablation_raw_reward",
        "score_source",
    ]
    for variant in config["variants"]:
        values = variant_values[variant]
        out_dir = output_root / variant
        out_csv = out_dir / "curiosity_ablation_scores.csv"
        out_summary = out_dir / "curiosity_learning_progress_summary.json"
        out_rows = []
        for row, value in zip(all_rows, values, strict=True):
            key = (row["split"], row["run_tag"], row["timestep_index"])
            base = base_scores.get(key, {})
            out_rows.append(
                {
                    "split": row["split"],
                    "run_tag": row["run_tag"],
                    "cell": row["cell"],
                    "timestep_index": row["timestep_index"],
                    "initial_prediction_error": base.get("initial_prediction_error", ""),
                    "trained_prediction_error": base.get("trained_prediction_error", ""),
                    "learning_progress": 0.0 if variant == "no_learning_progress" else base.get("learning_progress", ""),
                    "bounded_curiosity_reward": value,
                    "contact_loss_risk": row.get("curiosity.contact_loss_risk_next", ""),
                    "slip_risk": row.get("curiosity.slip_risk_next", ""),
                    "ablation_name": variant,
                    "ablation_raw_reward": value,
                    "score_source": "phase07_real_newton_transition_ablation",
                }
            )
        _write_csv(out_csv, fieldnames, out_rows)
        train_values = [float(row["bounded_curiosity_reward"]) for row in out_rows if row["split"] == "train"]
        val_values = [float(row["bounded_curiosity_reward"]) for row in out_rows if row["split"] == "validation"]
        variant_summary = {
            "classification": "curiosity_ablation_scores_v1",
            "status": "pass",
            "run_tag": run_tag,
            "ablation_name": variant,
            "output_csv": _rel(out_csv, root),
            "score_count": len(out_rows),
            "train_score_count": len(train_values),
            "validation_score_count": len(val_values),
            "mean_train_score": sum(train_values) / max(1, len(train_values)),
            "mean_validation_score": sum(val_values) / max(1, len(val_values)),
            "preflight_manifest": config["preflight_manifest"],
            "base_learning_progress_summary": config["base_learning_progress_summary"],
            "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
            "schema_promotion": "blocked",
            "generated_trex_fields": [],
            "policy_updated": False,
            "ablation_not_success_claim": True,
            "failures": [],
        }
        out_summary.write_text(json.dumps(variant_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        generated[variant] = _rel(out_summary, root)

    payload = {
        "classification": "curiosity_ablation_score_manifest_v1",
        "status": "pass",
        "run_tag": run_tag,
        "variants": generated,
        "variant_count": len(generated),
        "fresh_official_newton_sanity": _rel(fresh_sanity_json, root),
        "schema_promotion": "blocked",
        "generated_trex_fields": [],
        "policy_updated": False,
        "failures": [],
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--fresh-sanity-json", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    payload = run(args.config, args.root, args.fresh_sanity_json, args.run_tag)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
