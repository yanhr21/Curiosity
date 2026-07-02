#!/usr/bin/env python3
"""Validate a future Newton-to-T-Rex dataset metadata contract.

This is a lightweight schema checker. It does not load videos, tensors, models,
or checkpoints; full dataset validation must run on compute resources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACTION_DIM = 62
ACTION_CHUNK = 16
N_FINGERS = 10
F6_PER_FINGER = 6

KEY_HEAD = "observation.images.head"
KEY_WRIST_R = "observation.images.wrist_right"
KEY_WRIST_L = "observation.images.wrist_left"
KEY_STATE = "observation.state"
KEY_ACTION = "action"
KEY_ACTION_ABS = "action_abs"
KEY_TACF6 = "observation.tactile_f6"
DEFORM_KEYS = [f"observation.tactile_deform.l{i}" for i in range(5)] + [
    f"observation.tactile_deform.r{i}" for i in range(5)
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def feature_shape(feature: dict[str, Any]) -> list[Any]:
    return as_list(feature.get("shape"))


def check_video_feature(name: str, feature: dict[str, Any], failures: list[str]) -> None:
    shape = feature_shape(feature)
    if feature.get("dtype") != "video":
        failures.append(f"{name}: expected dtype=video, got {feature.get('dtype')}")
    if len(shape) != 3 or shape[0] != 3:
        failures.append(f"{name}: expected shape [3,H,W], got {shape}")


def check_num_feature(name: str, feature: dict[str, Any], expected_shape: list[int], failures: list[str]) -> None:
    shape = feature_shape(feature)
    if feature.get("dtype") != "float32":
        failures.append(f"{name}: expected dtype=float32, got {feature.get('dtype')}")
    if shape != expected_shape:
        failures.append(f"{name}: expected shape {expected_shape}, got {shape}")


def check_stats_block(stats: dict[str, Any], failures: list[str]) -> None:
    if not stats:
        failures.append("stats: empty or missing stats block")
        return
    block = stats[next(iter(stats))]
    required = {
        "action": [ACTION_CHUNK, ACTION_DIM],
        "state": [ACTION_DIM],
        "tactile_f6": [N_FINGERS * F6_PER_FINGER],
    }
    for key, expected_shape in required.items():
        item = block.get(key)
        if not isinstance(item, dict):
            failures.append(f"stats.{key}: missing")
            continue
        for subkey in ("q01", "q99", "mask"):
            value = item.get(subkey)
            if value is None:
                failures.append(f"stats.{key}.{subkey}: missing")
                continue
            if key == "action":
                ok = len(value) == expected_shape[0] and all(len(row) == expected_shape[1] for row in value)
            else:
                ok = len(value) == expected_shape[0]
            if not ok:
                failures.append(f"stats.{key}.{subkey}: expected shape {expected_shape}, got top length {len(value)}")


def validate(info_json: Path, stats_json: Path | None = None) -> dict[str, Any]:
    info = load_json(info_json)
    features = info.get("features", {})
    failures: list[str] = []

    for name in (KEY_HEAD, KEY_WRIST_R, KEY_WRIST_L):
        feature = features.get(name)
        if not isinstance(feature, dict):
            failures.append(f"{name}: missing")
        else:
            check_video_feature(name, feature, failures)

    numeric = {
        KEY_STATE: [ACTION_DIM],
        KEY_ACTION: [ACTION_CHUNK, ACTION_DIM],
        KEY_ACTION_ABS: [ACTION_DIM],
        KEY_TACF6: [N_FINGERS, F6_PER_FINGER],
    }
    for name, shape in numeric.items():
        feature = features.get(name)
        if not isinstance(feature, dict):
            failures.append(f"{name}: missing")
        else:
            check_num_feature(name, feature, shape, failures)

    for name in DEFORM_KEYS:
        feature = features.get(name)
        if not isinstance(feature, dict):
            failures.append(f"{name}: missing")
        else:
            check_video_feature(name, feature, failures)

    if stats_json is not None:
        check_stats_block(load_json(stats_json), failures)

    return {
        "classification": "trex_contract_metadata_validation_not_training_not_model_load",
        "info_json": str(info_json),
        "stats_json": str(stats_json) if stats_json else None,
        "status": "pass_trex_metadata_contract" if not failures else "fail_trex_metadata_contract",
        "failures": failures,
        "not_training": True,
        "not_model_load": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-json", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = validate(args.info_json, args.stats_json)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["status"].startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
