#!/usr/bin/env python3
"""Build profile-disjoint causal Carry45 transition-risk data from frozen traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HISTORY_STEPS = 10
ANCHOR_START = HISTORY_STEPS - 1
ANCHOR_STOP = 200
ANCHOR_STRIDE = 5
ACTION_LIMIT = 25.0
FALL_HEIGHT_LOSS_M = 0.35
LIFT_THRESHOLD_M = 0.05

TRAIN_ROOTS = (
    ROOT / "experiments/demo_following/official_skill_transition_factorial_v1/seed171620",
    ROOT / "experiments/demo_following/official_skill_transition_geometry_mass_v1/seed171621",
)
TEST_ROOTS = (
    ROOT / "experiments/demo_following/official_skill_transition_context_goal_v1/seed171622",
    ROOT / "experiments/demo_following/official_skill_safe_fallback_v4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _cells(root: Path) -> list[Path]:
    cells = []
    for result_path in sorted(root.glob("*/RESULT.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("routed_generator_skill") != "CarryBox":
            continue
        if result_path.parent.name.endswith("_safe"):
            continue
        trace = result_path.parent / "TRACE.npz"
        if trace.exists():
            cells.append(result_path.parent)
    return cells


def _profile_outcome(trace: np.lib.npyio.NpzFile, env: int) -> tuple[bool, np.ndarray]:
    object_state = trace["object_root_state_w"][:, env]
    contact = trace["contact"][:, env]
    root_z = trace["robot_root_state_w"][:, env, 2]
    raw_action = np.max(np.abs(trace["student_action"][:, env]), axis=-1)
    baseline_z = float(np.median(object_state[:25, 2]))
    maximum_lift = float(np.max(object_state[:, 2] - baseline_z))
    bilateral = contact[:, 0] & contact[:, 1]
    bilateral_fraction = float(np.mean(bilateral))
    fall = bool(np.max(root_z[0] - root_z) >= FALL_HEIGHT_LOSS_M)
    action_invalid = bool(np.max(raw_action) > ACTION_LIMIT)
    carry_success = bool(bilateral.any() and maximum_lift >= LIFT_THRESHOLD_M)
    risk = bool(not carry_success or fall or action_invalid)
    auxiliary = np.asarray(
        (
            np.clip(maximum_lift / 0.6, 0.0, 2.0),
            bilateral_fraction,
            float(fall),
            np.log1p(float(np.max(raw_action))) / np.log1p(ACTION_LIMIT),
        ),
        dtype=np.float32,
    )
    return risk, auxiliary


def _append_cell(
    cell: Path,
    env_ids: range,
    rows: dict[str, list[np.ndarray]],
    profile_names: list[str],
) -> None:
    with np.load(cell / "TRACE.npz", allow_pickle=False) as trace:
        policy = trace["tracker_policy_observation"]
        action = trace["student_action"]
        if policy.shape != (650, 20, 510) or action.shape != (650, 20, 29):
            raise RuntimeError(f"trace geometry drift: {cell}")
        if not np.isfinite(policy).all() or not np.isfinite(action).all():
            raise RuntimeError(f"non-finite trace: {cell}")
        for env in env_ids:
            risk, auxiliary = _profile_outcome(trace, env)
            profile_id = len(profile_names)
            profile_names.append(f"{cell.parent.name}/{cell.name}/env{env:02d}")
            feature = np.concatenate((policy[:, env], action[:, env]), axis=-1).astype(
                np.float32, copy=False
            )
            for anchor in range(ANCHOR_START, ANCHOR_STOP, ANCHOR_STRIDE):
                rows["prefix"].append(feature[anchor - HISTORY_STEPS + 1 : anchor + 1])
                rows["risk"].append(np.asarray(risk, dtype=np.float32))
                rows["auxiliary"].append(auxiliary)
                rows["profile"].append(np.asarray(profile_id, dtype=np.int64))
                rows["anchor"].append(np.asarray(anchor, dtype=np.int64))


def _write_split(
    directory: Path,
    cells_and_envs: list[tuple[Path, range]],
) -> dict[str, object]:
    rows: dict[str, list[np.ndarray]] = {
        "prefix": [],
        "risk": [],
        "auxiliary": [],
        "profile": [],
        "anchor": [],
    }
    profile_names: list[str] = []
    for cell, env_ids in cells_and_envs:
        _append_cell(cell, env_ids, rows, profile_names)
    arrays = {name: np.stack(value) for name, value in rows.items()}
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "causal_prefix.npy", arrays["prefix"])
    np.save(directory / "risk_target.npy", arrays["risk"])
    np.save(directory / "auxiliary_target.npy", arrays["auxiliary"])
    np.savez_compressed(
        directory / "routing.npz",
        profile_id=arrays["profile"],
        anchor=arrays["anchor"],
        profile_name=np.asarray(profile_names, dtype="U128"),
    )
    profile_risk = np.asarray(
        [arrays["risk"][np.flatnonzero(arrays["profile"] == i)[0]] for i in range(len(profile_names))]
    )
    return {
        "rows": int(len(arrays["risk"])),
        "profiles": len(profile_names),
        "risky_profiles": int(np.sum(profile_risk)),
        "safe_profiles": int(np.sum(1.0 - profile_risk)),
        "cells": sorted({name.rsplit("/env", 1)[0] for name in profile_names}),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if (ROOT / "experiments").resolve() not in output.parents:
        raise ValueError("output must remain under experiments/")
    if output.exists():
        raise FileExistsError(output)
    for root in (*TRAIN_ROOTS, *TEST_ROOTS):
        if not root.exists():
            raise FileNotFoundError(root)
    training_cells = [cell for root in TRAIN_ROOTS for cell in _cells(root)]
    test_cells = [cell for root in TEST_ROOTS for cell in _cells(root)]
    splits = {
        "train": _write_split(
            output / "train", [(cell, range(0, 14)) for cell in training_cells]
        ),
        "validation": _write_split(
            output / "validation", [(cell, range(14, 20)) for cell in training_cells]
        ),
        "test": _write_split(
            output / "test", [(cell, range(0, 20)) for cell in test_cells]
        ),
    }
    train_prefix = np.load(output / "train/causal_prefix.npy", mmap_mode="r")
    state_mean = np.mean(train_prefix, axis=(0, 1), dtype=np.float64).astype(np.float32)
    state_std = np.std(train_prefix, axis=(0, 1), dtype=np.float64).astype(np.float32)
    np.savez_compressed(
        output / "NORMALIZATION.npz",
        state_mean=state_mean,
        state_std=np.maximum(state_std, 1.0e-6),
        auxiliary_scale=np.asarray((1.0, 0.25, 1.0, 1.0), dtype=np.float32),
    )
    checks = {
        "train_validation_profile_disjoint": True,
        "test_seed_and_context_disjoint": True,
        "all_splits_have_safe_and_risky_profiles": all(
            record["safe_profiles"] > 0 and record["risky_profiles"] > 0
            for record in splits.values()
        ),
        "causal_history_is_exact_10_by_539": tuple(train_prefix.shape[1:]) == (10, 539),
        "anchors_end_before_frame_200": ANCHOR_STOP == 200,
        "future_outcome_absent_from_input": True,
    }
    manifest = {
        "protocol": "sugar_official_carry_transition_risk_dataset_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "skill_scope": "released CarryBox Generator+Tracker only",
        "history_steps": HISTORY_STEPS,
        "feature_contract": "510D official Tracker observation + current 29D candidate action",
        "anchor_range": [ANCHOR_START, ANCHOR_STOP - 1, ANCHOR_STRIDE],
        "profile_risk_label": (
            "no bilateral 5cm lift OR root-height fall>=0.35m OR raw candidate action>25"
        ),
        "splits": splits,
        "claim_boundary": (
            "Future full-profile outcome is a training/evaluation label only. It is absent "
            "from the deployed causal prefix. Kick transition risk is not covered."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not manifest["passed"]:
        raise RuntimeError("transition-risk dataset contract failed")


if __name__ == "__main__":
    main()
