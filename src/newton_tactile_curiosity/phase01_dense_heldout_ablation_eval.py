#!/usr/bin/env python3
"""Held-out cell and feature-ablation evaluation for Phase01 dense probing.

This script evaluates an existing checkpoint. It does not train and must not be
reported as a one-hour attempt or final curiosity success.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from newton_tactile_curiosity.phase01_dense_closed_loop_eval import load_best_params
from newton_tactile_curiosity.phase01_dense_closed_loop_probe import (
    FEATURE_ABLATION_MODES,
    PARAM_NAMES,
    OnlineLinearPredictor,
    result_to_row,
    run_episode,
)


@dataclass(frozen=True)
class EvalCell:
    name: str
    scene: str
    mu: float
    kh: float
    heldout: bool


def parse_cell_spec(spec: str) -> EvalCell:
    parts = spec.split(":")
    if len(parts) not in {4, 5}:
        raise ValueError(f"cell spec must be name:scene:mu:kh[:heldout], got {spec!r}")
    name, scene, mu, kh = parts[:4]
    if scene not in {"cube", "pen"}:
        raise ValueError(f"unsupported scene in cell spec {spec!r}")
    heldout = True if len(parts) == 4 else parts[4].lower() in {"1", "true", "yes", "heldout"}
    return EvalCell(name=name, scene=scene, mu=float(mu), kh=float(kh), heldout=heldout)


def method_specs(checkpoint_params: np.ndarray) -> list[tuple[str, np.ndarray, str]]:
    zero = np.zeros(len(PARAM_NAMES), dtype=np.float32)
    no_probe = checkpoint_params.copy()
    no_probe[PARAM_NAMES.index("probe_amplitude")] = 0.0
    no_balance = checkpoint_params.copy()
    no_balance[PARAM_NAMES.index("balance_gain")] = 0.0
    return [
        ("base_zero_action", zero, "none"),
        ("checkpoint_full_dense", checkpoint_params, "none"),
        ("checkpoint_vision_only_proxy", checkpoint_params, "vision_only_proxy"),
        ("checkpoint_tactile_only_proxy", checkpoint_params, "tactile_only_proxy"),
        ("checkpoint_noisy_tactile", checkpoint_params, "noisy_tactile"),
        ("checkpoint_shuffled_lr_tactile", checkpoint_params, "shuffled_lr_tactile"),
        ("checkpoint_no_probe_action", no_probe, "none"),
        ("checkpoint_no_balance_action", no_balance, "none"),
    ]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return float(sum(vals) / max(1, len(vals)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cells", nargs="+", default=[
        "train_like_pen_mu005:pen:0.05:1.0e12:false",
        "heldout_pen_mu004:pen:0.04:1.0e12:true",
        "heldout_pen_mu006:pen:0.06:1.0e12:true",
        "heldout_pen_mu003:pen:0.03:1.0e12:true",
    ])
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--score-lift-weight", type=float, default=4.0)
    parser.add_argument("--score-final-lift-weight", type=float, default=0.0)
    parser.add_argument("--score-hold-weight", type=float, default=0.03)
    parser.add_argument("--score-tail-hold-weight", type=float, default=0.0)
    parser.add_argument("--score-drop-weight", type=float, default=6.0)
    parser.add_argument("--hold-lift-threshold", type=float, default=0.08)
    parser.add_argument("--stable-tail-frames", type=int, default=60)
    parser.add_argument("--max-safe-slip-proxy", type=float, default=0.85)
    parser.add_argument("--max-safe-fn", type=float, default=180.0)
    parser.add_argument("--feature-noise-std", type=float, default=0.15)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.perf_counter()
    checkpoint_params = load_best_params(args.checkpoint)
    cells = [parse_cell_spec(spec) for spec in args.cells]
    methods = method_specs(checkpoint_params)
    invalid_modes = [mode for _, _, mode in methods if mode not in FEATURE_ABLATION_MODES]
    if invalid_modes:
        raise ValueError(f"invalid feature ablation modes: {invalid_modes}")

    rows: list[dict[str, Any]] = []
    episode = 0
    for cell_idx, cell in enumerate(cells):
        for method_name, params, feature_ablation in methods:
            for rep in range(args.repetitions):
                predictor = OnlineLinearPredictor(16, 12, lr=0.0)
                run_args = SimpleNamespace(**vars(args))
                run_args.scene = cell.scene
                run_args.override_mu = cell.mu
                run_args.override_kh = cell.kh
                run_args.predictor_lr = 0.0
                run_args.intrinsic_weight = 0.0
                run_args.safety_weight = 1.0
                run_args.min_duration_s = 0.0
                run_args.feature_ablation = feature_ablation
                result = run_episode(run_args, params, predictor, episode, cell_idx, rep)
                row = result_to_row(result)
                row.update(
                    {
                        "cell": cell.name,
                        "scene": cell.scene,
                        "override_mu": cell.mu,
                        "override_kh": cell.kh,
                        "heldout_cell": cell.heldout,
                        "method": method_name,
                        "feature_ablation": feature_ablation,
                        "repetition": rep,
                    }
                )
                rows.append(row)
                episode += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dense_heldout_ablation_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    cell_results: list[dict[str, Any]] = []
    for cell in cells:
        aggregate[cell.name] = {}
        full_rows = [row for row in rows if row["cell"] == cell.name and row["method"] == "checkpoint_full_dense"]
        base_rows = [row for row in rows if row["cell"] == cell.name and row["method"] == "base_zero_action"]
        for method_name, _, _ in methods:
            selected = [row for row in rows if row["cell"] == cell.name and row["method"] == method_name]
            aggregate[cell.name][method_name] = {
                "mean_max_lift": mean(selected, "max_lift"),
                "mean_hold_frames": mean(selected, "hold_frames"),
                "mean_safety_cost": mean(selected, "safety_cost"),
                "mean_drop_after_lift": mean(selected, "drop_after_lift"),
            }
        full = aggregate[cell.name]["checkpoint_full_dense"]
        base = aggregate[cell.name]["base_zero_action"]
        beats_base = (
            full["mean_max_lift"] > base["mean_max_lift"] + 1.0e-6
            and full["mean_hold_frames"] >= base["mean_hold_frames"]
            and full["mean_safety_cost"] <= base["mean_safety_cost"] + 1.0e-6
        )
        cell_results.append(
            {
                "cell": cell.name,
                "heldout": cell.heldout,
                "base_mean_max_lift": base["mean_max_lift"],
                "checkpoint_mean_max_lift": full["mean_max_lift"],
                "delta_max_lift": full["mean_max_lift"] - base["mean_max_lift"],
                "base_mean_hold_frames": base["mean_hold_frames"],
                "checkpoint_mean_hold_frames": full["mean_hold_frames"],
                "delta_hold_frames": full["mean_hold_frames"] - base["mean_hold_frames"],
                "base_mean_safety_cost": base["mean_safety_cost"],
                "checkpoint_mean_safety_cost": full["mean_safety_cost"],
                "safety_regression": full["mean_safety_cost"] > base["mean_safety_cost"] + 1.0e-6,
                "checkpoint_full_beats_base_zero_action": beats_base,
            }
        )

    heldout_results = [item for item in cell_results if item["heldout"]]
    heldout_all_beat_base = bool(heldout_results) and all(item["checkpoint_full_beats_base_zero_action"] for item in heldout_results)
    safety_regression_any = any(item["safety_regression"] for item in cell_results)
    elapsed_s = float(time.perf_counter() - started)
    summary = {
        "classification": "phase01_dense_heldout_ablation_eval_v1",
        "run_tag": args.run_tag,
        "status": "complete_heldout_ablation_eval",
        "checkpoint": str(args.checkpoint),
        "not_training_result": True,
        "real_training_attempt": False,
        "not_final_curiosity_success": True,
        "success_claim_allowed": False,
        "success_claim_blockers": [
            "strongest available baseline in this suite is only base_zero_action; scripted/no-curiosity learned dense baseline is still missing",
            "vision_only_proxy is not true camera vision because the current controller uses dense tactile/mechanics features but not image observations",
            "this is evaluation-only and does not add tactile-mask training to the policy optimization loop",
            "serious official reference-method comparison or faithful blocker is still missing",
        ],
        "cells": [cell.__dict__ for cell in cells],
        "methods": [name for name, _, _ in methods],
        "repetitions": int(args.repetitions),
        "heldout_all_beat_base_zero_action": heldout_all_beat_base,
        "safety_regression_any": safety_regression_any,
        "cell_results": cell_results,
        "aggregate": aggregate,
        "metrics_csv": str(csv_path),
        "elapsed_s": elapsed_s,
    }
    summary_path = args.output_dir / "dense_heldout_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report_path = args.report_dir / "dense_heldout_ablation_eval.md"
    report_path.write_text(
        "# Phase01 Dense Held-Out Ablation Eval\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{summary['status']}`\n"
        f"- heldout_all_beat_base_zero_action: `{heldout_all_beat_base}`\n"
        f"- safety_regression_any: `{safety_regression_any}`\n"
        f"- metrics: `{csv_path}`\n"
        f"- summary: `{summary_path}`\n\n"
        "This is an evaluation-only held-out and ablation suite. It is not final curiosity success because stronger baselines, true vision integration, tactile-mask training, and serious-method comparison remain open.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
