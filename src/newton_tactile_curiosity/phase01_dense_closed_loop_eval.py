#!/usr/bin/env python3
"""Evaluate a Phase01 dense closed-loop probe checkpoint against base action."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from newton_tactile_curiosity.phase01_dense_closed_loop_probe import (
    FEATURE_ABLATION_MODES,
    PARAM_NAMES,
    OnlineLinearPredictor,
    result_to_row,
    run_episode,
)


def load_best_params(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["best_params"], dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=["cube", "pen"], default="cube")
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument("--override-mu", type=float, default=0.3)
    parser.add_argument("--override-kh", type=float, default=1.0e12)
    parser.add_argument("--score-lift-weight", type=float, default=4.0)
    parser.add_argument("--score-final-lift-weight", type=float, default=0.0)
    parser.add_argument("--score-hold-weight", type=float, default=0.01)
    parser.add_argument("--score-tail-hold-weight", type=float, default=0.0)
    parser.add_argument("--score-drop-weight", type=float, default=2.0)
    parser.add_argument("--hold-lift-threshold", type=float, default=0.08)
    parser.add_argument("--stable-tail-frames", type=int, default=60)
    parser.add_argument("--max-safe-slip-proxy", type=float, default=0.85)
    parser.add_argument("--max-safe-fn", type=float, default=180.0)
    parser.add_argument("--feature-ablation", choices=FEATURE_ABLATION_MODES, default="none")
    parser.add_argument("--feature-noise-std", type=float, default=0.15)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.perf_counter()
    checkpoint_params = load_best_params(args.checkpoint)
    zero_params = np.zeros(len(PARAM_NAMES), dtype=np.float32)
    methods = [
        ("base_zero_action", zero_params),
        ("checkpoint_policy", checkpoint_params),
    ]

    rows: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in methods}
    episode = 0
    for method_name, params in methods:
        for rep in range(args.repetitions):
            predictor = OnlineLinearPredictor(16, 12, lr=0.0)
            run_args = SimpleNamespace(**vars(args))
            run_args.predictor_lr = 0.0
            run_args.intrinsic_weight = 0.0
            run_args.safety_weight = 1.0
            run_args.min_duration_s = 0.0
            result = run_episode(run_args, params, predictor, episode, rep, 0)
            row = result_to_row(result)
            row["method"] = method_name
            row["repetition"] = rep
            rows.append(row)
            summaries[method_name].append(row)
            episode += 1

    def mean_metric(method: str, metric: str) -> float:
        vals = [float(row[metric]) for row in summaries[method]]
        return float(sum(vals) / max(1, len(vals)))

    base_lift = mean_metric("base_zero_action", "max_lift")
    ckpt_lift = mean_metric("checkpoint_policy", "max_lift")
    base_hold = mean_metric("base_zero_action", "hold_frames")
    ckpt_hold = mean_metric("checkpoint_policy", "hold_frames")
    base_safety = mean_metric("base_zero_action", "safety_cost")
    ckpt_safety = mean_metric("checkpoint_policy", "safety_cost")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dense_closed_loop_eval_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elapsed_s = float(time.perf_counter() - started)
    safety_regression = ckpt_safety > base_safety + 1.0e-6
    status = "pass_eval_smoke_metrics_ready" if args.smoke else "complete_eval_metrics"
    summary = {
        "classification": "phase01_dense_closed_loop_eval_v1",
        "run_tag": args.run_tag,
        "status": status,
        "smoke": bool(args.smoke),
        "not_training_result": True,
        "not_curiosity_success": True,
        "checkpoint": str(args.checkpoint),
        "methods": [name for name, _ in methods],
        "feature_ablation": args.feature_ablation,
        "feature_noise_std": float(args.feature_noise_std),
        "repetitions": int(args.repetitions),
        "base_mean_max_lift": base_lift,
        "checkpoint_mean_max_lift": ckpt_lift,
        "delta_max_lift": ckpt_lift - base_lift,
        "base_mean_hold_frames": base_hold,
        "checkpoint_mean_hold_frames": ckpt_hold,
        "delta_hold_frames": ckpt_hold - base_hold,
        "base_mean_safety_cost": base_safety,
        "checkpoint_mean_safety_cost": ckpt_safety,
        "safety_regression": bool(safety_regression),
        "metrics_csv": str(csv_path),
        "elapsed_s": elapsed_s,
        "success_claim_allowed": False,
        "next_required_step": "one-hour training attempt plus held-out strongest-baseline evaluation and videos",
    }
    summary_path = args.output_dir / "dense_closed_loop_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report_path = args.report_dir / "dense_closed_loop_eval.md"
    report_path.write_text(
        "# Phase01 Dense Closed-Loop Eval\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{status}`\n"
        f"- smoke: `{summary['smoke']}`\n"
        f"- delta max lift: `{summary['delta_max_lift']}`\n"
        f"- delta hold frames: `{summary['delta_hold_frames']}`\n"
        f"- safety regression: `{summary['safety_regression']}`\n"
        f"- metrics: `{csv_path}`\n\n"
        "This is evaluation plumbing only. It is not a success claim because it is a smoke comparison, not a one-hour trained policy on harder held-out tasks with videos.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status.startswith("pass_") or status.startswith("complete_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
