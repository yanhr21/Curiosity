#!/usr/bin/env python3
"""Visualize a saved official SUGAR refiner rollout sample."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--expected-windows", type=int, default=16)
    parser.add_argument("--checkpoint-label", default="model_5000")
    parser.add_argument("--policy-stage", default="refiner")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if socket.gethostname().startswith("mgmtserver"):
        raise SystemExit("Refusing rollout visualization generation on a login/management node")

    import matplotlib.pyplot as plt
    import numpy as np

    args = parse_args()
    files = sorted((args.rollout_dir / "trajectory_complete").glob("*.npz"))
    if not files:
        raise SystemExit(f"No official completed rollout NPZ files under {args.rollout_dir}")

    records = []
    for path in files:
        with np.load(path) as data:
            required = {"obj_pos_b", "target_obj_pos_b", "motion_id", "fps", "num_timesteps"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise SystemExit(f"Missing keys in {path}: {missing}")
            final_relative_position_discrepancy = float(
                np.linalg.norm(data["obj_pos_b"][-1] - data["target_obj_pos_b"][-1])
            )
            fps = int(data["fps"])
            num_timesteps = int(data["num_timesteps"])
            records.append(
                {
                    "motion_id": int(data["motion_id"]),
                    "num_timesteps": num_timesteps,
                    "duration_seconds": num_timesteps / fps,
                    "final_relative_position_discrepancy": final_relative_position_discrepancy,
                    "file": str(path),
                }
            )

    records.sort(key=lambda record: record["motion_id"])
    complete = len(records)
    incomplete_or_unsaved = max(args.expected_windows - complete, 0)
    labels = [f"motion {record['motion_id']}" for record in records]
    errors = [record["final_relative_position_discrepancy"] for record in records]
    durations = [record["duration_seconds"] for record in records]

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    axes[0].barh(
        ["sampled windows"],
        [complete],
        label=f"trajectory_complete ({complete})",
        color="#2ca02c",
    )
    axes[0].barh(
        ["sampled windows"],
        [incomplete_or_unsaved],
        left=[complete],
        label=f"incomplete / not saved ({incomplete_or_unsaved})",
        color="#d62728",
    )
    axes[0].set_xlim(0, args.expected_windows)
    axes[0].set_xlabel("window count")
    axes[0].set_title(
        f"Official SUGAR CarryBox {args.policy_stage} {args.checkpoint_label}: sampled rollout completion "
        f"({complete}/{args.expected_windows}, {100.0 * complete / args.expected_windows:.2f}%)"
    )
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.25)

    if complete <= 64:
        axes[1].bar(labels, errors, color="#1f77b4")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].hist(errors, bins=30, color="#1f77b4", edgecolor="white")
        axes[1].axvline(float(np.mean(errors)), color="#d62728", linestyle="--", label="mean")
        axes[1].axvline(float(np.median(errors)), color="#2ca02c", linestyle=":", label="median")
        axes[1].legend()
    axes[1].set_ylabel("final ||obj_pos_b - target_obj_pos_b||")
    axes[1].set_title(
        "Derived final relative-position discrepancy for completed trajectories"
        if complete <= 64
        else "Distribution of derived final relative-position discrepancy"
    )
    axes[1].grid(axis="y", alpha=0.25)

    if complete <= 64:
        axes[2].bar(labels, durations, color="#ff7f0e")
        axes[2].tick_params(axis="x", rotation=45)
    else:
        axes[2].hist(durations, bins=30, color="#ff7f0e", edgecolor="white")
        axes[2].axvline(float(np.mean(durations)), color="#d62728", linestyle="--", label="mean")
        axes[2].axvline(float(np.median(durations)), color="#2ca02c", linestyle=":", label="median")
        axes[2].legend()
    axes[2].set_ylabel("duration (seconds)")
    axes[2].set_title(
        "Completed trajectory durations"
        if complete <= 64
        else "Distribution of completed trajectory durations"
    )
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Intermediate official {args.policy_stage} sample only — not full SUGAR and not the paper CarryBox SR/Err",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white", transparent=False)
    plt.close(fig)

    summary = {
        "scope": (
            f"official_sugar_carrybox_{args.policy_stage}_"
            f"{args.checkpoint_label}_sampled_rollout_windows"
        ),
        "checkpoint_label": args.checkpoint_label,
        "policy_stage": args.policy_stage,
        "comparable_to_paper": False,
        "expected_windows": args.expected_windows,
        "trajectory_complete": complete,
        "completion_rate_percent": 100.0 * complete / args.expected_windows,
        "derived_metric": "final_l2_norm_obj_pos_b_minus_target_obj_pos_b",
        "mean_final_relative_position_discrepancy": float(np.mean(errors)),
        "median_final_relative_position_discrepancy": float(np.median(errors)),
        "records": records,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[SUGAR-REFINER-ROLLOUT-VIS] checkpoint={args.checkpoint_label}")
    print(f"[SUGAR-REFINER-ROLLOUT-VIS] wrote={args.output}")
    print(f"[SUGAR-REFINER-ROLLOUT-VIS] summary={args.summary_json}")


if __name__ == "__main__":
    main()
