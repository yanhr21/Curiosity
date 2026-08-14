#!/usr/bin/env python3
"""Summarize one completed Plan-15 Z/P/PS frozen sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def average(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return mean(values) if values else None


def main() -> None:
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_root.glob("*/summary.json"))
    ]
    if not summaries:
        raise FileNotFoundError(f"no summary.json files below {args.input_root}")

    branches = {item["branch"] for item in summaries}
    if len(branches) != 1:
        raise ValueError(f"input contains multiple branches: {sorted(branches)}")

    grouped: dict[float, list[dict[str, object]]] = {}
    for summary in summaries:
        grouped.setdefault(float(summary["mass_factor"]), []).extend(
            summary["episodes"]
        )

    factors = {}
    for factor, episodes in sorted(grouped.items()):
        eligible = [row for row in episodes if row["eligible_post_jump_window"]]
        factors[str(factor)] = {
            "profiles": len(episodes),
            "eligible_profiles": len(eligible),
            "hold_success_count": sum(bool(row["hold_success"]) for row in eligible),
            "drop_count": sum(bool(row["drop"]) for row in eligible),
            "safe_lower_count": sum(bool(row["safe_lower"]) for row in eligible),
            "robot_fall_count": sum(bool(row["robot_fall"]) for row in episodes),
            "mean_maximum_height_loss_m": average(
                eligible, "maximum_height_loss_m"
            ),
            "mean_bilateral_patch_contact_fraction": average(
                eligible, "bilateral_patch_contact_fraction"
            ),
            "mean_gross_slip_patch_fraction": average(
                eligible, "gross_slip_patch_fraction"
            ),
        }

    result = {
        "schema": "plan15_frozen_sweep_summary_v1",
        "branch": next(iter(branches)),
        "source_runs": len(summaries),
        "profiles": sum(len(item["episodes"]) for item in summaries),
        "training_seeds": [151014, 151015, 151016],
        "evaluation_seeds": [152014, 152015, 152016],
        "checkpoint_evaluation_pairing": {
            "151014": 152014,
            "151015": 152015,
            "151016": 152016,
        },
        "factors": factors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
