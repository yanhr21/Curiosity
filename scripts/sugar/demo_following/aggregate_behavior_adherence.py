#!/usr/bin/env python3
"""Aggregate the three predeclared independent training seeds.

Inputs are the predictor-independent per-seed behavior audits.  Physics
profiles are summarized within each seed first; the training seed is the unit
of replication.  No predictor or demo-reward value is consumed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / (
    "experiments/demo_following/matched_reward_identity_same_teacher_v1"
)
SEEDS = (161581, 161583, 161585)
METRICS = {
    "lifted_fraction": {
        "label": "Lifted-frame fraction",
        "expected_delta": "negative",
    },
    "lifted_transport_fraction": {
        "label": "Lifted transport fraction",
        "expected_delta": "negative",
    },
    "ground_transport_fraction": {
        "label": "Ground transport fraction",
        "expected_delta": "positive",
    },
    "root_orbit_rate_rad_s": {
        "label": "Root orbit rate (rad/s)",
        "expected_delta": "positive",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "multiseed_behavior_adherence_v1",
    )
    return parser.parse_args()


def load_seed(run_root: Path, seed: int) -> dict[str, object]:
    path = run_root / f"seed{seed}/behavior_adherence_audit_v1/RESULT.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing completed seed audit: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "same_teacher_predictor_independent_behavior_audit_v1":
        raise RuntimeError(f"unexpected behavior protocol in {path}")
    evidence = payload.get("evidence_contract", {})
    if evidence.get("uses_predictor_output") is not False or evidence.get("uses_demo_reward") is not False:
        raise RuntimeError(f"non-independent behavior evidence in {path}")
    return payload


def expected_direction(delta: float, expected: str) -> bool:
    if expected == "negative":
        return delta < 0.0
    if expected == "positive":
        return delta > 0.0
    raise ValueError(expected)


def seed_outcomes(run_root: Path, seed: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for arm in ("correct", "unrelated"):
        path = run_root / f"seed{seed}/evaluation_update0064/{arm}/RESULT.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries = payload["summaries"]
        result[arm] = {
            "success_profiles": int(
                sum(record["termination_counts"]["success"] > 0 for record in summaries)
            ),
            "physical_fall_profiles": int(
                sum(bool(record["physical_robot_fall"]) for record in summaries)
            ),
            "profile_count": len(summaries),
        }
    return result


def write_figure(path: Path, deltas: dict[str, list[float]]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
    for axis, (metric, contract) in zip(axes, METRICS.items()):
        values = np.asarray(deltas[metric])
        axis.axhline(0.0, color="black", lw=1.0)
        if contract["expected_delta"] == "negative":
            axis.axhspan(min(float(values.min()), 0.0) - 0.01, 0.0, color="#dff0d8")
        else:
            axis.axhspan(0.0, max(float(values.max()), 0.0) + 0.01, color="#dff0d8")
        axis.plot(range(len(SEEDS)), values, color="0.55", lw=1.0)
        axis.scatter(range(len(SEEDS)), values, color="#1f77b4", s=36, zorder=2)
        axis.set_xticks(range(len(SEEDS)), [str(seed) for seed in SEEDS], rotation=35)
        axis.set_title(str(contract["label"]), fontsize=9)
        axis.grid(axis="y", color="0.9")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Unrelated minus correct reward arm by independent training seed\n"
        "Green is the predeclared Kick-like direction",
        fontsize=12,
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    seed_payloads = {seed: load_seed(run_root, seed) for seed in SEEDS}
    deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
    per_seed: dict[str, object] = {}
    for seed, payload in seed_payloads.items():
        comparison = payload["paired_profile_comparison"]
        metric_records: dict[str, object] = {}
        for metric, contract in METRICS.items():
            delta = float(comparison[metric]["unrelated_minus_correct_mean"])
            deltas[metric].append(delta)
            metric_records[metric] = {
                "unrelated_minus_correct_mean": delta,
                "expected_direction_observed": expected_direction(
                    delta, str(contract["expected_delta"])
                ),
            }
        per_seed[str(seed)] = {
            "behavior": metric_records,
            "task_outcomes": seed_outcomes(run_root, seed),
        }

    consistency = {
        metric: {
            "expected_delta": contract["expected_delta"],
            "seed_deltas": deltas[metric],
            "seeds_in_expected_direction": int(
                sum(
                    expected_direction(delta, str(contract["expected_delta"]))
                    for delta in deltas[metric]
                )
            ),
            "all_three_seeds": bool(
                all(
                    expected_direction(delta, str(contract["expected_delta"]))
                    for delta in deltas[metric]
                )
            ),
        }
        for metric, contract in METRICS.items()
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_figure(output_dir / "multiseed_behavior_deltas.png", deltas)
    result = {
        "protocol": "same_teacher_predictor_independent_multiseed_audit_v1",
        "training_seeds": list(SEEDS),
        "replication_unit": "independent policy training seed",
        "profile_role": "within-seed matched physics variation",
        "uses_predictor_output": False,
        "uses_demo_reward": False,
        "per_seed": per_seed,
        "direction_consistency": consistency,
        "stable_semantic_following": bool(
            all(record["all_three_seeds"] for record in consistency.values())
        ),
        "claim_rule": (
            "All three independent training seeds must show the predeclared "
            "direction on every reported semantic endpoint. Lifted and ground "
            "transport are complementary and are not counted as independent tests."
        ),
        "artifacts": {"figure": "multiseed_behavior_deltas.png"},
    }
    (output_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "stable_semantic_following": result["stable_semantic_following"],
        "direction_consistency": consistency,
    }, indent=2))


if __name__ == "__main__":
    main()
