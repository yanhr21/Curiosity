#!/usr/bin/env python3
"""Assess corrected online phase semantics from matched frozen Carry traces.

The evaluator has already run both frozen selected-demo scorers on each exact
policy trajectory. This assessor reads those online signals directly; it does
not re-run a model, simulator, policy, or optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


ARMS = ("correct", "unrelated")
DEMOS = ("correct", "unrelated")
UPDATES = (32, 64)
PROFILES_PER_UPDATE = 20
NUM_ENVS = len(UPDATES) * PROFILES_PER_UPDATE
STEPS = 400
REFERENCE_FRAME = 197
PHASE_HORIZON_STEPS = 650


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def first_episode_mask(done: np.ndarray) -> np.ndarray:
    done = np.asarray(done, dtype=bool)
    if done.shape != (STEPS, NUM_ENVS):
        raise ValueError(f"done geometry drift: {done.shape}")
    mask = np.zeros_like(done)
    for env_index in range(NUM_ENVS):
        hits = np.flatnonzero(done[:, env_index])
        stop = int(hits[0]) + 1 if hits.size else STEPS
        mask[:stop, env_index] = True
    return mask


def load_arm(root: Path, arm: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result = json.loads((root / arm / "RESULT.json").read_text(encoding="utf-8"))
    if result.get("protocol") != "sugar_phase_event_reward_matched_frozen_eval_32_64_v2":
        raise RuntimeError(f"{arm}: unexpected corrected evaluator protocol")
    if result.get("passed") is not True or not all(result.get("checks", {}).values()):
        raise RuntimeError(f"{arm}: corrected frozen evaluation did not pass")
    with np.load(root / arm / "TRACE.npz", allow_pickle=False) as archive:
        trace = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"done": (STEPS, NUM_ENVS)}
    for demo in DEMOS:
        required[f"demo_{demo}_phase"] = (STEPS, NUM_ENVS)
        required[f"demo_{demo}_ready"] = (STEPS, NUM_ENVS)
        required[f"demo_{demo}_risk"] = (STEPS, NUM_ENVS)
    for name, shape in required.items():
        if trace.get(name, np.empty(0)).shape != shape:
            raise RuntimeError(f"{arm}: {name} geometry drift")
    return result, trace


def semantic_blocks(trace: dict[str, np.ndarray]) -> dict[str, Any]:
    carry_phase = trace["demo_correct_phase"]
    kick_phase = trace["demo_unrelated_phase"]
    carry_ready = trace["demo_correct_ready"].astype(bool)
    kick_ready = trace["demo_unrelated_ready"].astype(bool)
    if not np.array_equal(carry_phase, kick_phase):
        raise RuntimeError("selected demo changed the causal phase clock")
    if not np.array_equal(carry_ready, kick_ready):
        raise RuntimeError("selected demo changed causal prefix readiness")
    expected_first_phase = np.float32((REFERENCE_FRAME + 2) / PHASE_HORIZON_STEPS)
    if not np.allclose(
        carry_phase[0], expected_first_phase, rtol=0.0, atol=np.finfo(np.float32).eps
    ):
        raise RuntimeError("corrected evaluator did not start from reference frame 197")

    first = first_episode_mask(trace["done"])
    margin = trace["demo_unrelated_risk"] - trace["demo_correct_risk"]
    output = {}
    for update_index, update in enumerate(UPDATES):
        start = update_index * PROFILES_PER_UPDATE
        stop = start + PROFILES_PER_UPDATE
        valid = first[:, start:stop] & carry_ready[:, start:stop]
        profile_means = []
        for profile in range(PROFILES_PER_UPDATE):
            selected = valid[:, profile]
            if not selected.any():
                raise RuntimeError("profile contains no ready first-episode scores")
            profile_means.append(
                float(np.mean(margin[:, start + profile][selected]))
            )
        values = margin[:, start:stop][valid]
        output[f"update_{update:04d}"] = {
            "ready_frame_count": int(values.size),
            "mean_kick_minus_carry_risk": float(np.mean(values)),
            "carry_preferred_frame_fraction": float(np.mean(values > 0.0)),
            "carry_preferred_profile_count": int(
                np.count_nonzero(np.asarray(profile_means) > 0.0)
            ),
            "profile_count": PROFILES_PER_UPDATE,
            "profile_mean_margins": profile_means,
        }
    return output


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    results = {}
    traces = {}
    blocks = {}
    for arm in ARMS:
        results[arm], traces[arm] = load_arm(root, arm)
        blocks[arm] = semantic_blocks(traces[arm])

    checks = {
        "both_frozen_evaluations_pass": all(
            result["passed"] and all(result["checks"].values())
            for result in results.values()
        ),
        "phase_initialization_metadata_exact": all(
            result.get("phase_initialization")
            == {
                "mode": "reference-aware",
                "reference_frame": REFERENCE_FRAME,
                "initial_episode_steps": REFERENCE_FRAME,
            }
            for result in results.values()
        ),
        "both_predictors_frozen_and_reference_aware": all(
            audit["model_frozen"] is True
            and audit["trainable_parameters"] == 0
            and audit["phase_source"]
            == "reset_reference_frame_plus_causal_control_clock"
            and audit["initial_episode_steps_supplied"] is True
            and audit["initial_episode_steps_min"] == REFERENCE_FRAME
            and audit["initial_episode_steps_max"] == REFERENCE_FRAME
            for result in results.values()
            for audit in result["demo_predictor_audits"].values()
        ),
        "actual_rollouts_are_physical_carry": all(
            result["final_update_aggregate"]["maximum_lift_height_m"] >= 0.05
            and result["final_update_aggregate"]["bilateral_rigid_contact_frames"] > 0
            and result["final_update_aggregate"]["physical_robot_fall"] == 0
            for result in results.values()
        ),
        "all_four_blocks_prefer_carry": all(
            block["mean_kick_minus_carry_risk"] > 0.0
            and block["carry_preferred_frame_fraction"] > 0.5
            and block["carry_preferred_profile_count"] == PROFILES_PER_UPDATE
            for arm_blocks in blocks.values()
            for block in arm_blocks.values()
        ),
        "no_model_policy_or_optimizer_execution": True,
    }
    payload = {
        "protocol": "sugar_corrected_phase_frozen_carry_gate_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "claim_scope": (
            "Direct assessment of online frozen scorer signals on two Carry-policy arms "
            "at updates 32 and 64. Passing establishes the corrected Carry-domain "
            "necessary gate, not Kick-domain transfer or policy semantic following."
        ),
        "evaluation_root": str(root),
        "reference_frame": REFERENCE_FRAME,
        "phase_horizon_steps": PHASE_HORIZON_STEPS,
        "semantic_blocks": blocks,
        "final_update_behavior": {
            arm: results[arm]["final_update_aggregate"] for arm in ARMS
        },
    }
    if not payload["passed"]:
        raise RuntimeError(
            "corrected frozen Carry gate failed: "
            f"{[name for name, value in checks.items() if not value]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (staging / "RESULT.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
