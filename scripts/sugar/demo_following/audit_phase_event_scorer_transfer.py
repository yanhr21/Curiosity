#!/usr/bin/env python3
"""Audit phase and rollout-domain transfer on frozen phase-event policy traces.

This script never creates an environment, changes a policy, or runs an optimizer.
It feeds the exact archived 121-D prefixes from the matched frozen evaluator into
the released project predictor under two causal clocks: the deployed reset-zero
clock and a clock initialized from the recorded CarryBox reference frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "SUGAR/source/sugar_rl"))

from sugar_rl.utils.demo_event_reward_runtime import (  # noqa: E402
    FrozenDemoEventReward,
    FrozenDemoEventRewardCfg,
)


ARMS = ("correct", "unrelated")
DEMOS = ("correct", "unrelated")
PHASE_VARIANTS = ("reset_zero", "reference_aware")
UPDATES = (32, 64)
PROFILES_PER_UPDATE = 20
NUM_ENVS = len(UPDATES) * PROFILES_PER_UPDATE
POLICY_DIM = 121
PHASE_HORIZON_STEPS = 650
RUNTIME_REPRODUCTION_ATOL = 2.0e-5
TERM_SLICES = {
    "projected_gravity": (0, 3),
    "base_height": (3, 4),
    "base_linear_velocity_body": (4, 7),
    "base_angular_velocity_body": (7, 10),
    "joint_position_relative": (10, 39),
    "joint_velocity": (39, 68),
    "previous_applied_action_policy_units": (68, 97),
    "box_position_body": (97, 100),
    "box_orientation_tangent_normal_body": (100, 106),
    "box_linear_velocity_body": (106, 109),
    "box_angular_velocity_body": (109, 112),
    "goal_position_body": (112, 115),
    "goal_orientation_tangent_normal_body": (115, 121),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=ROOT
        / "experiments/demo_following/matched_phase_event_reward_v1/seed161587/"
        "scorer_transfer_source_trace_v1",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT
        / "experiments/demo_following/contact_event_reward_redesign_v1/"
        "phase_aware_dense_feedback_scale_audit_v1/RUNTIME_CONFIG.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "experiments/demo_following/matched_phase_event_reward_v1/seed161587/"
        "scorer_transfer_phase_ablation_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_runtime_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "sugar_dense_demo_event_feedback_runtime_v1":
        raise RuntimeError("unexpected phase-event runtime protocol")
    if payload.get("potential_difference_shaping_used") is not False:
        raise RuntimeError("transfer audit requires dense compatibility feedback")
    if payload.get("future_actual_events_enter_runtime") is not False:
        raise RuntimeError("future labels may not enter the scorer transfer audit")
    return payload


def scorer_cfg(payload: dict[str, Any], selected_option: str) -> FrozenDemoEventRewardCfg:
    selected = payload["selected_demo_options"][selected_option]
    return FrozenDemoEventRewardCfg(
        dataset_root=str(payload["dataset_root"]),
        predictor_dir=str(payload["predictor_dir"]),
        selected_task=str(selected["selected_task"]),
        selected_motion_id=int(selected["selected_motion_id"]),
        compatibility_baseline=float(payload["compatibility_baseline"]),
        eta=float(payload["eta"]),
        uncertainty_beta=float(payload["uncertainty_beta"]),
        reward_clip=float(payload["reward_clip"]),
        per_target_risk_clip=float(payload["per_target_risk_clip"]),
        target_weights=tuple(float(value) for value in payload["target_weights"]),
    )


def causal_phase_step(
    episode_steps: np.ndarray,
    done: np.ndarray,
    *,
    horizon_steps: int = PHASE_HORIZON_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one deployed causal clock step and return phase plus next state."""
    episode_steps = np.asarray(episode_steps, dtype=np.int64).copy() + 1
    done = np.asarray(done, dtype=bool)
    if episode_steps.shape != done.shape:
        raise ValueError("episode-step and done geometry differ")
    phase = np.clip(
        (episode_steps.astype(np.float32) + np.float32(1.0))
        / np.float32(horizon_steps),
        0.0,
        1.0,
    )
    phase[done] = np.float32(1.0 / horizon_steps)
    episode_steps[done] = 0
    return phase.astype(np.float32), episode_steps


def first_episode_transition_mask(done: np.ndarray) -> np.ndarray:
    done = np.asarray(done, dtype=bool)
    if done.ndim != 2:
        raise ValueError("done must be [transition, env]")
    mask = np.zeros_like(done)
    for env in range(done.shape[1]):
        hits = np.flatnonzero(done[:, env])
        stop = int(hits[0]) + 1 if hits.size else done.shape[0]
        mask[:stop, env] = True
    return mask


def load_trace(root: Path, arm: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result_path = root / arm / "RESULT.json"
    trace_path = root / arm / "TRACE.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("protocol") != "sugar_phase_event_reward_matched_frozen_eval_32_64_v2":
        raise RuntimeError(f"{arm}: evaluator trace does not contain the v2 transfer contract")
    if result.get("passed") is not True or not all(result.get("checks", {}).values()):
        raise RuntimeError(f"{arm}: source frozen evaluation did not pass")
    phase_initialization = result.get("phase_initialization")
    if phase_initialization is not None and phase_initialization.get("mode") != (
        "reset-zero-diagnostic"
    ):
        raise RuntimeError(
            f"{arm}: source trace must explicitly reproduce the historical zero-phase runtime"
        )
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {name: np.asarray(archive[name]) for name in archive.files}
    required_shapes = {
        "goal_policy_core_observation": (401, NUM_ENVS, POLICY_DIM),
        "done": (400, NUM_ENVS),
    }
    for name, shape in required_shapes.items():
        if trace.get(name, np.empty(0)).shape != shape:
            raise RuntimeError(
                f"{arm}: {name} shape drift: {trace.get(name, np.empty(0)).shape}"
            )
    for demo in DEMOS:
        for suffix in ("reward", "phase", "ready", "risk", "weighted_uncertainty"):
            name = f"demo_{demo}_{suffix}"
            if trace.get(name, np.empty(0)).shape != (400, NUM_ENVS):
                raise RuntimeError(f"{arm}: missing runtime signal {name}")
    if not np.isfinite(trace["goal_policy_core_observation"]).all():
        raise RuntimeError(f"{arm}: archived 121-D core contains non-finite values")
    return result, trace


@torch.no_grad()
def score_phase_variants(
    trace: dict[str, np.ndarray],
    *,
    reference_frame: int,
    payload: dict[str, Any],
    selected_demo: str,
    device: torch.device,
) -> dict[str, np.ndarray | dict[str, Any]]:
    core = np.asarray(trace["goal_policy_core_observation"], dtype=np.float32)
    done = np.asarray(trace["done"], dtype=bool)
    expanded_core = np.concatenate((core, core), axis=1)
    expanded_done = np.concatenate((done, done), axis=1)
    scorer = FrozenDemoEventReward(
        num_envs=2 * NUM_ENVS,
        device=device,
        cfg=scorer_cfg(payload, selected_demo),
    )
    scorer.begin(torch.from_numpy(expanded_core[0]).to(device))
    episode_steps = np.concatenate(
        (
            np.zeros(NUM_ENVS, dtype=np.int64),
            np.full(NUM_ENVS, reference_frame, dtype=np.int64),
        )
    )
    records = {name: [] for name in ("risk", "reward", "ready", "phase", "uncertainty")}
    failure = torch.zeros(2 * NUM_ENVS, dtype=torch.bool, device=device)
    for transition in range(done.shape[0]):
        phase, episode_steps = causal_phase_step(
            episode_steps,
            expanded_done[transition],
        )
        signal = scorer.process_step(
            torch.from_numpy(expanded_core[transition + 1]).to(device),
            torch.from_numpy(phase).to(device),
            torch.from_numpy(expanded_done[transition]).to(device),
            failure,
        )
        records["risk"].append(signal.next_risk.cpu().numpy())
        records["reward"].append(signal.reward.cpu().numpy())
        records["ready"].append(signal.next_ready.cpu().numpy())
        records["phase"].append(signal.selected_demo_phase.cpu().numpy())
        records["uncertainty"].append(
            signal.next_weighted_uncertainty.cpu().numpy()
        )
    output: dict[str, np.ndarray | dict[str, Any]] = {}
    for name, values in records.items():
        stacked = np.stack(values).reshape(400, len(PHASE_VARIANTS), NUM_ENVS)
        output[name] = np.transpose(stacked, (1, 0, 2))
    output["audit"] = scorer.audit()
    return output


def runtime_reproduction(
    trace: dict[str, np.ndarray],
    scores: dict[str, np.ndarray | dict[str, Any]],
    selected_demo: str,
) -> dict[str, Any]:
    comparisons = {
        "risk": np.asarray(scores["risk"])[0],
        "reward": np.asarray(scores["reward"])[0],
        "ready": np.asarray(scores["ready"])[0],
        "phase": np.asarray(scores["phase"])[0],
        "weighted_uncertainty": np.asarray(scores["uncertainty"])[0],
    }
    maximum_absolute_error = {}
    exact_equal = {}
    for name, rescored in comparisons.items():
        source = np.asarray(trace[f"demo_{selected_demo}_{name}"])
        if source.dtype == np.bool_:
            exact_equal[name] = bool(np.array_equal(source, rescored))
            maximum_absolute_error[name] = 0.0 if exact_equal[name] else 1.0
        else:
            maximum_absolute_error[name] = float(
                np.max(np.abs(source.astype(np.float64) - rescored.astype(np.float64)))
            )
            exact_equal[name] = bool(np.array_equal(source, rescored))
    return {
        "maximum_absolute_error": maximum_absolute_error,
        "exact_equal": exact_equal,
        "passed": (
            exact_equal["ready"]
            and maximum_absolute_error["phase"]
            <= float(np.finfo(np.float32).eps)
            and all(
                maximum_absolute_error[name] <= RUNTIME_REPRODUCTION_ATOL
                for name in ("risk", "reward", "weighted_uncertainty")
            )
        ),
    }


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "count": int(absolute.size),
        "mean_absolute_z": float(np.mean(absolute)),
        "p95_absolute_z": float(np.percentile(absolute, 95)),
        "p99_absolute_z": float(np.percentile(absolute, 99)),
        "maximum_absolute_z": float(np.max(absolute)),
        "fraction_absolute_z_gt_3": float(np.mean(absolute > 3.0)),
        "fraction_absolute_z_gt_5": float(np.mean(absolute > 5.0)),
        "fraction_absolute_z_gt_10": float(np.mean(absolute > 10.0)),
    }


def block_semantics(
    score_by_demo: dict[str, dict[str, np.ndarray | dict[str, Any]]],
    trace: dict[str, np.ndarray],
) -> dict[str, Any]:
    first_episode = first_episode_transition_mask(trace["done"])
    output: dict[str, Any] = {}
    for phase_index, phase_name in enumerate(PHASE_VARIANTS):
        phase_output: dict[str, Any] = {}
        carry_risk = np.asarray(score_by_demo["correct"]["risk"])[phase_index]
        kick_risk = np.asarray(score_by_demo["unrelated"]["risk"])[phase_index]
        ready = np.asarray(score_by_demo["correct"]["ready"])[phase_index].astype(bool)
        if not np.array_equal(
            ready,
            np.asarray(score_by_demo["unrelated"]["ready"])[phase_index],
        ):
            raise RuntimeError("selected demo changed prefix readiness")
        margin = kick_risk - carry_risk
        for update_index, update in enumerate(UPDATES):
            start = update_index * PROFILES_PER_UPDATE
            stop = start + PROFILES_PER_UPDATE
            valid = first_episode[:, start:stop] & ready[:, start:stop]
            profile_means = []
            for profile in range(PROFILES_PER_UPDATE):
                profile_mask = valid[:, profile]
                if not profile_mask.any():
                    raise RuntimeError("profile contains no ready first-episode scores")
                profile_means.append(
                    float(np.mean(margin[:, start + profile][profile_mask]))
                )
            block_values = margin[:, start:stop][valid]
            phase_output[f"update_{update:04d}"] = {
                "ready_frame_count": int(block_values.size),
                "mean_kick_minus_carry_risk": float(np.mean(block_values)),
                "median_kick_minus_carry_risk": float(np.median(block_values)),
                "carry_preferred_frame_fraction": float(np.mean(block_values > 0.0)),
                "carry_preferred_profile_count": int(
                    np.count_nonzero(np.asarray(profile_means) > 0.0)
                ),
                "profile_count": PROFILES_PER_UPDATE,
                "profile_mean_margins": profile_means,
            }
        output[phase_name] = phase_output
    return output


def main() -> None:
    args = parse_args()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    runtime_config = args.runtime_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    payload = load_runtime_payload(runtime_config)
    source_results: dict[str, dict[str, Any]] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    reference_frames = {}
    for arm in ARMS:
        source_results[arm], traces[arm] = load_trace(evaluation_root, arm)
        reference_frames[arm] = int(source_results[arm]["reset"]["reference_frame"])
    if len(set(reference_frames.values())) != 1:
        raise RuntimeError("matched arms use different source reference frames")

    score_records: dict[str, dict[str, dict[str, np.ndarray | dict[str, Any]]]] = {}
    reproduction: dict[str, dict[str, Any]] = {}
    semantic_blocks: dict[str, Any] = {}
    archive_arrays: dict[str, np.ndarray] = {}
    for arm in ARMS:
        score_records[arm] = {}
        reproduction[arm] = {}
        for demo in DEMOS:
            scores = score_phase_variants(
                traces[arm],
                reference_frame=reference_frames[arm],
                payload=payload,
                selected_demo=demo,
                device=device,
            )
            score_records[arm][demo] = scores
            reproduction[arm][demo] = runtime_reproduction(
                traces[arm], scores, demo
            )
            for signal in ("risk", "reward", "ready", "phase", "uncertainty"):
                archive_arrays[f"{arm}_{demo}_{signal}"] = np.asarray(scores[signal])
        semantic_blocks[arm] = block_semantics(score_records[arm], traces[arm])

    dataset_root = Path(payload["dataset_root"])
    with np.load(dataset_root / "NORMALIZATION.npz", allow_pickle=False) as normalization:
        state_mean = np.asarray(normalization["state_mean"], dtype=np.float32)
        state_std = np.asarray(normalization["state_std"], dtype=np.float32)
    tracker_test_history = np.load(
        dataset_root / "test/policy_prefix.npy", mmap_mode="r", allow_pickle=False
    )
    tracker_z = (np.asarray(tracker_test_history, dtype=np.float32) - state_mean) / state_std
    distribution = {"official_tracker_test": distribution_summary(tracker_z)}
    for arm in ARMS:
        ready = np.asarray(score_records[arm]["correct"]["ready"])[0].astype(bool)
        first = first_episode_transition_mask(traces[arm]["done"])
        core_tp1 = traces[arm]["goal_policy_core_observation"][1:]
        values = core_tp1[first & ready]
        z = (values - state_mean) / state_std
        distribution[arm] = {
            "all_terms": distribution_summary(z),
            "by_term": {
                name: distribution_summary(z[:, start:stop])
                for name, (start, stop) in TERM_SLICES.items()
            },
        }

    phase_gate_blocks = []
    for arm in ARMS:
        for update in UPDATES:
            block = semantic_blocks[arm]["reference_aware"][f"update_{update:04d}"]
            phase_gate_blocks.append(
                {
                    "arm": arm,
                    "update": update,
                    "mean_margin_positive": block["mean_kick_minus_carry_risk"] > 0.0,
                    "frame_majority_prefers_carry": (
                        block["carry_preferred_frame_fraction"] > 0.5
                    ),
                    "profile_majority_prefers_carry": (
                        block["carry_preferred_profile_count"]
                        > block["profile_count"] / 2
                    ),
                }
            )
    phase_only_necessary_gate = all(
        all(
            record[name]
            for name in (
                "mean_margin_positive",
                "frame_majority_prefers_carry",
                "profile_majority_prefers_carry",
            )
        )
        for record in phase_gate_blocks
    )
    execution_checks = {
        "both_v2_source_evaluations_pass": all(
            result["passed"] and all(result["checks"].values())
            for result in source_results.values()
        ),
        "source_reference_frame_is_matched": len(set(reference_frames.values())) == 1,
        "source_reference_frame_is_nonzero": next(iter(reference_frames.values())) > 0,
        "all_runtime_current_clock_scores_reproduced": all(
            reproduction[arm][demo]["passed"] for arm in ARMS for demo in DEMOS
        ),
        "all_predictors_frozen": all(
            score_records[arm][demo]["audit"]["model_training"] is False
            and int(score_records[arm][demo]["audit"]["trainable_parameters"]) == 0
            for arm in ARMS
            for demo in DEMOS
        ),
        "all_score_arrays_finite": all(
            np.isfinite(value).all()
            for value in archive_arrays.values()
            if np.issubdtype(value.dtype, np.number)
        ),
        "no_policy_training_or_environment_execution": True,
    }
    result = {
        "protocol": "sugar_phase_event_scorer_transfer_audit_v1",
        "passed": all(execution_checks.values()),
        "checks": execution_checks,
        "claim_scope": (
            "Scorer-only ablation on exact frozen-policy 121-D prefixes. It tests whether "
            "initializing the causal clock from the recorded reference frame is sufficient "
            "to recover Carry45 preference on Carry rollouts. It does not test Kick-policy "
            "rollouts or establish policy semantic following."
        ),
        "evaluation_root": str(evaluation_root),
        "runtime_config": str(runtime_config),
        "device": str(device),
        "phase_horizon_steps": PHASE_HORIZON_STEPS,
        "phase_variants": {
            "reset_zero": "episode_steps starts at 0, exactly reproducing deployed runtime",
            "reference_aware": (
                "first episode starts at the source reference frame; later resets restart at 0"
            ),
        },
        "reference_frames": reference_frames,
        "runtime_reproduction": reproduction,
        "semantic_blocks": semantic_blocks,
        "phase_gate_blocks": phase_gate_blocks,
        "phase_only_necessary_carry_gate_passed": phase_only_necessary_gate,
        "scientific_verdict": (
            "Reference-aware phase passes the necessary Carry-domain gate; Kick-domain and "
            "independent rollout transfer remain required."
            if phase_only_necessary_gate
            else "Reference-aware phase is insufficient; rollout-domain transfer must be addressed."
        ),
        "normalized_state_distribution": distribution,
        "artifacts": {"scores": "SCORES.npz"},
    }
    if not result["passed"]:
        raise RuntimeError(
            "scorer transfer audit execution checks failed: "
            f"{[name for name, value in execution_checks.items() if not value]}; "
            f"runtime_reproduction={json.dumps(reproduction, sort_keys=True)}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        np.savez_compressed(staging / "SCORES.npz", **archive_arrays)
        (staging / "RESULT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
