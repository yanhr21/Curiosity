#!/usr/bin/env python3
"""Freeze the dense event-feedback baseline, scale and matched stopping rule."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "SUGAR/source/sugar_rl"))
sys.path.insert(0, str(ROOT / "scripts/sugar/demo_reward"))

from audit_actual_contact_event_corpus import motion_split  # noqa: E402
from sugar_rl.utils.demo_event_reward_potential import (  # noqa: E402
    DEFAULT_EVENT_WEIGHTS,
    calibrated_event_risk,
    compatibility_potential,
)
from train_actual_contact_event_predictor import (  # noqa: E402
    forward_model,
    model_from_normalization,
)


HISTORY_STEPS = 10
DEFAULT_DEMO_SPECS = {"carry45": (0, 45), "kick": (1, 21)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-evaluation-root",
        type=Path,
        default=ROOT
        / "experiments/demo_following/matched_reward_identity_same_teacher_v1",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--uncertainty-beta", type=float, default=1.0)
    parser.add_argument("--target-mean-absolute-ratio", type=float, default=0.25)
    parser.add_argument("--unrelated-motion-id", type=int, default=21)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_demo_conditions(
    dataset_root: Path,
    device: torch.device,
    demo_specs: dict[str, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    split = dataset_root / "train"
    with np.load(split / "routing.npz", allow_pickle=False) as routing:
        task = np.asarray(routing["demo_task"], dtype=np.int64)
        motion = np.asarray(routing["demo_source_motion_id"], dtype=np.int64)
    bank = np.load(split / "demo_bank.npy", mmap_mode="r", allow_pickle=False)
    result = {}
    for name, (task_index, motion_id) in demo_specs.items():
        rows = np.flatnonzero((task == task_index) & (motion == motion_id))
        if rows.size != 1:
            raise RuntimeError(f"{name} does not map to exactly one train demo")
        result[name] = torch.from_numpy(
            np.array(bank[int(rows[0])], dtype=np.float32, copy=True)
        ).to(device)
    return result


@torch.no_grad()
def score_prefixes(
    model,
    policy_core: np.ndarray,
    demo: torch.Tensor,
    normalized_phase: np.ndarray,
    multiplier: torch.Tensor,
    uncertainty_beta: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    frames, envs, policy_dim = policy_core.shape
    prefixes = np.stack(
        [policy_core[frame - HISTORY_STEPS + 1 : frame + 1] for frame in range(HISTORY_STEPS - 1, frames)],
        axis=0,
    ).transpose(0, 2, 1, 3)
    flat = prefixes.reshape(-1, HISTORY_STEPS, policy_dim)
    phase_flat = normalized_phase.reshape(-1)
    if phase_flat.shape != (len(flat),):
        raise RuntimeError("normalized phase geometry drift")
    potentials, uncertainties = [], []
    for begin in range(0, len(flat), batch_size):
        policy = torch.from_numpy(flat[begin : begin + batch_size]).to(device)
        phase = torch.from_numpy(
            phase_flat[begin : begin + batch_size].astype(np.float32, copy=False)
        ).to(device)
        selected = demo.unsqueeze(0).expand(len(policy), -1, -1, -1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = forward_model(
                model,
                policy_prefix=policy,
                selected_demo_condition=selected,
                selected_demo_phase=phase,
            )
        record = calibrated_event_risk(
            output["mean_log1p_scaled"].float(),
            output["log_variance_log1p_scaled"].float(),
            multiplier,
            uncertainty_beta=uncertainty_beta,
            target_weights=DEFAULT_EVENT_WEIGHTS,
        )
        potentials.append(compatibility_potential(record["risk"]).cpu().numpy())
        uncertainties.append(record["weighted_uncertainty"].cpu().numpy())
    shape = (frames - HISTORY_STEPS + 1, envs)
    return np.concatenate(potentials).reshape(shape), np.concatenate(uncertainties).reshape(shape)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def base_reward_reference(root: Path) -> np.ndarray:
    values = []
    for path in sorted(root.glob("seed*/evaluation_update0064/*/TRACE.npz")):
        with np.load(path, allow_pickle=False) as trace:
            values.append(
                np.asarray(trace["weighted_task_outcome_reward"], dtype=np.float32)
                + np.asarray(trace["external_constraint_reward"], dtype=np.float32)
            )
    if not values:
        raise FileNotFoundError("no frozen base-reward traces found")
    return np.concatenate([value.reshape(-1) for value in values])


def main() -> None:
    args = parse_args()
    if socket.gethostname().startswith(("mgmtserver", "login")):
        raise RuntimeError("reward audit must run inside a compute allocation")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("retained CUDA Slurm allocation required")
    corpus_root = args.corpus_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    predictor_dir = args.predictor_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = json.loads((dataset_root / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["model_inputs"].get("policy_observation_key") != "goal_policy_core_observation":
        raise RuntimeError("reward audit requires the deployable goal-policy core dataset")
    if manifest["model_inputs"]["policy_prefix"] != [10, 121]:
        raise RuntimeError("deployable policy-prefix geometry drift")
    if manifest.get("alignment_mode") != "clock_phase":
        raise RuntimeError("reward audit rejects free-window target alignment")
    training = json.loads((predictor_dir / "RESULT.json").read_text(encoding="utf-8"))
    calibration = json.loads(
        (predictor_dir / "CALIBRATION_RESULT.json").read_text(encoding="utf-8")
    )
    if training.get("passed") is not True or calibration.get("passed") is not True:
        raise RuntimeError("predictor training/calibration gates did not pass")
    device = torch.device(args.device)
    model = model_from_normalization(dataset_root / "NORMALIZATION.npz", device)
    checkpoint = torch.load(predictor_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    with np.load(predictor_dir / "UNCERTAINTY_CALIBRATION.npz", allow_pickle=False) as archive:
        multiplier = torch.from_numpy(
            np.asarray(archive["variance_multiplier"], dtype=np.float32)
        ).to(device)
    demo_specs = dict(DEFAULT_DEMO_SPECS)
    demo_specs["kick"] = (1, int(args.unrelated_motion_id))
    demos = load_demo_conditions(dataset_root, device, demo_specs)

    grouped: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        split: {
            task: {name: [] for name in demo_specs}
            | {f"{name}_uncertainty": [] for name in demo_specs}
            for task in ("CarryBox", "KickBox")
        }
        for split in ("train", "validation", "test")
    }
    for trace_path in sorted(corpus_root.glob("*/TRACE.npz")):
        result = json.loads((trace_path.parent / "RESULT.json").read_text(encoding="utf-8"))
        task_name = str(result["task_family"])
        with np.load(trace_path, allow_pickle=False) as trace:
            policy = np.asarray(trace["goal_policy_core_observation"], dtype=np.float32)
            source = np.asarray(trace["source_motion_id"], dtype=np.int64)
            local_motion = np.asarray(trace["local_motion_id"], dtype=np.int64)
            reference_steps = np.asarray(
                trace["source_reference_steps_by_local_motion"], dtype=np.int64
            )
            if np.count_nonzero(trace["reset_before_frame"]):
                raise RuntimeError(f"{trace_path}: reset contaminates causal prefixes")
        anchors = np.arange(HISTORY_STEPS - 1, policy.shape[0], dtype=np.float32)[:, None]
        per_env_reference_steps = reference_steps[local_motion[0]][None, :]
        normalized_phase = np.clip(
            (anchors + 1.0) / np.maximum(per_env_reference_steps - 10, 1),
            0.0,
            1.0,
        ).astype(np.float32)
        scored = {
            name: score_prefixes(
                model,
                policy,
                demo,
                normalized_phase,
                multiplier,
                args.uncertainty_beta,
                args.batch_size,
                device,
            )
            for name, demo in demos.items()
        }
        for env in range(policy.shape[1]):
            ids = np.unique(source[:, env])
            if ids.size != 1:
                raise RuntimeError("source motion changed within one environment")
            split_name = motion_split(int(ids[0]))
            last_anchor = min(policy.shape[0] - 11, int(per_env_reference_steps[0, env]) - 11)
            valid_rows = max(0, last_anchor - (HISTORY_STEPS - 1) + 1)
            if valid_rows == 0:
                raise RuntimeError("source motion has no valid causal reward rows")
            for name, (potential, uncertainty) in scored.items():
                grouped[split_name][task_name][name].append(potential[:valid_rows, env])
                grouped[split_name][task_name][f"{name}_uncertainty"].append(
                    uncertainty[:valid_rows, env]
                )
        print(f"REWARD_AUDIT_SCORED {trace_path.parent.name}", flush=True)

    arrays = {
        split: {
            task: {
                name: np.concatenate(chunks)
                for name, chunks in records.items()
            }
            for task, records in tasks.items()
        }
        for split, tasks in grouped.items()
    }
    baseline_source = np.concatenate(
        (
            arrays["train"]["CarryBox"]["carry45"],
            arrays["train"]["CarryBox"]["kick"],
        )
    )
    compatibility_baseline = float(np.median(baseline_source))
    base_reward = base_reward_reference(args.base_evaluation_root.expanduser().resolve())
    base_mean_abs = float(np.mean(np.abs(base_reward)))
    base_p95_abs = float(np.quantile(np.abs(base_reward), 0.95))
    train_unit = np.concatenate(
        (
            arrays["train"]["CarryBox"]["carry45"] - compatibility_baseline,
            arrays["train"]["CarryBox"]["kick"] - compatibility_baseline,
        )
    )
    unit_mean_abs = float(np.mean(np.abs(train_unit)))
    eta_unclipped = (
        args.target_mean_absolute_ratio * base_mean_abs / max(unit_mean_abs, 1.0e-8)
    )
    eta = float(np.clip(eta_unclipped, 0.1, 20.0))
    reward_clip = float(max(0.02, 0.5 * base_p95_abs))

    statistics = {}
    reward_statistics = {}
    for split, tasks in arrays.items():
        statistics[split] = {}
        reward_statistics[split] = {}
        for task, records in tasks.items():
            statistics[split][task] = {
                name: summarize(value) for name, value in records.items()
            }
            reward_statistics[split][task] = {
                name: summarize(
                    np.clip(
                        eta * (records[name] - compatibility_baseline),
                        -reward_clip,
                        reward_clip,
                    )
                )
                for name in demo_specs
            }

    scaled_train = np.clip(eta * train_unit, -reward_clip, reward_clip)
    achieved_ratio = float(np.mean(np.abs(scaled_train)) / base_mean_abs)
    checks = {
        "predictor_and_calibration_passed": True,
        "deployable_input_is_ten_by_121": True,
        "baseline_and_scale_fit_use_train_carry_only": True,
        "validation_carry_prefers_carry45": (
            statistics["validation"]["CarryBox"]["carry45"]["mean"]
            > statistics["validation"]["CarryBox"]["kick"]["mean"]
        ),
        "test_carry_prefers_carry45": (
            statistics["test"]["CarryBox"]["carry45"]["mean"]
            > statistics["test"]["CarryBox"]["kick"]["mean"]
        ),
        "validation_kick_prefers_selected_kick": (
            statistics["validation"]["KickBox"]["kick"]["mean"]
            > statistics["validation"]["KickBox"]["carry45"]["mean"]
        ),
        "test_kick_prefers_selected_kick": (
            statistics["test"]["KickBox"]["kick"]["mean"]
            > statistics["test"]["KickBox"]["carry45"]["mean"]
        ),
        "heldout_carry_feedback_has_opposite_sign": (
            reward_statistics["test"]["CarryBox"]["carry45"]["mean"] > 0
            and reward_statistics["test"]["CarryBox"]["kick"]["mean"] < 0
        ),
        "scaled_mean_absolute_feedback_is_ten_to_thirtyfive_percent_of_base": (
            0.10 <= achieved_ratio <= 0.35
        ),
        "all_values_finite": all(
            np.isfinite(value).all()
            for tasks in arrays.values()
            for records in tasks.values()
            for value in records.values()
        ),
    }
    runtime_config = {
        "protocol": "sugar_dense_demo_event_feedback_runtime_v1",
        "dataset_root": str(dataset_root),
        "predictor_dir": str(predictor_dir),
        "selected_demo_options": {
            "correct": {"selected_task": "CarryBox", "selected_motion_id": 45},
            "unrelated": {
                "selected_task": "KickBox",
                "selected_motion_id": int(args.unrelated_motion_id),
            },
        },
        "compatibility_baseline": compatibility_baseline,
        "eta": eta,
        "uncertainty_beta": args.uncertainty_beta,
        "reward_clip": reward_clip,
        "per_target_risk_clip": 5.0,
        "target_weights": list(DEFAULT_EVENT_WEIGHTS),
        "feedback_definition": "eta * (exp(-calibrated_event_risk) - fixed_train_baseline)",
        "future_actual_events_enter_runtime": False,
        "potential_difference_shaping_used": False,
    }
    stopping_rule = {
        "first_endpoint_updates": 64,
        "inspection_checkpoints": [32, 64],
        "do_not_extend_if": [
            "either arm loses bilateral contact and 5 cm lift on the frozen gate",
            "correct-demo frozen success falls by more than 20 percentage points from teacher-only",
            "update-64 predictor-independent behavior shows no Carry-versus-Kick semantic separation",
        ],
        "extend_to_three_seeds_only_if": (
            "both arms retain physical validity and the unrelated arm changes at least two "
            "predeclared contact/motion-regime directions without using predictor scores"
        ),
    }
    result = {
        "protocol": "sugar_deployable_demo_event_feedback_scale_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "runtime_config": runtime_config,
        "stopping_rule": stopping_rule,
        "base_reward_reference": {
            "definition": "frozen weighted task outcome plus external constraint reward",
            "mean_absolute": base_mean_abs,
            "p95_absolute": base_p95_abs,
        },
        "scale_fit": {
            "target_mean_absolute_ratio": args.target_mean_absolute_ratio,
            "unit_feedback_mean_absolute": unit_mean_abs,
            "eta_unclipped": eta_unclipped,
            "eta": eta,
            "reward_clip": reward_clip,
            "achieved_mean_absolute_ratio": achieved_ratio,
        },
        "potential_statistics": statistics,
        "reward_statistics": reward_statistics,
        "claim_boundary": (
            "Passing establishes a causal deployable reward scale and held-out semantic "
            "direction before policy optimization. It does not establish policy-level demo following."
        ),
    }
    (output / "RUNTIME_CONFIG.json").write_text(
        json.dumps(runtime_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": result["passed"], "checks": checks, "scale_fit": result["scale_fit"]}, indent=2))


if __name__ == "__main__":
    main()
