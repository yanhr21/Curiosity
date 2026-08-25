#!/usr/bin/env python3
"""Score feature-complete prefix-recovery rollouts with official TinyMDM priors.

This is a frozen-policy audit.  It converts actual IsaacLab state histories with
MimicKit's official ``compute_disc_obs`` adapter, then evaluates the already
trained official Carry45 and Kick21 TinyMDM ESM/SDS energies.  Outcome labels
are used only after scoring to summarize safe/fall groups; they never enter the
features or the prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_selected_demo_tinymdm import (  # noqa: E402
    OUTPUT_ROOT as PRIOR_ROOT,
    build_feature_windows,
    load_prior,
    raw_energy,
    require_compute_gpu,
)
from sugar_g1_box_schema import (  # noqa: E402
    FEATURE_DIM,
    G1_JOINT_NAMES,
    TRACKED_BODY_NAMES,
    WINDOW_SIZE,
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("zero-norm object quaternion")
    q = quaternion / norm
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix.astype(np.float32)


def exact_indices(actual: np.ndarray, required: tuple[str, ...], label: str) -> list[int]:
    names = [str(value) for value in actual.tolist()]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate {label} names")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"missing {label} names: {missing}")
    return [names.index(name) for name in required]


def load_feature_complete_trace(
    trace_path: Path, device: torch.device
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    required = {
        "robot_body_position_w",
        "robot_body_quaternion_w",
        "robot_body_linear_velocity_w",
        "robot_body_angular_velocity_w",
        "robot_joint_position",
        "robot_joint_velocity",
        "object_root_state_w",
        "robot_body_names",
        "robot_joint_names",
    }
    with np.load(trace_path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{trace_path}: missing feature-complete fields {missing}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    body_ids = exact_indices(
        arrays["robot_body_names"], TRACKED_BODY_NAMES, "tracked body"
    )
    joint_ids = exact_indices(arrays["robot_joint_names"], G1_JOINT_NAMES, "joint")
    steps, profiles = arrays["object_root_state_w"].shape[:2]
    if steps < WINDOW_SIZE or profiles != 20:
        raise ValueError(f"unexpected trace geometry {(steps, profiles)}")
    windows_by_profile: list[np.ndarray] = []
    for profile in range(profiles):
        root = arrays["object_root_state_w"][:, profile]
        robot = {
            "joint_pos": arrays["robot_joint_position"][:, profile, joint_ids],
            "joint_vel": arrays["robot_joint_velocity"][:, profile, joint_ids],
            "body_pos_w": arrays["robot_body_position_w"][:, profile, body_ids],
            "body_quat_w": arrays["robot_body_quaternion_w"][:, profile, body_ids],
            "body_lin_vel_w": arrays["robot_body_linear_velocity_w"][:, profile, body_ids],
            "body_ang_vel_w": arrays["robot_body_angular_velocity_w"][:, profile, body_ids],
        }
        obj = {
            "obj_trans": root[:, 0:3],
            "obj_rot": quaternion_wxyz_to_matrix(root[:, 3:7]),
            "obj_lin_vel": root[:, 7:10],
            "obj_ang_vel": root[:, 10:13],
        }
        windows_by_profile.append(build_feature_windows(robot, obj, device))
    windows = np.stack(windows_by_profile, axis=1)
    expected = (steps - WINDOW_SIZE + 1, profiles, WINDOW_SIZE, FEATURE_DIM)
    if windows.shape != expected or not np.isfinite(windows).all():
        raise RuntimeError(f"feature window geometry {windows.shape} != {expected}")
    metadata = {
        "steps": steps,
        "profiles": profiles,
        "windows_per_profile": expected[0],
        "feature_geometry": [WINDOW_SIZE, FEATURE_DIM],
        "tracked_body_names_exact": list(TRACKED_BODY_NAMES),
        "joint_names_exact": list(G1_JOINT_NAMES),
    }
    labels = {
        "done": arrays["done"],
        "robot_root_state_w": arrays["robot_root_state_w"],
        "object_root_state_w": arrays["object_root_state_w"],
        "foot_contact": arrays["foot_contact"],
    }
    return windows, metadata, labels


def chunked_energy(
    model: Any, windows: np.ndarray, device: torch.device, chunk_size: int
) -> np.ndarray:
    flat = windows.transpose(1, 0, 2, 3).reshape(-1, WINDOW_SIZE, FEATURE_DIM)
    chunks = [
        raw_energy(model, flat[start : start + chunk_size], device)
        for start in range(0, flat.shape[0], chunk_size)
    ]
    # Return clock-major [windows, profiles], matching the physical trace clock.
    return np.concatenate(chunks).reshape(windows.shape[1], windows.shape[0]).T


def outcome_labels(labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    root = labels["robot_root_state_w"]
    obj = labels["object_root_state_w"]
    foot = labels["foot_contact"]
    root_loss = root[0, :, 2] - root[:, :, 2].min(axis=0)
    planar_net = np.linalg.norm(obj[-1, :, :2] - obj[0, :, :2], axis=-1)
    any_foot = foot.any(axis=-1).any(axis=0)
    fall = root_loss >= 0.35
    kick = any_foot & (planar_net >= 0.05)
    return {
        "physical_fall": fall,
        "safe_kick": kick & ~fall,
        "kick_success": kick,
        "root_height_loss_m": root_loss,
        "planar_net_displacement_m": planar_net,
    }


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def pairwise_probability_lower(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    return float(np.mean(left[:, None] < right[None, :]))


def summarize_arm(
    carry: np.ndarray, kick: np.ndarray, outcomes: dict[str, np.ndarray]
) -> dict[str, Any]:
    margin = kick - carry
    profile_kick_energy = kick.mean(axis=0)
    safe = outcomes["safe_kick"]
    fall = outcomes["physical_fall"]
    windows = carry.shape[0]
    sections = {
        "early_first_50": slice(0, min(50, windows)),
        "middle": slice(min(50, windows), min(150, windows)),
        "late": slice(min(150, windows), windows),
        "full": slice(0, windows),
    }
    return {
        "carry_energy": describe(carry),
        "kick_energy": describe(kick),
        "kick_minus_carry_energy_margin": describe(margin),
        "kick_preferred_window_fraction": float(np.mean(margin < 0.0)),
        "kick_preferred_profile_count": int(np.sum(margin.mean(axis=0) < 0.0)),
        "sections": {
            name: {
                "kick_minus_carry_energy_margin_mean": float(margin[section].mean()),
                "kick_preferred_window_fraction": float(np.mean(margin[section] < 0.0)),
            }
            for name, section in sections.items()
            if margin[section].size
        },
        "outcomes": {
            "safe_kick_count": int(safe.sum()),
            "physical_fall_count": int(fall.sum()),
            "kick_success_count": int(outcomes["kick_success"].sum()),
        },
        "evaluation_label_only_safety_separation": {
            "safe_mean_kick_energy": (
                float(profile_kick_energy[safe].mean()) if safe.any() else None
            ),
            "fall_mean_kick_energy": (
                float(profile_kick_energy[fall].mean()) if fall.any() else None
            ),
            "pairwise_probability_safe_energy_lower_than_fall": (
                pairwise_probability_lower(profile_kick_energy[safe], profile_kick_energy[fall])
            ),
            "note": "Outcome labels are post-score audit labels and are not prior inputs.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        nargs=2,
        metavar=("NAME", "TRACE"),
        required=True,
        help="Repeat for each named feature-complete frozen trace.",
    )
    parser.add_argument("--prior-root", type=Path, default=PRIOR_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if not output.is_relative_to((ROOT / "experiments").resolve()):
        raise ValueError("output must remain under ignored experiments/")
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    require_compute_gpu(device)
    prior_root = args.prior_root.expanduser().resolve()
    if (prior_root / "motion_disjoint_score/RESULT.json").is_file():
        semantic_result_path = prior_root / "motion_disjoint_score/RESULT.json"
        prior_dirs = {"carry": "carry", "kick": "kick"}
        semantic_gate_key = "passed"
    else:
        semantic_result_path = prior_root / "cross_score/RESULT.json"
        prior_dirs = {"carry": "carry45", "kick": "kick21"}
        semantic_gate_key = "semantic_extension_gate_passed"
    semantic_result = json.loads(semantic_result_path.read_text(encoding="utf-8"))
    arm_data: dict[str, tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]] = {}
    for name, trace_value in args.arm:
        if name in arm_data:
            raise ValueError(f"duplicate arm {name}")
        arm_data[name] = load_feature_complete_trace(Path(trace_value).resolve(), device)

    energies: dict[str, dict[str, np.ndarray]] = {name: {} for name in arm_data}
    for prior_name in ("carry", "kick"):
        model = load_prior(prior_root / "priors" / prior_dirs[prior_name], device)
        for arm_name, (windows, _, _) in arm_data.items():
            energies[arm_name][prior_name] = chunked_energy(
                model, windows, device, args.chunk_size
            )
        del model
        torch.cuda.empty_cache()

    arm_summaries: dict[str, Any] = {}
    output.mkdir(parents=True)
    for arm_name, (_, metadata, labels) in arm_data.items():
        outcomes = outcome_labels(labels)
        carry = energies[arm_name]["carry"]
        kick = energies[arm_name]["kick"]
        arm_summaries[arm_name] = {
            "feature_contract": metadata,
            **summarize_arm(carry, kick, outcomes),
        }
        atomic_npz(
            output / f"{arm_name}_energies.npz",
            carry_energy=carry,
            kick_energy=kick,
            kick_minus_carry_energy=kick - carry,
            **outcomes,
        )

    semantic_gate = bool(semantic_result.get(semantic_gate_key, False))
    actual_kick_gate = all(
        summary["kick_preferred_window_fraction"] >= 0.50
        for summary in arm_summaries.values()
    )
    result = {
        "protocol": "sugar_cross_skill_recovery_official_tinymdm_state_audit_v1",
        "official_prior_root": str(prior_root),
        "arms": arm_summaries,
        "checks": {
            "all_actual_features_are_online_isaaclab_state": True,
            "official_compute_disc_obs_used": True,
            "official_tinymdm_esm_sds_used": True,
            "motion_disjoint_semantic_gate_passed": semantic_gate,
            "actual_recovery_majority_kick_preference_all_arms": actual_kick_gate,
        },
        "policy_integration_supported": bool(semantic_gate and actual_kick_gate),
        "decision": (
            "current_single_clip_priors_admitted_for_state_aware_controller"
            if semantic_gate and actual_kick_gate
            else "current_single_clip_priors_rejected_for_state_aware_controller"
        ),
        "claim_scope": (
            "Frozen official TinyMDM energy audit on actual prefix-recovery states. "
            "This does not train or improve a policy and is not an official latent embedding."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
