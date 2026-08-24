#!/usr/bin/env python3
"""Audit demo semantics from frozen rollouts without using predictor outputs.

The current causal pair holds the CarryBox45 teacher fixed and changes only the
selected reward demo (CarryBox45 versus KickBox21).  This analysis deliberately
uses physical rollout state and rigid hand-contact force only.  Demo rewards,
predicted losses, reward terms, and policy losses are forbidden as evidence.

The script does not launch IsaacLab and does not train or evaluate a policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / (
    "experiments/demo_following/matched_reward_identity_same_teacher_v1/"
    "seed161581/evaluation_update0064"
)
DEFAULT_OUTPUT = DEFAULT_RUN.parent / "behavior_adherence_audit_v1"
CORRECT_REFERENCE = ROOT / "SUGAR/data/CarryBox/data_045"
UNRELATED_REFERENCE = ROOT / "SUGAR/data/KickBox/data_021"
CONTROL_DT_S = 0.02
LIFT_THRESHOLD_M = 0.05
HAND_CONTACT_THRESHOLD_N = 0.1
REFERENCE_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "pelvis_contour_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "head_link",
    "left_shoulder_pitch_link",
    "logo_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_rubber_hand",
    "right_rubber_hand",
)

REQUIRED_TRACE_KEYS = {
    "done",
    "robot_root_state_w",
    "robot_joint_pos",
    "robot_joint_vel",
    "object_root_state_w",
    "lift_height_m",
    "left_hand_rigid_contact_force_w",
    "right_hand_rigid_contact_force_w",
}
OPTIONAL_TRACE_KEYS = {
    "policy_updates",
    "robot_body_position_w",
    "left_foot_box_contact_force_w",
    "right_foot_box_contact_force_w",
}
FORBIDDEN_EVIDENCE_PREFIXES = (
    "demo_",
    "reward",
    "manager_reward",
    "weighted_task_outcome_reward",
    "external_constraint_reward",
)

# These four directions are defined from the task semantics before reading the
# arm comparison: Carry means lifted, bilateral transport; Kick means ground
# interaction and motion around the box.  They are reported separately rather
# than collapsed into a tunable score.
PRIMARY_DIRECTIONS = {
    "correct_has_more_lifted_time": ("lifted_fraction", "correct_gt_unrelated"),
    "correct_has_more_lifted_transport": (
        "lifted_transport_fraction",
        "correct_gt_unrelated",
    ),
    "unrelated_has_more_ground_transport": (
        "ground_transport_fraction",
        "unrelated_gt_correct",
    ),
    "unrelated_has_more_orbiting": (
        "root_orbit_rate_rad_s",
        "unrelated_gt_correct",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correct-trace", type=Path, default=DEFAULT_RUN / "correct/TRACE.npz"
    )
    parser.add_argument(
        "--unrelated-trace",
        type=Path,
        default=DEFAULT_RUN / "unrelated/TRACE.npz",
    )
    parser.add_argument("--correct-reference", type=Path, default=CORRECT_REFERENCE)
    parser.add_argument(
        "--unrelated-reference", type=Path, default=UNRELATED_REFERENCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--policy-update",
        type=int,
        default=None,
        help="select one checkpoint block from a multi-update frozen trace",
    )
    parser.add_argument(
        "--same-checkpoint-condition-swap",
        action="store_true",
        help="Label the comparison as two conditions of one frozen checkpoint.",
    )
    return parser.parse_args()


def load_trace(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_TRACE_KEYS.difference(archive.files)
        if missing:
            raise KeyError(f"{path} is missing required trace fields: {sorted(missing)}")
        selected = REQUIRED_TRACE_KEYS | OPTIONAL_TRACE_KEYS.intersection(archive.files)
        return {name: np.asarray(archive[name]) for name in selected}


def select_policy_update(
    trace: dict[str, np.ndarray], update: int | None
) -> dict[str, np.ndarray]:
    if update is None:
        return trace
    if "policy_updates" not in trace:
        raise KeyError("--policy-update requires policy_updates in the trace")
    updates = np.asarray(trace["policy_updates"], dtype=np.int64)
    rows = np.flatnonzero(updates == int(update))
    if rows.size != 1 or trace["done"].shape[1] % len(updates):
        raise ValueError("policy update blocks are not uniquely addressable")
    profiles_per_update = trace["done"].shape[1] // len(updates)
    start = int(rows[0]) * profiles_per_update
    stop = start + profiles_per_update
    selected: dict[str, np.ndarray] = {}
    for name, value in trace.items():
        if name == "policy_updates":
            selected[name] = np.asarray([update], dtype=np.int64)
        elif value.ndim >= 2 and value.shape[1] == trace["done"].shape[1]:
            selected[name] = value[:, start:stop]
        else:
            selected[name] = value
    return selected


def load_reference(path: Path) -> dict[str, np.ndarray]:
    robot_path = path / "robot_50hz.npz"
    object_path = path / "obj_motion_global_50hz.pkl"
    contact_path = path / "contact_labels_50hz.npy"
    if not robot_path.is_file() or not object_path.is_file() or not contact_path.is_file():
        raise FileNotFoundError(f"incomplete reference motion directory: {path}")
    with np.load(robot_path, allow_pickle=False) as archive:
        robot = {name: np.asarray(archive[name]) for name in archive.files}
    with object_path.open("rb") as stream:
        obj = {name: np.asarray(value) for name, value in pickle.load(stream).items()}
    contact = np.load(contact_path, allow_pickle=False)
    length = min(
        robot["joint_pos"].shape[0], obj["obj_trans"].shape[0], contact.shape[0]
    )
    if robot["body_pos_w"].shape[1] != len(REFERENCE_BODY_NAMES):
        raise RuntimeError("reference G1 body order no longer matches the 35-body contract")
    return {
        "robot_position_w": robot["body_pos_w"][:length, 0].astype(np.float64),
        "robot_body_position_w": robot["body_pos_w"][:length].astype(np.float64),
        "object_position_w": obj["obj_trans"][:length].astype(np.float64),
        "object_linear_velocity_w": obj["obj_lin_vel"][:length].astype(np.float64),
        "contact_label": contact[:length].astype(bool),
        "joint_pos": robot["joint_pos"][:length].astype(np.float64),
        "joint_vel": robot["joint_vel"][:length].astype(np.float64),
    }


def first_episode(trace: dict[str, np.ndarray], env_index: int) -> dict[str, np.ndarray]:
    done = trace["done"][:, env_index]
    hits = np.flatnonzero(done)
    last_transition = int(hits[0]) if hits.size else done.shape[0] - 1
    frame_count = last_transition + 2
    return {
        name: value[:frame_count, env_index]
        for name, value in trace.items()
        if (
            name != "done"
            and value.ndim >= 2
            and value.shape[1] == trace["done"].shape[1]
        )
    }


def longest_true_run(mask: np.ndarray) -> int:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    if edges.size == 0:
        return 0
    return int(np.max(edges[1::2] - edges[::2]))


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive [start, end] intervals for a boolean sequence."""
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(stop - 1)) for start, stop in zip(edges[::2], edges[1::2])]


def first_sustained(mask: np.ndarray, frames: int = 5) -> int | None:
    if frames <= 0:
        raise ValueError("sustained-event length must be positive")
    values = np.asarray(mask, dtype=np.int8)
    if values.size < frames:
        return None
    hits = np.flatnonzero(np.convolve(values, np.ones(frames, dtype=np.int8), mode="valid") == frames)
    return int(hits[0]) if hits.size else None


def reference_event_timeline(
    reference: dict[str, np.ndarray], *, contact_role: str
) -> dict[str, Any]:
    if contact_role not in ("hands", "feet"):
        raise ValueError(contact_role)
    object_position = reference["object_position_w"]
    baseline_z = float(np.median(object_position[:25, 2]))
    lift = object_position[:, 2] - baseline_z
    speed = np.linalg.norm(reference["object_linear_velocity_w"], axis=-1)
    labels = reference["contact_label"]
    runs = true_runs(labels)
    lift_frames = np.flatnonzero(lift >= LIFT_THRESHOLD_M)
    body_names = (
        ("left_rubber_hand", "right_rubber_hand")
        if contact_role == "hands"
        else ("left_ankle_roll_link", "right_ankle_roll_link")
    )
    body_ids = [REFERENCE_BODY_NAMES.index(name) for name in body_names]
    distances = np.stack(
        [
            np.linalg.norm(
                reference["robot_body_position_w"][:, body_id] - object_position,
                axis=-1,
            )
            for body_id in body_ids
        ],
        axis=-1,
    )
    frame, side = (
        int(value)
        for value in np.unravel_index(int(np.argmin(distances)), distances.shape)
    )
    labeled_distances = distances[labels]
    return {
        "contact_role": contact_role,
        "contact_label_is_binary_proxy": True,
        "contact_run_count": len(runs),
        "contact_runs": [
            {
                "start_frame": start,
                "end_frame": end,
                "start_s": start * CONTROL_DT_S,
                "end_s": end * CONTROL_DT_S,
            }
            for start, end in runs
        ],
        "first_contact_frame": runs[0][0] if runs else None,
        "last_contact_frame": runs[-1][1] if runs else None,
        "contact_fraction": float(np.mean(labels)),
        "first_sustained_object_motion_frame": first_sustained(speed >= 0.05),
        "first_sustained_lift_frame": first_sustained(lift >= LIFT_THRESHOLD_M),
        "last_lift_frame": int(lift_frames[-1]) if lift_frames.size else None,
        "peak_lift_frame": int(np.argmax(lift)),
        "peak_lift_m": float(np.max(lift)),
        "closest_effector": body_names[side],
        "closest_effector_frame": frame,
        "closest_effector_center_distance_m": float(distances[frame, side]),
        "minimum_effector_center_distance_during_labeled_contact_m": (
            float(np.min(labeled_distances)) if labeled_distances.size else None
        ),
    }


def kinematic_metrics(
    robot_position_w: np.ndarray,
    object_position_w: np.ndarray,
    lift_height_m: np.ndarray,
    *,
    dt_s: float = CONTROL_DT_S,
) -> dict[str, float]:
    if robot_position_w.shape != object_position_w.shape:
        raise ValueError("robot and object position arrays must have the same shape")
    if robot_position_w.ndim != 2 or robot_position_w.shape[1] != 3:
        raise ValueError("positions must have shape [frames, 3]")
    if lift_height_m.shape != (robot_position_w.shape[0],):
        raise ValueError("lift height must have one scalar per frame")
    if robot_position_w.shape[0] < 2:
        raise ValueError("an episode must contain at least two frames")

    object_step_xy = np.linalg.norm(np.diff(object_position_w[:, :2], axis=0), axis=1)
    midpoint_lift = 0.5 * (lift_height_m[:-1] + lift_height_m[1:])
    lifted_step = midpoint_lift >= LIFT_THRESHOLD_M
    total_xy = float(object_step_xy.sum())
    lifted_xy = float(object_step_xy[lifted_step].sum())
    ground_xy = float(object_step_xy[~lifted_step].sum())

    relative_xy = robot_position_w[:, :2] - object_position_w[:, :2]
    bearing = np.unwrap(np.arctan2(relative_xy[:, 1], relative_xy[:, 0]))
    orbit_total = float(np.abs(np.diff(bearing)).sum())
    duration_s = (robot_position_w.shape[0] - 1) * dt_s
    return {
        "duration_s": float(duration_s),
        "maximum_lift_m": float(np.max(lift_height_m)),
        "lifted_fraction": float(np.mean(lift_height_m >= LIFT_THRESHOLD_M)),
        "object_horizontal_path_m": total_xy,
        "lifted_horizontal_path_m": lifted_xy,
        "ground_horizontal_path_m": ground_xy,
        "lifted_transport_fraction": lifted_xy / max(total_xy, 1.0e-12),
        "ground_transport_fraction": ground_xy / max(total_xy, 1.0e-12),
        "root_orbit_total_rad": orbit_total,
        "root_orbit_rate_rad_s": orbit_total / max(duration_s, 1.0e-12),
        "robot_horizontal_path_m": float(
            np.linalg.norm(np.diff(robot_position_w[:, :2], axis=0), axis=1).sum()
        ),
    }


def actual_metrics(episode: dict[str, np.ndarray]) -> dict[str, float]:
    robot_position = episode["robot_root_state_w"][:, :3].astype(np.float64)
    object_position = episode["object_root_state_w"][:, :3].astype(np.float64)
    lift = episode["lift_height_m"].astype(np.float64)
    metrics = kinematic_metrics(robot_position, object_position, lift)

    left_force = np.linalg.norm(
        episode["left_hand_rigid_contact_force_w"].astype(np.float64), axis=-1
    )
    right_force = np.linalg.norm(
        episode["right_hand_rigid_contact_force_w"].astype(np.float64), axis=-1
    )
    left = left_force > HAND_CONTACT_THRESHOLD_N
    right = right_force > HAND_CONTACT_THRESHOLD_N
    bilateral = left & right
    unilateral = left ^ right
    lifted = lift >= LIFT_THRESHOLD_M
    object_step_xy = np.linalg.norm(np.diff(object_position[:, :2], axis=0), axis=1)
    midpoint_lift = 0.5 * (lift[:-1] + lift[1:])
    carried_step = bilateral[:-1] & (midpoint_lift >= LIFT_THRESHOLD_M)
    metrics.update(
        {
            "bilateral_contact_fraction": float(np.mean(bilateral)),
            "unilateral_contact_fraction": float(np.mean(unilateral)),
            "bilateral_lift_fraction": float(np.mean(bilateral & lifted)),
            "longest_bilateral_contact_s": float(
                longest_true_run(bilateral) * CONTROL_DT_S
            ),
            "bilateral_lifted_transport_fraction": float(
                object_step_xy[carried_step].sum()
                / max(float(object_step_xy.sum()), 1.0e-12)
            ),
            "peak_left_hand_force_n": float(left_force.max()),
            "peak_right_hand_force_n": float(right_force.max()),
        }
    )
    if {
        "left_foot_box_contact_force_w",
        "right_foot_box_contact_force_w",
    }.issubset(episode):
        left_foot = (
            np.linalg.norm(episode["left_foot_box_contact_force_w"], axis=-1)
            > HAND_CONTACT_THRESHOLD_N
        )
        right_foot = (
            np.linalg.norm(episode["right_foot_box_contact_force_w"], axis=-1)
            > HAND_CONTACT_THRESHOLD_N
        )
        metrics.update(
            {
                "left_foot_box_contact_fraction": float(np.mean(left_foot)),
                "right_foot_box_contact_fraction": float(np.mean(right_foot)),
                "any_foot_box_contact_fraction": float(
                    np.mean(left_foot | right_foot)
                ),
            }
        )
    return metrics


def reference_metrics(reference: dict[str, np.ndarray]) -> dict[str, float]:
    object_position = reference["object_position_w"]
    # Both source clips begin with the object supported by the ground.  The
    # median of the first 0.5 s makes the physical 5 cm threshold insensitive
    # to a single mocap frame without fitting anything to the policy traces.
    baseline_z = float(np.median(object_position[:25, 2]))
    lift = object_position[:, 2] - baseline_z
    return kinematic_metrics(
        reference["robot_position_w"], object_position, lift
    )


def numeric_summary(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = records[0].keys()
    return {
        key: {
            "mean": float(np.mean([record[key] for record in records])),
            "median": float(np.median([record[key] for record in records])),
            "min": float(np.min([record[key] for record in records])),
            "max": float(np.max([record[key] for record in records])),
        }
        for key in keys
    }


def carry_structure_preserved(summary: dict[str, dict[str, float]]) -> bool:
    """Require sustained bilateral hold, lift and airborne transport."""
    return bool(
        summary["bilateral_contact_fraction"]["mean"] >= 0.50
        and summary["lifted_fraction"]["mean"] >= 0.50
        and summary["lifted_transport_fraction"]["mean"] >= 0.80
    )


def compare_profiles(
    correct: list[dict[str, float]], unrelated: list[dict[str, float]]
) -> dict[str, dict[str, float | int]]:
    comparison: dict[str, dict[str, float | int]] = {}
    for key in correct[0]:
        correct_values = np.asarray([record[key] for record in correct])
        unrelated_values = np.asarray([record[key] for record in unrelated])
        delta = unrelated_values - correct_values
        comparison[key] = {
            "correct_mean": float(correct_values.mean()),
            "unrelated_mean": float(unrelated_values.mean()),
            "unrelated_minus_correct_mean": float(delta.mean()),
            "unrelated_minus_correct_median": float(np.median(delta)),
            "profiles_positive": int(np.count_nonzero(delta > 0.0)),
            "profiles_negative": int(np.count_nonzero(delta < 0.0)),
            "profiles_equal": int(np.count_nonzero(delta == 0.0)),
        }
    return comparison


def primary_checks(comparison: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, (metric, direction) in PRIMARY_DIRECTIONS.items():
        delta = float(comparison[metric]["unrelated_minus_correct_mean"])
        passed = delta < 0.0 if direction == "correct_gt_unrelated" else delta > 0.0
        checks[name] = {
            "metric": metric,
            "predeclared_direction": direction,
            "unrelated_minus_correct_mean": delta,
            "direction_observed": bool(passed),
        }
    return checks


def summarize_behavior_shift(
    observed: int, total: int, comparison_subject: str = "unrelated reward arm"
) -> str:
    if observed == 0:
        return (
            f"The {comparison_subject} does not move toward Kick21 on any "
            "predeclared primary direction."
        )
    if observed < total:
        return (
            f"The {comparison_subject} moves toward Kick21 on {observed}/{total} "
            "predeclared primary directions, but does not reproduce the complete "
            "Kick21 interaction structure."
        )
    return (
        f"The {comparison_subject} moves toward Kick21 on all predeclared "
        "primary directions in this training seed; independent seed "
        "replication is still required."
    )


def write_profiles_csv(
    path: Path,
    correct: list[dict[str, float]],
    unrelated: list[dict[str, float]],
) -> None:
    fieldnames = ["arm", "profile_index", *correct[0].keys()]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for arm, records in (("correct", correct), ("unrelated", unrelated)):
            for profile_index, record in enumerate(records):
                writer.writerow(
                    {"arm": arm, "profile_index": profile_index, **record}
                )


def write_figure(
    path: Path,
    correct: list[dict[str, float]],
    unrelated: list[dict[str, float]],
    references: dict[str, dict[str, float]],
) -> None:
    panels = (
        ("maximum_lift_m", "Maximum box lift (m)"),
        ("lifted_transport_fraction", "Lifted share of box path"),
        ("ground_transport_fraction", "Ground-level share of box path"),
        ("root_orbit_rate_rad_s", "Robot orbit rate (rad/s)"),
        ("bilateral_contact_fraction", "Bilateral hand-contact fraction"),
        ("unilateral_contact_fraction", "Unilateral hand-contact fraction"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)
    colors = ("#1f77b4", "#d62728")
    rng = np.random.default_rng(20260823)
    for axis, (metric, label) in zip(axes.flat, panels):
        values = (
            np.asarray([record[metric] for record in correct]),
            np.asarray([record[metric] for record in unrelated]),
        )
        for profile in range(len(correct)):
            axis.plot((0, 1), (values[0][profile], values[1][profile]), color="0.82", lw=0.7, zorder=1)
        for index, arm_values in enumerate(values):
            jitter = rng.uniform(-0.045, 0.045, size=arm_values.shape[0])
            axis.scatter(
                index + jitter,
                arm_values,
                s=15,
                color=colors[index],
                alpha=0.8,
                zorder=2,
            )
            axis.scatter(index, arm_values.mean(), s=65, marker="D", color="black", zorder=3)
        if metric in references["carry45"]:
            axis.axhline(
                references["carry45"][metric], color=colors[0], ls="--", lw=1.2,
                label="Carry45 reference",
            )
            axis.axhline(
                references["kick21"][metric], color=colors[1], ls=":", lw=1.5,
                label="Kick21 reference",
            )
        axis.set_xticks((0, 1), ("Carry reward", "Kick reward"))
        axis.set_title(label, fontsize=10)
        axis.grid(axis="y", color="0.9", lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Same Carry45 teacher: predictor-independent physical behavior audit\n"
        "Each line is one matched physics profile; diamonds are arm means",
        fontsize=13,
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_reference_timeline(
    path: Path,
    references: dict[str, dict[str, np.ndarray]],
    timelines: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, constrained_layout=True)
    contracts = (
        ("carry45", "CarryBox45: binary hand-contact proxy", "#1f77b4"),
        ("kick21", "KickBox21: binary foot-contact proxy", "#d62728"),
    )
    for axis, (name, title, color) in zip(axes, contracts):
        reference = references[name]
        object_position = reference["object_position_w"]
        lift = object_position[:, 2] - float(np.median(object_position[:25, 2]))
        speed_xy = np.linalg.norm(reference["object_linear_velocity_w"][:, :2], axis=-1)
        time_s = np.arange(lift.shape[0]) * CONTROL_DT_S
        axis.plot(time_s, lift, color="black", lw=1.6, label="box lift")
        for index, run in enumerate(timelines[name]["contact_runs"]):
            axis.axvspan(
                run["start_s"],
                run["end_s"],
                color=color,
                alpha=0.18,
                label="contact proxy active" if index == 0 else None,
            )
        axis.axhline(LIFT_THRESHOLD_M, color="0.4", ls="--", lw=1.0, label="5 cm lift")
        velocity_axis = axis.twinx()
        velocity_axis.plot(time_s, speed_xy, color=color, alpha=0.55, lw=1.0, label="box XY speed")
        velocity_axis.set_ylabel("XY speed (m/s)", color=color)
        axis.set_ylabel("Lift (m)")
        axis.set_title(title)
        axis.grid(axis="y", color="0.9")
        axis.spines["top"].set_visible(False)
        velocity_axis.spines["top"].set_visible(False)
        lines, labels = axis.get_legend_handles_labels()
        velocity_lines, velocity_labels = velocity_axis.get_legend_handles_labels()
        axis.legend(lines + velocity_lines, labels + velocity_labels, frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("Reference time (s)")
    fig.suptitle(
        "Concrete reference interaction timelines (50 Hz)\n"
        "Shading is the official binary contact proxy, not tactile force",
        fontsize=12,
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def initial_state_checks(
    correct: dict[str, np.ndarray], unrelated: dict[str, np.ndarray]
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for key in ("robot_root_state_w", "robot_joint_pos", "object_root_state_w"):
        delta = np.abs(correct[key][0] - unrelated[key][0])
        checks[f"initial_{key}_max_abs"] = float(delta.max())
    checks["profile_count_equal"] = bool(
        correct["done"].shape[1] == unrelated["done"].shape[1]
    )
    checks["frame_budget_equal"] = bool(
        correct["done"].shape[0] == unrelated["done"].shape[0]
    )
    checks["initial_state_exact_match"] = bool(
        all(
            checks[f"initial_{key}_max_abs"] == 0.0
            for key in ("robot_root_state_w", "robot_joint_pos", "object_root_state_w")
        )
    )
    return checks


def main() -> None:
    args = parse_args()
    correct_trace = select_policy_update(
        load_trace(args.correct_trace.resolve()), args.policy_update
    )
    unrelated_trace = select_policy_update(
        load_trace(args.unrelated_trace.resolve()), args.policy_update
    )
    state_checks = initial_state_checks(correct_trace, unrelated_trace)
    if not state_checks["profile_count_equal"] or not state_checks["frame_budget_equal"]:
        raise RuntimeError("correct and unrelated traces are not profile-matched")

    profile_count = correct_trace["done"].shape[1]
    correct_records = [
        actual_metrics(first_episode(correct_trace, index))
        for index in range(profile_count)
    ]
    unrelated_records = [
        actual_metrics(first_episode(unrelated_trace, index))
        for index in range(profile_count)
    ]
    reference_data = {
        "carry45": load_reference(args.correct_reference.resolve()),
        "kick21": load_reference(args.unrelated_reference.resolve()),
    }
    references = {
        name: reference_metrics(reference)
        for name, reference in reference_data.items()
    }
    reference_timelines = {
        "carry45": reference_event_timeline(reference_data["carry45"], contact_role="hands"),
        "kick21": reference_event_timeline(reference_data["kick21"], contact_role="feet"),
    }
    comparison = compare_profiles(correct_records, unrelated_records)
    checks = primary_checks(comparison)
    directions_observed = sum(
        int(record["direction_observed"]) for record in checks.values()
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_profiles_csv(output_dir / "profiles.csv", correct_records, unrelated_records)
    write_figure(
        output_dir / "behavior_adherence.png",
        correct_records,
        unrelated_records,
        references,
    )
    write_reference_timeline(
        output_dir / "reference_semantic_timeline.png",
        reference_data,
        reference_timelines,
    )

    behavior_conclusion = summarize_behavior_shift(
        directions_observed,
        len(checks),
        (
            "unrelated demo condition"
            if args.same_checkpoint_condition_swap
            else "unrelated reward arm"
        ),
    )
    correct_summary = numeric_summary(correct_records)
    unrelated_summary = numeric_summary(unrelated_records)
    correct_carry_preserved = carry_structure_preserved(correct_summary)
    unrelated_carry_preserved = carry_structure_preserved(unrelated_summary)
    if correct_carry_preserved and unrelated_carry_preserved:
        if args.same_checkpoint_condition_swap:
            conclusion = (
                "The same frozen policy retains a usable Carry solution under both "
                f"demo conditions. {behavior_conclusion} Swapping only the selected "
                "demo condition changes behavior within the Carry solution family, "
                "but this seed alone does not establish semantic demo following."
            )
        else:
            conclusion = (
                "Both learned policies retain a usable Carry solution under the common "
                f"teacher. {behavior_conclusion} The selected reward changes behavior "
                "within the Carry solution family, but this seed alone does not establish "
                "semantic demo following."
            )
    elif correct_carry_preserved:
        subject = (
            "Swapping only the selected demo condition"
            if args.same_checkpoint_condition_swap
            else "Selecting the unrelated reward arm"
        )
        conclusion = (
            "The correct condition preserves the predeclared Carry contact/lift/"
            f"transport structure, while the unrelated condition does not. {behavior_conclusion} "
            f"{subject} therefore creates a strong physical behavior split, but the "
            "unrelated condition is a failed/falling response rather than verified "
            "KickBox imitation. This endpoint establishes condition use, not semantic "
            "demo following."
        )
    else:
        conclusion = (
            "The correct arm does not preserve the predeclared Carry contact/lift/"
            "transport structure, so this teacher-authority setting causes behavioral "
            f"collapse rather than a valid Carry-versus-Kick separation. {behavior_conclusion} "
            "This endpoint cannot establish semantic demo following."
        )

    result = {
        "protocol": (
            "same_checkpoint_condition_swap_behavior_audit_v1"
            if args.same_checkpoint_condition_swap
            else "same_teacher_predictor_independent_behavior_audit_v1"
        ),
        "status": "complete_existing_trace_audit",
        "question": (
            "With CarryBox45 teacher fixed, did selecting KickBox21 as reward demo "
            "produce Kick-like physical behavior rather than merely another Carry solution?"
        ),
        "evidence_contract": {
            "used_trace_keys": sorted(
                set(correct_trace).intersection(unrelated_trace)
            ),
            "forbidden_as_behavior_evidence": list(FORBIDDEN_EVIDENCE_PREFIXES),
            "uses_predictor_output": False,
            "uses_demo_reward": False,
            "lift_threshold_m": LIFT_THRESHOLD_M,
            "hand_contact_threshold_n": HAND_CONTACT_THRESHOLD_N,
            "control_dt_s": CONTROL_DT_S,
            "same_checkpoint_condition_swap": args.same_checkpoint_condition_swap,
        },
        "matched_checks": state_checks,
        "profile_count": profile_count,
        "policy_update": args.policy_update,
        "reference_motion_semantics": {
            "carry45": references["carry45"],
            "kick21": references["kick21"],
            "interpretation": (
                "Carry45 lifts and transports the box; Kick21 keeps it below the "
                "5 cm lift threshold and has more robot orbiting around the box."
            ),
        },
        "reference_event_timeline": reference_timelines,
        "actual_arm_summary": {
            "correct": correct_summary,
            "unrelated": unrelated_summary,
        },
        "paired_profile_comparison": comparison,
        "predeclared_semantic_directions": checks,
        "semantic_directions_observed": directions_observed,
        "semantic_directions_total": len(checks),
        "correct_carry_structure_preserved": correct_carry_preserved,
        "unrelated_carry_structure_preserved": unrelated_carry_preserved,
        "conclusion": conclusion,
        "limitations": [
            "Twenty profiles are matched physics variations from one training seed, not twenty independent policy seeds.",
            "Lifted-transport and ground-transport fractions are complementary views of the same object path, not independent statistical tests.",
            "The frozen evaluation starts from the Carry motion state; the common teacher schedule can still dominate or destabilize both selected-demo arms.",
            "Reference hand-contact forces are unavailable, so contact metrics compare learned arms but are not assigned a numeric reference target.",
        ],
        "required_next_trace_fields": [],
        "artifacts": {
            "profiles_csv": "profiles.csv",
            "figure": "behavior_adherence.png",
            "reference_timeline_figure": "reference_semantic_timeline.png",
        },
    }
    required_trace_fields = {
        "robot_body_position_w": "robot_body_position_w for named feet and hands",
        "left_foot_box_contact_force_w": "left_foot_box_contact_force_w",
        "right_foot_box_contact_force_w": "right_foot_box_contact_force_w",
        "left_hand_rigid_contact_force_w": "left hand-to-box contact force",
        "right_hand_rigid_contact_force_w": "right hand-to-box contact force",
    }
    missing_trace_fields = [
        description
        for key, description in required_trace_fields.items()
        if key not in correct_trace or key not in unrelated_trace
    ]
    result["required_next_trace_fields"] = missing_trace_fields
    if missing_trace_fields:
        result["limitations"].append(
            "Direct contact-role scoring is incomplete because the listed trace fields are missing."
        )
    (output_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "semantic_directions_observed": directions_observed,
        "semantic_directions_total": len(checks),
        "conclusion": result["conclusion"],
    }, indent=2))


if __name__ == "__main__":
    main()
