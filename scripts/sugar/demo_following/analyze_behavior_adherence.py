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


def load_reference(path: Path) -> dict[str, np.ndarray]:
    robot_path = path / "robot_50hz.npz"
    object_path = path / "obj_motion_global_50hz.pkl"
    if not robot_path.is_file() or not object_path.is_file():
        raise FileNotFoundError(f"incomplete reference motion directory: {path}")
    with np.load(robot_path, allow_pickle=False) as archive:
        robot = {name: np.asarray(archive[name]) for name in archive.files}
    with object_path.open("rb") as stream:
        obj = {name: np.asarray(value) for name, value in pickle.load(stream).items()}
    length = min(robot["joint_pos"].shape[0], obj["obj_trans"].shape[0])
    return {
        "robot_position_w": robot["body_pos_w"][:length, 0].astype(np.float64),
        "object_position_w": obj["obj_trans"][:length].astype(np.float64),
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
        if name != "done"
    }


def longest_true_run(mask: np.ndarray) -> int:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    if edges.size == 0:
        return 0
    return int(np.max(edges[1::2] - edges[::2]))


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
    correct_trace = load_trace(args.correct_trace.resolve())
    unrelated_trace = load_trace(args.unrelated_trace.resolve())
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
    references = {
        "carry45": reference_metrics(load_reference(args.correct_reference.resolve())),
        "kick21": reference_metrics(load_reference(args.unrelated_reference.resolve())),
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

    result = {
        "protocol": "same_teacher_predictor_independent_behavior_audit_v1",
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
        },
        "matched_checks": state_checks,
        "profile_count": profile_count,
        "reference_motion_semantics": {
            "carry45": references["carry45"],
            "kick21": references["kick21"],
            "interpretation": (
                "Carry45 lifts and transports the box; Kick21 keeps it below the "
                "5 cm lift threshold and has more robot orbiting around the box."
            ),
        },
        "actual_arm_summary": {
            "correct": numeric_summary(correct_records),
            "unrelated": numeric_summary(unrelated_records),
        },
        "paired_profile_comparison": comparison,
        "predeclared_semantic_directions": checks,
        "semantic_directions_observed": directions_observed,
        "semantic_directions_total": len(checks),
        "conclusion": (
            "Both learned policies remain strongly Carry-like under the common "
            "Carry45 teacher. The unrelated reward arm does not move toward the "
            "defining Kick21 semantics on any predeclared primary direction. "
            "The selected reward changes behavior within the Carry solution family, "
            "but this seed does not establish semantic demo following."
        ),
        "limitations": [
            "Twenty profiles are matched physics variations from one training seed, not twenty independent policy seeds.",
            "Lifted-transport and ground-transport fractions are complementary views of the same object path, not independent statistical tests.",
            "The trace has no per-body pose or foot-contact fields, so foot identity and direct kick contact cannot be scored retrospectively.",
            "The frozen evaluation starts from Carry motion state and keeps teacher coefficient at one; teacher behavior can dominate the selected reward.",
            "Reference hand-contact forces are unavailable, so contact metrics compare learned arms but are not assigned a numeric reference target.",
        ],
        "required_next_trace_fields": [
            "robot_body_position_w for named feet and hands",
            "left_foot_box_contact_force_w",
            "right_foot_box_contact_force_w",
            "hand_box_contact force separated from other contacts",
        ],
        "artifacts": {
            "profiles_csv": "profiles.csv",
            "figure": "behavior_adherence.png",
        },
    }
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
