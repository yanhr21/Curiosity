from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/sugar/demo_following/analyze_behavior_adherence.py"
)
SPEC = importlib.util.spec_from_file_location("behavior_adherence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

AGGREGATE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/sugar/demo_following/aggregate_behavior_adherence.py"
)
AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "aggregate_behavior_adherence", AGGREGATE_SCRIPT
)
assert AGGREGATE_SPEC is not None and AGGREGATE_SPEC.loader is not None
AGGREGATE = importlib.util.module_from_spec(AGGREGATE_SPEC)
AGGREGATE_SPEC.loader.exec_module(AGGREGATE)


def test_longest_true_run() -> None:
    assert MODULE.longest_true_run(np.array([False, True, True, False, True])) == 2
    assert MODULE.longest_true_run(np.zeros(4, dtype=bool)) == 0
    assert MODULE.longest_true_run(np.ones(4, dtype=bool)) == 4
    assert MODULE.true_runs(np.array([False, True, True, False, True])) == [(1, 2), (4, 4)]
    assert MODULE.first_sustained(np.array([False, True, True, True]), frames=3) == 1
    assert MODULE.first_sustained(np.array([False, True, False]), frames=2) is None


def test_lifted_and_ground_transport_partition() -> None:
    robot = np.zeros((4, 3), dtype=np.float64)
    obj = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.1], [3.0, 0.0, 0.1]]
    )
    lift = obj[:, 2]
    metrics = MODULE.kinematic_metrics(robot, obj, lift)
    assert metrics["object_horizontal_path_m"] == 3.0
    assert metrics["ground_horizontal_path_m"] == 1.0
    assert metrics["lifted_horizontal_path_m"] == 2.0
    assert np.isclose(metrics["ground_transport_fraction"], 1.0 / 3.0)
    assert np.isclose(metrics["lifted_transport_fraction"], 2.0 / 3.0)


def test_primary_directions_are_not_a_predictor_score() -> None:
    comparison = {
        "lifted_fraction": {"unrelated_minus_correct_mean": -0.2},
        "lifted_transport_fraction": {"unrelated_minus_correct_mean": -0.3},
        "ground_transport_fraction": {"unrelated_minus_correct_mean": 0.3},
        "root_orbit_rate_rad_s": {"unrelated_minus_correct_mean": 0.1},
    }
    checks = MODULE.primary_checks(comparison)
    assert all(record["direction_observed"] for record in checks.values())
    assert all("predict" not in record["metric"] for record in checks.values())


def test_future_foot_box_contact_is_evaluation_only_metric() -> None:
    frames = 4
    episode = {
        "robot_root_state_w": np.zeros((frames, 13), dtype=np.float32),
        "object_root_state_w": np.zeros((frames, 13), dtype=np.float32),
        "lift_height_m": np.zeros(frames, dtype=np.float32),
        "left_hand_rigid_contact_force_w": np.zeros((frames, 3), dtype=np.float32),
        "right_hand_rigid_contact_force_w": np.zeros((frames, 3), dtype=np.float32),
        "left_foot_box_contact_force_w": np.array(
            [[0, 0, 0], [0.2, 0, 0], [0.2, 0, 0], [0, 0, 0]],
            dtype=np.float32,
        ),
        "right_foot_box_contact_force_w": np.zeros((frames, 3), dtype=np.float32),
    }
    metrics = MODULE.actual_metrics(episode)
    assert metrics["left_foot_box_contact_fraction"] == 0.5
    assert metrics["right_foot_box_contact_fraction"] == 0.0
    assert metrics["any_foot_box_contact_fraction"] == 0.5


def test_multiseed_direction_contract() -> None:
    assert AGGREGATE.expected_direction(-0.1, "negative")
    assert not AGGREGATE.expected_direction(0.1, "negative")
    assert AGGREGATE.expected_direction(0.1, "positive")
    assert not AGGREGATE.expected_direction(-0.1, "positive")


def test_behavior_conclusion_matches_observed_direction_count() -> None:
    assert "does not move" in MODULE.summarize_behavior_shift(0, 4)
    assert "3/4" in MODULE.summarize_behavior_shift(3, 4)
    assert "all predeclared" in MODULE.summarize_behavior_shift(4, 4)
