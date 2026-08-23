from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/sugar/demo_reward/collect_official_tracker_contact_events.py"
CORPUS_AUDIT = ROOT / "scripts/sugar/demo_reward/audit_actual_contact_event_corpus.py"
DATASET_BUILDER = ROOT / "scripts/sugar/demo_reward/build_actual_contact_event_predictor_dataset.py"


def _functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _load_pure_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_event_durations", "_motion_regime", "_summarize"}
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "np": np,
        "LIFT_THRESHOLD_M": 0.05,
        "MOVE_THRESHOLD_MPS": 0.05,
        "SENSOR_ROLES": ("left_hand", "right_hand", "left_foot", "right_foot"),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


def test_collector_reads_physical_filtered_force_and_not_reference_proxy():
    source = SOURCE.read_text(encoding="utf-8")
    assert "force_matrix_w_history" in source
    assert "reference_binary_proxy_used_as_target\": False" in source
    assert "contact_labels_50hz" not in source
    assert "np.max(source_reference_steps_by_local)" in source
    assert {"_event_durations", "_motion_regime", "_summarize"}.issubset(_functions())


def test_event_duration_stops_before_reset_state_frame():
    functions = _load_pure_functions()
    contact = np.zeros((6, 1, 4), dtype=bool)
    contact[1:5, 0, 0] = True
    reset_before = np.zeros((6, 1), dtype=bool)
    reset_before[3, 0] = True
    total, remaining = functions["_event_durations"](contact, reset_before)
    assert total[:, 0, 0].tolist() == [0, 2, 2, 2, 2, 0]
    assert remaining[:, 0, 0].tolist() == [0, 2, 1, 2, 1, 0]


def test_motion_regime_is_episode_relative():
    functions = _load_pure_functions()
    state = np.zeros((6, 1, 13), dtype=np.float32)
    state[:3, 0, 2] = [0.2, 0.2, 0.26]
    state[:3, 0, 7] = 0.1
    state[3:, 0, 2] = [1.0, 1.0, 1.0]
    reset_before = np.zeros((6, 1), dtype=bool)
    reset_before[3, 0] = True
    lift, regime = functions["_motion_regime"](state, reset_before)
    assert lift[2, 0] > 0.05
    assert np.allclose(lift[3:, 0], 0.0)
    assert regime[2, 0] == 3
    assert np.all(regime[3:, 0] == 0)


def test_corpus_split_is_motion_disjoint_and_stable():
    tree = ast.parse(CORPUS_AUDIT.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"motion_split", "_longest_true_run"}
    ]
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(CORPUS_AUDIT), "exec"),
        namespace,
    )
    split = namespace["motion_split"]
    assert [split(index) for index in (7, 8, 9, 17, 18, 19)] == [
        "train",
        "validation",
        "test",
        "train",
        "validation",
        "test",
    ]
    longest = namespace["_longest_true_run"]
    assert longest(np.asarray([0, 1, 1, 0, 1, 1, 1], dtype=bool)) == 3


def test_event_dataset_geometry_and_reference_role_semantics():
    tree = ast.parse(DATASET_BUILDER.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "quaternion_wxyz_to_rotation6d",
            "event_remaining",
            "reference_contact_roles",
        }
    ]
    namespace = {"np": np}
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(DATASET_BUILDER),
            "exec",
        ),
        namespace,
    )
    rotation = namespace["quaternion_wxyz_to_rotation6d"](
        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    )
    assert np.allclose(rotation, [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    remaining = namespace["event_remaining"](
        np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bool)
    )
    assert remaining.tolist() == [[0, 0], [2, 0], [1, 2], [0, 1]]
    carry = namespace["reference_contact_roles"](
        "CarryBox",
        np.asarray([0, 1, 1], dtype=bool),
        np.zeros((3, 35, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
    )
    assert carry[:, :2].tolist() == [[False, False], [True, True], [True, True]]
    assert not carry[:, 2:].any()
