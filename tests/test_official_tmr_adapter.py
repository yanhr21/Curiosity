from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/sugar/demo_following/audit_official_tmr_motion_latent.py"
SPEC = importlib.util.spec_from_file_location("official_tmr_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_g1_adapter_has_explicit_nondegenerate_humanml_topology() -> None:
    frames = 8
    body = np.zeros((frames, len(MODULE.SUGAR_BODY_NAMES), 3), dtype=np.float32)
    ids = {name: index for index, name in enumerate(MODULE.SUGAR_BODY_NAMES)}
    points = {
        "pelvis": (0.0, 0.0, 0.8),
        "left_hip_roll_link": (0.0, 0.1, 0.75),
        "right_hip_roll_link": (0.0, -0.1, 0.75),
        "left_knee_link": (0.0, 0.1, 0.4),
        "right_knee_link": (0.0, -0.1, 0.4),
        "left_ankle_pitch_link": (0.0, 0.1, 0.1),
        "right_ankle_pitch_link": (0.0, -0.1, 0.1),
        "left_ankle_roll_link": (0.1, 0.1, 0.08),
        "right_ankle_roll_link": (0.1, -0.1, 0.08),
        "left_shoulder_pitch_link": (0.0, 0.25, 1.2),
        "right_shoulder_pitch_link": (0.0, -0.25, 1.2),
        "left_shoulder_roll_link": (0.0, 0.3, 1.2),
        "right_shoulder_roll_link": (0.0, -0.3, 1.2),
        "left_elbow_link": (0.0, 0.5, 1.0),
        "right_elbow_link": (0.0, -0.5, 1.0),
        "left_wrist_yaw_link": (0.0, 0.6, 0.8),
        "right_wrist_yaw_link": (0.0, -0.6, 0.8),
    }
    for name, point in points.items():
        body[:, ids[name]] = point
    joints = MODULE.g1_bodies_to_humanml_joints(body, MODULE.SUGAR_BODY_NAMES)
    assert joints.shape == (frames, 22, 3)
    audit = MODULE.geometry_audit(joints)
    assert audit["passed"]
    # Proper Z-up -> Y-up rotation keeps the pelvis height in coordinate 1.
    np.testing.assert_allclose(joints[:, 0, 1], 0.8)


def test_windowing_includes_terminal_window_without_duplicates() -> None:
    joints = np.zeros((264, 22, 3), dtype=np.float32)
    result = MODULE.windows(joints)
    assert len(result) == 6
    assert all(item.shape == (80, 22, 3) for item in result)
