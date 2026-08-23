"""Frozen representation constants for the SUGAR G1+box TinyMDM adapter."""

from __future__ import annotations


# Runtime order of the official SUGAR Isaac articulation, action term, and
# serialized Refiner ``joint_pos``/``joint_vel`` columns.  This is deliberately
# not ``joint_sdk_names``: the URDF traversal interleaves left/right chains and
# SUGAR's MotionLoader copies serialized columns directly into this order.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# URDF axes in G1_JOINT_NAMES order.
G1_JOINT_AXES = (
    (0, 1, 0), (0, 1, 0), (0, 0, 1),
    (1, 0, 0), (1, 0, 0), (1, 0, 0),
    (0, 0, 1), (0, 0, 1), (0, 1, 0),
    (0, 1, 0), (0, 1, 0),
    (0, 1, 0), (0, 1, 0),
    (0, 1, 0), (0, 1, 0),
    (1, 0, 0), (1, 0, 0),
    (1, 0, 0), (1, 0, 0),
    (0, 0, 1), (0, 0, 1),
    (0, 1, 0), (0, 1, 0),
    (1, 0, 0), (1, 0, 0),
    (0, 1, 0), (0, 1, 0),
    (0, 0, 1), (0, 0, 1),
)

TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

# The official SUGAR motion archives store all 35 rigid bodies in IsaacLab's
# imported-articulation order. The TinyMDM representation uses the same 14
# bodies selected by MotionCommandCfg. Fixed links with mass remain in the
# 35-body order; the four massless sensor links are merged by the URDF importer.
SOURCE_BODY_NAMES = (
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
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
    "logo_link",
    "head_link",
    "left_shoulder_pitch_link",
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
SOURCE_BODY_INDICES = tuple(SOURCE_BODY_NAMES.index(name) for name in TRACKED_BODY_NAMES)
ROOT_BODY_INDEX = TRACKED_BODY_NAMES.index("torso_link")
KEY_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
KEY_BODY_INDICES = tuple(TRACKED_BODY_NAMES.index(name) for name in KEY_BODY_NAMES)

WINDOW_SIZE = 10
CONTROL_FREQUENCY_HZ = 50
CHARACTER_FEATURE_DIM = 201
OBJECT_FEATURE_DIM = 15
FEATURE_DIM = CHARACTER_FEATURE_DIM + OBJECT_FEATURE_DIM
