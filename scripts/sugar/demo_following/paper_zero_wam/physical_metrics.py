"""Pure-NumPy physical outcome reducer shared by rollout and final audits."""

from __future__ import annotations

from typing import Any

import numpy as np


LIFT_THRESHOLD_M = 0.05
FALL_HEIGHT_LOSS_M = 0.35
FALL_ROOT_TILT_DEG = 60.0
KICK_NET_DISPLACEMENT_M = 0.05
KICK_CONTACT_COUPLED_PATH_M = 0.01
KICK_POST_CONTACT_PATH_M = 0.03

RESTORE_PAYLOAD_KEYS = (
    "object_materials",
    "robot_materials",
    "object_masses",
    "object_inertias",
    "object_coms",
    "robot_masses",
    "robot_inertias",
    "robot_coms",
    "robot_root_state_w",
    "robot_joint_pos",
    "robot_joint_vel",
    "object_root_state_w",
    "default_joint_pos",
    "previous_action",
    "action_manager_prev_action",
)


def planar_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left[:2] - right[:2]))


def physical_summary(
    robot_root: np.ndarray, object_root: np.ndarray, contact: np.ndarray
) -> dict[str, Any]:
    """Recompute the frozen Carry/Kick/fall decision from one complete trace."""

    objects = object_root[:, :3]
    heights = robot_root[:, 2]
    quaternions = robot_root[:, 3:7]
    baseline_z = float(np.median(objects[:25, 2]))
    maximum_lift = float(np.max(objects[:, 2] - baseline_z))
    root_height_loss = float(heights[0] - np.min(heights))
    up_z = np.clip(
        1.0 - 2.0 * (quaternions[:, 1] ** 2 + quaternions[:, 2] ** 2),
        -1.0,
        1.0,
    )
    maximum_tilt = float(np.degrees(np.arccos(up_z)).max())
    fall = root_height_loss >= FALL_HEIGHT_LOSS_M or maximum_tilt >= FALL_ROOT_TILT_DEG
    bilateral = contact[:, 0] & contact[:, 1]
    foot = contact[:, 2] | contact[:, 3]
    step_paths = np.linalg.norm(np.diff(objects[:, :2], axis=0), axis=1)
    planar_net = planar_distance(objects[-1], objects[0])
    coupled = float(
        sum(
            step_paths[index - 1]
            for index in range(1, len(objects))
            if foot[index - 1] or foot[index]
        )
    )
    if np.any(foot):
        first_contact = int(np.flatnonzero(foot)[0])
        post_contact = float(step_paths[max(first_contact - 1, 0) :].sum())
    else:
        post_contact = 0.0
    carry = bool(np.any(bilateral) and maximum_lift >= LIFT_THRESHOLD_M)
    kick = bool(
        np.any(foot)
        and planar_net >= KICK_NET_DISPLACEMENT_M
        and coupled >= KICK_CONTACT_COUPLED_PATH_M
        and post_contact >= KICK_POST_CONTACT_PATH_M
    )
    return {
        "maximum_lift_m": maximum_lift,
        "bilateral_contact_frames": int(bilateral.sum()),
        "foot_contact_frames": int(foot.sum()),
        "planar_object_net_displacement_m": planar_net,
        "contact_coupled_planar_path_m": coupled,
        "post_first_contact_planar_path_m": post_contact,
        "maximum_robot_root_height_loss_m": root_height_loss,
        "maximum_robot_root_tilt_deg": maximum_tilt,
        "physical_fall": fall,
        "safe_carry_success": carry and not fall,
        "safe_kick_success": kick and not fall,
    }
