"""Yaw-frame identifiability diagnostic, using actual PhysX future records.

This is a geometry counterexample outside the fixed recorded reset poses, not
an observed duplicate-input pair or a proof of the unique model failure cause.
"""
from __future__ import annotations

import json
import numpy as np

from .data import ROOT, OUTPUT, PARENT, CORPUS, TASKS, actual_entries, actual_future, mismatch, write_json
from build_actual_contact_event_predictor_dataset import quaternion_wxyz_to_rotation6d


def rotation6d_matrix(value):
    # Dataset convention is first and second matrix columns, not actor tangent/normal.
    x, y = value[..., :3], value[..., 3:6]
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def main():
    yaw = np.pi / 2
    rotation = np.asarray([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], dtype=np.float32)
    entries = {(TASKS.index(e["task"]), e["source_id"]): e for e in actual_entries(CORPUS)}
    records = []
    with np.load(PARENT / "NORMALIZATION.npz", allow_pickle=False) as z:
        scale = z["target_scale"]
    for split in ("validation", "test"):
        with np.load(OUTPUT / split / "routing.npz", allow_pickle=False) as z:
            route = {k: z[k] for k in z.files}
        bank = np.load(OUTPUT / split / "demo_bank.npy")
        for task in (0, 1):
            # Fixed first motion, beginning/middle/last valid anchor.
            source = int(np.min(route["base_source_motion_id"][route["base_task"] == task]))
            eligible = np.flatnonzero((route["base_task"] == task) & (route["base_source_motion_id"] == source))
            entry = entries[(task, source)]
            with np.load(entry["trace"], allow_pickle=False) as z:
                trace = {k: z[k] for k in ("object_root_state_w", "robot_body_position_w", "robot_root_state_w", "contact", "contact_event_remaining_frames", "motion_regime")}
            for row in eligible[[0, len(eligible) // 2, -1]]:
                anchor, env = int(route["base_anchor_frame"][row]), entry["env"]
                actual = actual_future(trace, env, anchor)
                transformed = {key: value.copy() for key, value in actual.items()}
                continuous = transformed["continuous"]
                continuous[:, :105] = (continuous[:, :105].reshape(10, 35, 3) @ rotation.T).reshape(10, 105)
                continuous[:, 105:108] = continuous[:, 105:108] @ rotation.T
                matrix = rotation6d_matrix(continuous[:, 108:114])
                matrix = rotation @ matrix
                continuous[:, 108:114] = np.swapaxes(matrix[..., :, :2], -1, -2).reshape(10, 6)
                continuous[:, 114:117] = continuous[:, 114:117] @ rotation.T
                continuous[:, 117:120] = continuous[:, 117:120] @ rotation.T
                alignment = int(np.rint(route["horizon_phase"][row, 0] * np.float32(31)))
                demo = bank[route["selected_demo"][row], alignment]
                original_error = mismatch(actual, demo)
                rotated_error = mismatch(transformed, demo)
                # Verify frame-relative pose and velocity invariance numerically
                # with actual root/object poses. The goal terms obey the same
                # exact relation when both task goal and robot are yaw-rotated.
                robot = trace["robot_root_state_w"][anchor, env]
                obj = trace["object_root_state_w"][anchor, env]
                robot_r = rotation6d_matrix(quaternion_wxyz_to_rotation6d(robot[3:7]))
                obj_r = rotation6d_matrix(quaternion_wxyz_to_rotation6d(obj[3:7]))
                relative_position = robot_r.T @ (obj[:3] - robot[:3])
                transformed_position = (rotation @ robot_r).T @ (rotation @ obj[:3] - rotation @ robot[:3])
                relative_rotation = robot_r.T @ obj_r
                transformed_rotation = (rotation @ robot_r).T @ (rotation @ obj_r)
                relative_velocity = robot_r.T @ obj[7:10]
                transformed_velocity = (rotation @ robot_r).T @ (rotation @ obj[7:10])
                max_invariance_error = max(float(np.max(np.abs(a - b))) for a, b in
                                           ((relative_position, transformed_position), (relative_rotation, transformed_rotation), (relative_velocity, transformed_velocity)))
                label_delta = np.abs(np.log1p(rotated_error / scale) - np.log1p(original_error / scale))
                records.append({"split": split, "task": TASKS[task], "source": source, "anchor": anchor,
                                "relative_geometry_max_absolute_change": max_invariance_error,
                                "fixed_demo_continuous_label_change_mean": float(label_delta[:, :4].mean()),
                                "fixed_demo_continuous_label_change_max": float(label_delta[:, :4].max()),
                                "contact_duration_regime_labels_unchanged": bool(np.array_equal(original_error[:, 4:], rotated_error[:, 4:]))})
    result = {"execution_completed": True, "optimizer_updates": 0, "yaw_degrees": 90,
              "records": records, "relative_geometry_invariance_passed": all(r["relative_geometry_max_absolute_change"] < 1e-5 for r in records),
              "fixed_demo_world_label_changed_every_case": all(r["fixed_demo_continuous_label_change_max"] > 0.01 for r in records),
              "input_contract_source": "SUGAR/source/sugar_rl/sugar_rl/utils/demo_event_reward_runtime.py:GOAL_POLICY_CORE_TERM_NAMES; goal_carry_mdp.py body-frame box and goal transforms",
              "input_invariance_derivation": "Joint state/action stay fixed; gravity and height are yaw-invariant; body-frame linear/angular velocity and relative box/goal poses are unchanged when robot, object, velocities and task goal rotate together. Fixed numeric selected demo and normalized phase are unchanged.",
              "limitation": "Numeric check uses recorded root-relative geometry, not reconstruction of all 121 runtime channels; full observation invariance follows the source formulas for any anchor frame. Rotated examples are constructed coordinate counterexamples outside recorded reset support, not additional physics rollouts or an empirical proof of the sole cause.",
              "interpretation": "World-frame mismatch requires a relation between reference and rollout frames. The current input API has no explicit transform specifying that relation. Audit/canonicalize it before treating future-loss gains as a deployable selected-demo metric."}
    write_json(OUTPUT.parent / "COORDINATE_CONTRACT_PROBE.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
