#!/usr/bin/env python3
"""Audit one native anatomical whole-hand SUGAR CarryBox trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PATCHES = (
    *(f"palm_r{row}_c{column}" for row in range(4) for column in range(3)),
    *(
        f"{digit}_{segment}"
        for digit in ("thumb", "index", "middle", "ring", "little")
        for segment in ("proximal", "middle", "distal")
    ),
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def quat_apply_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[..., 1:]
    t = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[..., :1] * t + np.cross(xyz, t)


def longest_true_run(mask: np.ndarray) -> list[int]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        index = int(raw)
        if index != previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    best = max(runs, key=lambda item: item[1] - item[0])
    return [best[0], best[1], best[1] - best[0] + 1]


def main() -> None:
    run_root = args.run_root.resolve()
    trace_path = run_root / "whole_hand_trace.npz"
    summary_path = run_root / "summary.json"
    world_path = run_root / "world_carrybox.mp4"
    if args.output.exists():
        raise FileExistsError(f"Refusing overwrite: {args.output}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}

    normal = arrays["normal_force"].astype(np.float64)
    shear = arrays["signed_shear"].astype(np.float64)
    penetration = arrays["penetration"].astype(np.float64)
    position = arrays["taxel_position_w"].astype(np.float64)
    quaternion = arrays["taxel_quaternion_w"].astype(np.float64)
    optical_rgb = arrays["optical_rgb"]
    optical_depth = arrays["optical_depth"].astype(np.float64)
    optical_baseline_rgb = arrays["optical_baseline_rgb"]
    optical_baseline_depth = arrays["optical_baseline_depth"].astype(np.float64)
    object_state = arrays["object_state_w"].astype(np.float64)
    object_velocity = arrays["object_velocity_w"].astype(np.float64)
    patch_force = arrays["patch_box_force_w"].astype(np.float64)
    patch_friction = arrays["patch_box_friction_force_w"].astype(np.float64)
    robot_force = arrays["robot_box_force_w"].astype(np.float64)
    robot_friction = arrays["robot_box_friction_force_w"].astype(np.float64)
    body_names = arrays["robot_box_force_body_names"].astype(str)
    time_steps = arrays["source_step"]
    frames = len(normal)
    dt = float(arrays["control_dt_s"])
    gravity = arrays["gravity_w"].astype(np.float64)
    mass = float(summary["box_mass_readback_kg"])

    capture = cv2.VideoCapture(str(world_path))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()

    expected_shapes = {
        "normal_force": (frames, 2, 27, 20, 25),
        "signed_shear": (frames, 2, 27, 20, 25, 2),
        "penetration": (frames, 2, 27, 20, 25),
        "taxel_position_w": (frames, 2, 27, 20, 25, 3),
        "taxel_quaternion_w": (frames, 2, 27, 20, 25, 4),
        "optical_rgb": (frames, 2, 320, 240, 3),
        "optical_depth": (frames, 2, 320, 240, 1),
        "optical_baseline_rgb": (2, 320, 240, 3),
        "optical_baseline_depth": (2, 320, 240, 1),
        "patch_box_force_w": (frames, 2, 27, 3),
        "patch_box_friction_force_w": (frames, 2, 27, 3),
    }
    shape_checks = {
        key: tuple(arrays[key].shape) == shape
        for key, shape in expected_shapes.items()
    }
    shape_checks["robot_box_force_w"] = (
        robot_force.ndim == 3
        and robot_force.shape[0] == frames
        and robot_force.shape[2] == 3
        and robot_force.shape[1] == len(body_names)
    )
    shape_checks["robot_box_friction_force_w"] = (
        robot_friction.shape == robot_force.shape
    )

    contact = penetration > 0.0
    normal_nonzero = normal != 0.0
    shear_nonzero = np.any(shear != 0.0, axis=-1)
    no_contact = ~contact
    active_patch = np.any(contact, axis=(-2, -1))
    bilateral = np.all(np.any(active_patch, axis=-1), axis=-1)
    lift = object_state[:, 2] - object_state[0, 2]
    lifted = lift >= 0.20

    local_force = np.concatenate((shear, normal[..., None]), axis=-1)
    taxel_force_on_sensor_w = quat_apply_wxyz(quaternion, local_force)
    tactile_force_on_object_w = -taxel_force_on_sensor_w.sum(
        axis=(1, 2, 3, 4)
    )
    patch_normal_force_on_object_w = -patch_force.sum(axis=(1, 2))
    patch_total_force_on_object_w = -(patch_force + patch_friction).sum(
        axis=(1, 2)
    )
    robot_normal_force_on_object_w = -robot_force.sum(axis=1)
    robot_total_force_on_object_w = -(robot_force + robot_friction).sum(axis=1)
    acceleration = np.full((frames, 3), np.nan, dtype=np.float64)
    if frames >= 3:
        acceleration[1:-1] = (
            object_velocity[2:, :3] - object_velocity[:-2, :3]
        ) / (2.0 * dt)
    required_contact_force_w = mass * (acceleration - gravity[None])
    weight = mass * float(np.linalg.norm(gravity))
    valid_dynamic = np.isfinite(acceleration).all(axis=-1) & lifted

    def residual(force: np.ndarray) -> np.ndarray:
        result = np.full(frames, np.nan, dtype=np.float64)
        result[valid_dynamic] = np.linalg.norm(
            force[valid_dynamic] - required_contact_force_w[valid_dynamic],
            axis=-1,
        ) / weight
        return result

    tactile_residual = residual(tactile_force_on_object_w)
    tactile_sign_reversed_residual = residual(-tactile_force_on_object_w)
    patch_normal_residual = residual(patch_normal_force_on_object_w)
    patch_total_residual = residual(patch_total_force_on_object_w)
    robot_normal_residual = residual(robot_normal_force_on_object_w)
    robot_total_residual = residual(robot_total_force_on_object_w)

    per_body_impulse = np.linalg.norm(
        robot_force + robot_friction, axis=-1
    ).sum(axis=0)
    body_order = np.argsort(per_body_impulse)[::-1]
    body_support = [
        {
            "body": str(body_names[index]),
            "integrated_force_norm_n_frames": float(per_body_impulse[index]),
            "is_anatomical_patch": "_anatomical_" in str(body_names[index]),
        }
        for index in body_order[:20]
        if per_body_impulse[index] > 0.0
    ]

    patch_activity = {}
    for hand_index, hand in enumerate(("left", "right")):
        patch_activity[hand] = {
            PATCHES[index]: int(active_patch[:, hand_index, index].sum())
            for index in range(27)
        }

    rgb_delta = np.abs(
        optical_rgb.astype(np.int16)
        - optical_baseline_rgb[None].astype(np.int16)
    )
    depth_delta = np.abs(optical_depth - optical_baseline_depth[None])

    checks = {
        "all_required_shapes_exact": all(shape_checks.values()),
        "patch_order_exact": tuple(arrays["patch_order"].astype(str)) == PATCHES,
        "source_steps_contiguous": np.array_equal(
            time_steps, np.arange(frames, dtype=time_steps.dtype)
        ),
        "world_video_frame_count_exact": video_frames == frames,
        "world_video_fully_decodes": decoded == frames,
        "all_arrays_finite": all(
            np.isfinite(value).all()
            for value in (
                normal,
                shear,
                penetration,
                position,
                quaternion,
                optical_depth,
                optical_baseline_depth,
                object_state,
                object_velocity,
                patch_force,
                patch_friction,
                robot_force,
                robot_friction,
            )
        ),
        "no_contact_normal_exact_zero": not np.any(normal[no_contact]),
        "no_contact_shear_exact_zero": not np.any(shear[no_contact]),
        "contact_and_normal_support_identical": np.array_equal(
            contact, normal_nonzero
        ),
        "shear_only_where_contact": not np.any(shear_nonzero & no_contact),
        "bilateral_contact_exists": bool(np.any(bilateral)),
        "lifted_bilateral_contact_exists": bool(np.any(lifted & bilateral)),
        "optical_rgb_nonblank": bool(np.std(optical_rgb) > 0.0),
        "optical_baseline_rgb_nonblank": bool(np.std(optical_baseline_rgb) > 0.0),
        "optical_depth_finite": bool(np.isfinite(optical_depth).all()),
        "robot_contact_audit_nonzero_while_lifted": bool(
            np.any(
                np.linalg.norm(robot_total_force_on_object_w[lifted], axis=-1)
                > 0
            )
        ),
    }

    def distribution(values: np.ndarray) -> dict[str, float | int | None]:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return {"count": 0, "median": None, "p95": None, "maximum": None}
        return {
            "count": int(len(finite)),
            "median": float(np.median(finite)),
            "p95": float(np.quantile(finite, 0.95)),
            "maximum": float(np.max(finite)),
        }

    payload = {
        "schema": "sugar_whole_hand_carrybox_native_audit_v1",
        "run_root": str(run_root),
        "scenario": summary["scenario"],
        "checks": checks,
        "structural_passed": all(checks.values()),
        "shape_checks": shape_checks,
        "frames": frames,
        "mass_readback_kg": mass,
        "weight_n": weight,
        "maximum_lift_m": float(np.max(lift)),
        "bilateral_contact_frames": int(bilateral.sum()),
        "lifted_bilateral_frames": int((lifted & bilateral).sum()),
        "longest_lifted_bilateral_run": longest_true_run(lifted & bilateral),
        "normal_sign_counts": {
            "negative": int(np.count_nonzero(normal < 0.0)),
            "zero": int(np.count_nonzero(normal == 0.0)),
            "positive": int(np.count_nonzero(normal > 0.0)),
        },
        "patch_activity_frames": patch_activity,
        "top_robot_contact_bodies": body_support,
        "dynamic_force_residual_over_weight": {
            "tacsl_reaction_on_object": distribution(tactile_residual),
            "tacsl_sign_reversed_hypothesis": distribution(
                tactile_sign_reversed_residual
            ),
            "physx_anatomical_patches_normal_only": distribution(
                patch_normal_residual
            ),
            "physx_anatomical_patches_normal_plus_friction": distribution(
                patch_total_residual
            ),
            "physx_all_robot_bodies_normal_only": distribution(
                robot_normal_residual
            ),
            "physx_all_robot_bodies_normal_plus_friction": distribution(
                robot_total_residual
            ),
        },
        "lifted_dynamic_vertical_force_n": {
            "required": distribution(required_contact_force_w[:, 2][valid_dynamic]),
            "tacsl_reaction_on_object": distribution(
                tactile_force_on_object_w[:, 2][valid_dynamic]
            ),
            "tacsl_sign_reversed_hypothesis": distribution(
                -tactile_force_on_object_w[:, 2][valid_dynamic]
            ),
            "physx_normal_only": distribution(
                robot_normal_force_on_object_w[:, 2][valid_dynamic]
            ),
            "physx_normal_plus_friction": distribution(
                robot_total_force_on_object_w[:, 2][valid_dynamic]
            ),
        },
        "optical_response": {
            "baseline_source": "archived official no-contact get_initial_render",
            "maximum_rgb_absolute_delta": int(rgb_delta.max()),
            "maximum_depth_absolute_delta_m": float(depth_delta.max()),
        },
        "claim_boundary": (
            "Native sensor and audit reconstruction only. TacSL force balance, "
            "coverage, success/failure semantics, and human review are reported "
            "separately; structural_passed is not whole-hand admission."
        ),
        "force_direction_convention": (
            "TacSL SDF penalty and PhysX ContactSensor vectors act on the "
            "sensor/elastomer; the reported reaction on the CarryBox is their "
            "negative. The opposite TacSL sign is also archived as an explicit "
            "diagnostic hypothesis and cannot silently replace that convention."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
