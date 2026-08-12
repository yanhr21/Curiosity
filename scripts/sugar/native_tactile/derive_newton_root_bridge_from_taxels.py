#!/usr/bin/env python3
"""Recover Newton's SUGAR root trajectory from saved IsaacLab taxel world poses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def _matrix_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quaternion = np.asarray(
                ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale)
            )
        elif axis == 1:
            scale = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quaternion = np.asarray(
                ((matrix[0, 2] - matrix[2, 0]) / scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale)
            )
        else:
            scale = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quaternion = np.asarray(
                ((matrix[1, 0] - matrix[0, 1]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    residual = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
    return rotation, translation, residual


def _physical_taxel_centers_hand(
    asset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ray-sample the exact exported custom collision fronts at the official grid."""

    with np.load(asset_path.resolve(), allow_pickle=False) as archive:
        names = np.asarray(archive["patch_names"]).astype(str)
        sides = np.asarray(archive["sides"]).astype(str)
        sizes = np.asarray(archive["patch_size_m"], dtype=np.float64)
        origins = np.asarray(archive["patch_frame_origin_hand_m"], dtype=np.float64)
        rotations = np.asarray(archive["patch_frame_rotation_hand"], dtype=np.float64)
        vertices = np.asarray(archive["vertices_hand_m"], dtype=np.float64)
        vertex_offsets = np.asarray(archive["vertex_offsets"], dtype=np.int64)
        triangles = np.asarray(archive["triangles"], dtype=np.int64)
        triangle_offsets = np.asarray(archive["triangle_offsets"], dtype=np.int64)
    centers = np.full((len(names), 3), np.nan, dtype=np.float64)
    for index, name in enumerate(names):
        if name == "palm_r1_c1":
            continue
        local_vertices = (
            vertices[vertex_offsets[index] : vertex_offsets[index + 1]] - origins[index]
        ) @ rotations[index]
        local_triangles = triangles[triangle_offsets[index] : triangle_offsets[index + 1]]
        mesh = trimesh.Trimesh(local_vertices, local_triangles, process=False)
        divisions = np.asarray((20, 25), dtype=np.int64)
        division_size = (sizes[index] - 2.0 * 0.0004) / (divisions + 1)
        pitch = float(np.min(division_size))
        x = np.linspace(-0.5 * pitch * 21, 0.5 * pitch * 21, 22)[1:-1]
        z = np.linspace(-0.5 * pitch * 26, 0.5 * pitch * 26, 27)[1:-1]
        x_values, z_values = np.meshgrid(x, z, indexing="ij")
        rays = np.column_stack((x_values.reshape(-1), np.full(500, -1.0), z_values.reshape(-1)))
        directions = np.zeros_like(rays)
        directions[:, 1] = 1.0
        _, ray_ids, locations = trimesh.ray.ray_triangle.RayMeshIntersector(mesh).intersects_id(
            rays,
            directions,
            return_locations=True,
            multiple_hits=False,
        )
        order = np.argsort(ray_ids)
        if not np.array_equal(ray_ids[order], np.arange(500)):
            raise RuntimeError(f"Physical taxel sampling failed for {sides[index]}/{name}.")
        local_center = locations[order].mean(axis=0)
        centers[index] = origins[index] + rotations[index] @ local_center
    return names, sides, origins, centers


def _angular_velocity_world(rotations: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.zeros((len(rotations), 3), dtype=np.float64)
    for index in range(len(rotations) - 1):
        relative = rotations[index + 1] @ rotations[index].T
        angle = float(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
        if angle > 1.0e-10:
            axis = np.asarray(
                (relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1])
            ) / (2.0 * np.sin(angle))
            velocity[index] = axis * angle / dt
    if len(rotations) > 1:
        velocity[-1] = velocity[-2]
    return velocity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bridge", type=Path, required=True)
    parser.add_argument("--newton-trace", type=Path, required=True)
    parser.add_argument("--isaaclab-trace", type=Path, required=True)
    parser.add_argument("--anatomical-patch-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.base_bridge.resolve(), allow_pickle=False) as archive:
        base = {key: np.asarray(archive[key]).copy() for key in archive.files}
    with np.load(args.newton_trace.resolve(), allow_pickle=False) as archive:
        source_frames = np.asarray(archive["source_frame"], dtype=np.int64)
        newton_positions = np.asarray(archive["taxel_position_w_m"], dtype=np.float64)
        patch_names = np.asarray(archive["patch_names"]).astype(str)
    with np.load(args.isaaclab_trace.resolve(), allow_pickle=False) as archive:
        isaac_positions = np.asarray(archive["taxel_position_w"][source_frames], dtype=np.float64)

    newton_centers = newton_positions.mean(axis=(2, 3))
    isaac_centers = isaac_positions.reshape(len(source_frames), 54, 20, 25, 3).mean(axis=(2, 3))
    asset_names, asset_sides, patch_origins_hand, physical_centers_hand = (
        _physical_taxel_centers_hand(args.anatomical_patch_asset)
    )
    if not np.array_equal(np.asarray([name.split("_", 1)[1] for name in patch_names]), asset_names):
        raise RuntimeError("Newton trace and anatomical asset patch order disagree.")
    retained = np.asarray([not name.endswith("palm_r1_c1") for name in patch_names])
    if retained.sum() != 52:
        raise RuntimeError("Expected to exclude exactly the two R15 center-palm patches.")

    root_state = np.asarray(base["robot_root_state_w"], dtype=np.float64).copy()
    corrected_rotations: list[np.ndarray] = []
    residual_rows: list[np.ndarray] = []
    correction_translation_rows: list[np.ndarray] = []
    correction_rotation_rows: list[np.ndarray] = []
    hand_frame_residual_rows: list[np.ndarray] = []
    for row, source_frame in enumerate(source_frames):
        physical_centers_world = np.full((54, 3), np.nan, dtype=np.float64)
        hand_frame_residuals: list[np.ndarray] = []
        for side in ("left", "right"):
            side_mask = (asset_sides == side) & retained
            hand_rotation, hand_translation, hand_residual = _rigid_alignment(
                patch_origins_hand[side_mask],
                newton_centers[row, side_mask],
            )
            physical_centers_world[side_mask] = (
                hand_rotation @ physical_centers_hand[side_mask].T
            ).T + hand_translation
            hand_frame_residuals.append(hand_residual)
        correction_rotation, correction_translation, residual = _rigid_alignment(
            physical_centers_world[retained],
            isaac_centers[row, retained],
        )
        source_rotation = _matrix_from_wxyz(root_state[source_frame, 3:7])
        corrected_rotation = correction_rotation @ source_rotation
        root_state[source_frame, :3] = correction_rotation @ root_state[source_frame, :3] + correction_translation
        root_state[source_frame, 3:7] = _wxyz_from_matrix(corrected_rotation)
        corrected_rotations.append(corrected_rotation)
        residual_rows.append(residual)
        correction_translation_rows.append(correction_translation)
        correction_rotation_rows.append(correction_rotation)
        hand_frame_residual_rows.append(np.concatenate(hand_frame_residuals))

    dt = 1.0 / 50.0
    root_velocity = np.asarray(base["robot_root_velocity_w"], dtype=np.float64).copy()
    corrected_positions = root_state[source_frames, :3]
    linear_velocity = np.gradient(corrected_positions, dt, axis=0, edge_order=1)
    angular_velocity = _angular_velocity_world(np.stack(corrected_rotations), dt)
    root_velocity[source_frames, :3] = linear_velocity
    root_velocity[source_frames, 3:6] = angular_velocity
    base["robot_root_state_w"] = root_state.astype(np.float32)
    base["robot_root_velocity_w"] = root_velocity.astype(np.float32)
    base["root_alignment_source_frame"] = source_frames.astype(np.int32)
    base["root_alignment_residual_m"] = np.stack(residual_rows).astype(np.float32)
    base["root_alignment_correction_translation_m"] = np.stack(correction_translation_rows).astype(np.float32)
    base["root_alignment_correction_rotation"] = np.stack(correction_rotation_rows).astype(np.float32)
    base["root_alignment_hand_frame_residual_m"] = np.stack(hand_frame_residual_rows).astype(np.float32)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **base)
    residuals = np.stack(residual_rows)
    print(
        {
            "output": str(output),
            "source_frame_interval": [int(source_frames[0]), int(source_frames[-1]) + 1],
            "retained_patch_centers": int(retained.sum()),
            "median_alignment_residual_mm": float(np.median(residuals) * 1000.0),
            "maximum_alignment_residual_mm": float(np.max(residuals) * 1000.0),
            "maximum_hand_frame_reconstruction_residual_mm": float(
                np.max(np.stack(hand_frame_residual_rows)) * 1000.0
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
