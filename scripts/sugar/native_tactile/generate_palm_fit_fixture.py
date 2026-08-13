#!/usr/bin/env python3
"""Build a rigid palm-contact fixture from one accepted full-G1 tactile frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return rotation_wxyz(np.asarray((w, x, y, z)))


def format_points(points: list[np.ndarray]) -> str:
    return ",\n            ".join(
        f"({point[0]:.9f}, {point[1]:.9f}, {point[2]:.9f})" for point in points
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=291)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--penetration-m", type=float, default=0.0005)
    parser.add_argument("--fixture-depth-m", type=float, default=0.018)
    parser.add_argument(
        "--collision-connector",
        action="store_true",
        help="Include the central beam in the one free rigid collision mesh.",
    )
    args = parser.parse_args()

    with np.load(args.trace) as trace:
        names = trace["patch_order"].astype(str)
        palm = np.flatnonzero(np.char.startswith(names, "palm_"))
        if len(palm) != 12:
            raise RuntimeError(f"Expected twelve palm patches, found {len(palm)}")
        object_states = trace["object_state_w"]
        reset_geometry = object_states.ndim == 1
        object_state = np.asarray(
            object_states if reset_geometry else object_states[args.frame], np.float64
        )
        object_rotation = rotation_wxyz(object_state[3:7])
        position_source = trace["taxel_position_w"]
        quaternion_source = trace["taxel_quaternion_w"]
        taxel_position = np.asarray(
            (position_source if reset_geometry else position_source[args.frame])[:, palm],
            np.float64,
        )
        taxel_quaternion = np.asarray(
            (quaternion_source if reset_geometry else quaternion_source[args.frame])[:, palm],
            np.float64,
        )
        patch_size = (
            np.asarray(trace["tactile_patch_size_m"][:, palm], np.float64)
            if "tactile_patch_size_m" in trace.files
            else None
        )

    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    palm_centers_object: list[list[np.ndarray]] = [[], []]
    box_faces = (
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3),
        (1, 2, 6), (1, 6, 5),
    )
    center_row, center_column = taxel_position.shape[-3] // 2, taxel_position.shape[-2] // 2
    for side in range(2):
        for local_patch in range(12):
            point_world = taxel_position[side, local_patch, center_row, center_column]
            quaternion_world = taxel_quaternion[
                side, local_patch, center_row, center_column
            ]
            patch_rotation_object = object_rotation.T @ rotation_xyzw(quaternion_world)
            tangent_x = patch_rotation_object[:, 0]
            tangent_y = patch_rotation_object[:, 1]
            toward_object = patch_rotation_object[:, 2]
            point_object = object_rotation.T @ (point_world - object_state[:3])
            palm_centers_object[side].append(point_object)
            if patch_size is None:
                surface_points = taxel_position[side, local_patch].reshape(-1, 3)
                surface_local = (surface_points - point_world) @ rotation_xyzw(
                    quaternion_world
                )
                rows, columns = taxel_position.shape[-3:-1]
                size_x = np.ptp(surface_local[:, 0]) * rows / (rows - 1)
                size_y = np.ptp(surface_local[:, 1]) * columns / (columns - 1)
            else:
                size_x, size_y = patch_size[side, local_patch]
            center = point_object + toward_object * (
                0.5 * args.fixture_depth_m - args.penetration_m
            )
            half_x = 0.47 * float(size_x)
            half_y = 0.47 * float(size_y)
            half_z = 0.5 * args.fixture_depth_m
            base = len(vertices)
            for sx, sy, sz in (
                (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
            ):
                vertices.append(
                    center
                    + sx * half_x * tangent_x
                    + sy * half_y * tangent_y
                    + sz * half_z * toward_object
                )
            triangles.extend(tuple(base + index for index in face) for face in box_faces)

    left_center = np.mean(palm_centers_object[0], axis=0)
    right_center = np.mean(palm_centers_object[1], axis=0)
    connector_axis = right_center - left_center
    center_distance = float(np.linalg.norm(connector_axis))
    connector_axis /= center_distance
    reference_axis = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(reference_axis, connector_axis))) > 0.9:
        reference_axis = np.asarray((0.0, 1.0, 0.0))
    connector_y = np.cross(connector_axis, reference_axis)
    connector_y /= np.linalg.norm(connector_y)
    connector_z = np.cross(connector_axis, connector_y)
    connector_center = 0.5 * (left_center + right_center)
    connector_half_length = max(0.5 * center_distance, 0.04)
    connector_vertices: list[np.ndarray] = []
    for sx, sy, sz in (
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ):
        connector_vertices.append(
            connector_center
            + sx * connector_half_length * connector_axis
            + sy * 0.022 * connector_y
            + sz * 0.022 * connector_z
        )
    if args.collision_connector:
        base = len(vertices)
        vertices.extend(connector_vertices)
        triangles.extend(tuple(base + index for index in face) for face in box_faces)
    connector_counts = ", ".join("3" for _ in box_faces)
    connector_indices = ", ".join(str(index) for face in box_faces for index in face)

    counts = ", ".join("3" for _ in triangles)
    indices = ", ".join(str(index) for face in triangles for index in face)
    output = f'''#usda 1.0
(
    defaultPrim = "PalmFitFixture"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "PalmFitFixture" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    bool physics:rigidBodyEnabled = 1
    float physics:mass = 0.3023375869

    def Mesh "fixture" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        bool physics:collisionEnabled = 1
        uniform token physics:approximation = "sdf"
        point3f[] points = [
            {format_points(vertices)}
        ]
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        uniform token subdivisionScheme = "none"
        uniform token orientation = "rightHanded"
        color3f[] primvars:displayColor = [(0.16, 0.48, 0.72)]
    }}

    def Mesh "visual_connector" ()
    {{
        point3f[] points = [
            {format_points(connector_vertices)}
        ]
        int[] faceVertexCounts = [{connector_counts}]
        int[] faceVertexIndices = [{connector_indices}]
        uniform token subdivisionScheme = "none"
        uniform token orientation = "rightHanded"
        color3f[] primvars:displayColor = [(0.95, 0.45, 0.08)]
    }}
}}
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    print(
        f"wrote {args.output} with {len(vertices)} vertices and "
        f"{len(triangles)} triangles"
    )


if __name__ == "__main__":
    main()
