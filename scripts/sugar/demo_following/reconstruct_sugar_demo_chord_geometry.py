#!/usr/bin/env python3
"""Reconstruct CHORD-style contact geometry for exact SUGAR demonstrations.

This uses NVIDIA CHORD's released ``approximate_contact_with_id`` implementation
and its released 1 cm threshold.  SUGAR provides exact retargeted G1 link poses,
object poses, and collision assets but no per-frame contact points.  We therefore
sample the known collision surfaces once and apply the official nearest-surface
contact rule at every demonstration frame.

The result is kinematic mesh-proximity reconstruction.  It is neither measured
force nor online PhysX contact.  SUGAR's binary contact label is read only after
reconstruction as an independent timing check; it never enters the geometry.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_CHORD_COMMIT = "5654c50edc1f3dea8e3145bf2dbfc277dbf27b4c"
OFFICIAL_THRESHOLD_M = 0.01
SURFACE_POINTS = 4096
SURFACE_SEED = 26082601

# IsaacLab's articulation order, verified against the live 35-body G1 asset.
G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "pelvis_contour_link",
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
    "head_link",
    "left_shoulder_pitch_link",
    "logo_link",
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

ROLE_LINKS = {
    "CarryBox": ("left_rubber_hand", "right_rubber_hand"),
    "KickBox": ("left_ankle_roll_link", "right_ankle_roll_link"),
}

TASK_OBJECT_USD = {
    "CarryBox": ROOT / "SUGAR/descriptions/objects/small_box/obj_aligned.usd",
    "KickBox": ROOT / "SUGAR/descriptions/objects/big_box/obj_aligned.usd",
}

FOOT_SPHERES = (
    (-0.05, 0.025, -0.03, 0.005),
    (-0.05, -0.025, -0.03, 0.005),
    (0.12, 0.03, -0.03, 0.005),
    (0.12, -0.03, -0.03, 0.005),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(ROLE_LINKS), required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--official-chord-root",
        type=Path,
        default=ROOT / "experiments/runtime_assets/official_chord_5654c50e",
    )
    parser.add_argument(
        "--object-usd",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--g1-mesh-dir",
        type=Path,
        default=ROOT / "SUGAR/descriptions/robots/g1/meshes",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _load_official_contact_utils(checkout: Path):
    source_root = checkout / "robotic_grounding/source/robotic_grounding"
    source = (
        checkout
        / "reconstruction/modules/v2d_task_library_loader/lib/contact_utils.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"official CHORD contact utility missing: {source}")
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location("official_chord_contact_utils", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official CHORD utility: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts.tolist():
        polygon = indices[cursor : cursor + count]
        cursor += count
        for offset in range(1, count - 1):
            faces.append((int(polygon[0]), int(polygon[offset]), int(polygon[offset + 1])))
    return np.asarray(faces, dtype=np.int64)


def _load_usd_mesh(path: Path) -> trimesh.Trimesh:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"cannot open object USD: {path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        transform = cache.GetLocalToWorldTransform(prim)
        world = np.asarray(
            [
                transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                for point in points
            ],
            dtype=np.float64,
        )
        triangles = _triangulate(np.asarray(counts), np.asarray(indices))
        vertices.append(world)
        faces.append(triangles + vertex_offset)
        vertex_offset += world.shape[0]
    if not vertices:
        raise RuntimeError(f"object USD contains no triangle mesh: {path}")
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices),
        faces=np.concatenate(faces),
        process=False,
    )


def _sample_mesh_surface(
    mesh: trimesh.Trimesh, *, count: int, seed: int, inward: bool
) -> tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_ids = trimesh.sample.sample_surface_even(mesh, count)
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True).clip(1.0e-8)
    if inward:
        normals = -normals
    return np.asarray(points, dtype=np.float32), normals


def _fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.stack((radius * np.cos(angle), radius * np.sin(angle), z), axis=-1)


def _role_surface(
    task: str, side: str, mesh_dir: Path
) -> tuple[np.ndarray, np.ndarray, str]:
    if task == "CarryBox":
        mesh_path = mesh_dir / f"{side}_rubber_hand.STL"
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        points, normals = _sample_mesh_surface(
            mesh, count=SURFACE_POINTS, seed=SURFACE_SEED + (side == "right"), inward=False
        )
        return points, normals, str(mesh_path)

    unit = _fibonacci_sphere(SURFACE_POINTS // len(FOOT_SPHERES))
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for x, y, z, radius in FOOT_SPHERES:
        points.append(unit * radius + np.asarray((x, y, z)))
        normals.append(unit)
    return (
        np.concatenate(points).astype(np.float32),
        np.concatenate(normals).astype(np.float32),
        "URDF ankle-roll collision: four radius-0.005m spheres",
    )


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q).clip(1.0e-12)
    w, x, y, z = q
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _transform(
    points: torch.Tensor,
    normals: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return points @ rotation.T + translation, normals @ rotation.T


def _minimum_surface_distance(object_points: torch.Tensor, role_points: torch.Tensor) -> float:
    minimum = torch.tensor(float("inf"), device=object_points.device)
    for chunk in object_points.split(512):
        minimum = torch.minimum(minimum, torch.cdist(chunk, role_points).amin())
    return float(minimum.item())


def _binary_metrics(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, float | int]:
    true_positive = int(np.logical_and(reference, reconstructed).sum())
    false_positive = int(np.logical_and(~reference, reconstructed).sum())
    false_negative = int(np.logical_and(reference, ~reconstructed).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    union = int(np.logical_or(reference, reconstructed).sum())
    return {
        "reference_active_frames": int(reference.sum()),
        "reconstructed_active_frames": int(reconstructed.sum()),
        "true_positive_frames": true_positive,
        "false_positive_frames": false_positive,
        "false_negative_frames": false_negative,
        "precision": precision,
        "recall": recall,
        "intersection_over_union": true_positive / max(1, union),
    }


def main() -> None:
    args = _arguments()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    object_usd = args.object_usd or TASK_OBJECT_USD[args.task]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    contact_utils, official_source = _load_official_contact_utils(args.official_chord_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    robot_path = args.motion_dir / "robot_50hz.npz"
    object_path = args.motion_dir / "obj_motion_global_50hz.pkl"
    label_path = args.motion_dir / "contact_labels_50hz.npy"
    with np.load(robot_path, allow_pickle=False) as archive:
        robot = {name: archive[name] for name in archive.files}
    with object_path.open("rb") as handle:
        object_motion = pickle.load(handle)
    labels = np.asarray(np.load(label_path), dtype=bool)
    frame_count = min(
        int(robot["body_pos_w"].shape[0]),
        int(object_motion["obj_trans"].shape[0]),
        int(labels.shape[0]),
    )
    if robot["body_pos_w"].shape[1] != len(G1_BODY_NAMES):
        raise RuntimeError("SUGAR motion no longer contains the exact 35-body G1 order")

    object_mesh = _load_usd_mesh(object_usd)
    object_points_local, object_normals_local = _sample_mesh_surface(
        object_mesh, count=SURFACE_POINTS, seed=SURFACE_SEED, inward=True
    )
    object_scale = float(object_motion["obj_scale"])
    object_points_local *= object_scale
    object_points_t = torch.as_tensor(object_points_local, device=device)
    object_normals_t = torch.as_tensor(object_normals_local, device=device)
    part_ids = torch.ones(object_points_t.shape[0], dtype=torch.long, device=device)

    role_data: list[tuple[torch.Tensor, torch.Tensor, str]] = []
    for side in ("left", "right"):
        points, normals, source = _role_surface(args.task, side, args.g1_mesh_dir)
        role_data.append(
            (
                torch.as_tensor(points, device=device),
                torch.as_tensor(normals, device=device),
                source,
            )
        )

    contact_active = np.zeros((frame_count, 2), dtype=bool)
    contact_count = np.zeros((frame_count, 2), dtype=np.int32)
    minimum_distance_m = np.full((frame_count, 2), np.nan, dtype=np.float32)
    mean_contact_distance_m = np.full((frame_count, 2), np.nan, dtype=np.float32)
    hand_position_w = np.full((frame_count, 2, 3), np.nan, dtype=np.float32)
    hand_normal_w = np.full((frame_count, 2, 3), np.nan, dtype=np.float32)
    object_position_w = np.full((frame_count, 2, 3), np.nan, dtype=np.float32)
    object_normal_w = np.full((frame_count, 2, 3), np.nan, dtype=np.float32)

    link_indices = [G1_BODY_NAMES.index(name) for name in ROLE_LINKS[args.task]]
    with torch.inference_mode():
        for frame in range(frame_count):
            object_rotation = torch.as_tensor(
                object_motion["obj_rot"][frame], dtype=torch.float32, device=device
            )
            object_translation = torch.as_tensor(
                object_motion["obj_trans"][frame], dtype=torch.float32, device=device
            )
            object_points_w, object_normals_w = _transform(
                object_points_t, object_normals_t, object_rotation, object_translation
            )
            for side, (role_points_t, role_normals_t, _) in enumerate(role_data):
                link = link_indices[side]
                link_rotation = torch.as_tensor(
                    _quat_wxyz_to_matrix(robot["body_quat_w"][frame, link]),
                    device=device,
                )
                link_translation = torch.as_tensor(
                    robot["body_pos_w"][frame, link], device=device
                )
                role_points_w, role_normals_w = _transform(
                    role_points_t, role_normals_t, link_rotation, link_translation
                )
                minimum = _minimum_surface_distance(object_points_w, role_points_w)
                minimum_distance_m[frame, side] = minimum
                if minimum >= OFFICIAL_THRESHOLD_M:
                    continue
                (
                    object_contacts,
                    object_contact_normals,
                    _,
                    hand_contacts,
                    hand_contact_normals,
                    contact_distances,
                ) = contact_utils.approximate_contact_with_id(
                    object_points_w,
                    object_normals_w,
                    part_ids,
                    role_points_w,
                    role_normals_w,
                    threshold=OFFICIAL_THRESHOLD_M,
                )
                count = int(contact_distances.numel())
                if count == 0:
                    raise RuntimeError("official contact function disagrees with exact minimum")
                weights = torch.softmax(-contact_distances, dim=0)
                contact_active[frame, side] = True
                contact_count[frame, side] = count
                mean_contact_distance_m[frame, side] = float(contact_distances.mean())
                hand_position_w[frame, side] = (hand_contacts * weights[:, None]).sum(0).cpu()
                hand_normal_w[frame, side] = (hand_contact_normals * weights[:, None]).sum(0).cpu()
                object_position_w[frame, side] = (object_contacts * weights[:, None]).sum(0).cpu()
                object_normal_w[frame, side] = (
                    object_contact_normals * weights[:, None]
                ).sum(0).cpu()

    reconstructed_any = contact_active.any(axis=1)
    label_metrics = _binary_metrics(labels[:frame_count], reconstructed_any)
    finite_contact_geometry = bool(
        np.isfinite(hand_position_w[contact_active]).all()
        and np.isfinite(hand_normal_w[contact_active]).all()
        and np.isfinite(object_position_w[contact_active]).all()
        and np.isfinite(object_normal_w[contact_active]).all()
    )
    independent_overlap = label_metrics["true_positive_frames"] > 0
    passed = bool(finite_contact_geometry and contact_active.any() and independent_overlap)

    np.savez_compressed(
        args.output_dir / "contact_geometry.npz",
        frame=np.arange(frame_count, dtype=np.int32),
        role_names=np.asarray(ROLE_LINKS[args.task]),
        contact_active=contact_active,
        contact_count=contact_count,
        minimum_distance_m=minimum_distance_m,
        mean_contact_distance_m=mean_contact_distance_m,
        hand_contact_position_w=hand_position_w,
        hand_contact_normal_w=hand_normal_w,
        object_contact_position_w=object_position_w,
        object_contact_normal_w=object_normal_w,
        object_contact_part_id=np.where(contact_active, 1, 0).astype(np.int16),
        sugar_binary_contact_label=labels[:frame_count],
        robot_body_pos_w=robot["body_pos_w"][:frame_count],
        robot_body_quat_w=robot["body_quat_w"][:frame_count],
        object_position_w=np.asarray(object_motion["obj_trans"][:frame_count]),
        object_rotation_w=np.asarray(object_motion["obj_rot"][:frame_count]),
    )
    result = {
        "passed": passed,
        "task": args.task,
        "motion_dir": str(args.motion_dir.resolve()),
        "frames": frame_count,
        "representation": "kinematic G1/object mesh-proximity reconstruction",
        "not_claimed": [
            "measured human contact",
            "measured force",
            "online PhysX contact",
            "CHORD policy training",
        ],
        "official_chord": {
            "commit": OFFICIAL_CHORD_COMMIT,
            "source": str(official_source.resolve()),
            "function": "approximate_contact_with_id",
            "threshold_m": OFFICIAL_THRESHOLD_M,
            "object_surface_points": int(object_points_t.shape[0]),
        },
        "sources": {
            "robot_pose": str(robot_path.resolve()),
            "object_pose": str(object_path.resolve()),
            "object_collision_mesh": str(object_usd.resolve()),
            "role_collision_surfaces": [item[2] for item in role_data],
            "binary_label_used_only_for_independent_validation": str(label_path.resolve()),
        },
        "contact": {
            "left_frames": int(contact_active[:, 0].sum()),
            "right_frames": int(contact_active[:, 1].sum()),
            "bilateral_frames": int(contact_active.all(axis=1).sum()),
            "any_frames": int(reconstructed_any.sum()),
            "global_minimum_distance_m": float(np.nanmin(minimum_distance_m)),
        },
        "independent_binary_timing_check": label_metrics,
        "checks": {
            "geometry_finite_on_active_frames": finite_contact_geometry,
            "official_threshold_produces_contact": bool(contact_active.any()),
            "reconstruction_overlaps_original_binary_timing": independent_overlap,
            "binary_label_was_not_an_input_to_geometry": True,
        },
    }
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    simulation_app.close()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
