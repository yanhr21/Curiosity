#!/usr/bin/env python3
"""Render one Plan-10 static feasibility scan as an audited H.264 review movie."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
import textwrap
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pyrender
import trimesh


HOST = socket.gethostname()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def pose_matrix(pose_wxyz: np.ndarray) -> np.ndarray:
    px, py, pz, w, x, y, z = (float(value) for value in pose_wxyz)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    result[:3, 3] = (px, py, pz)
    return result


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(np.asarray((0.0, 0.0, 1.0)), back)
    if np.linalg.norm(right) < 1.0e-6:
        right = np.asarray((1.0, 0.0, 0.0))
    else:
        right /= np.linalg.norm(right)
    up = np.cross(back, right)
    result = np.eye(4, dtype=np.float64)
    result[:3, 0] = right
    result[:3, 1] = up
    result[:3, 2] = back
    result[:3, 3] = eye
    return result


def body_color(name: str) -> tuple[float, float, float, float]:
    if name == "CarryBox":
        return (0.86, 0.48, 0.12, 1.0)
    if "hand_camera_base" in name:
        return (0.92, 0.08, 0.08, 1.0)
    if "wrist" in name:
        return (0.95, 0.72, 0.08, 1.0)
    if name.startswith("L_") or name == "left_hand_base_link":
        return (0.18, 0.60, 0.96, 1.0)
    if name.startswith("R_") or name == "right_hand_base_link":
        return (0.16, 0.82, 0.63, 1.0)
    if "head" in name or "d435" in name or "mid360" in name:
        return (0.15, 0.18, 0.22, 1.0)
    return (0.54, 0.58, 0.63, 1.0)


def add_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def box_outline_mesh(points_body: np.ndarray) -> trimesh.Trimesh:
    minimum = points_body.min(axis=0)
    maximum = points_body.max(axis=0)
    corners = np.asarray(
        [
            (x, y, z)
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    edges = [
        (start, stop)
        for start in range(8)
        for stop in range(start + 1, 8)
        if np.count_nonzero(corners[start] != corners[stop]) == 1
    ]
    cylinders = []
    for start, stop in edges:
        first, second = corners[start], corners[stop]
        direction = second - first
        transform = trimesh.geometry.align_vectors((0.0, 0.0, 1.0), direction)
        transform[:3, 3] = 0.5 * (first + second)
        cylinders.append(
            trimesh.creation.cylinder(
                radius=0.0035,
                height=float(np.linalg.norm(direction)),
                sections=10,
                transform=transform,
            )
        )
    return trimesh.util.concatenate(cylinders)


def telemetry_panel(replay: np.lib.npyio.NpzFile, index: int, label: str) -> np.ndarray:
    panel = np.full((270, 420, 3), (24, 28, 34), dtype=np.uint8)
    position_mm = replay["position_error_m"][index] * 1000.0
    rotation = replay["rotation_error_rad"][index]
    gap_mm = float(replay["head_gap_m"][index] * 1000.0)
    inside = int(replay["head_inside_vertices"][index])
    nonhand = float(replay["nonhand_load_n"][index])
    tilt = float(replay["tilt_deg"][index])
    box_delta = replay["box_translation_m"][index]
    root_delta = replay["root_delta_m"][index]
    active = str(replay["active_contact_bodies"][index])
    add_label(panel, label, (12, 23), 0.47, (230, 230, 230))
    add_label(panel, f"candidate {index + 1:02d}/{len(replay['tilt_deg']):02d}   tilt={tilt:+.0f} deg", (12, 48), 0.46)
    add_label(panel, f"hand pos L/R = {position_mm[0]:.2f} / {position_mm[1]:.2f} mm", (12, 73), 0.43)
    add_label(panel, f"hand rot L/R = {rotation[0]:.3f} / {rotation[1]:.3f} rad", (12, 96), 0.43)
    head_text = f"head gap={gap_mm:.1f} mm" if inside == 0 else f"head overlap vertices={inside}"
    add_label(panel, head_text, (12, 119), 0.43, (80, 220, 80) if inside == 0 else (70, 90, 255))
    add_label(panel, f"non-hand collision load={nonhand:.2f} N", (12, 142), 0.43, (70, 90, 255) if nonhand > 0.01 else (80, 220, 80))
    add_label(panel, f"box dY={box_delta[1]:+.2f} m   root dZ={root_delta[2]:+.2f} m", (12, 165), 0.40)
    add_label(panel, "active collision bodies:", (12, 189), 0.38)
    lines = textwrap.wrap(active if active else "none", width=52)[:3]
    for row, line in enumerate(lines):
        add_label(panel, line, (12, 209 + row * 17), 0.31, (90, 150, 255))
    add_label(panel, "STATIC ADMISSION: FAIL", (230, 260), 0.43, (60, 70, 255), 2)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--hold-frames", type=int, default=20)
    args = parser.parse_args()
    if HOST.startswith(("mgmtserver", "login")):
        raise SystemExit(f"Refusing EGL render on login node: {HOST}")
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("EGL render requires a retained allocation")
    replay_root = args.replay_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if args.fps <= 0 or args.hold_frames <= 0:
        raise ValueError("fps and hold-frames must be positive")
    output_root.mkdir(parents=True)
    replay_manifest_path = replay_root / "manifest.json"
    replay_manifest = json.loads(replay_manifest_path.read_text())
    if not replay_manifest["export_passed"]:
        raise RuntimeError("Refusing failed static replay export")
    geometry_manifest_path = Path(replay_manifest["geometry_manifest"])
    geometry_manifest = json.loads(geometry_manifest_path.read_text())
    geometry_path = Path(replay_manifest["geometry_npz"])
    replay_path = Path(replay_manifest["replay"])
    if sha256(geometry_path) != replay_manifest["geometry_npz_sha256"]:
        raise RuntimeError("Geometry hash mismatch")
    if sha256(replay_path) != replay_manifest["replay_sha256"]:
        raise RuntimeError("Replay hash mismatch")
    geometry = np.load(geometry_path, allow_pickle=False)
    replay = np.load(replay_path, allow_pickle=False)
    body_names = replay["body_names"].astype(str).tolist()
    label = replay_manifest["label"]

    scene = pyrender.Scene(
        bg_color=(0.055, 0.065, 0.078, 1.0), ambient_light=(0.32, 0.34, 0.38)
    )
    nodes: list[tuple[pyrender.Node, int, np.ndarray, str]] = []
    box_points_body = []
    for row in geometry_manifest["meshes"]:
        key = row["key"]
        tri = trimesh.Trimesh(
            vertices=geometry[f"{key}_vertices"],
            faces=geometry[f"{key}_faces"],
            process=False,
        )
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=body_color(row["owner"]),
            metallicFactor=0.05,
            roughnessFactor=0.70,
        )
        node = scene.add(
            pyrender.Mesh.from_trimesh(tri, material=material, smooth=False)
        )
        nodes.append(
            (
                node,
                int(row["body_index"]),
                geometry[f"{key}_body_from_mesh"].astype(np.float64),
                row["owner"],
            )
        )
        if row["owner"] == "CarryBox":
            vertices = geometry[f"{key}_vertices"].astype(np.float64)
            body_from_mesh = geometry[f"{key}_body_from_mesh"].astype(np.float64)
            box_points_body.append(
                vertices @ body_from_mesh[:3, :3].T + body_from_mesh[:3, 3]
            )
    outline_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(1.0, 0.58, 0.10, 1.0),
        emissiveFactor=(0.45, 0.18, 0.02),
        roughnessFactor=0.8,
    )
    outline_node = scene.add(
        pyrender.Mesh.from_trimesh(
            box_outline_mesh(np.concatenate(box_points_body)),
            material=outline_material,
            smooth=False,
        )
    )
    scene_center = np.median(replay["body_pose_wxyz"][:, -1, :3], axis=0)
    floor_pose = np.eye(4)
    floor_pose[:3, 3] = (scene_center[0], scene_center[1], -0.025)
    scene.add(
        pyrender.Mesh.from_trimesh(
            trimesh.creation.box(extents=(3.2, 3.2, 0.03)),
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(0.16, 0.18, 0.21, 1.0), roughnessFactor=0.92
            ),
            smooth=False,
        ),
        pose=floor_pose,
    )
    camera_node = scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(44.0)))
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.2),
        pose=look_at(scene_center + np.asarray((0.6, -1.7, 1.6)), scene_center),
    )
    scene.add(
        pyrender.DirectionalLight(color=(0.75, 0.82, 1.0), intensity=1.2),
        pose=look_at(scene_center + np.asarray((-1.4, 1.6, 0.9)), scene_center),
    )
    renderer = pyrender.OffscreenRenderer(1280, 420)
    video_path = output_root / "plan10_static_feasibility_human_review.mp4"
    middle_frame_path = output_root / "middle_frame.png"
    encoder = imageio_ffmpeg.write_frames(
        str(video_path),
        (1280, 720),
        fps=args.fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=7,
        output_params=["-movflags", "+faststart"],
    )
    encoder.send(None)
    hidden_pose = np.eye(4)
    hidden_pose[:3, 3] = (1000.0, 1000.0, 1000.0)
    left_ids = [
        index
        for index, name in enumerate(body_names)
        if name.startswith("L_") or name == "left_hand_base_link"
    ]
    right_ids = [
        index
        for index, name in enumerate(body_names)
        if name.startswith("R_") or name == "right_hand_base_link"
    ]
    candidate_hashes = []
    try:
        for index, poses in enumerate(replay["body_pose_wxyz"]):
            scene.set_pose(outline_node, hidden_pose)
            node_pose = {}
            for node, owner_index, body_from_mesh, _owner in nodes:
                current = pose_matrix(poses[owner_index]) @ body_from_mesh
                node_pose[node] = current
                scene.set_pose(node, current)
            pelvis = poses[body_names.index("pelvis"), :3]
            box = poses[-1, :3]
            target = 0.5 * (pelvis + box)
            scale = max(1.0, float(np.linalg.norm(pelvis - box)) / 0.9)
            scene.set_pose(
                camera_node,
                look_at(target + scale * np.asarray((1.35, -1.65, 0.75)), target),
            )
            renderer.viewport_width = 1280
            renderer.viewport_height = 420
            wide, _ = renderer.render(scene)
            for node, _owner_index, _body_from_mesh, owner in nodes:
                visible = (
                    owner.startswith(("L_", "R_"))
                    or owner.endswith("hand_base_link")
                    or "hand_camera_base" in owner
                    or "wrist" in owner
                )
                if not visible:
                    scene.set_pose(node, hidden_pose)
            scene.set_pose(outline_node, pose_matrix(poses[-1]))
            renderer.viewport_width = 420
            renderer.viewport_height = 270
            left_center = poses[left_ids, :3].mean(axis=0)
            right_center = poses[right_ids, :3].mean(axis=0)
            for node, _owner_index, _body_from_mesh, owner in nodes:
                right_specific = owner.startswith("R_") or owner.startswith("right_")
                if right_specific:
                    scene.set_pose(node, hidden_pose)
                elif owner.startswith("L_") or owner.startswith("left_"):
                    scene.set_pose(node, node_pose[node])
            scene.set_pose(
                camera_node,
                look_at(left_center + np.asarray((-0.40, 0.55, 0.55)), left_center),
            )
            left, _ = renderer.render(scene)
            for node, _owner_index, _body_from_mesh, owner in nodes:
                left_specific = owner.startswith("L_") or owner.startswith("left_")
                if left_specific:
                    scene.set_pose(node, hidden_pose)
                elif owner.startswith("R_") or owner.startswith("right_"):
                    scene.set_pose(node, node_pose[node])
            scene.set_pose(
                camera_node,
                look_at(right_center + np.asarray((0.40, 0.55, 0.55)), right_center),
            )
            right, _ = renderer.render(scene)
            panel = telemetry_panel(replay, index, label)
            frame = np.full((720, 1280, 3), (18, 21, 26), dtype=np.uint8)
            frame[:420] = wide
            frame[430:700, 0:420] = left
            frame[430:700, 430:850] = right
            frame[430:700, 860:1280] = panel
            add_label(frame, "WORLD / OFFICIAL G1-INSPIRE + CARRYBOX", (14, 26), 0.63)
            add_label(frame, "LEFT HAND", (12, 456), 0.52, (245, 154, 52))
            add_label(frame, "RIGHT HAND", (442, 456), 0.52, (98, 209, 162))
            add_label(frame, "RED=HAND CAMERA  YELLOW=WRIST", (12, 690), 0.34, (70, 150, 255))
            add_label(frame, "RED=HAND CAMERA  YELLOW=WRIST", (442, 690), 0.34, (70, 150, 255))
            candidate_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            if index == len(replay["body_pose_wxyz"]) // 2:
                if not cv2.imwrite(str(middle_frame_path), frame[:, :, ::-1]):
                    raise RuntimeError("Failed to write middle frame")
            for _ in range(args.hold_frames):
                encoder.send(np.ascontiguousarray(frame))
    finally:
        encoder.close()
        renderer.delete()
    checks = {
        "replay_export_passed": bool(replay_manifest["export_passed"]),
        "official_geometry_hash_exact": sha256(geometry_path)
        == replay_manifest["geometry_npz_sha256"],
        "all_scan_candidates_rendered": len(candidate_hashes)
        == replay_manifest["frame_count"],
        "all_candidate_frames_visually_distinct": len(set(candidate_hashes))
        == len(candidate_hashes),
        "middle_frame_written": middle_frame_path.is_file(),
        "video_written": video_path.is_file(),
        "negative_static_gate_preserved": not bool(replay["static_candidate"].any()),
    }
    payload = {
        "schema": "plan10_static_feasibility_h264_human_review_v1",
        "claim_boundary": "Official visual-geometry review of an audited no-learning static scan. Red/yellow geometry identifies the official camera/wrist topology; displayed loads remain rigid-contact diagnostics, not tactile output.",
        "host": HOST,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "label": label,
        "replay_manifest": str(replay_manifest_path),
        "replay_manifest_sha256": sha256(replay_manifest_path),
        "candidate_count": len(candidate_hashes),
        "hold_frames_per_candidate": args.hold_frames,
        "frame_count": len(candidate_hashes) * args.hold_frames,
        "fps": args.fps,
        "video": str(video_path),
        "video_sha256": sha256(video_path),
        "middle_frame": str(middle_frame_path),
        "middle_frame_sha256": sha256(middle_frame_path),
        "checks": checks,
    }
    payload["render_passed"] = all(checks.values())
    atomic_json(output_root / "manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["render_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
