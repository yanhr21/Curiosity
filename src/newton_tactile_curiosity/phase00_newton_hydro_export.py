#!/usr/bin/env python3
"""Export Newton official Panda hydro rollout mechanics and tactile maps.

This script intentionally reuses the official Newton Panda hydro Example. It
adds provenance-preserving diagnostics around contact arrays and hydroelastic
surface fields; it does not replace the controller or physics with a toy model.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp


class SurfaceNullViewer:
    """Small wrapper that enables Newton hydro surface buffers without rendering."""

    def __init__(self, num_frames: int):
        import newton

        self._viewer = newton.viewer.ViewerNull(num_frames=num_frames)
        # The official Panda hydro example uses hasattr(viewer, "renderer") to
        # enable HydroelasticSDF output_contact_surface.
        self.renderer = object()

    def register_ui_callback(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return getattr(self._viewer, name)


@dataclass(frozen=True)
class ExportConfig:
    root: Path
    output_dir: Path
    visual_dir: Path
    report_dir: Path
    run_tag: str
    device: str
    scene: str
    num_frames: int
    map_size: int


def _quat_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1.0e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _world_to_body(points: np.ndarray, body_q_row: np.ndarray) -> np.ndarray:
    t = np.asarray(body_q_row[:3], dtype=np.float32)
    r = _quat_to_matrix_xyzw(np.asarray(body_q_row[3:7], dtype=np.float32))
    return (np.asarray(points, dtype=np.float32) - t) @ r


def _body_label(model, body_idx: int) -> str:
    if body_idx < 0:
        return "world"
    if body_idx < len(model.body_label):
        return model.body_label[body_idx]
    return f"body_{body_idx}"


def _classify_shape(shape_idx: int, shape_body: np.ndarray, model) -> str:
    if shape_idx < 0 or shape_idx >= len(shape_body):
        return "invalid"
    label = _body_label(model, int(shape_body[shape_idx])).lower()
    if "leftfinger" in label:
        return "left_pad_or_finger"
    if "rightfinger" in label:
        return "right_pad_or_finger"
    if label.endswith("object") or "/object" in label:
        return "object"
    if label.endswith("cup") or "/cup" in label:
        return "cup"
    if int(shape_body[shape_idx]) < 0:
        return "static_world"
    return "other"


def _accumulate_pad_map(
    pad_map: np.ndarray,
    local_points: np.ndarray,
    weights: np.ndarray,
    map_size: int,
    yz_extent_m: tuple[float, float] = (0.08, 0.08),
) -> None:
    if local_points.size == 0:
        return
    # Diagnostic tactile parameterization: project hydro surface centroids into
    # finger-body local y/z. This is provenance-labeled, not a real gel sensor.
    y = local_points[:, 1]
    z = local_points[:, 2]
    half_y, half_z = yz_extent_m[0] / 2.0, yz_extent_m[1] / 2.0
    iy = np.floor((np.clip(y, -half_y, half_y) + half_y) / (2.0 * half_y + 1e-9) * map_size).astype(np.int32)
    iz = np.floor((np.clip(z, -half_z, half_z) + half_z) / (2.0 * half_z + 1e-9) * map_size).astype(np.int32)
    iy = np.clip(iy, 0, map_size - 1)
    iz = np.clip(iz, 0, map_size - 1)
    for row, col, weight in zip(iz, iy, weights, strict=False):
        pad_map[row, col] += float(max(weight, 0.0))


def _write_plots(
    visual_dir: Path,
    frame_index: np.ndarray,
    object_z: np.ndarray,
    contact_count: np.ndarray,
    hydro_face_count: np.ndarray,
    left_mass: np.ndarray,
    right_mass: np.ndarray,
    left_maps: np.ndarray,
    right_maps: np.ndarray,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    visual_dir.mkdir(parents=True, exist_ok=True)

    def write_svg_metric(path: Path) -> None:
        width, height = 960, 540
        margin = 48
        panels = [
            ("object_z_m", object_z.astype(np.float32), "#2f6f9f"),
            ("hydro_faces", hydro_face_count.astype(np.float32), "#b55d2a"),
            ("left_plus_right_mass", (left_mass + right_mass).astype(np.float32), "#7c4d8f"),
        ]
        panel_h = (height - 2 * margin) / len(panels)

        def polyline(values: np.ndarray, y0: float, h: float) -> str:
            vmax = float(values.max(initial=0.0))
            vmin = float(values.min(initial=0.0))
            if math.isclose(vmax, vmin):
                vmax = vmin + 1.0
            pts = []
            denom = max(1, len(values) - 1)
            for i, value in enumerate(values):
                x = margin + (width - 2 * margin) * i / denom
                y = y0 + h - (h - 28) * (float(value) - vmin) / (vmax - vmin) - 12
                pts.append(f"{x:.2f},{y:.2f}")
            return " ".join(pts)

        lines = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">',
            '<rect width="960" height="540" fill="#f7f7f3"/>',
        ]
        for panel_idx, (name, values, color) in enumerate(panels):
            y0 = margin + panel_idx * panel_h
            lines.append(f'<text x="18" y="{y0 + 18:.1f}" font-size="15" fill="#222">{name}</text>')
            lines.append(
                f'<rect x="{margin}" y="{y0:.1f}" width="{width - 2 * margin}" height="{panel_h - 10:.1f}" '
                'fill="#ffffff" stroke="#d5d5ce"/>'
            )
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline(values, y0, panel_h - 10)}"/>'
            )
            lines.append(
                f'<text x="{width - margin + 8}" y="{y0 + 18:.1f}" font-size="12" fill="#555">'
                f'max={float(values.max(initial=0.0)):.5g}</text>'
            )
        lines.append("</svg>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics_svg = visual_dir / "metrics.svg"
    write_svg_metric(metrics_svg)
    paths["metrics_svg"] = str(metrics_svg)

    def colorize(values: np.ndarray, vmax: float) -> np.ndarray:
        x = np.clip(values.astype(np.float32) / max(vmax, 1.0e-12), 0.0, 1.0)
        r = np.clip(4.0 * x - 1.2, 0.0, 1.0)
        g = np.clip(3.0 * x - 0.6, 0.0, 1.0) * np.clip(1.4 - x, 0.0, 1.0)
        b = np.clip(1.8 - 3.0 * x, 0.0, 1.0) * 0.45 + 0.05
        rgb = np.stack([r, g, b], axis=-1)
        return (255.0 * rgb).astype(np.uint8)

    def resize_nn(img: np.ndarray, scale: int) -> np.ndarray:
        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

    def write_ppm(path: Path, img: np.ndarray) -> None:
        img = np.ascontiguousarray(img, dtype=np.uint8)
        header = f"P6\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii")
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(img.tobytes())

    def frame_image(idx: int) -> np.ndarray:
        left = resize_nn(colorize(left_maps[idx], vmax), scale)
        right = resize_nn(colorize(right_maps[idx], vmax), scale)
        frame = np.full((tile_h, 2 * tile_w + gap, 3), 245, dtype=np.uint8)
        frame[:, :tile_w] = left
        frame[:, tile_w + gap :] = right
        return frame

    def write_uncompressed_avi(path: Path, nframes: int, fps: int = 30) -> None:
        import struct

        first = frame_image(0)
        height, width = first.shape[:2]
        row_bytes = width * 3
        stride = (row_bytes + 3) & ~3
        image_size = stride * height
        idx_entries: list[tuple[bytes, int, int, int]] = []

        def chunk(handle, fourcc: bytes, payload: bytes) -> None:
            handle.write(fourcc)
            handle.write(struct.pack("<I", len(payload)))
            handle.write(payload)
            if len(payload) % 2:
                handle.write(b"\x00")

        def list_start(handle, list_type: bytes) -> int:
            handle.write(b"LIST")
            pos = handle.tell()
            handle.write(b"\x00\x00\x00\x00")
            handle.write(list_type)
            return pos

        def list_end(handle, pos: int) -> None:
            end = handle.tell()
            handle.seek(pos)
            handle.write(struct.pack("<I", end - pos - 4))
            handle.seek(end)

        def dib_payload(rgb: np.ndarray) -> bytes:
            bgr_bottom_up = rgb[::-1, :, ::-1]
            if stride == row_bytes:
                return np.ascontiguousarray(bgr_bottom_up).tobytes()
            rows = []
            pad = b"\x00" * (stride - row_bytes)
            for row in bgr_bottom_up:
                rows.append(np.ascontiguousarray(row).tobytes() + pad)
            return b"".join(rows)

        with path.open("wb") as handle:
            handle.write(b"RIFF")
            riff_size_pos = handle.tell()
            handle.write(b"\x00\x00\x00\x00")
            handle.write(b"AVI ")

            hdrl_pos = list_start(handle, b"hdrl")
            avih = struct.pack(
                "<IIIIIIIIII4I",
                int(1_000_000 / fps),
                image_size * fps,
                0,
                0x10,
                nframes,
                0,
                1,
                image_size,
                width,
                height,
                0,
                0,
                0,
                0,
            )
            chunk(handle, b"avih", avih)

            strl_pos = list_start(handle, b"strl")
            strh = struct.pack(
                "<4s4sIHHIIIIIIIIhhhh",
                b"vids",
                b"DIB ",
                0,
                0,
                0,
                0,
                1,
                fps,
                0,
                nframes,
                image_size,
                0xFFFFFFFF,
                0,
                0,
                0,
                width,
                height,
            )
            chunk(handle, b"strh", strh)
            strf = struct.pack(
                "<IiiHHIIiiII",
                40,
                width,
                height,
                1,
                24,
                0,
                image_size,
                0,
                0,
                0,
                0,
            )
            chunk(handle, b"strf", strf)
            list_end(handle, strl_pos)
            list_end(handle, hdrl_pos)

            movi_pos = list_start(handle, b"movi")
            movi_data_start = handle.tell()
            for idx in range(nframes):
                payload = dib_payload(frame_image(idx))
                chunk_start = handle.tell()
                handle.write(b"00db")
                handle.write(struct.pack("<I", len(payload)))
                data_start = handle.tell()
                handle.write(payload)
                if len(payload) % 2:
                    handle.write(b"\x00")
                idx_entries.append((b"00db", 0x10, data_start - movi_data_start - 8, len(payload)))
            list_end(handle, movi_pos)

            idx_payload = b"".join(struct.pack("<4sIII", *entry) for entry in idx_entries)
            chunk(handle, b"idx1", idx_payload)
            end = handle.tell()
            handle.seek(riff_size_pos)
            handle.write(struct.pack("<I", end - 8))
            handle.seek(end)

    sample_frames = np.linspace(0, len(frame_index) - 1, min(8, len(frame_index)), dtype=int)
    vmax = float(max(left_maps.max(initial=0.0), right_maps.max(initial=0.0), 1.0e-9))
    scale = 8
    gap = 8
    tile_h = left_maps.shape[1] * scale
    tile_w = left_maps.shape[2] * scale
    sheet = np.full((2 * tile_h + gap, len(sample_frames) * tile_w + (len(sample_frames) - 1) * gap, 3), 245, dtype=np.uint8)
    for col, frame in enumerate(sample_frames):
        x0 = col * (tile_w + gap)
        sheet[0:tile_h, x0 : x0 + tile_w] = resize_nn(colorize(left_maps[frame], vmax), scale)
        sheet[tile_h + gap : 2 * tile_h + gap, x0 : x0 + tile_w] = resize_nn(colorize(right_maps[frame], vmax), scale)
    sheet_path = visual_dir / "tactile_sheet.ppm"
    write_ppm(sheet_path, sheet)
    paths["tactile_sheet_ppm"] = str(sheet_path)

    frames_dir = visual_dir / "tactile_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(frame_index)):
        write_ppm(frames_dir / f"frame_{idx:04d}.ppm", frame_image(idx))
    paths["dense_frame_dir"] = str(frames_dir)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        mp4 = visual_dir / "tactile_maps.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                "30",
                "-i",
                str(frames_dir / "frame_%04d.ppm"),
                "-vf",
                "format=yuv420p",
                str(mp4),
            ],
            check=True,
        )
        paths["tactile_maps_mp4"] = str(mp4)
    else:
        avi = visual_dir / "tactile_maps.avi"
        write_uncompressed_avi(avi, len(frame_index), fps=30)
        paths["tactile_maps_avi"] = str(avi)
    return paths


def run_export(cfg: ExportConfig) -> dict:
    from newton.examples.robot.example_robot_panda_hydro import Example

    wp.set_device(cfg.device)
    viewer = SurfaceNullViewer(num_frames=cfg.num_frames)
    args = SimpleNamespace(scene=cfg.scene, test=True, world_count=1)
    example = Example(viewer, args)

    shape_body = example.model.shape_body.numpy()
    shape_classes = [_classify_shape(i, shape_body, example.model) for i in range(example.model.shape_count)]
    left_bodies = [i for i, label in enumerate(example.model.body_label) if "leftfinger" in label.lower()]
    right_bodies = [i for i, label in enumerate(example.model.body_label) if "rightfinger" in label.lower()]
    left_body = left_bodies[0] if left_bodies else None
    right_body = right_bodies[0] if right_bodies else None

    map_size = cfg.map_size
    left_maps = np.zeros((cfg.num_frames, map_size, map_size), dtype=np.float32)
    right_maps = np.zeros((cfg.num_frames, map_size, map_size), dtype=np.float32)
    contact_count = np.zeros(cfg.num_frames, dtype=np.int32)
    hydro_face_count = np.zeros(cfg.num_frames, dtype=np.int32)
    max_penetration = np.zeros(cfg.num_frames, dtype=np.float32)
    left_mass = np.zeros(cfg.num_frames, dtype=np.float32)
    right_mass = np.zeros(cfg.num_frames, dtype=np.float32)
    object_z = np.zeros(cfg.num_frames, dtype=np.float32)
    force_norm_sum = np.zeros(cfg.num_frames, dtype=np.float32)
    pad_pair_faces = np.zeros((cfg.num_frames, 2), dtype=np.int32)

    for frame in range(cfg.num_frames):
        example.step()
        wp.synchronize()

        body_q = example.state_0.body_q.numpy()
        object_body_idx = example.object_body_local
        object_z[frame] = float(body_q[object_body_idx][2])

        n_contact = int(example.contacts.rigid_contact_count.numpy()[0])
        contact_count[frame] = n_contact
        if n_contact > 0:
            forces = example.contacts.rigid_contact_force.numpy()[:n_contact]
            force_norm_sum[frame] = float(np.linalg.norm(forces, axis=1).sum())

        hydro = (
            example.collision_pipeline.hydroelastic_sdf.get_contact_surface()
            if example.collision_pipeline.hydroelastic_sdf is not None
            else None
        )
        if hydro is None:
            continue
        n_face = int(hydro.face_contact_count.numpy()[0])
        hydro_face_count[frame] = n_face
        if n_face <= 0:
            continue

        depths = hydro.contact_surface_depth.numpy()[:n_face].astype(np.float32)
        pairs = hydro.contact_surface_shape_pair.numpy()[:n_face].astype(np.int32)
        vertices = hydro.contact_surface_point.numpy()[: 3 * n_face].astype(np.float32).reshape(n_face, 3, 3)
        centroids = vertices.mean(axis=1)
        penetration = np.maximum(-depths, 0.0)
        max_penetration[frame] = float(penetration.max(initial=0.0))

        left_points: list[np.ndarray] = []
        left_weights: list[float] = []
        right_points: list[np.ndarray] = []
        right_weights: list[float] = []
        for face_idx, (shape_a, shape_b) in enumerate(pairs):
            classes = {shape_classes[int(shape_a)], shape_classes[int(shape_b)]}
            weight = float(penetration[face_idx])
            if weight <= 0.0:
                continue
            if "left_pad_or_finger" in classes and left_body is not None:
                left_points.append(centroids[face_idx])
                left_weights.append(weight)
            if "right_pad_or_finger" in classes and right_body is not None:
                right_points.append(centroids[face_idx])
                right_weights.append(weight)

        if left_points and left_body is not None:
            local = _world_to_body(np.asarray(left_points, dtype=np.float32), body_q[left_body])
            weights = np.asarray(left_weights, dtype=np.float32)
            _accumulate_pad_map(left_maps[frame], local, weights, map_size)
            pad_pair_faces[frame, 0] = len(left_points)
        if right_points and right_body is not None:
            local = _world_to_body(np.asarray(right_points, dtype=np.float32), body_q[right_body])
            weights = np.asarray(right_weights, dtype=np.float32)
            _accumulate_pad_map(right_maps[frame], local, weights, map_size)
            pad_pair_faces[frame, 1] = len(right_points)

        left_mass[frame] = float(left_maps[frame].sum())
        right_mass[frame] = float(right_maps[frame].sum())

    example.test_final()
    viewer.close()

    frame_index = np.arange(cfg.num_frames, dtype=np.int32)
    npz_path = cfg.output_dir / "hydro_tactile_timeseries.npz"
    np.savez_compressed(
        npz_path,
        frame_index=frame_index,
        left_pressure_map=left_maps,
        right_pressure_map=right_maps,
        contact_count=contact_count,
        hydro_face_count=hydro_face_count,
        max_penetration=max_penetration,
        left_pressure_mass=left_mass,
        right_pressure_mass=right_mass,
        object_z=object_z,
        force_norm_sum=force_norm_sum,
        pad_pair_faces=pad_pair_faces,
    )

    plot_paths = _write_plots(
        cfg.visual_dir,
        frame_index,
        object_z,
        contact_count,
        hydro_face_count,
        left_mass,
        right_mass,
        left_maps,
        right_maps,
    )
    initial_z = float(object_z[0])
    max_lift = float(object_z.max(initial=initial_z) - initial_z)
    summary = {
        "classification": "phase00_newton_official_panda_hydro_tactile_export_v1",
        "run_tag": cfg.run_tag,
        "status": "pass",
        "not_training_result": True,
        "not_curiosity_success": True,
        "official_example": "newton.examples.robot.example_robot_panda_hydro",
        "scene": cfg.scene,
        "num_frames": cfg.num_frames,
        "map_size": map_size,
        "tactile_representation": (
            "hydroelastic_contact_surface_centroids_projected_to_left_right_finger_body_yz_"
            "weighted_by_positive_penetration_depth"
        ),
        "npz_path": str(npz_path),
        "visuals": plot_paths,
        "video_path": plot_paths.get("tactile_maps_mp4") or plot_paths.get("tactile_maps_avi"),
        "max_object_lift_m": max_lift,
        "max_contact_count": int(contact_count.max(initial=0)),
        "max_hydro_face_count": int(hydro_face_count.max(initial=0)),
        "max_penetration_m": float(max_penetration.max(initial=0.0)),
        "left_pressure_mass_max": float(left_mass.max(initial=0.0)),
        "right_pressure_mass_max": float(right_mass.max(initial=0.0)),
        "force_norm_sum_max": float(force_norm_sum.max(initial=0.0)),
        "left_touch_frames": int((left_mass > 0.0).sum()),
        "right_touch_frames": int((right_mass > 0.0).sum()),
        "shape_classes": shape_classes,
        "body_labels": list(example.model.body_label),
        "missing_for_curiosity_success": [
            "closed_loop_curiosity_training",
            "harder_heldout_tasks",
            "baseline_and_ablation_comparison",
            "manual_visual_inspection",
        ],
    }
    if summary["max_hydro_face_count"] <= 0 or (summary["left_touch_frames"] + summary["right_touch_frames"]) <= 0:
        summary["status"] = "partial_no_pad_resolved_hydro_touch"

    summary_path = cfg.output_dir / "hydro_tactile_summary.json"
    report_path = cfg.report_dir / "hydro_tactile_export.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        "# Phase 00 Newton Hydro Tactile Export\n\n"
        f"- run_tag: `{cfg.run_tag}`\n"
        f"- status: `{summary['status']}`\n"
        f"- official example: `{summary['official_example']}`\n"
        f"- max object lift m: `{summary['max_object_lift_m']:.6f}`\n"
        f"- max hydro face count: `{summary['max_hydro_face_count']}`\n"
        f"- left/right touch frames: `{summary['left_touch_frames']}` / `{summary['right_touch_frames']}`\n"
        f"- NPZ: `{npz_path}`\n"
        f"- metrics: `{plot_paths.get('metrics_svg')}`\n"
        f"- tactile sheet: `{plot_paths.get('tactile_sheet_ppm')}`\n"
        f"- tactile video: `{plot_paths.get('tactile_maps_mp4') or plot_paths.get('tactile_maps_avi')}`\n"
        "\nThis is dense tactile-environment evidence, not training and not curiosity success.\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str]) -> ExportConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=["cube", "pen"], default="cube")
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return ExportConfig(
        root=args.root,
        output_dir=args.output_dir,
        visual_dir=args.visual_dir,
        report_dir=args.report_dir,
        run_tag=args.run_tag,
        device=args.device,
        scene=args.scene,
        num_frames=args.num_frames,
        map_size=args.map_size,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.visual_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    summary = run_export(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
