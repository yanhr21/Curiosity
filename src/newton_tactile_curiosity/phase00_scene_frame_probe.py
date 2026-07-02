#!/usr/bin/env python3
"""Render real Newton SensorTiledCamera frames for Phase 00 scene evidence."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw


def normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("cannot normalize near-zero vector")
    return value / norm


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def look_at_transform(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = normalize(target - position)
    up_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize(np.cross(forward, up_hint))
    up = np.cross(right, forward)
    rotation = np.column_stack([right, up, -forward])
    return position.astype(np.float32), matrix_to_quat_xyzw(rotation).astype(np.float32)


def camera_transforms(step: int, total_steps: int, world_count: int):
    import warp as wp

    phase = step / max(1, total_steps - 1)
    target = np.asarray([-0.18, -0.50, 0.28], dtype=np.float32)
    poses = [
        look_at_transform(np.asarray([0.55, -1.25, 0.72], dtype=np.float32), target),
        look_at_transform(np.asarray([0.18, -0.88, 0.44 + 0.04 * math.sin(math.pi * phase)], dtype=np.float32), target),
        look_at_transform(np.asarray([-0.62, -0.88, 0.48 + 0.04 * math.cos(math.pi * phase)], dtype=np.float32), target),
    ]
    rows = []
    meta = []
    for name, (pos, quat) in zip(("head", "right_wrist", "left_wrist"), poses, strict=True):
        rows.append(
            [
                wp.transformf(
                    wp.vec3f(float(pos[0]), float(pos[1]), float(pos[2])),
                    wp.quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                )
            ]
            * world_count
        )
        meta.append({"name": name, "position": pos.tolist(), "target": target.tolist(), "quat_xyzw": quat.tolist()})
    return wp.array(rows, dtype=wp.transformf), meta


def write_uncompressed_avi(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("cannot write AVI without frames")
    height, width = frames[0].shape[:2]
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
        pad = b"\x00" * (stride - row_bytes)
        return b"".join(np.ascontiguousarray(row).tobytes() + pad for row in bgr_bottom_up)

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
            len(frames),
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
            len(frames),
            image_size,
            0xFFFFFFFF,
            0,
            0,
            0,
            width,
            height,
        )
        chunk(handle, b"strh", strh)
        strf = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, image_size, 0, 0, 0, 0)
        chunk(handle, b"strf", strf)
        list_end(handle, strl_pos)
        list_end(handle, hdrl_pos)
        movi_pos = list_start(handle, b"movi")
        movi_data_start = movi_pos + 4
        for frame in frames:
            offset = handle.tell() - movi_data_start
            payload = dib_payload(frame)
            chunk(handle, b"00db", payload)
            idx_entries.append((b"00db", 0x10, offset, len(payload)))
        list_end(handle, movi_pos)
        idx_payload = b"".join(struct.pack("<4sIII", *entry) for entry in idx_entries)
        chunk(handle, b"idx1", idx_payload)
        end = handle.tell()
        handle.seek(riff_size_pos)
        handle.write(struct.pack("<I", end - 8))


def compose_triptych(rgba: np.ndarray, title: str) -> Image.Image:
    names = ("head", "right_wrist", "left_wrist")
    panels = []
    for image, name in zip(rgba, names, strict=True):
        panel = Image.fromarray(image[..., :3], mode="RGB")
        canvas = Image.new("RGB", (panel.width, panel.height + 24), "white")
        canvas.paste(panel, (0, 0))
        ImageDraw.Draw(canvas).text((6, panel.height + 5), name, fill=(0, 0, 0))
        panels.append(canvas)
    out = Image.new("RGB", (sum(p.width for p in panels), panels[0].height + 28), "white")
    ImageDraw.Draw(out).text((8, 8), title, fill=(0, 0, 0))
    x = 0
    for panel in panels:
        out.paste(panel, (x, 28))
        x += panel.width
    return out


def write_contact_sheet(frames: list[Image.Image], path: Path, cols: int = 3) -> None:
    thumbs = []
    for i, frame in enumerate(frames):
        thumb = frame.copy()
        thumb.thumbnail((720, 260), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (740, 290), "white")
        canvas.paste(thumb, ((740 - thumb.width) // 2, 0))
        ImageDraw.Draw(canvas).text((8, 266), f"frame_{i:04d}", fill=(0, 0, 0))
        thumbs.append(canvas)
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 740, rows * 290), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 740, (i // cols) * 290))
    sheet.save(path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--scene", default="cube", choices=["cube", "pen"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    import newton
    import newton.viewer
    import warp as wp
    from newton.examples.robot.example_robot_panda_hydro import Example
    from newton.sensors import SensorTiledCamera

    wp.set_device(args.device)
    output_dir = Path(args.output_dir)
    visual_dir = Path(args.visual_dir)
    report_dir = Path(args.report_dir)
    frame_dir = visual_dir / "frames"
    for path in (output_dir, visual_dir, report_dir, frame_dir):
        path.mkdir(parents=True, exist_ok=True)

    viewer = newton.viewer.ViewerNull(num_frames=args.steps)
    example_args = SimpleNamespace(scene=args.scene, test=True, world_count=1)
    example = Example(viewer, example_args)

    sensor = SensorTiledCamera(model=example.model)
    sensor.utils.create_default_light(enable_shadows=True)
    rays = sensor.utils.compute_pinhole_camera_rays(args.width, args.height, [math.radians(45.0)] * 3)
    color_image = sensor.utils.create_color_image_output(args.width, args.height, camera_count=3)
    sample_indices = set(np.linspace(0, args.steps - 1, args.samples, dtype=np.int32).tolist())

    frame_paths: list[str] = []
    sheet_frames: list[Image.Image] = []
    avi_frames: list[np.ndarray] = []
    camera_meta = None
    object_z = []
    for step in range(args.steps):
        example.step()
        body_q = example.state_0.body_q.numpy()
        object_idx = example.object_body_local
        object_z.append(float(body_q[object_idx][2]))
        if step not in sample_indices:
            continue
        example.model.bvh_refit_shapes(example.state_0)
        transforms, camera_meta = camera_transforms(step, args.steps, example.world_count)
        sensor.update(
            example.state_0,
            transforms,
            rays,
            color_image=color_image,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        rgba = sensor.utils.to_rgba_from_color(color_image).numpy().copy()
        triptych = compose_triptych(rgba, f"{args.run_tag} step={step}")
        frame_path = frame_dir / f"scene_{step:04d}.png"
        triptych.save(frame_path)
        frame_paths.append(str(frame_path))
        sheet_frames.append(triptych)
        avi_frames.append(np.asarray(triptych, dtype=np.uint8))

    sheet_path = visual_dir / "scene_camera_sheet.jpg"
    avi_path = visual_dir / "scene_camera.avi"
    write_contact_sheet(sheet_frames, sheet_path)
    write_uncompressed_avi(avi_path, avi_frames, fps=12)

    first = np.asarray(sheet_frames[0], dtype=np.uint8)
    stack = np.stack([np.asarray(frame, dtype=np.uint8) for frame in sheet_frames], axis=0)
    summary = {
        "classification": "phase00_newton_main_sensor_tiled_camera_probe",
        "run_tag": args.run_tag,
        "not_training_result": True,
        "not_curiosity_success": True,
        "scene": args.scene,
        "steps": args.steps,
        "sampled_frames": len(frame_paths),
        "camera_names": ["head", "right_wrist", "left_wrist"],
        "camera_meta_last": camera_meta,
        "frame_paths": frame_paths,
        "contact_sheet": str(sheet_path),
        "avi": str(avi_path),
        "first_frame_shape": list(first.shape),
        "pixel_min": int(stack.min()),
        "pixel_max": int(stack.max()),
        "pixel_std": float(stack.std()),
        "nonblank": bool(stack.max() > stack.min() and stack.std() > 1.0),
        "object_initial_z": float(object_z[0]) if object_z else None,
        "object_max_z": float(max(object_z)) if object_z else None,
        "object_lift": float(max(object_z) - object_z[0]) if object_z else None,
        "source": "official_newton_main_panda_hydro_plus_SensorTiledCamera",
    }
    (output_dir / "scene_frame_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_dir / "scene_frame_probe.md").write_text(
        "# Phase 00 Scene Frame Probe\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- scene: `{args.scene}`\n"
        f"- sampled frames: `{len(frame_paths)}`\n"
        f"- nonblank: `{summary['nonblank']}`\n"
        f"- pixel std: `{summary['pixel_std']}`\n"
        f"- object lift: `{summary['object_lift']}`\n"
        f"- sheet: `{sheet_path}`\n"
        f"- avi: `{avi_path}`\n\n"
        "Classification: real SensorTiledCamera scene-frame probe only. "
        "This is not dense tactile success, training, or curiosity success.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["nonblank"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
