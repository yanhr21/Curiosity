#!/usr/bin/env python3
"""Synchronized Newton hydro scene+tactile diagnostic.

This reuses the official Newton Panda hydro example and exports synchronized
scene schematic frames, left/right hydro-derived tactile maps, and mechanics
statistics. Fields that are not direct sensor measurements are explicitly
named `hydro_proxy.*`.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
from PIL import Image, ImageDraw


class SurfaceNullViewer:
    """Null viewer that enables Newton HydroelasticSDF contact-surface buffers."""

    def __init__(self, num_frames: int):
        import newton

        self._viewer = newton.viewer.ViewerNull(num_frames=num_frames)
        self.renderer = object()

    def register_ui_callback(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return getattr(self._viewer, name)


@dataclass(frozen=True)
class Config:
    root: Path
    run_tag: str
    output_dir: Path
    visual_dir: Path
    report_dir: Path
    device: str
    scene: str
    num_frames: int
    map_size: int
    fps: int
    material_label: str
    override_mu: float | None
    override_kh: float | None
    scene_camera: bool
    scene_camera_width: int
    scene_camera_height: int


def quat_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
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


def world_to_body(points: np.ndarray, body_q_row: np.ndarray) -> np.ndarray:
    t = np.asarray(body_q_row[:3], dtype=np.float32)
    r = quat_to_matrix_xyzw(np.asarray(body_q_row[3:7], dtype=np.float32))
    return (np.asarray(points, dtype=np.float32) - t) @ r


def body_vector_to_world(local_vectors: np.ndarray, body_q_row: np.ndarray) -> np.ndarray:
    r = quat_to_matrix_xyzw(np.asarray(body_q_row[3:7], dtype=np.float32))
    return np.asarray(local_vectors, dtype=np.float32) @ r.T


def decode_oct(oct_xy: np.ndarray) -> np.ndarray:
    x = oct_xy[:, 0].astype(np.float32)
    y = oct_xy[:, 1].astype(np.float32)
    z = 1.0 - np.abs(x) - np.abs(y)
    neg = z < 0.0
    old_x = x.copy()
    x[neg] = (1.0 - np.abs(y[neg])) * np.sign(old_x[neg])
    y[neg] = (1.0 - np.abs(old_x[neg])) * np.sign(y[neg])
    n = np.stack([x, y, z], axis=1)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1.0e-12)


def body_label(model, body_idx: int) -> str:
    if body_idx < 0:
        return "world"
    if body_idx < len(model.body_label):
        return model.body_label[body_idx]
    return f"body_{body_idx}"


def classify_shape(shape_idx: int, shape_body: np.ndarray, model) -> str:
    if shape_idx < 0 or shape_idx >= len(shape_body):
        return "invalid"
    label = body_label(model, int(shape_body[shape_idx])).lower()
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


def pair_has(classes: list[str], pair: np.ndarray, name: str) -> np.ndarray:
    return np.array([(classes[int(a)] == name or classes[int(b)] == name) for a, b in pair], dtype=bool)


def accumulate_wrench(points: np.ndarray, forces: np.ndarray, origin: np.ndarray) -> np.ndarray:
    if points.size == 0 or forces.size == 0:
        return np.zeros(6, dtype=np.float32)
    f = forces.sum(axis=0)
    tau = np.cross(points - origin[None, :], forces).sum(axis=0)
    return np.concatenate([f, tau]).astype(np.float32)


def accumulate_map(
    pad_map: np.ndarray,
    local_points: np.ndarray,
    weights: np.ndarray,
    extent=(0.08, 0.08),
    center_yz=(0.0, 0.0),
) -> None:
    if local_points.size == 0:
        return
    size = pad_map.shape[0]
    y = local_points[:, 1] - float(center_yz[0])
    z = local_points[:, 2] - float(center_yz[1])
    hy, hz = extent[0] / 2.0, extent[1] / 2.0
    iy = np.floor((np.clip(y, -hy, hy) + hy) / (2.0 * hy + 1e-9) * size).astype(np.int32)
    iz = np.floor((np.clip(z, -hz, hz) + hz) / (2.0 * hz + 1e-9) * size).astype(np.int32)
    iy = np.clip(iy, 0, size - 1)
    iz = np.clip(iz, 0, size - 1)
    for row, col, weight in zip(iz, iy, weights, strict=False):
        pad_map[row, col] += float(max(weight, 0.0))


def accumulate_gaussian_scalar(
    pad_map: np.ndarray,
    local_points: np.ndarray,
    weights: np.ndarray,
    extent=(0.08, 0.08),
    center_yz=(0.0, 0.0),
    sigma_cells: float = 1.35,
) -> None:
    if local_points.size == 0:
        return
    size = pad_map.shape[0]
    y = local_points[:, 1] - float(center_yz[0])
    z = local_points[:, 2] - float(center_yz[1])
    hy, hz = extent[0] / 2.0, extent[1] / 2.0
    fy = (np.clip(y, -hy, hy) + hy) / (2.0 * hy + 1e-9) * (size - 1)
    fz = (np.clip(z, -hz, hz) + hz) / (2.0 * hz + 1e-9) * (size - 1)
    radius = max(1, int(math.ceil(3.0 * sigma_cells)))
    denom = 2.0 * sigma_cells * sigma_cells
    for cy, cz, weight in zip(fy, fz, weights, strict=False):
        w = float(weight)
        if not math.isfinite(w) or w <= 0.0:
            continue
        c_col = int(round(float(cy)))
        c_row = int(round(float(cz)))
        for row in range(max(0, c_row - radius), min(size, c_row + radius + 1)):
            dz = row - float(cz)
            for col in range(max(0, c_col - radius), min(size, c_col + radius + 1)):
                dy = col - float(cy)
                pad_map[row, col] += w * math.exp(-(dy * dy + dz * dz) / denom)


def accumulate_gaussian_vector(
    y_map: np.ndarray,
    z_map: np.ndarray,
    local_points: np.ndarray,
    weights: np.ndarray,
    vector_yz: np.ndarray,
    extent=(0.08, 0.08),
    center_yz=(0.0, 0.0),
    sigma_cells: float = 1.35,
) -> None:
    if local_points.size == 0:
        return
    size = y_map.shape[0]
    y = local_points[:, 1] - float(center_yz[0])
    z = local_points[:, 2] - float(center_yz[1])
    hy, hz = extent[0] / 2.0, extent[1] / 2.0
    fy = (np.clip(y, -hy, hy) + hy) / (2.0 * hy + 1e-9) * (size - 1)
    fz = (np.clip(z, -hz, hz) + hz) / (2.0 * hz + 1e-9) * (size - 1)
    radius = max(1, int(math.ceil(3.0 * sigma_cells)))
    denom = 2.0 * sigma_cells * sigma_cells
    vy, vz = float(vector_yz[0]), float(vector_yz[1])
    if not (math.isfinite(vy) and math.isfinite(vz)):
        return
    for cy, cz, weight in zip(fy, fz, weights, strict=False):
        w = float(weight)
        if not math.isfinite(w) or w <= 0.0:
            continue
        c_col = int(round(float(cy)))
        c_row = int(round(float(cz)))
        for row in range(max(0, c_row - radius), min(size, c_row + radius + 1)):
            dz = row - float(cz)
            for col in range(max(0, c_col - radius), min(size, c_col + radius + 1)):
                dy = col - float(cy)
                g = w * math.exp(-(dy * dy + dz * dz) / denom)
                y_map[row, col] += g * vy
                z_map[row, col] += g * vz


def contact_view_window(frame_samples: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, tuple[float, float], bool]:
    local_parts = [sample["local"][:, 1:3] for sample in frame_samples if sample["local"].size > 0]
    if not local_parts:
        return np.zeros(2, dtype=np.float32), (0.08, 0.08), False
    yz = np.concatenate(local_parts, axis=0).astype(np.float32)
    lo = np.quantile(yz, 0.01, axis=0)
    hi = np.quantile(yz, 0.99, axis=0)
    span = np.maximum(hi - lo, np.array([0.006, 0.006], dtype=np.float32))
    center = ((lo + hi) / 2.0).astype(np.float32)
    extent = tuple(float(v) for v in np.maximum(span * 1.8, np.array([0.012, 0.012], dtype=np.float32)))
    return center, extent, True


def colorize(values: np.ndarray, vmax: float) -> np.ndarray:
    x = np.clip(values.astype(np.float32) / max(vmax, 1.0e-12), 0.0, 1.0)
    x = np.power(x, 0.35)
    r = np.clip(2.2 * x, 0.0, 1.0)
    g = np.clip(1.7 * x - 0.15, 0.0, 1.0)
    b = np.clip(1.0 - 1.6 * x, 0.0, 0.25) + 0.15 * x
    return (255.0 * np.stack([r, g, b], axis=-1)).astype(np.uint8)


def resize_nn(img: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def tactile_panel(values: np.ndarray, vmax: float, scale: int, arrows: tuple[np.ndarray, np.ndarray] | None = None) -> Image.Image:
    img = Image.fromarray(resize_nn(colorize(values, vmax), scale))
    if arrows is None:
        return img
    y_map, z_map = arrows
    mag = np.sqrt(y_map * y_map + z_map * z_map)
    if float(mag.max(initial=0.0)) <= 1.0e-12:
        return img
    draw = ImageDraw.Draw(img)
    step = max(4, values.shape[0] // 6)
    norm = float(np.quantile(mag[mag > 0.0], 0.90)) if np.any(mag > 0.0) else 1.0
    for row in range(step // 2, values.shape[0], step):
        for col in range(step // 2, values.shape[1], step):
            m = float(mag[row, col])
            if m <= 1.0e-12:
                continue
            x0 = col * scale + scale // 2
            y0 = row * scale + scale // 2
            length = min(0.9 * step * scale, 0.2 * step * scale + 0.7 * step * scale * m / max(norm, 1.0e-12))
            dx = float(y_map[row, col]) / m * length
            dy = -float(z_map[row, col]) / m * length
            x1 = x0 + dx
            y1 = y0 + dy
            draw.line((x0, y0, x1, y1), fill=(20, 20, 20), width=2)
            angle = math.atan2(dy, dx)
            head = 4
            for offset in (2.5, -2.5):
                hx = x1 - head * math.cos(angle + offset)
                hy = y1 - head * math.sin(angle + offset)
                draw.line((x1, y1, hx, hy), fill=(20, 20, 20), width=2)
    return img


def sparkline(values: np.ndarray, width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (width, height), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(190, 190, 180))
    if values.size < 2:
        return img
    vmax = float(values.max(initial=0.0))
    vmin = float(values.min(initial=0.0))
    if math.isclose(vmax, vmin):
        vmax = vmin + 1.0
    pts = []
    denom = max(1, values.size - 1)
    for i, value in enumerate(values):
        x = int((width - 1) * i / denom)
        y = int((height - 8) - (height - 14) * (float(value) - vmin) / (vmax - vmin) + 3)
        pts.append((x, y))
    draw.line(pts, fill=color, width=2)
    return img


def draw_scene_panel(
    body_q: np.ndarray,
    labels: list[str],
    object_idx: int,
    cup_idx: int | None,
    left_idx: int | None,
    right_idx: int | None,
    width: int,
    height: int,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (248, 248, 242))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(165, 165, 155))
    draw.text((8, 6), "scene schematic: top view x-y", fill=(35, 35, 35))
    xs = [-0.08, 0.2]
    ys = [-0.62, -0.38]

    def project(pos):
        x = (float(pos[0]) - xs[0]) / (xs[1] - xs[0])
        y = (float(pos[1]) - ys[0]) / (ys[1] - ys[0])
        px = 25 + x * (width - 50)
        py = height - 25 - y * (height - 55)
        return int(px), int(py)

    for gx in np.linspace(xs[0], xs[1], 5):
        x, _ = project((gx, ys[0], 0))
        draw.line((x, 32, x, height - 20), fill=(222, 222, 214))
    for gy in np.linspace(ys[0], ys[1], 5):
        _, y = project((xs[0], gy, 0))
        draw.line((20, y, width - 20, y), fill=(222, 222, 214))

    if cup_idx is not None:
        c = project(body_q[cup_idx, :3])
        draw.ellipse((c[0] - 14, c[1] - 14, c[0] + 14, c[1] + 14), outline=(70, 120, 180), width=3)
        draw.text((c[0] + 16, c[1] - 6), "cup", fill=(70, 120, 180))
    o = project(body_q[object_idx, :3])
    draw.rectangle((o[0] - 9, o[1] - 9, o[0] + 9, o[1] + 9), fill=(80, 80, 80), outline=(10, 10, 10))
    draw.text((o[0] + 12, o[1] - 6), "object", fill=(20, 20, 20))
    for idx, name, color in [(left_idx, "L", (180, 70, 60)), (right_idx, "R", (60, 120, 180))]:
        if idx is None:
            continue
        p = project(body_q[idx, :3])
        draw.ellipse((p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6), fill=color)
        draw.text((p[0] + 8, p[1] - 7), name, fill=color)
    return img


def normalize_vec(value: np.ndarray) -> np.ndarray:
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
    return quat / np.maximum(np.linalg.norm(quat), 1.0e-12)


def look_at_transform(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = normalize_vec(target - position)
    up_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize_vec(np.cross(forward, up_hint))
    up = np.cross(right, forward)
    rotation = np.column_stack([right, up, -forward])
    return position.astype(np.float32), matrix_to_quat_xyzw(rotation).astype(np.float32)


def scene_camera_transforms(frame: int, num_frames: int, world_count: int):
    phase = frame / max(1, num_frames - 1)
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


def compose_scene_camera_triptych(rgba: np.ndarray, title: str) -> Image.Image:
    panels = []
    for image, name in zip(rgba, ("head", "right_wrist", "left_wrist"), strict=True):
        panel = Image.fromarray(image[..., :3], mode="RGB")
        canvas = Image.new("RGB", (panel.width, panel.height + 22), "white")
        canvas.paste(panel, (0, 0))
        ImageDraw.Draw(canvas).text((6, panel.height + 4), name, fill=(0, 0, 0))
        panels.append(canvas)
    out = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height + 28), "white")
    ImageDraw.Draw(out).text((8, 8), title, fill=(0, 0, 0))
    x = 0
    for panel in panels:
        out.paste(panel, (x, 28))
        x += panel.width
    return out


def write_uncompressed_avi(path: Path, frames: list[np.ndarray], fps: int) -> None:
    first = frames[0]
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
        pad = b"\x00" * (stride - row_bytes)
        return b"".join(np.ascontiguousarray(row).tobytes() + pad for row in bgr_bottom_up)

    with path.open("wb") as handle:
        handle.write(b"RIFF")
        riff_size_pos = handle.tell()
        handle.write(b"\x00\x00\x00\x00")
        handle.write(b"AVI ")
        hdrl_pos = list_start(handle, b"hdrl")
        chunk(
            handle,
            b"avih",
            struct.pack(
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
            ),
        )
        strl_pos = list_start(handle, b"strl")
        chunk(
            handle,
            b"strh",
            struct.pack(
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
            ),
        )
        chunk(handle, b"strf", struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, image_size, 0, 0, 0, 0))
        list_end(handle, strl_pos)
        list_end(handle, hdrl_pos)
        movi_pos = list_start(handle, b"movi")
        movi_data_start = handle.tell()
        for frame in frames:
            payload = dib_payload(frame)
            handle.write(b"00db")
            handle.write(struct.pack("<I", len(payload)))
            data_start = handle.tell()
            handle.write(payload)
            if len(payload) % 2:
                handle.write(b"\x00")
            idx_entries.append((b"00db", 0x10, data_start - movi_data_start - 8, len(payload)))
        list_end(handle, movi_pos)
        chunk(handle, b"idx1", b"".join(struct.pack("<4sIII", *entry) for entry in idx_entries))
        end = handle.tell()
        handle.seek(riff_size_pos)
        handle.write(struct.pack("<I", end - 8))


def write_mp4_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"new videos must be MP4, got: {path}")
    if not frames:
        raise ValueError("cannot write MP4 without frames")
    height, width = frames[0].shape[:2]
    try:
        import cv2  # type: ignore

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open MP4 writer: {path}")
        try:
            for frame in frames:
                arr = np.asarray(frame, dtype=np.uint8)
                if arr.shape[:2] != (height, width):
                    raise ValueError("all MP4 frames must have the same dimensions")
                writer.write(cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    except ModuleNotFoundError:
        import imageio.v2 as imageio  # type: ignore

        with imageio.get_writer(str(path), fps=float(fps), codec="libx264", macro_block_size=1) as writer:
            for frame in frames:
                arr = np.asarray(frame, dtype=np.uint8)
                if arr.shape[:2] != (height, width):
                    raise ValueError("all MP4 frames must have the same dimensions")
                writer.append_data(arr[..., :3])


def render_sync_frames(
    cfg: Config,
    body_q_series: np.ndarray,
    labels: list[str],
    object_idx: int,
    cup_idx: int | None,
    left_idx: int | None,
    right_idx: int | None,
    left_maps: np.ndarray,
    right_maps: np.ndarray,
    left_fn_maps: np.ndarray,
    right_fn_maps: np.ndarray,
    left_deform_maps: np.ndarray,
    right_deform_maps: np.ndarray,
    left_shear_y_maps: np.ndarray,
    left_shear_z_maps: np.ndarray,
    right_shear_y_maps: np.ndarray,
    right_shear_z_maps: np.ndarray,
    object_z: np.ndarray,
    area_sum: np.ndarray,
    fn_proxy: np.ndarray,
    stress_proxy: np.ndarray,
    ft_capacity_proxy: np.ndarray,
    shear_proxy: np.ndarray,
    scene_camera_frames: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    fn_nonzero = np.concatenate([left_fn_maps[left_fn_maps > 0.0], right_fn_maps[right_fn_maps > 0.0]])
    deform_nonzero = np.concatenate([left_deform_maps[left_deform_maps > 0.0], right_deform_maps[right_deform_maps > 0.0]])
    shear_mag_left = np.sqrt(left_shear_y_maps * left_shear_y_maps + left_shear_z_maps * left_shear_z_maps)
    shear_mag_right = np.sqrt(right_shear_y_maps * right_shear_y_maps + right_shear_z_maps * right_shear_z_maps)
    shear_nonzero = np.concatenate([shear_mag_left[shear_mag_left > 0.0], shear_mag_right[shear_mag_right > 0.0]])
    fn_vmax = float(np.quantile(fn_nonzero, 0.95)) if fn_nonzero.size else 1.0
    deform_vmax = float(np.quantile(deform_nonzero, 0.95)) if deform_nonzero.size else 1.0
    shear_vmax = float(np.quantile(shear_nonzero, 0.95)) if shear_nonzero.size else 1.0
    scale = 5
    tactile_w = cfg.map_size * scale
    tactile_h = cfg.map_size * scale
    frames: list[np.ndarray] = []
    for frame_idx in range(cfg.num_frames):
        canvas = Image.new("RGB", (1180, 940), (236, 236, 230))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f"{cfg.run_tag} frame {frame_idx:03d}", fill=(25, 25, 25))
        if scene_camera_frames is not None:
            scene = Image.fromarray(scene_camera_frames[frame_idx]).resize((560, 330), Image.Resampling.LANCZOS)
            draw.text((16, 28), "scene: Newton SensorTiledCamera head/right/left", fill=(35, 35, 35))
        else:
            scene = draw_scene_panel(
                body_q_series[frame_idx],
                labels,
                object_idx,
                cup_idx,
                left_idx,
                right_idx,
                560,
                330,
            )
        canvas.paste(scene, (16, 42))
        left_fn_img = tactile_panel(left_fn_maps[frame_idx], fn_vmax, scale)
        right_fn_img = tactile_panel(right_fn_maps[frame_idx], fn_vmax, scale)
        left_shear_img = tactile_panel(
            shear_mag_left[frame_idx],
            shear_vmax,
            scale,
            (left_shear_y_maps[frame_idx], left_shear_z_maps[frame_idx]),
        )
        right_shear_img = tactile_panel(
            shear_mag_right[frame_idx],
            shear_vmax,
            scale,
            (right_shear_y_maps[frame_idx], right_shear_z_maps[frame_idx]),
        )
        left_deform_img = tactile_panel(left_deform_maps[frame_idx], deform_vmax, scale)
        right_deform_img = tactile_panel(right_deform_maps[frame_idx], deform_vmax, scale)
        x_l = 610
        x_r = 610 + tactile_w + 36
        y_top = 74
        y_mid = 270
        y_bot = 466
        canvas.paste(left_fn_img, (x_l, y_top))
        canvas.paste(right_fn_img, (x_r, y_top))
        canvas.paste(left_shear_img, (x_l, y_mid))
        canvas.paste(right_shear_img, (x_r, y_mid))
        canvas.paste(left_deform_img, (x_l, y_bot))
        canvas.paste(right_deform_img, (x_r, y_bot))
        for x, y, label in [
            (x_l, y_top, "L Fn map"),
            (x_r, y_top, "R Fn map"),
            (x_l, y_mid, "L shear vector"),
            (x_r, y_mid, "R shear vector"),
            (x_l, y_bot, "L deform map"),
            (x_r, y_bot, "R deform map"),
        ]:
            draw.text((x, y - 18), f"hydro_proxy.calib {label}", fill=(25, 25, 25))
            draw.rectangle((x, y, x + tactile_w, y + tactile_h), outline=(60, 60, 60))
        y0 = 650
        curves = [
            ("object_z", object_z, (45, 100, 150)),
            ("contact_area_proxy", area_sum, (160, 95, 45)),
            ("Fn_proxy", fn_proxy, (120, 65, 145)),
            ("stress_proxy", stress_proxy, (170, 55, 45)),
            ("Ft_capacity_proxy", ft_capacity_proxy, (45, 120, 120)),
            ("shear_motion_proxy", shear_proxy, (65, 130, 80)),
        ]
        for i, (name, values, color) in enumerate(curves):
            col = i % 3
            row = i // 3
            x = 30 + col * 380
            y = y0 + row * 145
            draw.text((x, y - 20), name, fill=(35, 35, 35))
            curve = sparkline(values[: frame_idx + 1], 330, 90, color)
            canvas.paste(curve, (x, y))
            draw.text(
                (x, y + 96),
                f"now={float(values[frame_idx]):.5g} max={float(values.max(initial=0.0)):.5g}",
                fill=(35, 35, 35),
            )
        draw.text(
            (30, 905),
            (
                "Scene panel uses Newton SensorTiledCamera; tactile maps are calibrated-view hydro proxies, not final gel sensor output."
                if scene_camera_frames is not None
                else "Scene panel is schematic; tactile maps are calibrated-view Newton hydro proxies, not final gel sensor output."
            ),
            fill=(45, 45, 45),
        )
        frames.append(np.asarray(canvas, dtype=np.uint8))
    return frames


def run(cfg: Config) -> dict:
    from newton.examples.robot.example_robot_panda_hydro import Example

    total_start = time.perf_counter()
    wp.set_device(cfg.device)
    viewer = SurfaceNullViewer(num_frames=cfg.num_frames)
    example = Example(viewer, SimpleNamespace(scene=cfg.scene, test=True, world_count=1))
    material_override_applied = cfg.override_mu is not None or cfg.override_kh is not None
    if cfg.override_mu is not None:
        example.model.shape_material_mu.fill_(float(cfg.override_mu))
    if cfg.override_kh is not None:
        example.model.shape_material_kh.fill_(float(cfg.override_kh))
    if material_override_applied:
        wp.synchronize()

    scene_camera_sensor = None
    scene_camera_rays = None
    scene_camera_color_image = None
    scene_camera_frames: list[np.ndarray] | None = None
    scene_camera_meta = None
    if cfg.scene_camera:
        from newton.sensors import SensorTiledCamera

        scene_camera_sensor = SensorTiledCamera(model=example.model)
        scene_camera_sensor.utils.create_default_light(enable_shadows=True)
        scene_camera_rays = scene_camera_sensor.utils.compute_pinhole_camera_rays(
            cfg.scene_camera_width,
            cfg.scene_camera_height,
            [math.radians(45.0)] * 3,
        )
        scene_camera_color_image = scene_camera_sensor.utils.create_color_image_output(
            cfg.scene_camera_width,
            cfg.scene_camera_height,
            camera_count=3,
        )
        scene_camera_frames = []

    labels = list(example.model.body_label)
    shape_body = example.model.shape_body.numpy()
    shape_classes = [classify_shape(i, shape_body, example.model) for i in range(example.model.shape_count)]
    left_body = next((i for i, label in enumerate(labels) if "leftfinger" in label.lower()), None)
    right_body = next((i for i, label in enumerate(labels) if "rightfinger" in label.lower()), None)
    cup_body = next((i for i, label in enumerate(labels) if label.endswith("cup")), None)
    object_body = example.object_body_local

    left_maps = np.zeros((cfg.num_frames, cfg.map_size, cfg.map_size), dtype=np.float32)
    right_maps = np.zeros_like(left_maps)
    left_fn_maps = np.zeros_like(left_maps)
    right_fn_maps = np.zeros_like(left_maps)
    left_stress_maps = np.zeros_like(left_maps)
    right_stress_maps = np.zeros_like(left_maps)
    left_deform_maps = np.zeros_like(left_maps)
    right_deform_maps = np.zeros_like(left_maps)
    left_shear_y_maps = np.zeros_like(left_maps)
    left_shear_z_maps = np.zeros_like(left_maps)
    right_shear_y_maps = np.zeros_like(left_maps)
    right_shear_z_maps = np.zeros_like(left_maps)
    left_shear_magnitude_maps = np.zeros_like(left_maps)
    right_shear_magnitude_maps = np.zeros_like(left_maps)
    left_f6_normal_proxy = np.zeros((cfg.num_frames, 6), dtype=np.float32)
    right_f6_normal_proxy = np.zeros((cfg.num_frames, 6), dtype=np.float32)
    left_f6_ft_capacity_proxy = np.zeros((cfg.num_frames, 6), dtype=np.float32)
    right_f6_ft_capacity_proxy = np.zeros((cfg.num_frames, 6), dtype=np.float32)
    left_f6_combined_proxy = np.zeros((cfg.num_frames, 6), dtype=np.float32)
    right_f6_combined_proxy = np.zeros((cfg.num_frames, 6), dtype=np.float32)
    left_frame_samples: list[list[dict[str, np.ndarray]]] = [[] for _ in range(cfg.num_frames)]
    right_frame_samples: list[list[dict[str, np.ndarray]]] = [[] for _ in range(cfg.num_frames)]
    object_z = np.zeros(cfg.num_frames, dtype=np.float32)
    hydro_face_count = np.zeros(cfg.num_frames, dtype=np.int32)
    raw_contact_count = np.zeros(cfg.num_frames, dtype=np.int32)
    contact_area_sum = np.zeros(cfg.num_frames, dtype=np.float32)
    contact_area_left = np.zeros(cfg.num_frames, dtype=np.float32)
    contact_area_right = np.zeros(cfg.num_frames, dtype=np.float32)
    fn_proxy = np.zeros(cfg.num_frames, dtype=np.float32)
    fn_proxy_left = np.zeros(cfg.num_frames, dtype=np.float32)
    fn_proxy_right = np.zeros(cfg.num_frames, dtype=np.float32)
    ft_capacity_proxy = np.zeros(cfg.num_frames, dtype=np.float32)
    ft_capacity_proxy_left = np.zeros(cfg.num_frames, dtype=np.float32)
    ft_capacity_proxy_right = np.zeros(cfg.num_frames, dtype=np.float32)
    stress_proxy = np.zeros(cfg.num_frames, dtype=np.float32)
    stress_proxy_left = np.zeros(cfg.num_frames, dtype=np.float32)
    stress_proxy_right = np.zeros(cfg.num_frames, dtype=np.float32)
    normal_mean = np.zeros((cfg.num_frames, 3), dtype=np.float32)
    normal_concentration = np.zeros(cfg.num_frames, dtype=np.float32)
    force_balance_ratio = np.zeros(cfg.num_frames, dtype=np.float32)
    shear_motion_proxy = np.zeros(cfg.num_frames, dtype=np.float32)
    max_penetration = np.zeros(cfg.num_frames, dtype=np.float32)
    body_q_series = np.zeros((cfg.num_frames, len(labels), 7), dtype=np.float32)
    object_pos = np.zeros((cfg.num_frames, 3), dtype=np.float32)
    center_prev: dict[str, np.ndarray] = {}

    shape_kh = example.model.shape_material_kh.numpy()
    shape_mu = example.model.shape_material_mu.numpy()
    sim_start = time.perf_counter()

    for frame in range(cfg.num_frames):
        example.step()
        wp.synchronize()
        body_q = example.state_0.body_q.numpy().astype(np.float32)
        body_q_series[frame] = body_q
        object_pos[frame] = body_q[object_body, :3]
        object_z[frame] = float(body_q[object_body][2])

        if scene_camera_sensor is not None and scene_camera_rays is not None and scene_camera_color_image is not None:
            from newton.sensors import SensorTiledCamera

            example.model.bvh_refit_shapes(example.state_0)
            transforms, scene_camera_meta = scene_camera_transforms(frame, cfg.num_frames, example.world_count)
            scene_camera_sensor.update(
                example.state_0,
                transforms,
                scene_camera_rays,
                color_image=scene_camera_color_image,
                clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
            )
            rgba = scene_camera_sensor.utils.to_rgba_from_color(scene_camera_color_image).numpy().copy()
            scene_camera_frames.append(
                np.asarray(compose_scene_camera_triptych(rgba, f"{cfg.run_tag} frame={frame}"), dtype=np.uint8)
            )

        hydro = (
            example.collision_pipeline.hydroelastic_sdf.get_contact_surface()
            if example.collision_pipeline.hydroelastic_sdf is not None
            else None
        )
        if hydro is not None:
            n_face = int(hydro.face_contact_count.numpy()[0])
            hydro_face_count[frame] = n_face
            if n_face > 0:
                depths = hydro.contact_surface_depth.numpy()[:n_face].astype(np.float32)
                pairs = hydro.contact_surface_shape_pair.numpy()[:n_face].astype(np.int32)
                vertices = hydro.contact_surface_point.numpy()[: 3 * n_face].astype(np.float32).reshape(n_face, 3, 3)
                centroids = vertices.mean(axis=1)
                penetration = np.maximum(-depths, 0.0)
                max_penetration[frame] = float(penetration.max(initial=0.0))
                for side, body_idx, target in [
                    ("left", left_body, left_maps[frame]),
                    ("right", right_body, right_maps[frame]),
                ]:
                    if body_idx is None:
                        continue
                    mask = pair_has(shape_classes, pairs, f"{side}_pad_or_finger") & (penetration > 0.0)
                    if np.any(mask):
                        local = world_to_body(centroids[mask], body_q[body_idx])
                        weights = penetration[mask]
                        accumulate_map(target, local, weights)
                        center = np.average(local[:, 1:3], axis=0, weights=np.maximum(weights, 1.0e-12))
                        prev = center_prev.get(side)
                        if prev is not None:
                            shear_motion_proxy[frame] += float(np.linalg.norm(center - prev))
                        center_prev[side] = center

        reducer = example.collision_pipeline.hydroelastic_sdf.contact_reduction.reducer
        n_raw = int(reducer.contact_count.numpy()[0])
        raw_contact_count[frame] = n_raw
        if n_raw > 0:
            pairs = reducer.shape_pairs.numpy()[:n_raw].astype(np.int32)
            pd = reducer.position_depth.numpy()[:n_raw].astype(np.float32)
            area = reducer.contact_area.numpy()[:n_raw].astype(np.float32)
            penetration = np.maximum(-pd[:, 3], 0.0)
            k_a = shape_kh[pairs[:, 0]]
            k_b = shape_kh[pairs[:, 1]]
            k_eff = np.where((k_a + k_b) > 0.0, (k_a * k_b) / np.maximum(k_a + k_b, 1.0e-12), 0.0)
            mu_eff = np.maximum(shape_mu[pairs[:, 0]], shape_mu[pairs[:, 1]])
            fn = area * penetration * k_eff
            active = penetration > 0.0
            contact_area_sum[frame] = float(area[active].sum(initial=0.0))
            fn_proxy[frame] = float(fn.sum(initial=0.0))
            ft_capacity_proxy[frame] = float((mu_eff * fn).sum(initial=0.0))
            if contact_area_sum[frame] > 0.0:
                stress_proxy[frame] = float(fn_proxy[frame] / contact_area_sum[frame])
            normals = decode_oct(reducer.normal.numpy()[:n_raw].astype(np.float32))
            weights = np.maximum(fn, 0.0)
            if float(weights.sum(initial=0.0)) <= 0.0:
                weights = np.maximum(area * penetration, 0.0)
            if float(weights.sum(initial=0.0)) > 0.0:
                mean_n = np.average(normals, axis=0, weights=weights)
                mean_n_norm = float(np.linalg.norm(mean_n))
                normal_concentration[frame] = mean_n_norm
                if mean_n_norm > 1.0e-12:
                    normal_mean[frame] = (mean_n / mean_n_norm).astype(np.float32)
            left_mask = pair_has(shape_classes, pairs, "left_pad_or_finger") & (penetration > 0.0)
            right_mask = pair_has(shape_classes, pairs, "right_pad_or_finger") & (penetration > 0.0)
            contact_points = pd[:, :3]
            contact_area_left[frame] = float(area[left_mask].sum(initial=0.0))
            contact_area_right[frame] = float(area[right_mask].sum(initial=0.0))
            fn_proxy_left[frame] = float(fn[left_mask].sum(initial=0.0))
            fn_proxy_right[frame] = float(fn[right_mask].sum(initial=0.0))
            ft_capacity_proxy_left[frame] = float((mu_eff[left_mask] * fn[left_mask]).sum(initial=0.0))
            ft_capacity_proxy_right[frame] = float((mu_eff[right_mask] * fn[right_mask]).sum(initial=0.0))
            if contact_area_left[frame] > 0.0:
                stress_proxy_left[frame] = float(fn_proxy_left[frame] / contact_area_left[frame])
            if contact_area_right[frame] > 0.0:
                stress_proxy_right[frame] = float(fn_proxy_right[frame] / contact_area_right[frame])
            max_side_fn = max(float(fn_proxy_left[frame]), float(fn_proxy_right[frame]))
            if max_side_fn > 0.0:
                force_balance_ratio[frame] = min(float(fn_proxy_left[frame]), float(fn_proxy_right[frame])) / max_side_fn
            for (
                side,
                body_idx,
                side_mask,
                fn_map,
                stress_map,
                deform_map,
                shear_y_map,
                shear_z_map,
                f6_normal,
                f6_ft_capacity,
                f6_combined,
            ) in [
                (
                    "left",
                    left_body,
                    left_mask,
                    left_fn_maps[frame],
                    left_stress_maps[frame],
                    left_deform_maps[frame],
                    left_shear_y_maps[frame],
                    left_shear_z_maps[frame],
                    left_f6_normal_proxy,
                    left_f6_ft_capacity_proxy,
                    left_f6_combined_proxy,
                ),
                (
                    "right",
                    right_body,
                    right_mask,
                    right_fn_maps[frame],
                    right_stress_maps[frame],
                    right_deform_maps[frame],
                    right_shear_y_maps[frame],
                    right_shear_z_maps[frame],
                    right_f6_normal_proxy,
                    right_f6_ft_capacity_proxy,
                    right_f6_combined_proxy,
                ),
            ]:
                if body_idx is None or not np.any(side_mask):
                    continue
                local = world_to_body(contact_points[side_mask], body_q[body_idx])
                side_fn = np.maximum(fn[side_mask], 0.0)
                side_area = np.maximum(area[side_mask], 1.0e-12)
                side_penetration = np.maximum(penetration[side_mask], 0.0)
                side_stress = side_fn / side_area
                side_points = contact_points[side_mask]
                side_normals = normals[side_mask]
                side_normal_forces = side_normals * side_fn[:, None]
                side_origin = body_q[body_idx, :3]
                f6_normal[frame] = accumulate_wrench(side_points, side_normal_forces, side_origin)
                accumulate_gaussian_scalar(fn_map, local, side_fn)
                accumulate_gaussian_scalar(stress_map, local, side_stress)
                accumulate_gaussian_scalar(deform_map, local, side_penetration)
                center = np.average(local[:, 1:3], axis=0, weights=np.maximum(side_fn, 1.0e-12))
                prev = center_prev.get(f"{side}_reducer")
                if prev is not None:
                    shear_vec = center - prev
                    local_tangent = np.array([[0.0, shear_vec[0], shear_vec[1]]], dtype=np.float32)
                    world_tangent = body_vector_to_world(local_tangent, body_q[body_idx])[0]
                    tangent_norm = float(np.linalg.norm(world_tangent))
                    if tangent_norm > 1.0e-12:
                        tangent_dir = world_tangent / tangent_norm
                        side_ft_capacity = np.maximum(mu_eff[side_mask] * side_fn, 0.0)
                        side_tangent_forces = tangent_dir[None, :] * side_ft_capacity[:, None]
                        f6_ft_capacity[frame] = accumulate_wrench(side_points, side_tangent_forces, side_origin)
                    accumulate_gaussian_vector(
                        shear_y_map,
                        shear_z_map,
                        local,
                        np.maximum(side_fn, 0.0),
                        shear_vec,
                    )
                else:
                    shear_vec = np.zeros(2, dtype=np.float32)
                center_prev[f"{side}_reducer"] = center
                f6_combined[frame] = f6_normal[frame] + f6_ft_capacity[frame]
                sample = {
                    "local": local.astype(np.float32),
                    "fn": side_fn.astype(np.float32),
                    "stress": side_stress.astype(np.float32),
                    "penetration": side_penetration.astype(np.float32),
                    "shear_vec": np.asarray(shear_vec, dtype=np.float32),
                }
                if side == "left":
                    left_frame_samples[frame].append(sample)
                else:
                    right_frame_samples[frame].append(sample)
            left_shear_magnitude_maps[frame] = np.sqrt(
                left_shear_y_maps[frame] * left_shear_y_maps[frame]
                + left_shear_z_maps[frame] * left_shear_z_maps[frame]
            )
            right_shear_magnitude_maps[frame] = np.sqrt(
                right_shear_y_maps[frame] * right_shear_y_maps[frame]
                + right_shear_z_maps[frame] * right_shear_z_maps[frame]
            )

    left_calib_center, left_calib_extent, left_calib_valid = contact_view_window(
        [sample for frame_samples in left_frame_samples for sample in frame_samples]
    )
    right_calib_center, right_calib_extent, right_calib_valid = contact_view_window(
        [sample for frame_samples in right_frame_samples for sample in frame_samples]
    )
    left_fn_maps_calib = np.zeros_like(left_maps)
    right_fn_maps_calib = np.zeros_like(left_maps)
    left_stress_maps_calib = np.zeros_like(left_maps)
    right_stress_maps_calib = np.zeros_like(left_maps)
    left_deform_maps_calib = np.zeros_like(left_maps)
    right_deform_maps_calib = np.zeros_like(left_maps)
    left_shear_y_maps_calib = np.zeros_like(left_maps)
    left_shear_z_maps_calib = np.zeros_like(left_maps)
    right_shear_y_maps_calib = np.zeros_like(left_maps)
    right_shear_z_maps_calib = np.zeros_like(left_maps)
    left_shear_magnitude_maps_calib = np.zeros_like(left_maps)
    right_shear_magnitude_maps_calib = np.zeros_like(left_maps)
    for frame in range(cfg.num_frames):
        for sample in left_frame_samples[frame]:
            accumulate_gaussian_scalar(left_fn_maps_calib[frame], sample["local"], sample["fn"], extent=left_calib_extent, center_yz=left_calib_center)
            accumulate_gaussian_scalar(left_stress_maps_calib[frame], sample["local"], sample["stress"], extent=left_calib_extent, center_yz=left_calib_center)
            accumulate_gaussian_scalar(left_deform_maps_calib[frame], sample["local"], sample["penetration"], extent=left_calib_extent, center_yz=left_calib_center)
            accumulate_gaussian_vector(
                left_shear_y_maps_calib[frame],
                left_shear_z_maps_calib[frame],
                sample["local"],
                sample["fn"],
                sample["shear_vec"],
                extent=left_calib_extent,
                center_yz=left_calib_center,
            )
        for sample in right_frame_samples[frame]:
            accumulate_gaussian_scalar(right_fn_maps_calib[frame], sample["local"], sample["fn"], extent=right_calib_extent, center_yz=right_calib_center)
            accumulate_gaussian_scalar(right_stress_maps_calib[frame], sample["local"], sample["stress"], extent=right_calib_extent, center_yz=right_calib_center)
            accumulate_gaussian_scalar(right_deform_maps_calib[frame], sample["local"], sample["penetration"], extent=right_calib_extent, center_yz=right_calib_center)
            accumulate_gaussian_vector(
                right_shear_y_maps_calib[frame],
                right_shear_z_maps_calib[frame],
                sample["local"],
                sample["fn"],
                sample["shear_vec"],
                extent=right_calib_extent,
                center_yz=right_calib_center,
            )
        left_shear_magnitude_maps_calib[frame] = np.sqrt(
            left_shear_y_maps_calib[frame] * left_shear_y_maps_calib[frame]
            + left_shear_z_maps_calib[frame] * left_shear_z_maps_calib[frame]
        )
        right_shear_magnitude_maps_calib[frame] = np.sqrt(
            right_shear_y_maps_calib[frame] * right_shear_y_maps_calib[frame]
            + right_shear_z_maps_calib[frame] * right_shear_z_maps_calib[frame]
        )

    example.test_final()
    sim_end = time.perf_counter()
    viewer.close()

    frame_dt = float(getattr(example, "frame_dt", 1.0 / 60.0))
    initial_z = float(object_z[0])
    lift_threshold_m = 0.15
    drop_threshold_m = 0.05
    lifted_mask = object_z >= initial_z + lift_threshold_m
    lift_success = bool(lifted_mask.any())
    first_lift_frame = int(np.argmax(lifted_mask)) if lift_success else None
    if lift_success and first_lift_frame is not None:
        after_lift_z = object_z[first_lift_frame:]
        hold_frames_above_threshold = int((after_lift_z >= initial_z + lift_threshold_m).sum())
        max_lift_frame = int(object_z.argmax())
        max_drop_after_lift_m = float(object_z[max_lift_frame:].max(initial=object_z[max_lift_frame]) - object_z[max_lift_frame:].min(initial=object_z[max_lift_frame]))
        drop_detected = bool((object_z[first_lift_frame:] < initial_z + drop_threshold_m).any())
        xy_anchor = object_pos[first_lift_frame, :2]
        xy_drift_after_lift_m = float(np.linalg.norm(object_pos[first_lift_frame:, :2] - xy_anchor, axis=1).max(initial=0.0))
    else:
        hold_frames_above_threshold = 0
        max_drop_after_lift_m = 0.0
        drop_detected = False
        xy_drift_after_lift_m = 0.0
    if cfg.num_frames >= 3:
        velocity = np.gradient(object_pos, frame_dt, axis=0)
        acceleration = np.gradient(velocity, frame_dt, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        accel = np.linalg.norm(acceleration, axis=1)
        z_accel = np.abs(acceleration[:, 2])
    else:
        speed = np.zeros(cfg.num_frames, dtype=np.float32)
        accel = np.zeros(cfg.num_frames, dtype=np.float32)
        z_accel = np.zeros(cfg.num_frames, dtype=np.float32)
    shear_jump = np.abs(np.diff(shear_motion_proxy, prepend=shear_motion_proxy[0]))
    area_jump = np.abs(np.diff(contact_area_sum, prepend=contact_area_sum[0]))
    sim_wall_s = sim_end - sim_start
    cell_count = float(cfg.map_size * cfg.map_size)

    def max_cell_ratio(maps: np.ndarray) -> float:
        return float(((maps > 0.0).sum(axis=(1, 2)) / cell_count).max(initial=0.0))

    npz_path = cfg.output_dir / "sync_hydro_timeseries.npz"
    np.savez_compressed(
        npz_path,
        left_pressure_map=left_maps,
        right_pressure_map=right_maps,
        left_fn_map=left_fn_maps,
        right_fn_map=right_fn_maps,
        left_stress_map=left_stress_maps,
        right_stress_map=right_stress_maps,
        left_deform_proxy_map=left_deform_maps,
        right_deform_proxy_map=right_deform_maps,
        left_shear_vector_y_map=left_shear_y_maps,
        left_shear_vector_z_map=left_shear_z_maps,
        right_shear_vector_y_map=right_shear_y_maps,
        right_shear_vector_z_map=right_shear_z_maps,
        left_shear_magnitude_map=left_shear_magnitude_maps,
        right_shear_magnitude_map=right_shear_magnitude_maps,
        left_calibrated_view_fn_map=left_fn_maps_calib,
        right_calibrated_view_fn_map=right_fn_maps_calib,
        left_calibrated_view_stress_map=left_stress_maps_calib,
        right_calibrated_view_stress_map=right_stress_maps_calib,
        left_calibrated_view_deform_proxy_map=left_deform_maps_calib,
        right_calibrated_view_deform_proxy_map=right_deform_maps_calib,
        left_calibrated_view_shear_vector_y_map=left_shear_y_maps_calib,
        left_calibrated_view_shear_vector_z_map=left_shear_z_maps_calib,
        right_calibrated_view_shear_vector_y_map=right_shear_y_maps_calib,
        right_calibrated_view_shear_vector_z_map=right_shear_z_maps_calib,
        left_calibrated_view_shear_magnitude_map=left_shear_magnitude_maps_calib,
        right_calibrated_view_shear_magnitude_map=right_shear_magnitude_maps_calib,
        left_calibrated_view_center_yz=left_calib_center,
        right_calibrated_view_center_yz=right_calib_center,
        left_calibrated_view_extent_yz=np.asarray(left_calib_extent, dtype=np.float32),
        right_calibrated_view_extent_yz=np.asarray(right_calib_extent, dtype=np.float32),
        left_f6_normal_proxy=left_f6_normal_proxy,
        right_f6_normal_proxy=right_f6_normal_proxy,
        left_f6_ft_capacity_proxy=left_f6_ft_capacity_proxy,
        right_f6_ft_capacity_proxy=right_f6_ft_capacity_proxy,
        left_f6_combined_proxy=left_f6_combined_proxy,
        right_f6_combined_proxy=right_f6_combined_proxy,
        object_z=object_z,
        hydro_face_count=hydro_face_count,
        raw_contact_count=raw_contact_count,
        contact_area_sum=contact_area_sum,
        contact_area_left=contact_area_left,
        contact_area_right=contact_area_right,
        fn_proxy=fn_proxy,
        fn_proxy_left=fn_proxy_left,
        fn_proxy_right=fn_proxy_right,
        ft_capacity_proxy=ft_capacity_proxy,
        ft_capacity_proxy_left=ft_capacity_proxy_left,
        ft_capacity_proxy_right=ft_capacity_proxy_right,
        stress_proxy=stress_proxy,
        stress_proxy_left=stress_proxy_left,
        stress_proxy_right=stress_proxy_right,
        normal_mean=normal_mean,
        normal_concentration=normal_concentration,
        force_balance_ratio=force_balance_ratio,
        shear_motion_proxy=shear_motion_proxy,
        max_penetration=max_penetration,
        object_pos=object_pos,
        object_speed=speed,
        object_acceleration=accel,
        object_z_acceleration=z_accel,
        body_q=body_q_series,
    )

    render_start = time.perf_counter()
    frames = render_sync_frames(
        cfg,
        body_q_series,
        labels,
        object_body,
        cup_body,
        left_body,
        right_body,
        left_maps,
        right_maps,
        left_fn_maps_calib,
        right_fn_maps_calib,
        left_deform_maps_calib,
        right_deform_maps_calib,
        left_shear_y_maps_calib,
        left_shear_z_maps_calib,
        right_shear_y_maps_calib,
        right_shear_z_maps_calib,
        object_z,
        contact_area_sum,
        fn_proxy,
        stress_proxy,
        ft_capacity_proxy,
        shear_motion_proxy,
        scene_camera_frames,
    )
    video_path = cfg.visual_dir / "sync_scene_tactile.mp4"
    write_mp4_video(video_path, frames, cfg.fps)
    sheet_idx = np.linspace(0, cfg.num_frames - 1, 8, dtype=int)
    sheet = Image.new("RGB", (frames[0].shape[1] * 2, frames[0].shape[0] * 4), (236, 236, 230))
    for slot, idx in enumerate(sheet_idx):
        img = Image.fromarray(frames[int(idx)])
        img = img.resize((frames[0].shape[1], frames[0].shape[0]))
        sheet.paste(img, ((slot % 2) * frames[0].shape[1], (slot // 2) * frames[0].shape[0]))
    sheet_path = cfg.visual_dir / "sync_scene_tactile_sheet.jpg"
    sheet.save(sheet_path, quality=92)
    render_end = time.perf_counter()
    total_end = time.perf_counter()

    instrumented_fps = float(cfg.num_frames / sim_wall_s) if sim_wall_s > 0.0 else 0.0
    render_fps = float(cfg.num_frames / (render_end - render_start)) if render_end > render_start else 0.0
    if scene_camera_frames:
        scene_camera_stack = np.stack(scene_camera_frames, axis=0)
        scene_camera_pixel_std = float(scene_camera_stack.std())
        scene_camera_nonblank = bool(scene_camera_stack.max() > scene_camera_stack.min() and scene_camera_pixel_std > 1.0)
        scene_camera_frame_shape = list(scene_camera_frames[0].shape)
    else:
        scene_camera_pixel_std = 0.0
        scene_camera_nonblank = False
        scene_camera_frame_shape = None

    summary = {
        "classification": "phase00_sync_newton_hydro_scene_tactile_diagnostic_v1",
        "run_tag": cfg.run_tag,
        "status": "pass",
        "not_training_result": True,
        "not_curiosity_success": True,
        "official_example": "newton.examples.robot.example_robot_panda_hydro",
        "material_label": cfg.material_label,
        "material_override_applied": material_override_applied,
        "requested_override_mu": cfg.override_mu,
        "requested_override_kh": cfg.override_kh,
        "scene_panel": (
            "synchronized Newton SensorTiledCamera head/right_wrist/left_wrist scene frames"
            if scene_camera_frames is not None
            else "synchronized schematic top view from Newton body_q, not photoreal render"
        ),
        "scene_camera_enabled": cfg.scene_camera,
        "scene_camera_nonblank": scene_camera_nonblank,
        "scene_camera_pixel_std": scene_camera_pixel_std,
        "scene_camera_frame_shape": scene_camera_frame_shape,
        "scene_camera_meta_last": scene_camera_meta,
        "scene_camera_width": cfg.scene_camera_width,
        "scene_camera_height": cfg.scene_camera_height,
        "tactile_representation": "hydro_proxy raw pressure maps plus reducer-derived Gaussian grid maps for Fn, stress, deformation, and contact-center-motion shear vectors in finger local y/z",
        "rendered_tactile_view": "calibrated visualization view using per-run 1%-99% contact local-yz window; raw maps remain exported separately",
        "normal_force_field": "hydro_proxy.Fn = contact_area * penetration * effective_hydro_stiffness from Newton reducer buffers",
        "stress_field": "hydro_proxy.stress = hydro_proxy.Fn / contact_area, proxy pressure/stress from Newton hydro reducer buffers",
        "normal_field": "hydro_proxy.normal_mean = force-weighted mean normal decoded from Newton hydro reducer octahedral normals",
        "deformation_field": "hydro_proxy.deform_proxy_map = Gaussian-spread reducer penetration/compression in finger local tactile plane",
        "tangential_field": "hydro_proxy.Ft_capacity = max(shape_material_mu_pair) * hydro_proxy.Fn; hydro_proxy.shear_vector_map = Gaussian-spread frame-to-frame weighted contact-center shift in finger-local tactile plane",
        "f6_proxy_field": "hydro_proxy.F6 normal wrench from reducer normal forces plus Ft-capacity wrench along contact-center-motion tangent; this is T-Rex-aligned export shape, not official T-Rex tactile force",
        "grid_tactile_maps": [
            "left_pressure_map",
            "right_pressure_map",
            "left_fn_map",
            "right_fn_map",
            "left_stress_map",
            "right_stress_map",
            "left_deform_proxy_map",
            "right_deform_proxy_map",
            "left_shear_vector_y_map",
            "left_shear_vector_z_map",
            "right_shear_vector_y_map",
            "right_shear_vector_z_map",
            "left_shear_magnitude_map",
            "right_shear_magnitude_map",
            "left_calibrated_view_fn_map",
            "right_calibrated_view_fn_map",
            "left_calibrated_view_stress_map",
            "right_calibrated_view_stress_map",
            "left_calibrated_view_deform_proxy_map",
            "right_calibrated_view_deform_proxy_map",
            "left_calibrated_view_shear_vector_y_map",
            "left_calibrated_view_shear_vector_z_map",
            "right_calibrated_view_shear_vector_y_map",
            "right_calibrated_view_shear_vector_z_map",
            "left_calibrated_view_shear_magnitude_map",
            "right_calibrated_view_shear_magnitude_map",
        ],
        "f6_proxy_arrays": [
            "left_f6_normal_proxy",
            "right_f6_normal_proxy",
            "left_f6_ft_capacity_proxy",
            "right_f6_ft_capacity_proxy",
            "left_f6_combined_proxy",
            "right_f6_combined_proxy",
        ],
        "missing_for_reference_level": [
            "direct solver tangential force Ft",
            "direct pad-resolved shear force vector field",
            "photoreal or direct USD-raster scene panel",
            "validated gel/marker-style tactile camera output comparable to the reference video",
        ],
        "num_frames": cfg.num_frames,
        "fps": cfg.fps,
        "frame_dt_s": frame_dt,
        "npz_path": str(npz_path),
        "video_path": str(video_path),
        "sheet_path": str(sheet_path),
        "simulation_wall_s_instrumented": sim_wall_s,
        "instrumented_sim_export_fps": instrumented_fps,
        "render_wall_s": render_end - render_start,
        "render_fps": render_fps,
        "total_wall_s": total_end - total_start,
        "fps_note": "FPS is instrumented Python export throughput with per-frame GPU synchronization and video rendering separated; it is not a clean Newton solver benchmark.",
        "max_object_lift_m": float(object_z.max(initial=object_z[0]) - object_z[0]),
        "lift_success_threshold_m": lift_threshold_m,
        "lift_success": lift_success,
        "first_lift_frame": first_lift_frame,
        "hold_frames_above_lift_threshold": hold_frames_above_threshold,
        "drop_detected_after_lift": drop_detected,
        "max_drop_after_lift_m": max_drop_after_lift_m,
        "final_object_height_m": float(object_z[-1]),
        "xy_drift_after_lift_m": xy_drift_after_lift_m,
        "max_object_speed_m_per_s": float(speed.max(initial=0.0)),
        "max_object_accel_m_per_s2": float(accel.max(initial=0.0)),
        "max_object_z_accel_m_per_s2": float(z_accel.max(initial=0.0)),
        "max_shear_motion_jump_proxy_m": float(shear_jump.max(initial=0.0)),
        "max_contact_area_jump_proxy_m2": float(area_jump.max(initial=0.0)),
        "max_hydro_face_count": int(hydro_face_count.max(initial=0)),
        "max_raw_hydro_contact_count": int(raw_contact_count.max(initial=0)),
        "max_contact_area_sum_m2": float(contact_area_sum.max(initial=0.0)),
        "max_fn_proxy": float(fn_proxy.max(initial=0.0)),
        "max_fn_proxy_left": float(fn_proxy_left.max(initial=0.0)),
        "max_fn_proxy_right": float(fn_proxy_right.max(initial=0.0)),
        "max_stress_proxy": float(stress_proxy.max(initial=0.0)),
        "max_stress_proxy_left": float(stress_proxy_left.max(initial=0.0)),
        "max_stress_proxy_right": float(stress_proxy_right.max(initial=0.0)),
        "max_ft_capacity_proxy": float(ft_capacity_proxy.max(initial=0.0)),
        "max_ft_capacity_proxy_left": float(ft_capacity_proxy_left.max(initial=0.0)),
        "max_ft_capacity_proxy_right": float(ft_capacity_proxy_right.max(initial=0.0)),
        "max_normal_concentration": float(normal_concentration.max(initial=0.0)),
        "mean_force_balance_ratio_when_active": float(force_balance_ratio[force_balance_ratio > 0.0].mean()) if np.any(force_balance_ratio > 0.0) else 0.0,
        "max_shear_motion_proxy_m": float(shear_motion_proxy.max(initial=0.0)),
        "max_left_fn_map": float(left_fn_maps.max(initial=0.0)),
        "max_right_fn_map": float(right_fn_maps.max(initial=0.0)),
        "max_left_stress_map": float(left_stress_maps.max(initial=0.0)),
        "max_right_stress_map": float(right_stress_maps.max(initial=0.0)),
        "max_left_deform_proxy_map": float(left_deform_maps.max(initial=0.0)),
        "max_right_deform_proxy_map": float(right_deform_maps.max(initial=0.0)),
        "max_left_shear_magnitude_map": float(left_shear_magnitude_maps.max(initial=0.0)),
        "max_right_shear_magnitude_map": float(right_shear_magnitude_maps.max(initial=0.0)),
        "left_calibrated_view_valid": left_calib_valid,
        "right_calibrated_view_valid": right_calib_valid,
        "left_calibrated_view_center_yz": left_calib_center.tolist(),
        "right_calibrated_view_center_yz": right_calib_center.tolist(),
        "left_calibrated_view_extent_yz": [float(v) for v in left_calib_extent],
        "right_calibrated_view_extent_yz": [float(v) for v in right_calib_extent],
        "max_left_calibrated_view_fn_map": float(left_fn_maps_calib.max(initial=0.0)),
        "max_right_calibrated_view_fn_map": float(right_fn_maps_calib.max(initial=0.0)),
        "max_left_calibrated_view_shear_magnitude_map": float(left_shear_magnitude_maps_calib.max(initial=0.0)),
        "max_right_calibrated_view_shear_magnitude_map": float(right_shear_magnitude_maps_calib.max(initial=0.0)),
        "max_left_raw_fn_nonzero_cell_ratio": max_cell_ratio(left_fn_maps),
        "max_right_raw_fn_nonzero_cell_ratio": max_cell_ratio(right_fn_maps),
        "max_left_calibrated_fn_nonzero_cell_ratio": max_cell_ratio(left_fn_maps_calib),
        "max_right_calibrated_fn_nonzero_cell_ratio": max_cell_ratio(right_fn_maps_calib),
        "max_left_f6_normal_proxy_norm": float(np.linalg.norm(left_f6_normal_proxy, axis=1).max(initial=0.0)),
        "max_right_f6_normal_proxy_norm": float(np.linalg.norm(right_f6_normal_proxy, axis=1).max(initial=0.0)),
        "max_left_f6_ft_capacity_proxy_norm": float(np.linalg.norm(left_f6_ft_capacity_proxy, axis=1).max(initial=0.0)),
        "max_right_f6_ft_capacity_proxy_norm": float(np.linalg.norm(right_f6_ft_capacity_proxy, axis=1).max(initial=0.0)),
        "max_left_f6_combined_proxy_norm": float(np.linalg.norm(left_f6_combined_proxy, axis=1).max(initial=0.0)),
        "max_right_f6_combined_proxy_norm": float(np.linalg.norm(right_f6_combined_proxy, axis=1).max(initial=0.0)),
        "left_active_frames": int((left_maps.sum(axis=(1, 2)) > 0.0).sum()),
        "right_active_frames": int((right_maps.sum(axis=(1, 2)) > 0.0).sum()),
        "left_grid_fn_active_frames": int((left_fn_maps.sum(axis=(1, 2)) > 0.0).sum()),
        "right_grid_fn_active_frames": int((right_fn_maps.sum(axis=(1, 2)) > 0.0).sum()),
        "left_grid_shear_active_frames": int((left_shear_magnitude_maps.sum(axis=(1, 2)) > 0.0).sum()),
        "right_grid_shear_active_frames": int((right_shear_magnitude_maps.sum(axis=(1, 2)) > 0.0).sum()),
        "observed_shape_material_mu_unique": sorted({float(v) for v in shape_mu.tolist()}),
        "observed_shape_material_kh_unique": sorted({float(v) for v in shape_kh.tolist()}),
        "body_labels": labels,
    }
    summary_path = cfg.output_dir / "sync_hydro_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = cfg.report_dir / "sync_hydro_diagnostic.md"
    report.write_text(
        "# Phase 00 Synchronized Newton Hydro Diagnostic\n\n"
        f"- run_tag: `{cfg.run_tag}`\n"
        f"- status: `{summary['status']}`\n"
        f"- video: `{video_path}`\n"
        f"- sheet: `{sheet_path}`\n"
        f"- source arrays: `{npz_path}`\n"
        f"- scene camera enabled: `{summary['scene_camera_enabled']}`\n"
        f"- scene camera nonblank: `{summary['scene_camera_nonblank']}`\n"
        f"- scene camera pixel std: `{summary['scene_camera_pixel_std']:.6g}`\n"
        f"- max object lift m: `{summary['max_object_lift_m']:.6f}`\n"
        f"- max contact area sum m2: `{summary['max_contact_area_sum_m2']:.6g}`\n"
        f"- max Fn proxy: `{summary['max_fn_proxy']:.6g}`\n"
        f"- max stress proxy: `{summary['max_stress_proxy']:.6g}`\n"
        f"- max Ft capacity proxy: `{summary['max_ft_capacity_proxy']:.6g}`\n"
        f"- max shear motion proxy m: `{summary['max_shear_motion_proxy_m']:.6g}`\n"
        f"- max left/right grid Fn map: `{summary['max_left_fn_map']:.6g}` / `{summary['max_right_fn_map']:.6g}`\n"
        f"- max left/right calibrated-view Fn map: `{summary['max_left_calibrated_view_fn_map']:.6g}` / `{summary['max_right_calibrated_view_fn_map']:.6g}`\n"
        f"- max left/right calibrated Fn cell ratio: `{summary['max_left_calibrated_fn_nonzero_cell_ratio']:.6g}` / `{summary['max_right_calibrated_fn_nonzero_cell_ratio']:.6g}`\n"
        f"- max left/right grid shear magnitude map: `{summary['max_left_shear_magnitude_map']:.6g}` / `{summary['max_right_shear_magnitude_map']:.6g}`\n"
        f"- max left/right F6 combined proxy norm: `{summary['max_left_f6_combined_proxy_norm']:.6g}` / `{summary['max_right_f6_combined_proxy_norm']:.6g}`\n"
        f"- lift success over {lift_threshold_m:.2f} m: `{summary['lift_success']}`\n"
        f"- hold frames above threshold: `{summary['hold_frames_above_lift_threshold']}`\n"
        f"- drop detected after lift: `{summary['drop_detected_after_lift']}`\n"
        f"- max object acceleration m/s^2: `{summary['max_object_accel_m_per_s2']:.6g}`\n"
        f"- instrumented sim/export fps: `{summary['instrumented_sim_export_fps']:.6g}`\n"
        "\nThis is environment/base diagnostic evidence, not training and not curiosity success.\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str]) -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/public/home/yanhongru/Curiosity"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=["cube", "pen"], default="cube")
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--material-label", default="official_default")
    parser.add_argument("--override-mu", type=float, default=None)
    parser.add_argument("--override-kh", type=float, default=None)
    parser.add_argument("--scene-camera", action="store_true")
    parser.add_argument("--scene-camera-width", type=int, default=256)
    parser.add_argument("--scene-camera-height", type=int, default=256)
    args = parser.parse_args(argv)
    return Config(
        root=args.root,
        run_tag=args.run_tag,
        output_dir=args.output_dir,
        visual_dir=args.visual_dir,
        report_dir=args.report_dir,
        device=args.device,
        scene=args.scene,
        num_frames=args.num_frames,
        map_size=args.map_size,
        fps=args.fps,
        material_label=args.material_label,
        override_mu=args.override_mu,
        override_kh=args.override_kh,
        scene_camera=args.scene_camera,
        scene_camera_width=args.scene_camera_width,
        scene_camera_height=args.scene_camera_height,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.visual_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    summary = run(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
