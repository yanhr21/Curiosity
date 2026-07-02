#!/usr/bin/env python3
"""Create contrast-enhanced tactile visuals from a Phase 00 tactile NPZ."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def colorize(values: np.ndarray, vmax: float) -> np.ndarray:
    x = np.clip(values.astype(np.float32) / max(vmax, 1.0e-12), 0.0, 1.0)
    x = np.power(x, 0.35)
    r = np.clip(2.2 * x, 0.0, 1.0)
    g = np.clip(1.7 * x - 0.15, 0.0, 1.0)
    b = np.clip(1.0 - 1.6 * x, 0.0, 0.25) + 0.15 * x
    return (255.0 * np.stack([r, g, b], axis=-1)).astype(np.uint8)


def resize_nn(img: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def frame_image(left: np.ndarray, right: np.ndarray, idx: int, vmax: float, scale: int) -> np.ndarray:
    gap = 12
    top = 26
    left_rgb = resize_nn(colorize(left, vmax), scale)
    right_rgb = resize_nn(colorize(right, vmax), scale)
    h, w = left_rgb.shape[:2]
    canvas = np.full((h + top, 2 * w + gap, 3), 20, dtype=np.uint8)
    canvas[top:, :w] = left_rgb
    canvas[top:, w + gap :] = right_rgb
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((6, 5), f"frame {idx}  left", fill=(235, 235, 235))
    draw.text((w + gap + 6, 5), "right", fill=(235, 235, 235))
    return np.asarray(img, dtype=np.uint8)


def write_uncompressed_avi(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=10)
    args = parser.parse_args()

    args.visual_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz)
    left = data["left_pressure_map"]
    right = data["right_pressure_map"]
    mass = data["left_pressure_mass"] + data["right_pressure_mass"]
    nonzero = np.concatenate([left[left > 0.0], right[right > 0.0]])
    vmax = float(np.quantile(nonzero, 0.95)) if nonzero.size else 1.0
    active = np.where(mass > 0.0)[0]
    if active.size:
        sample_frames = np.linspace(int(active[0]), int(active[-1]), min(8, active.size), dtype=int)
    else:
        sample_frames = np.linspace(0, left.shape[0] - 1, min(8, left.shape[0]), dtype=int)

    video_frames = [frame_image(left[i], right[i], i, vmax, args.scale) for i in range(left.shape[0])]
    avi_path = args.visual_dir / "tactile_maps_enhanced.avi"
    write_uncompressed_avi(avi_path, video_frames, fps=30)

    tiles = [frame_image(left[i], right[i], int(i), vmax, args.scale) for i in sample_frames]
    gap = 10
    h, w = tiles[0].shape[:2]
    sheet = np.full((h, len(tiles) * w + (len(tiles) - 1) * gap, 3), 20, dtype=np.uint8)
    for col, tile in enumerate(tiles):
        x0 = col * (w + gap)
        sheet[:, x0 : x0 + w] = tile
    sheet_path = args.visual_dir / "tactile_sheet_enhanced.png"
    Image.fromarray(sheet).save(sheet_path)

    report = {
        "source_npz": str(args.npz),
        "enhanced_video": str(avi_path),
        "enhanced_sheet": str(sheet_path),
        "display_vmax_quantile_95": vmax,
        "sample_frames": [int(v) for v in sample_frames],
        "left_map_max": float(left.max(initial=0.0)),
        "right_map_max": float(right.max(initial=0.0)),
        "active_frames": int(active.size),
    }
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
