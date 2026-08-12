#!/usr/bin/env python3
"""Render SUGAR CarryBox plus both complete anatomical tactile hands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


PATCHES = (
    *(f"palm_r{row}_c{column}" for row in range(4) for column in range(3)),
    *(
        f"{digit}_{segment}"
        for digit in ("thumb", "index", "middle", "ring", "little")
        for segment in ("proximal", "middle", "distal")
    ),
)
DIGITS = ("thumb", "index", "middle", "ring", "little")
SEGMENTS_TOP_DOWN = ("distal", "middle", "proximal")
WIDTH, HEIGHT = 2560, 1440
WORLD_HEIGHT = 720
HAND_WIDTH = WIDTH // 2
SLIP_SHORT = ("-", "K", "I", "G")
SLIP_BORDER_BGR = (
    (155, 155, 155),
    (55, 145, 55),
    (0, 155, 230),
    (40, 40, 210),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--normal-max", type=float, default=None)
    parser.add_argument("--shear-max", type=float, default=None)
    parser.add_argument("--scale-note", default="per-trace automatic quantile")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    return parser.parse_args()


def signed_normal_bgr(values: np.ndarray, maximum: float) -> np.ndarray:
    """Map the untouched signed local-Z force to a diverging color field."""

    scaled = np.clip(np.asarray(values, np.float32) / maximum, -1.0, 1.0)
    magnitude = np.abs(scaled)
    image = np.full((*scaled.shape, 3), 255, dtype=np.uint8)
    negative = scaled < 0.0
    positive = scaled > 0.0
    # Negative local Z (compression in this mounted geometry) is red;
    # positive local Z is blue.  Neither sign is discarded or redefined.
    image[..., 0][negative] = np.rint(255.0 * (1.0 - magnitude[negative])).astype(np.uint8)
    image[..., 1][negative] = np.rint(255.0 * (1.0 - 0.70 * magnitude[negative])).astype(np.uint8)
    image[..., 1][positive] = np.rint(255.0 * (1.0 - 0.70 * magnitude[positive])).astype(np.uint8)
    image[..., 2][positive] = np.rint(255.0 * (1.0 - magnitude[positive])).astype(np.uint8)
    return image


def quat_apply_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[..., :3]
    cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[..., 3:] * cross + np.cross(xyz, cross)


def put(
    image: np.ndarray,
    text: str,
    point: tuple[int, int],
    scale: float = 0.45,
    thickness: int = 1,
    color: tuple[int, int, int] = (25, 25, 25),
) -> None:
    cv2.putText(
        image,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_tile(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    normal: np.ndarray,
    shear: np.ndarray,
    penetration: np.ndarray,
    slip_state: int,
    normal_max: float,
    shear_max: float,
) -> None:
    contact_mask = penetration > 0.0
    active = int(np.count_nonzero(contact_mask))
    border_color = SLIP_BORDER_BGR[int(slip_state)] if active else SLIP_BORDER_BGR[0]
    border_thickness = 3 if active else 1
    cv2.rectangle(
        canvas,
        (x, y),
        (x + w, y + h),
        border_color,
        border_thickness,
    )
    peak = float(normal.reshape(-1)[np.argmax(np.abs(normal))])
    put(
        canvas,
        f"{label} {SLIP_SHORT[int(slip_state)]} n={active:03d} Z={peak:+.2f}",
        (x + 4, y + 16),
        0.31,
    )
    map_y = y + 22
    map_h = h - 26
    heat = signed_normal_bgr(normal, normal_max)
    heat = cv2.resize(heat, (w - 6, map_h), interpolation=cv2.INTER_NEAREST)
    canvas[map_y : map_y + map_h, x + 3 : x + w - 3] = heat
    cv2.rectangle(
        canvas,
        (x + 3, map_y),
        (x + w - 4, map_y + map_h - 1),
        (120, 120, 120),
        1,
    )
    vector = (
        shear[contact_mask].mean(axis=0)
        if np.any(contact_mask)
        else np.zeros(2, np.float32)
    )
    center = (x + w // 2, map_y + map_h // 2)
    delta = np.clip(vector / shear_max, -1.0, 1.0) * 18.0
    # Released TacSL order is row/local-X then column/local-Y.  Video X is
    # therefore shear Y; video Y is shear X (up is positive).
    end = (int(round(center[0] + delta[1])), int(round(center[1] - delta[0])))
    if end != center:
        cv2.arrowedLine(canvas, center, end, (20, 20, 20), 2, cv2.LINE_AA, 0, 0.30)
    if active:
        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            border_color,
            border_thickness,
        )


def draw_hand(
    canvas: np.ndarray,
    side_index: int,
    normal: np.ndarray,
    shear: np.ndarray,
    penetration: np.ndarray,
    slip_state: np.ndarray,
    normal_max: float,
    shear_max: float,
    center_patch_note: str = "CENTER r1c1 = R15 RGB/DEPTH",
    center_patch_label: str = "R15",
) -> None:
    x0 = side_index * HAND_WIDTH
    side_name = "LEFT HAND" if side_index == 0 else "RIGHT HAND"
    total_active = int(np.count_nonzero(penetration > 0.0))
    active_patches = int(np.count_nonzero(np.any(penetration > 0.0, axis=(-2, -1))))
    slip_counts = [int(np.count_nonzero(slip_state == value)) for value in (1, 2, 3)]
    cv2.rectangle(
        canvas,
        (x0, WORLD_HEIGHT),
        (x0 + HAND_WIDTH - 1, HEIGHT - 1),
        (65, 65, 65),
        1,
    )
    put(
        canvas,
        f"{side_name} | ACTIVE {active_patches:02d} PATCHES / {total_active:04d} TAXELS | SLIP K/I/G {slip_counts[0]:02d}/{slip_counts[1]:02d}/{slip_counts[2]:02d}",
        (x0 + 24, 752),
        0.64,
        2,
    )
    handed_order = (
        "LITTLE  RING  MIDDLE  INDEX   +   THUMB (OUTSIDE)"
        if side_index == 0
        else "THUMB (OUTSIDE)   +   INDEX  MIDDLE  RING  LITTLE"
    )
    put(canvas, handed_order, (x0 + 24, 779), 0.48, 1, (80, 80, 80))

    # The overview is deliberately hand-shaped instead of a generic 5 x 3
    # matrix.  Four upright fingers align with the four palm columns; the
    # thumb is attached on the outside and the entire right hand is mirrored.
    # This keeps all 27 raw 20 x 25 fields visible while making the anatomy
    # immediately legible in a video player.
    tile_w, tile_h, gap = 162, 80, 16
    palm_start = x0 + (HAND_WIDTH - (4 * tile_w + 3 * gap)) // 2
    finger_y = (820, 904, 988)
    upright_digits = (
        ("little", "ring", "middle", "index")
        if side_index == 0
        else ("index", "middle", "ring", "little")
    )
    for column, digit in enumerate(upright_digits):
        x = palm_start + column * (tile_w + gap)
        text_size = cv2.getTextSize(
            digit.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )[0]
        put(
            canvas,
            digit.upper(),
            (x + (tile_w - text_size[0]) // 2, 812),
            0.48,
            1,
            (45, 45, 45),
        )
        for row, segment in enumerate(SEGMENTS_TOP_DOWN):
            patch_name = f"{digit}_{segment}"
            patch_index = PATCHES.index(patch_name)
            y = finger_y[row]
            draw_tile(
                canvas,
                x,
                y,
                tile_w,
                tile_h,
                {"distal": "D", "middle": "M", "proximal": "P"}[segment],
                normal[patch_index],
                shear[patch_index],
                penetration[patch_index],
                int(slip_state[patch_index]),
                normal_max,
                shear_max,
            )

    thumb_x = (
        palm_start + 4 * tile_w + 3 * gap + 42
        if side_index == 0
        else palm_start - tile_w - 42
    )
    thumb_label_size = cv2.getTextSize(
        "THUMB", cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2
    )[0]
    put(
        canvas,
        "THUMB",
        (thumb_x + (tile_w - thumb_label_size[0]) // 2, 812),
        0.54,
        2,
        (45, 45, 45),
    )
    thumb_y = finger_y
    for row, segment in enumerate(SEGMENTS_TOP_DOWN):
        patch_name = f"thumb_{segment}"
        patch_index = PATCHES.index(patch_name)
        draw_tile(
            canvas,
            thumb_x,
            thumb_y[row],
            tile_w,
            tile_h,
            {"distal": "D", "middle": "M", "proximal": "P"}[segment],
            normal[patch_index],
            shear[patch_index],
            penetration[patch_index],
            int(slip_state[patch_index]),
            normal_max,
            shear_max,
        )

    # A short connector makes the separate thumb column read as part of the
    # same hand without covering any taxel field.
    thumb_inner_x = thumb_x if side_index == 0 else thumb_x + tile_w
    palm_outer_x = palm_start + 4 * tile_w + 3 * gap if side_index == 0 else palm_start
    cv2.line(
        canvas,
        (palm_outer_x, 1078),
        (thumb_inner_x, 1078),
        (115, 115, 115),
        2,
        cv2.LINE_AA,
    )

    put(
        canvas,
        f"PALM - 4 ACROSS x 3 FROM FINGERS TO WRIST | {center_patch_note}",
        (palm_start, 1090),
        0.50,
        1,
    )
    palm_y = (1096, 1180, 1264)
    palm_rows_across = (
        (0, 1, 2, 3) if side_index == 0 else (3, 2, 1, 0)
    )
    # In the physical asset, c2 is nearest the fingers and c0 is nearest
    # the wrist.  r0..r3 run little-side to index-side.  The right panel is
    # mirrored only for a natural bilateral palm-facing display; indexing
    # into the archived raw tensor remains unchanged.
    for palm_y_index, column in enumerate((2, 1, 0)):
        for display_column, row in enumerate(palm_rows_across):
            patch_index = row * 3 + column
            draw_tile(
                canvas,
                palm_start + display_column * (tile_w + gap),
                palm_y[palm_y_index],
                tile_w,
                tile_h,
                f"r{row}c{column}",
                normal[patch_index],
                shear[patch_index],
                penetration[patch_index],
                int(slip_state[patch_index]),
                normal_max,
                shear_max,
            )
            if row == 1 and column == 1:
                tile_x = palm_start + display_column * (tile_w + gap)
                cv2.rectangle(
                    canvas,
                    (tile_x + 3, palm_y[palm_y_index] + 22),
                    (tile_x + tile_w - 3, palm_y[palm_y_index] + tile_h - 3),
                    (0, 155, 225),
                    2,
                )
                put(
                    canvas,
                    center_patch_label,
                    (tile_x + tile_w - 39, palm_y[palm_y_index] + tile_h - 8),
                    0.36,
                    1,
                    (0, 110, 180),
                )


def fit_world(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    scale = min(target_w / frame.shape[1], target_h / frame.shape[0])
    resized = cv2.resize(
        frame,
        (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    result = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    x = (target_w - resized.shape[1]) // 2
    y = (target_h - resized.shape[0]) // 2
    result[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return result


def close_crop(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    crop_w = int(round(width * 0.66))
    crop_h = int(round(crop_w * size[1] / size[0]))
    crop_h = min(crop_h, height)
    x0 = (width - crop_w) // 2
    # The fixed SUGAR CarryBox camera keeps the robot, hands, and box in the
    # lower part of the frame.  Bottom alignment makes the second panel a true
    # hand--box close view instead of enlarging empty background.
    y0 = height - crop_h
    crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    trace_path = run_root / "whole_hand_trace.npz"
    world_path = run_root / "world_carrybox.mp4"
    if not trace_path.is_file() or not world_path.is_file():
        raise FileNotFoundError("The trace and world video must both exist")
    with np.load(trace_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    if tuple(arrays["patch_order"].astype(str)) != PATCHES:
        raise RuntimeError("Patch ordering does not match the anatomical layout")
    normal = np.asarray(arrays["normal_force"], np.float32)
    shear = np.asarray(arrays["signed_shear"], np.float32)
    penetration = np.asarray(arrays["penetration"], np.float32)
    slip_state = np.asarray(arrays["tactile_only_slip_state"], np.int8)
    if normal.shape[1:] != (2, 27, 20, 25):
        raise RuntimeError(f"Unexpected normal shape: {normal.shape}")
    if shear.shape[1:] != (2, 27, 20, 25, 2):
        raise RuntimeError(f"Unexpected shear shape: {shear.shape}")
    if penetration.shape != normal.shape:
        raise RuntimeError(f"Unexpected penetration shape: {penetration.shape}")
    if slip_state.shape != normal.shape[:3]:
        raise RuntimeError(f"Unexpected slip-state shape: {slip_state.shape}")
    nonzero_normal = np.abs(normal[normal != 0.0])
    normal_max = args.normal_max or (
        float(np.quantile(nonzero_normal, 0.995)) if len(nonzero_normal) else 1.0
    )
    shear_magnitude = np.linalg.norm(shear, axis=-1)
    positive_shear = shear_magnitude[shear_magnitude > 0.0]
    shear_max = args.shear_max or (
        float(np.quantile(positive_shear, 0.995)) if len(positive_shear) else 1.0
    )
    normal_max = max(normal_max, 1.0e-9)
    shear_max = max(shear_max, 1.0e-9)

    start = int(args.start_frame)
    end = len(normal) if args.end_frame is None else int(args.end_frame)
    if not (0 <= start < end <= len(normal)):
        raise RuntimeError(f"Invalid frame interval [{start}, {end}) for {len(normal)} frames")

    capture = cv2.VideoCapture(str(world_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count != len(normal):
        raise RuntimeError(f"World/tactile frame mismatch: {frame_count} vs {len(normal)}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    object_z = arrays["object_state_w"][:, 2]
    object_velocity = np.asarray(arrays["object_velocity_w"], np.float64)
    gravity = np.asarray(arrays["gravity_w"], np.float64)
    mass = float(summary["box_mass_readback_kg"])
    control_dt = float(arrays["control_dt_s"])
    acceleration = np.full((len(normal), 3), np.nan, dtype=np.float64)
    if len(normal) >= 3:
        acceleration[1:-1] = (
            object_velocity[2:, :3] - object_velocity[:-2, :3]
        ) / (2.0 * control_dt)
    required_force = mass * (acceleration - gravity[None])
    physx_force = -(
        np.asarray(arrays["robot_box_force_w"], np.float64)
        + np.asarray(arrays["robot_box_friction_force_w"], np.float64)
    ).sum(axis=1)
    weight_n = mass * float(np.linalg.norm(gravity))
    try:
        for step in range(start, end):
            ok, world = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode world frame {step}")
            canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
            canvas[:WORLD_HEIGHT, :HAND_WIDTH] = fit_world(
                world, (HAND_WIDTH, WORLD_HEIGHT)
            )
            canvas[:WORLD_HEIGHT, HAND_WIDTH:] = close_crop(
                world, (HAND_WIDTH, WORLD_HEIGHT)
            )
            cv2.rectangle(
                canvas, (0, 0), (WIDTH - 1, WORLD_HEIGHT - 1), (55, 55, 55), 1
            )
            cv2.line(
                canvas,
                (HAND_WIDTH, 0),
                (HAND_WIDTH, WORLD_HEIGHT),
                (255, 255, 255),
                3,
            )
            cv2.rectangle(canvas, (14, 14), (1170, 58), (255, 255, 255), -1)
            put(canvas, args.title, (28, 46), 0.82, 2)
            cv2.rectangle(
                canvas,
                (HAND_WIDTH + 14, 14),
                (WIDTH - 14, 58),
                (255, 255, 255),
                -1,
            )
            put(
                canvas,
                "SAME FRAME - CLOSE VIEW OF BOTH HANDS AND BOX",
                (HAND_WIDTH + 30, 46),
                0.72,
                2,
            )
            lift = float(object_z[step] - object_z[0])
            local_force = np.concatenate(
                (shear[step], normal[step, ..., None]), axis=-1
            )
            tacsl_reaction = -quat_apply_xyzw(
                np.asarray(arrays["taxel_quaternion_w"][step], np.float64),
                np.asarray(local_force, np.float64),
            ).sum(axis=(0, 1, 2, 3))
            cv2.rectangle(
                canvas, (14, 620), (WIDTH - 14, 664), (25, 25, 25), -1
            )
            put(
                canvas,
                (
                    "same-step Fz on box: "
                    f"TacSL reaction {tacsl_reaction[2]:+.2f} N   "
                    f"PhysX normal+friction {physx_force[step, 2]:+.2f} N   "
                    f"required m(a-g) {required_force[step, 2]:+.2f} N   "
                    f"m*g {weight_n:.2f} N"
                ),
                (28, 650),
                0.56,
                1,
                (255, 255, 255),
            )
            cv2.rectangle(
                canvas, (14, 668), (WIDTH - 14, 712), (25, 25, 25), -1
            )
            put(
                canvas,
                f"frame {step:03d}   box lift {lift:+.3f} m   red=negative raw local-Z   blue=positive Z   arrow=signed XY   fixed scales: |Z| {normal_max:.3f} N, |XY| {shear_max:.3f} N",
                (28, 698),
                0.54,
                1,
                (255, 255, 255),
            )
            draw_hand(
                canvas,
                0,
                normal[step, 0],
                shear[step, 0],
                penetration[step, 0],
                slip_state[step, 0],
                normal_max,
                shear_max,
            )
            draw_hand(
                canvas,
                1,
                normal[step, 1],
                shear[step, 1],
                penetration[step, 1],
                slip_state[step, 1],
                normal_max,
                shear_max,
            )
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin is closed")
            process.stdin.write(np.ascontiguousarray(canvas).tobytes())
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")

    decoded_capture = cv2.VideoCapture(str(args.output))
    declared_frames = int(decoded_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_frames = 0
    decoded_shapes = set()
    try:
        while True:
            decoded_ok, decoded_frame = decoded_capture.read()
            if not decoded_ok:
                break
            decoded_frames += 1
            decoded_shapes.add(tuple(int(value) for value in decoded_frame.shape))
    finally:
        decoded_capture.release()
    expected_frames = int(end - start)
    if decoded_frames != expected_frames or declared_frames != expected_frames:
        raise RuntimeError(
            "Rendered video does not fully decode: "
            f"expected={expected_frames}, declared={declared_frames}, "
            f"decoded={decoded_frames}"
        )
    if decoded_shapes != {(HEIGHT, WIDTH, 3)}:
        raise RuntimeError(f"Unexpected decoded frame shapes: {sorted(decoded_shapes)}")

    render_record = {
        "schema": "sugar_whole_hand_carrybox_render_v2",
        "output": str(args.output.resolve()),
        "frames": int(end - start),
        "source_frame_interval": [start, end],
        "resolution": [WIDTH, HEIGHT],
        "fps": args.fps,
        "full_decode": {
            "passed": True,
            "declared_frames": declared_frames,
            "decoded_frames": decoded_frames,
            "decoded_frame_shape": [HEIGHT, WIDTH, 3],
        },
        "normal_scale_max_n_per_taxel": normal_max,
        "shear_scale_max_n_per_taxel": shear_max,
        "scale_selection": args.scale_note,
        "normal_display_convention": (
            "untouched signed local-Z force: negative/red, positive/blue, zero/white"
        ),
        "layout": (
            "top: full SUGAR CarryBox and same-frame hand-box crop; bottom: "
            "both complete hands, five fingers x three segments plus an "
            "anatomically oriented palm with four patches across and three "
            "patches from fingers to wrist; K/I/G are tactile-history-only "
            "stick, incipient-slip, and gross-slip states"
        ),
    }
    args.output.with_suffix(".render.json").write_text(
        json.dumps(render_record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(render_record, indent=2))


if __name__ == "__main__":
    main()
