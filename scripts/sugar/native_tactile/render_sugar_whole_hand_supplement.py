#!/usr/bin/env python3
"""Render detailed, optical, or force-audit views of one CarryBox trace."""

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
REGIONS = ("palm", *DIGITS)
WIDTH, HEIGHT = 2560, 1440
FORCE_LIMIT_N = 30.0
RESIDUAL_LIMIT = 20.0
OPTICAL_DEPTH_LIMIT_M = 0.050
OPTICAL_DELTA_LIMIT_M = 0.002

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--kind",
    choices=("left_detail", "right_detail", "palm_optical", "force_balance"),
    required=True,
)
parser.add_argument("--title", required=True)
parser.add_argument("--normal-max", type=float, required=True)
parser.add_argument("--shear-max", type=float, required=True)
parser.add_argument("--scale-note", required=True)
parser.add_argument("--start-frame", type=int, required=True)
parser.add_argument("--end-frame", type=int, required=True)
parser.add_argument("--fps", type=int, default=50)
args = parser.parse_args()


def put(
    image: np.ndarray,
    text: str,
    point: tuple[int, int],
    scale: float = 0.48,
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


def signed_bgr(values: np.ndarray, maximum: float) -> np.ndarray:
    scaled = np.clip(np.asarray(values, np.float32) / maximum, -1.0, 1.0)
    magnitude = np.abs(scaled)
    image = np.full((*scaled.shape, 3), 255, dtype=np.uint8)
    negative = scaled < 0.0
    positive = scaled > 0.0
    image[..., 0][negative] = np.rint(
        255.0 * (1.0 - magnitude[negative])
    ).astype(np.uint8)
    image[..., 1][negative] = np.rint(
        255.0 * (1.0 - 0.70 * magnitude[negative])
    ).astype(np.uint8)
    image[..., 1][positive] = np.rint(
        255.0 * (1.0 - 0.70 * magnitude[positive])
    ).astype(np.uint8)
    image[..., 2][positive] = np.rint(
        255.0 * (1.0 - magnitude[positive])
    ).astype(np.uint8)
    return image


def quat_apply_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[..., :3]
    cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[..., 3:] * cross + np.cross(xyz, cross)


def fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    result = np.full((height, width, 3), 245, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    result[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return result


def crop(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    target_ratio = width / height
    source_ratio = source_w / source_h
    if source_ratio > target_ratio:
        crop_w = int(round(source_h * target_ratio))
        x0 = (source_w - crop_w) // 2
        source = frame[:, x0 : x0 + crop_w]
    else:
        crop_h = int(round(source_w / target_ratio))
        y0 = max(0, min(source_h - crop_h, int(source_h * 0.42 - crop_h / 2)))
        source = frame[y0 : y0 + crop_h]
    return cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)


def header(canvas: np.ndarray, title: str, step: int, subtitle: str) -> None:
    cv2.rectangle(canvas, (14, 14), (WIDTH - 14, 58), (255, 255, 255), -1)
    put(canvas, title, (28, 46), 0.82, 2)
    cv2.rectangle(canvas, (14, 64), (WIDTH - 14, 102), (255, 255, 255), -1)
    put(canvas, f"source frame {step:03d} | {subtitle}", (28, 91), 0.54, 1)


def draw_detail_tile(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    name: str,
    normal: np.ndarray,
    shear: np.ndarray,
    penetration: np.ndarray,
) -> None:
    x, y, width, height = box
    active_mask = penetration > 0.0
    active = int(np.count_nonzero(active_mask))
    peak = float(normal.reshape(-1)[np.argmax(np.abs(normal))])
    border = (0, 70, 190) if active else (145, 145, 145)
    thickness = 3 if active else 1
    cv2.rectangle(canvas, (x, y), (x + width, y + height), border, thickness)
    put(
        canvas,
        f"{name} | active {active:03d} | Z peak {peak:+.3f} N",
        (x + 6, y + 21),
        0.37,
        1,
    )
    map_y = y + 28
    map_h = height - 34
    heat = cv2.resize(
        signed_bgr(normal, args.normal_max),
        (width - 10, map_h),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas[map_y : map_y + map_h, x + 5 : x + width - 5] = heat
    for row in range(2, 20, 4):
        for column in range(2, 25, 5):
            vector = shear[row, column]
            magnitude = float(np.linalg.norm(vector))
            if magnitude <= 1.0e-12:
                continue
            center = (
                x + 5 + int((column + 0.5) * (width - 10) / 25),
                map_y + int((row + 0.5) * map_h / 20),
            )
            direction = vector / magnitude
            length = 0.28 * min(width, map_h) * min(
                magnitude / args.shear_max, 1.0
            )
            end = (
                int(round(center[0] + direction[0] * length)),
                int(round(center[1] - direction[1] * length)),
            )
            cv2.arrowedLine(canvas, center, end, (20, 20, 20), 1, cv2.LINE_AA, 0, 0.30)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), border, thickness)


def detail_layout(side: str) -> dict[str, tuple[int, int, int, int]]:
    layout: dict[str, tuple[int, int, int, int]] = {}
    tile_w, tile_h = 420, 130
    finger_gap = 60
    x0 = (WIDTH - (5 * tile_w + 4 * finger_gap)) // 2
    digits = (
        ("little", "ring", "middle", "index", "thumb")
        if side == "left"
        else ("thumb", "index", "middle", "ring", "little")
    )
    for column, digit in enumerate(digits):
        x = x0 + column * (tile_w + finger_gap)
        for row, segment in enumerate(("distal", "middle", "proximal")):
            layout[f"{digit}_{segment}"] = (x, 440 + row * 136, tile_w, tile_h)
    palm_gap = 60
    palm_x0 = (WIDTH - (4 * tile_w + 3 * palm_gap)) // 2
    rows = (0, 1, 2, 3) if side == "left" else (3, 2, 1, 0)
    for y_index, column in enumerate((2, 1, 0)):
        for x_index, row in enumerate(rows):
            layout[f"palm_r{row}_c{column}"] = (
                palm_x0 + x_index * (tile_w + palm_gap),
                930 + y_index * 136,
                tile_w,
                tile_h,
            )
    return layout


def build_detail(
    world: np.ndarray,
    arrays: dict[str, np.ndarray],
    step: int,
    side: str,
) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    canvas[:360, :1280] = fit(world, 1280, 360)
    canvas[:360, 1280:] = crop(world, 1280, 360)
    header(
        canvas,
        args.title,
        step,
        (
            "raw 20x25 signed local-Z plus spatial signed-XY arrows | "
            f"fixed Z {args.normal_max:.3f} N, XY {args.shear_max:.3f} N"
        ),
    )
    hand = 0 if side == "left" else 1
    normal = arrays["normal_force"][step, hand]
    shear = arrays["signed_shear"][step, hand]
    penetration = arrays["penetration"][step, hand]
    active_patches = int(np.count_nonzero(np.any(penetration > 0, axis=(-2, -1))))
    active_taxels = int(np.count_nonzero(penetration > 0))
    put(
        canvas,
        f"{side.upper()} HAND | all 27 patches | active {active_patches:02d} patches / {active_taxels:04d} taxels",
        (30, 400),
        0.68,
        2,
    )
    layout = detail_layout(side)
    for patch_index, patch in enumerate(PATCHES):
        draw_detail_tile(
            canvas,
            layout[patch],
            patch.replace("proximal", "prox"),
            normal[patch_index],
            shear[patch_index],
            penetration[patch_index],
        )
    put(
        canvas,
        "FINGERS: five columns x distal/middle/proximal | PALM: 4 across x 3 fingers-to-wrist | r1c1 = R15",
        (30, 1418),
        0.50,
        1,
    )
    return canvas


def depth_bgr(depth: np.ndarray, maximum: float) -> np.ndarray:
    normalized = np.clip(np.nan_to_num(depth, nan=maximum) / maximum, 0.0, 1.0)
    return cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )


def build_optical(
    world: np.ndarray,
    arrays: dict[str, np.ndarray],
    step: int,
) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    canvas[:720, :1280] = fit(world, 1280, 720)
    canvas[:720, 1280:] = crop(world, 1280, 720)
    header(
        canvas,
        args.title,
        step,
        "official center-palm R15 RGB and raw camera depth | zeros are preserved",
    )
    panel_w, panel_h = 600, 560
    for hand, side in enumerate(("left", "right")):
        base_x = 40 + hand * 1280
        rgb = arrays["optical_rgb"][step, hand][..., ::-1]
        rgb_view = cv2.resize(rgb, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        canvas[820 : 820 + panel_h, base_x : base_x + panel_w] = rgb_view
        depth = arrays["optical_depth"][step, hand, ..., 0]
        baseline = arrays["optical_baseline_depth"][hand, ..., 0]
        absolute = cv2.resize(
            depth_bgr(depth, OPTICAL_DEPTH_LIMIT_M),
            (panel_w, 265),
            interpolation=cv2.INTER_AREA,
        )
        delta = cv2.resize(
            signed_bgr(depth - baseline, OPTICAL_DELTA_LIMIT_M),
            (panel_w, 265),
            interpolation=cv2.INTER_AREA,
        )
        depth_x = base_x + 620
        canvas[820:1085, depth_x : depth_x + panel_w] = absolute
        canvas[1115:1380, depth_x : depth_x + panel_w] = delta
        put(canvas, f"{side.upper()} R15 RGB", (base_x, 800), 0.62, 2)
        put(
            canvas,
            f"{side.upper()} raw depth 0..{OPTICAL_DEPTH_LIMIT_M:.3f} m",
            (depth_x, 800),
            0.52,
            1,
        )
        put(
            canvas,
            (
                f"depth - no-contact baseline +/-{OPTICAL_DELTA_LIMIT_M:.3f} m | "
                f"peak {float(np.max(np.abs(depth - baseline))):.2e} m"
            ),
            (depth_x, 1105),
            0.42,
            1,
        )
    return canvas


def compute_force_series(arrays: dict[str, np.ndarray], mass: float) -> dict[str, np.ndarray]:
    normal = np.asarray(arrays["normal_force"], np.float64)
    shear = np.asarray(arrays["signed_shear"], np.float64)
    quaternion = np.asarray(arrays["taxel_quaternion_w"], np.float64)
    frames = len(normal)
    tacsl_hand = np.zeros((frames, 2, 3), dtype=np.float64)
    tacsl_region = np.zeros((frames, 2, 6, 3), dtype=np.float64)
    region_slices = (slice(0, 12), *(slice(12 + 3 * i, 15 + 3 * i) for i in range(5)))
    for step in range(frames):
        local = np.concatenate((shear[step], normal[step, ..., None]), axis=-1)
        reaction = -quat_apply_xyzw(quaternion[step], local)
        tacsl_hand[step] = reaction.sum(axis=(1, 2, 3))
        for hand in range(2):
            for region_index, patch_slice in enumerate(region_slices):
                tacsl_region[step, hand, region_index] = reaction[
                    hand, patch_slice
                ].sum(axis=(0, 1, 2))
    patch_total = -(
        np.asarray(arrays["patch_box_force_w"], np.float64)
        + np.asarray(arrays["patch_box_friction_force_w"], np.float64)
    )
    physx_hand = patch_total.sum(axis=2)
    physx_region = np.zeros((frames, 2, 6, 3), dtype=np.float64)
    for region_index, patch_slice in enumerate(region_slices):
        physx_region[:, :, region_index] = patch_total[:, :, patch_slice].sum(axis=2)
    physx = -(
        np.asarray(arrays["robot_box_force_w"], np.float64)
        + np.asarray(arrays["robot_box_friction_force_w"], np.float64)
    ).sum(axis=1)
    velocity = np.asarray(arrays["object_velocity_w"], np.float64)
    gravity = np.asarray(arrays["gravity_w"], np.float64)
    dt = float(arrays["control_dt_s"])
    acceleration = np.full((frames, 3), np.nan, dtype=np.float64)
    acceleration[1:-1] = (velocity[2:, :3] - velocity[:-2, :3]) / (2.0 * dt)
    required = mass * (acceleration - gravity[None])
    weight = mass * float(np.linalg.norm(gravity))
    tacsl = tacsl_hand.sum(axis=1)
    return {
        "tacsl": tacsl,
        "tacsl_hand": tacsl_hand,
        "tacsl_region": tacsl_region,
        "physx": physx,
        "physx_hand": physx_hand,
        "physx_region": physx_region,
        "required": required,
        "tacsl_residual": np.linalg.norm(tacsl - required, axis=-1) / weight,
        "physx_residual": np.linalg.norm(physx - required, axis=-1) / weight,
        "weight": np.asarray(weight),
    }


def draw_series(
    canvas: np.ndarray,
    series: tuple[tuple[np.ndarray, tuple[int, int, int], str], ...],
    step: int,
    interval: tuple[int, int],
    box: tuple[int, int, int, int],
    minimum: float,
    maximum: float,
    title: str,
) -> None:
    x, y, width, height = box
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (130, 130, 130), 1)
    put(canvas, title, (x + 8, y + 24), 0.46, 1)
    start, end = interval
    for index, (values, color, label) in enumerate(series):
        data = np.nan_to_num(values[start:end], nan=minimum)
        xs = np.linspace(x + 5, x + width - 5, len(data))
        ys = y + height - 24 - np.clip(
            (data - minimum) / max(maximum - minimum, 1.0e-12), 0.0, 1.0
        ) * (height - 58)
        points = np.stack((xs, ys), axis=-1).round().astype(np.int32)
        cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        put(canvas, label, (x + 8 + index * 145, y + height - 7), 0.35, 1, color)
    cursor = x + int(round((step - start) * width / max(end - start - 1, 1)))
    cv2.line(canvas, (cursor, y + 30), (cursor, y + height - 25), (20, 20, 20), 1)


def build_force_balance(
    world: np.ndarray,
    arrays: dict[str, np.ndarray],
    force: dict[str, np.ndarray],
    step: int,
) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    canvas[:540, :1280] = fit(world, 1280, 540)
    header(
        canvas,
        args.title,
        step,
        "forces on CarryBox: raw TacSL reaction vs independent PhysX vs m(a-g)",
    )
    current = (
        f"TacSL ({force['tacsl'][step,0]:+.2f}, {force['tacsl'][step,1]:+.2f}, {force['tacsl'][step,2]:+.2f}) N   "
        f"PhysX ({force['physx'][step,0]:+.2f}, {force['physx'][step,1]:+.2f}, {force['physx'][step,2]:+.2f}) N   "
        f"required ({force['required'][step,0]:+.2f}, {force['required'][step,1]:+.2f}, {force['required'][step,2]:+.2f}) N"
    )
    put(canvas, current, (30, 575), 0.47, 1)
    put(
        canvas,
        "TacSL = spatial sensor model | PhysX = actual box support",
        (1300, 130),
        0.48,
        1,
        (70, 70, 70),
    )
    for hand, side in enumerate(("LEFT", "RIGHT")):
        x = 1300 + hand * 620
        put(canvas, f"{side} current Fz: TacSL / PhysX [N]", (x, 180), 0.48, 2)
        for region_index, region in enumerate(REGIONS):
            put(
                canvas,
                (
                    f"{region:>6s}: "
                    f"{force['tacsl_region'][step,hand,region_index,2]:+8.3f} / "
                    f"{force['physx_region'][step,hand,region_index,2]:+8.3f}"
                ),
                (x, 215 + region_index * 42),
                0.44,
                1,
            )
        hand_t = force["tacsl_hand"][step, hand]
        hand_p = force["physx_hand"][step, hand]
        put(
            canvas,
            f"hand total Fz: {hand_t[2]:+.3f} / {hand_p[2]:+.3f}",
            (x, 490),
            0.45,
            2,
        )
    interval = (args.start_frame, args.end_frame)
    colors = ((20, 20, 210), (210, 130, 20), (20, 160, 190))
    for component, label in enumerate(("Fx", "Fy", "Fz")):
        draw_series(
            canvas,
            (
                (force["tacsl"][:, component], colors[0], "TacSL"),
                (force["physx"][:, component], colors[1], "PhysX"),
                (force["required"][:, component], colors[2], "required"),
            ),
            step,
            interval,
            (30 + component * 840, 620, 800, 330),
            -FORCE_LIMIT_N,
            FORCE_LIMIT_N,
            f"{label} [N] fixed +/-{FORCE_LIMIT_N:.0f}",
        )
    draw_series(
        canvas,
        (
            (force["tacsl_residual"], colors[0], "TacSL / weight"),
            (force["physx_residual"], colors[1], "PhysX / weight"),
        ),
        step,
        interval,
        (30, 990, 1220, 390),
        0.0,
        RESIDUAL_LIMIT,
        f"normalized vector residual fixed 0..{RESIDUAL_LIMIT:.0f}",
    )
    height = arrays["object_state_w"][:, 2] - arrays["object_state_w"][0, 2]
    draw_series(
        canvas,
        ((height, (40, 140, 200), "box lift [m]"),),
        step,
        interval,
        (1310, 990, 1220, 390),
        -0.10,
        0.90,
        f"box lift | m*g={float(force['weight']):.3f} N",
    )
    return canvas


def main() -> None:
    run_root = args.run_root.resolve()
    trace_path = run_root / "whole_hand_trace.npz"
    world_path = run_root / "world_carrybox.mp4"
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if tuple(arrays["patch_order"].astype(str)) != PATCHES:
        raise RuntimeError("Patch order drift")
    frames = len(arrays["normal_force"])
    if not (0 <= args.start_frame < args.end_frame <= frames):
        raise RuntimeError("Invalid frame interval")
    capture = cv2.VideoCapture(str(world_path))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != frames:
        raise RuntimeError("World/tactile frame mismatch")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    force = (
        compute_force_series(arrays, float(summary["box_mass_readback_kg"]))
        if args.kind == "force_balance"
        else None
    )
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
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for step in range(args.start_frame, args.end_frame):
            ok, world = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode world frame {step}")
            if args.kind in ("left_detail", "right_detail"):
                canvas = build_detail(world, arrays, step, args.kind.split("_")[0])
            elif args.kind == "palm_optical":
                canvas = build_optical(world, arrays, step)
            else:
                assert force is not None
                canvas = build_force_balance(world, arrays, force, step)
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin closed")
            process.stdin.write(np.ascontiguousarray(canvas).tobytes())
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with {return_code}")
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
    expected_frames = int(args.end_frame - args.start_frame)
    if decoded_frames != expected_frames or declared_frames != expected_frames:
        raise RuntimeError(
            "Rendered supplement does not fully decode: "
            f"expected={expected_frames}, declared={declared_frames}, "
            f"decoded={decoded_frames}"
        )
    if decoded_shapes != {(HEIGHT, WIDTH, 3)}:
        raise RuntimeError(f"Unexpected decoded frame shapes: {sorted(decoded_shapes)}")
    record = {
        "schema": "sugar_whole_hand_carrybox_supplement_render_v1",
        "kind": args.kind,
        "output": str(args.output.resolve()),
        "frames": args.end_frame - args.start_frame,
        "source_frame_interval": [args.start_frame, args.end_frame],
        "resolution": [WIDTH, HEIGHT],
        "fps": args.fps,
        "full_decode": {
            "passed": True,
            "declared_frames": declared_frames,
            "decoded_frames": decoded_frames,
            "decoded_frame_shape": [HEIGHT, WIDTH, 3],
        },
        "normal_scale_max_n_per_taxel": args.normal_max,
        "shear_scale_max_n_per_taxel": args.shear_max,
        "scale_selection": args.scale_note,
    }
    args.output.with_suffix(".render.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
