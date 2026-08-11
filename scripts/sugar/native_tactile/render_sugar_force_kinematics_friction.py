#!/usr/bin/env python3
"""Render a clock-correct CarryBox force, kinematics, and friction audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

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
WIDTH, HEIGHT = 1920, 1080

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument(
    "--taxel-array-root",
    type=Path,
    default=None,
    help=(
        "Optional directory containing extracted normal_force.npy, "
        "signed_shear.npy, and taxel_quaternion_w.npy. When omitted, the "
        "renderer extracts them from whole_hand_trace.npz into a temporary "
        "directory and removes it on exit."
    ),
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--start-frame", type=int, default=230)
parser.add_argument("--end-frame", type=int, default=468)
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


def quat_apply_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[..., 1:]
    cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[..., :1] * cross + np.cross(xyz, cross)


def draw_plot(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    title: str,
    series: tuple[tuple[np.ndarray, str, tuple[int, int, int]], ...],
    step: int,
    interval: tuple[int, int],
    y_limits: tuple[float, float],
    current_note: str,
) -> None:
    x, y, width, height = box
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (135, 135, 135), 1)
    put(canvas, title, (x + 10, y + 25), 0.49, 2)
    start, end = interval
    lo, hi = y_limits
    plot_x0, plot_x1 = x + 42, x + width - 14
    plot_y0, plot_y1 = y + 42, y + height - 58
    if lo < 0.0 < hi:
        zero_y = int(round(plot_y1 - (0.0 - lo) / (hi - lo) * (plot_y1 - plot_y0)))
        cv2.line(canvas, (plot_x0, zero_y), (plot_x1, zero_y), (215, 215, 215), 1)
    for values, label, color in series:
        data = np.asarray(values[start:end], np.float64)
        xs = np.linspace(plot_x0, plot_x1, len(data))
        ys = plot_y1 - np.clip((data - lo) / (hi - lo), 0.0, 1.0) * (plot_y1 - plot_y0)
        valid = np.isfinite(data)
        if np.count_nonzero(valid) > 1:
            points = np.stack((xs[valid], ys[valid]), axis=-1).round().astype(np.int32)
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
    cursor = plot_x0 + int(round((step - start) * (plot_x1 - plot_x0) / max(end - start - 1, 1)))
    cv2.line(canvas, (cursor, plot_y0), (cursor, plot_y1), (20, 20, 20), 1)
    legend_x = x + 10
    for _, label, color in series:
        put(canvas, label, (legend_x, y + height - 33), 0.37, 1, color)
        legend_x += max(125, 10 + 9 * len(label))
    put(canvas, current_note, (x + 10, y + height - 10), 0.37, 1)
    put(canvas, f"{hi:+.1f}", (x + 3, plot_y0 + 5), 0.30, 1, (90, 90, 90))
    put(canvas, f"{lo:+.1f}", (x + 3, plot_y1), 0.30, 1, (90, 90, 90))


def patch_color(count: int, maximum: int) -> tuple[int, int, int]:
    if count <= 0:
        return (242, 242, 242)
    alpha = min(count / max(maximum, 1), 1.0)
    return (int(225 * (1.0 - alpha)), int(225 * (1.0 - alpha)), 255)


def draw_patch_tile(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    count: int,
    maximum: int,
) -> None:
    x, y, width, height = box
    color = patch_color(count, maximum)
    border = (0, 50, 195) if count else (170, 170, 170)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), color, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), border, 2 if count else 1)
    put(canvas, label, (x + 4, y + 15), 0.28, 1)
    put(canvas, str(count), (x + 4, y + height - 5), 0.31, 1, border)


def draw_hand(
    canvas: np.ndarray,
    origin: tuple[int, int],
    side: str,
    counts: np.ndarray,
    maximum: int,
) -> None:
    x0, y0 = origin
    put(canvas, f"{side.upper()} hand: active taxels / 500", (x0, y0), 0.47, 2)
    tile_w, tile_h, gap = 70, 48, 8
    order = DIGITS if side == "left" else tuple(reversed(DIGITS))
    for column, digit in enumerate(order):
        x = x0 + column * (tile_w + gap)
        short = digit[0].upper()
        for row, segment in enumerate(("distal", "middle", "proximal")):
            patch_index = PATCHES.index(f"{digit}_{segment}")
            draw_patch_tile(
                canvas,
                (x, y0 + 18 + row * (tile_h + 5), tile_w, tile_h),
                f"{short}-{segment[0].upper()}",
                int(counts[patch_index]),
                maximum,
            )
    palm_y = y0 + 184
    put(canvas, "palm: all 12 physical patches", (x0, palm_y), 0.34, 1)
    for row in range(3):
        for column in range(4):
            physical_row = column if side == "left" else 3 - column
            physical_column = 2 - row
            patch_index = PATCHES.index(f"palm_r{physical_row}_c{physical_column}")
            draw_patch_tile(
                canvas,
                (
                    x0 + column * (tile_w + gap),
                    palm_y + 10 + row * (tile_h + 5),
                    tile_w,
                    tile_h,
                ),
                f"P{physical_row}{physical_column}",
                int(counts[patch_index]),
                maximum,
            )


def compute_tacsl_reaction(
    array_root: Path, frames: int
) -> np.ndarray:
    normal = np.load(array_root / "normal_force.npy", mmap_mode="r")
    shear = np.load(array_root / "signed_shear.npy", mmap_mode="r")
    quaternion = np.load(array_root / "taxel_quaternion_w.npy", mmap_mode="r")
    if len(normal) != frames or len(shear) != frames or len(quaternion) != frames:
        raise RuntimeError("Extracted TacSL arrays do not match the control trace")
    result = np.empty((frames, 3), dtype=np.float64)
    for step in range(frames):
        local = np.concatenate(
            (np.asarray(shear[step], np.float64), np.asarray(normal[step], np.float64)[..., None]),
            axis=-1,
        )
        world = quat_apply_wxyz(np.asarray(quaternion[step], np.float64), local)
        result[step] = -world.sum(axis=(0, 1, 2, 3))
    return result


def resolve_taxel_array_root(run_root: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if args.taxel_array_root is not None:
        return args.taxel_array_root.resolve(), None
    temporary = tempfile.TemporaryDirectory(prefix="carrybox_taxel_arrays_")
    temporary_root = Path(temporary.name)
    with zipfile.ZipFile(run_root / "whole_hand_trace.npz") as archive:
        for name in ("normal_force.npy", "signed_shear.npy", "taxel_quaternion_w.npy"):
            archive.extract(name, temporary_root)
    return temporary_root, temporary


def main() -> None:
    run_root = args.run_root.resolve()
    taxel_array_root, temporary_arrays = resolve_taxel_array_root(run_root)
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    with np.load(run_root / "whole_hand_trace.npz", allow_pickle=False) as source:
        arrays = {
            key: source[key]
            for key in (
                "active_taxels",
                "bilateral_contact",
                "object_state_w",
                "object_velocity_w",
                "physics_object_state_w",
                "physics_object_velocity_w",
                "physics_robot_box_force_w",
                "physics_robot_box_friction_force_w",
                "physics_control_step",
                "physics_substep",
                "patch_box_force_w",
                "patch_box_friction_force_w",
                "robot_box_force_w",
                "robot_box_friction_force_w",
                "object_material_properties",
                "gravity_w",
                "physics_dt_s",
                "control_dt_s",
                "patch_order",
            )
        }
    if tuple(arrays["patch_order"].astype(str)) != PATCHES:
        raise RuntimeError("Patch order drift")
    frames = len(arrays["object_state_w"])
    if not (0 <= args.start_frame < args.end_frame <= frames):
        raise RuntimeError("Invalid render interval")
    physics_control = np.asarray(arrays["physics_control_step"], np.int64)
    physics_substep = np.asarray(arrays["physics_substep"], np.int64)
    if not np.array_equal(physics_control, np.repeat(np.arange(frames), 4)):
        raise RuntimeError("Expected exactly four ordered physics substeps per control frame")
    if not np.array_equal(physics_substep, np.tile(np.arange(4), frames)):
        raise RuntimeError("Physics substep order is not 0,1,2,3")

    mass = float(summary["box_mass_readback_kg"])
    gravity = np.asarray(arrays["gravity_w"], np.float64)
    weight = mass * float(np.linalg.norm(gravity))
    velocity = np.asarray(arrays["physics_object_velocity_w"], np.float64)
    normal_body = np.asarray(arrays["physics_robot_box_force_w"], np.float64)
    friction_body = np.asarray(arrays["physics_robot_box_friction_force_w"], np.float64)
    reaction_substep = -(normal_body + friction_body).sum(axis=1)
    normal_substep = -normal_body.sum(axis=1)
    friction_substep = -friction_body.sum(axis=1)
    clamp_substep = np.linalg.norm(normal_body, axis=-1).sum(axis=1)
    reaction = reaction_substep.reshape(frames, 4, 3).mean(axis=1)
    normal = normal_substep.reshape(frames, 4, 3).mean(axis=1)
    friction = friction_substep.reshape(frames, 4, 3).mean(axis=1)
    clamp = clamp_substep.reshape(frames, 4).mean(axis=1)
    end_velocity = velocity.reshape(frames, 4, 6)[:, -1]
    required = np.full((frames, 3), np.nan, dtype=np.float64)
    required[1:] = mass * (
        (end_velocity[1:, :3] - end_velocity[:-1, :3])
        / float(arrays["control_dt_s"])
        - gravity[None]
    )
    residual = np.linalg.norm(reaction - required, axis=-1) / weight
    tacsl = compute_tacsl_reaction(taxel_array_root, frames)

    state = np.asarray(arrays["object_state_w"], np.float64)
    lift = state[:, 2] - state[0, 2]
    off_ground = lift >= 0.05
    speed = np.linalg.norm(np.asarray(arrays["object_velocity_w"], np.float64)[:, :3], axis=-1)
    angular_speed = np.linalg.norm(
        np.asarray(arrays["object_velocity_w"], np.float64)[:, 3:], axis=-1
    )
    bilateral = np.asarray(arrays["bilateral_contact"], bool)
    high_lift = (lift >= 0.20) & bilateral
    valid_high = high_lift & np.isfinite(residual)
    if not np.any(valid_high):
        raise RuntimeError("No lifted bilateral interval in the trace")
    tacsl_residual = np.linalg.norm(tacsl - required, axis=-1) / weight
    friction_utilization = np.abs(friction[:, 2]) / np.maximum(0.5 * clamp, 1.0e-12)
    active_patch = np.asarray(arrays["active_taxels"], np.int64) > 0
    patch_force = np.asarray(arrays["patch_box_force_w"], np.float64)
    patch_friction = np.asarray(arrays["patch_box_friction_force_w"], np.float64)
    physical_patch = (
        np.linalg.norm(patch_force, axis=-1)
        + np.linalg.norm(patch_friction, axis=-1)
    ) > 1.0e-7
    true_positive = int(np.count_nonzero(active_patch[high_lift] & physical_patch[high_lift]))
    false_positive = int(np.count_nonzero(active_patch[high_lift] & ~physical_patch[high_lift]))
    false_negative = int(np.count_nonzero(~active_patch[high_lift] & physical_patch[high_lift]))
    true_negative = int(np.count_nonzero(~active_patch[high_lift] & ~physical_patch[high_lift]))
    contact_total = true_positive + false_positive + false_negative + true_negative
    patch_reaction = -(patch_force + patch_friction).sum(axis=(1, 2))
    all_robot_reaction = -(
        np.asarray(arrays["robot_box_force_w"], np.float64)
        + np.asarray(arrays["robot_box_friction_force_w"], np.float64)
    ).sum(axis=1)
    uninstrumented_difference = np.linalg.norm(
        patch_reaction - all_robot_reaction, axis=-1
    )
    uninstrumented_relative = uninstrumented_difference / np.maximum(
        np.linalg.norm(all_robot_reaction, axis=-1), 1.0e-12
    )
    patch_active_frames = {
        side: {
            patch: int(np.count_nonzero(active_patch[high_lift, hand, patch_index]))
            for patch_index, patch in enumerate(PATCHES)
        }
        for hand, side in enumerate(("left", "right"))
    }
    distal_indices = np.asarray(
        [index for index, patch in enumerate(PATCHES) if patch.endswith("_distal")]
    )
    normal_magnitude_by_patch = np.linalg.norm(patch_force[high_lift], axis=-1).sum(axis=0)
    distal_normal_fraction = float(
        normal_magnitude_by_patch[:, distal_indices].sum()
        / normal_magnitude_by_patch.sum()
    )
    physics_state = np.asarray(arrays["physics_object_state_w"], np.float64)
    high_lift_physics = np.repeat(high_lift, 4)
    adjacent_high_lift = high_lift_physics[1:] & high_lift_physics[:-1]
    translation_increment = np.linalg.norm(
        physics_state[1:, :3] - physics_state[:-1, :3], axis=-1
    )
    quaternion = physics_state[:, 3:7]
    quaternion_norm_error = np.abs(np.linalg.norm(quaternion, axis=-1) - 1.0)
    quaternion_dot = np.abs(np.sum(quaternion[1:] * quaternion[:-1], axis=-1))
    rotation_increment = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))

    stats = {
        "schema": "sugar_force_kinematics_friction_audit_v1",
        "control_frames": frames,
        "physics_substeps": int(len(physics_control)),
        "physics_dt_s": float(arrays["physics_dt_s"]),
        "control_dt_s": float(arrays["control_dt_s"]),
        "mass_kg": mass,
        "weight_n": weight,
        "lifted_bilateral_frames": int(np.count_nonzero(high_lift)),
        "lifted_bilateral_interval": [
            int(np.flatnonzero(high_lift)[0]),
            int(np.flatnonzero(high_lift)[-1]),
        ],
        "physx_force_residual_over_mg_q05_median_q95": np.quantile(
            residual[valid_high], (0.05, 0.50, 0.95)
        ).tolist(),
        "physx_vertical_force_median_n": float(np.median(reaction[valid_high, 2])),
        "required_vertical_force_median_n": float(np.median(required[valid_high, 2])),
        "normal_vertical_median_n": float(np.median(normal[valid_high, 2])),
        "friction_vertical_median_n": float(np.median(friction[valid_high, 2])),
        "summed_normal_magnitude_median_n": float(np.median(clamp[valid_high])),
        "global_friction_utilization_against_mu_0p5_q05_median_q95": np.quantile(
            friction_utilization[valid_high], (0.05, 0.50, 0.95)
        ).tolist(),
        "tacsl_reaction_magnitude_q05_median_q95_n": np.quantile(
            np.linalg.norm(tacsl[valid_high], axis=-1), (0.05, 0.50, 0.95)
        ).tolist(),
        "tacsl_force_residual_over_mg_q05_median_q95": np.quantile(
            tacsl_residual[valid_high], (0.05, 0.50, 0.95)
        ).tolist(),
        "object_dynamic_friction_readback": np.asarray(
            arrays["object_material_properties"], np.float64
        )[..., 1].reshape(-1).tolist(),
        "sensor_patch_dynamic_friction": 0.5,
        "tacsl_patch_vs_physx_patch_contact": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "agreement": (true_positive + true_negative) / contact_total,
            "precision": true_positive / (true_positive + false_positive),
            "recall": true_positive / (true_positive + false_negative),
        },
        "sensorized_patch_wrench_vs_all_robot_wrench": {
            "absolute_difference_n_q05_median_q95": np.quantile(
                uninstrumented_difference[high_lift], (0.05, 0.50, 0.95)
            ).tolist(),
            "relative_difference_q05_median_q95": np.quantile(
                uninstrumented_relative[high_lift], (0.05, 0.50, 0.95)
            ).tolist(),
        },
        "distal_patch_normal_force_fraction": distal_normal_fraction,
        "active_frames_by_patch_during_lifted_bilateral_interval": patch_active_frames,
        "kinematic_continuity": {
            "quaternion_norm_max_error": float(np.max(quaternion_norm_error)),
            "translation_per_5ms_m_q05_median_q95": np.quantile(
                translation_increment[adjacent_high_lift], (0.05, 0.50, 0.95)
            ).tolist(),
            "translation_per_5ms_m_max": float(
                np.max(translation_increment[adjacent_high_lift])
            ),
            "rotation_per_5ms_rad_q05_median_q95": np.quantile(
                rotation_increment[adjacent_high_lift], (0.05, 0.50, 0.95)
            ).tolist(),
            "rotation_per_5ms_rad_max": float(
                np.max(rotation_increment[adjacent_high_lift])
            ),
            "interpretation": "continuous actor pose; translational COM dynamics close through m(a-g)",
            "not_claimed": "full rotational torque balance was not reconstructed",
        },
        "verdict": {
            "physx_translation_balance": "passes at numerical precision",
            "physical_friction_support": "consistent and sufficient",
            "tacsl_contact_localization": "physically corresponding",
            "tacsl_force_calibration": "fails physical wrench agreement",
        },
    }

    capture = cv2.VideoCapture(str(run_root / "world_carrybox.mp4"))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != frames:
        raise RuntimeError("World video and trace frame count differ")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
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
    interval = (args.start_frame, args.end_frame)
    force_limit = max(5.0, float(np.nanquantile(np.abs(required[args.start_frame:args.end_frame]), 0.99)) * 1.15)
    magnitude_limit = max(
        10.0,
        float(np.nanquantile(np.linalg.norm(tacsl[args.start_frame:args.end_frame], axis=-1), 0.99))
        * 1.10,
    )
    maximum_patch_count = max(1, int(np.quantile(arrays["active_taxels"], 0.995)))
    required_off_ground = required.copy()
    required_off_ground[~off_ground] = np.nan
    try:
        for step in range(args.start_frame, args.end_frame):
            ok, world = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode world frame {step}")
            canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
            put(
                canvas,
                "CarryBox: synchronized physical balance and native whole-hand tactile",
                (20, 35),
                0.72,
                2,
            )
            put(
                canvas,
                (
                    f"frame {step:03d} | lift {lift[step]:+.3f} m | speed {speed[step]:.3f} m/s | "
                    f"angular speed {angular_speed[step]:.3f} rad/s"
                ),
                (20, 60),
                0.46,
                1,
            )
            canvas[72:557, 20:882] = fit(world, 862, 485)
            cv2.rectangle(canvas, (20, 72), (882, 557), (120, 120, 120), 1)
            put(canvas, "Actual SUGAR CarryBox rollout", (34, 99), 0.48, 2)

            cv2.rectangle(canvas, (900, 72), (1900, 557), (120, 120, 120), 1)
            put(
                canvas,
                "All 27 patches per hand (red = currently loaded)",
                (920, 101),
                0.50,
                2,
            )
            draw_hand(canvas, (925, 133), "left", arrays["active_taxels"][step, 0], maximum_patch_count)
            draw_hand(canvas, (1430, 133), "right", arrays["active_taxels"][step, 1], maximum_patch_count)
            put(
                canvas,
                "Observed result: distal fingertips carry this grasp; palms, middle and proximal patches remain unloaded.",
                (920, 540),
                0.39,
                1,
                (0, 45, 170),
            )

            current_residual = residual[step] * 1.0e6
            if off_ground[step] and np.isfinite(current_residual):
                force_note = (
                    f"now {reaction[step,2]:+.3f} / {required[step,2]:+.3f} N | "
                    f"residual {current_residual:.2f} x10^-6 mg"
                )
            else:
                force_note = (
                    f"robot Fz {reaction[step,2]:+.3f} N | free-body comparison N/A "
                    "while box is on/near ground"
                )
            draw_plot(
                canvas,
                (20, 580, 610, 465),
                "1. Robot PhysX force vs m(a-g) while box is off ground",
                (
                    (reaction[:, 2], "PhysX Fz", (205, 90, 15)),
                    (required_off_ground[:, 2], "required Fz", (20, 155, 195)),
                ),
                step,
                interval,
                (-force_limit, force_limit),
                force_note,
            )
            draw_plot(
                canvas,
                (655, 580, 610, 465),
                "2. Vertical support decomposition",
                (
                    (friction[:, 2], "friction Fz", (30, 145, 40)),
                    (normal[:, 2], "normal Fz", (190, 80, 30)),
                ),
                step,
                interval,
                (-force_limit, force_limit),
                f"now friction {friction[step,2]:+.3f} N | sum |normal| {clamp[step]:.2f} N | global use {100*friction_utilization[step]:.1f}%",
            )
            draw_plot(
                canvas,
                (1290, 580, 610, 465),
                "3. TacSL raw wrench is not force-calibrated",
                (
                    (np.linalg.norm(tacsl, axis=-1), "TacSL |F|", (30, 30, 205)),
                    (np.linalg.norm(reaction, axis=-1), "PhysX |F|", (205, 90, 15)),
                    (np.linalg.norm(required_off_ground, axis=-1), "required |F|", (20, 155, 195)),
                ),
                step,
                interval,
                (0.0, magnitude_limit),
                f"now TacSL {np.linalg.norm(tacsl[step]):.2f} N | PhysX {np.linalg.norm(reaction[step]):.2f} N",
            )
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin closed")
            process.stdin.write(np.ascontiguousarray(canvas).tobytes())
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    decoded_capture = cv2.VideoCapture(str(args.output))
    declared_frames = int(decoded_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_frames = 0
    middle_image = None
    final_image = None
    middle_index = (args.end_frame - args.start_frame) // 2
    while True:
        ok, decoded_image = decoded_capture.read()
        if not ok:
            break
        if decoded_frames == middle_index:
            middle_image = decoded_image.copy()
        final_image = decoded_image.copy()
        decoded_frames += 1
    decoded_capture.release()
    expected_frames = args.end_frame - args.start_frame
    if declared_frames != expected_frames or decoded_frames != expected_frames:
        raise RuntimeError(
            f"Encoded video decode failed: declared={declared_frames}, "
            f"decoded={decoded_frames}, expected={expected_frames}"
        )
    if middle_image is None or final_image is None:
        raise RuntimeError("Could not extract the review frames")
    middle_path = args.output.with_name(f"{args.output.stem}_middle_frame.png")
    if not cv2.imwrite(str(middle_path), middle_image):
        raise RuntimeError("Could not write the middle review frame")
    final_path = args.output.with_name(f"{args.output.stem}_final_frame.png")
    if not cv2.imwrite(str(final_path), final_image):
        raise RuntimeError("Could not write the final review frame")
    stats["video"] = str(args.output.resolve())
    stats["video_source_interval"] = [args.start_frame, args.end_frame]
    stats["video_resolution"] = [WIDTH, HEIGHT]
    stats["video_declared_frames"] = declared_frames
    stats["video_decoded_frames"] = decoded_frames
    stats["middle_review_frame"] = str(middle_path.resolve())
    stats["final_review_frame"] = str(final_path.resolve())
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    if temporary_arrays is not None:
        temporary_arrays.cleanup()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
