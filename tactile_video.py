# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render native solved-force tactile fields for the Panda box/pen scenes.

This replaces the historical ``kh * depth`` heatmap and aggregate
``SensorContact`` path. Every displayed cell comes from public
``SensorTactile`` output after ``solver.update_contacts``.

Example:
    python tactile_video.py --scene cube --frames 420 --output out/cube.mp4
"""

from __future__ import annotations

import faulthandler
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pyglet

pyglet.options["headless"] = True

import imageio_ffmpeg
import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont

import newton
import newton.examples
from newton.examples.robot.example_robot_panda_hydro import Example
from newton.sensors import SensorTactile
from scripts.sugar.native_tactile.slip import SlipState, TactileSlipDetector
from scripts.sugar.native_tactile.universal import NewtonTactileAdapter


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[:3]
    cross = np.cross(xyz, vector)
    return vector + 2.0 * (quaternion[3] * cross + np.cross(xyz, cross))


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _compose_transform(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate((a[:3] + _quat_rotate(a[3:7], b[:3]), _quat_multiply(a[3:7], b[3:7])))


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.astype(np.float32)


def _pad_patch_geometry(example: Example) -> tuple[list[int], list[wp.transform], list[tuple[float, float]]]:
    model = example.model
    shape_body = model.shape_body.numpy()
    shape_transform = model.shape_transform.numpy()
    shape_scale = model.shape_scale.numpy()
    body_q = example.state_0.body_q.numpy()
    object_body = next(index for index, label in enumerate(model.body_label) if label.endswith("object"))
    object_position = body_q[object_body, :3]

    pad_shapes: list[int] = []
    patch_transforms: list[wp.transform] = []
    patch_sizes: list[tuple[float, float]] = []
    for finger_name in ("leftfinger", "rightfinger"):
        finger_body = next(index for index, label in enumerate(model.body_label) if finger_name in label)
        candidates = np.flatnonzero(shape_body == finger_body)
        pad_shape = int(candidates.max())
        vertices = np.asarray(model.shape_source[pad_shape].vertices, dtype=np.float64) * shape_scale[pad_shape]
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        center = 0.5 * (lower + upper)
        extent = upper - lower
        thin_axis = int(np.argmin(extent))
        in_plane = [axis for axis in range(3) if axis != thin_axis]

        X_ws = _compose_transform(body_q[finger_body], shape_transform[pad_shape])
        thin_world = _quat_rotate(X_ws[3:7], np.eye(3)[thin_axis])
        center_world = X_ws[:3] + _quat_rotate(X_ws[3:7], center)
        sign = 1.0 if np.dot(object_position - center_world, thin_world) >= 0.0 else -1.0
        z_axis = sign * np.eye(3)[thin_axis]
        x_axis = np.eye(3)[in_plane[0]]
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.stack((x_axis, y_axis, z_axis), axis=1)
        origin = center + 0.5 * sign * extent[thin_axis] * np.eye(3)[thin_axis]

        pad_shapes.append(pad_shape)
        patch_transforms.append(wp.transform(wp.vec3(*origin), wp.quat(*_matrix_to_quaternion(rotation))))
        patch_sizes.append((float(extent[in_plane[0]]), float(extent[in_plane[1]])))
    return pad_shapes, patch_transforms, patch_sizes


def _object_shape(example: Example) -> int:
    model = example.model
    object_body = next(index for index, label in enumerate(model.body_label) if label.endswith("object"))
    return int(np.flatnonzero(model.shape_body.numpy() == object_body)[0])


def _state_world_frame(example: Example) -> np.ndarray:
    """Render synchronized orthographic views from the actual Newton body state."""
    width, height = 1280, 500
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    body_q = example.state_0.body_q.numpy()
    joint_parents = example.model.joint_parent.numpy()
    joint_children = example.model.joint_child.numpy()
    labels = tuple(example.model.body_label)
    object_body = next(index for index, label in enumerate(labels) if label.endswith("object"))
    cup_body = next((index for index, label in enumerate(labels) if label.endswith("cup")), None)

    views = (
        ("TOP VIEW  (world X / Y)", (18, 36, 630, 478), 0, 1, (-0.75, 0.45), (-0.92, 0.08)),
        ("SIDE VIEW  (world X / Z)", (650, 36, 1262, 478), 0, 2, (-0.75, 0.45), (-0.03, 0.85)),
    )

    def map_point(
        point: np.ndarray,
        bounds: tuple[int, int, int, int],
        horizontal_axis: int,
        vertical_axis: int,
        horizontal_range: tuple[float, float],
        vertical_range: tuple[float, float],
    ) -> tuple[int, int]:
        left, top, right, bottom = bounds
        x = (float(point[horizontal_axis]) - horizontal_range[0]) / (horizontal_range[1] - horizontal_range[0])
        y = (float(point[vertical_axis]) - vertical_range[0]) / (vertical_range[1] - vertical_range[0])
        return (
            int(round(left + x * (right - left))),
            int(round(bottom - y * (bottom - top))),
        )

    for title, bounds, horizontal_axis, vertical_axis, horizontal_range, vertical_range in views:
        left, top, right, bottom = bounds
        draw.rectangle(bounds, fill=(255, 255, 255), outline=(80, 80, 80), width=2)
        draw.text((left + 8, 12), title, fill=(20, 20, 20), font=font)
        for value in np.linspace(horizontal_range[0], horizontal_range[1], 7):
            point = np.zeros(3, dtype=np.float32)
            point[horizontal_axis] = value
            x, _ = map_point(point, bounds, horizontal_axis, vertical_axis, horizontal_range, vertical_range)
            draw.line((x, top, x, bottom), fill=(230, 230, 230), width=1)
        for value in np.linspace(vertical_range[0], vertical_range[1], 6):
            point = np.zeros(3, dtype=np.float32)
            point[vertical_axis] = value
            _, y = map_point(point, bounds, horizontal_axis, vertical_axis, horizontal_range, vertical_range)
            draw.line((left, y, right, y), fill=(230, 230, 230), width=1)

        table_center = np.asarray((0.08, -0.5, 0.05), dtype=np.float32)
        if vertical_axis == 1:
            corners = [
                table_center + np.asarray((dx, dy, 0.0), dtype=np.float32)
                for dx, dy in ((-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1))
            ]
            draw.polygon(
                [
                    map_point(point, bounds, horizontal_axis, vertical_axis, horizontal_range, vertical_range)
                    for point in corners
                ],
                fill=(225, 225, 225),
                outline=(125, 125, 125),
            )
        else:
            table_lower = map_point(
                table_center + np.asarray((-0.1, 0.0, -0.05), dtype=np.float32),
                bounds,
                horizontal_axis,
                vertical_axis,
                horizontal_range,
                vertical_range,
            )
            table_upper = map_point(
                table_center + np.asarray((0.1, 0.0, 0.05), dtype=np.float32),
                bounds,
                horizontal_axis,
                vertical_axis,
                horizontal_range,
                vertical_range,
            )
            draw.rectangle(
                (table_lower[0], table_upper[1], table_upper[0], table_lower[1]),
                fill=(225, 225, 225),
                outline=(125, 125, 125),
            )

        for parent_index, body_index in zip(joint_parents, joint_children, strict=True):
            parent_index = int(parent_index)
            body_index = int(body_index)
            if body_index in (object_body, cup_body) or parent_index < 0:
                continue
            start = map_point(
                body_q[parent_index, :3], bounds, horizontal_axis, vertical_axis, horizontal_range, vertical_range
            )
            end = map_point(
                body_q[body_index, :3], bounds, horizontal_axis, vertical_axis, horizontal_range, vertical_range
            )
            draw.line((*start, *end), fill=(45, 55, 70), width=7)

        for body_index, label in enumerate(labels):
            if body_index in (object_body, cup_body):
                continue
            point = map_point(
                body_q[body_index, :3], bounds, horizontal_axis, vertical_axis, horizontal_range, vertical_range
            )
            if "leftfinger" in label:
                color, radius = (20, 120, 230), 9
            elif "rightfinger" in label:
                color, radius = (20, 180, 120), 9
            else:
                color, radius = (60, 70, 85), 5
            draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color)

        object_position = body_q[object_body, :3]
        object_pixel = map_point(
            object_position, bounds, horizontal_axis, vertical_axis, horizontal_range, vertical_range
        )
        if example.scene.value == "cube":
            radius = 11
            draw.rectangle(
                (
                    object_pixel[0] - radius,
                    object_pixel[1] - radius,
                    object_pixel[0] + radius,
                    object_pixel[1] + radius,
                ),
                fill=(220, 55, 45),
                outline=(120, 20, 20),
                width=2,
            )
        else:
            axis_world = _quat_rotate(body_q[object_body, 3:7], np.asarray((0.0, 0.0, 0.07), dtype=np.float32))
            endpoints = [object_position - axis_world, object_position + axis_world]
            endpoint_pixels = [
                map_point(point, bounds, horizontal_axis, vertical_axis, horizontal_range, vertical_range)
                for point in endpoints
            ]
            draw.line((*endpoint_pixels[0], *endpoint_pixels[1]), fill=(220, 55, 45), width=9)

        if cup_body is not None:
            cup_pixel = map_point(
                body_q[cup_body, :3], bounds, horizontal_axis, vertical_axis, horizontal_range, vertical_range
            )
            draw.ellipse(
                (cup_pixel[0] - 14, cup_pixel[1] - 10, cup_pixel[0] + 14, cup_pixel[1] + 10),
                outline=(120, 70, 20),
                width=4,
            )

        draw.text(
            (left + 8, bottom - 20),
            "red=object  blue/green=finger bodies  gray=Panda links",
            fill=(35, 35, 35),
            font=font,
        )
    return np.asarray(image)


def _render_frame(
    scene: np.ndarray,
    normal: np.ndarray,
    shear: np.ndarray,
    evidence,
    frame_index: int,
    timestamp_s: float,
    object_lift_m: float,
    raw_count: int,
    conservation_residual_n: float,
    normal_scale_n: float,
    output: Path,
) -> None:
    canvas = Image.new("RGB", (1280, 720), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 20)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 15)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        title_font = label_font = small_font = ImageFont.load_default()
    world = Image.fromarray(scene).resize((1280, 380), Image.Resampling.LANCZOS)
    canvas.paste(world, (0, 26))
    draw.text(
        (640, 3),
        "Newton Panda | actual state projections + native solved-contact tactile",
        fill=(20, 20, 20),
        font=title_font,
        anchor="ma",
    )
    states = [SlipState(int(value)).name for value in evidence.state[0]]
    for patch_index, name in enumerate(("left pad", "right pad")):
        field = normal[patch_index]
        panel_left = 18 + patch_index * 630
        panel_top = 414
        panel_right = panel_left + 612
        panel_bottom = 686
        draw.rectangle(
            (panel_left, panel_top, panel_right, panel_bottom),
            fill=(250, 250, 250),
            outline=(90, 90, 90),
            width=2,
        )
        draw.text(
            (panel_left + 12, panel_top + 8),
            f"{name.upper()} | {states[patch_index]} | "
            f"Fn={evidence.normal_load_n[0, patch_index]:.3f} N | "
            f"Ft={evidence.tangential_load_n[0, patch_index]:.3f} N",
            fill=(20, 20, 20),
            font=label_font,
        )

        normalized = np.clip(field / max(normal_scale_n, 1.0e-9), -1.0, 1.0)
        heatmap = np.full((*field.shape, 3), 255.0, dtype=np.float32)
        positive = normalized > 0.0
        negative = normalized < 0.0
        heatmap[positive, 1] = 255.0 * (1.0 - normalized[positive])
        heatmap[positive, 2] = 255.0 * (1.0 - normalized[positive])
        heatmap[negative, 0] = 255.0 * (1.0 + normalized[negative])
        heatmap[negative, 1] = 255.0 * (1.0 + normalized[negative])
        heatmap_image = Image.fromarray(np.flipud(heatmap).astype(np.uint8)).resize(
            (500, 200), Image.Resampling.NEAREST
        )
        heat_left = panel_left + 72
        heat_top = panel_top + 42
        canvas.paste(heatmap_image, (heat_left, heat_top))
        draw.rectangle((heat_left, heat_top, heat_left + 500, heat_top + 200), outline=(60, 60, 60), width=2)
        rows, columns = field.shape
        cell_width = 500.0 / columns
        cell_height = 200.0 / rows
        shear_field = shear[patch_index]
        for row in range(0, rows, 2):
            for column in range(0, columns, 2):
                vector = shear_field[row, column]
                magnitude = float(np.linalg.norm(vector))
                if magnitude < normal_scale_n * 0.015:
                    continue
                center_x = heat_left + (column + 0.5) * cell_width
                center_y = heat_top + (rows - row - 0.5) * cell_height
                delta_x = float(vector[1] / max(normal_scale_n, 1.0e-9)) * cell_width * 2.5
                delta_y = -float(vector[0] / max(normal_scale_n, 1.0e-9)) * cell_height * 2.5
                draw.line(
                    (center_x, center_y, center_x + delta_x, center_y + delta_y),
                    fill=(10, 10, 10),
                    width=2,
                )
        draw.text(
            (heat_left + 250, panel_bottom - 23),
            "columns = local Y; rows = local X; arrows = signed XY shear",
            fill=(35, 35, 35),
            font=small_font,
            anchor="ma",
        )
        draw.text(
            (panel_left + 8, heat_top + 100),
            f"+{normal_scale_n:g} N red\n0 N white\n-{normal_scale_n:g} N blue",
            fill=(35, 35, 35),
            font=small_font,
            anchor="lm",
        )
    draw.text(
        (640, 703),
        f"frame={frame_index}  t={timestamp_s:.3f}s  lift={object_lift_m:+.3f}m  "
        f"raw samples={raw_count}  force conservation residual={conservation_residual_n:.2e}N  "
        "optical unavailable in Newton",
        fill=(25, 25, 25),
        font=small_font,
        anchor="ma",
    )
    canvas.save(output)


def main() -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    parser = Example.create_parser()
    parser.add_argument("--frames", type=int, default=420)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normal-scale-n", type=float, default=5.0)
    parser.add_argument("--friction-coefficient", type=float, default=0.5)
    parser.set_defaults(viewer="gl", headless=True, world_count=1)
    startup_time = time.monotonic()
    print("newton_tactile_stage=initialize_viewer", flush=True)
    viewer, args = newton.examples.init(parser)
    if args.world_count != 1:
        raise ValueError("The evidence video renders one world; use --world-count 1.")

    print("newton_tactile_stage=initialize_official_panda_hydro", flush=True)
    example = Example(viewer, args)
    print(
        f"newton_tactile_stage=official_panda_hydro_ready elapsed_s={time.monotonic() - startup_time:.3f}",
        flush=True,
    )
    pad_shapes, patch_transforms, patch_sizes = _pad_patch_geometry(example)
    sensor = SensorTactile(
        example.model,
        sensing_shapes=pad_shapes,
        counterpart_shapes=[_object_shape(example)],
        grid_shape=(20, 25),
        patch_size=patch_sizes,
        patch_transform_shape=patch_transforms,
    )
    example.contacts = example.collision_pipeline.contacts()
    print("newton_tactile_stage=native_sensor_ready", flush=True)

    adapter = NewtonTactileAdapter(sensor, ("left_pad", "right_pad"))
    detector = TactileSlipDetector(
        adapter.patch_names,
        friction_coefficient=args.friction_coefficient,
    )
    object_body = next(index for index, label in enumerate(example.model.body_label) if label.endswith("object"))
    initial_object_z = float(example.state_0.body_q.numpy()[object_body, 2])

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / f".{output.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    records = []
    max_residual = 0.0
    max_normal_cell = 0.0
    max_shear_cell = 0.0
    contact_frames = np.zeros(2, dtype=np.int64)
    state_counts = np.zeros((2, 4), dtype=np.int64)
    max_lift = 0.0
    for frame_index in range(args.frames):
        example.step()
        example.solver.update_contacts(example.contacts, example.state_0)
        sensor.update(example.state_0, example.contacts, timestamp=example.sim_time)
        tactile = adapter.frame()
        evidence = detector.update(tactile)
        scene = _state_world_frame(example)

        normal = tactile.normal_force_n[0]
        shear = tactile.shear_force_xy_n[0]
        dense_sum = sensor.force.numpy().sum(axis=1)
        residual = sensor.total_force_patch.numpy() - dense_sum - sensor.unmapped_force_patch.numpy()
        residual_max = float(np.abs(residual).max())
        max_residual = max(max_residual, residual_max)
        max_normal_cell = max(max_normal_cell, float(np.abs(normal).max()))
        max_shear_cell = max(max_shear_cell, float(np.linalg.norm(shear, axis=-1).max()))
        contact_frames += tactile.active[0].any(axis=(1, 2))
        for patch_index in range(2):
            state_counts[patch_index, int(evidence.state[0, patch_index])] += 1
        object_z = float(example.state_0.body_q.numpy()[object_body, 2])
        object_lift = object_z - initial_object_z
        max_lift = max(max_lift, object_lift)
        raw_count = int(sensor.raw_count.numpy()[0])
        records.append(
            {
                "frame": frame_index,
                "timestamp_s": example.sim_time,
                "tactile_sequence": tactile.clock.sequence,
                "tactile_timestamp_s": tactile.clock.timestamp_s,
                "tactile_dt_s": tactile.clock.dt_s,
                "object_lift_m": object_lift,
                "raw_sample_count": raw_count,
                "force_conservation_residual_n": residual_max,
                "slip_state": evidence.state[0].astype(int).tolist(),
                "normal_load_n": evidence.normal_load_n[0].tolist(),
                "tangential_load_n": evidence.tangential_load_n[0].tolist(),
                "cop_speed_m_s": evidence.center_of_pressure_speed_m_s[0].tolist(),
            }
        )
        _render_frame(
            scene,
            normal,
            shear,
            evidence,
            frame_index,
            example.sim_time,
            object_lift,
            raw_count,
            residual_max,
            args.normal_scale_n,
            frame_dir / f"frame_{frame_index:05d}.png",
        )

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-framerate",
        str(example.fps),
        "-i",
        str(frame_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(frame_dir)

    summary = {
        "scene": args.scene,
        "frames": args.frames,
        "fps": example.fps,
        "sensor": "newton.sensors.SensorTactile",
        "source": "Contacts.force after solver.update_contacts",
        "clock_fields": [
            "tactile_sequence",
            "tactile_timestamp_s",
            "tactile_dt_s",
        ],
        "patch_shapes": pad_shapes,
        "patch_sizes_m": patch_sizes,
        "grid_shape": [20, 25],
        "contact_frames_by_patch": contact_frames.tolist(),
        "state_counts_by_patch": state_counts.tolist(),
        "maximum_object_lift_m": max_lift,
        "maximum_force_conservation_residual_n": max_residual,
        "maximum_abs_normal_cell_n": max_normal_cell,
        "maximum_shear_cell_n": max_shear_cell,
        "normal_render_scale_n": args.normal_scale_n,
        "optical_available": False,
        "records": records,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    print(f"video={output}")
    faulthandler.cancel_dump_traceback_later()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
