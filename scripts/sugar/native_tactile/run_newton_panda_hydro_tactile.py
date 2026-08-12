#!/usr/bin/env python3
"""Run Newton's dynamic Panda hydroelastic pickup with native tactile output."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import newton
import newton.examples
from newton.examples.robot.example_robot_panda_hydro import Example
from newton.sensors import SensorTactile

from tactile_video import _object_shape, _pad_patch_geometry
from scripts.sugar.native_tactile.run_newton_softbody_franka_tactile import NewtonVTKRenderer
from scripts.sugar.native_tactile.slip import SlipState, TactileSlipDetector
from scripts.sugar.native_tactile.universal import NewtonTactileAdapter


FONT = ImageFont.load_default()


class TactilePandaHydroExample(Example):
    """Allocate force-reporting contacts before the official CUDA graph capture."""

    def capture(self) -> None:
        self.model.request_contact_attributes("force")
        self.contacts = self.collision_pipeline.contacts()
        super().capture()


def _signed_normal_image(values: np.ndarray, scale: float, size: tuple[int, int]) -> Image.Image:
    normalized = np.clip(values / max(scale, 1.0e-9), -1.0, 1.0)
    rgb = np.full((*values.shape, 3), 255.0, dtype=np.float32)
    positive = normalized > 0.0
    negative = normalized < 0.0
    rgb[positive, 1] = 255.0 * (1.0 - normalized[positive])
    rgb[positive, 2] = 255.0 * (1.0 - normalized[positive])
    rgb[negative, 0] = 255.0 * (1.0 + normalized[negative])
    rgb[negative, 1] = 255.0 * (1.0 + normalized[negative])
    return Image.fromarray(np.flipud(rgb).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)


def _compose(
    world: np.ndarray,
    tactile,
    evidence,
    *,
    frame: int,
    timestamp_s: float,
    lift_m: float,
    raw_count: int,
    residual_n: float,
    normal_scale_n: float,
) -> np.ndarray:
    canvas = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(world).resize((1280, 500), Image.Resampling.LANCZOS), (0, 28))
    draw.text((20, 8), "Newton official Panda hydro pickup | exact geometry + solved native tactile", fill="black", font=FONT)
    draw.text(
        (850, 8),
        f"frame {frame:04d}  t={timestamp_s:.2f}s  lift={lift_m:+.3f}m  raw={raw_count}  residual={residual_n:.2e}N",
        fill="black",
        font=FONT,
    )

    for patch, side in enumerate(("LEFT PAD", "RIGHT PAD")):
        x = 20 + patch * 640
        normal = tactile.normal_force_n[0, patch]
        shear = tactile.shear_force_xy_n[0, patch]
        canvas.paste(_signed_normal_image(normal, normal_scale_n, (560, 145)), (x + 40, 555))
        state = SlipState(int(evidence.state[0, patch])).name
        draw.text(
            (x, 535),
            f"{side} | {state} | signed normal red/blue | shear arrows | "
            f"Fn={evidence.normal_load_n[0, patch]:.3f}N Ft={evidence.tangential_load_n[0, patch]:.3f}N",
            fill="black",
            font=FONT,
        )
        rows, columns = normal.shape
        for row in range(0, rows, 2):
            for column in range(0, columns, 2):
                vector = shear[row, column]
                magnitude = float(np.linalg.norm(vector))
                if magnitude < normal_scale_n * 0.015:
                    continue
                cx = x + 40 + (column + 0.5) * 560.0 / columns
                cy = 555 + (rows - row - 0.5) * 145.0 / rows
                dx = float(vector[1]) / max(normal_scale_n, 1.0e-9) * 16.0
                dy = -float(vector[0]) / max(normal_scale_n, 1.0e-9) * 16.0
                draw.line((cx, cy, cx + dx, cy + dy), fill="black", width=2)
    draw.text(
        (20, 704),
        "Every colored cell is a conservative raster of Contacts.force after SolverMuJoCo.update_contacts; Newton optical is unavailable.",
        fill="black",
        font=FONT,
    )
    return np.asarray(canvas)


def _pack_raw(raw_rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    frames = len(raw_rows)
    capacity = max((len(row["contact_index"]) for row in raw_rows), default=0)
    packed = {
        "raw_count": np.zeros(frames, dtype=np.int32),
        "raw_contact_index": np.full((frames, capacity), -1, dtype=np.int32),
        "raw_contact_kind": np.full((frames, capacity), -1, dtype=np.int32),
        "raw_patch": np.full((frames, capacity), -1, dtype=np.int32),
        "raw_counterpart_shape": np.full((frames, capacity), -1, dtype=np.int32),
        "raw_counterpart_particle": np.full((frames, capacity), -1, dtype=np.int32),
        "raw_sensor_is_shape0": np.zeros((frames, capacity), dtype=bool),
        "raw_point_world_m": np.zeros((frames, capacity, 3), dtype=np.float32),
        "raw_point_patch_m": np.zeros((frames, capacity, 3), dtype=np.float32),
        "raw_force_world_n": np.zeros((frames, capacity, 3), dtype=np.float32),
        "raw_force_patch_n": np.zeros((frames, capacity, 3), dtype=np.float32),
        "raw_native_wrench_body0": np.zeros((frames, capacity, 6), dtype=np.float32),
        "raw_penetration_m": np.zeros((frames, capacity), dtype=np.float32),
    }
    for frame, row in enumerate(raw_rows):
        count = len(row["contact_index"])
        packed["raw_count"][frame] = count
        for key in packed:
            if key == "raw_count":
                continue
            source_key = key.removeprefix("raw_")
            packed[key][frame, :count] = row[source_key]
    return packed


def main() -> None:
    parser = Example.create_parser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--normal-scale-n", type=float, default=5.0)
    parser.set_defaults(viewer="null", headless=True, world_count=1, num_frames=420)
    viewer, args = newton.examples.init(parser)
    if args.world_count != 1:
        raise ValueError("The tactile evidence runner requires world-count=1.")
    frame_stop = min(args.num_frames, args.frame_stop or args.num_frames)
    if not (0 <= args.frame_start < frame_stop):
        raise ValueError("The selected simulation-frame interval is empty.")
    rendered_frames = frame_stop - args.frame_start

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / ".frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir()

    example = TactilePandaHydroExample(viewer, args)
    pad_shapes, patch_transforms, patch_sizes = _pad_patch_geometry(example)
    sensor = SensorTactile(
        example.model,
        sensing_shapes=pad_shapes,
        counterpart_shapes=[_object_shape(example)],
        grid_shape=(20, 25),
        patch_size=patch_sizes,
        patch_transform_shape=patch_transforms,
    )
    adapter = NewtonTactileAdapter(sensor, ("left_pad", "right_pad"))
    detector = TactileSlipDetector(adapter.patch_names, friction_coefficient=0.5)
    renderer = NewtonVTKRenderer(
        example.model,
        camera_position=(0.65, 0.25, 0.65),
        camera_target=(-0.05, -0.50, 0.28),
    )
    object_body = next(index for index, label in enumerate(example.model.body_label) if label.endswith("object"))
    initial_object_z = float(example.state_0.body_q.numpy()[object_body, 2])

    force_rows: list[np.ndarray] = []
    penetration_rows: list[np.ndarray] = []
    active_rows: list[np.ndarray] = []
    taxel_position_rows: list[np.ndarray] = []
    taxel_orientation_rows: list[np.ndarray] = []
    sequence_rows: list[int] = []
    timestamp_rows: list[float] = []
    dt_rows: list[float] = []
    object_position_rows: list[np.ndarray] = []
    raw_rows: list[dict[str, np.ndarray]] = []
    records: list[dict] = []
    contact_frames = np.zeros(2, dtype=np.int64)
    max_residual = 0.0

    for source_frame in range(frame_stop):
        example.step()
        example.solver.update_contacts(example.contacts, example.state_0)
        sensor.update(example.state_0, example.contacts, timestamp=example.sim_time)
        tactile = adapter.frame()
        evidence = detector.update(tactile)
        if source_frame < args.frame_start:
            continue

        output_frame = source_frame - args.frame_start
        force = sensor.force.numpy().reshape(2, 20, 25, 3).copy()
        penetration = sensor.max_penetration.numpy().reshape(2, 20, 25).copy()
        active = sensor.active.numpy().reshape(2, 20, 25).astype(bool).copy()
        dense_sum = force.sum(axis=(1, 2))
        residual = sensor.total_force_patch.numpy() - dense_sum
        residual_n = float(np.abs(residual).max())
        max_residual = max(max_residual, residual_n)
        contact_frames += (np.linalg.norm(force, axis=-1) > 1.0e-8).any(axis=(1, 2))

        raw = tactile.raw_samples
        if raw is None:
            raise RuntimeError("Newton universal frame did not preserve native raw samples.")
        raw_count = len(raw.contact_index)
        raw_rows.append(
            {
                "contact_index": raw.contact_index,
                "contact_kind": raw.contact_kind,
                "patch": raw.patch_index,
                "counterpart_shape": raw.counterpart_shape,
                "counterpart_particle": raw.counterpart_particle,
                "sensor_is_shape0": raw.sensor_is_shape0,
                "point_world_m": raw.point_world_m,
                "point_patch_m": raw.point_patch_m,
                "force_world_n": raw.force_world_n,
                "force_patch_n": raw.force_patch_n,
                "native_wrench_body0": raw.native_wrench_body0,
                "penetration_m": raw.penetration_m,
            }
        )
        object_position = example.state_0.body_q.numpy()[object_body, :3].copy()
        lift_m = float(object_position[2] - initial_object_z)
        force_rows.append(force)
        penetration_rows.append(penetration)
        active_rows.append(active)
        taxel_position_rows.append(np.asarray(tactile.taxel_position_w_m[0]).copy())
        taxel_orientation_rows.append(np.asarray(tactile.taxel_orientation_w_xyzw[0]).copy())
        sequence_rows.append(tactile.clock.sequence)
        timestamp_rows.append(tactile.clock.timestamp_s)
        dt_rows.append(tactile.clock.dt_s)
        object_position_rows.append(object_position)
        records.append(
            {
                "source_frame": source_frame,
                "timestamp_s": tactile.clock.timestamp_s,
                "raw_sample_count": raw_count,
                "force_conservation_residual_n": residual_n,
                "object_lift_m": lift_m,
                "slip_state": evidence.state[0].astype(int).tolist(),
            }
        )
        world = renderer.render(example.state_0)
        frame = _compose(
            world,
            tactile,
            evidence,
            frame=source_frame,
            timestamp_s=tactile.clock.timestamp_s,
            lift_m=lift_m,
            raw_count=raw_count,
            residual_n=residual_n,
            normal_scale_n=args.normal_scale_n,
        )
        Image.fromarray(frame).save(frame_dir / f"frame_{output_frame:05d}.png")
        if output_frame % 50 == 0:
            print(
                f"panda_hydro frame={source_frame} output={output_frame}/{rendered_frames} "
                f"lift={lift_m:+.4f} raw={raw_count} residual={residual_n:.3e}",
                flush=True,
            )

    trace = {
        "force_patch_n": np.stack(force_rows),
        "penetration_m": np.stack(penetration_rows),
        "active": np.stack(active_rows),
        "taxel_position_w_m": np.stack(taxel_position_rows),
        "taxel_orientation_w_xyzw": np.stack(taxel_orientation_rows),
        "tactile_sequence": np.asarray(sequence_rows, dtype=np.int64),
        "tactile_timestamp_s": np.asarray(timestamp_rows, dtype=np.float64),
        "tactile_dt_s": np.asarray(dt_rows, dtype=np.float64),
        "source_frame": np.arange(args.frame_start, frame_stop, dtype=np.int32),
        "object_position_w_m": np.stack(object_position_rows),
        "patch_names": np.asarray(adapter.patch_names),
        "patch_size_m": tactile.patch_size_m.copy(),
        "backend": np.asarray(tactile.backend),
        "optical_available": np.asarray(tactile.optical.available, dtype=bool),
        **_pack_raw(raw_rows),
    }
    np.savez_compressed(output_root / "trace.npz", **trace)
    summary = {
        "schema": "newton_mujoco_official_panda_hydro_native_tactile_v1",
        "scene": args.scene,
        "frames": rendered_frames,
        "source_frame_interval": [args.frame_start, frame_stop],
        "fps": example.fps,
        "patch_names": list(adapter.patch_names),
        "patch_shapes": pad_shapes,
        "patch_sizes_m": patch_sizes,
        "grid_shape": [20, 25],
        "taxel_position_shape": list(trace["taxel_position_w_m"].shape),
        "taxel_orientation_shape": list(trace["taxel_orientation_w_xyzw"].shape),
        "taxel_quaternion_order": "xyzw",
        "tactile_clock_fields": ["tactile_sequence", "tactile_timestamp_s", "tactile_dt_s"],
        "raw_sample_fields": [key for key in trace if key.startswith("raw_")],
        "contact_frames_per_patch": contact_frames.tolist(),
        "maximum_object_lift_m": float(np.max(np.stack(object_position_rows)[:, 2] - initial_object_z)),
        "maximum_force_conservation_residual_n": max_residual,
        "native_force_source": "SolverMuJoCo constraint force exported by update_contacts",
        "world_renderer": "VTK EGL rendering of exact Newton model geometry and live dynamic state",
        "optical_available": False,
        "training": False,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "frames.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    video = output_root / "newton_panda_hydro_native_tactile.mp4"
    subprocess.run(
        [
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
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    shutil.rmtree(frame_dir)
    renderer.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
