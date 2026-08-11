# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native stick-to-slide evidence for ``SensorTactile``."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont

import newton
from newton.sensors import SensorTactile
from scripts.sugar.native_tactile.slip import SlipState, TactileSlipDetector
from scripts.sugar.native_tactile.universal import NewtonTactileAdapter


def _plate_motion(frame: int, fps: int) -> tuple[str, float, float]:
    if frame < 60:
        return "stationary", 0.0, 0.0
    if frame < 120:
        t = (frame - 60) / fps
        omega = 2.0 * np.pi
        return "slow", 0.0005 * (1.0 - np.cos(omega * t)), 0.0005 * omega * np.sin(omega * t)
    if frame < 180:
        t = (frame - 120) / fps
        omega = 2.0 * np.pi
        return "incipient", 0.003 * np.sin(omega * t), 0.003 * omega * np.cos(omega * t)
    if frame < 240:
        t = (frame - 180) / fps
        omega = 10.0 * np.pi
        return "fast", 0.025 * np.sin(omega * t), 0.025 * omega * np.cos(omega * t)
    return "stationary_after", 0.0, 0.0


def _render(
    output: Path,
    *,
    frame: int,
    fps: int,
    phase: str,
    plate_x: float,
    object_position: np.ndarray,
    normal: np.ndarray,
    shear: np.ndarray,
    state: SlipState,
    normal_load: float,
    tangential_load: float,
    relative_speed: float,
    active_taxels: int,
) -> None:
    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    try:
        title = ImageFont.truetype("DejaVuSans.ttf", 22)
        label = ImageFont.truetype("DejaVuSans.ttf", 17)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        title = label = small = ImageFont.load_default()

    draw.text((640, 15), "Newton tactile-only slip", fill=(20, 20, 20), font=title, anchor="ma")
    draw.text(
        (640, 48),
        f"{phase} | {state.name} | held-out vrel={relative_speed:.4f} m/s",
        fill=(20, 20, 20),
        font=label,
        anchor="ma",
    )

    world = (55, 82, 1225, 335)
    draw.rectangle(world, fill=(249, 249, 249), outline=(70, 70, 70), width=2)
    x_min, x_max = -0.16, 0.16
    z_min, z_max = -0.03, 0.14

    def pixel(x: float, z: float) -> tuple[int, int]:
        px = world[0] + int((x - x_min) / (x_max - x_min) * (world[2] - world[0]))
        pz = world[3] - int((z - z_min) / (z_max - z_min) * (world[3] - world[1]))
        return px, pz

    for x in np.linspace(x_min, x_max, 9):
        px, _ = pixel(float(x), 0.0)
        draw.line((px, world[1], px, world[3]), fill=(230, 230, 230), width=1)
    for z in np.linspace(z_min, z_max, 6):
        _, pz = pixel(0.0, float(z))
        draw.line((world[0], pz, world[2], pz), fill=(230, 230, 230), width=1)
    plate_left, plate_top = pixel(plate_x - 0.10, 0.01)
    plate_right, plate_bottom = pixel(plate_x + 0.10, -0.01)
    draw.rectangle(
        (plate_left, plate_top, plate_right, plate_bottom),
        fill=(80, 130, 210),
        outline=(20, 60, 130),
        width=3,
    )
    object_left, object_top = pixel(float(object_position[0]) - 0.03, float(object_position[2]) + 0.03)
    object_right, object_bottom = pixel(float(object_position[0]) + 0.03, float(object_position[2]) - 0.03)
    draw.rectangle(
        (object_left, object_top, object_right, object_bottom),
        fill=(220, 65, 45),
        outline=(120, 20, 20),
        width=3,
    )
    draw.text((70, 95), "actual state: blue=plate, red=cube", fill=(30, 30, 30), font=small)

    panel = (55, 365, 1225, 680)
    draw.rectangle(panel, fill=(250, 250, 250), outline=(70, 70, 70), width=2)
    draw.text(
        (75, 382),
        f"solved field | Fn={normal_load:.3f} N | Ft={tangential_load:.3f} N | active={active_taxels}",
        fill=(20, 20, 20),
        font=label,
    )
    scale = 2.0
    normalized = np.clip(normal / scale, -1.0, 1.0)
    heatmap = np.full((*normal.shape, 3), 255.0, dtype=np.float32)
    positive = normalized > 0.0
    negative = normalized < 0.0
    heatmap[positive, 1:] = 255.0 * (1.0 - normalized[positive, None])
    heatmap[negative, :2] = 255.0 * (1.0 + normalized[negative, None])
    heat = Image.fromarray(np.flipud(heatmap).astype(np.uint8)).resize((900, 225), Image.Resampling.NEAREST)
    heat_left, heat_top = 285, 425
    image.paste(heat, (heat_left, heat_top))
    draw.rectangle((heat_left, heat_top, heat_left + 900, heat_top + 225), outline=(60, 60, 60), width=2)
    rows, columns = normal.shape
    cell_width, cell_height = 900.0 / columns, 225.0 / rows
    for row in range(0, rows, 2):
        for column in range(0, columns, 2):
            vector = shear[row, column]
            if float(np.linalg.norm(vector)) < 0.03:
                continue
            x = heat_left + (column + 0.5) * cell_width
            y = heat_top + (rows - row - 0.5) * cell_height
            dx = float(vector[1] / scale) * cell_width * 2.0
            dy = -float(vector[0] / scale) * cell_height * 2.0
            draw.line((x, y, x + dx, y + dy), fill=(10, 10, 10), width=2)
    draw.text((75, 520), "+2 N red\n0 N white\n-2 N blue", fill=(30, 30, 30), font=small, anchor="lm")
    draw.text((735, 660), "row=X; column=Y; arrows=XY shear", fill=(30, 30, 30), font=small, anchor="ma")
    draw.text(
        (640, 702),
        f"frame={frame} t={(frame + 1) / fps:.3f}s | vrel is held out from detection",
        fill=(25, 25, 25),
        font=small,
        anchor="ma",
    )
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    wp.set_device(args.device)

    friction = 0.005
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=friction, ke=1.0e5, kd=1.0e3)
    pad_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(0.0),
        inertia=wp.mat33(np.eye(3)),
        is_kinematic=True,
        label="tactile_plate",
    )
    pad_shape = builder.add_shape_box(pad_body, hx=0.10, hy=0.10, hz=0.01, cfg=cfg, label="tactile_plate_shape")
    object_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.045), wp.quat_identity()),
        mass=0.25,
        com=wp.vec3(0.0),
        inertia=wp.mat33(np.eye(3) * 1.5e-4),
        label="dynamic_cube",
    )
    object_shape = builder.add_shape_box(object_body, hx=0.03, hy=0.03, hz=0.03, cfg=cfg, label="dynamic_cube_shape")
    model = builder.finalize(device=args.device)
    sensor = SensorTactile(
        model,
        sensing_shapes=[pad_shape],
        counterpart_shapes=[object_shape],
        grid_shape=(20, 25),
        patch_size=(0.20, 0.20),
        patch_transform_shape=[wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity())],
    )
    contacts = model.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    solver = newton.solvers.SolverXPBD(model, iterations=20)
    adapter = NewtonTactileAdapter(sensor, ("tactile_plate",))
    detector = TactileSlipDetector(("tactile_plate",), friction_coefficient=friction)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / f".{output.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    fps, substeps = 60, 4
    dt = 1.0 / fps / substeps
    records = []

    for frame in range(args.frames):
        phase, plate_x, plate_vx = _plate_motion(frame, fps)
        for _ in range(substeps):
            poses = state_0.body_q.numpy()
            velocities = state_0.body_qd.numpy()
            poses[pad_body, :3] = (plate_x, 0.0, 0.0)
            poses[pad_body, 3:7] = (0.0, 0.0, 0.0, 1.0)
            velocities[pad_body] = 0.0
            velocities[pad_body, 0] = plate_vx
            state_0.body_q.assign(poses)
            state_0.body_qd.assign(velocities)
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        solver.update_contacts(contacts, state_0)
        sensor.update(state_0, contacts, timestamp=(frame + 1) / fps)
        tactile = adapter.frame()
        evidence = detector.update(tactile)
        body_q = state_0.body_q.numpy()
        body_qd = state_0.body_qd.numpy()
        actual_plate_x = float(body_q[pad_body, 0])
        actual_plate_velocity_xy = body_qd[pad_body, :2]
        relative_xy = body_qd[object_body, :2] - actual_plate_velocity_xy
        relative_speed = float(np.linalg.norm(relative_xy))
        active = bool(tactile.active[0, 0].any())
        state = SlipState(int(evidence.state[0, 0]))
        records.append(
            {
                "frame": frame,
                "phase": phase,
                "commanded_plate_x_m": plate_x,
                "commanded_plate_vx_m_s": plate_vx,
                "actual_plate_x_m": actual_plate_x,
                "actual_plate_vx_m_s": float(actual_plate_velocity_xy[0]),
                "object_x_m": float(body_q[object_body, 0]),
                "heldout_relative_tangential_speed_m_s": relative_speed,
                "contact": active,
                "detector_state": int(state),
                "normal_load_n": float(evidence.normal_load_n[0, 0]),
                "tangential_load_n": float(evidence.tangential_load_n[0, 0]),
            }
        )
        _render(
            frame_dir / f"frame_{frame:05d}.png",
            frame=frame,
            fps=fps,
            phase=phase,
            plate_x=actual_plate_x,
            object_position=body_q[object_body, :3],
            normal=tactile.normal_force_n[0, 0],
            shear=tactile.shear_force_xy_n[0, 0],
            state=state,
            normal_load=float(evidence.normal_load_n[0, 0]),
            tangential_load=float(evidence.tangential_load_n[0, 0]),
            relative_speed=relative_speed,
            active_taxels=int(tactile.active[0, 0].sum()),
        )

    contact = np.asarray([bool(row["contact"]) for row in records])
    truth = np.asarray([float(row["heldout_relative_tangential_speed_m_s"]) >= 0.005 for row in records]) & contact
    predicted = np.asarray([int(row["detector_state"]) >= int(SlipState.INCIPIENT) for row in records]) & contact
    speed = np.asarray([float(row["heldout_relative_tangential_speed_m_s"]) for row in records])
    truth_state = np.full(args.frames, int(SlipState.STICK), dtype=np.int8)
    truth_state[~contact] = int(SlipState.NO_CONTACT)
    truth_state[contact & (speed >= 0.005)] = int(SlipState.INCIPIENT)
    truth_state[contact & (speed >= 0.02)] = int(SlipState.GROSS)
    predicted_state = np.asarray([int(row["detector_state"]) for row in records], dtype=np.int8)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))
    summary = {
        "frames": args.frames,
        "fps": fps,
        "sensor": "newton.sensors.SensorTactile",
        "tactile_input": "Contacts.force",
        "heldout_label": "vrel>=.005 m/s on contact",
        "mu": friction,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "ordinal_accuracy": float(np.mean(predicted_state == truth_state)),
        "contact_frames": int(contact.sum()),
        "state_counts": np.bincount([int(row["detector_state"]) for row in records], minlength=4).tolist(),
        "max_heldout_vrel_m_s": max(float(row["heldout_relative_tangential_speed_m_s"]) for row in records),
        "records": records,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-framerate",
        str(fps),
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
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    print(f"video={output}")


if __name__ == "__main__":
    main()
