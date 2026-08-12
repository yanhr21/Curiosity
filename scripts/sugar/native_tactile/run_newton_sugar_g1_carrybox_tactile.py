#!/usr/bin/env python3
"""Replay official SUGAR CarryBox geometry in Newton with native solved tactile force."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Usd
import warp as wp

import newton
from newton.sensors import SensorTactile

from scripts.sugar.native_tactile.run_newton_softbody_franka_tactile import NewtonVTKRenderer
from scripts.sugar.native_tactile.slip import SlipState, TactileSlipDetector
from scripts.sugar.native_tactile.universal import NewtonTactileAdapter
from scripts.sugar.smp.sugar_g1_box_schema import G1_JOINT_NAMES


ROOT = Path(os.environ.get("CURIOSITY_ROOT", Path(__file__).resolve().parents[3])).resolve()
URDF = ROOT / "SUGAR/descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
BOX_USD = ROOT / "SUGAR/descriptions/objects/small_box/obj_aligned.usd"
MOTION = ROOT / "SUGAR/data/CarryBox/data_045"
FONT = ImageFont.load_default()


def _matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float32,
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
                ],
                dtype=np.float32,
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=np.float32,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=np.float32,
            )
    return quaternion / np.linalg.norm(quaternion)


def _load_box_mesh() -> newton.Mesh:
    stage = Usd.Stage.Open(str(BOX_USD))
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() == "Mesh":
            return newton.usd.get_mesh(prim)
    raise RuntimeError(f"No mesh found in {BOX_USD}")


def _hand_collision_shapes(builder: newton.ModelBuilder) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    shapes: list[int] = []
    centers: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    for side in ("left", "right"):
        candidates = []
        for shape, body in enumerate(builder.shape_body):
            body_name = builder.body_label[body] if body >= 0 else ""
            flags = int(builder.shape_flags[shape])
            if body_name.endswith(f"/{side}_rubber_hand") and flags & int(newton.ShapeFlags.COLLIDE_SHAPES):
                candidates.append(shape)
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one exact {side} rubber-hand collision mesh, got {candidates}.")
        shape = candidates[0]
        vertices = np.asarray(builder.shape_source[shape].vertices, dtype=np.float32)
        vertices *= np.asarray(builder.shape_scale[shape], dtype=np.float32)[None]
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        shapes.append(shape)
        centers.append(0.5 * (lower + upper))
        sizes.append(upper - lower)
    return shapes, centers, sizes


def _hand_masks(
    model: newton.Model,
    hand_shapes: list[int],
    centers: list[np.ndarray],
    sizes: list[np.ndarray],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    rows, columns = grid_shape
    masks = np.zeros((2, rows, columns), dtype=bool)
    scale = model.shape_scale.numpy()
    for hand, shape in enumerate(hand_shapes):
        vertices = np.asarray(model.shape_source[shape].vertices, dtype=np.float32) * scale[shape][None]
        row = np.rint((vertices[:, 0] - centers[hand][0]) / sizes[hand][0] * (rows - 1) + 0.5 * (rows - 1))
        column = np.rint(
            (vertices[:, 1] - centers[hand][1]) / sizes[hand][1] * (columns - 1) + 0.5 * (columns - 1)
        )
        row = np.clip(row.astype(np.int32), 0, rows - 1)
        column = np.clip(column.astype(np.int32), 0, columns - 1)
        masks[hand, row, column] = True
        expanded = masks[hand].copy()
        for row_shift in (-1, 0, 1):
            for column_shift in (-1, 0, 1):
                expanded |= np.roll(np.roll(masks[hand], row_shift, axis=0), column_shift, axis=1)
        masks[hand] = expanded
    return masks


def _hand_map(force: np.ndarray, mask: np.ndarray, scale_n: float, size: tuple[int, int]) -> Image.Image:
    magnitude = np.linalg.norm(force, axis=-1)
    normalized = np.clip(magnitude / max(scale_n, 1.0e-9), 0.0, 1.0)
    rgb = np.full((*magnitude.shape, 3), 255, dtype=np.uint8)
    rgb[mask] = np.asarray((225, 228, 232), dtype=np.uint8)
    active = magnitude > 0.0
    rgb[..., 0][active] = 255
    rgb[..., 1][active] = np.asarray(220.0 * (1.0 - normalized[active]), dtype=np.uint8)
    rgb[..., 2][active] = np.asarray(40.0 * (1.0 - normalized[active]), dtype=np.uint8)
    return Image.fromarray(np.flipud(rgb)).resize(size, Image.Resampling.NEAREST)


def _compose(
    world: np.ndarray,
    force: np.ndarray,
    masks: np.ndarray,
    evidence,
    *,
    source_frame: int,
    timestamp_s: float,
    raw_count: int,
    residual_n: float,
    force_scale_n: float,
) -> np.ndarray:
    canvas = Image.new("RGB", (1280, 720), "white")
    canvas.paste(Image.fromarray(world).resize((1280, 500), Image.Resampling.LANCZOS), (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 9),
        "Newton VBD | official SUGAR G1 + CarryBox geometry | native solved hand contact force",
        fill="black",
        font=FONT,
    )
    draw.text(
        (930, 9),
        f"source frame {source_frame:03d}  t={timestamp_s:5.2f}s  raw={raw_count:3d}  residual={residual_n:.1e} N",
        fill="black",
        font=FONT,
    )
    for hand, side in enumerate(("LEFT", "RIGHT")):
        x = 20 + hand * 640
        canvas.paste(_hand_map(force[hand], masks[hand], force_scale_n, (600, 145)), (x, 552))
        state = SlipState(int(evidence.state[0, hand])).name
        normal = float(evidence.normal_load_n[0, hand])
        shear = float(evidence.tangential_load_n[0, hand])
        draw.text(
            (x, 536),
            f"{side} exact rubber-hand surface projection | {state} | Fn={normal:.2f} N | Ft={shear:.2f} N",
            fill="black",
            font=FONT,
        )
    draw.text(
        (20, 704),
        "Gray is the projected exact rubber-hand mesh; orange/red is solved force. Replay is kinematic; force is VBD penalty+damping+friction, not a measured hardware calibration.",
        fill="black",
        font=FONT,
    )
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=660)
    parser.add_argument("--force-scale-n", type=float, default=25.0)
    parser.add_argument("--renderer-refresh-frames", type=int, default=40)
    parser.add_argument("--dynamic-box", action="store_true")
    args = parser.parse_args()

    with np.load(MOTION / "robot_50hz.npz") as archive:
        robot = {key: np.asarray(archive[key]) for key in archive.files}
    with (MOTION / "obj_motion_global_50hz.pkl").open("rb") as stream:
        object_motion = pickle.load(stream)
    frame_stop = min(args.frame_stop, len(robot["joint_pos"]))
    if not (0 <= args.frame_start < frame_stop):
        raise ValueError("The selected source-frame interval is empty.")

    builder = newton.ModelBuilder(gravity=-9.81 if args.dynamic_box else 0.0)
    builder.add_urdf(
        str(URDF),
        floating=True,
        collapse_fixed_joints=False,
        force_show_colliders=False,
        enable_self_collisions=False,
    )
    robot_body_count = builder.body_count
    for body in range(robot_body_count):
        builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
    hand_shapes, hand_centers, hand_sizes = _hand_collision_shapes(builder)
    for shape in range(builder.shape_count):
        visible = int(builder.shape_flags[shape]) & int(newton.ShapeFlags.VISIBLE)
        builder.shape_flags[shape] = visible
    for shape in hand_shapes:
        builder.shape_flags[shape] = int(newton.ShapeFlags.COLLIDE_SHAPES)

    box_mesh = _load_box_mesh()
    box_vertices = np.asarray(box_mesh.vertices, dtype=np.float32)
    box_lower = box_vertices.min(axis=0)
    box_upper = box_vertices.max(axis=0)
    box_center = 0.5 * (box_lower + box_upper)
    box_half_extent = 0.5 * (box_upper - box_lower)
    first_box_frame = args.frame_start if args.dynamic_box else 0
    first_box_quaternion = _matrix_to_xyzw(np.asarray(object_motion["obj_rot"][first_box_frame]))
    box_body = builder.add_body(
        xform=wp.transform(object_motion["obj_trans"][first_box_frame], first_box_quaternion),
        is_kinematic=not args.dynamic_box,
        label="carrybox",
    )
    visual_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
    )
    box_mass_kg = 0.3023375868797302
    box_volume_m3 = float(8.0 * np.prod(box_half_extent))
    collision_cfg = newton.ModelBuilder.ShapeConfig(
        density=box_mass_kg / box_volume_m3,
        ke=2.0e4,
        kd=200.0,
        kf=200.0,
        mu=1.0,
        has_shape_collision=True,
        has_particle_collision=False,
        is_visible=False,
    )
    builder.add_shape_mesh(box_body, mesh=box_mesh, cfg=visual_cfg)
    box_collision_shape = builder.add_shape_box(
        box_body,
        xform=wp.transform(box_center, wp.quat_identity()),
        hx=float(box_half_extent[0]),
        hy=float(box_half_extent[1]),
        hz=float(box_half_extent[2]),
        cfg=collision_cfg,
    )
    floor_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
    )
    builder.add_ground_plane(cfg=floor_cfg)
    builder.color()
    model = builder.finalize(device=args.device)

    grid_shape = (20, 25)
    patch_transforms = [wp.transform(center, wp.quat_identity()) for center in hand_centers]
    patch_sizes = [(float(size[0]), float(size[1])) for size in hand_sizes]
    sensor = SensorTactile(
        model,
        sensing_shapes=hand_shapes,
        counterpart_shapes=[box_collision_shape],
        grid_shape=grid_shape,
        patch_size=patch_sizes,
        patch_transform_shape=patch_transforms,
    )
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=2, rigid_body_contact_buffer_size=256)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    adapter = NewtonTactileAdapter(sensor, ("left_rubber_hand", "right_rubber_hand"))
    detector = TactileSlipDetector(adapter.patch_names, friction_coefficient=1.0)
    masks = _hand_masks(model, hand_shapes, hand_centers, hand_sizes, grid_shape)
    renderer = NewtonVTKRenderer(
        model,
        camera_position=(3.1, 3.2, 2.5),
        camera_target=(-0.15, 0.55, 0.65),
    )

    joint_labels = [label.split("/")[-1] for label in model.joint_label]
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    source_joint_to_model = [joint_labels.index(name) for name in G1_JOINT_NAMES]
    root_joint = next(index for index, name in enumerate(joint_labels) if name == "floating_base")
    box_joint = joint_labels.index("carrybox_free_joint")
    q = model.joint_q.numpy().copy()
    qd = model.joint_qd.numpy().copy()
    joint_q = wp.array(q, dtype=float, device=args.device)
    joint_qd = wp.array(qd, dtype=float, device=args.device)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame_dir = output_root / ".frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir()

    force_rows: list[np.ndarray] = []
    penetration_rows: list[np.ndarray] = []
    clock_rows: list[tuple[int, float, float]] = []
    record_rows: list[dict] = []
    max_residual = 0.0
    contact_frames = np.zeros(2, dtype=np.int64)
    max_raw_count = 0
    box_position_rows: list[np.ndarray] = []
    rendered_frames = frame_stop - args.frame_start
    for output_frame, source_frame in enumerate(range(args.frame_start, frame_stop)):
        if (
            args.renderer_refresh_frames > 0
            and output_frame > 0
            and output_frame % args.renderer_refresh_frames == 0
        ):
            renderer.close()
            del renderer
            gc.collect()
            renderer = NewtonVTKRenderer(
                model,
                camera_position=(3.1, 3.2, 2.5),
                camera_target=(-0.15, 0.55, 0.65),
            )
        saved_box_transform = None
        saved_box_velocity = None
        if args.dynamic_box and output_frame > 0:
            saved_box_transform = state_0.body_q.numpy()[box_body].copy()
            saved_box_velocity = state_0.body_qd.numpy()[box_body].copy()
        q.fill(0.0)
        qd.fill(0.0)
        root_q = int(q_start[root_joint])
        root_qd = int(qd_start[root_joint])
        q[root_q : root_q + 3] = robot["body_pos_w"][source_frame, 0]
        root_wxyz = robot["body_quat_w"][source_frame, 0]
        q[root_q + 3 : root_q + 7] = root_wxyz[[1, 2, 3, 0]]
        qd[root_qd : root_qd + 3] = robot["body_lin_vel_w"][source_frame, 0]
        qd[root_qd + 3 : root_qd + 6] = robot["body_ang_vel_w"][source_frame, 0]
        for source_joint, model_joint in enumerate(source_joint_to_model):
            q[int(q_start[model_joint])] = robot["joint_pos"][source_frame, source_joint]
            qd[int(qd_start[model_joint])] = robot["joint_vel"][source_frame, source_joint]
        object_q = int(q_start[box_joint])
        object_qd = int(qd_start[box_joint])
        replay_box_frame = args.frame_start if args.dynamic_box else source_frame
        q[object_q : object_q + 3] = object_motion["obj_trans"][replay_box_frame]
        q[object_q + 3 : object_q + 7] = _matrix_to_xyzw(object_motion["obj_rot"][replay_box_frame])
        qd[object_qd : object_qd + 3] = object_motion["obj_lin_vel"][replay_box_frame]
        qd[object_qd + 3 : object_qd + 6] = object_motion["obj_ang_vel"][replay_box_frame]
        joint_q.assign(q)
        joint_qd.assign(qd)
        newton.eval_fk(model, joint_q, joint_qd, state_0)
        if saved_box_transform is not None:
            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            body_q[box_body] = saved_box_transform
            body_qd[box_body] = saved_box_velocity
            state_0.body_q.assign(body_q)
            state_0.body_qd.assign(body_qd)
        state_0.clear_forces()
        state_1.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 50.0)
        solver.update_contacts(contacts, state_1)
        timestamp_s = (source_frame + 1) / 50.0
        sensor.update(state_1, contacts, timestamp=timestamp_s)
        tactile = adapter.frame()
        evidence = detector.update(tactile)

        force = sensor.force.numpy().reshape(2, *grid_shape, 3).copy()
        penetration = sensor.max_penetration.numpy().reshape(2, *grid_shape).copy()
        dense_sum = force.sum(axis=(1, 2))
        residual = sensor.total_force_patch.numpy() - dense_sum - sensor.unmapped_force_patch.numpy()
        residual_n = float(np.abs(residual).max())
        max_residual = max(max_residual, residual_n)
        contact_frames += (np.linalg.norm(force, axis=-1) > 1.0e-8).any(axis=(1, 2))
        raw_count = int(sensor.raw_count.numpy()[0])
        max_raw_count = max(max_raw_count, raw_count)
        force_rows.append(force)
        penetration_rows.append(penetration)
        clock_rows.append((tactile.clock.sequence, tactile.clock.timestamp_s, tactile.clock.dt_s))
        record_rows.append(
            {
                "source_frame": source_frame,
                "timestamp_s": timestamp_s,
                "raw_sample_count": raw_count,
                "force_conservation_residual_n": residual_n,
                "slip_state": evidence.state[0].astype(int).tolist(),
                "box_position_w_m": state_1.body_q.numpy()[box_body, :3].tolist(),
            }
        )
        box_position_rows.append(state_1.body_q.numpy()[box_body, :3].copy())
        world = renderer.render(state_1)
        frame = _compose(
            world,
            force,
            masks,
            evidence,
            source_frame=source_frame,
            timestamp_s=timestamp_s,
            raw_count=raw_count,
            residual_n=residual_n,
            force_scale_n=args.force_scale_n,
        )
        Image.fromarray(frame).save(frame_dir / f"frame_{output_frame:05d}.png")
        if output_frame % 50 == 0:
            print(
                f"newton_sugar frame={output_frame}/{rendered_frames} source={source_frame} "
                f"raw={raw_count} residual_n={residual_n:.3e}",
                flush=True,
            )
        if args.dynamic_box:
            state_0, state_1 = state_1, state_0

    clocks = np.asarray(clock_rows, dtype=np.float64)
    np.savez_compressed(
        output_root / "trace.npz",
        force_patch_n=np.stack(force_rows),
        penetration_m=np.stack(penetration_rows),
        tactile_sequence=clocks[:, 0].astype(np.int64),
        tactile_timestamp_s=clocks[:, 1],
        tactile_dt_s=clocks[:, 2],
        source_frame=np.arange(args.frame_start, frame_stop, dtype=np.int32),
        box_position_w_m=np.stack(box_position_rows),
        patch_names=np.asarray(adapter.patch_names),
    )
    summary = {
        "schema": "newton_vbd_official_sugar_g1_carrybox_native_tactile_v1",
        "frames": rendered_frames,
        "source_frame_interval": [args.frame_start, frame_stop],
        "fps": 50,
        "robot": str(URDF.relative_to(ROOT)),
        "object_visual": str(BOX_USD.relative_to(ROOT)),
        "object_collision": "bounding box of the exact CarryBox visual mesh",
        "box_mass_kg": box_mass_kg,
        "hand_collision_shapes": hand_shapes,
        "patch_names": list(adapter.patch_names),
        "patch_sizes_m": patch_sizes,
        "grid_shape": list(grid_shape),
        "contact_frames_per_hand": contact_frames.tolist(),
        "maximum_raw_samples_per_frame": max_raw_count,
        "maximum_force_conservation_residual_n": max_residual,
        "native_force_source": "SolverVBD rigid solved penalty+damping+friction force",
        "world_renderer": "VTK offscreen rendering of exact Newton model geometry and live state",
        "kinematic_robot_replay": True,
        "kinematic_box_replay": not args.dynamic_box,
        "maximum_box_displacement_from_initial_m": float(
            np.linalg.norm(np.stack(box_position_rows) - box_position_rows[0], axis=1).max()
        ),
        "optical_available": False,
        "training": False,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "frames.json").write_text(json.dumps(record_rows, indent=2) + "\n", encoding="utf-8")
    video = output_root / "newton_sugar_g1_carrybox_native_tactile.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-framerate",
            "50",
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
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
