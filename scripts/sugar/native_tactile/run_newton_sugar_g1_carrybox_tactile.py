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
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Usd
import trimesh
import warp as wp

import newton
from newton.geometry import HydroelasticSDF
from newton.sensors import SensorTactile

from scripts.sugar.native_tactile.run_newton_softbody_franka_tactile import NewtonVTKRenderer
from scripts.sugar.native_tactile.render_sugar_whole_hand_carrybox import draw_hand, fit_world
from scripts.sugar.native_tactile.slip import SlipState, TactileSlipDetector
from scripts.sugar.native_tactile.universal import NewtonTactileAdapter
from scripts.sugar.smp.sugar_g1_box_schema import G1_JOINT_NAMES


ROOT = Path(os.environ.get("CURIOSITY_ROOT", Path(__file__).resolve().parents[3])).resolve()
URDF = ROOT / "SUGAR/descriptions/robots/g1/g1_29dof_rev_1_0_with_rubber_hand.urdf"
BOX_USD = ROOT / "SUGAR/descriptions/objects/small_box/obj_aligned.usd"
MOTION = ROOT / "SUGAR/data/CarryBox/data_045"
FONT = ImageFont.load_default()
ANATOMICAL_PATCHES_PER_HAND = 27


def _rubber_hand_bodies(builder: newton.ModelBuilder) -> dict[str, int]:
    bodies: dict[str, int] = {}
    for side in ("left", "right"):
        matches = [
            body
            for body, label in enumerate(builder.body_label)
            if label.endswith(f"/{side}_rubber_hand")
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {side} rubber-hand body, got {matches}.")
        bodies[side] = matches[0]
    return bodies


def _add_anatomical_patch_shapes(
    builder: newton.ModelBuilder,
    asset_path: Path,
    *,
    hydroelastic: bool,
    device: str,
    contact_ke: float,
    contact_kd: float,
    contact_mu: float,
) -> tuple[list[int], list[str], list[tuple[float, float]], list[wp.transform]]:
    """Attach the exact IsaacLab-spawned load-bearing meshes to Newton hand bodies."""

    with np.load(asset_path.resolve(), allow_pickle=False) as archive:
        required = (
            "patch_names",
            "sides",
            "patch_size_m",
            "patch_frame_origin_hand_m",
            "patch_frame_rotation_hand",
            "vertices_hand_m",
            "vertex_offsets",
            "triangles",
            "triangle_offsets",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(f"Anatomical patch asset is missing {missing}.")
        patch_names = np.asarray(archive["patch_names"]).astype(str)
        sides = np.asarray(archive["sides"]).astype(str)
        patch_sizes = np.asarray(archive["patch_size_m"], dtype=np.float32)
        frame_origins = np.asarray(archive["patch_frame_origin_hand_m"], dtype=np.float32)
        source_rotations = np.asarray(archive["patch_frame_rotation_hand"], dtype=np.float32)
        vertices = np.asarray(archive["vertices_hand_m"], dtype=np.float32)
        vertex_offsets = np.asarray(archive["vertex_offsets"], dtype=np.int64)
        triangles = np.asarray(archive["triangles"], dtype=np.int32)
        triangle_offsets = np.asarray(archive["triangle_offsets"], dtype=np.int64)

    patch_count = 2 * ANATOMICAL_PATCHES_PER_HAND
    if len(patch_names) != patch_count or tuple(sides) != (
        *("left",) * ANATOMICAL_PATCHES_PER_HAND,
        *("right",) * ANATOMICAL_PATCHES_PER_HAND,
    ):
        raise ValueError("The anatomical asset must contain left 27 then right 27 patches.")
    if patch_sizes.shape != (patch_count, 2):
        raise ValueError(f"Unexpected anatomical patch-size shape {patch_sizes.shape}.")
    if frame_origins.shape != (patch_count, 3) or source_rotations.shape != (patch_count, 3, 3):
        raise ValueError("Unexpected anatomical patch-frame arrays.")
    if len(vertex_offsets) != patch_count + 1 or len(triangle_offsets) != patch_count + 1:
        raise ValueError("Unexpected anatomical mesh offset arrays.")

    hand_bodies = _rubber_hand_bodies(builder)
    patch_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=contact_ke,
        kd=contact_kd,
        kf=300.0,
        kh=1.0e11,
        gap=0.001,
        mu=contact_mu,
        is_hydroelastic=hydroelastic,
        has_shape_collision=True,
        has_particle_collision=False,
        is_visible=False,
    )
    # IsaacLab authors width along source X, length along source Z, and its
    # object-facing surface normal along -source Y. SensorTactile uses local
    # X/Y for the grid and local Z for signed normal force.
    source_to_sensor = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    shapes: list[int] = []
    labels: list[str] = []
    sizes: list[tuple[float, float]] = []
    transforms: list[wp.transform] = []
    for index in range(patch_count):
        vertex_start, vertex_stop = vertex_offsets[index : index + 2]
        triangle_start, triangle_stop = triangle_offsets[index : index + 2]
        patch_vertices = vertices[vertex_start:vertex_stop]
        patch_triangles = triangles[triangle_start:triangle_stop]
        if len(patch_vertices) == 0 or len(patch_triangles) == 0:
            raise ValueError(f"Anatomical patch {index} has empty collision geometry.")
        label = f"{sides[index]}_{patch_names[index]}"
        mesh = newton.Mesh(patch_vertices, patch_triangles.reshape(-1))
        if hydroelastic:
            mesh.build_sdf(
                device=device,
                max_resolution=128,
                narrow_band_range=(-0.01, 0.01),
                margin=patch_cfg.gap,
            )
        shape = builder.add_shape_mesh(
            hand_bodies[sides[index]],
            mesh=mesh,
            cfg=patch_cfg,
            label=f"anatomical_tactile/{label}",
        )
        sensor_rotation = source_rotations[index] @ source_to_sensor
        shapes.append(shape)
        labels.append(label)
        sizes.append((float(patch_sizes[index, 0]), float(patch_sizes[index, 1])))
        transforms.append(
            wp.transform(frame_origins[index], _matrix_to_xyzw(sensor_rotation))
        )
    return shapes, labels, sizes, transforms


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


def _slerp_xyzw(start: np.ndarray, stop: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate unit quaternions along their shortest arc."""

    q0 = np.asarray(start, dtype=np.float64)
    q1 = np.asarray(stop, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + float(alpha) * (q1 - q0)
        return (result / np.linalg.norm(result)).astype(np.float32)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    scale = np.sin(angle)
    result = (
        np.sin((1.0 - float(alpha)) * angle) * q0
        + np.sin(float(alpha) * angle) * q1
    ) / scale
    return result.astype(np.float32)


def _load_box_meshes() -> tuple[newton.Mesh, newton.Mesh, int, float]:
    stage = Usd.Stage.Open(str(BOX_USD))
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() == "Mesh":
            visual = newton.usd.get_mesh(prim)
            topology = trimesh.Trimesh(
                vertices=np.asarray(visual.vertices, dtype=np.float64),
                faces=np.asarray(visual.indices, dtype=np.int64).reshape(-1, 3),
                process=False,
            )
            components = list(topology.split(only_watertight=False))
            positive = [
                component
                for component in components
                if component.is_watertight
                and component.is_winding_consistent
                and float(component.volume) > 0.0
            ]
            if len(positive) != 1:
                raise RuntimeError(
                    "Expected one positive-volume CarryBox exterior component, "
                    f"found {len(positive)} of {len(components)}."
                )
            outer = positive[0]
            outer_bounds = np.asarray(outer.bounds, dtype=np.float64)
            for component in components:
                if component is outer:
                    continue
                component_bounds = np.asarray(component.bounds, dtype=np.float64)
                if np.any(component_bounds[0] < outer_bounds[0]) or np.any(
                    component_bounds[1] > outer_bounds[1]
                ):
                    raise RuntimeError("The selected CarryBox exterior does not enclose every other component.")
            collision = newton.Mesh(
                np.asarray(outer.vertices, dtype=np.float32),
                np.asarray(outer.faces, dtype=np.int32).reshape(-1),
            )
            return visual, collision, len(components), float(outer.volume)
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


def _pack_raw(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    frame_count = len(rows)
    capacity = max((len(row["contact_index"]) for row in rows), default=0)
    packed = {
        "raw_count": np.zeros(frame_count, dtype=np.int32),
        "raw_contact_index": np.full((frame_count, capacity), -1, dtype=np.int32),
        "raw_contact_kind": np.full((frame_count, capacity), -1, dtype=np.int32),
        "raw_patch": np.full((frame_count, capacity), -1, dtype=np.int32),
        "raw_counterpart_shape": np.full((frame_count, capacity), -1, dtype=np.int32),
        "raw_counterpart_particle": np.full((frame_count, capacity), -1, dtype=np.int32),
        "raw_sensor_is_shape0": np.zeros((frame_count, capacity), dtype=bool),
        "raw_point_world_m": np.zeros((frame_count, capacity, 3), dtype=np.float32),
        "raw_point_patch_m": np.zeros((frame_count, capacity, 3), dtype=np.float32),
        "raw_force_world_n": np.zeros((frame_count, capacity, 3), dtype=np.float32),
        "raw_force_patch_n": np.zeros((frame_count, capacity, 3), dtype=np.float32),
        "raw_native_wrench_body0": np.zeros((frame_count, capacity, 6), dtype=np.float32),
        "raw_penetration_m": np.zeros((frame_count, capacity), dtype=np.float32),
    }
    for frame, row in enumerate(rows):
        count = len(row["contact_index"])
        packed["raw_count"][frame] = count
        for key, target in packed.items():
            if key != "raw_count":
                target[frame, :count] = row[key.removeprefix("raw_")]
    return packed


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
    dynamic_box: bool,
    solver_name: str,
) -> np.ndarray:
    canvas = Image.new("RGB", (1280, 720), "white")
    canvas.paste(Image.fromarray(world).resize((1280, 500), Image.Resampling.LANCZOS), (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 9),
        f"Newton {solver_name} | official SUGAR G1 + CarryBox geometry | native solved hand contact force",
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
        "Gray is the exact rubber-hand projection; orange/red is solved force. "
        + (
            "Robot follows the SUGAR motion; the box is dynamically solved."
            if dynamic_box
            else "Robot and box follow the SUGAR reference motion."
        ),
        fill="black",
        font=FONT,
    )
    return np.asarray(canvas)


def _compose_anatomical(
    world: np.ndarray,
    force: np.ndarray,
    penetration: np.ndarray,
    evidence,
    *,
    source_frame: int,
    timestamp_s: float,
    raw_count: int,
    residual_n: float,
    force_scale_n: float,
    solver_name: str,
) -> np.ndarray:
    """Show the real Newton world and all 27 physical patches on both hands."""

    normal = force[..., 2].reshape(2, ANATOMICAL_PATCHES_PER_HAND, 20, 25)
    shear = force[..., :2].reshape(2, ANATOMICAL_PATCHES_PER_HAND, 20, 25, 2)
    depth = penetration.reshape(2, ANATOMICAL_PATCHES_PER_HAND, 20, 25)
    slip_state = np.asarray(evidence.state[0]).reshape(2, ANATOMICAL_PATCHES_PER_HAND)
    canvas = np.full((1440, 2560, 3), 255, dtype=np.uint8)
    canvas[:720] = fit_world(world[..., ::-1], (2560, 720))
    draw_hand(
        canvas,
        0,
        normal[0],
        shear[0],
        depth[0],
        slip_state[0],
        force_scale_n,
        force_scale_n,
        "CENTER r1c1 = OFFICIAL R15 FORCE FOOTPRINT; OPTICAL N/A",
        "R15",
    )
    draw_hand(
        canvas,
        1,
        normal[1],
        shear[1],
        depth[1],
        slip_state[1],
        force_scale_n,
        force_scale_n,
        "CENTER r1c1 = OFFICIAL R15 FORCE FOOTPRINT; OPTICAL N/A",
        "R15",
    )
    cv2.rectangle(canvas, (0, 0), (2559, 66), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "Newton native tactile | exact IsaacLab anatomical 27-patch hands | dynamic CarryBox",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"{solver_name} | source {source_frame:03d} | t={timestamp_s:.2f}s | "
            f"raw={raw_count} | force residual={residual_n:.2e} N | Newton optical unavailable"
        ),
        (18, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return canvas[..., ::-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=660)
    parser.add_argument("--force-scale-n", type=float, default=25.0)
    parser.add_argument("--vbd-contact-ke", type=float, default=1.5)
    parser.add_argument("--vbd-contact-kd", type=float, default=0.0)
    parser.add_argument("--contact-friction", type=float, default=1.0)
    parser.add_argument("--renderer-refresh-frames", type=int, default=40)
    parser.add_argument("--dynamic-box", action="store_true")
    parser.add_argument(
        "--robot-collisions",
        choices=("auto", "official", "sensor-only"),
        default="auto",
    )
    parser.add_argument("--solver", choices=("vbd", "mujoco"), default="vbd")
    parser.add_argument("--physics-substeps", type=int, default=0)
    parser.add_argument("--solver-iterations", type=int, default=0)
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--render-frame-start",
        type=int,
        default=None,
        help="Simulate from frame-start but begin video rendering at this source frame.",
    )
    parser.add_argument(
        "--robot-state-trace",
        type=Path,
        help="IsaacLab successful CarryBox trace containing actual robot root/joint states.",
    )
    parser.add_argument(
        "--anatomical-patch-asset",
        type=Path,
        help="NPZ exported from the actual IsaacLab 54-patch collision geometry.",
    )
    parser.add_argument(
        "--box-collision",
        choices=("auto", "outer-sdf", "bounding-box"),
        default="auto",
    )
    args = parser.parse_args()
    robot_collision_mode = (
        "official" if args.dynamic_box and args.robot_collisions == "auto" else args.robot_collisions
    )
    if robot_collision_mode == "auto":
        robot_collision_mode = "sensor-only"
    physics_substeps = args.physics_substeps or (
        6 if args.dynamic_box and args.solver == "mujoco" else 4 if args.dynamic_box else 1
    )
    solver_iterations = args.solver_iterations or (
        100 if args.solver == "mujoco" else 8 if args.dynamic_box else 2
    )
    if physics_substeps < 1 or solver_iterations < 1 or args.render_stride < 1:
        raise ValueError("physics-substeps, solver-iterations, and render-stride must be positive")
    if args.vbd_contact_ke <= 0.0 or args.vbd_contact_kd < 0.0 or args.contact_friction < 0.0:
        raise ValueError("Contact stiffness must be positive; damping and friction must be nonnegative")

    with np.load(MOTION / "robot_50hz.npz") as archive:
        robot = {key: np.asarray(archive[key]) for key in archive.files}
    with (MOTION / "obj_motion_global_50hz.pkl").open("rb") as stream:
        object_motion = pickle.load(stream)
    state_bridge = None
    if args.robot_state_trace is not None:
        with np.load(args.robot_state_trace.resolve(), allow_pickle=False) as archive:
            required = (
                "robot_root_state_w",
                "robot_root_velocity_w",
                "robot_joint_position",
                "robot_joint_velocity",
                "robot_joint_names",
                "object_state_w",
                "object_velocity_w",
            )
            missing = [key for key in required if key not in archive.files]
            if missing:
                raise ValueError(f"Robot-state trace is missing {missing}.")
            state_bridge = {key: np.asarray(archive[key]).copy() for key in required}
        bridge_joint_names = tuple(str(name) for name in state_bridge["robot_joint_names"])
        bridge_joint_indices = [bridge_joint_names.index(name) for name in G1_JOINT_NAMES]
        state_bridge["robot_joint_position"] = state_bridge["robot_joint_position"][:, bridge_joint_indices]
        state_bridge["robot_joint_velocity"] = state_bridge["robot_joint_velocity"][:, bridge_joint_indices]
    source_length = len(state_bridge["robot_joint_position"]) if state_bridge is not None else len(robot["joint_pos"])
    frame_stop = min(args.frame_stop, source_length)
    if not (0 <= args.frame_start < frame_stop):
        raise ValueError("The selected source-frame interval is empty.")
    render_frame_start = args.frame_start if args.render_frame_start is None else args.render_frame_start
    if not (args.frame_start <= render_frame_start < frame_stop):
        raise ValueError("render-frame-start must lie inside the simulation interval.")

    builder = newton.ModelBuilder(gravity=-9.81 if args.dynamic_box else 0.0)
    if args.solver == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
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
    original_hand_shapes, hand_centers, hand_sizes = _hand_collision_shapes(builder)
    anatomical = args.anatomical_patch_asset is not None
    hydroelastic = args.solver == "mujoco" and anatomical
    contact_ke = 1.0e11 if hydroelastic else args.vbd_contact_ke if anatomical else 1500.0
    contact_kd = 0.0 if hydroelastic else args.vbd_contact_kd if anatomical else 300.0
    if anatomical:
        hand_shapes, patch_names, patch_sizes, patch_transforms = _add_anatomical_patch_shapes(
            builder,
            args.anatomical_patch_asset,
            hydroelastic=hydroelastic,
            device=args.device,
            contact_ke=contact_ke,
            contact_kd=contact_kd,
            contact_mu=args.contact_friction,
        )
    else:
        hand_shapes = original_hand_shapes
        patch_names = ["left_rubber_hand", "right_rubber_hand"]
        patch_sizes = [(float(size[0]), float(size[1])) for size in hand_sizes]
        patch_transforms = [
            wp.transform(center, wp.quat_identity()) for center in hand_centers
        ]
    if robot_collision_mode == "sensor-only":
        for shape in range(builder.shape_count):
            visible = int(builder.shape_flags[shape]) & int(newton.ShapeFlags.VISIBLE)
            builder.shape_flags[shape] = visible
        for shape in hand_shapes:
            builder.shape_flags[shape] = int(newton.ShapeFlags.COLLIDE_SHAPES) | (
                int(newton.ShapeFlags.HYDROELASTIC) if hydroelastic else 0
            )
    elif anatomical:
        # The IsaacLab spawner makes the 54 tactile patches the only exterior
        # hand collision owners while retaining the rest of the robot.
        for shape in original_hand_shapes:
            visible = int(builder.shape_flags[shape]) & int(newton.ShapeFlags.VISIBLE)
            builder.shape_flags[shape] = visible

    box_collision_kind = (
        "outer-sdf" if args.dynamic_box and args.box_collision == "auto" else args.box_collision
    )
    if box_collision_kind == "auto":
        box_collision_kind = "bounding-box"
    box_mesh, box_outer_mesh, box_component_count, box_outer_volume_m3 = _load_box_meshes()
    box_vertices = np.asarray(box_mesh.vertices, dtype=np.float32)
    box_lower = box_vertices.min(axis=0)
    box_upper = box_vertices.max(axis=0)
    box_center = 0.5 * (box_lower + box_upper)
    box_half_extent = 0.5 * (box_upper - box_lower)
    first_box_frame = args.frame_start if args.dynamic_box else 0
    if state_bridge is None:
        first_box_position = object_motion["obj_trans"][first_box_frame]
        first_box_quaternion = _matrix_to_xyzw(np.asarray(object_motion["obj_rot"][first_box_frame]))
    else:
        first_box_state = state_bridge["object_state_w"][first_box_frame]
        first_box_position = first_box_state[:3]
        first_box_quaternion = first_box_state[3:7][[1, 2, 3, 0]]
    box_body = builder.add_body(
        xform=wp.transform(first_box_position, first_box_quaternion),
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
    box_volume_m3 = (
        box_outer_volume_m3
        if box_collision_kind == "outer-sdf"
        else float(8.0 * np.prod(box_half_extent))
    )
    collision_cfg = newton.ModelBuilder.ShapeConfig(
        density=box_mass_kg / box_volume_m3,
        ke=contact_ke if anatomical else 2.0e4,
        kd=contact_kd if anatomical else 200.0,
        kf=200.0,
        kh=1.0e11,
        gap=0.001,
        mu=args.contact_friction if anatomical else 1.0,
        is_hydroelastic=hydroelastic,
        has_shape_collision=True,
        has_particle_collision=False,
        is_visible=False,
    )
    builder.add_shape_mesh(box_body, mesh=box_mesh, cfg=visual_cfg)
    if box_collision_kind == "outer-sdf":
        box_outer_mesh.build_sdf(
            device=args.device,
            max_resolution=128,
            narrow_band_range=(-0.01, 0.01),
            margin=collision_cfg.gap if hydroelastic else 0.01,
        )
        box_collision_shape = builder.add_shape_mesh(
            box_body,
            mesh=box_outer_mesh,
            cfg=collision_cfg,
        )
    else:
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
        has_shape_collision=args.dynamic_box,
        has_particle_collision=False,
        is_visible=True,
    )
    builder.add_ground_plane(cfg=floor_cfg)
    builder.color()
    model = builder.finalize(device=args.device)

    grid_shape = (20, 25)
    sensor = SensorTactile(
        model,
        sensing_shapes=hand_shapes,
        counterpart_shapes=[box_collision_shape],
        grid_shape=grid_shape,
        patch_size=patch_sizes,
        patch_transform_shape=patch_transforms,
    )
    collision_pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=8192 if anatomical else None,
        broad_phase="explicit" if hydroelastic else None,
        sdf_hydroelastic_config=(
            HydroelasticSDF.Config(output_contact_surface=False) if hydroelastic else None
        ),
    )
    contacts = collision_pipeline.contacts()
    if args.solver == "mujoco":
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            njmax=3000,
            nconmax=1000,
            cone="elliptic",
            impratio=100,
            iterations=solver_iterations,
            ls_iterations=50,
            use_mujoco_contacts=False,
        )
    else:
        solver = newton.solvers.SolverVBD(
            model,
            iterations=solver_iterations,
            rigid_contact_hard=not anatomical,
            rigid_body_contact_buffer_size=(
                8192 if anatomical else 2048 if box_collision_kind == "outer-sdf" else 256
            ),
        )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    adapter = NewtonTactileAdapter(sensor, patch_names)
    detector = TactileSlipDetector(
        adapter.patch_names,
        friction_coefficient=args.contact_friction if anatomical else 1.0,
    )
    masks = None if anatomical else _hand_masks(
        model,
        hand_shapes,
        hand_centers,
        hand_sizes,
        grid_shape,
    )
    renderer = None

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
    active_rows: list[np.ndarray] = []
    taxel_position_rows: list[np.ndarray] = []
    taxel_orientation_rows: list[np.ndarray] = []
    raw_rows: list[dict[str, np.ndarray]] = []
    clock_rows: list[tuple[int, float, float]] = []
    record_rows: list[dict] = []
    max_residual = 0.0
    contact_frames = np.zeros(len(hand_shapes), dtype=np.int64)
    hand_contact_frames = np.zeros(2, dtype=np.int64)
    max_raw_count = 0
    box_position_rows: list[np.ndarray] = []
    box_orientation_rows: list[np.ndarray] = []
    box_velocity_rows: list[np.ndarray] = []
    simulation_frames = frame_stop - args.frame_start
    rendered_frames = 0
    for output_frame, source_frame in enumerate(range(args.frame_start, frame_stop)):
        if (
            renderer is not None
            and args.renderer_refresh_frames > 0
            and rendered_frames > 0
            and rendered_frames % args.renderer_refresh_frames == 0
            and source_frame >= render_frame_start
            and (source_frame - render_frame_start) % args.render_stride == 0
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
        if state_bridge is None:
            root_state = np.concatenate(
                (
                    robot["body_pos_w"][source_frame, 0],
                    robot["body_quat_w"][source_frame, 0],
                )
            )
            root_velocity = np.concatenate(
                (
                    robot["body_lin_vel_w"][source_frame, 0],
                    robot["body_ang_vel_w"][source_frame, 0],
                )
            )
            source_joint_position = robot["joint_pos"][source_frame]
            source_joint_velocity = robot["joint_vel"][source_frame]
        else:
            root_state = state_bridge["robot_root_state_w"][source_frame]
            root_velocity = state_bridge["robot_root_velocity_w"][source_frame]
            source_joint_position = state_bridge["robot_joint_position"][source_frame]
            source_joint_velocity = state_bridge["robot_joint_velocity"][source_frame]
        q[root_q : root_q + 3] = root_state[:3]
        root_wxyz = root_state[3:7]
        q[root_q + 3 : root_q + 7] = root_wxyz[[1, 2, 3, 0]]
        qd[root_qd : root_qd + 3] = root_velocity[:3]
        qd[root_qd + 3 : root_qd + 6] = root_velocity[3:6]
        for source_joint, model_joint in enumerate(source_joint_to_model):
            q[int(q_start[model_joint])] = source_joint_position[source_joint]
            qd[int(qd_start[model_joint])] = source_joint_velocity[source_joint]
        object_q = int(q_start[box_joint])
        object_qd = int(qd_start[box_joint])
        replay_box_frame = args.frame_start if args.dynamic_box else source_frame
        if state_bridge is None:
            q[object_q : object_q + 3] = object_motion["obj_trans"][replay_box_frame]
            q[object_q + 3 : object_q + 7] = _matrix_to_xyzw(object_motion["obj_rot"][replay_box_frame])
            qd[object_qd : object_qd + 3] = object_motion["obj_lin_vel"][replay_box_frame]
            qd[object_qd + 3 : object_qd + 6] = object_motion["obj_ang_vel"][replay_box_frame]
        else:
            bridge_object_state = state_bridge["object_state_w"][replay_box_frame]
            bridge_object_velocity = state_bridge["object_velocity_w"][replay_box_frame]
            q[object_q : object_q + 3] = bridge_object_state[:3]
            q[object_q + 3 : object_q + 7] = bridge_object_state[3:7][[1, 2, 3, 0]]
            qd[object_qd : object_qd + 3] = bridge_object_velocity[:3]
            qd[object_qd + 3 : object_qd + 6] = bridge_object_velocity[3:6]
        target_q = q.copy()
        target_qd = qd.copy()
        previous_q = None
        previous_qd = None
        if output_frame > 0 and physics_substeps > 1:
            previous_source_frame = source_frame - 1
            if state_bridge is None:
                previous_root_state = np.concatenate(
                    (
                        robot["body_pos_w"][previous_source_frame, 0],
                        robot["body_quat_w"][previous_source_frame, 0],
                    )
                )
                previous_root_velocity = np.concatenate(
                    (
                        robot["body_lin_vel_w"][previous_source_frame, 0],
                        robot["body_ang_vel_w"][previous_source_frame, 0],
                    )
                )
                previous_joint_position = robot["joint_pos"][previous_source_frame]
                previous_joint_velocity = robot["joint_vel"][previous_source_frame]
            else:
                previous_root_state = state_bridge["robot_root_state_w"][previous_source_frame]
                previous_root_velocity = state_bridge["robot_root_velocity_w"][previous_source_frame]
                previous_joint_position = state_bridge["robot_joint_position"][previous_source_frame]
                previous_joint_velocity = state_bridge["robot_joint_velocity"][previous_source_frame]
            previous_q = target_q.copy()
            previous_qd = target_qd.copy()
            previous_q[root_q : root_q + 3] = previous_root_state[:3]
            previous_q[root_q + 3 : root_q + 7] = previous_root_state[3:7][[1, 2, 3, 0]]
            previous_qd[root_qd : root_qd + 3] = previous_root_velocity[:3]
            previous_qd[root_qd + 3 : root_qd + 6] = previous_root_velocity[3:6]
            for source_joint, model_joint in enumerate(source_joint_to_model):
                previous_q[int(q_start[model_joint])] = previous_joint_position[source_joint]
                previous_qd[int(qd_start[model_joint])] = previous_joint_velocity[source_joint]

        # The first output frame establishes the replay pose without presenting
        # it to VBD as a one-step kinematic teleport. Later multi-substep frames
        # are advanced below from the previous recorded pose to this target.
        if output_frame == 0 or physics_substeps == 1:
            joint_q.assign(target_q)
            joint_qd.assign(target_qd)
            newton.eval_fk(model, joint_q, joint_qd, state_0)
        if saved_box_transform is not None and physics_substeps == 1:
            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            body_q[box_body] = saved_box_transform
            body_qd[box_body] = saved_box_velocity
            state_0.body_q.assign(body_q)
            state_0.body_qd.assign(body_qd)
        if output_frame == 0 and args.solver == "vbd":
            solver.body_q_prev.assign(state_0.body_q)
        for substep in range(physics_substeps):
            if previous_q is not None:
                alpha = float(substep + 1) / float(physics_substeps)
                substep_q = previous_q + alpha * (target_q - previous_q)
                substep_q[root_q + 3 : root_q + 7] = _slerp_xyzw(
                    previous_q[root_q + 3 : root_q + 7],
                    target_q[root_q + 3 : root_q + 7],
                    alpha,
                )
                substep_qd = previous_qd + alpha * (target_qd - previous_qd)
                box_transform = state_0.body_q.numpy()[box_body].copy()
                box_velocity = state_0.body_qd.numpy()[box_body].copy()
                joint_q.assign(substep_q)
                joint_qd.assign(substep_qd)
                newton.eval_fk(model, joint_q, joint_qd, state_0)
                body_q = state_0.body_q.numpy()
                body_qd = state_0.body_qd.numpy()
                body_q[box_body] = box_transform
                body_qd[box_body] = box_velocity
                state_0.body_q.assign(body_q)
                state_0.body_qd.assign(body_qd)
            state_0.clear_forces()
            state_1.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / (50.0 * physics_substeps))
            # MuJoCo/VBD advance the free box, but all G1 bodies are prescribed
            # by the recorded articulation state. Preserve those exact body
            # transforms in the output state so SensorTactile computes its
            # geometry-fixed patch frames in world coordinates.
            wp.copy(state_1.body_q, state_0.body_q, count=robot_body_count)
            wp.copy(state_1.body_qd, state_0.body_qd, count=robot_body_count)
            if substep == physics_substeps - 1:
                solver.update_contacts(contacts, state_1)
            state_0, state_1 = state_1, state_0
        timestamp_s = (source_frame + 1) / 50.0
        sensor.update(state_0, contacts, timestamp=timestamp_s)
        tactile = adapter.frame()
        evidence = detector.update(tactile)

        force = sensor.force.numpy().reshape(len(hand_shapes), *grid_shape, 3).copy()
        penetration = sensor.max_penetration.numpy().reshape(len(hand_shapes), *grid_shape).copy()
        dense_sum = force.sum(axis=(1, 2))
        residual = sensor.total_force_patch.numpy() - dense_sum
        residual_n = float(np.abs(residual).max())
        max_residual = max(max_residual, residual_n)
        patch_active = (np.linalg.norm(force, axis=-1) > 1.0e-8).any(axis=(1, 2))
        contact_frames += patch_active
        if anatomical:
            hand_contact_frames += patch_active.reshape(2, ANATOMICAL_PATCHES_PER_HAND).any(axis=1)
        else:
            hand_contact_frames += patch_active
        raw_count = int(sensor.raw_count.numpy()[0])
        max_raw_count = max(max_raw_count, raw_count)
        force_rows.append(force)
        penetration_rows.append(penetration)
        active_rows.append(np.asarray(tactile.active[0]).copy())
        taxel_position_rows.append(np.asarray(tactile.taxel_position_w_m[0]).copy())
        taxel_orientation_rows.append(np.asarray(tactile.taxel_orientation_w_xyzw[0]).copy())
        raw = tactile.raw_samples
        if raw is None:
            raise RuntimeError("Newton universal frame did not preserve native raw samples.")
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
        clock_rows.append((tactile.clock.sequence, tactile.clock.timestamp_s, tactile.clock.dt_s))
        record_rows.append(
            {
                "source_frame": source_frame,
                "timestamp_s": timestamp_s,
                "raw_sample_count": raw_count,
                "force_conservation_residual_n": residual_n,
                "slip_state": evidence.state[0].astype(int).tolist(),
                "box_position_w_m": state_0.body_q.numpy()[box_body, :3].tolist(),
            }
        )
        box_position_rows.append(state_0.body_q.numpy()[box_body, :3].copy())
        box_orientation_rows.append(state_0.body_q.numpy()[box_body, 3:7].copy())
        box_velocity_rows.append(state_0.body_qd.numpy()[box_body].copy())
        if (
            not args.no_render
            and source_frame >= render_frame_start
            and (source_frame - render_frame_start) % args.render_stride == 0
        ):
            if renderer is None:
                renderer = NewtonVTKRenderer(
                    model,
                    camera_position=(3.1, 3.2, 2.5),
                    camera_target=(-0.15, 0.55, 0.65),
                )
            world = renderer.render(state_0)
            if anatomical:
                frame = _compose_anatomical(
                    world,
                    force,
                    penetration,
                    evidence,
                    source_frame=source_frame,
                    timestamp_s=timestamp_s,
                    raw_count=raw_count,
                    residual_n=residual_n,
                    force_scale_n=args.force_scale_n,
                    solver_name=args.solver,
                )
            else:
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
                    dynamic_box=args.dynamic_box,
                    solver_name=args.solver,
                )
            Image.fromarray(frame).save(frame_dir / f"frame_{rendered_frames:05d}.png")
            rendered_frames += 1
        if output_frame % 50 == 0:
            print(
                f"newton_sugar frame={output_frame}/{simulation_frames} source={source_frame} "
                f"raw={raw_count} residual_n={residual_n:.3e}",
                flush=True,
            )

    if renderer is not None:
        renderer.close()
        del renderer
        gc.collect()
    clocks = np.asarray(clock_rows, dtype=np.float64)
    np.savez_compressed(
        output_root / "trace.npz",
        force_patch_n=np.stack(force_rows),
        penetration_m=np.stack(penetration_rows),
        active=np.stack(active_rows),
        taxel_position_w_m=np.stack(taxel_position_rows),
        taxel_orientation_w_xyzw=np.stack(taxel_orientation_rows),
        tactile_sequence=clocks[:, 0].astype(np.int64),
        tactile_timestamp_s=clocks[:, 1],
        tactile_dt_s=clocks[:, 2],
        source_frame=np.arange(args.frame_start, frame_stop, dtype=np.int32),
        box_position_w_m=np.stack(box_position_rows),
        box_orientation_w_xyzw=np.stack(box_orientation_rows),
        box_velocity_w=np.stack(box_velocity_rows),
        patch_names=np.asarray(adapter.patch_names),
        patch_size_m=tactile.patch_size_m.copy(),
        backend=np.asarray(tactile.backend),
        optical_available=np.asarray(tactile.optical.available, dtype=bool),
        **_pack_raw(raw_rows),
    )
    box_positions = np.stack(box_position_rows)
    summary = {
        "schema": "newton_official_sugar_g1_carrybox_native_tactile_v2",
        "frames": simulation_frames,
        "video_frames": rendered_frames,
        "source_frame_interval": [args.frame_start, frame_stop],
        "fps": 50,
        "video_fps": 50.0 / args.render_stride,
        "render_stride": args.render_stride,
        "render_frame_start": render_frame_start,
        "robot": str(URDF.relative_to(ROOT)),
        "robot_motion_source": (
            str(args.robot_state_trace.resolve())
            if args.robot_state_trace is not None
            else str((MOTION / "robot_50hz.npz").relative_to(ROOT))
        ),
        "object_visual": str(BOX_USD.relative_to(ROOT)),
        "object_collision": (
            "official positive-volume exterior component, Newton 128-resolution SDF"
            if box_collision_kind == "outer-sdf"
            else "bounding box of the exact CarryBox visual mesh"
        ),
        "object_source_component_count": box_component_count,
        "box_mass_kg": box_mass_kg,
        "box_collision_volume_m3": box_volume_m3,
        "hand_collision_source": (
            str(args.anatomical_patch_asset.resolve())
            if anatomical
            else "original URDF rubber-hand collision meshes"
        ),
        "original_rubber_hand_collision_shapes": original_hand_shapes,
        "hand_collision_shapes": hand_shapes,
        "patch_names": list(adapter.patch_names),
        "patch_sizes_m": patch_sizes,
        "grid_shape": list(grid_shape),
        "taxel_position_shape": [simulation_frames, len(adapter.patch_names), *grid_shape, 3],
        "taxel_orientation_shape": [simulation_frames, len(adapter.patch_names), *grid_shape, 4],
        "taxel_quaternion_order": "xyzw",
        "tactile_clock_fields": ["tactile_sequence", "tactile_timestamp_s", "tactile_dt_s"],
        "raw_sample_fields": [
            "raw_contact_index",
            "raw_contact_kind",
            "raw_patch",
            "raw_counterpart_shape",
            "raw_counterpart_particle",
            "raw_sensor_is_shape0",
            "raw_point_world_m",
            "raw_point_patch_m",
            "raw_force_world_n",
            "raw_force_patch_n",
            "raw_native_wrench_body0",
            "raw_penetration_m",
        ],
        "contact_frames_per_hand": hand_contact_frames.tolist(),
        "contact_frames_per_patch": contact_frames.tolist(),
        "maximum_raw_samples_per_frame": max_raw_count,
        "maximum_force_conservation_residual_n": max_residual,
        "native_force_source": (
            "SolverMuJoCo constraint force exported by update_contacts"
            if args.solver == "mujoco"
            else "SolverVBD rigid solved penalty+damping+friction force"
        ),
        "solver": args.solver,
        "world_renderer": (
            "disabled for continuous physics trace"
            if args.no_render
            else "VTK offscreen rendering of exact Newton model geometry and live state"
        ),
        "kinematic_robot_replay": True,
        "robot_collision_mode": robot_collision_mode,
        "anatomical_patches_per_hand": ANATOMICAL_PATCHES_PER_HAND if anatomical else 1,
        "kinematic_box_replay": not args.dynamic_box,
        "physics_substeps": physics_substeps,
        "solver_iterations": solver_iterations,
        "vbd_contact_ke_n_per_m": args.vbd_contact_ke if args.solver == "vbd" else None,
        "vbd_contact_kd_n_s_per_m": args.vbd_contact_kd if args.solver == "vbd" else None,
        "vbd_contact_mode": "soft penalty" if args.solver == "vbd" and anatomical else None,
        "contact_friction_coefficient": args.contact_friction if anatomical else 1.0,
        "ground_collision_enabled": bool(args.dynamic_box),
        "maximum_box_displacement_from_initial_m": float(
            np.linalg.norm(box_positions - box_positions[0], axis=1).max()
        ),
        "maximum_box_lift_from_initial_m": float(
            np.max(box_positions[:, 2] - box_positions[0, 2])
        ),
        "optical_available": False,
        "training": False,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "frames.json").write_text(json.dumps(record_rows, indent=2) + "\n", encoding="utf-8")
    if rendered_frames:
        video = output_root / "newton_sugar_g1_carrybox_native_tactile.mp4"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-framerate",
                str(50.0 / args.render_stride),
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
