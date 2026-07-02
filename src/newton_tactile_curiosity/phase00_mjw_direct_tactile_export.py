#!/usr/bin/env python3
"""Candidate MJWarp direct-force tactile export for official Panda hydro.

This diagnostic maps MJWarp EFC contact forces into left/right pad tactile
planes. It is explicitly a candidate direct-force bridge, not a final tactile
sensor claim, because the official hydro ``SensorContact/update_contacts`` path
still needs a compatible validation route.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp
from PIL import Image, ImageDraw

from newton_tactile_curiosity.phase00_mjw_force_audit import force_table_for_world, to_numpy
from newton_tactile_curiosity.phase00_sync_hydro_diagnostic import (
    SurfaceNullViewer,
    body_label,
    classify_shape,
    compose_scene_camera_triptych,
    contact_view_window,
    quat_to_matrix_xyzw,
    scene_camera_transforms,
    sparkline,
    tactile_panel,
    world_to_body,
    write_mp4_video,
)


def world_vector_to_body(vectors: np.ndarray, body_q_row: np.ndarray) -> np.ndarray:
    r = quat_to_matrix_xyzw(np.asarray(body_q_row[3:7], dtype=np.float32))
    return np.asarray(vectors, dtype=np.float32) @ r


def frame_matrix(frame_row: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame_row)
    if arr.shape == (3, 3):
        return arr.astype(np.float32)
    flat = arr.reshape(-1)
    if flat.size >= 9:
        return flat[:9].reshape(3, 3).astype(np.float32)
    return np.zeros((3, 3), dtype=np.float32)


def geom_pair_to_shapes(geom_pair: np.ndarray, world: int, geom_to_shape: np.ndarray) -> tuple[int, int]:
    geom = np.asarray(geom_pair).reshape(-1)
    g0 = int(geom[0]) if geom.size > 0 else -1
    g1 = int(geom[1]) if geom.size > 1 else -1
    if geom_to_shape.ndim >= 2 and 0 <= world < geom_to_shape.shape[0]:
        s0 = int(geom_to_shape[world, g0]) if 0 <= g0 < geom_to_shape.shape[1] else -1
        s1 = int(geom_to_shape[world, g1]) if 0 <= g1 < geom_to_shape.shape[1] else -1
    elif geom_to_shape.ndim == 1:
        s0 = int(geom_to_shape[g0]) if 0 <= g0 < geom_to_shape.shape[0] else -1
        s1 = int(geom_to_shape[g1]) if 0 <= g1 < geom_to_shape.shape[0] else -1
    else:
        s0 = -1
        s1 = -1
    return s0, s1


def shape_class(shape_classes: list[str], shape: int) -> str:
    return shape_classes[shape] if 0 <= shape < len(shape_classes) else "invalid"


def pad_side_for_pair(shape_classes: list[str], shape0: int, shape1: int) -> tuple[str, bool] | None:
    c0 = shape_class(shape_classes, shape0)
    c1 = shape_class(shape_classes, shape1)
    objects = {"object", "cup"}
    if c0 == "left_pad_or_finger" and c1 in objects:
        return "left", True
    if c1 == "left_pad_or_finger" and c0 in objects:
        return "left", False
    if c0 == "right_pad_or_finger" and c1 in objects:
        return "right", True
    if c1 == "right_pad_or_finger" and c0 in objects:
        return "right", False
    return None


def accumulate_gaussian_per_point(
    scalar_map: np.ndarray,
    y_map: np.ndarray,
    z_map: np.ndarray,
    local_points: np.ndarray,
    scalar_weights: np.ndarray,
    vector_yz: np.ndarray,
    extent: tuple[float, float],
    center_yz: np.ndarray,
    sigma_cells: float = 1.35,
) -> None:
    if local_points.size == 0:
        return
    size = scalar_map.shape[0]
    y = local_points[:, 1] - float(center_yz[0])
    z = local_points[:, 2] - float(center_yz[1])
    hy, hz = extent[0] / 2.0, extent[1] / 2.0
    fy = (np.clip(y, -hy, hy) + hy) / (2.0 * hy + 1.0e-9) * (size - 1)
    fz = (np.clip(z, -hz, hz) + hz) / (2.0 * hz + 1.0e-9) * (size - 1)
    radius = max(1, int(math.ceil(3.0 * sigma_cells)))
    denom = 2.0 * sigma_cells * sigma_cells
    for cy, cz, weight, vec in zip(fy, fz, scalar_weights, vector_yz, strict=False):
        w = float(weight)
        vy, vz = float(vec[0]), float(vec[1])
        if not (math.isfinite(w) and math.isfinite(vy) and math.isfinite(vz)):
            continue
        if w <= 0.0 and abs(vy) <= 0.0 and abs(vz) <= 0.0:
            continue
        c_col = int(round(float(cy)))
        c_row = int(round(float(cz)))
        for row in range(max(0, c_row - radius), min(size, c_row + radius + 1)):
            dz = row - float(cz)
            for col in range(max(0, c_col - radius), min(size, c_col + radius + 1)):
                dy = col - float(cy)
                g = math.exp(-(dy * dy + dz * dz) / denom)
                scalar_map[row, col] += max(w, 0.0) * g
                y_map[row, col] += vy * g
                z_map[row, col] += vz * g


def marker_flow_from_fields(
    shear_y: np.ndarray,
    shear_z: np.ndarray,
    normal_y: np.ndarray,
    normal_z: np.ndarray,
    area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shear_norm = np.sqrt(shear_y * shear_y + shear_z * shear_z)
    normal_norm = np.sqrt(normal_y * normal_y + normal_z * normal_z)
    shear_scale = float(np.quantile(shear_norm[shear_norm > 0.0], 0.95)) if np.any(shear_norm > 0.0) else 1.0
    normal_scale = float(np.quantile(normal_norm[normal_norm > 0.0], 0.95)) if np.any(normal_norm > 0.0) else 1.0
    area_scale = float(np.quantile(area[area > 0.0], 0.95)) if np.any(area > 0.0) else 1.0
    area_gate = np.clip(area / max(area_scale, 1.0e-12), 0.0, 1.0)
    marker_y = 0.65 * shear_y / max(shear_scale, 1.0e-12) + 0.25 * area_gate * normal_y / max(normal_scale, 1.0e-12)
    marker_z = 0.65 * shear_z / max(shear_scale, 1.0e-12) + 0.25 * area_gate * normal_z / max(normal_scale, 1.0e-12)
    return marker_y.astype(np.float32), marker_z.astype(np.float32)


def center_of_pressure_proxy_from_map(
    force_map: np.ndarray,
    extent: tuple[float, float],
    center_yz: np.ndarray,
) -> np.ndarray:
    """Return candidate pad-local Y/Z center from force-map weights per frame."""
    maps = np.asarray(force_map, dtype=np.float32)
    frames, rows, cols = maps.shape
    y_axis = np.linspace(
        float(center_yz[0]) - float(extent[0]) / 2.0,
        float(center_yz[0]) + float(extent[0]) / 2.0,
        cols,
        dtype=np.float32,
    )
    z_axis = np.linspace(
        float(center_yz[1]) - float(extent[1]) / 2.0,
        float(center_yz[1]) + float(extent[1]) / 2.0,
        rows,
        dtype=np.float32,
    )
    yy = np.broadcast_to(y_axis[None, :], (rows, cols))
    zz = np.broadcast_to(z_axis[:, None], (rows, cols))
    cop = np.full((frames, 2), np.nan, dtype=np.float32)
    for frame in range(frames):
        weights = np.maximum(maps[frame], 0.0)
        total = float(weights.sum())
        if total <= 1.0e-12:
            continue
        cop[frame, 0] = float((weights * yy).sum() / total)
        cop[frame, 1] = float((weights * zz).sum() / total)
    return cop


def candidate_gel_marker_panel(
    pressure: np.ndarray,
    area: np.ndarray,
    flow_y: np.ndarray,
    flow_z: np.ndarray,
    pressure_vmax: float,
    area_vmax: float,
    scale: int,
) -> Image.Image:
    pressure_norm = np.clip(np.asarray(pressure, dtype=np.float32) / max(pressure_vmax, 1.0e-12), 0.0, 1.0)
    area_norm = np.clip(np.asarray(area, dtype=np.float32) / max(area_vmax, 1.0e-12), 0.0, 1.0)
    p = np.power(np.maximum(pressure_norm, 0.7 * area_norm), 0.35)
    rgb = np.zeros((*p.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (30 + 60 * p).astype(np.uint8)
    rgb[..., 1] = (70 + 90 * p).astype(np.uint8)
    rgb[..., 2] = (130 + 110 * p).astype(np.uint8)
    img = Image.fromarray(np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1))
    draw = ImageDraw.Draw(img)
    size = pressure.shape[0]
    step = max(4, size // 7)
    flow_mag = np.sqrt(flow_y * flow_y + flow_z * flow_z)
    flow_scale = float(np.quantile(flow_mag[flow_mag > 0.0], 0.90)) if np.any(flow_mag > 0.0) else 1.0
    for row in range(step // 2, size, step):
        for col in range(step // 2, size, step):
            x0 = col * scale + scale // 2
            y0 = row * scale + scale // 2
            fy = float(flow_y[row, col])
            fz = float(flow_z[row, col])
            mag = math.sqrt(fy * fy + fz * fz)
            dx = 0.0
            dy = 0.0
            if mag > 1.0e-9:
                length = min(0.85 * step * scale, 0.15 * step * scale + 0.7 * step * scale * mag / max(flow_scale, 1.0e-12))
                dx = fy / mag * length
                dy = -fz / mag * length
            fill = (235, 245, 250) if float(area[row, col]) <= 0.0 else (60, 245, 165)
            draw.ellipse((x0 + dx - 2, y0 + dy - 2, x0 + dx + 2, y0 + dy + 2), fill=fill, outline=(5, 35, 60))
            if mag > 1.0e-9:
                draw.line((x0, y0, x0 + dx, y0 + dy), fill=(245, 245, 245), width=1)
    return img


def render_candidate_frames(
    run_tag: str,
    fps: int,
    scene_frames: list[np.ndarray] | None,
    left_fn: np.ndarray,
    right_fn: np.ndarray,
    left_ft: np.ndarray,
    right_ft: np.ndarray,
    left_area: np.ndarray,
    right_area: np.ndarray,
    left_y: np.ndarray,
    left_z: np.ndarray,
    right_y: np.ndarray,
    right_z: np.ndarray,
    left_normal_y: np.ndarray,
    left_normal_z: np.ndarray,
    right_normal_y: np.ndarray,
    right_normal_z: np.ndarray,
    left_marker_y: np.ndarray,
    left_marker_z: np.ndarray,
    right_marker_y: np.ndarray,
    right_marker_z: np.ndarray,
    object_z: np.ndarray,
    pad_object_count: np.ndarray,
    pad_object_fn: np.ndarray,
    pad_object_ft: np.ndarray,
) -> list[np.ndarray]:
    del fps
    fn_nonzero = np.concatenate([left_fn[left_fn > 0.0], right_fn[right_fn > 0.0]])
    ft_nonzero = np.concatenate([left_ft[left_ft > 0.0], right_ft[right_ft > 0.0]])
    area_nonzero = np.concatenate([left_area[left_area > 0.0], right_area[right_area > 0.0]])
    fn_vmax = float(np.quantile(fn_nonzero, 0.95)) if fn_nonzero.size else 1.0
    ft_vmax = float(np.quantile(ft_nonzero, 0.95)) if ft_nonzero.size else 1.0
    area_vmax = float(np.quantile(area_nonzero, 0.95)) if area_nonzero.size else 1.0
    scale = 5
    map_size = left_fn.shape[1]
    tactile_w = map_size * scale
    frames: list[np.ndarray] = []
    for frame_idx in range(left_fn.shape[0]):
        canvas = Image.new("RGB", (1180, 980), (236, 236, 230))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f"{run_tag} frame {frame_idx:03d}", fill=(25, 25, 25))
        if scene_frames is not None:
            scene = Image.fromarray(scene_frames[frame_idx]).resize((560, 330), Image.Resampling.LANCZOS)
            draw.text((16, 30), "scene: Newton SensorTiledCamera", fill=(35, 35, 35))
        else:
            scene = Image.new("RGB", (560, 330), (248, 248, 242))
            ImageDraw.Draw(scene).text((12, 12), "scene camera disabled", fill=(35, 35, 35))
        canvas.paste(scene, (16, 50))

        marker_y = 405
        left_marker = candidate_gel_marker_panel(
            left_fn[frame_idx],
            left_area[frame_idx],
            left_marker_y[frame_idx],
            left_marker_z[frame_idx],
            fn_vmax,
            area_vmax,
            scale,
        )
        right_marker = candidate_gel_marker_panel(
            right_fn[frame_idx],
            right_area[frame_idx],
            right_marker_y[frame_idx],
            right_marker_z[frame_idx],
            fn_vmax,
            area_vmax,
            scale,
        )
        draw.text((24, marker_y - 18), "L candidate gel/marker render", fill=(25, 25, 25))
        draw.text((214, marker_y - 18), "R candidate gel/marker render", fill=(25, 25, 25))
        canvas.paste(left_marker, (24, marker_y))
        canvas.paste(right_marker, (214, marker_y))
        draw.rectangle((24, marker_y, 24 + tactile_w, marker_y + tactile_w), outline=(60, 60, 60))
        draw.rectangle((214, marker_y, 214 + tactile_w, marker_y + tactile_w), outline=(60, 60, 60))

        x_l = 620
        x_r = 620 + tactile_w + 36
        y_fn = 72
        y_ft = 280
        y_area = 488
        panels = [
            (
                x_l,
                y_fn,
                "L candidate Fn + normal",
                tactile_panel(left_fn[frame_idx], fn_vmax, scale, (left_normal_y[frame_idx], left_normal_z[frame_idx])),
            ),
            (
                x_r,
                y_fn,
                "R candidate Fn + normal",
                tactile_panel(right_fn[frame_idx], fn_vmax, scale, (right_normal_y[frame_idx], right_normal_z[frame_idx])),
            ),
            (
                x_l,
                y_ft,
                "L candidate Ft vector",
                tactile_panel(left_ft[frame_idx], ft_vmax, scale, (left_y[frame_idx], left_z[frame_idx])),
            ),
            (
                x_r,
                y_ft,
                "R candidate Ft vector",
                tactile_panel(right_ft[frame_idx], ft_vmax, scale, (right_y[frame_idx], right_z[frame_idx])),
            ),
            (
                x_l,
                y_area,
                "L contact area proxy + normal",
                tactile_panel(left_area[frame_idx], area_vmax, scale, (left_normal_y[frame_idx], left_normal_z[frame_idx])),
            ),
            (
                x_r,
                y_area,
                "R contact area proxy + normal",
                tactile_panel(right_area[frame_idx], area_vmax, scale, (right_normal_y[frame_idx], right_normal_z[frame_idx])),
            ),
        ]
        for x, y, label, panel in panels:
            draw.text((x, y - 18), label, fill=(25, 25, 25))
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + tactile_w, y + tactile_w), outline=(60, 60, 60))

        curves = [
            ("object_z", object_z, (45, 100, 150)),
            ("pad_object_count", pad_object_count.astype(np.float32), (160, 95, 45)),
            ("candidate Fn sum", pad_object_fn, (120, 65, 145)),
            ("candidate Ft sum", pad_object_ft, (45, 120, 120)),
        ]
        for i, (name, values, color) in enumerate(curves):
            x = 30 + (i % 2) * 560
            y = 585 + (i // 2) * 140
            draw.text((x, y - 20), name, fill=(35, 35, 35))
            canvas.paste(sparkline(values[: frame_idx + 1], 480, 85, color), (x, y))
            draw.text((x, y + 92), f"now={float(values[frame_idx]):.5g} max={float(values.max(initial=0.0)):.5g}", fill=(35, 35, 35))

        draw.text(
            (24, 945),
            "candidate_mjw_direct_force.*; area is point-contact proxy, not final tactile sensor area.",
            fill=(45, 45, 45),
        )
        frames.append(np.asarray(canvas, dtype=np.uint8))
    return frames


def run(args: argparse.Namespace) -> dict:
    from newton.examples.robot.example_robot_panda_hydro import Example
    import newton

    started = time.perf_counter()
    wp.set_device(args.device)
    viewer = SurfaceNullViewer(num_frames=args.num_frames)
    example = Example(viewer, SimpleNamespace(scene=args.scene, test=True, world_count=1))

    material_override_applied = args.override_mu is not None or args.override_kh is not None
    material_notify_status = "not_needed"
    material_notify_error = None
    if args.override_mu is not None:
        example.model.shape_material_mu.fill_(float(args.override_mu))
    if args.override_kh is not None:
        example.model.shape_material_kh.fill_(float(args.override_kh))
    if material_override_applied:
        try:
            example.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
            material_notify_status = "pass"
        except Exception as exc:  # noqa: BLE001
            material_notify_status = "failed_nonblocking"
            material_notify_error = f"{type(exc).__name__}: {exc}"
        wp.synchronize()

    labels = list(example.model.body_label)
    shape_body = example.model.shape_body.numpy()
    shape_classes = [classify_shape(i, shape_body, example.model) for i in range(example.model.shape_count)]
    left_body = next((i for i, label in enumerate(labels) if "leftfinger" in label.lower()), None)
    right_body = next((i for i, label in enumerate(labels) if "rightfinger" in label.lower()), None)
    object_body = example.object_body_local

    scene_camera_sensor = None
    scene_camera_rays = None
    scene_camera_color_image = None
    scene_frames: list[np.ndarray] | None = None
    scene_camera_meta = None
    if args.scene_camera:
        from newton.sensors import SensorTiledCamera

        scene_camera_sensor = SensorTiledCamera(model=example.model)
        scene_camera_sensor.utils.create_default_light(enable_shadows=True)
        scene_camera_rays = scene_camera_sensor.utils.compute_pinhole_camera_rays(
            args.scene_camera_width,
            args.scene_camera_height,
            [math.radians(45.0)] * 3,
        )
        scene_camera_color_image = scene_camera_sensor.utils.create_color_image_output(
            args.scene_camera_width,
            args.scene_camera_height,
            camera_count=3,
        )
        scene_frames = []

    left_samples: list[list[dict[str, np.ndarray]]] = [[] for _ in range(args.num_frames)]
    right_samples: list[list[dict[str, np.ndarray]]] = [[] for _ in range(args.num_frames)]
    object_z = np.zeros(args.num_frames, dtype=np.float32)
    nacon_series = np.zeros(args.num_frames, dtype=np.int32)
    pad_object_count = np.zeros(args.num_frames, dtype=np.int32)
    pad_object_fn = np.zeros(args.num_frames, dtype=np.float32)
    pad_object_ft = np.zeros(args.num_frames, dtype=np.float32)
    left_fn_sum = np.zeros(args.num_frames, dtype=np.float32)
    right_fn_sum = np.zeros(args.num_frames, dtype=np.float32)
    left_ft_sum = np.zeros(args.num_frames, dtype=np.float32)
    right_ft_sum = np.zeros(args.num_frames, dtype=np.float32)
    read_errors: list[str] = []

    for frame in range(args.num_frames):
        try:
            example.step()
            wp.synchronize()
            body_q = example.state_0.body_q.numpy().astype(np.float32)
            object_z[frame] = float(body_q[object_body, 2])

            if scene_camera_sensor is not None and scene_camera_rays is not None and scene_camera_color_image is not None:
                from newton.sensors import SensorTiledCamera

                example.model.bvh_refit_shapes(example.state_0)
                transforms, scene_camera_meta = scene_camera_transforms(frame, args.num_frames, example.world_count)
                scene_camera_sensor.update(
                    example.state_0,
                    transforms,
                    scene_camera_rays,
                    color_image=scene_camera_color_image,
                    clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
                )
                rgba = scene_camera_sensor.utils.to_rgba_from_color(scene_camera_color_image).numpy().copy()
                scene_frames.append(
                    np.asarray(compose_scene_camera_triptych(rgba, f"{args.run_tag} frame={frame}"), dtype=np.uint8)
                )

            solver = example.solver
            mjw_data = solver.mjw_data
            contact = mjw_data.contact
            nacon = int(to_numpy(mjw_data.nacon).reshape(-1)[0])
            nacon = max(0, min(nacon, int(mjw_data.naconmax)))
            nacon_series[frame] = nacon
            if nacon <= 0:
                continue

            geom = to_numpy(contact.geom)[:nacon]
            pos = to_numpy(contact.pos)[:nacon].astype(np.float32)
            frames_mj = to_numpy(contact.frame)[:nacon]
            efc_address = to_numpy(contact.efc_address)[:nacon]
            worldid = to_numpy(contact.worldid)[:nacon].reshape(-1)
            efc_force = to_numpy(mjw_data.efc.force)
            geom_to_shape = to_numpy(solver.mjc_geom_to_newton_shape)

            for cidx in range(nacon):
                world = int(worldid[cidx]) if cidx < worldid.size else 0
                shape0, shape1 = geom_pair_to_shapes(geom[cidx], world, geom_to_shape)
                side_pair = pad_side_for_pair(shape_classes, shape0, shape1)
                if side_pair is None:
                    continue
                side, pad_is_shape0 = side_pair
                force_row = force_table_for_world(efc_force, world)
                addresses = [int(a) for a in np.asarray(efc_address[cidx]).reshape(-1) if 0 <= int(a) < force_row.size]
                if not addresses:
                    continue
                values = np.asarray([float(force_row[a]) for a in addresses], dtype=np.float32)
                fn = abs(float(values[0]))
                frame_mat = frame_matrix(frames_mj[cidx])
                tangent_world = np.zeros(3, dtype=np.float32)
                for value, basis in zip(values[1:], frame_mat[1 : 1 + max(0, len(values) - 1)], strict=False):
                    tangent_world += float(value) * basis.astype(np.float32)
                tangent_world *= -1.0 if pad_is_shape0 else 1.0
                normal_world = frame_mat[0].astype(np.float32)
                normal_world *= -1.0 if pad_is_shape0 else 1.0
                ft = float(np.linalg.norm(tangent_world))
                body_idx = left_body if side == "left" else right_body
                if body_idx is None:
                    continue
                local_point = world_to_body(pos[cidx : cidx + 1], body_q[body_idx])[0]
                local_tangent = world_vector_to_body(tangent_world[None, :], body_q[body_idx])[0]
                local_normal = world_vector_to_body(normal_world[None, :], body_q[body_idx])[0]
                local_normal_norm = float(np.linalg.norm(local_normal))
                if local_normal_norm > 1.0e-12:
                    local_normal = local_normal / local_normal_norm
                sample = {
                    "local": local_point.astype(np.float32),
                    "fn": np.asarray(fn, dtype=np.float32),
                    "ft": np.asarray(ft, dtype=np.float32),
                    "tangent_yz": local_tangent[1:3].astype(np.float32),
                    "normal_yz": local_normal[1:3].astype(np.float32),
                    "shape0": np.asarray(shape0, dtype=np.int32),
                    "shape1": np.asarray(shape1, dtype=np.int32),
                }
                if side == "left":
                    left_samples[frame].append(sample)
                    left_fn_sum[frame] += fn
                    left_ft_sum[frame] += ft
                else:
                    right_samples[frame].append(sample)
                    right_fn_sum[frame] += fn
                    right_ft_sum[frame] += ft
                pad_object_count[frame] += 1
                pad_object_fn[frame] += fn
                pad_object_ft[frame] += ft
        except Exception as exc:  # noqa: BLE001
            read_errors.append(f"frame {frame}: {type(exc).__name__}: {exc}")
            continue

    official_final_test_status = "not_run"
    official_final_test_error = None
    try:
        example.test_final()
        official_final_test_status = "pass"
    except AssertionError as exc:
        official_final_test_status = "failed_nonblocking"
        official_final_test_error = str(exc)
    viewer.close()

    left_center, left_extent, left_calib_valid = contact_view_window(
        [
            {"local": np.stack([sample["local"] for sample in frame_samples], axis=0)}
            for frame_samples in left_samples
            if frame_samples
        ]
    )
    right_center, right_extent, right_calib_valid = contact_view_window(
        [
            {"local": np.stack([sample["local"] for sample in frame_samples], axis=0)}
            for frame_samples in right_samples
            if frame_samples
        ]
    )

    shape = (args.num_frames, args.map_size, args.map_size)
    left_fn_map = np.zeros(shape, dtype=np.float32)
    right_fn_map = np.zeros(shape, dtype=np.float32)
    left_ft_map = np.zeros(shape, dtype=np.float32)
    right_ft_map = np.zeros(shape, dtype=np.float32)
    left_area_map = np.zeros(shape, dtype=np.float32)
    right_area_map = np.zeros(shape, dtype=np.float32)
    left_shear_y_map = np.zeros(shape, dtype=np.float32)
    left_shear_z_map = np.zeros(shape, dtype=np.float32)
    right_shear_y_map = np.zeros(shape, dtype=np.float32)
    right_shear_z_map = np.zeros(shape, dtype=np.float32)
    left_normal_y_map = np.zeros(shape, dtype=np.float32)
    left_normal_z_map = np.zeros(shape, dtype=np.float32)
    right_normal_y_map = np.zeros(shape, dtype=np.float32)
    right_normal_z_map = np.zeros(shape, dtype=np.float32)

    for frame, frame_samples in enumerate(left_samples):
        if not frame_samples:
            continue
        local = np.stack([sample["local"] for sample in frame_samples], axis=0)
        fn = np.asarray([float(sample["fn"]) for sample in frame_samples], dtype=np.float32)
        ft = np.asarray([float(sample["ft"]) for sample in frame_samples], dtype=np.float32)
        tangent = np.stack([sample["tangent_yz"] for sample in frame_samples], axis=0).astype(np.float32)
        normal = np.stack([sample["normal_yz"] for sample in frame_samples], axis=0).astype(np.float32)
        accumulate_gaussian_per_point(left_fn_map[frame], left_shear_y_map[frame], left_shear_z_map[frame], local, fn, np.zeros_like(tangent), left_extent, left_center)
        accumulate_gaussian_per_point(left_ft_map[frame], left_shear_y_map[frame], left_shear_z_map[frame], local, ft, tangent, left_extent, left_center)
        accumulate_gaussian_per_point(left_area_map[frame], left_normal_y_map[frame], left_normal_z_map[frame], local, np.ones_like(fn), normal, left_extent, left_center)
    for frame, frame_samples in enumerate(right_samples):
        if not frame_samples:
            continue
        local = np.stack([sample["local"] for sample in frame_samples], axis=0)
        fn = np.asarray([float(sample["fn"]) for sample in frame_samples], dtype=np.float32)
        ft = np.asarray([float(sample["ft"]) for sample in frame_samples], dtype=np.float32)
        tangent = np.stack([sample["tangent_yz"] for sample in frame_samples], axis=0).astype(np.float32)
        normal = np.stack([sample["normal_yz"] for sample in frame_samples], axis=0).astype(np.float32)
        accumulate_gaussian_per_point(right_fn_map[frame], right_shear_y_map[frame], right_shear_z_map[frame], local, fn, np.zeros_like(tangent), right_extent, right_center)
        accumulate_gaussian_per_point(right_ft_map[frame], right_shear_y_map[frame], right_shear_z_map[frame], local, ft, tangent, right_extent, right_center)
        accumulate_gaussian_per_point(right_area_map[frame], right_normal_y_map[frame], right_normal_z_map[frame], local, np.ones_like(fn), normal, right_extent, right_center)

    left_marker_y_map, left_marker_z_map = marker_flow_from_fields(
        left_shear_y_map,
        left_shear_z_map,
        left_normal_y_map,
        left_normal_z_map,
        left_area_map,
    )
    right_marker_y_map, right_marker_z_map = marker_flow_from_fields(
        right_shear_y_map,
        right_shear_z_map,
        right_normal_y_map,
        right_normal_z_map,
        right_area_map,
    )
    left_cop_proxy_yz = center_of_pressure_proxy_from_map(left_fn_map, left_extent, left_center)
    right_cop_proxy_yz = center_of_pressure_proxy_from_map(right_fn_map, right_extent, right_center)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    npz_path = args.output_dir / "candidate_mjw_direct_tactile_timeseries.npz"
    np.savez_compressed(
        npz_path,
        object_z=object_z,
        nacon=nacon_series,
        pad_object_contact_count=pad_object_count,
        pad_object_candidate_fn_sum=pad_object_fn,
        pad_object_candidate_ft_sum=pad_object_ft,
        left_candidate_fn_sum=left_fn_sum,
        right_candidate_fn_sum=right_fn_sum,
        left_candidate_ft_sum=left_ft_sum,
        right_candidate_ft_sum=right_ft_sum,
        left_candidate_fn_map=left_fn_map,
        right_candidate_fn_map=right_fn_map,
        left_candidate_ft_map=left_ft_map,
        right_candidate_ft_map=right_ft_map,
        left_candidate_contact_area_proxy_map=left_area_map,
        right_candidate_contact_area_proxy_map=right_area_map,
        left_candidate_center_of_pressure_proxy_yz=left_cop_proxy_yz,
        right_candidate_center_of_pressure_proxy_yz=right_cop_proxy_yz,
        left_candidate_shear_y_map=left_shear_y_map,
        left_candidate_shear_z_map=left_shear_z_map,
        right_candidate_shear_y_map=right_shear_y_map,
        right_candidate_shear_z_map=right_shear_z_map,
        left_candidate_normal_y_map=left_normal_y_map,
        left_candidate_normal_z_map=left_normal_z_map,
        right_candidate_normal_y_map=right_normal_y_map,
        right_candidate_normal_z_map=right_normal_z_map,
        left_candidate_marker_flow_y_map=left_marker_y_map,
        left_candidate_marker_flow_z_map=left_marker_z_map,
        right_candidate_marker_flow_y_map=right_marker_y_map,
        right_candidate_marker_flow_z_map=right_marker_z_map,
    )

    video_frames = render_candidate_frames(
        args.run_tag,
        args.fps,
        scene_frames,
        left_fn_map,
        right_fn_map,
        left_ft_map,
        right_ft_map,
        left_area_map,
        right_area_map,
        left_shear_y_map,
        left_shear_z_map,
        right_shear_y_map,
        right_shear_z_map,
        left_normal_y_map,
        left_normal_z_map,
        right_normal_y_map,
        right_normal_z_map,
        left_marker_y_map,
        left_marker_z_map,
        right_marker_y_map,
        right_marker_z_map,
        object_z,
        pad_object_count,
        pad_object_fn,
        pad_object_ft,
    )
    video_path = args.visual_dir / "candidate_mjw_direct_tactile.mp4"
    write_mp4_video(video_path, video_frames, args.fps)
    sample_indices = np.linspace(0, len(video_frames) - 1, min(args.sheet_frames, len(video_frames)), dtype=int)
    sheet_frames = [Image.fromarray(video_frames[int(i)]).resize((590, 410), Image.Resampling.LANCZOS) for i in sample_indices]
    sheet_cols = 2
    sheet_rows = int(math.ceil(len(sheet_frames) / sheet_cols))
    sheet = Image.new("RGB", (sheet_cols * 590, sheet_rows * 410), (240, 240, 236))
    for idx, image in enumerate(sheet_frames):
        sheet.paste(image, ((idx % sheet_cols) * 590, (idx // sheet_cols) * 410))
    sheet_path = args.visual_dir / "candidate_mjw_direct_tactile_sheet.jpg"
    sheet.save(sheet_path, quality=92)

    cell_count = float(args.map_size * args.map_size)
    max_cell_ratio = lambda maps: float(((maps > 0.0).sum(axis=(1, 2)) / cell_count).max(initial=0.0))
    summary = {
        "classification": "phase00_candidate_mjwarp_direct_force_tactile_export_v1",
        "run_tag": args.run_tag,
        "status": "pass_candidate_direct_force_export" if float(pad_object_fn.max(initial=0.0)) > 0.0 and float(pad_object_ft.max(initial=0.0)) > 0.0 else "blocked_no_candidate_pad_object_force",
        "not_training_result": True,
        "not_curiosity_success": True,
        "direct_tactile_claim_allowed": False,
        "official_example": "newton.examples.robot.example_robot_panda_hydro",
        "method": "map MJWarp contact EFC normal/tangent components into left/right finger-local tactile grids for pad-object contacts",
        "normal_area_overlay": "candidate contact normals from MJWarp contact.frame and contact-area proxy from pad-object point-contact density are rendered in the same direct-force video",
        "center_of_pressure_proxy": "candidate center-of-pressure proxy is the force-map weighted center in each pad-local Y/Z plane; it is not validated hardware CoP",
        "candidate_gel_marker_render": "blue gel-like candidate rendering with marker displacement overlay derived from candidate Fn/Ft/normal/area-proxy fields; not validated Taccel or hardware photometric marker output",
        "remaining_gap": "compatible SensorContact alignment passed in p00_mjw_align_v1_20260701_055200; active hydro output remains candidate until validated gel/marker photometric semantics, real contact-area validation, and final gate review",
        "num_frames": int(args.num_frames),
        "scene": args.scene,
        "material_label": args.material_label,
        "requested_override_mu": args.override_mu,
        "requested_override_kh": args.override_kh,
        "observed_shape_material_mu_unique": sorted({float(v) for v in example.model.shape_material_mu.numpy().tolist()}),
        "observed_shape_material_kh_unique": sorted({float(v) for v in example.model.shape_material_kh.numpy().tolist()}),
        "material_notify_status": material_notify_status,
        "material_notify_error": material_notify_error,
        "force_sign_convention": "shape0_negative validated by p00_mjw_align_v1_20260701_055200 on compatible SensorContact scene",
        "scene_camera_enabled": bool(args.scene_camera),
        "scene_camera_meta_last": scene_camera_meta,
        "read_error_count": len(read_errors),
        "read_errors_first": read_errors[:5],
        "official_final_test_status": official_final_test_status,
        "official_final_test_error": official_final_test_error,
        "max_nacon": int(nacon_series.max(initial=0)),
        "frames_with_pad_object_contacts": int((pad_object_count > 0).sum()),
        "max_pad_object_contact_count": int(pad_object_count.max(initial=0)),
        "max_pad_object_candidate_fn_sum": float(pad_object_fn.max(initial=0.0)),
        "max_pad_object_candidate_ft_sum": float(pad_object_ft.max(initial=0.0)),
        "max_left_candidate_fn_sum": float(left_fn_sum.max(initial=0.0)),
        "max_right_candidate_fn_sum": float(right_fn_sum.max(initial=0.0)),
        "max_left_candidate_ft_sum": float(left_ft_sum.max(initial=0.0)),
        "max_right_candidate_ft_sum": float(right_ft_sum.max(initial=0.0)),
        "max_left_candidate_fn_map": float(left_fn_map.max(initial=0.0)),
        "max_right_candidate_fn_map": float(right_fn_map.max(initial=0.0)),
        "max_left_candidate_ft_map": float(left_ft_map.max(initial=0.0)),
        "max_right_candidate_ft_map": float(right_ft_map.max(initial=0.0)),
        "max_left_candidate_contact_area_proxy_map": float(left_area_map.max(initial=0.0)),
        "max_right_candidate_contact_area_proxy_map": float(right_area_map.max(initial=0.0)),
        "max_left_candidate_fn_nonzero_cell_ratio": max_cell_ratio(left_fn_map),
        "max_right_candidate_fn_nonzero_cell_ratio": max_cell_ratio(right_fn_map),
        "max_left_candidate_contact_area_proxy_cell_ratio": max_cell_ratio(left_area_map),
        "max_right_candidate_contact_area_proxy_cell_ratio": max_cell_ratio(right_area_map),
        "left_candidate_center_of_pressure_proxy_valid_frames": int(np.isfinite(left_cop_proxy_yz).all(axis=1).sum()),
        "right_candidate_center_of_pressure_proxy_valid_frames": int(np.isfinite(right_cop_proxy_yz).all(axis=1).sum()),
        "max_left_candidate_normal_yz_norm": float(np.sqrt(left_normal_y_map * left_normal_y_map + left_normal_z_map * left_normal_z_map).max(initial=0.0)),
        "max_right_candidate_normal_yz_norm": float(np.sqrt(right_normal_y_map * right_normal_y_map + right_normal_z_map * right_normal_z_map).max(initial=0.0)),
        "max_left_candidate_marker_flow_norm": float(np.sqrt(left_marker_y_map * left_marker_y_map + left_marker_z_map * left_marker_z_map).max(initial=0.0)),
        "max_right_candidate_marker_flow_norm": float(np.sqrt(right_marker_y_map * right_marker_y_map + right_marker_z_map * right_marker_z_map).max(initial=0.0)),
        "left_calibrated_view_valid": bool(left_calib_valid),
        "right_calibrated_view_valid": bool(right_calib_valid),
        "left_calibrated_view_center_yz": left_center.tolist(),
        "right_calibrated_view_center_yz": right_center.tolist(),
        "left_calibrated_view_extent_yz": [float(v) for v in left_extent],
        "right_calibrated_view_extent_yz": [float(v) for v in right_extent],
        "max_object_lift_m": float(object_z.max(initial=object_z[0]) - object_z[0]),
        "npz_path": str(npz_path),
        "video_path": str(video_path),
        "sheet_path": str(sheet_path),
        "elapsed_s": float(time.perf_counter() - started),
    }
    summary_path = args.output_dir / "candidate_mjw_direct_tactile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    report_path = args.report_dir / "candidate_mjw_direct_tactile.md"
    report_path.write_text(
        "# Phase 00 Candidate MJWarp Direct-Force Tactile Export\n\n"
        f"- run_tag: `{args.run_tag}`\n"
        f"- status: `{summary['status']}`\n"
        f"- max pad-object candidate Fn sum: `{summary['max_pad_object_candidate_fn_sum']}`\n"
        f"- max pad-object candidate Ft sum: `{summary['max_pad_object_candidate_ft_sum']}`\n"
        f"- max left/right candidate Fn map: `{summary['max_left_candidate_fn_map']}` / `{summary['max_right_candidate_fn_map']}`\n"
        f"- max left/right candidate Ft map: `{summary['max_left_candidate_ft_map']}` / `{summary['max_right_candidate_ft_map']}`\n"
        f"- max left/right contact area proxy map: `{summary['max_left_candidate_contact_area_proxy_map']}` / `{summary['max_right_candidate_contact_area_proxy_map']}`\n"
        f"- max left/right normal-yz norm: `{summary['max_left_candidate_normal_yz_norm']}` / `{summary['max_right_candidate_normal_yz_norm']}`\n"
        f"- max left/right candidate marker-flow norm: `{summary['max_left_candidate_marker_flow_norm']}` / `{summary['max_right_candidate_marker_flow_norm']}`\n"
        f"- summary: `{summary_path}`\n"
        f"- video: `{video_path}`\n"
        f"- sheet: `{sheet_path}`\n\n"
        "This is candidate direct-force tactile evidence only. Contact area is a point-contact-density proxy, and the gel/marker panel is a candidate rendering derived from direct-force fields, not validated sensor photometry. It is not training, not curiosity success, and not final tactile-sensor validation.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", choices=["cube", "pen"], default="cube")
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--map-size", type=int, default=32)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sheet-frames", type=int, default=12)
    parser.add_argument("--material-label", default="official_default")
    parser.add_argument("--override-mu", type=float, default=None)
    parser.add_argument("--override-kh", type=float, default=None)
    parser.add_argument("--scene-camera", action="store_true")
    parser.add_argument("--scene-camera-width", type=int, default=256)
    parser.add_argument("--scene-camera-height", type=int, default=256)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    summary = run(args)
    return 0 if str(summary["status"]).startswith("pass_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
