#!/usr/bin/env python3
"""Render native VBD tactile force while the official Franka grasps a soft duck."""

from __future__ import annotations

import json
import gc
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import warp as wp

import newton
import newton.examples
from newton.examples.softbody.example_softbody_franka import Example
from newton.sensors import SensorTactile

from scripts.sugar.native_tactile.slip import SlipState, TactileSlipDetector
from scripts.sugar.native_tactile.universal import NewtonTactileAdapter


FONT = ImageFont.load_default()
STATE_NAMES = tuple(state.name for state in SlipState)


class TactileExample(Example):
    """Capture the official graph with force-reporting contacts from the start."""

    def capture(self) -> None:
        self.model.request_contact_attributes("force")
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.soft_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()
        super().capture()


def _finger_collision_shapes(model: newton.Model) -> tuple[list[int], list[str]]:
    body_indices = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    shapes: list[int] = []
    names: list[str] = []
    side_counts = {"left": 0, "right": 0}
    for shape, (body, shape_flags) in enumerate(zip(body_indices, flags)):
        if body < 0 or not (int(shape_flags) & int(newton.ShapeFlags.COLLIDE_PARTICLES)):
            continue
        body_name = model.body_label[int(body)]
        side = "left" if "leftfinger" in body_name else "right" if "rightfinger" in body_name else None
        if side is None:
            continue
        shapes.append(shape)
        names.append(f"{side}_finger_surface_{side_counts[side]}")
        side_counts[side] += 1
    if side_counts != {"left": 4, "right": 4}:
        raise RuntimeError(f"Expected four collision surfaces per finger, got {side_counts}.")
    return shapes, names


def _heatmap(values: np.ndarray, scale: float, size: tuple[int, int]) -> Image.Image:
    normalized = np.clip(values / max(scale, 1.0e-9), 0.0, 1.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.asarray(255.0 * normalized, dtype=np.uint8)
    rgb[..., 1] = np.asarray(220.0 * np.sqrt(normalized), dtype=np.uint8)
    rgb[..., 2] = np.asarray(255.0 * (1.0 - normalized), dtype=np.uint8)
    return Image.fromarray(np.flipud(rgb), mode="RGB").resize(size, Image.Resampling.NEAREST)


def _transform_matrix(transform: np.ndarray) -> np.ndarray:
    """Convert Newton's [xyz, xyzw] transform to a 4x4 matrix."""

    px, py, pz, x, y, z, w = (float(value) for value in transform)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), px],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), py],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), pz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _vtk_polydata(vertices: np.ndarray, indices: np.ndarray):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(vertices, dtype=np.float32), deep=True))
    triangles = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    packed = np.column_stack((np.full(len(triangles), 3, dtype=np.int64), triangles)).reshape(-1)
    cells = vtk.vtkCellArray()
    cells.SetCells(len(triangles), numpy_to_vtkIdTypeArray(packed, deep=True))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    return polydata


class NewtonVTKRenderer:
    """Offscreen renderer of the exact Newton model geometry and live state."""

    def __init__(
        self,
        model: newton.Model,
        *,
        width: int = 1280,
        height: int = 720,
        camera_position: tuple[float, float, float] = (-1.25, 0.75, 1.05),
        camera_target: tuple[float, float, float] = (-0.20, -0.50, 0.35),
    ):
        import vtk

        self.vtk = vtk
        self.width = width
        self.height = height
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.92, 0.94, 0.97)
        self.window = vtk.vtkEGLRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetSize(width, height)
        self.window.SetMultiSamples(0)
        self.window.AddRenderer(self.renderer)
        self.capture = vtk.vtkWindowToImageFilter()
        self.capture.SetInput(self.window)
        self.capture.SetInputBufferTypeToRGB()

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*camera_position)
        camera.SetFocalPoint(*camera_target)
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.SetViewAngle(42.0)

        self.shape_body = model.shape_body.numpy().astype(np.int32)
        self.shape_transform = model.shape_transform.numpy().copy()
        shape_type = model.shape_type.numpy().astype(np.int32)
        shape_scale = model.shape_scale.numpy().copy()
        shape_flags = model.shape_flags.numpy().astype(np.int32)
        shape_color = model.shape_color.numpy().copy()
        self.shape_actors: list[tuple[int, object]] = []

        for shape_index in range(model.shape_count):
            flags = int(shape_flags[shape_index])
            body = int(self.shape_body[shape_index])
            visible = bool(flags & int(newton.ShapeFlags.VISIBLE))
            static_collision = body < 0 and bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES))
            if not (visible or static_collision):
                continue
            geo_type = int(shape_type[shape_index])
            scale = np.asarray(shape_scale[shape_index], dtype=np.float32)
            source = model.shape_source[shape_index]
            if geo_type in (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)):
                vertices = np.asarray(source.vertices, dtype=np.float32) * scale[None]
                indices = np.asarray(source.indices, dtype=np.int64).reshape(-1, 3)
                if float(np.prod(scale)) < 0.0:
                    indices = indices[:, [0, 2, 1]]
            else:
                mesh = self._primitive_mesh(geo_type, scale)
                if mesh is None:
                    continue
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                indices = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)

            polydata = _vtk_polydata(vertices, indices)
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.ComputePointNormalsOn()
            normals.SplittingOff()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            color = np.clip(shape_color[shape_index], 0.0, 1.0)
            if not visible:
                color = np.asarray((0.65, 0.68, 0.72), dtype=np.float32)
            actor.GetProperty().SetColor(*(float(value) for value in color))
            actor.GetProperty().SetRoughness(0.65)
            self.renderer.AddActor(actor)
            self.shape_actors.append((shape_index, actor))

        self.duck_polydata = None
        self.duck_points = None
        if model.tri_count > 0:
            particle_q = model.particle_q.numpy()
            triangle_indices = model.tri_indices.numpy()
            self.duck_polydata = _vtk_polydata(particle_q, triangle_indices)
            self.duck_points = self.duck_polydata.GetPoints()
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(self.duck_polydata)
            normals.ComputePointNormalsOn()
            normals.SplittingOff()
            duck_mapper = vtk.vtkPolyDataMapper()
            duck_mapper.SetInputConnection(normals.GetOutputPort())
            duck_actor = vtk.vtkActor()
            duck_actor.SetMapper(duck_mapper)
            duck_actor.GetProperty().SetColor(1.0, 0.72, 0.08)
            duck_actor.GetProperty().SetRoughness(0.8)
            self.renderer.AddActor(duck_actor)

        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(-0.8, 0.4, 1.8)
        light.SetFocalPoint(-0.2, -0.5, 0.25)
        light.SetIntensity(0.9)
        self.renderer.AddLight(light)
        fill = vtk.vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(0.8, -0.4, 1.0)
        fill.SetFocalPoint(-0.2, -0.5, 0.25)
        fill.SetIntensity(0.55)
        self.renderer.AddLight(fill)

    @staticmethod
    def _primitive_mesh(geo_type: int, scale: np.ndarray):
        if geo_type == int(newton.GeoType.PLANE):
            width = float(scale[0]) if scale[0] > 0.0 else 4.0
            length = float(scale[1]) if scale[1] > 0.0 else 4.0
            return newton.Mesh.create_plane(width, length, compute_inertia=False)
        if geo_type == int(newton.GeoType.SPHERE):
            return newton.Mesh.create_sphere(float(scale[0]), compute_inertia=False)
        if geo_type == int(newton.GeoType.CAPSULE):
            return newton.Mesh.create_capsule(
                float(scale[0]), float(scale[1]), up_axis=newton.Axis.Z, compute_inertia=False
            )
        if geo_type == int(newton.GeoType.CYLINDER):
            return newton.Mesh.create_cylinder(
                float(scale[0]), float(scale[1]), up_axis=newton.Axis.Z, compute_inertia=False
            )
        if geo_type == int(newton.GeoType.CONE):
            return newton.Mesh.create_cone(
                float(scale[0]), float(scale[1]), up_axis=newton.Axis.Z, compute_inertia=False
            )
        if geo_type == int(newton.GeoType.BOX):
            return newton.Mesh.create_box(*[float(value) for value in scale], compute_inertia=False)
        if geo_type == int(newton.GeoType.ELLIPSOID):
            return newton.Mesh.create_ellipsoid(*[float(value) for value in scale], compute_inertia=False)
        return None

    def render(self, state: newton.State) -> np.ndarray:
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        body_q = state.body_q.numpy()
        for shape_index, actor in self.shape_actors:
            shape_matrix = _transform_matrix(self.shape_transform[shape_index])
            body = int(self.shape_body[shape_index])
            world_matrix = shape_matrix if body < 0 else _transform_matrix(body_q[body]) @ shape_matrix
            vtk_matrix = self.vtk.vtkMatrix4x4()
            vtk_matrix.DeepCopy(world_matrix.reshape(-1))
            actor.SetUserMatrix(vtk_matrix)

        if self.duck_points is not None:
            self.duck_points.SetData(numpy_to_vtk(state.particle_q.numpy().astype(np.float32), deep=True))
            self.duck_points.Modified()
            self.duck_polydata.Modified()

        self.renderer.ResetCameraClippingRange()
        self.window.Render()
        self.capture.Modified()
        self.capture.Update()
        image = self.capture.GetOutput()
        rgb = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(self.height, self.width, 3)
        return np.flipud(rgb).copy()

    def close(self) -> None:
        self.window.Finalize()


def _compose(
    world: np.ndarray,
    force: np.ndarray,
    evidence,
    *,
    frame: int,
    timestamp: float,
    raw_count: int,
    residual: float,
    force_scale: float,
) -> np.ndarray:
    canvas = Image.new("RGB", (1280, 720), "white")
    world_image = Image.fromarray(world).resize((1280, 470), Image.Resampling.LANCZOS)
    canvas.paste(world_image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1279, 34), fill=(255, 255, 255))
    draw.text((12, 8), "Newton VBD | official Franka + deformable rubber duck | solved native tactile", fill="black", font=FONT)
    draw.text(
        (840, 8),
        f"frame {frame:04d}  t={timestamp:6.2f}s  raw={raw_count:3d}  conservation={residual:.2e} N",
        fill="black",
        font=FONT,
    )

    magnitude = np.linalg.norm(force, axis=-1)
    for side_index, (side, patch_indices) in enumerate((("LEFT", range(0, 4)), ("RIGHT", range(4, 8)))):
        base_x = 20 + side_index * 640
        draw.text((base_x, 485), f"{side} FRANKA FINGER: 4 real collision surfaces", fill="black", font=FONT)
        for local_index, patch_index in enumerate(patch_indices):
            x = base_x + local_index * 150
            y = 510
            canvas.paste(_heatmap(magnitude[patch_index], force_scale, (130, 130)), (x, y))
            state = STATE_NAMES[int(evidence.state[0, patch_index])]
            load = float(np.linalg.norm(force[patch_index].sum(axis=(0, 1))))
            draw.text((x, 645), f"surface {local_index} | {state}", fill="black", font=FONT)
            draw.text((x, 660), f"|sum F|={load:7.2f} N", fill="black", font=FONT)
    draw.text(
        (20, 695),
        "Heatmaps show |signed local XYZ force| at fixed taxels; world view renders exact Newton geometry/state with VTK offscreen.",
        fill="black",
        font=FONT,
    )
    return np.asarray(canvas)


def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--force-scale-n", type=float, default=10.0)
    parser.add_argument("--renderer-refresh-frames", type=int, default=40)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.set_defaults(viewer="null", headless=True, num_frames=720)
    viewer, args = newton.examples.init(parser)
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

    example = TactileExample(viewer, args)
    world_renderer = NewtonVTKRenderer(example.model)
    shape_flags = example.model.shape_flags.numpy()
    print(
        "softbody_tactile vtk_render_objects "
        f"up_axis={example.model.up_axis} shapes={example.model.shape_count} "
        f"visible_shapes={int(np.count_nonzero(shape_flags & int(newton.ShapeFlags.VISIBLE)))} "
        f"rendered_shapes={len(world_renderer.shape_actors)} soft_triangles={example.model.tri_count}",
        flush=True,
    )

    sensing_shapes, patch_names = _finger_collision_shapes(example.model)
    sensor = SensorTactile(
        example.model,
        sensing_shapes=sensing_shapes,
        grid_shape=(20, 25),
        patch_size=(0.08, 0.08),
    )
    adapter = NewtonTactileAdapter(sensor, patch_names)
    detector = TactileSlipDetector(patch_names, friction_coefficient=1.5)

    force_rows: list[np.ndarray] = []
    penetration_rows: list[np.ndarray] = []
    sequence_rows: list[int] = []
    timestamp_rows: list[float] = []
    dt_rows: list[float] = []
    records: list[dict] = []
    raw_rows: list[dict[str, np.ndarray]] = []
    max_residual = 0.0
    contact_frames = np.zeros(len(patch_names), dtype=np.int64)

    for frame_index in range(frame_stop):
        output_frame = frame_index - args.frame_start
        if (
            args.renderer_refresh_frames > 0
            and output_frame > 0
            and output_frame % args.renderer_refresh_frames == 0
        ):
            world_renderer.close()
            del world_renderer
            gc.collect()
            world_renderer = NewtonVTKRenderer(example.model)
        example.step()
        if frame_index == 0:
            wp.synchronize_device()
            print("softbody_tactile stage=official_step_cuda_ok", flush=True)
        example.soft_solver.update_contacts(example.contacts, example.state_0)
        if frame_index == 0:
            wp.synchronize_device()
            print("softbody_tactile stage=native_force_write_cuda_ok", flush=True)
            print(
                "softbody_tactile contacts "
                f"rigid={int(example.contacts.rigid_contact_count.numpy()[0])}/{example.contacts.rigid_contact_max} "
                f"soft={int(example.contacts.soft_contact_count.numpy()[0])}/{example.contacts.soft_contact_max}",
                flush=True,
            )
        sensor.update(example.state_0, example.contacts, timestamp=example.sim_time)
        if frame_index == 0:
            print("softbody_tactile stage=sensor_returned", flush=True)
        tactile = adapter.frame()
        if frame_index == 0:
            print("softbody_tactile stage=adapter_returned", flush=True)
        evidence = detector.update(tactile)
        if frame_index == 0:
            print("softbody_tactile stage=slip_returned", flush=True)

        if frame_index < args.frame_start:
            continue

        force = sensor.force.numpy().reshape((len(patch_names), 20, 25, 3)).copy()
        penetration = sensor.max_penetration.numpy().reshape((len(patch_names), 20, 25)).copy()
        dense_sum = force.sum(axis=(1, 2))
        total = sensor.total_force_patch.numpy()
        unmapped = sensor.unmapped_force_patch.numpy()
        residual = float(np.max(np.abs(total - dense_sum - unmapped)))
        max_residual = max(max_residual, residual)
        contact_frames += (np.linalg.norm(force, axis=-1) > 1.0e-8).any(axis=(1, 2))

        raw_count = int(sensor.raw_count.numpy()[0])
        raw_rows.append(
            {
                "contact_index": sensor.raw_contact_index.numpy()[:raw_count].copy(),
                "contact_kind": sensor.raw_contact_kind.numpy()[:raw_count].copy(),
                "patch": sensor.raw_patch.numpy()[:raw_count].copy(),
                "counterpart_particle": sensor.raw_counterpart_particle.numpy()[:raw_count].copy(),
                "point_world_m": sensor.raw_point_world.numpy()[:raw_count].copy(),
                "force_world_n": sensor.raw_force_world.numpy()[:raw_count].copy(),
                "force_patch_n": sensor.raw_force_patch.numpy()[:raw_count].copy(),
                "penetration_m": sensor.raw_penetration.numpy()[:raw_count].copy(),
            }
        )
        force_rows.append(force)
        penetration_rows.append(penetration)
        sequence_rows.append(tactile.clock.sequence)
        timestamp_rows.append(tactile.clock.timestamp_s)
        dt_rows.append(tactile.clock.dt_s)
        records.append(
            {
                "frame": frame_index,
                "timestamp_s": tactile.clock.timestamp_s,
                "raw_sample_count": raw_count,
                "force_conservation_residual_n": residual,
                "slip_state": evidence.state[0].astype(int).tolist(),
            }
        )

        world = world_renderer.render(example.state_0)
        if output_frame == 0:
            print(f"softbody_tactile stage=vtk_world_frame shape={world.shape}", flush=True)
        composed = _compose(
            world,
            force,
            evidence,
            frame=frame_index,
            timestamp=example.sim_time,
            raw_count=raw_count,
            residual=residual,
            force_scale=args.force_scale_n,
        )
        Image.fromarray(composed).save(frame_dir / f"frame_{output_frame:05d}.png")
        if output_frame % 50 == 0:
            print(
                f"softbody_tactile frame={frame_index} output={output_frame}/{rendered_frames} "
                f"raw={raw_count} residual_n={residual:.3e}",
                flush=True,
            )

    max_raw = max((len(row["contact_index"]) for row in raw_rows), default=0)
    raw_contact_kind = np.full((rendered_frames, max_raw), -1, dtype=np.int32)
    raw_patch = np.full((rendered_frames, max_raw), -1, dtype=np.int32)
    raw_counterpart_particle = np.full((rendered_frames, max_raw), -1, dtype=np.int32)
    raw_point_world = np.zeros((rendered_frames, max_raw, 3), dtype=np.float32)
    raw_force_world = np.zeros((rendered_frames, max_raw, 3), dtype=np.float32)
    raw_force_patch = np.zeros((rendered_frames, max_raw, 3), dtype=np.float32)
    raw_penetration = np.zeros((rendered_frames, max_raw), dtype=np.float32)
    raw_count = np.zeros(rendered_frames, dtype=np.int32)
    for frame_index, row in enumerate(raw_rows):
        count = len(row["contact_index"])
        raw_count[frame_index] = count
        raw_contact_kind[frame_index, :count] = row["contact_kind"]
        raw_patch[frame_index, :count] = row["patch"]
        raw_counterpart_particle[frame_index, :count] = row["counterpart_particle"]
        raw_point_world[frame_index, :count] = row["point_world_m"]
        raw_force_world[frame_index, :count] = row["force_world_n"]
        raw_force_patch[frame_index, :count] = row["force_patch_n"]
        raw_penetration[frame_index, :count] = row["penetration_m"]

    np.savez_compressed(
        output_root / "trace.npz",
        force_patch_n=np.stack(force_rows),
        penetration_m=np.stack(penetration_rows),
        tactile_sequence=np.asarray(sequence_rows, dtype=np.int64),
        tactile_timestamp_s=np.asarray(timestamp_rows, dtype=np.float64),
        tactile_dt_s=np.asarray(dt_rows, dtype=np.float64),
        patch_names=np.asarray(patch_names),
        raw_count=raw_count,
        raw_contact_kind=raw_contact_kind,
        raw_patch=raw_patch,
        raw_counterpart_particle=raw_counterpart_particle,
        raw_point_world_m=raw_point_world,
        raw_force_world_n=raw_force_world,
        raw_force_patch_n=raw_force_patch,
        raw_penetration_m=raw_penetration,
    )
    summary = {
        "schema": "newton_vbd_official_franka_soft_duck_native_tactile_v1",
        "frames": rendered_frames,
        "source_frame_interval": [args.frame_start, frame_stop],
        "patch_names": patch_names,
        "contact_frames_per_patch": contact_frames.tolist(),
        "frames_with_any_finger_contact": int(np.count_nonzero(raw_count)),
        "maximum_force_conservation_residual_n": max_residual,
        "native_force_source": "SolverVBD particle-rigid solved penalty+damping+friction force",
        "world_renderer": "VTK offscreen rendering of exact Newton model geometry and live state",
        "optical_available": False,
        "training": False,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "frames.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    video = output_root / "newton_softbody_franka_native_tactile.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-framerate",
            str(args.fps),
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
