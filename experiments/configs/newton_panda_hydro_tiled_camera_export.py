#!/usr/bin/env python3
"""Export real SensorTiledCamera views from official Newton Panda hydro.

This diagnostic uses Newton's official Panda hydro example plus the official
SensorTiledCamera path. Outputs are namespaced as `newton.camera.*`; no T-Rex
image keys or fake tensors are created.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

import newton
import newton.viewer
from newton.examples.robot.example_robot_panda_hydro import Example
from newton.sensors import SensorTiledCamera


CAMERA_NAMES = ("head_proxy", "right_wrist_proxy", "left_wrist_proxy")
TRACKED_OBJECTS = ("official_object", "existing_cup_asset")


def _look_at_transform(position: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack([right, up, -forward])
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return position.astype(np.float32), quat_xyzw.astype(np.float32)


def _camera_transforms(step: int, num_steps: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict]]:
    # Fixed world-space proxy cameras around the Panda hydro table scene.
    target = np.asarray([-0.18, -0.50, 0.28], dtype=np.float32)
    phase = step / max(1, num_steps - 1)
    head_pos = np.asarray([0.55, -1.25, 0.72], dtype=np.float32)
    right_pos = np.asarray([0.22, -0.86, 0.42 + 0.04 * math.sin(math.pi * phase)], dtype=np.float32)
    left_pos = np.asarray([-0.58, -0.86, 0.48 + 0.04 * math.cos(math.pi * phase)], dtype=np.float32)
    positions = (head_pos, right_pos, left_pos)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    transforms = [_look_at_transform(pos, target, up) for pos in positions]
    meta = [
        {"name": name, "position": pos.tolist(), "target": target.tolist(), "quat_xyzw": quat.tolist()}
        for name, (pos, quat) in zip(CAMERA_NAMES, transforms, strict=True)
    ]
    return transforms, meta


def _wp_camera_array(transforms: list[tuple[np.ndarray, np.ndarray]], world_count: int):
    import warp as wp

    rows = []
    for pos, quat in transforms:
        transform = wp.transformf(
            wp.vec3f(float(pos[0]), float(pos[1]), float(pos[2])),
            wp.quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        )
        rows.append([transform] * world_count)
    return wp.array(rows, dtype=wp.transformf)


def _save_triptych(images: np.ndarray, names: tuple[str, ...], output: Path, title: str) -> None:
    panels = []
    for image, name in zip(images, names, strict=True):
        rgb = image[..., :3]
        panel = Image.fromarray(rgb, mode="RGB")
        canvas = Image.new("RGB", (panel.width, panel.height + 24), "white")
        canvas.paste(panel, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, panel.height + 5), name, fill=(0, 0, 0))
        panels.append(canvas)
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels) + 32
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 32))
        x += panel.width
    sheet.save(output)


def _write_browser(frame_paths: list[Path], output: Path) -> None:
    names = [path.name for path in frame_paths]
    options = "\n".join(f'      "{name}",' for name in names)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Newton Panda Hydro Camera Browser</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f7f7f7; color: #111; }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    img {{ width: 100%; max-height: 78vh; object-fit: contain; background: white; border: 1px solid #ccc; }}
    input[type="range"] {{ width: 100%; margin: 16px 0; }}
    code {{ background: #eee; padding: 2px 4px; }}
  </style>
</head>
<body>
  <main>
    <h1>Newton Panda Hydro Camera Browser</h1>
    <div>Frame: <code id="label"></code> / Total: <code>{len(names)}</code></div>
    <input id="slider" type="range" min="0" max="{max(len(names) - 1, 0)}" value="0" step="1">
    <img id="frame" src="{names[0] if names else ''}" alt="Newton camera frame">
  </main>
  <script>
    const frames = [
{options}
    ];
    const slider = document.getElementById("slider");
    const img = document.getElementById("frame");
    const label = document.getElementById("label");
    function setFrame(i) {{
      img.src = frames[i];
      label.textContent = frames[i];
    }}
    slider.addEventListener("input", () => setFrame(Number(slider.value)));
    setFrame(0);
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def _write_contact_sheet(frame_paths: list[Path], output: Path, cols: int = 3) -> None:
    thumbs = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((480, 220), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (500, 250), "white")
        canvas.paste(img, ((500 - img.width) // 2, 0))
        ImageDraw.Draw(canvas).text((8, 226), path.stem, fill=(0, 0, 0))
        thumbs.append(canvas)
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 500, rows * 250), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 500, (i // cols) * 250))
    sheet.save(output)


def _summary(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std()) if arr.size else None,
        "nonzero": int(np.count_nonzero(arr)) if arr.size else 0,
    }


def _as_np(value) -> np.ndarray:
    arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    return np.asarray(arr).copy()


def _object_pose(example: Example) -> np.ndarray:
    body_q = example.state_0.body_q.numpy()
    poses = []
    for world_idx in range(example.world_count):
        object_body_idx = world_idx * example.bodies_per_world + example.object_body_local
        poses.append(body_q[object_body_idx].copy())
    return np.asarray(poses, dtype=np.float32)


def _ee_pose(example: Example) -> np.ndarray:
    body_q = example.state_0.body_q.numpy()
    poses = []
    for world_idx in range(example.world_count):
        ee_body_idx = world_idx * example.bodies_per_world + example.ee_index
        poses.append(body_q[ee_body_idx].copy())
    return np.asarray(poses, dtype=np.float32)


def _find_local_body_index(example: Example, label_suffix: str) -> int:
    for idx, label in enumerate(example.model_single.body_label):
        if label == label_suffix or label.endswith(f"/{label_suffix}"):
            return idx
    raise ValueError(f"Could not find local body label ending with {label_suffix!r}")


def _retarget_existing_cup_as_object(example: Example) -> dict:
    """Track and lift the cup body already created by the official example.

    The official Panda hydro cube scene loads `manipulation_objects/cup` as the
    placement target. For the Phase 01 cup-asset gate we reuse that official
    asset and retarget the example's object bookkeeping and IK waypoints to the
    cup body. This is a scene adapter only; it does not create T-Rex fields or a
    learned model.
    """

    cup_body_local = _find_local_body_index(example, "cup")
    original_object_body_local = int(example.object_body_local)
    original_object_pos = list(example.object_pos)
    example.object_body_local = cup_body_local
    example.object_pos = list(example.cup_pos)
    example.grasping_offset = [0.0, 0.0, 0.18]
    example.place_offset = 0.0
    example.put_in_cup = False
    example.object_max_z = [example.object_pos[2]] * example.world_count if example.test_mode else None
    example.setup_ik()
    example.capture_ik()
    return {
        "adapter": "retarget_existing_official_cup_asset_as_object",
        "original_object_body_local": original_object_body_local,
        "original_object_pos": original_object_pos,
        "cup_body_local": cup_body_local,
        "cup_pos": list(example.cup_pos),
        "grasping_offset": list(example.grasping_offset),
        "body_label": example.model_single.body_label[cup_body_local],
        "put_in_cup_after_retarget": bool(example.put_in_cup),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=240)
    parser.add_argument("--sample-steps", type=str, default="0,60,120,180,239")
    parser.add_argument("--scene", choices=["pen", "cube"], default="cube")
    parser.add_argument("--tracked-object", choices=TRACKED_OBJECTS, default="official_object")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    args_in = parser.parse_args()

    import warp as wp

    wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)
    wp.set_device("cuda:0")

    viewer = newton.viewer.ViewerNull(num_frames=args_in.num_steps, benchmark=False)
    example_args = SimpleNamespace(
        device="cuda:0",
        viewer="null",
        output_path="None",
        num_frames=args_in.num_steps,
        render_fps=None,
        headless=True,
        test=False,
        quiet=True,
        paused=False,
        benchmark=False,
        warp_config=[],
        realtime=False,
        world_count=1,
        scene=args_in.scene,
    )
    example = Example(viewer, example_args)
    object_adapter_meta = {
        "adapter": "official_example_default_object",
        "body_label": example.model_single.body_label[int(example.object_body_local)],
        "object_body_local": int(example.object_body_local),
        "object_pos": list(example.object_pos),
    }
    if args_in.tracked_object == "existing_cup_asset":
        if args_in.scene != "cube":
            raise ValueError("existing_cup_asset gate currently requires --scene cube so the official cup asset is loaded")
        object_adapter_meta = _retarget_existing_cup_as_object(example)

    sensor = SensorTiledCamera(model=example.model)
    sensor.utils.create_default_light(enable_shadows=True)
    camera_count = len(CAMERA_NAMES)
    camera_rays = sensor.utils.compute_pinhole_camera_rays(
        args_in.width,
        args_in.height,
        [math.radians(55.0), math.radians(65.0), math.radians(65.0)],
    )
    color = sensor.utils.create_color_image_output(args_in.width, args_in.height, camera_count)
    depth = sensor.utils.create_depth_image_output(args_in.width, args_in.height, camera_count)

    requested = sorted({int(x) for x in args_in.sample_steps.split(",") if x.strip()})
    requested = [min(max(step, 0), args_in.num_steps - 1) for step in requested]

    args_in.output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    color_records = []
    depth_records = []
    camera_meta_records = []
    object_z_records = []
    rollout_records = {
        "step": [],
        "sim_time": [],
        "joint_q": [],
        "joint_qd": [],
        "joint_target_q": [],
        "object_body_q": [],
        "ee_body_q": [],
        "rigid_contact_count": [],
    }

    for step in range(args_in.num_steps):
        example.step()
        rollout_records["step"].append(step + 1)
        rollout_records["sim_time"].append(float(example.sim_time))
        rollout_records["joint_q"].append(_as_np(example.state_0.joint_q))
        rollout_records["joint_qd"].append(_as_np(example.state_0.joint_qd))
        rollout_records["joint_target_q"].append(_as_np(example.control.joint_target_q))
        rollout_records["object_body_q"].append(_object_pose(example))
        rollout_records["ee_body_q"].append(_ee_pose(example))
        rollout_records["rigid_contact_count"].append(_as_np(example.contacts.rigid_contact_count))
        if step not in requested:
            continue
        example.model.bvh_refit_shapes(example.state_0)
        example.model.bvh_refit_particles(example.state_0)
        transforms, camera_meta = _camera_transforms(step, args_in.num_steps)
        sensor.update(
            example.state_0,
            _wp_camera_array(transforms, example.world_count),
            camera_rays,
            color_image=color,
            depth_image=depth,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        rgba = sensor.utils.to_rgba_from_color(color).numpy().copy()
        depth_np = depth.numpy().copy()
        png = args_in.output_dir / f"frame_{step:04d}.png"
        _save_triptych(rgba, CAMERA_NAMES, png, title=f"step {step}")
        frame_paths.append(png)
        color_records.append(rgba)
        depth_records.append(depth_np)
        camera_meta_records.append({"step": step, "cameras": camera_meta})
        body_q = example.state_0.body_q.numpy()
        object_body_idx = example.object_body_local
        object_z_records.append(float(body_q[object_body_idx][2]))

    _write_browser(frame_paths, args_in.output_dir / "frame_browser.html")
    _write_contact_sheet(frame_paths, args_in.output_dir / "contact_sheet.png")

    color_arr = np.asarray(color_records, dtype=np.uint8)
    depth_arr = np.asarray(depth_records, dtype=np.float32)
    object_z_arr = np.asarray(object_z_records, dtype=np.float32)
    rollout_arrays = {
        "newton.panda.step": np.asarray(rollout_records["step"], dtype=np.int32),
        "newton.panda.sim_time": np.asarray(rollout_records["sim_time"], dtype=np.float32),
        "newton.panda.joint_q": np.asarray(rollout_records["joint_q"], dtype=np.float32),
        "newton.panda.joint_qd": np.asarray(rollout_records["joint_qd"], dtype=np.float32),
        "newton.panda.joint_target_q": np.asarray(rollout_records["joint_target_q"], dtype=np.float32),
        "newton.panda.object_body_q": np.asarray(rollout_records["object_body_q"], dtype=np.float32),
        "newton.panda.ee_body_q": np.asarray(rollout_records["ee_body_q"], dtype=np.float32),
        "newton.panda.rigid_contact_count": np.asarray(rollout_records["rigid_contact_count"], dtype=np.int32),
    }
    args_in.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_in.npz,
        **{
            "newton.camera.color_rgba": color_arr,
            "newton.camera.depth": depth_arr,
            "newton.camera.object_z": object_z_arr,
            **rollout_arrays,
        },
    )
    full_object_z = rollout_arrays["newton.panda.object_body_q"][:, :, 2]
    initial_z = full_object_z[0]
    max_lift = full_object_z.max(axis=0) - initial_z

    summary = {
        "status": "pass",
        "classification": "official_newton_sensor_tiled_camera_export_not_trex_schema",
        "note": "Real SensorTiledCamera output from official Panda hydro. No T-Rex image keys are created.",
        "newton_version": getattr(newton, "__version__", "unknown"),
        "device": str(wp.get_device()),
        "scene": args_in.scene,
        "tracked_object": args_in.tracked_object,
        "object_adapter": object_adapter_meta,
        "num_steps": args_in.num_steps,
        "sample_steps": requested,
        "camera_names": list(CAMERA_NAMES),
        "width": args_in.width,
        "height": args_in.height,
        "output_dir": str(args_in.output_dir),
        "frame_browser": str(args_in.output_dir / "frame_browser.html"),
        "contact_sheet": str(args_in.output_dir / "contact_sheet.png"),
        "npz": str(args_in.npz),
        "camera_meta": camera_meta_records,
        "array_summaries": {
            "newton.camera.color_rgba": _summary(color_arr),
            "newton.camera.depth": _summary(depth_arr),
            "newton.camera.object_z": _summary(object_z_arr),
            **{key: _summary(value) for key, value in rollout_arrays.items()},
        },
        "initial_object_z": [float(x) for x in initial_z],
        "final_object_z": [float(x) for x in full_object_z[-1]],
        "max_object_z": [float(x) for x in full_object_z.max(axis=0)],
        "max_lift": [float(x) for x in max_lift],
        "trex_missing_by_design": [
            "observation.state[62]",
            "action[16,62]",
            "action_abs[62]",
            "observation.tactile_f6[10,6]",
            "ten tactile_deform streams",
        ],
    }
    args_in.summary.parent.mkdir(parents=True, exist_ok=True)
    args_in.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args_in.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "pass", "summary": str(args_in.summary)}, indent=2))
    viewer.close()


if __name__ == "__main__":
    main()
