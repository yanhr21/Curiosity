#!/usr/bin/env python3
"""Export real SensorTiledCamera views from official Newton Panda hydro.

This diagnostic uses Newton's official Panda hydro example plus the official
SensorTiledCamera path. Outputs are namespaced as `newton.camera.*`; no T-Rex
image keys or fake tensors are created.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
CONTROLLER_MODES = ("official_pick_place", "lift_hold", "lift_hold_feedback", "lift_hold_learned_residual")


def _controller_phase(example: Example) -> tuple[int, str]:
    idx = int(example.current_waypoint)
    labels = (
        "approach_rest",
        "approach_pre_grasp",
        "close_gripper",
        "lift_to_rest",
        "transport_above_cup",
        "loosen_gripper",
        "recover_grip",
        "lower_to_place",
        "release",
    )
    if idx < len(labels):
        return idx, labels[idx]
    return idx, f"waypoint_{idx}"


def _commanded_gripper_target(example: Example) -> float:
    if example.waypoints:
        t = example.time_in_waypoint / max(float(example.waypoints[example.current_waypoint][1]), 1e-6)
        next_waypoint = (example.current_waypoint + 1) % len(example.waypoints)
        t_gripper = example.waypoints[example.current_waypoint][2] * (1 - t) + example.waypoints[next_waypoint][2] * t
        return float(0.06 * (1 - t_gripper))
    return float("nan")


def _commanded_lift_target(example: Example) -> float:
    if example.waypoints:
        target_position = example.waypoints[example.current_waypoint][0]
        return float(target_position[2])
    return float("nan")


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


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _task_metrics(
    full_object_q: np.ndarray,
    fps: float,
    lift_height_min: float,
    hold_duration_min: float,
    drop_height_loss: float,
) -> dict:
    xyz = full_object_q[:, :, :3]
    initial_xyz = xyz[0]
    final_xyz = xyz[-1]
    z = xyz[:, :, 2]
    initial_z = initial_xyz[:, 2]
    max_z = z.max(axis=0)
    final_z = final_xyz[:, 2]
    threshold_z = initial_z + lift_height_min
    per_world = []
    success_flags = []
    for world_idx in range(z.shape[1]):
        lifted = z[:, world_idx] >= threshold_z[world_idx]
        longest_hold_frames = _longest_true_run(lifted)
        longest_hold_s = longest_hold_frames / fps
        max_lift = max_z[world_idx] - initial_z[world_idx]
        final_lift = final_z[world_idx] - initial_z[world_idx]
        drop_from_max = max_z[world_idx] - final_z[world_idx]
        xy_drift = np.linalg.norm(xyz[:, world_idx, :2] - initial_xyz[world_idx, :2], axis=1)
        failures = []
        if max_lift < lift_height_min:
            failures.append("lift_height_below_min")
        if longest_hold_s < hold_duration_min:
            failures.append("hold_duration_below_min")
        if drop_from_max > drop_height_loss:
            failures.append("drop_height_loss_above_threshold")
        success = not failures
        success_flags.append(success)
        per_world.append(
            {
                "world_idx": int(world_idx),
                "success": bool(success),
                "failure_reasons": failures,
                "initial_z": float(initial_z[world_idx]),
                "max_z": float(max_z[world_idx]),
                "final_z": float(final_z[world_idx]),
                "max_lift": float(max_lift),
                "final_lift": float(final_lift),
                "drop_from_max": float(drop_from_max),
                "lift_threshold_z": float(threshold_z[world_idx]),
                "longest_hold_frames": int(longest_hold_frames),
                "longest_hold_s": float(longest_hold_s),
                "max_xy_drift": float(xy_drift.max()),
                "final_xy_drift": float(xy_drift[-1]),
            }
        )
    return {
        "classification": "newton_native_lift_hold_success_metrics",
        "fps": float(fps),
        "thresholds": {
            "lift_height_m_min": float(lift_height_min),
            "hold_duration_s_min": float(hold_duration_min),
            "drop_height_loss_m": float(drop_height_loss),
        },
        "success_all_worlds": bool(all(success_flags)),
        "per_world": per_world,
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


def _find_builder_body_index(builder, label_suffix: str) -> int:
    for idx, label in enumerate(builder.body_label):
        if label == label_suffix or label.endswith(f"/{label_suffix}"):
            return idx
    raise ValueError(f"Could not find builder body label ending with {label_suffix!r}")


def _find_final_model_body_index(example: Example, label_suffix: str) -> int:
    for idx, label in enumerate(example.model.body_label):
        if label == label_suffix or label.endswith(f"/{label_suffix}"):
            return idx
    raise ValueError(f"Could not find final model body label ending with {label_suffix!r}")


def _scale_mat33(value, scale: float):
    import warp as wp

    return wp.mat33(np.asarray(value, dtype=np.float32) * float(scale))


def _inv_mat33_or_zero(value):
    import warp as wp

    return wp.inverse(value) if np.any(np.asarray(value, dtype=np.float32)) else value


def _apply_builder_physics_variant(
    builder,
    *,
    target_label_suffix: str,
    physics_variant_label: str,
    body_mass_scale: float,
    shape_friction_scale: float,
    object_mass_kg: float | None,
    object_friction_mu: float | None,
) -> dict:
    if body_mass_scale <= 0.0:
        raise ValueError(f"body_mass_scale must be positive, got {body_mass_scale}")
    if shape_friction_scale < 0.0:
        raise ValueError(f"shape_friction_scale must be nonnegative, got {shape_friction_scale}")
    if object_mass_kg is not None and object_mass_kg <= 0.0:
        raise ValueError(f"object_mass_kg must be positive, got {object_mass_kg}")
    if object_friction_mu is not None and object_friction_mu < 0.0:
        raise ValueError(f"object_friction_mu must be nonnegative, got {object_friction_mu}")

    body_idx = _find_builder_body_index(builder, target_label_suffix)
    shape_indices = list(builder.body_shapes.get(body_idx, []))
    original_body_mass = float(builder.body_mass[body_idx])
    original_body_inv_mass = float(builder.body_inv_mass[body_idx])
    original_shape_mu = {str(idx): float(builder.shape_material_mu[idx]) for idx in shape_indices}
    requested = {
        "physics_variant_label": physics_variant_label,
        "target_label_suffix": target_label_suffix,
        "body_mass_scale": float(body_mass_scale),
        "shape_friction_scale": float(shape_friction_scale),
        "object_mass_kg": None if object_mass_kg is None else float(object_mass_kg),
        "object_friction_mu": None if object_friction_mu is None else float(object_friction_mu),
    }

    should_update_mass = object_mass_kg is not None or not np.isclose(body_mass_scale, 1.0)
    should_update_friction = object_friction_mu is not None or not np.isclose(shape_friction_scale, 1.0)
    if should_update_mass:
        if original_body_mass <= 0.0:
            raise ValueError(f"cannot scale non-dynamic body {body_idx} with mass {original_body_mass}")
        new_mass = float(object_mass_kg) if object_mass_kg is not None else original_body_mass * float(body_mass_scale)
        mass_ratio = new_mass / original_body_mass
        builder.body_mass[body_idx] = new_mass
        builder.body_inv_mass[body_idx] = 1.0 / new_mass
        builder.body_inertia[body_idx] = _scale_mat33(builder.body_inertia[body_idx], mass_ratio)
        builder.body_inv_inertia[body_idx] = _inv_mat33_or_zero(builder.body_inertia[body_idx])

    if should_update_friction:
        for shape_idx in shape_indices:
            builder.shape_material_mu[shape_idx] = (
                float(object_friction_mu)
                if object_friction_mu is not None
                else float(builder.shape_material_mu[shape_idx]) * float(shape_friction_scale)
            )

    return {
        "adapter": "pre_finalize_builder_body_mass_inertia_and_shape_friction",
        "applied": bool(should_update_mass or should_update_friction),
        "builder_stage": "before_scene_replicate_and_before_final_model_finalize",
        "requested": requested,
        "body_label": builder.body_label[body_idx],
        "body_index_local": int(body_idx),
        "shape_indices_local": [int(idx) for idx in shape_indices],
        "original_body_mass_kg": original_body_mass,
        "original_body_inv_mass": original_body_inv_mass,
        "updated_body_mass_kg": float(builder.body_mass[body_idx]),
        "updated_body_inv_mass": float(builder.body_inv_mass[body_idx]),
        "original_shape_material_mu": original_shape_mu,
        "updated_shape_material_mu": {str(idx): float(builder.shape_material_mu[idx]) for idx in shape_indices},
        "source_namespace": "candidate.physics.*",
        "generated_trex_fields": [],
        "schema_promotion": "blocked",
        "learned_policy": False,
    }


@contextmanager
def _pre_finalize_physics_variant_context(
    *,
    tracked_object: str,
    physics_variant_label: str,
    body_mass_scale: float,
    shape_friction_scale: float,
    object_mass_kg: float | None,
    object_friction_mu: float | None,
):
    should_apply = (
        object_mass_kg is not None
        or object_friction_mu is not None
        or not np.isclose(body_mass_scale, 1.0)
        or not np.isclose(shape_friction_scale, 1.0)
    )
    target_label_suffix = "cup" if tracked_object == "existing_cup_asset" else "object"
    meta = {
        "adapter": "pre_finalize_builder_body_mass_inertia_and_shape_friction",
        "applied": False,
        "builder_stage": "not_requested",
        "requested": {
            "physics_variant_label": physics_variant_label,
            "target_label_suffix": target_label_suffix,
            "body_mass_scale": float(body_mass_scale),
            "shape_friction_scale": float(shape_friction_scale),
            "object_mass_kg": None if object_mass_kg is None else float(object_mass_kg),
            "object_friction_mu": None if object_friction_mu is None else float(object_friction_mu),
        },
        "body_index_local": None,
        "shape_indices_local": [],
        "source_namespace": "candidate.physics.*",
        "generated_trex_fields": [],
        "schema_promotion": "blocked",
        "learned_policy": False,
    }
    if not should_apply:
        yield meta
        return

    original_replicate = newton.ModelBuilder.replicate
    holder = {"meta": meta}

    def wrapped_replicate(scene_builder, template_builder, *args, **kwargs):
        if not holder["meta"].get("applied", False):
            holder["meta"] = _apply_builder_physics_variant(
                template_builder,
                target_label_suffix=target_label_suffix,
                physics_variant_label=physics_variant_label,
                body_mass_scale=body_mass_scale,
                shape_friction_scale=shape_friction_scale,
                object_mass_kg=object_mass_kg,
                object_friction_mu=object_friction_mu,
            )
        return original_replicate(scene_builder, template_builder, *args, **kwargs)

    newton.ModelBuilder.replicate = wrapped_replicate
    try:
        yield holder
    finally:
        newton.ModelBuilder.replicate = original_replicate


def _observed_physics_from_model(example: Example, adapter_meta: dict) -> dict:
    body_label = adapter_meta.get("body_label")
    if body_label is None:
        return {}
    body_idx = _find_final_model_body_index(example, str(body_label).split("/")[-1])
    shape_indices = list(example.model.body_shapes.get(int(body_idx), []))
    body_mass = example.model.body_mass.numpy().copy()
    body_inv_mass = example.model.body_inv_mass.numpy().copy()
    shape_mu = example.model.shape_material_mu.numpy().copy()
    return {
        "observed_body_index_final_model": int(body_idx),
        "observed_shape_indices_final_model": [int(idx) for idx in shape_indices],
        "observed_body_mass_kg": {str(body_idx): float(body_mass[int(body_idx)])},
        "observed_body_inv_mass": {str(body_idx): float(body_inv_mass[int(body_idx)])},
        "observed_shape_material_mu": {str(idx): float(shape_mu[int(idx)]) for idx in shape_indices},
    }


def _retarget_existing_cup_as_object(example: Example, final_hold_duration: float) -> dict:
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
    if final_hold_duration > 0.0 and example.waypoints:
        example.waypoints[-1][1] = float(final_hold_duration)
    example.capture_ik()
    return {
        "adapter": "retarget_existing_official_cup_asset_as_object",
        "original_object_body_local": original_object_body_local,
        "original_object_pos": original_object_pos,
        "cup_body_local": cup_body_local,
        "cup_pos": list(example.cup_pos),
        "grasping_offset": list(example.grasping_offset),
        "final_hold_duration": float(final_hold_duration),
        "body_label": example.model_single.body_label[cup_body_local],
        "put_in_cup_after_retarget": bool(example.put_in_cup),
    }


def _configure_lift_hold_waypoints(example: Example, hold_duration: float) -> dict:
    """Keep the official approach/grasp/lift prior but hold instead of placing."""

    if len(example.waypoints) < 4:
        raise ValueError("official Panda hydro example did not create enough waypoints for lift-hold mode")
    lift_wp = list(example.waypoints[3])
    hold_wp = [lift_wp[0], float(hold_duration), lift_wp[2], lift_wp[3]]
    guard_wp = [lift_wp[0], 999.0, lift_wp[2], lift_wp[3]]
    original_waypoint_count = len(example.waypoints)
    example.waypoints = [list(example.waypoints[0]), list(example.waypoints[1]), list(example.waypoints[2]), hold_wp, guard_wp]
    example.current_waypoint = 0
    example.time_in_waypoint = 0.0
    example.capture_ik()
    return {
        "adapter": "official_panda_hydro_waypoints_lift_hold_no_release",
        "original_waypoint_count": int(original_waypoint_count),
        "new_waypoint_count": int(len(example.waypoints)),
        "hold_duration_s": float(hold_duration),
        "guard_duration_s": 999.0,
        "learned_policy": False,
        "feedback_adaptation": False,
    }


def _configure_lift_hold_feedback_waypoints(
    example: Example,
    hold_duration: float,
    lift_duration_scale: float,
    stabilization_duration: float,
) -> dict:
    meta = _configure_lift_hold_waypoints(example, hold_duration)
    if lift_duration_scale <= 0.0:
        raise ValueError("feedback lift duration scale must be positive")
    if stabilization_duration < 0.0:
        raise ValueError("feedback stabilization duration must be nonnegative")
    example.waypoints[2][1] = float(example.waypoints[2][1]) * float(lift_duration_scale)
    example.waypoints[3][1] = float(hold_duration) + float(stabilization_duration)
    example.capture_ik()
    return {
        **meta,
        "adapter": "official_panda_hydro_waypoints_lift_hold_scripted_feedback",
        "feedback_adaptation": True,
        "feedback_type": "scripted_contact_object_motion_controller",
        "initial_lift_duration_scale": float(lift_duration_scale),
        "initial_stabilization_duration_s": float(stabilization_duration),
        "learned_policy": False,
        "curiosity_reward": "none",
    }


def _feedback_state() -> dict:
    return {
        "prev_object_z": None,
        "prev_object_vz": None,
        "lift_velocity_scale": 1.0,
        "hold_height_offset_m": 0.0,
        "applied_hold_height_offset_m": 0.0,
        "stabilization_extension_s": 0.0,
        "trigger_count": 0,
        "active_reason": "none",
        "original_lift_duration_s": None,
        "base_hold_duration_s": None,
    }


def _load_residual_adapter(checkpoint_path: Path, active_threshold: float):
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on compute env
        raise RuntimeError(
            "learned residual adapter evaluation requires torch in the active export venv"
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("classification") != "newton_native_residual_controller_adapter_v1_checkpoint":
        raise ValueError(f"unexpected residual adapter checkpoint classification: {checkpoint.get('classification')}")
    target_columns = list(checkpoint["target_columns"])
    feature_columns = list(checkpoint["feature_columns"])
    config = checkpoint["config"]
    architecture = config["architecture"]

    class ResidualControllerAdapter(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, gru_layers: int) -> None:
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=gru_layers, batch_first=True)
            self.active_head = nn.Linear(hidden_dim, 1)
            self.continuous_head = nn.Linear(hidden_dim, len(target_columns) - 1)

        def forward(self, features, hidden=None):
            output, hidden = self.gru(self.input_norm(features), hidden)
            return self.active_head(output), self.continuous_head(output), hidden

    model = ResidualControllerAdapter(
        input_dim=int(architecture["input_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        gru_layers=int(architecture["gru_layers"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32)
    feature_std = torch.tensor(checkpoint["feature_std"], dtype=torch.float32)
    continuous_mean = torch.tensor(checkpoint["continuous_mean"], dtype=torch.float32)
    continuous_std = torch.tensor(checkpoint["continuous_std"], dtype=torch.float32)

    return {
        "torch": torch,
        "model": model,
        "hidden": None,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "continuous_mean": continuous_mean,
        "continuous_std": continuous_std,
        "active_threshold": float(active_threshold),
        "checkpoint_classification": checkpoint["classification"],
        "checkpoint_config_method": config.get("method_name"),
    }


def _apply_learned_residual_adapter(
    example: Example,
    adapter: dict,
    state: dict,
    feature_values: dict[str, float],
    lift_duration_scale_max: float,
    hold_height_offset_max: float,
    stabilization_max: float,
) -> dict:
    import warp as wp

    torch = adapter["torch"]
    ordered_features = [float(feature_values[column]) for column in adapter["feature_columns"]]
    features = torch.tensor(ordered_features, dtype=torch.float32).view(1, 1, -1)
    features = (features - adapter["feature_mean"].view(1, 1, -1)) / adapter["feature_std"].view(1, 1, -1)
    with torch.no_grad():
        active_logits, continuous_pred, adapter["hidden"] = adapter["model"](features, adapter["hidden"])
        active_prob = float(torch.sigmoid(active_logits[0, 0, 0]).cpu())
        continuous = continuous_pred[0, 0].cpu() * adapter["continuous_std"] + adapter["continuous_mean"]
    values = [float(x) for x in continuous.tolist()]
    active = active_prob >= float(adapter["active_threshold"])

    lift_velocity_scale = float(np.clip(values[0], 0.35, 1.0))
    hold_height_offset_m = float(np.clip(values[1], -abs(hold_height_offset_max), abs(hold_height_offset_max)))
    stabilization_extension_s = float(np.clip(values[2], 0.0, abs(stabilization_max)))

    if state["original_lift_duration_s"] is None:
        state["original_lift_duration_s"] = float(example.waypoints[2][1])
    if state["base_hold_duration_s"] is None:
        state["base_hold_duration_s"] = float(example.waypoints[3][1])

    reason = "none"
    if active:
        reason = "learned_residual_adapter_active"
        state["trigger_count"] += 1
        state["active_reason"] = reason
        state["lift_velocity_scale"] = lift_velocity_scale
        state["hold_height_offset_m"] = hold_height_offset_m
        state["stabilization_extension_s"] = stabilization_extension_s

        lift_duration = min(
            float(state["original_lift_duration_s"]) / max(lift_velocity_scale, 1e-3),
            float(state["original_lift_duration_s"]) * float(lift_duration_scale_max),
        )
        example.waypoints[2][1] = float(max(example.waypoints[2][1], lift_duration))
        offset_delta = state["hold_height_offset_m"] - state["applied_hold_height_offset_m"]
        for idx in (3, 4):
            target_pos = list(example.waypoints[idx][0])
            target_pos[2] = float(target_pos[2]) + offset_delta
            example.waypoints[idx][0] = wp.vec3(target_pos)
        state["applied_hold_height_offset_m"] = state["hold_height_offset_m"]
        example.waypoints[3][1] = float(state["base_hold_duration_s"]) + stabilization_extension_s
        example.capture_ik()
    else:
        state["active_reason"] = "none"

    return {
        "feedback_active": int(active),
        "feedback_reason": reason,
        "feedback_lift_velocity_scale": float(state["lift_velocity_scale"]),
        "feedback_hold_height_offset_m": float(state["hold_height_offset_m"]),
        "feedback_stabilization_extension_s": float(state["stabilization_extension_s"]),
        "feedback_trigger_count": int(state["trigger_count"]),
        "feedback_observed_object_vz_m_s": float(feature_values.get("_object_vz_m_s", 0.0)),
        "feedback_observed_object_accel_m_s2": float(feature_values.get("_object_accel_m_s2", 0.0)),
        "feedback_active_probability": active_prob,
        "feedback_raw_lift_velocity_scale": values[0],
        "feedback_raw_hold_height_offset_m": values[1],
        "feedback_raw_stabilization_extension_s": values[2],
    }


def _pre_record_warmup(example: Example, warmup_steps: int) -> None:
    """Settle physics before recording without advancing scripted waypoints."""

    for _ in range(max(0, int(warmup_steps))):
        if example.graph:
            import warp as wp

            wp.capture_launch(example.graph)
        else:
            example.simulate()
        example.sim_time += example.frame_dt


def _apply_scripted_feedback(
    example: Example,
    state: dict,
    object_pose: np.ndarray,
    contact_proxy: np.ndarray,
    frame_dt: float,
    min_contact_count: int,
    accel_threshold: float,
    height_drop_threshold: float,
    lift_duration_scale_max: float,
    hold_height_step: float,
    hold_height_offset_max: float,
    stabilization_step: float,
    stabilization_max: float,
) -> dict:
    import warp as wp

    object_z = float(object_pose[0, 2])
    prev_z = state["prev_object_z"]
    prev_vz = state["prev_object_vz"]
    vz = 0.0 if prev_z is None else (object_z - float(prev_z)) / max(frame_dt, 1e-6)
    accel = 0.0 if prev_vz is None else (vz - float(prev_vz)) / max(frame_dt, 1e-6)
    contact_count = int(np.max(contact_proxy)) if np.size(contact_proxy) else 0

    reason = "none"
    if contact_count < min_contact_count and int(example.current_waypoint) >= 2:
        reason = "low_contact_count"
    elif abs(accel) > accel_threshold and int(example.current_waypoint) >= 2:
        reason = "object_acceleration_above_feedback_threshold"
    elif (
        int(example.current_waypoint) >= 3
        and prev_z is not None
        and (float(prev_z) - object_z) > height_drop_threshold
    ):
        reason = "object_height_drop"

    if reason != "none":
        if state["original_lift_duration_s"] is None:
            state["original_lift_duration_s"] = float(example.waypoints[2][1])
        state["trigger_count"] += 1
        state["active_reason"] = reason
        state["lift_velocity_scale"] = max(
            0.35,
            state["lift_velocity_scale"] * 0.92,
        )
        state["hold_height_offset_m"] = max(
            -abs(hold_height_offset_max),
            state["hold_height_offset_m"] - abs(hold_height_step),
        )
        state["stabilization_extension_s"] = min(
            stabilization_max,
            state["stabilization_extension_s"] + stabilization_step,
        )

        lift_duration = min(
            float(example.waypoints[2][1]) / 0.92,
            float(state["original_lift_duration_s"]) * lift_duration_scale_max,
        )
        example.waypoints[2][1] = float(max(example.waypoints[2][1], lift_duration))
        offset_delta = state["hold_height_offset_m"] - state["applied_hold_height_offset_m"]
        for idx in (3, 4):
            target_pos = list(example.waypoints[idx][0])
            target_pos[2] = float(target_pos[2]) + offset_delta
            example.waypoints[idx][0] = wp.vec3(target_pos)
        state["applied_hold_height_offset_m"] = state["hold_height_offset_m"]
        example.waypoints[3][1] = float(example.waypoints[3][1]) + stabilization_step
        example.capture_ik()
    else:
        state["active_reason"] = "none"

    state["prev_object_z"] = object_z
    state["prev_object_vz"] = vz
    return {
        "feedback_active": int(reason != "none"),
        "feedback_reason": reason,
        "feedback_lift_velocity_scale": float(state["lift_velocity_scale"]),
        "feedback_hold_height_offset_m": float(state["hold_height_offset_m"]),
        "feedback_stabilization_extension_s": float(state["stabilization_extension_s"]),
        "feedback_trigger_count": int(state["trigger_count"]),
        "feedback_observed_object_vz_m_s": float(vz),
        "feedback_observed_object_accel_m_s2": float(accel),
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
    parser.add_argument("--controller-mode", choices=CONTROLLER_MODES, default="official_pick_place")
    parser.add_argument("--final-hold-duration", type=float, default=1.0)
    parser.add_argument("--lift-height-min", type=float, default=0.12)
    parser.add_argument("--hold-duration-min", type=float, default=2.0)
    parser.add_argument("--drop-height-loss", type=float, default=0.05)
    parser.add_argument("--physics-variant-label", type=str, default="nominal")
    parser.add_argument("--body-mass-scale", type=float, default=1.0)
    parser.add_argument("--shape-friction-scale", type=float, default=1.0)
    parser.add_argument("--object-mass-kg", type=float, default=None)
    parser.add_argument("--object-friction-mu", type=float, default=None)
    parser.add_argument("--feedback-min-contact-count", type=int, default=20)
    parser.add_argument("--feedback-accel-threshold", type=float, default=6.5)
    parser.add_argument("--feedback-height-drop-threshold", type=float, default=0.015)
    parser.add_argument("--feedback-initial-lift-duration-scale", type=float, default=1.35)
    parser.add_argument("--feedback-lift-duration-scale-max", type=float, default=2.25)
    parser.add_argument("--feedback-hold-height-step", type=float, default=0.003)
    parser.add_argument("--feedback-hold-height-offset-max", type=float, default=0.03)
    parser.add_argument("--feedback-stabilization-step", type=float, default=0.25)
    parser.add_argument("--feedback-stabilization-max", type=float, default=2.0)
    parser.add_argument("--pre-record-warmup-steps", type=int, default=0)
    parser.add_argument("--residual-adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--residual-adapter-active-threshold", type=float, default=0.5)
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
    with _pre_finalize_physics_variant_context(
        tracked_object=args_in.tracked_object,
        physics_variant_label=args_in.physics_variant_label,
        body_mass_scale=args_in.body_mass_scale,
        shape_friction_scale=args_in.shape_friction_scale,
        object_mass_kg=args_in.object_mass_kg,
        object_friction_mu=args_in.object_friction_mu,
    ) as object_physics_adapter_holder:
        example = Example(viewer, example_args)
    object_physics_adapter_meta = object_physics_adapter_holder.get("meta", object_physics_adapter_holder)
    object_physics_adapter_meta = {
        **object_physics_adapter_meta,
        **_observed_physics_from_model(example, object_physics_adapter_meta),
    }
    object_adapter_meta = {
        "adapter": "official_example_default_object",
        "body_label": example.model_single.body_label[int(example.object_body_local)],
        "object_body_local": int(example.object_body_local),
        "object_pos": list(example.object_pos),
    }
    if args_in.tracked_object == "existing_cup_asset":
        if args_in.scene != "cube":
            raise ValueError("existing_cup_asset gate currently requires --scene cube so the official cup asset is loaded")
        object_adapter_meta = _retarget_existing_cup_as_object(example, args_in.final_hold_duration)
    controller_adapter_meta = {"adapter": "official_panda_hydro_waypoints_unmodified"}
    if args_in.controller_mode == "lift_hold":
        controller_adapter_meta = _configure_lift_hold_waypoints(example, args_in.final_hold_duration)
    if args_in.controller_mode in {"lift_hold_feedback", "lift_hold_learned_residual"}:
        controller_adapter_meta = _configure_lift_hold_feedback_waypoints(
            example,
            args_in.final_hold_duration,
            args_in.feedback_initial_lift_duration_scale,
            args_in.feedback_stabilization_step,
        )
    residual_adapter = None
    if args_in.controller_mode == "lift_hold_learned_residual":
        if args_in.residual_adapter_checkpoint is None:
            raise ValueError("lift_hold_learned_residual requires --residual-adapter-checkpoint")
        residual_adapter = _load_residual_adapter(
            args_in.residual_adapter_checkpoint,
            active_threshold=args_in.residual_adapter_active_threshold,
        )
        controller_adapter_meta = {
            **controller_adapter_meta,
            "adapter": "official_panda_hydro_waypoints_lift_hold_learned_residual",
            "feedback_adaptation": True,
            "feedback_type": "learned_newton_native_residual_controller_adapter",
            "learned_policy": True,
            "checkpoint": str(args_in.residual_adapter_checkpoint),
            "checkpoint_classification": residual_adapter["checkpoint_classification"],
            "checkpoint_config_method": residual_adapter["checkpoint_config_method"],
            "active_threshold": float(args_in.residual_adapter_active_threshold),
            "not_official_trex_method": True,
            "not_trex_schema": True,
        }

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
        "controller_phase_index": [],
        "commanded_gripper_target": [],
        "commanded_lift_target": [],
        "feedback_active": [],
        "feedback_reason_id": [],
        "feedback_lift_velocity_scale": [],
        "feedback_hold_height_offset_m": [],
        "feedback_stabilization_extension_s": [],
        "feedback_trigger_count": [],
        "feedback_observed_object_vz_m_s": [],
        "feedback_observed_object_accel_m_s2": [],
        "feedback_active_probability": [],
        "feedback_raw_lift_velocity_scale": [],
        "feedback_raw_hold_height_offset_m": [],
        "feedback_raw_stabilization_extension_s": [],
    }
    controller_phase_labels = {}
    feedback_state = _feedback_state()
    feedback_reason_labels = {"0": "none"}
    feedback_reason_to_id = {"none": 0}

    if args_in.pre_record_warmup_steps > 0:
        _pre_record_warmup(example, args_in.pre_record_warmup_steps)

    for step in range(args_in.num_steps):
        example.step()
        object_pose = _object_pose(example)
        contact_proxy = _as_np(example.contacts.rigid_contact_count)
        phase_idx, phase_label = _controller_phase(example)
        controller_phase_labels[str(phase_idx)] = phase_label
        if args_in.controller_mode == "lift_hold_feedback":
            feedback_record = _apply_scripted_feedback(
                example=example,
                state=feedback_state,
                object_pose=object_pose,
                contact_proxy=contact_proxy,
                frame_dt=float(example.frame_dt),
                min_contact_count=args_in.feedback_min_contact_count,
                accel_threshold=args_in.feedback_accel_threshold,
                height_drop_threshold=args_in.feedback_height_drop_threshold,
                lift_duration_scale_max=args_in.feedback_lift_duration_scale_max,
                hold_height_step=args_in.feedback_hold_height_step,
                hold_height_offset_max=args_in.feedback_hold_height_offset_max,
                stabilization_step=args_in.feedback_stabilization_step,
                stabilization_max=args_in.feedback_stabilization_max,
            )
        elif args_in.controller_mode == "lift_hold_learned_residual":
            object_z = float(object_pose[0, 2])
            prev_z = feedback_state["prev_object_z"]
            prev_vz = feedback_state["prev_object_vz"]
            vz = 0.0 if prev_z is None else (object_z - float(prev_z)) / max(float(example.frame_dt), 1e-6)
            accel = 0.0 if prev_vz is None else (vz - float(prev_vz)) / max(float(example.frame_dt), 1e-6)
            feedback_state["prev_object_z"] = object_z
            feedback_state["prev_object_vz"] = vz
            assert residual_adapter is not None
            feedback_record = _apply_learned_residual_adapter(
                example=example,
                adapter=residual_adapter,
                state=feedback_state,
                feature_values={
                    "newton.panda.sim_time": float(example.sim_time),
                    "newton.contact.rigid_contact_count": float(int(np.max(contact_proxy)) if np.size(contact_proxy) else 0),
                    "newton.object.body_q.z": object_z,
                    "candidate.controller.phase_index": float(phase_idx),
                    "candidate.controller.commanded_gripper_target": float(_commanded_gripper_target(example)),
                    "candidate.controller.commanded_lift_target": float(_commanded_lift_target(example)),
                    "_object_vz_m_s": vz,
                    "_object_accel_m_s2": accel,
                },
                lift_duration_scale_max=args_in.feedback_lift_duration_scale_max,
                hold_height_offset_max=args_in.feedback_hold_height_offset_max,
                stabilization_max=args_in.feedback_stabilization_max,
            )
        else:
            feedback_record = {
                "feedback_active": 0,
                "feedback_reason": "none",
                "feedback_lift_velocity_scale": 1.0,
                "feedback_hold_height_offset_m": 0.0,
                "feedback_stabilization_extension_s": 0.0,
                "feedback_trigger_count": 0,
                "feedback_observed_object_vz_m_s": 0.0,
                "feedback_observed_object_accel_m_s2": 0.0,
                "feedback_active_probability": 0.0,
                "feedback_raw_lift_velocity_scale": 1.0,
                "feedback_raw_hold_height_offset_m": 0.0,
                "feedback_raw_stabilization_extension_s": 0.0,
            }
        reason = feedback_record["feedback_reason"]
        if reason not in feedback_reason_to_id:
            reason_id = len(feedback_reason_to_id)
            feedback_reason_to_id[reason] = reason_id
            feedback_reason_labels[str(reason_id)] = reason
        reason_id = feedback_reason_to_id[reason]
        rollout_records["step"].append(step + 1)
        rollout_records["sim_time"].append(float(example.sim_time))
        rollout_records["joint_q"].append(_as_np(example.state_0.joint_q))
        rollout_records["joint_qd"].append(_as_np(example.state_0.joint_qd))
        rollout_records["joint_target_q"].append(_as_np(example.control.joint_target_q))
        rollout_records["object_body_q"].append(object_pose)
        rollout_records["ee_body_q"].append(_ee_pose(example))
        rollout_records["rigid_contact_count"].append(contact_proxy)
        rollout_records["controller_phase_index"].append(phase_idx)
        rollout_records["commanded_gripper_target"].append(_commanded_gripper_target(example))
        rollout_records["commanded_lift_target"].append(_commanded_lift_target(example))
        rollout_records["feedback_active"].append(feedback_record["feedback_active"])
        rollout_records["feedback_reason_id"].append(reason_id)
        rollout_records["feedback_lift_velocity_scale"].append(feedback_record["feedback_lift_velocity_scale"])
        rollout_records["feedback_hold_height_offset_m"].append(feedback_record["feedback_hold_height_offset_m"])
        rollout_records["feedback_stabilization_extension_s"].append(
            feedback_record["feedback_stabilization_extension_s"]
        )
        rollout_records["feedback_trigger_count"].append(feedback_record["feedback_trigger_count"])
        rollout_records["feedback_observed_object_vz_m_s"].append(
            feedback_record["feedback_observed_object_vz_m_s"]
        )
        rollout_records["feedback_observed_object_accel_m_s2"].append(
            feedback_record["feedback_observed_object_accel_m_s2"]
        )
        rollout_records["feedback_active_probability"].append(feedback_record.get("feedback_active_probability", 0.0))
        rollout_records["feedback_raw_lift_velocity_scale"].append(
            feedback_record.get("feedback_raw_lift_velocity_scale", feedback_record["feedback_lift_velocity_scale"])
        )
        rollout_records["feedback_raw_hold_height_offset_m"].append(
            feedback_record.get("feedback_raw_hold_height_offset_m", feedback_record["feedback_hold_height_offset_m"])
        )
        rollout_records["feedback_raw_stabilization_extension_s"].append(
            feedback_record.get(
                "feedback_raw_stabilization_extension_s",
                feedback_record["feedback_stabilization_extension_s"],
            )
        )
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
        "candidate.controller.phase_index": np.asarray(rollout_records["controller_phase_index"], dtype=np.int32),
        "candidate.controller.commanded_gripper_target": np.asarray(
            rollout_records["commanded_gripper_target"], dtype=np.float32
        ),
        "candidate.controller.commanded_lift_target": np.asarray(
            rollout_records["commanded_lift_target"], dtype=np.float32
        ),
        "candidate.controller.feedback_active": np.asarray(rollout_records["feedback_active"], dtype=np.int32),
        "candidate.controller.feedback_reason_id": np.asarray(rollout_records["feedback_reason_id"], dtype=np.int32),
        "candidate.controller.feedback_lift_velocity_scale": np.asarray(
            rollout_records["feedback_lift_velocity_scale"], dtype=np.float32
        ),
        "candidate.controller.feedback_hold_height_offset_m": np.asarray(
            rollout_records["feedback_hold_height_offset_m"], dtype=np.float32
        ),
        "candidate.controller.feedback_stabilization_extension_s": np.asarray(
            rollout_records["feedback_stabilization_extension_s"], dtype=np.float32
        ),
        "candidate.controller.feedback_trigger_count": np.asarray(
            rollout_records["feedback_trigger_count"], dtype=np.int32
        ),
        "candidate.controller.feedback_observed_object_vz_m_s": np.asarray(
            rollout_records["feedback_observed_object_vz_m_s"], dtype=np.float32
        ),
        "candidate.controller.feedback_observed_object_accel_m_s2": np.asarray(
            rollout_records["feedback_observed_object_accel_m_s2"], dtype=np.float32
        ),
        "candidate.controller.feedback_active_probability": np.asarray(
            rollout_records["feedback_active_probability"], dtype=np.float32
        ),
        "candidate.controller.feedback_raw_lift_velocity_scale": np.asarray(
            rollout_records["feedback_raw_lift_velocity_scale"], dtype=np.float32
        ),
        "candidate.controller.feedback_raw_hold_height_offset_m": np.asarray(
            rollout_records["feedback_raw_hold_height_offset_m"], dtype=np.float32
        ),
        "candidate.controller.feedback_raw_stabilization_extension_s": np.asarray(
            rollout_records["feedback_raw_stabilization_extension_s"], dtype=np.float32
        ),
        "candidate.physics.tracked_body_indices": np.asarray(
            []
            if object_physics_adapter_meta.get("body_index_local") is None
            else [object_physics_adapter_meta["body_index_local"]],
            dtype=np.int32,
        ),
        "candidate.physics.tracked_shape_indices": np.asarray(
            object_physics_adapter_meta.get("shape_indices_local") or [], dtype=np.int32
        ),
        "candidate.physics.body_mass_scale": np.asarray([args_in.body_mass_scale], dtype=np.float32),
        "candidate.physics.shape_friction_scale": np.asarray([args_in.shape_friction_scale], dtype=np.float32),
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
    metrics = _task_metrics(
        rollout_arrays["newton.panda.object_body_q"],
        fps=float(example.fps),
        lift_height_min=args_in.lift_height_min,
        hold_duration_min=args_in.hold_duration_min,
        drop_height_loss=args_in.drop_height_loss,
    )

    summary = {
        "status": "pass",
        "classification": "official_newton_sensor_tiled_camera_export_not_trex_schema",
        "note": "Real SensorTiledCamera output from official Panda hydro. No T-Rex image keys are created.",
        "newton_version": getattr(newton, "__version__", "unknown"),
        "device": str(wp.get_device()),
        "scene": args_in.scene,
        "tracked_object": args_in.tracked_object,
        "controller_mode": args_in.controller_mode,
        "final_hold_duration": args_in.final_hold_duration,
        "physics_variant": object_physics_adapter_meta["requested"],
        "object_adapter": object_adapter_meta,
        "object_physics_adapter": object_physics_adapter_meta,
        "controller_adapter": controller_adapter_meta,
        "num_steps": args_in.num_steps,
        "pre_record_warmup_steps": args_in.pre_record_warmup_steps,
        "sample_steps": requested,
        "controller_type": (
            "newton_native_residual_controller_adapter_evaluation"
            if args_in.controller_mode == "lift_hold_learned_residual"
            else "official_newton_panda_hydro_scripted_no_adaptation"
        ),
        "controller_phase_labels": controller_phase_labels,
        "feedback_reason_labels": feedback_reason_labels,
        "scripted_feedback": {
            "enabled": args_in.controller_mode in {"lift_hold_feedback", "lift_hold_learned_residual"},
            "learned_policy": args_in.controller_mode == "lift_hold_learned_residual",
            "curiosity_reward": "none",
            "min_contact_count": args_in.feedback_min_contact_count,
            "accel_threshold_m_s2": args_in.feedback_accel_threshold,
            "height_drop_threshold_m": args_in.feedback_height_drop_threshold,
            "initial_lift_duration_scale": args_in.feedback_initial_lift_duration_scale,
            "lift_duration_scale_max": args_in.feedback_lift_duration_scale_max,
            "hold_height_step_m": args_in.feedback_hold_height_step,
            "hold_height_offset_max_m": args_in.feedback_hold_height_offset_max,
            "stabilization_step_s": args_in.feedback_stabilization_step,
            "stabilization_max_s": args_in.feedback_stabilization_max,
            "final_trigger_count": int(feedback_state["trigger_count"]),
            "source": "candidate.controller.*",
            "residual_adapter_checkpoint": str(args_in.residual_adapter_checkpoint)
            if args_in.residual_adapter_checkpoint
            else None,
            "residual_adapter_active_threshold": float(args_in.residual_adapter_active_threshold),
        },
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
        "task_metrics": metrics,
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
