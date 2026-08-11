#!/usr/bin/env python3
"""Run the official R15 on a swept capsule and render native slip evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--frames", type=int, default=240)
parser.add_argument("--fps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.frames < 240:
    parser.error("--frames must be at least 240 to include every declared slip phase")
simulation_app = AppLauncher(args).app

import json
import math
import subprocess
import sys
import traceback

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from pxr import UsdGeom, UsdPhysics

import omni.replicator.core as rep
import isaacsim.core.utils.stage as stage_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
import isaaclab.utils.math as math_utils
from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor, VisuoTactileSensorCfg

from scripts.sugar.native_tactile.slip import SlipState, TactileSlipDetector
from scripts.sugar.native_tactile.universal import IsaacLabTacSLAdapter


ROOT = Path(__file__).resolve().parents[3]
R15_USD = (
    ROOT
    / "experiments/sugar_reproduction/assets/official_tacsl"
    / "gelsight_r15_finger/gelsight_r15_finger.usd"
)
CALIBRATION = (
    ROOT
    / "experiments/sugar_reproduction/assets/official_tacsl/calibration"
)
WIDTH, HEIGHT = 1280, 720
OBJECT_RADIUS_M = 0.006
CONTACT_INDENTATION_M = 0.0008
FORCE_SCALE_N_PER_TAXEL = 0.0012
SLOW_STEP_M = 0.00012  # 0.006 m/s at 50 Hz
FAST_STEP_M = 0.00060  # 0.030 m/s, deliberately above the 0.020 m/s gross-slip label
RETURN_STEP_M = 0.00020  # 0.010 m/s
SLOW_END_M = SLOW_STEP_M * 50
FAST_END_M = SLOW_END_M - FAST_STEP_M * 20


class VideoWriter:
    def __init__(self, path: Path, fps: int) -> None:
        self.process = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{WIDTH}x{HEIGHT}",
                "-r",
                str(fps),
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
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def append(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def put(frame: np.ndarray, text: str, point: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(
        frame,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    output = np.full((height, width, 3), 245, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return output


def force_rgb(normal: np.ndarray, maximum: float) -> np.ndarray:
    scaled = np.clip(normal / max(maximum, 1.0e-9), -1.0, 1.0)
    magnitude = np.abs(scaled)
    image = np.full((*normal.shape, 3), 255, dtype=np.uint8)
    negative = scaled < 0.0
    positive = scaled > 0.0
    image[..., 1][negative] = np.rint(255.0 * (1.0 - 0.7 * magnitude[negative])).astype(np.uint8)
    image[..., 2][negative] = np.rint(255.0 * (1.0 - magnitude[negative])).astype(np.uint8)
    image[..., 0][positive] = np.rint(255.0 * (1.0 - magnitude[positive])).astype(np.uint8)
    image[..., 1][positive] = np.rint(255.0 * (1.0 - 0.7 * magnitude[positive])).astype(np.uint8)
    return image


def object_x_and_label(frame: int) -> tuple[float, str]:
    if frame < 40:
        return 0.0, "fixed contact"
    if frame < 90:
        return SLOW_STEP_M * (frame - 40), "slow sweep"
    if frame < 120:
        return SLOW_END_M, "fixed after slow"
    if frame < 140:
        return SLOW_END_M - FAST_STEP_M * (frame - 120), "fast sweep"
    if frame < 180:
        return FAST_END_M, "fixed after sweep"
    return FAST_END_M + RETURN_STEP_M * (frame - 180), "return sweep"


def onset_delay(
    oracle_state: np.ndarray,
    predicted_state: np.ndarray,
    *,
    start: int,
    end: int,
    threshold: SlipState,
    fps: int,
) -> dict[str, int | float | None]:
    """Measure causal detector onset after the first held-out phase crossing."""
    oracle_indices = np.flatnonzero(oracle_state[start:end] >= int(threshold))
    if not len(oracle_indices):
        return {
            "oracle_onset_frame": None,
            "predicted_onset_frame": None,
            "delay_frames": None,
            "delay_s": None,
        }
    oracle_frame = start + int(oracle_indices[0])
    predicted_indices = np.flatnonzero(
        predicted_state[oracle_frame:end] >= int(threshold)
    )
    predicted_frame = (
        oracle_frame + int(predicted_indices[0]) if len(predicted_indices) else None
    )
    delay_frames = None if predicted_frame is None else predicted_frame - oracle_frame
    return {
        "oracle_onset_frame": oracle_frame,
        "predicted_onset_frame": predicted_frame,
        "delay_frames": delay_frames,
        "delay_s": None if delay_frames is None else delay_frames / fps,
    }


def main() -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    video_path = output_root / "isaaclab_r15_capsule_tactile_slip.mp4"
    trace_path = output_root / "trace.npz"
    summary_path = output_root / "summary.json"

    stage_utils.create_new_stage()
    dt = 1.0 / args.fps
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=dt, device=args.device)
    )

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=str(R15_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            compliant_contact_stiffness=10.0,
            compliant_contact_damping=1.0,
            physics_material_prim_path="elastomer",
        ),
        actuators={},
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(math.sqrt(2.0) / 2.0, -math.sqrt(2.0) / 2.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )
    object_cfg = RigidObjectCfg(
        prim_path="/World/Capsule",
        spawn=sim_utils.MeshCapsuleCfg(
            radius=OBJECT_RADIUS_M,
            height=0.024,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.20, 0.52),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    render_cfg = GELSIGHT_R15_CFG.replace(base_data_path=str(CALIBRATION))
    sensor_cfg = VisuoTactileSensorCfg(
        prim_path="/World/Robot/elastomer/tactile_sensor",
        update_period=dt,
        debug_vis=False,
        enable_camera_tactile=True,
        enable_force_field=True,
        camera_cfg=TiledCameraCfg(
            height=render_cfg.image_height,
            width=render_cfg.image_width,
            prim_path="/World/Robot/elastomer_tip/cam",
            update_period=dt,
            data_types=["distance_to_image_plane"],
            spawn=None,
        ),
        render_cfg=render_cfg,
        tactile_array_size=(20, 25),
        tactile_margin=0.003,
        contact_object_prim_path_expr="/World/Capsule",
        normal_contact_stiffness=1.0,
        friction_coefficient=0.5,
        tangential_stiffness=0.1,
    )
    world_camera_cfg = TiledCameraCfg(
        prim_path="/World/WorldCamera",
        update_period=dt,
        height=360,
        width=640,
        data_types=["rgb"],
        offset=TiledCameraCfg.OffsetCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=30.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 5.0),
        ),
    )
    sim_utils.update_stage()
    robot = Articulation(robot_cfg)
    contact_object = RigidObject(object_cfg)
    collision_mesh_path = "/World/Capsule/geometry/mesh"
    collision_mesh = stage_utils.get_current_stage().GetPrimAtPath(collision_mesh_path)
    UsdPhysics.MeshCollisionAPI(collision_mesh).GetApproximationAttr().Set("sdf")
    sim_utils.define_mesh_collision_properties(
        collision_mesh_path,
        sim_utils.SDFMeshPropertiesCfg(
            sdf_margin=0.02,
            sdf_narrow_band_thickness=0.05,
            sdf_resolution=64,
            sdf_subgrid_resolution=6,
        ),
    )
    sensor = VisuoTactileSensor(sensor_cfg)
    world_camera = TiledCamera(world_camera_cfg)
    sim.reset()
    world_camera.set_world_poses_from_view(
        torch.tensor([[0.34, -0.38, 0.72]], device=args.device),
        torch.tensor([[0.0, 0.04, 0.52]], device=args.device),
    )
    for _ in range(4):
        sim.step()
        sensor.update(dt, force_recompute=True)
        world_camera.update(dt, force_recompute=True)
    sensor.get_initial_render()
    tactile_points = sensor.data.tactile_points_pos_w[0]
    tactile_quaternions = sensor.data.tactile_points_quat_w[0]
    local_z = torch.zeros_like(tactile_points)
    local_z[:, 2] = 1.0
    patch_normal_w = math_utils.quat_apply(tactile_quaternions, local_z).mean(dim=0)
    patch_normal_w = patch_normal_w / torch.linalg.norm(patch_normal_w)
    preferred_tangent_w = torch.tensor((1.0, 0.0, 0.0), device=args.device)
    sweep_tangent_w = preferred_tangent_w - patch_normal_w * torch.dot(
        preferred_tangent_w, patch_normal_w
    )
    sweep_tangent_w = sweep_tangent_w / torch.linalg.norm(sweep_tangent_w)
    tactile_center_w = tactile_points.mean(dim=0)
    normal_coordinates = tactile_points @ patch_normal_w
    contact_center_w = tactile_center_w + patch_normal_w * (
        normal_coordinates.max() - torch.dot(tactile_center_w, patch_normal_w)
    )
    object_mesh_points = torch.tensor(
        np.asarray(UsdGeom.Mesh(collision_mesh).GetPointsAttr().Get()),
        dtype=torch.float32,
        device=args.device,
    )
    object_support_distance_m = torch.max(-(object_mesh_points @ patch_normal_w))
    resting_object_center_w = contact_center_w + patch_normal_w * (
        object_support_distance_m - CONTACT_INDENTATION_M
    )

    adapter = IsaacLabTacSLAdapter(
        ("r15",),
        grid_shape=(20, 25),
        patch_size_m=((0.023977, 0.032001),),
    )
    detector = TactileSlipDetector(("r15",), friction_coefficient=0.5)
    writer = VideoWriter(video_path, args.fps)
    normal_rows = []
    shear_rows = []
    penetration_rows = []
    position_rows = []
    orientation_rows = []
    optical_rgb_rows = []
    optical_depth_rows = []
    tactile_sequence_rows = []
    tactile_timestamp_rows = []
    tactile_dt_rows = []
    optical_sequence_rows = []
    optical_timestamp_rows = []
    optical_dt_rows = []
    slip_rows = []
    oracle_speed_rows = []
    x_rows = []
    phase_rows = []
    previous_x = 0.0
    for frame in range(args.frames):
        x, label = object_x_and_label(frame)
        object_position_w = resting_object_center_w + sweep_tangent_w * x
        pose = torch.cat(
            (
                object_position_w,
                torch.tensor((1.0, 0.0, 0.0, 0.0), device=args.device),
            )
        ).unsqueeze(0)
        velocity = torch.zeros((1, 6), dtype=torch.float32, device=args.device)
        velocity[0, :3] = sweep_tangent_w * ((x - previous_x) / dt)
        contact_object.write_root_pose_to_sim(pose)
        contact_object.write_root_velocity_to_sim(velocity)
        previous_x = x
        sim.step()
        robot.update(dt)
        contact_object.update(dt)
        sensor.update(dt, force_recompute=True)
        world_camera.update(dt, force_recompute=True)
        tactile = adapter.update(
            {"capsule": [sensor.data]},
            timestamp_s=(frame + 1) * dt,
            optical_timestamp_s=(frame + 1) * dt,
        )
        if tactile.optical.clock is None:
            raise RuntimeError("Available official RGB/depth has no optical clock")
        optical_tensor = tactile.optical.rgb[0]
        depth_tensor = tactile.optical.depth[0]
        if optical_tensor is None or depth_tensor is None:
            raise RuntimeError("Official R15 optical data is unavailable")
        evidence = detector.update(tactile)

        normal = tactile.normal_force_n[0, 0].detach().cpu().numpy()
        shear = tactile.shear_force_xy_n[0, 0].detach().cpu().numpy()
        penetration = tactile.penetration_m[0, 0].detach().cpu().numpy()
        position = tactile.taxel_position_w_m[0, 0].detach().cpu().numpy()
        orientation = (
            tactile.taxel_orientation_w_xyzw[0, 0].detach().cpu().numpy()
        )
        optical = optical_tensor[0].detach().cpu().numpy().astype(np.uint8)
        depth = depth_tensor[0, ..., 0].detach().cpu().numpy()
        relative = sensor.data.tactile_relative_tangential_velocity_w[0].reshape(
            20, 25, 3
        )
        relative = relative.detach().cpu().numpy()
        oracle_speed = float(
            np.linalg.norm(relative[penetration > 0.0], axis=-1).max()
        ) if np.any(penetration > 0.0) else 0.0
        normal_rows.append(normal)
        shear_rows.append(shear)
        penetration_rows.append(penetration)
        position_rows.append(position)
        orientation_rows.append(orientation)
        optical_rgb_rows.append(optical)
        optical_depth_rows.append(depth)
        tactile_sequence_rows.append(tactile.clock.sequence)
        tactile_timestamp_rows.append(tactile.clock.timestamp_s)
        tactile_dt_rows.append(tactile.clock.dt_s)
        optical_sequence_rows.append(tactile.optical.clock.sequence)
        optical_timestamp_rows.append(tactile.optical.clock.timestamp_s)
        optical_dt_rows.append(tactile.optical.clock.dt_s)
        slip_rows.append(int(evidence.state[0, 0]))
        oracle_speed_rows.append(oracle_speed)
        x_rows.append(x)
        phase_rows.append(label)

        canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        world = world_camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        canvas[:360, :640] = fit(world, 640, 360)
        canvas[:360, 640:] = fit(optical, 640, 360)
        force = cv2.resize(
            force_rgb(normal, FORCE_SCALE_N_PER_TAXEL),
            (640, 360),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas[360:, :640] = force
        finite = depth[np.isfinite(depth)]
        low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        depth_u8 = np.rint(255.0 * np.clip((depth - low) / max(high - low, 1.0e-9), 0.0, 1.0)).astype(np.uint8)
        depth_rgb = np.repeat(depth_u8[..., None], 3, axis=-1)
        canvas[360:, 640:] = fit(depth_rgb, 640, 360)
        cv2.rectangle(canvas, (0, 0), (1279, 719), (50, 50, 50), 1)
        cv2.line(canvas, (640, 0), (640, 720), (70, 70, 70), 2)
        cv2.line(canvas, (0, 360), (1280, 360), (70, 70, 70), 2)
        for x0, title in ((0, "WORLD: official R15 + local capsule"), (640, "OFFICIAL R15 RGB")):
            cv2.rectangle(canvas, (x0 + 8, 8), (x0 + 625, 38), (255, 255, 255), -1)
            put(canvas, title, (x0 + 18, 30), 0.6)
        cv2.rectangle(canvas, (8, 368), (628, 414), (255, 255, 255), -1)
        put(canvas, f"SIGNED NORMAL FIELD | active {int(np.count_nonzero(penetration > 0.0))}/500", (18, 390), 0.52)
        put(canvas, f"Fn {evidence.normal_load_n[0,0]:.3f} N | Ft {evidence.tangential_load_n[0,0]:.3f} N", (18, 410), 0.45)
        cv2.rectangle(canvas, (648, 368), (1272, 452), (255, 255, 255), -1)
        put(canvas, "OFFICIAL R15 DEPTH", (658, 390), 0.55)
        put(canvas, f"frame {frame:03d} | command {label} | capsule x {x:+.4f} m", (658, 414), 0.46)
        put(canvas, f"tactile-only {SlipState(int(evidence.state[0,0])).name}", (658, 434), 0.52)
        put(canvas, f"held-out relative speed {oracle_speed:.4f} m/s", (658, 450), 0.46)
        writer.append(canvas)
    writer.close()

    normal_array = np.stack(normal_rows).astype(np.float32)
    penetration_array = np.stack(penetration_rows).astype(np.float32)
    slip_array = np.asarray(slip_rows, dtype=np.int8)
    oracle_speed_array = np.asarray(oracle_speed_rows, dtype=np.float32)
    tactile_sequence_array = np.asarray(tactile_sequence_rows, dtype=np.int64)
    tactile_timestamp_array = np.asarray(tactile_timestamp_rows, dtype=np.float64)
    tactile_dt_array = np.asarray(tactile_dt_rows, dtype=np.float64)
    optical_sequence_array = np.asarray(optical_sequence_rows, dtype=np.int64)
    optical_timestamp_array = np.asarray(optical_timestamp_rows, dtype=np.float64)
    optical_dt_array = np.asarray(optical_dt_rows, dtype=np.float64)
    expected_sequence = np.arange(args.frames, dtype=np.int64)
    expected_timestamp = (expected_sequence + 1) * dt
    expected_dt = np.full(args.frames, dt, dtype=np.float64)
    expected_dt[0] = 0.0
    if not np.array_equal(tactile_sequence_array, expected_sequence):
        raise RuntimeError("Tactile sequence is not contiguous from zero")
    if not np.array_equal(optical_sequence_array, expected_sequence):
        raise RuntimeError("Optical sequence is not contiguous from zero")
    if not np.allclose(
        tactile_timestamp_array, expected_timestamp, atol=1.0e-12
    ):
        raise RuntimeError("Tactile timestamps do not match the source clock")
    if not np.allclose(
        optical_timestamp_array, expected_timestamp, atol=1.0e-12
    ):
        raise RuntimeError("Optical timestamps do not match the source clock")
    if not np.allclose(tactile_dt_array, expected_dt, atol=1.0e-12):
        raise RuntimeError("Tactile elapsed times are inconsistent")
    if not np.allclose(optical_dt_array, expected_dt, atol=1.0e-12):
        raise RuntimeError("Optical elapsed times are inconsistent")
    np.savez_compressed(
        trace_path,
        normal_force=normal_array,
        signed_shear=np.stack(shear_rows).astype(np.float32),
        penetration=penetration_array,
        taxel_position_w=np.stack(position_rows).astype(np.float32),
        taxel_orientation_w_xyzw=np.stack(orientation_rows).astype(np.float32),
        optical_rgb=np.stack(optical_rgb_rows).astype(np.uint8),
        optical_depth=np.stack(optical_depth_rows).astype(np.float32),
        tactile_sequence=tactile_sequence_array,
        tactile_timestamp_s=tactile_timestamp_array,
        tactile_dt_s=tactile_dt_array,
        optical_sequence=optical_sequence_array,
        optical_timestamp_s=optical_timestamp_array,
        optical_dt_s=optical_dt_array,
        tactile_only_slip_state=slip_array,
        heldout_relative_tangential_speed_m_s=oracle_speed_array,
        object_x_m=np.asarray(x_rows, dtype=np.float32),
        command_phase=np.asarray(phase_rows),
    )
    contact_array = np.any(penetration_array > 0.0, axis=(1, 2))
    oracle_state = np.full(args.frames, int(SlipState.NO_CONTACT), dtype=np.int8)
    oracle_state[contact_array] = int(SlipState.STICK)
    oracle_state[contact_array & (oracle_speed_array >= 0.005)] = int(SlipState.INCIPIENT)
    oracle_state[contact_array & (oracle_speed_array >= 0.020)] = int(SlipState.GROSS)
    oracle_slip = oracle_state >= int(SlipState.INCIPIENT)
    predicted_slip = slip_array >= int(SlipState.INCIPIENT)
    true_positive = int(np.count_nonzero(oracle_slip & predicted_slip))
    false_positive = int(np.count_nonzero(~oracle_slip & predicted_slip))
    false_negative = int(np.count_nonzero(oracle_slip & ~predicted_slip))
    true_negative = int(np.count_nonzero(~oracle_slip & ~predicted_slip))
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (oracle_state, slip_array), 1)
    phase_intervals = {
        "fixed_contact": (0, 40),
        "slow_sweep": (40, 90),
        "fixed_after_slow": (90, 120),
        "fast_sweep": (120, 140),
        "fixed_after_sweep": (140, 180),
        "return_sweep": (180, args.frames),
    }
    phase_evaluation = {}
    for phase, (start, end) in phase_intervals.items():
        phase_evaluation[phase] = {
            "frame_interval": [start, end],
            "command_speed_m_s": float(
                np.median(np.abs(np.diff(np.asarray(x_rows)[max(start - 1, 0) : end])))
                * args.fps
            ) if end - start > 1 else 0.0,
            "heldout_speed_m_s_median": float(np.median(oracle_speed_array[start:end])),
            "heldout_speed_m_s_maximum": float(np.max(oracle_speed_array[start:end])),
            "oracle_state_counts": {
                SlipState(state).name: int(np.count_nonzero(oracle_state[start:end] == state))
                for state in range(4)
            },
            "predicted_state_counts": {
                SlipState(state).name: int(np.count_nonzero(slip_array[start:end] == state))
                for state in range(4)
            },
        }
    onset = {
        "incipient_slow_sweep": onset_delay(
            oracle_state,
            slip_array,
            start=40,
            end=90,
            threshold=SlipState.INCIPIENT,
            fps=args.fps,
        ),
        "gross_fast_sweep": onset_delay(
            oracle_state,
            slip_array,
            start=120,
            end=140,
            threshold=SlipState.GROSS,
            fps=args.fps,
        ),
    }
    summary = {
        "schema": "isaaclab_r15_capsule_native_tactile_slip_v2",
        "frames": args.frames,
        "fps": args.fps,
        "backend": "official isaaclab v2.3.2 VisuoTactileSensor",
        "sensor": "official GELSIGHT_R15_CFG",
        "object": "local IsaacLab MeshCapsuleCfg with PhysX SDF collision",
        "grid_shape": [20, 25],
        "taxel_position_shape": list(np.stack(position_rows).shape),
        "taxel_orientation_shape": list(np.stack(orientation_rows).shape),
        "taxel_quaternion_order": (
            "xyzw (official IsaacLab wxyz reordered by common adapter)"
        ),
        "optical_rgb_shape": list(np.stack(optical_rgb_rows).shape),
        "optical_depth_shape": list(np.stack(optical_depth_rows).shape),
        "tactile_clock_fields": [
            "tactile_sequence",
            "tactile_timestamp_s",
            "tactile_dt_s",
        ],
        "optical_clock_fields": [
            "optical_sequence",
            "optical_timestamp_s",
            "optical_dt_s",
        ],
        "persisted_clocks": {
            "tactile_sequence": [
                int(tactile_sequence_array[0]),
                int(tactile_sequence_array[-1]),
            ],
            "tactile_timestamp_s": [
                float(tactile_timestamp_array[0]),
                float(tactile_timestamp_array[-1]),
            ],
            "optical_sequence": [
                int(optical_sequence_array[0]),
                int(optical_sequence_array[-1]),
            ],
            "optical_timestamp_s": [
                float(optical_timestamp_array[0]),
                float(optical_timestamp_array[-1]),
            ],
            "elapsed_time_matches_source_clock": True,
        },
        "contact_indentation_m": CONTACT_INDENTATION_M,
        "force_display_scale_n_per_taxel": FORCE_SCALE_N_PER_TAXEL,
        "patch_normal_w": [float(value) for value in patch_normal_w.tolist()],
        "sweep_tangent_w": [float(value) for value in sweep_tangent_w.tolist()],
        "sensor_surface_span_along_normal_m": float(
            normal_coordinates.max() - normal_coordinates.min()
        ),
        "object_support_distance_along_negative_normal_m": float(
            object_support_distance_m
        ),
        "contact_frames": int(np.count_nonzero(np.any(penetration_array > 0.0, axis=(1, 2)))),
        "slip_state_counts": {
            SlipState(state).name: int(np.count_nonzero(slip_array == state))
            for state in range(4)
        },
        "heldout_state_confusion_rows_oracle_columns_prediction": confusion.tolist(),
        "heldout_binary": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": float(true_positive / max(true_positive + false_positive, 1)),
            "recall": float(true_positive / max(true_positive + false_negative, 1)),
        },
        "phase_evaluation": phase_evaluation,
        "onset_delay": onset,
        "maximum_heldout_relative_speed_m_s": float(oracle_speed_array.max()),
        "video": str(video_path),
        "trace": str(trace_path),
        "claim_boundary": "No policy and no hardware claim; relative velocity is evaluation-only.",
    }
    capture = cv2.VideoCapture(str(video_path))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if decoded != args.frames:
        raise RuntimeError(f"Video decoded {decoded}/{args.frames} frames")
    summary["fully_decoded_frames"] = decoded
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    rep.vp_manager.destroy_hydra_textures("Replicator")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        simulation_app.close()
