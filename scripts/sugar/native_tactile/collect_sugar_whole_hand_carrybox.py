#!/usr/bin/env python3
"""Record native anatomical whole-hand tactile fields on SUGAR CarryBox.

This is a no-learning visualization collector. The official frozen SUGAR
Refiner controls the sensorized G1. A release failure relaxes every joint
target after ``release_step``. A closure failure returns only the right arm
to its neutral action before box contact, so the left and right native tactile
maps expose the resulting partial/missed closure. The box remains a dynamic
PhysX body in every condition.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback


HOST = socket.gethostname()
if HOST.startswith(("mgmtserver", "login")):
    raise SystemExit(f"Refusing IsaacLab collection on login node: {HOST}")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("A retained Slurm allocation is required")

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = (
    "Sugar-G129dof-CarryBox-Official-Refiner-Anatomical27-"
    "WholeHand-TacSL-Audit"
)
CHECKPOINT = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/"
    "baseline/ckpts/refiner_model10000.pt"
)
PATCHES = (
    *(f"palm_r{row}_c{column}" for row in range(4) for column in range(3)),
    *(
        f"{digit}_{segment}"
        for digit in ("thumb", "index", "middle", "ring", "little")
        for segment in ("proximal", "middle", "distal")
    ),
)
SIDES = ("left", "right")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument(
    "--scenario",
    choices=("successful_grasp", "failed_grasp", "failed_closure"),
    required=True,
)
parser.add_argument("--seed", type=int, default=4263)
parser.add_argument("--motion-id", type=int, default=45)
parser.add_argument("--max-steps", type=int, default=660)
parser.add_argument("--release-step", type=int, default=360)
parser.add_argument("--closure-fault-step", type=int, default=210)
parser.add_argument("--mass-kg", type=float, default=0.3023375868797302)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--physical-stiffness", type=float, default=1500.0)
parser.add_argument("--physical-damping", type=float, default=300.0)
parser.add_argument("--normal-stiffness", type=float, default=199.35014495534745)
parser.add_argument("--tangential-stiffness", type=float, default=19.935014495534745)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

output_root = args.output_root.expanduser().resolve()
experiment_root = (ROOT / "experiments").resolve()
if not output_root.is_relative_to(experiment_root):
    raise SystemExit("Output must remain below experiments/")
if output_root.exists():
    raise SystemExit(f"Refusing overwrite: {output_root}")
if not CHECKPOINT.is_file():
    raise SystemExit(f"Missing official SUGAR checkpoint: {CHECKPOINT}")
if args.max_steps < 120:
    raise SystemExit("At least 120 recorded control steps are required")
if args.scenario == "failed_grasp" and not 1 <= args.release_step < args.max_steps:
    raise SystemExit("release_step must lie inside the failed-grasp rollout")
if args.scenario == "failed_closure" and not 1 <= args.closure_fault_step < args.max_steps:
    raise SystemExit("closure_fault_step must lie inside the failed-closure rollout")

os.environ["CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_STIFFNESS"] = str(
    args.physical_stiffness
)
os.environ["CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_DAMPING"] = str(
    args.physical_damping
)
os.environ["CURIOSITY_ANATOMICAL_TACSL_NORMAL_STIFFNESS"] = str(
    args.normal_stiffness
)
os.environ["CURIOSITY_ANATOMICAL_TACSL_TANGENTIAL_STIFFNESS"] = str(
    args.tangential_stiffness
)
os.environ["CURIOSITY_ANATOMICAL_TACSL_FRICTION_COEFFICIENT"] = "0.5"
os.environ["CURIOSITY_ENABLE_ANATOMICAL27_WHOLE_HAND_TACSL_AUDIT"] = "1"
os.environ["CURIOSITY_TACSL_CALIBRATION_DIR"] = str(
    ROOT / "experiments/sugar_reproduction/assets/official_tacsl/calibration"
)
os.environ["ISAACLAB_GROUND_PLANE_USD"] = str(
    ROOT / "SUGAR/descriptions/terrain/sugar_ground_plane.usda"
)
os.environ["ISAACLAB_USE_LOCAL_FRAME_MARKER"] = "1"
os.chdir(ROOT / "SUGAR")
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT))
from scripts.sugar.native_tactile.slip import TactileSlipDetector  # noqa: E402
from scripts.sugar.native_tactile.universal import IsaacLabTacSLAdapter  # noqa: E402

from sugar_rl.assets.robots.anatomical_whole_hand_tacsl_g1 import (  # noqa: E402
    ANATOMICAL_WHOLE_HAND_PATCH_SPECS,
    anatomical_whole_hand_sensor_names,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_official_refiner_anatomical_whole_hand_tacsl_audit_env_cfg import (  # noqa: E402
    OfficialRefinerAnatomicalWholeHandTacSLAuditEnvCfg,
)
from sugar_rl.utils.official_refiner_nominal_teacher import (  # noqa: E402
    FrozenOfficialRefinerTeacher,
)

sys.path.insert(0, str(ROOT / "SUGAR/scripts/sugar_rl"))
from official_refiner_anatomical_whole_hand_tacsl_audit_task_registration import (  # noqa: E402
    register_official_refiner_anatomical_whole_hand_tacsl_audit_task,
)


def cpu(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=dtype).numpy()


def termination_after_grace(
    env,
    original_func,
    original_params: dict[str, object],
    grace_steps: int,
) -> torch.Tensor:
    return (env.episode_length_buf > grace_steps) & original_func(
        env, **original_params
    )


class FfmpegRgbWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        self.path = path
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
                f"{width}x{height}",
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

    def append(self, rgb: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}")


def main() -> None:
    output_root.mkdir(parents=True)
    trace_path = output_root / "whole_hand_trace.npz"
    world_path = output_root / "world_carrybox.mp4"
    summary_path = output_root / "summary.json"

    register_official_refiner_anatomical_whole_hand_tacsl_audit_task()
    cfg = OfficialRefinerAnatomicalWholeHandTacSLAuditEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.commands.motion.motion_folder = "data/CarryBox"
    cfg.commands.motion.init_with_ref = True
    cfg.commands.motion.start_init_env_ratio = 1.0
    cfg.commands.motion.pose_range = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.commands.motion.joint_position_range = (0.0, 0.0)
    cfg.events.push_robot = None
    cfg.events.push_object = None
    cfg.scene.left_hand_camera = None
    cfg.scene.right_hand_camera = None
    mass_scale = float(args.mass_kg) / 0.5
    cfg.events.obj_mass.params["mass_distribution_params"] = (
        mass_scale,
        mass_scale,
    )
    for group_name in ("policy", "critic"):
        group = getattr(cfg.observations, group_name, None)
        if group is not None:
            group.enable_corruption = False
    for termination_name in (
        "anchor_ori",
        "ee_body_pos",
        "obj_pos",
        "obj_ori",
        "anchor_pos",
    ):
        term = getattr(cfg.terminations, termination_name, None)
        if term is None:
            continue
        original_func = term.func
        original_params = dict(term.params)
        term.func = termination_after_grace
        term.params = {
            "original_func": original_func,
            "original_params": original_params,
            "grace_steps": 2,
        }
    for sensor_name in anatomical_whole_hand_sensor_names():
        getattr(cfg.scene, sensor_name).update_period = float(cfg.sim.dt)

    env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array")
    writer: FfmpegRgbWriter | None = None
    original_reset_idx = None
    original_scene_update = None
    try:
        base_env = env.unwrapped
        command = base_env.command_manager.get_term("motion")

        def fixed_start(env_ids) -> None:
            ids = (
                env_ids
                if isinstance(env_ids, torch.Tensor)
                else torch.as_tensor(env_ids, dtype=torch.long, device=base_env.device)
            )
            command.motion_id[ids] = args.motion_id
            command.time_steps[ids] = 0
            command._use_motion_data[ids] = True

        command._sample_init_state = fixed_start
        env.reset()
        original_reset_idx = base_env._reset_idx
        base_env._reset_idx = lambda env_ids: None
        # This visualization task loads the official checkpoint directly and
        # deliberately skips the historical hash gate.  Architecture, weights,
        # observations, and deterministic inference remain unchanged.
        teacher = FrozenOfficialRefinerTeacher(
            base_env,
            CHECKPOINT,
            expected_sha256=None,
        )
        action_term = base_env.action_manager.get_term("JointPositionAction")
        action_joint_names = tuple(action_term._joint_names)
        right_arm_action_indices = tuple(
            index
            for index, name in enumerate(action_joint_names)
            if name.startswith("right_")
            and any(part in name for part in ("shoulder", "elbow", "wrist"))
        )
        if len(right_arm_action_indices) != 7:
            raise RuntimeError(
                "Expected seven right-arm action indices, got "
                f"{right_arm_action_indices} from {action_joint_names}"
            )
        sensors = [base_env.scene[name] for name in anatomical_whole_hand_sensor_names()]
        if len(sensors) != 54:
            raise RuntimeError(f"Expected 54 sensors, found {len(sensors)}")
        common_patch_names = tuple(
            f"{side}_{patch}" for side in SIDES for patch in PATCHES
        )
        common_patch_sizes_m = [
            (spec.width_m, spec.length_m)
            for _side in SIDES
            for spec in ANATOMICAL_WHOLE_HAND_PATCH_SPECS
        ]
        tactile_adapter = IsaacLabTacSLAdapter(
            common_patch_names,
            grid_shape=(20, 25),
            patch_size_m=common_patch_sizes_m,
        )
        slip_detector = TactileSlipDetector(
            common_patch_names,
            friction_coefficient=0.5,
        )
        center_optical = [sensors[4], sensors[31]]
        optical_baseline_rgb: list[np.ndarray] = []
        optical_baseline_depth: list[np.ndarray] = []
        for side, sensor in zip(SIDES, center_optical, strict=True):
            if int(torch.count_nonzero(sensor.data.tactile_normal_force).item()) != 0:
                raise RuntimeError(
                    f"{side} center R15 baseline was requested under contact"
                )
            camera = sensor._camera_sensor
            depth = None
            for _ in range(16):
                base_env.sim.render()
                camera.update(0.0, force_recompute=True)
                depth = camera.data.output["distance_to_image_plane"]
                if bool(torch.isfinite(depth).all().item()):
                    break
            if depth is None or not bool(torch.isfinite(depth).all().item()):
                raise RuntimeError(f"{side} R15 no-contact baseline is non-finite")
            sensor.get_initial_render()
            zero_deformation = torch.zeros_like(depth[..., 0])
            optical_baseline_rgb.append(
                cpu(
                    sensor._tactile_rgb_render.render(zero_deformation)[0],
                    torch.uint8,
                )
            )
            optical_baseline_depth.append(cpu(depth[0]))
        world_camera = base_env.scene["world_camera"]
        robot = base_env.scene["robot"]
        obj = base_env.scene["obj"]
        writer = FfmpegRgbWriter(world_path, 1280, 720, args.fps)

        normal_rows: list[np.ndarray] = []
        shear_rows: list[np.ndarray] = []
        penetration_rows: list[np.ndarray] = []
        object_rows: list[np.ndarray] = []
        object_velocity_rows: list[np.ndarray] = []
        joint_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        position_rows: list[np.ndarray] = []
        quaternion_rows: list[np.ndarray] = []
        contact_normal_rows: list[np.ndarray] = []
        relative_velocity_rows: list[np.ndarray] = []
        optical_rgb_rows: list[np.ndarray] = []
        optical_depth_rows: list[np.ndarray] = []
        slip_state_rows: list[np.ndarray] = []
        slip_normal_load_rows: list[np.ndarray] = []
        slip_tangential_load_rows: list[np.ndarray] = []
        slip_friction_utilization_rows: list[np.ndarray] = []
        slip_cop_speed_rows: list[np.ndarray] = []
        slip_footprint_rate_rows: list[np.ndarray] = []
        slip_normal_loss_rate_rows: list[np.ndarray] = []
        patch_box_force_rows: list[np.ndarray] = []
        patch_box_friction_rows: list[np.ndarray] = []
        robot_box_force_rows: list[np.ndarray] = []
        robot_box_friction_rows: list[np.ndarray] = []
        physics_object_state_rows: list[np.ndarray] = []
        physics_object_velocity_rows: list[np.ndarray] = []
        physics_robot_box_force_rows: list[np.ndarray] = []
        physics_robot_box_friction_rows: list[np.ndarray] = []
        physics_control_steps: list[int] = []
        physics_substeps: list[int] = []
        source_frames: list[int] = []
        terminated_rows: list[bool] = []
        truncated_rows: list[bool] = []
        termination_names = tuple(base_env.termination_manager.active_terms)
        termination_rows = {name: [] for name in termination_names}
        all_robot_box_contact = base_env.scene["all_robot_box_contact"]
        object_material_properties = cpu(obj.root_physx_view.get_material_properties())
        capture_state = {"enabled": False, "control_step": -1, "substep": 0}
        original_scene_update = base_env.scene.update

        def scene_update_with_physics_balance(dt: float) -> None:
            original_scene_update(dt)
            if not capture_state["enabled"]:
                return
            contact = all_robot_box_contact.data
            if contact.force_matrix_w is None or contact.friction_forces_w is None:
                raise RuntimeError("Substep robot/box force data is absent")
            physics_object_state_rows.append(cpu(obj.data.root_state_w[0]))
            physics_object_velocity_rows.append(cpu(obj.data.root_vel_w[0]))
            physics_robot_box_force_rows.append(
                cpu(contact.force_matrix_w[0, :, 0])
            )
            physics_robot_box_friction_rows.append(
                cpu(contact.friction_forces_w[0, :, 0])
            )
            physics_control_steps.append(int(capture_state["control_step"]))
            physics_substeps.append(int(capture_state["substep"]))
            capture_state["substep"] += 1

        base_env.scene.update = scene_update_with_physics_balance

        for source_step in range(args.max_steps):
            source_frames.append(int(command.time_steps[0]))
            _, action = teacher.action()
            if args.scenario == "failed_grasp" and source_step >= args.release_step:
                action = torch.zeros_like(action)
            elif (
                args.scenario == "failed_closure"
                and source_step >= args.closure_fault_step
            ):
                action = action.clone()
                action[:, right_arm_action_indices] = 0.0
            capture_state.update(
                enabled=True,
                control_step=source_step,
                substep=0,
            )
            try:
                _, _, terminated, truncated, _ = env.step(action)
            finally:
                capture_state["enabled"] = False
            if capture_state["substep"] != int(cfg.decimation):
                raise RuntimeError(
                    "Expected "
                    f"{cfg.decimation} physics samples, got "
                    f"{capture_state['substep']} at control step {source_step}"
                )

            tactile_frame = tactile_adapter.update(
                {"carrybox": [sensor.data for sensor in sensors]},
                timestamp_s=(source_step + 1) * float(cfg.decimation * cfg.sim.dt),
            )
            slip_evidence = slip_detector.update(tactile_frame)
            normal = cpu(tactile_frame.normal_force_n[0]).reshape(
                2, 27, 20, 25
            )
            shear = cpu(tactile_frame.shear_force_xy_n[0]).reshape(
                2, 27, 20, 25, 2
            )
            penetration = cpu(tactile_frame.penetration_m[0]).reshape(
                2, 27, 20, 25
            )
            position = cpu(tactile_frame.taxel_position_w_m[0]).reshape(
                2, 27, 20, 25, 3
            )
            quaternion = cpu(
                tactile_frame.taxel_orientation_w_xyzw[0]
            ).reshape(2, 27, 20, 25, 4)
            contact_normal = np.stack(
                [cpu(sensor.data.tactile_contact_normal_w[0]) for sensor in sensors]
            ).reshape(2, 27, 20, 25, 3)
            relative_velocity = np.stack(
                [
                    cpu(sensor.data.tactile_relative_tangential_velocity_w[0])
                    for sensor in sensors
                ]
            ).reshape(2, 27, 20, 25, 3)
            normal_rows.append(normal)
            shear_rows.append(shear)
            penetration_rows.append(penetration)
            position_rows.append(position)
            quaternion_rows.append(quaternion)
            contact_normal_rows.append(contact_normal)
            relative_velocity_rows.append(relative_velocity)
            slip_state_rows.append(slip_evidence.state.reshape(2, 27))
            slip_normal_load_rows.append(
                slip_evidence.normal_load_n.reshape(2, 27)
            )
            slip_tangential_load_rows.append(
                slip_evidence.tangential_load_n.reshape(2, 27)
            )
            slip_friction_utilization_rows.append(
                slip_evidence.friction_utilization.reshape(2, 27)
            )
            slip_cop_speed_rows.append(
                slip_evidence.center_of_pressure_speed_m_s.reshape(2, 27)
            )
            slip_footprint_rate_rows.append(
                slip_evidence.footprint_change_rate_s.reshape(2, 27)
            )
            slip_normal_loss_rate_rows.append(
                slip_evidence.normal_loss_rate_s.reshape(2, 27)
            )
            object_rows.append(cpu(obj.data.root_state_w[0]))
            object_velocity_rows.append(cpu(obj.data.root_vel_w[0]))
            joint_rows.append(cpu(robot.data.joint_pos[0]))
            action_rows.append(cpu(action[0]))
            optical_rgb_rows.append(
                np.stack(
                    [
                        cpu(
                            sensors[hand_index * 27 + 4].data.tactile_rgb_image[0],
                            torch.uint8,
                        )
                        for hand_index in range(2)
                    ]
                )
            )
            optical_depth_rows.append(
                np.stack(
                    [
                        cpu(
                            sensors[hand_index * 27 + 4].data.tactile_depth_image[0]
                        )
                        for hand_index in range(2)
                    ]
                )
            )
            robot_contact = base_env.scene["all_robot_box_contact"].data
            if robot_contact.force_matrix_w is None:
                raise RuntimeError("All-robot box-contact force matrix is absent")
            robot_box_force_rows.append(cpu(robot_contact.force_matrix_w[0, :, 0]))
            if robot_contact.friction_forces_w is None:
                raise RuntimeError("All-robot box friction-force matrix is absent")
            robot_box_friction_rows.append(
                cpu(robot_contact.friction_forces_w[0, :, 0])
            )
            patch_force_by_hand = []
            patch_friction_by_hand = []
            for side in SIDES:
                patch_contact = base_env.scene[f"{side}_patch_box_contact"].data
                if patch_contact.force_matrix_w is None:
                    raise RuntimeError(f"{side} patch box-contact force matrix is absent")
                if patch_contact.friction_forces_w is None:
                    raise RuntimeError(
                        f"{side} patch box friction-force matrix is absent"
                    )
                patch_force_by_hand.append(
                    cpu(patch_contact.force_matrix_w[0, :, 0])
                )
                patch_friction_by_hand.append(
                    cpu(patch_contact.friction_forces_w[0, :, 0])
                )
            patch_box_force_rows.append(np.stack(patch_force_by_hand))
            patch_box_friction_rows.append(np.stack(patch_friction_by_hand))
            terminated_rows.append(bool(terminated[0].item()))
            truncated_rows.append(bool(truncated[0].item()))
            for name in termination_names:
                termination_rows[name].append(
                    bool(base_env.termination_manager.get_term(name)[0].item())
                )

            rgb = cpu(world_camera.data.output["rgb"][0, ..., :3], torch.uint8)
            writer.append(rgb)
            if source_step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "source_step": source_step,
                            "box_z_m": float(object_rows[-1][2]),
                            "active_taxels_left": int(np.count_nonzero(normal[0])),
                            "active_taxels_right": int(np.count_nonzero(normal[1])),
                        }
                    ),
                    flush=True,
                )
            if (
                args.scenario != "failed_closure"
                and (terminated_rows[-1] or truncated_rows[-1])
            ):
                break

        writer.close()
        writer = None
        normal_array = np.stack(normal_rows).astype(np.float32)
        shear_array = np.stack(shear_rows).astype(np.float32)
        penetration_array = np.stack(penetration_rows).astype(np.float32)
        object_array = np.stack(object_rows).astype(np.float32)
        object_velocity_array = np.stack(object_velocity_rows).astype(np.float32)
        active = np.count_nonzero(penetration_array > 0.0, axis=(-1, -2))
        bilateral = np.all(np.any(active > 0, axis=-1), axis=-1)
        relative_lift = object_array[:, 2] - object_array[0, 2]
        termination_arrays = {
            f"termination_{name}": np.asarray(values, dtype=np.bool_)
            for name, values in termination_rows.items()
        }
        np.savez_compressed(
            trace_path,
            normal_force=normal_array,
            signed_shear=shear_array,
            penetration=penetration_array,
            taxel_position_w=np.stack(position_rows).astype(np.float32),
            taxel_quaternion_w=np.stack(quaternion_rows).astype(np.float32),
            tactile_contact_normal_w=np.stack(contact_normal_rows).astype(np.float32),
            tactile_relative_tangential_velocity_w=np.stack(
                relative_velocity_rows
            ).astype(np.float32),
            tactile_only_slip_state=np.stack(slip_state_rows).astype(np.int8),
            tactile_only_slip_normal_load_n=np.stack(slip_normal_load_rows).astype(
                np.float32
            ),
            tactile_only_slip_tangential_load_n=np.stack(
                slip_tangential_load_rows
            ).astype(np.float32),
            tactile_only_slip_friction_utilization=np.stack(
                slip_friction_utilization_rows
            ).astype(np.float32),
            tactile_only_slip_cop_speed_m_s=np.stack(slip_cop_speed_rows).astype(
                np.float32
            ),
            tactile_only_slip_footprint_rate_s=np.stack(
                slip_footprint_rate_rows
            ).astype(np.float32),
            tactile_only_slip_normal_loss_rate_s=np.stack(
                slip_normal_loss_rate_rows
            ).astype(np.float32),
            optical_rgb=np.stack(optical_rgb_rows).astype(np.uint8),
            optical_depth=np.stack(optical_depth_rows).astype(np.float32),
            optical_baseline_rgb=np.stack(optical_baseline_rgb).astype(np.uint8),
            optical_baseline_depth=np.stack(optical_baseline_depth).astype(
                np.float32
            ),
            active_taxels=active.astype(np.int32),
            bilateral_contact=bilateral.astype(np.bool_),
            object_state_w=object_array,
            object_velocity_w=object_velocity_array,
            patch_box_force_w=np.stack(patch_box_force_rows).astype(np.float32),
            patch_box_friction_force_w=np.stack(patch_box_friction_rows).astype(
                np.float32
            ),
            robot_box_force_w=np.stack(robot_box_force_rows).astype(np.float32),
            robot_box_friction_force_w=np.stack(robot_box_friction_rows).astype(
                np.float32
            ),
            robot_box_force_body_names=np.asarray(
                base_env.scene["all_robot_box_contact"].body_names
            ),
            robot_joint_position=np.stack(joint_rows).astype(np.float32),
            applied_action=np.stack(action_rows).astype(np.float32),
            source_step=np.arange(len(normal_array), dtype=np.int32),
            motion_frame_before_action=np.asarray(source_frames, dtype=np.int32),
            terminated=np.asarray(terminated_rows, dtype=np.bool_),
            truncated=np.asarray(truncated_rows, dtype=np.bool_),
            patch_order=np.asarray(PATCHES),
            side_order=np.asarray(SIDES),
            sensor_names=np.asarray(anatomical_whole_hand_sensor_names()),
            tactile_patch_size_m=np.asarray(common_patch_sizes_m, dtype=np.float32).reshape(
                2, 27, 2
            ),
            gravity_w=np.asarray(cfg.sim.gravity, dtype=np.float32),
            physics_dt_s=np.asarray(cfg.sim.dt, dtype=np.float64),
            control_dt_s=np.asarray(cfg.decimation * cfg.sim.dt, dtype=np.float64),
            physics_object_state_w=np.stack(physics_object_state_rows).astype(
                np.float32
            ),
            physics_object_velocity_w=np.stack(
                physics_object_velocity_rows
            ).astype(np.float32),
            physics_robot_box_force_w=np.stack(
                physics_robot_box_force_rows
            ).astype(np.float32),
            physics_robot_box_friction_force_w=np.stack(
                physics_robot_box_friction_rows
            ).astype(np.float32),
            physics_control_step=np.asarray(physics_control_steps, dtype=np.int32),
            physics_substep=np.asarray(physics_substeps, dtype=np.int8),
            object_material_properties=object_material_properties.astype(np.float32),
            **termination_arrays,
        )
        reasons = [
            name for name, values in termination_rows.items() if any(values)
        ]
        lifted_and_bilateral = (relative_lift >= 0.20) & bilateral
        summary = {
            "schema": "sugar_whole_hand_carrybox_native_tactile_v2",
            "scenario": args.scenario,
            "host": HOST,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "task_id": TASK_ID,
            "motion_id": args.motion_id,
            "seed": args.seed,
            "source_frames": int(len(normal_array)),
            "release_step": (
                args.release_step if args.scenario == "failed_grasp" else None
            ),
            "closure_fault_step": (
                args.closure_fault_step
                if args.scenario == "failed_closure"
                else None
            ),
            "closure_fault_action_joint_names": (
                [action_joint_names[index] for index in right_arm_action_indices]
                if args.scenario == "failed_closure"
                else None
            ),
            "continued_after_task_termination_for_visualization": (
                args.scenario == "failed_closure"
            ),
            "box_mass_requested_kg": args.mass_kg,
            "box_mass_readback_kg": float(
                cpu(obj.root_physx_view.get_masses())[0].sum()
            ),
            "normal_shape": list(normal_array.shape),
            "shear_shape": list(shear_array.shape),
            "common_tactile_backend": "isaaclab_tacsl",
            "common_tactile_patch_count": len(common_patch_names),
            "common_tactile_patch_size_shape": [2, 27, 2],
            "common_tactile_patch_size_order": "row/local-X then column/local-Y",
            "tactile_only_slip_state_shape": list(
                np.stack(slip_state_rows).shape
            ),
            "tactile_only_slip_inputs": [
                "signed_local_z_force",
                "signed_local_xy_shear",
                "penetration",
                "timestamps",
            ],
            "taxel_position_shape": [len(normal_array), 2, 27, 20, 25, 3],
            "taxel_quaternion_shape": [len(normal_array), 2, 27, 20, 25, 4],
            "optical_rgb_shape": list(np.stack(optical_rgb_rows).shape),
            "optical_depth_shape": list(np.stack(optical_depth_rows).shape),
            "optical_baseline_rgb_shape": list(
                np.stack(optical_baseline_rgb).shape
            ),
            "optical_baseline_depth_shape": list(
                np.stack(optical_baseline_depth).shape
            ),
            "patch_box_force_shape": list(np.stack(patch_box_force_rows).shape),
            "patch_box_friction_force_shape": list(
                np.stack(patch_box_friction_rows).shape
            ),
            "robot_box_force_shape": list(np.stack(robot_box_force_rows).shape),
            "robot_box_friction_force_shape": list(
                np.stack(robot_box_friction_rows).shape
            ),
            "physics_substeps_per_control_step": int(cfg.decimation),
            "physics_robot_box_force_shape": list(
                np.stack(physics_robot_box_force_rows).shape
            ),
            "maximum_relative_lift_m": float(relative_lift.max()),
            "final_relative_lift_m": float(relative_lift[-1]),
            "bilateral_contact_frames": int(bilateral.sum()),
            "lifted_bilateral_frames": int(lifted_and_bilateral.sum()),
            "maximum_active_taxels_left": int(active[:, 0].sum(axis=-1).max()),
            "maximum_active_taxels_right": int(active[:, 1].sum(axis=-1).max()),
            "termination_reasons": reasons,
            "world_video": str(world_path),
            "trace": str(trace_path),
            "claim_boundary": (
                "No-learning native-sensor behavior trace. Final success/failure "
                "and visible correspondence require the paired video review."
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if original_scene_update is not None:
            base_env.scene.update = original_scene_update
        if original_reset_idx is not None:
            base_env._reset_idx = original_reset_idx
        if writer is not None:
            writer.process.kill()
            writer.process.wait()
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
