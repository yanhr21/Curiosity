#!/usr/bin/env python3
"""Run one no-learning live CarryBox mass jump with online patch telemetry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--motion-id", type=int, default=45)
parser.add_argument("--seed", type=int, default=150814)
parser.add_argument("--mass-factor", type=float, default=3.0)
parser.add_argument(
    "--action-trace",
    type=Path,
    default=None,
    help=(
        "Optional nominal trace containing applied_action.  Replaying this "
        "same sequence across mass factors isolates observation leakage from "
        "controller action changes."
    ),
)
parser.add_argument("--max-steps", type=int, default=450)
parser.add_argument("--minimum-lift", type=float, default=0.05)
parser.add_argument("--stable-frames", type=int, default=10)
parser.add_argument("--delay-frames", type=int, nargs=2, default=(10, 50))
parser.add_argument("--object-static-friction", type=float, default=0.5)
parser.add_argument("--object-dynamic-friction", type=float, default=0.5)
parser.add_argument(
    "--fixed-response",
    choices=("none", "squeeze_lower"),
    default="none",
    help=(
        "Optional mass-independent feasibility response applied at one fixed "
        "absolute frame. It never reads mass, jump state, tactile, or object state."
    ),
)
parser.add_argument("--fixed-response-frame", type=int, default=300)
parser.add_argument("--fixed-response-ramp-frames", type=int, default=20)
parser.add_argument(
    "--record-world",
    action="store_true",
    help="Record the synchronized G1 CarryBox world camera as H.264.",
)
parser.add_argument("--fps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.record_world:
    args.enable_cameras = True

if args.mass_factor < 1.0:
    raise SystemExit("preflight mass factor must be at least one")
if args.max_steps < 1:
    raise SystemExit("max-steps must be positive")
if args.fixed_response_frame < 0:
    raise SystemExit("fixed-response-frame must be non-negative")
if args.fixed_response_ramp_frames < 1:
    raise SystemExit("fixed-response-ramp-frames must be positive")
if not (
    0.0
    <= args.object_dynamic_friction
    <= args.object_static_friction
    < float("inf")
):
    raise SystemExit("object friction must satisfy 0 <= dynamic <= static")

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_STIFFNESS", "100")
os.environ.setdefault("CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_DAMPING", "20")
os.environ.setdefault("CURIOSITY_ANATOMICAL_TACSL_NORMAL_STIFFNESS", "20")
os.environ.setdefault("CURIOSITY_ANATOMICAL_TACSL_TANGENTIAL_STIFFNESS", "2")
os.environ.setdefault("CURIOSITY_ANATOMICAL_TACSL_FRICTION_COEFFICIENT", "0.5")
os.environ.setdefault(
    "CURIOSITY_TACSL_CALIBRATION_DIR",
    str(ROOT / "experiments/sugar_reproduction/assets/official_tacsl/calibration"),
)
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ["CURIOSITY_ENABLE_ANATOMICAL27_WHOLE_HAND_TACSL_AUDIT"] = "1"
os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(ROOT / "SUGAR/descriptions/terrain/sugar_ground_plane.usda"),
)
cached_g1_usd = (
    ROOT
    / "experiments/online_patch_tactile_mass_adaptation/runtime_assets"
    / "g1_29dof_preconverted_isaacsim510"
    / "g1_29dof_rev_1_0_with_rubber_hand.usd"
)
if cached_g1_usd.is_file():
    os.environ.setdefault("CURIOSITY_G1_PRECONVERTED_USD", str(cached_g1_usd))
os.chdir(ROOT / "SUGAR")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT / "SUGAR/scripts/sugar_rl"))
from official_refiner_anatomical_whole_hand_tacsl_audit_task_registration import (  # noqa: E402
    TASK_ID as REFINER_TASK_ID,
    register_official_refiner_anatomical_whole_hand_tacsl_audit_task,
)

from sugar_rl.tasks.locomanip.online_patch_tactile import (  # noqa: E402
    BASE_PATCH_CHANNELS,
    current_whole_hand_patch_features,
    current_whole_hand_patch_oracle_tangential_speed,
    current_whole_hand_patch_timestamps_s,
    online_patch_tactile_contract,
)
import sugar_rl.tasks.locomanip.mdp as mdp  # noqa: E402
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_online_patch_tactile_mass_env_cfg import (  # noqa: E402
    OnlinePatchSlipMassRobotPlayEnvCfg,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg import (  # noqa: E402
    _review_camera,
)
from sugar_rl.utils.official_refiner_nominal_teacher import (  # noqa: E402
    FrozenOfficialRefinerTeacher,
)


REFINER_CHECKPOINT = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/"
    "baseline/ckpts/refiner_model10000.pt"
)


def cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(device="cpu", dtype=torch.float32).numpy()


def cpu_native(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(device="cpu").numpy()


class FfmpegRgbWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
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
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=False)
    action_trace = args.action_trace
    if action_trace is not None:
        action_trace = action_trace.expanduser()
        if not action_trace.is_absolute():
            action_trace = ROOT / action_trace
        action_trace = action_trace.resolve()

    register_official_refiner_anatomical_whole_hand_tacsl_audit_task()
    cfg = OnlinePatchSlipMassRobotPlayEnvCfg()
    cfg.seed = int(args.seed)
    cfg.sim.device = args.device
    cfg.commands.motion.motion_folder = "data/CarryBox"
    cfg.commands.motion.pose_range = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.commands.motion.joint_position_range = (0.0, 0.0)
    if args.record_world:
        cfg.scene.world_camera = _review_camera(
            name="WorldCamera",
            position=(3.6, 3.6, 2.4),
            quaternion_wxyz=(
                0.3043649418,
                0.2319667899,
                0.5600173703,
                0.7348019703,
            ),
            width=1280,
            height=720,
        )
        cfg.sim.render_interval = cfg.decimation
    cfg.events.obj_physics_material.params.update(
        static_friction_range=(
            float(args.object_static_friction),
            float(args.object_static_friction),
        ),
        dynamic_friction_range=(
            float(args.object_dynamic_friction),
            float(args.object_dynamic_friction),
        ),
        restitution_range=(0.0, 0.0),
        num_buckets=1,
    )
    for term_name in ("reset_mass_jump", "step_mass_jump"):
        params = getattr(cfg.events, term_name).params
        params["mass_factors"] = (float(args.mass_factor),)
        params["minimum_lift_m"] = float(args.minimum_lift)
        params["stable_lift_frames"] = int(args.stable_frames)
        params["delay_frames"] = tuple(int(value) for value in args.delay_frames)
        params["seed"] = int(args.seed)

    env = gym.make(
        REFINER_TASK_ID,
        cfg=cfg,
        render_mode="rgb_array" if args.record_world else None,
    )
    original_reset_idx = None
    writer = None
    try:
        base_env = env.unwrapped
        command = base_env.command_manager.get_term("motion")

        def fixed_start(env_ids) -> None:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=base_env.device)
            command.motion_id[ids] = int(args.motion_id)
            command.time_steps[ids] = 0
            command._use_motion_data[ids] = True

        command._sample_init_state = fixed_start
        env.reset()
        reset_ids = torch.arange(base_env.num_envs, device=base_env.device)
        command._resample_command(reset_ids)
        base_env.scene.write_data_to_sim()
        base_env.sim.forward()
        original_reset_idx = base_env._reset_idx
        base_env._reset_idx = lambda env_ids: None

        teacher = (
            FrozenOfficialRefinerTeacher(
                base_env,
                REFINER_CHECKPOINT,
                expected_sha256=None,
            )
            if args.action_trace is None
            else None
        )
        obj = base_env.scene["obj"]
        object_material = obj.root_physx_view.get_material_properties().detach().cpu()
        if not bool(
            (object_material[..., 0] == float(args.object_static_friction)).all()
        ):
            raise RuntimeError("CarryBox static-friction readback mismatch")
        if not bool(
            (object_material[..., 1] == float(args.object_dynamic_friction)).all()
        ):
            raise RuntimeError("CarryBox dynamic-friction readback mismatch")
        robot = base_env.scene["robot"]
        world_camera = base_env.scene["world_camera"] if args.record_world else None
        action_term = base_env.action_manager.get_term("JointPositionAction")
        fixed_response_raw_delta = torch.zeros(
            29, dtype=torch.float32, device=base_env.device
        )
        fixed_response_joint_delta_rad: dict[str, float] = {}
        if args.fixed_response == "squeeze_lower":
            # One fixed response for every mass condition. Shoulder-roll
            # targets move both rigid hands inward; the symmetric leg targets
            # lower the support posture slightly. These are target offsets,
            # not a tactile controller and not a claim about optimal action.
            fixed_response_joint_delta_rad = {
                "left_shoulder_roll_joint": -0.10,
                "right_shoulder_roll_joint": 0.10,
                "left_hip_pitch_joint": -0.08,
                "right_hip_pitch_joint": -0.08,
                "left_knee_joint": 0.15,
                "right_knee_joint": 0.15,
                "left_ankle_pitch_joint": -0.07,
                "right_ankle_pitch_joint": -0.07,
            }
            action_joint_names = list(action_term._joint_names)
            scale = action_term._scale
            if not isinstance(scale, torch.Tensor):
                raise RuntimeError("fixed response requires tensor joint-action scale")
            scale = scale[0] if scale.ndim == 2 else scale
            for joint_name, delta_rad in fixed_response_joint_delta_rad.items():
                if joint_name not in action_joint_names:
                    raise RuntimeError(
                        f"fixed response joint is absent from action term: {joint_name}"
                    )
                action_index = action_joint_names.index(joint_name)
                joint_scale = float(scale[action_index].item())
                if joint_scale == 0.0:
                    raise RuntimeError(
                        f"fixed response joint has zero action scale: {joint_name}"
                    )
                fixed_response_raw_delta[action_index] = delta_rad / joint_scale
        replay_actions = None
        if action_trace is not None:
            if not action_trace.is_file():
                raise FileNotFoundError(action_trace)
            with np.load(action_trace, allow_pickle=False) as replay:
                action_key = (
                    "applied_action" if "applied_action" in replay.files else "action"
                )
                replay_actions = np.asarray(replay[action_key], dtype=np.float32)
            if replay_actions.ndim != 2 or replay_actions.shape[1] != 29:
                raise RuntimeError(
                    f"action trace must be [frames,29], got {replay_actions.shape}"
                )
            if len(replay_actions) < args.max_steps:
                raise RuntimeError(
                    f"action trace has {len(replay_actions)} frames, need {args.max_steps}"
                )

        rows: dict[str, list[np.ndarray | int | float | bool]] = {
            "patch_features": [],
            "slip_features": [],
            "slip_state": [],
            "oracle_patch_tangential_speed_m_s": [],
            "patch_sensor_timestamp_s": [],
            "actor_policy_observation": [],
            "actor_patch_history": [],
            "object_pos_w": [],
            "object_quat_w": [],
            "object_lin_vel_w": [],
            "object_ang_vel_w": [],
            "object_pos_b": [],
            "object_ori_b": [],
            "object_lin_vel_b": [],
            "object_ang_vel_b": [],
            "joint_pos": [],
            "joint_vel": [],
            "applied_action": [],
            "fixed_response_action_delta": [],
            "mass_readback_kg": [],
            "inertia_readback_kg_m2": [],
            "target_factor": [],
            "qualified": [],
            "pending": [],
            "pending_step": [],
            "jump_applied": [],
            "mass_changed": [],
            "jump_step": [],
        }
        for step in range(args.max_steps):
            if replay_actions is None:
                assert teacher is not None
                _, action = teacher.action()
            else:
                action = torch.as_tensor(
                    replay_actions[step : step + 1],
                    dtype=torch.float32,
                    device=base_env.device,
                )
            response_delta = torch.zeros_like(action)
            if (
                args.fixed_response != "none"
                and step >= args.fixed_response_frame
            ):
                ramp = min(
                    (step - args.fixed_response_frame + 1)
                    / float(args.fixed_response_ramp_frames),
                    1.0,
                )
                response_delta[0] = fixed_response_raw_delta * ramp
                action = action + response_delta
            observation, _, _, _, _ = env.step(action)
            patches = current_whole_hand_patch_features(base_env)
            patch_timestamp = current_whole_hand_patch_timestamps_s(base_env)
            oracle_speed = current_whole_hand_patch_oracle_tangential_speed(
                base_env
            )
            diagnostics = base_env._online_mass_jump_diagnostics
            slip = base_env._online_patch_slip_diagnostics
            slip_features = torch.stack(
                (
                    slip["slip_score"],
                    slip["incipient_slip"].to(torch.float32),
                    slip["gross_slip"].to(torch.float32),
                ),
                dim=-1,
            )
            rows["patch_features"].append(cpu(patches[0]))
            rows["slip_features"].append(cpu(slip_features[0]))
            rows["slip_state"].append(cpu_native(slip["state"][0]))
            rows["oracle_patch_tangential_speed_m_s"].append(
                cpu(oracle_speed[0])
            )
            rows["patch_sensor_timestamp_s"].append(cpu(patch_timestamp[0]))
            rows["actor_policy_observation"].append(cpu(observation["policy"][0]))
            rows["actor_patch_history"].append(
                cpu(observation["online_patch_tactile_history"][0])
            )
            rows["object_pos_w"].append(cpu(obj.data.root_pos_w[0]))
            rows["object_quat_w"].append(cpu(obj.data.root_quat_w[0]))
            rows["object_lin_vel_w"].append(cpu(obj.data.root_lin_vel_w[0]))
            rows["object_ang_vel_w"].append(cpu(obj.data.root_ang_vel_w[0]))
            rows["object_pos_b"].append(cpu(mdp.obj_pos_b(base_env, "motion")[0]))
            rows["object_ori_b"].append(cpu(mdp.obj_ori_b(base_env, "motion")[0]))
            rows["object_lin_vel_b"].append(
                cpu(mdp.obj_lin_vel_b(base_env, "motion")[0])
            )
            rows["object_ang_vel_b"].append(
                cpu(mdp.obj_ang_vel_b(base_env, "motion")[0])
            )
            rows["joint_pos"].append(cpu(robot.data.joint_pos[0]))
            rows["joint_vel"].append(cpu(robot.data.joint_vel[0]))
            rows["applied_action"].append(cpu(action[0]))
            rows["fixed_response_action_delta"].append(cpu(response_delta[0]))
            rows["mass_readback_kg"].append(float(diagnostics["mass_readback_kg"][0]))
            rows["inertia_readback_kg_m2"].append(
                cpu(diagnostics["inertia_readback_kg_m2"][0])
            )
            rows["target_factor"].append(float(diagnostics["target_factor"][0]))
            rows["qualified"].append(bool(diagnostics["qualified"][0]))
            rows["pending"].append(bool(diagnostics["pending"][0]))
            rows["pending_step"].append(int(diagnostics["pending_step"][0]))
            rows["jump_applied"].append(bool(diagnostics["jump_applied"][0]))
            rows["mass_changed"].append(bool(diagnostics["mass_changed"][0]))
            rows["jump_step"].append(int(diagnostics["jump_step"][0]))
            if world_camera is not None:
                rgb = cpu_native(world_camera.data.output["rgb"][0, ..., :3])
                if writer is None:
                    height, width = rgb.shape[:2]
                    writer = FfmpegRgbWriter(
                        output_root / "world_carrybox.mp4",
                        width,
                        height,
                        args.fps,
                    )
                writer.append(rgb)
            if step % 20 == 0 or step + 1 == args.max_steps:
                print(
                    "[PLAN15] live frame",
                    step + 1,
                    "/",
                    args.max_steps,
                    "mass_kg=",
                    rows["mass_readback_kg"][-1],
                    "jump=",
                    rows["jump_applied"][-1],
                    flush=True,
                )

        arrays = {name: np.asarray(values) for name, values in rows.items()}
        np.savez_compressed(output_root / "online_mass_jump_trace.npz", **arrays)
        contact = arrays["patch_features"][..., 0] > 0.5
        normal_load = arrays["patch_features"][..., 1]
        pressure = arrays["patch_features"][..., 2]
        jump_indices = np.flatnonzero(arrays["jump_applied"])
        first_jump_frame = int(jump_indices[0]) if len(jump_indices) else None
        jump_height_m = (
            None
            if first_jump_frame is None
            else float(arrays["object_pos_w"][first_jump_frame, 2])
        )
        minimum_post_jump_height_m = (
            None
            if first_jump_frame is None
            else float(arrays["object_pos_w"][first_jump_frame:, 2].min())
        )
        post_jump_frames = (
            0
            if first_jump_frame is None
            else int(len(arrays["object_pos_w"]) - first_jump_frame - 1)
        )
        outcome_window_complete = post_jump_frames >= 80
        maximum_post_jump_height_loss_m = (
            None
            if first_jump_frame is None
            else float(jump_height_m - minimum_post_jump_height_m)
        )
        bilateral = contact[:, 0].any(axis=-1) & contact[:, 1].any(axis=-1)
        bilateral_at_jump = (
            False if first_jump_frame is None else bool(bilateral[first_jump_frame])
        )
        pre_jump_bilateral_10 = False
        if first_jump_frame is not None and first_jump_frame >= 10:
            pre_jump_bilateral_10 = bool(
                np.all(bilateral[first_jump_frame - 10 : first_jump_frame])
            )
        patch_timestamps = arrays["patch_sensor_timestamp_s"]
        timestamp_spread_s = np.ptp(patch_timestamps, axis=(1, 2))
        mean_patch_timestamp_s = patch_timestamps.mean(axis=(1, 2))
        timestamp_steps_s = np.diff(mean_patch_timestamp_s)
        timestamp_synchronized = bool(np.max(timestamp_spread_s) <= 1.0e-6)
        timestamp_strictly_online = bool(
            len(timestamp_steps_s) > 0 and np.all(timestamp_steps_s > 0.0)
        )
        minimum_frame_advance_s = (
            None
            if len(timestamp_steps_s) == 0
            else float(np.min(timestamp_steps_s))
        )
        summary = {
            "schema": "plan15_online_patch_mass_jump_preflight_v1",
            "semantics": "live IsaacLab rollout; no learning; no offline replay",
            "action_source": (
                (
                    "online_frozen_official_refiner"
                    if args.fixed_response == "none"
                    else "online_frozen_official_refiner_plus_fixed_response"
                )
                if replay_actions is None
                else "fixed_nominal_applied_action_trace"
            ),
            "action_trace": (
                None if action_trace is None else str(action_trace)
            ),
            "fixed_response": {
                "name": args.fixed_response,
                "absolute_start_frame": int(args.fixed_response_frame),
                "ramp_frames": int(args.fixed_response_ramp_frames),
                "joint_target_delta_rad": fixed_response_joint_delta_rad,
                "reads_mass_factor": False,
                "reads_jump_flag": False,
                "reads_tactile": False,
                "reads_object_state": False,
            },
            "motion_id": int(args.motion_id),
            "seed": int(args.seed),
            "source_frames": int(args.max_steps),
            "nominal_mass_kg": float(arrays["mass_readback_kg"][0]),
            "target_mass_factor": float(args.mass_factor),
            "object_static_friction_readback": float(
                object_material[0, 0, 0].item()
            ),
            "object_dynamic_friction_readback": float(
                object_material[0, 0, 1].item()
            ),
            "first_jump_frame": first_jump_frame,
            "post_jump_frames": post_jump_frames,
            "minimum_required_post_jump_frames": 80,
            "outcome_window_complete": outcome_window_complete,
            "jump_height_m": jump_height_m,
            "minimum_post_jump_height_m": minimum_post_jump_height_m,
            "maximum_post_jump_height_loss_m": maximum_post_jump_height_loss_m,
            "post_jump_hold_5cm": (
                None
                if maximum_post_jump_height_loss_m is None or not outcome_window_complete
                else maximum_post_jump_height_loss_m <= 0.05
            ),
            "post_jump_drop_15cm": (
                None
                if maximum_post_jump_height_loss_m is None or not outcome_window_complete
                else maximum_post_jump_height_loss_m >= 0.15
            ),
            "mass_changed": bool(arrays["mass_changed"][-1]),
            "final_mass_readback_kg": float(arrays["mass_readback_kg"][-1]),
            "bilateral_contact_frames": int(
                np.count_nonzero(bilateral)
            ),
            "bilateral_contact_at_jump": bilateral_at_jump,
            "bilateral_contact_for_10_frames_before_jump": pre_jump_bilateral_10,
            "maximum_active_patches_left": int(contact[:, 0].sum(axis=-1).max()),
            "maximum_active_patches_right": int(contact[:, 1].sum(axis=-1).max()),
            "maximum_patch_normal_load_n": float(normal_load.max()),
            "maximum_patch_pressure_pa": float(pressure.max()),
            "maximum_object_lift_m": float(
                arrays["object_pos_w"][:, 2].max() - arrays["object_pos_w"][0, 2]
            ),
            "patch_sensor_clock": {
                "source": "official TacSL SensorBase._timestamp_last_update",
                "first_timestamp_s": float(mean_patch_timestamp_s[0]),
                "last_timestamp_s": float(mean_patch_timestamp_s[-1]),
                "maximum_bilateral_54_patch_skew_s": float(
                    np.max(timestamp_spread_s)
                ),
                "minimum_frame_advance_s": minimum_frame_advance_s,
                "all_54_patches_synchronized": timestamp_synchronized,
                "strictly_advances_each_control_frame": timestamp_strictly_online,
            },
            "actor_mass_observation": False,
            "actor_jump_flag_observation": False,
            "actor_measured_object_state": False,
            "patch_contract": online_patch_tactile_contract(),
            "base_patch_channels": list(BASE_PATCH_CHANNELS),
            "slip_callable_live": True,
            "world_video": (
                "world_carrybox.mp4" if args.record_world else None
            ),
        }
        (output_root / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        if first_jump_frame is None:
            raise RuntimeError("mass jump never triggered in the requested rollout")
        if not pre_jump_bilateral_10:
            raise RuntimeError(
                "CarryBox was not in bilateral TacSL contact for the ten frames "
                "before the lift-timed mass event"
            )
        if not timestamp_synchronized or not timestamp_strictly_online:
            raise RuntimeError(
                "official TacSL timestamps are not synchronized and online"
            )
    finally:
        if writer is not None:
            writer.close()
        if original_reset_idx is not None:
            env.unwrapped._reset_idx = original_reset_idx
        env.close()


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
