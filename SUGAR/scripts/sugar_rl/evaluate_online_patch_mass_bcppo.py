#!/usr/bin/env python3
"""Frozen online evaluation for one Plan-15 Z/P/PS checkpoint."""

from __future__ import annotations

import argparse
import builtins
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[3]
BRANCH_TASKS = {
    "Z": "Sugar-G129dof-CarryBox-OnlineMass-Patch-Z-BCPPO",
    "P": "Sugar-G129dof-CarryBox-OnlineMass-Patch-P-BCPPO",
    "PS": "Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-BCPPO",
}
FIXED_OVERFIT_TASK = "Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-Overfit-BCPPO"
FIXED_OVERFIT_AUDIT_TASK = (
    "Sugar-G129dof-CarryBox-OnlineMass-Patch-PS-Overfit-Audit-BCPPO"
)
EVALUATION_SEEDS = (152014, 152015, 152016)
TRAINING_SEEDS = (151014, 151015, 151016)
MASS_FACTORS = (1.0, 1.5, 3.0, 6.0, 10.0)
CORRECTED_SCALE_SCHEMA = (
    "plan15_live_patch_channel_scales_v3_extent_offset_calibrated"
)
OFFICIAL_REFINER = (
    ROOT
    / "experiments/sugar_reproduction/outputs/final/official_sugar/"
    "baseline/ckpts/refiner_model10000.pt"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--branch", choices=tuple(BRANCH_TASKS), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--patch-scale-file", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--seed", type=int, choices=EVALUATION_SEEDS, required=True)
parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS)
parser.add_argument("--mass-factor", type=float, choices=MASS_FACTORS, required=True)
parser.add_argument(
    "--motion-folder",
    type=Path,
    default=ROOT / "SUGAR/data/CarryBox/data_045",
)
parser.add_argument("--motion-id", type=int, default=0)
parser.add_argument("--profiles", type=int, default=20)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--max-steps", type=int, default=450)
parser.add_argument("--post-jump-window", type=int, default=80)
parser.add_argument(
    "--fixed-3x-overfit-gate",
    action="store_true",
    help=(
        "Use the exact fixed-condition overfit config: PS, 3x mass, "
        "20-frame delay and zero reset-pose noise."
    ),
)
parser.add_argument(
    "--audit-contact-forces",
    action="store_true",
    help="Record evaluator-only independent PhysX box/pad normal and friction forces.",
)
parser.add_argument(
    "--object-static-friction",
    type=float,
    default=None,
    help="Evaluation-only CarryBox static friction override.",
)
parser.add_argument(
    "--object-dynamic-friction",
    type=float,
    default=None,
    help="Evaluation-only CarryBox dynamic friction override.",
)
parser.add_argument(
    "--ignore-object-reference-termination",
    action="store_true",
    help=(
        "Diagnostic only: keep physical rollout alive when only object "
        "position/orientation reference tracking exceeds its threshold."
    ),
)
parser.add_argument(
    "--physical-outcome-view",
    action="store_true",
    help=(
        "Continue the same physical episode after any SUGAR reference "
        "termination while retaining the would-have-terminated terms as "
        "labels. Use this view for post-jump hold/drop/fall outcomes."
    ),
)
parser.add_argument(
    "--record-world",
    action="store_true",
    help=(
        "Record the synchronized world camera for one profile. This is the "
        "endpoint-video path; matched statistical sweeps remain camera-free."
    ),
)
parser.add_argument(
    "--record-profile-index",
    type=int,
    default=0,
    help="Environment index inside the selected batch to record.",
)
parser.add_argument(
    "--record-batch-index",
    type=int,
    default=0,
    help="Zero-based evaluation batch whose world camera is recorded.",
)
parser.add_argument("--fps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

host = socket.gethostname()
if host.startswith(("mgmtserver", "login")):
    raise SystemExit(f"refusing Plan-15 evaluation on login node {host}")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Plan-15 evaluation requires a retained Slurm allocation")
if os.environ.get("PLAN15_PIPELINE_LOCK_HELD") != "1":
    raise SystemExit(
        "Plan-15 evaluation requires the locked formal launcher "
        "(PLAN15_PIPELINE_LOCK_HELD=1)"
    )

if args.profiles < 1 or args.num_envs < 1 or args.profiles % args.num_envs:
    parser.error("profiles must be positive and divisible by num-envs")
if args.max_steps < args.post_jump_window:
    parser.error("max-steps must cover the post-jump window")
if args.ignore_object_reference_termination and args.physical_outcome_view:
    parser.error(
        "ignore-object-reference-termination and physical-outcome-view are "
        "mutually exclusive"
    )
if args.record_world and not 0 <= args.record_profile_index < args.num_envs:
    parser.error("record-profile-index must select an environment in the recorded batch")
if args.record_world and not 0 <= args.record_batch_index < args.profiles // args.num_envs:
    parser.error("record-batch-index must select an evaluation batch")
if args.fps < 1:
    parser.error("fps must be positive")
if args.fixed_3x_overfit_gate and (
    args.branch != "PS" or args.mass_factor != 3.0 or args.motion_id != 0
):
    parser.error("fixed 3x overfit gate requires branch PS, mass 3x and local motion 0")
if args.audit_contact_forces and not args.fixed_3x_overfit_gate:
    parser.error("contact-force audit is currently restricted to the fixed overfit gate")
if (args.object_static_friction is None) != (
    args.object_dynamic_friction is None
):
    parser.error(
        "object-static-friction and object-dynamic-friction must be specified together"
    )
if args.object_static_friction is not None and not (
    0.0
    <= args.object_dynamic_friction
    <= args.object_static_friction
    < float("inf")
):
    parser.error("object friction must satisfy 0 <= dynamic <= static")
if args.record_world:
    args.enable_cameras = True
    # Cluster H200 rendering requires the system NVIDIA Vulkan ICD. Without
    # this exact path Isaac Sim's renderer crashes during plugin startup even
    # though camera-free PhysX evaluation is healthy.
    vulkan_icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    if not vulkan_icd.is_file():
        raise FileNotFoundError(vulkan_icd)
    os.environ["VK_ICD_FILENAMES"] = str(vulkan_icd)

checkpoint = args.checkpoint.expanduser().resolve()
scale_file = args.patch_scale_file.expanduser().resolve()
output_root = args.output_root.expanduser().resolve()
for required_path in (checkpoint, scale_file, OFFICIAL_REFINER):
    if not required_path.is_file():
        raise FileNotFoundError(required_path)
if output_root.exists():
    raise FileExistsError(output_root)
scale_payload = json.loads(scale_file.read_text(encoding="utf-8"))
if scale_payload.get("schema") != CORRECTED_SCALE_SCHEMA:
    raise ValueError(
        "frozen Plan-15 evaluation requires corrected-force patch scales"
    )
scales = scale_payload.get("patch_channel_scales")
if not isinstance(scales, list) or len(scales) != 9:
    raise ValueError("patch scale file must contain nine patch_channel_scales")
scales = [float(value) for value in scales]
if any(not math.isfinite(value) or value <= 0.0 for value in scales):
    raise ValueError("patch channel scales must be positive and finite")
os.environ["SUGAR_ONLINE_PATCH_CHANNEL_SCALES"] = json.dumps(scales)
os.environ["SUGAR_INIT_AT_RANDOM_EP_LEN"] = "0"
os.environ["SUGAR_PLAN15_LIVE_HANDOFF"] = "1"
os.environ["SUGAR_PLAN15_HANDOFF_TEACHER_CKPT"] = str(OFFICIAL_REFINER)
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_STIFFNESS", "100")
os.environ.setdefault("CURIOSITY_ANATOMICAL_PHYSX_COMPLIANT_DAMPING", "20")
os.environ["CURIOSITY_ANATOMICAL_TACSL_CONTACT_OFFSET_M"] = "0.0003"
os.environ["CURIOSITY_ANATOMICAL_TACSL_NORMAL_STIFFNESS"] = "7294.8755"
os.environ["CURIOSITY_ANATOMICAL_TACSL_TANGENTIAL_STIFFNESS"] = "9"
os.environ.setdefault("CURIOSITY_ANATOMICAL_TACSL_FRICTION_COEFFICIENT", "0.5")
os.environ.setdefault(
    "CURIOSITY_TACSL_CALIBRATION_DIR",
    str(ROOT / "experiments/sugar_reproduction/assets/official_tacsl/calibration"),
)
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ.setdefault("CURIOSITY_ENABLE_ANATOMICAL27_WHOLE_HAND_TACSL_AUDIT", "1")
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
import rsl_rl.algorithms  # noqa: E402
import rsl_rl.runners.on_policy_runner as on_policy_runner_module  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from sugar_rl.tasks.locomanip.online_patch_tactile import (  # noqa: E402
    BASE_PATCH_CHANNELS,
    SENSOR_NAMES_BY_HAND,
    current_whole_hand_patch_features,
)
from sugar_rl.tasks.locomanip.patch_slip import PatchSlipDetector  # noqa: E402
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg import (  # noqa: E402
    _review_camera,
)
from sugar_rl.utils.online_patch_tactile_actor_critic import (  # noqa: E402
    OnlinePatchTactileActorCritic,
)
from sugar_rl.utils.rsl_rl_bcppo import BCPPO  # noqa: E402
from sugar_rl.utils.online_teacher_handoff_wrapper import (  # noqa: E402
    OnlineTeacherHandoffVecEnvWrapper,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_patch_mass_bcppo_task_registration import (  # noqa: E402
    register_online_patch_mass_bcppo_tasks,
)


def cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def observations(wrapper) -> dict[str, torch.Tensor]:
    value = wrapper.get_observations()
    return value[0] if isinstance(value, tuple) else value


class FfmpegRgbWriter:
    """Stream synchronized world frames directly to a playable H.264 file."""

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
        self.process.stdin.write(
            np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        )

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}")


def profile_summary(
    trace: dict[str, np.ndarray],
    profile: int,
    termination_names: tuple[str, ...],
) -> dict[str, object]:
    valid = trace["valid_frame"][:, profile]
    handoff_indices = np.flatnonzero(
        trace["handoff_active"][:, profile] & valid
    )
    handoff_frame = int(handoff_indices[0]) if len(handoff_indices) else None
    jump_indices = np.flatnonzero(trace["jump_applied"][:, profile] & valid)
    jump_frame = int(jump_indices[0]) if len(jump_indices) else None
    termination_indices = np.flatnonzero(
        trace["termination_any"][:, profile] & valid
    )
    termination_frame = (
        int(termination_indices[0]) if len(termination_indices) else None
    )
    termination_terms = (
        [
            name
            for index, name in enumerate(termination_names)
            if bool(trace["termination_terms"][termination_frame, profile, index])
        ]
        if termination_frame is not None
        else []
    )
    initial_z = float(trace["object_pos_w"][0, profile, 2])
    if jump_frame is None:
        return {
            "profile": profile,
            "handoff_frame": handoff_frame,
            "jump_frame": None,
            "first_termination_frame": termination_frame,
            "first_termination_terms": termination_terms,
            "eligible_post_jump_window": False,
            "strict_sugar_eligible_post_jump_window": False,
            "hold_success": False,
            "strict_sugar_hold_success": False,
            "drop": False,
            "safe_lower": False,
            "robot_fall": bool(trace["robot_fall"][:, profile][valid].any()),
            "reference_robot_deviation": bool(
                trace["reference_robot_deviation"][:, profile][valid].any()
            ),
            "bilateral_patch_contact_frames": int(
                np.count_nonzero(
                    trace["bilateral_patch_contact"][:, profile] & valid
                )
            ),
        }
    stop = min(jump_frame + int(args.post_jump_window), len(trace["object_pos_w"]))
    window_valid = valid[jump_frame:stop]
    invalid_indices = np.flatnonzero(~window_valid)
    if len(invalid_indices):
        stop = jump_frame + int(invalid_indices[0])
        window_valid = valid[jump_frame:stop]
    eligible = (
        stop - jump_frame == int(args.post_jump_window)
        and bool(window_valid.all())
    )
    strict_sugar_eligible = bool(
        eligible
        and (
            termination_frame is None
            or termination_frame
            >= jump_frame + int(args.post_jump_window) - 1
        )
    )
    z = trace["object_pos_w"][jump_frame:stop, profile, 2]
    vz = trace["object_lin_vel_w"][jump_frame:stop, profile, 2]
    jump_z = float(trace["object_pos_w"][jump_frame, profile, 2])
    height_loss = float(jump_z - np.min(z))
    robot_fall = bool(trace["robot_fall"][jump_frame:stop, profile].any())
    reference_robot_deviation = bool(
        trace["reference_robot_deviation"][jump_frame:stop, profile].any()
    )
    drop = bool(height_loss >= 0.15 or np.min(z) <= initial_z + 0.03)
    hold_success = bool(eligible and height_loss <= 0.05 and not robot_fall)
    strict_sugar_hold_success = bool(
        strict_sugar_eligible and height_loss <= 0.05 and not robot_fall
    )
    safe_lower = bool(
        eligible
        and not hold_success
        and not drop
        and not robot_fall
        and np.min(z) <= initial_z + 0.08
        and np.min(vz) >= -0.35
        and np.max(trace["reference_orientation_error_rad"][jump_frame:stop, profile]) <= 0.8
    )
    bilateral = trace["bilateral_patch_contact"][jump_frame:stop, profile]
    return {
        "profile": profile,
        "handoff_frame": handoff_frame,
        "jump_frame": jump_frame,
        "first_termination_frame": termination_frame,
        "first_termination_terms": termination_terms,
        "eligible_post_jump_window": eligible,
        "strict_sugar_eligible_post_jump_window": strict_sugar_eligible,
        "hold_success": hold_success,
        "strict_sugar_hold_success": strict_sugar_hold_success,
        "drop": drop,
        "safe_lower": safe_lower,
        "robot_fall": robot_fall,
        "reference_robot_deviation": reference_robot_deviation,
        "minimum_robot_root_height_m": float(
            np.min(trace["robot_root_height_m"][jump_frame:stop, profile])
        ),
        "minimum_robot_root_up_z": float(
            np.min(trace["robot_root_up_z"][jump_frame:stop, profile])
        ),
        "jump_height_m": jump_z,
        "minimum_post_jump_height_m": float(np.min(z)),
        "maximum_height_loss_m": height_loss,
        "maximum_reference_position_error_m": float(
            np.max(trace["reference_position_error_m"][jump_frame:stop, profile])
        ),
        "maximum_reference_orientation_error_rad": float(
            np.max(trace["reference_orientation_error_rad"][jump_frame:stop, profile])
        ),
        "bilateral_patch_contact_fraction": float(np.mean(bilateral)),
        "gross_slip_patch_fraction": float(
            np.mean(trace["slip_state"][jump_frame:stop, profile] == 3)
        ),
        "reward_sum": float(
            np.sum(
                trace["reward"][:, profile][
                    np.isfinite(trace["reward"][:, profile])
                ]
            )
        ),
    }


def main() -> None:
    register_online_patch_mass_bcppo_tasks()
    setattr(builtins, "BCPPO", BCPPO)
    setattr(rsl_rl.algorithms, "BCPPO", BCPPO)
    setattr(builtins, "OnlinePatchTactileActorCritic", OnlinePatchTactileActorCritic)
    setattr(on_policy_runner_module, "OnlinePatchTactileActorCritic", OnlinePatchTactileActorCritic)

    if args.audit_contact_forces:
        task = FIXED_OVERFIT_AUDIT_TASK
    elif args.fixed_3x_overfit_gate:
        task = FIXED_OVERFIT_TASK
    else:
        task = BRANCH_TASKS[args.branch]
    task_spec = gym.spec(task)
    env_cfg_type = task_spec.kwargs["play_env_cfg_entry_point"]
    module_name, class_name = env_cfg_type.split(":")
    module = __import__(module_name, fromlist=[class_name])
    env_cfg = getattr(module, class_name)()
    env_cfg.seed = int(args.seed)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.sim.device = args.device
    motion_folder = args.motion_folder.expanduser().resolve()
    if not motion_folder.is_dir():
        raise FileNotFoundError(motion_folder)
    env_cfg.commands.motion.motion_folder = str(motion_folder)
    if args.ignore_object_reference_termination:
        env_cfg.terminations.obj_pos = None
        env_cfg.terminations.obj_ori = None
    if args.record_world:
        env_cfg.scene.world_camera = _review_camera(
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
        env_cfg.sim.render_interval = env_cfg.decimation
    for term_name in ("reset_mass_jump", "step_mass_jump"):
        params = getattr(env_cfg.events, term_name).params
        params["mass_factors"] = (float(args.mass_factor),)
        params["seed"] = int(args.seed)
    mass_jump_delay_frames = tuple(
        int(value) for value in env_cfg.events.step_mass_jump.params["delay_frames"]
    )
    pose_ranges = tuple(env_cfg.commands.motion.pose_range.values())
    joint_position_range = env_cfg.commands.motion.joint_position_range
    reset_pose_noise_disabled = bool(
        all(tuple(value) == (0.0, 0.0) for value in pose_ranges)
        and tuple(joint_position_range) == (0.0, 0.0)
    )
    if args.fixed_3x_overfit_gate and (
        mass_jump_delay_frames != (20, 20) or not reset_pose_noise_disabled
    ):
        raise RuntimeError("fixed overfit evaluation config was not preserved")
    for group_name in (
        "policy",
        "online_patch_tactile_history",
        "critic",
        "teacher",
        "training_handoff_mask",
    ):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None:
            group.enable_corruption = False

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = int(args.seed)
    agent_cfg.device = args.device
    agent_cfg.algorithm.teacher_ckpt = str(OFFICIAL_REFINER)
    gym_env = gym.make(
        task,
        cfg=env_cfg,
        render_mode="rgb_array" if args.record_world else None,
    )
    base_env = gym_env.unwrapped
    env = OnlineTeacherHandoffVecEnvWrapper(
        gym_env,
        clip_actions=agent_cfg.clip_actions,
        teacher_checkpoint=OFFICIAL_REFINER,
    )
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=base_env.device)
    output_root.mkdir(parents=True)
    obj = base_env.scene["obj"]
    robot = base_env.scene["robot"]
    command = base_env.command_manager.get_term("motion")
    world_camera = base_env.scene["world_camera"] if args.record_world else None
    physx_pad_contact = (
        base_env.scene["all_pads_box_contact"]
        if args.audit_contact_forces
        else None
    )
    physx_box_all_normal = (
        base_env.scene["box_all_normal_contact"]
        if args.audit_contact_forces
        else None
    )
    tacsl_audit_sensors = (
        [
            base_env.scene[name]
            for names in SENSOR_NAMES_BY_HAND
            for name in names
        ]
        if args.audit_contact_forces
        else []
    )
    world_writer = None
    friction_readback: np.ndarray | None = None
    robot_friction_readback: np.ndarray | None = None
    tacsl_feature_read_calls = 0
    slip_detector_update_calls = 0

    def fixed_start(env_ids) -> None:
        ids = torch.as_tensor(
            env_ids, dtype=torch.int64, device=base_env.device
        )
        command.motion_id[ids] = int(args.motion_id)
        command.time_steps[ids] = 0
        command._use_motion_data[ids] = True

    command._sample_init_state = fixed_start
    termination_names = tuple(base_env.termination_manager.active_terms)
    reference_robot_deviation_names = tuple(
        name
        for name in termination_names
        if name in {"anchor_ori", "anchor_pos", "ee_body_pos"}
    )
    original_termination_compute = base_env.termination_manager.compute
    if args.physical_outcome_view:
        termination_manager = base_env.termination_manager

        def compute_without_reference_reset() -> torch.Tensor:
            """Evaluate every strict term, but do not reset the physical state."""

            original_termination_compute()
            termination_manager._terminated_buf.zero_()
            termination_manager._truncated_buf.zero_()
            return torch.zeros(
                base_env.num_envs, dtype=torch.bool, device=base_env.device
            )

        termination_manager.compute = compute_without_reference_reset

    total_profiles = int(args.profiles)
    traces: dict[str, list[np.ndarray]] = {
        "action": [],
        "policy_action": [],
        "teacher_control": [],
        "handoff_active": [],
        "reward": [],
        "object_pos_w": [],
        "object_lin_vel_w": [],
        "reference_position_error_m": [],
        "reference_orientation_error_rad": [],
        "patch_features": [],
        "slip_state": [],
        "bilateral_patch_contact": [],
        "jump_applied": [],
        "mass_changed": [],
        "mass_readback_kg": [],
        "robot_fall": [],
        "reference_robot_deviation": [],
        "robot_root_height_m": [],
        "robot_root_up_z": [],
        "termination_any": [],
        "termination_terms": [],
        "reference_frame": [],
        "valid_frame": [],
    }
    if args.audit_contact_forces:
        traces.update(
            {
                "physx_box_normal_force_from_all_pads_w": [],
                "physx_box_friction_force_from_all_pads_w": [],
                "physx_box_normal_force_per_pad_w": [],
                "physx_box_friction_force_per_pad_w": [],
                "physx_box_contact_position_per_pad_w": [],
                "physx_contact_to_nearest_taxel_distance_m": [],
                "tacsl_active_taxel_count_per_pad": [],
                "tacsl_min_signed_distance_per_pad_m": [],
                "physx_box_all_contact_normal_force_w": [],
            }
        )
    original_reset_idx = base_env._reset_idx
    try:
        for batch in range(total_profiles // int(args.num_envs)):
            print(
                f"[PLAN15 EVAL] batch {batch + 1}/"
                f"{total_profiles // int(args.num_envs)} reset begin",
                flush=True,
            )
            base_env._reset_idx = original_reset_idx
            with torch.inference_mode():
                reset_value = env.reset()
                if args.physical_outcome_view:
                    base_env.termination_manager._term_dones.zero_()
                material = obj.root_physx_view.get_material_properties().detach().cpu()
                if args.object_static_friction is not None:
                    material[..., 0] = float(args.object_static_friction)
                    material[..., 1] = float(args.object_dynamic_friction)
                    env_ids_cpu = torch.arange(
                        base_env.num_envs, dtype=torch.int64, device="cpu"
                    )
                    obj.root_physx_view.set_material_properties(material, env_ids_cpu)
                    material = (
                        obj.root_physx_view.get_material_properties().detach().cpu()
                    )
                    if not bool(
                        (material[..., 0] == float(args.object_static_friction)).all()
                    ):
                        raise RuntimeError("CarryBox static-friction readback mismatch")
                    if not bool(
                        (material[..., 1] == float(args.object_dynamic_friction)).all()
                    ):
                        raise RuntimeError("CarryBox dynamic-friction readback mismatch")
                current_friction = material[..., :2].numpy().copy()
                if friction_readback is None:
                    friction_readback = current_friction
                elif not np.array_equal(friction_readback, current_friction):
                    raise RuntimeError("CarryBox friction changed between profile batches")
                current_robot_friction = (
                    robot.root_physx_view.get_material_properties()
                    .detach()
                    .cpu()[..., :2]
                    .numpy()
                    .copy()
                )
                if not np.all(current_robot_friction == 0.5):
                    raise RuntimeError("G1/pad friction is not fixed at TacSL mu=0.5")
                if robot_friction_readback is None:
                    robot_friction_readback = current_robot_friction
                elif not np.array_equal(
                    robot_friction_readback, current_robot_friction
                ):
                    raise RuntimeError("G1/pad friction changed between profile batches")
            print(
                f"[PLAN15 EVAL] batch {batch + 1}/"
                f"{total_profiles // int(args.num_envs)} reset complete",
                flush=True,
            )
            obs = reset_value[0] if isinstance(reset_value, tuple) else reset_value
            if not bool((command.motion_id == int(args.motion_id)).all()):
                raise RuntimeError("frozen evaluation motion id did not survive reset")
            if not bool((command.time_steps == 0).all()):
                raise RuntimeError("frozen evaluation did not start from motion frame 0")

            # Reset writes the requested physical motion state, but the official
            # SUGAR command's relative-body buffers are refreshed only by
            # _update_command().  Synchronize them before the first policy
            # observation, following the repository's frozen evaluator path.
            command.time_steps[:] = -1
            command._update_command()
            if not bool((command.time_steps == 0).all()):
                raise RuntimeError("command-buffer synchronization changed frame 0")
            reset_anchor_error = torch.linalg.vector_norm(
                command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1
            )
            print(
                f"[PLAN15 EVAL] batch {batch + 1}/"
                f"{total_profiles // int(args.num_envs)} frame0 anchor errors "
                f"{cpu(reset_anchor_error).tolist()}",
                flush=True,
            )
            base_env.obs_buf = base_env.observation_manager.compute(
                update_history=False
            )
            obs = observations(env)
            initial_robot_root_height = robot.data.root_pos_w[:, 2].clone()
            base_env._reset_idx = lambda env_ids: None
            # Z is the exact-zero sensing control.  Its actor already receives
            # zero patch/slip tensors, and the evaluator must not silently
            # read TacSL merely to populate plots or outcome labels.  P/PS may
            # recompute causal slip as an evaluator-only label; it is never
            # fed back into either actor.
            detector = (
                None
                if args.branch == "Z"
                else PatchSlipDetector(base_env.num_envs, device=base_env.device)
            )
            done_latched = torch.zeros(
                base_env.num_envs, dtype=torch.bool, device=base_env.device
            )
            batch_rows = {name: [] for name in traces}
            for step in range(int(args.max_steps)):
                active_before_step = ~done_latched
                policy_obs = {
                    name: torch.where(
                        done_latched.reshape(
                            (base_env.num_envs,)
                            + (1,) * (value.ndim - 1)
                        ),
                        torch.zeros_like(value),
                        value,
                    )
                    for name, value in obs.items()
                }
                with torch.inference_mode():
                    action = policy(policy_obs)
                    action = torch.where(
                        done_latched[:, None], torch.zeros_like(action), action
                    )
                    obs, reward, done, _ = env.step(action)
                executed_action = env.last_executed_action
                teacher_control = env.last_teacher_control_mask
                if args.branch == "Z":
                    patch = torch.zeros(
                        (
                            base_env.num_envs,
                            2,
                            27,
                            len(BASE_PATCH_CHANNELS),
                        ),
                        dtype=torch.float32,
                        device=base_env.device,
                    )
                    slip_state = torch.zeros(
                        (base_env.num_envs, 2, 27),
                        dtype=torch.long,
                        device=base_env.device,
                    )
                else:
                    if detector is None:
                        raise RuntimeError("P/PS evaluation requires a slip detector")
                    patch = current_whole_hand_patch_features(base_env)
                    tacsl_feature_read_calls += 1
                    timestamp = torch.full(
                        (base_env.num_envs,),
                        (step + 1) * float(base_env.step_dt),
                        dtype=torch.float32,
                        device=base_env.device,
                    )
                    slip = detector.update(
                        contact=patch[..., 0].bool(),
                        normal_load_n=patch[..., 1],
                        mean_pressure_pa=patch[..., 2],
                        shear_xy_n=patch[..., 3:5],
                        friction_utilization=patch[..., 5],
                        timestamp_s=timestamp,
                        reset_mask=torch.full(
                            (base_env.num_envs,),
                            step == 0,
                            dtype=torch.bool,
                            device=base_env.device,
                        ),
                    )
                    slip_detector_update_calls += 1
                    slip_state = slip.state
                diagnostics = base_env._online_mass_jump_diagnostics
                ref_pos_error = torch.linalg.vector_norm(
                    command.obj_pos_w - command.obj_ref_pos_w, dim=-1
                )
                dot = torch.sum(command.obj_quat_w * command.obj_ref_quat_w, dim=-1).abs()
                ref_ori_error = 2.0 * torch.acos(torch.clamp(dot, 0.0, 1.0))
                termination_terms = torch.stack(
                    [
                        base_env.termination_manager.get_term(name)
                        for name in termination_names
                    ],
                    dim=-1,
                )
                strict_termination = termination_terms.any(dim=-1)
                reference_robot_deviation = torch.zeros_like(
                    done, dtype=torch.bool
                )
                for name in reference_robot_deviation_names:
                    reference_robot_deviation |= (
                        base_env.termination_manager.get_term(name)
                    )
                robot_root_height = robot.data.root_pos_w[:, 2]
                robot_root_quat = robot.data.root_quat_w
                robot_root_up_z = 1.0 - 2.0 * (
                    robot_root_quat[:, 1].square()
                    + robot_root_quat[:, 2].square()
                )
                robot_fall = (
                    robot_root_height < initial_robot_root_height - 0.35
                ) | (robot_root_up_z < 0.5)
                batch_rows["action"].append(cpu(executed_action))
                batch_rows["policy_action"].append(cpu(action))
                batch_rows["teacher_control"].append(cpu(teacher_control))
                batch_rows["handoff_active"].append(
                    cpu(~teacher_control)
                )
                batch_rows["reward"].append(cpu(reward))
                batch_rows["object_pos_w"].append(cpu(obj.data.root_pos_w))
                batch_rows["object_lin_vel_w"].append(cpu(obj.data.root_lin_vel_w))
                batch_rows["reference_position_error_m"].append(cpu(ref_pos_error))
                batch_rows["reference_orientation_error_rad"].append(cpu(ref_ori_error))
                batch_rows["patch_features"].append(cpu(patch))
                batch_rows["slip_state"].append(cpu(slip_state))
                batch_rows["bilateral_patch_contact"].append(
                    cpu((patch[..., 0] > 0.5).any(dim=-1).all(dim=-1))
                )
                batch_rows["jump_applied"].append(cpu(diagnostics["jump_applied"]))
                batch_rows["mass_changed"].append(cpu(diagnostics["mass_changed"]))
                batch_rows["mass_readback_kg"].append(cpu(diagnostics["mass_readback_kg"]))
                batch_rows["robot_fall"].append(cpu(robot_fall))
                batch_rows["reference_robot_deviation"].append(
                    cpu(reference_robot_deviation)
                )
                batch_rows["robot_root_height_m"].append(cpu(robot_root_height))
                batch_rows["robot_root_up_z"].append(cpu(robot_root_up_z))
                batch_rows["termination_any"].append(cpu(strict_termination))
                batch_rows["termination_terms"].append(cpu(termination_terms))
                batch_rows["reference_frame"].append(cpu(command.time_steps))
                batch_rows["valid_frame"].append(cpu(active_before_step))
                if physx_pad_contact is not None:
                    batch_rows["physx_box_normal_force_from_all_pads_w"].append(
                        cpu(physx_pad_contact.data.force_matrix_w.sum(dim=(1, 2)))
                    )
                    batch_rows["physx_box_friction_force_from_all_pads_w"].append(
                        cpu(physx_pad_contact.data.friction_forces_w.sum(dim=(1, 2)))
                    )
                    batch_rows["physx_box_normal_force_per_pad_w"].append(
                        cpu(physx_pad_contact.data.force_matrix_w[:, 0])
                    )
                    batch_rows["physx_box_friction_force_per_pad_w"].append(
                        cpu(physx_pad_contact.data.friction_forces_w[:, 0])
                    )
                    contact_position = physx_pad_contact.data.contact_pos_w[:, 0]
                    taxel_positions = torch.stack(
                        [sensor.data.tactile_points_pos_w for sensor in tacsl_audit_sensors],
                        dim=1,
                    )
                    active_taxels = torch.stack(
                        [sensor.data.penetration_depth > 0.0 for sensor in tacsl_audit_sensors],
                        dim=1,
                    )
                    minimum_signed_distance = torch.stack(
                        [
                            sensor.data.tactile_signed_distance_m.amin(dim=-1)
                            for sensor in tacsl_audit_sensors
                        ],
                        dim=1,
                    )
                    contact_is_finite = torch.isfinite(contact_position).all(dim=-1)
                    nearest_taxel = torch.linalg.vector_norm(
                        taxel_positions - contact_position.unsqueeze(-2), dim=-1
                    ).amin(dim=-1)
                    nearest_taxel = torch.where(
                        contact_is_finite,
                        nearest_taxel,
                        torch.full_like(nearest_taxel, float("nan")),
                    )
                    batch_rows["physx_box_contact_position_per_pad_w"].append(
                        cpu(contact_position)
                    )
                    batch_rows[
                        "physx_contact_to_nearest_taxel_distance_m"
                    ].append(cpu(nearest_taxel))
                    batch_rows["tacsl_active_taxel_count_per_pad"].append(
                        cpu(active_taxels.sum(dim=-1))
                    )
                    batch_rows["tacsl_min_signed_distance_per_pad_m"].append(
                        cpu(minimum_signed_distance)
                    )
                    batch_rows["physx_box_all_contact_normal_force_w"].append(
                        cpu(physx_box_all_normal.data.net_forces_w.sum(dim=1))
                    )
                if world_camera is not None and batch == int(args.record_batch_index):
                    rgb = cpu(
                        world_camera.data.output["rgb"][
                            int(args.record_profile_index), ..., :3
                        ]
                    ).astype(np.uint8)
                    if world_writer is None:
                        height, width = rgb.shape[:2]
                        world_writer = FfmpegRgbWriter(
                            output_root / "world_carrybox.mp4",
                            width,
                            height,
                            int(args.fps),
                        )
                    world_writer.append(rgb)
                if not args.physical_outcome_view:
                    done_latched |= done.bool()
            for name in traces:
                traces[name].append(np.stack(batch_rows[name], axis=0))
            print(
                f"[PLAN15 EVAL] batch {batch + 1}/{total_profiles // int(args.num_envs)} complete",
                flush=True,
            )
    finally:
        if world_writer is not None:
            world_writer.close()
        base_env.termination_manager.compute = original_termination_compute
        base_env._reset_idx = original_reset_idx
        env.close()

    arrays = {
        name: np.concatenate(values, axis=1)
        for name, values in traces.items()
    }
    if any(value.shape[1] != total_profiles for value in arrays.values()):
        raise RuntimeError("profile count mismatch in frozen evaluation trace")
    np.savez_compressed(output_root / "frozen_evaluation_trace.npz", **arrays)
    episodes = [
        profile_summary(arrays, index, termination_names)
        for index in range(total_profiles)
    ]
    eligible = [item for item in episodes if item["eligible_post_jump_window"]]
    strict_sugar_eligible = [
        item
        for item in episodes
        if item["strict_sugar_eligible_post_jump_window"]
    ]
    summary = {
        "schema": "plan15_frozen_online_patch_mass_evaluation_v3_live_handoff",
        "branch": args.branch,
        "checkpoint": str(checkpoint),
        "training_seed": args.training_seed,
        "seed": int(args.seed),
        "mass_factor": float(args.mass_factor),
        "object_friction_explicitly_overridden": (
            args.object_static_friction is not None
        ),
        "object_static_friction_readback": (
            friction_readback[..., 0].tolist()
            if friction_readback is not None
            else None
        ),
        "object_dynamic_friction_readback": (
            friction_readback[..., 1].tolist()
            if friction_readback is not None
            else None
        ),
        "robot_pad_material_contract": {
            "all_robot_shapes_static_friction": 0.5,
            "all_robot_shapes_dynamic_friction": 0.5,
            "anatomical_pad_combine_mode": "average",
            "effective_pad_box_friction": 0.5,
        },
        "motion_id": int(args.motion_id),
        "motion_folder": str(motion_folder),
        "start_frame": 0,
        "profiles": total_profiles,
        "num_envs": int(args.num_envs),
        "max_steps": int(args.max_steps),
        "post_jump_window_frames": int(args.post_jump_window),
        "fixed_3x_overfit_gate": bool(args.fixed_3x_overfit_gate),
        "evaluator_only_physx_contact_force_audit": bool(
            args.audit_contact_forces
        ),
        "tacsl_effective_grid_margin_m_by_pad": (
            [
                float(sensor.tactile_grid_effective_margin_m)
                for sensor in tacsl_audit_sensors
            ]
            if args.audit_contact_forces
            else None
        ),
        "mass_jump_delay_frames": list(mass_jump_delay_frames),
        "reset_pose_noise_disabled": reset_pose_noise_disabled,
        "actor_mass_or_jump_input": False,
        "online_teacher_handoff": True,
        "pre_handoff_actor_control": False,
        # The branch actor consumes its own live rollout observation. The
        # separately recomputed evaluator tensors below are outcome labels and
        # are never fed back into the policy.
        "actor_receives_live_patch_observation": args.branch in {"P", "PS"},
        "actor_receives_causal_slip_observation": args.branch == "PS",
        "evaluator_reads_tacsl": tacsl_feature_read_calls > 0,
        "evaluator_tacsl_feature_read_calls": tacsl_feature_read_calls,
        "evaluator_slip_detector_update_calls": slip_detector_update_calls,
        "trace_patch_semantics": (
            "exact_zero_control"
            if args.branch == "Z"
            else "online_tacsl_evaluator_label"
        ),
        "evaluation_recomputed_patch_slip_labels_feed_actor": False,
        "diagnostic_ignore_object_reference_termination": bool(
            args.ignore_object_reference_termination
        ),
        "physical_outcome_view": bool(args.physical_outcome_view),
        "evaluation_view": (
            "physical_outcome"
            if args.physical_outcome_view
            else (
                "object_reference_termination_ablation"
                if args.ignore_object_reference_termination
                else "strict_sugar_reference"
            )
        ),
        "diagnostic_only": bool(
            args.ignore_object_reference_termination
            or args.physical_outcome_view
        ),
        "formal_termination_contract": not bool(
            args.ignore_object_reference_termination
            or args.physical_outcome_view
        ),
        "physical_outcome_contract": bool(args.physical_outcome_view),
        "physical_robot_fall_definition": (
            "root height loss >= 0.35 m or root up-axis world-z < 0.5"
        ),
        "termination_term_names": list(termination_names),
        "world_video": "world_carrybox.mp4" if args.record_world else None,
        "world_video_fps": int(args.fps) if args.record_world else None,
        "world_video_profile_index": (
            int(args.record_batch_index) * int(args.num_envs)
            + int(args.record_profile_index)
            if args.record_world
            else None
        ),
        "world_video_batch_index": (
            int(args.record_batch_index) if args.record_world else None
        ),
        "policy_spatial_unit": "27 physical patches per hand; no taxel policy units",
        "eligible_profiles": len(eligible),
        "strict_sugar_eligible_profiles": len(strict_sugar_eligible),
        "hold_success_count": sum(bool(item["hold_success"]) for item in eligible),
        "strict_sugar_hold_success_count": sum(
            bool(item["strict_sugar_hold_success"])
            for item in strict_sugar_eligible
        ),
        "drop_count": sum(bool(item["drop"]) for item in eligible),
        "safe_lower_count": sum(bool(item["safe_lower"]) for item in eligible),
        "robot_fall_count": sum(bool(item["robot_fall"]) for item in episodes),
        "reference_robot_deviation_count": sum(
            bool(item["reference_robot_deviation"]) for item in episodes
        ),
        "episodes": episodes,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(
            f"[PLAN15 EVAL] uncaught {type(error).__name__}: {error!r}",
            flush=True,
        )
        try:
            simulation_app.close()
        except BaseException:
            pass
        # Isaac Kit installs process-level exception handling that can turn a
        # re-raised Python exception into exit code 0 during shutdown.  A
        # formal launcher must never accept an exception as a successful
        # evaluation, so leave with an unambiguous nonzero status after the
        # diagnostic line has been flushed.
        os._exit(1)
    else:
        simulation_app.close()
