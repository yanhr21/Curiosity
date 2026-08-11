#!/usr/bin/env python3
"""Frozen CarryBox evaluation and tactile-input dependence audit."""

from __future__ import annotations

import argparse
import builtins
import json
from pathlib import Path
import subprocess

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--arm",
    choices=(
        "tactile",
        "zero",
        "bounded_tactile",
        "bounded_zero",
        "residual_tactile",
        "residual_zero",
    ),
    required=True,
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--teacher_checkpoint", type=Path, required=True)
parser.add_argument("--motion_folder", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=660)
parser.add_argument("--seed", type=int, default=13011)
parser.add_argument("--condition_label", default="nominal")
parser.add_argument("--mass_scale", type=float, default=1.0)
parser.add_argument("--static_friction", type=float, default=None)
parser.add_argument("--dynamic_friction", type=float, default=None)
parser.add_argument(
    "--actor_tactile_mode",
    choices=("live", "zeroed", "patch_permuted"),
    default="live",
    help="Evaluation-time actor input; the physical scene and checkpoint are unchanged.",
)
parser.add_argument("--tactile_permutation_seed", type=int, default=13012)
parser.add_argument(
    "--tactile_authority_scale",
    type=float,
    default=1.0,
    help=(
        "Evaluation-only multiplier on actor.0 tactile-feature columns after "
        "checkpoint loading; 0 exactly removes tactile action authority."
    ),
)
parser.add_argument(
    "--record_bundle",
    type=Path,
    default=None,
    help="Optional actual world plus physical-taxel recording directory.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.start_frame < 0 or args.max_steps < 1:
    parser.error("start_frame must be non-negative and max_steps must be positive")
if args.mass_scale <= 0.0:
    parser.error("mass_scale must be positive")
if not 0.0 <= args.tactile_authority_scale <= 1.0:
    parser.error("tactile_authority_scale must be in [0, 1]")
if (args.static_friction is None) != (args.dynamic_friction is None):
    parser.error("static_friction and dynamic_friction must be specified together")
if args.static_friction is not None and not (
    0.0 <= args.dynamic_friction <= args.static_friction
):
    parser.error("friction must satisfy 0 <= dynamic <= static")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio_ffmpeg
import numpy as np
import rsl_rl.algorithms
import rsl_rl.runners.on_policy_runner as on_policy_runner_module
import torch
from rsl_rl.runners import OnPolicyRunner

import sugar_rl.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from sugar_rl.utils.native_tactile_training_bcppo import (
    NativeTactileTrainingBCPPO,
)
from sugar_rl.utils.parser_cfg import parse_env_cfg
from sugar_rl.utils.reference_only_tactile_actor_critic import (
    ReferenceOnlyTactileActorCritic,
)
from sugar_rl.tasks.locomanip.native_whole_hand_tactile_history import (
    NATIVE_TACTILE_NORMAL_SCALE_N,
    NATIVE_TACTILE_SENSOR_NAMES,
    NATIVE_TACTILE_SHEAR_SCALE_N,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_official_refiner_anatomical_whole_hand_tacsl_env_cfg import (
    _review_camera,
)

from native_whole_hand_tactile_bcppo_task_registration import (
    register_native_whole_hand_tactile_bcppo_tasks,
)


TASKS = {
    "tactile": "Sugar-G129dof-CarryBox-NativeWholeHand-ProprioTaskTacSL-BCPPO",
    "zero": "Sugar-G129dof-CarryBox-NativeWholeHand-ProprioTaskZero-BCPPO",
    "bounded_tactile": (
        "Sugar-G129dof-CarryBox-BoundedNativeWholeHand-ProprioTaskTacSL-BCPPO"
    ),
    "bounded_zero": (
        "Sugar-G129dof-CarryBox-BoundedNativeWholeHand-ProprioTaskZero-BCPPO"
    ),
    "residual_tactile": (
        "Sugar-G129dof-CarryBox-ActionResidualNativeWholeHand-ProprioTaskTacSL-BCPPO"
    ),
    "residual_zero": (
        "Sugar-G129dof-CarryBox-ActionResidualNativeWholeHand-ProprioTaskZero-BCPPO"
    ),
}
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class FfmpegRgbWriter:
    """Write fixed-size RGB frames as browser-compatible H.264."""

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


def configure_runtime_classes() -> None:
    register_native_whole_hand_tactile_bcppo_tasks()
    setattr(builtins, "ReferenceOnlyTactileActorCritic", ReferenceOnlyTactileActorCritic)
    setattr(
        on_policy_runner_module,
        "ReferenceOnlyTactileActorCritic",
        ReferenceOnlyTactileActorCritic,
    )
    setattr(builtins, "NativeTactileTrainingBCPPO", NativeTactileTrainingBCPPO)
    setattr(rsl_rl.algorithms, "NativeTactileTrainingBCPPO", NativeTactileTrainingBCPPO)
    setattr(
        on_policy_runner_module,
        "NativeTactileTrainingBCPPO",
        NativeTactileTrainingBCPPO,
    )


def disable_randomization_and_pushes(env_cfg) -> list[str]:
    disabled = []
    for name in (
        "robot_physics_material",
        "obj_physics_material",
        "obj_mass",
        "add_joint_default_pos",
        "base_com",
        "push_robot",
        "push_object",
    ):
        if getattr(env_cfg.events, name, None) is not None:
            setattr(env_cfg.events, name, None)
            disabled.append(name)
    return disabled


def physical_taxel_counts(base_env) -> tuple[np.ndarray, np.ndarray]:
    counts = []
    maxima = []
    for side in ("left", "right"):
        side_count = torch.zeros(base_env.num_envs, dtype=torch.int64, device=base_env.device)
        side_max = torch.zeros(base_env.num_envs, dtype=torch.float32, device=base_env.device)
        for name, sensor in base_env.scene.sensors.items():
            if not name.startswith(f"{side}_") or not name.endswith("_tactile"):
                continue
            normal = sensor.data.tactile_normal_force
            side_count += torch.count_nonzero(normal.abs() > 1.0e-6, dim=-1)
            side_max = torch.maximum(side_max, normal.abs().amax(dim=-1))
        counts.append(side_count.detach().cpu().numpy())
        maxima.append(side_max.detach().cpu().numpy())
    return np.stack(counts, axis=-1), np.stack(maxima, axis=-1)


def force_reference_start(base_env, frame: int) -> dict[str, object]:
    command = base_env.command_manager.get_term("motion")
    if command.motion.num_motion != 1:
        raise RuntimeError(
            f"Expected exactly one motion in data_045 folder, got {command.motion.num_motion}"
        )
    total = int(command.motion.time_step_total_permotion[0].item())
    if frame >= total:
        raise ValueError(f"start frame {frame} outside the {total}-frame motion")

    def sample_frame(env_ids) -> None:
        ids = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(env_ids, dtype=torch.long, device=base_env.device)
        )
        command.motion_id[ids] = 0
        command.time_steps[ids] = frame
        command._use_motion_data[ids] = True

    command._sample_init_state = sample_frame
    base_env.reset()
    if int(command.motion_id[0]) != 0 or int(command.time_steps[0]) != frame:
        raise RuntimeError("Forced official motion frame did not survive reset")

    # Reset writes the correct physical state but the official relative-body
    # command buffer is updated only by _update_command.  Rebuild that buffer
    # without a physics step and finish at the requested frame.
    command.time_steps[:] = frame - 1
    command._update_command()
    if int(command.time_steps[0]) != frame:
        raise RuntimeError("Command-buffer synchronization changed the start frame")
    base_env.obs_buf = base_env.observation_manager.compute(update_history=False)
    return {"motion_id": 0, "start_frame": frame, "motion_frames": total}


def apply_object_condition(base_env) -> dict[str, object]:
    """Apply one explicit evaluation-only mass/material condition."""

    obj = base_env.scene["obj"]
    env_ids_cpu = torch.arange(base_env.num_envs, dtype=torch.int64, device="cpu")
    default_mass = obj.data.default_mass.detach().cpu().clone()
    default_inertia = obj.data.default_inertia.detach().cpu().clone()
    requested_mass = default_mass * float(args.mass_scale)
    requested_inertia = default_inertia * float(args.mass_scale)
    obj.root_physx_view.set_masses(requested_mass, env_ids_cpu)
    obj.root_physx_view.set_inertias(requested_inertia, env_ids_cpu)

    requested_material = obj.root_physx_view.get_material_properties().detach().cpu()
    if args.static_friction is not None:
        requested_material[..., 0] = float(args.static_friction)
        requested_material[..., 1] = float(args.dynamic_friction)
        obj.root_physx_view.set_material_properties(requested_material, env_ids_cpu)

    mass = obj.root_physx_view.get_masses().detach().cpu()
    inertia = obj.root_physx_view.get_inertias().detach().cpu()
    material = obj.root_physx_view.get_material_properties().detach().cpu()
    if not torch.allclose(mass, requested_mass, rtol=2.0e-6, atol=1.0e-7):
        raise RuntimeError("object mass readback does not match the requested condition")
    if not torch.allclose(inertia, requested_inertia, rtol=2.0e-6, atol=5.0e-8):
        raise RuntimeError("object inertia readback does not match the requested condition")
    if args.static_friction is not None:
        if not bool((material[..., 0] == float(args.static_friction)).all()):
            raise RuntimeError("object static-friction readback mismatch")
        if not bool((material[..., 1] == float(args.dynamic_friction)).all()):
            raise RuntimeError("object dynamic-friction readback mismatch")
    return {
        "label": args.condition_label,
        "mass_scale": float(args.mass_scale),
        "mass_kg": float(mass[0].sum().item()),
        "inertia_scaled_with_mass": True,
        "static_friction": float(material[0, 0, 0].item()),
        "dynamic_friction": float(material[0, 0, 1].item()),
        "friction_explicitly_overridden": args.static_friction is not None,
    }


def build_patch_permutations() -> tuple[torch.Tensor, dict[str, object]]:
    """Create fixed, independently permuted anatomical patch orders."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.tactile_permutation_seed))
    permutations = torch.stack(
        [torch.randperm(27, generator=generator) for _ in range(2)], dim=0
    )
    identity = torch.arange(27, dtype=torch.int64)
    metadata = {
        "seed": int(args.tactile_permutation_seed),
        "left": permutations[0].tolist(),
        "right": permutations[1].tolist(),
        "left_fixed_points": int((permutations[0] == identity).sum().item()),
        "right_fixed_points": int((permutations[1] == identity).sum().item()),
        "semantics": (
            "fixed independent permutation of the 27 anatomical patches in each "
            "hand; history, normal/shear channel, and 20x25 taxel coordinates "
            "inside each patch are preserved"
        ),
    }
    return permutations, metadata


def tactile_input_variants(
    raw: torch.Tensor,
    permutations_cpu: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return live, exact-zero, and fixed-patch-permuted actor inputs."""

    expected_width = 2 * 4 * 27 * 3 * 20 * 25
    if raw.ndim != 2 or raw.shape[-1] != expected_width:
        raise RuntimeError(
            f"unexpected native tactile actor tensor {tuple(raw.shape)}"
        )
    maps = raw.reshape(raw.shape[0], 2, 4, 27, 3, 20, 25)
    permuted = torch.empty_like(maps)
    for hand in range(2):
        permutation = permutations_cpu[hand].to(device=raw.device)
        permuted[:, hand] = maps[:, hand].index_select(2, permutation)
    return {
        "live": raw,
        "zeroed": torch.zeros_like(raw),
        "patch_permuted": permuted.reshape_as(raw),
    }


def main() -> None:
    configure_runtime_classes()
    task = TASKS[args.arm]
    checkpoint = args.checkpoint.expanduser().resolve()
    teacher = args.teacher_checkpoint.expanduser().resolve()
    motion_folder = args.motion_folder.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    record_bundle = (
        args.record_bundle.expanduser().resolve()
        if args.record_bundle is not None
        else None
    )
    if record_bundle is not None:
        if not record_bundle.is_relative_to((PROJECT_ROOT / "experiments").resolve()):
            raise ValueError("record_bundle must remain below experiments")
        if record_bundle.exists():
            raise FileExistsError(f"refusing to overwrite {record_bundle}")
        if not args.enable_cameras:
            raise ValueError("record_bundle requires --enable_cameras")
    for path in (checkpoint, teacher):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not motion_folder.is_dir():
        raise FileNotFoundError(motion_folder)

    env_cfg = parse_env_cfg(
        task,
        device=args.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args.seed
    env_cfg.commands.motion.motion_folder = str(motion_folder)
    env_cfg.commands.motion.rollout_traj = False
    env_cfg.commands.motion.init_with_ref = False
    env_cfg.commands.motion.start_init_env_ratio = 1.0
    if record_bundle is not None:
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
    disabled_events = disable_randomization_and_pushes(env_cfg)
    for group_name in ("policy", "native_whole_hand_tactile_history", "critic", "teacher"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None:
            group.enable_corruption = False

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.algorithm.teacher_ckpt = str(teacher)

    gym_env = gym.make(
        task,
        cfg=env_cfg,
        render_mode="rgb_array" if record_bundle is not None else None,
    )
    base_env = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    actor_first = runner.alg.policy.actor[0]
    actor_base_width = int(runner.alg.policy.num_actor_base_obs)
    if not isinstance(actor_first, torch.nn.Linear):
        raise RuntimeError("actor.0 is not the expected Linear layer")
    if actor_first.weight.shape[1] <= actor_base_width:
        raise RuntimeError("actor.0 has no appended tactile-feature columns")
    with torch.no_grad():
        actor_first.weight[:, actor_base_width:].mul_(
            float(args.tactile_authority_scale)
        )
    start = force_reference_start(base_env, args.start_frame)
    physical_condition = apply_object_condition(base_env)
    obs = env.get_observations()
    policy = runner.get_inference_policy(device=base_env.device)
    patch_permutations, patch_permutation_metadata = build_patch_permutations()
    command = base_env.command_manager.get_term("motion")
    obj = base_env.scene["obj"]
    robot = base_env.scene["robot"]

    initial = {
        "robot_root_state_w": robot.data.root_state_w.detach().cpu().numpy().copy(),
        "joint_pos": robot.data.joint_pos.detach().cpu().numpy().copy(),
        "joint_vel": robot.data.joint_vel.detach().cpu().numpy().copy(),
        "object_root_state_w": obj.data.root_state_w.detach().cpu().numpy().copy(),
        "last_action": base_env.action_manager.action.detach().cpu().numpy().copy(),
    }
    actor_tactile = obs["native_whole_hand_tactile_history"]
    initial_actor_tactile_nonzero = int(torch.count_nonzero(actor_tactile).item())

    record_writer: FfmpegRgbWriter | None = None
    record_rows: dict[str, list[np.ndarray]] | None = None
    record_world_path: Path | None = None
    if record_bundle is not None:
        record_bundle.mkdir(parents=True)
        record_world_path = record_bundle / "world_carrybox.mp4"
        record_writer = FfmpegRgbWriter(record_world_path, 1280, 720, 50)
        record_rows = {
            "normal_force": [],
            "signed_shear": [],
            "penetration": [],
            "object_state_w": [],
            "reward": [],
            "cumulative_reward_before_action": [],
            "raw_actor_tactile_nonzero_values": [],
            "fed_actor_tactile_nonzero_values": [],
            "same_state_live_zero_action_abs_max": [],
            "same_state_live_patch_permuted_action_abs_max": [],
        }
        world_camera = base_env.scene["world_camera"]
        base_env.sim.render()
        world_camera.update(0.0, force_recompute=True)

    def append_record_frame(
        raw_tactile: torch.Tensor,
        fed_tactile: torch.Tensor,
        same_state_actions: dict[str, torch.Tensor],
        cumulative_reward_value: float,
    ) -> None:
        if record_rows is None or record_writer is None:
            return
        normal = torch.stack(
            [
                base_env.scene[name].data.tactile_normal_force[0]
                for names in NATIVE_TACTILE_SENSOR_NAMES
                for name in names
            ],
            dim=0,
        ).reshape(2, 27, 20, 25)
        shear = torch.stack(
            [
                base_env.scene[name].data.tactile_shear_force[0]
                for names in NATIVE_TACTILE_SENSOR_NAMES
                for name in names
            ],
            dim=0,
        ).reshape(2, 27, 20, 25, 2)
        penetration = torch.stack(
            [
                base_env.scene[name].data.penetration_depth[0]
                for names in NATIVE_TACTILE_SENSOR_NAMES
                for name in names
            ],
            dim=0,
        ).reshape(2, 27, 20, 25)
        rgb = (
            base_env.scene["world_camera"]
            .data.output["rgb"][0, ..., :3]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8, copy=True)
        )
        record_writer.append(rgb)
        record_rows["normal_force"].append(
            normal.detach().cpu().numpy().astype(np.float32, copy=True)
        )
        record_rows["signed_shear"].append(
            shear.detach().cpu().numpy().astype(np.float32, copy=True)
        )
        record_rows["penetration"].append(
            penetration.detach().cpu().numpy().astype(np.float32, copy=True)
        )
        record_rows["object_state_w"].append(
            obj.data.root_state_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        record_rows["cumulative_reward_before_action"].append(
            np.asarray(cumulative_reward_value, dtype=np.float64)
        )
        record_rows["raw_actor_tactile_nonzero_values"].append(
            np.asarray(torch.count_nonzero(raw_tactile).item(), dtype=np.int64)
        )
        record_rows["fed_actor_tactile_nonzero_values"].append(
            np.asarray(torch.count_nonzero(fed_tactile).item(), dtype=np.int64)
        )
        record_rows["same_state_live_zero_action_abs_max"].append(
            np.asarray(
                (same_state_actions["live"] - same_state_actions["zeroed"])
                .abs()
                .max()
                .item(),
                dtype=np.float32,
            )
        )
        record_rows["same_state_live_patch_permuted_action_abs_max"].append(
            np.asarray(
                (
                    same_state_actions["live"]
                    - same_state_actions["patch_permuted"]
                )
                .abs()
                .max()
                .item(),
                dtype=np.float32,
            )
        )

    rows: dict[str, list[np.ndarray]] = {
        "action": [],
        "teacher_action": [],
        "same_state_action_live": [],
        "same_state_action_zeroed": [],
        "same_state_action_patch_permuted": [],
        "reward": [],
        "object_pos_w": [],
        "object_ref_pos_w": [],
        "object_pos_error": [],
        "physical_active_taxels": [],
        "physical_normal_abs_max_n": [],
        "reference_frame": [],
        "raw_actor_tactile_nonzero_values": [],
        "raw_actor_tactile_abs_max": [],
        "fed_actor_tactile_nonzero_values": [],
        "fed_actor_tactile_abs_max": [],
        "fed_actor_tactile_rms": [],
        "fed_actor_tactile_feature_abs_max": [],
        "fed_actor_tactile_feature_l2": [],
    }
    deferred_resets: list[int] = []
    original_reset_idx = base_env._reset_idx

    def defer_reset(env_ids: torch.Tensor) -> None:
        deferred_resets.extend(int(value) for value in env_ids.detach().cpu().tolist())

    base_env._reset_idx = defer_reset
    cumulative_reward = 0.0
    done = False
    try:
        for _ in range(args.max_steps):
            with torch.inference_mode():
                raw_tactile = obs["native_whole_hand_tactile_history"]
                tactile_variants = tactile_input_variants(
                    raw_tactile, patch_permutations
                )
                same_state_actions = {}
                for mode, tactile_input in tactile_variants.items():
                    counterfactual_obs = dict(obs)
                    counterfactual_obs["native_whole_hand_tactile_history"] = (
                        tactile_input
                    )
                    same_state_actions[mode] = policy(counterfactual_obs)
                fed_tactile = tactile_variants[args.actor_tactile_mode]
                action = same_state_actions[args.actor_tactile_mode]
                fed_features = runner.alg.policy.actor_tactile_encoder(fed_tactile)
                teacher_obs = torch.cat(
                    [obs[name] for name in runner.alg.policy.obs_groups["teacher"]],
                    dim=-1,
                )
                teacher_action = runner.alg.teacher_model(teacher_obs)
                append_record_frame(
                    raw_tactile,
                    fed_tactile,
                    same_state_actions,
                    cumulative_reward,
                )
                obs, reward, dones, _ = env.step(action)

            if record_rows is not None:
                record_rows["reward"].append(
                    reward[0].detach().cpu().numpy().astype(np.float32, copy=True)
                )

            active, normal_max = physical_taxel_counts(base_env)
            pos_error = torch.linalg.vector_norm(
                obj.data.root_pos_w - command.obj_ref_pos_w, dim=-1
            )
            rows["action"].append(action.detach().cpu().numpy().copy())
            rows["teacher_action"].append(teacher_action.detach().cpu().numpy().copy())
            rows["same_state_action_live"].append(
                same_state_actions["live"].detach().cpu().numpy().copy()
            )
            rows["same_state_action_zeroed"].append(
                same_state_actions["zeroed"].detach().cpu().numpy().copy()
            )
            rows["same_state_action_patch_permuted"].append(
                same_state_actions["patch_permuted"].detach().cpu().numpy().copy()
            )
            rows["reward"].append(reward.detach().cpu().numpy().copy())
            rows["object_pos_w"].append(obj.data.root_pos_w.detach().cpu().numpy().copy())
            rows["object_ref_pos_w"].append(command.obj_ref_pos_w.detach().cpu().numpy().copy())
            rows["object_pos_error"].append(pos_error.detach().cpu().numpy().copy())
            rows["physical_active_taxels"].append(active)
            rows["physical_normal_abs_max_n"].append(normal_max)
            rows["reference_frame"].append(command.time_steps.detach().cpu().numpy().copy())
            rows["raw_actor_tactile_nonzero_values"].append(
                torch.count_nonzero(raw_tactile, dim=-1).detach().cpu().numpy().copy()
            )
            rows["raw_actor_tactile_abs_max"].append(
                raw_tactile.abs().amax(dim=-1).detach().cpu().numpy().copy()
            )
            rows["fed_actor_tactile_nonzero_values"].append(
                torch.count_nonzero(fed_tactile, dim=-1).detach().cpu().numpy().copy()
            )
            rows["fed_actor_tactile_abs_max"].append(
                fed_tactile.abs().amax(dim=-1).detach().cpu().numpy().copy()
            )
            rows["fed_actor_tactile_rms"].append(
                fed_tactile.square().mean(dim=-1).sqrt().detach().cpu().numpy().copy()
            )
            rows["fed_actor_tactile_feature_abs_max"].append(
                fed_features.abs().amax(dim=-1).detach().cpu().numpy().copy()
            )
            rows["fed_actor_tactile_feature_l2"].append(
                torch.linalg.vector_norm(fed_features, dim=-1)
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            cumulative_reward += float(reward[0].item())
            done = bool(dones[0].item())
            if done:
                break
    finally:
        base_env._reset_idx = original_reset_idx

    if record_rows is not None:
        with torch.inference_mode():
            final_raw_tactile = obs["native_whole_hand_tactile_history"]
            final_tactile_variants = tactile_input_variants(
                final_raw_tactile, patch_permutations
            )
            final_same_state_actions = {}
            for mode, tactile_input in final_tactile_variants.items():
                final_obs = dict(obs)
                final_obs["native_whole_hand_tactile_history"] = tactile_input
                final_same_state_actions[mode] = policy(final_obs)
            append_record_frame(
                final_raw_tactile,
                final_tactile_variants[args.actor_tactile_mode],
                final_same_state_actions,
                cumulative_reward,
            )
        record_rows["reward"].append(np.asarray(0.0, dtype=np.float32))

    arrays = {name: np.stack(values, axis=0) for name, values in rows.items()}
    actions = arrays["action"][:, 0]
    teachers = arrays["teacher_action"][:, 0]
    action_abs_error = np.abs(actions - teachers)
    initial_z = float(initial["object_root_state_w"][0, 2])
    object_z = arrays["object_pos_w"][:, 0, 2]
    active = arrays["physical_active_taxels"][:, 0]
    raw_supported = arrays["raw_actor_tactile_nonzero_values"][:, 0] > 0
    same_state_live = arrays["same_state_action_live"][:, 0]
    same_state_zeroed = arrays["same_state_action_zeroed"][:, 0]
    same_state_permuted = arrays["same_state_action_patch_permuted"][:, 0]
    live_zero_action_delta = np.max(
        np.abs(same_state_live - same_state_zeroed), axis=-1
    )
    live_permuted_action_delta = np.max(
        np.abs(same_state_live - same_state_permuted), axis=-1
    )

    def supported_mean(values: np.ndarray) -> float:
        return float(values[raw_supported].mean()) if np.any(raw_supported) else 0.0
    termination_terms = {
        name: bool(base_env.termination_manager.get_term(name)[0].item())
        for name in base_env.termination_manager.active_terms
    }

    npz_path = output.with_suffix(".npz")
    np.savez_compressed(npz_path, **initial, **arrays)
    result = {
        "semantics": (
            f"frozen {args.condition_label} physical rollout from official "
            f"motion-45 frame with actor tactile mode {args.actor_tactile_mode}; "
            "no RGB, no actor-visible object state, no learning"
        ),
        "arm": args.arm,
        "actor_tactile_mode": args.actor_tactile_mode,
        "tactile_authority_scale": float(args.tactile_authority_scale),
        "patch_permutation": patch_permutation_metadata,
        "task": task,
        "checkpoint": str(checkpoint),
        "teacher_checkpoint": str(teacher),
        "seed": args.seed,
        "physical_condition": physical_condition,
        "disabled_events": disabled_events,
        "reference": start,
        "completed_steps": int(actions.shape[0]),
        "done": done,
        "deferred_reset_env_ids": deferred_resets,
        "termination_terms": termination_terms,
        "cumulative_reward": cumulative_reward,
        "student_teacher_action_mae": float(action_abs_error.mean()),
        "student_teacher_action_abs_max": float(action_abs_error.max()),
        "initial_actor_tactile_nonzero_values": initial_actor_tactile_nonzero,
        "raw_actor_tactile_supported_frames": int(np.count_nonzero(raw_supported)),
        "fed_actor_tactile_supported_frames": int(
            np.count_nonzero(arrays["fed_actor_tactile_nonzero_values"][:, 0] > 0)
        ),
        "same_state_live_vs_zeroed_action_abs_max": float(
            live_zero_action_delta.max()
        ),
        "same_state_live_vs_zeroed_action_abs_max_supported_mean": supported_mean(
            live_zero_action_delta
        ),
        "same_state_live_vs_patch_permuted_action_abs_max": float(
            live_permuted_action_delta.max()
        ),
        "same_state_live_vs_patch_permuted_action_abs_max_supported_mean": supported_mean(
            live_permuted_action_delta
        ),
        "maximum_relative_lift_m": float(np.max(object_z - initial_z)),
        "final_relative_lift_m": float(object_z[-1] - initial_z),
        "bilateral_physical_tactile_frames": int(np.count_nonzero(np.all(active > 0, axis=-1))),
        "maximum_active_taxels_left": int(active[:, 0].max()),
        "maximum_active_taxels_right": int(active[:, 1].max()),
        "final_object_position_error_m": float(arrays["object_pos_error"][-1, 0]),
        "trace": str(npz_path),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if record_rows is not None and record_writer is not None and record_bundle is not None:
        record_writer.close()
        record_writer = None
        record_trace_path = record_bundle / "tactile_trace.npz"
        np.savez_compressed(
            record_trace_path,
            **{name: np.stack(values) for name, values in record_rows.items()},
        )
        record_summary = {
            "schema": "native_whole_hand_tactile_policy_rollout_bundle_v1",
            "actor_tactile_mode": args.actor_tactile_mode,
            "tactile_authority_scale": float(args.tactile_authority_scale),
            "checkpoint": str(checkpoint),
            "evaluation_result": str(output),
            "frames": int(len(record_rows["normal_force"])),
            "executed_actions": int(actions.shape[0]),
            "includes_final_post_action_state": True,
            "fps": 50,
            "world_video": str(record_world_path),
            "tactile_trace": str(record_trace_path),
            "normal_display_scale_n": float(NATIVE_TACTILE_NORMAL_SCALE_N),
            "shear_display_scale_n": float(NATIVE_TACTILE_SHEAR_SCALE_N),
            "physical_condition": physical_condition,
            "display_boundary": (
                "The saved anatomical fields are the physical sensors. The actor "
                "input mode is recorded separately and may be live, zeroed, or "
                "patch-permuted."
            ),
        }
        (record_bundle / "summary.json").write_text(
            json.dumps(record_summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2), flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
