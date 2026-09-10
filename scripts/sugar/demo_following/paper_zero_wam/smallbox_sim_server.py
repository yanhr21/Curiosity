#!/usr/bin/env python3
"""Isaac Sim server for paired paper Zero-WAM SMALLBOX rollouts."""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing.connection import Listener
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SUGAR = ROOT / "SUGAR"
SUGAR_SCRIPT = SUGAR / "scripts/sugar_rl"
sys.path.insert(0, str(SUGAR_SCRIPT))

os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(SUGAR / "descriptions/terrain/sugar_ground_plane.usda"),
)
os.environ.setdefault("ISAACLAB_USE_LOCAL_FRAME_MARKER", "1")
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
job_id = os.environ.get("SLURM_JOB_ID", "local")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
os.environ.setdefault(
    "ISAACLAB_TMP_ROOT", f"/tmp/paper_zero_wam_sim_{job_id}_{array_id}"
)
os.environ.setdefault(
    "SUGAR_UNITREE_TMP_ROOT",
    f"/tmp/paper_zero_wam_unitree_{job_id}_{array_id}",
)

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--address", type=Path, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--copies-per-profile", type=int, choices=(2, 4), default=2)
parser.add_argument(
    "--route",
    choices=("model", "CarryBox_endpoint", "KickBox_endpoint"),
    default="model",
)
parser.add_argument("--contact-threshold-n", type=float, default=0.1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 8:
    parser.error("the paired FSDP physical batch requires exactly eight environments")
args.task = "Sugar-G129dof-CarryBox-Inference"
args.enable_cameras = True
args.device = "cuda:0"
args.kit_args = " ".join(
    part
    for part in (
        getattr(args, "kit_args", ""),
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/maxGpuCount=1",
    )
    if part
)
app_launcher = AppLauncher(args, multi_gpu=False, max_gpu_count=1)
simulation_app = app_launcher.app

import builtins  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import rsl_rl.algorithms  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import sugar_rl.tasks  # noqa: F401,E402
from sugar_rl.utils.parser_cfg import parse_env_cfg  # noqa: E402
from sugar_rl.utils.rsl_rl_bcppo import BCPPO  # noqa: E402
from .physical_metrics import RESTORE_PAYLOAD_KEYS  # noqa: E402


setattr(builtins, "BCPPO", BCPPO)
setattr(rsl_rl.algorithms, "BCPPO", BCPPO)

AUTHKEY = b"paper-zero-wam-smallbox-v1"
CONTROL_HZ = 50
RGB_STRIDE = 5
STATE_READBACK_TOLERANCE_M = 1.0e-5
SENSOR_NAMES = (
    "left_hand_forces",
    "right_hand_forces",
    "left_foot_forces",
    "right_foot_forces",
)
ENDPOINTS = {
    "CarryBox_endpoint": {
        "motion_folder": SUGAR / "data/CarryBox/data_045",
        "generator": SUGAR / "demo_ckpts/CarryBox/generator.ckpt",
        "tracker": SUGAR / "demo_ckpts/CarryBox/tracker.pt",
    },
    "KickBox_endpoint": {
        "motion_folder": SUGAR / "data/KickBox/data_021",
        "generator": SUGAR / "demo_ckpts/KickBox/generator.ckpt",
        "tracker": SUGAR / "demo_ckpts/KickBox/tracker.pt",
    },
}


def camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/PaperZeroWamWorldCamera",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(3.6, 3.6, 2.4),
            rot=(0.3043649418, 0.2319667899, 0.5600173703, 0.7348019703),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=4.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        width=640,
        height=640,
    )


def disable_resets(cfg) -> None:
    if hasattr(cfg, "terminations"):
        for name in vars(cfg.terminations):
            if not name.startswith("_"):
                setattr(cfg.terminations, name, None)
    for name in ("push_robot", "push_object"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.episode_length_s = 30.0


def _latest_filtered_force(sensor) -> torch.Tensor:
    force = sensor.data.force_matrix_w_history
    if force is None or force.ndim != 5 or force.shape[2:4] != (1, 1):
        raise RuntimeError("unexpected filtered contact-force geometry")
    return force[:, -1, 0, 0]


def paired_rows(value: torch.Tensor, copies_per_profile: int) -> torch.Tensor:
    sources = torch.arange(
        0, 8, copies_per_profile, dtype=torch.long, device=value.device
    ).repeat_interleave(copies_per_profile)
    return value.index_select(0, sources).clone()


def install_paired_profiles(base, copies_per_profile: int) -> dict[str, object]:
    robot = base.scene["robot"]
    obj = base.scene["obj"]
    ids = torch.arange(8, dtype=torch.int32)
    physics = {
        "object_materials": paired_rows(obj.root_physx_view.get_material_properties(), copies_per_profile),
        "robot_materials": paired_rows(robot.root_physx_view.get_material_properties(), copies_per_profile),
        "object_masses": paired_rows(obj.root_physx_view.get_masses(), copies_per_profile),
        "object_inertias": paired_rows(obj.root_physx_view.get_inertias(), copies_per_profile),
        "object_coms": paired_rows(obj.root_physx_view.get_coms(), copies_per_profile),
        "robot_masses": paired_rows(robot.root_physx_view.get_masses(), copies_per_profile),
        "robot_inertias": paired_rows(robot.root_physx_view.get_inertias(), copies_per_profile),
        "robot_coms": paired_rows(robot.root_physx_view.get_coms(), copies_per_profile),
    }
    obj.root_physx_view.set_material_properties(physics["object_materials"], ids)
    robot.root_physx_view.set_material_properties(physics["robot_materials"], ids)
    obj.root_physx_view.set_masses(physics["object_masses"], ids)
    obj.root_physx_view.set_inertias(physics["object_inertias"], ids)
    obj.root_physx_view.set_coms(physics["object_coms"], ids)
    robot.root_physx_view.set_masses(physics["robot_masses"], ids)
    robot.root_physx_view.set_inertias(physics["robot_inertias"], ids)
    robot.root_physx_view.set_coms(physics["robot_coms"], ids)

    sources = torch.arange(
        0, 8, copies_per_profile, dtype=torch.long, device=base.device
    ).repeat_interleave(copies_per_profile)
    origins = base.scene.env_origins
    robot_root = robot.data.root_state_w.index_select(0, sources).clone()
    object_root = obj.data.root_state_w.index_select(0, sources).clone()
    source_origins = origins.index_select(0, sources)
    translation = origins - source_origins
    robot_root[:, :3] += translation
    object_root[:, :3] += translation
    # The robot RGB training corpus applies this same one-time centering before
    # every exact-state render.  Keep the live camera distribution aligned
    # without introducing per-frame tracking or future information.
    # Algebraically this is the same midpoint subtraction as the corpus
    # renderer.  Expressing it through the local robot-object displacement
    # avoids cancellation between 30 m-spaced FP32 world coordinates.
    object_from_robot_xy = object_root[:, :2] - robot_root[:, :2]
    robot_root[:, :2] = origins[:, :2] - 0.5 * object_from_robot_xy
    object_root[:, :2] = origins[:, :2] + 0.5 * object_from_robot_xy
    joint_pos = robot.data.joint_pos.index_select(0, sources).clone()
    joint_vel = robot.data.joint_vel.index_select(0, sources).clone()
    robot.write_root_state_to_sim(robot_root)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    obj.write_root_state_to_sim(object_root)
    if hasattr(robot.data, "default_joint_pos"):
        robot.data.default_joint_pos.copy_(
            robot.data.default_joint_pos.index_select(0, sources)
        )
    paired_previous_action = base.action_manager.action.index_select(0, sources).clone()
    base.action_manager.process_action(paired_previous_action)
    base.episode_length_buf.zero_()
    base.scene.write_data_to_sim()
    base.sim.forward()
    base.scene.update(dt=0.0)
    for name in SENSOR_NAMES:
        base.scene.sensors[name].reset()
    base.scene.update(dt=0.0)

    readback = {
        "object_materials": obj.root_physx_view.get_material_properties(),
        "robot_materials": robot.root_physx_view.get_material_properties(),
        "object_masses": obj.root_physx_view.get_masses(),
        "object_inertias": obj.root_physx_view.get_inertias(),
        "object_coms": obj.root_physx_view.get_coms(),
        "robot_masses": robot.root_physx_view.get_masses(),
        "robot_inertias": robot.root_physx_view.get_inertias(),
        "robot_coms": robot.root_physx_view.get_coms(),
    }
    readback_exact = all(torch.equal(readback[name], physics[name]) for name in physics)
    relative_robot = robot.data.root_state_w.clone()
    relative_object = obj.data.root_state_w.clone()
    relative_robot[:, :3] -= origins
    relative_object[:, :3] -= origins
    initial_focus_error = float(
        (0.5 * (relative_robot[:, :2] + relative_object[:, :2])).abs().max()
    )
    pair_state_error = 0.0
    pair_physics_error = 0.0
    group_starts = list(range(0, 8, copies_per_profile))
    for start in group_starts:
        for copy_index in range(start + 1, start + copies_per_profile):
            pair_state_error = max(
                pair_state_error,
                float((relative_robot[start] - relative_robot[copy_index]).abs().max()),
                float((relative_object[start] - relative_object[copy_index]).abs().max()),
                float((robot.data.joint_pos[start] - robot.data.joint_pos[copy_index]).abs().max()),
                float((robot.data.joint_vel[start] - robot.data.joint_vel[copy_index]).abs().max()),
                float(
                    (
                        robot.data.default_joint_pos[start]
                        - robot.data.default_joint_pos[copy_index]
                    ).abs().max()
                ),
                float(
                    (
                        base.action_manager.action[start]
                        - base.action_manager.action[copy_index]
                    ).abs().max()
                ),
                float(
                    (
                        base.action_manager.prev_action[start]
                        - base.action_manager.prev_action[copy_index]
                    ).abs().max()
                ),
            )
            for value in readback.values():
                pair_physics_error = max(
                    pair_physics_error,
                    float((value[start] - value[copy_index]).abs().max()),
                )
    profile_vectors = []
    for even in group_starts:
        profile_vectors.append(
            torch.cat(
                [value[even].reshape(-1).float().cpu() for value in readback.values()]
                + [
                    relative_robot[even].reshape(-1).float().cpu(),
                    relative_object[even].reshape(-1).float().cpu(),
                    robot.data.joint_pos[even].reshape(-1).float().cpu(),
                    robot.data.joint_vel[even].reshape(-1).float().cpu(),
                    robot.data.default_joint_pos[even].reshape(-1).float().cpu(),
                    base.action_manager.action[even].reshape(-1).float().cpu(),
                    base.action_manager.prev_action[even].reshape(-1).float().cpu(),
                ]
            )
        )
    distinct_profile_pairs = sum(
        not torch.equal(profile_vectors[left], profile_vectors[right])
        for left in range(len(group_starts))
        for right in range(left + 1, len(group_starts))
    )
    profile_vectors_finite = all(torch.isfinite(value).all() for value in profile_vectors)
    return {
        "physics_readback_exact": readback_exact,
        "maximum_pair_state_error": pair_state_error,
        "maximum_pair_physics_error": pair_physics_error,
        "maximum_initial_focus_xy_error": initial_focus_error,
        "state_readback_tolerance_m": STATE_READBACK_TOLERANCE_M,
        "distinct_profile_pair_comparisons": distinct_profile_pairs,
        "copies_per_profile": copies_per_profile,
        "profile_count": len(group_starts),
        "profile_vectors": [value.tolist() for value in profile_vectors],
        "profile_vectors_finite": bool(profile_vectors_finite),
        "paired_profiles_passed": (
            readback_exact
            and pair_state_error <= STATE_READBACK_TOLERANCE_M
            and pair_physics_error == 0.0
            and initial_focus_error <= STATE_READBACK_TOLERANCE_M
            and distinct_profile_pairs == len(group_starts) * (len(group_starts) - 1) // 2
            and profile_vectors_finite
        ),
    }


def capture_restore_payload(base) -> dict[str, np.ndarray]:
    robot = base.scene["robot"]
    obj = base.scene["obj"]
    payload = {
        "object_materials": obj.root_physx_view.get_material_properties().detach().cpu().numpy().copy(),
        "robot_materials": robot.root_physx_view.get_material_properties().detach().cpu().numpy().copy(),
        "object_masses": obj.root_physx_view.get_masses().detach().cpu().numpy().copy(),
        "object_inertias": obj.root_physx_view.get_inertias().detach().cpu().numpy().copy(),
        "object_coms": obj.root_physx_view.get_coms().detach().cpu().numpy().copy(),
        "robot_masses": robot.root_physx_view.get_masses().detach().cpu().numpy().copy(),
        "robot_inertias": robot.root_physx_view.get_inertias().detach().cpu().numpy().copy(),
        "robot_coms": robot.root_physx_view.get_coms().detach().cpu().numpy().copy(),
        "robot_root_state_w": robot.data.root_state_w.detach().cpu().numpy().copy(),
        "robot_joint_pos": robot.data.joint_pos.detach().cpu().numpy().copy(),
        "robot_joint_vel": robot.data.joint_vel.detach().cpu().numpy().copy(),
        "object_root_state_w": obj.data.root_state_w.detach().cpu().numpy().copy(),
        "default_joint_pos": robot.data.default_joint_pos.detach().cpu().numpy().copy(),
        "previous_action": base.action_manager.action.detach().cpu().numpy().copy(),
        "action_manager_prev_action": base.action_manager.prev_action.detach().cpu().numpy().copy(),
    }
    if tuple(payload) != RESTORE_PAYLOAD_KEYS:
        raise RuntimeError("physical restore payload topology changed")
    return payload


def reinitialize_tracker_observation_history(base) -> dict[str, bool]:
    """Fill every released Tracker history slot from the restored live state."""

    manager = base.observation_manager
    manager.reset()
    manager.compute(update_history=True)
    expected_policy_terms = {
        "base_ang_vel_history",
        "joint_pos_history",
        "joint_vel_history",
        "actions_history",
        "project_gravity",
    }
    buffers = manager._group_obs_term_history_buffer
    policy_buffers = buffers.get("policy", {})
    if set(policy_buffers) != expected_policy_terms:
        raise RuntimeError(
            "released Tracker observation-history topology changed: "
            f"{sorted(policy_buffers)}"
        )
    evidence: dict[str, bool] = {}
    for term_name, circular_buffer in policy_buffers.items():
        value = circular_buffer.buffer
        exact = bool(
            value.ndim >= 2
            and value.shape[1] == 5
            and torch.equal(value, value[:, :1].expand_as(value))
        )
        evidence[term_name] = exact
    if not all(evidence.values()):
        raise RuntimeError(
            f"released Tracker history was not reinitialized exactly: {evidence}"
        )
    return dict(sorted(evidence.items()))


def restore_profiles(base, payload: dict[str, np.ndarray]) -> dict[str, object]:
    """Restore the adapted arm's exact physical state before endpoint rollout."""

    robot = base.scene["robot"]
    obj = base.scene["obj"]
    ids = torch.arange(8, dtype=torch.int32)

    def tensor(name: str, reference: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(payload[name], device=reference.device, dtype=reference.dtype)
        if value.shape != reference.shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid restore payload for {name}: {tuple(value.shape)}")
        return value

    properties = {
        "object_materials": (obj.root_physx_view, "get_material_properties", "set_material_properties"),
        "robot_materials": (robot.root_physx_view, "get_material_properties", "set_material_properties"),
        "object_masses": (obj.root_physx_view, "get_masses", "set_masses"),
        "object_inertias": (obj.root_physx_view, "get_inertias", "set_inertias"),
        "object_coms": (obj.root_physx_view, "get_coms", "set_coms"),
        "robot_masses": (robot.root_physx_view, "get_masses", "set_masses"),
        "robot_inertias": (robot.root_physx_view, "get_inertias", "set_inertias"),
        "robot_coms": (robot.root_physx_view, "get_coms", "set_coms"),
    }
    for name, (view, getter_name, setter_name) in properties.items():
        reference = getattr(view, getter_name)()
        getattr(view, setter_name)(tensor(name, reference), ids)
    robot_root = tensor("robot_root_state_w", robot.data.root_state_w)
    joint_pos = tensor("robot_joint_pos", robot.data.joint_pos)
    joint_vel = tensor("robot_joint_vel", robot.data.joint_vel)
    object_root = tensor("object_root_state_w", obj.data.root_state_w)
    robot.write_root_state_to_sim(robot_root)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    obj.write_root_state_to_sim(object_root)
    robot.data.default_joint_pos.copy_(
        tensor("default_joint_pos", robot.data.default_joint_pos)
    )
    # ActionManager keeps two history slots.  Restore the older slot first,
    # then the currently exposed ``last_action`` slot, so endpoint and adapted
    # routes begin from the same complete action history rather than merely
    # the same policy observation.
    base.action_manager.process_action(
        tensor("action_manager_prev_action", base.action_manager.prev_action)
    )
    base.action_manager.process_action(
        tensor("previous_action", base.action_manager.action)
    )
    base.episode_length_buf.zero_()
    base.scene.write_data_to_sim()
    base.sim.forward()
    base.scene.update(dt=0.0)
    for name in SENSOR_NAMES:
        base.scene.sensors[name].reset()
    base.scene.update(dt=0.0)
    command = base.command_manager.get_term("motion")
    if hasattr(command, "time_steps"):
        command.time_steps.zero_()
    environment_ids = torch.arange(8, dtype=torch.long, device=base.device)
    command._fill_generator_obs_buffer(environment_ids)
    command._call_generator(environment_ids)
    observation_history = reinitialize_tracker_observation_history(base)
    readback = capture_restore_payload(base)
    exact = all(np.array_equal(readback[name], payload[name]) for name in payload)
    if not exact:
        raise RuntimeError("endpoint server did not exactly restore adapted profile payload")
    return {
        "status": "restored",
        "restore_exact": True,
        "tracker_observation_history_reinitialized": observation_history,
        "restore_payload": readback,
        "initial_rgb": capture_rgb(base),
        "initial_state": {
            "robot_root_state_w": readback["robot_root_state_w"],
            "robot_joint_pos": readback["robot_joint_pos"],
            "robot_joint_vel": readback["robot_joint_vel"],
            "object_root_state_w": readback["object_root_state_w"],
            "previous_action": readback["previous_action"],
            "action_manager_prev_action": readback["action_manager_prev_action"],
        },
    }


def capture_rgb(base) -> np.ndarray:
    base.sim.render()
    camera = base.scene["world_camera"]
    camera.update(0.0, force_recompute=True)
    value = camera.data.output["rgb"][:, ..., :3]
    if tuple(value.shape) != (8, 640, 640, 3):
        raise RuntimeError(f"live camera geometry mismatch: {tuple(value.shape)}")
    resized = F.interpolate(
        value.permute(0, 3, 1, 2).float(),
        size=(320, 320),
        mode="area",
    ).round().clamp_(0, 255).to(torch.uint8)
    output = resized.permute(0, 2, 3, 1).cpu().numpy()
    if any(int(frame.max()) == int(frame.min()) for frame in output):
        raise RuntimeError("live camera returned a constant frame")
    return output


def step_chunk(base, env, actions: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    if actions.ndim != 3 or actions.shape[0] != 8 or actions.shape[2] != 29:
        raise ValueError(f"action chunk geometry mismatch: {actions.shape}")
    records: dict[str, list[np.ndarray]] = {
        "robot_root_state_w": [],
        "robot_joint_pos": [],
        "robot_joint_vel": [],
        "object_root_state_w": [],
        "contact": [],
        "requested_action": [],
        "executed_action": [],
        "done": [],
    }
    frames: list[np.ndarray] = []
    for index in range(actions.shape[1]):
        requested = torch.as_tensor(actions[:, index], device=base.device).clone()
        # Preserve the exact model-decoder command before handing a tensor to
        # the environment wrapper; a wrapper is allowed to transform its input
        # tensor in place, so comparing against that same object after step()
        # would not independently prove decoder -> PhysX action fidelity.
        submitted = requested.clone()
        with torch.inference_mode():
            _, _, done, _ = env.step(requested)
        executed = base.action_manager.action
        if executed.shape != submitted.shape or not torch.equal(executed, submitted):
            raise RuntimeError(
                "PhysX action input differs from the requested 29-D model action; "
                f"max_error={float((executed - submitted).abs().max())}"
            )
        force = torch.stack(
            [_latest_filtered_force(base.scene.sensors[name]) for name in SENSOR_NAMES],
            dim=1,
        )
        records["robot_root_state_w"].append(
            base.scene["robot"].data.root_state_w.detach().cpu().numpy().copy()
        )
        records["robot_joint_pos"].append(
            base.scene["robot"].data.joint_pos.detach().cpu().numpy().copy()
        )
        records["robot_joint_vel"].append(
            base.scene["robot"].data.joint_vel.detach().cpu().numpy().copy()
        )
        records["object_root_state_w"].append(
            base.scene["obj"].data.root_state_w.detach().cpu().numpy().copy()
        )
        records["contact"].append(
            (torch.linalg.vector_norm(force, dim=-1) > threshold)
            .detach().cpu().numpy().copy()
        )
        records["requested_action"].append(submitted.detach().cpu().numpy().copy())
        records["executed_action"].append(executed.detach().cpu().numpy().copy())
        records["done"].append(done.detach().cpu().numpy().astype(bool, copy=True))
        if (index + 1) % RGB_STRIDE == 0:
            frames.append(capture_rgb(base))
    return {
        **{name: np.stack(values) for name, values in records.items()},
        "rgb": np.stack(frames),
    }


def step_endpoint_chunk(
    base, env, policy, observation, count: int, threshold: float
) -> tuple[dict[str, np.ndarray], object]:
    """Execute an exact released Generator+Tracker endpoint for ``count`` steps."""

    if count <= 0 or count > 40:
        raise ValueError(f"endpoint chunk count must be in 1..40, got {count}")
    records: dict[str, list[np.ndarray]] = {
        "robot_root_state_w": [],
        "robot_joint_pos": [],
        "robot_joint_vel": [],
        "object_root_state_w": [],
        "contact": [],
        "requested_action": [],
        "executed_action": [],
        "done": [],
    }
    frames: list[np.ndarray] = []
    current = observation
    for index in range(count):
        with torch.inference_mode():
            requested = policy(current).clone()
            submitted = requested.clone()
            current, _, done, _ = env.step(requested)
        executed = base.action_manager.action
        if executed.shape != submitted.shape or not torch.equal(executed, submitted):
            raise RuntimeError(
                "released endpoint action differs from PhysX action readback; "
                f"max_error={float((executed - submitted).abs().max())}"
            )
        force = torch.stack(
            [_latest_filtered_force(base.scene.sensors[name]) for name in SENSOR_NAMES],
            dim=1,
        )
        records["robot_root_state_w"].append(
            base.scene["robot"].data.root_state_w.detach().cpu().numpy().copy()
        )
        records["robot_joint_pos"].append(
            base.scene["robot"].data.joint_pos.detach().cpu().numpy().copy()
        )
        records["robot_joint_vel"].append(
            base.scene["robot"].data.joint_vel.detach().cpu().numpy().copy()
        )
        records["object_root_state_w"].append(
            base.scene["obj"].data.root_state_w.detach().cpu().numpy().copy()
        )
        records["contact"].append(
            (torch.linalg.vector_norm(force, dim=-1) > threshold)
            .detach().cpu().numpy().copy()
        )
        records["requested_action"].append(submitted.detach().cpu().numpy().copy())
        records["executed_action"].append(executed.detach().cpu().numpy().copy())
        records["done"].append(done.detach().cpu().numpy().astype(bool, copy=True))
        if (index + 1) % RGB_STRIDE == 0:
            frames.append(capture_rgb(base))
    return (
        {
            **{name: np.stack(values) for name, values in records.items()},
            "rgb": np.stack(frames),
        },
        current,
    )


def main() -> None:
    output_error: dict[str, str] | None = None
    listener = None
    connection = None
    env = None
    try:
        os.chdir(SUGAR)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=True,
            entry_point_key="play_env_cfg_entry_point",
        )
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = 8
        env_cfg.scene.env_spacing = 30.0
        env_cfg.scene.world_camera = camera_cfg()
        env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.observations.policy.enable_corruption = False
        disable_resets(env_cfg)
        endpoint = ENDPOINTS.get(args.route)
        if endpoint is not None:
            for path in endpoint.values():
                if not Path(path).exists():
                    raise FileNotFoundError(path)
            env_cfg.commands.motion.generator_checkpoint_path = str(
                endpoint["generator"]
            )
            env_cfg.commands.motion.motion_folder = str(endpoint["motion_folder"])
            env_cfg.commands.motion.eval_random_motion = False
            env_cfg.commands.motion.eval_mode = True
            env_cfg.commands.motion.eval_max_time = 660
            args.checkpoint = str(endpoint["tracker"])
        agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
        gym_env = gym.make(args.task, cfg=env_cfg)
        if isinstance(gym_env.unwrapped, DirectMARLEnv):
            gym_env = multi_agent_to_single_agent(gym_env)
        base = gym_env.unwrapped
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        env.get_observations()
        profile_result = install_paired_profiles(base, args.copies_per_profile)
        endpoint_policy = None
        endpoint_observation = None
        if endpoint is not None:
            official_runner = OnPolicyRunner(
                env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
            )
            official_runner.load(str(endpoint["tracker"]))
            endpoint_policy = official_runner.get_inference_policy(device=base.device)
            endpoint_observation = env.get_observations()
            if isinstance(endpoint_observation, tuple):
                endpoint_observation = endpoint_observation[0]
        initial_rgb = capture_rgb(base)
        initial_state = {
            "robot_root_state_w": base.scene["robot"].data.root_state_w.detach().cpu().numpy().copy(),
            "robot_joint_pos": base.scene["robot"].data.joint_pos.detach().cpu().numpy().copy(),
            "robot_joint_vel": base.scene["robot"].data.joint_vel.detach().cpu().numpy().copy(),
            "object_root_state_w": base.scene["obj"].data.root_state_w.detach().cpu().numpy().copy(),
            "previous_action": base.action_manager.action.detach().cpu().numpy().copy(),
            "action_manager_prev_action": base.action_manager.prev_action.detach().cpu().numpy().copy(),
        }
        pixel_pair_max_abs = max(
            int(
                np.abs(
                    initial_rgb[start].astype(np.int16)
                    - initial_rgb[copy_index].astype(np.int16)
                ).max()
            )
            for start in range(0, 8, args.copies_per_profile)
            for copy_index in range(start + 1, start + args.copies_per_profile)
        )
        profile_result["maximum_initial_pair_pixel_error"] = pixel_pair_max_abs
        profile_result["paired_initial_rgb_exact"] = pixel_pair_max_abs == 0
        profile_result["paired_profiles_passed"] = bool(
            profile_result["paired_profiles_passed"]
            and profile_result["paired_initial_rgb_exact"]
        )
        args.address.parent.mkdir(parents=True, exist_ok=True)
        if args.address.exists():
            args.address.unlink()
        listener = Listener(str(args.address), family="AF_UNIX", authkey=AUTHKEY)
        connection = listener.accept()
        connection.send(
            {
                "status": "ready",
                "seed": args.seed,
                "route": args.route,
                "endpoint_files": (
                    {name: str(path) for name, path in endpoint.items()}
                    if endpoint is not None
                    else None
                ),
                "control_hz": CONTROL_HZ,
                "rgb_hz": CONTROL_HZ // RGB_STRIDE,
                "profile_result": profile_result,
                "restore_payload": capture_restore_payload(base),
                "initial_rgb": initial_rgb,
                "initial_state": initial_state,
            }
        )
        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "restore_profiles":
                if endpoint is None:
                    raise ValueError("restore command sent to model server")
                restored = restore_profiles(
                    base,
                    {
                        name: np.asarray(value)
                        for name, value in request["restore_payload"].items()
                    },
                )
                endpoint_observation = env.get_observations()
                if isinstance(endpoint_observation, tuple):
                    endpoint_observation = endpoint_observation[0]
                connection.send(restored)
            elif command == "step_chunk":
                if endpoint is not None:
                    raise ValueError("model action command sent to endpoint server")
                connection.send(
                    step_chunk(
                        base,
                        env,
                        np.asarray(request["actions"], dtype=np.float32),
                        args.contact_threshold_n,
                    )
                )
            elif command == "step_endpoint_chunk":
                if endpoint_policy is None or endpoint_observation is None:
                    raise ValueError("endpoint action command sent to model server")
                response, endpoint_observation = step_endpoint_chunk(
                    base,
                    env,
                    endpoint_policy,
                    endpoint_observation,
                    int(request["count"]),
                    args.contact_threshold_n,
                )
                connection.send(response)
            elif command == "close":
                connection.send({"status": "closed"})
                break
            else:
                raise ValueError(f"unknown simulator command: {command}")
    except Exception as error:
        output_error = {
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        if connection is not None:
            try:
                connection.send({"status": "error", **output_error})
            except Exception:
                pass
        raise
    finally:
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()
        if env is not None:
            env.close()
        if args.address.exists():
            args.address.unlink()
        if output_error is not None:
            error_path = args.address.with_suffix(".error.json")
            error_path.write_text(json.dumps(output_error, indent=2) + "\n")
        simulation_app.close()


if __name__ == "__main__":
    main()
