#!/usr/bin/env python3
"""Collect actual body-to-box contact/event targets from official SUGAR inference.

This collector runs the released task-specific Tracker and Generator in the
IsaacLab/PhysX inference environment.  It records the four filtered physical
ContactSensors (left/right hand and foot), body/object state, actions, motion
frame and reset boundary on the same control clock.  Reference binary labels
are never used as actual-contact targets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUGAR = ROOT / "SUGAR"
SUGAR_SCRIPT = SUGAR / "scripts/sugar_rl"
if str(SUGAR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(SUGAR_SCRIPT))

# Match the H200 IsaacLab runtime used by the project's verified SUGAR runs.
os.environ.setdefault(
    "ISAACLAB_GROUND_PLANE_USD",
    str(SUGAR / "descriptions/terrain/sugar_ground_plane.usda"),
)
os.environ.setdefault("ISAACLAB_USE_LOCAL_FRAME_MARKER", "1")
os.environ.setdefault("SUGAR_DISABLE_TRAIN_DEBUG_VIS", "1")
os.environ.setdefault("DISPLAY", "")
vulkan_icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
if vulkan_icd.is_file():
    os.environ.setdefault("VK_ICD_FILENAMES", str(vulkan_icd))
job_id = os.environ.get("SLURM_JOB_ID", "local")
os.environ.setdefault("ISAACLAB_TMP_ROOT", f"/tmp/Curiosity_contact_event_{job_id}")
os.environ.setdefault(
    "SUGAR_UNITREE_TMP_ROOT",
    f"/tmp/Curiosity_contact_event_unitree_{job_id}",
)

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task-family", choices=("CarryBox", "KickBox"), required=True)
parser.add_argument("--motion-folder", type=Path, required=True)
parser.add_argument("--generator-checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--source-motion-id",
    type=int,
    default=None,
    help="Expected source ID for a single-motion canary.",
)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=700)
parser.add_argument("--seed", type=int, default=271001)
parser.add_argument("--contact-threshold-n", type=float, default=0.1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.checkpoint is None:
    parser.error("--checkpoint is required")
if args.num_envs <= 0 or args.steps < 100:
    parser.error("--num-envs must be positive and --steps must be at least 100")
if args.contact_threshold_n <= 0:
    parser.error("--contact-threshold-n must be positive")
args.task = f"Sugar-G129dof-{args.task_family}-Inference"
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import builtins  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import rsl_rl.algorithms  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import sugar_rl.tasks  # noqa: F401,E402
from sugar_rl.utils.parser_cfg import parse_env_cfg  # noqa: E402
from sugar_rl.utils.rsl_rl_bcppo import BCPPO  # noqa: E402


setattr(builtins, "BCPPO", BCPPO)
setattr(rsl_rl.algorithms, "BCPPO", BCPPO)

SENSOR_NAMES = (
    "left_hand_forces",
    "right_hand_forces",
    "left_foot_forces",
    "right_foot_forces",
)
SENSOR_ROLES = ("left_hand", "right_hand", "left_foot", "right_foot")
CONTROL_DT_S = 0.02
LIFT_THRESHOLD_M = 0.05
MOVE_THRESHOLD_MPS = 0.05


def _latest_filtered_force(sensor) -> torch.Tensor:
    force = sensor.data.force_matrix_w_history
    if force is None or force.ndim != 5 or force.shape[2:4] != (1, 1):
        raise RuntimeError(
            f"filtered ContactSensor shape drift: {None if force is None else tuple(force.shape)}"
        )
    return force[:, -1, 0, 0, :]


def _event_durations(contact: np.ndarray, reset_before: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return total and remaining event frames without crossing reset boundaries."""
    total = np.zeros(contact.shape, dtype=np.int16)
    remaining = np.zeros(contact.shape, dtype=np.int16)
    frames, envs, roles = contact.shape
    for env in range(envs):
        start = 0
        # ManagerBasedRLEnv resets before returning observations and scene
        # state, so a true flag means that the recorded frame starts a new
        # episode.  The preceding episode therefore stops at this index.
        boundaries = np.flatnonzero(reset_before[:, env])
        stops = [int(value) for value in boundaries if value > start]
        if not stops or stops[-1] != frames:
            stops.append(frames)
        for stop in stops:
            for role in range(roles):
                values = contact[start:stop, env, role]
                padded = np.pad(values.astype(np.int8), (1, 1))
                edges = np.flatnonzero(np.diff(padded))
                for event_start, event_stop in zip(edges[::2], edges[1::2]):
                    length = int(event_stop - event_start)
                    sl = slice(start + int(event_start), start + int(event_stop))
                    total[sl, env, role] = length
                    remaining[sl, env, role] = np.arange(
                        length, 0, -1, dtype=np.int16
                    )
            start = stop
    return total, remaining


def _motion_regime(
    object_state: np.ndarray, reset_before: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    frames, envs = reset_before.shape
    lift = np.zeros((frames, envs), dtype=np.float32)
    regime = np.zeros((frames, envs), dtype=np.uint8)
    for env in range(envs):
        start = 0
        boundaries = np.flatnonzero(reset_before[:, env])
        stops = [int(value) for value in boundaries if value > start]
        if not stops or stops[-1] != frames:
            stops.append(frames)
        for stop in stops:
            count = min(25, stop - start)
            baseline = float(np.median(object_state[start : start + count, env, 2]))
            lift[start:stop, env] = object_state[start:stop, env, 2] - baseline
            speed = np.linalg.norm(object_state[start:stop, env, 7:10], axis=-1)
            lifted = lift[start:stop, env] >= LIFT_THRESHOLD_M
            moving = speed >= MOVE_THRESHOLD_MPS
            # 0 ground-static, 1 ground-moving, 2 lifted-static, 3 lifted-moving
            regime[start:stop, env] = lifted.astype(np.uint8) * 2 + moving.astype(np.uint8)
            start = stop
    return lift, regime


def _summarize(
    *,
    task_family: str,
    force: np.ndarray,
    contact: np.ndarray,
    lift: np.ndarray,
    regime: np.ndarray,
    done: np.ndarray,
    threshold_n: float,
) -> dict[str, object]:
    fractions = contact.mean(axis=(0, 1))
    bilateral_hand = contact[:, :, 0] & contact[:, :, 1]
    any_hand = contact[:, :, 0] | contact[:, :, 1]
    any_foot = contact[:, :, 2] | contact[:, :, 3]
    max_lift = float(np.max(lift))
    measurements = {
        "left_hand_contact_fraction": float(fractions[0]),
        "right_hand_contact_fraction": float(fractions[1]),
        "left_foot_contact_fraction": float(fractions[2]),
        "right_foot_contact_fraction": float(fractions[3]),
        "bilateral_hand_contact_fraction": float(np.mean(bilateral_hand)),
        "any_hand_contact_fraction": float(np.mean(any_hand)),
        "any_foot_contact_fraction": float(np.mean(any_foot)),
        "lifted_fraction": float(np.mean(lift >= LIFT_THRESHOLD_M)),
        "ground_moving_fraction": float(np.mean(regime == 1)),
        "lifted_moving_fraction": float(np.mean(regime == 3)),
        "maximum_lift_m": max_lift,
        "peak_force_n_by_role": {
            role: float(np.max(np.linalg.norm(force[:, :, index], axis=-1)))
            for index, role in enumerate(SENSOR_ROLES)
        },
        "reset_count": int(np.count_nonzero(done)),
    }
    structural = {
        "all_arrays_finite": bool(np.isfinite(force).all() and np.isfinite(lift).all()),
        "physical_force_vectors_recorded": force.ndim == 4 and force.shape[-2:] == (4, 3),
        "contact_is_exact_threshold_of_force": bool(
            np.array_equal(contact, np.linalg.norm(force, axis=-1) > threshold_n)
        ),
    }
    if task_family == "CarryBox":
        behavioral = {
            "carry_has_bilateral_hand_contact": measurements[
                "bilateral_hand_contact_fraction"
            ]
            >= 0.02,
            "carry_lifts_box_five_centimeters": max_lift >= LIFT_THRESHOLD_M,
        }
    else:
        behavioral = {
            "kick_has_foot_box_contact": measurements["any_foot_contact_fraction"] >= 0.01,
            "kick_moves_box_at_ground_level": measurements["ground_moving_fraction"] >= 0.05,
        }
    checks = {**structural, **behavioral}
    return {
        "protocol": "sugar_official_tracker_actual_contact_event_canary_v1",
        "passed": all(checks.values()),
        "task_family": task_family,
        "checks": checks,
        "measurements": measurements,
        "contact_threshold_n": threshold_n,
        "contact_source": (
            "IsaacLab filtered ContactSensor.force_matrix_w_history for each named "
            "hand/foot body against /Obj"
        ),
        "reference_binary_proxy_used_as_target": False,
        "claim_boundary": (
            "This canary validates actual rollout target collection for the released "
            "official Tracker/Generator pair. It does not establish reward-predictor "
            "generalization or demo-following policy benefit."
        ),
        "automatic_next_branch": (
            "build_motion_disjoint_actual_contact_event_dataset"
            if all(checks.values())
            else "repair_actual_rollout_collection_or_task_policy_before_predictor_training"
        ),
    }


def main() -> None:
    output = args.output_dir.expanduser().resolve()
    motion_folder = args.motion_folder.expanduser().resolve()
    generator_checkpoint = args.generator_checkpoint.expanduser().resolve()
    tracker_checkpoint = Path(args.checkpoint).expanduser().resolve()
    for path in (motion_folder, generator_checkpoint, tracker_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if (
        motion_folder.name.startswith("data_")
        and (motion_folder / "robot_50hz.npz").is_file()
    ):
        motion_paths = [motion_folder]
    else:
        motion_paths = sorted(motion_folder.glob("data_*"))
    if not motion_paths:
        raise FileNotFoundError(f"no data_* motions under {motion_folder}")
    source_motion_id_by_local = np.asarray(
        [int(path.name.split("_")[1]) for path in motion_paths],
        dtype=np.int16,
    )
    source_reference_steps_by_local = []
    for path in motion_paths:
        with np.load(path / "robot_50hz.npz", allow_pickle=False) as archive:
            source_reference_steps_by_local.append(int(archive["joint_pos"].shape[0]))
    source_reference_steps_by_local = np.asarray(
        source_reference_steps_by_local, dtype=np.int32
    )
    if (
        args.source_motion_id is not None
        and (
            len(source_motion_id_by_local) != 1
            or int(source_motion_id_by_local[0]) != args.source_motion_id
        )
    ):
        raise ValueError(
            "--source-motion-id is only a single-motion identity assertion"
        )
    output.mkdir(parents=True, exist_ok=False)
    # Official SUGAR asset configs use paths relative to the SUGAR checkout.
    os.chdir(SUGAR)
    print("COLLECTOR_PHASE=paths_validated", flush=True)

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
    env_cfg.commands.motion.generator_checkpoint_path = str(generator_checkpoint)
    env_cfg.commands.motion.motion_folder = str(motion_folder)
    env_cfg.commands.motion.eval_random_motion = False
    env_cfg.commands.motion.eval_mode = True
    # MotionLoader uses eval_max_time as its allocation width.  It therefore
    # must cover the longest source motion even when this rollout is shorter.
    env_cfg.commands.motion.eval_max_time = max(
        args.steps + 1, int(np.max(source_reference_steps_by_local))
    )
    os.environ["SUGAR_DISABLE_TRAIN_DEBUG_VIS"] = "1"
    for value in vars(env_cfg.scene).values():
        if hasattr(value, "debug_vis"):
            value.debug_vis = False

    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.seed = args.seed
    print("COLLECTOR_PHASE=before_gym_make", flush=True)
    gym_env = gym.make(args.task, cfg=env_cfg)
    print("COLLECTOR_PHASE=after_gym_make", flush=True)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    base = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(tracker_checkpoint))
    policy = runner.get_inference_policy(device=base.device)
    print("COLLECTOR_PHASE=policy_loaded", flush=True)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    body_names = np.asarray(base.scene["robot"].body_names, dtype="U64")
    records: dict[str, list[np.ndarray]] = {
        "robot_root_state_w": [],
        "robot_body_position_w": [],
        "object_root_state_w": [],
        "contact_force_w": [],
        "action": [],
        "done": [],
        "motion_frame": [],
        "local_motion_id": [],
        "policy_observation": [],
    }
    started = time.time()
    print("COLLECTOR_PHASE=rollout_started", flush=True)
    for step in range(args.steps):
        with torch.inference_mode():
            action = policy(obs)
            obs, _, done, _ = env.step(action)
        records["robot_root_state_w"].append(
            base.scene["robot"].data.root_state_w.detach().cpu().numpy().copy()
        )
        records["robot_body_position_w"].append(
            base.scene["robot"].data.body_pos_w.detach().cpu().numpy().copy()
        )
        records["object_root_state_w"].append(
            base.scene["obj"].data.root_state_w.detach().cpu().numpy().copy()
        )
        records["contact_force_w"].append(
            torch.stack(
                [_latest_filtered_force(base.scene.sensors[name]) for name in SENSOR_NAMES],
                dim=1,
            ).detach().cpu().numpy().copy()
        )
        records["action"].append(action.detach().cpu().numpy().copy())
        records["done"].append(done.detach().cpu().numpy().astype(bool, copy=True))
        command = base.command_manager.get_term("motion")
        records["motion_frame"].append(command.time_steps.detach().cpu().numpy().copy())
        records["local_motion_id"].append(command.motion_id.detach().cpu().numpy().copy())
        records["policy_observation"].append(
            obs["policy"].detach().cpu().numpy().copy()
        )
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            print(
                f"COLLECTOR_PROGRESS={step + 1}/{args.steps} "
                f"elapsed_s={time.time() - started:.1f}",
                flush=True,
            )

    arrays = {name: np.stack(values) for name, values in records.items()}
    local_motion_id = arrays["local_motion_id"].astype(np.int64)
    if np.any(local_motion_id < 0) or np.any(
        local_motion_id >= len(source_motion_id_by_local)
    ):
        raise RuntimeError("local motion ID is outside the enumerated motion folder")
    arrays["source_motion_id"] = source_motion_id_by_local[local_motion_id]
    force = arrays["contact_force_w"].astype(np.float32)
    reset_before = arrays["done"].astype(bool)
    contact = np.linalg.norm(force, axis=-1) > args.contact_threshold_n
    event_total, event_remaining = _event_durations(contact, reset_before)
    lift, regime = _motion_regime(arrays["object_root_state_w"], reset_before)
    result = _summarize(
        task_family=args.task_family,
        force=force,
        contact=contact,
        lift=lift,
        regime=regime,
        done=reset_before,
        threshold_n=args.contact_threshold_n,
    )
    result.update(
        {
            "host": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "elapsed_s": time.time() - started,
            "source_motion_ids_available": source_motion_id_by_local.tolist(),
            "source_reference_steps_by_local": (
                source_reference_steps_by_local.tolist()
            ),
            "source_motion_ids_observed": sorted(
                int(value) for value in np.unique(arrays["source_motion_id"])
            ),
            "local_motion_ids_observed": sorted(
                int(value) for value in np.unique(arrays["local_motion_id"])
            ),
            "tracker_checkpoint": str(tracker_checkpoint),
            "generator_checkpoint": str(generator_checkpoint),
            "motion_folder": str(motion_folder),
            "artifacts": {"trace": "TRACE.npz", "result": "RESULT.json"},
        }
    )
    np.savez_compressed(
        output / "TRACE.npz",
        **arrays,
        contact=contact,
        contact_role_names=np.asarray(SENSOR_ROLES, dtype="U16"),
        contact_event_total_frames=event_total,
        contact_event_remaining_frames=event_remaining,
        reset_before_frame=reset_before,
        lift_height_m=lift,
        motion_regime=regime,
        motion_regime_names=np.asarray(
            ("ground_static", "ground_moving", "lifted_static", "lifted_moving"),
            dtype="U16",
        ),
        robot_body_names=body_names,
        source_motion_id_by_local_motion=source_motion_id_by_local,
        source_reference_steps_by_local_motion=source_reference_steps_by_local,
        control_dt_s=np.asarray([CONTROL_DT_S], dtype=np.float32),
    )
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("COLLECTOR_PHASE=result_written", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # Isaac Sim can otherwise rewrite startup failures to exit code 0.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        simulation_app.close()
