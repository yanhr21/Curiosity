#!/usr/bin/env python3
"""Run one no-learning live CarryBox mass jump with online patch telemetry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

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
parser.add_argument("--max-steps", type=int, default=420)
parser.add_argument("--minimum-lift", type=float, default=0.05)
parser.add_argument("--stable-frames", type=int, default=10)
parser.add_argument("--delay-frames", type=int, nargs=2, default=(10, 50))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.mass_factor < 1.0:
    raise SystemExit("preflight mass factor must be at least one")
if args.max_steps < 1:
    raise SystemExit("max-steps must be positive")

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
os.chdir(ROOT / "SUGAR")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
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
    online_patch_tactile_contract,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_online_patch_tactile_mass_env_cfg import (  # noqa: E402
    OnlinePatchSlipMassRobotPlayEnvCfg,
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


def main() -> None:
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=False)

    register_official_refiner_anatomical_whole_hand_tacsl_audit_task()
    cfg = OnlinePatchSlipMassRobotPlayEnvCfg()
    cfg.seed = int(args.seed)
    cfg.sim.device = args.device
    cfg.commands.motion.motion_folder = "data/CarryBox"
    cfg.commands.motion.pose_range = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.commands.motion.joint_position_range = (0.0, 0.0)
    cfg.events.obj_physics_material.params.update(
        static_friction_range=(0.5, 0.5),
        dynamic_friction_range=(0.5, 0.5),
        restitution_range=(0.0, 0.0),
        num_buckets=1,
    )
    for term_name in ("reset_mass_jump", "step_mass_jump"):
        params = getattr(cfg.events, term_name).params
        params["mass_factors"] = (float(args.mass_factor),)
        params["minimum_lift_m"] = float(args.minimum_lift)
        params["stable_bilateral_frames"] = int(args.stable_frames)
        params["delay_frames"] = tuple(int(value) for value in args.delay_frames)
        params["seed"] = int(args.seed)

    env = gym.make(REFINER_TASK_ID, cfg=cfg, render_mode=None)
    original_reset_idx = None
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
        robot = base_env.scene["robot"]
        replay_actions = None
        if args.action_trace is not None:
            action_trace = args.action_trace.expanduser().resolve()
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
            "actor_policy_observation": [],
            "actor_patch_history": [],
            "object_pos_w": [],
            "object_quat_w": [],
            "object_lin_vel_w": [],
            "object_ang_vel_w": [],
            "joint_pos": [],
            "joint_vel": [],
            "applied_action": [],
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
            observation, _, _, _, _ = env.step(action)
            patches = current_whole_hand_patch_features(base_env)
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
            rows["actor_policy_observation"].append(cpu(observation["policy"][0]))
            rows["actor_patch_history"].append(
                cpu(observation["online_patch_tactile_history"][0])
            )
            rows["object_pos_w"].append(cpu(obj.data.root_pos_w[0]))
            rows["object_quat_w"].append(cpu(obj.data.root_quat_w[0]))
            rows["object_lin_vel_w"].append(cpu(obj.data.root_lin_vel_w[0]))
            rows["object_ang_vel_w"].append(cpu(obj.data.root_ang_vel_w[0]))
            rows["joint_pos"].append(cpu(robot.data.joint_pos[0]))
            rows["joint_vel"].append(cpu(robot.data.joint_vel[0]))
            rows["applied_action"].append(cpu(action[0]))
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

        arrays = {name: np.asarray(values) for name, values in rows.items()}
        np.savez_compressed(output_root / "online_mass_jump_trace.npz", **arrays)
        contact = arrays["patch_features"][..., 0] > 0.5
        normal_load = arrays["patch_features"][..., 1]
        pressure = arrays["patch_features"][..., 2]
        jump_indices = np.flatnonzero(arrays["jump_applied"])
        first_jump_frame = int(jump_indices[0]) if len(jump_indices) else None
        summary = {
            "schema": "plan15_online_patch_mass_jump_preflight_v1",
            "semantics": "live IsaacLab rollout; no learning; no offline replay",
            "action_source": (
                "online_frozen_official_refiner"
                if replay_actions is None
                else "fixed_nominal_applied_action_trace"
            ),
            "action_trace": (
                None if args.action_trace is None else str(args.action_trace.expanduser().resolve())
            ),
            "motion_id": int(args.motion_id),
            "seed": int(args.seed),
            "source_frames": int(args.max_steps),
            "nominal_mass_kg": float(arrays["mass_readback_kg"][0]),
            "target_mass_factor": float(args.mass_factor),
            "first_jump_frame": first_jump_frame,
            "mass_changed": bool(arrays["mass_changed"][-1]),
            "final_mass_readback_kg": float(arrays["mass_readback_kg"][-1]),
            "bilateral_contact_frames": int(
                np.count_nonzero(contact[:, 0].any(axis=-1) & contact[:, 1].any(axis=-1))
            ),
            "maximum_active_patches_left": int(contact[:, 0].sum(axis=-1).max()),
            "maximum_active_patches_right": int(contact[:, 1].sum(axis=-1).max()),
            "maximum_patch_normal_load_n": float(normal_load.max()),
            "maximum_patch_pressure_pa": float(pressure.max()),
            "maximum_object_lift_m": float(
                arrays["object_pos_w"][:, 2].max() - arrays["object_pos_w"][0, 2]
            ),
            "actor_mass_observation": False,
            "actor_jump_flag_observation": False,
            "actor_measured_object_state": False,
            "patch_contract": online_patch_tactile_contract(),
            "base_patch_channels": list(BASE_PATCH_CHANNELS),
            "slip_callable_live": True,
        }
        (output_root / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        if first_jump_frame is None:
            raise RuntimeError("mass jump never triggered in the requested rollout")
    finally:
        if original_reset_idx is not None:
            env.unwrapped._reset_idx = original_reset_idx
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
