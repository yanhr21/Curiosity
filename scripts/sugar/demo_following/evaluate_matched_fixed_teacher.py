#!/usr/bin/env python3
"""Frozen matched evaluation for correct versus unrelated demo packages.

The policy, official Refiner, and selected-demo scorers are read-only.
All actor/critic/ICM/predictor tactile inputs stay exact zero and no TacSL
sensor is read.  PhysX hand contact is archived only as a labelled audit
diagnostic; it is never an actor or reward-model input.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import traceback

# Cluster H200 jobs require the system NVIDIA ICD before AppLauncher imports
# Isaac Sim.  Callers may still override this explicitly for another host.
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("DISPLAY", "")

from isaaclab.app import AppLauncher


HOST = socket.gethostname()
if HOST.startswith(("mgmtserver", "login")):
    raise SystemExit(f"Refusing matched demo evaluation on {HOST}")
if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Matched demo evaluation requires a retained Slurm allocation")

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / (
    "experiments/demo_following/matched_reward_identity_same_teacher_v1/"
    "seed161581/correct/update_0064/protocol.json"
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
parser.add_argument(
    "--arm",
    choices=(
        "correct_demo",
        "wrong_demo",
        "zero_demo",
        "task_only",
        "unrelated_demo",
        "wrong_teacher_correct_reward",
        "wrong_teacher_unrelated_reward",
        "same_teacher_correct_reward",
        "same_teacher_unrelated_reward",
    ),
    required=True,
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--seed", type=int, default=120381)
parser.add_argument(
    "--updates",
    default="1,128,512",
    help="Frozen checkpoints to evaluate; 128 alone is allowed for a labelled preview.",
)
parser.add_argument(
    "--teacher-only-zero-residual",
    action="store_true",
    help=(
        "Correct-teacher prerequisite gate: load the admitted environment and "
        "physics package but zero the frozen actor output layer after loading, "
        "so every deterministic learned residual is exactly zero."
    ),
)
parser.add_argument(
    "--phase-initialization",
    choices=("reference-aware", "reset-zero-diagnostic"),
    default="reference-aware",
    help=(
        "Phase-event only. The reset-zero option exists solely to reproduce "
        "the historical phase bug for the matched scorer ablation."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
try:
    REQUESTED_UPDATES = tuple(int(value) for value in args.updates.split(","))
except ValueError as error:
    raise SystemExit(f"invalid --updates value: {args.updates}") from error
if not (
    REQUESTED_UPDATES == (1, 128, 512)
    or REQUESTED_UPDATES == (32, 64)
    or (
        len(REQUESTED_UPDATES) == 1
        and REQUESTED_UPDATES[0] >= 64
        and REQUESTED_UPDATES[0] % 64 == 0
    )
):
    raise SystemExit(
        "--updates must be 1,128,512, 32,64, or one positive 64-update endpoint"
    )
if args.teacher_only_zero_residual and args.arm not in (
    "wrong_teacher_correct_reward",
    "same_teacher_correct_reward",
):
    raise SystemExit(
        "teacher-only zero-residual gate requires the correct CarryBox teacher arm"
    )
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402

import sugar_rl.tasks  # noqa: E402,F401
from sugar_rl.tasks.locomanip.agents.rsl_rl_smp_icm_cfg import (  # noqa: E402
    SMPICMSugarNativeZeroPreservingFixedLowLrRunnerCfg,
)
from sugar_rl.tasks.locomanip.direct_tactile_history import (  # noqa: E402
    explicit_zero_tactile_force_history,
)
from sugar_rl.tasks.locomanip.goal_carry_mdp import (  # noqa: E402
    previous_applied_action_policy_units,
)
from sugar_rl.tasks.locomanip.goal_tactile_strategy import (  # noqa: E402
    explicit_zero_anti_repeat_strategy_observation,
    explicit_zero_tactile_external_cost,
    explicit_zero_tactile_slip_observation,
)
from sugar_rl.tasks.locomanip.latent_contact_dynamics_events import (  # noqa: E402
    apply_stratified_latent_contact_dynamics,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_smp_icm_goal_env_cfg import (  # noqa: E402
    NoTactileGoalRobotEnvCfg,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_smp_icm_goal_coherent_env_cfg import (  # noqa: E402
    GoalCoherentLatentRobotEnvCfg,
)
from sugar_rl.utils.demo_reward_runtime import (  # noqa: E402
    FrozenDemoRewardRuntimeCfg,
    FrozenDemoRewardScorer,
)
from sugar_rl.utils.demo_event_reward_runtime import (  # noqa: E402
    FrozenPhaseAwareDemoEventScorer,
    FrozenPhaseAwareDemoEventScorerCfg,
    extract_goal_policy_core,
)
from sugar_rl.utils.official_refiner_nominal_teacher import (  # noqa: E402
    OfficialRefinerResidualVecEnvWrapper,
)
from sugar_rl.utils.wrong_demo_teacher_anneal import (  # noqa: E402
    WrongReferenceFixedOfficialRefinerResidualVecEnvWrapper,
    WrongReferenceScheduledOfficialRefinerResidualVecEnvWrapper,
)
from sugar_rl.utils.raw_termination_capture import RawTerminationCapture  # noqa: E402
from sugar_rl.utils.sugar_native_curiosity_ppo import (  # noqa: E402
    SugarNativeZeroPreservingTactileActorCritic,
)


TASK_ID = "Sugar-G129dof-CarryBox-SMP-ICM-Goal-Coherent-Latent"
UPDATES = REQUESTED_UPDATES
PROFILES_PER_UPDATE = 20
NUM_ENVS = len(UPDATES) * PROFILES_PER_UPDATE
PREVIEW_UPDATE128 = UPDATES == (128,)
TERMINATION_NAMES = (
    "time_out",
    "success",
    "unsafe_fall",
    "box_out_of_workspace",
)
TASK_REWARD_TERMS = (
    "goal_position",
    "goal_orientation",
    "lift_fraction",
    "goal_stability",
)


def expand_fixed_one_wrapper_batch_state(
    state: dict[str, object], *, evaluation_num_envs: int
) -> tuple[dict[str, object], dict[str, object]]:
    """Expand invariant fixed-teacher authority state to evaluation profiles.

    Training stores one wrapper value per training environment.  The phase-event
    evaluator runs two frozen checkpoints in one scene, so its batch is twice as
    large.  Only the three fixed-one authority tensors are batch-shaped; their
    values must be exactly false/zero/one before they may be replicated.
    """
    if evaluation_num_envs <= 0:
        raise ValueError("evaluation_num_envs must be positive")
    if state.get("teacher_authority_contract") != "fixed_one":
        raise ValueError("phase-event evaluation requires fixed-one authority")
    expanded = dict(state)
    expected = {
        "release_latched": False,
        "release_progress": 0,
        "teacher_coefficient": 1.0,
    }
    source_num_envs = None
    for name, fill_value in expected.items():
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.numel():
            raise ValueError(f"invalid fixed-one wrapper tensor: {name}")
        if source_num_envs is None:
            source_num_envs = int(value.numel())
        elif int(value.numel()) != source_num_envs:
            raise ValueError("fixed-one wrapper batch tensors disagree")
        expected_value = torch.full_like(value, fill_value)
        if not torch.equal(value, expected_value):
            raise ValueError(f"non-invariant fixed-one wrapper tensor: {name}")
        expanded[name] = torch.full(
            (evaluation_num_envs,),
            fill_value,
            dtype=value.dtype,
            device=value.device,
        )
    audit = {
        "protocol": "fixed_one_wrapper_batch_expansion_v1",
        "source_num_envs": source_num_envs,
        "evaluation_num_envs": evaluation_num_envs,
        "expanded_fields": sorted(expected),
        "values_preserved": True,
        "passed": True,
    }
    return expanded, audit


def clone_tensor_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def tensor_states_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def enforce_determinism() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def workspace_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_runtime_overlay(path: Path) -> dict[str, object]:
    overlay = json.loads(path.read_text(encoding="utf-8"))
    if "base_config" not in overlay:
        return overlay
    base_path = workspace_path(str(overlay["base_config"]))
    base = json.loads(base_path.read_text(encoding="utf-8"))

    def merge(left: dict[str, object], right: dict[str, object]):
        result = dict(left)
        for name, value in right.items():
            if isinstance(value, dict) and isinstance(result.get(name), dict):
                result[name] = merge(result[name], value)
            else:
                result[name] = value
        return result

    return merge(base, overlay)


def configure_explicit_zero(
    cfg: NoTactileGoalRobotEnvCfg | GoalCoherentLatentRobotEnvCfg,
) -> None:
    cfg.observations.tactile_history.force_history.func = (
        explicit_zero_tactile_force_history
    )
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.v16_tactile_slip_belief.func = (
            explicit_zero_tactile_slip_observation
        )
        group.anti_repeat_strategy_state.func = (
            explicit_zero_anti_repeat_strategy_observation
        )
    cfg.rewards.tactile_slip.func = explicit_zero_tactile_external_cost
    cfg.rewards.repeated_failed_strategy.func = (
        explicit_zero_tactile_external_cost
    )


def apply_repeated_training_physics(
    base_env, proof: dict[str, object]
) -> dict[str, object]:
    source = proof["coherent_latent_dynamics"]
    if source.get("passed") is not True:
        raise RuntimeError("training coherent-physics audit did not pass")
    tuple20 = source["tuple_by_environment"]
    required = (
        "mass_scale",
        "static_friction",
        "dynamic_friction",
        "com_y_m",
        "pulse_delta_velocity_w_mps",
    )
    if set(tuple20) != set(required):
        raise RuntimeError("training physics tuple schema drift")
    declared: dict[str, torch.Tensor] = {}
    for name in required:
        value = torch.as_tensor(tuple20[name], dtype=torch.float32)
        if value.shape[0] != PROFILES_PER_UPDATE:
            raise RuntimeError(f"training physics {name} is not 20-way")
        declared[name] = value.repeat(
            (len(UPDATES),) + (1,) * (value.ndim - 1)
        )
    term_cfg = base_env.event_manager.get_term_cfg("latent_contact_dynamics")
    term = term_cfg.func
    if not isinstance(term, apply_stratified_latent_contact_dynamics):
        raise TypeError("latent dynamics event class drift")
    term._tuple_cpu = {name: value.clone() for name, value in declared.items()}
    env_ids = torch.arange(NUM_ENVS, dtype=torch.long)
    term(base_env, env_ids, **term_cfg.params)
    observed = term.tuple_for_device("cpu")
    readback = {
        name: value.detach().cpu().clone()
        for name, value in term.last_readback.items()
    }
    exact = all(
        torch.equal(observed[name], declared[name])
        and torch.equal(readback[name], declared[name])
        for name in required
    ) and torch.equal(readback["env_ids"], env_ids)
    if not exact:
        raise RuntimeError("evaluation physics application/readback mismatch")
    return {
        "passed": True,
        "source_training_distribution_seed": int(source["distribution_seed"]),
        "updates": list(UPDATES),
        "profiles_per_update": PROFILES_PER_UPDATE,
        "tuple_by_environment": {
            name: value.tolist() for name, value in declared.items()
        },
    }


def apply_no_tactile_training_physics(
    base_env, proof: dict[str, object]
) -> dict[str, object]:
    """Restore the exact standard-SUGAR startup physics saved by training."""

    source = proof.get("no_tactile_startup_physics")
    if not isinstance(source, dict) or source.get("passed") is not True:
        raise RuntimeError("training no-tactile startup physics proof is missing")
    values = source.get("values")
    if not isinstance(values, dict):
        raise RuntimeError("training no-tactile startup physics values are missing")
    required = {
        "object_materials",
        "robot_materials",
        "object_masses",
        "object_inertias",
        "object_coms",
    }
    if set(values) != required:
        raise RuntimeError("training no-tactile startup physics schema drift")

    obj = base_env.scene["obj"]
    robot = base_env.scene["robot"]
    env_ids = torch.arange(NUM_ENVS, dtype=torch.long)
    current = {
        "object_materials": obj.root_physx_view.get_material_properties(),
        "robot_materials": robot.root_physx_view.get_material_properties(),
        "object_masses": obj.root_physx_view.get_masses(),
        "object_inertias": obj.root_physx_view.get_inertias(),
        "object_coms": obj.root_physx_view.get_coms(),
    }
    expected = {}
    for name, tensor in current.items():
        training_value = torch.as_tensor(
            values[name], dtype=tensor.dtype, device=tensor.device
        )
        training_shape = (PROFILES_PER_UPDATE, *tensor.shape[1:])
        if training_value.shape != training_shape:
            raise RuntimeError(
                f"training no-tactile physics {name} source shape drift: "
                f"expected {training_shape}, got {tuple(training_value.shape)}"
            )
        expected[name] = training_value.repeat(
            (len(UPDATES),) + (1,) * (training_value.ndim - 1)
        )
    for name in required:
        if expected[name].shape != current[name].shape:
            raise RuntimeError(
                f"training no-tactile physics {name} shape drift: "
                f"expected {tuple(current[name].shape)}, got {tuple(expected[name].shape)}"
            )

    obj.root_physx_view.set_material_properties(expected["object_materials"], env_ids)
    robot.root_physx_view.set_material_properties(expected["robot_materials"], env_ids)
    obj.root_physx_view.set_masses(expected["object_masses"], env_ids)
    obj.root_physx_view.set_inertias(expected["object_inertias"], env_ids)
    obj.root_physx_view.set_coms(expected["object_coms"], env_ids)

    observed = {
        "object_materials": obj.root_physx_view.get_material_properties(),
        "robot_materials": robot.root_physx_view.get_material_properties(),
        "object_masses": obj.root_physx_view.get_masses(),
        "object_inertias": obj.root_physx_view.get_inertias(),
        "object_coms": obj.root_physx_view.get_coms(),
    }
    readback_atol = {
        name: (
            float(torch.finfo(observed[name].dtype).eps)
            if name == "object_coms"
            else 1.0e-7
        )
        for name in required
    }
    readback_max_abs = {
        name: float((observed[name] - expected[name]).abs().max())
        for name in required
    }
    failed = [
        name
        for name in required
        if not torch.allclose(
            observed[name],
            expected[name],
            rtol=0.0,
            atol=readback_atol[name],
        )
    ]
    if failed:
        max_abs = {
            name: float((observed[name] - expected[name]).abs().max())
            for name in failed
        }
        max_abs_by_update = {
            name: [
                float(
                    (
                        observed[name][
                            index * PROFILES_PER_UPDATE :
                            (index + 1) * PROFILES_PER_UPDATE
                        ]
                        - expected[name][
                            index * PROFILES_PER_UPDATE :
                            (index + 1) * PROFILES_PER_UPDATE
                        ]
                    )
                    .abs()
                    .max()
                )
                for index in range(len(UPDATES))
            ]
            for name in failed
        }
        raise RuntimeError(
            "no-tactile training physics readback mismatch: "
            f"failed={failed}, max_abs={max_abs}, "
            f"max_abs_by_update={max_abs_by_update}"
        )
    return {
        "passed": True,
        "protocol": source.get("protocol"),
        "restored_from_training_proof": True,
        "same_profiles_repeated_exactly_across_updates": True,
        "readback_atol": readback_atol,
        "readback_max_abs": readback_max_abs,
        "updates": list(UPDATES),
        "profiles_per_update": PROFILES_PER_UPDATE,
        "values": {
            name: tensor.detach().cpu().tolist()
            for name, tensor in expected.items()
        },
    }


def audit_reconstructed_training_physics(
    base_env, expected_distribution_seed: int
) -> dict[str, object]:
    """Read back the deterministic 20-way startup tuple for a live preview."""

    term_cfg = base_env.event_manager.get_term_cfg("latent_contact_dynamics")
    term = term_cfg.func
    if not isinstance(term, apply_stratified_latent_contact_dynamics):
        raise TypeError("latent dynamics event class drift")
    values = {
        name: value.detach().cpu().clone()
        for name, value in term.tuple_for_device("cpu").items()
    }
    readback = {
        name: value.detach().cpu().clone()
        for name, value in term.last_readback.items()
    }
    required = (
        "mass_scale",
        "static_friction",
        "dynamic_friction",
        "com_y_m",
        "pulse_delta_velocity_w_mps",
    )
    expected_ids = torch.arange(NUM_ENVS, dtype=torch.long)
    exact = (
        set(values) == set(required)
        and all(value.shape[0] == NUM_ENVS for value in values.values())
        and all(
            name in readback and torch.equal(readback[name], values[name])
            for name in required
        )
        and torch.equal(readback.get("env_ids"), expected_ids)
        and int(term_cfg.params["distribution_seed"])
        == int(expected_distribution_seed)
    )
    if not exact:
        raise RuntimeError("preview physics startup/readback mismatch")
    return {
        "passed": True,
        "source_training_distribution_seed": int(expected_distribution_seed),
        "reconstructed_from_same_deterministic_event": True,
        "updates": list(UPDATES),
        "profiles_per_update": PROFILES_PER_UPDATE,
        "tuple_by_environment": {
            name: value.tolist() for name, value in values.items()
        },
    }


def apply_teacher_gate_nominal_physics(base_env) -> dict[str, object]:
    """Apply one repeated nominal tuple for the teacher prerequisite gate."""

    obj = base_env.scene["obj"]
    robot = base_env.scene["robot"]
    env_ids = torch.arange(NUM_ENVS, dtype=torch.long)

    # The explicit-zero matched scene intentionally has no TacSL-coupled
    # latent-contact event. Set the nominal tuple directly through PhysX and
    # verify the resulting object and robot material/mass state.
    default_mass = obj.data.default_mass.detach().clone()
    default_inertia = obj.data.default_inertia.detach().clone()
    obj.root_physx_view.set_masses(default_mass, env_ids)
    obj.root_physx_view.set_inertias(default_inertia, env_ids)

    object_materials = obj.root_physx_view.get_material_properties()
    object_materials[env_ids, :, 0] = 0.5
    object_materials[env_ids, :, 1] = 0.5
    object_materials[env_ids, :, 2] = 0.0
    obj.root_physx_view.set_material_properties(object_materials, env_ids)

    robot_materials = robot.root_physx_view.get_material_properties()
    robot_materials[env_ids, :, 0] = 0.5
    robot_materials[env_ids, :, 1] = 0.5
    robot_materials[env_ids, :, 2] = 0.0
    robot.root_physx_view.set_material_properties(robot_materials, env_ids)

    declared = {
        "mass_scale": torch.ones(NUM_ENVS, dtype=torch.float32),
        "static_friction": torch.full((NUM_ENVS,), 0.5, dtype=torch.float32),
        "dynamic_friction": torch.full((NUM_ENVS,), 0.5, dtype=torch.float32),
        "com_y_m": torch.zeros(NUM_ENVS, dtype=torch.float32),
        "pulse_delta_velocity_w_mps": torch.zeros(
            NUM_ENVS, 3, dtype=torch.float32
        ),
    }
    mass_after = obj.root_physx_view.get_masses().detach().cpu()
    inertia_after = obj.root_physx_view.get_inertias().detach().cpu()
    object_after = obj.root_physx_view.get_material_properties().detach().cpu()
    robot_after = robot.root_physx_view.get_material_properties().detach().cpu()
    physics_checks = {
        "mass": torch.allclose(
            mass_after,
            default_mass.detach().cpu(),
            rtol=0.0,
            atol=1.0e-7,
        ),
        "inertia": torch.allclose(
            inertia_after,
            default_inertia.detach().cpu(),
            rtol=0.0,
            atol=1.0e-7,
        ),
        "object_friction": torch.allclose(
            object_after[:, :, :2],
            torch.full_like(object_after[:, :, :2], 0.5),
            rtol=0.0,
            atol=1.0e-7,
        ),
        "object_restitution": torch.allclose(
            object_after[:, :, 2],
            torch.zeros_like(object_after[:, :, 2]),
            rtol=0.0,
            atol=1.0e-7,
        ),
        "robot_friction": torch.allclose(
            robot_after[:, :, :2],
            torch.full_like(robot_after[:, :, :2], 0.5),
            rtol=0.0,
            atol=1.0e-7,
        ),
        "robot_restitution": torch.allclose(
            robot_after[:, :, 2],
            torch.zeros_like(robot_after[:, :, 2]),
            rtol=0.0,
            atol=1.0e-7,
        ),
    }
    failed = [name for name, passed in physics_checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "teacher-gate nominal physics readback mismatch: "
            f"failed={failed}, "
            f"object_material_range="
            f"({float(object_after.min())}, {float(object_after.max())}), "
            f"robot_material_range="
            f"({float(robot_after.min())}, {float(robot_after.max())}), "
            f"mass_max_abs="
            f"{float((mass_after - default_mass.detach().cpu()).abs().max())}, "
            f"inertia_max_abs="
            f"{float((inertia_after - default_inertia.detach().cpu()).abs().max())}"
        )
    return {
        "passed": True,
        "teacher_prerequisite_nominal_physics": True,
        "updates": list(UPDATES),
        "profiles_per_update": PROFILES_PER_UPDATE,
        "tuple_by_environment": {
            name: value.tolist() for name, value in declared.items()
        },
    }


def restore_state_action_boundary(
    base_env,
    source_path: Path,
    selected_frame: int = 103,
    motion_loader_index: int = 45,
) -> tuple[dict[str, object], np.ndarray]:
    required = (
        "robot_root_state_w",
        "robot_joint_pos",
        "robot_joint_vel",
        "object_root_state_w",
        "motion_frame",
        "source_environment_origin_w",
        "selected_motion_id",
        "policy_actions_unclipped",
        "applied_actions_policy_units",
        "native_sample_phase",
    )
    with np.load(source_path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise KeyError(f"state/action source missing {missing}")
        source = {name: np.asarray(archive[name]) for name in required}
    if int(source["selected_motion_id"].reshape(-1)[0]) != 45:
        raise RuntimeError("evaluation reset is not official motion 45")
    if str(source["native_sample_phase"].reshape(-1)[0]) != "pre_action":
        raise RuntimeError("evaluation source is not pre-action state")
    reference_frame = int(source["motion_frame"].reshape(-1)[selected_frame])
    if selected_frame == 103 and reference_frame != 299:
        raise RuntimeError("evaluation source reference frame is not 299")
    policy_actions = np.asarray(source["policy_actions_unclipped"], np.float32)
    applied_actions = np.asarray(
        source["applied_actions_policy_units"], np.float32
    )
    conversion_error = np.abs(policy_actions - applied_actions)
    if (
        policy_actions.shape != applied_actions.shape
        or policy_actions.shape[1] != 29
        or float(conversion_error.max()) > 2.0e-6
    ):
        raise RuntimeError("official source action conversion drift")

    origins = base_env.scene.env_origins.detach().cpu().numpy()
    source_origin = np.asarray(
        source["source_environment_origin_w"], np.float32
    )
    translation = origins - source_origin[None]
    translation_norm = np.linalg.norm(translation, axis=1)
    canonical_environment_index = int(np.argmin(translation_norm))
    robot_root = np.repeat(
        source["robot_root_state_w"][selected_frame : selected_frame + 1],
        NUM_ENVS,
        axis=0,
    ).astype(np.float32, copy=True)
    object_root = np.repeat(
        source["object_root_state_w"][selected_frame : selected_frame + 1],
        NUM_ENVS,
        axis=0,
    ).astype(np.float32, copy=True)
    robot_root[:, :3] += translation
    object_root[:, :3] += translation
    joint_pos = np.repeat(
        source["robot_joint_pos"][selected_frame : selected_frame + 1],
        NUM_ENVS,
        axis=0,
    ).astype(np.float32, copy=True)
    joint_vel = np.repeat(
        source["robot_joint_vel"][selected_frame : selected_frame + 1],
        NUM_ENVS,
        axis=0,
    ).astype(np.float32, copy=True)
    ids = torch.arange(NUM_ENVS, dtype=torch.long, device=base_env.device)
    base_env.scene["robot"].write_root_state_to_sim(
        torch.as_tensor(robot_root, device=base_env.device), env_ids=ids
    )
    base_env.scene["robot"].write_joint_state_to_sim(
        torch.as_tensor(joint_pos, device=base_env.device),
        torch.as_tensor(joint_vel, device=base_env.device),
        env_ids=ids,
    )
    base_env.scene["obj"].write_root_state_to_sim(
        torch.as_tensor(object_root, device=base_env.device), env_ids=ids
    )
    command = base_env.command_manager.get_term("motion")
    command.motion_id.fill_(motion_loader_index)
    command.time_steps.fill_(reference_frame)
    command.last_reset_timestep.fill_(reference_frame)
    command._use_motion_data.fill_(True)
    command._record_reference_targets(ids)
    object_position = torch.as_tensor(object_root[:, :3], device=base_env.device)
    command.initial_obj_pos_w.copy_(object_position)
    command.initial_obj_height_w.copy_(object_position[:, 2])
    command.ever_lifted.zero_()
    command.goal_stable_counter.zero_()
    command.episode_steps.zero_()
    base_env.episode_length_buf.fill_(1)
    base_env._sugar_direct_tactile_history_cache = {}
    base_env.sim.forward()
    base_env.scene.update(dt=0.0)

    previous = torch.as_tensor(
        policy_actions[selected_frame - 1],
        device=base_env.device,
    ).reshape(1, -1).expand(NUM_ENVS, -1)
    base_env.action_manager.process_action(previous)
    previous_applied = torch.as_tensor(
        applied_actions[selected_frame - 1], device=base_env.device
    ).reshape(1, -1).expand(NUM_ENVS, -1)
    actor_exact = torch.equal(base_env.action_manager.action, previous)
    applied_exact = torch.equal(
        previous_applied_action_policy_units(base_env), previous_applied
    )
    if not actor_exact or not applied_exact:
        raise RuntimeError("previous action 102 did not reach both views")
    return (
        {
            "selected_motion_id": 45,
            "source_index": selected_frame,
            "reference_frame": reference_frame,
            "previous_action_source_index": selected_frame - 1,
            "previous_actor_action_exact": bool(actor_exact),
            "previous_applied_action_exact": bool(applied_exact),
            "tactile_arrays_loaded": False,
            "tactile_sensor_data_read": False,
            "source_action_conversion_max_abs": float(conversion_error.max()),
            "canonical_environment_index": canonical_environment_index,
            "canonical_origin_translation_norm_m": float(
                translation_norm[canonical_environment_index]
            ),
        },
        policy_actions[selected_frame].copy(),
    )


def construct_policy(
    observations: dict[str, torch.Tensor],
    env,
    state: dict[str, torch.Tensor],
) -> SugarNativeZeroPreservingTactileActorCritic:
    runner = SMPICMSugarNativeZeroPreservingFixedLowLrRunnerCfg().to_dict()
    policy_cfg = dict(runner["policy"])
    if policy_cfg.pop("class_name") != (
        "SugarNativeZeroPreservingTactileActorCritic"
    ):
        raise RuntimeError("matched demo actor class drift")
    policy = SugarNativeZeroPreservingTactileActorCritic(
        observations,
        runner["obs_groups"],
        env.num_actions,
        **policy_cfg,
    ).to(env.device)
    policy.initialize_residual_mean_exact_zero()
    policy.load_state_dict(state, strict=True)
    policy.eval()
    policy.requires_grad_(False)
    return policy


def observation_subset(
    observations: dict[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    return {name: value[start:stop] for name, value in observations.items()}


def deterministic_action(policy, observations) -> torch.Tensor:
    actor_obs = policy.actor_obs_normalizer(policy.get_actor_obs(observations))
    policy.update_distribution(actor_obs)
    return policy.action_mean.detach().clone()


def filtered_contact_vector(base_env, name: str) -> torch.Tensor:
    value = base_env.scene[name].data.force_matrix_w_history
    if value is None or value.shape[0] != NUM_ENVS or value.shape[-1] != 3:
        raise RuntimeError(f"rigid contact sensor {name} geometry drift")
    # Select the strongest vector in the short sensor history/body/filter set.
    flat = value.reshape(NUM_ENVS, -1, 3)
    index = torch.linalg.vector_norm(flat, dim=-1).argmax(dim=-1)
    return flat[torch.arange(NUM_ENVS, device=flat.device), index]


def ordered_joint_ids(base_env, env) -> tuple[torch.Tensor, list[str]]:
    term = base_env.action_manager.get_term("JointPositionAction")
    names = list(term._joint_names)
    if names != list(env.teacher_joint_names):
        raise RuntimeError("policy/teacher joint order drift")
    raw = term._joint_ids
    if isinstance(raw, slice):
        ids = torch.arange(
            base_env.scene["robot"].data.joint_pos.shape[1],
            dtype=torch.long,
            device=base_env.device,
        )[raw]
    else:
        ids = torch.as_tensor(raw, dtype=torch.long, device=base_env.device)
    if tuple(ids.shape) != (29,):
        raise RuntimeError("matched demo evaluation requires 29 controlled joints")
    return ids, names


def capture_state(base_env, joint_ids: torch.Tensor) -> dict[str, np.ndarray]:
    command = base_env.command_manager.get_term("motion")
    return {
        "robot_root_state_w": base_env.scene["robot"].data.root_state_w.detach().cpu().numpy().astype(np.float32),
        "robot_joint_pos": base_env.scene["robot"].data.joint_pos[:, joint_ids].detach().cpu().numpy().astype(np.float32),
        "robot_joint_vel": base_env.scene["robot"].data.joint_vel[:, joint_ids].detach().cpu().numpy().astype(np.float32),
        # Evaluation-only whole-body geometry. It is archived after the actor
        # call and never enters observations, rewards, or the predictor.
        "robot_body_position_w": base_env.scene["robot"].data.body_pos_w.detach().cpu().numpy().astype(np.float32),
        "object_root_state_w": base_env.scene["obj"].data.root_state_w.detach().cpu().numpy().astype(np.float32),
        "goal_position_w": command.obj_target_pos_w.detach().cpu().numpy().astype(np.float32),
        "goal_orientation_wxyz": command.obj_target_quat_w.detach().cpu().numpy().astype(np.float32),
        "goal_position_error_m": torch.linalg.vector_norm(command.obj_pos_w - command.obj_target_pos_w, dim=-1).detach().cpu().numpy().astype(np.float32),
        "goal_orientation_error_rad": math_utils.quat_error_magnitude(command.obj_quat_w, command.obj_target_quat_w).detach().cpu().numpy().astype(np.float32),
        "lift_height_m": (command.obj_pos_w[:, 2] - command.initial_obj_height_w).detach().cpu().numpy().astype(np.float32),
        "left_hand_rigid_contact_force_w": filtered_contact_vector(base_env, "left_hand_forces").detach().cpu().numpy().astype(np.float32),
        "right_hand_rigid_contact_force_w": filtered_contact_vector(base_env, "right_hand_forces").detach().cpu().numpy().astype(np.float32),
        "left_foot_box_contact_force_w": filtered_contact_vector(base_env, "left_foot_forces").detach().cpu().numpy().astype(np.float32),
        "right_foot_box_contact_force_w": filtered_contact_vector(base_env, "right_foot_forces").detach().cpu().numpy().astype(np.float32),
    }


def replace_envs(
    current: dict[str, np.ndarray], terminal: dict[str, np.ndarray], ids: np.ndarray
) -> None:
    if set(current) != set(terminal):
        raise RuntimeError("terminal/current state capture schema drift")
    for name in current:
        current[name][ids] = terminal[name][ids]


def make_scorer(
    path: Path,
    num_envs: int,
    device,
    *,
    phase_event: bool = False,
    selected_option: str | None = None,
    phase_horizon_steps: int = 650,
):
    if phase_event:
        if selected_option not in {"correct", "unrelated"}:
            raise ValueError("phase-event scorer requires a selected option")
        return FrozenPhaseAwareDemoEventScorer(
            num_envs=num_envs,
            device=device,
            cfg=FrozenPhaseAwareDemoEventScorerCfg(
                runtime_config_path=str(path),
                selected_option=selected_option,
                phase_horizon_steps=phase_horizon_steps,
            ),
        )
    config = load_runtime_overlay(path)
    return FrozenDemoRewardScorer(
        num_envs=num_envs,
        device=device,
        cfg=FrozenDemoRewardRuntimeCfg(
            config_path=str(path),
            gamma=float(config["potential"]["gamma"]),
            eta=float(config["potential"]["eta"]),
            failure_closed_policy_index=int(
                config["potential"]["failure_closed_policy_index"]
            ),
        ),
    )


def summaries(
    arrays: dict[str, np.ndarray], selected_key: str
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    done = arrays["done"]
    for update_index, update in enumerate(UPDATES):
        for profile in range(PROFILES_PER_UPDATE):
            env_index = update_index * PROFILES_PER_UPDATE + profile
            hits = np.flatnonzero(done[:, env_index])
            last_transition = int(hits[0]) if hits.size else args.steps - 1
            valid_t = np.arange(args.steps) <= last_transition
            last_frame = min(last_transition + 1, args.steps)
            valid_f = np.arange(args.steps + 1) <= last_frame
            task = arrays["weighted_task_outcome_reward"][:, env_index]
            selected = arrays[f"demo_{selected_key}_reward"][:, env_index]
            conflict = valid_t & (task * selected < 0.0)
            left = np.linalg.norm(
                arrays["left_hand_rigid_contact_force_w"][:, env_index],
                axis=-1,
            )
            right = np.linalg.norm(
                arrays["right_hand_rigid_contact_force_w"][:, env_index],
                axis=-1,
            )
            root = arrays["robot_root_state_w"][valid_f, env_index]
            root_quat_wxyz = root[:, 3:7]
            root_up_z = 1.0 - 2.0 * (
                np.square(root_quat_wxyz[:, 1])
                + np.square(root_quat_wxyz[:, 2])
            )
            maximum_root_height_loss_m = float(
                root[0, 2] - root[:, 2].min()
            )
            minimum_root_up_z = float(root_up_z.min())
            physical_robot_fall = bool(
                maximum_root_height_loss_m >= 0.35
            )
            comparison_key = (
                "wrong"
                if "demo_wrong_component_mse" in arrays
                else "unrelated"
            )
            mean_comparison_loss = float(
                arrays[f"demo_{comparison_key}_component_mse"][
                    valid_f, env_index
                ].sum(axis=-1).mean()
            )
            records.append(
                {
                    "policy_update": update,
                    "profile_index": profile,
                    "first_done_step": int(hits[0]) if hits.size else -1,
                    "termination_counts": {
                        name: int(
                            np.count_nonzero(
                                arrays["raw_termination"][:, env_index, i]
                                & valid_t
                            )
                        )
                        for i, name in enumerate(TERMINATION_NAMES)
                    },
                    "maximum_lift_height_m": float(
                        arrays["lift_height_m"][valid_f, env_index].max()
                    ),
                    "final_lift_height_m": float(
                        arrays["lift_height_m"][last_frame, env_index]
                    ),
                    "minimum_goal_position_error_m": float(
                        arrays["goal_position_error_m"][valid_f, env_index].min()
                    ),
                    "final_goal_position_error_m": float(
                        arrays["goal_position_error_m"][last_frame, env_index]
                    ),
                    "cumulative_weighted_task_outcome_reward": float(
                        task[valid_t].sum()
                    ),
                    "cumulative_selected_demo_feedback": float(
                        selected[valid_t].sum()
                    ),
                    "task_demo_conflict_steps": int(conflict.sum()),
                    "task_demo_conflict_fraction": float(
                        conflict.sum() / max(1, valid_t.sum())
                    ),
                    "mean_correct_demo_predicted_loss": float(
                        arrays["demo_correct_component_mse"][
                            valid_f, env_index
                        ].sum(axis=-1).mean()
                    ),
                    "mean_wrong_demo_predicted_loss": float(
                        mean_comparison_loss
                    ),
                    # Explicit name for the fixed-teacher identity experiment;
                    # the legacy field above is retained for older result
                    # consumers and contains this same value in that branch.
                    "mean_unrelated_demo_predicted_loss": float(
                        mean_comparison_loss
                    ),
                    "mean_selected_demo_predicted_loss": float(
                        arrays[f"demo_{selected_key}_component_mse"][
                            valid_f, env_index
                        ].sum(axis=-1).mean()
                    ),
                    "left_rigid_contact_frames": int(
                        np.count_nonzero(left[valid_f] > 0.1)
                    ),
                    "right_rigid_contact_frames": int(
                        np.count_nonzero(right[valid_f] > 0.1)
                    ),
                    "bilateral_rigid_contact_frames": int(
                        np.count_nonzero(
                            (left[valid_f] > 0.1) & (right[valid_f] > 0.1)
                        )
                    ),
                    "minimum_robot_root_height_m": float(root[:, 2].min()),
                    "maximum_robot_root_height_loss_m": (
                        maximum_root_height_loss_m
                    ),
                    "minimum_robot_root_up_z": minimum_root_up_z,
                    "physical_robot_fall": physical_robot_fall,
                }
            )
    return records


def main() -> None:
    if args.steps < 64:
        raise ValueError("matched demo frozen evaluation requires at least 64 steps")
    enforce_determinism()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("strict evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    for name in (
        "CURIOSITY_TACSL_R15_USD",
        "CURIOSITY_TACSL_LEFT_MOUNT_TRANSLATION_OFFSET",
        "CURIOSITY_TACSL_RIGHT_MOUNT_TRANSLATION_OFFSET",
    ):
        if os.environ.get(name) is not None:
            raise RuntimeError(f"explicit-zero evaluation forbids {name}")

    config_path = args.config.expanduser().resolve()
    config = load_runtime_overlay(config_path)
    config_protocol = config.get("protocol")
    is_authority_rework_v3 = (
        config_protocol
        == "sugar_plan11_demo_conflict_authority_rework_matched_v3"
    )
    is_wrong_teacher_reward_conflict = (
        config_protocol == "sugar_plan11_wrong_teacher_reward_conflict_v1"
    )
    is_fixed_teacher_identity = (
        config_protocol == "sugar_plan11_fixed_teacher_demo_identity_v2"
    )
    is_teacher_floor_overfit = (
        config_protocol == "sugar_plan11_teacher_floor_overfit_v1"
    )
    is_phase_event_reward = (
        config_protocol == "sugar_phase_event_reward_matched_policy_v1"
    )
    is_teacher_demo64 = (
        is_wrong_teacher_reward_conflict
        or is_fixed_teacher_identity
        or is_teacher_floor_overfit
        or is_phase_event_reward
    )
    legacy_preview_update128 = PREVIEW_UPDATE128 and not is_teacher_demo64
    unrelated_teacher_arm = (
        is_wrong_teacher_reward_conflict
        or (
            is_fixed_teacher_identity
            and args.arm == "wrong_teacher_unrelated_reward"
        )
    )
    if (
        config_protocol
        not in {
            "sugar_plan11_demo_conflict_zero_tactile_matched_v2",
            "sugar_plan11_demo_conflict_authority_rework_matched_v3",
            "sugar_plan11_wrong_teacher_reward_conflict_v1",
            "sugar_plan11_fixed_teacher_demo_identity_v2",
            "sugar_plan11_teacher_floor_overfit_v1",
            "sugar_phase_event_reward_matched_policy_v1",
        }
        or config.get("execution_ready") is not True
        or config["shared_runtime"].get("tactile_regime")
        != "explicit_zero_control"
    ):
        raise RuntimeError("matched demo training config is not admitted")
    for record in config["artifacts"].values():
        if "path" in record and not workspace_path(record["path"]).exists():
            raise FileNotFoundError(workspace_path(record["path"]))
    if legacy_preview_update128:
        training_record = config.get("preview_training_config", {})
        training_config_path = workspace_path(training_record.get("path", ""))
        if (
            not training_config_path.is_file()
        ):
            raise RuntimeError("update-128 preview training-config binding drift")

    output_dir = args.output_dir.expanduser().resolve()
    experiment_root = (ROOT / "experiments").resolve()
    if not output_dir.is_relative_to(experiment_root):
        raise ValueError("evaluation output must remain under experiments/")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    proof_path = workspace_path(config["arms"][args.arm]["output"])
    admission_path = proof_path.parent / "POSTCHECK_ADMISSION.json"
    proof = None
    admission = None
    if is_teacher_demo64:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if (
            proof.get("num_updates") != UPDATES[-1]
            or proof.get("tactile_regime") != "explicit_zero_control"
            or proof.get("protocol_config", {}).get("arm") != args.arm
            or (
                (
                    is_fixed_teacher_identity
                    or is_teacher_floor_overfit
                    or is_phase_event_reward
                )
                and UPDATES[-1] > 64
                and (
                    proof.get("resume_update") != UPDATES[-1] - 64
                    or proof.get("updates_executed_this_process") != 64
                )
            )
        ):
            raise RuntimeError("fixed-teacher interval training proof drift")
        if (
            is_fixed_teacher_identity
            or is_teacher_floor_overfit
            or is_phase_event_reward
        ):
            failed = [
                name
                for name, passed in proof.get("checks", {}).items()
                if passed is not True
            ]
            if (
                proof.get("passed") is not True
                or failed
                or proof.get("checks", {}).get(
                    "teacher_floor_schedule_reaches_exact_nonzero_floor"
                    if is_teacher_floor_overfit
                    else "explicit_zero_control_keeps_teacher_authority_fixed"
                ) is not True
                or proof.get("protocol") != config_protocol
                or (
                    is_phase_event_reward
                    and (
                        not isinstance(
                            proof.get("no_tactile_startup_physics"), dict
                        )
                        or proof["no_tactile_startup_physics"].get("passed")
                        is not True
                    )
                )
            ):
                raise RuntimeError("matched teacher-control proof is not admitted")
    elif not legacy_preview_update128:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        if (
            admission.get("protocol")
            != (
                "sugar_plan11_demo_conflict_authority_rework_training_postcheck_v3"
                if is_authority_rework_v3
                else "sugar_plan11_demo_conflict_explicit_zero_training_postcheck_v2r1"
            )
            or admission.get("passed") is not True
            or not all(admission.get("checks", {}).values())
            or admission.get("arm") != args.arm
            or proof.get("num_updates") != 512
            or proof.get("tactile_regime") != "explicit_zero_control"
            or proof.get("protocol_config", {}).get("arm") != args.arm
        ):
            raise RuntimeError("training arm proof is not admitted")
    elif args.arm != "unrelated_demo":
        raise RuntimeError("the update-128 preview is only admitted for unrelated_demo")
    checkpoint_paths: dict[int, Path] = {}
    checkpoints: dict[int, dict[str, object]] = {}
    for update in UPDATES:
        if legacy_preview_update128:
            path = proof_path.parent / f"policy_update{update}.pt"
        else:
            record = proof["checkpoint_records"][str(update)]
            path = Path(record["path"]).resolve()
        checkpoint = torch.load(path, map_location=args.device, weights_only=True)
        if int(checkpoint.get("iteration", -1)) != update:
            raise RuntimeError(f"checkpoint {update} iteration drift")
        checkpoint_paths[update] = path
        checkpoints[update] = checkpoint

    shared = config["shared_runtime"]
    # The active phase-event and teacher-only controls use the original SUGAR
    # scene with no TacSL assets. Historical packages retain their archived
    # scene so old traces are not silently reinterpreted.
    active_no_tactile_scene = args.teacher_only_zero_residual or is_phase_event_reward
    cfg = (
        NoTactileGoalRobotEnvCfg()
        if active_no_tactile_scene
        else GoalCoherentLatentRobotEnvCfg()
    )
    cfg.terminations.dropped_after_lift = None
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = args.seed
    cfg.sim.device = args.device
    causal_same_teacher = "same_teacher_correct_reward" in config["arms"]
    if args.teacher_only_zero_residual or causal_same_teacher:
        correct_arm = (
            "same_teacher_correct_reward"
            if causal_same_teacher
            else "wrong_teacher_correct_reward"
        )
        task_motion_folder = workspace_path(
            config["arms"][correct_arm]["teacher_motion_folder"]
        )
    else:
        task_motion_folder = ROOT / "SUGAR/data/CarryBox"
    cfg.commands.motion.motion_folder = str(task_motion_folder)
    cfg.commands.motion.teacher_motion_folder = (
        str(
            workspace_path(
                config["arms"][args.arm].get(
                    "teacher_motion_folder",
                    shared.get("wrong_teacher_motion_folder"),
                )
            )
        )
        if is_teacher_demo64
        else None
    )
    cfg.commands.motion.use_generator = False
    cfg.commands.motion.generator_checkpoint_path = None
    cfg.commands.motion.start_init_env_ratio = 0.0
    cfg.commands.motion.init_with_ref = True
    if not active_no_tactile_scene:
        cfg.events.latent_contact_dynamics.params["distribution_seed"] = int(
            shared["latent_physics_distribution_seed"]
        )
        if is_teacher_floor_overfit and not active_no_tactile_scene:
            fixed = shared["fixed_physics_profile"]
            event = cfg.events.latent_contact_dynamics
            event.params["mass_scale_range"] = (
                float(fixed["mass_scale"]), float(fixed["mass_scale"])
            )
            event.params["static_friction_range"] = (
                float(fixed["static_friction"]),
                float(fixed["static_friction"]),
            )
            event.params["dynamic_friction_range"] = (
                float(fixed["dynamic_friction"]),
                float(fixed["dynamic_friction"]),
            )
            event.params["com_y_range_m"] = (
                float(fixed["com_y_m"]), float(fixed["com_y_m"])
            )
            event.params["pulse_magnitude_range_mps"] = (0.0, 0.0)
    configure_explicit_zero(cfg)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
    )
    gym_env = None
    raw_capture = None
    reset_binding = None
    try:
        gym_env = gym.make(TASK_ID, cfg=cfg)
        enforce_determinism()
        teacher_path = workspace_path(
            config["artifacts"]["official_refiner_teacher"]["path"]
        )
        env = (
            WrongReferenceFixedOfficialRefinerResidualVecEnvWrapper(
                gym_env,
                teacher_path,
                residual_scale=float(shared["residual_scale"]),
                clip_actions=None,
            )
            if (is_fixed_teacher_identity or is_phase_event_reward)
            else (
            WrongReferenceScheduledOfficialRefinerResidualVecEnvWrapper(
                gym_env,
                teacher_path,
                residual_scale=float(shared["residual_scale"]),
                teacher_anneal_control_steps=(
                    int(shared["teacher_anneal_updates"]) * 24
                ),
                teacher_final_coefficient=float(
                    shared.get("teacher_final_coefficient", 0.0)
                ),
                clip_actions=None,
            )
            if (is_wrong_teacher_reward_conflict or is_teacher_floor_overfit)
            else OfficialRefinerResidualVecEnvWrapper(
                gym_env,
                teacher_path,
                residual_scale=float(shared["residual_scale"]),
                release_mode="linear",
                linear_release_steps=4,
                teacher_release_scope="arm_only",
                support_teacher_mode="advancing",
                drop_grace_steps=0,
                post_release_residual_scale=float(
                    shared["postfailure_arm_residual_scale"]
                ),
                clip_actions=None,
            )
            )
        )
        wrapper_state_batch_audit = None
        if is_teacher_demo64:
            wrapper_state = checkpoints[UPDATES[-1]][
                "residual_wrapper_state_dict"
            ]
            if is_phase_event_reward:
                wrapper_state, wrapper_state_batch_audit = (
                    expand_fixed_one_wrapper_batch_state(
                        wrapper_state,
                        evaluation_num_envs=NUM_ENVS,
                    )
                )
            env.load_checkpoint_state_dict(wrapper_state)
            if is_wrong_teacher_reward_conflict and (
                env.release.global_control_steps
                != env.release.linear_release_steps
                or bool(torch.count_nonzero(env.release.coefficient))
            ):
                raise RuntimeError(
                    "update-64 checkpoint did not restore zero teacher authority"
                )
            if is_teacher_floor_overfit and (
                env.release.global_control_steps
                != env.release.linear_release_steps
                or not bool(torch.all(env.release.coefficient == 0.25))
            ):
                raise RuntimeError(
                    "update-128 checkpoint did not restore teacher floor"
                )
            if (is_fixed_teacher_identity or is_phase_event_reward) and (
                env.release.mode != "fixed_one"
                or not bool(torch.all(env.release.coefficient == 1.0))
                or bool(env.release.release_latched.any())
                or bool(env.release.release_progress.any())
            ):
                raise RuntimeError(
                    "update-64 checkpoint did not restore fixed teacher authority"
                )
        base_env = env.unwrapped
        scene_sensor_names = tuple(sorted(base_env.scene.sensors.keys()))
        robot_body_names = tuple(base_env.scene["robot"].body_names)
        forbidden_tactile_sensor_names = {
            "left_palm_tactile",
            "right_palm_tactile",
        }
        forbidden_tactile_body_fragments = (
            "tacsl",
            "elastomer",
            "anatomical_",
        )
        no_tactile_scene_proof = {
            "protocol": "sugar_demo_no_tactile_scene_v1",
            "scene_sensor_names": list(scene_sensor_names),
            "robot_body_count": len(robot_body_names),
            "passed": bool(
                forbidden_tactile_sensor_names.isdisjoint(scene_sensor_names)
                and not any(
                    fragment in body_name
                    for body_name in robot_body_names
                    for fragment in forbidden_tactile_body_fragments
                )
            ),
        }
        physics = (
            apply_teacher_gate_nominal_physics(base_env)
            if args.teacher_only_zero_residual
            else apply_no_tactile_training_physics(base_env, proof)
            if is_phase_event_reward
            else audit_reconstructed_training_physics(
                base_env, int(shared["latent_physics_distribution_seed"])
            )
            if legacy_preview_update128
            else apply_repeated_training_physics(base_env, proof)
        )
        if args.teacher_only_zero_residual:
            command = base_env.command_manager.get_term("motion")

            def fixed_motion45_start(env_ids) -> None:
                ids = torch.as_tensor(
                    env_ids, dtype=torch.long, device=base_env.device
                )
                command.motion_id[ids] = 0
                command.time_steps[ids] = 0
                command._use_motion_data[ids] = True

            command._sample_init_state = fixed_motion45_start
            observations, _ = env.reset()
            previous_action = base_env.action_manager.action
            previous_applied = previous_applied_action_policy_units(base_env)
            reset_record = {
                "selected_motion_id": 45,
                "source_index": None,
                "reference_frame": 0,
                "previous_action_source_index": None,
                "previous_actor_action_exact": bool(
                    torch.count_nonzero(previous_action) == 0
                ),
                "previous_applied_action_exact": bool(
                    torch.count_nonzero(previous_applied) == 0
                ),
                "tactile_arrays_loaded": False,
                "tactile_sensor_data_read": False,
                "source_action_conversion_max_abs": None,
                "canonical_environment_index": 0,
                "canonical_origin_translation_norm_m": None,
                "reset_semantics": "motion45_frame0_standard_environment_reset",
            }
            source_action103 = None
        else:
            reset_record, source_action103 = restore_state_action_boundary(
                base_env,
                workspace_path(config["artifacts"]["state_action_source"]["path"]),
                selected_frame=(
                    int(shared["explicit_zero_source_frame"])
                    if is_teacher_demo64
                    else 103
                ),
                motion_loader_index=(0 if is_teacher_demo64 else 45),
            )
            observations = env.get_observations()
        if (
            observations["policy"].shape != (NUM_ENVS, 175)
            or observations["icm_vector"].shape != (NUM_ENVS, 115)
            or observations["tactile_history"].shape != (NUM_ENVS, 12000)
            or torch.count_nonzero(observations["tactile_history"]) != 0
        ):
            raise RuntimeError("explicit-zero observation schema/value drift")
        initial_goal_policy_core = None
        if is_phase_event_reward:
            initial_goal_policy_core = extract_goal_policy_core(
                observations["policy"],
                list(base_env.observation_manager.active_terms["policy"]),
            ).detach().cpu().numpy().astype(np.float32)
        with torch.inference_mode():
            teacher_observation, first_teacher_action = env.teacher.action()
        if source_action103 is None:
            first_teacher_error = torch.zeros_like(first_teacher_action)
        else:
            expected_action = torch.as_tensor(
                source_action103,
                device=base_env.device,
                dtype=first_teacher_action.dtype,
            ).reshape(1, -1).expand_as(first_teacher_action)
            first_teacher_error = first_teacher_action - expected_action
        first_teacher_max_abs_by_env = (
            first_teacher_error.abs().amax(dim=1).detach().cpu().numpy()
        )
        first_teacher_canonical_environment_index = int(
            reset_record["canonical_environment_index"]
        )
        first_teacher_canonical_max_abs = float(
            first_teacher_max_abs_by_env[
                first_teacher_canonical_environment_index
            ]
        )
        first_teacher_all_env_max_abs = float(
            first_teacher_max_abs_by_env.max()
        )
        teacher_observation_drift_from_env0 = (
            (teacher_observation - teacher_observation[0:1])
            .abs()
            .amax(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        # Replicating the source state into a different environment grid changes
        # the world-origin subtraction used to construct the same local 890-D
        # observation.  Environment zero is not necessarily the original source
        # origin once two checkpoint batches share a scene.  Gate the replica
        # closest to the recorded source origin at the original 2e-6 tolerance
        # and retain every translated-replica error as diagnostic evidence.
        source_action_tolerance = 2.0e-6
        if (
            not args.teacher_only_zero_residual
            and not unrelated_teacher_arm
            and first_teacher_canonical_max_abs > source_action_tolerance
        ):
            raise RuntimeError(
                "first teacher action does not reproduce the source boundary: "
                f"canonical_max_abs={first_teacher_canonical_max_abs}, "
                f"all_env_max_abs={first_teacher_all_env_max_abs}, "
                f"tolerance={source_action_tolerance}, "
                f"source_index={reset_record['source_index']}"
            )
        if (
            not args.teacher_only_zero_residual
            and unrelated_teacher_arm
            and first_teacher_canonical_max_abs <= 1.0e-3
        ):
            raise RuntimeError("wrong teacher is not distinct from CarryBox source")
        if is_wrong_teacher_reward_conflict and bool(
            torch.count_nonzero(env.release.coefficient)
        ):
            raise RuntimeError("teacher authority changed during reset")
        if is_teacher_floor_overfit and not bool(
            torch.all(env.release.coefficient == 0.25)
        ):
            raise RuntimeError("teacher floor changed during reset")
        if (is_fixed_teacher_identity or is_phase_event_reward) and not bool(
            torch.all(env.release.coefficient == 1.0)
        ):
            raise RuntimeError("fixed teacher authority changed during reset")

        policies = []
        policy_states = []
        for index, update in enumerate(UPDATES):
            start = index * PROFILES_PER_UPDATE
            stop = start + PROFILES_PER_UPDATE
            policy = construct_policy(
                observation_subset(observations, start, stop),
                env,
                checkpoints[update]["policy_state_dict"],
            )
            if args.teacher_only_zero_residual:
                policy.initialize_residual_mean_exact_zero()
            policies.append(policy)
            policy_states.append(clone_tensor_state(policy.state_dict()))

        if (
            is_fixed_teacher_identity
            or is_teacher_floor_overfit
            or is_phase_event_reward
        ):
            fixed_correct_key = (
                "same_teacher_correct_reward"
                if "same_teacher_correct_reward" in config["arms"]
                else "wrong_teacher_correct_reward"
            )
            fixed_unrelated_key = (
                "same_teacher_unrelated_reward"
                if "same_teacher_unrelated_reward" in config["arms"]
                else "wrong_teacher_unrelated_reward"
            )
            runtime_paths = {
                "correct": workspace_path(
                    config["arms"][fixed_correct_key]["demo_runtime_config"]
                ),
                "unrelated": workspace_path(
                    config["arms"][fixed_unrelated_key]["demo_runtime_config"]
                ),
            }
        else:
            runtime_paths = (
            {
                "correct": workspace_path(
                    config["arms"]["correct_demo"]["demo_runtime_config"]
                ),
                "wrong": workspace_path(
                    config["arms"]["wrong_demo"]["demo_runtime_config"]
                ),
                "zero": workspace_path(
                    config["arms"]["zero_demo"]["demo_runtime_config"]
                ),
            }
            )
        # The fixed-teacher V2 branch already binds the unrelated runtime
        # above through its own arm.  Only the legacy update-128 preview uses
        # the old ``unrelated_demo`` key.
        if args.arm == "unrelated_demo":
            runtime_paths["unrelated"] = workspace_path(
                config["arms"]["unrelated_demo"]["demo_runtime_config"]
            )
        scorers = {
            name: make_scorer(
                path,
                NUM_ENVS,
                base_env.device,
                phase_event=is_phase_event_reward,
                selected_option=name if is_phase_event_reward else None,
                phase_horizon_steps=int(
                    shared.get("demo_event_phase_horizon_steps", 650)
                ),
            )
            for name, path in runtime_paths.items()
        }
        if is_phase_event_reward:
            command = base_env.command_manager.get_term("motion")
            initial_phase_steps = command.last_reset_timestep.detach().clone()
            if args.phase_initialization == "reset-zero-diagnostic":
                initial_phase_steps.zero_()
            initial_demo = {
                name: scorer.begin(
                    observations,
                    initial_episode_steps=initial_phase_steps,
                )
                for name, scorer in scorers.items()
            }
        else:
            initial_demo = {
                name: scorer.begin(observations)
                for name, scorer in scorers.items()
            }
        selected_key = {
            "correct_demo": "correct",
            "wrong_demo": "wrong",
            "zero_demo": "zero",
            "task_only": "correct",
            "unrelated_demo": "unrelated",
            "wrong_teacher_correct_reward": "correct",
            "wrong_teacher_unrelated_reward": "unrelated",
            "same_teacher_correct_reward": "correct",
            "same_teacher_unrelated_reward": "unrelated",
        }[args.arm]

        joint_ids, joint_names = ordered_joint_ids(base_env, env)
        raw_capture = RawTerminationCapture(base_env.termination_manager)
        pre_reset: dict[str, object] = {}
        original_reset_idx = base_env._reset_idx
        reset_binding = (base_env, original_reset_idx)

        def capture_before_reset(env_ids):
            if pre_reset:
                raise RuntimeError("multiple reset calls within one transition")
            ids = (
                torch.as_tensor(env_ids, device=base_env.device)
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            pre_reset["ids"] = ids
            pre_reset["state"] = capture_state(base_env, joint_ids)
            return original_reset_idx(env_ids)

        base_env._reset_idx = capture_before_reset
        frame_lists = {name: [value] for name, value in capture_state(base_env, joint_ids).items()}
        if is_phase_event_reward:
            if initial_goal_policy_core is None:
                raise RuntimeError("phase-event initial policy core was not captured")
            frame_lists["goal_policy_core_observation"] = [
                initial_goal_policy_core
            ]
        demo_component_lists = {
            name: [
                np.zeros((NUM_ENVS, 1), dtype=np.float32)
                if is_phase_event_reward
                else initial_demo[name]["component_mse"].detach().cpu().numpy()
            ]
            for name in scorers
        }
        transition_lists: dict[str, list[np.ndarray]] = {
            "residual_action_mean": [],
            "teacher_action": [],
            "executed_action": [],
            "teacher_coefficient": [],
            "failure_closed": [],
            "done": [],
            "raw_termination": [],
            "manager_reward": [],
            "reward_terms": [],
            "weighted_task_outcome_reward": [],
            "external_constraint_reward": [],
            "terminal_pre_reset_state": [],
        }
        for name in scorers:
            transition_lists[f"demo_{name}_reward"] = []
            if is_phase_event_reward:
                transition_lists[f"demo_{name}_phase"] = []
                transition_lists[f"demo_{name}_ready"] = []
                transition_lists[f"demo_{name}_risk"] = []
                transition_lists[f"demo_{name}_weighted_uncertainty"] = []
        reward_term_names = list(base_env.reward_manager.active_terms)
        task_term_indices = [reward_term_names.index(name) for name in TASK_REWARD_TERMS]
        tactile_nonzero = 0
        tactile_abs_max = 0.0
        residual_abs_max = 0.0
        for _step in range(args.steps):
            pre_reset.clear()
            with torch.inference_mode():
                means = []
                for index, policy in enumerate(policies):
                    start = index * PROFILES_PER_UPDATE
                    stop = start + PROFILES_PER_UPDATE
                    means.append(
                        deterministic_action(
                            policy, observation_subset(observations, start, stop)
                        )
                    )
                residual = torch.cat(means, dim=0)
                residual_abs_max = max(
                    residual_abs_max, float(residual.abs().max())
                )
                before = raw_capture.completed_compute_count
                observations_next, manager_reward, rsl_done, _extras = env.step(
                    residual
                )
                done = rsl_done.to(dtype=torch.bool)
                raw_termination = raw_capture.snapshot_after_step(
                    done,
                    completed_compute_count_before_step=before,
                )
                demo_signals = {
                    name: scorer.process_step(observations_next, done)
                    for name, scorer in scorers.items()
                }
                goal_policy_core_next = (
                    extract_goal_policy_core(
                        observations_next["policy"],
                        list(base_env.observation_manager.active_terms["policy"]),
                    )
                    if is_phase_event_reward
                    else None
                )
            runtime = env.latest_step
            if runtime is None:
                raise RuntimeError("official residual wrapper omitted runtime")
            if bool(runtime.failure_closed.any()):
                raise RuntimeError("explicit-zero evaluation produced a failure event")
            tactile = observations_next["tactile_history"]
            tactile_nonzero += int(torch.count_nonzero(tactile))
            tactile_abs_max = max(tactile_abs_max, float(tactile.abs().max()))

            reward_terms = np.stack(
                [
                    (
                        base_env.reward_manager._step_reward[:, index]
                        * float(base_env.step_dt)
                    ).detach().cpu().numpy()
                    for index in range(len(reward_term_names))
                ],
                axis=-1,
            ).astype(np.float32)
            task_outcome = reward_terms[:, task_term_indices].sum(axis=-1)
            external = reward_terms.sum(axis=-1) - task_outcome
            transition_lists["residual_action_mean"].append(residual.detach().cpu().numpy())
            transition_lists["teacher_action"].append(runtime.teacher_action.detach().cpu().numpy())
            transition_lists["executed_action"].append(runtime.executed_action.detach().cpu().numpy())
            transition_lists["teacher_coefficient"].append(runtime.teacher_coefficient.detach().cpu().numpy())
            transition_lists["failure_closed"].append(runtime.failure_closed.detach().cpu().numpy())
            transition_lists["done"].append(done.detach().cpu().numpy())
            transition_lists["raw_termination"].append(raw_termination.detach().cpu().numpy())
            transition_lists["manager_reward"].append(manager_reward.detach().cpu().numpy())
            transition_lists["reward_terms"].append(reward_terms)
            transition_lists["weighted_task_outcome_reward"].append(
                (float(shared["reward_mix_without_demo"]["task_outcome"]) * task_outcome).astype(np.float32)
            )
            transition_lists["external_constraint_reward"].append(external.astype(np.float32))
            for name in scorers:
                transition_lists[f"demo_{name}_reward"].append(
                    demo_signals[name].reward.detach().cpu().numpy()
                )
                if is_phase_event_reward:
                    transition_lists[f"demo_{name}_phase"].append(
                        demo_signals[name]
                        .selected_demo_phase.detach()
                        .cpu()
                        .numpy()
                    )
                    transition_lists[f"demo_{name}_ready"].append(
                        demo_signals[name].next_ready.detach().cpu().numpy()
                    )
                    transition_lists[f"demo_{name}_risk"].append(
                        demo_signals[name].next_risk.detach().cpu().numpy()
                    )
                    transition_lists[
                        f"demo_{name}_weighted_uncertainty"
                    ].append(
                        demo_signals[name]
                        .next_weighted_uncertainty.detach()
                        .cpu()
                        .numpy()
                    )
                demo_component_lists[name].append(
                    (
                        demo_signals[name].next_risk[:, None]
                        if is_phase_event_reward
                        else demo_signals[name].component_mse
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

            state = capture_state(base_env, joint_ids)
            done_np = done.detach().cpu().numpy().astype(bool)
            terminal_mask = np.zeros(NUM_ENVS, dtype=bool)
            if done_np.any():
                if not pre_reset:
                    raise RuntimeError("done transition escaped terminal capture")
                ids = np.asarray(pre_reset["ids"], dtype=np.int64)
                if not np.array_equal(ids, np.flatnonzero(done_np)):
                    raise RuntimeError("reset IDs differ from returned done IDs")
                replace_envs(state, pre_reset["state"], ids)
                terminal_mask[ids] = True
            elif pre_reset:
                raise RuntimeError("terminal capture occurred without done")
            transition_lists["terminal_pre_reset_state"].append(terminal_mask)
            for name, value in state.items():
                frame_lists[name].append(value)
            if is_phase_event_reward:
                if goal_policy_core_next is None:
                    raise RuntimeError("phase-event next policy core was not captured")
                frame_lists["goal_policy_core_observation"].append(
                    goal_policy_core_next.detach().cpu().numpy().astype(np.float32)
                )
            observations = observations_next

        base_env._reset_idx = original_reset_idx
        reset_binding = None
        arrays = {
            name: np.stack(values)
            for name, values in frame_lists.items()
        }
        arrays.update(
            {
                name: np.stack(values)
                for name, values in transition_lists.items()
            }
        )
        for name, values in demo_component_lists.items():
            arrays[f"demo_{name}_component_mse"] = np.stack(values)
        arrays["policy_updates"] = np.asarray(UPDATES, dtype=np.int64)
        arrays["ordered_joint_names"] = np.asarray(joint_names)
        arrays["ordered_body_names"] = np.asarray(base_env.scene["robot"].body_names)
        arrays["reward_term_names"] = np.asarray(reward_term_names)
        arrays["termination_names"] = np.asarray(TERMINATION_NAMES)

        summary_records = summaries(arrays, selected_key)
        final_summaries = [
            record
            for record in summary_records
            if record["policy_update"] == UPDATES[-1]
        ]
        scorer_audits = {
            name: scorer.frozen_model_audit()
            for name, scorer in scorers.items()
        }
        checks = {
            "training_checkpoint_admitted_for_requested_scope": (
                legacy_preview_update128
                or is_teacher_demo64
                or (
                    admission.get("passed") is True
                    and all(admission.get("checks", {}).values())
                )
            ),
            "requested_frozen_checkpoints_loaded": len(policies) == len(UPDATES),
            "phase_event_fixed_one_wrapper_batch_restored": (
                wrapper_state_batch_audit is not None
                and wrapper_state_batch_audit["passed"] is True
                if is_phase_event_reward
                else True
            ),
            "policy_parameters_frozen": all(
                not parameter.requires_grad
                for policy in policies
                for parameter in policy.parameters()
            ),
            "policy_state_unchanged": all(
                tensor_states_equal(before, policy.state_dict())
                for before, policy in zip(policy_states, policies, strict=True)
            ),
            "teacher_only_residual_exact_zero": (
                residual_abs_max == 0.0
                if args.teacher_only_zero_residual
                else True
            ),
            "official_teacher_frozen": env.teacher.frozen_audit()["passed"],
            (
                "teacher_motion45_frame0_start_action_is_finite"
                if args.teacher_only_zero_residual
                else "wrong_teacher_first_action_differs_from_carrybox_source"
                if unrelated_teacher_arm
                else "closest_origin_first_teacher_action_matches_source"
                if is_phase_event_reward
                else "canonical_first_teacher_action_matches_source103"
            ): (
                bool(torch.isfinite(first_teacher_action).all())
                if args.teacher_only_zero_residual
                else first_teacher_canonical_max_abs > 1.0e-3
                if unrelated_teacher_arm
                else first_teacher_canonical_max_abs <= source_action_tolerance
            ),
            "all_first_teacher_actions_finite": bool(
                np.isfinite(first_teacher_max_abs_by_env).all()
            ),
            (
                "motion45_frame0_previous_action_exact_zero"
                if args.teacher_only_zero_residual
                else "previous_action102_restored_exact"
            ): reset_record["previous_actor_action_exact"] and reset_record["previous_applied_action_exact"],
            (
                "teacher_gate_nominal_physics_readback_exact"
                if args.teacher_only_zero_residual
                else "physics_profiles_repeat_exactly_across_updates"
            ): physics["passed"],
            "all_tactile_inputs_exact_zero": tactile_nonzero == 0 and tactile_abs_max == 0.0,
            "no_tactile_arrays_or_sensor_read": not reset_record["tactile_arrays_loaded"] and not reset_record["tactile_sensor_data_read"],
            "demo_control_has_no_tactile_scene": no_tactile_scene_proof["passed"],
            (
                "frozen_evaluation_teacher_coefficient_exact_floor"
                if is_teacher_floor_overfit
                else (
                    "frozen_evaluation_teacher_coefficient_exact_zero"
                    if is_wrong_teacher_reward_conflict
                    else "frozen_evaluation_teacher_coefficient_exact_one"
                )
            ): (
                np.count_nonzero(arrays["failure_closed"]) == 0
                and np.all(
                    arrays["teacher_coefficient"]
                    == (
                        0.25
                        if is_teacher_floor_overfit
                        else (0.0 if is_wrong_teacher_reward_conflict else 1.0)
                    )
                )
            ),
            "all_demo_predictors_frozen": all(
                (
                    audit["model_frozen"]
                    and audit["future_actual_events_used"] is False
                    if is_phase_event_reward
                    else (
                        audit["model_bitwise_frozen"]
                        and audit["all_parameters_require_grad_false"]
                        and audit["training_mode_false"] is True
                    )
                )
                for audit in scorer_audits.values()
            ),
            "phase_event_exact_policy_core_archived": (
                arrays.get("goal_policy_core_observation", np.empty(0)).shape
                == (args.steps + 1, NUM_ENVS, 121)
                if is_phase_event_reward
                else True
            ),
            "phase_event_runtime_signals_archived": (
                all(
                    arrays.get(f"demo_{name}_phase", np.empty(0)).shape
                    == (args.steps, NUM_ENVS)
                    and arrays.get(f"demo_{name}_ready", np.empty(0)).shape
                    == (args.steps, NUM_ENVS)
                    and arrays.get(f"demo_{name}_risk", np.empty(0)).shape
                    == (args.steps, NUM_ENVS)
                    and arrays.get(
                        f"demo_{name}_weighted_uncertainty", np.empty(0)
                    ).shape
                    == (args.steps, NUM_ENVS)
                    for name in scorers
                )
                if is_phase_event_reward
                else True
            ),
            "phase_event_initial_phase_contract_explicit": (
                args.phase_initialization
                in {"reference-aware", "reset-zero-diagnostic"}
                if is_phase_event_reward
                else True
            ),
            "all_numeric_arrays_finite": all(
                np.isfinite(value).all()
                for value in arrays.values()
                if np.issubdtype(value.dtype, np.number)
            ),
            "terminal_state_capture_matches_done": np.array_equal(
                arrays["terminal_pre_reset_state"], arrays["done"]
            ),
            "no_learning_or_optimizer_constructed": True,
            "teacher_only_bilateral_contact": (
                all(
                    record["bilateral_rigid_contact_frames"] > 0
                    for record in final_summaries
                )
                if args.teacher_only_zero_residual
                else True
            ),
            "teacher_only_lifts_box_at_least_5cm": (
                all(
                    record["maximum_lift_height_m"] >= 0.05
                    for record in final_summaries
                )
                if args.teacher_only_zero_residual
                else True
            ),
        }
        checks = {name: bool(value) for name, value in checks.items()}
        trace_path = staging / "TRACE.npz"
        np.savez_compressed(trace_path, **arrays)
        if args.teacher_only_zero_residual:
            result_protocol = "sugar_plan11_correct_teacher_zero_residual_gate_v1"
        elif is_phase_event_reward:
            result_protocol = (
                "sugar_phase_event_reward_matched_frozen_eval_32_64_v2"
            )
        elif is_fixed_teacher_identity:
            result_protocol = (
                "sugar_plan11_fixed_teacher_demo_identity_frozen_eval_interval_v3"
            )
        elif is_teacher_floor_overfit:
            result_protocol = (
                "sugar_plan11_teacher_floor_overfit_frozen_eval_v1"
            )
        elif is_wrong_teacher_reward_conflict:
            result_protocol = (
                "sugar_plan11_wrong_teacher_reward_conflict_frozen_eval64_v1"
            )
        elif legacy_preview_update128:
            result_protocol = "sugar_plan11_unrelated_kickbox21_update128_preview_v1"
        elif is_authority_rework_v3:
            result_protocol = (
                "sugar_plan11_demo_conflict_authority_rework_frozen_eval_v3"
            )
        else:
            result_protocol = "sugar_plan11_demo_conflict_zero_tactile_frozen_eval_v2"
        result = {
            "protocol": result_protocol,
            "passed": all(checks.values()),
            "preview_not_final_512_result": legacy_preview_update128,
            "claim_scope": (
                "Frozen deterministic CarryBox behavior evaluation under exact-zero "
                "tactile. The fixed official Refiner and selected internal demo "
                "remain read-only; PhysX hand force is a non-tactile audit "
                "diagnostic. This result cannot establish tactile usefulness or "
                "whole-hand sensor admission."
            ),
            "host": HOST,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "arm": args.arm,
            "selected_demo_telemetry": selected_key,
            "selected_demo_metric": (
                "calibrated_event_risk"
                if is_phase_event_reward
                else "predicted_component_mse"
            ),
            "selected_demo_feedback_applied_during_training": args.arm != "task_only",
            "teacher_only_zero_residual": args.teacher_only_zero_residual,
            "teacher_only_admission_rule": (
                "all profiles show bilateral contact and at least 0.05 m lift; "
                "physical fall remains reported as baseline outcome, not an "
                "admission condition for residual-policy learning"
                if args.teacher_only_zero_residual
                else None
            ),
            "residual_action_abs_max": residual_abs_max,
            "steps": args.steps,
            "seed": args.seed,
            "num_envs": NUM_ENVS,
            "policy_updates": list(UPDATES),
            "profiles_per_update": PROFILES_PER_UPDATE,
            "wrapper_state_batch_restore": wrapper_state_batch_audit,
            "checks": checks,
            "reset": reset_record,
            "phase_initialization": (
                {
                    "mode": args.phase_initialization,
                    "reference_frame": int(reset_record["reference_frame"]),
                    "initial_episode_steps": (
                        int(reset_record["reference_frame"])
                        if args.phase_initialization == "reference-aware"
                        else 0
                    ),
                }
                if is_phase_event_reward
                else None
            ),
            "first_teacher_observation_shape": list(teacher_observation.shape),
            "first_teacher_canonical_environment_index": (
                first_teacher_canonical_environment_index
            ),
            "first_teacher_action_vs_source_canonical_max_abs": (
                first_teacher_canonical_max_abs
            ),
            "first_teacher_action_vs_source_all_env_max_abs": (
                first_teacher_all_env_max_abs
            ),
            "first_teacher_action_vs_source_tolerance": source_action_tolerance,
            "first_teacher_action_vs_source_max_abs_by_env": (
                first_teacher_max_abs_by_env.tolist()
            ),
            "first_teacher_observation_drift_from_env0_max_abs_by_env": (
                teacher_observation_drift_from_env0.tolist()
            ),
            "physics": physics,
            "no_tactile_scene": no_tactile_scene_proof,
            "reward_term_names": reward_term_names,
            "summaries": summary_records,
            "final_update_aggregate": {
                name: float(np.mean([record[name] for record in final_summaries]))
                for name in (
                    "maximum_lift_height_m",
                    "final_goal_position_error_m",
                    "cumulative_weighted_task_outcome_reward",
                    "cumulative_selected_demo_feedback",
                    "task_demo_conflict_fraction",
                    "mean_correct_demo_predicted_loss",
                    "mean_wrong_demo_predicted_loss",
                    "mean_selected_demo_predicted_loss",
                    "bilateral_rigid_contact_frames",
                    "maximum_robot_root_height_loss_m",
                    "minimum_robot_root_up_z",
                    "physical_robot_fall",
                )
            },
            "demo_predictor_audits": scorer_audits,
            "training": {
                "config": str(config_path),
                "proof": None if legacy_preview_update128 else str(proof_path),
                "postcheck_admission": (
                    None
                    if (legacy_preview_update128 or is_teacher_demo64)
                    else str(admission_path)
                ),
                "checkpoints": {
                    str(update): {"path": str(path)}
                    for update, path in checkpoint_paths.items()
                },
            },
            "trace": {
                "path": str((output_dir / "TRACE.npz").relative_to(ROOT)),
                "bytes": trace_path.stat().st_size,
            },
            "source": {
                "path": str(Path(__file__).resolve()),
            },
        }
        (staging / "RESULT.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not result["passed"]:
            failed = [name for name, value in checks.items() if not value]
            raise RuntimeError(f"matched demo frozen evaluation failed: {failed}")
        raw_capture.restore()
        staging.rename(output_dir)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if reset_binding is not None:
            reset_binding[0]._reset_idx = reset_binding[1]
        if raw_capture is not None:
            raw_capture.restore()
        if gym_env is not None:
            gym_env.close()
        if staging.exists():
            failed_root = ROOT.parent / "Curiosity_archive" / "invalid_transient" / staging.name
            failed_root.parent.mkdir(parents=True, exist_ok=True)
            if failed_root.exists():
                shutil.rmtree(failed_root)
            shutil.move(str(staging), str(failed_root))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac Kit may replace an active exception with a successful exit
        # while closing.  A frozen evaluator failure must remain non-zero for
        # the retained-child status contract, so fail before Kit shutdown can
        # mask it.
        os._exit(1)
    simulation_app.close()
