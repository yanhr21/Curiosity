#!/usr/bin/env python3
"""Compute-only SUGAR/SMP/ICM policy integration and training runner.

The historical default runs the pure-discovery SUGAR environment.  The
explicit ``goal_recovery_multiphysics`` objective instead runs the full
goal-based coherent-latent task, separates task and constraint ledgers, keeps
original ICM outcome-independent, and removes only the drop-after-lift
termination so that set-down/regrasp remains possible.  Short endpoints are
diagnostics; only a predeclared long endpoint plus fresh deterministic
multiphysics evaluation can provide behavior evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import traceback

from isaaclab.app import AppLauncher


if socket.gethostname().startswith("mgmtserver"):
    raise SystemExit("Refusing Stage-H integration diagnostic on a login node")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion-folder", type=Path, required=True)
parser.add_argument("--prior-dir", type=Path, required=True)
parser.add_argument("--contact-source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--resume-checkpoint",
    type=Path,
    default=None,
    help=(
        "optional admitted rollout-boundary combined checkpoint; restores "
        "policy, Adam, SMP/ICM, demo accounting, and residual-wrapper state "
        "before continuing at the next global update"
    ),
)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--num-updates", type=int, default=1)
parser.add_argument(
    "--training-objective",
    choices=(
        "pure_discovery",
        "goal_recovery_multiphysics",
        "goal_recovery_native_authority",
    ),
    default="pure_discovery",
    help=(
        "pure discovery preserves the historical zero-outcome contract; "
        "goal recovery activates the full carrying objective and coherent "
        "mass/friction/COM training distribution"
    ),
)
parser.add_argument(
    "--policy-contract",
    choices=(
        "strict_mimickit",
        "sugar_native",
        "sugar_native_tactile_floor_lr",
        "sugar_native_zero_preserving_tactile_floor_lr",
        "sugar_native_zero_preserving_tactile_fixed_low_lr",
    ),
    default="strict_mimickit",
    help="named policy optimizer contract; SMP and ICM remain unchanged",
)
parser.add_argument(
    "--checkpoint-updates",
    type=str,
    default="1",
    help="comma-separated one-indexed updates to checkpoint and reload-audit",
)
parser.add_argument(
    "--tactile-regime",
    choices=("nominal", "h2r1_five_role", "explicit_zero_control"),
    default="nominal",
    help=(
        "direct-TacSL input regime; H2R1 uses all five locked stress roles, "
        "while explicit_zero_control is a declared no-sensor ablation"
    ),
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--action-seed",
    type=int,
    default=None,
    help=(
        "optional rollout-action RNG reset after all frozen model loading; "
        "used only by predeclared matched controls"
    ),
)
parser.add_argument(
    "--strict-deterministic-torch",
    action="store_true",
    help=(
        "require deterministic PyTorch/CuDNN algorithms and disabled TF32; "
        "CUBLAS_WORKSPACE_CONFIG must also be set before process startup"
    ),
)
parser.add_argument(
    "--causal-contact-bootstrap-v2",
    action="store_true",
    help=(
        "restore the recorded previous official action and four real TacSL "
        "frames when the diagnostic starts from a mid-trajectory contact "
        "state; required by the corrected posture-adaptive redo"
    ),
)
parser.add_argument(
    "--demo-reward-config",
    type=Path,
    default=None,
    help=(
        "optional frozen selected-demo potential runtime configuration; "
        "kept separate from original ICM and SMP"
    ),
)
parser.add_argument(
    "--demo-reward-telemetry-config",
    type=Path,
    default=None,
    help=(
        "load the same frozen predictor read-only without adding its score "
        "to policy reward; used by matched no-demo posture capacity/formal "
        "arms"
    ),
)
parser.add_argument(
    "--demo-event-reward-config",
    type=Path,
    default=None,
    help="frozen phase-aware dense contact/event reward scale config",
)
parser.add_argument(
    "--demo-event-selected-option",
    choices=("correct", "unrelated"),
    default=None,
)
parser.add_argument(
    "--demo-event-phase-horizon-steps",
    type=int,
    default=650,
    help="shared causal clock horizon; Carry45 and Kick21 both contain 660 frames",
)
parser.add_argument(
    "--admission-only",
    action="store_true",
    help=(
        "validate the phase-event protocol and load its frozen model, then "
        "exit before environment creation or any PPO update"
    ),
)
parser.add_argument(
    "--protocol-config",
    type=Path,
    default=None,
    help="optional hash-bound protocol record for a matched experiment",
)
parser.add_argument(
    "--protocol-arm",
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
        "icm_policy_on",
        "icm_policy_weight_zero",
    ),
    default=None,
    help=(
        "explicit arm name for the Plan-11 four-way demonstration-conflict "
        "protocol; older matched protocols continue to infer their two arms"
    ),
)
parser.add_argument(
    "--reference-waypoint-foundation-config",
    type=Path,
    default=None,
    help=(
        "hash-bound W1 nominal/3x-mass source-reset and object-waypoint "
        "foundation contract; requires a passing paired W0 audit"
    ),
)
parser.add_argument(
    "--paper-cws-runtime-config",
    type=Path,
    default=None,
    help=(
        "optional hash-bound public-paper CWS/TacSL runtime reward config; "
        "training-only and separate from actor inputs and original ICM"
    ),
)
parser.add_argument(
    "--paper-cws-guidance-weight",
    type=float,
    default=0.0,
    help="non-negative external paper-CWS policy-reward weight",
)
parser.add_argument(
    "--nominal-teacher-checkpoint",
    type=Path,
    default=None,
    help="enable the frozen official-Refiner residual action wrapper",
)
parser.add_argument(
    "--teacher-wrapper-mode",
    choices=(
        "arm_only_v1",
        "posture_adaptive_v1",
        "wrong_reference_anneal_v1",
        "wrong_reference_fixed_v1",
    ),
    default="arm_only_v1",
    help=(
        "explicit action-routing adapter; posture_adaptive_v1 retains the "
        "same serious policy/optimizer and opens only the frozen 14/11/4 "
        "authority route"
    ),
)
parser.add_argument(
    "--wrong-teacher-motion-folder",
    type=Path,
    default=None,
    help="motion-id-compatible folder used only by the frozen wrong teacher",
)
parser.add_argument(
    "--teacher-anneal-updates",
    type=int,
    default=0,
    help="global full-body wrong-teacher anneal horizon in PPO updates",
)
parser.add_argument(
    "--teacher-final-coefficient",
    type=float,
    default=0.0,
    help="nonzero floor for the declared global teacher schedule",
)
parser.add_argument(
    "--explicit-zero-source-frame",
    type=int,
    default=103,
    help="pre-action source index used by the exact-zero state/action reset",
)
parser.add_argument("--residual-scale", type=float, default=0.05)
parser.add_argument(
    "--post-release-residual-scale",
    type=float,
    default=None,
    help=(
        "optional causal arm residual scale reached only after the direct-"
        "TacSL failure release; support joints retain --residual-scale"
    ),
)
parser.add_argument(
    "--teacher-release-mode",
    choices=("immediate", "linear", "fixed_one"),
    default="linear",
)
parser.add_argument("--teacher-linear-release-steps", type=int, default=4)
parser.add_argument(
    "--teacher-reference-advance-mode",
    choices=(
        "legacy_pre_step",
        "command_manager_only",
        "goal_teacher_post_step_once",
    ),
    default="legacy_pre_step",
    help=(
        "legacy_pre_step preserves earlier residual studies; "
        "command_manager_only lets the native SUGAR command term advance "
        "the official reference; goal_teacher_post_step_once advances only "
        "the frozen nominal teacher after each non-reset goal-task step"
    ),
)
parser.add_argument(
    "--teacher-release-scope",
    choices=("full_body", "arm_only"),
    default="full_body",
)
parser.add_argument(
    "--support-teacher-mode",
    choices=("advancing", "failure_latched"),
    default="advancing",
)
parser.add_argument(
    "--posture-pre-failure-residual-scale",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--posture-post-failure-residual-scale",
    type=float,
    default=0.40,
)
parser.add_argument(
    "--posture-post-failure-teacher-floor",
    type=float,
    default=0.65,
)
parser.add_argument(
    "--drop-grace-steps",
    type=int,
    default=0,
    help=(
        "recurring per-episode bounded window opened by the first raw "
        "dropped_after_lift; only the locked 64-step exposure control may "
        "set this nonzero"
    ),
)
parser.add_argument(
    "--reward-control",
    choices=(
        "full",
        "smp_policy_weight_zero",
        "icm_policy_weight_zero",
        "foundation_icm_policy_weight_zero",
        "plan11_icm_policy_weight_zero",
    ),
    default="full",
    help=(
        "locked policy-credit control; both SMP and original ICM are still "
        "scored and updated in every arm"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _enforce_strict_torch_determinism() -> None:
    """Reapply the frozen deterministic contract after runtime imports/init."""

    if not args.strict_deterministic_torch:
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


_enforce_strict_torch_determinism()

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import sugar_rl.tasks  # noqa: E402,F401
from sugar_rl.tasks.locomanip.goal_carry_mdp import (  # noqa: E402
    previous_applied_action_policy_units,
)
from sugar_rl.tasks.locomanip.direct_tactile_stress import (  # noqa: E402
    H2_TACTILE_STRESS_ROLES,
    configure_h2_direct_tactile_stress,
    direct_tactile_stress_audit,
)
from sugar_rl.tasks.locomanip.direct_tactile_history import (  # noqa: E402
    direct_tactile_force_history,
    explicit_zero_tactile_force_history,
)
from sugar_rl.tasks.locomanip.goal_tactile_strategy import (  # noqa: E402
    ExplicitZeroTactileStrategyControlRuntime,
    explicit_zero_anti_repeat_strategy_observation,
    explicit_zero_tactile_external_cost,
    explicit_zero_tactile_slip_observation,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_smp_icm_goal_env_cfg import (  # noqa: E402
    PureDiscoveryRobotEnvCfg,
    TACTILE_RUNTIME_PARAMS,
)
from sugar_rl.tasks.locomanip.robots.g129dof.train_refiner.carry_box_smp_icm_goal_coherent_env_cfg import (  # noqa: E402
    GoalCoherentLatentRobotEnvCfg,
)
from sugar_rl.tasks.locomanip.latent_contact_dynamics_events import (  # noqa: E402
    apply_stratified_latent_contact_dynamics,
)
from sugar_rl.tasks.locomanip.agents.rsl_rl_smp_icm_cfg import (  # noqa: E402
    SMPICMPureDiscoveryRunnerCfg,
    SMPICMSugarNativeFloorLrRunnerCfg,
    SMPICMSugarNativeRunnerCfg,
    SMPICMSugarNativeZeroPreservingFloorLrRunnerCfg,
    SMPICMSugarNativeZeroPreservingFixedLowLrRunnerCfg,
)
from sugar_rl.utils.official_smp_policy_optimizer import (  # noqa: E402
    OfficialSMPPolicyOptimizerAdapter,
    OfficialSMPTactileActorCritic,
)
from sugar_rl.utils.original_icm_trainer import (  # noqa: E402
    ICMTransitionBatch,
)
from sugar_rl.utils.sugar_native_curiosity_ppo import (  # noqa: E402
    SugarNativeCuriosityPPO,
    SugarNativeTactileFloorLrPPO,
    SugarNativeTactileActorCritic,
    SugarNativeZeroPreservingTactileActorCritic,
    SugarNativeZeroPreservingTactileFixedLowLrPPO,
    SugarNativeZeroPreservingTactileFloorLrPPO,
)
from sugar_rl.utils.smp_icm_reward_integration import (  # noqa: E402
    EXTERNAL_CONSTRAINT_TERMS,
    OUTCOME_REWARD_TERMS,
    SMPICMRewardMixCfg,
    SMPICMRolloutIntegrator,
)
from sugar_rl.utils.official_refiner_nominal_teacher import (  # noqa: E402
    OfficialRefinerResidualVecEnvWrapper,
)
from sugar_rl.utils.wrong_demo_teacher_anneal import (  # noqa: E402
    WrongReferenceFixedOfficialRefinerResidualVecEnvWrapper,
    WrongReferenceScheduledOfficialRefinerResidualVecEnvWrapper,
)
from sugar_rl.utils.reference_waypoint_foundation import (  # noqa: E402
    ReferenceWaypointFoundationReset,
    ReferenceWaypointSource,
)
from sugar_rl.utils.posture_adaptive_refiner_teacher import (  # noqa: E402
    PostureAdaptiveOfficialRefinerResidualVecEnvWrapper,
)
from sugar_rl.utils.demo_reward_runtime import (  # noqa: E402
    DemoRewardAugmentedSMPICMRolloutIntegrator,
    FrozenDemoRewardRuntimeCfg,
    FrozenDemoRewardScorer,
)
from sugar_rl.utils.demo_event_reward_runtime import (  # noqa: E402
    DemoEventRewardAugmentedSMPICMRolloutIntegrator,
    FrozenPhaseAwareDemoEventScorer,
    FrozenPhaseAwareDemoEventScorerCfg,
    extract_goal_policy_core,
)
from sugar_rl.utils.paper_cws_rollout_integration import (  # noqa: E402
    PaperCWSAugmentedSMPICMRolloutIntegrator,
)
from sugar_rl.utils.paper_cws_runtime_reward import (  # noqa: E402
    OfficialTacSLPaperCWSReward,
    PaperCWSRuntimeCfg,
)

# Some Isaac/RSL transitive imports set global cuDNN policy. The strict
# matched-arm contract must be the last writer before any experiment work.
_enforce_strict_torch_determinism()


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PURE_DISCOVERY_TASK_ID = "Sugar-G129dof-CarryBox-SMP-ICM-Pure-Discovery"
GOAL_RECOVERY_TASK_ID = "Sugar-G129dof-CarryBox-SMP-ICM-Goal-Coherent-Latent"
TEACHER_FLOOR_FIXED_PHYSICS_PROFILE = {
    "mass_scale": 1.0,
    "static_friction": 0.6,
    "dynamic_friction": 0.5,
    "com_y_m": 0.0,
    "pulse_delta_velocity_w_mps": [0.0, 0.0, 0.0],
}
EXPECTED_TACSL_MOUNT_ENVIRONMENT = {
    "CURIOSITY_TACSL_R15_USD": str(
        (
            WORKSPACE_ROOT
            / "experiments/sugar_reproduction/assets/official_tacsl/"
            "gelsight_r15_finger/gelsight_r15_finger.usd"
        ).resolve()
    ),
    "CURIOSITY_TACSL_LEFT_MOUNT_TRANSLATION_OFFSET": (
        "-0.004606,-0.041890,0.005119"
    ),
    "CURIOSITY_TACSL_RIGHT_MOUNT_TRANSLATION_OFFSET": (
        "-0.005480,0.063320,0.025027"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deep_update(
    base: dict[str, object], update: dict[str, object]
) -> dict[str, object]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_demo_reward_config(path: Path) -> dict[str, object]:
    overlay = json.loads(path.read_text(encoding="utf-8"))
    if "base_config" not in overlay:
        return overlay
    base_path = Path(str(overlay["base_config"])).expanduser()
    if not base_path.is_absolute():
        base_path = (WORKSPACE_ROOT / base_path).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    return _deep_update(base, overlay)


def _workspace_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (WORKSPACE_ROOT / path).resolve()
    )


def _load_paper_cws_runtime_config(
    path: Path,
) -> tuple[dict[str, object], PaperCWSRuntimeCfg]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol",
        "status",
        "paper",
        "runtime",
        "bound_offline_report",
        "claim_boundary",
    }
    if set(payload) != required:
        raise ValueError("paper-CWS runtime config schema drift")
    if (
        payload["protocol"] != "sugar_paper_cws_runtime_reward_config_v3"
        or payload["status"]
        != "paper_formula_reproduction_not_official_chord_code"
        or payload["paper"] != "arXiv:2607.00033v1"
        or payload["claim_boundary"]
        != {
            "training_only_privileged_reward": True,
            "actor_receives_sdf_normals": False,
            "original_icm_receives_sdf_normals": False,
            "out_of_support_reward_policy": "strict_zero",
            "invalid_transition_reward_policy": (
                "strict_zero_and_excluded_from_contact_geometry"
            ),
            "slip_is_separate": True,
            "task_and_safety_are_separate": True,
            "policy_effectiveness_claimed": False,
        }
    ):
        raise ValueError("paper-CWS runtime semantic boundary drift")
    runtime = dict(payload["runtime"])
    reference_path = _workspace_path(runtime["reference_arrays_path"])
    runtime["reference_arrays_path"] = str(reference_path)
    cfg = PaperCWSRuntimeCfg(**runtime)
    expected = {
        "reference_arrays_sha256": (
            "ee5b5e0ba3bda1224baeed6f6541b13445a96b6a5d0a1bcd4911fc1d8e958b12"
        ),
        "reference_motion_frame_offset": 196,
        "friction_coefficient": 0.8,
        "friction_cone_edges": 8,
        "relative_tolerance": 0.2,
        "reward_variance": 32.0,
        "support_direction_seed": 260700033,
        "reference_support_key": "support_aggregate_nominal",
        "reference_normal_force_key": "normal_force_nominal",
        "active_force_epsilon": 0.0,
        "support_direction_count": 512,
        "support_basis_chunk_size": 64,
        "contact_epsilon": 1.0e-9,
    }
    if any(getattr(cfg, name) != value for name, value in expected.items()):
        raise ValueError("paper-CWS primary reproduction settings drift")
    report_binding = payload["bound_offline_report"]
    report_path = _workspace_path(report_binding["path"])
    if (
        not report_path.is_file()
        or _sha256(report_path) != report_binding["sha256"]
        or report_path.stat().st_size != int(report_binding["size_bytes"])
    ):
        raise RuntimeError("paper-CWS offline report binding drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("protocol")
        != "paper_cws_exact_official_sugar_tacsl_offline_v1"
        or report.get("passed") is not True
        or not all(report.get("checks", {}).values())
        or report["artifacts"]["arrays_sha256"]
        != cfg.reference_arrays_sha256
        or _workspace_path(report["artifacts"]["arrays"])
        != Path(cfg.reference_arrays_path)
    ):
        raise RuntimeError("paper-CWS offline admission record drift")
    return payload, cfg


def _load_reference_waypoint_foundation_config(
    path: Path,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol",
        "explicit_user_authorization_reference",
        "w0_pair_audit",
        "runner_source",
        "output",
        "checkpoint",
        "num_envs",
        "num_updates",
        "checkpoint_updates",
        "seed",
        "sources",
        "teacher",
        "policy",
        "reward",
        "claim_boundary",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(
            f"reference-waypoint W1 config is missing {missing}"
        )
    endpoint_by_protocol = {
        "sugar_reference_waypoint_foundation_w1_wiring_1_update_v1": (
            1,
            [1],
        ),
        "sugar_reference_waypoint_foundation_w1_64_update_v1": (
            64,
            [1, 16, 64],
        ),
        (
            "sugar_reference_waypoint_foundation_w1_relative_"
            "wiring_1_update_v2"
        ): (1, [1]),
        "sugar_reference_waypoint_foundation_w1_relative_64_update_v2": (
            64,
            [1, 16, 64],
        ),
        (
            "sugar_reference_waypoint_foundation_w1_relative_"
            "wiring_1_update_v3"
        ): (1, [1]),
        "sugar_reference_waypoint_foundation_w1_relative_64_update_v3": (
            64,
            [1, 16, 64],
        ),
    }
    if payload["protocol"] not in endpoint_by_protocol:
        raise ValueError("reference-waypoint W1 protocol drift")
    expected_updates, expected_checkpoints = endpoint_by_protocol[
        payload["protocol"]
    ]
    relative_foundation = "_relative_" in payload["protocol"]
    if (
        expected_updates == 64
        and "wiring_audit" not in payload
    ):
        raise KeyError(
            "formal reference-waypoint W1 config must bind the passing "
            "one-update wiring audit"
        )
    if (
        int(payload["num_envs"]) != 20
        or int(payload["num_updates"]) != expected_updates
        or payload["checkpoint_updates"] != expected_checkpoints
    ):
        raise ValueError(
            "reference-waypoint W1 endpoint/config geometry drift"
        )
    expected_reference_advance_mode = (
        "goal_teacher_post_step_once"
        if payload["protocol"].endswith("_v3")
        else "command_manager_only"
    )
    if payload["teacher"] != {
        "accepted_refiner_iteration": 10000,
        "coefficient": 1.0,
        "release_mode": "fixed_one",
        "reference_advance_mode": expected_reference_advance_mode,
        "residual_scale": 0.05,
    }:
        raise ValueError("reference-waypoint W1 teacher contract drift")
    if payload["policy"] != {
        "contract": "sugar_native_zero_preserving_tactile_fixed_low_lr",
        "direct_tacsl_history": [4, 2, 3, 20, 25],
        "mass_label_visible": False,
        "rgb_visible": False,
    }:
        raise ValueError("reference-waypoint W1 policy contract drift")
    reward = payload["reward"]
    if relative_foundation:
        if (
            set(reward)
            != {
                "task_outcome_weight",
                "smp_policy_weight",
                "icm_learner_active",
                "icm_policy_weight",
                "external_constraint_weight",
                "demo_reward_loaded",
                "paper_cws_guidance_weight",
                "paper_cws_runtime_config",
            }
            or reward["task_outcome_weight"] != 10.0
            or reward["smp_policy_weight"] != 0.5
            or reward["icm_learner_active"] is not True
            or reward["icm_policy_weight"] not in (0.0, 1.0)
            or reward["external_constraint_weight"] != 1.0
            or reward["demo_reward_loaded"] is not False
            or float(reward["paper_cws_guidance_weight"]) < 0.0
            or (
                (float(reward["paper_cws_guidance_weight"]) > 0.0)
                != (reward["paper_cws_runtime_config"] is not None)
            )
        ):
            raise ValueError(
                "relative reference-waypoint W1 reward contract drift"
            )
    elif reward != {
        "task_outcome_weight": 10.0,
        "smp_policy_weight": 0.5,
        "icm_learner_active": True,
        "icm_policy_weight": 0.0,
        "external_constraint_weight": 1.0,
        "demo_reward_loaded": False,
    }:
        raise ValueError("historical reference-waypoint W1 reward drift")
    expected_claim_boundary = {
        "training_evidence_only": expected_updates == 64,
        "wiring_diagnostic_only": expected_updates == 1,
        "stable_nominal_carrying_claimed": False,
        "recovery_claimed": False,
        "alternative_strategy_claimed": False,
        "icm_progress_claimed": False,
        "project_completion_claimed": False,
    }
    if payload["claim_boundary"] != expected_claim_boundary:
        raise ValueError("reference-waypoint W1 claim boundary drift")
    expected_sources = {
        "nominal_mass1": {
            "sha256": (
                "9f963be6e9b8079462d2947d2c246bde11580028316f79ff36c74cf84f70df27"
            ),
            "initial_frame": 103,
            "reference_frame": 299,
            "waypoint_reference_frame": (
                None if relative_foundation else 303
            ),
            "waypoint_relative_lift_m": (
                0.04 if relative_foundation else None
            ),
            "mass_scale": 1.0,
        },
        "heavy_mass3": {
            "sha256": (
                "e4fc7c20e2d9c5488d5480bcf42a216adb99586f8ae3fa715aca50744bc42261"
            ),
            "initial_frame": 112,
            "reference_frame": 308,
            "waypoint_reference_frame": (
                None if relative_foundation else 312
            ),
            "waypoint_relative_lift_m": (
                0.04 if relative_foundation else None
            ),
            "mass_scale": 3.0,
        },
    }
    sources = payload["sources"]
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("reference-waypoint W1 requires two sources")
    if [source.get("source_id") for source in sources] != [
        "nominal_mass1",
        "heavy_mass3",
    ]:
        raise ValueError(
            "reference-waypoint W1 source order must be nominal then heavy"
        )
    for source in sources:
        source_id = source.get("source_id")
        expected = expected_sources.get(source_id)
        if expected is None or any(
            source.get(name) != value for name, value in expected.items()
        ):
            raise ValueError(
                f"reference-waypoint W1 source drift: {source_id}"
            )
        if (
            float(source.get("static_friction", -1.0)) != 0.6
            or float(source.get("dynamic_friction", -1.0)) != 0.5
            or float(source.get("com_y_m", 1.0)) != 0.0
        ):
            raise ValueError(
                f"reference-waypoint W1 source physics drift: {source_id}"
            )
        source_path = _workspace_path(source["path"])
        if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
            raise RuntimeError(
                f"reference-waypoint W1 source binding drift: {source_path}"
            )
    w0 = payload["w0_pair_audit"]
    w0_path = _workspace_path(w0["path"])
    if (
        not w0_path.is_file()
        or _sha256(w0_path) != w0["sha256"]
        or w0_path.stat().st_size != int(w0["size_bytes"])
    ):
        raise RuntimeError("reference-waypoint W0 pair-audit binding drift")
    w0_record = json.loads(w0_path.read_text(encoding="utf-8"))
    if not (
        w0_record.get("protocol")
        == (
            "sugar_reference_waypoint_w0_relative_lift_pair_audit_v3"
            if relative_foundation
            else "sugar_reference_waypoint_w0_pair_audit_v1"
        )
        and w0_record.get("passed") is True
        and all(w0_record.get("checks", {}).values())
        and w0_record.get("claims", {}).get(
            "w0_no_learning_reachability_gate_passed"
        )
        is True
    ):
        raise RuntimeError(
            "reference-waypoint W1 requires a passing paired W0 audit"
        )
    if (
        expected_updates == 64
    ):
        wiring = payload["wiring_audit"]
        wiring_path = _workspace_path(wiring["path"])
        if (
            not wiring_path.is_file()
            or _sha256(wiring_path) != wiring["sha256"]
            or wiring_path.stat().st_size != int(wiring["size_bytes"])
        ):
            raise RuntimeError(
                "reference-waypoint W1 wiring-audit binding drift"
            )
        wiring_record = json.loads(
            wiring_path.read_text(encoding="utf-8")
        )
        if not (
            wiring_record.get("protocol")
            == (
                "sugar_reference_waypoint_foundation_w1_relative_"
                "wiring_audit_v3"
                if payload["protocol"].endswith("_v3")
                else "sugar_reference_waypoint_foundation_w1_wiring_audit_v1"
            )
            and wiring_record.get("passed") is True
            and all(wiring_record.get("checks", {}).values())
            and wiring_record.get("claims", {}).get("w1_wiring_passed")
            is True
        ):
            raise RuntimeError(
                "formal reference-waypoint W1 requires a passing "
                "one-update wiring audit"
            )
    runner = payload["runner_source"]
    runner_path = Path(__file__).resolve()
    if not (
        _workspace_path(runner["path"]) == runner_path
        and _sha256(runner_path) == runner["sha256"]
        and runner_path.stat().st_size == int(runner["size_bytes"])
    ):
        raise RuntimeError("reference-waypoint W1 runner binding drift")
    return payload


def _tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tactile_nonzero_by_role(
    tactile: torch.Tensor,
    role_names_by_env: tuple[str, ...] | None,
) -> dict[str, int] | None:
    if role_names_by_env is None:
        return None
    if tactile.shape[0] != len(role_names_by_env):
        raise ValueError("H2 tactile role/environment count drift")
    role_count = len(H2_TACTILE_STRESS_ROLES)
    if (
        len(role_names_by_env) < role_count
        or len(role_names_by_env) % role_count != 0
        or set(role_names_by_env) != set(H2_TACTILE_STRESS_ROLES)
    ):
        raise RuntimeError(
            "H2 telemetry requires a positive equal-size assignment across "
            "the five frozen roles"
        )
    expected_per_role = len(role_names_by_env) // role_count
    output: dict[str, int] = {}
    for role in H2_TACTILE_STRESS_ROLES:
        indices = [
            index
            for index, name in enumerate(role_names_by_env)
            if name == role
        ]
        if len(indices) != expected_per_role:
            raise RuntimeError(
                f"H2 role {role} does not have {expected_per_role} envs"
            )
        output[role] = int(torch.count_nonzero(tactile[indices]))
    return output


def _state_tree_sha256(value) -> str:
    """Hash nested tensor/optimizer state without using pickle serialization."""

    digest = hashlib.sha256()

    def update(item) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode("utf-8"))
            for child in item:
                update(child)
        elif item is None:
            digest.update(b"none")
        elif isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode("utf-8"))
            digest.update(repr(item).encode("utf-8"))
        else:
            raise TypeError(f"unsupported checkpoint hash type {type(item)!r}")

    update(value)
    return digest.hexdigest()


def _numeric_leaves(value) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Mapping):
        return [
            number
            for child in value.values()
            for number in _numeric_leaves(child)
        ]
    if isinstance(value, (tuple, list)):
        return [
            number for child in value for number in _numeric_leaves(child)
        ]
    return []


def _checkpoint_path(base: Path, update: int, final_update: int) -> Path:
    if update == final_update:
        return base
    return base.with_name(f"{base.stem}_update{update}{base.suffix}")


def _state_max_abs(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> float:
    if set(left) != set(right):
        raise ValueError("state dictionaries have different keys")
    maximum = 0.0
    for name in left:
        lhs = left[name].detach()
        rhs = right[name].detach().to(lhs.device)
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            raise ValueError(f"state tensor schema drift for {name}")
        if lhs.numel():
            if lhs.is_floating_point() or lhs.is_complex():
                difference = float(torch.abs(lhs - rhs).max())
            else:
                difference = 0.0 if torch.equal(lhs, rhs) else 1.0
            maximum = max(maximum, difference)
    return maximum


def _assert_experiment_output(path: Path) -> None:
    root = (WORKSPACE_ROOT / "experiments").resolve()
    if root not in path.resolve().parents:
        raise ValueError(f"Stage-H diagnostic must stay below experiments/: {path}")


def _valid_icm_batch(
    observation_t,
    action_t: torch.Tensor,
    observation_tp1,
    valid: torch.Tensor,
) -> ICMTransitionBatch:
    return ICMTransitionBatch(
        vector_obs_t=observation_t["icm_vector"],
        tactile_history_t=observation_t["tactile_history"].reshape(
            -1, 4, 2, 3, 20, 25
        ),
        applied_action_policy_units_t=action_t,
        vector_obs_tp1=observation_tp1["icm_vector"],
        tactile_history_tp1=observation_tp1["tactile_history"].reshape(
            -1, 4, 2, 3, 20, 25
        ),
        transition_valid=valid,
    ).select_valid()


def _construct_policy_algorithm(observations, env, runner_cfg):
    config = runner_cfg.to_dict()
    obs_groups = config["obs_groups"]
    policy_cfg = dict(config["policy"])
    policy_class_name = policy_cfg.pop("class_name")
    if policy_class_name == "OfficialSMPTactileActorCritic":
        policy_class = OfficialSMPTactileActorCritic
        algorithm_class_name = "OfficialSMPPolicyOptimizerAdapter"
        algorithm_class = OfficialSMPPolicyOptimizerAdapter
    elif policy_class_name in (
        "SugarNativeTactileActorCritic",
        "SugarNativeZeroPreservingTactileActorCritic",
    ):
        policy_class = (
            SugarNativeZeroPreservingTactileActorCritic
            if policy_class_name
            == "SugarNativeZeroPreservingTactileActorCritic"
            else SugarNativeTactileActorCritic
        )
        if (
            args.policy_contract
            == "sugar_native_zero_preserving_tactile_fixed_low_lr"
        ):
            algorithm_class_name = (
                "SugarNativeZeroPreservingTactileFixedLowLrPPO"
            )
            algorithm_class = (
                SugarNativeZeroPreservingTactileFixedLowLrPPO
            )
        elif (
            args.policy_contract
            == "sugar_native_zero_preserving_tactile_floor_lr"
        ):
            algorithm_class_name = (
                "SugarNativeZeroPreservingTactileFloorLrPPO"
            )
            algorithm_class = SugarNativeZeroPreservingTactileFloorLrPPO
        elif args.policy_contract == "sugar_native_tactile_floor_lr":
            algorithm_class_name = "SugarNativeTactileFloorLrPPO"
            algorithm_class = SugarNativeTactileFloorLrPPO
        else:
            algorithm_class_name = "SugarNativeCuriosityPPO"
            algorithm_class = SugarNativeCuriosityPPO
    else:
        raise ValueError(f"unexpected Stage-H policy class {policy_class_name}")
    policy = policy_class(
        observations, obs_groups, env.num_actions, **policy_cfg
    ).to(env.device)
    algorithm_cfg = dict(config["algorithm"])
    actual_algorithm_class_name = algorithm_cfg.pop("class_name")
    if actual_algorithm_class_name != algorithm_class_name:
        raise ValueError(
            "Stage-H policy/optimizer class mismatch: "
            f"{policy_class_name}/{actual_algorithm_class_name}"
        )
    algorithm = algorithm_class(
        policy,
        device=env.device,
        **algorithm_cfg,
    )
    algorithm.init_storage(
        "rl",
        env.num_envs,
        config["num_steps_per_env"],
        observations,
        [env.num_actions],
    )
    return policy, algorithm, config


def _runner_cfg():
    if args.policy_contract == "strict_mimickit":
        return SMPICMPureDiscoveryRunnerCfg()
    if args.policy_contract == "sugar_native_tactile_floor_lr":
        return SMPICMSugarNativeFloorLrRunnerCfg()
    if (
        args.policy_contract
        == "sugar_native_zero_preserving_tactile_floor_lr"
    ):
        return SMPICMSugarNativeZeroPreservingFloorLrRunnerCfg()
    if (
        args.policy_contract
        == "sugar_native_zero_preserving_tactile_fixed_low_lr"
    ):
        return SMPICMSugarNativeZeroPreservingFixedLowLrRunnerCfg()
    return SMPICMSugarNativeRunnerCfg()


def _restore_audited_contact_state(
    base_env, source_path: Path
) -> tuple[
    dict[str, object],
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Seed every diagnostic env from one real direct-contact replay frame."""

    with np.load(source_path, allow_pickle=False) as archive:
        source = {name: np.asarray(archive[name]) for name in archive.files}
    required = (
        "robot_root_state_w",
        "robot_joint_pos",
        "robot_joint_vel",
        "object_root_state_w",
        "normal_force",
        "shear_force",
        "motion_frame",
        "source_environment_origin_w",
        "selected_motion_id",
    )
    if args.causal_contact_bootstrap_v2:
        required += (
            "policy_actions_unclipped",
            "applied_actions_policy_units",
        )
    missing = [name for name in required if name not in source]
    if missing:
        raise KeyError(f"contact source is missing fields: {missing}")
    selected_motion_id = int(source["selected_motion_id"].reshape(-1)[0])
    if selected_motion_id != 45:
        raise ValueError("Stage-H contact diagnostic is locked to audited motion 45")
    integrated = source["normal_force"].sum(axis=(-2, -1))
    bilateral_load = np.min(integrated, axis=-1)
    selected_frame = int(np.argmax(bilateral_load))
    if bilateral_load[selected_frame] <= 0.0:
        raise RuntimeError("audited Stage-H contact source has no bilateral frame")
    selected_reference_frame = int(
        source["motion_frame"].reshape(-1)[selected_frame]
    )
    if selected_reference_frame < 0:
        raise RuntimeError("audited Stage-H reference frame is negative")
    if args.causal_contact_bootstrap_v2 and selected_frame < 3:
        raise RuntimeError(
            "causal contact bootstrap requires three preceding TacSL frames"
        )

    num_envs = base_env.num_envs
    source_origin = source["source_environment_origin_w"].astype(np.float32)
    target_origins = base_env.scene.env_origins.detach().cpu().numpy()
    translations = target_origins - source_origin[None, :]
    robot_root = np.repeat(
        source["robot_root_state_w"][selected_frame : selected_frame + 1],
        num_envs,
        axis=0,
    ).astype(np.float32, copy=True)
    object_root = np.repeat(
        source["object_root_state_w"][selected_frame : selected_frame + 1],
        num_envs,
        axis=0,
    ).astype(np.float32, copy=True)
    robot_root[:, :3] += translations
    object_root[:, :3] += translations
    joint_pos = np.repeat(
        source["robot_joint_pos"][selected_frame : selected_frame + 1],
        num_envs,
        axis=0,
    ).astype(np.float32, copy=True)
    joint_vel = np.repeat(
        source["robot_joint_vel"][selected_frame : selected_frame + 1],
        num_envs,
        axis=0,
    ).astype(np.float32, copy=True)
    env_ids = torch.arange(num_envs, device=base_env.device)
    robot = base_env.scene["robot"]
    obj = base_env.scene["obj"]
    robot.write_root_state_to_sim(
        torch.as_tensor(robot_root, device=base_env.device), env_ids=env_ids
    )
    robot.write_joint_state_to_sim(
        torch.as_tensor(joint_pos, device=base_env.device),
        torch.as_tensor(joint_vel, device=base_env.device),
        env_ids=env_ids,
    )
    obj.write_root_state_to_sim(
        torch.as_tensor(object_root, device=base_env.device), env_ids=env_ids
    )

    command = base_env.command_manager.get_term("motion")
    command.motion_id.fill_(selected_motion_id)
    command.time_steps.fill_(selected_reference_frame)
    command._use_motion_data.fill_(True)
    command._record_reference_targets(env_ids)
    current_object_position = torch.as_tensor(
        object_root[:, :3], device=base_env.device
    )
    command.initial_obj_pos_w.copy_(current_object_position)
    command.initial_obj_height_w.copy_(current_object_position[:, 2])
    command.ever_lifted.zero_()
    command.goal_stable_counter.zero_()
    command.episode_steps.zero_()
    base_env.episode_length_buf.fill_(1)
    base_env._sugar_direct_tactile_history_cache = {}
    base_env.sim.forward()
    base_env.sim.render()
    base_env.scene.update(dt=0.0)
    for sensor_name in ("left_palm_tactile", "right_palm_tactile"):
        base_env.scene[sensor_name].update(
            float(base_env.step_dt), force_recompute=True
        )
    raw_tactile = torch.cat(
        [
            torch.cat(
                (
                    base_env.scene[sensor_name].data.tactile_normal_force.reshape(
                        num_envs, -1
                    ),
                    base_env.scene[sensor_name].data.tactile_shear_force.reshape(
                        num_envs, -1
                    ),
                ),
                dim=-1,
            )
            for sensor_name in ("left_palm_tactile", "right_palm_tactile")
        ],
        dim=-1,
    )
    live_raw_nonzero_values = int(torch.count_nonzero(raw_tactile))
    live_raw_abs_max = float(raw_tactile.abs().max())

    # The restored robot/object state and this force frame come from the same
    # audited official TacSL replay sample.  A single live SDF recomputation
    # immediately after USD restoration is not a deterministic initialization
    # boundary: depending on the asynchronous SDF/scene warm-up, it can return
    # either the correct contact or an all-zero frame.  Seed only the first
    # observation from the exact recorded spatial pressure/signed-shear fields;
    # every subsequent control step remains live TacSL.
    replay_normal = source["normal_force"][selected_frame].astype(
        np.float32, copy=False
    )
    replay_shear = source["shear_force"][selected_frame].astype(
        np.float32, copy=False
    )
    if replay_normal.shape != (2, 20, 25):
        raise ValueError(
            "audited direct-TacSL normal-force shape drift: "
            f"{replay_normal.shape}"
        )
    if replay_shear.shape != (2, 20, 25, 2):
        raise ValueError(
            "audited direct-TacSL signed-shear shape drift: "
            f"{replay_shear.shape}"
        )
    if not np.isfinite(replay_normal).all() or not np.isfinite(
        replay_shear
    ).all():
        raise RuntimeError("audited direct-TacSL replay seed is non-finite")
    if np.min(replay_normal) < 0.0:
        raise RuntimeError(
            "audited direct-TacSL replay has negative normal force"
        )
    if np.min(replay_normal.sum(axis=(-2, -1))) <= 0.0:
        raise RuntimeError(
            "audited direct-TacSL replay seed is not bilateral"
        )
    for hand_index, sensor_name in enumerate(
        ("left_palm_tactile", "right_palm_tactile")
    ):
        sensor_data = base_env.scene[sensor_name].data
        normal_seed = torch.as_tensor(
            replay_normal[hand_index].reshape(1, -1),
            device=base_env.device,
        ).expand(num_envs, -1)
        shear_seed = torch.as_tensor(
            replay_shear[hand_index].reshape(1, -1, 2),
            device=base_env.device,
        ).expand(num_envs, -1, -1)
        sensor_data.tactile_normal_force.copy_(normal_seed)
        sensor_data.tactile_shear_force.copy_(shear_seed)
    seeded_raw_tactile = torch.cat(
        [
            torch.cat(
                (
                    base_env.scene[
                        sensor_name
                    ].data.tactile_normal_force.reshape(num_envs, -1),
                    base_env.scene[
                        sensor_name
                    ].data.tactile_shear_force.reshape(num_envs, -1),
                ),
                dim=-1,
            )
            for sensor_name in ("left_palm_tactile", "right_palm_tactile")
        ],
        dim=-1,
    )
    previous_action = None
    current_action = None
    source_tactile_history = None
    previous_action_exact = None
    if args.causal_contact_bootstrap_v2:
        policy_actions = np.asarray(
            source["policy_actions_unclipped"], dtype=np.float32
        )
        applied_actions = np.asarray(
            source["applied_actions_policy_units"], dtype=np.float32
        )
        if (
            policy_actions.shape != applied_actions.shape
            or policy_actions.ndim != 2
            or policy_actions.shape[1] != 29
            or selected_frame >= policy_actions.shape[0]
        ):
            raise ValueError(
                "causal contact bootstrap source action geometry drift"
            )
        action_conversion_error = np.abs(
            policy_actions - applied_actions
        )
        if (
            not np.isfinite(action_conversion_error).all()
            or float(action_conversion_error.max()) > 2.0e-6
        ):
            raise RuntimeError(
                "source policy/applied action conversion exceeds the frozen "
                "float32 round-trip tolerance"
            )
        # The official source is pre-action.  One process_action call supplies
        # two different causal views that must not be conflated:
        #
        # * official Refiner mdp.last_action is the raw actor action 102;
        # * the goal policy's custom previous-applied-action observation is
        #   the inverse-scaled physical target from that same action.
        #
        # The source records both fields.  Restore through the raw official
        # action, then verify both downstream views against their own record.
        previous_action = policy_actions[selected_frame - 1].copy()
        previous_applied_action = applied_actions[
            selected_frame - 1
        ].copy()
        current_action = policy_actions[selected_frame].copy()
        repeated_previous = torch.as_tensor(
            previous_action,
            dtype=torch.float32,
            device=base_env.device,
        ).reshape(1, -1).expand(num_envs, -1)
        base_env.action_manager.process_action(repeated_previous)
        observed_previous_actor = (
            base_env.action_manager.action.detach()
        )
        repeated_previous_applied = torch.as_tensor(
            previous_applied_action,
            dtype=torch.float32,
            device=base_env.device,
        ).reshape(1, -1).expand(num_envs, -1)
        observed_previous_applied = previous_applied_action_policy_units(
            base_env
        ).detach()
        previous_actor_action_exact = bool(
            torch.equal(observed_previous_actor, repeated_previous)
        )
        previous_applied_action_exact = bool(
            torch.equal(
                observed_previous_applied,
                repeated_previous_applied,
            )
        )
        previous_action_exact = (
            previous_actor_action_exact
            and previous_applied_action_exact
        )
        if not previous_action_exact:
            raise RuntimeError(
                "recorded previous action did not reproduce both official "
                "Refiner last_action and goal previous-applied-action views"
            )

        normal_history = np.asarray(
            source["normal_force"][selected_frame - 3 : selected_frame + 1],
            dtype=np.float32,
        )
        shear_history = np.asarray(
            source["shear_force"][selected_frame - 3 : selected_frame + 1],
            dtype=np.float32,
        )
        if (
            normal_history.shape != (4, 2, 20, 25)
            or shear_history.shape != (4, 2, 20, 25, 2)
        ):
            raise ValueError(
                "causal contact bootstrap source tactile history drift"
            )
        source_tactile_history = np.concatenate(
            (
                normal_history[:, :, None],
                shear_history.transpose(0, 1, 4, 2, 3),
            ),
            axis=2,
        )
        source_tactile_history = (
            source_tactile_history
            / float(TACTILE_RUNTIME_PARAMS["taxel_area_m2"])
            * float(TACTILE_RUNTIME_PARAMS["stress_scale"])
        ).astype(np.float32, copy=False)

    record = {
        "selected_motion_id": selected_motion_id,
        "selected_contact_source_frame": selected_frame,
        "selected_reference_frame": selected_reference_frame,
        "initial_tactile_seed_source": (
            "audited_official_direct_tacsl_replay_frame"
        ),
        "live_raw_sensor_nonzero_values_before_replay_seed": (
            live_raw_nonzero_values
        ),
        "live_raw_sensor_abs_max_before_replay_seed": live_raw_abs_max,
        "raw_sensor_nonzero_values_after_restore": int(
            torch.count_nonzero(seeded_raw_tactile)
        ),
        "raw_sensor_abs_max_after_restore": float(
            seeded_raw_tactile.abs().max()
        ),
        "causal_contact_bootstrap_v2": (
            args.causal_contact_bootstrap_v2
        ),
        "source_previous_action_frame": (
            selected_frame - 1
            if args.causal_contact_bootstrap_v2
            else None
        ),
        "source_current_action_frame": (
            selected_frame
            if args.causal_contact_bootstrap_v2
            else None
        ),
        "previous_action_reaches_observation_bitwise": (
            previous_action_exact
        ),
        "previous_actor_action_reaches_official_last_action_bitwise": (
            previous_actor_action_exact
        ),
        "previous_applied_action_reaches_goal_observation_bitwise": (
            previous_applied_action_exact
        ),
        "source_tactile_history_frames": (
            list(range(selected_frame - 3, selected_frame + 1))
            if args.causal_contact_bootstrap_v2
            else None
        ),
        "source_policy_applied_action_roundtrip": (
            {
                "global_l2_max": float(
                    np.linalg.norm(
                        policy_actions - applied_actions, axis=1
                    ).max()
                ),
                "global_max_abs": float(
                    action_conversion_error.max()
                ),
                "previous_frame_l2": float(
                    np.linalg.norm(
                        policy_actions[selected_frame - 1]
                        - applied_actions[selected_frame - 1]
                    )
                ),
                "previous_frame_max_abs": float(
                    action_conversion_error[selected_frame - 1].max()
                ),
                "current_frame_l2": float(
                    np.linalg.norm(
                        policy_actions[selected_frame]
                        - applied_actions[selected_frame]
                    )
                ),
                "current_frame_max_abs": float(
                    action_conversion_error[selected_frame].max()
                ),
                "tolerance_max_abs": 2.0e-6,
                "official_last_action_source_field": (
                    "policy_actions_unclipped"
                ),
                "goal_previous_applied_source_field": (
                    "applied_actions_policy_units"
                ),
                "current_teacher_source_field": (
                    "policy_actions_unclipped"
                ),
            }
            if args.causal_contact_bootstrap_v2
            else None
        ),
    }
    return (
        record,
        previous_action,
        current_action,
        source_tactile_history,
    )


def _restore_explicit_zero_control_state(
    base_env, source_path: Path, *, selected_frame: int = 103
) -> tuple[
    dict[str, object],
    np.ndarray,
    np.ndarray,
    None,
]:
    """Restore the official state/action boundary without reading tactile.

    The source NPZ is used only for official robot/object state, command frame,
    and native actions.  Its historical tactile arrays are neither loaded nor
    used to select the frame.  The declared source index and its immediately
    preceding action are restored. Historical controls use index 103; the
    wrong-teacher experiment starts at pre-grasp index 1.
    """

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
            raise KeyError(
                f"state/action reset source is missing fields: {missing}"
            )
        # Deliberately load only the declared state/action fields.  In
        # particular normal_force and shear_force remain unopened.
        source = {name: np.asarray(archive[name]) for name in required}
    if int(source["selected_motion_id"].reshape(-1)[0]) != 45:
        raise ValueError("explicit-zero control is locked to official motion 45")
    if str(source["native_sample_phase"].reshape(-1)[0]) != "pre_action":
        raise ValueError("explicit-zero control requires pre-action source records")
    frame_count = int(source["motion_frame"].reshape(-1).shape[0])
    if not 1 <= selected_frame < frame_count:
        raise ValueError("explicit-zero source frame is outside the action sequence")
    selected_reference_frame = int(
        source["motion_frame"].reshape(-1)[selected_frame]
    )

    policy_actions = np.asarray(
        source["policy_actions_unclipped"], dtype=np.float32
    )
    applied_actions = np.asarray(
        source["applied_actions_policy_units"], dtype=np.float32
    )
    if (
        policy_actions.shape != applied_actions.shape
        or policy_actions.ndim != 2
        or policy_actions.shape[1] != 29
        or selected_frame >= policy_actions.shape[0]
    ):
        raise ValueError("explicit-zero source action geometry drift")
    action_conversion_error = np.abs(policy_actions - applied_actions)
    if (
        not np.isfinite(action_conversion_error).all()
        or float(action_conversion_error.max()) > 2.0e-6
    ):
        raise RuntimeError(
            "source policy/applied action conversion exceeds 2e-6"
        )

    num_envs = base_env.num_envs
    source_origin = source["source_environment_origin_w"].astype(np.float32)
    target_origins = base_env.scene.env_origins.detach().cpu().numpy()
    translations = target_origins - source_origin[None, :]
    robot_root = np.repeat(
        source["robot_root_state_w"][selected_frame : selected_frame + 1],
        num_envs,
        axis=0,
    ).astype(np.float32, copy=True)
    object_root = np.repeat(
        source["object_root_state_w"][selected_frame : selected_frame + 1],
        num_envs,
        axis=0,
    ).astype(np.float32, copy=True)
    robot_root[:, :3] += translations
    object_root[:, :3] += translations
    joint_pos = np.repeat(
        source["robot_joint_pos"][selected_frame : selected_frame + 1],
        num_envs,
        axis=0,
    ).astype(np.float32, copy=True)
    joint_vel = np.repeat(
        source["robot_joint_vel"][selected_frame : selected_frame + 1],
        num_envs,
        axis=0,
    ).astype(np.float32, copy=True)
    env_ids = torch.arange(num_envs, device=base_env.device)
    base_env.scene["robot"].write_root_state_to_sim(
        torch.as_tensor(robot_root, device=base_env.device), env_ids=env_ids
    )
    base_env.scene["robot"].write_joint_state_to_sim(
        torch.as_tensor(joint_pos, device=base_env.device),
        torch.as_tensor(joint_vel, device=base_env.device),
        env_ids=env_ids,
    )
    base_env.scene["obj"].write_root_state_to_sim(
        torch.as_tensor(object_root, device=base_env.device), env_ids=env_ids
    )

    command = base_env.command_manager.get_term("motion")
    selected_motion_loader_index = (
        0 if args.wrong_teacher_motion_folder is not None else 45
    )
    if selected_motion_loader_index >= command.motion.num_motion:
        raise ValueError("explicit-zero motion loader index is out of range")
    command.motion_id.fill_(selected_motion_loader_index)
    command.time_steps.fill_(selected_reference_frame)
    command._use_motion_data.fill_(True)
    command._record_reference_targets(env_ids)
    current_object_position = torch.as_tensor(
        object_root[:, :3], device=base_env.device
    )
    command.initial_obj_pos_w.copy_(current_object_position)
    command.initial_obj_height_w.copy_(current_object_position[:, 2])
    command.ever_lifted.zero_()
    command.goal_stable_counter.zero_()
    command.episode_steps.zero_()
    base_env.episode_length_buf.fill_(1)
    base_env._sugar_direct_tactile_history_cache = {}
    base_env.sim.forward()
    base_env.scene.update(dt=0.0)

    previous_action = policy_actions[selected_frame - 1].copy()
    previous_applied_action = applied_actions[selected_frame - 1].copy()
    current_action = policy_actions[selected_frame].copy()
    repeated_previous = torch.as_tensor(
        previous_action,
        dtype=torch.float32,
        device=base_env.device,
    ).reshape(1, -1).expand(num_envs, -1)
    base_env.action_manager.process_action(repeated_previous)
    observed_previous_actor = base_env.action_manager.action.detach()
    repeated_previous_applied = torch.as_tensor(
        previous_applied_action,
        dtype=torch.float32,
        device=base_env.device,
    ).reshape(1, -1).expand(num_envs, -1)
    observed_previous_applied = previous_applied_action_policy_units(
        base_env
    ).detach()
    previous_actor_exact = bool(
        torch.equal(observed_previous_actor, repeated_previous)
    )
    previous_applied_exact = bool(
        torch.equal(observed_previous_applied, repeated_previous_applied)
    )
    if not (previous_actor_exact and previous_applied_exact):
        raise RuntimeError(
            "explicit-zero reset did not reproduce both previous-action views"
        )

    record = {
        "selected_motion_id": 45,
        "selected_motion_loader_index": selected_motion_loader_index,
        "selected_contact_source_frame": selected_frame,
        "selected_reference_frame": selected_reference_frame,
        "initial_tactile_seed_source": "none_explicit_zero_control",
        "tactile_arrays_loaded": False,
        "tactile_sensor_data_read": False,
        "live_raw_sensor_nonzero_values_before_replay_seed": 0,
        "live_raw_sensor_abs_max_before_replay_seed": 0.0,
        "raw_sensor_nonzero_values_after_restore": 0,
        "raw_sensor_abs_max_after_restore": 0.0,
        "causal_contact_bootstrap_v2": False,
        "source_previous_action_frame": selected_frame - 1,
        "source_current_action_frame": selected_frame,
        "previous_action_reaches_observation_bitwise": True,
        "previous_actor_action_reaches_official_last_action_bitwise": (
            previous_actor_exact
        ),
        "previous_applied_action_reaches_goal_observation_bitwise": (
            previous_applied_exact
        ),
        "source_tactile_history_frames": None,
        "source_policy_applied_action_roundtrip": {
            "global_max_abs": float(action_conversion_error.max()),
            "previous_frame_max_abs": float(
                action_conversion_error[selected_frame - 1].max()
            ),
            "current_frame_max_abs": float(
                action_conversion_error[selected_frame].max()
            ),
            "tolerance_max_abs": 2.0e-6,
        },
    }
    return record, previous_action, current_action, None


def _coherent_latent_dynamics_audit(
    base_env,
    expected_distribution_seed: int,
    *,
    fixed_profile: bool = False,
) -> dict[str, object]:
    """Freeze the exact training physics tuple and its startup readback."""

    term = base_env.event_manager.get_term_cfg("latent_contact_dynamics").func
    if not isinstance(term, apply_stratified_latent_contact_dynamics):
        raise TypeError("goal-recovery startup dynamics term type drift")
    values = term.tuple_for_device("cpu")
    readback = {
        name: value.detach().cpu().clone()
        for name, value in term.last_readback.items()
    }
    expected_env_ids = torch.arange(base_env.num_envs, dtype=torch.long)
    tuple_readback_exact = (
        isinstance(readback.get("env_ids"), torch.Tensor)
        and torch.equal(readback["env_ids"], expected_env_ids)
        and all(
            name in readback and torch.equal(readback[name], tensor)
            for name, tensor in values.items()
        )
    )
    mass = values["mass_scale"]
    static = values["static_friction"]
    dynamic = values["dynamic_friction"]
    com_y = values["com_y_m"]
    pulse = values["pulse_delta_velocity_w_mps"]
    policy_terms = list(base_env.observation_manager.active_terms["policy"])
    icm_terms = list(base_env.observation_manager.active_terms["icm_vector"])
    hidden_fields = (
        "mass",
        "friction",
        "com_y",
        "center_of_mass",
        "latent_contact",
    )
    checks = {
        "startup_tuple_readback_bitwise_exact": tuple_readback_exact,
        "mass_scale_range_exact": bool(
            (mass >= 0.5).all() and (mass <= 2.0).all()
        ),
        "friction_range_and_order_exact": bool(
            (static >= 0.2).all()
            and (static <= 0.8).all()
            and (dynamic >= 0.2).all()
            and (dynamic <= 0.8).all()
            and (dynamic <= static).all()
        ),
        "com_range_exact": bool(
            (com_y >= -0.04).all() and (com_y <= 0.04).all()
        ),
        "reference_pulse_disabled": bool(torch.count_nonzero(pulse) == 0),
        "physics_tuple_hidden_from_policy_and_icm": not any(
            fragment in name.lower()
            for name in policy_terms + icm_terms
            for fragment in hidden_fields
        ),
    }
    if fixed_profile:
        checks["fixed_nominal_profile_exact_across_all_envs"] = bool(
            torch.all(mass == 1.0)
            and torch.all(static == 0.6)
            and torch.all(dynamic == 0.5)
            and torch.all(com_y == 0.0)
            and torch.count_nonzero(pulse) == 0
        )
    else:
        checks["twenty_distinct_mass_dynamic_com_strata"] = (
            int(torch.unique(mass).numel()) == base_env.num_envs
            and int(torch.unique(dynamic).numel()) == base_env.num_envs
            and int(torch.unique(com_y).numel()) == base_env.num_envs
        )
    return {
        "passed": all(checks.values()),
        "profile_mode": "fixed_nominal" if fixed_profile else "stratified",
        "event_class": (
            f"{term.__class__.__module__}.{term.__class__.__qualname__}"
        ),
        "distribution_seed": expected_distribution_seed,
        "tuple_by_environment": {
            name: tensor.tolist() for name, tensor in values.items()
        },
        "last_readback_env_ids": (
            readback["env_ids"].tolist() if "env_ids" in readback else None
        ),
        "checks": checks,
    }


def main() -> None:
    if args.num_envs < 2:
        raise ValueError("Stage-H diagnostic requires at least two environments")
    if args.num_updates < 1:
        raise ValueError("Stage-H diagnostic requires at least one update")
    resume_checkpoint_path = (
        args.resume_checkpoint.expanduser().resolve()
        if args.resume_checkpoint is not None
        else None
    )
    resume_payload = None
    resume_update = 0
    if resume_checkpoint_path is not None:
        _assert_experiment_output(resume_checkpoint_path)
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(resume_checkpoint_path)
        resume_payload = torch.load(
            resume_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            resume_payload.get("protocol")
            != "sugar_stage_i_official_refiner_residual_multistep_checkpoint_v1"
        ):
            raise ValueError("unexpected continuation checkpoint protocol")
        resume_update = int(resume_payload.get("iteration", -1))
        if resume_update < 1 or args.num_updates <= resume_update:
            raise ValueError(
                "resume checkpoint iteration must precede --num-updates"
            )
    updates_executed = args.num_updates - resume_update
    h2_contract = args.tactile_regime == "h2r1_five_role"
    explicit_zero_tactile_contract = (
        args.tactile_regime == "explicit_zero_control"
    )
    residual_teacher_contract = args.nominal_teacher_checkpoint is not None
    reference_waypoint_foundation_contract = (
        args.reference_waypoint_foundation_config is not None
    )
    paper_cws_contract = args.paper_cws_runtime_config is not None
    if (
        paper_cws_contract
        != (args.paper_cws_guidance_weight > 0.0)
        or args.paper_cws_guidance_weight < 0.0
    ):
        raise ValueError(
            "paper-CWS config and a positive guidance weight must be "
            "declared together"
        )
    posture_adaptive_contract = (
        args.teacher_wrapper_mode == "posture_adaptive_v1"
    )
    wrong_reference_wrapper_contract = (
        args.teacher_wrapper_mode == "wrong_reference_anneal_v1"
    )
    wrong_reference_fixed_wrapper_contract = (
        args.teacher_wrapper_mode == "wrong_reference_fixed_v1"
    )
    if posture_adaptive_contract and not residual_teacher_contract:
        raise ValueError(
            "posture_adaptive_v1 requires the frozen official Refiner"
        )
    if args.causal_contact_bootstrap_v2 and not posture_adaptive_contract:
        raise ValueError(
            "the corrected causal contact bootstrap is admitted only for "
            "the posture-adaptive official-Refiner redo"
        )
    if reference_waypoint_foundation_contract and (
        posture_adaptive_contract or args.causal_contact_bootstrap_v2
    ):
        raise ValueError(
            "reference-waypoint W1 uses the exact two-source reset, not the "
            "withdrawn posture-adaptive/bootstrap route"
        )
    legacy_goal_recovery_contract = (
        args.training_objective == "goal_recovery_multiphysics"
    )
    native_authority_contract = (
        args.training_objective == "goal_recovery_native_authority"
    )
    demo_reward_contract = args.demo_reward_config is not None
    demo_reward_telemetry_contract = (
        args.demo_reward_telemetry_config is not None
    )
    demo_event_reward_contract = args.demo_event_reward_config is not None
    if sum(
        bool(value)
        for value in (
            demo_reward_contract,
            demo_reward_telemetry_contract,
            demo_event_reward_contract,
        )
    ) > 1:
        raise ValueError(
            "legacy demo, demo telemetry and phase-event reward are mutually exclusive"
        )
    if demo_event_reward_contract != (args.demo_event_selected_option is not None):
        raise ValueError("phase-event config and selected option must be declared together")
    demo_predictor_loaded_contract = (
        demo_reward_contract
        or demo_reward_telemetry_contract
        or demo_event_reward_contract
    )
    active_demo_reward_contract = (
        demo_reward_contract or demo_event_reward_contract
    )
    early_protocol_config = None
    if args.protocol_config is not None:
        early_protocol_path = _workspace_path(args.protocol_config)
        if early_protocol_path.is_file():
            early_protocol_config = json.loads(
                early_protocol_path.read_text(encoding="utf-8")
            )
    demo_authority_rework_contract = (
        early_protocol_config is not None
        and early_protocol_config.get("protocol")
        == "sugar_plan11_demo_conflict_authority_rework_matched_v3"
    )
    wrong_teacher_reward_conflict_contract = (
        early_protocol_config is not None
        and early_protocol_config.get("protocol")
        in {
            "sugar_plan11_wrong_teacher_reward_conflict_v1",
            "sugar_plan11_fixed_teacher_demo_identity_v2",
            "sugar_plan11_teacher_floor_overfit_v1",
            "sugar_phase_event_reward_matched_policy_v1",
        }
    )
    fixed_teacher_demo_identity_contract = (
        early_protocol_config is not None
        and early_protocol_config.get("protocol")
        in {
            "sugar_plan11_fixed_teacher_demo_identity_v2",
            "sugar_phase_event_reward_matched_policy_v1",
        }
    )
    phase_event_protocol_contract = (
        early_protocol_config is not None
        and early_protocol_config.get("protocol")
        == "sugar_phase_event_reward_matched_policy_v1"
    )
    teacher_floor_overfit_contract = (
        early_protocol_config is not None
        and early_protocol_config.get("protocol")
        == "sugar_plan11_teacher_floor_overfit_v1"
    )
    annealed_wrong_teacher_contract = (
        wrong_teacher_reward_conflict_contract
        and not fixed_teacher_demo_identity_contract
        and not teacher_floor_overfit_contract
    )
    scheduled_teacher_contract = (
        annealed_wrong_teacher_contract or teacher_floor_overfit_contract
    )
    fixed_teacher_interval_resume_contract = (
        fixed_teacher_demo_identity_contract
        and resume_payload is not None
        and updates_executed == 64
        and resume_update % 64 == 0
        and args.num_updates % 64 == 0
    )
    teacher_floor_resume_contract = (
        teacher_floor_overfit_contract
        and resume_payload is not None
        and resume_update == 64
        and updates_executed == 64
        and args.num_updates == 128
    )
    if resume_payload is not None and not (
        fixed_teacher_interval_resume_contract or teacher_floor_resume_contract
    ):
        raise ValueError(
            "checkpoint continuation must be a consecutive 64-update "
            "fixed-teacher segment or the declared update64-to-floor overfit"
        )
    if wrong_reference_wrapper_contract != scheduled_teacher_contract:
        raise ValueError(
            "scheduled wrong-reference wrapper/protocol mismatch"
        )
    if (
        wrong_reference_fixed_wrapper_contract
        != fixed_teacher_demo_identity_contract
    ):
        raise ValueError(
            "fixed wrong-reference wrapper/protocol mismatch"
        )
    if paper_cws_contract and demo_predictor_loaded_contract:
        raise ValueError(
            "paper-CWS and demo-predictor branches are separate experiments"
        )
    if paper_cws_contract and not reference_waypoint_foundation_contract:
        raise ValueError(
            "first paper-CWS runtime is admitted only on the corrected W1 "
            "relative-waypoint foundation"
        )
    goal_recovery_contract = (
        legacy_goal_recovery_contract or native_authority_contract
    )
    expected_base_residual_scale = (
        1.0
        if wrong_teacher_reward_conflict_contract
        else (0.50 if demo_authority_rework_contract else 0.05)
    )
    if residual_teacher_contract and not (
        reference_waypoint_foundation_contract
    ) and not wrong_teacher_reward_conflict_contract and (
        args.policy_contract
        != "sugar_native_zero_preserving_tactile_fixed_low_lr"
        or args.residual_scale != expected_base_residual_scale
        or args.teacher_release_mode != "linear"
        or args.teacher_linear_release_steps != 4
    ):
        raise ValueError(
            "residual-teacher multistep contract is locked to the "
            "zero-preserving fixed-low-LR policy, its hash-bound protocol "
            "residual scale, and the four-step linear release"
        )
    if reference_waypoint_foundation_contract and (
        not residual_teacher_contract
        or not native_authority_contract
        or args.policy_contract
        != "sugar_native_zero_preserving_tactile_fixed_low_lr"
        or args.tactile_regime != "nominal"
        or args.num_envs != 20
        or (
            (args.num_updates, args.checkpoint_updates)
            not in ((1, "1"), (64, "1,16,64"))
        )
        or args.residual_scale != 0.05
        or args.post_release_residual_scale is not None
        or args.teacher_release_mode != "fixed_one"
        or args.teacher_release_scope != "full_body"
        or args.support_teacher_mode != "advancing"
        or args.teacher_reference_advance_mode
        not in (
            "command_manager_only",
            "goal_teacher_post_step_once",
        )
        or args.drop_grace_steps != 0
        or args.reward_control
        not in (
            "foundation_icm_policy_weight_zero",
            "full",
        )
        or demo_predictor_loaded_contract
        or args.action_seed is not None
        or args.strict_deterministic_torch
        or args.protocol_config is not None
    ):
        raise ValueError(
            "reference-waypoint W1 is locked to its one-update wiring or "
            "64-update training endpoint, serious native direct-TacSL actor, "
            "fixed full-body live teacher, single native reference advance, "
            "scale 0.05, active original ICM learner with an explicitly "
            "matched zero/full policy weight, and no demo/RGB branch"
        )
    if (
        not reference_waypoint_foundation_contract
        and not wrong_teacher_reward_conflict_contract
        and args.teacher_reference_advance_mode != "legacy_pre_step"
    ):
        raise ValueError(
            "command-manager-only reference advance is locked to the "
            "reference-waypoint W1 foundation"
        )
    if (
        args.reward_control == "foundation_icm_policy_weight_zero"
        and not reference_waypoint_foundation_contract
    ):
        raise ValueError(
            "foundation ICM policy-weight-zero mix is locked to W1"
        )
    if residual_teacher_contract and (
        (
            native_authority_contract
            and args.post_release_residual_scale != 1.0
        )
        or (
            not native_authority_contract
            and args.post_release_residual_scale is not None
        )
    ) and not reference_waypoint_foundation_contract and not wrong_teacher_reward_conflict_contract:
        raise ValueError(
            "only goal_recovery_native_authority may set the locked "
            "post-release residual scale, and it must be exactly 1.0"
        )
    residual_long_contract = (
        residual_teacher_contract
        and args.num_updates == 64
        and args.checkpoint_updates == "1,16,64"
    )
    goal_recovery_smoke_contract = (
        legacy_goal_recovery_contract
        and args.num_updates == 8
        and args.checkpoint_updates == "1,4,8"
    )
    goal_recovery_formal_contract = (
        legacy_goal_recovery_contract
        and args.num_updates == 256
        and args.checkpoint_updates == "1,64,256"
    )
    native_authority_smoke_contract = (
        native_authority_contract
        and args.num_updates == 8
        and args.checkpoint_updates == "1,4,8"
    )
    native_authority_formal_contract = (
        native_authority_contract
        and args.num_updates == 512
        and args.checkpoint_updates == "1,128,512"
        and not posture_adaptive_contract
    )
    wrong_teacher_reward_conflict_64_contract = (
        wrong_teacher_reward_conflict_contract
        and native_authority_contract
        and residual_teacher_contract
        and active_demo_reward_contract
        and explicit_zero_tactile_contract
        and args.num_envs == 20
        and (
            (
                (
                    fixed_teacher_interval_resume_contract
                    or teacher_floor_resume_contract
                )
                and args.checkpoint_updates == str(args.num_updates)
            )
            or (
                resume_payload is None
                and args.num_updates == 64
                and args.checkpoint_updates
                == ("32,64" if phase_event_protocol_contract else "1,64")
            )
        )
        and args.policy_contract
        == "sugar_native_zero_preserving_tactile_fixed_low_lr"
        and args.residual_scale == 1.0
        and args.post_release_residual_scale is None
        and args.teacher_release_scope == "full_body"
        and args.support_teacher_mode == "advancing"
        and args.teacher_reference_advance_mode
        == "goal_teacher_post_step_once"
        and (
            (
                annealed_wrong_teacher_contract
                and args.teacher_anneal_updates == 64
                and args.teacher_final_coefficient == 0.0
                and args.teacher_release_mode == "linear"
            )
            or (
                teacher_floor_resume_contract
                and args.teacher_anneal_updates == 64
                and args.teacher_final_coefficient == 0.25
                and args.teacher_release_mode == "linear"
            )
            or (
                fixed_teacher_demo_identity_contract
                and args.teacher_anneal_updates == 0
                and args.teacher_final_coefficient == 0.0
                and args.teacher_release_mode == "fixed_one"
            )
        )
        and args.explicit_zero_source_frame == 1
        and args.wrong_teacher_motion_folder is not None
        and args.drop_grace_steps == 0
        and args.reward_control == "full"
        and args.action_seed is not None
        and args.strict_deterministic_torch
        and args.protocol_config is not None
    )
    posture_capacity_contract = (
        posture_adaptive_contract
        and native_authority_contract
        and not active_demo_reward_contract
        and args.num_envs in (20, 40, 60, 80)
        and args.num_updates == 2
        and args.checkpoint_updates == "1,2"
    )
    posture_formal_num_updates = (
        (1_000_000 + 24 * args.num_envs - 1)
        // (24 * args.num_envs)
        if args.num_envs in (20, 40, 60, 80)
        else -1
    )
    posture_formal_checkpoint_updates = (
        "1,"
        f"{(posture_formal_num_updates + 3) // 4},"
        f"{(posture_formal_num_updates + 1) // 2},"
        f"{posture_formal_num_updates}"
    )
    posture_formal_contract = (
        posture_adaptive_contract
        and native_authority_contract
        and args.num_envs in (20, 40, 60, 80)
        and args.num_updates == posture_formal_num_updates
        and args.checkpoint_updates
        == posture_formal_checkpoint_updates
    )
    if posture_adaptive_contract and not (
        posture_capacity_contract or posture_formal_contract
    ):
        raise ValueError(
            "posture_adaptive_v1 is locked to the result-blind 2-update "
            "20/40/60/80 capacity sweep or the exact >=1M-transition "
            "matched formal endpoint"
        )
    if posture_adaptive_contract and (
        args.posture_pre_failure_residual_scale != 0.05
        or args.posture_post_failure_residual_scale != 0.40
        or args.posture_post_failure_teacher_floor != 0.65
        or args.teacher_release_scope != "arm_only"
        or args.support_teacher_mode != "advancing"
        or args.drop_grace_steps != 0
        or args.reward_control != "full"
    ):
        raise ValueError(
            "posture-adaptive authority is frozen to 0.05->0.40, teacher "
            "floor 0.65, arm-only release, advancing support, no grace, "
            "and the full separately logged reward mix"
        )
    if active_demo_reward_contract and not (
        native_authority_contract
        and (
            native_authority_smoke_contract
            or native_authority_formal_contract
            or posture_formal_contract
            or wrong_teacher_reward_conflict_64_contract
        )
        and args.action_seed is not None
    ):
        raise ValueError(
            "the frozen demo-potential adapter is locked to a declared "
            "native-authority arm-only or posture-formal endpoint with an "
            "explicit post-model-load action seed"
        )
    if args.action_seed is not None and not native_authority_contract:
        raise ValueError(
            "the explicit matched action seed is locked to native authority"
        )
    if args.strict_deterministic_torch and args.action_seed is None:
        raise ValueError(
            "strict deterministic Torch is locked to a matched action seed"
        )
    if (
        (args.action_seed is not None or active_demo_reward_contract)
        and args.protocol_config is None
    ):
        raise ValueError(
            "matched action/demo experiments require --protocol-config"
        )
    goal_recovery_fixed_contract = (
        goal_recovery_smoke_contract
        or goal_recovery_formal_contract
        or native_authority_smoke_contract
        or native_authority_formal_contract
        or posture_capacity_contract
        or posture_formal_contract
        or reference_waypoint_foundation_contract
        or wrong_teacher_reward_conflict_64_contract
    )
    if goal_recovery_contract and not (
        reference_waypoint_foundation_contract
        or wrong_teacher_reward_conflict_64_contract
        or (
            goal_recovery_fixed_contract
            and residual_teacher_contract
            and (h2_contract or explicit_zero_tactile_contract)
            and (
                args.num_envs == 20
                if not posture_adaptive_contract
                else args.num_envs in (20, 40, 60, 80)
            )
            and args.teacher_release_scope == "arm_only"
            and args.support_teacher_mode == "advancing"
            and args.drop_grace_steps == 0
            and (
                args.reward_control == "full"
                or (
                    explicit_zero_tactile_contract
                    and native_authority_formal_contract
                    and args.reward_control
                    == "plan11_icm_policy_weight_zero"
                )
            )
        )
    ):
        raise ValueError(
            "goal-recovery multiphysics is locked to the official residual "
            "teacher, H2R1 or the declared Plan-11 explicit-zero control, a "
            "declared arm-only or posture-adaptive endpoint, "
            "four-step release, advancing support, no drop grace, and the "
            "full reward mix"
        )
    postfailure_exposure_contract = (
        residual_long_contract
        and args.teacher_release_scope == "full_body"
        and args.drop_grace_steps == 64
    )
    blockwise_teacher_contract = (
        residual_long_contract
        and args.teacher_release_scope == "arm_only"
        and args.support_teacher_mode == "advancing"
        and args.drop_grace_steps == 0
        and args.reward_control == "full"
    )
    failure_latched_support_contract = (
        residual_long_contract
        and args.teacher_release_scope == "arm_only"
        and args.support_teacher_mode == "failure_latched"
        and args.drop_grace_steps == 0
        and args.reward_control == "full"
    )
    supported_postdrop_exposure_contract = (
        residual_long_contract
        and args.teacher_release_scope == "arm_only"
        and args.support_teacher_mode == "advancing"
        and args.drop_grace_steps == 64
        and args.reward_control == "full"
    )
    if args.teacher_release_scope == "arm_only" and not (
        blockwise_teacher_contract
        or failure_latched_support_contract
        or supported_postdrop_exposure_contract
        or goal_recovery_fixed_contract
    ):
        raise ValueError(
            "arm-only teacher release is locked to the full-reward residual "
            "H2R1 64-update endpoint, with either no grace or the admitted "
            "64-step supported post-drop exposure"
        )
    if args.support_teacher_mode == "failure_latched" and not (
        residual_long_contract
        and args.teacher_release_scope == "arm_only"
        and args.drop_grace_steps == 0
        and args.reward_control == "full"
    ):
        raise ValueError(
            "failure-latched support is locked to arm-only, full-reward "
            "residual H2R1 64-update no-grace training"
        )
    if args.drop_grace_steps not in (0, 64):
        raise ValueError("drop grace is locked to disabled or 64 steps")
    if args.drop_grace_steps > 0 and (
        not (
            postfailure_exposure_contract
            or supported_postdrop_exposure_contract
        )
        or args.reward_control != "full"
    ):
        raise ValueError(
            "bounded drop grace is locked to the full-reward residual H2R1 "
            "64-update exposure control"
        )
    if args.reward_control != "full" and not (
        residual_long_contract
        or reference_waypoint_foundation_contract
        or (
            explicit_zero_tactile_contract
            and native_authority_formal_contract
            and args.reward_control == "plan11_icm_policy_weight_zero"
        )
    ):
        raise ValueError(
            "policy-credit controls are locked to the residual H2R1 "
            "64-update endpoint"
        )
    reward_mix_by_control = {
        "full": (
            SMPICMRewardMixCfg(
                task_outcome_weight=(
                    10.0 if native_authority_contract else 1.0
                ),
                smp_reward_weight=0.5,
                icm_reward_weight=1.0,
                external_constraint_weight=1.0,
                require_zero_outcome_rewards=False,
                require_no_success_termination=False,
            )
            if goal_recovery_contract
            else SMPICMRewardMixCfg()
        ),
        "smp_policy_weight_zero": SMPICMRewardMixCfg(
            smp_reward_weight=0.0,
            icm_reward_weight=1.0,
            external_constraint_weight=1.0,
        ),
        "icm_policy_weight_zero": SMPICMRewardMixCfg(
            smp_reward_weight=0.5,
            icm_reward_weight=0.0,
            external_constraint_weight=1.0,
        ),
        "foundation_icm_policy_weight_zero": SMPICMRewardMixCfg(
            task_outcome_weight=10.0,
            smp_reward_weight=0.5,
            icm_reward_weight=0.0,
            external_constraint_weight=1.0,
            require_zero_outcome_rewards=False,
            require_no_success_termination=False,
        ),
        "plan11_icm_policy_weight_zero": SMPICMRewardMixCfg(
            task_outcome_weight=10.0,
            smp_reward_weight=0.5,
            icm_reward_weight=0.0,
            external_constraint_weight=1.0,
            require_zero_outcome_rewards=False,
            require_no_success_termination=False,
        ),
    }
    reward_mix_cfg = reward_mix_by_control[args.reward_control]
    h2_fixed_endpoint = (
        (
            args.num_updates == 8
            and args.checkpoint_updates == "1,4,8"
        )
        or residual_long_contract
        or goal_recovery_formal_contract
        or native_authority_formal_contract
        or posture_capacity_contract
        or posture_formal_contract
    )
    if h2_contract and (
        args.policy_contract
        != "sugar_native_zero_preserving_tactile_fixed_low_lr"
        or (
            args.num_envs != 20
            if not posture_adaptive_contract
            else args.num_envs not in (20, 40, 60, 80)
        )
        or not h2_fixed_endpoint
    ):
        raise ValueError(
            "H2R1 is locked to the zero-preserving fixed-low-LR contract, "
            "an admitted fixed endpoint, and either the legacy 20-env route "
            "or a frozen posture capacity/formal environment count"
        )
    if explicit_zero_tactile_contract and (
        not (
            native_authority_formal_contract
            or wrong_teacher_reward_conflict_64_contract
        )
        or args.policy_contract
        != "sugar_native_zero_preserving_tactile_fixed_low_lr"
        or args.num_envs != 20
        or (
            not wrong_teacher_reward_conflict_64_contract
            and (
                args.num_updates != 512
                or args.checkpoint_updates != "1,128,512"
            )
        )
        or not residual_teacher_contract
        or (
            not wrong_teacher_reward_conflict_64_contract
            and args.teacher_release_scope != "arm_only"
        )
        or args.support_teacher_mode != "advancing"
        or args.drop_grace_steps != 0
        or args.reward_control
        not in ("full", "plan11_icm_policy_weight_zero")
        or args.action_seed is None
        or not args.strict_deterministic_torch
        or args.protocol_config is None
    ):
        raise ValueError(
            "Plan-11 explicit-zero tactile is locked to the matched 20-env, "
            "512-update serious native-authority protocol; the frozen demo "
            "predictor is required only by the four-demo experiment"
        )
    try:
        checkpoint_updates = {
            int(value)
            for value in args.checkpoint_updates.split(",")
            if value.strip()
        }
    except ValueError as error:
        raise ValueError("--checkpoint-updates must be comma-separated integers") from error
    if (
        not checkpoint_updates
        or args.num_updates not in checkpoint_updates
        or min(checkpoint_updates) < 1
        or max(checkpoint_updates) > args.num_updates
    ):
        raise ValueError(
            "checkpoint updates must be within the run and include the final update"
        )
    motion_folder = args.motion_folder.expanduser().resolve()
    prior_dir = args.prior_dir.expanduser().resolve()
    contact_source = args.contact_source.expanduser().resolve()
    legacy_demo_reward_config_path = (
        (
            args.demo_reward_config
            if demo_reward_contract
            else args.demo_reward_telemetry_config
        )
        .expanduser()
        .resolve()
        if (demo_reward_contract or demo_reward_telemetry_contract)
        else None
    )
    demo_event_reward_config_path = (
        args.demo_event_reward_config.expanduser().resolve()
        if demo_event_reward_contract
        else None
    )
    demo_predictor_config_path = (
        demo_event_reward_config_path
        if demo_event_reward_contract
        else legacy_demo_reward_config_path
    )
    protocol_config_path = (
        args.protocol_config.expanduser().resolve()
        if args.protocol_config is not None
        else None
    )
    reference_waypoint_foundation_config_path = (
        args.reference_waypoint_foundation_config.expanduser().resolve()
        if reference_waypoint_foundation_contract
        else None
    )
    paper_cws_runtime_config_path = (
        args.paper_cws_runtime_config.expanduser().resolve()
        if paper_cws_contract
        else None
    )
    demo_reward_config = (
        _load_demo_reward_config(legacy_demo_reward_config_path)
        if legacy_demo_reward_config_path is not None
        else None
    )
    demo_event_reward_config = (
        json.loads(demo_event_reward_config_path.read_text(encoding="utf-8"))
        if demo_event_reward_config_path is not None
        and demo_event_reward_config_path.is_file()
        else None
    )
    teacher_checkpoint = (
        args.nominal_teacher_checkpoint.expanduser().resolve()
        if residual_teacher_contract
        else None
    )
    output = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_paths = {
        update: _checkpoint_path(checkpoint, update, args.num_updates)
        for update in checkpoint_updates
    }
    _assert_experiment_output(output)
    for path in checkpoint_paths.values():
        _assert_experiment_output(path)
    existing = [path for path in (output, *checkpoint_paths.values()) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")
    if not motion_folder.is_dir():
        raise FileNotFoundError(motion_folder)
    for required in ("model.pt", "result.json", "diffusion_config.yaml"):
        if not (prior_dir / required).is_file():
            raise FileNotFoundError(prior_dir / required)
    if not contact_source.is_file():
        raise FileNotFoundError(contact_source)
    if (
        demo_predictor_config_path is not None
        and not demo_predictor_config_path.is_file()
    ):
        raise FileNotFoundError(demo_predictor_config_path)
    if (
        protocol_config_path is not None
        and not protocol_config_path.is_file()
    ):
        raise FileNotFoundError(protocol_config_path)
    protocol_config = (
        json.loads(protocol_config_path.read_text(encoding="utf-8"))
        if protocol_config_path is not None
        else None
    )
    if (
        reference_waypoint_foundation_config_path is not None
        and not reference_waypoint_foundation_config_path.is_file()
    ):
        raise FileNotFoundError(
            reference_waypoint_foundation_config_path
        )
    reference_waypoint_foundation_config = (
        _load_reference_waypoint_foundation_config(
            reference_waypoint_foundation_config_path
        )
        if reference_waypoint_foundation_config_path is not None
        else None
    )
    paper_cws_runtime_payload, paper_cws_runtime_cfg = (
        _load_paper_cws_runtime_config(paper_cws_runtime_config_path)
        if paper_cws_runtime_config_path is not None
        else (None, None)
    )
    if (
        reference_waypoint_foundation_config is not None
        and "_relative_"
        in reference_waypoint_foundation_config["protocol"]
    ):
        declared_reward = reference_waypoint_foundation_config["reward"]
        declared_cws = declared_reward["paper_cws_runtime_config"]
        if (
            reference_waypoint_foundation_config["teacher"][
                "reference_advance_mode"
            ]
            != args.teacher_reference_advance_mode
            or
            float(declared_reward["paper_cws_guidance_weight"])
            != float(args.paper_cws_guidance_weight)
            or float(declared_reward["icm_policy_weight"])
            != float(reward_mix_cfg.icm_reward_weight)
            or (
                paper_cws_contract
                and (
                    declared_cws is None
                    or _workspace_path(declared_cws["path"])
                    != paper_cws_runtime_config_path
                    or _sha256(paper_cws_runtime_config_path)
                    != declared_cws["sha256"]
                    or paper_cws_runtime_config_path.stat().st_size
                    != int(declared_cws["size_bytes"])
                )
            )
            or (not paper_cws_contract and declared_cws is not None)
        ):
            raise ValueError(
                "reference-waypoint reward/CWS runtime binding drift"
            )
    if reference_waypoint_foundation_config is not None and (
        int(reference_waypoint_foundation_config["num_envs"])
        != args.num_envs
        or int(reference_waypoint_foundation_config["num_updates"])
        != args.num_updates
        or reference_waypoint_foundation_config["checkpoint_updates"]
        != sorted(checkpoint_updates)
        or int(reference_waypoint_foundation_config["seed"]) != args.seed
        or _workspace_path(
            reference_waypoint_foundation_config["output"]
        )
        != output
        or _workspace_path(
            reference_waypoint_foundation_config["checkpoint"]
        )
        != checkpoint
    ):
        raise ValueError(
            "reference-waypoint W1 runtime/config binding drift"
        )
    if reference_waypoint_foundation_config is not None and (
        contact_source
        != _workspace_path(
            reference_waypoint_foundation_config["sources"][0]["path"]
        )
    ):
        raise ValueError(
            "W1 compatibility --contact-source must equal source zero"
        )
    observed_tacsl_mount_environment = {
        name: os.environ.get(name)
        for name in EXPECTED_TACSL_MOUNT_ENVIRONMENT
    }
    if protocol_config is not None:
        protocol_name = protocol_config.get("protocol")
        plan11_demo_conflict_protocol = (
            protocol_name
            in {
                "sugar_plan11_demo_conflict_matched_v1",
                "sugar_plan11_demo_conflict_zero_tactile_matched_v2",
                "sugar_plan11_demo_conflict_authority_rework_matched_v3",
                "sugar_plan11_wrong_teacher_reward_conflict_v1",
                "sugar_plan11_fixed_teacher_demo_identity_v2",
                "sugar_plan11_teacher_floor_overfit_v1",
                "sugar_phase_event_reward_matched_policy_v1",
            }
        )
        plan11_icm_policy_credit_protocol = (
            protocol_name
            == "sugar_plan11_original_icm_policy_credit_zero_tactile_v1"
        )
        plan11_protocol_arm = (
            plan11_demo_conflict_protocol
            or plan11_icm_policy_credit_protocol
        )
        plan11_zero_tactile_protocol = (
            protocol_name
            in {
                "sugar_plan11_demo_conflict_zero_tactile_matched_v2",
                "sugar_plan11_demo_conflict_authority_rework_matched_v3",
                "sugar_plan11_wrong_teacher_reward_conflict_v1",
                "sugar_plan11_fixed_teacher_demo_identity_v2",
                "sugar_plan11_teacher_floor_overfit_v1",
                "sugar_plan11_original_icm_policy_credit_zero_tactile_v1",
                "sugar_phase_event_reward_matched_policy_v1",
            }
        )
        if plan11_protocol_arm != (args.protocol_arm is not None):
            raise ValueError(
                "Plan-11 protocol and --protocol-arm must be declared together"
            )
        arm_name = (
            args.protocol_arm
            if plan11_protocol_arm
            else ("demo_eta2" if demo_reward_contract else "no_demo")
        )
        arm = protocol_config["arms"][arm_name]
        shared = protocol_config["shared_runtime"]
        deterministic_protocol = protocol_name in {
            "sugar_demo_reward_matched_policy_diagnostic_8update_deterministic_v2",
            "sugar_demo_reward_matched_policy_formal_512_seed92781_deterministic_v2",
            "sugar_posture_adaptive_capacity_20env_seed95781_v1",
            "sugar_posture_adaptive_capacity_40env_seed95781_v1",
            "sugar_posture_adaptive_capacity_60env_seed95781_v1",
            "sugar_posture_adaptive_capacity_80env_seed95781_v1",
            "sugar_posture_adaptive_matched_policy_formal_seed96781_v1",
            "sugar_posture_adaptive_causal_bootstrap_capacity_seed105781_v2",
            "sugar_posture_adaptive_causal_bootstrap_matched_formal_seed106781_v2",
            "sugar_plan11_demo_conflict_matched_v1",
            "sugar_plan11_demo_conflict_zero_tactile_matched_v2",
            "sugar_plan11_demo_conflict_authority_rework_matched_v3",
            "sugar_plan11_wrong_teacher_reward_conflict_v1",
            "sugar_plan11_fixed_teacher_demo_identity_v2",
            "sugar_plan11_teacher_floor_overfit_v1",
            "sugar_plan11_original_icm_policy_credit_zero_tactile_v1",
            "sugar_phase_event_reward_matched_policy_v1",
        }
        mount_contract_exact = (
            (
                shared.get("tactile_mount_environment") is None
                and all(
                    value is None
                    for value in observed_tacsl_mount_environment.values()
                )
            )
            if plan11_zero_tactile_protocol
            else (
                shared.get("tactile_mount_environment")
                == EXPECTED_TACSL_MOUNT_ENVIRONMENT
                and observed_tacsl_mount_environment
                == EXPECTED_TACSL_MOUNT_ENVIRONMENT
                and Path(
                    EXPECTED_TACSL_MOUNT_ENVIRONMENT[
                        "CURIOSITY_TACSL_R15_USD"
                    ]
                ).is_file()
            )
        )
        if not (
            protocol_name in {
                "sugar_demo_reward_matched_policy_diagnostic_8update_v1",
                "sugar_demo_reward_matched_policy_formal_512_seed92771_v1",
                "sugar_demo_reward_matched_policy_diagnostic_8update_deterministic_v2",
                "sugar_demo_reward_matched_policy_formal_512_seed92781_deterministic_v2",
                "sugar_posture_adaptive_capacity_20env_seed95781_v1",
                "sugar_posture_adaptive_capacity_40env_seed95781_v1",
                "sugar_posture_adaptive_capacity_60env_seed95781_v1",
                "sugar_posture_adaptive_capacity_80env_seed95781_v1",
                "sugar_posture_adaptive_matched_policy_formal_seed96781_v1",
                "sugar_posture_adaptive_causal_bootstrap_capacity_seed105781_v2",
                "sugar_posture_adaptive_causal_bootstrap_matched_formal_seed106781_v2",
                "sugar_plan11_demo_conflict_matched_v1",
                "sugar_plan11_demo_conflict_zero_tactile_matched_v2",
                "sugar_plan11_demo_conflict_authority_rework_matched_v3",
                "sugar_plan11_wrong_teacher_reward_conflict_v1",
                "sugar_plan11_fixed_teacher_demo_identity_v2",
                "sugar_plan11_teacher_floor_overfit_v1",
                "sugar_plan11_original_icm_policy_credit_zero_tactile_v1",
                "sugar_phase_event_reward_matched_policy_v1",
            }
            and int(shared["sim_and_policy_seed"]) == args.seed
            and int(shared["action_seed"]) == args.action_seed
            and int(shared["num_envs"]) == args.num_envs
            and int(shared["num_updates"]) == args.num_updates
            and int(shared.get("resume_update", 0)) == resume_update
            and (
                (
                    _workspace_path(shared["resume_checkpoint"]["path"])
                    == resume_checkpoint_path
                )
                if resume_checkpoint_path is not None
                else shared.get("resume_checkpoint") is None
            )
            and list(shared["checkpoint_updates"])
            == sorted(checkpoint_updates)
            and _workspace_path(arm["output"]) == output
            and _workspace_path(arm["checkpoint"]) == checkpoint
            and bool(arm["demo_reward_enabled"])
            is active_demo_reward_contract
            and bool(
                arm.get(
                    "demo_predictor_telemetry_loaded",
                    demo_reward_contract,
                )
            )
            is demo_predictor_loaded_contract
            and (
                not plan11_icm_policy_credit_protocol
                or (
                    arm["reward_control"] == args.reward_control
                    and float(arm["icm_policy_weight"])
                    == float(reward_mix_cfg.icm_reward_weight)
                    and demo_predictor_loaded_contract is False
                )
            )
            and (
                _workspace_path(arm["demo_runtime_config"])
                == demo_predictor_config_path
                if demo_predictor_loaded_contract
                else arm.get("demo_runtime_config") is None
            )
            and (
                not phase_event_protocol_contract
                or (
                    arm.get("demo_reward_kind")
                    == "phase_aware_dense_event"
                    and arm.get("selected_option")
                    == args.demo_event_selected_option
                    and int(shared.get("demo_event_phase_horizon_steps", -1))
                    == args.demo_event_phase_horizon_steps
                )
            )
            and _workspace_path(
                protocol_config["artifacts"]["runner_source"]["path"]
            )
            == Path(__file__).resolve()
            and bool(shared.get("strict_deterministic_torch", False))
            is args.strict_deterministic_torch
            and bool(
                shared.get("causal_contact_bootstrap_v2", False)
            )
            is args.causal_contact_bootstrap_v2
            and deterministic_protocol
            is args.strict_deterministic_torch
            and shared.get("teacher_wrapper_mode", "arm_only_v1")
            == args.teacher_wrapper_mode
            and (
                _workspace_path(
                    arm.get(
                        "teacher_motion_folder",
                        shared.get("wrong_teacher_motion_folder"),
                    )
                )
                if arm.get(
                    "teacher_motion_folder",
                    shared.get("wrong_teacher_motion_folder"),
                )
                is not None
                else None
            )
            == (
                args.wrong_teacher_motion_folder.expanduser().resolve()
                if args.wrong_teacher_motion_folder is not None
                else None
            )
            and int(shared.get("teacher_anneal_updates", 0))
            == args.teacher_anneal_updates
            and float(shared.get("teacher_final_coefficient", 0.0))
            == args.teacher_final_coefficient
            and (
                not teacher_floor_overfit_contract
                or shared.get("fixed_physics_profile")
                == TEACHER_FLOOR_FIXED_PHYSICS_PROFILE
            )
            and int(shared.get("explicit_zero_source_frame", 103))
            == args.explicit_zero_source_frame
            and float(shared.get("residual_scale", 0.05))
            == float(args.residual_scale)
            and shared.get("tactile_regime") == args.tactile_regime
            and float(
                shared.get(
                    "posture_pre_failure_residual_scale",
                    0.05,
                )
            )
            == args.posture_pre_failure_residual_scale
            and float(
                shared.get(
                    "posture_post_failure_residual_scale",
                    0.40,
                )
            )
            == args.posture_post_failure_residual_scale
            and float(
                shared.get(
                    "posture_post_failure_teacher_floor",
                    0.65,
                )
            )
            == args.posture_post_failure_teacher_floor
            and mount_contract_exact
            and plan11_zero_tactile_protocol
            is explicit_zero_tactile_contract
        ):
            raise ValueError("matched demo-policy protocol/config drift")
        if args.strict_deterministic_torch and (
            shared.get("cublas_workspace_config") != ":4096:8"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError(
                "strict deterministic protocol requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )
    elif explicit_zero_tactile_contract:
        raise ValueError(
            "explicit-zero tactile is admitted only by the hash-bound "
            "Plan-11 zero-tactile demo-conflict protocol"
        )
    if demo_reward_config is not None and (
        demo_reward_config.get("protocol")
        != "sugar_demo_reward_runtime_frozen_scale_v1"
        or float(demo_reward_config["potential"]["eta"])
        != (
            10.0
            if (
                demo_authority_rework_contract
                or wrong_teacher_reward_conflict_contract
            )
            else 2.0
        )
        or float(demo_reward_config["potential"]["gamma"]) != 0.99
    ):
        raise ValueError("unexpected frozen demo-potential runtime config")
    if demo_event_reward_contract:
        if (
            demo_event_reward_config is None
            or demo_event_reward_config.get("protocol")
            != "sugar_dense_demo_event_feedback_runtime_v1"
            or demo_event_reward_config.get("potential_difference_shaping_used")
            is not False
            or demo_event_reward_config.get("future_actual_events_enter_runtime")
            is not False
            or args.demo_event_selected_option
            not in demo_event_reward_config.get("selected_demo_options", {})
        ):
            raise ValueError("unexpected phase-aware demo-event runtime config")
        for source_name in ("dataset_root", "predictor_dir"):
            if not Path(demo_event_reward_config[source_name]).is_dir():
                raise FileNotFoundError(demo_event_reward_config[source_name])
    if teacher_checkpoint is not None and not teacher_checkpoint.is_file():
        raise FileNotFoundError(teacher_checkpoint)
    if args.admission_only:
        if not (
            phase_event_protocol_contract
            and demo_event_reward_contract
            and wrong_teacher_reward_conflict_64_contract
        ):
            raise ValueError(
                "admission-only is scoped to the phase-event matched policy protocol"
            )
        scorer = FrozenPhaseAwareDemoEventScorer(
            num_envs=args.num_envs,
            device=args.device,
            cfg=FrozenPhaseAwareDemoEventScorerCfg(
                runtime_config_path=str(demo_event_reward_config_path),
                selected_option=args.demo_event_selected_option,
                phase_horizon_steps=args.demo_event_phase_horizon_steps,
            ),
        )
        audit = scorer.frozen_model_audit()
        payload = {
            "protocol": "sugar_phase_event_policy_admission_only_v1",
            "passed": bool(
                audit["model_frozen"]
                and audit["policy_dim"] == 121
                and audit["alignment_mode"] == "clock_phase"
                and audit["future_actual_events_used"] is False
            ),
            "selected_option": args.demo_event_selected_option,
            "policy_updates_executed": 0,
            "environment_created": False,
            "frozen_model_audit": audit,
        }
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if not payload["passed"]:
            raise RuntimeError("phase-event admission-only probe failed")
        return

    if goal_recovery_contract:
        cfg = GoalCoherentLatentRobotEnvCfg()
        # A drop is a recoverable state in this branch: set-down, regrasp, and
        # bottom support must remain observable.  Success, unsafe fall,
        # workspace, and timeout remain active.
        cfg.terminations.dropped_after_lift = None
        latent_distribution_seed = (
            52017 if native_authority_contract else 42017
        )
        cfg.events.latent_contact_dynamics.params[
            "distribution_seed"
        ] = latent_distribution_seed
        if teacher_floor_overfit_contract:
            fixed = TEACHER_FLOOR_FIXED_PHYSICS_PROFILE
            event = cfg.events.latent_contact_dynamics
            event.params["mass_scale_range"] = (
                fixed["mass_scale"], fixed["mass_scale"]
            )
            event.params["static_friction_range"] = (
                fixed["static_friction"], fixed["static_friction"]
            )
            event.params["dynamic_friction_range"] = (
                fixed["dynamic_friction"], fixed["dynamic_friction"]
            )
            event.params["com_y_range_m"] = (
                fixed["com_y_m"], fixed["com_y_m"]
            )
            event.params["pulse_magnitude_range_mps"] = (0.0, 0.0)
        if reference_waypoint_foundation_contract:
            foundation_event = cfg.events.latent_contact_dynamics
            foundation_event.params["mass_scale_range"] = (1.0, 3.0)
            foundation_event.params["static_friction_range"] = (0.6, 0.6)
            foundation_event.params["dynamic_friction_range"] = (0.5, 0.5)
            foundation_event.params["com_y_range_m"] = (0.0, 0.0)
            foundation_event.params["pulse_magnitude_range_mps"] = (
                0.0,
                0.0,
            )
            material = cfg.events.robot_physics_material
            material.params["static_friction_range"] = (0.6, 0.6)
            material.params["dynamic_friction_range"] = (0.5, 0.5)
            material.params["restitution_range"] = (0.0, 0.0)
        task_id = GOAL_RECOVERY_TASK_ID
    else:
        cfg = PureDiscoveryRobotEnvCfg()
        task_id = PURE_DISCOVERY_TASK_ID
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.commands.motion.motion_folder = str(motion_folder)
    cfg.commands.motion.teacher_motion_folder = (
        str(args.wrong_teacher_motion_folder.expanduser().resolve())
        if args.wrong_teacher_motion_folder is not None
        else None
    )
    cfg.commands.motion.use_generator = False
    cfg.commands.motion.generator_checkpoint_path = None
    # Contact-rich official SUGAR reset for this integration diagnostic only.
    cfg.commands.motion.start_init_env_ratio = 0.0
    cfg.commands.motion.init_with_ref = True
    if explicit_zero_tactile_contract:
        # Preserve every observation width while making the control causally
        # independent of the rejected historical TacSL installation.  The
        # manager never calls a tactile sensor or a proxy contact function in
        # this branch; tactile-derived reward terms remain present but return
        # exact zero so the non-tactile task/safety ledger is unchanged.
        cfg.observations.tactile_history.force_history.func = (
            explicit_zero_tactile_force_history
        )
        for group in (
            cfg.observations.policy,
            cfg.observations.critic,
        ):
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

    gym_env = None
    reference_waypoint_foundation_reset = None
    try:
        gym_env = gym.make(task_id, cfg=cfg)
        # Environment construction can also initialize libraries with their
        # own global backend defaults. Reassert the immutable pair contract
        # before policy/model construction and every training update.
        _enforce_strict_torch_determinism()
        if fixed_teacher_demo_identity_contract:
            env = WrongReferenceFixedOfficialRefinerResidualVecEnvWrapper(
                gym_env,
                teacher_checkpoint,
                residual_scale=args.residual_scale,
                clip_actions=None,
            )
        elif scheduled_teacher_contract:
            env = WrongReferenceScheduledOfficialRefinerResidualVecEnvWrapper(
                gym_env,
                teacher_checkpoint,
                residual_scale=args.residual_scale,
                teacher_anneal_control_steps=(
                    args.teacher_anneal_updates * 24
                ),
                teacher_final_coefficient=(
                    args.teacher_final_coefficient
                ),
                clip_actions=None,
            )
        elif posture_adaptive_contract:
            env = PostureAdaptiveOfficialRefinerResidualVecEnvWrapper(
                gym_env,
                teacher_checkpoint,
                residual_scale=args.residual_scale,
                release_mode=args.teacher_release_mode,
                linear_release_steps=args.teacher_linear_release_steps,
                post_release_residual_scale=(
                    args.post_release_residual_scale
                ),
                posture_pre_failure_residual_scale=(
                    args.posture_pre_failure_residual_scale
                ),
                posture_post_failure_residual_scale=(
                    args.posture_post_failure_residual_scale
                ),
                posture_post_failure_teacher_floor=(
                    args.posture_post_failure_teacher_floor
                ),
                drop_grace_steps=args.drop_grace_steps,
                clip_actions=None,
            )
        elif residual_teacher_contract:
            env = OfficialRefinerResidualVecEnvWrapper(
                gym_env,
                teacher_checkpoint,
                residual_scale=args.residual_scale,
                release_mode=args.teacher_release_mode,
                linear_release_steps=args.teacher_linear_release_steps,
                teacher_release_scope=args.teacher_release_scope,
                support_teacher_mode=args.support_teacher_mode,
                drop_grace_steps=args.drop_grace_steps,
                post_release_residual_scale=(
                    args.post_release_residual_scale
                ),
                teacher_reference_advance_mode=(
                    args.teacher_reference_advance_mode
                ),
                clip_actions=None,
            )
        else:
            env = RslRlVecEnvWrapper(gym_env, clip_actions=None)
        base_env = env.unwrapped
        if reference_waypoint_foundation_contract:
            foundation_sources = tuple(
                ReferenceWaypointSource.from_mapping(
                    source, WORKSPACE_ROOT
                )
                for source in reference_waypoint_foundation_config["sources"]
            )
            reference_waypoint_foundation_reset = (
                ReferenceWaypointFoundationReset(
                    base_env, foundation_sources
                )
            )
            all_env_ids = torch.arange(
                base_env.num_envs,
                dtype=torch.long,
                device=base_env.device,
            )
            reference_waypoint_foundation_reset.restore(all_env_ids)
            reference_waypoint_foundation_reset.install()
            restored_raw_tactile = torch.cat(
                [
                    torch.cat(
                        (
                            base_env.scene[
                                sensor_name
                            ].data.tactile_normal_force.reshape(
                                base_env.num_envs, -1
                            ),
                            base_env.scene[
                                sensor_name
                            ].data.tactile_shear_force.reshape(
                                base_env.num_envs, -1
                            ),
                        ),
                        dim=-1,
                    )
                    for sensor_name in (
                        "left_palm_tactile",
                        "right_palm_tactile",
                    )
                ],
                dim=-1,
            )
            contact_seed = {
                **reference_waypoint_foundation_reset.audit_state(),
                "raw_sensor_nonzero_values_after_restore": int(
                    torch.count_nonzero(restored_raw_tactile)
                ),
                "raw_sensor_abs_max_after_restore": float(
                    restored_raw_tactile.abs().max()
                ),
            }
            source_previous_action = None
            source_current_action = None
            source_tactile_history = None
        else:
            (
                contact_seed,
                source_previous_action,
                source_current_action,
                source_tactile_history,
            ) = (
                _restore_explicit_zero_control_state(
                    base_env,
                    contact_source,
                    selected_frame=args.explicit_zero_source_frame,
                )
                if explicit_zero_tactile_contract
                else _restore_audited_contact_state(
                    base_env, contact_source
                )
            )
        latent_dynamics_proof = (
            _coherent_latent_dynamics_audit(
                base_env,
                latent_distribution_seed,
                fixed_profile=teacher_floor_overfit_contract,
            )
            if goal_recovery_contract
            else None
        )
        tactile_stress_runtime = (
            configure_h2_direct_tactile_stress(
                base_env,
                taxel_area_m2=TACTILE_RUNTIME_PARAMS["taxel_area_m2"],
                stress_scale=TACTILE_RUNTIME_PARAMS["stress_scale"],
            )
            if h2_contract
            else None
        )
        role_names_by_env = (
            tactile_stress_runtime.role_names_by_env
            if tactile_stress_runtime is not None
            else None
        )
        causal_bootstrap_tactile_expected = None
        causal_bootstrap_tactile_max_abs = None
        causal_bootstrap_nominal_source_max_abs = None
        if args.causal_contact_bootstrap_v2:
            if (
                source_previous_action is None
                or source_current_action is None
                or source_tactile_history is None
            ):
                raise RuntimeError(
                    "causal contact bootstrap omitted source fields"
                )
            repeated_history = torch.as_tensor(
                source_tactile_history,
                dtype=torch.float32,
                device=base_env.device,
            ).unsqueeze(0).expand(args.num_envs, -1, -1, -1, -1, -1)
            causal_bootstrap_tactile_expected = (
                tactile_stress_runtime.bootstrap_history(
                    repeated_history,
                    current_step=int(base_env.common_step_counter),
                )
                if tactile_stress_runtime is not None
                else repeated_history.clone()
            )
            direct_tactile_force_history(
                base_env,
                left_sensor_name="left_palm_tactile",
                right_sensor_name="right_palm_tactile",
                history_steps=4,
                grid_shape=(20, 25),
                taxel_area_m2=TACTILE_RUNTIME_PARAMS["taxel_area_m2"],
                stress_scale=TACTILE_RUNTIME_PARAMS["stress_scale"],
            )
            history_cache = base_env._sugar_direct_tactile_history_cache
            if len(history_cache) != 1:
                raise RuntimeError(
                    "causal contact bootstrap found unexpected tactile cache"
                )
            history_entry = next(iter(history_cache.values()))
            if tuple(history_entry["history"].shape) != (
                args.num_envs,
                4,
                2,
                3,
                20,
                25,
            ):
                raise RuntimeError(
                    "causal contact bootstrap cache geometry drift"
                )
            history_entry["history"].copy_(
                causal_bootstrap_tactile_expected
            )
        observations = env.get_observations()
        explicit_zero_teacher_action_l2 = None
        explicit_zero_teacher_action_max_abs = None
        explicit_zero_teacher_action_canonical_max_abs = None
        explicit_zero_teacher_action_max_abs_by_env = None
        explicit_zero_teacher_observation_shape = None
        if explicit_zero_tactile_contract:
            if source_current_action is None or not residual_teacher_contract:
                raise RuntimeError(
                    "explicit-zero control omitted source action or official teacher"
                )
            with torch.inference_mode():
                teacher_observation, teacher_action = env.teacher.action()
            expected_teacher_action = torch.as_tensor(
                source_current_action,
                dtype=teacher_action.dtype,
                device=teacher_action.device,
            ).reshape(1, -1).expand_as(teacher_action)
            teacher_action_error = teacher_action - expected_teacher_action
            explicit_zero_teacher_action_l2 = float(
                torch.linalg.vector_norm(teacher_action_error, dim=-1).max()
            )
            explicit_zero_teacher_action_max_abs = float(
                teacher_action_error.abs().max()
            )
            explicit_zero_teacher_action_max_abs_by_env = (
                teacher_action_error.abs().amax(dim=-1).detach().cpu().tolist()
            )
            explicit_zero_teacher_action_canonical_max_abs = float(
                teacher_action_error[0].abs().max()
            )
            explicit_zero_teacher_observation_shape = list(
                teacher_observation.shape
            )
            unrelated_teacher_arm = (
                annealed_wrong_teacher_contract
                or (
                    fixed_teacher_demo_identity_contract
                    and args.protocol_arm
                    == "wrong_teacher_unrelated_reward"
                )
            )
            if (
                unrelated_teacher_arm
                and explicit_zero_teacher_action_canonical_max_abs <= 1.0e-3
            ):
                raise RuntimeError(
                    "declared wrong teacher is not behaviorally distinct from "
                    "the CarryBox source action at the pre-grasp boundary"
                )
            if (
                not unrelated_teacher_arm
                and explicit_zero_teacher_action_canonical_max_abs > 2.0e-6
            ):
                raise RuntimeError(
                    "explicit-zero causal reset does not reproduce official "
                    "source action within 2e-6: "
                    "canonical_max_abs="
                    f"{explicit_zero_teacher_action_canonical_max_abs:.9g}, "
                    f"all_env_max_abs={explicit_zero_teacher_action_max_abs:.9g}, "
                    f"l2={explicit_zero_teacher_action_l2:.9g}"
                )
        causal_bootstrap_teacher_action = None
        causal_bootstrap_teacher_action_l2 = None
        causal_bootstrap_teacher_action_max_abs = None
        causal_bootstrap_teacher_action_error_by_env = None
        causal_bootstrap_teacher_observation_spread_max_abs = None
        causal_bootstrap_teacher_action_spread_max_abs = None
        causal_bootstrap_teacher_observation_reference = None
        if args.causal_contact_bootstrap_v2:
            observed_history = observations["tactile_history"].reshape(
                args.num_envs, 4, 2, 3, 20, 25
            )
            causal_bootstrap_tactile_max_abs = float(
                torch.abs(
                    observed_history - causal_bootstrap_tactile_expected
                ).max()
            )
            if tactile_stress_runtime is not None:
                nominal_mask = torch.tensor(
                    [
                        role == "nominal"
                        for role in role_names_by_env
                    ],
                    dtype=torch.bool,
                    device=base_env.device,
                )
                causal_bootstrap_nominal_source_max_abs = float(
                    torch.abs(
                        observed_history[nominal_mask]
                        - repeated_history[nominal_mask]
                    ).max()
                )
            else:
                causal_bootstrap_nominal_source_max_abs = float(
                    torch.abs(
                        observed_history - repeated_history
                    ).max()
                )
            (
                causal_bootstrap_teacher_observation,
                causal_bootstrap_teacher_action,
            ) = env.teacher.action()
            expected_teacher_action = torch.as_tensor(
                source_current_action,
                dtype=torch.float32,
                device=base_env.device,
            ).reshape(1, -1).expand(args.num_envs, -1)
            action_error = (
                causal_bootstrap_teacher_action
                - expected_teacher_action
            )
            causal_bootstrap_teacher_action_l2 = float(
                torch.linalg.vector_norm(action_error, dim=-1).max()
            )
            causal_bootstrap_teacher_action_max_abs = float(
                action_error.abs().max()
            )
            action_error_l2_by_env = torch.linalg.vector_norm(
                action_error, dim=-1
            )
            action_error_max_abs_by_env = action_error.abs().amax(dim=-1)
            causal_bootstrap_teacher_action_error_by_env = {
                "l2": action_error_l2_by_env.detach().cpu().tolist(),
                "max_abs": (
                    action_error_max_abs_by_env.detach().cpu().tolist()
                ),
                "maximum_l2_environment": int(
                    torch.argmax(action_error_l2_by_env)
                ),
                "maximum_abs_environment": int(
                    torch.argmax(action_error_max_abs_by_env)
                ),
                "environment_zero_l2": float(
                    action_error_l2_by_env[0]
                ),
                "environment_zero_max_abs": float(
                    action_error_max_abs_by_env[0]
                ),
            }
            causal_bootstrap_teacher_observation_spread_max_abs = float(
                (
                    causal_bootstrap_teacher_observation
                    - causal_bootstrap_teacher_observation[0:1]
                )
                .abs()
                .max()
            )
            causal_bootstrap_teacher_action_spread_max_abs = float(
                (
                    causal_bootstrap_teacher_action
                    - causal_bootstrap_teacher_action[0:1]
                )
                .abs()
                .max()
            )
            reference_trace_path = WORKSPACE_ROOT / (
                "experiments/sugar_smp_exploration/stage_i/"
                "live_official_refiner_teacher_seed4263_v2/"
                "LIVE_OFFICIAL_REFINER_TEACHER_TRACE.npz"
            )
            with np.load(reference_trace_path, allow_pickle=False) as archive:
                reference_observation_np = np.asarray(
                    archive["official_observation"][0], dtype=np.float32
                )
            reference_observation = torch.as_tensor(
                reference_observation_np,
                dtype=torch.float32,
                device=base_env.device,
            ).reshape(1, -1)
            reference_difference = (
                causal_bootstrap_teacher_observation
                - reference_observation
            ).abs()
            term_names = env.teacher.observation_manager.active_terms[
                "policy"
            ]
            term_dims = env.teacher.observation_manager.group_obs_term_dim[
                "policy"
            ]
            offset = 0
            term_differences = {}
            for term_name, term_shape in zip(
                term_names, term_dims, strict=True
            ):
                width = int(math.prod(term_shape))
                term_difference = reference_difference[
                    :, offset : offset + width
                ]
                term_differences[term_name] = {
                    "environment_zero_max_abs": float(
                        term_difference[0].max()
                    ),
                    "all_environment_max_abs": float(
                        term_difference.max()
                    ),
                }
                offset += width
            if offset != 890:
                raise RuntimeError(
                    "official observation term widths do not sum to 890"
                )
            with torch.inference_mode():
                reference_action = env.teacher.actor.act_inference(
                    {"policy": reference_observation}
                )
            reference_action_error = (
                reference_action
                - expected_teacher_action[0:1]
            )
            causal_bootstrap_teacher_observation_reference = {
                "path": str(reference_trace_path),
                "sha256": _sha256(reference_trace_path),
                "environment_zero_max_abs": float(
                    reference_difference[0].max()
                ),
                "all_environment_max_abs": float(
                    reference_difference.max()
                ),
                "term_max_abs": term_differences,
                "reference_action_vs_source_l2": float(
                    torch.linalg.vector_norm(reference_action_error)
                ),
                "reference_action_vs_source_max_abs": float(
                    reference_action_error.abs().max()
                ),
            }
        initial_tactile = observations["tactile_history"]
        initial_tactile_nonzero_values = int(
            torch.count_nonzero(initial_tactile)
        )
        initial_tactile_abs_max = float(initial_tactile.abs().max())
        initial_tactile_nonzero_by_role = _tactile_nonzero_by_role(
            initial_tactile,
            role_names_by_env,
        )
        posture_joint_names: list[str] = []
        posture_robot_joint_ids: torch.Tensor | None = None
        posture_joint_position_initial: torch.Tensor | None = None
        posture_joint_position_min: torch.Tensor | None = None
        posture_joint_position_max: torch.Tensor | None = None
        if posture_adaptive_contract:
            action_term = base_env.action_manager.get_term(
                "JointPositionAction"
            )
            action_joint_names = list(action_term._joint_names)
            if action_joint_names != list(env.teacher_joint_names):
                raise RuntimeError(
                    "posture motion audit action/teacher joint order drift"
                )
            robot_joint_count = int(
                base_env.scene["robot"].data.joint_pos.shape[1]
            )
            raw_joint_ids = action_term._joint_ids
            if isinstance(raw_joint_ids, slice):
                ordered_robot_joint_ids = torch.arange(
                    robot_joint_count,
                    dtype=torch.long,
                    device=base_env.device,
                )[raw_joint_ids]
            else:
                ordered_robot_joint_ids = torch.as_tensor(
                    raw_joint_ids,
                    dtype=torch.long,
                    device=base_env.device,
                )
            if ordered_robot_joint_ids.shape != (29,):
                raise RuntimeError(
                    "posture motion audit requires 29 ordered robot joints"
                )
            posture_action_columns = torch.as_tensor(
                env.teacher_posture_indices,
                dtype=torch.long,
                device=base_env.device,
            )
            posture_robot_joint_ids = ordered_robot_joint_ids[
                posture_action_columns
            ]
            posture_joint_names = [
                env.teacher_joint_names[index]
                for index in env.teacher_posture_indices
            ]
            posture_joint_position_initial = (
                base_env.scene["robot"]
                .data.joint_pos[:, posture_robot_joint_ids]
                .detach()
                .clone()
            )
            posture_joint_position_min = (
                posture_joint_position_initial.amin(dim=0)
            )
            posture_joint_position_max = (
                posture_joint_position_initial.amax(dim=0)
            )
        # Freeze fresh policy initialization independently of simulator RNG
        # consumption. HN0/HN1 explicitly start policy and Adam from scratch.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        runner_cfg = _runner_cfg()
        policy, algorithm, runner_dict = _construct_policy_algorithm(
            observations, env, runner_cfg
        )
        residual_zero_initialization = (
            policy.initialize_residual_mean_exact_zero()
            if residual_teacher_contract
            else None
        )
        initial_action_std = (
            policy.std.detach().clone()
            if hasattr(policy, "std")
            else torch.exp(policy.log_std.detach().clone())
        )
        initial_zero_tactile_causal_audit = (
            policy.zero_tactile_causal_audit()
            if hasattr(policy, "zero_tactile_causal_audit")
            else None
        )
        base_integrator = SMPICMRolloutIntegrator(
            base_env,
            prior_dir=str(prior_dir),
            mix_cfg=reward_mix_cfg,
        )
        paper_cws_scorer = (
            OfficialTacSLPaperCWSReward(
                base_env,
                paper_cws_runtime_cfg,
            )
            if paper_cws_contract
            else None
        )
        demo_scorer = (
            FrozenDemoRewardScorer(
                num_envs=base_env.num_envs,
                device=base_env.device,
                cfg=FrozenDemoRewardRuntimeCfg(
                    config_path=str(legacy_demo_reward_config_path),
                    gamma=float(
                        demo_reward_config["potential"]["gamma"]
                    ),
                    eta=float(demo_reward_config["potential"]["eta"]),
                    failure_closed_policy_index=int(
                        demo_reward_config["potential"][
                            "failure_closed_policy_index"
                        ]
                    ),
                ),
            )
            if (demo_reward_contract or demo_reward_telemetry_contract)
            else None
        )
        demo_event_scorer = (
            FrozenPhaseAwareDemoEventScorer(
                num_envs=base_env.num_envs,
                device=base_env.device,
                cfg=FrozenPhaseAwareDemoEventScorerCfg(
                    runtime_config_path=str(demo_event_reward_config_path),
                    selected_option=args.demo_event_selected_option,
                    phase_horizon_steps=args.demo_event_phase_horizon_steps,
                ),
            )
            if demo_event_reward_contract
            else None
        )
        if demo_event_reward_contract:
            extract_goal_policy_core(
                observations["policy"],
                list(base_env.observation_manager.active_terms["policy"]),
            )
        integrator = (
            PaperCWSAugmentedSMPICMRolloutIntegrator(
                base=base_integrator,
                paper_cws=paper_cws_scorer,
                guidance_weight=args.paper_cws_guidance_weight,
            )
            if paper_cws_contract
            else (
                DemoEventRewardAugmentedSMPICMRolloutIntegrator(
                    base=base_integrator,
                    demo=demo_event_scorer,
                )
                if demo_event_reward_contract
                else (
                    DemoRewardAugmentedSMPICMRolloutIntegrator(
                        base=base_integrator,
                        demo=demo_scorer,
                    )
                    if demo_reward_contract
                    else base_integrator
                )
            )
        )
        initial_begin = (
            integrator.begin(observations)
            if active_demo_reward_contract
            else integrator.begin()
        )
        initial_smp_window = (
            initial_begin["smp_window"]
            if active_demo_reward_contract
            else initial_begin
        )
        resume_rng_mode = "fresh_action_seed"
        resume_temporal_boundary = None
        resume_restore_record = None
        resume_wrapper_transition_valid = None
        if resume_payload is not None:
            if int(resume_payload["iteration"]) != resume_update:
                raise RuntimeError("resume iteration changed after admission")
            policy.load_state_dict(
                resume_payload["policy_state_dict"], strict=True
            )
            algorithm.load_checkpoint_state_dict(
                resume_payload["policy_optimizer_state_dict"]
            )
            # The saved learner state is complete, but the old checkpoint did
            # not archive a PhysX scene snapshot.  Continue at a declared fresh
            # episode boundary: retain SMP/ICM model, optimizer, normalizer and
            # accounting state while keeping the just-created live SMP window
            # and demo prefix aligned with the freshly reset environment.
            resume_integration = dict(
                resume_payload["integration_state_dict"]
            )
            if active_demo_reward_contract:
                fresh_demo_state = (
                    demo_event_scorer.state_dict()
                    if demo_event_reward_contract
                    else demo_scorer.state_dict()
                )
                resume_demo_state = dict(resume_integration["demo"])
                for name in (
                    "policy_prefix",
                    "prefix_valid_mask",
                    "current_component_mse",
                    "imitation_active",
                ):
                    resume_demo_state[name] = fresh_demo_state[name]
                resume_integration["demo"] = resume_demo_state
            integrator.load_state_dict(resume_integration)
            if residual_teacher_contract:
                source_wrapper_state = resume_payload[
                    "residual_wrapper_state_dict"
                ]
                if teacher_floor_resume_contract:
                    resume_wrapper_transition_valid = bool(
                        source_wrapper_state.get("protocol")
                        == "sugar_wrong_reference_fixed_official_refiner_v1"
                        and source_wrapper_state.get(
                            "teacher_authority_contract"
                        )
                        == "fixed_one"
                        and torch.all(
                            source_wrapper_state["teacher_coefficient"]
                            == 1.0
                        )
                        and not bool(
                            source_wrapper_state["release_latched"].any()
                        )
                        and not bool(
                            source_wrapper_state["release_progress"].any()
                        )
                        and env.release.global_control_steps == 0
                        and torch.all(env.release.coefficient == 1.0)
                        and env.release.final_coefficient == 0.25
                    )
                    if not resume_wrapper_transition_valid:
                        raise RuntimeError(
                            "fixed-one to teacher-floor wrapper transition drift"
                        )
                else:
                    env.load_checkpoint_state_dict(source_wrapper_state)
            if (
                algorithm.completed_updates != resume_update
                or base_integrator.rollouts_completed != resume_update
                or base_integrator.icm_trainer.optimizer_updates
                != resume_update
                or base_integrator.smp_scorer.normalizer_updates
                != resume_update
            ):
                raise RuntimeError(
                    "resume learner/accounting state does not match iteration"
                )
            resume_restore_record = {
                "checkpoint_path": str(resume_checkpoint_path),
                "checkpoint_sha256": _sha256(resume_checkpoint_path),
                "iteration": resume_update,
                "policy_state_exact": _state_tree_sha256(
                    policy.state_dict()
                )
                == _state_tree_sha256(
                    resume_payload["policy_state_dict"]
                ),
                "optimizer_state_exact": _state_tree_sha256(
                    algorithm.checkpoint_state_dict()
                )
                == _state_tree_sha256(
                    resume_payload["policy_optimizer_state_dict"]
                ),
                "hybrid_integration_state_exact": _state_tree_sha256(
                    integrator.state_dict()
                )
                == _state_tree_sha256(resume_integration),
                "residual_wrapper_state_exact": (
                    _state_tree_sha256(env.checkpoint_state_dict())
                    == _state_tree_sha256(
                        resume_payload["residual_wrapper_state_dict"]
                    )
                    if residual_teacher_contract
                    and not teacher_floor_resume_contract
                    else (None if teacher_floor_resume_contract else True)
                ),
                "residual_wrapper_transition_valid": (
                    resume_wrapper_transition_valid
                    if teacher_floor_resume_contract
                    else None
                ),
            }
            resume_temporal_boundary = {
                "mode": "fresh_episode_boundary_without_physx_snapshot",
                "policy_optimizer_icm_smp_restored_exact": True,
                "wrapper_boundary": (
                    "fixed_one_to_declared_nonzero_floor_schedule"
                    if teacher_floor_resume_contract
                    else "exact_wrapper_restore"
                ),
                "smp_window_reinitialized_from_live_reset": True,
                "demo_prefix_reinitialized_from_live_reset": bool(
                    active_demo_reward_contract
                ),
            }
            initial_action_std = (
                policy.std.detach().clone()
                if hasattr(policy, "std")
                else torch.exp(policy.log_std.detach().clone())
            )
            initial_zero_tactile_causal_audit = (
                policy.zero_tactile_causal_audit()
                if hasattr(policy, "zero_tactile_causal_audit")
                else None
            )
        if args.action_seed is not None:
            if (
                resume_payload is not None
                and "torch_cpu_rng_state" in resume_payload
                and "torch_cuda_rng_state_all" in resume_payload
            ):
                torch.set_rng_state(resume_payload["torch_cpu_rng_state"])
                torch.cuda.set_rng_state_all(
                    resume_payload["torch_cuda_rng_state_all"]
                )
                resume_rng_mode = "restored_checkpoint_rng_state"
            else:
                segment_action_seed = args.action_seed + resume_update
                torch.manual_seed(segment_action_seed)
                torch.cuda.manual_seed_all(segment_action_seed)
                resume_rng_mode = (
                    "derived_segment_boundary_seed"
                    if resume_payload is not None
                    else "fresh_action_seed"
                )
        segment_initial_learning_rate = float(algorithm.learning_rate)

        prior_hash_before = _tensor_state_sha256(
            base_integrator.smp_scorer.prior.state_dict()
        )
        teacher_hash_before = (
            _tensor_state_sha256(env.teacher.actor.state_dict())
            if residual_teacher_contract
            else None
        )
        policy_state_before = {
            name: tensor.detach().clone()
            for name, tensor in policy.state_dict().items()
        }
        actor_state_before = {
            name: tensor.detach().clone()
            for name, tensor in policy.actor.state_dict().items()
        }
        critic_state_before = {
            name: tensor.detach().clone()
            for name, tensor in policy.critic.state_dict().items()
        }

        ledgers: list[dict[str, float]] = []
        icm_means: list[float] = []
        smp_means: list[float] = []
        policy_base_reward_means: list[float] = []
        policy_reward_means: list[float] = []
        task_outcome_means: list[float] = []
        external_means: list[float] = []
        raw_sds_means: list[float] = []
        demo_reward_means: list[float] = []
        demo_unit_reward_means: list[float] = []
        demo_event_potential_means: list[float] = []
        demo_event_risk_means: list[float] = []
        demo_event_uncertainty_means: list[float] = []
        demo_event_ready_fractions: list[float] = []
        demo_event_phase_means: list[float] = []
        paper_cws_reward_means: list[float] = []
        paper_cws_weighted_reward_means: list[float] = []
        paper_cws_missed_contact_counts: list[int] = []
        paper_cws_unintended_contact_counts: list[int] = []
        paper_cws_clamped_reference_counts: list[int] = []
        paper_cws_out_of_support_reward_abs_max = 0.0
        paper_cws_invalid_transition_count = 0
        paper_cws_invalid_transition_reward_abs_max = 0.0
        reward_reconstruction_max_abs = 0.0
        demo_icm_bitwise_unchanged = True
        paper_cws_icm_bitwise_unchanged = True
        valid_transition_count = 0
        bootstrap_count = 0
        tactile_nonzero_values = initial_tactile_nonzero_values
        tactile_abs_max = initial_tactile_abs_max
        reload_probe: ICMTransitionBatch | None = None
        policy_update_metrics_by_update: list[dict[str, object]] = []
        integration_metrics_by_update: list[dict[str, object]] = []
        valid_transition_count_by_update: list[int] = []
        tactile_nonzero_values_by_update: list[int] = []
        tactile_nonzero_by_role_by_update: list[dict[str, int]] = []
        checkpoint_records: dict[int, dict[str, object]] = {}
        residual_action_storage_exact = True
        teacher_coefficients: list[float] = []
        support_teacher_coefficients: list[float] = []
        manipulation_teacher_coefficients: list[float] = []
        posture_teacher_coefficients: list[float] = []
        balance_teacher_coefficients: list[float] = []
        support_residual_scales: list[float] = []
        manipulation_residual_scales: list[float] = []
        posture_residual_scales: list[float] = []
        balance_residual_scales: list[float] = []
        native_authority_scale_routing_exact = True
        per_joint_teacher_routing_exact = True
        support_hold_routing_exact = True
        support_hold_trigger_capture_exact = True
        support_hold_reset_rearm_exact = True
        support_hold_arm_teacher_unchanged_exact = True
        support_hold_trigger_count = 0
        support_hold_applied_env_steps = 0
        teacher_release_events = 0
        teacher_zero_control_steps = 0
        action_sha256_by_update: list[str] = []

        for update_index in range(resume_update + 1, args.num_updates + 1):
            update_valid_transition_count = 0
            update_tactile_nonzero_values = 0
            update_tactile_nonzero_by_role = (
                {role: 0 for role in H2_TACTILE_STRESS_ROLES}
                if h2_contract
                else None
            )
            update_actions: list[torch.Tensor] = []
            with torch.inference_mode():
                for step_index in range(runner_dict["num_steps_per_env"]):
                    observation_t = observations.clone()
                    actions = algorithm.act(observation_t)
                    update_actions.append(actions.detach().cpu().clone())
                    observations_tp1, external_reward, dones, extras = env.step(
                        actions
                    )
                    if posture_adaptive_contract:
                        posture_joint_position = (
                            base_env.scene["robot"]
                            .data.joint_pos[:, posture_robot_joint_ids]
                            .detach()
                        )
                        posture_joint_position_min = torch.minimum(
                            posture_joint_position_min,
                            posture_joint_position.amin(dim=0),
                        )
                        posture_joint_position_max = torch.maximum(
                            posture_joint_position_max,
                            posture_joint_position.amax(dim=0),
                        )
                    if residual_teacher_contract:
                        runtime_step = env.latest_step
                        if runtime_step is None:
                            raise RuntimeError(
                                "residual wrapper omitted runtime record"
                            )
                        residual_action_storage_exact &= bool(
                            torch.equal(
                                actions, runtime_step.residual_action
                            )
                            and torch.equal(
                                algorithm.transition.actions,
                                runtime_step.residual_action,
                            )
                        )
                        applied_coefficient = (
                            runtime_step.teacher_coefficient.detach()
                        )
                        if args.teacher_release_scope == "arm_only":
                            support = applied_coefficient[
                                :, env.teacher_support_indices
                            ]
                            manipulation = applied_coefficient[
                                :, env.teacher_manipulation_indices
                            ]
                            scalar = manipulation[:, 0]
                            if posture_adaptive_contract:
                                posture = applied_coefficient[
                                    :, env.teacher_posture_indices
                                ]
                                balance = applied_coefficient[
                                    :, env.teacher_balance_indices
                                ]
                                expected_posture = (
                                    args.posture_post_failure_teacher_floor
                                    + (
                                        1.0
                                        - args.posture_post_failure_teacher_floor
                                    )
                                    * scalar
                                )
                                per_joint_teacher_routing_exact &= bool(
                                    torch.equal(
                                        manipulation,
                                        scalar[:, None].expand_as(
                                            manipulation
                                        ),
                                    )
                                    and torch.equal(
                                        posture,
                                        expected_posture[:, None].expand_as(
                                            posture
                                        ),
                                    )
                                    and torch.equal(
                                        balance,
                                        torch.ones_like(balance),
                                    )
                                )
                            else:
                                per_joint_teacher_routing_exact &= bool(
                                    torch.equal(
                                        support,
                                        torch.ones_like(support),
                                    )
                                    and torch.equal(
                                        manipulation,
                                        scalar[:, None].expand_as(
                                            manipulation
                                        ),
                                    )
                                )
                            support_indices = torch.as_tensor(
                                env.teacher_support_indices,
                                dtype=torch.long,
                                device=applied_coefficient.device,
                            )
                            manipulation_indices = torch.as_tensor(
                                env.teacher_manipulation_indices,
                                dtype=torch.long,
                                device=applied_coefficient.device,
                            )
                            effective_support_action = (
                                runtime_step.teacher_action[
                                    :, support_indices
                                ]
                            )
                            advancing_support_action = (
                                runtime_step.advancing_teacher_action[
                                    :, support_indices
                                ]
                            )
                            expected_support_action = (
                                advancing_support_action.clone()
                            )
                            expected_support_action[
                                runtime_step.support_hold_valid
                            ] = runtime_step.support_hold_action[
                                runtime_step.support_hold_valid
                            ]
                            support_hold_routing_exact &= bool(
                                torch.equal(
                                    effective_support_action,
                                    expected_support_action,
                                )
                            )
                            support_hold_arm_teacher_unchanged_exact &= bool(
                                torch.equal(
                                    runtime_step.teacher_action[
                                        :, manipulation_indices
                                    ],
                                    runtime_step.advancing_teacher_action[
                                        :, manipulation_indices
                                    ],
                                )
                            )
                            trigger = runtime_step.support_hold_trigger
                            support_hold_trigger_count += int(trigger.sum())
                            support_hold_applied_env_steps += int(
                                runtime_step.support_hold_valid.sum()
                            )
                            if bool(trigger.any()):
                                support_hold_trigger_capture_exact &= bool(
                                    torch.equal(
                                        runtime_step.next_support_hold_action[
                                            trigger
                                        ],
                                        advancing_support_action[trigger],
                                    )
                                    and bool(
                                        runtime_step.next_support_hold_valid[
                                            trigger
                                        ].all()
                                    )
                                )
                            reset = runtime_step.reset_mask
                            if bool(reset.any()):
                                support_hold_reset_rearm_exact &= bool(
                                    not bool(
                                        runtime_step.next_support_hold_valid[
                                            reset
                                        ].any()
                                    )
                                    and bool(
                                        torch.equal(
                                            runtime_step.next_support_hold_action[
                                                reset
                                            ],
                                            torch.zeros_like(
                                                runtime_step.next_support_hold_action[
                                                    reset
                                                ]
                                            ),
                                        )
                                    )
                                    and bool(
                                        (
                                            runtime_step.support_hold_trigger_control_step[
                                                reset
                                            ]
                                            == -1
                                        ).all()
                                    )
                                )
                            support_teacher_coefficients.extend(
                                support.cpu().reshape(-1).tolist()
                            )
                            manipulation_teacher_coefficients.extend(
                                manipulation.cpu().reshape(-1).tolist()
                            )
                            if posture_adaptive_contract:
                                posture_teacher_coefficients.extend(
                                    posture.cpu().reshape(-1).tolist()
                                )
                                balance_teacher_coefficients.extend(
                                    balance.cpu().reshape(-1).tolist()
                                )
                            applied_residual_scale = (
                                runtime_step.residual_scale.detach()
                            )
                            support_scale = applied_residual_scale[
                                :, env.teacher_support_indices
                            ]
                            manipulation_scale = applied_residual_scale[
                                :, env.teacher_manipulation_indices
                            ]
                            support_residual_scales.extend(
                                support_scale.cpu().reshape(-1).tolist()
                            )
                            manipulation_residual_scales.extend(
                                manipulation_scale.cpu().reshape(-1).tolist()
                            )
                            if posture_adaptive_contract:
                                posture_scale = applied_residual_scale[
                                    :, env.teacher_posture_indices
                                ]
                                balance_scale = applied_residual_scale[
                                    :, env.teacher_balance_indices
                                ]
                                posture_residual_scales.extend(
                                    posture_scale.cpu().reshape(-1).tolist()
                                )
                                balance_residual_scales.extend(
                                    balance_scale.cpu().reshape(-1).tolist()
                                )
                            if reference_waypoint_foundation_contract:
                                native_authority_scale_routing_exact &= bool(
                                    torch.equal(
                                        applied_residual_scale,
                                        torch.full_like(
                                            applied_residual_scale,
                                            args.residual_scale,
                                        ),
                                    )
                                )
                            elif native_authority_contract:
                                expected_manipulation_scale = (
                                    args.residual_scale
                                    + (1.0 - manipulation)
                                    * (
                                        args.post_release_residual_scale
                                        - args.residual_scale
                                    )
                                )
                                if posture_adaptive_contract:
                                    expected_posture_scale = (
                                        args.posture_pre_failure_residual_scale
                                        + (1.0 - scalar)
                                        * (
                                            args.posture_post_failure_residual_scale
                                            - args.posture_pre_failure_residual_scale
                                        )
                                    )
                                    native_authority_scale_routing_exact &= bool(
                                        torch.equal(
                                            manipulation_scale,
                                            expected_manipulation_scale,
                                        )
                                        and torch.equal(
                                            posture_scale,
                                            expected_posture_scale[
                                                :, None
                                            ].expand_as(posture_scale),
                                        )
                                        and torch.equal(
                                            balance_scale,
                                            torch.full_like(
                                                balance_scale,
                                                args.residual_scale,
                                            ),
                                        )
                                    )
                                else:
                                    native_authority_scale_routing_exact &= bool(
                                        torch.equal(
                                            support_scale,
                                            torch.full_like(
                                                support_scale,
                                                args.residual_scale,
                                            ),
                                        )
                                        and torch.equal(
                                            manipulation_scale,
                                            expected_manipulation_scale,
                                        )
                                    )
                        else:
                            scalar = applied_coefficient.reshape(-1)
                            if reference_waypoint_foundation_contract:
                                applied_residual_scale = (
                                    runtime_step.residual_scale.detach()
                                )
                                support_scale = applied_residual_scale[
                                    :, env.teacher_support_indices
                                ]
                                manipulation_scale = applied_residual_scale[
                                    :, env.teacher_manipulation_indices
                                ]
                                support_residual_scales.extend(
                                    support_scale.cpu().reshape(-1).tolist()
                                )
                                manipulation_residual_scales.extend(
                                    manipulation_scale.cpu().reshape(-1).tolist()
                                )
                                native_authority_scale_routing_exact &= bool(
                                    torch.equal(
                                        applied_residual_scale,
                                        torch.full_like(
                                            applied_residual_scale,
                                            args.residual_scale,
                                        ),
                                    )
                                )
                        teacher_coefficients.extend(
                            scalar.cpu().reshape(-1).tolist()
                        )
                        teacher_release_events += int(
                            runtime_step.failure_closed.sum()
                        )
                        teacher_zero_control_steps += int(
                            (scalar == 0.0).sum()
                        )
                    applied_action = (
                        previous_applied_action_policy_units(base_env)
                        .detach()
                        .clone()
                    )
                    signals = integrator.process_step(
                        observation_t=observation_t,
                        applied_action_policy_units_t=applied_action,
                        observation_tp1=observations_tp1,
                        external_reward=external_reward,
                        dones=dones,
                    )
                    if paper_cws_contract:
                        if integrator.last_base_signals is None:
                            raise RuntimeError(
                                "paper-CWS adapter omitted unchanged base "
                                "signals"
                            )
                        paper_cws_icm_bitwise_unchanged &= bool(
                            torch.equal(
                                signals.icm_discovery_reward,
                                integrator.last_base_signals.icm_discovery_reward,
                            )
                        )
                    elif active_demo_reward_contract:
                        if integrator.last_base_signals is None:
                            raise RuntimeError(
                                "demo adapter omitted unchanged base signals"
                            )
                        demo_icm_bitwise_unchanged &= bool(
                            torch.equal(
                                signals.icm_discovery_reward,
                                integrator.last_base_signals.icm_discovery_reward,
                            )
                        )
                    reconstructed = (
                        base_integrator.mix_cfg.task_outcome_weight
                        * signals.task_outcome_reward
                        + base_integrator.mix_cfg.external_constraint_weight
                        * signals.external_constraint_reward
                        + base_integrator.mix_cfg.smp_reward_weight
                        * signals.smp_reward
                        + base_integrator.mix_cfg.icm_reward_weight
                        * signals.icm_discovery_reward
                        + (
                            signals.demo_reward
                            if active_demo_reward_contract
                            else torch.zeros_like(signals.policy_reward)
                        )
                        + (
                            signals.paper_cws_weighted_reward
                            if paper_cws_contract
                            else torch.zeros_like(signals.policy_reward)
                        )
                    )
                    reward_reconstruction_max_abs = max(
                        reward_reconstruction_max_abs,
                        float(
                            torch.abs(
                                reconstructed - signals.policy_reward
                            ).max()
                        ),
                    )
                    algorithm.process_env_step(
                        observations_tp1,
                        signals.policy_reward,
                        dones,
                        extras,
                    )
                    ledgers.append(
                        {
                            name: float(value.mean())
                            for name, value in signals.reward_terms.items()
                        }
                    )
                    icm_means.append(
                        float(signals.icm_discovery_reward.mean())
                    )
                    smp_means.append(float(signals.smp_reward.mean()))
                    policy_base_reward_means.append(
                        float(
                            (
                                integrator.last_base_signals.policy_reward
                                if (
                                    active_demo_reward_contract
                                    or paper_cws_contract
                                )
                                else signals.policy_reward
                            ).mean()
                        )
                    )
                    policy_reward_means.append(
                        float(signals.policy_reward.mean())
                    )
                    task_outcome_means.append(
                        float(signals.task_outcome_reward.mean())
                    )
                    external_means.append(
                        float(signals.external_constraint_reward.mean())
                    )
                    raw_sds_means.append(
                        float(signals.smp_raw_sds_mean.mean())
                    )
                    if active_demo_reward_contract:
                        demo_reward_means.append(
                            float(signals.demo_reward.mean())
                        )
                        demo_unit_reward_means.append(
                            float(signals.demo_unit_eta_reward.mean())
                        )
                    if demo_event_reward_contract:
                        demo_event_potential_means.append(
                            float(signals.demo_event_potential.mean())
                        )
                        demo_event_risk_means.append(
                            float(signals.demo_event_risk.mean())
                        )
                        demo_event_uncertainty_means.append(
                            float(signals.demo_event_uncertainty.mean())
                        )
                        demo_event_ready_fractions.append(
                            float(signals.demo_event_ready.float().mean())
                        )
                        demo_event_phase_means.append(
                            float(signals.demo_event_phase.mean())
                        )
                    if paper_cws_contract:
                        paper_cws_reward_means.append(
                            float(signals.paper_cws_reward.mean())
                        )
                        paper_cws_weighted_reward_means.append(
                            float(
                                signals.paper_cws_weighted_reward.mean()
                            )
                        )
                        paper_cws_missed_contact_counts.append(
                            int(signals.paper_cws_missed_contact.sum())
                        )
                        paper_cws_unintended_contact_counts.append(
                            int(
                                signals.paper_cws_unintended_contact.sum()
                            )
                        )
                        paper_cws_clamped_reference_counts.append(
                            int(
                                signals.paper_cws_reference_index_clamped.sum()
                            )
                        )
                        if bool(
                            signals.paper_cws_reference_index_clamped.any()
                        ):
                            outside = (
                                signals.paper_cws_reference_index_clamped
                            )
                            paper_cws_out_of_support_reward_abs_max = max(
                                paper_cws_out_of_support_reward_abs_max,
                                float(
                                    signals.paper_cws_reward[outside]
                                    .abs()
                                    .max()
                                ),
                                float(
                                    signals.paper_cws_weighted_reward[
                                        outside
                                    ]
                                    .abs()
                                    .max()
                                ),
                            )
                        invalid_cws = ~signals.transition_valid
                        paper_cws_invalid_transition_count += int(
                            invalid_cws.sum()
                        )
                        if bool(invalid_cws.any()):
                            paper_cws_invalid_transition_reward_abs_max = max(
                                paper_cws_invalid_transition_reward_abs_max,
                                float(
                                    signals.paper_cws_reward[
                                        invalid_cws
                                    ].abs().max()
                                ),
                                float(
                                    signals.paper_cws_weighted_reward[
                                        invalid_cws
                                    ].abs().max()
                                ),
                            )
                    valid_count = int(signals.transition_valid.sum())
                    valid_transition_count += valid_count
                    update_valid_transition_count += valid_count
                    bootstrap_count += int(signals.icm_normalizer_bootstrap)
                    tactile = observations_tp1["tactile_history"]
                    tactile_count = int(torch.count_nonzero(tactile))
                    tactile_nonzero_values += tactile_count
                    update_tactile_nonzero_values += tactile_count
                    step_tactile_nonzero_by_role = (
                        _tactile_nonzero_by_role(
                            tactile,
                            role_names_by_env,
                        )
                    )
                    if (
                        update_tactile_nonzero_by_role is not None
                        and step_tactile_nonzero_by_role is not None
                    ):
                        for role, count in (
                            step_tactile_nonzero_by_role.items()
                        ):
                            update_tactile_nonzero_by_role[role] += count
                    tactile_abs_max = max(
                        tactile_abs_max, float(tactile.abs().max())
                    )
                    if reload_probe is None and (
                        signals.transition_valid.any()
                        and not signals.icm_normalizer_bootstrap
                    ):
                        reload_probe = _valid_icm_batch(
                            observation_t,
                            applied_action,
                            observations_tp1,
                            signals.transition_valid,
                        )
                    observations = observations_tp1
                algorithm.compute_returns(observations)
            action_sha256_by_update.append(
                _tensor_state_sha256(
                    {"actions": torch.stack(update_actions)}
                )
            )

            loss_metrics = algorithm.update()
            if hasattr(policy, "zero_tactile_causal_audit"):
                loss_metrics["zero_tactile_causal_audit"] = (
                    policy.zero_tactile_causal_audit()
                )
            integration_metrics = integrator.finish_rollout()
            policy_update_metrics_by_update.append(
                {"update": update_index, "metrics": loss_metrics}
            )
            integration_metrics_by_update.append(
                {"update": update_index, "metrics": integration_metrics}
            )
            valid_transition_count_by_update.append(
                update_valid_transition_count
            )
            tactile_nonzero_values_by_update.append(
                update_tactile_nonzero_values
            )
            if update_tactile_nonzero_by_role is not None:
                tactile_nonzero_by_role_by_update.append(
                    update_tactile_nonzero_by_role
                )

            if update_index in checkpoint_updates:
                if reload_probe is None:
                    raise RuntimeError(
                        "no valid post-bootstrap ICM transition for reload probe"
                    )
                checkpoint_probe = (
                    base_integrator.icm_trainer.module.transition(
                        reload_probe.vector_obs_t,
                        reload_probe.tactile_history_t,
                        reload_probe.applied_action_policy_units_t,
                        reload_probe.vector_obs_tp1,
                        reload_probe.tactile_history_tp1,
                    )["intrinsic_reward"]
                    .detach()
                    .cpu()
                )
                policy_checkpoint_state = policy.state_dict()
                optimizer_checkpoint_state = (
                    algorithm.checkpoint_state_dict()
                )
                integration_checkpoint_state = integrator.state_dict()
                combined_state = {
                    "protocol": (
                        "sugar_stage_i_official_refiner_residual_"
                        "multistep_checkpoint_v1"
                        if residual_teacher_contract
                        else "sugar_stage_h_combined_checkpoint_v2"
                    ),
                    "policy_state_dict": policy_checkpoint_state,
                    "policy_optimizer_state_dict": optimizer_checkpoint_state,
                    "integration_state_dict": integration_checkpoint_state,
                    "icm_probe_reward": checkpoint_probe,
                    "iteration": update_index,
                    "torch_cpu_rng_state": torch.get_rng_state(),
                    "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                }
                if residual_teacher_contract:
                    combined_state["residual_wrapper_state_dict"] = (
                        env.checkpoint_state_dict()
                    )
                checkpoint_path = checkpoint_paths[update_index]
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(combined_state, checkpoint_path)
                checkpoint_records[update_index] = {
                    "path": str(checkpoint_path),
                    "sha256": _sha256(checkpoint_path),
                    "policy_state_sha256": _state_tree_sha256(
                        policy_checkpoint_state
                    ),
                    "policy_optimizer_state_sha256": _state_tree_sha256(
                        optimizer_checkpoint_state
                    ),
                    "integration_state_sha256": _state_tree_sha256(
                        integration_checkpoint_state
                    ),
                    "icm_probe_reward": checkpoint_probe.tolist(),
                }
                if residual_teacher_contract:
                    checkpoint_records[update_index][
                        "residual_wrapper_state_sha256"
                    ] = _state_tree_sha256(
                        combined_state["residual_wrapper_state_dict"]
                    )

        prior_hash_after = _tensor_state_sha256(
            base_integrator.smp_scorer.prior.state_dict()
        )
        actor_delta = _state_max_abs(actor_state_before, policy.actor.state_dict())
        critic_delta = _state_max_abs(
            critic_state_before, policy.critic.state_dict()
        )
        policy_delta = _state_max_abs(policy_state_before, policy.state_dict())

        if reload_probe is None:
            raise RuntimeError("no valid post-bootstrap ICM transition for reload probe")
        checkpoint_reload_records: dict[int, dict[str, object]] = {}
        reload_policy_max_abs = 0.0
        reload_icm_reward_max_abs = 0.0
        reload_prior_hash = ""
        for update_index in sorted(checkpoint_updates):
            loaded = torch.load(
                checkpoint_paths[update_index],
                map_location=base_env.device,
                weights_only=True,
            )
            expected_checkpoint_protocol = (
                "sugar_stage_i_official_refiner_residual_"
                "multistep_checkpoint_v1"
                if residual_teacher_contract
                else "sugar_stage_h_combined_checkpoint_v2"
            )
            if (
                loaded.get("protocol") != expected_checkpoint_protocol
                or int(loaded.get("iteration", -1)) != update_index
            ):
                raise ValueError(
                    f"checkpoint protocol/iteration mismatch at {update_index}"
                )
            reload_policy, reload_algorithm, _ = _construct_policy_algorithm(
                observations, env, _runner_cfg()
            )
            reload_base_integrator = SMPICMRolloutIntegrator(
                base_env,
                prior_dir=str(prior_dir),
                mix_cfg=reward_mix_cfg,
            )
            reload_paper_cws_scorer = (
                OfficialTacSLPaperCWSReward(
                    base_env,
                    paper_cws_runtime_cfg,
                )
                if paper_cws_contract
                else None
            )
            reload_demo_scorer = (
                FrozenDemoRewardScorer(
                    num_envs=base_env.num_envs,
                    device=base_env.device,
                    cfg=demo_scorer.cfg,
                )
                if demo_reward_contract
                else None
            )
            reload_demo_event_scorer = (
                FrozenPhaseAwareDemoEventScorer(
                    num_envs=base_env.num_envs,
                    device=base_env.device,
                    cfg=demo_event_scorer.cfg,
                )
                if demo_event_reward_contract
                else None
            )
            reload_integrator = (
                PaperCWSAugmentedSMPICMRolloutIntegrator(
                    base=reload_base_integrator,
                    paper_cws=reload_paper_cws_scorer,
                    guidance_weight=args.paper_cws_guidance_weight,
                )
                if reload_paper_cws_scorer is not None
                else (
                    DemoEventRewardAugmentedSMPICMRolloutIntegrator(
                        base=reload_base_integrator,
                        demo=reload_demo_event_scorer,
                    )
                    if reload_demo_event_scorer is not None
                    else (
                        DemoRewardAugmentedSMPICMRolloutIntegrator(
                            base=reload_base_integrator,
                            demo=reload_demo_scorer,
                        )
                        if reload_demo_scorer is not None
                        else reload_base_integrator
                    )
                )
            )
            if active_demo_reward_contract:
                reload_integrator.begin(observations)
            else:
                reload_integrator.begin()
            reload_policy.load_state_dict(
                loaded["policy_state_dict"], strict=True
            )
            reload_algorithm.load_checkpoint_state_dict(
                loaded["policy_optimizer_state_dict"]
            )
            reload_integrator.load_state_dict(
                loaded["integration_state_dict"]
            )
            wrapper_state_exact = True
            if residual_teacher_contract:
                env.load_checkpoint_state_dict(
                    loaded["residual_wrapper_state_dict"]
                )
                wrapper_state_exact = (
                    _state_tree_sha256(env.checkpoint_state_dict())
                    == checkpoint_records[update_index][
                        "residual_wrapper_state_sha256"
                    ]
                )
            reload_zero_tactile_causal_audit = (
                reload_policy.zero_tactile_causal_audit()
                if hasattr(
                    reload_policy, "zero_tactile_causal_audit"
                )
                else None
            )
            reloaded_probe = (
                reload_base_integrator.icm_trainer.module.transition(
                    reload_probe.vector_obs_t,
                    reload_probe.tactile_history_t,
                    reload_probe.applied_action_policy_units_t,
                    reload_probe.vector_obs_tp1,
                    reload_probe.tactile_history_tp1,
                )["intrinsic_reward"]
                .detach()
                .cpu()
            )
            expected = checkpoint_records[update_index]
            policy_hash_matches = (
                _state_tree_sha256(reload_policy.state_dict())
                == expected["policy_state_sha256"]
            )
            optimizer_hash_matches = (
                _state_tree_sha256(
                    reload_algorithm.checkpoint_state_dict()
                )
                == expected["policy_optimizer_state_sha256"]
            )
            integration_hash_matches = (
                _state_tree_sha256(reload_integrator.state_dict())
                == expected["integration_state_sha256"]
            )
            probe_max_abs = float(
                torch.abs(
                    loaded["icm_probe_reward"].detach().cpu()
                    - reloaded_probe
                ).max()
            )
            current_reload_prior_hash = _tensor_state_sha256(
                reload_base_integrator.smp_scorer.prior.state_dict()
            )
            final_policy_max_abs = (
                _state_max_abs(
                    policy.state_dict(), reload_policy.state_dict()
                )
                if update_index == args.num_updates
                else None
            )
            checkpoint_reload_records[update_index] = {
                "policy_state_exact": policy_hash_matches,
                "policy_optimizer_state_exact": optimizer_hash_matches,
                "integration_state_exact": integration_hash_matches,
                "residual_wrapper_state_exact": wrapper_state_exact,
                "icm_probe_reward_max_abs": probe_max_abs,
                "prior_state_sha256": current_reload_prior_hash,
                "final_live_policy_max_abs": final_policy_max_abs,
                "zero_tactile_causal_audit": (
                    reload_zero_tactile_causal_audit
                ),
            }
            if final_policy_max_abs is not None:
                reload_policy_max_abs = final_policy_max_abs
            reload_icm_reward_max_abs = max(
                reload_icm_reward_max_abs, probe_max_abs
            )
            reload_prior_hash = current_reload_prior_hash
            del reload_integrator, reload_algorithm, reload_policy, loaded
            torch.cuda.empty_cache()

        tactile_stress_proof = direct_tactile_stress_audit(base_env)
        drop_grace_proof = (
            env.drop_grace_audit_state()
            if residual_teacher_contract
            else None
        )
        teacher_partition_proof = (
            env.teacher_partition_audit_state()
            if residual_teacher_contract
            else None
        )
        teacher_reference_advance_proof = (
            env.teacher_reference_advance_audit_state()
            if residual_teacher_contract
            else None
        )
        support_hold_proof = (
            env.support_hold_audit_state()
            if residual_teacher_contract
            else None
        )
        reference_waypoint_foundation_proof = (
            reference_waypoint_foundation_reset.audit_state()
            if reference_waypoint_foundation_reset is not None
            else None
        )
        policy_observation_terms = list(
            base_env.observation_manager.active_terms["policy"]
        )
        icm_observation_terms = list(
            base_env.observation_manager.active_terms["icm_vector"]
        )
        outcome_weight_map = {
            name: float(base_env.reward_manager.get_term_cfg(name).weight)
            for name in OUTCOME_REWARD_TERMS
        }
        external_constraint_weights = {
            name: float(base_env.reward_manager.get_term_cfg(name).weight)
            for name in EXTERNAL_CONSTRAINT_TERMS
        }
        termination_names = list(base_env.termination_manager.active_terms)
        transition_fields = [field.name for field in fields(ICMTransitionBatch)]
        forbidden_icm_fields = sorted(
            name
            for name in transition_fields
            if any(
                fragment in name.lower()
                for fragment in (
                    "reward",
                    "success",
                    "failure",
                    "lift",
                    "slip",
                    "strategy",
                    "mass",
                    "friction",
                    "oracle",
                )
            )
        )
        outcome_ledger_abs_max = max(
            abs(step[name]) for step in ledgers for name in OUTCOME_REWARD_TERMS
        )
        all_scalar_values = (
            icm_means
            + smp_means
            + policy_base_reward_means
            + policy_reward_means
            + task_outcome_means
            + external_means
            + raw_sds_means
            + demo_reward_means
            + demo_unit_reward_means
            + demo_event_potential_means
            + demo_event_risk_means
            + demo_event_uncertainty_means
            + demo_event_ready_fractions
            + demo_event_phase_means
            + paper_cws_reward_means
            + paper_cws_weighted_reward_means
            + _numeric_leaves(policy_update_metrics_by_update)
            + _numeric_leaves(integration_metrics_by_update)
        )
        expected_samples_per_update = (
            args.num_envs * runner_dict["num_steps_per_env"]
        )
        strict_mimickit = args.policy_contract == "strict_mimickit"
        if strict_mimickit:
            optimizer_epoch_telemetry = [
                epoch
                for record in policy_update_metrics_by_update
                for epoch in record["metrics"]["actor_epoch_telemetry"]
            ]
            epoch_approx_kl_key = "approx_kl_mean"
        else:
            optimizer_epoch_telemetry = [
                epoch
                for record in policy_update_metrics_by_update
                for epoch in record["metrics"]["epoch_telemetry"]
            ]
            epoch_approx_kl_key = "sampled_approx_kl_mean"
        h1_numerical_stability = (
            all(
                float(record["metrics"]["clip_fraction"]) <= 0.5
                for record in policy_update_metrics_by_update
            )
            and all(
                float(epoch["clip_fraction_max"]) <= 0.8
                and float(epoch[epoch_approx_kl_key]) <= 0.1
                for epoch in optimizer_epoch_telemetry
            )
        )
        checkpoint_reload_exact = all(
            bool(record["policy_state_exact"])
            and bool(record["policy_optimizer_state_exact"])
            and bool(record["integration_state_exact"])
            and float(record["icm_probe_reward_max_abs"]) == 0.0
            and record["prior_state_sha256"] == prior_hash_before
            and bool(record["residual_wrapper_state_exact"])
            for record in checkpoint_reload_records.values()
        )
        teacher_hash_after = (
            _tensor_state_sha256(env.teacher.actor.state_dict())
            if residual_teacher_contract
            else None
        )
        demo_final_audit = (
            demo_event_scorer.frozen_model_audit()
            if demo_event_scorer is not None
            else demo_scorer.frozen_model_audit()
            if demo_scorer is not None
            else None
        )
        explicit_zero_runtime_audit = None
        if explicit_zero_tactile_contract:
            zero_entry = getattr(
                base_env, "_sugar_goal_tactile_strategy_runtime", None
            )
            if (
                not isinstance(zero_entry, tuple)
                or len(zero_entry) != 2
                or not isinstance(
                    zero_entry[1],
                    ExplicitZeroTactileStrategyControlRuntime,
                )
            ):
                raise RuntimeError(
                    "explicit-zero tactile strategy runtime was not preserved"
                )
            explicit_zero_runtime_audit = zero_entry[1].audit_state()
        paper_cws_final_audit = (
            paper_cws_scorer.audit_state()
            if paper_cws_scorer is not None
            else None
        )
        posture_joint_motion = None
        if posture_adaptive_contract:
            posture_joint_range = (
                posture_joint_position_max - posture_joint_position_min
            )
            posture_joint_motion = {
                "joint_names": posture_joint_names,
                "robot_joint_ids": (
                    posture_robot_joint_ids.detach().cpu().tolist()
                ),
                "initial_min_rad": (
                    posture_joint_position_initial.amin(dim=0)
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "initial_max_rad": (
                    posture_joint_position_initial.amax(dim=0)
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "observed_min_rad": (
                    posture_joint_position_min.detach().cpu().tolist()
                ),
                "observed_max_rad": (
                    posture_joint_position_max.detach().cpu().tolist()
                ),
                "observed_range_rad": (
                    posture_joint_range.detach().cpu().tolist()
                ),
                "maximum_observed_range_rad": float(
                    posture_joint_range.max()
                ),
            }
        if strict_mimickit:
            optimizer_checks = {
                "official_smp_actor_update_count_exact": (
                    algorithm.actor_optimizer_steps == 40 * args.num_updates
                ),
                "official_smp_critic_update_count_exact": (
                    algorithm.critic_optimizer_steps == 32 * args.num_updates
                ),
                "official_smp_preupdate_logprob_identity_each_rollout": all(
                    abs(
                        float(
                            record["metrics"][
                                "preupdate_importance_ratio_mean"
                            ]
                        )
                        - 1.0
                    )
                    <= 1.0e-3
                    and float(
                        record["metrics"][
                            "preupdate_importance_ratio_max_abs_from_one"
                        ]
                    )
                    <= 1.0e-2
                    and float(
                        record["metrics"]["preupdate_clip_fraction"]
                    )
                    == 0.0
                    for record in policy_update_metrics_by_update
                ),
                "official_smp_policy_normalizer_rollout_delayed": (
                    all(
                        int(
                            record["metrics"][
                                "policy_normalizer_samples_committed"
                            ]
                        )
                        == expected_samples_per_update
                        and int(
                            record["metrics"][
                                "policy_normalizer_total_samples"
                            ]
                        )
                        == expected_samples_per_update
                        * int(record["update"])
                        for record in policy_update_metrics_by_update
                    )
                    and int(policy._pending_actor_count.item()) == 0
                    and int(policy._pending_critic_count.item()) == 0
                ),
                "official_smp_telemetry_complete": all(
                    record["metrics"]["telemetry_schema"]
                    == "official_smp_policy_update_telemetry_v1"
                    and len(
                        record["metrics"]["actor_epoch_telemetry"]
                    )
                    == 5
                    and len(
                        record["metrics"]["actor_mini_batch_telemetry"]
                    )
                    == 40
                    and len(
                        record["metrics"]["critic_epoch_telemetry"]
                    )
                    == 2
                    and len(
                        record["metrics"]["critic_mini_batch_telemetry"]
                    )
                    == 32
                    for record in policy_update_metrics_by_update
                ),
                "official_smp_fixed_action_std_exact": all(
                    abs(
                        float(
                            record["metrics"]["rollout_telemetry"][
                                "fixed_action_std_min"
                            ]
                        )
                        - 0.05
                    )
                    <= 1.0e-8
                    and abs(
                        float(
                            record["metrics"]["rollout_telemetry"][
                                "fixed_action_std_max"
                            ]
                        )
                        - 0.05
                    )
                    <= 1.0e-8
                    for record in policy_update_metrics_by_update
                ),
                "official_smp_action_bound_loss_zero": all(
                    float(record["metrics"]["action_bound"]) == 0.0
                    for record in policy_update_metrics_by_update
                ),
            }
        else:
            expected_native_contract = (
                args.policy_contract
                if args.policy_contract
                in {
                    "sugar_native_tactile_floor_lr",
                    "sugar_native_zero_preserving_tactile_floor_lr",
                    "sugar_native_zero_preserving_tactile_fixed_low_lr",
                }
                else "sugar_native_base_ppo"
            )
            expected_initial_learning_rate = (
                1.0e-5
                if args.policy_contract
                in {
                    "sugar_native_tactile_floor_lr",
                    "sugar_native_zero_preserving_tactile_floor_lr",
                    "sugar_native_zero_preserving_tactile_fixed_low_lr",
                }
                else 1.0e-3
            )
            optimizer_checks = {
                "sugar_native_adam_update_count_exact": (
                    algorithm.optimizer_steps == 20 * args.num_updates
                    and algorithm.completed_updates == args.num_updates
                ),
                "sugar_native_preupdate_logprob_identity_each_rollout": all(
                    abs(
                        float(
                            record["metrics"]["rollout_telemetry"][
                                "preupdate_importance_ratio_mean"
                            ]
                        )
                        - 1.0
                    )
                    <= 1.0e-3
                    and float(
                        record["metrics"]["rollout_telemetry"][
                            "preupdate_importance_ratio_max_abs_from_one"
                        ]
                    )
                    <= 1.0e-2
                    and float(
                        record["metrics"]["rollout_telemetry"][
                            "preupdate_clip_fraction"
                        ]
                    )
                    == 0.0
                    for record in policy_update_metrics_by_update
                ),
                "sugar_native_upstream_ppo_implementation_exact": all(
                    record["metrics"]["optimizer_implementation"]
                    == "rsl_rl.algorithms.PPO.update"
                    and record["metrics"]["upstream_rsl_rl_version"]
                    == "3.0.1"
                    and record["metrics"]["upstream_ppo_sha256"]
                    == (
                        "deafc8c947eba4df3e91b393869426cdab8d7b71e05974c"
                        "3734125d2331d7d1c"
                    )
                    for record in policy_update_metrics_by_update
                ),
                "sugar_native_named_optimizer_contract_exact": (
                    algorithm.contract_name == expected_native_contract
                    and all(
                        float(record["metrics"]["learning_rate_start"])
                        == (
                            segment_initial_learning_rate
                            if position == 0
                            else float(
                                policy_update_metrics_by_update[
                                    position - 1
                                ]["metrics"]["learning_rate_end"]
                            )
                        )
                        for position, record in enumerate(
                            policy_update_metrics_by_update
                        )
                    )
                ),
                "sugar_native_telemetry_complete": all(
                    len(record["metrics"]["epoch_telemetry"]) == 5
                    and len(record["metrics"]["mini_batch_telemetry"]) == 20
                    and record["metrics"]["optimizer_steps_this_update"] == 20
                    for record in policy_update_metrics_by_update
                ),
                "sugar_native_learning_rate_within_upstream_bounds": all(
                    1.0e-5
                    <= float(mini_batch["learning_rate_at_step"])
                    <= 1.0e-2
                    for record in policy_update_metrics_by_update
                    for mini_batch in record["metrics"][
                        "mini_batch_telemetry"
                    ]
                ),
                "sugar_native_learning_rate_schedule_exact": all(
                    (
                        float(mini_batch["learning_rate_at_step"])
                        == 1.0e-5
                        and float(
                            mini_batch[
                                "learning_rate_before_adaptation"
                            ]
                        )
                        == 1.0e-5
                    )
                    if args.policy_contract
                    == "sugar_native_zero_preserving_tactile_fixed_low_lr"
                    else True
                    for record in policy_update_metrics_by_update
                    for mini_batch in record["metrics"][
                        "mini_batch_telemetry"
                    ]
                ),
                "sugar_native_initial_action_std_exact": bool(
                    (
                        torch.isfinite(initial_action_std).all()
                        and torch.all(initial_action_std > 0.0)
                    )
                    if resume_payload is not None
                    else torch.all(initial_action_std == 1.0)
                ),
                "sugar_native_action_is_not_artificially_bounded": (
                    env.clip_actions is None
                ),
            }
        checks = {
            "tactile_mount_environment_contract_exact": (
                (
                    all(
                        value is None
                        for value in observed_tacsl_mount_environment.values()
                    )
                    and protocol_config["shared_runtime"][
                        "tactile_mount_environment"
                    ]
                    is None
                )
                if explicit_zero_tactile_contract
                else (
                    (
                        observed_tacsl_mount_environment
                        == protocol_config["shared_runtime"][
                            "tactile_mount_environment"
                        ]
                        == EXPECTED_TACSL_MOUNT_ENVIRONMENT
                    )
                    if protocol_config is not None
                    else True
                )
            ),
            "strict_torch_determinism_runtime_exact": (
                (
                    args.strict_deterministic_torch
                    and torch.are_deterministic_algorithms_enabled()
                    and torch.backends.cudnn.deterministic
                    and not torch.backends.cudnn.benchmark
                    and not torch.backends.cuda.matmul.allow_tf32
                    and not torch.backends.cudnn.allow_tf32
                    and os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                    == ":4096:8"
                )
                if args.strict_deterministic_torch
                else True
            ),
            "external_constraints_present": all(
                name in base_env.reward_manager.active_terms
                for name in EXTERNAL_CONSTRAINT_TERMS
            ),
            "icm_transition_api_has_no_outcome_or_result": not forbidden_icm_fields,
            "icm_first_valid_step_is_normalizer_bootstrap_only": (
                (
                    bootstrap_count == 0
                    and base_integrator.icm_bootstrap_steps == 1
                )
                if resume_payload is not None
                else (bootstrap_count == 1 and icm_means[0] == 0.0)
            ),
            "icm_postbootstrap_discovery_positive": max(icm_means[1:]) > 0.0,
            "policy_reward_reconstruction_exact": (
                reward_reconstruction_max_abs == 0.0
            ),
            "smp_live_window_shape_exact": list(initial_smp_window.shape)
            == [args.num_envs, 10, 216],
            "smp_esm_scored_every_step": (
                base_integrator.smp_scorer.transitions_scored
                == expected_samples_per_update * args.num_updates
            ),
            "smp_rollout_delayed_normalizer_updated_each_rollout": (
                base_integrator.smp_scorer.normalizer_updates
                == args.num_updates
            ),
            "smp_prior_bitwise_frozen": prior_hash_before == prior_hash_after,
            "tactile_stream_matches_declared_regime": (
                (
                    contact_seed["tactile_arrays_loaded"] is False
                    and contact_seed["tactile_sensor_data_read"] is False
                    and contact_seed[
                        "raw_sensor_nonzero_values_after_restore"
                    ]
                    == 0
                    and contact_seed["raw_sensor_abs_max_after_restore"]
                    == 0.0
                    and tactile_nonzero_values == 0
                    and tactile_abs_max == 0.0
                )
                if explicit_zero_tactile_contract
                else (
                    contact_seed[
                        "raw_sensor_nonzero_values_after_restore"
                    ]
                    > 0
                    and contact_seed[
                        "raw_sensor_abs_max_after_restore"
                    ]
                    > 0.0
                    and tactile_nonzero_values > 0
                    and tactile_abs_max > 0.0
                )
            ),
            "icm_independent_optimizer_updated_each_rollout": (
                base_integrator.icm_trainer.optimizer_updates
                == args.num_updates
            ),
            "icm_scored_before_update": all(
                record["metrics"]["icm_scored_before_update"] is True
                for record in integration_metrics_by_update
            ),
            "all_valid_transitions_accounted": (
                sum(
                    int(record["metrics"]["rollout_valid_transitions"])
                    for record in integration_metrics_by_update
                )
                == valid_transition_count
                and [
                    int(record["metrics"]["rollout_valid_transitions"])
                    for record in integration_metrics_by_update
                ]
                == valid_transition_count_by_update
            ),
            **optimizer_checks,
            "resume_checkpoint_state_restored_exact": (
                resume_restore_record is None
                or (
                    all(
                        bool(resume_restore_record[name])
                        for name in (
                            "policy_state_exact",
                            "optimizer_state_exact",
                            "hybrid_integration_state_exact",
                        )
                    )
                    and (
                        bool(
                            resume_restore_record[
                                "residual_wrapper_transition_valid"
                            ]
                        )
                        if teacher_floor_resume_contract
                        else bool(
                            resume_restore_record[
                                "residual_wrapper_state_exact"
                            ]
                        )
                    )
                )
            ),
            "policy_actor_parameters_changed": actor_delta > 0.0,
            "policy_critic_parameters_changed": critic_delta > 0.0,
            "combined_policy_parameters_changed": policy_delta > 0.0,
            "combined_checkpoints_reload_exact": checkpoint_reload_exact,
            "all_reported_scalars_finite": all(
                torch.isfinite(torch.tensor(value, dtype=torch.float64)).item()
                for value in all_scalar_values
            ),
        }
        if goal_recovery_contract:
            checks.update(
                {
                    "goal_outcome_weights_exact": outcome_weight_map
                    == {
                        "goal_position": 1.0,
                        "goal_orientation": 0.2,
                        "lift_fraction": 0.5,
                        "goal_stability": 0.25,
                    },
                    "goal_success_termination_retained": (
                        "success" in termination_names
                    ),
                    "goal_drop_termination_removed_only": (
                        "dropped_after_lift" not in termination_names
                        and {
                            "time_out",
                            "success",
                            "unsafe_fall",
                            "box_out_of_workspace",
                        }.issubset(termination_names)
                    ),
                    "goal_outcome_ledger_live": (
                        outcome_ledger_abs_max > 0.0
                        and max(abs(value) for value in task_outcome_means)
                        > 0.0
                    ),
                    "goal_task_constraint_mix_exact": (
                        base_integrator.mix_cfg.task_outcome_weight
                        == (10.0 if native_authority_contract else 1.0)
                        and base_integrator.mix_cfg.external_constraint_weight
                        == 1.0
                        and base_integrator.mix_cfg.smp_reward_weight == 0.5
                        and base_integrator.mix_cfg.icm_reward_weight
                        == (
                            float(
                                reference_waypoint_foundation_config[
                                    "reward"
                                ]["icm_policy_weight"]
                            )
                            if reference_waypoint_foundation_contract
                            else (
                                0.0
                                if args.reward_control
                                == "plan11_icm_policy_weight_zero"
                                else 1.0
                            )
                        )
                        and not (
                            base_integrator.mix_cfg.require_zero_outcome_rewards
                        )
                        and not (
                            base_integrator.mix_cfg.require_no_success_termination
                        )
                    ),
                    "goal_coherent_latent_dynamics_pass": (
                        latent_dynamics_proof is not None
                        and (
                            (
                                latent_dynamics_proof["checks"][
                                    "startup_tuple_readback_bitwise_exact"
                                ]
                                is True
                                and latent_dynamics_proof["checks"][
                                    "physics_tuple_hidden_from_policy_and_icm"
                                ]
                                is True
                                and latent_dynamics_proof["checks"][
                                    "reference_pulse_disabled"
                                ]
                                is True
                            )
                            if reference_waypoint_foundation_contract
                            else bool(latent_dynamics_proof["passed"])
                        )
                    ),
                    "v16_direct_tacsl_slip_belief_is_policy_input": (
                        "v16_tactile_slip_belief"
                        in policy_observation_terms
                    ),
                    "v16_slip_cost_is_external_and_separate_from_icm": (
                        external_constraint_weights.get("tactile_slip")
                        == -0.25
                        and "v16_tactile_slip_belief"
                        not in icm_observation_terms
                        and not any(
                            "slip" in name.lower()
                            for name in transition_fields
                        )
                    ),
                }
            )
        else:
            checks.update(
                {
                    "pure_discovery_outcome_weights_zero": all(
                        value == 0.0
                        for value in outcome_weight_map.values()
                    ),
                    "pure_discovery_success_termination_absent": (
                        "success" not in termination_names
                    ),
                    "outcome_reward_ledger_exactly_zero": (
                        outcome_ledger_abs_max == 0.0
                    ),
                }
            )
        if reference_waypoint_foundation_contract:
            expected_source_index = [
                index % 2 for index in range(args.num_envs)
            ]
            expected_mass = [
                1.0 if index == 0 else 3.0
                for index in expected_source_index
            ]
            checks.update(
                {
                    "w0_pair_audit_bound_and_passed_before_w1": (
                        reference_waypoint_foundation_config is not None
                        and reference_waypoint_foundation_config[
                            "w0_pair_audit"
                        ]["sha256"]
                        == _sha256(
                            _workspace_path(
                                reference_waypoint_foundation_config[
                                    "w0_pair_audit"
                                ]["path"]
                            )
                        )
                    ),
                    "foundation_two_source_assignment_exact": (
                        reference_waypoint_foundation_proof is not None
                        and reference_waypoint_foundation_proof[
                            "source_ids"
                        ]
                        == ["nominal_mass1", "heavy_mass3"]
                        and reference_waypoint_foundation_proof[
                            "source_index_by_env"
                        ]
                        == expected_source_index
                        and reference_waypoint_foundation_proof[
                            "reference_frame_by_source"
                        ]
                        == [299, 308]
                        and reference_waypoint_foundation_proof[
                            "waypoint_reference_frame_by_source"
                        ]
                        == [None, None]
                        and reference_waypoint_foundation_proof[
                            "waypoint_relative_lift_m_by_source"
                        ]
                        == [0.04, 0.04]
                    ),
                    "foundation_source_consistent_physics_exact": (
                        reference_waypoint_foundation_proof is not None
                        and np.allclose(
                            reference_waypoint_foundation_proof[
                                "mass_scale_by_env"
                            ],
                            expected_mass,
                            rtol=0.0,
                            atol=1.0e-7,
                        )
                        and np.allclose(
                            reference_waypoint_foundation_proof[
                                "static_friction_by_env"
                            ],
                            [0.6] * args.num_envs,
                            rtol=0.0,
                            atol=1.0e-7,
                        )
                        and np.allclose(
                            reference_waypoint_foundation_proof[
                                "dynamic_friction_by_env"
                            ],
                            [0.5] * args.num_envs,
                            rtol=0.0,
                            atol=1.0e-7,
                        )
                        and np.allclose(
                            reference_waypoint_foundation_proof[
                                "com_y_m_by_env"
                            ],
                            [0.0] * args.num_envs,
                            rtol=0.0,
                            atol=1.0e-7,
                        )
                        and not any(
                            any(value != 0.0 for value in vector)
                            for vector in reference_waypoint_foundation_proof[
                                "pulse_delta_velocity_w_mps_by_env"
                            ]
                        )
                    ),
                    "foundation_reset_hook_live_and_repeated": (
                        reference_waypoint_foundation_proof is not None
                        and reference_waypoint_foundation_proof[
                            "hook_installed"
                        ]
                        is True
                        and reference_waypoint_foundation_proof[
                            "reset_calls"
                        ]
                        >= (2 if args.num_updates == 64 else 1)
                        and reference_waypoint_foundation_proof[
                            "reset_environment_steps"
                        ]
                        >= (
                            args.num_envs + 1
                            if args.num_updates == 64
                            else args.num_envs
                        )
                    ),
                    "foundation_native_reference_advances_once": (
                        teacher_reference_advance_proof is not None
                        and teacher_reference_advance_proof["mode"]
                        == (
                            "goal_teacher_post_step_once"
                            if reference_waypoint_foundation_config[
                                "protocol"
                            ].endswith("_v3")
                            else "command_manager_only"
                        )
                        and teacher_reference_advance_proof[
                            "nonreset_environment_steps"
                        ]
                        > 0
                        and teacher_reference_advance_proof[
                            "nonreset_exactly_one_native_frame"
                        ]
                        is True
                    ),
                    "foundation_icm_learner_active_and_policy_weight_matched": (
                        base_integrator.mix_cfg.icm_reward_weight
                        == float(
                            reference_waypoint_foundation_config[
                                "reward"
                            ]["icm_policy_weight"]
                        )
                        and base_integrator.icm_trainer.optimizer_updates
                        == args.num_updates
                        and max(icm_means[1:]) > 0.0
                    ),
                    "foundation_residual_scale_fixed_low_exact": (
                        native_authority_scale_routing_exact
                        and bool(support_residual_scales)
                        and bool(manipulation_residual_scales)
                        and min(support_residual_scales)
                        == float(
                            torch.tensor(
                                args.residual_scale,
                                dtype=torch.float32,
                            )
                        )
                        and max(support_residual_scales)
                        == float(
                            torch.tensor(
                                args.residual_scale,
                                dtype=torch.float32,
                            )
                        )
                        and min(manipulation_residual_scales)
                        == float(
                            torch.tensor(
                                args.residual_scale,
                                dtype=torch.float32,
                            )
                        )
                        and max(manipulation_residual_scales)
                        == float(
                            torch.tensor(
                                args.residual_scale,
                                dtype=torch.float32,
                            )
                        )
                    ),
                }
            )
        if h2_contract:
            expected_stress_steps = (
                (4 if args.causal_contact_bootstrap_v2 else 1)
                + args.num_updates * runner_dict["num_steps_per_env"]
            )
            expected_stress_cache_hits = (
                (2 if args.causal_contact_bootstrap_v2 else 0)
                + args.num_updates * runner_dict["num_steps_per_env"]
            )
            checks.update(
                {
                    "h2r1_stress_runtime_self_audit_passes": (
                        tactile_stress_proof is not None
                        and bool(tactile_stress_proof["passed"])
                    ),
                    "h2r1_stress_active_before_first_icm_transition": (
                        initial_tactile_nonzero_by_role is not None
                        and all(
                            count > 0
                            for count in (
                                initial_tactile_nonzero_by_role.values()
                            )
                        )
                    ),
                    "h2r1_all_roles_have_raw_tacsl_provenance": (
                        tactile_stress_proof is not None
                        and all(
                            int(
                                tactile_stress_proof[
                                    "raw_nonzero_values_by_role"
                                ][role]
                            )
                            > 0
                            and int(
                                tactile_stress_proof[
                                    "raw_nonzero_steps_by_role"
                                ][role]
                            )
                            > 0
                            for role in H2_TACTILE_STRESS_ROLES
                        )
                    ),
                    "h2r1_initial_contact_seed_reaches_all_roles": (
                        tactile_stress_proof is not None
                        and all(
                            int(
                                tactile_stress_proof[
                                    "initial_raw_nonzero_by_role"
                                ][role]
                            )
                            > 0
                            for role in H2_TACTILE_STRESS_ROLES
                        )
                    ),
                    "h2r1_exactly_one_stress_frame_per_control_step": (
                        tactile_stress_proof is not None
                        and int(tactile_stress_proof["generated_steps"])
                        == expected_stress_steps
                    ),
                    "h2r1_role_not_actor_or_icm_input": not any(
                        fragment in name.lower()
                        for name in (
                            policy_observation_terms
                            + icm_observation_terms
                            + transition_fields
                        )
                        for fragment in ("stress_role", "tactile_role")
                    ),
                    "h2r1_shared_history_boundary_reuse_exact": (
                        tactile_stress_proof is not None
                        and int(tactile_stress_proof["cache_hits"])
                        == expected_stress_cache_hits
                        and int(tactile_stress_proof["apply_calls"])
                        == int(tactile_stress_proof["generated_steps"])
                        + int(tactile_stress_proof["cache_hits"])
                    ),
                }
            )
        elif explicit_zero_tactile_contract:
            checks.update(
                {
                    "explicit_zero_has_no_tactile_stress_runtime": (
                        tactile_stress_proof is None
                    ),
                    "explicit_zero_observation_exact_all_steps": (
                        initial_tactile_nonzero_values == 0
                        and initial_tactile_abs_max == 0.0
                        and all(
                            count == 0
                            for count in tactile_nonzero_values_by_update
                        )
                    ),
                    "explicit_zero_strategy_runtime_no_sensor_reads": (
                        explicit_zero_runtime_audit is not None
                        and explicit_zero_runtime_audit["sensor_read_count"]
                        == 0
                        and explicit_zero_runtime_audit[
                            "all_outputs_exact_zero"
                        ]
                        is True
                        and explicit_zero_runtime_audit[
                            "slip_observation_shape"
                        ]
                        == [args.num_envs, 14]
                        and explicit_zero_runtime_audit[
                            "strategy_observation_shape"
                        ]
                        == [args.num_envs, 40]
                    ),
                    "explicit_zero_tactile_rewards_exact_zero": all(
                        record["tactile_slip"] == 0.0
                        and record["repeated_failed_strategy"] == 0.0
                        for record in ledgers
                    ),
                    "explicit_zero_previous_action_restored_bitwise": (
                        contact_seed[
                            "previous_action_reaches_observation_bitwise"
                        ]
                        is True
                        and contact_seed[
                            "previous_actor_action_reaches_official_"
                            "last_action_bitwise"
                        ]
                        is True
                        and contact_seed[
                            "previous_applied_action_reaches_goal_"
                            "observation_bitwise"
                        ]
                        is True
                    ),
                    (
                        "wrong_teacher_first_action_differs_from_carrybox_source"
                        if unrelated_teacher_arm
                        else "explicit_zero_first_teacher_action_matches_source"
                    ): (
                        explicit_zero_teacher_observation_shape
                        == [args.num_envs, 890]
                        and explicit_zero_teacher_action_l2 is not None
                        and explicit_zero_teacher_action_max_abs is not None
                        and explicit_zero_teacher_action_canonical_max_abs
                        is not None
                        and (
                            explicit_zero_teacher_action_canonical_max_abs
                            > 1.0e-3
                            if unrelated_teacher_arm
                            else explicit_zero_teacher_action_canonical_max_abs
                            <= 2.0e-6
                        )
                    ),
                }
            )
        else:
            checks["nominal_run_has_no_tactile_stress_runtime"] = (
                tactile_stress_proof is None
            )
        if residual_teacher_contract:
            checks.update(
                {
                    "residual_mean_initialized_exact_zero": (
                        (
                            resume_restore_record is not None
                            and resume_restore_record["policy_state_exact"]
                        )
                        if resume_payload is not None
                        else (
                            residual_zero_initialization is not None
                            and bool(residual_zero_initialization["passed"])
                        )
                    ),
                    "ppo_storage_variable_is_residual_action": (
                        residual_action_storage_exact
                    ),
                    "teacher_actor_bitwise_frozen": (
                        teacher_hash_before == teacher_hash_after
                        and env.teacher.frozen_audit()["passed"]
                    ),
                    "original_icm_has_no_teacher_or_release_field": (
                        not any(
                            fragment in name.lower()
                            for name in transition_fields
                            for fragment in (
                                "teacher",
                                "coefficient",
                                "failure",
                            )
                        )
                    ),
                }
            )
            if explicit_zero_tactile_contract:
                checks[
                    (
                        "teacher_floor_schedule_reaches_exact_nonzero_floor"
                        if teacher_floor_overfit_contract
                        else (
                            "wrong_teacher_global_schedule_reaches_zero"
                            if annealed_wrong_teacher_contract
                            else "explicit_zero_control_keeps_teacher_authority_fixed"
                        )
                    )
                ] = (
                    (
                        teacher_coefficients
                        and max(teacher_coefficients) <= 1.0
                        and min(teacher_coefficients) >= 0.25
                        and teacher_zero_control_steps == 0
                        and env.release.global_control_steps
                        == args.teacher_anneal_updates * 24
                        and bool(
                            torch.all(env.release.coefficient == 0.25)
                        )
                    )
                    if teacher_floor_overfit_contract
                    else (
                    (
                        teacher_coefficients
                        and max(teacher_coefficients) <= 1.0
                        and min(teacher_coefficients) >= 0.0
                        and env.release.global_control_steps
                        == args.teacher_anneal_updates * 24
                        and float(env.release.coefficient.max()) == 0.0
                    )
                    if annealed_wrong_teacher_contract
                    else (
                        teacher_coefficients
                        and teacher_release_events == 0
                        and teacher_zero_control_steps == 0
                        and min(teacher_coefficients) == 1.0
                        and max(teacher_coefficients) == 1.0
                        and not bool(env.release.release_latched.any())
                        and not bool(env.release.release_progress.any())
                    )
                    )
                )
            elif reference_waypoint_foundation_contract:
                checks[
                    "foundation_teacher_authority_fixed_bitwise_one"
                ] = (
                    teacher_coefficients
                    and teacher_zero_control_steps == 0
                    and min(teacher_coefficients) == 1.0
                    and max(teacher_coefficients) == 1.0
                    and env.release.mode == "fixed_one"
                    and not bool(env.release.release_latched.any())
                    and not bool(env.release.release_progress.any())
                )
            else:
                checks["real_tactile_failure_releases_teacher"] = (
                    teacher_release_events > 0
                    and teacher_zero_control_steps > 0
                    and min(teacher_coefficients) == 0.0
                    and max(teacher_coefficients) == 1.0
                )
            if args.causal_contact_bootstrap_v2:
                checks.update(
                    {
                        "causal_contact_previous_action_restored_bitwise": (
                            contact_seed[
                                "previous_action_reaches_observation_bitwise"
                            ]
                            is True
                            and contact_seed[
                                "previous_actor_action_reaches_official_"
                                "last_action_bitwise"
                            ]
                            is True
                            and contact_seed[
                                "previous_applied_action_reaches_goal_"
                                "observation_bitwise"
                            ]
                            is True
                        ),
                        "causal_contact_four_real_tacsl_frames_exact": (
                            contact_seed["source_tactile_history_frames"]
                            is not None
                            and len(
                                contact_seed[
                                    "source_tactile_history_frames"
                                ]
                            )
                            == 4
                            and causal_bootstrap_tactile_max_abs == 0.0
                            and causal_bootstrap_nominal_source_max_abs
                            == 0.0
                        ),
                        "causal_contact_first_teacher_action_matches_source": (
                            causal_bootstrap_teacher_action_l2 is not None
                            and causal_bootstrap_teacher_action_l2
                            <= 5.0e-6
                            and causal_bootstrap_teacher_action_max_abs
                            is not None
                            and causal_bootstrap_teacher_action_max_abs
                            <= 3.0e-6
                            and causal_bootstrap_teacher_observation_reference
                            is not None
                            and causal_bootstrap_teacher_observation_reference[
                                "reference_action_vs_source_l2"
                            ]
                            <= 5.0e-6
                            and causal_bootstrap_teacher_observation_reference[
                                "reference_action_vs_source_max_abs"
                            ]
                            <= 2.0e-6
                            and causal_bootstrap_teacher_observation_reference[
                                "all_environment_max_abs"
                            ]
                            <= 2.0e-6
                        ),
                    }
                )
            if (
                native_authority_contract
                and not reference_waypoint_foundation_contract
                and not wrong_teacher_reward_conflict_contract
            ):
                base_scale_float32 = float(
                    torch.tensor(args.residual_scale, dtype=torch.float32)
                )
                post_scale_float32 = float(
                    torch.tensor(
                        args.post_release_residual_scale,
                        dtype=torch.float32,
                    )
                )
                if posture_adaptive_contract:
                    posture_scale_float32 = float(
                        torch.tensor(
                            args.posture_post_failure_residual_scale,
                            dtype=torch.float32,
                        )
                    )
                    posture_floor_float32 = float(
                        torch.tensor(
                            args.posture_post_failure_teacher_floor,
                            dtype=torch.float32,
                        )
                    )
                    checks.update(
                        {
                            "posture_three_way_named_partition_exact": (
                                teacher_partition_proof[
                                    "three_way_complete_disjoint_partition"
                                ]
                                is True
                                and len(
                                    teacher_partition_proof[
                                        "manipulation_indices"
                                    ]
                                )
                                == 14
                                and len(
                                    teacher_partition_proof[
                                        "posture_indices"
                                    ]
                                )
                                == 11
                                and len(
                                    teacher_partition_proof[
                                        "balance_indices"
                                    ]
                                )
                                == 4
                            ),
                            "posture_per_joint_teacher_and_scale_routing_exact": (
                                per_joint_teacher_routing_exact
                                and native_authority_scale_routing_exact
                            ),
                            "posture_manipulation_authority_spans_exact": (
                                min(manipulation_teacher_coefficients)
                                == 0.0
                                and max(manipulation_teacher_coefficients)
                                == 1.0
                                and min(manipulation_residual_scales)
                                == base_scale_float32
                                and max(manipulation_residual_scales)
                                == post_scale_float32
                            ),
                            "posture_support_authority_spans_exact": (
                                min(posture_teacher_coefficients)
                                == posture_floor_float32
                                and max(posture_teacher_coefficients)
                                == 1.0
                                and min(posture_residual_scales)
                                == base_scale_float32
                                and max(posture_residual_scales)
                                == posture_scale_float32
                            ),
                            "posture_balance_route_fixed_exact": (
                                min(balance_teacher_coefficients) == 1.0
                                and max(balance_teacher_coefficients) == 1.0
                                and min(balance_residual_scales)
                                == base_scale_float32
                                and max(balance_residual_scales)
                                == base_scale_float32
                            ),
                            "posture_joint_motion_is_live_and_finite": (
                                posture_joint_motion is not None
                                and len(
                                    posture_joint_motion["joint_names"]
                                )
                                == 11
                                and np.isfinite(
                                    posture_joint_motion[
                                        "observed_range_rad"
                                    ]
                                ).all()
                                and posture_joint_motion[
                                    "maximum_observed_range_rad"
                                ]
                                > 1.0e-6
                            ),
                        }
                    )
                else:
                    checks.update(
                        {
                            "postfailure_native_authority_scale_routing_exact": (
                                native_authority_scale_routing_exact
                                and bool(support_residual_scales)
                                and bool(manipulation_residual_scales)
                            ),
                            "support_residual_authority_remains_fixed_low": (
                                min(support_residual_scales)
                                == base_scale_float32
                                and max(support_residual_scales)
                                == base_scale_float32
                            ),
                            "arm_residual_authority_spans_low_to_native": (
                                min(manipulation_residual_scales)
                                == base_scale_float32
                                and max(manipulation_residual_scales)
                                == post_scale_float32
                            ),
                        }
                    )
        if (
            postfailure_exposure_contract
            or supported_postdrop_exposure_contract
        ):
            grace_counts = drop_grace_proof["counts"]
            checks.update(
                {
                    "bounded_drop_grace_contract_exact": (
                        drop_grace_proof["enabled"] is True
                        and int(drop_grace_proof["drop_grace_steps"]) == 64
                        and int(grace_counts["evaluation_calls"])
                        == args.num_updates
                        * runner_dict["num_steps_per_env"]
                        and int(grace_counts["window_starts"]) > 0
                        and int(grace_counts["suppressed"]) > 0
                        and int(grace_counts["window_active"])
                        >= int(grace_counts["suppressed"])
                        and int(grace_counts["raw_true"])
                        == int(grace_counts["suppressed"])
                        + int(grace_counts["effective_true"])
                    ),
                    "bounded_drop_grace_changes_only_drop_function": (
                        drop_grace_proof["unchanged_non_drop_terms"] is True
                        and drop_grace_proof["drop_config_unchanged"] is True
                        and (
                            drop_grace_proof[
                                "termination_identities_before"
                            ]["dropped_after_lift"]["func"]
                            != drop_grace_proof[
                                "termination_identities_after"
                            ]["dropped_after_lift"]["func"]
                        )
                    ),
                }
            )
        elif residual_teacher_contract:
            checks["bounded_drop_grace_disabled_for_parent_contract"] = (
                drop_grace_proof["enabled"] is False
                and int(drop_grace_proof["drop_grace_steps"]) == 0
            )
        if (
            blockwise_teacher_contract
            or failure_latched_support_contract
            or supported_postdrop_exposure_contract
        ):
            checks.update(
                {
                    "blockwise_named_partition_complete_exact": (
                        teacher_partition_proof["scope"] == "arm_only"
                        and teacher_partition_proof[
                            "complete_disjoint_partition"
                        ]
                        is True
                        and len(
                            teacher_partition_proof[
                                "ordered_joint_names"
                            ]
                        )
                        == 29
                        and len(
                            teacher_partition_proof["support_indices"]
                        )
                        == 15
                        and len(
                            teacher_partition_proof[
                                "manipulation_indices"
                            ]
                        )
                        == 14
                    ),
                    "blockwise_support_one_arm_follows_release_exact": (
                        per_joint_teacher_routing_exact
                        and min(support_teacher_coefficients) == 1.0
                        and max(support_teacher_coefficients) == 1.0
                        and min(manipulation_teacher_coefficients) == 0.0
                        and max(manipulation_teacher_coefficients) == 1.0
                        and teacher_zero_control_steps > 0
                    ),
                }
            )
        if (
            blockwise_teacher_contract
            or supported_postdrop_exposure_contract
        ):
            checks.update(
                {
                    "advancing_support_mode_exact": (
                        teacher_partition_proof[
                            "support_teacher_mode"
                        ]
                        == "advancing"
                        and support_hold_proof["mode"] == "advancing"
                        and support_hold_trigger_count == 0
                        and support_hold_applied_env_steps == 0
                        and support_hold_routing_exact
                        and support_hold_arm_teacher_unchanged_exact
                    ),
                }
            )
        if failure_latched_support_contract:
            support_counts = support_hold_proof["counts"]
            checks.update(
                {
                    "failure_latched_support_routing_exact": (
                        teacher_partition_proof[
                            "support_teacher_mode"
                        ]
                        == "failure_latched"
                        and support_hold_proof["mode"]
                        == "failure_latched"
                        and support_hold_routing_exact
                        and support_hold_trigger_capture_exact
                        and support_hold_arm_teacher_unchanged_exact
                        and support_hold_trigger_count > 0
                        and support_hold_applied_env_steps > 0
                        and int(support_counts["triggers"])
                        == support_hold_trigger_count
                        and int(support_counts["applied_env_steps"])
                        == support_hold_applied_env_steps
                    ),
                    "failure_latched_support_reset_rearm_exact": (
                        support_hold_reset_rearm_exact
                        and int(support_counts["reset_rearms"]) > 0
                    ),
                }
            )
        checks["policy_credit_reward_mix_exact"] = (
            base_integrator.mix_cfg == reward_mix_cfg
        )
        if paper_cws_contract:
            checks.update(
                {
                    "paper_cws_scored_every_transition": (
                        paper_cws_final_audit is not None
                        and paper_cws_final_audit["steps_scored"]
                        == runner_dict["num_steps_per_env"]
                        * args.num_updates
                    ),
                    "paper_cws_reward_finite_nonzero": (
                        bool(paper_cws_reward_means)
                        and all(
                            np.isfinite(value)
                            for value in (
                                paper_cws_reward_means
                                + paper_cws_weighted_reward_means
                            )
                        )
                        and max(paper_cws_reward_means) > 0.0
                    ),
                    "paper_cws_weight_reconstructs_exactly": all(
                        abs(
                            weighted
                            - args.paper_cws_guidance_weight * raw
                        )
                        <= 1.0e-7
                        for raw, weighted in zip(
                            paper_cws_reward_means,
                            paper_cws_weighted_reward_means,
                            strict=True,
                        )
                    ),
                    "paper_cws_does_not_modify_original_icm_output": (
                        paper_cws_icm_bitwise_unchanged
                    ),
                    "paper_cws_privileged_fields_absent_from_actor_and_icm": (
                        paper_cws_final_audit[
                            "actor_receives_sdf_normals"
                        ]
                        is False
                        and paper_cws_final_audit[
                            "original_icm_receives_sdf_normals"
                        ]
                        is False
                    ),
                    "paper_cws_out_of_support_reward_strict_zero": (
                        paper_cws_final_audit[
                            "out_of_support_reward_policy"
                        ]
                        == "strict_zero"
                        and paper_cws_out_of_support_reward_abs_max == 0.0
                    ),
                    "paper_cws_invalid_transition_reward_strict_zero": (
                        paper_cws_final_audit[
                            "invalid_transition_reward_policy"
                        ]
                        == "strict_zero_and_excluded_from_contact_geometry"
                        and paper_cws_final_audit[
                            "invalid_environment_steps_masked"
                        ]
                        == paper_cws_invalid_transition_count
                        and paper_cws_invalid_transition_reward_abs_max == 0.0
                    ),
                }
            )
        else:
            checks["no_paper_cws_reward_added_to_control"] = (
                not paper_cws_reward_means
                and not paper_cws_weighted_reward_means
                and paper_cws_scorer is None
            )
        if demo_reward_contract:
            checks.update(
                {
                    "demo_predictor_bitwise_frozen_every_rollout": (
                        demo_final_audit is not None
                        and demo_final_audit["model_bitwise_frozen"] is True
                        and all(
                            record["metrics"][
                                "demo_predictor_bitwise_frozen"
                            ]
                            is True
                            and record["metrics"][
                                "demo_predictor_updated"
                            ]
                            is False
                            for record in integration_metrics_by_update
                        )
                    ),
                    "demo_reward_scored_every_transition": (
                        demo_final_audit["transitions_scored"]
                        == expected_samples_per_update * args.num_updates
                    ),
                    "demo_reward_nonzero_and_finite": (
                        bool(demo_reward_means)
                        and max(abs(value) for value in demo_reward_means)
                        > 0.0
                        and all(
                            np.isfinite(value)
                            for value in (
                                demo_reward_means
                                + demo_unit_reward_means
                            )
                        )
                    ),
                    "demo_eta_step_means_reconstruct": all(
                        np.isclose(
                            reward,
                            (
                                10.0
                                if (
                                    demo_authority_rework_contract
                                    or wrong_teacher_reward_conflict_contract
                                )
                                else 2.0
                            )
                            * unit,
                            rtol=1.0e-6,
                            atol=5.0e-7,
                        )
                        for reward, unit in zip(
                            demo_reward_means,
                            demo_unit_reward_means,
                            strict=True,
                        )
                    ),
                    "demo_does_not_modify_original_icm_output": (
                        demo_icm_bitwise_unchanged
                    ),
                    "demo_failure_mask_matches_tactile_regime": (
                        (
                            demo_final_audit["failure_boundaries"] == 0
                            and demo_final_audit[
                                "postfailure_zero_rewards"
                            ]
                            == 0
                        )
                        if explicit_zero_tactile_contract
                        else (
                            demo_final_audit["failure_boundaries"] > 0
                            and demo_final_audit[
                                "postfailure_zero_rewards"
                            ]
                            > 0
                        )
                    ),
                }
            )
        elif demo_event_reward_contract:
            event_eta = float(demo_event_reward_config["eta"])
            selected_event = demo_event_reward_config[
                "selected_demo_options"
            ][args.demo_event_selected_option]
            checks.update(
                {
                    "demo_event_predictor_frozen_every_rollout": (
                        demo_final_audit is not None
                        and demo_final_audit["model_frozen"] is True
                        and all(
                            record["metrics"]["demo_event_model_frozen"]
                            is True
                            and record["metrics"][
                                "demo_event_predictor_updated"
                            ]
                            is False
                            for record in integration_metrics_by_update
                        )
                    ),
                    "demo_event_scored_every_transition": (
                        demo_final_audit["transitions_scored"]
                        == expected_samples_per_update * args.num_updates
                    ),
                    "demo_event_reward_nonzero_and_finite": (
                        bool(demo_reward_means)
                        and max(abs(value) for value in demo_reward_means) > 0.0
                        and all(
                            np.isfinite(value)
                            for value in (
                                demo_reward_means
                                + demo_unit_reward_means
                                + demo_event_potential_means
                                + demo_event_risk_means
                                + demo_event_uncertainty_means
                                + demo_event_ready_fractions
                                + demo_event_phase_means
                            )
                        )
                    ),
                    "demo_event_eta_step_means_reconstruct": all(
                        np.isclose(
                            reward,
                            event_eta * unit,
                            rtol=1.0e-6,
                            atol=5.0e-7,
                        )
                        for reward, unit in zip(
                            demo_reward_means,
                            demo_unit_reward_means,
                            strict=True,
                        )
                    ),
                    "demo_event_does_not_modify_original_icm_output": (
                        demo_icm_bitwise_unchanged
                    ),
                    "demo_event_phase_and_prefix_are_causal": (
                        demo_final_audit["phase_source"]
                        == "reset_bounded_causal_control_clock"
                        and demo_final_audit["future_actual_events_used"]
                        is False
                        and demo_final_audit["history_steps"] == 10
                        and all(
                            0.0 <= value <= 1.0
                            for value in demo_event_phase_means
                        )
                        and all(
                            0.0 <= value <= 1.0
                            for value in demo_event_ready_fractions
                        )
                    ),
                    "demo_event_selected_demo_exact": (
                        demo_final_audit["selected_option"]
                        == args.demo_event_selected_option
                        and demo_final_audit["selected_task"]
                        == selected_event["selected_task"]
                        and int(demo_final_audit["selected_motion_id"])
                        == int(selected_event["selected_motion_id"])
                    ),
                }
            )
        else:
            checks["no_demo_reward_added_to_control"] = (
                not demo_reward_means
                and not demo_unit_reward_means
            )
            if demo_reward_telemetry_contract:
                checks[
                    "read_only_demo_predictor_loaded_and_frozen_without_reward"
                ] = (
                    demo_scorer is not None
                    and demo_final_audit is not None
                    and demo_final_audit["model_bitwise_frozen"] is True
                    and demo_final_audit["transitions_scored"] == 0
                    and demo_final_audit["failure_boundaries"] == 0
                    and demo_final_audit["postfailure_zero_rewards"] == 0
                )
            else:
                checks["no_demo_predictor_runtime_in_control"] = (
                    demo_scorer is None
                )
        zero_preserving_contract = (
            args.policy_contract
            in {
                "sugar_native_zero_preserving_tactile_floor_lr",
                "sugar_native_zero_preserving_tactile_fixed_low_lr",
            }
        )
        fixed_low_lr_contract = (
            args.policy_contract
            == "sugar_native_zero_preserving_tactile_fixed_low_lr"
        )
        if zero_preserving_contract:
            checks.update(
                {
                    "zero_tactile_causal_invariant_initial": (
                        initial_zero_tactile_causal_audit is not None
                        and bool(
                            initial_zero_tactile_causal_audit["passed"]
                        )
                    ),
                    "zero_tactile_causal_invariant_after_every_update": all(
                        bool(
                            record["metrics"][
                                "zero_tactile_causal_audit"
                            ]["passed"]
                        )
                        for record in policy_update_metrics_by_update
                    ),
                    "zero_tactile_causal_invariant_after_every_reload": all(
                        record["zero_tactile_causal_audit"] is not None
                        and bool(
                            record["zero_tactile_causal_audit"]["passed"]
                        )
                        for record in checkpoint_reload_records.values()
                    ),
                }
            )
        if (
            args.policy_contract
            in {
                "sugar_native_tactile_floor_lr",
                "sugar_native_zero_preserving_tactile_floor_lr",
                "sugar_native_zero_preserving_tactile_fixed_low_lr",
            }
            and args.num_updates > 1
        ):
            tactile_rollout_check = (
                "explicit_zero_each_rollout_has_exact_zero_tactile"
                if explicit_zero_tactile_contract
                else (
                    "h2r1_each_rollout_has_transformed_spatial_tactile"
                    if h2_contract
                    else (
                        "zf1_each_rollout_has_direct_tacsl"
                        if fixed_low_lr_contract
                        else (
                            "zlr1_each_rollout_has_direct_tacsl"
                            if zero_preserving_contract
                            else "hlr1_each_rollout_has_direct_tacsl"
                        )
                    )
                )
            )
            checks[tactile_rollout_check] = all(
                (
                    count == 0
                    if explicit_zero_tactile_contract
                    else count > 0
                )
                for count in tactile_nonzero_values_by_update
            )
        if args.num_updates > 1:
            if strict_mimickit:
                endpoint_check_name = "h1_fixed_endpoint_is_eight_updates"
                numerical_check_name = "h1_numerical_stability_pass"
            elif goal_recovery_contract:
                endpoint_check_name = (
                    (
                        "teacher_floor_overfit_advances_exactly_64_updates"
                        if teacher_floor_resume_contract
                        else (
                            "fixed_teacher_continuation_advances_exactly_64_updates"
                            if fixed_teacher_interval_resume_contract
                            else "wrong_teacher_reward_conflict_endpoint_is_64_updates"
                        )
                    )
                    if wrong_teacher_reward_conflict_64_contract
                    else (
                    "posture_capacity_fixed_endpoint_is_two_updates"
                    if posture_capacity_contract
                    else (
                        "posture_formal_exposes_at_least_one_million_slots"
                        if posture_formal_contract
                        else (
                            "goal_recovery_native_authority_fixed_endpoint_is_512_updates"
                            if native_authority_formal_contract
                            else (
                                "goal_recovery_fixed_endpoint_is_256_updates"
                                if goal_recovery_formal_contract
                                else "goal_recovery_smoke_is_eight_updates"
                            )
                        )
                    )
                    )
                )
                numerical_check_name = (
                    "goal_recovery_multiphysics_numerical_stability_pass"
                )
            elif residual_long_contract:
                endpoint_check_name = (
                    "residual_h2r1_fixed_endpoint_is_64_updates"
                )
                numerical_check_name = (
                    "residual_h2r1_64_update_numerical_stability_pass"
                )
            elif h2_contract:
                endpoint_check_name = "h2r1_fixed_endpoint_is_eight_updates"
                numerical_check_name = "h2r1_numerical_stability_pass"
            elif fixed_low_lr_contract:
                endpoint_check_name = "zf1_fixed_endpoint_is_eight_updates"
                numerical_check_name = "zf1_numerical_stability_pass"
            elif zero_preserving_contract:
                endpoint_check_name = "zlr1_fixed_endpoint_is_eight_updates"
                numerical_check_name = "zlr1_numerical_stability_pass"
            elif args.policy_contract == "sugar_native_tactile_floor_lr":
                endpoint_check_name = "hlr1_fixed_endpoint_is_eight_updates"
                numerical_check_name = "hlr1_numerical_stability_pass"
            else:
                endpoint_check_name = "hn1_fixed_endpoint_is_eight_updates"
                numerical_check_name = "hn1_numerical_stability_pass"
            checks[endpoint_check_name] = (
                (
                    (
                        args.num_updates == resume_update + 64
                        and checkpoint_updates == {args.num_updates}
                        and len(policy_update_metrics_by_update) == 64
                    )
                    if (
                        fixed_teacher_interval_resume_contract
                        or teacher_floor_resume_contract
                    )
                    else (
                        args.num_updates == 64
                        and checkpoint_updates
                        == (
                            {32, 64}
                            if phase_event_protocol_contract
                            else {1, 64}
                        )
                        and len(policy_update_metrics_by_update) == 64
                    )
                )
                if wrong_teacher_reward_conflict_64_contract
                else (
                (
                    args.num_updates == 2
                    and checkpoint_updates == {1, 2}
                    and len(policy_update_metrics_by_update) == 2
                )
                if posture_capacity_contract
                else (
                (
                    args.num_updates == posture_formal_num_updates
                    and checkpoint_updates
                    == {
                        1,
                        (posture_formal_num_updates + 3) // 4,
                        (posture_formal_num_updates + 1) // 2,
                        posture_formal_num_updates,
                    }
                    and len(policy_update_metrics_by_update)
                    == posture_formal_num_updates
                    and (
                        args.num_envs
                        * runner_dict["num_steps_per_env"]
                        * args.num_updates
                    )
                    >= 1_000_000
                )
                if posture_formal_contract
                else (
                (
                    args.num_updates == 512
                    and checkpoint_updates == {1, 128, 512}
                    and len(policy_update_metrics_by_update) == 512
                )
                if native_authority_formal_contract
                else (
                (
                    args.num_updates == 256
                    and checkpoint_updates == {1, 64, 256}
                    and len(policy_update_metrics_by_update) == 256
                )
                if goal_recovery_formal_contract
                else (
                (
                    args.num_updates == 64
                    and checkpoint_updates == {1, 16, 64}
                    and len(policy_update_metrics_by_update) == 64
                )
                if residual_long_contract
                else (
                    args.num_updates == 8
                    and checkpoint_updates == {1, 4, 8}
                    and len(policy_update_metrics_by_update) == 8
                )
                )
                )
                )
                )
                )
            )
            checks[numerical_check_name] = h1_numerical_stability
        if strict_mimickit:
            protocol = (
                "sugar_stage_h_smp_original_icm_policy_integration_v1"
                if args.num_updates == 1
                else "sugar_stage_h_h1_multistep_stability_v1"
            )
        elif goal_recovery_contract:
            if phase_event_protocol_contract:
                protocol = "sugar_phase_event_reward_matched_policy_v1"
            elif fixed_teacher_demo_identity_contract:
                protocol = "sugar_plan11_fixed_teacher_demo_identity_v2"
            elif teacher_floor_overfit_contract:
                protocol = "sugar_plan11_teacher_floor_overfit_v1"
            elif wrong_teacher_reward_conflict_contract:
                protocol = "sugar_plan11_wrong_teacher_reward_conflict_v1"
            elif reference_waypoint_foundation_contract:
                protocol = reference_waypoint_foundation_config["protocol"]
            elif posture_capacity_contract:
                protocol = (
                    "sugar_stage_i_posture_adaptive_causal_bootstrap_"
                    "capacity_2update_v2"
                    if args.causal_contact_bootstrap_v2
                    else "sugar_stage_i_posture_adaptive_capacity_2update_v1"
                )
            elif posture_formal_contract:
                protocol = (
                    (
                        "sugar_stage_i_posture_adaptive_causal_bootstrap_"
                        "demo_reward_formal_v2"
                        if demo_reward_contract
                        else "sugar_stage_i_posture_adaptive_"
                        "causal_bootstrap_formal_v2"
                    )
                    if args.causal_contact_bootstrap_v2
                    else (
                        "sugar_stage_i_posture_adaptive_demo_reward_formal_v1"
                        if demo_reward_contract
                        else "sugar_stage_i_posture_adaptive_formal_v1"
                    )
                )
            else:
                protocol = (
                (
                    (
                        "sugar_stage_i_goal_recovery_native_authority_"
                        + (
                            "demo_reward_formal_512_deterministic_v2"
                            if args.strict_deterministic_torch
                            else "demo_reward_formal_512_v1"
                        )
                    )
                    if demo_reward_contract
                    else (
                        "sugar_stage_i_goal_recovery_native_authority_"
                        + (
                            "formal_512_deterministic_v2"
                            if args.strict_deterministic_torch
                            else "formal_512_v1"
                        )
                    )
                )
                if native_authority_formal_contract
                else (
                    (
                        "sugar_stage_i_goal_recovery_native_authority_"
                        + (
                            "demo_reward_smoke_deterministic_v2"
                            if args.strict_deterministic_torch
                            else "demo_reward_smoke_v1"
                        )
                    )
                    if (
                        native_authority_smoke_contract
                        and demo_reward_contract
                    )
                    else (
                        "sugar_stage_i_goal_recovery_native_authority_"
                        + (
                            "smoke_deterministic_v2"
                            if args.strict_deterministic_torch
                            else "smoke_v1"
                        )
                    )
                    if native_authority_smoke_contract
                    else (
                        "sugar_stage_i_goal_recovery_multiphysics_formal_256_v1"
                        if goal_recovery_formal_contract
                        else "sugar_stage_i_goal_recovery_multiphysics_smoke_v1"
                    )
                )
                )
        elif residual_long_contract:
            protocol = (
                "sugar_stage_i_supported_postdrop_exposure_64_update_v1"
                if supported_postdrop_exposure_contract
                else (
                "sugar_stage_i_failure_latched_support_hold_64_update_v1"
                if failure_latched_support_contract
                else (
                "sugar_stage_i_blockwise_teacher_authority_64_update_v1"
                if blockwise_teacher_contract
                else (
                "sugar_stage_i_official_refiner_residual_h2r1_"
                "64_update_postfailure_exposure_v1"
                if postfailure_exposure_contract
                else (
                "sugar_stage_i_official_refiner_residual_h2r1_"
                "64_update_discovery_v1"
                if args.reward_control == "full"
                else (
                    "sugar_stage_i_official_refiner_residual_h2r1_"
                    "64_update_policy_credit_control_v1"
                )
                )
                )
                )
                )
            )
        elif h2_contract and residual_teacher_contract:
            protocol = (
                "sugar_stage_i_official_refiner_residual_h2r1_"
                "eight_update_v1"
            )
        elif h2_contract:
            protocol = (
                "sugar_stage_h_h2r1_five_role_fixed_low_lr_stability_v1"
            )
        elif fixed_low_lr_contract:
            protocol = (
                "sugar_stage_h_zf0_zero_preserving_fixed_low_lr_"
                "integration_v1"
                if args.num_updates == 1
                else (
                    "sugar_stage_h_zf1_zero_preserving_fixed_low_lr_"
                    "stability_v1"
                )
            )
            if args.num_updates == 1:
                checks["zf0_numerical_stability_pass"] = (
                    h1_numerical_stability
                )
        elif zero_preserving_contract:
            protocol = (
                "sugar_stage_h_zlr0_zero_preserving_floor_lr_integration_v1"
                if args.num_updates == 1
                else (
                    "sugar_stage_h_zlr1_zero_preserving_floor_lr_"
                    "stability_v1"
                )
            )
            if args.num_updates == 1:
                checks["zlr0_numerical_stability_pass"] = (
                    h1_numerical_stability
                )
        elif args.policy_contract == "sugar_native_tactile_floor_lr":
            protocol = (
                "sugar_stage_h_hlr0_tactile_floor_lr_integration_v1"
                if args.num_updates == 1
                else "sugar_stage_h_hlr1_tactile_floor_lr_stability_v1"
            )
            # Unlike HN0, the isolation run explicitly applies every locked
            # numerical gate to its single update.
            if args.num_updates == 1:
                checks["hlr0_numerical_stability_pass"] = (
                    h1_numerical_stability
                )
        else:
            protocol = (
                "sugar_stage_h_hn0_sugar_native_integration_v1"
                if args.num_updates == 1
                else "sugar_stage_h_hn1_sugar_native_stability_v1"
            )
        source_paths = [
            Path(__file__).resolve(),
            WORKSPACE_ROOT
            / "SUGAR/source/sugar_rl/sugar_rl/utils/smp_icm_reward_integration.py",
            WORKSPACE_ROOT
            / "SUGAR/source/sugar_rl/sugar_rl/utils/official_smp_scorer.py",
            WORKSPACE_ROOT
            / "SUGAR/source/sugar_rl/sugar_rl/utils/original_icm_continuous.py",
            WORKSPACE_ROOT
            / "SUGAR/source/sugar_rl/sugar_rl/utils/original_icm_trainer.py",
            WORKSPACE_ROOT
            / (
                "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/"
                "direct_tactile_history.py"
            ),
            WORKSPACE_ROOT
            / (
                "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/"
                "g129dof/train_refiner/carry_box_smp_icm_goal_env_cfg.py"
            ),
            WORKSPACE_ROOT / "MimicKit/mimickit/learning/smp_agent.py",
        ]
        if goal_recovery_contract:
            source_paths.extend(
                [
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/"
                        "latent_contact_dynamics_events.py"
                    ),
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/"
                        "latent_contact_visuotactile_sensor.py"
                    ),
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/"
                        "direct_tactile_slip_spatiotemporal.py"
                    ),
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/"
                        "goal_tactile_strategy.py"
                    ),
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/"
                        "g129dof/train_refiner/"
                        "carry_box_smp_icm_goal_coherent_env_cfg.py"
                    ),
                ]
            )
        if wrong_teacher_reward_conflict_contract:
            source_paths.append(
                WORKSPACE_ROOT
                / "SUGAR/source/sugar_rl/sugar_rl/utils/"
                "wrong_demo_teacher_anneal.py"
            )
        if h2_contract:
            source_paths.extend(
                [
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/"
                        "direct_tactile_stress.py"
                    ),
                    WORKSPACE_ROOT
                    / (
                        "DOCS/"
                        "sugar_stage_h_h2r1_five_role_fixed_low_lr_protocol_"
                        "20260723.md"
                    ),
                    WORKSPACE_ROOT
                    / (
                        "scripts/sugar/smp/"
                        "audit_direct_tactile_h2_stress_runtime.py"
                    ),
                ]
            )
        if residual_teacher_contract:
            residual_protocol_path = (
                WORKSPACE_ROOT
                / "DOCS/sugar_reference_waypoint_foundation_protocol_20260727.md"
                if reference_waypoint_foundation_contract
                else (
                WORKSPACE_ROOT
                / (
                    "DOCS/sugar_supported_postdrop_exposure_protocol_"
                    "20260725.md"
                )
                if supported_postdrop_exposure_contract
                else (
                WORKSPACE_ROOT
                / (
                    "DOCS/sugar_failure_latched_support_hold_protocol_"
                    "20260725.md"
                )
                if failure_latched_support_contract
                else (
                WORKSPACE_ROOT
                / (
                    "DOCS/sugar_blockwise_teacher_authority_protocol_"
                    "20260725.md"
                )
                if blockwise_teacher_contract
                else (
                    WORKSPACE_ROOT
                    / (
                        "DOCS/sugar_official_refiner_residual_"
                        "postfailure_exposure_protocol_20260725.md"
                    )
                    if postfailure_exposure_contract
                    else (
                        WORKSPACE_ROOT
                        / (
                            "DOCS/sugar_official_refiner_residual_h2r1_64_"
                            "update_protocol_20260725.md"
                        )
                        if args.reward_control == "full"
                        else (
                            WORKSPACE_ROOT
                            / (
                                "DOCS/sugar_official_refiner_residual_"
                                "policy_credit_ablation_protocol_20260725.md"
                            )
                        )
                    )
                )
                )
                )
                )
            )
            residual_config_path = (
                reference_waypoint_foundation_config_path
                if reference_waypoint_foundation_contract
                else (
                WORKSPACE_ROOT
                / (
                    "scripts/sugar/smp/config/"
                    "stage_i_supported_postdrop_exposure_64_update_v1.json"
                )
                if supported_postdrop_exposure_contract
                else (
                WORKSPACE_ROOT
                / (
                    "scripts/sugar/smp/config/"
                    "stage_i_failure_latched_support_hold_64_update_v1.json"
                )
                if failure_latched_support_contract
                else (
                WORKSPACE_ROOT
                / (
                    "scripts/sugar/smp/config/"
                    "stage_i_blockwise_teacher_authority_64_update_v1.json"
                )
                if blockwise_teacher_contract
                else (
                    WORKSPACE_ROOT
                    / (
                        "scripts/sugar/smp/config/stage_i_official_refiner_"
                        "residual_h2r1_64_update_postfailure_exposure_v1.json"
                    )
                    if postfailure_exposure_contract
                    else (
                        WORKSPACE_ROOT
                        / (
                            "scripts/sugar/smp/config/stage_i_official_"
                            "refiner_residual_h2r1_64_update_v1.json"
                        )
                        if args.reward_control == "full"
                        else (
                            WORKSPACE_ROOT
                            / (
                                "scripts/sugar/smp/config/stage_i_official_"
                                "refiner_residual_h2r1_64_update_"
                                f"{args.reward_control}_v1.json"
                            )
                        )
                    )
                )
                )
                )
                )
            )
            source_paths.extend(
                [
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/utils/"
                        "official_refiner_nominal_teacher.py"
                    ),
                    (
                        (
                            WORKSPACE_ROOT
                            / (
                                "DOCS/"
                                "sugar_posture_adaptive_authority_protocol_"
                                "20260726.md"
                            )
                        )
                        if posture_adaptive_contract
                        else (
                        residual_protocol_path
                        if reference_waypoint_foundation_contract
                        else (
                        (
                            WORKSPACE_ROOT
                            / (
                                "DOCS/sugar_goal_recovery_native_authority_"
                                "protocol_20260725.md"
                            )
                        )
                        if native_authority_contract
                        else (
                            WORKSPACE_ROOT
                            / (
                                "DOCS/sugar_goal_recovery_multiphysics_redo_"
                                "protocol_20260725.md"
                            )
                        )
                        if goal_recovery_contract
                        else (
                            residual_protocol_path
                            if residual_long_contract
                            else (
                                WORKSPACE_ROOT
                                / (
                                    "DOCS/sugar_official_refiner_residual_"
                                    "h2r1_eight_update_protocol_20260725.md"
                                )
                            )
                        )
                        )
                        )
                    ),
                    (
                        (
                            protocol_config_path
                        )
                        if posture_adaptive_contract
                        else (
                        residual_config_path
                        if reference_waypoint_foundation_contract
                        else (
                        (
                            WORKSPACE_ROOT
                            / (
                                "scripts/sugar/smp/config/"
                                "stage_i_goal_recovery_native_authority_"
                                f"{'formal_512' if native_authority_formal_contract else 'smoke'}_v1.json"
                            )
                        )
                        if native_authority_contract
                        else (
                            WORKSPACE_ROOT
                            / (
                                "scripts/sugar/smp/config/"
                                "stage_i_goal_recovery_multiphysics_"
                                f"{'formal_256' if goal_recovery_formal_contract else 'smoke'}_v1.json"
                            )
                        )
                        if goal_recovery_contract
                        else (
                            residual_config_path
                            if residual_long_contract
                            else (
                                WORKSPACE_ROOT
                                / (
                                    "scripts/sugar/smp/config/stage_i_"
                                    "official_refiner_residual_h2r1_"
                                    "eight_update_v1.json"
                                )
                            )
                        )
                        )
                        )
                    ),
                ]
            )
            if reference_waypoint_foundation_contract:
                source_paths.append(
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/utils/"
                        "reference_waypoint_foundation.py"
                    )
                )
            if paper_cws_contract:
                source_paths.extend(
                    [
                        WORKSPACE_ROOT
                        / (
                            "SUGAR/source/sugar_rl/sugar_rl/utils/"
                            "paper_contact_wrench_support.py"
                        ),
                        WORKSPACE_ROOT
                        / (
                            "SUGAR/source/sugar_rl/sugar_rl/utils/"
                            "tacsl_paper_cws_adapter.py"
                        ),
                        WORKSPACE_ROOT
                        / (
                            "SUGAR/source/sugar_rl/sugar_rl/utils/"
                            "paper_cws_runtime_reward.py"
                        ),
                        WORKSPACE_ROOT
                        / (
                            "SUGAR/source/sugar_rl/sugar_rl/utils/"
                            "paper_cws_rollout_integration.py"
                        ),
                        paper_cws_runtime_config_path,
                    ]
                )
            if posture_adaptive_contract:
                source_paths.append(
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/utils/"
                        "posture_adaptive_refiner_teacher.py"
                    )
                )
        if strict_mimickit:
            source_paths.extend(
                [
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/utils/"
                        "official_smp_policy_optimizer.py"
                    ),
                    WORKSPACE_ROOT / "MimicKit/mimickit/learning/ppo_agent.py",
                ]
            )
        else:
            source_paths.extend(
                [
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/utils/"
                        "sugar_native_curiosity_ppo.py"
                    ),
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/"
                        "agents/rsl_rl_ppo_cfg.py"
                    ),
                ]
            )
        if demo_reward_contract or demo_reward_telemetry_contract:
            source_paths.extend(
                [
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/utils/"
                        "demo_reward_runtime.py"
                    ),
                    WORKSPACE_ROOT
                    / (
                        "SUGAR/source/sugar_rl/sugar_rl/utils/"
                        "demo_reward_potential.py"
                    ),
                    legacy_demo_reward_config_path,
                    WORKSPACE_ROOT
                    / (
                        "scripts/sugar/demo_reward/"
                        "demo_conditioned_causal_predictor_v1.py"
                    ),
                ]
            )
        if protocol_config_path is not None:
            source_paths.append(protocol_config_path)
        if args.causal_contact_bootstrap_v2:
            source_paths.append(
                WORKSPACE_ROOT
                / (
                    "experiments/sugar_smp_exploration/stage_i/"
                    "live_official_refiner_teacher_seed4263_v2/"
                    "LIVE_OFFICIAL_REFINER_TEACHER_TRACE.npz"
                )
            )
        payload = {
            "protocol": protocol,
            "passed": all(checks.values()),
            "claim_scope": (
                (
                    "Predeclared Plan-11 four-arm demo-conflict training "
                    "control with exact-zero actor, ICM, slip, failed-"
                    "strategy, and demo-predictor tactile inputs. The frozen "
                    "official Refiner, serious SUGAR-native PPO actor, "
                    "official SMP, original ICM, and 11.9M demo predictor are "
                    "unchanged. No tactile effectiveness or post-tactile-"
                    "failure behavior is claimed; matched frozen evaluation "
                    "and synchronized video remain required."
                )
                if explicit_zero_tactile_contract
                else (
                (
                    (
                        "Result-blind two-update posture-adaptive capacity "
                        "preflight. Only runtime, TacSL, frozen-model, "
                        "checkpoint, throughput, memory, and utilization "
                        "gates may select capacity; no behavior claim follows."
                    )
                    if posture_capacity_contract
                    else (
                    (
                        "Predeclared posture-adaptive >=1M-transition "
                        "from-scratch training endpoint using the unchanged "
                        "serious SUGAR policy, SMP, original ICM, TacSL, and "
                        "optional frozen demo potential. This is training "
                        "evidence only; behavior requires the fixed physics "
                        "grid, new seeds, and synchronized rendering."
                    )
                    if posture_formal_contract
                    else (
                    (
                        "Predeclared 512-update from-scratch goal-recovery "
                        "training endpoint with task credit 10 and causal "
                        "post-failure native arm authority. This is training "
                        "evidence only; recovery, physics-conditioned strategy, "
                        "and scientific success require fresh deterministic "
                        "multiphysics evaluation and synchronized visual proof."
                    )
                    if native_authority_formal_contract
                    else (
                        (
                            "Predeclared 256-update from-scratch goal-recovery "
                            "multiphysics training endpoint. This is training "
                            "evidence only; recovery, physics-conditioned "
                            "strategy, and scientific success require fresh "
                            "deterministic multiphysics evaluation and "
                            "synchronized visual proof."
                        )
                        if goal_recovery_formal_contract
                        else (
                            "Eight-update corrected-runtime diagnostic for the "
                            "goal-recovery contract; not formal training, "
                            "learned behavior, recovery, or strategy evidence."
                        )
                    )
                    )
                    )
                )
                if goal_recovery_contract
                else (
                (
                    (
                        "64-update named-joint lower-body support plus bounded "
                        "post-drop exposure with unchanged original ICM and "
                        "reward mix; exposure is trained, but discovery, "
                        "recovery, and strategy evidence require separate "
                        "no-grace frozen evaluation"
                    )
                    if supported_postdrop_exposure_contract
                    else (
                    (
                        "64-update failure-latched 15-D support hold with "
                        "unchanged 14-D arm release; original ICM and reward "
                        "mix remain exact, and discovery/recovery require "
                        "separate frozen evaluation"
                    )
                    if failure_latched_support_contract
                    else (
                    (
                        "64-update named-joint arm-only teacher release with "
                        "retained leg/waist support; original ICM and reward "
                        "mix remain exact, and discovery/recovery require "
                        "separate frozen evaluation"
                    )
                    if blockwise_teacher_contract
                    else (
                    (
                        "64-update recurring post-drop exposure run with "
                        "unchanged original ICM and reward mix; exposure is "
                        "trained, but discovery, recovery, and strategy "
                        "evidence require separate no-grace frozen evaluation"
                    )
                    if postfailure_exposure_contract
                    else (
                        "64-update bounded residual-policy training run with "
                        "real SUGAR/TacSL/SMP/ICM; training is performed, but "
                        "stable behavior, recovery, strategy discovery, and "
                        "outcome evidence require separate frozen evaluation"
                    )
                    )
                    )
                    )
                )
                if residual_long_contract
                else (
                    f"{args.num_updates}-update real SUGAR/TacSL/SMP/ICM "
                    "integration/stability diagnostic; not formal training, "
                    "learned behavior, strategy discovery, or outcome evidence"
                )
                )
                )
            ),
            "semantic_boundary": (
                "ICM is independently learned pre-update action-conditioned "
                "forward prediction error in inverse-model features. Task "
                "results do not define or gate it. PPO is only the numerical "
                "policy optimizer consuming the separate signal."
            ),
            "host": socket.gethostname(),
            "device": str(base_env.device),
            "task": task_id,
            "training_objective": args.training_objective,
            "policy_contract": args.policy_contract,
            "teacher_wrapper_mode": args.teacher_wrapper_mode,
            "tactile_regime": args.tactile_regime,
            "tactile_mount_environment": (
                observed_tacsl_mount_environment
            ),
            "reward_control": args.reward_control,
            "nominal_teacher_residual": (
                {
                    "teacher_checkpoint": str(teacher_checkpoint),
                    "teacher_checkpoint_sha256": _sha256(
                        teacher_checkpoint
                    ),
                    "residual_scale": args.residual_scale,
                    "post_release_residual_scale": (
                        args.post_release_residual_scale
                    ),
                    "release_mode": args.teacher_release_mode,
                    "scheduled_final_coefficient": (
                        args.teacher_final_coefficient
                    ),
                    "reference_advance_mode": (
                        args.teacher_reference_advance_mode
                    ),
                    "linear_release_steps": (
                        args.teacher_linear_release_steps
                    ),
                    "release_scope": args.teacher_release_scope,
                    "support_teacher_mode": args.support_teacher_mode,
                    "posture_pre_failure_residual_scale": (
                        args.posture_pre_failure_residual_scale
                    ),
                    "posture_post_failure_residual_scale": (
                        args.posture_post_failure_residual_scale
                    ),
                    "posture_post_failure_teacher_floor": (
                        args.posture_post_failure_teacher_floor
                    ),
                    "drop_grace_steps": args.drop_grace_steps,
                    "zero_initialization": residual_zero_initialization,
                    "coefficient_min": min(teacher_coefficients),
                    "coefficient_max": max(teacher_coefficients),
                    "failure_release_events": teacher_release_events,
                    "zero_teacher_control_steps": (
                        teacher_zero_control_steps
                    ),
                    "support_residual_scale_min": (
                        min(support_residual_scales)
                        if support_residual_scales
                        else None
                    ),
                    "support_residual_scale_max": (
                        max(support_residual_scales)
                        if support_residual_scales
                        else None
                    ),
                    "manipulation_residual_scale_min": (
                        min(manipulation_residual_scales)
                        if manipulation_residual_scales
                        else None
                    ),
                    "manipulation_residual_scale_max": (
                        max(manipulation_residual_scales)
                        if manipulation_residual_scales
                        else None
                    ),
                    "posture_teacher_coefficient_min": (
                        min(posture_teacher_coefficients)
                        if posture_teacher_coefficients
                        else None
                    ),
                    "posture_teacher_coefficient_max": (
                        max(posture_teacher_coefficients)
                        if posture_teacher_coefficients
                        else None
                    ),
                    "balance_teacher_coefficient_min": (
                        min(balance_teacher_coefficients)
                        if balance_teacher_coefficients
                        else None
                    ),
                    "balance_teacher_coefficient_max": (
                        max(balance_teacher_coefficients)
                        if balance_teacher_coefficients
                        else None
                    ),
                    "posture_residual_scale_min": (
                        min(posture_residual_scales)
                        if posture_residual_scales
                        else None
                    ),
                    "posture_residual_scale_max": (
                        max(posture_residual_scales)
                        if posture_residual_scales
                        else None
                    ),
                    "balance_residual_scale_min": (
                        min(balance_residual_scales)
                        if balance_residual_scales
                        else None
                    ),
                    "balance_residual_scale_max": (
                        max(balance_residual_scales)
                        if balance_residual_scales
                        else None
                    ),
                    "teacher_state_sha256_before": teacher_hash_before,
                    "teacher_state_sha256_after": teacher_hash_after,
                }
                if residual_teacher_contract
                else None
            ),
            "teacher_joint_partition": teacher_partition_proof,
            "teacher_reference_advance": (
                teacher_reference_advance_proof
            ),
            "posture_joint_motion": posture_joint_motion,
            "support_hold": support_hold_proof,
            "bounded_drop_grace": drop_grace_proof,
            "coherent_latent_dynamics": latent_dynamics_proof,
            "reference_waypoint_foundation": (
                reference_waypoint_foundation_proof
            ),
            "tactile_stress_runtime": tactile_stress_proof,
            "explicit_zero_tactile_control": (
                {
                    "strategy_runtime": explicit_zero_runtime_audit,
                    "official_teacher_observation_shape": (
                        explicit_zero_teacher_observation_shape
                    ),
                    "first_teacher_action_vs_source_l2": (
                        explicit_zero_teacher_action_l2
                    ),
                    "first_teacher_action_vs_source_max_abs": (
                        explicit_zero_teacher_action_max_abs
                    ),
                    "canonical_env0_action_vs_source_max_abs": (
                        explicit_zero_teacher_action_canonical_max_abs
                    ),
                    "action_vs_source_max_abs_by_env": (
                        explicit_zero_teacher_action_max_abs_by_env
                    ),
                    "tactile_arrays_loaded": contact_seed.get(
                        "tactile_arrays_loaded"
                    ),
                    "tactile_sensor_data_read": contact_seed.get(
                        "tactile_sensor_data_read"
                    ),
                    "failure_mask_policy": (
                        "remains_inactive_because_no_tactile_failure_"
                        "signal_is_substituted"
                    ),
                }
                if explicit_zero_tactile_contract
                else None
            ),
            "policy_observation_terms": policy_observation_terms,
            "icm_observation_terms": icm_observation_terms,
            "initial_zero_tactile_causal_audit": (
                initial_zero_tactile_causal_audit
            ),
            "seed": args.seed,
            "action_seed": args.action_seed,
            "torch_determinism": {
                "strict_requested": args.strict_deterministic_torch,
                "deterministic_algorithms_enabled": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cuda_matmul_allow_tf32": (
                    torch.backends.cuda.matmul.allow_tf32
                ),
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
            },
            "protocol_config": (
                {
                    "path": str(protocol_config_path),
                    "sha256": _sha256(protocol_config_path),
                    "arm": (
                        args.protocol_arm
                        if args.protocol_arm is not None
                        else (
                            "demo_eta2"
                            if demo_reward_contract
                            else "no_demo"
                        )
                    ),
                }
                if protocol_config_path is not None
                else None
            ),
            "reference_waypoint_foundation_config": (
                {
                    "path": str(
                        reference_waypoint_foundation_config_path
                    ),
                    "sha256": _sha256(
                        reference_waypoint_foundation_config_path
                    ),
                    "w0_pair_audit": (
                        reference_waypoint_foundation_config[
                            "w0_pair_audit"
                        ]
                    ),
                }
                if reference_waypoint_foundation_config_path is not None
                else None
            ),
            "paper_cws_runtime": (
                {
                    "runtime_config": str(
                        paper_cws_runtime_config_path
                    ),
                    "runtime_config_sha256": _sha256(
                        paper_cws_runtime_config_path
                    ),
                    "guidance_weight": (
                        args.paper_cws_guidance_weight
                    ),
                    "final_audit": paper_cws_final_audit,
                    "missed_contact_count_by_step": (
                        paper_cws_missed_contact_counts
                    ),
                    "unintended_contact_count_by_step": (
                        paper_cws_unintended_contact_counts
                    ),
                    "clamped_reference_count_by_step": (
                        paper_cws_clamped_reference_counts
                    ),
                    "out_of_support_reward_abs_max": (
                        paper_cws_out_of_support_reward_abs_max
                    ),
                    "invalid_transition_count": (
                        paper_cws_invalid_transition_count
                    ),
                    "invalid_transition_reward_abs_max": (
                        paper_cws_invalid_transition_reward_abs_max
                    ),
                }
                if paper_cws_contract
                else None
            ),
            "num_envs": args.num_envs,
            "num_updates": args.num_updates,
            "resume_update": resume_update,
            "updates_executed_this_process": updates_executed,
            "resume": (
                {
                    "restore": resume_restore_record,
                    "temporal_boundary": resume_temporal_boundary,
                    "rng_mode": resume_rng_mode,
                }
                if resume_payload is not None
                else None
            ),
            "steps_per_env": runner_dict["num_steps_per_env"],
            "motion_folder": str(motion_folder),
            "prior_dir": str(prior_dir),
            "contact_source": str(contact_source),
            "contact_seed": contact_seed,
            "causal_contact_bootstrap": (
                {
                    "protocol": (
                        "sugar_causal_contact_bootstrap_v2"
                    ),
                    "previous_action_frame": contact_seed[
                        "source_previous_action_frame"
                    ],
                    "current_action_frame": contact_seed[
                        "source_current_action_frame"
                    ],
                    "tactile_history_frames": contact_seed[
                        "source_tactile_history_frames"
                    ],
                    "previous_action_reaches_observation_bitwise": (
                        contact_seed[
                            "previous_action_reaches_observation_bitwise"
                        ]
                    ),
                    "previous_actor_action_reaches_official_last_action_bitwise": (
                        contact_seed[
                            "previous_actor_action_reaches_official_"
                            "last_action_bitwise"
                        ]
                    ),
                    "previous_applied_action_reaches_goal_observation_bitwise": (
                        contact_seed[
                            "previous_applied_action_reaches_goal_"
                            "observation_bitwise"
                        ]
                    ),
                    "observed_tactile_vs_transformed_source_max_abs": (
                        causal_bootstrap_tactile_max_abs
                    ),
                    "nominal_tactile_vs_unmodified_source_max_abs": (
                        causal_bootstrap_nominal_source_max_abs
                    ),
                    "first_teacher_action_vs_source_l2_max": (
                        causal_bootstrap_teacher_action_l2
                    ),
                    "first_teacher_action_vs_source_max_abs": (
                        causal_bootstrap_teacher_action_max_abs
                    ),
                    "first_teacher_action_error_by_environment": (
                        causal_bootstrap_teacher_action_error_by_env
                    ),
                    "teacher_observation_cross_environment_spread_max_abs": (
                        causal_bootstrap_teacher_observation_spread_max_abs
                    ),
                    "teacher_action_cross_environment_spread_max_abs": (
                        causal_bootstrap_teacher_action_spread_max_abs
                    ),
                    "teacher_observation_vs_passed_live_reference": (
                        causal_bootstrap_teacher_observation_reference
                    ),
                    "teacher_action_tolerance": {
                        "l2_max": 5.0e-6,
                        "restored_live_max_abs": 3.0e-6,
                        "passed_reference_max_abs": 2.0e-6,
                        "restored_observation_max_abs": 2.0e-6,
                    },
                }
                if args.causal_contact_bootstrap_v2
                else None
            ),
            "demo_reward": (
                {
                    "runtime_config": str(legacy_demo_reward_config_path),
                    "runtime_config_sha256": _sha256(
                        legacy_demo_reward_config_path
                    ),
                    "eta": float(
                        demo_reward_config["potential"]["eta"]
                    ),
                    "gamma": float(
                        demo_reward_config["potential"]["gamma"]
                    ),
                    "selected_demo_motion_id": (
                        demo_final_audit["selected_demo_motion_id"]
                    ),
                    "selected_demo_bank_row": (
                        demo_final_audit["selected_demo_bank_row"]
                    ),
                    "final_frozen_audit": demo_final_audit,
                }
                if demo_reward_contract
                else None
            ),
            "demo_reward_telemetry": (
                {
                    "runtime_config": str(legacy_demo_reward_config_path),
                    "runtime_config_sha256": _sha256(
                        legacy_demo_reward_config_path
                    ),
                    "reward_enabled": False,
                    "selected_demo_motion_id": (
                        demo_final_audit["selected_demo_motion_id"]
                    ),
                    "selected_demo_bank_row": (
                        demo_final_audit["selected_demo_bank_row"]
                    ),
                    "final_frozen_audit": demo_final_audit,
                }
                if demo_reward_telemetry_contract
                else None
            ),
            "demo_event_reward": (
                {
                    "runtime_config": str(demo_event_reward_config_path),
                    "selected_option": args.demo_event_selected_option,
                    "selected_demo_task": demo_final_audit["selected_task"],
                    "selected_demo_motion_id": demo_final_audit[
                        "selected_motion_id"
                    ],
                    "selected_demo_bank_row": demo_final_audit[
                        "selected_demo_row"
                    ],
                    "eta": float(demo_event_reward_config["eta"]),
                    "compatibility_baseline": float(
                        demo_event_reward_config["compatibility_baseline"]
                    ),
                    "phase_horizon_steps": args.demo_event_phase_horizon_steps,
                    "final_frozen_audit": demo_final_audit,
                }
                if demo_event_reward_contract
                else None
            ),
            "reward_mix": base_integrator.state_dict()["mix_config"],
            "outcome_reward_weights": outcome_weight_map,
            "external_constraint_weights": external_constraint_weights,
            "termination_terms": termination_names,
            "icm_transition_fields": transition_fields,
            "forbidden_icm_fields": forbidden_icm_fields,
            "means_by_step": {
                "icm_discovery": icm_means,
                "smp": smp_means,
                "task_outcome": task_outcome_means,
                "external_constraint": external_means,
                "policy_base": policy_base_reward_means,
                "policy_total": policy_reward_means,
                "raw_sds": raw_sds_means,
                "demo_reward": demo_reward_means,
                "demo_unit_eta_reward": demo_unit_reward_means,
                "demo_event_potential": demo_event_potential_means,
                "demo_event_risk": demo_event_risk_means,
                "demo_event_uncertainty": demo_event_uncertainty_means,
                "demo_event_ready_fraction": demo_event_ready_fractions,
                "demo_event_phase": demo_event_phase_means,
                "paper_cws_reward": paper_cws_reward_means,
                "paper_cws_weighted_reward": (
                    paper_cws_weighted_reward_means
                ),
            },
            "action_sha256_by_update": action_sha256_by_update,
            "outcome_ledger_abs_max": outcome_ledger_abs_max,
            "policy_reward_reconstruction_max_abs": (
                reward_reconstruction_max_abs
            ),
            "valid_transition_count": valid_transition_count,
            "valid_transition_count_by_update": valid_transition_count_by_update,
            "icm_bootstrap_step_count": bootstrap_count,
            "tactile_nonzero_values": tactile_nonzero_values,
            "tactile_nonzero_values_by_update": (
                tactile_nonzero_values_by_update
            ),
            "tactile_abs_max": tactile_abs_max,
            "initial_tactile_nonzero_values": initial_tactile_nonzero_values,
            "initial_tactile_abs_max": initial_tactile_abs_max,
            "initial_tactile_nonzero_by_role": (
                initial_tactile_nonzero_by_role
            ),
            "tactile_nonzero_by_role_by_update": (
                tactile_nonzero_by_role_by_update
            ),
            "policy_update_metrics": loss_metrics,
            "integration_update_metrics": integration_metrics,
            "policy_update_metrics_by_update": (
                policy_update_metrics_by_update
            ),
            "integration_update_metrics_by_update": (
                integration_metrics_by_update
            ),
            "stability_thresholds": {
                "maximum_update_mean_clip_fraction": 0.5,
                "maximum_policy_epoch_clip_fraction": 0.8,
                "maximum_policy_epoch_sampled_approx_kl": 0.1,
            },
            "numerical_stability_pass": (
                h1_numerical_stability if args.num_updates > 1 else None
            ),
            "parameter_change_max_abs": {
                "actor": actor_delta,
                "critic": critic_delta,
                "all_policy": policy_delta,
            },
            "initial_policy_state_sha256": _state_tree_sha256(
                policy_state_before
            ),
            "prior_state_sha256_before": prior_hash_before,
            "prior_state_sha256_after": prior_hash_after,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_records": checkpoint_records,
            "checkpoint_reload_records": checkpoint_reload_records,
            "checkpoint_reload_policy_max_abs": reload_policy_max_abs,
            "checkpoint_reload_icm_reward_max_abs": reload_icm_reward_max_abs,
            "checks": checks,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if not payload["passed"]:
            failed = sorted(
                name for name, passed in checks.items() if not passed
            )
            raise RuntimeError(f"Stage-H diagnostic failed checks: {failed}")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if reference_waypoint_foundation_reset is not None:
            reference_waypoint_foundation_reset.restore_original_reset()
        if gym_env is not None:
            gym_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
