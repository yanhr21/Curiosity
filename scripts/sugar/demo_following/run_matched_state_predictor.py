#!/usr/bin/env python3
"""Prepare or run same-teacher selected-demo comparisons in 64-update segments.

Both arms use the same correct CarryBox45 teacher and differ only in whether
the selected reward demo is CarryBox45 or KickBox21. The design retains the
serious SUGAR PPO, official frozen Refiner, official MimicKit prior, frozen
serious frozen predictor, matched seeds and reward weights. Active experiments
run autonomously to their predeclared scientific endpoint. Experiment files
remain below the ignored ``experiments/`` tree.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/sugar/smp/audit_stage_h_smp_icm_policy_integration.py"
PYTHON = Path("/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python")
CORRECT_TEACHER = ROOT / "experiments/demo_following/runtime_assets/correct_teacher"
CORRECT_DEMO = ROOT / (
    "scripts/sugar/demo_reward/config/"
    "plan11_demo_runtime_correct_authority_rework_v3.json"
)
UNRELATED_DEMO = ROOT / (
    "scripts/sugar/demo_reward/config/"
    "plan11_demo_runtime_unrelated_kickbox21_v1.json"
)
PHASE_EVENT_RUNTIME = ROOT / (
    "experiments/demo_following/contact_event_reward_redesign_v1/"
    "phase_aware_dense_feedback_scale_audit_v1/RUNTIME_CONFIG.json"
)
DESIGNS = {
    "phase_event_reward_only": {
        "seed": 161587,
        "action_seed": 161588,
        "protocol": "sugar_phase_event_reward_matched_policy_v1",
        "checkpoint_updates": [32, 64],
        "output": ROOT
        / "experiments/demo_following/matched_phase_event_reward_reference_aware_v2",
        "question": (
            "With one fixed CarryBox45 teacher and identical optimization, "
            "does causal phase-aware dense feedback produce behavior that "
            "depends on CarryBox45 versus unrelated KickBox21?"
        ),
        "arms": {
            "correct": {
                "protocol_arm": "same_teacher_correct_reward",
                "teacher": CORRECT_TEACHER,
                "event_runtime_config": PHASE_EVENT_RUNTIME,
                "selected_option": "correct",
                "meaning": "CarryBox45 teacher and CarryBox45 phase-event reward",
            },
            "unrelated": {
                "protocol_arm": "same_teacher_unrelated_reward",
                "teacher": CORRECT_TEACHER,
                "event_runtime_config": PHASE_EVENT_RUNTIME,
                "selected_option": "unrelated",
                "meaning": "CarryBox45 teacher and KickBox21 phase-event reward",
            },
        },
    },
    "same_teacher_reward_only": {
        "seed": 161581,
        "action_seed": 161582,
        "output": ROOT
        / "experiments/demo_following/matched_reward_identity_same_teacher_v1",
        "question": (
            "Causal selected-demo reward experiment: both arms use the same "
            "correct CarryBox motion45 teacher and differ only in selected demo."
        ),
        "arms": {
            "correct": {
                "protocol_arm": "same_teacher_correct_reward",
                "teacher": CORRECT_TEACHER,
                "demo_config": CORRECT_DEMO,
                "meaning": (
                    "CarryBox motion45 teacher with CarryBox motion45 selected demo"
                ),
            },
            "unrelated": {
                "protocol_arm": "same_teacher_unrelated_reward",
                "teacher": CORRECT_TEACHER,
                "demo_config": UNRELATED_DEMO,
                "meaning": (
                    "CarryBox motion45 teacher with unrelated KickBox motion21 "
                    "selected demo"
                ),
            },
        },
    },
    "teacher_floor_overfit": {
        "seed": 161581,
        "action_seed": 161582,
        "endpoint_updates": 128,
        "start_update": 128,
        "protocol": "sugar_plan11_teacher_floor_overfit_v1",
        "resume_source": ROOT
        / "experiments/demo_following/matched_reward_identity_same_teacher_v1",
        "output": ROOT
        / "experiments/demo_following/teacher_floor_overfit_v1",
        "teacher_wrapper_mode": "wrong_reference_anneal_v1",
        "teacher_anneal_updates": 64,
        "teacher_final_coefficient": 0.25,
        "fixed_physics_profile": {
            "mass_scale": 1.0,
            "static_friction": 0.6,
            "dynamic_friction": 0.5,
            "com_y_m": 0.0,
            "pulse_delta_velocity_w_mps": [0.0, 0.0, 0.0],
        },
        "question": (
            "Fixed-profile learnability diagnostic: after the matched update-64 "
            "endpoints, does the selected reward create Carry-versus-Kick "
            "interaction structure when the common Carry45 teacher is annealed "
            "identically to a nonzero 0.25 floor?"
        ),
        "arms": {
            "correct": {
                "protocol_arm": "same_teacher_correct_reward",
                "teacher": CORRECT_TEACHER,
                "demo_config": CORRECT_DEMO,
                "meaning": (
                    "CarryBox45 selected reward with the common CarryBox45 teacher"
                ),
            },
            "unrelated": {
                "protocol_arm": "same_teacher_unrelated_reward",
                "teacher": CORRECT_TEACHER,
                "demo_config": UNRELATED_DEMO,
                "meaning": (
                    "KickBox21 selected reward with the common CarryBox45 teacher"
                ),
            },
        },
    },
}
PREDECLARED_SEED_PAIRS = {
    (161581, 161582),
    (161583, 161584),
    (161585, 161586),
    (161587, 161588),
    (161589, 161590),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design", choices=tuple(DESIGNS), default="phase_event_reward_only"
    )
    parser.add_argument("--arm", choices=("correct", "unrelated"), required=True)
    parser.add_argument("--endpoint-updates", type=int)
    parser.add_argument("--segment-updates", type=int, default=64)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--action-seed", type=int)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--stop-after-segment",
        action="store_true",
        help="run only the next 64-update segment, then return for inspection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate assets and print the next command without IsaacLab",
    )
    parser.add_argument(
        "--runner-admission-only",
        action="store_true",
        help=(
            "on a retained GPU, load the frozen model through the formal inner "
            "runner and exit before environment creation or PPO"
        ),
    )
    parser.add_argument(
        "--runner-rollout-smoke-only",
        action="store_true",
        help=(
            "on a retained GPU, execute one formal 24-step online rollout "
            "through the actor and reward path, with no optimizer step"
        ),
    )
    parser.add_argument(
        "--probe-evidence-output",
        type=Path,
        help=(
            "persist the passing machine-readable inner-runner probe result; "
            "valid only with one runner probe mode"
        ),
    )
    return parser.parse_args()


def workspace_relative(path: Path) -> str:
    # Preserve stable workspace symlink entrypoints instead of serializing
    # their archive targets outside this checkout.
    return str(path.expanduser().absolute().relative_to(ROOT))


def require_inputs(contract: dict[str, object]) -> None:
    demo_input = contract.get("demo_config") or contract.get(
        "event_runtime_config"
    )
    files = (
        RUNNER,
        PYTHON,
        ROOT / "MimicKit/mimickit/learning/smp_agent.py",
        ROOT / "experiments/demo_following/smp_prior/model.pt",
        ROOT / "experiments/demo_following/smp_prior/result.json",
        ROOT / "experiments/demo_following/smp_prior/diffusion_config.yaml",
        ROOT / "experiments/demo_following/smp_prior/env_config.yaml",
        ROOT
        / "experiments/demo_following/runtime_assets/contact_source/env45_sequence.npz",
        ROOT
        / "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt",
        demo_input,
        contract["teacher"] / "data_045/robot_50hz.npz",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required matched-demo inputs missing: {missing}")
    if contract.get("event_runtime_config") is not None:
        runtime = json.loads(Path(demo_input).read_text(encoding="utf-8"))
        for path in (runtime["dataset_root"], runtime["predictor_dir"]):
            if not Path(path).is_dir():
                raise FileNotFoundError(path)
    if (ROOT / "MimicKit/.git/HEAD").read_text().strip() != (
        "2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69"
    ):
        raise RuntimeError("MimicKit must remain at the pinned detached commit")


def proof_passed(
    path: Path, *, require_reference_aware_phase: bool = False
) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("checks", {})
    if payload.get("passed") is not True or not checks or not all(checks.values()):
        return False
    if not require_reference_aware_phase:
        return True
    audit = payload.get("demo_event_reward", {}).get("final_frozen_audit", {})
    reference_frame = payload.get("contact_seed", {}).get(
        "selected_reference_frame"
    )
    return (
        payload.get("protocol") == "sugar_phase_event_reward_matched_policy_v1"
        and checks.get("demo_event_phase_and_prefix_are_causal") is True
        and reference_frame is not None
        and audit.get("phase_source")
        == "reset_reference_frame_plus_causal_control_clock"
        and audit.get("initial_episode_steps_supplied") is True
        and audit.get("initial_episode_steps_min") == reference_frame
        and audit.get("initial_episode_steps_max") == reference_frame
    )


def require_passing_probe_result(
    path: Path,
    *,
    returncode: int,
    admission_only: bool,
) -> dict[str, object]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("formal inner-runner produced no valid result") from error
    expected_protocol = (
        "sugar_phase_event_policy_admission_only_v1"
        if admission_only
        else "sugar_phase_event_online_rollout_gradient_authority_smoke_v3"
    )
    if (
        returncode != 0
        or result.get("passed") is not True
        or result.get("protocol") != expected_protocol
        or result.get("policy_updates_executed") != 0
    ):
        raise RuntimeError("formal inner-runner probe failed")
    return result


def segment_paths(
    output_root: Path, arm: str, update: int, seed: int
) -> dict[str, Path]:
    directory = output_root / f"seed{seed}/{arm}/update_{update:04d}"
    return {
        "directory": directory,
        "proof": directory / "proof.json",
        "checkpoint": directory / "policy.pt",
        "protocol": directory / "protocol.json",
        "console": directory / "console.log",
    }


def protocol_payload(
    *,
    arm: str,
    update: int,
    previous_update: int,
    previous_checkpoint: Path | None,
    output_root: Path,
    seed: int,
    action_seed: int,
    num_envs: int,
    design: dict[str, object],
) -> dict[str, object]:
    checkpoint_updates = list(
        design.get(
            "checkpoint_updates",
            [1, 64] if update == 64 else [update],
        )
    )
    arms: dict[str, object] = {}
    for name, contract in design["arms"].items():
        paths = segment_paths(output_root, name, update, seed)
        arms[contract["protocol_arm"]] = {
            "meaning": contract["meaning"],
            "teacher_motion_folder": workspace_relative(contract["teacher"]),
            "demo_reward_enabled": True,
            "demo_predictor_telemetry_loaded": True,
            "demo_runtime_config": workspace_relative(
                contract.get("demo_config")
                or contract["event_runtime_config"]
            ),
            "demo_reward_kind": (
                "phase_aware_dense_event"
                if contract.get("event_runtime_config") is not None
                else "legacy_potential_difference"
            ),
            "selected_option": contract.get("selected_option"),
            "output": workspace_relative(paths["proof"]),
            "checkpoint": workspace_relative(paths["checkpoint"]),
        }
    return {
        "protocol": design.get(
            "protocol", "sugar_plan11_fixed_teacher_demo_identity_v2"
        ),
        "execution_ready": True,
        "question": design["question"],
        "shared_runtime": {
            "task": "Sugar-G129dof-CarryBox-SMP-ICM-Goal-Coherent-Latent",
            "training_objective": "goal_recovery_native_authority",
            "sim_and_policy_seed": seed,
            "action_seed": action_seed,
            "num_envs": num_envs,
            "num_updates": update,
            "steps_per_env": 24,
            "policy_contract": "sugar_native_zero_preserving_tactile_fixed_low_lr",
            "latent_physics_distribution_seed": 52017,
            "resume_update": previous_update,
            "resume_checkpoint": (
                {"path": workspace_relative(previous_checkpoint)}
                if previous_checkpoint is not None
                else None
            ),
            "checkpoint_updates": checkpoint_updates,
            "strict_deterministic_torch": True,
            "cublas_workspace_config": ":4096:8",
            "headless_renderer": "disabled_for_policy_training",
            "headless_graphics_icd": "mesa_lavapipe_cpu_no_render",
            "process_shutdown": "fast_exit_after_passing_evidence",
            "teacher_wrapper_mode": design.get(
                "teacher_wrapper_mode", "wrong_reference_fixed_v1"
            ),
            "teacher_anneal_updates": int(
                design.get("teacher_anneal_updates", 0)
            ),
            "teacher_final_coefficient": float(
                design.get("teacher_final_coefficient", 0.0)
            ),
            "fixed_physics_profile": design.get("fixed_physics_profile"),
            "explicit_zero_source_frame": 1,
            "residual_scale": 1.0,
            "tactile_regime": "explicit_zero_control",
            "tactile_mount_environment": None,
            "reward_mix_without_demo": {
                "task_outcome": 10,
                "external_constraint": 1,
                "smp": 0.5,
                "original_icm": 1,
            },
            "demo_event_phase_horizon_steps": (
                650 if design.get("protocol") == "sugar_phase_event_reward_matched_policy_v1" else None
            ),
        },
        "arms": arms,
        "artifacts": {
            "runner_source": {
                "path": workspace_relative(RUNNER),
            },
            "official_refiner_teacher": {
                "path": workspace_relative(
                    ROOT
                    / "experiments/sugar_reproduction/outputs/final/"
                    "official_sugar/baseline/ckpts/refiner_model10000.pt"
                )
            },
            "state_action_source": {
                "path": workspace_relative(
                    ROOT
                    / "experiments/demo_following/runtime_assets/"
                    "contact_source/env45_sequence.npz"
                )
            },
        },
    }


def runner_command(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    protocol: Path,
    previous_checkpoint: Path | None,
    contract: dict[str, object],
) -> list[str]:
    task_motion_folder = CORRECT_TEACHER
    design = DESIGNS[args.design]
    portable_kit_root = (
        Path("/tmp")
        / (
            f"Curiosity_demo_kit_{os.environ.get('SLURM_JOB_ID', 'local')}_"
            f"{paths['directory'].parent.name}_{paths['directory'].name}"
        )
    )
    teacher_wrapper_mode = str(
        design.get("teacher_wrapper_mode", "wrong_reference_fixed_v1")
    )
    teacher_anneal_updates = int(design.get("teacher_anneal_updates", 0))
    teacher_final_coefficient = float(
        design.get("teacher_final_coefficient", 0.0)
    )
    command = [
        str(PYTHON),
        "-u",
        str(RUNNER),
        "--motion-folder",
        str(task_motion_folder),
        "--prior-dir",
        str(ROOT / "experiments/demo_following/smp_prior"),
        "--contact-source",
        str(
            ROOT
            / "experiments/demo_following/runtime_assets/contact_source/env45_sequence.npz"
        ),
        "--nominal-teacher-checkpoint",
        str(
            ROOT
            / "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt"
        ),
        "--protocol-config",
        str(protocol),
        "--protocol-arm",
        contract["protocol_arm"],
        "--training-objective",
        "goal_recovery_native_authority",
        "--policy-contract",
        "sugar_native_zero_preserving_tactile_fixed_low_lr",
        "--tactile-regime",
        "explicit_zero_control",
        "--num-envs",
        str(args.num_envs),
        "--num-updates",
        str(int(paths["directory"].name.removeprefix("update_"))),
        "--checkpoint-updates",
        ",".join(
            str(value)
            for value in design.get(
                "checkpoint_updates",
                (
                    [1, 64]
                    if paths["directory"].name == "update_0064"
                    else [int(paths["directory"].name.removeprefix("update_"))]
                ),
            )
        ),
        "--seed",
        str(args.seed),
        "--action-seed",
        str(args.action_seed),
        "--strict-deterministic-torch",
        "--teacher-wrapper-mode",
        teacher_wrapper_mode,
        "--wrong-teacher-motion-folder",
        str(contract["teacher"]),
        "--teacher-anneal-updates",
        str(teacher_anneal_updates),
        "--teacher-final-coefficient",
        str(teacher_final_coefficient),
        "--explicit-zero-source-frame",
        "1",
        "--residual-scale",
        "1.0",
        "--teacher-release-mode",
        "linear" if teacher_wrapper_mode == "wrong_reference_anneal_v1" else "fixed_one",
        "--teacher-linear-release-steps",
        "4",
        "--teacher-release-scope",
        "full_body",
        "--support-teacher-mode",
        "advancing",
        "--teacher-reference-advance-mode",
        "goal_teacher_post_step_once",
        "--drop-grace-steps",
        "0",
        "--reward-control",
        "full",
        "--output",
        str(paths["proof"]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--device",
        args.device,
        "--headless",
        "--fast-exit-after-evidence",
        "--kit_args",
        (
            f"--portable-root {portable_kit_root} "
            "--/renderer/enabled= --/app/vulkan=false "
            "--/renderer/multiGpu/enabled=false "
            "--/renderer/multiGpu/autoEnable=false "
            "--/renderer/multiGpu/maxGpuCount=1"
        ),
    ]
    if contract.get("event_runtime_config") is not None:
        command.extend(
            [
                "--demo-event-reward-config",
                str(contract["event_runtime_config"]),
                "--demo-event-selected-option",
                str(contract["selected_option"]),
                "--demo-event-phase-horizon-steps",
                "650",
            ]
        )
    else:
        command.extend(["--demo-reward-config", str(contract["demo_config"])])
    if previous_checkpoint is not None:
        command[3:3] = ["--resume-checkpoint", str(previous_checkpoint)]
    return command


def runtime_environment(args: argparse.Namespace, update: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": str(args.seed),
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "SUGAR_DISABLE_TRAIN_DEBUG_VIS": "1",
            "DISPLAY": "",
            "ISAACLAB_GROUND_PLANE_USD": str(
                ROOT / "SUGAR/descriptions/terrain/sugar_ground_plane.usda"
            ),
            "ISAACLAB_TMP_ROOT": (
                f"/tmp/Curiosity_demo_following_{args.arm}_{update}_"
                f"{os.environ.get('SLURM_JOB_ID', 'local')}"
            ),
            "SUGAR_UNITREE_TMP_ROOT": (
                f"/tmp/Curiosity_demo_following_unitree_{args.arm}_{update}_"
                f"{os.environ.get('SLURM_JOB_ID', 'local')}"
            ),
        }
    )
    # Policy training has no cameras or rendering consumers.  Keep PhysX and
    # torch on CUDA, but use the CPU Vulkan ICD for Kit's mandatory graphics
    # bootstrap: the current H200 NVIDIA Vulkan path device-loses after about
    # one minute even when no frame is rendered.
    env["VK_ICD_FILENAMES"] = (
        "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
    )
    python_paths = (
        ROOT / "scripts/sugar/smp",
        ROOT / "IsaacLab/source/isaaclab_contrib",
        ROOT / "IsaacLab/source/isaaclab_assets",
        ROOT / "IsaacLab/source/isaaclab",
        ROOT / "IsaacLab/source/isaaclab_tasks",
        ROOT / "IsaacLab/source/isaaclab_rl",
        ROOT / "SUGAR/source/sugar_rl",
        ROOT / "SUGAR/source/sugar_il",
    )
    env["PYTHONPATH"] = os.pathsep.join(map(str, python_paths))
    for name in (
        "CURIOSITY_TACSL_R15_USD",
        "CURIOSITY_TACSL_LEFT_MOUNT_TRANSLATION_OFFSET",
        "CURIOSITY_TACSL_RIGHT_MOUNT_TRANSLATION_OFFSET",
    ):
        env.pop(name, None)
    return env


def main() -> None:
    args = parse_args()
    design = DESIGNS[args.design]
    args.seed = design["seed"] if args.seed is None else args.seed
    args.action_seed = (
        design["action_seed"] if args.action_seed is None else args.action_seed
    )
    args.output_root = (
        design["output"] if args.output_root is None else args.output_root
    )
    args.endpoint_updates = (
        int(design.get("endpoint_updates", 64))
        if args.endpoint_updates is None
        else args.endpoint_updates
    )
    phase_event_design = args.design == "phase_event_reward_only"
    runner_probe_mode = bool(
        args.runner_admission_only or args.runner_rollout_smoke_only
    )
    if args.runner_admission_only and args.runner_rollout_smoke_only:
        raise ValueError("select exactly one formal runner probe mode")
    if runner_probe_mode and (not phase_event_design or args.dry_run):
        raise ValueError(
            "runner probes are phase-event, non-training, non-dry execution modes"
        )
    if args.probe_evidence_output is not None and not runner_probe_mode:
        raise ValueError(
            "--probe-evidence-output requires one runner probe mode"
        )
    if (
        args.probe_evidence_output is not None
        and args.probe_evidence_output.expanduser().exists()
    ):
        raise FileExistsError(args.probe_evidence_output)
    if (args.seed, args.action_seed) not in PREDECLARED_SEED_PAIRS or args.num_envs != 20:
        raise ValueError(
            "matched design requires one predeclared sim/action seed pair and 20 environments"
        )
    if args.endpoint_updates < 64 or args.endpoint_updates % args.segment_updates:
        raise ValueError("endpoint must be a positive multiple of segment updates")
    if args.segment_updates != 64:
        raise ValueError("the matched segment contract uses 64 updates")
    if phase_event_design and args.endpoint_updates != 64:
        raise ValueError("phase-event first endpoint is fixed at 64 updates")
    if args.design == "teacher_floor_overfit" and (
        args.seed, args.action_seed
    ) != (161581, 161582):
        raise ValueError("teacher-floor overfit is one fixed seed161581 diagnostic")
    arm_contract = design["arms"][args.arm]
    require_inputs(arm_contract)
    output_root = args.output_root.expanduser().resolve()
    previous_checkpoint: Path | None = None
    previous_update = 0
    start_update = int(design.get("start_update", 64))
    resume_source = design.get("resume_source")
    if resume_source is not None:
        source_paths = segment_paths(
            Path(resume_source).expanduser().resolve(),
            args.arm,
            start_update - 64,
            args.seed,
        )
        if not (
            proof_passed(source_paths["proof"])
            and source_paths["checkpoint"].is_file()
        ):
            raise RuntimeError(
                f"declared overfit resume source is incomplete: {source_paths['directory']}"
            )
        previous_checkpoint = source_paths["checkpoint"]
        previous_update = start_update - 64
    if start_update > args.endpoint_updates:
        raise ValueError("design start update exceeds requested endpoint")

    for update in range(start_update, args.endpoint_updates + 1, 64):
        paths = segment_paths(output_root, args.arm, update, args.seed)
        if proof_passed(
            paths["proof"],
            require_reference_aware_phase=phase_event_design,
        ) and paths["checkpoint"].is_file():
            previous_checkpoint = paths["checkpoint"]
            previous_update = update
            continue
        if paths["proof"].exists() or paths["checkpoint"].exists():
            raise RuntimeError(
                f"incomplete or failed segment requires inspection: {paths['directory']}"
            )
        payload = protocol_payload(
            arm=args.arm,
            update=update,
            previous_update=previous_update,
            previous_checkpoint=previous_checkpoint,
            output_root=output_root,
            seed=args.seed,
            action_seed=args.action_seed,
            num_envs=args.num_envs,
            design=design,
        )
        command = runner_command(
            args=args,
            paths=paths,
            protocol=paths["protocol"],
            previous_checkpoint=previous_checkpoint,
            contract=arm_contract,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {"update": update, "protocol": payload, "command": command},
                    indent=2,
                )
            )
            return
        if runner_probe_mode:
            if socket.gethostname().startswith(("mgmtserver", "login")):
                raise SystemExit("runner probe requires a retained compute allocation")
            probe_name = (
                "admission"
                if args.runner_admission_only
                else "rollout_smoke"
            )
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix=f"phase_event_{args.arm}_{probe_name}_",
                delete=False,
            ) as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                temporary_protocol = Path(stream.name)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix=f"phase_event_{args.arm}_{probe_name}_result_",
                delete=False,
            ) as stream:
                temporary_result = Path(stream.name)
            command[command.index("--protocol-config") + 1] = str(
                temporary_protocol
            )
            command.append(
                "--admission-only"
                if args.runner_admission_only
                else "--rollout-smoke-only"
            )
            command.extend(
                ["--probe-result-output", str(temporary_result)]
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT / "SUGAR",
                    env=runtime_environment(args, update),
                    check=False,
                )
                probe_result = require_passing_probe_result(
                    temporary_result,
                    returncode=completed.returncode,
                    admission_only=args.runner_admission_only,
                )
                if args.probe_evidence_output is not None:
                    evidence_output = (
                        args.probe_evidence_output.expanduser().resolve()
                    )
                    evidence_output.parent.mkdir(parents=True, exist_ok=True)
                    evidence_output.write_text(
                        json.dumps(probe_result, indent=2, sort_keys=True)
                        + "\n",
                        encoding="utf-8",
                    )
            finally:
                temporary_protocol.unlink(missing_ok=True)
                temporary_result.unlink(missing_ok=True)
            return
        paths["directory"].mkdir(parents=True, exist_ok=False)
        paths["protocol"].write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if socket.gethostname().startswith("mgmtserver"):
            raise SystemExit("refusing IsaacLab training on a login node")
        with paths["console"].open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=ROOT / "SUGAR",
                env=runtime_environment(args, update),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not proof_passed(
            paths["proof"],
            require_reference_aware_phase=phase_event_design,
        ):
            raise RuntimeError(
                f"segment {update} failed; inspect {paths['console']} and {paths['proof']}"
            )
        if update == 64:
            first_update = paths["checkpoint"].with_name("policy_update1.pt")
            first_update.unlink(missing_ok=True)
        previous_checkpoint = paths["checkpoint"]
        previous_update = update
        if args.stop_after_segment:
            return


if __name__ == "__main__":
    main()
