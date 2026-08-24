#!/usr/bin/env python3
"""Run one shared SUGAR actor conditioned on Carry45 and Kick21 demos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path("/public/home/yanhongru/envs/sugar_py311_isaacsim510/bin/python")
RUNNER = ROOT / "scripts/sugar/smp/audit_stage_h_smp_icm_policy_integration.py"
OUTPUT_ROOT = ROOT / (
    "experiments/demo_following/shared_actionable_demo_conditioning_v1/seed161591"
)
RUNTIME_CONFIG = ROOT / (
    "experiments/demo_following/contact_event_reward_redesign_v1/"
    "phase_aware_dense_feedback_scale_audit_v1/RUNTIME_CONFIG.json"
)
TEACHER = ROOT / "experiments/demo_following/runtime_assets/correct_teacher"
PRIOR = ROOT / "experiments/demo_following/smp_prior"
CONTACT = ROOT / (
    "experiments/demo_following/runtime_assets/contact_source/env45_sequence.npz"
)
REFINER = ROOT / (
    "experiments/sugar_reproduction/outputs/final/official_sugar/baseline/"
    "ckpts/refiner_model10000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--admission-only", action="store_true")
    mode.add_argument("--rollout-smoke-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=161591)
    parser.add_argument("--action-seed", type=int, default=161592)
    return parser.parse_args()


def relative(path: Path) -> str:
    # Keep the repository-facing logical path. Runtime assets may be stable
    # symlinks into the single local archive; resolving those links would turn
    # a valid workspace entry into an unrelated absolute provenance path.
    return str(path.expanduser().absolute().relative_to(ROOT))


def require_inputs() -> None:
    files = (
        PYTHON,
        RUNNER,
        RUNTIME_CONFIG,
        CONTACT,
        REFINER,
        PRIOR / "model.pt",
        PRIOR / "result.json",
        PRIOR / "diffusion_config.yaml",
        PRIOR / "env_config.yaml",
        TEACHER / "data_045/robot_50hz.npz",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing shared-conditioning inputs: {missing}")


def protocol_payload(
    output: Path, checkpoint: Path, seed: int, action_seed: int
) -> dict[str, object]:
    return {
        "protocol": "sugar_shared_actionable_demo_conditioning_v1",
        "execution_ready": True,
        "question": (
            "Can one shared SUGAR checkpoint change behavior when only the "
            "selected Carry45 versus Kick21 demo condition changes?"
        ),
        "shared_runtime": {
            "sim_and_policy_seed": seed,
            "action_seed": action_seed,
            "num_envs": 20,
            "num_updates": 64,
            "resume_update": 0,
            "resume_checkpoint": None,
            "checkpoint_updates": [1, 64],
            "strict_deterministic_torch": True,
            "cublas_workspace_config": ":4096:8",
            "teacher_wrapper_mode": "wrong_reference_fixed_v1",
            "teacher_anneal_updates": 0,
            "teacher_final_coefficient": 0.0,
            "explicit_zero_source_frame": 1,
            "residual_scale": 1.0,
            "tactile_regime": "explicit_zero_control",
            "tactile_mount_environment": None,
            "demo_event_phase_horizon_steps": 650,
            "reward_mix_without_demo": {
                "task_outcome": 10,
                "external_constraint": 1,
                "smp": 0.5,
                "original_icm": 1,
            },
            "actionable_demo_conditioning": {
                "dimension": 798,
                "selected_demo_assignment": "even_correct_odd_unrelated",
                "future_actual_events_used": False,
                "single_shared_checkpoint": True,
            },
        },
        "arms": {
            "shared_balanced_conditioning": {
                "meaning": (
                    "one actor; ten environments receive Carry45 and ten "
                    "receive Kick21 through the frozen causal predictor"
                ),
                "teacher_motion_folder": relative(TEACHER),
                "demo_reward_enabled": True,
                "demo_predictor_telemetry_loaded": True,
                "demo_runtime_config": relative(RUNTIME_CONFIG),
                "demo_reward_kind": "phase_aware_dense_event",
                "selected_option": "balanced",
                "output": relative(output),
                "checkpoint": relative(checkpoint),
            }
        },
        "artifacts": {
            "runner_source": {"path": relative(RUNNER)},
            "official_refiner_teacher": {"path": relative(REFINER)},
            "state_action_source": {"path": relative(CONTACT)},
        },
    }


def environment(seed: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": str(seed),
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "SUGAR_DISABLE_TRAIN_DEBUG_VIS": "1",
            "DISPLAY": "",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
            "ISAACLAB_GROUND_PLANE_USD": str(
                ROOT / "SUGAR/descriptions/terrain/sugar_ground_plane.usda"
            ),
            "ISAACLAB_TMP_ROOT": (
                f"/tmp/Curiosity_shared_demo_{os.environ.get('SLURM_JOB_ID', 'local')}"
            ),
            "SUGAR_UNITREE_TMP_ROOT": (
                f"/tmp/Curiosity_shared_demo_unitree_{os.environ.get('SLURM_JOB_ID', 'local')}"
            ),
        }
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
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    for name in (
        "CURIOSITY_TACSL_R15_USD",
        "CURIOSITY_TACSL_LEFT_MOUNT_TRANSLATION_OFFSET",
        "CURIOSITY_TACSL_RIGHT_MOUNT_TRANSLATION_OFFSET",
    ):
        env.pop(name, None)
    return env


def main() -> None:
    args = parse_args()
    require_inputs()
    root = args.output_root.expanduser().resolve()
    endpoint = root / "update_0064"
    output = endpoint / "proof.json"
    checkpoint = endpoint / "policy.pt"
    protocol = endpoint / "protocol.json"
    if output.exists() or checkpoint.exists():
        raise FileExistsError("shared-conditioning endpoint already exists")
    endpoint.mkdir(parents=True, exist_ok=True)
    protocol.write_text(
        json.dumps(
            protocol_payload(output, checkpoint, args.seed, args.action_seed),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    portable_root = Path("/tmp") / (
        f"Curiosity_shared_demo_kit_{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    command = [
        str(PYTHON),
        "-u",
        str(RUNNER),
        "--motion-folder",
        str(TEACHER),
        "--prior-dir",
        str(PRIOR),
        "--contact-source",
        str(CONTACT),
        "--nominal-teacher-checkpoint",
        str(REFINER),
        "--protocol-config",
        str(protocol),
        "--protocol-arm",
        "shared_balanced_conditioning",
        "--training-objective",
        "goal_recovery_native_authority",
        "--policy-contract",
        "sugar_native_zero_preserving_tactile_fixed_low_lr",
        "--tactile-regime",
        "explicit_zero_control",
        "--num-envs",
        "20",
        "--num-updates",
        "64",
        "--checkpoint-updates",
        "1,64",
        "--seed",
        str(args.seed),
        "--action-seed",
        str(args.action_seed),
        "--strict-deterministic-torch",
        "--teacher-wrapper-mode",
        "wrong_reference_fixed_v1",
        "--wrong-teacher-motion-folder",
        str(TEACHER),
        "--teacher-anneal-updates",
        "0",
        "--teacher-final-coefficient",
        "0.0",
        "--explicit-zero-source-frame",
        "1",
        "--residual-scale",
        "1.0",
        "--teacher-release-mode",
        "fixed_one",
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
        "--demo-event-reward-config",
        str(RUNTIME_CONFIG),
        "--demo-event-selected-option",
        "balanced",
        "--demo-event-phase-horizon-steps",
        "650",
        "--actionable-demo-conditioning",
        "--output",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.device,
        "--headless",
        "--fast-exit-after-evidence",
        "--kit_args",
        (
            f"--portable-root {portable_root} --/renderer/enabled= "
            "--/app/vulkan=false --/renderer/multiGpu/enabled=false "
            "--/renderer/multiGpu/autoEnable=false "
            "--/renderer/multiGpu/maxGpuCount=1"
        ),
    ]
    probe = endpoint / (
        "admission.json" if args.admission_only else "rollout_smoke.json"
    )
    if args.admission_only:
        command.extend(["--admission-only", "--probe-result-output", str(probe)])
    elif args.rollout_smoke_only:
        command.extend(
            ["--rollout-smoke-only", "--probe-result-output", str(probe)]
        )
    if args.dry_run:
        print(json.dumps({"protocol": str(protocol), "command": command}, indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run IsaacLab only inside a retained compute allocation")
    completed = subprocess.run(
        command, cwd=ROOT / "SUGAR", env=environment(args.seed)
    )
    evidence = probe if (args.admission_only or args.rollout_smoke_only) else output
    if completed.returncode != 0 or not evidence.is_file():
        raise RuntimeError(
            f"shared-conditioning runner failed without evidence: {evidence}"
        )
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise RuntimeError("shared-conditioning evidence did not pass")


if __name__ == "__main__":
    main()
