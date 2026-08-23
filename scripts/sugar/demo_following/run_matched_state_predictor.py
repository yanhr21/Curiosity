#!/usr/bin/env python3
"""Run the same-teacher selected-demo comparison in 64-update segments.

Both arms use the same correct CarryBox45 teacher and differ only in whether
the selected reward demo is CarryBox45 or KickBox21. The design retains the
serious SUGAR PPO, official frozen Refiner, official MimicKit prior, frozen
11.9M predictor, matched seeds and reward weights. Experiment files remain
below the ignored ``experiments/`` tree.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


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
DESIGNS = {
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
}
PREDECLARED_SEED_PAIRS = {
    (161581, 161582),
    (161583, 161584),
    (161585, 161586),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design", choices=tuple(DESIGNS), default="same_teacher_reward_only"
    )
    parser.add_argument("--arm", choices=("correct", "unrelated"), required=True)
    parser.add_argument("--endpoint-updates", type=int, default=64)
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
    return parser.parse_args()


def workspace_relative(path: Path) -> str:
    # Preserve stable workspace symlink entrypoints instead of serializing
    # their archive targets outside this checkout.
    return str(path.expanduser().absolute().relative_to(ROOT))


def require_inputs(contract: dict[str, object]) -> None:
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
        contract["demo_config"],
        contract["teacher"] / "data_045/robot_50hz.npz",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required matched-demo inputs missing: {missing}")
    if (ROOT / "MimicKit/.git/HEAD").read_text().strip() != (
        "2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69"
    ):
        raise RuntimeError("MimicKit must remain at the pinned detached commit")


def proof_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("passed") is True and all(payload.get("checks", {}).values())


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
    checkpoint_updates = [1, 64] if update == 64 else [update]
    arms: dict[str, object] = {}
    for name, contract in design["arms"].items():
        paths = segment_paths(output_root, name, update, seed)
        arms[contract["protocol_arm"]] = {
            "meaning": contract["meaning"],
            "teacher_motion_folder": workspace_relative(contract["teacher"]),
            "demo_reward_enabled": True,
            "demo_predictor_telemetry_loaded": True,
            "demo_runtime_config": workspace_relative(contract["demo_config"]),
            "output": workspace_relative(paths["proof"]),
            "checkpoint": workspace_relative(paths["checkpoint"]),
        }
    return {
        "protocol": "sugar_plan11_fixed_teacher_demo_identity_v2",
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
            "teacher_wrapper_mode": "wrong_reference_fixed_v1",
            "teacher_anneal_updates": 0,
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
        (
            "1,64"
            if paths["directory"].name == "update_0064"
            else str(int(paths["directory"].name.removeprefix("update_")))
        ),
        "--seed",
        str(args.seed),
        "--action-seed",
        str(args.action_seed),
        "--strict-deterministic-torch",
        "--teacher-wrapper-mode",
        "wrong_reference_fixed_v1",
        "--wrong-teacher-motion-folder",
        str(contract["teacher"]),
        "--teacher-anneal-updates",
        "0",
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
        "--demo-reward-config",
        str(contract["demo_config"]),
        "--output",
        str(paths["proof"]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--device",
        args.device,
        "--headless",
    ]
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
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
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
    if (args.seed, args.action_seed) not in PREDECLARED_SEED_PAIRS or args.num_envs != 20:
        raise ValueError(
            "matched design requires one predeclared sim/action seed pair and 20 environments"
        )
    if args.endpoint_updates < 64 or args.endpoint_updates % args.segment_updates:
        raise ValueError("endpoint must be a positive multiple of segment updates")
    if args.segment_updates != 64:
        raise ValueError("the matched segment contract uses 64 updates")
    arm_contract = design["arms"][args.arm]
    require_inputs(arm_contract)
    output_root = args.output_root.expanduser().resolve()
    previous_checkpoint: Path | None = None
    previous_update = 0

    for update in range(64, args.endpoint_updates + 1, 64):
        paths = segment_paths(output_root, args.arm, update, args.seed)
        if proof_passed(paths["proof"]) and paths["checkpoint"].is_file():
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
        if completed.returncode != 0 or not proof_passed(paths["proof"]):
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
