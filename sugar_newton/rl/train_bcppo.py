# SPDX-License-Identifier: BSD-3-Clause
"""Train the CarryBox tracker on Newton with SUGAR's own BCPPO, logging to Weights & Biases.

Nothing about the algorithm is reimplemented. This imports ``BCPPO`` from SUGAR and runs it
inside ``rsl_rl``'s ``OnPolicyRunner``, with the hyperparameters read out of SUGAR's
``BCPPORunnerCfg``. The local adapter
:class:`~sugar_newton.rl.vec_env.ActingTeacherHandoffVecEnv` presents Newton in the
shape rsl_rl expects, executes the admitted Refiner during the pickup prefix, and masks
that prefix out of optimization.

``BCPPO`` is a three-stage curriculum around the frozen refiner as teacher::

    stage 1   step < 500          loss = distill                    (LR schedule fixed)
    stage 2   500 <= step < 1000  loss = distill + alpha * value    (no policy gradient)
    stage 3   step >= 2000 ramp   loss = alpha * surrogate + value
                                         - alpha * entropy + distill * max(1-alpha, floor)

so the teacher checkpoint is not optional -- ``BCPPO.__init__`` asserts on a missing one,
and without it stages 1-2 have no loss at all.

Usage::

    python -m sugar_newton.rl.train_bcppo \
        --motion-root experiments/sugar_reproduction/outputs/newton_refiner_dataset_20260827/rollout_datasets/refiner/rl_dataset \
        --num-envs 8 --max-iterations 30001 \
        --teacher-ckpt experiments/.../newton_refiner/model_255.pt \
        --teacher-gate-result experiments/.../formal20/RESULT.json \
        --rsl-rl-root /path/to/SUGAR-compatible/site-packages/rsl_rl \
        --wandb-project sugar_newton --run-name carrybox_bcppo_$(date +%m%d_%H%M)

The runner writes wandb through rsl_rl's own ``WandbSummaryWriter`` (``logger: wandb``),
so the run name is the log directory's basename and the project comes from
``wandb_project`` in the runner config. Credentials follow the convention used elsewhere in
this workspace: ``WANDB_API_KEY`` from the environment, else ``~/.netrc`` for
``api.wandb.ai``. The key is never printed.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import warp as wp

HERE = Path(__file__).resolve().parent
SUGAR_SRC = HERE.parents[1] / "SUGAR" / "source" / "sugar_rl"
DEFAULT_TEACHER = (HERE.parents[1] / "experiments/sugar_reproduction/outputs/final"
                   / "official_sugar/baseline/ckpts/refiner_model10000.pt")
DEFAULT_TEACHER_MOTIONS = HERE.parents[1] / "SUGAR/data/CarryBox"


def activate_rsl_rl(package_root: str) -> None:
    """Select the official SUGAR-compatible rsl-rl package before importing BCPPO.

    The Newton runtime on this cluster carries rsl-rl 5.x, whose model/runner API is
    incompatible with SUGAR's released BCPPO.  SUGAR's environment already contains the
    required 3.0.1 package and it is pure Python, so load that exact package in the
    Newton Python process rather than modifying either implementation.
    """
    if package_root:
        root = Path(package_root).expanduser().resolve()
        package = root if root.name == "rsl_rl" else root / "rsl_rl"
        init = package / "__init__.py"
        if not init.is_file():
            raise SystemExit(f"rsl-rl package root has no rsl_rl/__init__.py: {root}")
        for name in tuple(sys.modules):
            if name == "rsl_rl" or name.startswith("rsl_rl."):
                del sys.modules[name]
        spec = importlib.util.spec_from_file_location(
            "rsl_rl", init, submodule_search_locations=[str(package)]
        )
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load rsl-rl package from {package}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["rsl_rl"] = module
        spec.loader.exec_module(module)

    try:
        import rsl_rl
        from rsl_rl.modules import ActorCritic  # noqa: F401
        from rsl_rl.runners import OnPolicyRunner  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "SUGAR BCPPO requires the rsl-rl 3.x ActorCritic/OnPolicyRunner API; "
            "pass --rsl-rl-root or SUGAR_RSL_RL_ROOT pointing at the compatible "
            f"rsl_rl package ({exc})"
        ) from exc
    print(f"[rsl-rl] SUGAR-compatible package: {Path(rsl_rl.__file__).resolve()}")


def sugar_bcppo():
    """Import SUGAR's BCPPO and register it where rsl_rl's runner will find it.

    ``OnPolicyRunner._construct_algorithm`` resolves the algorithm with
    ``eval(alg_cfg["class_name"])``, so the class has to be reachable by bare name. This is
    SUGAR's own mechanism, copied from ``scripts/sugar_rl/train.py:147-150``, not a new one.
    ``sugar_rl/utils/__init__.py`` is empty, so the module imports without pulling in
    IsaacLab.
    """
    sys.path.insert(0, str(SUGAR_SRC))
    import rsl_rl.algorithms
    from sugar_rl.utils.rsl_rl_bcppo import BCPPO

    builtins.BCPPO = BCPPO
    rsl_rl.algorithms.BCPPO = BCPPO
    return BCPPO


def runner_cfg(args) -> dict:
    """BCPPORunnerCfg, transcribed. Values are SUGAR's; only logging is added."""
    return {
        "num_steps_per_env": 24,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "experiment_name": args.run_name,
        "empirical_normalization": False,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"], "teacher": ["teacher"]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.5,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "BCPPO",
            "teacher_ckpt": args.teacher_ckpt,
            "stage3_distill_weight_floor": 0.0,
            # The mask is not an actor input.  It is stored beside the 510-D policy
            # observation and removes the physical Refiner-controlled pickup prefix
            # from PPO/value credit.  Official full-trajectory teacher distillation
            # remains active: the prefix is valid BC data even though the student did
            # not physically execute its sampled action there.
            "training_mask_obs_group": "training_handoff_mask",
            "distill_mask_start_step": args.max_iterations + 1,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "logger": args.logger,
        "wandb_project": args.wandb_project,
    }


def ensure_wandb_credentials() -> None:
    """Fail here rather than several minutes into a run with logging silently off."""
    if os.environ.get("WANDB_API_KEY"):
        return
    try:
        import netrc

        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth and auth[2]:
            os.environ["WANDB_API_KEY"] = auth[2]
            return
    except Exception:
        pass
    raise SystemExit("wandb: set WANDB_API_KEY or add api.wandb.ai to ~/.netrc "
                     "(or pass --logger tensorboard)")


def require_admitted_teacher(gate_result: str | Path, checkpoint: str | Path) -> dict:
    """Fail closed unless the exact acting checkpoint passed the frozen gate."""
    gate_path = Path(gate_result).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not gate_path.is_file():
        raise SystemExit(f"teacher gate result not found: {gate_path}")
    try:
        result = json.loads(gate_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid teacher gate result {gate_path}: {exc}") from exc

    from sugar_newton.validation.refiner_open_loop import sha256

    expected_sha = sha256(checkpoint_path)
    checks = result.get("checks", {})
    required = {
        "all_profiles_finished": True,
        "checkpoint_actor_is_exact_890_to_29_official_mlp": True,
        "no_active_profile_diverged": True,
        "physical_lift_fraction_passes": True,
        "strict_completion_fraction_passes": True,
    }
    failures = {
        key: checks.get(key)
        for key, expected in required.items()
        if checks.get(key) is not expected
    }
    if result.get("protocol") != "sugar_newton_official_refiner_open_loop_gate_v1":
        failures["protocol"] = result.get("protocol")
    if result.get("passed") is not True:
        failures["passed"] = result.get("passed")
    if result.get("checkpoint_sha256") != expected_sha:
        failures["checkpoint_sha256"] = {
            "gate": result.get("checkpoint_sha256"),
            "acting_checkpoint": expected_sha,
        }
    try:
        lifted_count = int(result["lifted_profile_count"])
        strict_count = int(result["strict_complete_profile_count"])
        required_count = int(result["required_profile_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"teacher gate result has invalid count fields: {gate_path}"
        ) from exc
    if lifted_count < required_count:
        failures["lifted_profile_count"] = lifted_count
    if strict_count < required_count:
        failures["strict_complete_profile_count"] = strict_count
    if failures:
        raise SystemExit(
            "acting Refiner is not admitted for Newton Tracker training: "
            + json.dumps(failures, sort_keys=True)
        )
    return {
        "path": str(gate_path),
        "protocol": result["protocol"],
        "checkpoint_sha256": expected_sha,
        "num_profiles": int(result["num_profiles"]),
        "required_profile_count": required_count,
        "lifted_profile_count": lifted_count,
        "strict_complete_profile_count": strict_count,
    }


def attach_video(runner, args, clip: str) -> None:
    """Render an evaluation rollout every ``--video-interval`` iterations.

    rsl_rl has no hook for this, so ``runner.save`` is wrapped: it already fires on
    ``save_interval``, and piggybacking keeps the cadence tied to something the runner
    controls rather than duplicating its iteration bookkeeping. A failure to render is
    logged and swallowed -- a missing video must never end a training run.
    """
    from sugar_newton.rl.video import VideoRecorder

    rec = VideoRecorder(clip=clip, frames=args.video_frames,
                        out_dir=str(Path(args.log_root) / args.run_name / "videos"),
                        motion_root=args.motion_root,
                        teacher_motion_root=args.teacher_motion_root,
                        mu=args.mu, substeps=args.substeps, device=args.device)
    original_save = runner.save
    state = {"last": -1}

    def save_and_record(path, infos=None):
        original_save(path, infos)
        it = runner.current_learning_iteration
        if it == state["last"] or it % args.video_interval:
            return
        state["last"] = it
        try:
            video_path, stats = rec.record(runner.alg.policy, it)
        except Exception as exc:
            print(f"[video] skipped at iter {it}: {type(exc).__name__}: {exc}")
            return
        if video_path is None:
            return
        print(f"[video] iter {it}: {video_path}  lift {stats.get('video/box_lift', 0):.3f} m "
              f"(reference {stats.get('video/box_lift_reference', 0):.3f} m)")
        if args.logger == "wandb":
            import wandb

            if wandb.run is not None:
                wandb.log({**stats,
                           "video/rollout": wandb.Video(video_path, fps=rec.fps,
                                                        format="mp4")},
                          step=it)

    runner.save = save_and_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8,
                    help="Newton physics worlds; 8 is validated, larger values require a fresh benchmark")
    ap.add_argument("--max-iterations", type=int, default=30001)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument(
        "--motion-root", required=True,
        help="Refiner rollout rl_dataset used by the Tracker student/critic; for an execution-only "
             "smoke this may point at raw data, but that is not faithful Tracker training",
    )
    ap.add_argument("--teacher-motion-root", default=str(DEFAULT_TEACHER_MOTIONS),
                    help="raw CarryBox motions consumed by the frozen Refiner teacher")
    ap.add_argument("--episode-length", type=int, default=300)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--teacher-ckpt", default=str(DEFAULT_TEACHER))
    ap.add_argument(
        "--teacher-gate-result",
        required=True,
        help="frozen 20-profile RESULT.json admitting this exact acting checkpoint",
    )
    ap.add_argument("--run-name", default="carrybox_bcppo")
    ap.add_argument("--log-root", default="logs/newton_bcppo")
    ap.add_argument("--logger", default="wandb", choices=("wandb", "tensorboard"))
    ap.add_argument("--wandb-project", default="sugar_newton")
    ap.add_argument("--video-interval", type=int, default=0,
                    help="reserved for a future matched handoff evaluator; must remain zero")
    ap.add_argument("--video-frames", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--rsl-rl-root",
        default=os.environ.get("SUGAR_RSL_RL_ROOT", ""),
        help="site-packages directory (or rsl_rl package directory) for SUGAR's compatible rsl-rl 3.x",
    )
    args = ap.parse_args()

    if not Path(args.teacher_ckpt).is_file():
        raise SystemExit(f"teacher checkpoint not found: {args.teacher_ckpt}\n"
                         "BCPPO's stages 1-2 have no loss without it.")
    teacher_gate = require_admitted_teacher(
        args.teacher_gate_result, args.teacher_ckpt
    )
    for label, root in (("student motion", args.motion_root),
                        ("teacher motion", args.teacher_motion_root)):
        root_path = Path(root)
        if not root_path.is_dir() or not any(root_path.glob("data_*")):
            raise SystemExit(f"{label} root has no data_* clips: {root_path}")
    # The port fails closed if a future edit removes any official reward term. A
    # three-iteration raw-motion run remains execution-smoke only.
    from sugar_newton.rl import rewards

    if rewards.OMITTED and args.max_iterations > 3:
        raise SystemExit(
            "formal training disabled: Newton reward still omits official terms "
            f"{rewards.OMITTED}; max_iterations<=3 is execution-smoke only"
        )
    if (Path(args.motion_root).resolve() == Path(args.teacher_motion_root).resolve()
            and args.max_iterations > 3):
        raise SystemExit(
            "formal training requires Refiner rollout motion for --motion-root; "
            "raw student==teacher data is execution-smoke only"
        )
    if args.logger == "wandb":
        ensure_wandb_credentials()
    if args.video_interval != 0:
        raise SystemExit(
            "the old video recorder starts the Tracker before a physical handoff; "
            "--video-interval must remain 0 until a matched handoff recorder exists"
        )

    activate_rsl_rl(args.rsl_rl_root)
    sugar_bcppo()
    from rsl_rl.runners import OnPolicyRunner

    from sugar_newton.rl.vec_env import OBS_DIMS, make_acting_teacher_handoff

    wp.init()
    torch.manual_seed(args.seed)

    env = make_acting_teacher_handoff(
        args.num_envs,
        teacher_checkpoint=args.teacher_ckpt,
        clip_names=args.clips,
        episode_length=args.episode_length,
        motion_root=args.motion_root,
        teacher_motion_root=args.teacher_motion_root,
        substeps=args.substeps,
        mu=args.mu,
        device=args.device,
        seed=args.seed,
    )
    print(f"[env] {args.num_envs} worlds, {len(env.env.clip_names)} clips, "
          f"obs {OBS_DIMS} + training_handoff_mask:1, act {env.num_actions}")

    log_dir = Path(args.log_root) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    initialization_audit = {
        "protocol": "sugar_newton_bcppo_acting_teacher_handoff_init_v1",
        "fresh_student_optimizer_and_iteration": True,
        "resume_checkpoint": None,
        "teacher_gate": teacher_gate,
        "acting_and_distillation_teacher_checkpoint": str(
            Path(args.teacher_ckpt).expanduser().resolve()
        ),
        "acting_teacher_hidden_dims": [512, 256, 128],
        "policy_observation_dim": OBS_DIMS["policy"],
        "critic_observation_dim": OBS_DIMS["critic"],
        "teacher_observation_dim": OBS_DIMS["teacher"],
        "training_mask_actor_input": False,
        "training_mask_obs_group": "training_handoff_mask",
        "full_trajectory_teacher_distillation_retained": True,
        "minimum_lift_m": env.MINIMUM_LIFT_M,
        "stable_lift_frames": env.STABLE_LIFT_FRAMES,
        "teacher_controls_prefix": True,
        "teacher_prefix_ppo_credit": False,
    }
    (log_dir / "INITIALIZATION_AUDIT.json").write_text(
        json.dumps(initialization_audit, indent=2, sort_keys=True) + "\n"
    )
    runner = OnPolicyRunner(env, runner_cfg(args), log_dir=str(log_dir), device=args.device)
    print(f"[alg] {type(runner.alg).__name__}  teacher={args.teacher_ckpt}")
    acting_state = env.acting_teacher.state_dict()
    distillation_state = runner.alg.teacher_model.state_dict()
    if acting_state.keys() != distillation_state.keys():
        raise RuntimeError("acting and distillation Refiner state geometry differs")
    teacher_pair_max_abs_delta = max(
        float((acting_state[key] - distillation_state[key]).abs().max())
        for key in acting_state
    )
    if teacher_pair_max_abs_delta != 0.0:
        raise RuntimeError(
            "acting and distillation Refiner parameters are not bitwise equal: "
            f"max delta {teacher_pair_max_abs_delta}"
        )
    initialization_audit["acting_vs_distillation_teacher_max_abs_delta"] = (
        teacher_pair_max_abs_delta
    )
    (log_dir / "INITIALIZATION_AUDIT.json").write_text(
        json.dumps(initialization_audit, indent=2, sort_keys=True) + "\n"
    )
    acting_teacher_initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in env.acting_teacher.named_parameters()
    }
    distillation_teacher_initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in runner.alg.teacher_model.named_parameters()
    }

    # The Newton environment already samples an exact random reference start frame.
    # Randomizing only rsl_rl's logging buffer would desynchronize episode length from
    # the physics and corrupt handoff-step diagnostics.
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=False,
    )

    acting_teacher_max_delta = max(
        float(
            (parameter.detach().cpu() - acting_teacher_initial[name])
            .abs()
            .max()
        )
        for name, parameter in env.acting_teacher.named_parameters()
    )
    distillation_teacher_max_delta = max(
        float(
            (parameter.detach().cpu() - distillation_teacher_initial[name])
            .abs()
            .max()
        )
        for name, parameter in runner.alg.teacher_model.named_parameters()
    )
    for parameter in env.acting_teacher.parameters():
        if parameter.requires_grad:
            raise RuntimeError("acting Refiner parameter unexpectedly requires grad")
        if not torch.isfinite(parameter).all():
            raise RuntimeError("acting Refiner parameter became non-finite")
    policy_parameters_finite = all(
        bool(torch.isfinite(parameter).all())
        for parameter in runner.alg.policy.parameters()
    )
    total_transitions = int(
        args.num_envs * runner.cfg["num_steps_per_env"] * args.max_iterations
    )
    divergence_rate = float(env.env.num_diverged / max(total_transitions, 1))
    result = {
        "protocol": "sugar_newton_bcppo_acting_teacher_handoff_result_v1",
        "teacher_checkpoint_sha256": env.teacher_checkpoint_sha256,
        "acting_teacher_parameter_max_abs_delta": acting_teacher_max_delta,
        "distillation_teacher_parameter_max_abs_delta": (
            distillation_teacher_max_delta
        ),
        "acting_teacher_parameters_frozen": acting_teacher_max_delta == 0.0,
        "distillation_teacher_parameters_frozen": (
            distillation_teacher_max_delta == 0.0
        ),
        "cumulative_teacher_control_steps": int(
            env.cumulative_teacher_control_steps.sum().item()
        ),
        "cumulative_policy_control_steps": int(
            env.cumulative_policy_control_steps.sum().item()
        ),
        "cumulative_handoffs": int(env.cumulative_handoffs.sum().item()),
        "divergence_total": int(env.env.num_diverged),
        "divergence_rate": divergence_rate,
        "total_transitions": total_transitions,
        "policy_parameters_finite": policy_parameters_finite,
        "training_mask_obs_group": "training_handoff_mask",
        "full_trajectory_teacher_distillation_retained": True,
        "teacher_prefix_ppo_credit": False,
        "passed": bool(
            env.cumulative_handoffs.sum().item() > 0
            and env.cumulative_policy_control_steps.sum().item() > 0
            and acting_teacher_max_delta == 0.0
            and distillation_teacher_max_delta == 0.0
            and policy_parameters_finite
            and divergence_rate <= 0.005
        ),
    }
    (log_dir / "TRAINING_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

# Keep this file newline-terminated: compute nodes execute it over the shared filesystem.
