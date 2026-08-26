# SPDX-License-Identifier: BSD-3-Clause
"""Train the CarryBox tracker on Newton with SUGAR's own BCPPO, logging to Weights & Biases.

Nothing about the algorithm is reimplemented. This imports ``BCPPO`` from SUGAR and runs it
inside ``rsl_rl``'s ``OnPolicyRunner``, with the hyperparameters read out of SUGAR's
``BCPPORunnerCfg``. The only local code in the loop is
:class:`~sugar_newton.rl.vec_env.CarryBoxVecEnv`, which presents the Newton environment in
the shape rsl_rl expects.

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
        --teacher-ckpt experiments/.../ckpts/refiner_model10000.pt \
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
            "training_mask_obs_group": None,
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
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    ap.add_argument("--run-name", default="carrybox_bcppo")
    ap.add_argument("--log-root", default="logs/newton_bcppo")
    ap.add_argument("--logger", default="wandb", choices=("wandb", "tensorboard"))
    ap.add_argument("--wandb-project", default="sugar_newton")
    ap.add_argument("--video-interval", type=int, default=100,
                    help="render an evaluation rollout to wandb every N iterations; 0 disables")
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

    activate_rsl_rl(args.rsl_rl_root)
    sugar_bcppo()
    from rsl_rl.runners import OnPolicyRunner

    from sugar_newton.rl.vec_env import OBS_DIMS, make

    wp.init()
    torch.manual_seed(args.seed)

    env = make(args.num_envs, clip_names=args.clips, episode_length=args.episode_length,
               motion_root=args.motion_root, teacher_motion_root=args.teacher_motion_root,
               substeps=args.substeps, mu=args.mu, device=args.device, seed=args.seed)
    print(f"[env] {args.num_envs} worlds, {len(env.env.clip_names)} clips, "
          f"obs {OBS_DIMS}, act {env.num_actions}")

    log_dir = Path(args.log_root) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg(args), log_dir=str(log_dir), device=args.device)
    print(f"[alg] {type(runner.alg).__name__}  teacher={args.teacher_ckpt}")
    if args.resume:
        runner.load(args.resume)
        print(f"[alg] resumed from {args.resume}")

    if args.video_interval > 0:
        attach_video(runner, args, env.env.clip_names[0])

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

# Keep this file newline-terminated: compute nodes execute it over the shared filesystem.
