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

**Tactile is not an observation here.** The policy reads SUGAR's 510-D
command/proprioception group and nothing else; the critic and teacher read the privileged
890-D group. The tactile field is computed only for the evaluation video, where it is drawn
as a per-hand heatmap so a grasp can be told apart from a wrist wedged under the box. Wiring
tactile into the policy is Plan 16 phase 2 and is deliberately not done here.

Usage::

    python -m sugar_newton.rl.train_bcppo \
        --num-envs 512 --max-iterations 30001 \
        --teacher-ckpt experiments/.../ckpts/refiner_model10000.pt \
        --eval-minutes 10 \
        --wandb-project sugar_newton --run-name carrybox_bcppo_$(date +%m%d_%H%M)

Evaluation runs on a wall-clock cadence (``--eval-minutes``, default 10) rather than an
iteration count, because iteration time changes over a run and a count does not hold a
cadence. Collider settings default to the ones the throughput and fidelity work settled on:
``--box-tris 2000``, full-resolution hands, ``--margin 0``. See ``rl/README.md``.

The runner writes wandb through rsl_rl's own ``WandbSummaryWriter`` (``logger: wandb``),
so the run name is the log directory's basename and the project comes from
``wandb_project`` in the runner config. Credentials follow the convention used elsewhere in
this workspace: ``WANDB_API_KEY`` from the environment, else ``~/.netrc`` for
``api.wandb.ai``. The key is never printed.
"""

from __future__ import annotations

import argparse
import builtins
import os
import sys
import time
from pathlib import Path

import torch
import warp as wp

HERE = Path(__file__).resolve().parent
SUGAR_SRC = HERE.parents[1] / "SUGAR" / "source" / "sugar_rl"
DEFAULT_TEACHER = (HERE.parents[1] / "experiments/sugar_reproduction/outputs/final"
                   / "official_sugar/baseline/ckpts/refiner_model10000.pt")


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


def attach_video(runner, args) -> None:
    """Render an evaluation rollout on a wall-clock cadence, or every N iterations.

    ``runner.log`` is wrapped rather than ``runner.save``: ``log`` is called once per
    iteration (``on_policy_runner.py:157``) whereas ``save`` fires only on
    ``save_interval``, which is far too coarse to honour a minutes-based interval. A
    failure to render is logged and swallowed -- a missing video must never end a run.

    ``--eval-minutes`` is wall clock, so the cadence holds regardless of how much the
    iteration time changes over a run; ``--video-interval`` is the iteration-count
    alternative. Whichever fires first wins, and the first evaluation happens at the first
    logged iteration so a broken video path is found immediately rather than in an hour.
    """
    from sugar_newton.rl.video import VideoRecorder

    rec = VideoRecorder(clip=(args.clips or ["data_000"])[0], frames=args.video_frames,
                        out_dir=str(Path(args.log_root) / args.run_name / "videos"),
                        mu=args.mu, substeps=args.substeps, device=args.device,
                        tactile=not args.no_tactile_video, canvas_tris=args.canvas_tris,
                        box_tris=args.box_tris, hand_tris=args.hand_tris,
                        margin=args.margin)
    period = args.eval_minutes * 60.0
    original_log = runner.log
    state = {"last_it": -1, "next_t": 0.0}      # next_t = 0 -> evaluate at the first log

    def log_and_record(locs, width: int = 80, pad: int = 35):
        original_log(locs, width, pad)
        it = runner.current_learning_iteration
        if it == state["last_it"]:
            return
        now = time.monotonic()
        due_time = period > 0 and now >= state["next_t"]
        due_iter = args.video_interval > 0 and it % args.video_interval == 0
        if not (due_time or due_iter):
            return
        state["last_it"] = it

        t0 = now
        try:
            video_path, stats = rec.record(runner.alg.policy, it)
        except Exception as exc:
            print(f"[eval] skipped at iter {it}: {type(exc).__name__}: {exc}")
            video_path, stats = None, {}
        # Schedule from the END of the evaluation: rendering plus compositing takes minutes,
        # and scheduling from the start would let a slow eval fire back-to-back forever.
        state["next_t"] = time.monotonic() + period
        if video_path is None:
            return
        took = time.monotonic() - t0
        print(f"[eval] iter {it} ({took:.0f} s): {video_path}  "
              f"lift {stats.get('video/box_lift', 0):.3f} m "
              f"(reference {stats.get('video/box_lift_reference', 0):.3f} m)  "
              f"load-bearing {stats.get('video/load_bearing_contacts', float('nan')):.1f}")
        if args.logger == "wandb":
            import wandb

            if wandb.run is not None:
                fmt = "gif" if video_path.endswith(".gif") else "mp4"
                wandb.log({**stats, "video/eval_seconds": took,
                           "video/rollout": wandb.Video(video_path, fps=rec.fps,
                                                        format=fmt)},
                          step=it)

    runner.log = log_and_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--max-iterations", type=int, default=3000,
                    help="ABSOLUTE target iteration, not a per-leg count: a resumed run "
                         "trains only the remainder, so chained legs converge on one end")
    ap.add_argument("--save-interval", type=int, default=25,
                    help="iterations between checkpoints. Must be well under what one "
                         "allocation reaches (~240 at 512 worlds in 4 h) or a wall-clock "
                         "kill loses the whole leg and a chained run cannot advance")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--episode-length", type=int, default=300)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--teacher-ckpt", default=str(DEFAULT_TEACHER))
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    ap.add_argument("--run-name", default="carrybox_bcppo")
    ap.add_argument("--log-root", default="logs/newton_bcppo")
    ap.add_argument("--logger", default="wandb", choices=("wandb", "tensorboard"))
    ap.add_argument("--wandb-project", default="sugar_newton")
    ap.add_argument("--eval-minutes", type=float, default=10.0,
                    help="wall-clock minutes between evaluation videos; 0 disables")
    ap.add_argument("--video-interval", type=int, default=0,
                    help="also evaluate every N iterations; 0 disables (see --eval-minutes)")
    ap.add_argument("--video-frames", type=int, default=400)
    ap.add_argument("--no-tactile-video", action="store_true",
                    help="scene only, no tactile heatmap panels in the evaluation video")
    ap.add_argument("--canvas-tris", type=int, default=3000,
                    help="triangle count of the flat hand canvas the heatmap is drawn on")
    ap.add_argument("--box-tris", type=int, default=2000,
                    help="decimated box collider triangles; 0 keeps the asset's ~100k")
    ap.add_argument("--hand-tris", type=int, default=0,
                    help="decimated hand collider triangles; 0 keeps the asset's ~45k")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="collider surface thickness [m]; Newton's default is 0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.logger == "wandb":
        ensure_wandb_credentials()
    if not Path(args.teacher_ckpt).is_file():
        raise SystemExit(f"teacher checkpoint not found: {args.teacher_ckpt}\n"
                         "BCPPO's stages 1-2 have no loss without it.")

    sugar_bcppo()
    from rsl_rl.runners import OnPolicyRunner

    from sugar_newton.rl.vec_env import OBS_DIMS, make

    # ---- multi-GPU ---------------------------------------------------------------------
    # Under torchrun, rsl_rl reads WORLD_SIZE/LOCAL_RANK/RANK itself and raises unless the
    # device is exactly cuda:LOCAL_RANK, so derive it here rather than trusting --device.
    # BCPPO subclasses PPO and forwards **kwargs, so the multi_gpu_cfg the runner passes
    # reaches the all-reduce paths already in its update loop; nothing else is needed.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        args.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

    wp.init()
    # Each rank must see DIFFERENT worlds. Data-parallel training averages gradients, so
    # eight ranks seeded alike would step eight identical rollouts and the 8x batch would
    # carry 1x the information -- a silent waste of seven GPUs rather than a crash.
    torch.manual_seed(args.seed + rank)

    env = make(args.num_envs, clip_names=args.clips, episode_length=args.episode_length,
               substeps=args.substeps, mu=args.mu, device=args.device,
               seed=args.seed + rank,
               box_tris=args.box_tris, hand_tris=args.hand_tris, margin=args.margin)
    if rank == 0:
        total = args.num_envs * world_size
        print(f"[env] {args.num_envs} worlds x {world_size} rank(s) = {total} worlds, "
              f"{len(env.env.clip_names)} clips, obs {OBS_DIMS}, act {env.num_actions}")
        print(f"[env] box_tris={args.box_tris} hand_tris={args.hand_tris} "
              f"margin={args.margin} mu={args.mu} substeps={args.substeps}")
        print(f"[env] batch/iteration = {24 * total} samples")

    log_dir = Path(args.log_root) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg(args), log_dir=str(log_dir), device=args.device)
    if rank == 0:
        print(f"[alg] {type(runner.alg).__name__}  teacher={args.teacher_ckpt}"
              + (f"  (DDP, {world_size} ranks)" if world_size > 1 else ""))
    if args.resume:
        runner.load(args.resume)
        if rank == 0:
            print(f"[alg] resumed from {args.resume} at iteration "
                  f"{runner.current_learning_iteration}")

    # Rank 0 only: the recorder builds a second environment and writes the video, and
    # rsl_rl calls `log` on rank 0 alone, so attaching elsewhere would allocate a spare env
    # per GPU that never renders. The other ranks wait at the next all-reduce while rank 0
    # records -- harmless at a few minutes, but it is why eval must stay well inside NCCL's
    # timeout (raised in the launcher).
    if rank == 0 and (args.eval_minutes > 0 or args.video_interval > 0):
        attach_video(runner, args)

    # `learn` takes a COUNT and computes `tot_iter = current + count`
    # (on_policy_runner.py:96), so passing --max-iterations after a resume would extend the
    # endpoint by that much again. Every 4 h leg of a chained run would then add another
    # full budget and the run would never reach a fixed end. --max-iterations is an ABSOLUTE
    # target here, so the count has to be the remainder.
    todo = args.max_iterations - runner.current_learning_iteration
    if todo <= 0:
        if rank == 0:
            print(f"[alg] already at iteration {runner.current_learning_iteration} of "
                  f"{args.max_iterations}; nothing to do")
        return
    if rank == 0:
        print(f"[alg] training {todo} iterations to reach {args.max_iterations}")
    runner.learn(num_learning_iterations=todo, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
