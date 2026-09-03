# SPDX-License-Identifier: BSD-3-Clause
"""Train SUGAR's Refiner on Newton, using SUGAR's own configuration objects.

    python -m sugar_swap.train --num-envs 512                     # one GPU
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
        -m sugar_swap.train --num-envs 4096                       # eight GPUs, 512 each

Every learning hyperparameter and every MDP term comes out of SUGAR's gym registry. Nothing
in this file names a reward weight, a learning rate or a network width, and that is the whole
point of it. The previous attempt (``sugar_newton/rl/train_refiner.py``) transcribed
``BasePPORunnerCfg`` field by field and hand-ported the environment; the transcription of the
algorithm survived audit, but the environment's reward function did not -- four terms were
missing, three regularisation weights were off by factors between 1e-7 and 0.1, and IsaacLab's
``weight * dt`` convention was never applied. Reading the configs instead of retyping them
makes that class of error impossible rather than merely unlikely.

Two entry points off one registry id, both fetched by name:

===========================  =============================================================
``env_cfg_entry_point``      ``...train_refiner.carry_box_refiner_env_cfg:RobotEnvCfg``
``rsl_rl_cfg_entry_point``   ``sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg``
``play_env_cfg_entry_point`` ``...carry_box_refiner_env_cfg:RobotPlayEnvCfg`` (evaluation only)
===========================  =============================================================

The agent config is handed to ``OnPolicyRunner`` through ``agent_cfg.to_dict()``, which is
exactly what SUGAR's own ``scripts/sugar_rl/train.py`` does. The refiner is **plain PPO**:
``BasePPORunnerCfg`` declares ``class_name="PPO"`` over an ``ActorCritic`` whose policy group
*is* the privileged group (``base_refiner_env_cfg.py`` sets ``policy = critic =
PrivilegedCfg``), so the actor reads the same 890-D vector as the critic and there is no
teacher. BCPPO is the tracker's distillation stage and has no business here.

What this file adds to SUGAR's configuration, and nothing else:

* ``--num-envs`` / ``--max-iterations`` / ``--seed`` / ``--save-interval`` overrides, applied
  the way ``SUGAR/scripts/sugar_rl/cli_args.py`` applies them.
* wandb logging with a run identity that survives a chained SLURM run
  (``sugar_newton.rl.run_dir``).
* A periodic evaluation video on a wall-clock cadence, rendered by
  ``sugar_newton.rl.video.VideoRecorder`` against a one-environment evaluation env.

Multi-GPU (see ``--num-envs`` and :func:`ddp_ranks`):

* ``--num-envs`` is a **TOTAL** across ranks, never per-rank, because that is what the config
  field it overrides means -- ``scene.num_envs=4096`` is SUGAR's whole batch. Under torchrun
  it is divided by ``WORLD_SIZE``, so 8 ranks of the default build 512 worlds each and
  reproduce SUGAR's 4096 x 24 batch exactly rather than an 8x larger one.
* rsl_rl's ``OnPolicyRunner`` reads ``WORLD_SIZE`` / ``LOCAL_RANK`` / ``RANK`` itself and
  raises unless ``device == cuda:LOCAL_RANK`` (``on_policy_runner.py:378``), so the device is
  *derived* from the environment here rather than taken from ``--device``. It then calls
  ``init_process_group``, ``broadcast_parameters()`` at the top of ``learn()``, and
  all-reduces the gradients inside ``PPO.update()``; nothing in this file re-implements any
  of that.
* Newton is pointed at the local rank's device three ways -- ``torch.cuda.set_device``,
  ``wp.set_device`` for Warp's default, and ``sim.device`` for the model itself. Getting the
  last one wrong puts all eight models on ``cuda:0`` and looks like a memory leak.
* Rank 0 alone mints the wandb run, dumps the configs and records evaluation video; rsl_rl
  disables its own logging and checkpointing on the other ranks (``disable_logs``).

Deviations from SUGAR, recorded rather than hidden:

* **Batch size.** SUGAR registers ``num_envs=4096``, so its PPO batch is 4096 x 24 = 98304
  samples per update. ``--num-envs`` changes that, and the batch is a learning hyperparameter,
  not an operational one. The value actually used is printed at startup and logged to wandb.
* **``--save-interval``.** SUGAR's is 1000. A 4 h SLURM leg does not reach 1000 iterations, so
  a preempted leg would lose everything; 25 is what the chained launcher uses. Checkpoint
  frequency does not affect the optimisation.
* **``--max-iterations`` is an absolute target, not an increment.** ``runner.learn()`` takes a
  *count* and computes ``tot_iter = current_learning_iteration + count``, so a resumed leg
  that asked for the full budget would extend the endpoint by a whole budget every leg and the
  run would never finish. The remainder is computed below.

``bootstrap.install()`` runs before any ``isaaclab`` import, as it must; see
``sugar_swap/README.md`` §1.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TASK = "Sugar-G129dof-CarryBox-Refiner"
ENV_CFG_KEY = "env_cfg_entry_point"
AGENT_CFG_KEY = "rsl_rl_cfg_entry_point"
EVAL_CFG_KEY = "play_env_cfg_entry_point"

# SUGAR leaves `motion_folder` as None in the config and supplies it from its launchers, which
# run with SUGAR/ as the working directory (`SUGAR/train.sh:16`).
DEFAULT_MOTION_FOLDER = REPO / "SUGAR" / "data" / "CarryBox"

# IsaacLab and SUGAR are importable by path rather than installed; `sugar_il` is needed even
# for a config-only run, because SUGAR's `locomanip/mdp/commands.py` imports its
# diffusion-policy wrapper at module scope (README §1).
_IMPORT_PATHS = (
    REPO,
    REPO / "IsaacLab" / "source" / "isaaclab",
    REPO / "IsaacLab" / "source" / "isaaclab_tasks",
    REPO / "IsaacLab" / "source" / "isaaclab_rl",
    REPO / "SUGAR" / "source" / "sugar_rl",
    REPO / "SUGAR" / "source" / "sugar_il",
)


# ---------------------------------------------------------------------------------------
# multi-GPU
# ---------------------------------------------------------------------------------------
def ddp_ranks() -> tuple[int, int, int]:
    """``(local_rank, global_rank, world_size)`` from torchrun's environment.

    Read here rather than passed on the command line, and read from the same three variables
    ``OnPolicyRunner._configure_multi_gpu`` reads, so there is exactly one source of truth for
    which GPU this process owns. ``(0, 0, 1)`` outside torchrun, which is what makes the
    single-process path below identical to what it was before DDP existed.
    """
    return (
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def pin_device(local_rank: int, world_size: int) -> str:
    """Point torch, Warp and this process's CUDA context at the local rank's GPU.

    Three separate defaults have to move, and the failure modes differ:

    * ``torch.cuda.set_device`` -- fixes where a bare ``cuda`` tensor and NCCL's own buffers
      land. Called before the first allocation so no context is ever created on ``cuda:0``.
    * ``wp.set_device`` -- Warp's default device, used by any allocation that does not name
      one. Newton is careful (``collide.py:652`` and ``solver_mujoco.py:6063`` both open a
      ``wp.ScopedDevice(model.device)``), so this is belt-and-braces rather than load-bearing;
      it costs nothing and it is the one that would otherwise fail silently.
    * the returned string, which becomes ``sim.device`` and therefore
      ``ModelBuilder.finalize(device=...)`` -- this is the load-bearing one. Get it wrong and
      all eight ranks build their Newton model on ``cuda:0``: no error, no NCCL complaint,
      just one GPU at 8x the memory and seven idle, which reads as a leak rather than as a
      placement bug.

    Returns the device string every later config field is set from.
    """
    import torch

    device = f"cuda:{local_rank}" if world_size > 1 else None
    if device is None:
        return ""
    torch.cuda.set_device(local_rank)
    import warp as wp

    wp.set_device(device)
    return device


def ddp_barrier() -> None:
    """Wait for every rank, if there are others. A no-op single-process."""
    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _digest(tensor) -> str:
    """A short exact hash of a tensor's bytes, for comparing ranks bit-for-bit.

    Bytes rather than a sum: two different parameter vectors can share a float32 sum, and the
    claim being checked -- "these ranks hold the same policy" -- is a bitwise one.
    """
    raw = tensor.detach().to("cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def attach_ddp_verify(runner, iterations: int, rank: int, world_size: int) -> None:
    """Print, per rank and per iteration, a hash of the rollout and a hash of the policy.

    This is the check that tells data-parallel PPO apart from ``world_size`` independent
    trainers that happen to share a log directory, which is a failure with no symptom until
    the reward curve is inexplicably worse than the single-GPU one:

    * the **rollout** hashes must DIFFER between ranks. Each rank seeds its environment with
      ``seed + rank`` precisely so they do; ranks stepping identical worlds would make the
      8x batch carry 1x the information.
    * the **policy** hashes must be IDENTICAL after the update. They can only be identical if
      the gradients were all-reduced, because the inputs to the update were different.

    Wrapped around ``alg.update`` rather than ``runner.log`` because ``log`` runs on rank 0
    alone, and a check that only one rank performs cannot compare ranks. The rollout hash is
    taken before ``update()`` because ``update()`` ends by clearing the storage.
    """
    import torch

    original = runner.alg.update
    # The iteration label is counted from here rather than read back from the runner:
    # `learn()` assigns `current_learning_iteration = it` only AFTER `update()` returns, so
    # reading it inside the wrapper is one behind and the error accumulates.
    state = {"n": 0, "start": int(runner.current_learning_iteration)}

    def update_and_report():
        if state["n"] >= iterations:
            return original()
        it = state["start"] + state["n"]
        rollout = _digest(runner.alg.storage.rewards)
        loss = original()
        params = _digest(
            torch.cat([p.detach().reshape(-1) for p in runner.alg.policy.parameters()])
        )
        print(f"[ddp-verify] iter {it} rank {rank}/{world_size} device "
              f"{runner.device} rollout {rollout} params {params}", flush=True)
        state["n"] += 1
        return loss

    runner.alg.update = update_and_report


# ---------------------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------------------
def install_swap(verbose: bool = False) -> None:
    """Put Newton behind IsaacLab's import graph, then let SUGAR register its tasks.

    Must precede every ``isaaclab`` import, and the order inside it matters as much as the
    order relative to it: importing ``sugar_rl.tasks`` is what populates the gym registry, and
    it has to happen after the substitution because SUGAR's config modules import
    ``isaaclab.assets`` and friends at module scope. Without that import
    ``load_cfg_from_registry`` raises ``KeyError`` on the task id -- the registry is simply
    empty. SUGAR's own ``scripts/sugar_rl/train.py:168`` has the same line for the same reason.
    """
    for path in _IMPORT_PATHS:
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from sugar_swap import bootstrap

    bootstrap.install(verbose=verbose)
    _shim_direct_rl_env()

    import sugar_rl.tasks  # noqa: F401


def _shim_direct_rl_env() -> None:
    """Bind the one name ``isaaclab_rl`` imports that sugar_swap's ``isaaclab.envs`` lacks.

    ``isaaclab_rl.rsl_rl.vecenv_wrapper`` opens with ``from isaaclab.envs import DirectRLEnv,
    ManagerBasedRLEnv`` and then uses ``DirectRLEnv`` only in one ``isinstance`` check.
    sugar_swap shadows ``isaaclab.envs`` with the manager-based workflow alone, so that import
    fails on a name nothing in this process can ever be an instance of.

    A placeholder is the cheap fix and reimplementing the wrapper is the expensive one:
    ``RslRlVecEnvWrapper`` is what converts the observation dict into rsl_rl's ``TensorDict``,
    derives ``dones`` from ``terminated | truncated``, and moves the time-outs into
    ``extras["time_outs"]`` -- and getting that last one wrong makes the critic bootstrap
    through a truncation, silently, forever. Reuse it.

    Guarded on ``hasattr`` so it becomes a no-op the moment ``sugar_swap/env.py`` grows a
    ``DirectRLEnv`` of its own.
    """
    import isaaclab.envs

    if hasattr(isaaclab.envs, "DirectRLEnv"):
        return

    class DirectRLEnv:
        """Unimplemented. sugar_swap provides IsaacLab's manager-based workflow only."""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "sugar_swap: the direct RL workflow has no Newton backend; SUGAR uses the "
                "manager-based one."
            )

    isaaclab.envs.DirectRLEnv = DirectRLEnv


# ---------------------------------------------------------------------------------------
# configuration, straight off the registry
# ---------------------------------------------------------------------------------------
def load_agent_cfg(task: str, args: argparse.Namespace):
    """SUGAR's ``BasePPORunnerCfg``, with only the fields ``cli_args.py`` overrides touched.

    The ``experiment_name`` default and the ``""`` -> task-id derivation are
    ``cli_args.parse_rsl_rl_cfg`` / ``update_rsl_rl_cfg``; keeping them means the agent.yaml
    this run dumps is comparable with one from SUGAR's own launcher.
    """
    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry(task, AGENT_CFG_KEY)

    if cfg.experiment_name == "":
        cfg.experiment_name = task.lower().replace("-", "_").removesuffix("_play")
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    if args.save_interval is not None:
        cfg.save_interval = args.save_interval
    cfg.device = args.device
    cfg.logger = args.logger
    cfg.wandb_project = args.wandb_project
    cfg.resume = bool(args.resume)
    return cfg


def load_env_cfg(task: str, key: str, args: argparse.Namespace, *, num_envs: int | None,
                 seed: int, world_size: int = 1):
    """One of SUGAR's registered env configs, with the run-time bindings SUGAR also sets.

    ``num_envs`` is the TOTAL across ranks -- ``None`` meaning SUGAR's registered total -- and
    what lands in ``scene.num_envs`` is that total divided by ``world_size``, because each
    rank builds its own model and steps its own share of the batch. Keeping the argument a
    total is what makes ``--num-envs`` mean the same thing at any rank count: it is an
    override of ``scene.num_envs=4096``, and 4096 is SUGAR's whole batch. Were it per-rank,
    ``--num-envs 4096`` on 8 GPUs would silently be an 8x larger batch than SUGAR's, i.e. a
    different experiment wearing the same command line.

    The division has to be exact. A remainder would give the ranks unequal env counts, and
    since rsl_rl averages gradients with an unweighted mean the ranks' samples would carry
    different weights -- an effective batch that is neither the requested one nor an obvious
    error, so it raises instead.
    """
    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry(task, key)
    total = int(cfg.scene.num_envs if num_envs is None else num_envs)
    if world_size > 1 and total % world_size:
        raise SystemExit(
            f"--num-envs is a TOTAL across ranks and must divide the world size: "
            f"{total} envs over {world_size} ranks leaves {total % world_size}. "
            f"Try {total - total % world_size} or {total + world_size - total % world_size}."
        )
    cfg.scene.num_envs = total // world_size
    cfg.commands.motion.motion_folder = str(args.motion_folder)
    # `train.py:205`: the environment seed is the agent's, because randomisations happen during
    # construction and have to be reproducible with the policy init.
    cfg.seed = seed
    cfg.sim.device = args.device
    return cfg


def eval_env_cfg(task: str, args: argparse.Namespace, *, seed: int):
    """SUGAR's play config plus the two changes an evaluation video needs.

    Returns ``(cfg, tracking_terms)`` where ``tracking_terms`` are the termination terms that
    were removed, so the recorder can still *report* tracking loss without acting on it.

    **Reference-tracking terminations off.** With them on, an untrained policy leaves the
    reference within a handful of frames, resets, and the video is the same half second on
    repeat -- which says nothing about what the policy does. Selection is by
    ``time_out=False``: of the six terms in ``BaseTerminationsCfg``, only
    ``trajectory_complete`` is a time-out, and it has to stay, because it is what stops the
    motion index running past the end of the clip.

    **Start state pinned.** ``eval_mode=True`` makes SUGAR's own ``MotionCommand`` treat every
    environment as a "protected" one (``commands.py:254``), which pins ``motion_id = env_id %
    num_motion`` and ``time_step = 0``. At ``num_envs=1`` that is exactly ``motion=0,
    start=0``, on every reset, and it is SUGAR's code doing it rather than a monkeypatch of it.
    Without this the single environment falls in the 75 % "pool" bucket and each evaluation
    starts from a different sampled state, so successive videos are not comparable.

    **Random pushes off**, for the same comparability reason and with the same precedent:
    SUGAR's own ``RobotRolloutPlayEnvCfg`` sets ``push_robot`` and ``push_object`` to None. At
    ``interval_range_s=(1.5, 3.0)`` a 400-frame rollout would otherwise absorb three or four
    random impulses that differ between evaluations.
    """
    cfg = load_env_cfg(task, EVAL_CFG_KEY, args, num_envs=1, seed=seed)
    cfg.commands.motion.eval_mode = True

    tracking: dict[str, object] = {}
    for name, term in list(vars(cfg.terminations).items()):
        if name.startswith("_") or term is None or getattr(term, "time_out", False):
            continue
        tracking[name] = term
        setattr(cfg.terminations, name, None)

    for name in ("push_robot", "push_object"):
        if getattr(cfg.events, name, None) is not None:
            setattr(cfg.events, name, None)

    return cfg, tracking


# ---------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------
class _EvalEnv:
    """A sugar_swap env wearing the attribute surface ``VideoRecorder`` was written against.

    ``sugar_newton.rl.video.VideoRecorder`` drives ``CarryBoxEnv``. Everything it does --
    aim the camera at the pelvis/box midpoint, build the per-hand tactile atlas, composite the
    HUD, fall back from mp4 to gif when the container has no ffmpeg -- is backend-independent.
    Only the handful of attributes below are not, so supplying them keeps ``record()``,
    ``_composite()``, ``_film()``, ``_aim()`` and ``_act_mean()`` in use verbatim instead of
    reimplemented. The tactile heatmap in particular is the thing that tells a real grasp from
    a wrist wedged under the box, and it is not worth rewriting.

    The Newton handles are properties, not attributes: the solver swaps ``state_0`` and
    ``state_1`` on every substep, so a captured reference goes stale after one step.
    """

    def __init__(self, env, tracking_terms: dict, *, start: int, motion: int):
        import torch

        from isaaclab.managers import TerminationManager

        self._env = env
        self._start, self._motion = start, motion
        self.device = env.device
        self._command = env.command_manager.get_term("motion")
        # `_body_index` is the per-environment body row; at num_envs=1 it is also the row in
        # Newton's flat body arrays, which is what `_aim` indexes.
        self.box_body = int(env.scene["obj"]._body_index)
        # (num_motion, T, 3), straight off SUGAR's motion loader. `record()` reads the peak of
        # the reference object height to report the lift as a fraction of what the clip does.
        self.ref = {"obj_pos": self._command.motion.obj_pos}

        # The five terms `eval_env_cfg` removed, rebuilt as a manager that only *reports*.
        # An IsaacLab manager rather than a direct call to `term.func`, because it is what
        # resolves the `SceneEntityCfg` bodies in each term's params -- the drift readout then
        # uses SUGAR's own functions with SUGAR's own thresholds and no transcribed 0.3.
        self._tracking = (
            TerminationManager(types.SimpleNamespace(**tracking_terms), env)
            if tracking_terms
            else None
        )
        self.drifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    @property
    def model(self):
        return self._env.scene.model

    @property
    def state_0(self):
        return self._env.scene.state_0

    @property
    def contacts(self):
        return self._env.scene.contacts

    @property
    def solver(self):
        return self._env.scene.solver

    @property
    def motion_id(self):
        return self._command.motion_id

    def _body_q(self):
        return self._env.scene.body_q()

    def observe(self):
        """The observation manager's own output, as the TensorDict rsl_rl's policy expects.

        Not a hand-assembled vector. The refiner's actor reads the ``policy`` group and that
        group *is* the 890-D privileged one, so anything reassembled here would risk feeding
        the network a differently-ordered input than the one it is being trained on -- which
        is exactly how the hand-written port's 890-D vector went wrong.
        """
        from tensordict import TensorDict

        obs = self._env.observation_manager.compute()
        return TensorDict(dict(obs), batch_size=[self._env.num_envs], device=self.device)

    def reset(self, env_ids=None, start: int | None = None, motion: int | None = None):
        """Reset and *check* the pin, rather than force it.

        ``eval_mode`` is what does the pinning, so this asserts SUGAR still behaves that way.
        Forcing the indices here instead would keep the video comparable while hiding a config
        regression that also changes what training sees.
        """
        start = self._start if start is None else start
        motion = self._motion if motion is None else motion
        self._env.reset()
        got = (int(self._command.time_steps[0]), int(self._command.motion_id[0]))
        if got != (start, motion):
            raise RuntimeError(
                f"sugar_swap.train: evaluation start state is not pinned -- wanted "
                f"(start={start}, motion={motion}), got (start={got[0]}, motion={got[1]}). "
                "Check commands.motion.eval_mode is still True on the evaluation config."
            )
        self._refresh_drift()

    def step(self, action):
        out = self._env.step(action)
        self._refresh_drift()
        return out

    def _refresh_drift(self) -> None:
        if self._tracking is not None:
            self.drifted = self._tracking.compute()


def build_recorder(args, task: str, seed: int):
    """A ``VideoRecorder`` whose environment is a one-world sugar_swap env."""
    from sugar_newton.rl.video import VideoRecorder

    class SwapVideoRecorder(VideoRecorder):
        """Only ``_ensure`` and ``_policy_obs`` differ from the Newton-native recorder."""

        def _ensure(self) -> bool:
            if self.viewer is not None:
                return True
            try:
                import pyglet

                if os.environ.get("G1_XVFB") != "1":
                    pyglet.options["headless"] = True
                from newton.viewer import ViewerGL

                from isaaclab.envs import ManagerBasedRLEnv

                cfg, tracking = eval_env_cfg(task, args, seed=seed)
                raw = ManagerBasedRLEnv(cfg, device=args.device)
                self.env = _EvalEnv(raw, tracking, start=self.start, motion=0)
                self.viewer = ViewerGL(headless=os.environ.get("G1_XVFB") != "1")
                self.viewer.set_model(self.env.model)
                if os.environ.get("PYOPENGL_PLATFORM") != "egl" and not self._warned:
                    print("[video] PYOPENGL_PLATFORM is not egl; rendering may fall back to "
                          "software and be very slow. Source slurm/render_env_egl.sh.")
                    self._warned = True
                return True
            except Exception as exc:              # never take training down for a video
                if not self._warned:
                    import traceback

                    print(f"[video] disabled: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    self._warned = True
                return False

        def _policy_obs(self, policy, env):
            return env.observe()

    return SwapVideoRecorder(
        # `clip` is only a HUD label here; `eval_mode` is what selects the motion, and at
        # num_envs=1 it selects index 0.
        clip="motion_id=0",
        start=0,
        frames=args.video_frames,
        out_dir=str(Path(args.log_root) / args.run_name / "videos"),
        device=args.device,
        tactile=not args.no_tactile_video,
        canvas_tris=args.canvas_tris,
        privileged_policy=True,
    )


def attach_eval(runner, recorder, args) -> None:
    """Render an evaluation rollout every ``--eval-minutes`` of wall clock.

    ``runner.log`` is wrapped rather than ``runner.save``: ``log`` fires once per iteration
    whereas ``save`` fires on ``save_interval``, which is far too coarse to hold a
    minutes-based cadence. Wall clock rather than an iteration count because iteration time
    changes a lot over a run and a count does not hold a cadence.

    Scheduling is from the *end* of the evaluation. Rendering plus the matplotlib composite
    takes minutes, and scheduling from the start would let a slow evaluation fire
    back-to-back forever. The first one happens at the first logged iteration, so a broken
    render path shows up in the first minute rather than in an hour.

    A failure to render is printed and swallowed. A missing video must never end a run.
    """
    period = args.eval_minutes * 60.0
    original_log = runner.log
    state = {"last_it": -1, "next_t": 0.0}

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
            video_path, stats = recorder.record(runner.alg.policy, it)
        except Exception as exc:
            print(f"[eval] skipped at iter {it}: {type(exc).__name__}: {exc}")
            video_path, stats = None, {}
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
                           "video/rollout": wandb.Video(video_path, fps=recorder.fps,
                                                        format=fmt)},
                          step=it)

    runner.log = log_and_record


# ---------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", default=TASK, help="gym id registered by sugar_rl.tasks")
    ap.add_argument("--num-envs", type=int, default=None,
                    help="TOTAL environments across all ranks, NOT per rank: overrides "
                         "SUGAR's registered scene.num_envs (4096). Under torchrun it is "
                         "divided by WORLD_SIZE, so --num-envs 4096 on 8 GPUs builds 512 "
                         "per rank and reproduces SUGAR's 4096 x 24 batch. This changes the "
                         "PPO batch (num_envs x 24) and is therefore a real deviation")
    ap.add_argument("--max-iterations", type=int, default=None,
                    help="ABSOLUTE target iteration, not a per-leg count: a resumed run "
                         "trains only the remainder. Default is SUGAR's own (30001)")
    ap.add_argument("--save-interval", type=int, default=25,
                    help="iterations between checkpoints. SUGAR's is 1000, which a 4 h SLURM "
                         "leg never reaches; 25 bounds what a preempted leg redoes")
    ap.add_argument("--seed", type=int, default=None,
                    help="default is SUGAR's agent-config seed (42)")
    ap.add_argument("--resume", default="", help="checkpoint (.pt) to resume from")
    ap.add_argument("--motion-folder", default=str(DEFAULT_MOTION_FOLDER),
                    help="reference motion clips; SUGAR passes this as --motion_folder")
    ap.add_argument("--run-name", default="carrybox_refiner_swap")
    ap.add_argument("--log-root", default="logs/swap_refiner")
    ap.add_argument("--logger", default="wandb", choices=("wandb", "tensorboard"))
    ap.add_argument("--wandb-project", default="sugar_newton")
    ap.add_argument("--eval-minutes", type=float, default=20.0,
                    help="wall-clock minutes between evaluation videos; 0 disables")
    ap.add_argument("--video-interval", type=int, default=0,
                    help="also evaluate every N iterations; 0 disables")
    ap.add_argument("--video-frames", type=int, default=400)
    ap.add_argument("--no-tactile-video", action="store_true",
                    help="scene only, no tactile heatmap panels")
    ap.add_argument("--canvas-tris", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0",
                    help="ignored under torchrun, where the device must be cuda:LOCAL_RANK")
    ap.add_argument("--ddp-verify", type=int, default=0, metavar="N",
                    help="for the first N updates, every rank prints a hash of its rollout "
                         "and of the policy after the update. The rollout hashes must DIFFER "
                         "and the policy hashes must MATCH; that is the check that catches a "
                         "'DDP' that is really N independent trainers")
    ap.add_argument("--report-ignored", action="store_true",
                    help="print the IsaacLab config fields the Newton backend drops")
    ap.add_argument("--dry-run", action="store_true",
                    help="load both configs, build the runner, then exit without training")
    return ap


def _count_terms(group) -> int:
    """Live terms in a manager config group; a term set to None is disabled, not present."""
    return sum(1 for n, t in vars(group).items() if not n.startswith("_") and t is not None)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.motion_folder = Path(args.motion_folder)
    if not args.motion_folder.is_dir():
        raise SystemExit(
            f"motion folder not found: {args.motion_folder}\n"
            "Run `bash SUGAR/_downloads/fetch_assets.sh` (see ASSETS.md)."
        )

    local_rank, rank, world_size = ddp_ranks()
    lead = rank == 0

    def say(*parts: object) -> None:
        """Print on rank 0 only. Eight copies of the config banner is noise, not evidence."""
        if lead:
            print(*parts)

    if args.logger == "wandb":
        # Fail here rather than several minutes into a run with logging silently off. The
        # launcher re-sources the credential file AFTER the container's profile scripts have
        # run, because /root/.bashrc exports a stale WANDB_API_KEY over the good one.
        # Checked on every rank although only rank 0 logs: the ranks share one environment,
        # so a rank that cannot see the key means rank 0 probably cannot either, and finding
        # that out now costs nothing.
        from sugar_newton.rl.train_bcppo import ensure_wandb_credentials

        ensure_wandb_credentials()

    import torch

    # Before install_swap, and before anything else can allocate: the point of pinning the
    # device is that no CUDA context is ever created on cuda:0 by a rank that does not own it.
    device = pin_device(local_rank, world_size)
    if device:
        args.device = device

    install_swap(verbose=args.report_ignored)

    # SUGAR sets these before building anything; they change the numerics of every matmul.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    agent_cfg = load_agent_cfg(args.task, args)
    env_cfg = load_env_cfg(args.task, ENV_CFG_KEY, args, num_envs=args.num_envs,
                           seed=agent_cfg.seed, world_size=world_size)

    # `+ rank`, so the ranks step DIFFERENT worlds. Data-parallel PPO averages gradients over
    # the ranks, so eight ranks seeded alike would collect eight copies of one rollout and the
    # 8x batch would carry 1x the information -- seven GPUs wasted with nothing to show for it
    # in any log. The policy is unaffected: `learn()` broadcasts rank 0's parameters to every
    # rank before the first iteration (`on_policy_runner.py:92`), so the differing init here
    # is overwritten and only the environment randomisation and motion sampling keep the
    # offset. `--seed` therefore still names one run, not one run per rank.
    torch.manual_seed(agent_cfg.seed + rank)

    num_envs = int(env_cfg.scene.num_envs)
    total_envs = num_envs * world_size
    steps = int(agent_cfg.num_steps_per_env)
    say(f"[cfg] task {args.task}")
    say(f"[cfg] env    {ENV_CFG_KEY} -> {type(env_cfg).__module__}:{type(env_cfg).__name__}")
    say(f"[cfg] agent  {AGENT_CFG_KEY} -> "
        f"{type(agent_cfg).__module__}:{type(agent_cfg).__name__}")
    # The batch line is the one a reader must not be able to misread, so it names the total,
    # the per-rank share and the rank count explicitly rather than leaving any of the three
    # to be inferred from --num-envs.
    say(f"[cfg] BATCH {total_envs} envs TOTAL = {world_size} rank(s) x {num_envs} envs/rank; "
        f"{total_envs} x {steps} steps = {total_envs * steps} samples/update "
        f"(SUGAR registers 4096 envs total = {4096 * steps})")
    say(f"[cfg] dt={env_cfg.sim.dt} decimation={env_cfg.decimation} -> "
        f"{1.0 / (env_cfg.sim.dt * env_cfg.decimation):.0f} Hz control, "
        f"episode {env_cfg.episode_length_s} s")
    say(f"[cfg] {_count_terms(env_cfg.rewards)} reward terms, "
        f"{_count_terms(env_cfg.terminations)} termination terms, "
        f"{_count_terms(env_cfg.events)} event terms")
    say(f"[alg] {agent_cfg.algorithm.class_name} / {agent_cfg.policy.class_name}  "
        f"lr={agent_cfg.algorithm.learning_rate} schedule={agent_cfg.algorithm.schedule} "
        f"desired_kl={agent_cfg.algorithm.desired_kl} gamma={agent_cfg.algorithm.gamma} "
        f"lam={agent_cfg.algorithm.lam} epochs={agent_cfg.algorithm.num_learning_epochs} "
        f"minibatches={agent_cfg.algorithm.num_mini_batches}")
    if world_size > 1:
        # Every rank prints this one, and it is the placement evidence: eight lines naming
        # eight distinct devices, cross-checkable against nvidia-smi.
        print(f"[ddp] rank {rank}/{world_size} local_rank {local_rank} -> {args.device}, "
              f"{num_envs} envs, seed {agent_cfg.seed + rank}", flush=True)

    env = ManagerBasedRLEnv(env_cfg, device=args.device)
    # `gym.Env.unwrapped` returns self, and `RslRlVecEnvWrapper` asks for it on the raw env.
    # sugar_swap's env is a plain object rather than a gym.Env, so supply the identity the
    # wrapper expects instead of routing the construction through `gym.make` -- which would
    # add gym's wrapper chain and its passive env checker for no benefit here.
    if not hasattr(env, "unwrapped"):
        env.unwrapped = env
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_dir = Path(args.log_root) / args.run_name
    if lead:
        log_dir.mkdir(parents=True, exist_ok=True)
    if args.logger == "wandb":
        # One wandb run across every leg of a chained SLURM run; the id lives in
        # run_meta.json beside the checkpoints. See sugar_newton/rl/run_dir.py. `rank` is
        # what stops the other seven ranks minting seven wandb runs: bind_wandb_run returns
        # immediately off rank 0, and rsl_rl builds no writer there either (`disable_logs`).
        from sugar_newton.rl.run_dir import bind_wandb_run

        bind_wandb_run(log_dir, project=args.wandb_project, stage="refiner_swap", rank=rank)

    # `to_dict()` is SUGAR's own conversion (`scripts/sugar_rl/train.py:312`). `obs_groups` is
    # unset on BasePPORunnerCfg and arrives as {}, which rsl_rl's `resolve_obs_groups` fills
    # from the environment's own group names -- {"policy": ["policy"], "critic": ["critic"]},
    # which is right for a refiner whose policy group *is* the privileged group.
    #
    # This constructor is also where distributed training is set up: it reads WORLD_SIZE /
    # LOCAL_RANK / RANK itself, calls init_process_group, and raises unless `device` is
    # exactly cuda:LOCAL_RANK -- which is why agent_cfg.device came from pin_device above.
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=str(log_dir),
                            device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    actor_in = runner.alg.policy.actor[0].in_features
    critic_in = runner.alg.policy.critic[0].in_features
    say(f"[alg] actor {actor_in} -> {vec_env.num_actions}, critic {critic_in} -> 1, "
        f"obs_groups {runner.cfg['obs_groups']}")
    if world_size > 1:
        say(f"[alg] DDP: {world_size} ranks, gradients all-reduced in PPO.update "
            f"(ppo.py:375), parameters broadcast from rank 0 at the top of learn()")

    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if not os.path.isfile(resume_path):
            raise SystemExit(f"resume checkpoint does not exist: {resume_path}")
        # EVERY rank loads, not just rank 0. `learn()` derives its loop bounds from
        # `current_learning_iteration`, so a rank that skipped the load would run a different
        # number of iterations and the run would hang at an all-reduce rather than fail.
        # `map_location` is the other half: the checkpoint was saved by rank 0 and its tensors
        # carry cuda:0, so an unmapped load has all eight ranks allocate on GPU 0 -- the same
        # silent single-GPU pile-up that a wrong sim.device causes, arriving by a different
        # route.
        runner.load(resume_path, map_location=args.device)
        # rsl_rl saves the label of the COMPLETED iteration (`on_policy_runner.py:153,160`
        # set `current_learning_iteration = it` and then write `model_{it}.pt`), and `load`
        # restores that same label. Its next `learn()` therefore starts at `it` again: one
        # duplicate optimizer update per leg, and `model_{it}.pt` rewritten. Advance the
        # label instead, exactly as SUGAR's own train.py does (`train.py:522`). It also makes
        # the "already finished" test below exact rather than one iteration short. Done on
        # every rank, for the same reason the load is.
        completed = int(runner.current_learning_iteration)
        runner.current_learning_iteration = completed + 1
        say(f"[alg] resumed from {resume_path}: checkpoint completed iteration "
            f"{completed}, next is {runner.current_learning_iteration}")

    if lead:
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)

    # Rank 0 only: the recorder builds a SECOND Newton environment and writes the video, so
    # attaching it elsewhere would allocate a spare env per GPU that never renders. rsl_rl
    # calls `log` on rank 0 alone anyway (`disable_logs`), so the wrapper would never fire.
    # The other ranks wait at the next all-reduce while rank 0 records, which is why the
    # launcher raises the NCCL heartbeat timeout well past the ~2 min an evaluation costs.
    if lead and (args.eval_minutes > 0 or args.video_interval > 0):
        attach_eval(runner, build_recorder(args, args.task, agent_cfg.seed), args)

    if args.ddp_verify > 0:
        attach_ddp_verify(runner, args.ddp_verify, rank, world_size)

    # `learn` takes a COUNT and computes `tot_iter = current + count`, so passing the absolute
    # target after a resume would extend the endpoint by a whole budget on every leg and a
    # chained run would never reach a fixed end.
    todo = int(agent_cfg.max_iterations) - int(runner.current_learning_iteration)
    if args.dry_run:
        say(f"[dry-run] would train {todo} iterations to reach {agent_cfg.max_iterations}")
        return 0
    if todo <= 0:
        say(f"[alg] already at iteration {runner.current_learning_iteration} of "
            f"{agent_cfg.max_iterations}; nothing to do")
        return 0

    say(f"[alg] training {todo} iterations to reach {agent_cfg.max_iterations}")
    sys.stdout.flush()
    runner.learn(num_learning_iterations=todo, init_at_random_ep_len=True)
    # Rank 0 writes the final checkpoint after its last iteration; the other ranks leave
    # `learn` immediately. Without the barrier they tear the process group down underneath it.
    ddp_barrier()
    env.close()
    if world_size > 1:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
