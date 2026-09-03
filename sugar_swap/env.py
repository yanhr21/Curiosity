"""`ManagerBasedRLEnv` driving Newton, with IsaacLab's manager pipeline kept intact.

The managers themselves are IsaacLab's own classes, imported unmodified: this class only
supplies the pieces they call into -- a scene, a simulation context, and a `step` that runs
them in IsaacLab's order. That ordering is the part worth being careful about, because the
observation a policy sees depends on it:

1. `process_action` once per policy step, then `apply_action` once per physics substep.
2. Physics advances `decimation` times at `sim.dt`.
3. Episode counters advance, then commands are resampled.
4. Rewards are computed over the post-step state.
5. Terminations are evaluated, and the terminated environments are reset.
6. Observations are computed *after* the reset, so a fresh episode's first observation
   describes the new state rather than the old one.

Getting step 6 wrong is the classic silent bug in a port like this -- the policy trains on
observations from the episode it already left -- so the reset happens before
`observation_manager.compute()`, as in IsaacLab.
"""

from __future__ import annotations

from typing import Any

import torch

from .lenient import LenientCfg
from .shadows import SimulationCfg


class SimulationContext:
    """The slice of IsaacLab's `SimulationContext` that the managers and terms consult.

    `is_playing()` returning True matters: `ManagerBase.__init__` uses it to decide whether to
    resolve scene entities immediately or defer them to an Omniverse timeline event. Returning
    True keeps resolution synchronous and means the timeline stub is never reached.
    """

    def __init__(self, dt: float, device: str, render_interval: int = 1):
        self.dt = dt
        self.device = device
        self.render_interval = render_interval
        self._playing = True

    def is_playing(self) -> bool:
        return self._playing

    def get_physics_dt(self) -> float:
        return self.dt

    def has_gui(self) -> bool:
        return False

    def has_rtx_sensors(self) -> bool:
        return False

    def step(self, render: bool = False) -> None:
        raise NotImplementedError(
            "sugar_swap: physics is advanced by ManagerBasedRLEnv._step_physics, not through "
            "SimulationContext.step()."
        )


class ManagerBasedEnvCfg(LenientCfg):
    """Consumed: `scene`, `sim`, `decimation`, and the manager config groups.

    `sim` defaults to an instance rather than None because SUGAR's `__post_init__` mutates it
    in place (`self.sim.dt = 0.005`) without ever assigning the group, exactly as it does
    against IsaacLab's base config.
    """

    scene: Any = None
    sim: Any = SimulationCfg()
    decimation: int = 1
    observations: Any = None
    actions: Any = None
    events: Any = None
    viewer: Any = LenientCfg()
    seed: int | None = None


class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    """Adds the RL-specific groups."""

    rewards: Any = None
    terminations: Any = None
    commands: Any = None
    curriculum: Any = None
    episode_length_s: float = 10.0
    is_finite_horizon: bool = False


class DirectRLEnvCfg(LenientCfg):
    """Referenced by SUGAR's task registry; the direct workflow is not used here."""


def _bind_event_term_globals() -> None:
    """Bind the two globals IsaacLab's verbatim-extracted event terms close over.

    `events.py` execs IsaacLab's definitions into a namespace it assembles itself, and two
    names their bodies reference are missing from it: `math_utils`, which every one of the
    four terms samples through, and `_validate_scale_range`, which
    `randomize_rigid_body_mass` calls whenever `operation="scale"` -- which is exactly how
    SUGAR configures the box's mass randomisation. Without them the event manager raises
    `NameError` while resolving its terms.

    Both belong in that module's `_WANTED` and `_build_namespace`. They are bound from here
    because the exec namespace is reachable only through an extracted term's `__globals__`,
    and the validator is lifted from IsaacLab's source rather than retyped so the accepted
    range stays IsaacLab's.
    """
    from isaaclab.envs.mdp import events as terms

    namespace = terms.randomize_rigid_body_mass.__init__.__globals__
    if "math_utils" in namespace:
        return

    import ast

    import isaaclab.utils.math as math_utils

    from . import events as extractor

    namespace["math_utils"] = math_utils

    path = extractor._events_source_path()
    source = path.read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_scale_range":
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
            return
    raise RuntimeError(
        "sugar_swap: IsaacLab's events.py no longer defines _validate_scale_range, which "
        "randomize_rigid_body_mass needs for a scale operation."
    )


class ManagerBasedEnv:
    """Base holding the scene, the simulation context and the non-RL managers."""

    def __init__(self, cfg: ManagerBasedEnvCfg, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = device

        sim_cfg = cfg.sim
        self.physics_dt = float(getattr(sim_cfg, "dt", 1.0 / 200.0))
        self.cfg.decimation = int(cfg.decimation)
        self.sim = SimulationContext(self.physics_dt, device)

        from .builder import build_scene

        self.scene = build_scene(cfg.scene, device)
        self.scene.physics_dt = self.physics_dt
        self.num_envs = self.scene.num_envs

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.extras: dict[str, Any] = {}
        # `_sim_step_counter` counts physics substeps and `common_step_counter` policy
        # decisions. IsaacLab's event manager gates reset terms on the first and SUGAR's
        # observation terms cache on the second.
        self._sim_step_counter = 0
        self.common_step_counter = 0
        self.obs_buf: dict[str, Any] = {}

        from isaaclab.managers import EventManager

        _bind_event_term_globals()
        # IsaacLab builds the event manager before every other manager, so that
        # `prestartup` terms can still change the asset before physics starts.
        self.event_manager = EventManager(self.cfg.events, self)
        if "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")

        # IsaacLab steps the scene once here, because the observation manager probes every
        # term for its dimension while constructing and a term reading an unpopulated
        # buffer would crash or report the wrong width.
        self.scene.update(self.physics_dt)

        self.load_managers()

    @property
    def step_dt(self) -> float:
        """Seconds per policy decision, which is what the reward and command terms use."""
        return self.physics_dt * self.cfg.decimation

    def load_managers(self) -> None:
        from isaaclab.managers import ActionManager, ObservationManager

        self.action_manager = ActionManager(self.cfg.actions, self)
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        if type(self) is ManagerBasedEnv and "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def _step_physics(self) -> None:
        """Advance Newton by a single physics step.

        The decimation loop lives in `step()`, which re-applies the action between steps the
        way IsaacLab does; advancing `decimation` steps here as well would run the physics
        `decimation**2` times per decision.
        """
        from .builder import step_physics

        step_physics(self.scene, 1)


class ManagerBasedRLEnv(ManagerBasedEnv):
    """The environment SUGAR's task registry instantiates."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.render_mode = render_mode
        # Set before super().__init__, because it calls load_managers() and the termination
        # and command terms read these buffers while resolving their dimensions.
        num_envs = int(cfg.scene.num_envs)
        device = kwargs.get("device", "cuda:0")
        self.max_episode_length = 1
        self.reward_buf = torch.zeros(num_envs, device=device)
        self.reset_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reset_time_outs = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)

        super().__init__(cfg, device=kwargs.pop("device", "cuda:0"))

        self.max_episode_length = max(int(cfg.episode_length_s / self.step_dt), 1)
        self.reset()

    def load_managers(self) -> None:
        """IsaacLab's order: commands first, then the base managers, then the RL managers.

        The ordering is load-bearing rather than stylistic: SUGAR's observation terms call
        `env.command_manager.get_term("motion")` while the observation manager is probing
        them for their width, and the reward terms resolve against the termination manager.
        """
        from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

        self.command_manager = CommandManager(self.cfg.commands, self)
        super().load_managers()
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        self.curriculum_manager = (
            None if self.cfg.curriculum is None else CurriculumManager(self.cfg.curriculum, self)
        )
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    @property
    def num_actions(self) -> int:
        return self.action_manager.total_action_dim

    @property
    def max_episode_length_s(self) -> float:
        """Episode length in seconds; the reward manager divides its episode sums by it."""
        return self.cfg.episode_length_s

    def reset(self, seed: int | None = None, options: dict | None = None):
        env_ids = self.scene.all_envs
        self._reset_idx(env_ids)
        self.scene.write_data_to_sim()
        # IsaacLab calls `sim.forward()` here to push the written joint coordinates through
        # forward kinematics before anything reads a body pose.
        self.scene.flush_kinematics()
        self.scene.update(self.physics_dt)
        self.obs_buf = self.observation_manager.compute()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """IsaacLab's reset, including its manager ordering and its `log` extras.

        The order is IsaacLab's comment "this is order-sensitive"; the per-manager return
        values are what populate `extras["log"]`, which is where the per-term episode reward
        breakdown comes from.
        """
        if self.curriculum_manager is not None:
            self.curriculum_manager.compute(env_ids=env_ids)
        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            # Gates IsaacLab's `min_step_count_between_reset`; without it a term that should
            # fire at most once per N env steps fires on every reset.
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
            )

        self.extras["log"] = dict()
        managers = [
            self.observation_manager,
            self.action_manager,
            self.reward_manager,
            self.curriculum_manager,
            self.command_manager,
            self.event_manager,
            self.termination_manager,
        ]
        for manager in managers:
            if manager is None:
                continue
            info = manager.reset(env_ids)
            if info:
                self.extras["log"].update(info)

        self.episode_length_buf[env_ids] = 0

    def step(self, action: torch.Tensor):
        """One policy decision, in IsaacLab's order.

        Every step below is ordered as `ManagerBasedRLEnv.step`, and three of those orderings
        are the kind that corrupt training silently rather than crashing:

        * the scene is updated inside the decimation loop, so the contact sensor reduces each
          physics step's contacts and `joint_acc` differences at `sim.dt` rather than at the
          policy step;
        * terminations are computed before rewards, and `reset_terminated` is the
          termination manager's `terminated` rather than its `compute()` result -- the latter
          also carries time-outs, which would make the critic bootstrap through a truncation;
        * commands are resampled *after* the reset, so a fresh episode's reference frame
          matches the state the observation then reports.
        """
        self.action_manager.process_action(action.to(self.device))

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self._step_physics()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.scene.flush_kinematics()
            self.scene.update(dt=self.physics_dt)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False):
        return None

    def close(self) -> None:
        pass


def build():
    """Construct the `isaaclab.envs` shadow module.

    `__path__` points at the real IsaacLab package so that submodules we want unmodified --
    `isaaclab.envs.mdp` and `isaaclab.envs.utils.io_descriptors` -- still resolve from the
    genuine source, while the top-level names come from here.
    """
    import pathlib
    import types

    mod = types.ModuleType("isaaclab.envs")
    real = (
        pathlib.Path(__file__).resolve().parent.parent
        / "IsaacLab" / "source" / "isaaclab" / "isaaclab" / "envs"
    )
    mod.__path__ = [str(real)]
    mod.ManagerBasedEnv = ManagerBasedEnv
    mod.ManagerBasedEnvCfg = ManagerBasedEnvCfg
    mod.ManagerBasedRLEnv = ManagerBasedRLEnv
    mod.ManagerBasedRLEnvCfg = ManagerBasedRLEnvCfg
    mod.DirectRLEnvCfg = DirectRLEnvCfg
    mod.VecEnvObs = dict
    mod.VecEnvStepReturn = tuple
    return {"isaaclab.envs": mod}
