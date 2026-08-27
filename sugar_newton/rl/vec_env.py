# SPDX-License-Identifier: BSD-3-Clause
"""Make :class:`CarryBoxEnv` an ``rsl_rl.env.VecEnv``, so SUGAR's own trainer can drive it.

The point of this file is that nothing about the *algorithm* is reimplemented here. SUGAR
trains the tracker with ``BCPPO`` (``sugar_rl/utils/rsl_rl_bcppo.py``) running inside
``rsl_rl``'s ``OnPolicyRunner``; both are imported and used as they are. This adapter is
the only thing in between, and all it does is present the Newton environment in the shape
rsl_rl expects.

That shape is three observation groups in a ``TensorDict``::

    policy   510-D   what the actor sees
    critic   890-D   privileged
    teacher  890-D   what the frozen refiner is asked to imitate

``BCPPORunnerCfg.obs_groups`` maps each to its consumer, and the teacher group is what
makes the distillation term possible at all.

``extras["time_outs"]`` is the rsl_rl convention for "this episode was truncated, not
failed", which lets the algorithm bootstrap the value target instead of cutting it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict

from sugar_newton.rl import obs_890
from sugar_newton.rl.carrybox_env import (
    N_DOF,
    OBS_DIM,
    TRACKER_COMMAND_DIM,
    CarryBoxEnv,
)


class EnvCfg(dict):
    """A dict that rsl_rl's wandb writer will accept.

    ``WandbSummaryWriter.store_config`` calls ``env_cfg.to_dict()`` and falls back to
    ``dataclasses.asdict``; a plain dict satisfies neither, and the run dies *after*
    wandb has already opened, which reads as a wandb problem when it is not.
    """

    def to_dict(self) -> dict:
        return dict(self)


class CarryBoxVecEnv:
    """rsl_rl VecEnv over :class:`CarryBoxEnv`."""

    def __init__(self, env: CarryBoxEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = N_DOF
        self.device = env.device
        self.max_episode_length = env.episode_length
        self.cfg = EnvCfg(num_envs=env.num_envs, episode_length=env.episode_length,
                          substeps=env.substeps, dt=env.dt, clips=list(env.clip_names))
        self.episode_length_buf = torch.zeros(env.num_envs, dtype=torch.long,
                                              device=env.device)

    def _obs(self, policy: torch.Tensor | None = None) -> TensorDict:
        # CarryBoxEnv.step() already advances the 510-D causal history exactly once
        # and returns that observation.  Recalling observe() here would duplicate the
        # same physical frame in every history channel.
        if policy is None:
            policy = self.env.observe()
        priv = obs_890.build(self.env, teacher=False)
        # Teacher and critic share a term list but read aligned raw and Refiner-rollout
        # motion sources respectively, matching the official SUGAR tracker contract.
        teacher = obs_890.build(self.env, teacher=True)
        return TensorDict(
            {"policy": policy, "critic": priv, "teacher": teacher},
            batch_size=[self.num_envs], device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    def reset(self) -> tuple[TensorDict, dict]:
        self.env.reset()
        self.episode_length_buf.zero_()
        return self._obs(), {}

    def step(self, actions: torch.Tensor):
        policy, reward, done, extras = self.env.step(actions)
        self.episode_length_buf += 1
        self.episode_length_buf[done] = 0
        info = {
            "time_outs": extras.get("timeout", torch.zeros_like(done)),
            "episode": {
                f"rew_{k}": v.mean() for k, v in extras.get("reward_terms", {}).items()
            },
        }
        info["episode"]["diverged_total"] = torch.tensor(
            float(self.env.num_diverged), device=self.device)
        return self._obs(policy), reward, done, info


def make(num_envs: int, **kwargs) -> CarryBoxVecEnv:
    return CarryBoxVecEnv(CarryBoxEnv(num_envs=num_envs, **kwargs))


OBS_DIMS = {"policy": OBS_DIM, "critic": obs_890.OBS_DIM_890, "teacher": obs_890.OBS_DIM_890}


class ActingTeacherHandoffVecEnv(CarryBoxVecEnv):
    """Execute the admitted Refiner until a physical no-reset handoff.

    The runner still samples a Tracker action every step because that is how
    ``OnPolicyRunner`` fills its rollout storage.  While the box has not remained at
    least five centimetres above its reset height for ten consecutive frames, this
    adapter executes the deterministic Refiner mean instead.  The extra
    ``training_handoff_mask`` observation is zero on those teacher-controlled
    transitions and one only when the sampled Tracker action is the action actually
    applied to Newton.  SUGAR's BCPPO consumes that key solely as its PPO/value mask;
    official teacher distillation still uses the full trajectory, and the key is never
    concatenated into the 510-D deployed actor observation.

    This class intentionally has no fallback timer.  A trajectory on which the
    admitted teacher cannot reach the physical gate remains teacher-controlled and
    contributes no Tracker PPO transition rather than fabricating a handoff.
    """

    MINIMUM_LIFT_M = 0.05
    STABLE_LIFT_FRAMES = 10

    def __init__(
        self,
        env: CarryBoxEnv,
        *,
        teacher_checkpoint: str | Path,
        reward_clip: float = 10.0,
        sync_divergence_reset: bool = True,
    ):
        super().__init__(env)
        checkpoint = Path(teacher_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if reward_clip <= 0.0:
            raise ValueError("reward_clip must be positive")

        # rsl_rl has already been pinned to SUGAR's compatible release by the
        # launcher before this adapter is constructed.  Reuse the exact strict loader
        # used by the frozen physical gate, so acting and evaluation cannot silently
        # disagree about architecture or checkpoint semantics.
        from sugar_newton.validation.refiner_open_loop import (
            load_official_teacher,
            sha256,
        )

        self.acting_teacher, hidden_dims = load_official_teacher(
            checkpoint, torch.device(env.device)
        )
        if hidden_dims != [512, 256, 128]:
            raise ValueError(
                f"acting Refiner hidden geometry drifted: {hidden_dims}"
            )
        self.teacher_checkpoint = checkpoint
        self.teacher_checkpoint_sha256 = sha256(checkpoint)
        self.reward_clip = float(reward_clip)
        self.sync_divergence_reset = bool(sync_divergence_reset)

        self.teacher_control = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.stable_lift_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.handoff_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.initial_box_z = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.cumulative_handoffs = torch.zeros_like(self.stable_lift_count)
        self.cumulative_teacher_control_steps = torch.zeros_like(
            self.stable_lift_count
        )
        self.cumulative_policy_control_steps = torch.zeros_like(
            self.stable_lift_count
        )
        self.last_policy_action: torch.Tensor | None = None
        self.last_teacher_action: torch.Tensor | None = None
        self.last_executed_action: torch.Tensor | None = None
        self.last_teacher_control_mask: torch.Tensor | None = None
        self._reset_handoff(torch.arange(self.num_envs, device=self.device))
        self.cfg.update(
            acting_teacher_checkpoint=str(checkpoint),
            acting_teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            handoff_minimum_lift_m=self.MINIMUM_LIFT_M,
            handoff_stable_lift_frames=self.STABLE_LIFT_FRAMES,
            training_mask_obs_group="training_handoff_mask",
            reward_clip=self.reward_clip,
            sync_divergence_reset=self.sync_divergence_reset,
        )

    def _current_box_z(self) -> torch.Tensor:
        return self.env._body_q()[:, self.env.box_body, 2]

    def _reset_handoff(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.teacher_control[env_ids] = True
        self.stable_lift_count[env_ids] = 0
        self.handoff_step[env_ids] = -1
        self.initial_box_z[env_ids] = self._current_box_z()[env_ids]

    def _obs(self, policy: torch.Tensor | None = None) -> TensorDict:
        observations = super()._obs(policy)
        observations.set(
            "training_handoff_mask",
            (~self.teacher_control).to(torch.float32).unsqueeze(-1),
        )
        return observations

    def reset(self) -> tuple[TensorDict, dict]:
        self.env.reset()
        self.episode_length_buf.zero_()
        self._reset_handoff(torch.arange(self.num_envs, device=self.device))
        return self._obs(), {}

    @torch.inference_mode()
    def step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Tracker action shape is {tuple(actions.shape)}, expected "
                f"{(self.num_envs, self.num_actions)}"
            )
        teacher_observation = obs_890.build(self.env, teacher=True)
        teacher_action = self.acting_teacher(teacher_observation)
        if not torch.isfinite(teacher_action).all():
            raise RuntimeError("acting Refiner emitted a non-finite action")

        teacher_control_before_step = self.teacher_control.clone()
        executed_action = torch.where(
            teacher_control_before_step[:, None], teacher_action, actions
        )
        self.last_policy_action = actions.detach().clone()
        self.last_teacher_action = teacher_action.detach().clone()
        self.last_executed_action = executed_action.detach().clone()
        self.last_teacher_control_mask = teacher_control_before_step
        self.cumulative_teacher_control_steps += teacher_control_before_step.long()
        self.cumulative_policy_control_steps += (~teacher_control_before_step).long()

        policy, reward, done, extras = self.env.step(executed_action)
        divergence = extras.get("termination_terms", {}).get(
            "diverged", torch.zeros_like(done)
        )

        if self.sync_divergence_reset and bool(divergence.any()):
            # Match the validated Refiner training safety contract: never mix a
            # post-divergence reset world with still-live worlds inside one PPO
            # horizon.  The frozen evaluator remains per-profile and does not use this
            # synchronized training-only reset.
            self.env.reset()
            policy = self.env.observe()
            done = torch.ones_like(done)
            extras["timeout"] = torch.zeros_like(done)

        active_teacher = teacher_control_before_step & ~done
        lifted = (
            self._current_box_z() - self.initial_box_z
            >= self.MINIMUM_LIFT_M
        )
        self.stable_lift_count = torch.where(
            active_teacher & lifted,
            self.stable_lift_count + 1,
            torch.where(
                active_teacher,
                torch.zeros_like(self.stable_lift_count),
                self.stable_lift_count,
            ),
        )
        newly_handed_off = active_teacher & (
            self.stable_lift_count >= self.STABLE_LIFT_FRAMES
        )
        if bool(newly_handed_off.any()):
            self.teacher_control[newly_handed_off] = False
            self.handoff_step[newly_handed_off] = self.episode_length_buf[
                newly_handed_off
            ] + 1
            self.cumulative_handoffs[newly_handed_off] += 1

        reset_ids = done.nonzero(as_tuple=False).flatten()
        if reset_ids.numel():
            # CarryBoxEnv has already auto-reset these worlds.  Start the replacement
            # episode under teacher control and measure lift from its new box height.
            self._reset_handoff(reset_ids)

        reward = torch.clamp(
            torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0),
            -self.reward_clip,
            self.reward_clip,
        )
        self.episode_length_buf += 1
        self.episode_length_buf[done] = 0
        info = {
            "time_outs": extras.get("timeout", torch.zeros_like(done)),
            "episode": {
                f"rew_{key}": torch.nan_to_num(
                    value, nan=0.0, posinf=0.0, neginf=0.0
                ).mean()
                for key, value in extras.get("reward_terms", {}).items()
            },
        }
        info["episode"].update(
            diverged_total=torch.tensor(
                float(self.env.num_diverged), device=self.device
            ),
            teacher_control_fraction=teacher_control_before_step.float().mean(),
            handoff_active_fraction=(~self.teacher_control).float().mean(),
            cumulative_handoffs=self.cumulative_handoffs.float().sum(),
        )
        return self._obs(policy), reward, done, info


def make_acting_teacher_handoff(
    num_envs: int,
    *,
    teacher_checkpoint: str | Path,
    reward_clip: float = 10.0,
    sync_divergence_reset: bool = True,
    **kwargs,
) -> ActingTeacherHandoffVecEnv:
    return ActingTeacherHandoffVecEnv(
        CarryBoxEnv(num_envs=num_envs, **kwargs),
        teacher_checkpoint=teacher_checkpoint,
        reward_clip=reward_clip,
        sync_divergence_reset=sync_divergence_reset,
    )


class RefinerVecEnv:
    """Official 890-D Refiner actor/critic contract over the Newton environment.

    This is deliberately separate from :class:`CarryBoxVecEnv`: the Refiner itself is
    trained with the privileged 890-D group for both actor and critic, exactly as
    ``BaseObservationsCfg`` and ``BasePPORunnerCfg`` declare.  No Tracker observation or
    distillation target is involved in this stage.
    """

    def __init__(
        self,
        env: CarryBoxEnv,
        *,
        reward_clip: float = 10.0,
        sync_divergence_reset: bool = True,
        policy_history_steps: int = 0,
        policy_command_dim: int = 0,
        tracker_teacher: bool = False,
    ):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = N_DOF
        self.device = env.device
        self.max_episode_length = env.episode_length
        if reward_clip <= 0.0:
            raise ValueError("reward_clip must be positive")
        self.reward_clip = float(reward_clip)
        self.sync_divergence_reset = bool(sync_divergence_reset)
        if policy_history_steps < 0:
            raise ValueError("policy_history_steps must be non-negative")
        self.policy_history_steps = int(policy_history_steps)
        if policy_command_dim not in (0, TRACKER_COMMAND_DIM):
            raise ValueError(
                f"policy_command_dim must be 0 or {TRACKER_COMMAND_DIM}"
            )
        self.policy_command_dim = int(policy_command_dim)
        self.tracker_teacher = bool(tracker_teacher)
        if self.policy_command_dim and not self.tracker_teacher:
            raise ValueError(
                "Refiner policy command conditioning requires the synchronized "
                "Tracker observation"
            )
        if self.policy_command_dim and not self.policy_history_steps:
            raise ValueError(
                "Refiner policy command conditioning requires causal history"
            )
        self.cfg = EnvCfg(
            num_envs=env.num_envs,
            episode_length=env.episode_length,
            substeps=env.substeps,
            dt=env.dt,
            clips=list(env.clip_names),
            observation_contract=(
                f"official_refiner_current_plus_{self.policy_history_steps}x890d_history"
                + (
                    f"_plus_current_{self.policy_command_dim}d_reference_command"
                    if self.policy_command_dim
                    else ""
                )
                if self.policy_history_steps
                else "official_refiner_890d"
            ),
            reward_clip=self.reward_clip,
            sync_divergence_reset=self.sync_divergence_reset,
            frame_zero_env_count=env.frame_zero_env_count,
            policy_history_steps=self.policy_history_steps,
            policy_command_dim=self.policy_command_dim,
            tracker_teacher=self.tracker_teacher,
        )
        self.episode_length_buf = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self._current_tracker = self.env.observe() if self.tracker_teacher else None
        self._current_command: torch.Tensor | None = None
        self._synchronize_current_command()
        self._current_privileged = obs_890.build(self.env, teacher=False)
        self._observation_epoch = 0
        self._policy_history: torch.Tensor | None = None
        self._advance_policy_history(self._current_privileged, reset_all=True)

    def _synchronize_current_command(self) -> None:
        if not self.policy_command_dim:
            self._current_command = None
            return
        if self._current_tracker is None:
            raise RuntimeError("Tracker observation is missing for command sync")
        command = self.env.tracker_command()
        if tuple(command.shape) != (self.num_envs, self.policy_command_dim):
            raise RuntimeError(f"Refiner policy command drift: {tuple(command.shape)}")
        prefix = self._current_tracker[:, :self.policy_command_dim]
        if not torch.equal(command, prefix):
            delta = float((command - prefix).abs().max().item())
            raise RuntimeError(
                f"Refiner policy command differs from Tracker prefix: {delta}"
            )
        self._current_command = command

    def _advance_policy_history(
        self,
        current: torch.Tensor,
        *,
        reset_ids: torch.Tensor | None = None,
        reset_all: bool = False,
    ) -> None:
        if tuple(current.shape) != (self.num_envs, obs_890.OBS_DIM_890):
            raise RuntimeError(f"Refiner current observation drift: {tuple(current.shape)}")
        self._current_privileged = current
        if not self.policy_history_steps:
            return
        repeated = current[:, None, :].expand(
            -1, self.policy_history_steps, -1
        ).clone()
        if reset_all or self._policy_history is None:
            self._policy_history = repeated
            return
        self._policy_history = torch.cat(
            (self._policy_history[:, 1:], current[:, None, :]), dim=1
        )
        if reset_ids is not None and reset_ids.numel():
            self._policy_history[reset_ids] = repeated[reset_ids]

    def _obs(self) -> TensorDict:
        privileged = self._current_privileged
        policy = privileged
        if self.policy_history_steps:
            if self._policy_history is None:
                raise RuntimeError("Refiner policy history was not initialized")
            if not torch.equal(self._policy_history[:, -1], privileged):
                raise RuntimeError("Refiner policy history does not end at current state")
            policy = torch.cat(
                (privileged, self._policy_history.reshape(self.num_envs, -1)),
                dim=-1,
            )
        if self.policy_command_dim:
            if self._current_command is None:
                raise RuntimeError("Refiner policy command was not synchronized")
            if self._current_tracker is None or not torch.equal(
                self._current_command,
                self._current_tracker[:, :self.policy_command_dim],
            ):
                raise RuntimeError("Refiner policy command/Tracker alignment drift")
            policy = torch.cat((policy, self._current_command), dim=-1)
        teacher = self._current_tracker if self.tracker_teacher else policy
        if self.tracker_teacher:
            if teacher is None or tuple(teacher.shape) != (self.num_envs, OBS_DIM):
                raise RuntimeError("Tracker teacher observation was not synchronized")
        return TensorDict(
            # The optional Refiner BCPPO transfer uses the same causal/current Newton
            # 890-D tensor to query a separately frozen official Refiner.  Keeping a
            # distinct observation-group name makes that teacher contract explicit while
            # leaving ordinary PPO's policy/critic groups unchanged.
            {"policy": policy, "critic": privileged, "teacher": teacher},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    def reset(self) -> tuple[TensorDict, dict]:
        self.env.reset()
        self.episode_length_buf.zero_()
        if self.tracker_teacher:
            self._current_tracker = self.env.observe()
        self._synchronize_current_command()
        self._advance_policy_history(
            obs_890.build(self.env, teacher=False), reset_all=True
        )
        self._observation_epoch += 1
        return self._obs(), {}

    def step(self, actions: torch.Tensor):
        tracker, reward, done, extras = self.env.step(actions)
        divergence = extras.get("termination_terms", {}).get(
            "diverged", torch.zeros_like(done)
        )
        # MuJoCo-Warp worlds are physically independent, but PPO collects one
        # synchronized horizon.  If one solve becomes non-finite, reset the whole
        # horizon boundary rather than mixing post-reset observations with the
        # still-live worlds.  This is a training-only safety rule; frozen evaluation
        # keeps the original per-profile termination behavior.
        if self.sync_divergence_reset and bool(divergence.any()):
            self.env.reset()
            tracker = self.env.observe()
            done = torch.ones_like(done)
            extras["timeout"] = torch.zeros_like(done)
        if self.tracker_teacher:
            self._current_tracker = tracker
        self._synchronize_current_command()
        reset_ids = done.nonzero(as_tuple=False).flatten()
        self._advance_policy_history(
            obs_890.build(self.env, teacher=False), reset_ids=reset_ids
        )
        self._observation_epoch += 1
        reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)
        self.episode_length_buf += 1
        self.episode_length_buf[done] = 0
        info = {
            "time_outs": extras.get("timeout", torch.zeros_like(done)),
            "episode": {
                f"rew_{key}": torch.nan_to_num(
                    value, nan=0.0, posinf=0.0, neginf=0.0
                ).mean()
                for key, value in extras.get("reward_terms", {}).items()
            },
        }
        info["episode"]["diverged_total"] = torch.tensor(
            float(self.env.num_diverged), device=self.device
        )
        return self._obs(), reward, done, info


def make_refiner(
    num_envs: int,
    *,
    reward_clip: float = 10.0,
    sync_divergence_reset: bool = True,
    policy_history_steps: int = 0,
    policy_command_dim: int = 0,
    tracker_teacher: bool = False,
    **kwargs,
) -> RefinerVecEnv:
    return RefinerVecEnv(
        CarryBoxEnv(num_envs=num_envs, **kwargs),
        reward_clip=reward_clip,
        sync_divergence_reset=sync_divergence_reset,
        policy_history_steps=policy_history_steps,
        policy_command_dim=policy_command_dim,
        tracker_teacher=tracker_teacher,
    )
