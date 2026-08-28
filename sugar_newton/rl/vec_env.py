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

from sugar_newton.rl import obs_890, rewards
from sugar_newton.rl.carrybox_env import (
    N_DOF,
    OBS_DIM,
    TRACKER_COMMAND_DIM,
    CarryBoxEnv,
)
from sugar_newton.rl.reference_phase import policy_reference_phase


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
        physical_recovery_objective: bool = False,
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
        self.physical_recovery_objective = bool(physical_recovery_objective)
        self.physical_recovery_calls = 0
        self.physical_recovery_max_abs = 0.0
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
            physical_recovery_objective=self.physical_recovery_objective,
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
        if self.physical_recovery_objective:
            recovery = extras.get("physical_recovery_terms")
            required = {
                "object_margin",
                "end_effector_margin",
                "bilateral_contact",
                "lifted_bilateral_hold",
                "failure",
            }
            if recovery is None or set(recovery) != required:
                raise RuntimeError("Newton physical-recovery reward contract drift")
            values = torch.stack(
                [
                    recovery["object_margin"],
                    recovery["end_effector_margin"],
                    recovery["bilateral_contact"],
                    recovery["lifted_bilateral_hold"],
                ],
                dim=-1,
            )
            if not torch.isfinite(values).all():
                raise RuntimeError("Newton physical-recovery labels are non-finite")
            physical_score = values.mean(dim=-1) - recovery["failure"]
            official_score = torch.clamp(
                reward / rewards.OFFICIAL_POSITIVE_REWARD_SCALE, -1.0, 1.0
            )
            reward = 0.5 * official_score + 0.5 * physical_score
            divergence = extras.get("termination_terms", {}).get(
                "diverged", torch.zeros_like(done)
            )
            reward = torch.where(divergence, torch.zeros_like(reward), reward)
            recovery["official_reward_normalized"] = official_score.detach()
            recovery["physical_score"] = physical_score.detach()
            recovery["combined_reward"] = reward.detach()
            self.physical_recovery_calls += int(self.num_envs)
            self.physical_recovery_max_abs = max(
                self.physical_recovery_max_abs,
                float(reward.abs().max().item()),
            )
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
        if self.physical_recovery_objective:
            info["episode"].update(
                {
                    f"physical_{key}": value.mean()
                    for key, value in extras["physical_recovery_terms"].items()
                }
            )
        info["episode"]["diverged_total"] = torch.tensor(
            float(self.env.num_diverged), device=self.device
        )
        return self._obs(), reward, done, info


class RefinerFailureFrontierVecEnv(RefinerVecEnv):
    """Train only after an exact-Refiner physical prefix, without resetting.

    The sampled student action is stored by rsl_rl throughout the rollout, but
    the exact released Refiner action is executed for the first ``prefix_steps``
    transitions of every episode.  ``training_handoff_mask`` is zero there and
    one only when the student's action is actually deployed.  The mask is a
    storage/loss key; it is never concatenated into the 9790-D temporal actor.
    """

    def __init__(
        self,
        env: CarryBoxEnv,
        *,
        teacher_checkpoint: str | Path,
        prefix_steps: int,
        **kwargs,
    ):
        if prefix_steps <= 0:
            raise ValueError("failure-frontier prefix must be positive")
        super().__init__(env, **kwargs)
        checkpoint = Path(teacher_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        from sugar_newton.validation.refiner_open_loop import (
            load_official_teacher,
            sha256,
        )

        self.acting_teacher, hidden_dims = load_official_teacher(
            checkpoint, torch.device(env.device)
        )
        if hidden_dims != [512, 256, 128]:
            raise ValueError(f"acting Refiner geometry drifted: {hidden_dims}")
        self.teacher_checkpoint = checkpoint
        self.teacher_checkpoint_sha256 = sha256(checkpoint)
        self.prefix_steps = int(prefix_steps)
        self.cumulative_teacher_control_steps = 0
        self.cumulative_policy_control_steps = 0
        self.maximum_teacher_execution_delta = 0.0
        self.maximum_policy_execution_delta = 0.0
        self.last_teacher_control_mask: torch.Tensor | None = None
        self.last_teacher_action: torch.Tensor | None = None
        self.last_policy_action: torch.Tensor | None = None
        self.last_executed_action: torch.Tensor | None = None
        self.cfg.update(
            failure_frontier_prefix_steps=self.prefix_steps,
            acting_teacher_checkpoint=str(checkpoint),
            acting_teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            training_mask_obs_group="training_handoff_mask",
            training_mask_actor_input=False,
        )

    def _teacher_control(self) -> torch.Tensor:
        return self.episode_length_buf < self.prefix_steps

    def _obs(self) -> TensorDict:
        observations = super()._obs()
        observations.set(
            "training_handoff_mask",
            (~self._teacher_control()).to(torch.float32).unsqueeze(-1),
        )
        return observations

    def step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Refiner action shape is {tuple(actions.shape)}, expected "
                f"{(self.num_envs, self.num_actions)}"
            )
        teacher_control = self._teacher_control().clone()
        with torch.inference_mode():
            teacher_action = self.acting_teacher(self._current_privileged)
        if not torch.isfinite(teacher_action).all():
            raise RuntimeError("failure-frontier Refiner action is non-finite")
        executed_action = torch.where(
            teacher_control[:, None], teacher_action, actions
        )
        if bool(teacher_control.any()):
            self.maximum_teacher_execution_delta = max(
                self.maximum_teacher_execution_delta,
                float(
                    (
                        executed_action[teacher_control]
                        - teacher_action[teacher_control]
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
        if bool((~teacher_control).any()):
            self.maximum_policy_execution_delta = max(
                self.maximum_policy_execution_delta,
                float(
                    (executed_action[~teacher_control] - actions[~teacher_control])
                    .abs()
                    .max()
                    .item()
                ),
            )
        self.last_teacher_control_mask = teacher_control
        self.last_teacher_action = teacher_action.detach().clone()
        self.last_policy_action = actions.detach().clone()
        self.last_executed_action = executed_action.detach().clone()
        self.cumulative_teacher_control_steps += int(teacher_control.sum().item())
        self.cumulative_policy_control_steps += int((~teacher_control).sum().item())

        observations, reward, done, info = super().step(executed_action)
        info["episode"].update(
            failure_frontier_teacher_control_fraction=teacher_control.float().mean(),
            failure_frontier_policy_control_fraction=(~teacher_control).float().mean(),
            failure_frontier_teacher_steps=torch.tensor(
                float(self.cumulative_teacher_control_steps), device=self.device
            ),
            failure_frontier_policy_steps=torch.tensor(
                float(self.cumulative_policy_control_steps), device=self.device
            ),
        )
        return observations, reward, done, info


class RefinerReferencePhaseVecEnv(RefinerVecEnv):
    """Learn one causal reference-phase scalar around an exact frozen Refiner.

    Physical time, reward, termination and timeout remain owned by
    :class:`CarryBoxEnv`.  The policy scalar only selects the future-reference
    indices used to build the frozen Refiner's 890-D input.  The fixed envelope
    is exactly zero at physical steps 200 and 235, so the controller cannot
    stall the clock or change the endpoint reference.
    """

    def __init__(
        self,
        env: CarryBoxEnv,
        *,
        teacher_checkpoint: str | Path,
        prefix_steps: int = 200,
        endpoint_step: int = 235,
        maximum_offset: int = 7,
        **kwargs,
    ) -> None:
        if prefix_steps != 200 or endpoint_step != 235 or maximum_offset != 7:
            raise ValueError("reference-phase recovery contract is fixed at 200/235/+/-7")
        super().__init__(env, **kwargs)
        if self.policy_history_steps != 10:
            raise ValueError("reference-phase controller requires exact 10-frame history")
        checkpoint = Path(teacher_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        from sugar_newton.validation.refiner_open_loop import (
            load_official_teacher,
            sha256,
        )

        self.acting_teacher, hidden_dims = load_official_teacher(
            checkpoint, torch.device(env.device)
        )
        if hidden_dims != [512, 256, 128]:
            raise ValueError(f"acting Refiner geometry drifted: {hidden_dims}")
        self.num_actions = 1
        self.teacher_checkpoint = checkpoint
        self.teacher_checkpoint_sha256 = sha256(checkpoint)
        self.prefix_steps = int(prefix_steps)
        self.endpoint_step = int(endpoint_step)
        self.maximum_offset = int(maximum_offset)
        self.cumulative_masked_steps = 0
        self.cumulative_retimed_steps = 0
        self.maximum_abs_phase_offset = 0
        self.last_raw_phase: torch.Tensor | None = None
        self.last_reference_t: torch.Tensor | None = None
        self.last_refiner_action: torch.Tensor | None = None
        self.cfg.update(
            reference_phase_retiming=True,
            reference_phase_handoff_step=self.prefix_steps,
            reference_phase_endpoint_step=self.endpoint_step,
            reference_phase_maximum_offset=self.maximum_offset,
            acting_teacher_checkpoint=str(checkpoint),
            acting_teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            training_mask_obs_group="training_handoff_mask",
            training_mask_actor_input=False,
            physical_clock_modified=False,
        )

    def _policy_control(self) -> torch.Tensor:
        return self.episode_length_buf >= self.prefix_steps

    def _obs(self) -> TensorDict:
        observations = super()._obs()
        observations.set(
            "training_handoff_mask",
            self._policy_control().to(torch.float32).unsqueeze(-1),
        )
        return observations

    def step(self, actions: torch.Tensor):
        if tuple(actions.shape) != (self.num_envs, 1):
            raise ValueError(
                f"reference-phase action shape is {tuple(actions.shape)}, expected "
                f"{(self.num_envs, 1)}"
            )
        if not torch.isfinite(actions).all():
            raise RuntimeError("reference-phase action is non-finite")
        physical_t = self.env.t.detach().clone()
        reference_t = policy_reference_phase(
            actions,
            physical_t,
            handoff_step=self.prefix_steps,
            endpoint_step=self.endpoint_step,
            maximum_offset=self.maximum_offset,
        )
        phase_offset = reference_t - physical_t
        with torch.inference_mode():
            retimed_obs = obs_890.build(
                self.env,
                teacher=False,
                reference_t=reference_t,
            )
            refiner_action = self.acting_teacher(retimed_obs)
        if not torch.isfinite(refiner_action).all():
            raise RuntimeError("reference-phase frozen Refiner action is non-finite")
        if not torch.equal(self.env.t, physical_t):
            raise RuntimeError("reference-phase construction modified physical time")

        policy_control = self._policy_control().clone()
        self.cumulative_masked_steps += int((~policy_control).sum().item())
        self.cumulative_retimed_steps += int(
            (policy_control & (phase_offset != 0)).sum().item()
        )
        self.maximum_abs_phase_offset = max(
            self.maximum_abs_phase_offset,
            int(phase_offset.abs().max().item()),
        )
        self.last_raw_phase = actions.detach().clone()
        self.last_reference_t = reference_t.detach().clone()
        self.last_refiner_action = refiner_action.detach().clone()

        observations, reward, done, info = super().step(refiner_action)
        info["episode"].update(
            reference_phase_policy_control_fraction=policy_control.float().mean(),
            reference_phase_nonzero_fraction=(phase_offset != 0).float().mean(),
            reference_phase_mean_offset=phase_offset.float().mean(),
            reference_phase_maximum_abs_offset=torch.tensor(
                float(self.maximum_abs_phase_offset), device=self.device
            ),
            reference_phase_retimed_steps=torch.tensor(
                float(self.cumulative_retimed_steps), device=self.device
            ),
        )
        return observations, reward, done, info


def make_refiner(
    num_envs: int,
    *,
    reward_clip: float = 10.0,
    sync_divergence_reset: bool = True,
    policy_history_steps: int = 0,
    policy_command_dim: int = 0,
    tracker_teacher: bool = False,
    physical_recovery_objective: bool = False,
    failure_frontier_prefix_steps: int = 0,
    failure_frontier_teacher_checkpoint: str | Path | None = None,
    reference_phase_retiming: bool = False,
    **kwargs,
) -> RefinerVecEnv:
    env = CarryBoxEnv(num_envs=num_envs, **kwargs)
    wrapper_kwargs = dict(
        reward_clip=reward_clip,
        sync_divergence_reset=sync_divergence_reset,
        policy_history_steps=policy_history_steps,
        policy_command_dim=policy_command_dim,
        tracker_teacher=tracker_teacher,
        physical_recovery_objective=physical_recovery_objective,
    )
    if reference_phase_retiming:
        if failure_frontier_prefix_steps != 200:
            raise ValueError("reference-phase retiming requires fixed prefix200")
        if failure_frontier_teacher_checkpoint is None:
            raise ValueError("reference-phase frozen Refiner checkpoint is required")
        return RefinerReferencePhaseVecEnv(
            env,
            teacher_checkpoint=failure_frontier_teacher_checkpoint,
            prefix_steps=failure_frontier_prefix_steps,
            **wrapper_kwargs,
        )
    if failure_frontier_prefix_steps:
        if failure_frontier_teacher_checkpoint is None:
            raise ValueError("failure-frontier teacher checkpoint is required")
        return RefinerFailureFrontierVecEnv(
            env,
            teacher_checkpoint=failure_frontier_teacher_checkpoint,
            prefix_steps=failure_frontier_prefix_steps,
            **wrapper_kwargs,
        )
    return RefinerVecEnv(env, **wrapper_kwargs)
