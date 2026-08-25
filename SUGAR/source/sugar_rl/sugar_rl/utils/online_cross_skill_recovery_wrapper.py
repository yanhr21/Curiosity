# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Online released-skill prefix for causal cross-skill recovery training.

The wrapper constructs every recovery start state by actually stepping PhysX:
one exact KickBox alignment action followed by a fixed or predeclared scheduled
number of exact CarryBox Generator+Tracker actions.  These prefix steps happen
between RSL-RL episodes and never enter PPO storage.  At the physical handoff,
the causal action composer receives the current 510-D Tracker observation,
current 36-D Carry command, current 36-D Kick command and selected-skill
two-way one-hot; no future or outcome label enters that 584-D actor input.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.networks.mlp import MLP
from sugar_il.wrapper.sugar_il_wrapper import GeneratorObs, GeneratorWrapper


TRACKER_OBSERVATION_DIM = 510
TRACKER_ACTION_DIM = 29
GENERATED_COMMAND_DIM = 36


def _load_released_tracker_actor(
    checkpoint_path: str | Path, device: str | torch.device
) -> MLP:
    """Load the exact released SUGAR Tracker actor into its official MLP."""

    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location=device, weights_only=True)
    state = payload["model_state_dict"]
    weight_keys = sorted(
        (
            name
            for name in state
            if name.startswith("actor.") and name.endswith("weight")
        ),
        key=lambda name: int(name.split(".")[1]),
    )
    if not weight_keys:
        raise RuntimeError(f"released Tracker has no actor weights: {path}")
    input_dim = int(state[weight_keys[0]].shape[1])
    output_dim = int(state[weight_keys[-1]].shape[0])
    hidden_dims = [int(state[name].shape[0]) for name in weight_keys[:-1]]
    if (
        input_dim != TRACKER_OBSERVATION_DIM
        or output_dim != TRACKER_ACTION_DIM
        or hidden_dims != [512, 256, 128]
    ):
        raise RuntimeError(
            "released Tracker geometry drift: "
            f"input={input_dim}, hidden={hidden_dims}, output={output_dim}"
        )
    actor = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation="elu",
    ).to(device)
    actor.load_state_dict(
        {
            name.removeprefix("actor."): value
            for name, value in state.items()
            if name.startswith("actor.")
        },
        strict=True,
    )
    actor.eval().requires_grad_(False)
    return actor


class _CausalShadowGenerator:
    """Run a released Generator causally without mutating the domain command."""

    def __init__(self, generator: GeneratorWrapper, command) -> None:
        self.generator = generator
        self.num_envs = command.num_envs
        self.device = command.device
        self.call_interval = int(command.generator_call_interval)
        if self.call_interval <= 0:
            raise RuntimeError("shadow Generator call interval drift")
        self._cpu_rng_state = torch.get_rng_state().clone()
        self._cuda_rng_state = (
            torch.cuda.get_rng_state(self.device).clone()
            if torch.cuda.is_available()
            else None
        )
        history = (generator.n_obs_steps - 1) * 5 + 1
        current = command._get_generator_obs()
        self.observation = {
            name: getattr(current, name).expand(-1, history, -1).clone()
            for name in (
                "obj_pos_b",
                "obj_ori_b",
                "joint_pos",
                "project_gravity",
                "target_obj_pos_b",
                "target_obj_ori_b",
                "last_command",
            )
        }
        self.generated = self._predict()
        self.last_command = current.last_command.clone()

    def _predict(self) -> torch.Tensor:
        global_cpu_state = torch.get_rng_state()
        global_cuda_state = (
            torch.cuda.get_rng_state(self.device)
            if self._cuda_rng_state is not None
            else None
        )
        try:
            torch.set_rng_state(self._cpu_rng_state)
            if self._cuda_rng_state is not None:
                torch.cuda.set_rng_state(self._cuda_rng_state, self.device)
            generated = self.generator.predict(GeneratorObs(**self.observation))
            self._cpu_rng_state = torch.get_rng_state().clone()
            if self._cuda_rng_state is not None:
                self._cuda_rng_state = torch.cuda.get_rng_state(self.device).clone()
        finally:
            torch.set_rng_state(global_cpu_state)
            if global_cuda_state is not None:
                torch.cuda.set_rng_state(global_cuda_state, self.device)
        expected_time = (self.generator.n_action_steps - 1) * 5 + 1
        if generated.shape != (
            self.num_envs,
            expected_time,
            GENERATED_COMMAND_DIM,
        ):
            raise RuntimeError(
                f"shadow Generator output geometry drift: {tuple(generated.shape)}"
            )
        if not torch.isfinite(generated).all():
            raise RuntimeError("shadow Generator produced non-finite commands")
        return generated

    def command_at(self, time_steps: torch.Tensor) -> torch.Tensor:
        index = (time_steps.to(dtype=torch.long) - 1) % self.call_interval
        env_index = torch.arange(self.num_envs, device=self.device)
        command = self.generated[env_index, index]
        self.last_command = command.unsqueeze(1).clone()
        return command

    def update_after_step(self, command) -> None:
        current = command._get_generator_obs()
        current.last_command = self.last_command
        for name, history in self.observation.items():
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = getattr(current, name).squeeze(1)
        refresh = (command.time_steps - 1) % self.call_interval == 0
        if torch.any(refresh):
            replacement = self._predict()
            self.generated[refresh] = replacement[refresh]


class OnlineCrossSkillRecoveryVecEnvWrapper(RslRlVecEnvWrapper):
    """Create a synchronized Carry-to-Kick physical handoff before each episode."""

    def __init__(
        self,
        env,
        *,
        clip_actions: float,
        carry_tracker_checkpoint: str | Path,
        kick_tracker_checkpoint: str | Path,
        carry_generator_checkpoint: str | Path,
        carry_prefix_steps: int = 9,
        carry_prefix_schedule: Sequence[int] | None = None,
        audit_path: str | Path | None = None,
        prefix_frame_callback: Callable[[str, int], None] | None = None,
        reward_clip: float | None = None,
        conditional_tinymdm_config: str | Path | None = None,
        conditional_tinymdm_checkpoint: str | Path | None = None,
        conditional_tinymdm_calibration: str | Path | None = None,
        conditional_tinymdm_class_id: int | None = None,
        conditional_tinymdm_reward_seed: int = 190001,
        conditional_tinymdm_reward_mode: str = "occupancy",
        conditional_tinymdm_task_reward_weight: float = 0.5,
        conditional_tinymdm_smp_reward_weight: float = 0.5,
        transition_selected_skill_id: int | None = None,
        transition_recovery_reward: bool = False,
    ) -> None:
        schedule = (
            (int(carry_prefix_steps),)
            if carry_prefix_schedule is None
            else tuple(int(step) for step in carry_prefix_schedule)
        )
        if not schedule or any(step <= 0 for step in schedule):
            raise ValueError("carry prefix schedule must contain positive steps")
        if len(set(schedule)) != len(schedule):
            raise ValueError("carry prefix schedule must not contain duplicates")
        super().__init__(env, clip_actions=clip_actions)
        self.base_env = env.unwrapped
        self.command = self.base_env.command_manager.get_term("motion")
        if self.command.generator is None:
            raise RuntimeError("recovery task requires the live KickBox Generator")
        self.carry_actor = _load_released_tracker_actor(
            carry_tracker_checkpoint, self.base_env.device
        )
        self.kick_actor = _load_released_tracker_actor(
            kick_tracker_checkpoint, self.base_env.device
        )
        self.carry_generator = GeneratorWrapper.load(
            checkpoint_path=str(Path(carry_generator_checkpoint).resolve()),
            device=self.base_env.device,
        )
        self.carry_prefix_schedule = schedule
        self.carry_prefix_steps = schedule[0]
        self.carry_prefix_install_counts = {step: 0 for step in schedule}
        self.audit_path = (
            Path(audit_path).expanduser().resolve() if audit_path is not None else None
        )
        self.prefix_frame_callback = prefix_frame_callback
        self.reward_clip = float(reward_clip) if reward_clip is not None else None
        if self.reward_clip is not None and self.reward_clip <= 0.0:
            raise ValueError("reward clip must be positive")
        conditional_values = (
            conditional_tinymdm_config,
            conditional_tinymdm_checkpoint,
            conditional_tinymdm_calibration,
            conditional_tinymdm_class_id,
        )
        if any(value is not None for value in conditional_values) and not all(
            value is not None for value in conditional_values
        ):
            raise ValueError("conditional TinyMDM reward configuration is incomplete")
        self.conditional_tinymdm_reward = None
        self.conditional_tinymdm_task_reward_weight = float(
            conditional_tinymdm_task_reward_weight
        )
        self.conditional_tinymdm_smp_reward_weight = float(
            conditional_tinymdm_smp_reward_weight
        )
        if (
            not math.isfinite(self.conditional_tinymdm_task_reward_weight)
            or not math.isfinite(self.conditional_tinymdm_smp_reward_weight)
            or self.conditional_tinymdm_task_reward_weight < 0.0
            or self.conditional_tinymdm_smp_reward_weight < 0.0
            or (
                self.conditional_tinymdm_task_reward_weight
                + self.conditional_tinymdm_smp_reward_weight
            )
            <= 0.0
        ):
            raise ValueError(
                "conditional TinyMDM reward weights must be finite and nonnegative"
            )
        if all(value is not None for value in conditional_values):
            from sugar_rl.utils.online_conditional_tinymdm_reward import (
                OnlineConditionalTinyMDMReward,
            )

            self.conditional_tinymdm_reward = OnlineConditionalTinyMDMReward(
                self.base_env,
                config_path=conditional_tinymdm_config,
                checkpoint_path=conditional_tinymdm_checkpoint,
                calibration_path=conditional_tinymdm_calibration,
                class_id=int(conditional_tinymdm_class_id),
                reward_seed=int(conditional_tinymdm_reward_seed),
                reward_mode=conditional_tinymdm_reward_mode,
            )
        if transition_selected_skill_id not in (None, -1, 0, 1):
            raise ValueError(
                "transition selected skill must be balanced=-1, Carry=0 or Kick=1"
            )
        self.transition_selected_skill_id = transition_selected_skill_id
        self.transition_recovery_reward_enabled = bool(transition_recovery_reward)
        if (
            self.transition_recovery_reward_enabled
            and self.conditional_tinymdm_reward is not None
        ):
            raise ValueError(
                "transition recovery reward and conditional TinyMDM reward are "
                "mutually exclusive diagnostics"
            )
        if self.transition_recovery_reward_enabled and transition_selected_skill_id is None:
            raise ValueError("transition recovery reward requires a selected skill")
        self.transition_selected_skill_ids = None
        self.transition_selected_skill_exposure = None
        if transition_selected_skill_id is not None:
            if transition_selected_skill_id == -1:
                self.transition_selected_skill_ids = (
                    torch.arange(self.num_envs, device=self.base_env.device) % 2
                ).to(dtype=torch.long)
            else:
                self.transition_selected_skill_ids = torch.full(
                    (self.num_envs,),
                    transition_selected_skill_id,
                    device=self.base_env.device,
                    dtype=torch.long,
                )
            self.transition_selected_skill_exposure = torch.zeros(
                (self.num_envs, 2),
                device=self.base_env.device,
                dtype=torch.long,
            )
        self.carry_shadow: _CausalShadowGenerator | None = None
        self._transition_handoff_object_xy: torch.Tensor | None = None
        self._transition_handoff_root_height: torch.Tensor | None = None
        self._transition_previous_displacement: torch.Tensor | None = None
        self.transition_recovery_reward_calls = 0
        self.transition_recovery_reward_max_abs = 0.0
        self.prefix_count = 0
        self.max_alignment_action_abs = 0.0
        self.max_carry_action_abs = 0.0
        self.max_handoff_observation_abs = 0.0
        self._install_prefix()

    def _augment_transition_observation(self, observations):
        if self.transition_selected_skill_id is None:
            return observations
        if self.carry_shadow is None:
            raise RuntimeError("transition observation requested before causal Generator")
        policy = observations["policy"]
        if self.transition_selected_skill_ids is None:
            raise RuntimeError("transition selected-skill tensor is missing")
        carry_command = self.carry_shadow.command_at(self.command.time_steps)
        kick_command = policy[:, :GENERATED_COMMAND_DIM].clone()
        selected_command = torch.where(
            self.transition_selected_skill_ids[:, None] == 0,
            carry_command,
            kick_command,
        )
        selected_skill = torch.zeros(
            (self.num_envs, 2), device=self.base_env.device, dtype=policy.dtype
        )
        selected_skill.scatter_(
            1, self.transition_selected_skill_ids[:, None], 1.0
        )
        observations["carry_skill_command"] = carry_command
        observations["kick_skill_command"] = kick_command
        observations["selected_skill_command"] = selected_command
        observations["selected_skill_id"] = selected_skill
        return observations

    def get_observations(self):
        return self._augment_transition_observation(super().get_observations())

    def _policy_observation(self):
        observations = super().get_observations()
        policy = observations["policy"]
        if policy.shape != (self.num_envs, TRACKER_OBSERVATION_DIM):
            raise RuntimeError(
                f"Tracker policy observation drift: {tuple(policy.shape)}"
            )
        if not torch.isfinite(policy).all():
            raise RuntimeError("Tracker policy observation is non-finite")
        return observations, policy

    def _assign_transition_conditions_for_prefix(self) -> None:
        """Keep each episode balanced while swapping every env's condition."""

        if self.transition_selected_skill_ids is None:
            return
        if self.transition_selected_skill_id == -1:
            parity = torch.arange(self.num_envs, device=self.base_env.device)
            self.transition_selected_skill_ids.copy_(
                (parity + self.prefix_count) % 2
            )
        if self.transition_selected_skill_exposure is None:
            raise RuntimeError("transition condition exposure tensor is missing")
        for skill in (0, 1):
            self.transition_selected_skill_exposure[:, skill] += (
                self.transition_selected_skill_ids == skill
            ).to(dtype=torch.long)

    @torch.inference_mode()
    def _install_prefix(self):
        """Advance all synchronized environments to the recovery handoff."""

        self._assign_transition_conditions_for_prefix()
        schedule_index = self.prefix_count % len(self.carry_prefix_schedule)
        self.carry_prefix_steps = self.carry_prefix_schedule[schedule_index]
        if self.conditional_tinymdm_reward is not None:
            self.conditional_tinymdm_reward.reset_history()
        observations, policy = self._policy_observation()
        alignment_action = self.kick_actor(policy)
        observations, _, dones, _ = super().step(alignment_action)
        if self.conditional_tinymdm_reward is not None:
            self.conditional_tinymdm_reward.observe_current_state()
        if torch.any(dones) or not torch.all(self.command.time_steps == 1):
            raise RuntimeError("one-step KickBox alignment prefix drift")
        self.max_alignment_action_abs = max(
            self.max_alignment_action_abs,
            float(torch.amax(torch.abs(alignment_action)).item()),
        )
        if self.prefix_frame_callback is not None:
            self.prefix_frame_callback("official Kick alignment", 0)

        shadow = _CausalShadowGenerator(self.carry_generator, self.command)
        for carry_step in range(self.carry_prefix_steps):
            policy = observations["policy"]
            selected_command = shadow.command_at(self.command.time_steps)
            carry_observation = policy.clone()
            carry_observation[:, :GENERATED_COMMAND_DIM] = selected_command
            carry_action = self.carry_actor(carry_observation)
            if not torch.isfinite(carry_action).all():
                raise RuntimeError("released CarryBox Tracker action became non-finite")
            observations, _, dones, _ = super().step(carry_action)
            if self.conditional_tinymdm_reward is not None:
                self.conditional_tinymdm_reward.observe_current_state()
            if torch.any(dones):
                raise RuntimeError("recovery prefix terminated before handoff")
            shadow.update_after_step(self.command)
            self.max_carry_action_abs = max(
                self.max_carry_action_abs,
                float(torch.amax(torch.abs(carry_action)).item()),
            )
            if self.prefix_frame_callback is not None:
                self.prefix_frame_callback("unrelated Carry prefix", carry_step)

        handoff_policy = observations["policy"]
        if (
            handoff_policy.shape != (self.num_envs, TRACKER_OBSERVATION_DIM)
            or not torch.isfinite(handoff_policy).all()
        ):
            raise RuntimeError("handoff observation contract failed")
        self.max_handoff_observation_abs = max(
            self.max_handoff_observation_abs,
            float(torch.amax(torch.abs(handoff_policy)).item()),
        )
        self.carry_shadow = shadow
        if self.transition_recovery_reward_enabled:
            self._transition_handoff_object_xy = (
                self.base_env.scene["obj"].data.root_pos_w[:, :2].clone()
            )
            self._transition_handoff_root_height = (
                self.base_env.scene["robot"].data.root_pos_w[:, 2].clone()
            )
            self._transition_previous_displacement = torch.zeros(
                self.num_envs,
                device=self.base_env.device,
                dtype=self._transition_handoff_object_xy.dtype,
            )
        observations = self._augment_transition_observation(observations)
        self.carry_prefix_install_counts[self.carry_prefix_steps] += 1
        self.prefix_count += 1
        if self.conditional_tinymdm_reward is not None:
            self.conditional_tinymdm_reward.prepare_reward()
        self._write_audit()
        return observations

    def _write_audit(self) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol": (
                "sugar_online_cross_skill_recovery_prefix_v3"
                if len(self.carry_prefix_schedule) > 1
                else "sugar_online_cross_skill_recovery_prefix_v2"
            ),
            "num_envs": int(self.num_envs),
            "prefix_count": int(self.prefix_count),
            "kick_alignment_steps": 1,
            "carry_prefix_steps": int(self.carry_prefix_steps),
            "carry_prefix_schedule": list(self.carry_prefix_schedule),
            "carry_prefix_install_counts": [
                self.carry_prefix_install_counts[step]
                for step in self.carry_prefix_schedule
            ],
            "prefix_schedule_is_episode_boundary_online": True,
            "ppo_prefix_transitions": 0,
            "state_teleport": False,
            "offline_replay": False,
            "max_alignment_action_abs": self.max_alignment_action_abs,
            "max_carry_action_abs": self.max_carry_action_abs,
            "max_handoff_observation_abs": self.max_handoff_observation_abs,
            "all_finite": True,
            "conditional_tinymdm_reward": (
                self.conditional_tinymdm_reward.audit()
                if self.conditional_tinymdm_reward is not None
                else None
            ),
            "conditional_tinymdm_task_reward_weight": (
                self.conditional_tinymdm_task_reward_weight
                if self.conditional_tinymdm_reward is not None
                else None
            ),
            "conditional_tinymdm_smp_reward_weight": (
                self.conditional_tinymdm_smp_reward_weight
                if self.conditional_tinymdm_reward is not None
                else None
            ),
            "transition_selected_skill_id": self.transition_selected_skill_id,
            "transition_selected_skill_counts": (
                [
                    int((self.transition_selected_skill_ids == skill).sum().item())
                    for skill in (0, 1)
                ]
                if self.transition_selected_skill_ids is not None
                else None
            ),
            "transition_selected_skill_assignment": (
                "env_parity_swapped_each_episode"
                if self.transition_selected_skill_id == -1
                else "fixed_condition"
                if self.transition_selected_skill_id in (0, 1)
                else None
            ),
            "transition_selected_skill_exposure_min_per_env": (
                [
                    int(
                        self.transition_selected_skill_exposure[:, skill]
                        .min()
                        .item()
                    )
                    for skill in (0, 1)
                ]
                if self.transition_selected_skill_exposure is not None
                else None
            ),
            "transition_selected_skill_exposure_max_per_env": (
                [
                    int(
                        self.transition_selected_skill_exposure[:, skill]
                        .max()
                        .item()
                    )
                    for skill in (0, 1)
                ]
                if self.transition_selected_skill_exposure is not None
                else None
            ),
            "transition_observation_is_causal": (
                self.transition_selected_skill_id is not None
            ),
            "transition_recovery_reward": {
                "enabled": self.transition_recovery_reward_enabled,
                "reward_calls": self.transition_recovery_reward_calls,
                "maximum_abs_reward": self.transition_recovery_reward_max_abs,
                "planar_progress_scale_m": 0.02,
                "foot_contact_threshold_n": 0.1,
                "foot_contact_bonus": 0.2,
                "upright_risk_start_m": 0.15,
                "upright_risk_full_m": 0.35,
                "future_or_outcome_labels_used": False,
                "actor_observation_augmented": False,
            },
        }
        temporary = self.audit_path.with_suffix(self.audit_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.audit_path)

    @staticmethod
    def _latest_filtered_force(sensor) -> torch.Tensor:
        force = sensor.data.force_matrix_w_history
        if force is None or force.ndim != 5 or force.shape[2:4] != (1, 1):
            raise RuntimeError("filtered foot-to-object ContactSensor geometry drift")
        return force[:, -1, 0, 0, :]

    def _transition_recovery_reward(self, dones: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Causal current-rollout reward; none of these labels enter the actor."""

        if (
            self._transition_handoff_object_xy is None
            or self._transition_handoff_root_height is None
            or self._transition_previous_displacement is None
            or self.transition_selected_skill_ids is None
        ):
            raise RuntimeError("transition recovery reward state is missing")
        zero = torch.zeros(self.num_envs, device=self.base_env.device)
        if torch.any(dones):
            return zero, {
                "progress": zero,
                "foot_contact": zero,
                "upright_risk": zero,
            }

        object_xy = self.base_env.scene["obj"].data.root_pos_w[:, :2]
        displacement = torch.linalg.vector_norm(
            object_xy - self._transition_handoff_object_xy, dim=-1
        )
        progress = torch.clamp(
            (displacement - self._transition_previous_displacement) / 0.02,
            -1.0,
            1.0,
        )
        self._transition_previous_displacement = displacement.clone()

        left_force = self._latest_filtered_force(
            self.base_env.scene.sensors["left_foot_forces"]
        )
        right_force = self._latest_filtered_force(
            self.base_env.scene.sensors["right_foot_forces"]
        )
        foot_contact = (
            (torch.linalg.vector_norm(left_force, dim=-1) > 0.1)
            | (torch.linalg.vector_norm(right_force, dim=-1) > 0.1)
        ).to(dtype=progress.dtype)

        root_height = self.base_env.scene["robot"].data.root_pos_w[:, 2]
        root_loss = self._transition_handoff_root_height - root_height
        upright_risk = torch.clamp((root_loss - 0.15) / 0.20, 0.0, 1.0)
        kick_mask = (self.transition_selected_skill_ids == 1).to(progress.dtype)
        reward = kick_mask * (progress + 0.2 * foot_contact - upright_risk)
        if not torch.isfinite(reward).all():
            raise RuntimeError("transition recovery reward became non-finite")
        self.transition_recovery_reward_calls += 1
        self.transition_recovery_reward_max_abs = max(
            self.transition_recovery_reward_max_abs,
            float(torch.amax(torch.abs(reward)).item()),
        )
        return reward, {
            "progress": progress,
            "foot_contact": foot_contact,
            "upright_risk": upright_risk,
        }

    @torch.inference_mode()
    def step(self, actions: torch.Tensor):
        observations, rewards, dones, extras = super().step(actions)
        if self.transition_recovery_reward_enabled:
            recovery_reward, components = self._transition_recovery_reward(dones)
            rewards = rewards + recovery_reward
            extras["transition_recovery_reward_mean"] = recovery_reward.mean()
            extras["transition_recovery_progress_mean"] = components["progress"].mean()
            extras["transition_recovery_foot_contact_mean"] = components[
                "foot_contact"
            ].mean()
            extras["transition_recovery_upright_risk_mean"] = components[
                "upright_risk"
            ].mean()
            if self.transition_recovery_reward_calls % 50 == 0:
                self._write_audit()
        if self.transition_selected_skill_id is not None and not torch.any(dones):
            if self.carry_shadow is None:
                raise RuntimeError("transition causal Generator state is missing")
            self.carry_shadow.update_after_step(self.command)
            observations = self._augment_transition_observation(observations)
        if self.conditional_tinymdm_reward is not None:
            if torch.any(dones):
                conditional_reward = torch.zeros_like(rewards)
            else:
                self.conditional_tinymdm_reward.observe_current_state()
                conditional_reward, sds_loss = self.conditional_tinymdm_reward.reward()
                extras["conditional_tinymdm_reward_mean"] = conditional_reward.mean()
                extras["conditional_tinymdm_sds_loss_mean"] = sds_loss.mean()
            rewards = (
                self.conditional_tinymdm_task_reward_weight * rewards
                + self.conditional_tinymdm_smp_reward_weight * conditional_reward
            )
            if self.conditional_tinymdm_reward.reward_calls % 50 == 0:
                self._write_audit()
        if self.reward_clip is not None:
            rewards = torch.clamp(rewards, -self.reward_clip, self.reward_clip)
        if torch.any(dones):
            if not torch.all(dones):
                raise RuntimeError(
                    "cross-skill recovery episodes lost synchronized timeout semantics"
                )
            # ManagerBasedRLEnv auto-resets timed-out environments and then
            # advances the motion command once before returning.  Re-enter the
            # public reset path at this synchronized episode boundary so every
            # recovery episode starts from the same clock-zero state as the
            # first episode.  This reset is outside PPO storage; only the
            # returned terminal reward/done belongs to the completed episode.
            super().reset()
            if not torch.all(self.command.time_steps == 0):
                raise RuntimeError("episode-boundary reset did not restore clock zero")
            observations = self._install_prefix()
        return observations, rewards, dones, extras
