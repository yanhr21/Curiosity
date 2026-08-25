# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Online released-skill prefix for causal cross-skill recovery training.

The wrapper constructs every recovery start state by actually stepping PhysX:
one exact KickBox alignment action followed by a fixed number of exact
CarryBox Generator+Tracker actions.  These prefix steps happen between RSL-RL
episodes and never enter PPO storage.  The trainable policy therefore receives
only the current 510-D Tracker observation at the physical handoff state.
"""

from __future__ import annotations

import json
import math
import os
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
        audit_path: str | Path | None = None,
        prefix_frame_callback: Callable[[str, int], None] | None = None,
        reward_clip: float | None = None,
        conditional_tinymdm_config: str | Path | None = None,
        conditional_tinymdm_checkpoint: str | Path | None = None,
        conditional_tinymdm_calibration: str | Path | None = None,
        conditional_tinymdm_class_id: int | None = None,
        conditional_tinymdm_reward_seed: int = 190001,
        conditional_tinymdm_task_reward_weight: float = 0.5,
        conditional_tinymdm_smp_reward_weight: float = 0.5,
    ) -> None:
        if carry_prefix_steps <= 0:
            raise ValueError("carry_prefix_steps must be positive")
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
        self.carry_prefix_steps = int(carry_prefix_steps)
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
            )
        self.prefix_count = 0
        self.max_alignment_action_abs = 0.0
        self.max_carry_action_abs = 0.0
        self.max_handoff_observation_abs = 0.0
        self._install_prefix()

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

    @torch.inference_mode()
    def _install_prefix(self):
        """Advance all synchronized environments to the recovery handoff."""

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
        self.prefix_count += 1
        self._write_audit()
        return observations

    def _write_audit(self) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol": "sugar_online_cross_skill_recovery_prefix_v2",
            "num_envs": int(self.num_envs),
            "prefix_count": int(self.prefix_count),
            "kick_alignment_steps": 1,
            "carry_prefix_steps": int(self.carry_prefix_steps),
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
            ),
            "conditional_tinymdm_smp_reward_weight": (
                self.conditional_tinymdm_smp_reward_weight
            ),
        }
        temporary = self.audit_path.with_suffix(self.audit_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.audit_path)

    @torch.inference_mode()
    def step(self, actions: torch.Tensor):
        observations, rewards, dones, extras = super().step(actions)
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
