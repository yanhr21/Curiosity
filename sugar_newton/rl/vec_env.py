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

import torch
from tensordict import TensorDict

from sugar_newton.rl import obs_890
from sugar_newton.rl.carrybox_env import N_DOF, OBS_DIM, CarryBoxEnv


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

    def _obs(self) -> TensorDict:
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
        _, reward, done, extras = self.env.step(actions)
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
        return self._obs(), reward, done, info


def make(num_envs: int, **kwargs) -> CarryBoxVecEnv:
    return CarryBoxVecEnv(CarryBoxEnv(num_envs=num_envs, **kwargs))


OBS_DIMS = {"policy": OBS_DIM, "critic": obs_890.OBS_DIM_890, "teacher": obs_890.OBS_DIM_890}


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
        self.cfg = EnvCfg(
            num_envs=env.num_envs,
            episode_length=env.episode_length,
            substeps=env.substeps,
            dt=env.dt,
            clips=list(env.clip_names),
            observation_contract="official_refiner_890d",
            reward_clip=self.reward_clip,
            sync_divergence_reset=self.sync_divergence_reset,
        )
        self.episode_length_buf = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def _obs(self) -> TensorDict:
        privileged = obs_890.build(self.env, teacher=False)
        return TensorDict(
            {"policy": privileged, "critic": privileged},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    def reset(self) -> tuple[TensorDict, dict]:
        self.env.reset()
        self.episode_length_buf.zero_()
        return self._obs(), {}

    def step(self, actions: torch.Tensor):
        _, reward, done, extras = self.env.step(actions)
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
            done = torch.ones_like(done)
            extras["timeout"] = torch.zeros_like(done)
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
    **kwargs,
) -> RefinerVecEnv:
    return RefinerVecEnv(
        CarryBoxEnv(num_envs=num_envs, **kwargs),
        reward_clip=reward_clip,
        sync_divergence_reset=sync_divergence_reset,
    )
