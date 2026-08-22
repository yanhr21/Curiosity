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


class CarryBoxVecEnv:
    """rsl_rl VecEnv over :class:`CarryBoxEnv`."""

    def __init__(self, env: CarryBoxEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = N_DOF
        self.device = env.device
        self.max_episode_length = env.episode_length
        self.cfg = {"num_envs": env.num_envs, "episode_length": env.episode_length}
        self.episode_length_buf = torch.zeros(env.num_envs, dtype=torch.long,
                                              device=env.device)

    def _obs(self) -> TensorDict:
        policy = self.env.observe()
        priv = obs_890.build(self.env, teacher=False)
        # Teacher and critic share a term list; they differ only in which motion they read.
        # Until the env carries a second (refined) motion set they are the same tensor,
        # which is correct exactly when teacher_motion_folder == motion_folder -- how
        # play.py was run here. See obs_890.build's docstring.
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
