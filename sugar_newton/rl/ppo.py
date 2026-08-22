# SPDX-License-Identifier: BSD-3-Clause
"""PPO with SUGAR's hyperparameters, small enough to carry no extra dependency.

``rsl-rl-lib`` is what SUGAR trains with, but the Newton container has no package mirror
and its interpreter is not executable from the login node, so installing it there is a
fight. The algorithm is standard, the hyperparameters are read straight from
``agents/rsl_rl_ppo_cfg.py:BasePPORunnerCfg``, and the network shapes are the checkpoint's
own -- which is the part that actually matters, because it lets training *start from
``tracker.pt``* instead of from scratch.

That warm start is the point. The official tracker already walks, squats and grips in
Newton (about a third of the reference lift); fine-tuning it under this contact model is a
far shorter path than 30k iterations from random weights.

Asymmetry note: SUGAR's critic takes an 890-D privileged observation built from future
reference frames and teacher terms. This critic takes the same 510-D actor observation and
is trained from scratch, so the pretrained critic weights are deliberately *not* loaded --
loading a critic whose input distribution differs would be worse than initialising fresh.
Building the privileged observation is the obvious next improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    """BasePPORunnerCfg / RslRlPpoAlgorithmCfg, verbatim."""

    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    entropy_coef: float = 0.005
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    max_grad_norm: float = 1.0
    init_noise_std: float = 1.0
    hidden_dims: tuple[int, ...] = (512, 256, 128)


def mlp(inp: int, hidden: tuple[int, ...], out: int) -> nn.Sequential:
    """Linear/ELU stack laid out so state_dict indices match the checkpoint's."""
    layers: list[nn.Module] = []
    last = inp
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        super().__init__()
        self.actor = mlp(obs_dim, cfg.hidden_dims, act_dim)
        self.critic = mlp(obs_dim, cfg.hidden_dims, 1)
        self.std = nn.Parameter(cfg.init_noise_std * torch.ones(act_dim))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(obs), self.std.expand_as(self.actor(obs)))

    def act(self, obs: torch.Tensor):
        d = self.distribution(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.critic(obs).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor):
        d = self.distribution(obs)
        return (d.log_prob(act).sum(-1), d.entropy().sum(-1),
                self.critic(obs).squeeze(-1), d.mean, d.stddev)

    def load_sugar_actor(self, path: str) -> None:
        """Warm-start the actor (and only the actor) from SUGAR's tracker checkpoint."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        mapped = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
        missing = self.actor.load_state_dict(mapped, strict=True)
        if "std" in sd:
            with torch.no_grad():
                self.std.copy_(sd["std"])
        print(f"[ppo] warm-started actor from {path} (iter {ck.get('iter')}); "
              f"critic left random{'' if not missing else f'; {missing}'}")


class RolloutBuffer:
    def __init__(self, steps: int, num_envs: int, obs_dim: int, act_dim: int, device):
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.obs = z(steps, num_envs, obs_dim)
        self.act = z(steps, num_envs, act_dim)
        self.logp = z(steps, num_envs)
        self.val = z(steps, num_envs)
        self.rew = z(steps, num_envs)
        self.done = z(steps, num_envs)
        self.adv = z(steps, num_envs)
        self.ret = z(steps, num_envs)
        self.steps, self.num_envs = steps, num_envs
        self.i = 0

    def add(self, obs, act, logp, val, rew, done) -> None:
        i = self.i
        self.obs[i], self.act[i], self.logp[i] = obs, act, logp
        self.val[i], self.rew[i], self.done[i] = val, rew, done.float()
        self.i += 1

    def finish(self, last_val: torch.Tensor, gamma: float, lam: float) -> None:
        """GAE-lambda. Bootstrapping is cut at `done`, timeouts included.

        Cutting at a timeout biases the value target low for episodes that were merely
        truncated. SUGAR handles this the same way at this level; the honest fix is to
        bootstrap on timeout specifically, which needs the pre-reset observation.
        """
        adv = 0.0
        for t in reversed(range(self.steps)):
            nv = last_val if t == self.steps - 1 else self.val[t + 1]
            nonterminal = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * nv * nonterminal - self.val[t]
            adv = delta + gamma * lam * nonterminal * adv
            self.adv[t] = adv
        self.ret = self.adv + self.val
        self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)
        self.i = 0

    def batches(self, num_mini_batches: int):
        n = self.steps * self.num_envs
        flat = lambda x: x.reshape(n, *x.shape[2:])  # noqa: E731
        obs, act = flat(self.obs), flat(self.act)
        logp, adv, ret, val = flat(self.logp), flat(self.adv), flat(self.ret), flat(self.val)
        perm = torch.randperm(n, device=obs.device)
        for chunk in perm.chunk(num_mini_batches):
            yield obs[chunk], act[chunk], logp[chunk], adv[chunk], ret[chunk], val[chunk]


class PPO:
    def __init__(self, policy: ActorCritic, cfg: PPOConfig, device):
        self.policy, self.cfg, self.device = policy, cfg, device
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        self.lr = cfg.learning_rate

    def update(self, buf: RolloutBuffer) -> dict[str, float]:
        c = self.cfg
        stats = {"surrogate": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0}
        count = 0
        for _ in range(c.num_learning_epochs):
            for obs, act, old_logp, adv, ret, old_val in buf.batches(c.num_mini_batches):
                logp, ent, val, mu, sigma = self.policy.evaluate(obs, act)

                ratio = (logp - old_logp).exp()
                surrogate = -torch.min(
                    adv * ratio, adv * ratio.clamp(1.0 - c.clip_param, 1.0 + c.clip_param)
                ).mean()

                if c.use_clipped_value_loss:
                    clipped = old_val + (val - old_val).clamp(-c.clip_param, c.clip_param)
                    value_loss = torch.max((val - ret).square(),
                                           (clipped - ret).square()).mean()
                else:
                    value_loss = (val - ret).square().mean()

                loss = surrogate + c.value_loss_coef * value_loss - c.entropy_coef * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm)
                self.opt.step()

                # rsl_rl's adaptive schedule: steer the step size by the observed KL
                with torch.no_grad():
                    kl = (torch.log(sigma / sigma.detach() + 1e-5)
                          + (sigma.detach().square() + (mu.detach() - mu).square())
                          / (2.0 * sigma.square()) - 0.5).sum(-1).mean()
                if c.schedule == "adaptive":
                    if kl > c.desired_kl * 2.0:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < c.desired_kl / 2.0 and kl > 0.0:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr

                stats["surrogate"] += float(surrogate)
                stats["value"] += float(value_loss)
                stats["entropy"] += float(ent.mean())
                stats["kl"] += float(kl)
                count += 1
        return {k: v / max(count, 1) for k, v in stats.items()} | {"lr": self.lr}
