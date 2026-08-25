import copy
from pathlib import Path

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks.mlp import MLP

class BCPPO(PPO):
    def __init__(
        self,
        policy,
        teacher_ckpt=None,
        stage3_distill_weight_floor=0.0,
        training_mask_obs_group=None,
        distill_mask_start_step=0,
        actor_hold_start_step=-1,
        actor_hold_end_step=-1,
        behavior_anchor_checkpoint=None,
        behavior_anchor_coef=0.0,
        behavior_anchor_start_step=0,
        stage3_tactile_only_actor=False,
        bc_only_steps=500,
        critic_warmup_steps=1000,
        full_ppo_warmup_steps=2000,
        teacher_mean_only=False,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)

        self.stage3_distill_weight_floor = float(stage3_distill_weight_floor)
        self.training_mask_obs_group = training_mask_obs_group
        self.distill_mask_start_step = int(distill_mask_start_step)
        self.actor_hold_start_step = int(actor_hold_start_step)
        self.actor_hold_end_step = int(actor_hold_end_step)
        self.behavior_anchor_coef = float(behavior_anchor_coef)
        self.behavior_anchor_start_step = int(behavior_anchor_start_step)
        self.stage3_tactile_only_actor = bool(stage3_tactile_only_actor)
        self.teacher_mean_only = bool(teacher_mean_only)
        self._actor_optimization_parameters = tuple(
            parameter
            for name, parameter in self.policy.named_parameters()
            if not name.startswith("critic.")
        )
        self._tactile_actor_parameters = tuple(
            parameter
            for name, parameter in self.policy.named_parameters()
            if name.startswith("actor_tactile_encoder.")
        )
        tactile_parameter_ids = {
            id(parameter) for parameter in self._tactile_actor_parameters
        }
        self._base_actor_parameters = tuple(
            parameter
            for parameter in self._actor_optimization_parameters
            if id(parameter) not in tactile_parameter_ids
        )
        self._actor_optimizer_reset_done = False
        if not 0.0 <= self.stage3_distill_weight_floor <= 1.0:
            raise ValueError(
                "stage3_distill_weight_floor must lie in [0, 1], got "
                f"{self.stage3_distill_weight_floor}"
            )
        if self.distill_mask_start_step < 0:
            raise ValueError("distill_mask_start_step must be non-negative")
        hold_disabled = (
            self.actor_hold_start_step == -1
            and self.actor_hold_end_step == -1
        )
        hold_valid = (
            self.actor_hold_start_step >= 0
            and self.actor_hold_end_step >= self.actor_hold_start_step
        )
        if not (hold_disabled or hold_valid):
            raise ValueError(
                "actor hold must be disabled with -1/-1 or use an ordered "
                "non-negative interval"
            )
        if self.behavior_anchor_coef < 0.0:
            raise ValueError("behavior_anchor_coef must be non-negative")
        if self.behavior_anchor_start_step < 0:
            raise ValueError("behavior_anchor_start_step must be non-negative")
        
        self.distill_loss_coef = 1.0
        self.bc_only_steps = int(bc_only_steps)
        self.critic_warmup_steps = int(critic_warmup_steps)
        self.full_ppo_warmup_steps = int(full_ppo_warmup_steps)
        if not (
            0 <= self.bc_only_steps <= self.critic_warmup_steps
            < self.full_ppo_warmup_steps
        ):
            raise ValueError(
                "BCPPO stage boundaries must satisfy "
                "0 <= bc_only <= critic_warmup < full_ppo_warmup"
            )

        self.distill_loss_fn = nn.MSELoss()
        self.update_step = 0

        if teacher_ckpt is not None:
            print(f"[Distill] 正在从 {teacher_ckpt} 加载 Teacher 模型...")
            checkpoint = torch.load(teacher_ckpt, map_location=self.device)
            state_dict = checkpoint['model_state_dict']

            actor_weight_keys = sorted(
                [k for k in state_dict.keys() if k.startswith('actor.') and 'weight' in k],
                key=lambda x: int(x.split('.')[1])
            )

            num_obs = state_dict[actor_weight_keys[0]].shape[1]
            num_actions = state_dict[actor_weight_keys[-1]].shape[0]
            hidden_dims = [state_dict[k].shape[0] for k in actor_weight_keys[:-1]]

            print(f"[Distill] 检测到 Teacher 架构: 输入={num_obs}, 隐藏层={hidden_dims}, 输出={num_actions}")

            self.teacher_model = MLP(
                input_dim=num_obs,
                output_dim=num_actions,
                hidden_dims=hidden_dims,
                activation="elu",
            ).to(self.device)

            mlp_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('actor.'):
                    new_key = k.replace('actor.', '')
                    mlp_state_dict[new_key] = v
            
            self.teacher_model.load_state_dict(mlp_state_dict)
            self.teacher_model.eval()
            
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            self.teacher_std = checkpoint['model_state_dict']['std'].detach().to(self.device)
            print("[Distill] Teacher loaded successfully and parameters frozen.")
        else:
            self.teacher_model = None
            print("[Warning] No teacher_ckpt provided")
            assert False

        self.behavior_anchor_policy = None
        self.behavior_anchor_std = None
        self.behavior_anchor_checkpoint = None
        if behavior_anchor_checkpoint is not None:
            anchor_path = Path(behavior_anchor_checkpoint).expanduser().resolve()
            if not anchor_path.is_file():
                raise FileNotFoundError(anchor_path)
            anchor_payload = torch.load(
                anchor_path, map_location=self.device, weights_only=False
            )
            anchor_state = anchor_payload.get("model_state_dict")
            if not isinstance(anchor_state, dict):
                raise KeyError("behavior anchor is missing model_state_dict")
            self.behavior_anchor_policy = copy.deepcopy(self.policy)
            self.behavior_anchor_policy.load_state_dict(anchor_state, strict=True)
            self.behavior_anchor_policy.eval()
            for parameter in self.behavior_anchor_policy.parameters():
                parameter.requires_grad_(False)
            self.behavior_anchor_checkpoint = str(anchor_path)
            if "std" in anchor_state:
                self.behavior_anchor_std = anchor_state["std"].detach().to(self.device)
            elif "log_std" in anchor_state:
                self.behavior_anchor_std = anchor_state["log_std"].detach().to(
                    self.device
                ).exp()
            else:
                raise KeyError("behavior anchor is missing std/log_std")
            print(
                "[BCPPO] Loaded frozen deployment behavior anchor: "
                f"{anchor_path}",
                flush=True,
            )
        elif self.behavior_anchor_coef > 0.0:
            raise ValueError(
                "positive behavior_anchor_coef requires a behavior anchor checkpoint"
            )

    def _reset_actor_optimizer_state(self) -> dict[str, int]:
        """Remove stale pre-hold Adam moments before PPO receives authority."""

        cleared_parameters = 0
        cleared_tensors = 0
        for parameter in self._actor_optimization_parameters:
            state = self.optimizer.state.get(parameter)
            if not state:
                continue
            cleared_parameters += 1
            for value in state.values():
                if torch.is_tensor(value):
                    value.zero_()
                    cleared_tensors += 1
        self._actor_optimizer_reset_done = True
        return {
            "cleared_parameters": cleared_parameters,
            "cleared_state_tensors": cleared_tensors,
        }

    def _reduce_distill_loss(self, per_sample_loss, obs_batch):
        """Reduce teacher KL on transitions where the student actually acts.

        Official SUGAR does not configure ``training_mask_obs_group`` and
        therefore retains its original full-batch mean.  The live-handoff
        Plan-15 tasks do configure the mask: their teacher-controlled pickup
        prefix is useful for reaching a physical handoff, but it is not part
        of the deployed student's state distribution and must not keep
        steering the actor after handoff training begins.
        """

        if (
            self.training_mask_obs_group is None
            or self.update_step < self.distill_mask_start_step
        ):
            return per_sample_loss.mean()
        active_weight = obs_batch[
            self.training_mask_obs_group
        ].reshape(per_sample_loss.shape[0], -1)[:, 0]
        active_weight = (active_weight > 0.5).to(per_sample_loss.dtype)
        return (per_sample_loss * active_weight).sum() / active_weight.sum().clamp_min(
            1.0
        )

    def update(self):  # noqa: C901
        actor_hold_active = (
            self.actor_hold_start_step >= 0
            and self.actor_hold_start_step
            <= self.update_step
            <= self.actor_hold_end_step
        )
        behavior_anchor_active = (
            self.behavior_anchor_policy is not None
            and self.behavior_anchor_coef > 0.0
            and self.update_step >= self.behavior_anchor_start_step
            and not actor_hold_active
        )
        actor_optimizer_reset_report = None
        if (
            self.actor_hold_end_step >= 0
            and self.update_step == self.actor_hold_end_step + 1
            and not self._actor_optimizer_reset_done
        ):
            actor_optimizer_reset_report = self._reset_actor_optimizer_state()
            print(
                "[BCPPO] Cleared stale actor optimizer state at PPO handoff: "
                f"{actor_optimizer_reset_report}",
                flush=True,
            )
        if self.update_step >= self.bc_only_steps:
            self.schedule = "adaptive"
        else:
            self.schedule = "fixed"

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_distill_loss = 0 # Teacher 统计
        mean_distill_weight = 0
        mean_behavior_anchor_loss = 0
        mean_actor_optimizer_active = 0
        mean_base_actor_optimizer_active = 0
        mean_tactile_actor_optimizer_active = 0

        if self.training_mask_obs_group is not None:
            rollout_mask = self.storage.observations[
                self.training_mask_obs_group
            ][: self.storage.step].reshape(-1)
            active_transitions = int((rollout_mask > 0.5).sum().item())
            total_transitions = int(rollout_mask.numel())
            self.last_training_mask_report = {
                "observation_group": self.training_mask_obs_group,
                "active_policy_transitions": active_transitions,
                "masked_teacher_transitions": (
                    total_transitions - active_transitions
                ),
                "total_transitions": total_transitions,
            }
        else:
            self.last_training_mask_report = None

        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            # we assume policy group is always there and needs augmentation
            original_batch_size = obs_batch.batch_size[0]

            if self.training_mask_obs_group is None:
                active_weight = torch.ones(
                    original_batch_size, device=self.device
                )
            else:
                active_weight = obs_batch[
                    self.training_mask_obs_group
                ][:original_batch_size].reshape(original_batch_size, -1)[:, 0]
                active_weight = (active_weight > 0.5).to(torch.float32)
            active_count = active_weight.sum()
            active_denom = active_count.clamp_min(1.0)

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            if self.training_mask_obs_group is not None and active_count > 1:
                with torch.no_grad():
                    flat_advantage = advantages_batch.reshape(-1)
                    active_mean = (flat_advantage * active_weight).sum() / active_count
                    active_variance = (
                        (flat_advantage - active_mean).square() * active_weight
                    ).sum() / active_count
                    advantages_batch = (
                        advantages_batch - active_mean
                    ) / torch.sqrt(active_variance + 1.0e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                assert False, "please check symmtry"
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # compute number of augmentations per sample
                # we assume policy group is always there and needs augmentation
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            behavior_anchor_loss = mu_batch.sum() * 0.0
            if behavior_anchor_active:
                with torch.no_grad():
                    anchor_mu = self.behavior_anchor_policy.act_inference(
                        obs_batch
                    )[:original_batch_size]
                    anchor_sigma = self.behavior_anchor_std.expand_as(anchor_mu)
                anchor_kl_per_sample = (
                    torch.log(sigma_batch / anchor_sigma + 1.0e-5)
                    + (
                        anchor_sigma.square()
                        + (anchor_mu - mu_batch).square()
                    )
                    / (2.0 * sigma_batch.square())
                    - 0.5
                ).sum(dim=-1)
                behavior_anchor_loss = (
                    anchor_kl_per_sample * active_weight
                ).sum() / active_denom

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = (kl * active_weight).sum() / active_denom

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = (
                torch.max(surrogate, surrogate_clipped) * active_weight
            ).sum() / active_denom

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss_per_sample = torch.max(
                    value_losses, value_losses_clipped
                ).reshape(original_batch_size, -1).mean(dim=-1)
            else:
                value_loss_per_sample = (
                    (returns_batch - value_batch)
                    .pow(2)
                    .reshape(original_batch_size, -1)
                    .mean(dim=-1)
                )
            value_loss = (
                value_loss_per_sample * active_weight
            ).sum() / active_denom
            entropy_mean = (
                entropy_batch.reshape(original_batch_size, -1).mean(dim=-1)
                * active_weight
            ).sum() / active_denom

            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_obs_list = []
                    for obs_group in self.policy.obs_groups["teacher"]:
                        teacher_obs_list.append(obs_batch[obs_group])
                        if torch.isnan(obs_batch[obs_group]).any():
                            assert False
                    teacher_obs_batch = torch.cat(teacher_obs_list, dim=-1)
                    # print("teacher obs batch shape:", teacher_obs_batch.shape)

                    teacher_action_mean = self.teacher_model(teacher_obs_batch)
                    teacher_action_std = self.teacher_std
                
                if self.teacher_mean_only:
                    # Recovery fine-tuning needs the released deterministic
                    # action as a behavior anchor while retaining its own low
                    # exploration noise.  Matching the teacher's historical
                    # stochastic std would reintroduce the OOD failures that
                    # the anchor is intended to prevent.
                    distill_loss_per_sample = (
                        teacher_action_mean - mu_batch
                    ).square().mean(dim=-1)
                else:
                    # Official SUGAR KL(T||S) behavior.
                    log_std_s = torch.log(sigma_batch + 1e-8)
                    log_std_t = torch.log(teacher_action_std + 1e-8)
                    distill_loss_per_sample = (
                        log_std_s - log_std_t
                        + (
                            teacher_action_std.pow(2)
                            + (teacher_action_mean - mu_batch).pow(2)
                        )
                        / (2.0 * (sigma_batch.pow(2) + 1e-7))
                        - 0.5
                    ).sum(dim=-1)
                distill_loss = self._reduce_distill_loss(
                    distill_loss_per_sample, obs_batch
                )
                mean_distill_loss += distill_loss.item()




            # =========================
            # Stage 1: Pure Distill
            # =========================
            if self.update_step < self.bc_only_steps:
                distill_weight = 1.0
                loss = self.distill_loss_coef * distill_loss

            # =========================
            # Stage 2: Distill + Critic
            # =========================
            elif self.update_step < self.critic_warmup_steps:
                alpha = min((self.update_step - self.bc_only_steps) / (self.critic_warmup_steps - self.bc_only_steps), 1.0)
                distill_weight = 1.0
                loss = (
                    self.distill_loss_coef * distill_loss 
                    + alpha * self.value_loss_coef * value_loss
                    )

            # =========================
            # Stage 3: Full PPO + Distill
            # =========================
            else:
                alpha = min((self.update_step - self.critic_warmup_steps) / (self.full_ppo_warmup_steps - self.critic_warmup_steps), 1.0)
                distill_weight = max(
                    1.0 - alpha, self.stage3_distill_weight_floor
                )
                loss = (
                    surrogate_loss * alpha
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_mean * alpha
                    + self.distill_loss_coef * distill_loss * distill_weight
                )

            # Once the live-handoff student reaches its admitted checkpoint,
            # the remainder of critic warmup must not keep moving the actor.
            # The actor and critic are disjoint modules, so a value-only loss
            # preserves the actor bit-for-bit while still completing critic
            # preparation for PPO.
            if actor_hold_active:
                distill_weight = 0.0
                if self.update_step < self.critic_warmup_steps:
                    critic_alpha = min(
                        (self.update_step - self.bc_only_steps)
                        / (self.critic_warmup_steps - self.bc_only_steps),
                        1.0,
                    )
                    loss = critic_alpha * self.value_loss_coef * value_loss
                else:
                    loss = self.value_loss_coef * value_loss
            if behavior_anchor_active:
                loss = loss + self.behavior_anchor_coef * behavior_anchor_loss


            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # A zero-valued gradient is not the same as no actor update for
            # Adam: old momentum still moves parameters. Before deployment-
            # aligned masking begins, full-trajectory distillation provides
            # actor credit. Afterwards, minibatches without a student-
            # controlled transition must set actor gradients to None.
            actor_optimizer_active = (
                not actor_hold_active
                and (
                    self.update_step < self.distill_mask_start_step
                    or active_count.item() > 0
                )
            )
            if not actor_optimizer_active:
                for parameter in self._actor_optimization_parameters:
                    parameter.grad = None
            tactile_only_active = (
                actor_optimizer_active
                and self.stage3_tactile_only_actor
                and self.update_step > self.actor_hold_end_step
            )
            if tactile_only_active:
                for parameter in self._base_actor_parameters:
                    parameter.grad = None
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_mean.item()
            mean_distill_weight += distill_weight
            mean_behavior_anchor_loss += behavior_anchor_loss.item()
            mean_actor_optimizer_active += float(actor_optimizer_active)
            mean_base_actor_optimizer_active += float(
                actor_optimizer_active and not tactile_only_active
            )
            mean_tactile_actor_optimizer_active += float(
                actor_optimizer_active
            )
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_distill_loss /= num_updates
        mean_distill_weight /= num_updates
        mean_behavior_anchor_loss /= num_updates
        mean_actor_optimizer_active /= num_updates
        mean_base_actor_optimizer_active /= num_updates
        mean_tactile_actor_optimizer_active /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "distill": mean_distill_loss,
            "distill_weight": mean_distill_weight,
            "actor_hold_active": float(actor_hold_active),
            "behavior_anchor": mean_behavior_anchor_loss,
            "behavior_anchor_active": float(behavior_anchor_active),
            "actor_optimizer_active_fraction": mean_actor_optimizer_active,
            "base_actor_optimizer_active_fraction": (
                mean_base_actor_optimizer_active
            ),
            "tactile_actor_optimizer_active_fraction": (
                mean_tactile_actor_optimizer_active
            ),
            "actor_optimizer_state_reset": float(
                actor_optimizer_reset_report is not None
            ),
        }
        if self.last_training_mask_report is not None:
            active_transitions = int(
                self.last_training_mask_report["active_policy_transitions"]
            )
            total_transitions = int(
                self.last_training_mask_report["total_transitions"]
            )
            loss_dict["post_handoff_transitions"] = active_transitions
            loss_dict["post_handoff_fraction"] = (
                float(active_transitions / total_transitions)
                if total_transitions > 0
                else 0.0
            )
            loss_dict["distill_post_handoff_only"] = float(
                self.update_step >= self.distill_mask_start_step
            )
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.update_step += 1

        return loss_dict
