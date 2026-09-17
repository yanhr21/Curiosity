"""Explicit independent-noise loss adaptation of the complete official Generator.

No model modules are created or replaced. This local method variant removes only
the target-dependent cross-example noise permutation from the official loss.
Normalization, timestep sampling, scheduler, complete denoiser and epsilon/sample
MSE remain the same. It must not be called the unchanged official training loss.
"""
import torch
import torch.nn.functional as F
from sugar_il.policy.generator import Generator


class IndependentNoiseGenerator(Generator):
    def compute_loss(self, batch, training=True):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        trajectory = self.normalizer['action'].normalize(batch['action'])
        obs_tokens = self.obs_encoder(nobs, training)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # The only intentional loss change: retain independently sampled noise
        # instead of applying official noise_assignment(trajectory, noise).
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                  (trajectory.shape[0],), device=trajectory.device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred, _ = self.model(noisy_trajectory, timesteps, cond=obs_tokens,
                             training=training, gen_attn_map=False)
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == 'epsilon':
            target = noise
        elif prediction_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f'Unsupported official prediction type {prediction_type}')
        return F.mse_loss(pred, target)


def configure_noise_coupling(policy, mode):
    if type(policy) is not Generator:
        raise TypeError('Adapt only a fully loaded original official Generator')
    if mode == 'independent':
        # Keep the complete loaded instance, modules and parameter identities;
        # only dispatch the explicitly recorded loss variant during training.
        policy.__class__ = IndependentNoiseGenerator
    elif mode != 'official_assignment':
        raise ValueError('Unknown Generator noise-coupling method')
    return policy
