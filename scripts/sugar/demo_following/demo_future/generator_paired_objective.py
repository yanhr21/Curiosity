"""Local paired-condition loss on the complete official SUGAR Generator.

Uses the existing full model and observed TRAIN targets; creates no modules.
The official IID-adapted base loss and its RNG stream are preserved. Paired
supervision lives outside obs and is never read by inference.
"""
import torch
import torch.nn.functional as F
from scripts.sugar.demo_following.demo_future.generator_noise_coupling import IndependentNoiseGenerator
from scripts.sugar.demo_following.demo_future.generator_demo_geometry import CONTEXT_KEY


def rng_state(device):
    return torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None


def restore_rng(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state(state[1], device)


class PairedConditionGenerator(IndependentNoiseGenerator):
    def compute_loss(self, batch, training=True):
        # The control has identically zero ranking gradient. Keeping the exact
        # original base call also permits endpoint-level control reproduction.
        if self.obs_encoder.condition_mode == 'zero_context' or self.paired_rank_weight == 0:
            return super().compute_loss(batch, training)
        base, rank = self.compute_paired_losses(batch, training)
        self.last_paired_loss = dict(base=float(base.detach()), rank=float(rank.detach()),
                                    weight=self.paired_rank_weight)
        return base + self.paired_rank_weight * rank

    def target_encoder_rank_prediction(self, nobs, noisy, timesteps,
                                       encoder_rng, denoiser_rng, training):
        """Original full forwards; only target encoder parameters keep rank grads.

        Detached denoiser weights still differentiate its conditioning input.
        The base path continues to update every original trainable parameter.
        No modules are constructed, mutated, or permanently frozen here.
        """
        encoder_params = {name: p if name.startswith('target_state_net.') else p.detach()
                          for name, p in self.obs_encoder.named_parameters()}
        encoder_buffers = {name: b.detach() for name, b in self.obs_encoder.named_buffers()}
        restore_rng(encoder_rng, noisy.device)
        tokens = torch.func.functional_call(self.obs_encoder,
            (encoder_params, encoder_buffers), (nobs, training), strict=True)
        restore_rng(denoiser_rng, noisy.device)
        return torch.func.functional_call(self.model,
            ({name: p.detach() for name, p in self.model.named_parameters()},
             {name: b.detach() for name, b in self.model.named_buffers()}),
            (noisy, timesteps), dict(cond=tokens, training=training, gen_attn_map=False),
            strict=True)[0]

    def compute_paired_losses(self, batch, training=True):
        """Separate differentiable components, also used by full-model preflight."""
        if self.noise_scheduler.config.prediction_type != 'epsilon':
            raise RuntimeError('Paired ranking is declared only for the original epsilon model')
        if 'paired_action' not in batch or 'paired_geometry' not in batch:
            raise RuntimeError('Require actual TRAIN paired supervision outside actor observations')
        nobs = self.normalizer.normalize(batch['obs'])
        trajectory = self.normalizer['action'].normalize(batch['action'])
        device = trajectory.device
        encoder_rng = rng_state(device)
        obs_tokens = self.obs_encoder(nobs, training)
        noise = torch.randn(trajectory.shape, device=device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                  (trajectory.shape[0],), device=device).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        denoiser_rng = rng_state(device)
        prediction, _ = self.model(noisy, timesteps, cond=obs_tokens, training=training, gen_attn_map=False)
        base = F.mse_loss(prediction, noise)
        final_rng = rng_state(device)
        wrong = dict(nobs)
        wrong[CONTEXT_KEY] = self.normalizer[CONTEXT_KEY].normalize(batch['paired_geometry'])
        # Reuse the corresponding stochastic masks, then restore the next-call
        # stream. Wrong-context computation cannot perturb future base batches.
        try:
            if self.paired_rank_gradient_scope == 'target_encoder_only':
                if self.obs_encoder.history_dropout_probability:
                    raise RuntimeError('Declared gradient routing retains intact history')
                rank_prediction = self.target_encoder_rank_prediction(
                    nobs, noisy, timesteps, encoder_rng, denoiser_rng, training)
                wrong_prediction = self.target_encoder_rank_prediction(
                    wrong, noisy, timesteps, encoder_rng, denoiser_rng, training)
            else:
                rank_prediction = prediction
                restore_rng(encoder_rng, device)
                wrong_tokens = self.obs_encoder(wrong, training)
                restore_rng(denoiser_rng, device)
                wrong_prediction, _ = self.model(noisy, timesteps, cond=wrong_tokens, training=training, gen_attn_map=False)
        finally:
            restore_rng(final_rng, device)
        paired = self.normalizer['action'].normalize(batch['paired_action'])
        alpha = self.noise_scheduler.alphas_cumprod.to(device)[timesteps]
        scale = (alpha / (1 - alpha) * (trajectory - paired).square().mean((1, 2))).detach()
        if not bool((scale > 0).all()) or not bool(torch.isfinite(scale).all()):
            raise RuntimeError('Actual paired future separation must be finite and positive')
        correct_error = (rank_prediction - noise).square().mean((1, 2))
        wrong_error = (wrong_prediction - noise).square().mean((1, 2))
        margin = self.paired_rank_margin + (correct_error - wrong_error) / scale
        penalty = F.relu(margin) if self.paired_rank_function == 'hinge' else F.softplus(margin)
        rank = (scale * penalty).mean()
        return base, rank


def configure_paired_objective(policy, settings):
    if type(policy) is not IndependentNoiseGenerator:
        raise TypeError('Pair only the already loaded complete IID-adapted official Generator')
    rank_ablation=settings['weight']==0 and settings.get('ablation')=='rank_removed_on_real18_replay'
    if (settings['weight'] not in (0.1, 0.25) and not rank_ablation) or settings['margin_fraction'] != 0.1:
        raise ValueError('Use a separately predeclared audited weight and the fixed margin')
    policy.__class__ = PairedConditionGenerator
    policy.paired_rank_weight = float(settings['weight'])
    policy.paired_rank_margin = float(settings['margin_fraction'])
    scope = settings.get('gradient_scope', 'full_model')
    if scope not in ('full_model', 'target_encoder_only') or (scope == 'target_encoder_only' and settings['weight'] != .25):
        raise ValueError('Use the predeclared gradient scope and coefficient')
    policy.paired_rank_gradient_scope = scope
    function = settings.get('rank_function', 'softplus')
    if function not in ('softplus', 'hinge') or (function == 'hinge' and (scope != 'full_model' or settings['weight'] != .25)):
        raise ValueError('Use only the separately audited full-model fixed0.25 hinge alternative')
    policy.paired_rank_function = function
    policy.last_paired_loss = None
    return policy
