"""Conditioning adapter for the full released SUGAR Generator.

Preserves all twelve official Transformer blocks, all original observation
branches, action normalizer and diffusion solver. Only the existing target
encoder's first Linear gains168 zero-initialized input columns. Project method,
not official Zero-WAM, not an SMP latent, and not a replacement generator.
"""
import copy

import torch
from torch import nn

from sugar_il.model.common.normalizer import SingleFieldLinearNormalizer
from sugar_il.model.encoder.generator_state_encoder import GeneratorStateObsEncoder

CONTEXT_KEY = 'original_demo_geometry'
CONTEXT_WIDTH = 8 * 21


class OriginalDemoGeometryEncoder(GeneratorStateObsEncoder):
    def __init__(self, original, mode, history_dropout_probability=0., history_dropout_seed=272077,
                 goal_condition_mode='provided'):
        if mode not in ('zero_context', 'demo_geometry'):
            raise ValueError('Unknown geometry condition arm')
        if not original.use_target or not original.use_last_action:
            raise RuntimeError('This adapter requires the audited released target/last-command interface')
        super().__init__(shape_meta=copy.deepcopy(original.shape_meta),
                         feature_dim=original.feature_dim,
                         use_last_action=True, use_target=True)
        self.condition_mode = mode
        if goal_condition_mode not in ('provided', 'zero_diagnostic'):
            raise ValueError('Unknown diagnostic goal-input mode')
        self.goal_condition_mode = goal_condition_mode
        if not 0. <= history_dropout_probability < 1.:
            raise ValueError('History dropout probability must be in [0,1)')
        self.history_dropout_probability = float(history_dropout_probability)
        # Independent CPU RNG preserves the official shuffle/noise RNG stream.
        self.history_dropout_generator = torch.Generator(device='cpu').manual_seed(history_dropout_seed)
        self.last_history_dropout_mask = None
        layer = original.target_state_net[0]
        if layer.in_features != 9 or layer.out_features != 256:
            raise RuntimeError('Official target encoder shape changed')
        self.target_state_net[0] = nn.Linear(9 + CONTEXT_WIDTH, 256,
                                            device=layer.weight.device, dtype=layer.weight.dtype)
        self.to(device=layer.weight.device, dtype=layer.weight.dtype)
        state = copy.deepcopy(original.state_dict())
        state['target_state_net.0.weight'] = torch.nn.functional.pad(
            state['target_state_net.0.weight'], (0, CONTEXT_WIDTH))
        self.load_state_dict(state, strict=True)

    def forward(self, obs_dict, training=True):
        # Use the unchanged official branch methods; only the target branch
        # receives additional normalized original-demo geometry.
        obj = self.forward_obj_state(torch.cat([obs_dict['obj_pos_b'], obs_dict['obj_ori_b']], dim=-1))
        history = obs_dict['last_action']
        self.last_history_dropout_mask = None
        if training and self.training and self.history_dropout_probability:
            mask = torch.rand(len(history), generator=self.history_dropout_generator) < self.history_dropout_probability
            self.last_history_dropout_mask = mask.clone()
            history = history.masked_fill(mask.to(history.device)[:, None, None], 0.)
        if getattr(self, 'use_measured_robot_state', False):
            measured = torch.cat([obs_dict['joint_pos'], obs_dict['project_gravity']], dim=-1)
            if measured.shape[:-1] != history.shape[:-1] or measured.shape[-1] != 32 or not torch.isfinite(measured).all():
                raise RuntimeError('Require the audited current29joint+3gravity observations')
            history = torch.cat([history, measured], dim=-1)
        robot = self.forward_robot_state(history)
        context = obs_dict[CONTEXT_KEY]
        if context.shape[:-1] != obs_dict['target_obj_pos_b'].shape[:-1] or context.shape[-1] != CONTEXT_WIDTH:
            raise RuntimeError('Context must align with the current observation frame')
        if not torch.isfinite(context).all():
            raise RuntimeError('Nonfinite original-demo context')
        if self.condition_mode == 'zero_context':
            context = torch.zeros_like(context)
        goal = torch.cat([obs_dict['target_obj_pos_b'], obs_dict['target_obj_ori_b']], dim=-1)
        if self.goal_condition_mode == 'zero_diagnostic':
            goal = torch.zeros_like(goal)
        target = self.forward_target_state(torch.cat([goal, context], dim=-1))
        return torch.cat([obj, robot, target], dim=1)


def install_geometry_context(policy, training_geometry, mode):
    """Extend one full loaded official policy; return a direct preservation audit."""
    if len(policy.model.blocks) != 12 or policy.action_dim != 36 or policy.action_horizon != 8:
        raise RuntimeError('Require the audited complete released Generator')
    if policy.obs_encoder.shape_meta['obs']['n_obs_steps'] != 1:
        raise RuntimeError('Require the audited one-observation-frame interface')
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    original = policy.obs_encoder
    policy.obs_encoder = OriginalDemoGeometryEncoder(original, mode)
    # Fit only the new field on the predeclared TRAIN context; all old
    # normalizer statistics and executable command semantics stay unchanged.
    normalizer = SingleFieldLinearNormalizer.create_fit(
        training_geometry.reshape(-1, CONTEXT_WIDTH), mode='limits')
    policy.normalizer[CONTEXT_KEY] = normalizer
    policy.to(next(original.parameters()).device)
    key = 'obs_encoder.target_state_net.0.weight'
    after = policy.state_dict()
    preserved = {
        k: bool(torch.equal(v, after[k][:, :9] if k == key else after[k]))
        for k, v in before.items()}
    extra = [k for k in after if k not in before]
    audit = dict(all_original_state_entries_exact=all(preserved.values()),
                 changed_original_entries=[k for k, same in preserved.items() if not same],
                 new_input_columns_zero=not bool(after[key][:, 9:].count_nonzero()),
                 only_new_normalizer_entries=all(k.startswith('normalizer.params_dict.' + CONTEXT_KEY + '.') for k in extra),
                 old_target_input_width=9, added_input_width=CONTEXT_WIDTH,
                 full_parameter_count=sum(p.numel() for p in policy.parameters()),
                 full_transformer_layers=len(policy.model.blocks),
                 condition_mode=mode,
                 optimizer_created=False,
                 scope='Exact old state preservation while extending the official target input. Forward/backward and real execution require separate actual checks.')
    if not all(audit[k] for k in ('all_original_state_entries_exact', 'new_input_columns_zero', 'only_new_normalizer_entries')):
        raise RuntimeError('Full released state was not preserved: ' + str(audit))
    return audit


def restore_geometry_state(policy, model_state, mode, goal_condition_mode='provided'):
    """Restore a complete adapted state into a freshly loaded official Generator.

    The official wrapper always constructs the original9-D target encoder.
    Rebuild only the audited input adapter before strict full-state loading;
    learned normalizer statistics come from the checkpoint, never a new fit.
    """
    if len(policy.model.blocks) != 12 or policy.action_dim != 36 or policy.action_horizon != 8:
        raise RuntimeError('Restore requires the full released Generator architecture')
    weight = model_state['obs_encoder.target_state_net.0.weight']
    if tuple(weight.shape) != (256, 9 + CONTEXT_WIDTH):
        raise RuntimeError('Saved original-demo adapter width differs')
    if not any(k.startswith('normalizer.params_dict.' + CONTEXT_KEY + '.') for k in model_state):
        raise RuntimeError('Saved original-demo normalizer is missing')
    policy.obs_encoder = OriginalDemoGeometryEncoder(policy.obs_encoder, mode, goal_condition_mode=goal_condition_mode)
    robot_width = model_state['obs_encoder.robot_state_net.0.weight'].shape
    if 'obs_encoder.robot_state_net.0.measured_weight' in model_state:
        if robot_width != (256,36) or model_state['obs_encoder.robot_state_net.0.measured_weight'].shape != (256,32):
            raise RuntimeError('Invalid partitioned measured-robot checkpoint')
        from scripts.sugar.demo_following.demo_future.generator_measured_robot import install_measured_robot_inputs
        install_measured_robot_inputs(policy.obs_encoder,parameter_layout='partitioned_leaf')
    elif robot_width == (256, 68):
        from scripts.sugar.demo_following.demo_future.generator_measured_robot import install_measured_robot_inputs
        install_measured_robot_inputs(policy.obs_encoder)
    elif robot_width != (256, 36):
        raise RuntimeError('Unknown complete robot-input checkpoint shape')
    # Official LinearNormalizer dynamically restores its complete nested state.
    policy.load_state_dict(model_state, strict=True)
    policy.normalizer.requires_grad_(False)
    restored = policy.state_dict()
    exact = restored.keys() == model_state.keys() and all(
        torch.equal(value.detach().cpu(), model_state[key].detach().cpu()) for key, value in restored.items())
    if not exact:
        raise RuntimeError('Full adapted Generator state did not restore exactly')
    return dict(full_state_exact=True, condition_mode=mode,
                full_parameter_count=sum(p.numel() for p in policy.parameters()),
                normalizer_refitted=False, optimizer_updates=0,
                scope='Strict full-model inference restoration; optimizer resumption and physical behavior require separate checks.')
