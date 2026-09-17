"""Warm-start and checkpoint glue around the unchanged official training loop.

This module preserves the complete Generator, AdamW, and the workspace loop.
An explicitly selected local noise-coupling adapter changes only that loss step;
the default retains the original official loss.
Native TRAIN collection and a matched optimization protocol must be completed
before calling run(). Checkpoints retain AdamW but do not yet implement exact
interrupted-run resumption of the official loop's local scheduler and sampler.
"""
from pathlib import Path

import torch

from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper
from sugar_il.workspace.train_generator_workspace import TrainGeneratorWorkspace

from scripts.sugar.demo_following.demo_future.generator_demo_geometry import (
    OriginalDemoGeometryEncoder,
)


def warm_start_geometry_policy(parent_checkpoint, condition_mode, start_ckpt_path=None,
                               history_dropout_probability=0., history_dropout_seed=272077,
                               goal_condition_mode='provided', noise_coupling='official_assignment', paired_objective=None, generated_state_objective=None, robot_state_conditioning=None, actual_state_supervision=None):
    """Hydra factory: load the entire released model and extend only its input.

    The official workspace obtains normalization from ActualDemoGeometryDataset
    before its first forward call. No temporary or synthetic statistics are fit.
    """
    if start_ckpt_path is not None:
        raise ValueError('Use the explicit released warm start, not implicit workspace resume')
    wrapper = GeneratorWrapper.load(str(parent_checkpoint), device='cpu')
    policy = wrapper.policy
    if len(policy.model.blocks) != 12 or policy.action_dim != 36 or policy.action_horizon != 8:
        raise RuntimeError('Require the complete released12-layer8x36 Generator')
    before = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    policy.obs_encoder = OriginalDemoGeometryEncoder(policy.obs_encoder, condition_mode,
        history_dropout_probability=history_dropout_probability, history_dropout_seed=history_dropout_seed,
        goal_condition_mode=goal_condition_mode)
    after = policy.state_dict()
    changed_key = 'obs_encoder.target_state_net.0.weight'
    if before.keys() != after.keys() or not all(
        torch.equal(value, after[key][:, :9] if key == changed_key else after[key])
        for key, value in before.items()
    ) or after[changed_key][:, 9:].count_nonzero():
        raise RuntimeError('Released full model or normalization was not preserved')
    policy.normalizer.requires_grad_(False)
    if robot_state_conditioning is not None:
        settings=robot_state_conditioning
        if not (settings['mode']=='previous_command_plus_measured32' and settings['old_input_width']==36 and
                settings['new_input_width']==68 and settings['added_input_fields']==['joint_pos','project_gravity'] and
                settings['added_columns_initialization']=='exact_zero' and settings['applies_to_both_arms']):
            raise ValueError('Use only the separately audited measured32 robot-input adaptation')
        from scripts.sugar.demo_following.demo_future.generator_measured_robot import install_measured_robot_inputs
        install_measured_robot_inputs(policy.obs_encoder,parameter_layout=settings.get('parameter_layout','wide68'))
    from scripts.sugar.demo_following.demo_future.generator_noise_coupling import configure_noise_coupling
    configure_noise_coupling(policy, noise_coupling)
    if paired_objective is not None:
        from scripts.sugar.demo_following.demo_future.generator_paired_objective import configure_paired_objective
        configure_paired_objective(policy, paired_objective)
    if generated_state_objective is not None:
        from scripts.sugar.demo_following.demo_future.generator_latent_replay import configure_generated_state_objective
        configure_generated_state_objective(policy, generated_state_objective)
    if actual_state_supervision is not None:
        from scripts.sugar.demo_following.demo_future.generator_actual_state_supervision import configure_actual_state_supervision
        configure_actual_state_supervision(policy, actual_state_supervision)
    return policy


class GeometryGeneratorWorkspace(TrainGeneratorWorkspace):
    """Keep official run() intact; retain optimizer and finish each save directly."""

    def __init__(self, cfg, output_dir=None):
        if cfg.training.resume or cfg.policy.start_ckpt_path is not None:
            raise ValueError('This glue supports an explicit released warm start only')
        if cfg.training.gradient_accumulate_every != 1:
            raise ValueError('Use one optimizer step per official batch for matched accounting')
        super().__init__(cfg)
        self._output_dir = None if output_dir is None else str(Path(output_dir))
        # Upstream excludes AdamW whenever resume=False. Preserve it explicitly;
        # this does not claim the released checkpoint contains an optimizer.
        self.exclude_keys = ()

    def save_checkpoint(self, path=None, tag='latest', exclude_keys=None,
                        include_keys=None, use_thread=True):
        if exclude_keys and 'optimizer' in exclude_keys:
            raise ValueError('Complete AdamW must be retained')
        if not hasattr(self, 'optimizer'):
            raise RuntimeError('Construct the official optimizer before checkpointing')
        if path is None and tag == 'latest':
            path = Path(self.output_dir) / 'checkpoints' / f'epoch_{self.epoch:04d}.ckpt'
        return super().save_checkpoint(path=path, tag=tag, exclude_keys=exclude_keys,
                                       include_keys=include_keys, use_thread=False)
