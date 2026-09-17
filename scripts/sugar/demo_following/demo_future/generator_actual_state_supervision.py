"""Expert command denoising on actual generated states, using the full existing model.

This adds no network, latent representation or physical-state prediction target.
"""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_latent_replay import LatentReplayGenerator
from scripts.sugar.demo_following.demo_future.generator_noise_coupling import IndependentNoiseGenerator
from scripts.sugar.demo_following.demo_future.generator_paired_objective import rng_state,restore_rng


class ActualStateSupervisedGenerator(LatentReplayGenerator):
    def expert_indices(self, step):
        return torch.cat([group[(torch.arange(36)+int(step)*36)%len(group)] for group in self.expert_groups])

    def expert_batch(self, step, device):
        indices=self.expert_indices(step)
        return dict(obs={k:v[indices].to(device) for k,v in self.expert_observations.items()},
                    action=self.expert_targets[indices].to(device)),indices

    def compute_loss(self,batch,training=True):
        original=super().compute_loss(batch,training)
        if self.expert_weight==0:return original
        step=self.expert_step
        device=batch['action'].device
        after_original=rng_state(device)
        extra,indices=self.expert_batch(step,device)
        try:
            torch.manual_seed(self.expert_seed+step)
            if device.type=='cuda':torch.cuda.manual_seed_all(self.expert_seed+step)
            auxiliary=IndependentNoiseGenerator.compute_loss(self,extra,training)
        finally:
            restore_rng(after_original,device)
        self.last_paired_loss=dict(self.last_paired_loss,expert=float(auxiliary.detach()),expert_weight=self.expert_weight,
                                  expert_step=step,expert_rows=len(indices),expert_seed=self.expert_seed+step)
        if training:
            self.expert_counts.index_add_(0,indices,torch.ones_like(indices))
            self.expert_step+=1
        return original+self.expert_weight*auxiliary


def configure_actual_state_supervision(policy,settings):
    assert type(policy) is LatentReplayGenerator
    assert settings['weight']==.1 and settings['rows_per_group']==36 and settings['batch_rows']==144
    assert settings['apply_to_both_arms'] and settings['seed']==272500
    root=Path(settings['corpus']);result=json.loads((root/'RESULT.json').read_text());plan=json.loads((root/'PROTOCOL.json').read_text())
    assert result['checks_passed'] and result['examples']==355 and len(plan['groups'])==4
    with np.load(root/'EXPERT_PLAN_ON_ACTUAL_STATES.npz') as a:arrays={k:torch.from_numpy(a[k].copy()) for k in a.files}
    assert arrays['expert_future_command_target'].shape==(355,8,36)
    assert all(torch.isfinite(v).all() for v in arrays.values())
    policy.__class__=ActualStateSupervisedGenerator
    policy.expert_targets=arrays.pop('expert_future_command_target')
    policy.expert_observations=arrays
    policy.expert_groups=[torch.tensor(g['indices'],dtype=torch.long) for g in plan['groups']]
    policy.expert_counts=torch.zeros(355,dtype=torch.long)
    policy.expert_weight=.1;policy.expert_step=0;policy.expert_seed=272500
    assert sorted(torch.cat(policy.expert_groups).tolist())==list(range(355))
    return policy
