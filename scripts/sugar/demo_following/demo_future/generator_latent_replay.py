"""TRAIN-only frozen diffusion replay loss on the complete official Generator.

No model modules are added. Actual physical futures remain supervision, and
saved diffusion states are not physical labels or actor observation fields.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.sugar.demo_following.demo_future.generator_dataset import ActualBranchGeometryDataset
from scripts.sugar.demo_following.demo_future.generator_paired_objective import PairedConditionGenerator, rng_state, restore_rng


class ActualBranchLatentReplayDataset(ActualBranchGeometryDataset):
    def __init__(self, replay_source, **kwargs):
        super().__init__(**kwargs)
        self.replay_source=Path(replay_source)
        result=json.loads((self.replay_source/'RESULT.json').read_text())
        gap_coverage=self.replay_source.name=='generator_train_diffusion_replay8_gaps2'
        self_replay=self.replay_source.name=='generator_train_diffusion_replay8_current_gaps'
        decision_source=self.replay_source
        expected=[158,178,197,245,277,298,318]
        if gap_coverage:
            protocol=json.loads((self.replay_source/'PROTOCOL.json').read_text())
            decision_source=Path(protocol['reused_replay_source'])
            expected=[158,178,197,221,245,261,277,298,318]
            if protocol['new_train_phases']!=[221,261] or Path(protocol['actual_corpus'])!=self.corpus or protocol['train_phases']!=expected:
                raise RuntimeError('New replay must remain bound to the inspected actual gap corpus')
        if self_replay:
            protocol=json.loads((self.replay_source/'PROTOCOL.json').read_text())
            expected=[158,178,197,221,245,261,277,298,318]
            if protocol['source_run']!=str(self.replay_source.parent/'matched_generator_branch_latent_replay01_gaps512') or protocol['train_phases']!=expected or self.corpus!=self.replay_source.parent/'generator_actual_branch_coverage_gaps2':
                raise RuntimeError('Own replay must bind the frozen ranked18case teacher and original actual corpus')
            audit=self.replay_source/'old_versus_own_gradient'
            gradient=json.loads((audit/'RESULT.json').read_text());readback=json.loads((audit/'SAVED_MEAN_READBACK.json').read_text())
            if not (gradient['checks_passed'] and gradient['plot_inspected'] and readback['checks_passed']):
                raise RuntimeError('Complete own-replay full gradient evidence before training adaptation')
            decision=json.loads((audit/'DECISION.json').read_text())
        else:
            decision=json.loads((decision_source/'generated_state_gradient_audit/DECISION.json').read_text())
        if not (result['checks_passed'] and result['plot_inspected'] and decision['permits_preparing_matched_objective_experiment']):
            raise RuntimeError('Require complete fresh TRAIN replay and fixed gradient decision')
        cases=2*len(expected)
        if not self.paired_supervision or self.real_case_count!=cases or self.repetitions!=8 or len(self)!=cases*8:
            raise RuntimeError('Preserve all declared real cases and eight replay seeds per TRAIN batch')
        with np.load(self.replay_source/'TRAIN_GENERATED_STATE_INPUTS.npz') as a:
            self.replay={k:a[k].copy() for k in a.files}
        if not (self.replay['xt'].shape==(8,16,cases,8,36) and self.replay['seeds'].tolist()==list(range(272230,272238))
                and self.replay['times'].tolist()==list(range(45,-1,-3))
                and self.replay['phases'].tolist()==[p for p in expected for _ in (0,1)]
                and self.replay['branches'].tolist()==[i%2 for i in range(cases)]
                and [r['shared_world_control_frame'] for r in self.timeline_metadata]==self.replay['phases'].tolist()):
            raise RuntimeError('Actual replay case, phase, seed or time mapping differs')
        if not all(np.isfinite(v).all() for v in self.replay.values()):
            raise RuntimeError('Nonfinite saved TRAIN replay')
        normalizer=self.get_normalizer()
        target=normalizer['action'].normalize(torch.from_numpy(self.arrays['future_command_target']))
        # Original replay normalization was CUDA float32. Exact raw targets and
        # case binding are checked below; cross-device normalization tolerates
        # only float32 arithmetic, never a refit or replacement target.
        if not np.allclose(target.numpy().astype(np.float64),self.replay['normalized_actual_target'],rtol=1e-6,atol=1e-6):
            raise RuntimeError('Saved latent supervision differs from actual TRAIN futures')
        pieces=[]
        for phase in expected:
            with np.load(self.replay_source/f'phase_{phase}_steps_16.npz') as a:
                pieces.append(a['xt'][:,0].transpose(0,2,1,3,4))
        if not np.array_equal(np.concatenate(pieces,axis=2),self.replay['xt']):
            raise RuntimeError('Combined replay differs from exact official recordings')

    def __getitem__(self,index):
        sample=super().__getitem__(index)
        seed_index=int(index)//self.real_case_count
        case=int(index)%self.real_case_count
        sample['replay_xt']=torch.from_numpy(self.replay['xt'][seed_index,:,case].copy())
        sample['replay_times']=torch.from_numpy(self.replay['times'].copy())
        sample['replay_seed']=torch.tensor(int(self.replay['seeds'][seed_index]),dtype=torch.int64)
        return sample


class LatentReplayGenerator(PairedConditionGenerator):
    def compute_replay_components(self,batch,training=True,time_index=None):
        # This exact previous full loss is always evaluated first, in both arms.
        original=super().compute_loss(batch,training)
        if time_index is None:time_index=self.generated_state_step%16
        device=batch['action'].device
        after_original=rng_state(device)
        try:
            auxiliary=self.compute_generated_state_loss(batch,training,time_index)
        finally:
            restore_rng(after_original,device)
        return original,auxiliary

    def compute_generated_state_loss(self,batch,training,time_index):
        if not 0<=time_index<16 or batch['replay_xt'].shape[1:]!=(16,8,36):
            raise RuntimeError('Use all original sixteen replay states and full8x36 commands')
        time=self.generated_state_times[time_index]
        if not bool((batch['replay_times'][:,time_index]==time).all()):
            raise RuntimeError('Replay time rows are not matched')
        xt=batch['replay_xt'][:,time_index].detach()
        target=self.normalizer['action'].normalize(batch['action']).detach()
        # Match official inference scalar arithmetic after add_noise has moved
        # scheduler arrays to CUDA. This snapshot has identical frozen values.
        alpha=self.generated_state_alphas_cpu[time]
        implied=((xt-(alpha**.5)*target)/((1-alpha)**.5)).detach()
        nobs=self.normalizer.normalize(batch['obs'])
        tokens=self.obs_encoder(nobs,training)
        timesteps=torch.full((len(xt),),time,device=xt.device,dtype=torch.long)
        prediction=self.model(xt,timesteps,cond=tokens,training=training,gen_attn_map=False)[0]
        return F.mse_loss(prediction,implied)

    def compute_loss(self,batch,training=True):
        if self.generated_state_weight==0:
            return super().compute_loss(batch,training)
        index=self.generated_state_step%16
        original,auxiliary=self.compute_replay_components(batch,training,index)
        old=dict(self.last_paired_loss) if self.obs_encoder.condition_mode!='zero_context' and self.paired_rank_weight!=0 and self.last_paired_loss is not None else dict(base=float(original.detach()),rank=0.,weight=self.paired_rank_weight,rank_not_evaluated_zero_context=self.obs_encoder.condition_mode=='zero_context')
        old.update(generated=float(auxiliary.detach()),generated_weight=self.generated_state_weight,
            generated_time_index=index,generated_time=self.generated_state_times[index],
            generated_step=self.generated_state_step,generated_rows=len(batch['action']))
        self.last_paired_loss=old
        if training:self.generated_state_step+=1
        return original+self.generated_state_weight*auxiliary


def configure_generated_state_objective(policy,settings):
    if type(policy) is not PairedConditionGenerator:
        raise TypeError('Extend only the complete already-loaded paired official Generator')
    if not (settings['weight']==.1 and settings['apply_to_both_arms'] and
            settings['times']==list(range(45,-1,-3)) and settings['seeds']==list(range(272230,272238))):
        raise ValueError('Use only the fixed predeclared both-arm replay experiment')
    rank_ablation=policy.paired_rank_weight==0 and settings.get('paired_rank_ablation')=='rank_removed_on_real18_replay'
    if not ((policy.paired_rank_weight==.25 or rank_ablation) and policy.paired_rank_function=='softplus' and policy.paired_rank_gradient_scope=='full_model'):
        raise ValueError('Preserve the full025 paired base objective')
    policy.__class__=LatentReplayGenerator
    policy.generated_state_weight=.1
    policy.generated_state_step=0
    policy.generated_state_times=tuple(settings['times'])
    policy.generated_state_alphas_cpu=policy.noise_scheduler.alphas_cumprod.detach().cpu().clone()
    return policy
