"""Frozen released Generator/Tracker checkpoint audit on actual stored inputs.

Uses official full classes and weights. Mask modifications below are temporary
zero-update diagnostics and are restored; no checkpoint or official code changes.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper, GeneratorObs, DOWNSAMPLE_RATE
from rsl_rl.networks.mlp import MLP

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'experiments/demo_following/demo_future_smp_v1/official_generator_audit'
CORPUS=ROOT/'experiments/demo_following/contact_event_reward_redesign_v1/deployable_goal_core_corpus_v1'
sys.path.insert(0,str(ROOT/'MimicKit/mimickit'))
from util import torch_util


def xyz_columns(xz):
    x,z=xz[...,:3],xz[...,3:]
    y=torch.cross(z,x,dim=-1)
    matrix=torch.stack([x,y,z],dim=-1)
    return matrix,matrix[...,:,:2].flatten(-2)


def load_actual_inputs(wrapper,task,device):
    seed={'CarryBox':271100,'KickBox':271200}[task]
    path=CORPUS/f'{task.lower()}_shard00_seed{seed}/TRACE.npz'
    archive=np.load(path,allow_pickle=False)
    core=archive['goal_policy_core_observation']
    policy=archive['policy_observation']
    times=np.arange(50,450,50)
    history=times[:,None]-DOWNSAMPLE_RATE*np.arange(wrapper.n_obs_steps-1,-1,-1)[None,:]
    if archive['done'][history,0].any():
        raise RuntimeError('Selected actual history crosses a reset')
    raw_core=torch.tensor(core[history,0],device=device)
    command=torch.tensor(policy[history,0,:36],device=device)
    _,obj_xy=xyz_columns(raw_core[...,100:106])
    _,goal_xy=xyz_columns(raw_core[...,115:121])
    if not wrapper.use_last_action:
        raise RuntimeError('This source lacks exact absolute joint calibration for a non-last-command checkpoint')
    raw={'obj_pos_b':raw_core[...,97:100],'obj_ori_b':obj_xy,'last_action':command}
    if wrapper.use_target:
        raw.update(target_obj_pos_b=raw_core[...,112:115],target_obj_ori_b=goal_xy)
    # Validate the conversion against the official wrapper on one current
    # frame. Inactive absolute-joint field is None, never invented joint data.
    current=raw_core[:,-1:]
    obj_matrix,_=xyz_columns(current[...,100:106]);goal_matrix,_=xyz_columns(current[...,115:121])
    oq=torch_util.matrix_to_quat(obj_matrix)[...,[3,0,1,2]]
    gq=torch_util.matrix_to_quat(goal_matrix)[...,[3,0,1,2]]
    obs=GeneratorObs(obj_pos_b=current[...,97:100],obj_ori_b=oq,joint_pos=None,
                     project_gravity=current[...,:3],target_obj_pos_b=current[...,112:115],
                     target_obj_ori_b=gq,last_command=command[:,-1:])
    wrapped=wrapper._prepare_obs_dict(obs)
    conversion_error=max(float((raw[k][:,-1:]-wrapped[k]).abs().max()) for k in wrapped)
    if conversion_error>1e-5:
        raise RuntimeError('Actual core orientation conversion differs from official wrapper')
    return raw,archive,times,{'trace':str(path.relative_to(ROOT)),'frames':times.tolist(),
                             'history_frames':history.tolist(),'environment':0,
                             'current_frame_wrapper_conversion_max_error':conversion_error,
                             'source':'Synchronized actual121-D core and stored510-D Tracker observation; first36 columns are recorded generated motion command. No actual actions are replaced with joint positions.'}


def audit(task,device,output):
    folder=ROOT/'SUGAR/demo_ckpts'/task
    wrapper=GeneratorWrapper.load(str(folder/'generator.ckpt'),device=str(device))
    generator=wrapper.policy.eval().requires_grad_(False)
    before={k:v.clone() for k,v in generator.state_dict().items()}
    raw,archive,times,inputs=load_actual_inputs(wrapper,task,device)
    with torch.no_grad():
        normalized=generator.normalizer.normalize(raw)
        condition=generator.obs_encoder(normalized,training=False)
        torch.manual_seed(272101)
        sample=torch.randn(len(times),generator.action_horizon,generator.action_dim,device=device)
        time=torch.full((len(times),),17,device=device,dtype=torch.long)
        prediction,_=generator.model(sample,time,condition,training=False)
        n=wrapper.n_obs_steps
        mask_rows=[]
        for i,block in enumerate(generator.model.blocks):
            ca=block.cross_attn
            mask_rows.append({'block':i,'use_attn_mask':ca.use_attn_mask,
                              'shape':list(ca.masks.shape) if ca.use_attn_mask else None,
                              'false_entries_per_mask':(~ca.masks).sum(dim=(1,2)).cpu().tolist() if ca.use_attn_mask else None,
                              'nominal_robot_columns':[n,2*n]})
        # Force selection of existing mask0; evaluation mode disables dropout.
        mask_diagnostic={'enabled':any(row['use_attn_mask'] for row in mask_rows),
                         'optimizer_updates':0}
        if mask_diagnostic['enabled']:
            active=[(i,b.cross_attn) for i,b in enumerate(generator.model.blocks) if b.cross_attn.use_attn_mask]
            original_probs=[ca.probs.clone() for _,ca in active]
            try:
                for _,ca in active:ca.probs.copy_(torch.tensor([1.,0.],device=device))
                current_mask,_=generator.model(sample,time,condition,training=True)
                for _,ca in active:ca.masks[0,:,n:2*n]=False
                repaired_mask,_=generator.model(sample,time,condition,training=True)
                mask_diagnostic.update(
                    forced_stored_mask0_vs_unmasked_max_error=float((current_mask-prediction).abs().max()),
                    temporary_correct_robot_slice_vs_unmasked_max_error=float((repaired_mask-prediction).abs().max()))
            finally:
                for (i,ca),probs in zip(active,original_probs):
                    ca.probs.copy_(probs)
                    ca.masks.copy_(before[f'model.blocks.{i}.cross_attn.masks'])
        else:
            mask_diagnostic['reason']='Released checkpoint disables attention masks; source default mask diagnostic is inapplicable.'
        zero_target=condition.clone()
        if wrapper.use_target:zero_target[:,2*n:3*n]=0
        target_removed,_=generator.model(sample,time,zero_target,training=False)
        torch.manual_seed(272103)
        sampled=generator.predict_action(raw)
        torch.manual_seed(272103)
        repeated=generator.predict_action(raw)
        dense=wrapper._parse_action(sampled)
    # Gradient-flow compatibility audit without an optimizer or any updates.
    generator.requires_grad_(True);generator.zero_grad(set_to_none=True)
    conditioned=generator.obs_encoder(generator.normalizer.normalize(raw),training=False)
    pred,_=generator.model(sample,time,conditioned,training=False)
    loss=pred.square().mean();loss.backward()
    gradient_names=[k for k,v in generator.named_parameters() if v.grad is not None]
    gradients_finite=all(torch.isfinite(v.grad).all().item() for v in generator.parameters() if v.grad is not None)
    generator.zero_grad(set_to_none=True);generator.requires_grad_(False)
    changed=[k for k,v in generator.state_dict().items() if not torch.equal(v,before[k])]
    payload=torch.load(folder/'tracker.pt',map_location=device,weights_only=True)
    state={k.removeprefix('actor.'):v for k,v in payload['model_state_dict'].items() if k.startswith('actor.')}
    actor=MLP(input_dim=510,output_dim=29,hidden_dims=[512,256,128],activation='elu').to(device)
    actor.load_state_dict(state,strict=True);actor.eval().requires_grad_(False)
    with torch.no_grad():
        observations=torch.tensor(archive['policy_observation'][times,0],device=device)
        actual=actor(observations)
        expected=torch.tensor(archive['action'][times+1,0],device=device)
        tracker_error=float((actual-expected).abs().max())
    checks={'strict_generator_loaded':True,'strict_tracker_actor_loaded':True,
            'generator_frozen_unchanged':not changed,'all_outputs_finite':all(torch.isfinite(x).all().item() for x in (prediction,sampled,dense,actual)),
            'fixed_seed_sampling_repeatable':torch.equal(sampled,repeated),
            'official_interpolation_geometry':tuple(dense.shape)==(len(times),(wrapper.n_action_steps-1)*5+1,36),
            'full_generator_backward_finite':gradients_finite and len(gradient_names)>0,
            'recorded_next_tracker_action_reproduced':tracker_error<1e-4}
    result={'task':task,'checks':checks,'passed':all(checks.values()),'inputs':inputs,
            'generator_parameters':sum(p.numel() for p in generator.parameters()),
            'tracker_actor_parameters':sum(p.numel() for p in actor.parameters()),
            'configuration':{'n_obs_steps':wrapper.n_obs_steps,'n_action_steps':wrapper.n_action_steps,
                             'use_last_action':wrapper.use_last_action,'use_target':wrapper.use_target,
                             'layers':len(generator.model.blocks),'width':condition.shape[-1],
                             'heads':generator.model.blocks[0].cross_attn.num_heads,
                             'stored_scheduler_target':str(wrapper.cfg.policy.noise_scheduler._target_),
                             'actual_wrapper_scheduler':type(generator.noise_scheduler).__name__,
                             'inference_steps':generator.num_inference_steps},
            'condition_shape':list(condition.shape),'command_prediction_shape':list(sampled.shape),
            'interpolated_command_shape':list(dense.shape),'checkpoint_mask_buffers':mask_rows,
            'mask_diagnostic':dict(mask_diagnostic,buffers_restored=not changed),
            'zero_target_token_diagnostic_max_error':float((target_removed-prediction).abs().max()),
            'tracker_actual_next_action_max_error':tracker_error,'backward_parameter_tensors':len(gradient_names),
            'changed_generator_state_keys':changed,
            'scope':'Official full released Generator and official released Tracker actor audit; mask repair is diagnostic only and does not alter saved weights, training, inference solver or physical policy behavior.'}
    np.savez(output/f'{task}_PROBES.npz',command_prediction=sampled.cpu().numpy(),
             interpolated_command=dense.cpu().numpy(),tracker_action=actual.cpu().numpy())
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    output=parser.parse_args().output
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login','mgmtserver')):
        raise RuntimeError('Run in the retained compute step')
    output.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    rows=[]
    for task in ('CarryBox','KickBox'):
        row=audit(task,torch.device('cuda:0'),output);rows.append(row)
        (output/f'{task}_RESULT.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    result={'execution_completed':True,'passed':all(r['passed'] for r in rows),'tasks':rows,
            'optimizer_updates':0,'next_action':'Use exact released full Generator/Tracker paths to prepare selected-demo conditioning and faithful motion-command data; keep36-D reference commands separate from29-D executed actions.'}
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
