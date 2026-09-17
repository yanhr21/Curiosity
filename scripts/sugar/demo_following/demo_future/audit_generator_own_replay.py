"""Frozen full-model gradients for replacing old teacher latents with own latents.

Only diagnostic forwards and derivatives are performed. Original actual TRAIN
targets and the complete official architecture, sampler and endpoints remain fixed.
"""
import json
import argparse
import os
import socket

import dill
import numpy as np
import torch
import torch.nn.functional as F

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import (
    BASE, PARENT, ActualBranchGeometryDataset, GeneratorWrapper, CONTEXT_KEY,
    restore_geometry_state, write,
)
from scripts.sugar.demo_following.demo_future.audit_generator_generated_state_gradient import pair_stats

RUN=BASE/'matched_generator_branch_latent_replay01_gaps512'
OWN=BASE/'generator_train_diffusion_replay8_current_gaps'
LEGACY=BASE/'generator_train_diffusion_replay8_gaps2'
OUT=OWN/'old_versus_own_gradient'


def compare(names,grads):
    supervised,old_aux,own_aux=grads
    old=[a+b for a,b in zip(supervised,old_aux)]
    candidate=[a+b for a,b in zip(supervised,own_aux)]
    pairs={'old_vs_candidate':(old,candidate),'own_vs_candidate':(own_aux,candidate),
           'supervised_vs_candidate':(supervised,candidate),'old_aux_vs_own_aux':(old_aux,own_aux)}
    result={}
    for key,(a,b) in pairs.items():
        values=pair_stats(names,a,b)
        result[key]={group:dict(left_norm=v['supervised_norm'],right_norm=v['generated_state_norm'],
                               dot=v['dot'],cosine=v['cosine']) for group,v in values.items()}
    return result


def read_saved_means():
    """Independent CPU readback; no model construction or derivatives."""
    torch.set_num_threads(8)
    result=json.loads((OUT/'RESULT.json').read_text())
    assert result['checks_passed'] and result['plot_inspected']
    saved=torch.load(OUT/'MEAN_GRADIENTS.pt',map_location='cpu')
    names=saved['parameter_names']
    grads=[[values[n] for n in names] for values in saved['mean_gradients']]
    current=compare(names,grads);checks={}
    for pair,groups in current.items():
        for group,values in groups.items():
            for key,value in values.items():
                expected=result['aggregate'][pair][group][key]
                checks[f'{pair}_{group}_{key}']=value==expected if value is None else bool(np.isclose(value,expected,rtol=1e-10,atol=1e-12))
    path=OUT/'SAVED_MEAN_READBACK.json';assert not path.exists()
    write(path,dict(checks_passed=all(checks.values()),checks=checks,recomputed=current,scope='Independent CPU readback of all saved full mean gradients, full and layer-group direction statistics. Zero model forwards, gradients, optimizer steps or samples.'))
    assert all(checks.values())
    print(json.dumps(dict(checks=len(checks),passed=True)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--readback',action='store_true')
    args=parser.parse_args()
    if args.readback:
        read_saved_means()
        return
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    record=json.loads((OWN/'RESULT.json').read_text())
    assert record['checks_passed'] and record['plot_inspected']
    protocol=json.loads((BASE/'matched_generator_branch_latent_only_gaps512/NEXT_TRAIN_REPLAY_PROTOCOL.json').read_text())['following_gradient_audit']
    assert protocol['source_run']==str(RUN) and protocol['candidate_replay_source']==str(OWN)
    assert protocol['gradient_rows']==128 and protocol['auxiliary_weight']==.1 and protocol['ranking_weight']==.25
    plan=json.loads((RUN/'PROTOCOL.json').read_text());phases=plan['data']['train']['phases']
    data=[]
    for folder in (LEGACY,OWN):
        with np.load(folder/'TRAIN_GENERATED_STATE_INPUTS.npz') as a:data.append({k:a[k].copy() for k in a.files})
    previous,current=data;seeds=current['seeds'].tolist();times=current['times'].tolist()
    assert seeds==protocol['seeds'] and times==protocol['times']
    assert len(phases)==9 and not set(phases).intersection((218,258))
    checks={key+'_exact_between_replays':np.array_equal(previous[key],current[key]) for key in current if key!='xt'}
    assert all(checks.values())
    reference=json.loads((RUN/'frozen_evaluation/gap_component_gradient/RESULT.json').read_text())
    assert reference['checks_passed'] and reference['plot_inspected']
    reference_rows={(r['seed'],r['time']):r for r in reference['rows']}
    OUT.mkdir(exist_ok=False);write(OUT/'PROTOCOL.json',protocol)
    samples=[];expected=[];clean=[]
    for phase in phases:
        ds=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state'])
        samples.extend(ds[i] for i in (0,1))
        with np.load(OWN/f'phase_{phase}_steps_16.npz') as a:
            expected.append(a['epsilon'][:,0].transpose(0,2,1,3,4))
            clean.append(a['pred_original'][:,0].transpose(0,2,1,3,4))
    expected=np.concatenate(expected,axis=2);clean=np.concatenate(clean,axis=2)
    obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
    actions=torch.stack([s['action'] for s in samples]).cuda()
    with (RUN/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as stream:
        state=torch.load(stream,pickle_module=dill,map_location='cpu')['state_dicts']['model']
    policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
    restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
    policy.cuda().eval();policy.normalizer.requires_grad_(False)
    names,parameters=zip(*[(n,p) for n,p in policy.named_parameters() if p.requires_grad])
    normalized=policy.normalizer.normalize(obs);target=policy.normalizer['action'].normalize(actions)
    permutation=torch.arange(18,device='cuda')^1
    wrong=dict(normalized);wrong[CONTEXT_KEY]=normalized[CONTEXT_KEY][permutation]
    checks['full8319216']=sum(p.numel() for p in policy.parameters())==8319216
    checks['actual_targets_exact']=np.array_equal(target.detach().cpu().numpy().astype(np.float64),current['normalized_actual_target'])
    checks['shared_causal_fields_exact']=all(torch.equal(obs[k],obs[k][permutation]) for k in ('obj_pos_b','obj_ori_b','last_action'))
    xt=[torch.from_numpy(a['xt']).cuda() for a in data]
    noise=torch.from_numpy(current['initial_gaussian']).cuda()
    scheduler=policy.noise_scheduler;alpha=scheduler.alphas_cumprod.detach().cpu().clone()
    assert scheduler.config.prediction_type=='epsilon' and scheduler.config.clip_sample
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
    with torch.no_grad():
        tokens=[policy.obs_encoder({k:v[i:i+1] for k,v in normalized.items()},training=False) for i in range(18)]
        for si,seed in enumerate(seeds):
            exact_epsilon=True;exact_clean=True
            for ti,time in enumerate(times):
                for case in range(18):
                    point=policy.model(xt[1][si,ti,case:case+1],torch.tensor(time,device='cuda'),cond=tokens[case],training=False,gen_attn_map=False)[0]
                    reconstructed=((xt[1][si,ti,case:case+1]-((1-alpha[time])**.5)*point)/(alpha[time]**.5)).clamp(-1,1)
                    exact_epsilon=exact_epsilon and np.array_equal(point[0].cpu().numpy(),expected[si,ti,case])
                    exact_clean=exact_clean and np.array_equal(reconstructed[0].cpu().numpy(),clean[si,ti,case])
            checks[f'{seed}_all288_own_epsilon_exact']=exact_epsilon
            checks[f'{seed}_all288_own_clean_exact']=exact_clean
            assert all(checks.values())
    aggregate=[[torch.zeros_like(p) for p in parameters] for _ in range(3)]
    rows=[];seed_means={}
    for si,seed in enumerate(seeds):
        accum=[[torch.zeros_like(p) for p in parameters] for _ in range(3)]
        for ti,time in enumerate(times):
            t=torch.full((18,),time,device='cuda',dtype=torch.long)
            q=scheduler.add_noise(target,noise[si],t)
            cond=policy.obs_encoder(normalized,training=False);other=policy.obs_encoder(wrong,training=False)
            pred=policy.model(q,t,cond=cond,training=False,gen_attn_map=False)[0]
            wrong_pred=policy.model(q,t,cond=other,training=False,gen_attn_map=False)[0]
            base=(pred-noise[si]).square().mean((1,2));wrong_error=(wrong_pred-noise[si]).square().mean((1,2))
            scale=(alpha[time]/(1-alpha[time])*(target-target[permutation]).square().mean((1,2))).detach()
            assert bool((scale>0).all())
            rank=scale*F.softplus(.1+(base-wrong_error)/scale)
            auxiliary=[]
            for points in xt:
                point=policy.model(points[si,ti],t,cond=cond,training=False,gen_attn_map=False)[0]
                implied=((points[si,ti]-(alpha[time]**.5)*target)/((1-alpha[time])**.5)).detach()
                auxiliary.append((point-implied).square().mean((1,2)))
            losses=[(base+.25*rank).mean(),.1*auxiliary[0].mean(),.1*auxiliary[1].mean()]
            grads=[]
            for i,loss in enumerate(losses):
                g=torch.autograd.grad(loss,parameters,retain_graph=i<2,allow_unused=True)
                grads.append([torch.zeros_like(p) if v is None else v.detach() for p,v in zip(parameters,g)])
            key=f'{seed}_{time}'
            checks[key+'_all_gradients_finite']=all(bool(torch.isfinite(g).all()) for group in grads for g in group)
            expected_own=((torch.from_numpy(expected[si,ti]).cuda()-implied)**2).mean((1,2))
            checks[key+'_all18_own_batch_point_errors_match']=torch.allclose(auxiliary[1].detach(),expected_own,rtol=1e-5,atol=1e-6)
            row=dict(seed=seed,time=time,groups=compare(names,grads),
                phase_components={str(p):dict(base=float(base[2*i:2*i+2].mean().detach()),rank=float(rank[2*i:2*i+2].mean().detach()),generated=float(auxiliary[0][2*i:2*i+2].mean().detach()),own_generated=float(auxiliary[1][2*i:2*i+2].mean().detach())) for i,p in enumerate(phases)})
            if (seed,time) in reference_rows:
                old=reference_rows[(seed,time)]
                checks[key+'_legacy_phase_components_exact']=all(all(v[k]==old['phase_components'][p][k] for k in ('base','rank','generated')) for p,v in row['phase_components'].items())
                checks[key+'_legacy_full_gradient_group_norms_match']=all(np.isclose(row['groups']['old_vs_candidate'][group]['left_norm']**2,v['total_norm_squared'],rtol=3e-5,atol=1e-10) for group,v in old['groups'].items())
            for gi,values in enumerate(grads):
                for i,g in enumerate(values):accum[gi][i]+=g/16;aggregate[gi][i]+=g/128
            rows.append(row);write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,rows=rows));assert all(checks.values())
        seed_means[str(seed)]=compare(names,accum)
        print(json.dumps(dict(seed=seed,all16rows_complete=True,mean=seed_means[str(seed)]['old_vs_candidate']['full_model'])),flush=True)
    policy.obs_encoder.condition_mode='zero_context'
    preds=[policy.model(q,t,cond=policy.obs_encoder(v,training=False),training=False,gen_attn_map=False)[0] for v in (normalized,wrong)]
    zero=(scale*F.softplus(.1+((preds[0]-noise[-1]).square().mean((1,2))-(preds[1]-noise[-1]).square().mean((1,2)))/scale)).mean()
    g=torch.autograd.grad(zero,parameters,allow_unused=True)
    checks['zero_context_rank_outputs_gradients_exact']=torch.equal(*preds) and all(v is None or bool((v==0).all()) for v in g)
    policy.obs_encoder.condition_mode='demo_geometry'
    checks['full_state_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
    checks['scheduler_unchanged']=torch.equal(alpha,scheduler.alphas_cumprod.cpu())
    checks['rng_unchanged']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    checks['no_accumulated_parameter_gradients']=all(p.grad is None for p in parameters)
    checks['all128rows']=len(rows)==128
    mean=compare(names,aggregate)
    saved=dict(parameter_names=names,component_names=['base_plus_rank','old_teacher_auxiliary','current_own_auxiliary'],mean_gradients=[{n:g.cpu() for n,g in zip(names,values)} for values in aggregate])
    torch.save(saved,OUT/'MEAN_GRADIENTS.pt');readback=torch.load(OUT/'MEAN_GRADIENTS.pt',map_location='cpu')
    checks['all_saved_full_mean_gradients_exact']=readback['parameter_names']==names and all(torch.equal(g.cpu(),readback['mean_gradients'][i][n]) for i,values in enumerate(aggregate) for n,g in zip(names,values))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for group,ax in zip(('full_model','geometry_columns','target_encoder','denoiser'),axes.flat):
        for pair in ('old_vs_candidate','own_vs_candidate','supervised_vs_candidate'):
            ax.plot(times,[np.mean([r['groups'][pair][group]['dot'] for r in rows if r['time']==time]) for time in times],label=pair)
        ax.axhline(0,color='black',linewidth=.7);ax.invert_xaxis();ax.set_yscale('symlog',linthresh=1e-4);ax.grid(alpha=.25);ax.set_title(group)
    axes[0,0].legend(fontsize=8);fig.suptitle('Candidate own-replay direction: uniform8TRAIN seeds, raw FP32 derivatives, no optimizer')
    fig.tight_layout();fig.savefig(OUT/'GRADIENT_DESCENT.png',dpi=140);plt.close(fig)
    rule={f'{label}_{pair}_descent':v[pair]['full_model']['dot']>0 for label,v in [('overall',mean),*seed_means.items()] for pair in ('old_vs_candidate','own_vs_candidate')}
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,rows=rows,aggregate=mean,seed_means=seed_means,plot_inspected=False,new_optimizer_updates=0,new_sample_paths=0,new_physics_steps=0))
    write(OUT/'DIRECTION_DECISION.json',dict(checks_passed=all(checks.values()),criteria=rule,raw_gradient_support=all(checks.values()) and all(rule.values()),scope='Fixed .1 own-replay replacement, full mean and all8seed means. No actual Adam/BF16 or performance claim. Complete plot inspection and independent saved-gradient readback before preparing any separate matched stage.'))
    assert all(checks.values())


if __name__=='__main__':
    main()
