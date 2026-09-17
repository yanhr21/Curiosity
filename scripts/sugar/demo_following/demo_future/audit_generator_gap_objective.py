"""Full frozen TRAIN gradient interaction of preserved14 versus added4 cases."""
import json
import argparse
import os
import socket

import dill
import numpy as np
import torch
import torch.nn.functional as F

from scripts.sugar.demo_following.demo_future.probe_generator_training_replay import (
    BASE, RUN, OUT as REPLAY_PROBE, PARENT, ActualBranchGeometryDataset,
    GeneratorWrapper, CONTEXT_KEY, restore_geometry_state, write,
)
from scripts.sugar.demo_following.demo_future.audit_generator_generated_state_gradient import pair_stats

OUT=RUN/'frozen_evaluation/gap_objective_gradient'


def stats(names,left,right):
    values=pair_stats(names,left,right)
    for value in values.values():
        value['old14_norm']=value.pop('supervised_norm')
        value['new4_norm']=value.pop('generated_state_norm')
        value['full_direction_old14_descent_dot']=(14*value['old14_norm']**2+4*value['dot'])/18
        value['full_direction_new4_descent_dot']=(14*value['dot']+4*value['new4_norm']**2)/18
    return values


def component_stats(names,grads):
    labels=('base','rank','generated');result={}
    for i,j in ((0,1),(0,2),(1,2)):
        for group,value in pair_stats(names,grads[i],grads[j]).items():
            item=result.setdefault(group,dict(norms={},dots={},cosines={}))
            item['norms'][labels[i]]=value['supervised_norm']
            item['norms'][labels[j]]=value['generated_state_norm']
            key=labels[i]+'_vs_'+labels[j]
            item['dots'][key]=value['dot'];item['cosines'][key]=value['cosine']
    for item in result.values():
        item['total_direction_descent_dot']={label:item['norms'][label]**2+sum(value for key,value in item['dots'].items() if label in key.split('_vs_')) for label in labels}
        item['total_norm_squared']=sum(v*v for v in item['norms'].values())+2*sum(item['dots'].values())
    return result


def stored_adam_direction(policy,payload):
    """Read moments through the official parameter grouping, without a step."""
    from omegaconf import OmegaConf
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    optimizer=policy.get_optimizer(**plan['optimizer'])
    state=payload['state_dicts']['optimizer']
    optimizer.load_state_dict(state)
    named={id(p):n for n,p in policy.named_parameters()}
    vectors={};bindings=[]
    for gi,(group,saved_group) in enumerate(zip(optimizer.param_groups,state['param_groups'])):
        assert len(group['params'])==len(saved_group['params'])
        for p,pid in zip(group['params'],saved_group['params']):
            n=named[id(p)];record=optimizer.state[p]
            assert pid in state['state'] and int(record['step'])==512
            assert all(torch.equal(v.cpu(),state['state'][pid][key].cpu()) for key,v in record.items() if torch.is_tensor(v))
            beta1,beta2=group['betas'];step=int(record['step'])
            m=record['exp_avg'].double()/(1-beta1**step)
            v=record['exp_avg_sq'].double()/(1-beta2**step)
            vectors[n]=m/(v.sqrt()+group['eps'])+group['weight_decay']*p.detach().double()
            bindings.append(dict(name=n,optimizer_id=pid,group=gi,shape=list(p.shape),step=step))
    return vectors,dict(bindings=bindings,param_groups=[{k:OmegaConf.to_container(v,resolve=True) if OmegaConf.is_config(v) else v for k,v in g.items() if k!='params'} for g in state['param_groups']],
        scope='Stored512 momentum bias correction and variance preconditioner evaluated with endpoint parameters for decay, FP64 per-unit-lr direction. No optimizer.step, new gradient insertion, update513, actual step512 reconstruction or training continuation.')


def adam_stats(names,grads,direction):
    result={}
    for label,grad in zip(('base','rank','generated'),grads):
        values=pair_stats(names,grad,[direction[n] for n in names])
        for group,v in values.items():
            result.setdefault(group,{})[label]=dict(gradient_norm=v['supervised_norm'],stored_direction_norm=v['generated_state_norm'],
                descent_dot=v['dot'],cosine=v['cosine'])
    return result


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--components',action='store_true');parser.add_argument('--adam-state',action='store_true')
    args=parser.parse_args();components=args.components or args.adam_state
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    prior=json.loads((REPLAY_PROBE/'RESULT.json').read_text())
    assert prior['checks_passed'] and prior['plot_inspected']
    decision=json.loads((REPLAY_PROBE/'DECISION.json').read_text())
    assert decision['next_action']=='full_train_old14_new4_objective_gradient_audit'
    plan=json.loads((RUN/'PROTOCOL.json').read_text());phases=plan['data']['train']['phases']
    replay=BASE/'generator_train_diffusion_replay8_gaps2'
    with np.load(replay/'TRAIN_GENERATED_STATE_INPUTS.npz') as a:saved={k:a[k].copy() for k in a.files}
    with np.load(REPLAY_PROBE/'new18.npz') as a:reference=a['epsilon'].copy();oracle=a['oracle'].copy()
    seeds=saved['seeds'][:4].tolist();times=saved['times'].tolist()
    old_indices=[i for i,p in enumerate(saved['phases']) if p not in (221,261)]
    new_indices=[i for i,p in enumerate(saved['phases']) if p in (221,261)]
    assert len(old_indices)==14 and len(new_indices)==4 and len(times)==16
    reference_rows=None
    if components:
        reference_result=json.loads((OUT/'RESULT.json').read_text())
        assert reference_result['checks_passed'] and reference_result['plot_inspected']
        component_protocol=json.loads((OUT/'NEXT_COMPONENT_GRADIENT_PROTOCOL.json').read_text())
        assert component_protocol['seeds']==seeds and component_protocol['diffusion_times']==times and component_protocol['actual_train_phases']==phases
        reference_rows={(row['seed'],row['time']):row for row in reference_result['rows']}
        OUT=RUN/'frozen_evaluation/gap_component_gradient'
        if args.adam_state:
            old_component=json.loads((OUT/'RESULT.json').read_text())
            assert old_component['checks_passed'] and old_component['plot_inspected']
            adam_protocol=json.loads((OUT/'NEXT_ADAM_DIRECTION_PROTOCOL.json').read_text())
            component_rows={(row['seed'],row['time']):row for row in old_component['rows']}
            OUT=RUN/'frozen_evaluation/gap_component_adam_direction'
    OUT.mkdir(exist_ok=False)
    protocol=dict(source_run=str(RUN),train_phases=phases,seeds=seeds,times=times,
        gradient_rows=64,old14_indices=old_indices,new4_indices=new_indices,
        objective='Per-case q epsilon MSE +0.25 scaled softplus(.1+paired difference/scale) +0.1 frozen teacher xt epsilon correction; original actual targets only.',
        scope='Full8319216 learned new18 endpoint, eval FP32 raw full-parameter gradients. First4 fixed TRAIN replay seeds/all16 times; q uses the saved initial Gaussian and same time as auxiliary, diagnostic coupling differs from randomized training. No dropout, optimizer, Adam, BF16, coefficient choice, check label or new trajectory. Every18case replay error must match saved point probe within stated floating batch tolerance; old14/new4 weighted gradients must recover complete objective gradient.',
        automatic_next_action='Inspect all64rows, four seed means, fullstate/RNG/scheduler/zero-controls and plots. If the full mean direction conflicts with old14, prepare a bounded gradient-preservation diagnostic; if only subgroup conflict but both descend, inspect gradient magnitudes and original loss weighting before any new budget. No automatic training or weight sweep from raw gradients alone.')
    if components:protocol.update(component_protocol)
    if args.adam_state:protocol.update(adam_protocol)
    write(OUT/'PROTOCOL.json',protocol)
    samples=[]
    for phase in phases:
        ds=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state'])
        samples.extend(ds[i] for i in (0,1))
    obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
    actions=torch.stack([s['action'] for s in samples]).cuda()
    with (RUN/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as stream:
        payload=torch.load(stream,pickle_module=dill,map_location='cpu');state=payload['state_dicts']['model']
    policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
    restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
    policy.cuda().eval();policy.normalizer.requires_grad_(False)
    named=[(n,p) for n,p in policy.named_parameters() if p.requires_grad];names,parameters=zip(*named)
    if args.adam_state:
        adam_direction,adam_metadata=stored_adam_direction(policy,payload)
        assert set(adam_direction)==set(names)
        write(OUT/'ADAM_PARAMETER_BINDING.json',adam_metadata)
    normalized=policy.normalizer.normalize(obs);target=policy.normalizer['action'].normalize(actions)
    wrong=dict(normalized);wrong[CONTEXT_KEY]=normalized[CONTEXT_KEY][torch.arange(18,device='cuda')^1]
    xt=torch.from_numpy(saved['xt'][:4]).cuda();noise=torch.from_numpy(saved['initial_gaussian'][:4]).cuda()
    implied=torch.from_numpy(oracle[:4]).cuda();scheduler=policy.noise_scheduler;alpha=scheduler.alphas_cumprod.detach().cpu().clone()
    checks=dict(full_parameter_count=sum(p.numel() for p in policy.parameters())==8319216,
        actual_targets_exact=np.array_equal(target.detach().cpu().numpy().astype(np.float64),saved['normalized_actual_target']),
        same_actual_causal_fields=all(torch.equal(obs[k][::2],obs[k][1::2]) for k in ('obj_pos_b','obj_ori_b','last_action')),
        no_check_labels=not set(phases).intersection([218,258]))
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
    group_count=3 if components else 2
    aggregate=[[torch.zeros_like(p) for p in parameters] for _ in range(group_count)]
    rows=[];seed_means={}
    for si,seed in enumerate(seeds):
        accum=[[torch.zeros_like(p) for p in parameters] for _ in range(group_count)]
        for ti,time in enumerate(times):
            t=torch.full((18,),time,device='cuda',dtype=torch.long)
            q=scheduler.add_noise(target,noise[si],t)
            cond=policy.obs_encoder(normalized,training=False);other=policy.obs_encoder(wrong,training=False)
            pred=policy.model(q,t,cond=cond,training=False,gen_attn_map=False)[0]
            wrong_pred=policy.model(q,t,cond=other,training=False,gen_attn_map=False)[0]
            point=policy.model(xt[si,ti],t,cond=cond,training=False,gen_attn_map=False)[0]
            base=(pred-noise[si]).square().mean((1,2));wrong_error=(wrong_pred-noise[si]).square().mean((1,2))
            paired=target[torch.arange(18,device='cuda')^1]
            scale=(alpha[time]/(1-alpha[time])*(target-paired).square().mean((1,2))).detach()
            assert bool((scale>0).all())
            rank=scale*F.softplus(.1+(base-wrong_error)/scale)
            generated=(point-implied[si,ti]).square().mean((1,2))
            expected=((reference[si,ti,0]-oracle[si,ti])**2).mean((1,2))
            key=f'{seed}_{time}'
            checks[key+'_all18_replay_errors_match']=bool(np.allclose(generated.detach().cpu().numpy(),expected,rtol=1e-5,atol=1e-6))
            total=base+.25*rank+.1*generated
            losses=[base.mean(),.25*rank.mean(),.1*generated.mean(),total.mean()] if components else [total[old_indices].mean(),total[new_indices].mean(),total.mean()]
            grads=[]
            for index,loss in enumerate(losses):
                grad=torch.autograd.grad(loss,parameters,retain_graph=index<len(losses)-1,allow_unused=True)
                grads.append([torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,grad)])
            if components:
                checks[key+'_three_components_sum_to_full_gradient']=all(torch.allclose(a+b+c,total_grad,rtol=3e-5,atol=1e-6) for a,b,c,total_grad in zip(*grads))
            else:
                checks[key+'_weighted_full_gradient_match']=all(torch.allclose((14*a+4*b)/18,c,rtol=3e-5,atol=1e-6) for a,b,c in zip(*grads))
            checks[key+'_all_full_gradients_finite']=all(bool(torch.isfinite(g).all()) for group in grads for g in group)
            for group in range(group_count):
                for i,g in enumerate(grads[group]):accum[group][i]+=g/16;aggregate[group][i]+=g/64
            row=dict(seed=seed,time=time,old14_loss=float(total[old_indices].mean().detach()),new4_loss=float(total[new_indices].mean().detach()),
                groups=component_stats(names,grads[:3]) if components else stats(names,grads[0],grads[1]),phase_components={str(p):dict(base=float(base[2*i:2*i+2].mean().detach()),rank=float(rank[2*i:2*i+2].mean().detach()),generated=float(generated[2*i:2*i+2].mean().detach())) for i,p in enumerate(phases)})
            if components:
                old=reference_rows[(seed,time)]
                checks[key+'_all_old_phase_loss_scalars_exact']=row['phase_components']==old['phase_components'] and row['old14_loss']==old['old14_loss'] and row['new4_loss']==old['new4_loss']
                checks[key+'_all_old_full_gradient_group_norms_match']=all(np.isclose(row['groups'][group]['total_norm_squared'],(14/18)**2*v['old14_norm']**2+(4/18)**2*v['new4_norm']**2+2*14*4/18**2*v['dot'],rtol=3e-5,atol=1e-10) for group,v in old['groups'].items())
            if args.adam_state:
                checks[key+'_all_component_reference_statistics_exact']=row==component_rows[(seed,time)]
                row['stored_adam_direction']=adam_stats(names,grads[:3],adam_direction)
            rows.append(row)
            write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,rows=rows));assert all(checks.values())
            del grads
        seed_means[str(seed)]=component_stats(names,accum) if components else stats(names,*accum)
        if args.adam_state:seed_means[str(seed)]['stored_adam_direction']=adam_stats(names,accum,adam_direction)
        print(json.dumps(dict(seed=seed,all16gradient_rows_complete=True)),flush=True)
    policy.obs_encoder.condition_mode='zero_context'
    pred=[]
    for inputs in (normalized,wrong):
        pred.append(policy.model(q,t,cond=policy.obs_encoder(inputs,training=False),training=False,gen_attn_map=False)[0])
    zero=(scale*F.softplus(.1+((pred[0]-noise[si]).square().mean((1,2))-(pred[1]-noise[si]).square().mean((1,2)))/scale)).mean() if components else (pred[0]-pred[1]).square().mean()
    grad=torch.autograd.grad(zero,parameters,allow_unused=True)
    checks['zero_context_outputs_and_gradients_exact']=torch.equal(*pred) and all(g is None or bool((g==0).all()) for g in grad)
    policy.obs_encoder.condition_mode='demo_geometry'
    checks['full_state_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
    checks['scheduler_values_unchanged']=torch.equal(alpha,scheduler.alphas_cumprod.cpu())
    checks['rng_unchanged']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    checks['no_accumulated_gradients']=all(p.grad is None for p in policy.parameters())
    checks['all64rows']=len(rows)==64
    mean=component_stats(names,aggregate) if components else stats(names,*aggregate)
    if args.adam_state:
        mean['stored_adam_direction']=adam_stats(names,aggregate,adam_direction)
        torch.save(dict(parameter_names=names,component_names=['base','rank','generated'],mean_gradients=[{name:grad.cpu() for name,grad in zip(names,values)} for values in aggregate]),OUT/'MEAN_COMPONENT_GRADIENTS.pt')
        saved_gradients=torch.load(OUT/'MEAN_COMPONENT_GRADIENTS.pt',map_location='cpu')
        checks['full_mean_component_gradients_saved_exact']=saved_gradients['parameter_names']==names and all(torch.equal(g.cpu(),saved_gradients['mean_gradients'][i][name]) for i,values in enumerate(aggregate) for name,g in zip(names,values))
        with (RUN/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as stream:
            after=torch.load(stream,pickle_module=dill,map_location='cpu')
        a,b=payload['state_dicts']['optimizer'],after['state_dicts']['optimizer']
        checks['saved_full_adam_unchanged']=a['param_groups']==b['param_groups'] and a['state'].keys()==b['state'].keys() and all(torch.equal(v,b['state'][pid][k]) if torch.is_tensor(v) else v==b['state'][pid][k] for pid,record in a['state'].items() for k,v in record.items())
        checks['saved_full_checkpoint_model_unchanged']=state.keys()==after['state_dicts']['model'].keys() and all(torch.equal(v,after['state_dicts']['model'][k]) for k,v in state.items())
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for group,ax in zip(('full_model','geometry_columns','target_encoder','denoiser'),axes.flat):
        if components:
            for pair in ('base_vs_rank','base_vs_generated','rank_vs_generated'):
                ax.plot(times,[np.mean([r['groups'][group]['cosines'][pair] for r in rows if r['time']==time]) for time in times],label=pair)
        else:
            for seed in seeds:
                subset=[r for r in rows if r['seed']==seed]
                ax.plot(times,[r['groups'][group]['cosine'] for r in subset],label=str(seed))
        ax.axhline(0,color='black',linewidth=.7);ax.invert_xaxis();ax.set_title(group);ax.grid(alpha=.25)
    axes[0,0].legend();fig.suptitle('Three weighted TRAIN loss gradients; mean per-row cosines, no optimizer' if components else 'Old14 versus new4 realTRAIN full-objective raw gradients; no optimizer')
    fig.tight_layout();fig.savefig(OUT/'GRADIENTS.png',dpi=140);plt.close(fig)
    if components:
        fig,axes=plt.subplots(4,2,figsize=(15,15))
        for group,axis_row in zip(('full_model','geometry_columns','target_encoder','denoiser'),axes):
            for label in ('base','rank','generated'):
                for axis,key in zip(axis_row,('norms','total_direction_descent_dot')):
                    axis.plot(times,[np.mean([r['groups'][group][key][label] for r in rows if r['time']==time]) for time in times],label=label)
            for axis,key in zip(axis_row,('gradient norm','total direction descent dot')):
                axis.set_title(group+' '+key);axis.invert_xaxis();axis.set_yscale('symlog',linthresh=1e-5);axis.grid(alpha=.25)
            axis_row[1].axhline(0,color='black',linewidth=.7)
        axes[0,0].legend();fig.suptitle('Fixed weighted components; mean four TRAIN seeds at each diagnostic time')
        fig.tight_layout();fig.savefig(OUT/'COMPONENT_NORMS_AND_DESCENT.png',dpi=140);plt.close(fig)
    if args.adam_state:
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for group,ax in zip(('full_model','geometry_columns','target_encoder','denoiser'),axes.flat):
            for label in ('base','rank','generated'):
                ax.plot(times,[np.mean([r['stored_adam_direction'][group][label]['descent_dot'] for r in rows if r['time']==time]) for time in times],label=label)
            ax.axhline(0,color='black',linewidth=.7);ax.set_yscale('symlog',linthresh=1e-4);ax.invert_xaxis();ax.grid(alpha=.25);ax.set_title(group)
        axes[0,0].legend();fig.suptitle('Stored Adam512 direction versus current loss gradients; no optimizer step')
        fig.tight_layout();fig.savefig(OUT/'STORED_ADAM_DIRECTION.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,rows=rows,
        aggregate=mean,seed_means=seed_means,plot_inspected=False,new_optimizer_updates=0,new_sample_paths=0,new_physics_steps=0))
    assert all(checks.values())


if __name__=='__main__':main()
