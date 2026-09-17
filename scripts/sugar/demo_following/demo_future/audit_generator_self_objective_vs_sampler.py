"""Frozen full TRAIN objective gradients versus saved official sampler gradients."""
import argparse
import json
import os
import socket

import dill
import numpy as np
import torch
import torch.nn.functional as F

from scripts.sugar.demo_following.demo_future.probe_generator_reverse_process import (
    BASE, SELF_SOURCE, PAIRED_OUT, PARENT, ActualBranchGeometryDataset,
    GeneratorWrapper, CONTEXT_KEY, restore_geometry_state, write,
)
from scripts.sugar.demo_following.demo_future.audit_generator_generated_state_gradient import pair_stats

REFERENCE=PAIRED_OUT/'full_sampler_gradient'
OUT=PAIRED_OUT/'objective_vs_sampler_gradient'


def compare(names,components,reference):
    gradients=list(components)+[[a+b+c for a,b,c in zip(*components)]]
    result={}
    for phase,pair in reference.items():
        result[phase]={}
        for label,gradient in zip(('base','rank','replay','total'),gradients):
            values=pair_stats(names,gradient,pair[0])
            result[phase][label]={group:dict(objective_gradient_norm=v['supervised_norm'],sampler_gradient_norm=v['generated_state_norm'],descent_dot=v['dot'],cosine=v['cosine']) for group,v in values.items()}
    return result


def readback():
    torch.set_num_threads(8)
    report=json.loads((OUT/'RESULT.json').read_text());assert report['checks_passed'] and report['plot_inspected']
    saved=torch.load(OUT/'MEAN_GRADIENTS.pt',map_location='cpu')
    reference=torch.load(REFERENCE/'MEAN_GRADIENTS.pt',map_location='cpu')
    assert saved['parameter_names']==reference['parameter_names']
    checks={}
    for mean,components in saved['means'].items():
        measured=compare(saved['parameter_names'],components,reference['means'])
        for phase,items in measured.items():
            for component,groups in items.items():
                for group,values in groups.items():
                    for key,value in values.items():
                        expected=report['mean_comparisons'][mean][phase][component][group][key]
                        checks[f'{mean}_{phase}_{component}_{group}_{key}']=value==expected if value is None else bool(np.isclose(value,expected,rtol=1e-10,atol=1e-12))
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    write(OUT/'SAVED_MEAN_READBACK.json',dict(checks_passed=True,checks=checks,scope='Independent CPU full named gradient means and all9phase/overall dot products, all4seedmeans/overall. No model forwards or gradients.'))
    print(json.dumps(dict(checks=len(checks))),flush=True)


def run():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    for filename in ('RESULT.json','SAVED_MEAN_READBACK.json'):
        prior=json.loads((REFERENCE/filename).read_text());assert prior['checks_passed']
    assert json.loads((REFERENCE/'RESULT.json').read_text())['plot_inspected']
    protocol=json.loads((REFERENCE/'NEXT_OBJECTIVE_COMPARISON_PROTOCOL.json').read_text())
    plan=json.loads((SELF_SOURCE/'PROTOCOL.json').read_text());phases=plan['data']['train']['phases']
    assert phases==protocol['train_phases'] and not set(phases).intersection((218,258))
    replay=BASE/'generator_train_diffusion_replay8_current_gaps'
    with np.load(replay/'TRAIN_GENERATED_STATE_INPUTS.npz') as a:saved={k:a[k].copy() for k in a.files}
    with np.load(protocol['point_reference']) as a:point_reference={k:a[k].copy() for k in a.files}
    seeds=saved['seeds'][:4].tolist();times=saved['times'].tolist()
    assert seeds==protocol['q_and_replay_seeds'] and times==protocol['diffusion_times']
    OUT.mkdir(exist_ok=False);write(OUT/'PROTOCOL.json',protocol)
    reference=torch.load(REFERENCE/'MEAN_GRADIENTS.pt',map_location='cpu')
    reference_means={key:[[v.cuda() for v in group] for group in pair] for key,pair in reference['means'].items()}
    samples=[]
    for phase in phases:
        ds=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state'])
        samples.extend(ds[i] for i in (0,1))
    obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
    actions=torch.stack([s['action'] for s in samples]).cuda()
    with (SELF_SOURCE/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
        state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
    policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
    restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
    policy.cuda().eval();policy.normalizer.requires_grad_(False)
    names,params=zip(*[(n,p) for n,p in policy.named_parameters() if p.requires_grad])
    normalized=policy.normalizer.normalize(obs);target=policy.normalizer['action'].normalize(actions)
    permutation=torch.arange(18,device='cuda')^1;wrong=dict(normalized);wrong[CONTEXT_KEY]=normalized[CONTEXT_KEY][permutation]
    xt=torch.from_numpy(saved['xt'][:4]).cuda();noise=torch.from_numpy(saved['initial_gaussian'][:4]).cuda();oracle=torch.from_numpy(point_reference['oracle'][:4]).cuda()
    scheduler=policy.noise_scheduler;alpha=scheduler.alphas_cumprod.cpu().clone();config=dict(scheduler.config)
    checks=dict(full8319216=sum(p.numel() for p in policy.parameters())==8319216,parameter_names_exact=names==reference['parameter_names'],
        actual_targets_exact=np.array_equal(target.cpu().numpy().astype(np.float64),saved['normalized_actual_target']) and np.array_equal(target.cpu().numpy(),point_reference['target']),
        phase_binding_exact=saved['phases'].tolist()==[p for p in phases for _ in (0,1)],
        point_seed_time_exact=np.array_equal(saved['seeds'],point_reference['seeds']) and np.array_equal(saved['times'],point_reference['times']),
        shared_causal_fields_exact=all(torch.equal(obs[k][::2],obs[k][1::2]) for k in ('obj_pos_b','obj_ori_b','last_action')))
    assert all(checks.values())
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
    aggregate=[[torch.zeros_like(p) for p in params] for _ in range(3)];means={};comparisons={};rows=[]
    for si,seed in enumerate(seeds):
        accum=[[torch.zeros_like(p) for p in params] for _ in range(3)]
        for ti,time in enumerate(times):
            t=torch.full((18,),time,device='cuda',dtype=torch.long);q=scheduler.add_noise(target,noise[si],t)
            cond=policy.obs_encoder(normalized,training=False);other=policy.obs_encoder(wrong,training=False)
            pred=policy.model(q,t,cond=cond,training=False,gen_attn_map=False)[0]
            wrong_pred=policy.model(q,t,cond=other,training=False,gen_attn_map=False)[0]
            point=policy.model(xt[si,ti],t,cond=cond,training=False,gen_attn_map=False)[0]
            base=(pred-noise[si]).square().mean((1,2));wrong_error=(wrong_pred-noise[si]).square().mean((1,2))
            scale=(alpha[time]/(1-alpha[time])*(target-target[permutation]).square().mean((1,2))).detach();assert bool((scale>0).all())
            rank=scale*F.softplus(.1+(base-wrong_error)/scale);replay_loss=(point-oracle[si,ti]).square().mean((1,2))
            total=base+.25*rank+.1*replay_loss
            expected=((point_reference['epsilon'][si,ti,0]-point_reference['oracle'][si,ti])**2).mean((1,2))
            key=f'{seed}_{time}';checks[key+'_all18_replay_loss_match']=bool(np.allclose(replay_loss.detach().cpu().numpy(),expected,rtol=1e-5,atol=1e-6))
            gradients=[]
            for index,loss in enumerate((base.mean(),.25*rank.mean(),.1*replay_loss.mean(),total.mean())):
                grad=torch.autograd.grad(loss,params,retain_graph=index<3,allow_unused=True)
                gradients.append([torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,grad)])
            checks[key+'_component_full_gradient_sum']=all(torch.allclose(a+b+c,d,rtol=3e-5,atol=1e-6) for a,b,c,d in zip(*gradients))
            checks[key+'_finite_gradients']=all(bool(torch.isfinite(g).all()) for group in gradients for g in group)
            for group in range(3):
                for a,b,g in zip(accum[group],aggregate[group],gradients[group]):a.add_(g/16);b.add_(g/64)
            rows.append(dict(seed=seed,time=time,phase_losses={str(phase):dict(base=float(base[2*i:2*i+2].mean().detach()),rank=float(rank[2*i:2*i+2].mean().detach()),replay=float(replay_loss[2*i:2*i+2].mean().detach())) for i,phase in enumerate(phases)}))
            write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,rows=rows));assert all(checks.values())
            del gradients
        comparisons[str(seed)]=compare(names,accum,reference_means);means[str(seed)]=[[v.cpu() for v in group] for group in accum]
        print(json.dumps(dict(seed=seed,all16gradient_rows=True,overall={label:groups['full_model'] for label,groups in comparisons[str(seed)]['all_train'].items()})),flush=True)
    policy.obs_encoder.condition_mode='zero_context';predictions=[]
    for inputs in (normalized,wrong):predictions.append(policy.model(q,t,cond=policy.obs_encoder(inputs,training=False),training=False,gen_attn_map=False)[0])
    zero=(scale*F.softplus(.1+((predictions[0]-noise[si]).square().mean((1,2))-(predictions[1]-noise[si]).square().mean((1,2)))/scale)).mean()
    zero_grad=torch.autograd.grad(zero,params,allow_unused=True)
    checks['zero_context_rank_outputs_and_gradients_exact']=torch.equal(*predictions) and all(g is None or bool((g==0).all()) for g in zero_grad)
    policy.obs_encoder.condition_mode='demo_geometry'
    checks['fullstate_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
    checks['rng_unchanged']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    checks['scheduler_unchanged']=config==dict(scheduler.config) and torch.equal(alpha,scheduler.alphas_cumprod.cpu())
    checks['no_accumulated_gradients']=all(p.grad is None for p in policy.parameters());checks['declared64rows']=len(rows)==64
    comparisons['all_noise']=compare(names,aggregate,reference_means);means['all_noise']=[[v.cpu() for v in group] for group in aggregate]
    torch.save(dict(parameter_names=names,means=means),OUT/'MEAN_GRADIENTS.pt')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(17,12))
    for phase,ax in zip(phases,axes.flat):
        for i,label in enumerate(('base','rank','replay','total')):
            ax.plot(range(5),[comparisons[key][str(phase)][label]['full_model']['cosine'] for key in [str(s) for s in seeds]+['all_noise']],marker='o',label=label)
        ax.axhline(0,color='gray');ax.set_xticks(range(5),['seed0','seed1','seed2','seed3','mean']);ax.set_ylim(-1.05,1.05);ax.set_title(f'TRAIN{phase}');ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8);fig.suptitle('Training objective vs full final-sampler gradient; different seed sets, FP32 diagnostic only')
    fig.tight_layout();fig.savefig(OUT/'OBJECTIVE_VS_SAMPLER.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,rows=rows,mean_comparisons=comparisons,plot_inspected=False,new_optimizer_updates=0,new_sample_paths=0,new_physics_steps=0,scope=protocol['sampling_scope']))
    assert all(checks.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--readback',action='store_true')
    args=parser.parse_args();readback() if args.readback else run()


if __name__=='__main__':main()
