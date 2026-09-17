"""Diagnostic gradients through the unmodified complete official DDPM sampler."""
import argparse
import json
import os
import socket

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.probe_generator_reverse_process import (
    BASE, PAIRED_OUT, SELF_SOURCE, PARENT, ActualBranchGeometryDataset, GeneratorWrapper,
    restore_geometry_state, write,
)
from scripts.sugar.demo_following.demo_future.audit_generator_generated_state_gradient import pair_stats

EVIDENCE=PAIRED_OUT/'fixed_path_transfer'
OUT=PAIRED_OUT/'full_sampler_gradient'
FRESH=False


def readback():
    torch.set_num_threads(8)
    report=json.loads((OUT/'RESULT.json').read_text());assert report['checks_passed'] and report['plot_inspected']
    saved=torch.load(OUT/'MEAN_GRADIENTS.pt',map_location='cpu');checks={}
    for key,grads in saved['means'].items():
        measured=pair_stats(saved['parameter_names'],*grads)
        expected=report['mean_statistics'][key]
        for group,values in measured.items():
            for metric,value in values.items():
                ref=expected[group][metric]
                checks[f'{key}_{group}_{metric}']=value==ref if value is None else bool(np.isclose(value,ref,rtol=1e-10,atol=1e-12))
    write(OUT/'SAVED_MEAN_READBACK.json',dict(checks_passed=all(checks.values()),checks=checks,scope='Independent CPU fullnamed gradient-mean readback, fullsampler(left) versus last-step-local(right). No model forward or new gradients.'))
    assert all(checks.values());print(json.dumps(dict(checks=len(checks))),flush=True)


def run():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    for filename in ('RESULT.json','SAVED_MEAN_READBACK.json' if FRESH else 'SAVED_READBACK.json'):
        prior=json.loads((EVIDENCE/filename).read_text());assert prior['checks_passed']
    assert json.loads((EVIDENCE/'RESULT.json').read_text())['plot_inspected']
    protocol=json.loads((EVIDENCE/('NEXT_FRESH_SAMPLER_PROTOCOL.json' if FRESH else 'NEXT_FULL_SAMPLER_GRADIENT_PROTOCOL.json')).read_text())
    plan=json.loads((SELF_SOURCE/'PROTOCOL.json').read_text())
    assert protocol['train_phases']==plan['data']['train']['phases'] and not set(protocol['train_phases']).intersection((218,258))
    OUT.mkdir(exist_ok=False);write(OUT/'PROTOCOL.json',protocol)
    with (SELF_SOURCE/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
        state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
    policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
    restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
    policy.cuda().eval();policy.normalizer.requires_grad_(False);policy.num_inference_steps=16
    names,params=zip(*[(n,p) for n,p in policy.named_parameters() if p.requires_grad])
    checks=dict(full8319216=sum(p.numel() for p in policy.parameters())==8319216)
    objective_direction=None
    if FRESH:
        reference=torch.load(EVIDENCE/'MEAN_GRADIENTS.pt',map_location='cpu')
        checks['fixed_objective_parameter_names_exact']=names==reference['parameter_names']
        objective_direction=[(a+b+c).cuda() for a,b,c in zip(*reference['means']['all_noise'])]
        with np.load(BASE/'generator_train_diffusion_replay8_current_gaps/TRAIN_GENERATED_STATE_INPUTS.npz') as a:training_seeds=a['seeds'].tolist()
        checks['fresh_seeds_disjoint']=len(set(protocol['seeds']))==8 and not set(protocol['seeds']).intersection(training_seeds+plan['frozen_evaluation']['sample_seeds'])
    scheduler=policy.noise_scheduler;original_step=scheduler.step;config=dict(scheduler.config);alpha=scheduler.alphas_cumprod.cpu().clone()
    total=[[torch.zeros_like(p) for p in params] for _ in range(2)];means={};mean_statistics={};rows=[];path_calls=0
    seed_accum={seed:[[torch.zeros_like(p) for p in params] for _ in range(2)] for seed in protocol['seeds']} if FRESH else {}
    cases_per_phase=2*len(protocol['seeds']);total_cases=cases_per_phase*len(protocol['train_phases']);fresh_calls=0;primary_calls=0
    for phase in protocol['train_phases']:
        dataset=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state'])
        samples=[dataset[i] for i in (0,1)]
        obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
        target=policy.normalizer['action'].normalize(torch.stack([s['action'] for s in samples]).cuda())
        if FRESH:
            with np.load(SELF_SOURCE/f'frozen_evaluation/phase_{phase}/trained_demo_geometry_correct.npz') as a:primary=a['predictions'][:2].copy()
            with torch.no_grad():
                for si,seed in enumerate(plan['frozen_evaluation']['sample_seeds'][:2]):
                    for branch in (0,1):
                        torch.manual_seed(seed);action=policy.predict_action({k:v[branch:branch+1] for k,v in obs.items()});primary_calls+=1
                        checks[f'{phase}_{seed}_{branch}_primary_exact']=np.array_equal(action.cpu().numpy()[0],primary[si,branch])
                shape=(len(protocol['seeds']),1,2,16,8,36)
                recorded={key:np.empty(shape,dtype=np.float32) for key in ('xt','epsilon','pred_original','prev_sample')}
                expected_action=np.empty((len(protocol['seeds']),2,8,36),dtype=np.float32)
                for si,seed in enumerate(protocol['seeds']):
                    for branch in (0,1):
                        fresh_records=[]
                        def record_fresh(epsilon,time,xt,*args,**kwargs):
                            result=original_step(epsilon,time,xt,*args,**kwargs)
                            fresh_records.append((int(time),xt.cpu().clone(),epsilon.cpu().clone(),result.pred_original_sample.cpu().clone(),result.prev_sample.cpu().clone()))
                            return result
                        scheduler.step=record_fresh
                        try:
                            torch.manual_seed(seed);action=policy.predict_action({k:v[branch:branch+1] for k,v in obs.items()})
                        finally:scheduler.step=original_step
                        assert [r[0] for r in fresh_records]==list(range(45,-1,-3))
                        for i,key in enumerate(recorded):recorded[key][si,0,branch]=torch.cat([r[i+1] for r in fresh_records]).numpy()
                        expected_action[si,branch]=action.cpu().numpy()[0];fresh_calls+=1
            recorded.update(times=np.array(list(range(45,-1,-3))),seeds=np.array(protocol['seeds']),normalized_actual_target=target.cpu().numpy().astype(np.float64))
            np.savez_compressed(OUT/f'phase_{phase}_fresh_paths.npz',**recorded,predictions=expected_action)
            with np.load(OUT/f'phase_{phase}_fresh_paths.npz') as a:
                checks[f'{phase}_fresh_saved_arrays_exact']=all(np.array_equal(a[k],v) for k,v in dict(recorded,predictions=expected_action).items())
        else:
            with np.load(PAIRED_OUT/'self18'/f'phase_{phase}_steps_16.npz') as a:recorded={k:a[k].copy() for k in a.files}
            with np.load(SELF_SOURCE/f'frozen_evaluation/phase_{phase}/trained_demo_geometry_correct.npz') as a:expected_action=a['predictions'][:4].copy()
        checks[f'{phase}_target_exact']=np.array_equal(target.cpu().numpy().astype(np.float64),recorded['normalized_actual_target'])
        checks[f'{phase}_seeds_exact']=recorded['seeds'].tolist()==protocol['seeds']
        phase_mean=[[torch.zeros_like(p) for p in params] for _ in range(2)]
        for si,seed in enumerate(protocol['seeds']):
            for branch in (0,1):
                key=f'{phase}_{seed}_{branch}';records=[]
                def record_step(epsilon,time,xt,*args,**kwargs):
                    result=original_step(epsilon,time,xt,*args,**kwargs)
                    records.append((int(time),xt.detach().cpu().clone(),epsilon.detach().cpu().clone(),result.pred_original_sample.detach().cpu().clone(),result.prev_sample.detach().cpu().clone(),result.prev_sample))
                    return result
                scheduler.step=record_step
                try:
                    torch.manual_seed(seed)
                    action=policy.predict_action({k:v[branch:branch+1] for k,v in obs.items()})
                finally:scheduler.step=original_step
                checks[key+'_all16times_exact']=[r[0] for r in records]==recorded['times'].tolist()
                for i,name in enumerate(('xt','epsilon','pred_original','prev_sample')):
                    checks[key+'_'+name+'_exact']=np.array_equal(torch.cat([r[i+1] for r in records]).numpy(),recorded[name][si,0,branch])
                checks[key+'_physical_output_exact']=np.array_equal(action.detach().cpu().numpy()[0],expected_action[si,branch])
                assert all(checks.values()),[k for k,v in checks.items() if not v]
                final=records[-1][5];full_loss=(final-target[branch:branch+1]).square().mean()
                cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                full_grad=torch.autograd.grad(full_loss,params,allow_unused=True)
                full_grad=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,full_grad)]
                checks[key+'_full_backward_rng_exact']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                local_xt=torch.from_numpy(recorded['xt'][si,0,branch,-1:]).cuda()
                normalized=policy.normalizer.normalize({k:v[branch:branch+1] for k,v in obs.items()})
                cond=policy.obs_encoder(normalized,training=False)
                local_epsilon=policy.model(local_xt,torch.tensor(0,device='cuda'),cond=cond,training=False,gen_attn_map=False)[0]
                local_clean=((local_xt-(1-alpha[0])**.5*local_epsilon)/alpha[0]**.5).clamp(-1,1)
                local_loss=(local_clean-target[branch:branch+1]).square().mean()
                checks[key+'_local_epsilon_exact']=np.array_equal(local_epsilon.detach().cpu().numpy()[0],recorded['epsilon'][si,0,branch,-1])
                checks[key+'_local_loss_exact']=torch.equal(full_loss.detach(),local_loss.detach())
                local_grad=torch.autograd.grad(local_loss,params,allow_unused=True)
                local_grad=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,local_grad)]
                checks[key+'_local_forward_backward_rng_exact']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                checks[key+'_all_gradients_finite']=all(bool(torch.isfinite(g).all()) for group in (full_grad,local_grad) for g in group)
                stats=pair_stats(names,full_grad,local_grad)
                checks[key+'_nonzero_full_and_local']=stats['full_model']['supervised_norm']>0 and stats['full_model']['generated_state_norm']>0
                for group,grad in enumerate((full_grad,local_grad)):
                    for a,t,g in zip(phase_mean[group],total[group],grad):a.add_(g/cases_per_phase);t.add_(g/total_cases)
                    if FRESH:
                        for a,g in zip(seed_accum[seed][group],grad):a.add_(g/(2*len(protocol['train_phases'])))
                row=dict(phase=phase,seed=seed,branch=branch,loss=float(full_loss.detach()),statistics=stats)
                if FRESH:row['fixed_objective_vs_full_sampler']=pair_stats(names,objective_direction,full_grad)
                rows.append(row);path_calls+=1
                write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,rows=rows,exact_replayed_paths=path_calls))
                assert all(checks.values()),[k for k,v in checks.items() if not v]
                del action,records,final,full_loss,full_grad,local_loss,local_grad,local_clean,local_epsilon,cond
            print(json.dumps(dict(phase=phase,seed=seed,exact_paths=path_calls)),flush=True)
        mean_statistics[str(phase)]=pair_stats(names,*phase_mean)
        means[str(phase)]=[[v.cpu() for v in group] for group in phase_mean]
    mean_statistics['all_train']=pair_stats(names,*total);means['all_train']=[[v.cpu() for v in group] for group in total]
    for seed,gradients in seed_accum.items():
        mean_statistics['seed_'+str(seed)]=pair_stats(names,*gradients);means['seed_'+str(seed)]=[[v.cpu() for v in group] for group in gradients]
    checks['fullstate_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
    checks['no_accumulated_parameter_grads']=all(p.grad is None for p in policy.parameters())
    checks['scheduler_unchanged']=config==dict(scheduler.config) and torch.equal(alpha,scheduler.alphas_cumprod.cpu()) and scheduler.step==original_step
    checks['exact_path_budget']=path_calls==protocol['exact_replayed_paths']
    if FRESH:
        checks['fresh_path_budget']=fresh_calls==protocol['new_independent_paths']
        checks['primary_control_budget']=primary_calls==protocol['primary_control_paths']
    torch.save(dict(parameter_names=names,means=means),OUT/'MEAN_GRADIENTS.pt')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(16,12))
    for phase,ax in zip(protocol['train_phases'],axes.flat):
        selected=[r for r in rows if r['phase']==phase]
        for group,color in [('full_model','C0'),('target_encoder','C1'),('denoiser','C2')]:
            ax.plot(range(cases_per_phase),[r['statistics'][group]['cosine'] for r in selected],marker='o',color=color,label=group)
        ax.axhline(0,color='gray');ax.set_ylim(-1.05,1.05);ax.set_title(f'TRAIN{phase}');ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=7);fig.suptitle('Full16step vs detached last-step gradient; '+('fresh8TRAINseeds' if FRESH else 'fixed4primaryseeds')+', no optimization')
    fig.tight_layout();fig.savefig(OUT/'SAMPLER_GRADIENT.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,rows=rows,mean_statistics=mean_statistics,plot_inspected=False,new_optimizer_updates=0,new_physics_steps=0,exact_replayed_paths=path_calls,new_independent_paths=fresh_calls,primary_control_paths=primary_calls,official_forward_calls=16*(path_calls+fresh_calls+primary_calls),extra_local_forward_calls=path_calls,
        scope='FP32 eval gradients only: left full official16step sampler final MSE, right same final MSE at detached last state. Full architecture/targets/weights unchanged. This is neither existing q+rank+aux gradient nor Adam direction or evidence of training benefit. Only realTRAIN phases; '+('fresh8seeds with exact no-grad/grad paths and originalprimary controls, not replacement primary32 evaluation.' if FRESH else 'four existing primary seeds are diagnostic, not new training replicates.')))
    assert all(checks.values())


def main():
    global EVIDENCE,OUT,FRESH
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--readback',action='store_true');parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args()
    if args.fresh:
        FRESH=True;EVIDENCE=PAIRED_OUT/'objective_vs_sampler_gradient';OUT=PAIRED_OUT/'fresh_full_sampler_gradient'
    readback() if args.readback else run()


if __name__=='__main__':main()
