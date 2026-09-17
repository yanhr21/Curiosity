"""Full-model TRAIN gradients on saved official generated diffusion states."""
import argparse
import json
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

SOURCE=BASE/'matched_generator_branch_paired_rank025512'
RECORDED=BASE/'matched_generator_branch_rank025_solver50/frozen_evaluation/reverse_process_probe'
EVIDENCE=BASE/'matched_generator_branch_paired_hinge025512/frozen_evaluation/own_reverse_process/path_feedback_readback'
OUT=EVIDENCE/'generated_state_gradient_audit'


def pair_stats(names, left, right):
    result={}
    for group in ('full_model','geometry_columns','target_encoder','denoiser'):
        ll=torch.zeros((),device=left[0].device,dtype=torch.float64)
        rr=ll.clone();dot=ll.clone()
        for name,a,b in zip(names,left,right):
            if group=='geometry_columns':
                if name!='obs_encoder.target_state_net.0.weight':continue
                a,b=a[:,9:],b[:,9:]
            elif group=='target_encoder' and not name.startswith('obs_encoder.target_state_net.'):continue
            elif group=='denoiser' and not name.startswith('model.'):continue
            a,b=a.double(),b.double();ll+=a.square().sum();rr+=b.square().sum();dot+=(a*b).sum()
        ll,rr,dot=map(float,(ll,rr,dot))
        result[group]=dict(supervised_norm=ll**.5,generated_state_norm=rr**.5,dot=dot,
            cosine=dot/(ll*rr)**.5 if ll*rr else None)
    return result


def main():
    global EVIDENCE, RECORDED, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-replay',action='store_true')
    args=parser.parse_args()
    if args.fresh_replay:
        EVIDENCE=BASE/'generator_train_diffusion_replay8'
        RECORDED=EVIDENCE
        OUT=EVIDENCE/'generated_state_gradient_audit'
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    protocol=json.loads((EVIDENCE/'NEXT_GRADIENT_PROTOCOL.json').read_text())
    prior=json.loads((EVIDENCE/'RESULT.json').read_text())
    assert prior['checks_passed'] and prior['plot_inspected'] and all(protocol['checks'].values())
    row_count=len(protocol['inference_seeds'])*len(protocol['diffusion_times'])
    assert protocol['source_run']==str(SOURCE) and protocol['gradient_rows']==row_count
    assert row_count in (64,128) and len(protocol['diffusion_times'])==16
    OUT.mkdir(exist_ok=False);write(OUT/'PROTOCOL.json',protocol)
    with np.load(protocol['input_arrays']) as a:
        saved={k:a[k].copy() for k in a.files}
    plan=json.loads((SOURCE/'PROTOCOL.json').read_text())
    assert protocol['train_phases']==plan['data']['train']['phases']
    dataset=ActualBranchGeometryDataset(**plan['branch_dataset'])
    assert dataset.real_case_count==14
    samples=[dataset[i] for i in range(14)]
    obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
    actions=torch.stack([s['action'] for s in samples]).cuda()
    with (SOURCE/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
        state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
    policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
    restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
    policy.cuda().eval();policy.normalizer.requires_grad_(False)
    checks=dict(full_parameter_count=sum(p.numel() for p in policy.parameters())==8319216,
        no_heldout_phases=set(saved['phases'].tolist())==set(protocol['train_phases']),
        saved_seed_time_exact=saved['seeds'].tolist()==protocol['inference_seeds'] and saved['times'].tolist()==protocol['diffusion_times'])
    normalized=policy.normalizer.normalize(obs)
    target=policy.normalizer['action'].normalize(actions)
    checks['actual_normalized_targets_exact']=np.array_equal(target.detach().cpu().numpy().astype(np.float64),saved['normalized_actual_target'])
    permutation=torch.arange(14,device='cuda')^1
    checks['causal_fields_same_within_pair']=all(torch.equal(obs[k],obs[k][permutation]) for k in ('obj_pos_b','obj_ori_b','last_action'))
    expected=[];expected_clean=[];reference_xt=[]
    for phase in protocol['train_phases']:
        with np.load(RECORDED/f'phase_{phase}_steps_16.npz') as a:
            expected.append(a['epsilon'][:,0].transpose(0,2,1,3,4))
            expected_clean.append(a['pred_original'][:,0].transpose(0,2,1,3,4))
            reference_xt.append(a['xt'][:,0].transpose(0,2,1,3,4))
    expected=np.concatenate(expected,axis=2);expected_clean=np.concatenate(expected_clean,axis=2)
    checks['all_generated_xt_arrays_exact']=np.array_equal(saved['xt'],np.concatenate(reference_xt,axis=2))
    initial=torch.from_numpy(saved['initial_gaussian']).cuda()
    xt=torch.from_numpy(saved['xt']).cuda()
    checks['initial_gaussians_pair_exact']=torch.equal(initial[:,::2],initial[:,1::2])
    assert all(checks.values())
    scheduler=policy.noise_scheduler
    assert scheduler.config.prediction_type=='epsilon' and scheduler.config.num_train_timesteps==50 and scheduler.config.clip_sample
    alpha_before=scheduler.alphas_cumprod.clone().cpu()
    named=[(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    names,parameters=zip(*named)
    accum=[[torch.zeros_like(p) for p in parameters] for _ in range(3)]
    rows=[];seed_summaries={}
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
    for si,seed in enumerate(protocol['inference_seeds']):
        seed_accum=[[torch.zeros_like(p) for p in parameters] for _ in range(3)]
        for ti,time in enumerate(protocol['diffusion_times']):
            timesteps=torch.full((14,),time,device='cuda',dtype=torch.long)
            q=scheduler.add_noise(target,initial[si],timesteps)
            # Frozen inference uses CPU scheduler scalars. Preserve that exact
            # arithmetic even after official add_noise moves its alpha array.
            alpha=alpha_before[time]
            root_alpha=alpha**.5;root_beta=(1-alpha)**.5
            oracle=(xt[si,ti]-root_alpha*target)/root_beta
            base_errors=[];wrong_errors=[];generated_errors=[];clean_errors=[];captured=[];captured_clean=[]
            for case in range(14):
                correct={k:v[case:case+1] for k,v in normalized.items()}
                wrong=dict(correct);wrong[CONTEXT_KEY]=normalized[CONTEXT_KEY][case^1: (case^1)+1]
                cond=policy.obs_encoder(correct,training=False)
                wrong_cond=policy.obs_encoder(wrong,training=False)
                t=torch.tensor(time,device='cuda')
                pred=policy.model(q[case:case+1],t,cond=cond,training=False,gen_attn_map=False)[0]
                other=policy.model(q[case:case+1],t,cond=wrong_cond,training=False,gen_attn_map=False)[0]
                point=policy.model(xt[si,ti,case:case+1],t,cond=cond,training=False,gen_attn_map=False)[0]
                base_errors.append((pred-initial[si,case:case+1]).square().mean())
                wrong_errors.append((other-initial[si,case:case+1]).square().mean())
                generated_errors.append((point-oracle[case:case+1]).square().mean())
                clean=(xt[si,ti,case:case+1]-root_beta*point)/root_alpha
                clean_errors.append((clean-target[case:case+1]).square().mean())
                captured.append(point.detach());captured_clean.append(clean.detach().clamp(-1,1))
            base_errors=torch.stack(base_errors);wrong_errors=torch.stack(wrong_errors)
            generated_errors=torch.stack(generated_errors);clean_errors=torch.stack(clean_errors)
            key=f'{seed}_{time}'
            checks[key+'_all14point_epsilon_exact']=np.array_equal(torch.cat(captured).cpu().numpy(),expected[si,ti])
            checks[key+'_all14point_clean_exact']=np.array_equal(torch.cat(captured_clean).cpu().numpy(),expected_clean[si,ti])
            checks[key+'_epsilon_clean_loss_identity']=bool(torch.allclose(generated_errors.detach(),alpha/(1-alpha)*clean_errors.detach(),rtol=2e-5,atol=1e-6))
            scale=(alpha/(1-alpha)*(target-target[permutation]).square().mean((1,2))).detach()
            rank=scale*F.softplus(.1+(base_errors-wrong_errors)/scale)
            losses=(base_errors.mean(),(base_errors+.25*rank).mean(),generated_errors.mean())
            grads=[]
            for i,loss in enumerate(losses):
                g=torch.autograd.grad(loss,parameters,retain_graph=i<2,allow_unused=True)
                grads.append([torch.zeros_like(p) if v is None else v.detach() for p,v in zip(parameters,g)])
            checks[key+'_losses_gradients_finite']=all(bool(torch.isfinite(v)) for v in losses) and all(bool(torch.isfinite(g).all()) for group in grads for g in group)
            if not all(checks.values()):
                write(OUT/'FAILED_CHECKS.json',checks)
                raise RuntimeError('Generated-state full-model gradient replay or identity check failed')
            for i in range(3):
                for j,g in enumerate(grads[i]):
                    accum[i][j]+=g/row_count;seed_accum[i][j]+=g/16
            rows.append(dict(seed=seed,time=time,base_loss=float(losses[0].detach()),supervised_paired_loss=float(losses[1].detach()),generated_epsilon_loss=float(losses[2].detach()),
                phase_metrics={str(phase):dict(base_epsilon=base_errors[2*pi:2*pi+2].detach().cpu().tolist(),rank=rank[2*pi:2*pi+2].detach().cpu().tolist(),generated_epsilon=generated_errors[2*pi:2*pi+2].detach().cpu().tolist(),generated_unclipped_clean=clean_errors[2*pi:2*pi+2].detach().cpu().tolist()) for pi,phase in enumerate(protocol['train_phases'])},
                base_vs_generated=pair_stats(names,grads[0],grads[2]),paired_vs_generated=pair_stats(names,grads[1],grads[2])))
            write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,rows=rows))
        seed_summaries[str(seed)]=dict(base_vs_generated=pair_stats(names,seed_accum[0],seed_accum[2]),paired_vs_generated=pair_stats(names,seed_accum[1],seed_accum[2]))
        print(json.dumps(dict(seed=seed,all16times_complete=True)),flush=True)
    # Prompt-only zero control uses identical q states for both forward calls.
    policy.obs_encoder.condition_mode='zero_context'
    t=torch.full((14,),24,device='cuda',dtype=torch.long);q=scheduler.add_noise(target,initial[0],t)
    wrong=dict(normalized);wrong[CONTEXT_KEY]=normalized[CONTEXT_KEY][permutation]
    predictions=[policy.model(q,t,cond=policy.obs_encoder(o,training=False),training=False,gen_attn_map=False)[0] for o in (normalized,wrong)]
    zero=(predictions[0]-initial[0]).square().mean()-(predictions[1]-initial[0]).square().mean()
    zg=torch.autograd.grad(zero,parameters,allow_unused=True)
    checks['zero_context_prompt_outputs_exact']=torch.equal(*predictions)
    checks['zero_context_prompt_gradient_exact_zero']=all(g is None or bool((g==0).all()) for g in zg)
    policy.obs_encoder.condition_mode='demo_geometry'
    checks['full_state_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
    checks['scheduler_values_unchanged']=torch.equal(alpha_before,scheduler.alphas_cumprod.cpu())
    checks['rng_unchanged']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    checks['no_accumulated_gradients']=all(p.grad is None for p in policy.parameters())
    checks[f'all{row_count}rows']=len(rows)==row_count
    aggregate=dict(base_vs_generated=pair_stats(names,accum[0],accum[2]),paired_vs_generated=pair_stats(names,accum[1],accum[2]))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for group,ax in zip(('full_model','geometry_columns','target_encoder','denoiser'),axes.flat):
        for seed in protocol['inference_seeds']:
            subset=[r for r in rows if r['seed']==seed]
            ax.plot([r['time'] for r in subset],[r['paired_vs_generated'][group]['cosine'] for r in subset],label=str(seed))
        ax.axhline(0,color='black',linewidth=.7);ax.set_title(group);ax.invert_xaxis();ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7);fig.suptitle('Full TRAIN raw-gradient cosine: supervised+rank versus generated-state correction; no optimizer')
    fig.tight_layout();fig.savefig(OUT/'GRADIENT_COSINES.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,rows=rows,aggregate=aggregate,seed_summaries=seed_summaries,
        plot_inspected=False,full_parameter_count=8319216,actual_train_cases=14,gradient_rows=row_count,
        new_optimizer_updates=0,new_sample_paths=0,new_physics_steps=0,
        scope='Single full025 endpoint; eval-mode FP32 full gradients on detached saved diffusion states and original actualTRAIN targets. Uniform16 inference times, not an unbiased all50time training gradient. Initial Gaussian is common supervised noise, not each later innovation. No gradient through sampling, Adam step, coefficient selection or new training budget.'))
    assert all(checks.values())
    print(json.dumps(aggregate),flush=True)


if __name__=='__main__':main()
