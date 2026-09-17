"""Full official model point evaluation on two frozen saved sampling paths."""
import argparse
import hashlib
import json
import os
import socket

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.probe_generator_reverse_process import (
    PAIRED_OUT, SELF_SOURCE, RANKED_SOURCE, PARENT, ActualBranchGeometryDataset,
    GeneratorWrapper, CONTEXT_KEY, restore_geometry_state, write,
)

OUT=PAIRED_OUT/'fixed_path_transfer'


def prepare():
    for name in ('ranked18','self18'):
        result=json.loads((PAIRED_OUT/name/'RESULT.json').read_text())
        assert result['checks_passed'] and result['plot_inspected']
    readback=json.loads((PAIRED_OUT/'SAVED_READBACK.json').read_text())
    assert readback['checks_passed']
    protocol=json.loads((PAIRED_OUT/'PROTOCOL.json').read_text())
    OUT.mkdir(exist_ok=False)
    write(OUT/'PROTOCOL.json',dict(phases=protocol['phases'],seeds=protocol['seeds'],
        point_forward_calls=4224,zero_control_calls=2,source_hashes=readback['source_hashes'],
        scope='Correct-prompt paths only, all11phases/4existingprimaryseeds/2branches/16times. Exact fullranked18-on-ranked18 and fullself18-on-self18 epsilon/clean replays, plus self18-on-ranked18 point outputs. No new sampler calls, gradients, optimization or physics. Reused218/258 evaluation only. Error decomposition is anchored to ranked18 states and is descriptive, not unique causal attribution or a hybrid controller.',
        automatic_next_action='Require fullstate/RNG/scheduler/target/ownpoint exactness and all11plots plus saved readback. Compare direct model effect on fixed ranked18 states with residual from own state changes, especially277 and four passingTRAIN regressions. Select one bounded TRAIN diagnosis from evidence, without teacher iteration, coefficient/solver sweep, old512extension or selecting intermediate outputs.'))


def run():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    assert not (OUT/'PARTIAL_RESULT.json').exists() and not (OUT/'RESULT.json').exists()
    plans={name:json.loads((source/'PROTOCOL.json').read_text()) for name,source in [('ranked18',RANKED_SOURCE),('self18',SELF_SOURCE)]}
    checks={};outputs={};calls=0;zero_calls=0;norms={};scheduler_reference=None
    for name,source in [('ranked18',RANKED_SOURCE),('self18',SELF_SOURCE)]:
        with (source/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
            state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
        policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
        restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
        policy.cuda().eval().requires_grad_(False)
        assert sum(p.numel() for p in policy.parameters())==8319216
        scheduler=policy.noise_scheduler;config=dict(scheduler.config);alpha=scheduler.alphas_cumprod.cpu().clone()
        assert config['prediction_type']=='epsilon' and config['clip_sample'] and config['clip_sample_range']==1 and not config['thresholding']
        if scheduler_reference is None:scheduler_reference=(config,alpha)
        checks[name+'_scheduler_matched']=config==scheduler_reference[0] and torch.equal(alpha,scheduler_reference[1])
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        for phase in protocol['phases']:
            dataset=ActualBranchGeometryDataset(plans[name]['phase_corpora'][str(phase)],plans[name]['normalizer_state'])
            samples=[dataset[i] for i in (0,1)]
            obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
            normalized=policy.normalizer.normalize(obs)
            target=policy.normalizer['action'].normalize(torch.stack([s['action'] for s in samples]).cuda()).cpu().numpy()
            if name=='ranked18':norms[phase]={k:v.cpu() for k,v in normalized.items()}
            checks[f'{name}_{phase}_normalized_inputs_exact']=all(torch.equal(v.cpu(),norms[phase][k]) for k,v in normalized.items())
            pathnames=('ranked18',) if name=='ranked18' else ('self18','ranked18')
            with torch.inference_mode():
                tokens=[policy.obs_encoder({k:v[i:i+1] for k,v in normalized.items()},training=False) for i in (0,1)]
                for pathname in pathnames:
                    with np.load(PAIRED_OUT/pathname/f'phase_{phase}_steps_16.npz') as archive:
                        recorded={k:archive[k].copy() for k in archive.files}
                    key=name+'_on_'+pathname
                    checks[f'{phase}_{key}_target_exact']=np.array_equal(target.astype(np.float64),recorded['normalized_actual_target'])
                    checks[f'{phase}_{key}_seeds']=recorded['seeds'].tolist()==protocol['seeds']
                    xt=recorded['xt'][:,0];epsilon=np.empty_like(xt);clean=np.empty_like(xt)
                    for si in range(4):
                        for branch in (0,1):
                            for ti,time in enumerate(recorded['times']):
                                sample=torch.from_numpy(xt[si,branch,ti:ti+1]).cuda()
                                point=policy.model(sample,torch.tensor(int(time),device='cuda'),cond=tokens[branch],training=False,gen_attn_map=False)[0]
                                raw=(sample-(1-alpha[int(time)])**.5*point)/alpha[int(time)]**.5
                                epsilon[si,branch,ti]=point[0].cpu().numpy();clean[si,branch,ti]=raw[0].clamp(-1,1).cpu().numpy();calls+=1
                    if name==pathname:
                        checks[f'{phase}_{key}_all_epsilon_exact']=np.array_equal(epsilon,recorded['epsilon'][:,0])
                        checks[f'{phase}_{key}_all_clean_exact']=np.array_equal(clean,recorded['pred_original'][:,0])
                    checks[f'{phase}_{key}_finite']=bool(np.isfinite(epsilon).all() and np.isfinite(clean).all())
                    arrays=dict(epsilon=epsilon,clean=clean,target=target,times=recorded['times'],seeds=recorded['seeds'])
                    file=OUT/f'phase_{phase}_{key}.npz';np.savez_compressed(file,**arrays)
                    with np.load(file) as saved:checks[f'{phase}_{key}_saved_exact']=all(np.array_equal(saved[k],v) for k,v in arrays.items())
                    outputs.setdefault(str(phase),{})[key]=((clean.astype(np.float64)-target[None,:,None])**2).mean((-1,-2)).mean(0)
                if name=='self18' and phase==protocol['phases'][0]:
                    policy.obs_encoder.condition_mode='zero_context';pred=[]
                    for ci in (0,1):
                        inputs={k:v[:1] for k,v in normalized.items()};inputs[CONTEXT_KEY]=normalized[CONTEXT_KEY][ci:ci+1]
                        pred.append(policy.model(torch.from_numpy(xt[0,0,:1]).cuda(),torch.tensor(int(recorded['times'][0]),device='cuda'),cond=policy.obs_encoder(inputs,training=False),training=False,gen_attn_map=False)[0]);zero_calls+=1
                    checks['self18_zero_context_prompt_exact']=torch.equal(*pred)
                    policy.obs_encoder.condition_mode='demo_geometry'
            assert all(checks.values()),[k for k,v in checks.items() if not v]
            write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,point_forward_calls=calls,completed_model=name,completed_phase=phase))
            print(json.dumps(dict(model=name,phase=phase,point_forward_calls=calls)),flush=True)
        checks[name+'_fullstate_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
        checks[name+'_rng_unchanged']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        checks[name+'_scheduler_unchanged']=config==dict(scheduler.config) and torch.equal(alpha,scheduler.alphas_cumprod.cpu())
        checks[name+'_no_gradients']=all(p.grad is None for p in policy.parameters())
        del policy,state
    checks['point_budget_exact']=calls==protocol['point_forward_calls'];checks['zero_budget_exact']=zero_calls==protocol['zero_control_calls']
    for filename,digest in protocol['source_hashes'].items():checks['source_unchanged_'+filename]=hashlib.sha256(open(filename,'rb').read()).hexdigest()==digest
    rows={}
    for phase,values in outputs.items():
        old=values['ranked18_on_ranked18'];new=values['self18_on_self18'];transferred=values['self18_on_ranked18']
        direct=transferred-old;feedback=new-transferred;total=new-old
        checks[phase+'_decomposition_exact']=bool(np.allclose(total,direct+feedback,rtol=1e-12,atol=1e-15))
        rows[phase]=dict(clean_mse={k:v.tolist() for k,v in values.items()},direct_effect=direct.tolist(),state_feedback_residual=feedback.tolist(),total_change=total.tolist(),
            final_mean=dict(old=float(old[:,-1].mean()),new=float(new[:,-1].mean()),direct=float(direct[:,-1].mean()),feedback=float(feedback[:,-1].mean())))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(18,16))
    for phase,ax in zip(protocol['phases'],axes.flat):
        for key,color in [('ranked18_on_ranked18','C0'),('self18_on_self18','C1'),('self18_on_ranked18','C2')]:
            for branch,style in enumerate(('-','--')):ax.plot(recorded['times'],outputs[str(phase)][key][branch],color=color,ls=style,label=key+f' b{branch}')
        ax.set_title(f'Phase{phase}');ax.set_yscale('log');ax.invert_xaxis();ax.grid(alpha=.25)
    axes.flat[-1].set_visible(False);axes.flat[0].legend(fontsize=6)
    fig.suptitle('Official full models on frozen saved4seed paths; transfer is diagnostic only')
    fig.tight_layout();fig.savefig(OUT/'PATH_TRANSFER.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,point_forward_calls=calls,zero_control_calls=zero_calls,plot_inspected=False,new_sample_paths=0,new_optimizer_updates=0,new_physics_steps=0,scope=protocol['scope']))
    assert all(checks.values()),[k for k,v in checks.items() if not v]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['prepare','run'],required=True)
    args=parser.parse_args()
    prepare() if args.mode=='prepare' else run()


if __name__=='__main__':main()
