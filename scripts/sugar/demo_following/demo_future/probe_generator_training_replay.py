"""Frozen official model readback on the exact TRAIN latent replay inputs.

Point forwards preserve the original teacher recording arithmetic. Actual
TRAIN command targets stay fixed; diffusion states are never physical labels.
"""
import json
import argparse
import os
import socket

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import (
    BASE, PARENT, ActualBranchGeometryDataset, GeneratorWrapper, CONTEXT_KEY,
    restore_geometry_state, write,
)

RUN = BASE / 'matched_generator_branch_latent_replay01_gaps512'
OLD = BASE / 'matched_generator_branch_latent_replay01512'
TEACHER = BASE / 'matched_generator_branch_paired_rank025512'
OUT = RUN / 'frozen_evaluation/training_replay_probe'


def main():
    global RUN, TEACHER, OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-replay',action='store_true')
    args=parser.parse_args()
    if args.self_replay:
        TEACHER=RUN
        RUN=BASE/'matched_generator_branch_self_replay01_gaps512'
        OUT=RUN/'frozen_evaluation/training_replay_probe'
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    prior=json.loads((RUN/'frozen_evaluation/rank_component_audit/RESULT.json').read_text())
    assert prior['checks_passed'] and prior['plot_inspected']
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    replay=BASE/('generator_train_diffusion_replay8_current_gaps' if args.self_replay else 'generator_train_diffusion_replay8_gaps2')
    assert plan['generated_state_objective']['replay_source']==str(replay)
    with np.load(replay/'TRAIN_GENERATED_STATE_INPUTS.npz') as a:
        saved={k:a[k].copy() for k in a.files}
    phases=plan['data']['train']['phases'];seeds=saved['seeds'].tolist();times=saved['times'].tolist()
    assert phases==[158,178,197,221,245,261,277,298,318]
    assert seeds==list(range(272230,272238)) and times==list(range(45,-1,-3))
    OUT.mkdir(exist_ok=False)
    models=[('teacher',TEACHER),('self18',RUN)] if args.self_replay else [('teacher',TEACHER),('old14',OLD),('new18',RUN)]
    protocol=dict(models=[str(path) for _,path in models],replay_source=str(replay),
        phases=phases,seeds=seeds,times=times,teacher_exact_point_forwards=2304,
        learned_model_correct_wrong_point_forwards=9216,total_point_forwards=11520,zero_control_forwards=4,
        scope='Full8319216 official models. All18realTRAIN cases,8saved replay seeds,16saved times. Same oldfull025 teacher xt and actual targets for both learned models; no fresh paths, gradients, optimizer updates, new physical states, check labels, coefficient choice or BF16/Adam claim. Wrong prompt is fixed-xt sensitivity only. All2304teacher point predictions and official clean reconstruction must replay exactly.',
        automatic_next_action='If the failed TRAIN277/298 supervised and replay errors both improve while saved generation worsens, inspect exact own sampling paths. Otherwise audit old/new TRAIN objective interactions before another matched budget. Preserve both reused-check gains and all TRAIN regressions; no generated deployment admission.')
    if args.self_replay:
        protocol.update(learned_model_correct_wrong_point_forwards=4608,total_point_forwards=6912,zero_control_forwards=2,
            scope='Full8319216 ranked18teacher and self18learner. Exact current8seed16time18case TRAIN replay used by the completed teacher-refresh experiment. All2304teacher epsilon/clean point replays exact, learner4608correct/wrong forwards. No new paths, gradients, updates, check labels or physics. Fixed-xt point error is not own reverse-process performance.',
            automatic_next_action='Inspect all9TRAIN plots/fullstates/RNG/scheduler/teacher exactpoint and savedarray checks. Separate persistent277 common/contrast error from four passing TRAIN regressions. If replay fitting improves but generation regresses, record exact own saved inference paths for a bounded transfer diagnosis; otherwise audit the fixed TRAIN objective interaction. Do not iterate teachers, sweep weights, extend512 or substitute averaged outputs.')
    write(OUT/'PROTOCOL.json',protocol)
    samples=[];expected=[];expected_clean=[];reference_xt=[]
    for phase in phases:
        dataset=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state'])
        samples.extend(dataset[i] for i in (0,1))
        with np.load(replay/f'phase_{phase}_steps_16.npz') as a:
            expected.append(a['epsilon'][:,0].transpose(0,2,1,3,4))
            expected_clean.append(a['pred_original'][:,0].transpose(0,2,1,3,4))
            reference_xt.append(a['xt'][:,0].transpose(0,2,1,3,4))
    expected=np.concatenate(expected,axis=2);expected_clean=np.concatenate(expected_clean,axis=2)
    obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
    actions=torch.stack([s['action'] for s in samples]).cuda();xt=torch.from_numpy(saved['xt']).cuda()
    checks=dict(all18_recorded_xt_exact=np.array_equal(saved['xt'],np.concatenate(reference_xt,axis=2)),
        actual_train_phase_binding=saved['phases'].tolist()==[p for p in phases for _ in (0,1)],
        no_reused_check_targets=not set(phases).intersection([218,258]))
    outputs={};reference_norm=None;reference_scheduler=None
    for name,run in models:
        with (run/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as stream:
            state=torch.load(stream,pickle_module=dill,map_location='cpu')['state_dicts']['model']
        policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
        restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
        policy.cuda().eval().requires_grad_(False)
        assert sum(p.numel() for p in policy.parameters())==8319216
        normalized=policy.normalizer.normalize(obs);target=policy.normalizer['action'].normalize(actions)
        checks[name+'_actual_targets_exact']=np.array_equal(target.cpu().numpy().astype(np.float64),saved['normalized_actual_target'])
        norm={k:v.cpu() for k,v in normalized.items()}
        if reference_norm is None:reference_norm=norm
        checks[name+'_normalized_inputs_exact']=all(torch.equal(v,reference_norm[k]) for k,v in norm.items())
        scheduler=policy.noise_scheduler;alpha=scheduler.alphas_cumprod.detach().cpu().clone()
        sched=dict(scheduler.config)
        if reference_scheduler is None:reference_scheduler=(sched,alpha)
        checks[name+'_scheduler_exact']=sched==reference_scheduler[0] and torch.equal(alpha,reference_scheduler[1])
        assert sched['prediction_type']=='epsilon' and sched['clip_sample_range']==1 and sched['clip_sample']
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        conditions=1 if name=='teacher' else 2
        epsilon=np.empty((8,16,conditions,18,8,36),dtype=np.float32)
        clean=np.empty_like(epsilon);unclipped=np.empty_like(epsilon)
        with torch.inference_mode():
            tokens=[]
            for ci in range(conditions):
                values=[]
                for case in range(18):
                    inputs={k:v[case:case+1] for k,v in normalized.items()}
                    if ci:inputs[CONTEXT_KEY]=normalized[CONTEXT_KEY][case^1:(case^1)+1]
                    values.append(policy.obs_encoder(inputs,training=False))
                tokens.append(values)
            for si,seed in enumerate(seeds):
                for ti,time in enumerate(times):
                    t=torch.tensor(time,device='cuda');root_alpha=alpha[time]**.5;root_beta=(1-alpha[time])**.5
                    for ci in range(conditions):
                        for case in range(18):
                            point=policy.model(xt[si,ti,case:case+1],t,cond=tokens[ci][case],training=False,gen_attn_map=False)[0]
                            x0=(xt[si,ti,case:case+1]-root_beta*point)/root_alpha
                            epsilon[si,ti,ci,case]=point[0].cpu().numpy()
                            unclipped[si,ti,ci,case]=x0[0].cpu().numpy()
                            clean[si,ti,ci,case]=x0[0].clamp(-1,1).cpu().numpy()
                if name=='teacher':
                    checks[f'teacher_{seed}_all288_epsilon_exact']=np.array_equal(epsilon[si,:,0],expected[si])
                    checks[f'teacher_{seed}_all288_clean_exact']=np.array_equal(clean[si,:,0],expected_clean[si])
                    assert all(checks.values())
                print(json.dumps(dict(model=name,seed=seed,all16times_complete=True)),flush=True)
            if name!='teacher':
                policy.obs_encoder.condition_mode='zero_context';pred=[]
                for case_context in (0,1):
                    inputs={k:v[:1] for k,v in normalized.items()};inputs[CONTEXT_KEY]=normalized[CONTEXT_KEY][case_context:case_context+1]
                    pred.append(policy.model(xt[0,0,:1],torch.tensor(times[0],device='cuda'),cond=policy.obs_encoder(inputs,training=False),training=False,gen_attn_map=False)[0])
                checks[name+'_zero_context_prompt_exact']=torch.equal(*pred)
                policy.obs_encoder.condition_mode='demo_geometry'
        checks[name+'_full_state_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
        checks[name+'_rng_unchanged']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        checks[name+'_no_gradients']=all(p.grad is None for p in policy.parameters())
        checks[name+'_scheduler_values_unchanged']=torch.equal(alpha,scheduler.alphas_cumprod.cpu())
        checks[name+'_all_outputs_finite']=all(np.isfinite(a).all() for a in (epsilon,clean,unclipped))
        oracle=np.empty_like(saved['xt'])
        for ti,time in enumerate(times):
            oracle[:,ti]=((xt[:,ti]-(alpha[time]**.5)*target)/((1-alpha[time])**.5)).cpu().numpy()
        arrays=dict(epsilon=epsilon,clean=clean,unclipped=unclipped,oracle=oracle,target=target.cpu().numpy(),seeds=seeds,times=times,phases=saved['phases'])
        np.savez_compressed(OUT/(name+'.npz'),**arrays)
        with np.load(OUT/(name+'.npz')) as a:
            checks[name+'_saved_arrays_exact']=all(np.array_equal(a[k],v) for k,v in arrays.items())
        outputs[name]=dict(epsilon_mse=((epsilon-oracle[:,:,None])**2).mean((-1,-2)),
            clean_mse=((clean-target.cpu().numpy()[None,None,None])**2).mean((-1,-2)),
            unclipped_mse=((unclipped-target.cpu().numpy()[None,None,None])**2).mean((-1,-2)))
        write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,completed_models=list(outputs)))
        assert all(checks.values());del policy,state
    rows={}
    for pi,phase in enumerate(phases):
        rows[str(phase)]={name:{key:value[:,:,:,2*pi:2*pi+2].mean((0,1,3)).tolist() for key,value in metrics.items()} for name,metrics in outputs.items()}
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(16,12))
    for pi,(phase,ax) in enumerate(zip(phases,axes.flat)):
        for name,color in ([('teacher','C0'),('self18','C1')] if args.self_replay else [('teacher','C2'),('old14','C0'),('new18','C1')]):
            for branch,style in enumerate(('-','--')):
                ax.plot(times,outputs[name]['clean_mse'][:,:,0,2*pi+branch].mean(0),style,color=color,label=f'{name} b{branch}')
        ax.set_title(f'TRAIN{phase}');ax.set_yscale('log');ax.invert_xaxis();ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7);fig.suptitle('Same frozen teacher xt, actual TRAIN targets; point clean error, NOT own sampling')
    fig.tight_layout();fig.savefig(OUT/'REPLAY_ERRORS.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,
        plot_inspected=False,new_optimizer_updates=0,new_sample_paths=0,new_physics_steps=0,
        scope='Uniform8seed/16time frozen TRAIN replay input error. Descriptive eval FP32 point forwards, not actual training loss trajectory, own generated paths or Adam-gradient evidence.'))
    assert all(checks.values())


if __name__=='__main__':main()
