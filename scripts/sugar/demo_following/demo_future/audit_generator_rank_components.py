"""Matched frozen diagnostics of full official Generator objective adaptations.

No new network, optimization, controller, or physics. Saved generation errors
and target-corrupted denoising errors are deliberately reported separately.
"""
import argparse
import json
import os
import socket
from pathlib import Path

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import (
    BASE, PARENT, ActualBranchGeometryDataset, GeneratorWrapper, CONTEXT_KEY,
    restore_geometry_state, write,
)

RUNS = [BASE / name for name in (
    'matched_generator_branch_dense_fit512',
    'matched_generator_branch_paired_rank512',
    'matched_generator_branch_paired_rank025512',
)]
OUT = RUNS[-1] / 'frozen_evaluation/rank_component_audit'
CHANNELS = {'joint29': slice(0, 29), 'linear3': slice(29, 32),
            'angular3': slice(32, 35), 'contact1': slice(35, 36)}


def prepare():
    OUT.mkdir(exist_ok=False)
    plans = [json.loads((r / 'PROTOCOL.json').read_text()) for r in RUNS]
    ends = [json.loads((r / 'RESULT.json').read_text()) for r in RUNS]
    assert all(e['execution_completed'] and e['horizon_plots_inspected'] for e in ends)
    comparison='LATENT_REPLAY_COMPARISON.json' if plans[-1].get('generated_state_objective') else 'PAIRED_COMPARISON.json'
    assert json.loads((RUNS[-1] / 'frozen_evaluation' / comparison).read_text())['checks_passed']
    gap = RUNS[-1].name == 'matched_generator_branch_latent_replay01_gaps512'
    for key in (('normalizer_state', 'seed', 'actual_optimizer_updates') if gap else ('phase_corpora', 'normalizer_state', 'seed', 'batch_size', 'actual_optimizer_updates')):
        assert all(p[key] == plans[0][key] for p in plans)
    if gap:
        assert [p['batch_size'] for p in plans] == [112,144]
        # The expanded corpus preserves copied old arrays under new paths.
        # Verify their contents rather than requiring identical directory names.
        for phase in plans[0]['phase_corpora']:
            datasets=[ActualBranchGeometryDataset(p['phase_corpora'][phase],p['normalizer_state']) for p in plans]
            for branch in (0,1):
                a,b=[d[branch] for d in datasets]
                assert torch.equal(a['action'],b['action'])
                assert a['obs'].keys()==b['obs'].keys() and all(torch.equal(v,b['obs'][k]) for k,v in a['obs'].items())
    weights=[p.get('paired_objective',{}).get('weight',0.) for p in plans]
    scopes=[p.get('paired_objective',{}).get('gradient_scope','full_model') for p in plans]
    functions=[p.get('paired_objective',{}).get('rank_function','softplus') for p in plans]
    protocol = dict(runs=[str(r) for r in RUNS], weights=weights, gradient_scopes=scopes, rank_functions=functions,
        generated_state_weights=[p.get('generated_state_objective',{}).get('weight',0.) for p in plans],
        phases=plans[-1]['frozen_evaluation']['evaluation_phases'],
        noise_seeds=list(range(272190, 272194)), diffusion_times=list(range(50)),
        conditions=['correct', 'paired_wrong'],
        new_optimizer_updates=0, new_physics_steps=0,
        scope=f'Same18actual cases/order and saved common noise arrays across all{len(RUNS)}complete official endpoints. Target-corrupted inputs are supervised diagnostics, never generated futures. Saved32draw generation decomposition is separate. All known TRAIN motions and reused checks; no model/solver changes.',
        automatic_next_action='Inspect all full-state/normalizer/input/noise checks and all nine phase curves. Use TRAIN component evidence to choose one bounded diagnostic of denoising supervision or solver behavior; report reused checks separately. Do not automatically sweep ranking weights, extend512 endpoints, or infer generated physics/SMP benefit.')
    if gap:
        protocol.update(common_phases=plans[0]['frozen_evaluation']['evaluation_phases'], new_phases=[221,261],
            scope='Two full8319216 frozen latent-replay endpoints, old112/new144 training batches. All22actual cases in the new11phase order. Exact old18case noise and forward batch replay, separate4newcase forwards. Same targets/noises across models; no equal training FLOPs or old q-noise claim. Saved32draw decomposition only on available primary phases; no invented old221/261 samples.',
            automatic_next_action='Inspect all11phase denoising curves and exact old18case replay. Diagnose failed TRAIN277/298 and early TRAIN regressions separately from successful reused218/258 transfer before another training budget or generated physics.')
    if RUNS[-1].name=='matched_generator_branch_self_replay01_gaps512':
        previous=RUNS[0]/'frozen_evaluation/rank_component_audit'
        original=json.loads((previous/'PROTOCOL.json').read_text())
        evidence=json.loads((RUNS[-1]/'frozen_evaluation/centroid_error_readback/RESULT.json').read_text())
        assert evidence['checks_passed'] and evidence['plot_inspected']
        protocol.update(common_phases=original['common_phases'],new_phases=original['new_phases'],replay_previous22=True,
            scope='Two frozen complete18case8319216models, old and refreshed teacher adaptation. All22actual cases/11phases,4saved common noises/all50training times. Preserve exact old22noise arrays and original18+4forward chunks; reproduce all3old-model arrays exactly. No gradients, optimizer, new sample paths or physics. Known TRAIN sources/reused checks; target-corrupted denoising is not generated-state accuracy.',
            automatic_next_action='Inspect all11plots and fullstates/normalizer/scheduler/exactold22replay checks. Compare TRAIN277 and the4regressed passing TRAIN phases with saved common/contrast generation bias. Choose a bounded full-model conditioning or generated-state diagnosis; no teacher iteration, coefficient sweep or endpoint-budget extension from this probe alone.')
    write(OUT / 'PROTOCOL.json', protocol)
    state = torch.load(plans[0]['normalizer_state'], map_location='cpu')['model']
    scale = state['normalizer.params_dict.action.scale'].double().numpy()
    checks, rows = {}, {}
    for phase in protocol['phases']:
        row, reference = {}, None
        for run, endpoint in zip(RUNS, ends):
            if str(phase) not in endpoint['phases']:
                row[run.name] = dict(primary_generation_unavailable=True)
                continue
            with np.load(run / f'frozen_evaluation/phase_{phase}/trained_demo_geometry_correct.npz') as a:
                prediction = a['predictions'].astype(np.float64) * scale
                target = a['actual_future_command_target'].astype(np.float64) * scale
                seeds = a['sample_seeds'].copy()
            if reference is None:
                reference = target.copy(), seeds.copy()
            checks[f'{phase}_{run.name}_target_seed_exact'] = bool(np.array_equal(target, reference[0]) and np.array_equal(seeds, reference[1]))
            mean = prediction.mean(0)
            error = prediction - target[None]
            mse, bias, variance = float(np.mean(error**2)), float(np.mean((mean-target)**2)), float(np.mean((prediction-mean[None])**2))
            delta = (target[1] - target[0]).reshape(-1)
            direction = delta / np.linalg.norm(delta)
            flat = error.reshape(32, 2, -1)
            parallel = np.sum(flat * direction, axis=-1)
            parallel_mse = float(np.mean(parallel**2) / len(delta))
            checks[f'{phase}_{run.name}_decomposition'] = bool(np.isclose(mse, bias+variance, rtol=1e-10) and np.isclose(mse, endpoint['phases'][str(phase)]['variants']['trained_demo_geometry_correct']['normalized_mse'], rtol=1e-5) and mse >= parallel_mse - 1e-12)
            row[run.name] = dict(total_mse=mse, draw_mean_mse=bias, draw_variance=variance,
                variance_fraction=variance/mse, branch_axis_error_mse=parallel_mse,
                orthogonal_error_mse=mse-parallel_mse,
                mean_condition_response_projection=float(np.dot((mean[1]-mean[0]).reshape(-1), delta)/np.dot(delta, delta)),
                per_channel={k: dict(mse=float(np.mean(error[..., sl]**2)),
                    weighted_contribution=float(np.sum(error[..., sl]**2)/error.size)) for k, sl in CHANNELS.items()},
                primary_pass=endpoint['phases'][str(phase)]['branch_identifiability_passed'])
        rows[str(phase)] = row
    losses = {}
    for arm in ('zero_context', 'demo_geometry'):
        values = json.loads((RUNS[-1] / arm / 'PAIRED_LOSSES.json').read_text())
        checks[arm+'_512_loss_records'] = len(values) == 512
        checks[arm+'_loss_arithmetic'] = all(np.isfinite([v['base'],v['rank'],v['total']]).all() and v['weight']==.25 and np.isclose(v['total'],v['base']+.25*v['rank']+v.get('generated_weight',0.)*v.get('generated',0.),rtol=2e-6,atol=2e-6) for v in values)
        losses[arm] = dict(records=len(values), first=values[0], last=values[-1],
            scope='Different random noise/time at each update; first/last loss is not a fixed-condition performance comparison.')
    result = dict(checks=checks, checks_passed=all(checks.values()), phases=rows, loss_records=losses,
        scope='All saved draws, not best-draw selection. Mean/direction decompositions are descriptive, not substitute outputs or success metrics.',
        new_optimizer_updates=0, new_sample_draws=0, new_physics_steps=0)
    write(OUT / 'SAVED_COMPONENTS.json', result)
    assert result['checks_passed']
    print(json.dumps(dict(saved_components_passed=True, phases=rows)), flush=True)


def frozen():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login', 'mgmtserver'))
    torch.set_num_threads(8)
    protocol = json.loads((OUT / 'PROTOCOL.json').read_text())
    assert json.loads((OUT / 'SAVED_COMPONENTS.json').read_text())['checks_passed']
    assert not (OUT / 'COMMON_NOISE.npz').exists()
    phases, seeds, times = protocol['phases'], protocol['noise_seeds'], protocol['diffusion_times']
    plan = json.loads((RUNS[-1] / 'PROTOCOL.json').read_text())
    samples = []
    for phase in phases:
        dataset = ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)], plan['normalizer_state'])
        samples.extend([dataset[i] for i in (0, 1)])
    obs = {k: torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
    actions = torch.stack([s['action'] for s in samples]).cuda()
    noises = []
    gap = 'common_phases' in protocol
    noise_phases = protocol['common_phases'] + protocol['new_phases'] if gap else phases
    noise_order = [noise_phases.index(p) for p in phases]
    for seed in seeds:
        g = torch.Generator(device='cuda').manual_seed(seed)
        if gap:
            phase_noise = torch.cat([torch.randn((len(protocol['common_phases']),8,36), generator=g, device='cuda'),
                                     torch.randn((len(protocol['new_phases']),8,36), generator=g, device='cuda')])
        else:
            phase_noise = torch.randn((len(phases),8,36), generator=g, device='cuda')
        noises.append(phase_noise[noise_order].repeat_interleave(2,0))
    noise = torch.stack(noises)
    np.savez_compressed(OUT / 'COMMON_NOISE.npz', noise=noise.cpu().numpy(), seeds=seeds, phases=np.repeat(phases,2), branches=np.tile([0,1],len(phases)))
    with np.load(OUT / 'COMMON_NOISE.npz') as a:
        saved_noise_exact = np.array_equal(a['noise'], noise.cpu().numpy())
    checks = dict(saved_noise_exact=saved_noise_exact, paired_noise_exact=torch.equal(noise[:,::2],noise[:,1::2]))
    chunks = [list(range(len(samples)))]
    if gap:
        common_indices = [2*phases.index(p)+b for p in protocol['common_phases'] for b in (0,1)]
        new_indices = [2*phases.index(p)+b for p in protocol['new_phases'] for b in (0,1)]
        chunks = [common_indices,new_indices]
        with np.load(RUNS[0]/'frozen_evaluation/rank_component_audit/COMMON_NOISE.npz') as old:
            if protocol.get('replay_previous22'):
                checks['old22case_noise_exact']=np.array_equal(noise.cpu().numpy(),old['noise'])
            else:checks['old18case_noise_exact'] = np.array_equal(noise[:,common_indices].cpu().numpy(),old['noise'])
        assert all(checks.values())
    shape = (len(RUNS),len(seeds),len(times),2,len(samples))
    metrics = {k: np.empty(shape, dtype=np.float64) for k in ('epsilon_mse','clean_unclipped_mse','clean_clipped_mse')}
    reference_norm = None
    reference_scheduler = None
    for ri, run in enumerate(RUNS):
        with (run / 'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
            state = torch.load(f, pickle_module=dill, map_location='cpu')['state_dicts']['model']
        policy = GeneratorWrapper.load(str(PARENT),device='cpu').policy
        restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
        policy.cuda().eval().requires_grad_(False)
        assert sum(p.numel() for p in policy.parameters()) == 8319216
        normalized = policy.normalizer.normalize(obs)
        target = policy.normalizer['action'].normalize(actions)
        norm = {k:v.detach().cpu() for k,v in normalized.items()}; norm['actual_target']=target.cpu()
        if reference_norm is None: reference_norm = norm
        checks[run.name+'_all_normalized_inputs_targets_exact'] = all(torch.equal(v,reference_norm[k]) for k,v in norm.items())
        wrong = dict(normalized); wrong[CONTEXT_KEY] = normalized[CONTEXT_KEY][torch.arange(len(samples),device='cuda') ^ 1]
        checks[run.name+'_only_geometry_swapped'] = all(torch.equal(normalized[k],wrong[k]) for k in normalized if k!=CONTEXT_KEY)
        scheduler = policy.noise_scheduler
        assert scheduler.config.num_train_timesteps==50 and scheduler.config.prediction_type=='epsilon'
        actual_scheduler = (dict(scheduler.config), scheduler.alphas_cumprod.cpu().clone())
        if reference_scheduler is None:
            reference_scheduler = actual_scheduler
        checks[run.name+'_scheduler_exact'] = actual_scheduler[0] == reference_scheduler[0] and torch.equal(actual_scheduler[1], reference_scheduler[1])
        with torch.inference_mode():
            tokens=[[policy.obs_encoder({k:v[indices] for k,v in condition.items()},training=False) for indices in chunks] for condition in (normalized,wrong)]
            for si in range(len(seeds)):
                for ti,time in enumerate(times):
                    t=torch.full((len(samples),),time,dtype=torch.long,device='cuda')
                    noisy=scheduler.add_noise(target,noise[si],t)
                    alpha=scheduler.alphas_cumprod[time].cuda()
                    for ci,condition_chunks in enumerate(tokens):
                        pred=torch.empty_like(noisy)
                        for indices,cond in zip(chunks,condition_chunks):
                            pred[indices]=policy.model(noisy[indices],t[indices],cond=cond,training=False,gen_attn_map=False)[0]
                        clean=(noisy-(1-alpha).sqrt()*pred)/alpha.sqrt()
                        clipped=clean.clamp(-scheduler.config.clip_sample_range,scheduler.config.clip_sample_range)
                        for key,error in [('epsilon_mse',(pred-noise[si])**2),('clean_unclipped_mse',(clean-target)**2),('clean_clipped_mse',(clipped-target)**2)]:
                            metrics[key][ri,si,ti,ci]=error.mean((1,2)).cpu().numpy()
                print(json.dumps(dict(run=run.name,noise_seed=seeds[si],all50times_complete=True)),flush=True)
        checks[run.name+'_full_state_unchanged']=all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
        checks[run.name+'_no_parameter_gradients']=all(p.grad is None for p in policy.parameters())
        del policy,state
    checks['all_metrics_finite']=all(np.isfinite(v).all() for v in metrics.values())
    if gap:
        with np.load(RUNS[0]/'frozen_evaluation/rank_component_audit/MATCHED_DENOISING.npz') as old:
            for key,value in metrics.items():
                checks[('old22case_' if protocol.get('replay_previous22') else 'old18case_')+key+'_exact'] = np.array_equal(value[0] if protocol.get('replay_previous22') else value[0][...,common_indices],old[key][-1])
    np.savez_compressed(OUT/'MATCHED_DENOISING.npz',**metrics,times=times,seeds=seeds,phases=np.repeat(phases,2),branches=np.tile([0,1],len(phases)))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    columns=3; rows=(len(phases)+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,figsize=(17,4.4*rows))
    for pi,(phase,ax) in enumerate(zip(phases,axes.flat)):
        for ri,weight in enumerate(protocol['weights']):
            for branch,style in enumerate(('-','--')):
                scope=protocol.get('gradient_scopes',['full_model']*len(RUNS))[ri]
                auxiliary=protocol.get('generated_state_weights',[0.]*len(RUNS))[ri]
                function=protocol.get('rank_functions',['softplus']*len(RUNS))[ri]
                label=f'{"ranked18" if ri==0 else "self18"} branch{branch}' if protocol.get('replay_previous22') else f'{"old14" if ri==0 else "new18"} branch{branch}' if gap else f'{weight} {scope} {function} aux{auxiliary} branch{branch}'
                ax.plot(times,metrics['clean_clipped_mse'][ri,:,:,0,2*pi+branch].mean(0),color=f'C{ri}',linestyle=style,label=label)
        ax.set_title(f'Phase{phase}');ax.set_yscale('log');ax.set_xlabel('Diffusion training time');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    for ax in list(axes.flat)[len(phases):]:ax.set_visible(False)
    fig.suptitle('Full frozen models, same4saved noise seeds, correct prompt, target-corrupted input; NOT generation')
    fig.tight_layout();fig.savefig(OUT/'MATCHED_DENOISING.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks=checks,checks_passed=all(checks.values()),
        full_parameter_count=8319216,model_count=len(RUNS),actual_cases=len(samples),noise_seeds=seeds,
        metric_axes=['model','noise_seed','diffusion_time','condition','case'],plot_inspected=False,
        new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0,scope=protocol['scope']))
    assert all(checks.values())


def main():
    global RUNS, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','frozen'],required=True)
    parser.add_argument('--run',type=Path,default=RUNS[-1])
    args=parser.parse_args()
    run=args.run.resolve()
    if run == BASE/'matched_generator_branch_self_replay01_gaps512':
        RUNS=[BASE/'matched_generator_branch_latent_replay01_gaps512',run]
        OUT=run/'frozen_evaluation/rank_component_audit'
    elif run == BASE/'matched_generator_branch_latent_replay01_gaps512':
        RUNS=[BASE/'matched_generator_branch_latent_replay01512',run]
        OUT=run/'frozen_evaluation/rank_component_audit'
    elif run in (BASE/'matched_generator_branch_encoder_rank025512',BASE/'matched_generator_branch_paired_hinge025512',BASE/'matched_generator_branch_latent_replay01512'):
        RUNS=[BASE/'matched_generator_branch_paired_rank025512',run]
        OUT=run/'frozen_evaluation/rank_component_audit'
    elif run!=RUNS[-1]:
        raise RuntimeError('Use a declared full-model objective comparison')
    prepare() if args.mode=='prepare' else frozen()


if __name__=='__main__':
    main()
