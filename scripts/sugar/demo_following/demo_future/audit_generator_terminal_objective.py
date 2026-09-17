"""Qualify one fixed full-official-sampler loss replacement on two frozen students.

This is a TRAIN-only gradient diagnostic, not a new model or training run.
"""
import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn.functional as F

from scripts.sugar.demo_following.demo_future.replicate_generator_training_seed import GROUP,SOURCES,RUNS
from scripts.sugar.demo_following.demo_future.probe_generator_reverse_process import PAIRED_OUT,PARENT,BASE,ActualBranchGeometryDataset,GeneratorWrapper,restore_geometry_state,CONTEXT_KEY,write
from scripts.sugar.demo_following.demo_future.audit_generator_generated_state_gradient import pair_stats
from scripts.sugar.demo_following.demo_future.generator_paired_objective import rng_state
from scripts.sugar.demo_following.demo_future.preflight_generator_latent_replay import same_rng

OUT=GROUP/'full_terminal_objective_qualification'
OBJECTIVE=PAIRED_OUT/'objective_vs_sampler_gradient'
FRESH=PAIRED_OUT/'fresh_full_sampler_gradient'
MODELS={'self84':SOURCES[1],'self400':RUNS[1]}
PHASES=[158,178,197,221,245,261,277,298,318]
SEEDS=list(range(272300,272308))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def prepare():
    for directory in ('frozen_four_endpoint_train_centroid','frozen_four_endpoint_train_representation'):
        result=json.loads((GROUP/directory/'RESULT.json').read_text())
        assert result['checks_passed'] and result['plot_inspected']
        assert json.loads((GROUP/directory/'SAVED_READBACK.json').read_text())['checks_passed']
    comparison=json.loads((GROUP/'COMPARISON.json').read_text())
    assert comparison['relative_improvement_replicated'] and not comparison['both_seeds_prediction_requirements_passed']
    for folder in (OBJECTIVE,FRESH):
        assert json.loads((folder/'RESULT.json').read_text())['checks_passed']
        assert json.loads((folder/'RESULT.json').read_text())['plot_inspected']
        assert json.loads((folder/'SAVED_MEAN_READBACK.json').read_text())['checks_passed']
    centroid=json.loads((GROUP/'frozen_four_endpoint_train_centroid/RESULT.json').read_text())
    assert all(row['variance_eliminated_mean_still_fails'] for row in centroid['phases']['277'].values())
    OUT.mkdir(exist_ok=False)
    paths=[OBJECTIVE/'MEAN_GRADIENTS.pt',FRESH/'MEAN_GRADIENTS.pt',GROUP/'COMPARISON.json',
           GROUP/'frozen_four_endpoint_train_centroid/RESULT.json',GROUP/'frozen_four_endpoint_train_representation/RESULT.json']
    paths += [p/'demo_geometry/checkpoints/endpoint.ckpt' for p in MODELS.values()]
    paths += [BASE/'generator_train_diffusion_replay8_current_gaps/TRAIN_GENERATED_STATE_INPUTS.npz',Path(json.loads((OBJECTIVE/'PROTOCOL.json').read_text())['point_reference'])]
    write(OUT/'PROTOCOL.json',dict(models={k:str(v) for k,v in MODELS.items()},train_phases=PHASES,sampling_seeds=SEEDS,
        candidate='q epsilon MSE +0.25 paired scaled softplus rank +0.1 full official16step terminal normalized-command MSE; replaces frozen-state epsilon auxiliary, not an added fourth term.',
        rationale='Persistent277 mean bias above threshold in allfour endpoints;277/298 contrast improves across studentseeds but remains insufficient. Full initial-to-terminal differentiation directly targets the existing prediction metric. Token-distance correlations do not establish causality, and the earlier0/3 raw-conflict replication remains negative. This is a separate prospective compatibility test, not a reinterpretation of that failure.',
        coefficient=.1,coefficient_scope='One fixed inherited numeric coefficient; epsilon and terminal losses have different units and gradient scales. No coefficient search or claim of equal influence.',
        gradient_scope='Frozen learned endpoints, FP32 eval, original full8319216 modules and16step scheduler. Q/replay use existing4seeds x16times x18realTRAIN; not all50q times, dropout, actualAdam or BF16 training. Terminal uses existing eight diagnostic seeds, not fresh relative to previous analyses. Only new student paths are newly computed. No check218/258 labels.',
        reuse='Reuse oldself84 audited full component and terminal gradients; verify36primary and36fresh exact saved path controls. Newself400 computes64component rows,144new no-grad paths and144exact full-gradient replays plus36primary controls.',
        expected_new_paths=144,expected_gradient_replays=144,expected_control_paths=108,expected_official_sampler_forwards=6336,expected_component_forwards=192,
        machine_criteria=['Both students: candidate mean direction descends original q+rank base, existing q+rank+replay objective, and terminal TRAIN mean (dot>1e-8).',
                          'Both students: at least6of8 seed-mean terminal gradients have positive candidate descent dot>1e-8.',
                          'Both students: at least7of9 TRAIN phase-mean terminal gradients have positive candidate descent dot>1e-8, including277.'],
        automatic_next_action='If integrity, nineplots, independent fullgradient readback and allfixed criteria pass, prepare a distinct matched full-model terminal-supervision experiment with release initialization and both arms; first require actual full144row CPU/H200BF16 forward/gradient/RNG/Adam compatibility. No training budget is launched by this diagnostic. If criteria fail, reject this fixed replacement and inspect recorded causal state/selected-demo information coverage; do not adjust coefficient, add seed, swap teacher, change solver or extend an old512 endpoint.',
        source_hashes={str(p):digest(p) for p in paths},new_optimizer_updates=0,new_physics_steps=0))
    print(json.dumps(dict(prepared=str(OUT))),flush=True)


def old_gradients():
    objective=torch.load(OBJECTIVE/'MEAN_GRADIENTS.pt',map_location='cpu')
    terminal=torch.load(FRESH/'MEAN_GRADIENTS.pt',map_location='cpu')
    assert objective['parameter_names']==terminal['parameter_names']
    return objective['parameter_names'],objective['means']['all_noise'],{k:v[0] for k,v in terminal['means'].items()}


def compare(names,components,terminal):
    base=[a+b for a,b in zip(*components[:2])]
    original=[a+b+c for a,b,c in zip(*components)]
    candidate=[a+.1*b for a,b in zip(base,terminal['all_train'])]
    result={'base':pair_stats(names,candidate,base),'original':pair_stats(names,candidate,original),
            'terminal':{k:{'candidate':pair_stats(names,candidate,g),'original':pair_stats(names,original,g)} for k,g in terminal.items()}}
    positive_phases=[p for p in PHASES if result['terminal'][str(p)]['candidate']['full_model']['dot']>1e-8]
    positive_seeds=[s for s in SEEDS if result['terminal']['seed_'+str(s)]['candidate']['full_model']['dot']>1e-8]
    result['criteria']={'base_descent':result['base']['full_model']['dot']>1e-8,
        'original_objective_descent':result['original']['full_model']['dot']>1e-8,
        'terminal_mean_descent':result['terminal']['all_train']['candidate']['full_model']['dot']>1e-8,
        'at_least6of8_seed_descent':len(positive_seeds)>=6,'at_least7of9_phase_descent':len(positive_phases)>=7,'phase277_descent':277 in positive_phases}
    result['positive_phases']=positive_phases;result['positive_seeds']=positive_seeds
    return result


def run():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    protocol=json.loads((OUT/'PROTOCOL.json').read_text());assert not (OUT/'RESULT.json').exists()
    assert all(digest(p)==v for p,v in protocol['source_hashes'].items())
    names_old,old_components,old_terminal=old_gradients()
    comparisons={};checks={};new_paths=0;replays=0;controls=0;component_rows=[]
    replay_source=BASE/'generator_train_diffusion_replay8_current_gaps'
    with np.load(replay_source/'TRAIN_GENERATED_STATE_INPUTS.npz') as a:saved={k:a[k].copy() for k in a.files}
    point_path=Path(json.loads((OBJECTIVE/'PROTOCOL.json').read_text())['point_reference'])
    with np.load(point_path) as a:oracle=a['oracle'][:4].copy()
    for label,source in MODELS.items():
        plan=json.loads((source/'PROTOCOL.json').read_text());assert plan['data']['train']['phases']==PHASES
        dataset=ActualBranchGeometryDataset(**plan['branch_dataset']);assert dataset.real_case_count==18
        samples=[dataset[i] for i in range(18)]
        obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
        with (source/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
        policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
        restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic');policy.cuda().eval();policy.normalizer.requires_grad_(False)
        names,params=zip(*[(n,p) for n,p in policy.named_parameters() if p.requires_grad])
        checks[label+'_full8319216_and_named_parameters']=sum(p.numel() for p in policy.parameters())==8319216 and names==names_old
        target=policy.normalizer['action'].normalize(torch.stack([s['action'] for s in samples]).cuda())
        checks[label+'_actual_targets_exact']=np.array_equal(target.cpu().numpy().astype(np.float64),saved['normalized_actual_target'])
        checks[label+'_real_phase_binding']=saved['phases'].tolist()==[p for p in PHASES for _ in (0,1)]
        scheduler=policy.noise_scheduler;step=scheduler.step;config=dict(scheduler.config);alpha=scheduler.alphas_cumprod.cpu().clone()
        terminal={};all_mean=[torch.zeros_like(p) for p in params]
        seed_means={s:[torch.zeros_like(p) for p in params] for s in SEEDS} if label=='self400' else {}
        for pi,phase in enumerate(PHASES):
            with np.load(source/f'frozen_evaluation/phase_{phase}/trained_demo_geometry_correct.npz') as a:primary=a['predictions'][:2].copy()
            with torch.no_grad():
                for si,seed in enumerate(plan['frozen_evaluation']['sample_seeds'][:2]):
                    for branch in (0,1):
                        torch.manual_seed(seed);result=policy.predict_action({k:v[2*pi+branch:2*pi+branch+1] for k,v in obs.items()})
                        checks[f'{label}_{phase}_{seed}_{branch}_primary_exact']=np.array_equal(result.cpu().numpy()[0],primary[si,branch]);controls+=1
            if label=='self84':
                with np.load(FRESH/f'phase_{phase}_fresh_paths.npz') as a:reference={k:a[k].copy() for k in a.files}
            records_by_seed={k:[] for k in ('xt','epsilon','pred_original','prev_sample','predictions')}
            phase_mean=[torch.zeros_like(p) for p in params]
            for si,seed in enumerate(SEEDS[:2] if label=='self84' else SEEDS):
                branch_records={k:[] for k in records_by_seed}
                for branch in (0,1):
                    subobs={k:v[2*pi+branch:2*pi+branch+1] for k,v in obs.items()}
                    def sample(grad):
                        recorded=[]
                        def record(epsilon,time,xt,*args,**kwargs):
                            item=step(epsilon,time,xt,*args,**kwargs)
                            recorded.append((int(time),xt.detach().cpu().clone(),epsilon.detach().cpu().clone(),item.pred_original_sample.detach().cpu().clone(),item.prev_sample.detach().cpu().clone(),item.prev_sample))
                            return item
                        scheduler.step=record
                        try:
                            torch.manual_seed(seed)
                            with torch.set_grad_enabled(grad):action=policy.predict_action(subobs)
                        finally:scheduler.step=step
                        values={k:torch.cat([r[i+1] for r in recorded]).numpy() for i,k in enumerate(('xt','epsilon','pred_original','prev_sample'))}
                        values['predictions']=action.detach().cpu().numpy()[0]
                        assert [r[0] for r in recorded]==list(range(45,-1,-3))
                        return values,recorded[-1][-1]
                    expected,_=sample(False)
                    if label=='self84':
                        for k,v in expected.items():
                            ref=reference[k][si,branch] if k=='predictions' else reference[k][si,0,branch]
                            checks[f'{label}_{phase}_{seed}_{branch}_{k}_old_exact']=np.array_equal(v,ref)
                        controls+=1
                    else:
                        actual,final=sample(True);new_paths+=1;replays+=1
                        for k,v in expected.items():checks[f'{label}_{phase}_{seed}_{branch}_{k}_grad_replay_exact']=np.array_equal(v,actual[k])
                        before=rng_state(torch.device('cuda'));loss=(final-target[2*pi+branch:2*pi+branch+1]).square().mean()
                        gradients=torch.autograd.grad(loss,params,allow_unused=True)
                        gradients=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gradients)]
                        checks[f'{label}_{phase}_{seed}_{branch}_backward_rng_finite']=same_rng(before,rng_state(torch.device('cuda'))) and all(bool(torch.isfinite(g).all()) for g in gradients)
                        for a,b,c,g in zip(phase_mean,all_mean,seed_means[seed],gradients):a.add_(g/16);b.add_(g/144);c.add_(g/18)
                        for k,v in expected.items():branch_records[k].append(v)
                        del gradients,loss,final
                    assert all(checks.values()),[k for k,v in checks.items() if not v]
                if label=='self400':
                    for k in records_by_seed:records_by_seed[k].append(np.stack(branch_records[k]))
            if label=='self400':
                arrays={k:np.stack(v) for k,v in records_by_seed.items()};arrays.update(seeds=np.array(SEEDS),times=np.array(list(range(45,-1,-3))),target=target[2*pi:2*pi+2].cpu().numpy())
                np.savez_compressed(OUT/f'phase_{phase}_paths.npz',**arrays)
                with np.load(OUT/f'phase_{phase}_paths.npz') as a:checks[f'{phase}_all_saved_paths_exact']=all(np.array_equal(a[k],v) for k,v in arrays.items())
                terminal[str(phase)]=[p.cpu() for p in phase_mean]
            write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,new_paths=new_paths,gradient_replays=replays,control_paths=controls,current_model=label,current_phase=phase))
            print(json.dumps(dict(model=label,phase=phase,new_paths=new_paths,replays=replays,controls=controls)),flush=True)
        if label=='self84':components=old_components;terminal=old_terminal
        else:
            terminal['all_train']=[p.cpu() for p in all_mean]
            for seed,values in seed_means.items():terminal['seed_'+str(seed)]=[p.cpu() for p in values]
            del all_mean,seed_means
            normalized=policy.normalizer.normalize(obs);wrong=dict(normalized);permutation=torch.arange(18,device='cuda')^1;wrong[CONTEXT_KEY]=normalized[CONTEXT_KEY][permutation]
            xt=torch.from_numpy(saved['xt'][:4]).cuda();noise=torch.from_numpy(saved['initial_gaussian'][:4]).cuda();real_oracle=torch.from_numpy(oracle).cuda()
            aggregate=[[torch.zeros_like(p) for p in params] for _ in range(3)]
            before=rng_state(torch.device('cuda'))
            for si,seed in enumerate(saved['seeds'][:4].tolist()):
                for ti,time in enumerate(saved['times'].tolist()):
                    t=torch.full((18,),time,device='cuda',dtype=torch.long);q=scheduler.add_noise(target,noise[si],t)
                    cond=policy.obs_encoder(normalized,training=False);other=policy.obs_encoder(wrong,training=False)
                    pred=policy.model(q,t,cond=cond,training=False,gen_attn_map=False)[0]
                    wrong_pred=policy.model(q,t,cond=other,training=False,gen_attn_map=False)[0]
                    point=policy.model(xt[si,ti],t,cond=cond,training=False,gen_attn_map=False)[0]
                    base=(pred-noise[si]).square().mean((1,2));wrong_error=(wrong_pred-noise[si]).square().mean((1,2))
                    scale=(alpha[time]/(1-alpha[time])*(target-target[permutation]).square().mean((1,2))).detach();assert bool((scale>0).all())
                    rank=scale*F.softplus(.1+(base-wrong_error)/scale);replay=(point-real_oracle[si,ti]).square().mean((1,2))
                    gradients=[]
                    for i,loss in enumerate((base.mean(),.25*rank.mean(),.1*replay.mean(),(base+.25*rank+.1*replay).mean())):
                        values=torch.autograd.grad(loss,params,retain_graph=i<3,allow_unused=True)
                        gradients.append([torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,values)])
                    checks[f'{seed}_{time}_full_component_sum_finite']=all(torch.allclose(a+b+c,d,rtol=3e-5,atol=1e-6) for a,b,c,d in zip(*gradients)) and all(bool(torch.isfinite(g).all()) for group in gradients for g in group)
                    for group in range(3):
                        for a,g in zip(aggregate[group],gradients[group]):a.add_(g/64)
                    component_rows.append(dict(seed=seed,time=time,base=float(base.mean().detach()),rank=float(rank.mean().detach()),replay=float(replay.mean().detach())))
                    assert all(checks.values()),[k for k,v in checks.items() if not v]
                    del gradients
                print(json.dumps(dict(model=label,component_seed=seed,component_rows=len(component_rows))),flush=True)
            checks['component_rng_unchanged']=same_rng(before,rng_state(torch.device('cuda')))
            components=[[v.cpu() for v in group] for group in aggregate]
            torch.save(dict(parameter_names=names,components=components,terminal=terminal),OUT/'NEW_STUDENT_GRADIENTS.pt')
        checks[label+'_fullstate_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
        checks[label+'_scheduler_unchanged']=config==dict(scheduler.config) and torch.equal(alpha,scheduler.alphas_cumprod.cpu()) and scheduler.step==step
        checks[label+'_no_accumulated_gradients']=all(p.grad is None for p in policy.parameters())
        comparisons[label]=compare(names,components,terminal)
        del policy,state
    checks['exact_path_and_gradient_budgets']=(new_paths,replays,controls,len(component_rows))==(144,144,108,64)
    checks['all_source_hashes_unchanged']=all(digest(p)==v for p,v in protocol['source_hashes'].items())
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(17,12))
    for phase,ax in zip(PHASES,axes.flat):
        values=[comparisons[label]['terminal'][str(phase)][kind]['full_model']['cosine'] for label in MODELS for kind in ('original','candidate')]
        ax.bar(range(4),values);ax.set_xticks(range(4),['old84','terminal84','old400','terminal400']);ax.axhline(0,color='gray');ax.set_ylim(-1.05,1.05);ax.set_title(f'TRAIN{phase}');ax.grid(axis='y',alpha=.25)
    fig.suptitle('Fixed0.1 terminal-loss replacement: raw direction cosine to terminal gradient; no updates')
    fig.tight_layout();fig.savefig(OUT/'TERMINAL_COMPATIBILITY.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,comparisons=comparisons,component_rows=component_rows,plot_inspected=False,
        new_sample_paths=new_paths,exact_gradient_replays=replays,control_paths=controls,official_sampler_forwards=16*(new_paths+replays+controls),component_forwards=192,new_optimizer_updates=0,new_physics_steps=0))
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    print(json.dumps(dict(checks=len(checks),criteria={k:v['criteria'] for k,v in comparisons.items()})),flush=True)


def readback():
    torch.set_num_threads(8)
    result=json.loads((OUT/'RESULT.json').read_text());assert result['checks_passed'] and result['plot_inspected']
    assert not (OUT/'SAVED_READBACK.json').exists()
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    names,components,terminal=old_gradients();measured={'self84':compare(names,components,terminal)}
    saved=torch.load(OUT/'NEW_STUDENT_GRADIENTS.pt',map_location='cpu');assert names==saved['parameter_names']
    measured['self400']=compare(names,saved['components'],saved['terminal'])
    checks={'all_direction_metrics_and_criteria_exact':measured==result['comparisons'],
            'source_hashes_unchanged':all(digest(p)==v for p,v in protocol['source_hashes'].items())}
    for phase in PHASES:
        with np.load(OUT/f'phase_{phase}_paths.npz') as a:
            checks[f'{phase}_complete_saved_paths']=all(a[k].shape==(8,2,16,8,36) and np.isfinite(a[k]).all() for k in ('xt','epsilon','pred_original','prev_sample')) and a['seeds'].tolist()==SEEDS
            checks[f'{phase}_terminal_loss_finite']=bool(np.isfinite(((a['prev_sample'][:,:,-1]-a['target'][None])**2).mean()))
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    write(OUT/'SAVED_READBACK.json',dict(checks_passed=True,checks=checks,new_optimizer_updates=0,new_physics_steps=0))
    passed=all(all(v['criteria'].values()) for v in measured.values())
    write(OUT/'DECISION.json',dict(checks_passed=True,permits_preparing_matched_experiment=passed,fixed_terminal_weight=.1,
        per_student_criteria={k:v['criteria'] for k,v in measured.items()},new_training_launched=False,
        next_action='prepare_full_model_terminal_replacement_protocol_and_full144row_CPU_H200BF16_preflight' if passed else 'reject_fixed_terminal_replacement_and_audit_recorded_causal_state_information',
        interpretation='Local FP32 TRAIN compatibility only; no actualAdam/BF16 or prediction improvement claim. Original full-prediction and nonregression criteria remain unchanged; no old512 extension, coefficient search or thirdseed.'))
    print(json.dumps(dict(checks=len(checks),prepare_matched_experiment=passed)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['prepare','run','readback'],required=True)
    {'prepare':prepare,'run':run,'readback':readback}[parser.parse_args().mode]()


if __name__=='__main__':main()
