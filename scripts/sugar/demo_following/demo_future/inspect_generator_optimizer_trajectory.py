"""Read actual replayed updates and compare complete official496/512 samplers."""
import argparse
import hashlib
import json
import os
import socket
import dill

import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.probe_generator_reverse_process import (
    BASE,SELF_SOURCE,PAIRED_OUT,PARENT,ActualBranchGeometryDataset,
    GeneratorWrapper,restore_geometry_state,write,
)
from scripts.sugar.demo_following.demo_future.audit_generator_generated_state_gradient import pair_stats

RUN=BASE/'matched_generator_branch_self_replay01_gaps512_optimizer_trace_r1'
TRACE=RUN/'optimizer_trace'
FRESH=PAIRED_OUT/'fresh_full_sampler_gradient'
DELTAS=TRACE/'displacement_readback'
EVAL=TRACE/'frozen496_vs512'


def checkpoints():
    from scripts.sugar.demo_following.demo_future.compare_generator_rank_ablation import equal
    torch.set_num_threads(8)
    assert json.loads((TRACE/'RESULT.json').read_text())['checks_passed']
    names=sorted(p.name for p in (SELF_SOURCE/'demo_geometry/checkpoints').glob('epoch_*.ckpt'))
    checks={};rows=[]
    for name in names:
        paths=[source/'demo_geometry/checkpoints'/name for source in (SELF_SOURCE,RUN)]
        payloads=[]
        for path in paths:
            with path.open('rb') as f:payloads.append(torch.load(f,pickle_module=dill,map_location='cpu'))
        old,new=payloads
        checks[name+'_fullmodel_exact']=equal(old['state_dicts']['model'],new['state_dicts']['model'])
        checks[name+'_fulladam_exact']=equal(old['state_dicts']['optimizer'],new['state_dicts']['optimizer'])
        clocks=sorted(set(int(v['step']) for v in new['state_dicts']['optimizer']['state'].values()))
        epoch=dill.loads(new['pickles']['epoch']);global_step=dill.loads(new['pickles']['global_step'])
        checks[name+'_clock_binding']=len(clocks)==1 and clocks[0]==epoch+1==global_step+1
        rows.append(dict(filename=name,completed_optimizer_updates=clocks[0],saved_epoch=epoch,saved_global_step=global_step,
            original=str(paths[0]),replayed=str(paths[1]),sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))
        assert all(checks.values());print(json.dumps(rows[-1]),flush=True)
        del payloads,old,new
    checks['all8_existing_epochs']=len(rows)==8
    write(TRACE/'EXISTING_CHECKPOINT_BINDING.json',dict(checks_passed=all(checks.values()),checks=checks,checkpoints=rows,
        new_model_forwards=0,new_optimizer_updates=0,scope='Read-only binding of all existing official epoch checkpoints; full original/replay model+Adam exact. Filename epoch is not completed optimizer-update count. No checkpoint selection or new sampling.'))
    assert all(checks.values())


def deltas():
    torch.set_num_threads(8)
    result=json.loads((TRACE/'RESULT.json').read_text());assert result['checks_passed']
    protocol=json.loads((RUN/'REPLAY_PROTOCOL.json').read_text())
    DELTAS.mkdir(exist_ok=False)
    reference=torch.load(FRESH/'MEAN_GRADIENTS.pt',map_location='cpu')
    names=reference['parameter_names'];phases=protocol['next_frozen_comparison']['train_phases']
    labels=[str(p) for p in phases]+['all_train'];checks={};rows=[]
    before=torch.load(TRACE/'after_update_496.pt',map_location='cpu',weights_only=False)['model']
    after=torch.load(TRACE/'after_update_512.pt',map_location='cpu',weights_only=False)['model']
    total=[torch.zeros_like(before[n],dtype=torch.float64) for n in names]
    for step in range(497,513):
        saved=torch.load(TRACE/f'update_{step}_delta.pt',map_location='cpu')
        checks[f'{step}_parameter_names_exact']=saved['parameter_names']==names
        checks[f'{step}_clock_exact']=saved['completed_updates']==step
        for summed,value in zip(total,saved['descent_delta']):summed.add_(value)
        stats={key:pair_stats(names,reference['means'][key][0],saved['descent_delta']) for key in labels}
        rows.append(dict(step=step,applied_learning_rates=saved['applied_learning_rates'],endpoint_gradient_projection=stats))
    checks['all16_FP64_deltas_telescope_exact']=all(torch.equal(v,before[n].double()-after[n].double()) for n,v in zip(names,total))
    saved_total=torch.load(TRACE/'LAST16_DESCENT_DELTA.pt',map_location='cpu')
    checks['saved_total_exact']=saved_total['parameter_names']==names and all(torch.equal(a,b) for a,b in zip(total,saved_total['descent_delta']))
    totals={key:pair_stats(names,reference['means'][key][0],total) for key in labels}
    for key in labels:
        for group in totals[key]:
            checks[f'{key}_{group}_sum_projections_match']=bool(np.isclose(sum(row['endpoint_gradient_projection'][key][group]['dot'] for row in rows),totals[key][group]['dot'],rtol=1e-10,atol=1e-12))
    for filename,digest in protocol['source_hashes'].items():checks['source_unchanged_'+filename]=hashlib.sha256(open(filename,'rb').read()).hexdigest()==digest
    groups=['full_model','geometry_columns','target_encoder','denoiser']
    arrays=dict(descent_dots=np.array([[[row['endpoint_gradient_projection'][key][group]['dot'] for group in groups] for key in labels] for row in rows]),steps=np.arange(497,513),phases=labels,groups=groups)
    np.savez_compressed(DELTAS/'UPDATE_PROJECTIONS.npz',**arrays)
    with np.load(DELTAS/'UPDATE_PROJECTIONS.npz') as a:checks['saved_projection_arrays_exact']=all(np.array_equal(a[k],v) for k,v in arrays.items())
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(17,12))
    for phase,ax in zip(phases,axes.flat):
        ax.plot(range(497,513),[row['endpoint_gradient_projection'][str(phase)]['full_model']['dot'] for row in rows],marker='o');ax.axhline(0,color='gray');ax.set_title(f'TRAIN{phase}');ax.grid(alpha=.25)
    fig.suptitle('Actual replayed parameter displacement projected onto fixed endpoint gradient; not observed error change')
    fig.tight_layout();fig.savefig(DELTAS/'UPDATE_PROJECTIONS.png',dpi=140);plt.close(fig)
    write(DELTAS/'RESULT.json',dict(execution_completed=True,checks_passed=True,checks=checks,rows=rows,total_displacement_statistics=totals,plot_inspected=False,new_optimizer_updates=0,new_gradients=0,new_model_forwards=0,new_physics_steps=0,
        scope='Actual p_before-p_after from exact original497..512 updates. Projections use frozen512 freshTRAIN gradients and are endpoint-anchored first-order diagnostics, not measured loss changes or fresh updates. All original training/data files unchanged.'))
    print(json.dumps(dict(checks=len(checks),total_projection={key:v['full_model']['dot'] for key,v in totals.items()})),flush=True)


def evaluate():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    result=json.loads((DELTAS/'RESULT.json').read_text());assert result['checks_passed'] and result['plot_inspected']
    protocol=json.loads((RUN/'REPLAY_PROTOCOL.json').read_text())['next_frozen_comparison']
    plan=json.loads((SELF_SOURCE/'PROTOCOL.json').read_text());phases=protocol['train_phases'];seeds=protocol['seeds']
    assert len(seeds)==8 and not set(phases).intersection((218,258))
    EVAL.mkdir(exist_ok=False);write(EVAL/'PROTOCOL.json',protocol)
    checks={};rows={};new_paths=0;exact_paths=0;outputs={}
    for update in (496,512):
        snapshot=torch.load(TRACE/f'after_update_{update}.pt',map_location='cpu',weights_only=False);state=snapshot['model']
        policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
        restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
        policy.cuda().eval().requires_grad_(False);policy.num_inference_steps=16
        checks[f'{update}_full8319216']=sum(p.numel() for p in policy.parameters())==8319216
        scheduler=policy.noise_scheduler;config=dict(scheduler.config);alpha=scheduler.alphas_cumprod.cpu().clone()
        for phase in phases:
            ds=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state']);samples=[ds[i] for i in (0,1)]
            obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
            target=policy.normalizer['action'].normalize(torch.stack([s['action'] for s in samples]).cuda())
            with np.load(FRESH/f'phase_{phase}_fresh_paths.npz') as a:
                expected=a['predictions'].copy();checks[f'{update}_{phase}_target_exact']=np.array_equal(target.cpu().numpy().astype(np.float64),a['normalized_actual_target'])
                checks[f'{update}_{phase}_seed_exact']=a['seeds'].tolist()==seeds
            draws=[]
            with torch.inference_mode():
                for si,seed in enumerate(seeds if update==496 else seeds[:2]):
                    pair=[]
                    for branch in (0,1):
                        torch.manual_seed(seed);pair.append(policy.predict_action({k:v[branch:branch+1] for k,v in obs.items()}))
                        if update==496:new_paths+=1
                        else:exact_paths+=1
                    draws.append(torch.cat(pair))
            actual=torch.stack(draws).cpu().numpy()
            if update==512:
                checks[f'{phase}_four512_controls_exact']=np.array_equal(actual,expected[:2]);actual=expected
            predicted=policy.normalizer['action'].normalize(torch.from_numpy(actual).cuda())
            npred=predicted.cpu().numpy();ntarget=target.cpu().numpy()
            squared=(npred-ntarget[None])**2;other=(npred-ntarget[::-1][None])**2
            case=squared.mean((-1,-2));horizon=squared.mean(-1)
            arrays=dict(predictions=actual,normalized_predictions=npred,normalized_target=ntarget,case_mse=case,horizon_mse=horizon,preference=case<other.mean((-1,-2)),seeds=seeds)
            np.savez_compressed(EVAL/f'phase_{phase}_update_{update}.npz',**arrays)
            with np.load(EVAL/f'phase_{phase}_update_{update}.npz') as a:checks[f'{update}_{phase}_saved_arrays_exact']=all(np.array_equal(a[k],v) for k,v in arrays.items())
            checks[f'{update}_{phase}_finite']=all(np.isfinite(v).all() for v in arrays.values())
            row=dict(mean_mse=float(case.mean()),per_seed_mse=case.mean(1).tolist(),branch_preference=arrays['preference'].sum(0).tolist(),horizon_mse=horizon.mean(0).tolist())
            outputs.setdefault(str(phase),{})[str(update)]=arrays
            rows.setdefault(str(phase),{})[str(update)]=row
            write(EVAL/'PARTIAL_RESULT.json',dict(checks=checks,phases=rows));assert all(checks.values())
            print(json.dumps(dict(update=update,phase=phase,mean_mse=row['mean_mse'])),flush=True)
        checks[f'{update}_fullstate_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
        checks[f'{update}_scheduler_unchanged']=config==dict(scheduler.config) and torch.equal(alpha,scheduler.alphas_cumprod.cpu())
        checks[f'{update}_no_gradients']=all(p.grad is None for p in policy.parameters());del policy,state,snapshot
    checks['declared_path_counts']=new_paths==144 and exact_paths==36
    for phase in phases:
        left,right=[outputs[str(phase)][str(update)] for update in (496,512)]
        checks[f'{phase}_both_targets_seeds_exact']=all(np.array_equal(left[k],right[k]) for k in ('normalized_target','seeds'))
        rows[str(phase)]['change']=dict(mean_mse_change=rows[str(phase)]['512']['mean_mse']-rows[str(phase)]['496']['mean_mse'],
            improved_seed_means=int(np.sum(right['case_mse'].mean(1)<left['case_mse'].mean(1))))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(17,12))
    for phase,ax in zip(phases,axes.flat):
        for update,color in [('496','C0'),('512','C1')]:
            for branch,style in enumerate(('-','--')):ax.plot(range(8),rows[str(phase)][update]['horizon_mse'][branch],color=color,ls=style,label=f'{update} b{branch}')
        ax.set_title(f'TRAIN{phase}');ax.set_yscale('log');ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8);fig.suptitle('Actual before/after last16 updates; fixed8TRAINseeds, no checkpoint selection')
    fig.tight_layout();fig.savefig(EVAL/'HORIZON_CHANGE.png',dpi=140);plt.close(fig)
    write(EVAL/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,plot_inspected=False,new496_sample_paths=new_paths,exact512_reference_replays=exact_paths,new_optimizer_updates=0,new_physics_steps=0,
        scope='Complete official models at actual496 and512 from verified identical training trajectory; same8freshTRAINseeds, correctprompt only. Endpoint512 arrays reused after36exactreplays. All draws retained; no checkpointselection, checklabels, primary32replacement or generated physics.'))
    assert all(checks.values())


def saved():
    result=json.loads((EVAL/'RESULT.json').read_text());assert result['checks_passed'] and result['plot_inspected']
    protocol=json.loads((EVAL/'PROTOCOL.json').read_text());checks={}
    for phase in protocol['train_phases']:
        phase_means={}
        for update in (496,512):
            with np.load(EVAL/f'phase_{phase}_update_{update}.npz') as a:
                row=result['phases'][str(phase)][str(update)]
                square=(a['normalized_predictions']-a['normalized_target'][None])**2
                other=(a['normalized_predictions']-a['normalized_target'][::-1][None])**2
                checks[f'{phase}_{update}_case_errors_exact']=np.array_equal(square.mean((-1,-2)),a['case_mse'])
                checks[f'{phase}_{update}_horizon_errors_exact']=np.array_equal(square.mean(-1),a['horizon_mse'])
                checks[f'{phase}_{update}_mean_readback']=float(a['case_mse'].mean())==row['mean_mse']
                checks[f'{phase}_{update}_per_seed_mean_exact']=np.array_equal(a['case_mse'].mean(1),row['per_seed_mse'])
                checks[f'{phase}_{update}_horizon_mean_exact']=np.array_equal(a['horizon_mse'].mean(0),row['horizon_mse'])
                preference=square.mean((-1,-2))<other.mean((-1,-2))
                checks[f'{phase}_{update}_preference_exact']=np.array_equal(preference,a['preference']) and np.array_equal(preference.sum(0),row['branch_preference'])
                checks[f'{phase}_{update}_shape_and_seeds']=a['predictions'].shape==(8,2,8,36) and a['seeds'].tolist()==protocol['seeds']
                with np.load(FRESH/f'phase_{phase}_fresh_paths.npz') as ref:
                    checks[f'{phase}_{update}_reference_target_exact']=np.array_equal(a['normalized_target'].astype(np.float64),ref['normalized_actual_target'])
                    if update==512:checks[f'{phase}_all_saved512physical_outputs_exact']=np.array_equal(a['predictions'],ref['predictions'])
                phase_means[update]=a['case_mse'].mean(1)
        checks[f'{phase}_improvement_count_exact']=int(np.sum(phase_means[512]<phase_means[496]))==result['phases'][str(phase)]['change']['improved_seed_means']
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    write(EVAL/'SAVED_READBACK.json',dict(checks_passed=True,checks=checks,new_model_forwards=0))
    print(json.dumps(dict(checks=len(checks))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['deltas','evaluate','saved','checkpoints'],required=True)
    args=parser.parse_args();{'deltas':deltas,'evaluate':evaluate,'saved':saved,'checkpoints':checkpoints}[args.mode]()


if __name__=='__main__':main()
