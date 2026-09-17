"""Full official saved-checkpoint learning curve; no training or checkpoint selection."""
import argparse
import hashlib
import json
import os
import socket

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.inspect_generator_optimizer_trajectory import TRACE,EVAL
from scripts.sugar.demo_following.demo_future.probe_generator_reverse_process import SELF_SOURCE,PARENT,ActualBranchGeometryDataset,GeneratorWrapper,restore_geometry_state,write
from scripts.sugar.demo_following.demo_future.compare_generator_rank_ablation import equal

OUT=TRACE/'existing_checkpoint_learning_curve32'


def prepare():
    torch.set_num_threads(8)
    for name in ('RESULT.json','SAVED_READBACK.json'):
        result=json.loads((EVAL/name).read_text());assert result['checks_passed']
    assert json.loads((EVAL/'RESULT.json').read_text())['plot_inspected']
    binding=json.loads((TRACE/'EXISTING_CHECKPOINT_BINDING.json').read_text());assert binding['checks_passed']
    plan=json.loads((SELF_SOURCE/'PROTOCOL.json').read_text());checkpoints=binding['checkpoints']
    assert [r['completed_optimizer_updates'] for r in checkpoints]==[1,65,129,193,257,321,385,449]
    with open(checkpoints[0]['original'],'rb') as f:first=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
    initial=torch.load(SELF_SOURCE/'demo_geometry/INITIAL_MODEL.pt',map_location='cpu')['model']
    first_initial=equal(first,initial)
    entries=[dict(update=row['completed_optimizer_updates'],checkpoint=row['original'],reference_variant='initial_shared_goal_zero' if i==0 and first_initial else None) for i,row in enumerate(checkpoints)]
    entries.append(dict(update=512,checkpoint=str(SELF_SOURCE/'demo_geometry/checkpoints/endpoint.ckpt'),reference_variant='trained_demo_geometry_correct'))
    phases=plan['data']['train']['phases'];seeds=plan['frozen_evaluation']['sample_seeds']
    fresh=sum(e['reference_variant'] is None for e in entries)*len(phases)*len(seeds)*2
    replays=sum(e['reference_variant'] is not None for e in entries)*len(phases)*2*2
    OUT.mkdir(exist_ok=False)
    hashes={e['checkpoint']:hashlib.sha256(open(e['checkpoint'],'rb').read()).hexdigest() for e in entries}
    write(OUT/'PROTOCOL.json',dict(source_run=str(SELF_SOURCE),entries=entries,train_phases=phases,seeds=seeds,inference_steps=16,first_update_model_equals_initial=first_initial,
        new_paths=fresh,exact_reference_replays=replays,official_denoiser_forwards=16*(fresh+replays),checkpoint_hashes=hashes,new_optimizer_updates=0,new_physics_steps=0,
        scope='All8existing official epoch checkpoints at actualupdates1/65/129/193/257/321/385/449 plus complete512 endpoint. Full8319216 models, all9TRAIN, same32primary seeds, correctprompt only. Reuse initial/512 fullprimary arrays only after fullstate binding and first2seed/branch exactcontrols. No selected checkpoints, new seeds, primary endpoint reruns, checklabels or physics. Correctbranch preference and target-mean accuracy alone do not establish the fullfive-condition criterion.',
        automatic_next_action='Inspect all9 learningcurves and9 earliest-vs-final horizon panels plus savedreadback. If277 accuracy never passes across all recorded states, predeclare one independent matched training-seed replication of unchanged ranked18 versus self18 variants, reporting all outcomes without best-seed selection. If277 passes then regresses, inspect full-model objective/denoising differences across the adjacent saved states bounding that regression. No earlycheckpointdeployment, old513, weight/teacher/solver sweep or SMPbenefit claim.'))
    print(json.dumps(dict(first_initial_exact=first_initial,new_paths=fresh,exact_replays=replays)),flush=True)


def run():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    protocol=json.loads((OUT/'PROTOCOL.json').read_text());plan=json.loads((SELF_SOURCE/'PROTOCOL.json').read_text())
    assert not (OUT/'PARTIAL_RESULT.json').exists()
    checks={};rows={};new_paths=0;replays=0
    for entry in protocol['entries']:
        update=entry['update'];folder=OUT/f'update_{update:04d}';folder.mkdir()
        with open(entry['checkpoint'],'rb') as f:payload=torch.load(f,pickle_module=dill,map_location='cpu')
        state=payload['state_dicts']['model'];clocks=set(int(v['step']) for v in payload['state_dicts']['optimizer']['state'].values())
        checks[f'{update}_actual_adam_clock']=clocks=={update}
        policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
        restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
        policy.cuda().eval().requires_grad_(False);policy.num_inference_steps=16
        checks[f'{update}_full8319216']=sum(p.numel() for p in policy.parameters())==8319216
        scheduler=policy.noise_scheduler;config=dict(scheduler.config);alpha=scheduler.alphas_cumprod.cpu().clone()
        for phase in protocol['train_phases']:
            ds=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state']);samples=[ds[i] for i in (0,1)]
            obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']};target=torch.stack([s['action'] for s in samples]).cuda()
            expected=None;variant=entry['reference_variant'];seeds=protocol['seeds'][:2] if variant else protocol['seeds']
            if variant:
                with np.load(SELF_SOURCE/f'frozen_evaluation/phase_{phase}/{variant}.npz') as a:
                    expected=a['predictions'].copy();checks[f'{update}_{phase}_reference_target_seed_exact']=np.array_equal(a['actual_future_command_target'],target.cpu().numpy()) and a['sample_seeds'].tolist()==protocol['seeds']
            draws=[]
            with torch.inference_mode():
                for seed in seeds:
                    pair=[]
                    for branch in (0,1):
                        torch.manual_seed(seed);pair.append(policy.predict_action({k:v[branch:branch+1] for k,v in obs.items()}))
                        if variant:replays+=1
                        else:new_paths+=1
                    draws.append(torch.cat(pair))
                predicted=torch.stack(draws)
                if variant:
                    checks[f'{update}_{phase}_exact_reference_controls']=np.array_equal(predicted.cpu().numpy(),expected[:2]);predicted=torch.from_numpy(expected).cuda()
                normalized=policy.normalizer['action'].normalize(predicted);ntarget=policy.normalizer['action'].normalize(target)
                error=(normalized-ntarget[None]).square();other=(normalized-ntarget.flip(0)[None]).square()
                preference=error.mean((2,3))<other.mean((2,3));baseline=(ntarget-ntarget.mean(0)).square()
            metrics=dict(normalized_mse=float(error.mean()),per_branch_mse=error.mean((0,2,3)).cpu().tolist(),per_branch_correct_preference_draws=preference.sum(0).cpu().tolist(),normalized_mse_by_branch_horizon_dimension=error.mean(0).cpu().tolist())
            if variant:
                primary=json.loads((SELF_SOURCE/f'frozen_evaluation/phase_{phase}/RESULT.json').read_text())['variants'][variant]
                checks[f'{update}_{phase}_all_original_primary_metrics_exact']=all(metrics[k]==primary[k] for k in metrics)
            arrays=dict(predictions=predicted.cpu().numpy(),normalized_predictions=normalized.cpu().numpy(),normalized_target=ntarget.cpu().numpy(),error=error.cpu().numpy(),preference=preference.cpu().numpy(),seeds=protocol['seeds'])
            np.savez_compressed(folder/f'phase_{phase}.npz',**arrays)
            with np.load(folder/f'phase_{phase}.npz') as a:checks[f'{update}_{phase}_saved_arrays_exact']=all(np.array_equal(a[k],v) for k,v in arrays.items())
            checks[f'{update}_{phase}_finite']=all(np.isfinite(v).all() for v in arrays.values())
            rows.setdefault(str(phase),{})[str(update)]=dict(metrics=metrics,two_target_mean_mse=float(baseline.mean()),mean_ratio=metrics['normalized_mse']/float(baseline.mean()),
                accuracy_only_passed=metrics['normalized_mse']<=.5*float(baseline.mean()),correct_branch_only_passed=min(metrics['per_branch_correct_preference_draws'])>=28)
            write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,phases=rows,new_paths=new_paths,replays=replays));assert all(checks.values()),[k for k,v in checks.items() if not v]
            print(json.dumps(dict(update=update,phase=phase,mse=metrics['normalized_mse'],branch_draws=metrics['per_branch_correct_preference_draws'])),flush=True)
        checks[f'{update}_fullstate_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
        checks[f'{update}_scheduler_unchanged']=config==dict(scheduler.config) and torch.equal(alpha,scheduler.alphas_cumprod.cpu())
        checks[f'{update}_no_gradients']=all(p.grad is None for p in policy.parameters())
        checks[f'{update}_checkpoint_hash_unchanged']=hashlib.sha256(open(entry['checkpoint'],'rb').read()).hexdigest()==protocol['checkpoint_hashes'][entry['checkpoint']]
        del policy,payload,state
    checks['exact_declared_budgets']=new_paths==protocol['new_paths'] and replays==protocol['exact_reference_replays']
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    updates=[e['update'] for e in protocol['entries']]
    fig,axes=plt.subplots(3,3,figsize=(17,12))
    for phase,ax in zip(protocol['train_phases'],axes.flat):
        for branch,style in enumerate(('-','--')):ax.plot(updates,[rows[str(phase)][str(u)]['metrics']['per_branch_mse'][branch] for u in updates],style,marker='o',label=f'branch{branch}')
        ax.axhline(.5*rows[str(phase)]['512']['two_target_mean_mse'],ls=':',color='gray',label='accuracy threshold only');ax.set_yscale('log');ax.set_title(f'TRAIN{phase}');ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=7);fig.suptitle('All existing saved states, original32seeds; correctprompt only, no checkpoint selection')
    fig.tight_layout();fig.savefig(OUT/'LEARNING_CURVE.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(17,12))
    for phase,ax in zip(protocol['train_phases'],axes.flat):
        for update,color in ((updates[0],'tab:orange'),(512,'tab:blue')):
            horizon=np.asarray(rows[str(phase)][str(update)]['metrics']['normalized_mse_by_branch_horizon_dimension']).mean(-1)
            for branch,style in enumerate(('-','--')):
                ax.plot(range(1,9),horizon[branch],style,color=color,label=f'update{update} branch{branch}')
        ax.set_yscale('log');ax.set_title(f'TRAIN{phase}');ax.set_xlabel('future horizon');ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=7);fig.suptitle('Earliest saved state versus512: all32seeds, normalized error by horizon')
    fig.tight_layout();fig.savefig(OUT/'EARLIEST_VS_FINAL_HORIZONS.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,plot_inspected=False,new_paths=new_paths,exact_reference_replays=replays,new_optimizer_updates=0,new_physics_steps=0,scope=protocol['scope']))
    assert all(checks.values())


def readback():
    result=json.loads((OUT/'RESULT.json').read_text());assert result['checks_passed'] and result['plot_inspected']
    protocol=json.loads((OUT/'PROTOCOL.json').read_text());checks={}
    for entry in protocol['entries']:
        update=entry['update']
        for phase in protocol['train_phases']:
            with np.load(OUT/f'update_{update:04d}/phase_{phase}.npz') as a:
                error=(a['normalized_predictions']-a['normalized_target'][None])**2
                other=(a['normalized_predictions']-a['normalized_target'][::-1][None])**2
                checks[f'{update}_{phase}_error_exact']=np.array_equal(error,a['error'])
                checks[f'{update}_{phase}_preference_exact']=np.array_equal(error.mean((2,3))<other.mean((2,3)),a['preference'])
                checks[f'{update}_{phase}_mean_readback']=bool(np.isclose(error.mean(),result['phases'][str(phase)][str(update)]['metrics']['normalized_mse'],rtol=1e-5,atol=1e-12))
                checks[f'{update}_{phase}_all32_seeds']=a['seeds'].tolist()==protocol['seeds'] and a['predictions'].shape==(32,2,8,36)
                row=result['phases'][str(phase)][str(update)];metrics=row['metrics']
                checks[f'{update}_{phase}_branch_horizon_readback']=bool(np.allclose(error.mean(0),metrics['normalized_mse_by_branch_horizon_dimension'],rtol=1e-5,atol=1e-12))
                checks[f'{update}_{phase}_branch_mse_readback']=bool(np.allclose(error.mean((0,2,3)),metrics['per_branch_mse'],rtol=1e-5,atol=1e-12))
                checks[f'{update}_{phase}_branch_counts_exact']=a['preference'].sum(0).tolist()==metrics['per_branch_correct_preference_draws']
                baseline=float(((a['normalized_target']-a['normalized_target'].mean(0))**2).mean())
                checks[f'{update}_{phase}_baseline_readback']=bool(np.isclose(baseline,row['two_target_mean_mse'],rtol=1e-5,atol=1e-12))
                checks[f'{update}_{phase}_decision_readback']=(row['accuracy_only_passed']==(metrics['normalized_mse']<=.5*row['two_target_mean_mse']) and row['correct_branch_only_passed']==(min(metrics['per_branch_correct_preference_draws'])>=28) and row['mean_ratio']==metrics['normalized_mse']/row['two_target_mean_mse'])
        checks[f'{update}_source_hash_unchanged']=hashlib.sha256(open(entry['checkpoint'],'rb').read()).hexdigest()==protocol['checkpoint_hashes'][entry['checkpoint']]
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    write(OUT/'SAVED_READBACK.json',dict(checks_passed=True,checks=checks,new_model_forwards=0))
    updates=[e['update'] for e in protocol['entries']]
    passed=[u for u in updates if result['phases']['277'][str(u)]['accuracy_only_passed']]
    regressions=[[a,b] for a,b in zip(updates,updates[1:]) if result['phases']['277'][str(a)]['accuracy_only_passed'] and not result['phases']['277'][str(b)]['accuracy_only_passed']]
    assert not result['phases']['277']['512']['accuracy_only_passed']
    assert bool(passed)==bool(regressions)
    write(OUT/'DECISION.json',dict(checks_passed=True,phase277_accuracy_passing_updates=passed,all_adjacent_accuracy_regressions=regressions,
        selected_next_action='independent_matched_training_seed_replication' if not passed else 'full_model_objective_and_denoising_at_adjacent_regression_states',
        scope='Apply the predeclared branch to all saved states; correctprompt accuracy only, no fullfive-condition pass or selected deployment checkpoint. All old primary outcomes and budgets remain unchanged.',
        no_new_training_admitted_by_this_readback=True))
    print(json.dumps(dict(checks=len(checks))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['prepare','run','readback'],required=True)
    args=parser.parse_args();{'prepare':prepare,'run':run,'readback':readback}[args.mode]()


if __name__=='__main__':main()
