"""Exact saved-draw common/branch-contrast error decomposition; no forwards."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import BASE, write

RUN=BASE/'matched_generator_branch_self_replay01_gaps512'
OUT=RUN/'frozen_evaluation/centroid_error_readback'
MODELS={'fit14':BASE/'matched_generator_branch_latent_replay01512',
        'ranked18':BASE/'matched_generator_branch_latent_replay01_gaps512','self18':RUN}


def main():
    global OUT,MODELS
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replication',action='store_true')
    replication=parser.parse_args().replication
    torch.set_num_threads(1)
    if replication:
        from scripts.sugar.demo_following.demo_future.replicate_generator_training_seed import GROUP,RUNS,SOURCES
        comparison=json.loads((GROUP/'COMPARISON.json').read_text())
        assert comparison['checks_passed'] and not comparison['both_seeds_prediction_requirements_passed']
        MODELS=dict(ranked18=SOURCES[0],self18=SOURCES[1],ranked18_seed272400=RUNS[0],self18_seed272400=RUNS[1])
        OUT=GROUP/'frozen_four_endpoint_train_centroid'
    else:
        comparison=json.loads((RUN/'frozen_evaluation/LATENT_REPLAY_COMPARISON.json').read_text())
        assert comparison['checks_passed'] and not comparison['decision']['all_prediction_requirements_passed']
    OUT.mkdir(exist_ok=False)
    write(OUT/'PROTOCOL.json',dict(models={k:str(v) for k,v in MODELS.items()},draws=32,
        conditions=['correct','wrong','zero_context','demo_zero','initial'],new_model_forwards=0,new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0,
        scope=('Allfour fixed-teacher/student-seed endpoints,9TRAIN only/5conditions/32saved draws. ' if replication else 'All existing phases/5conditions/32saved draws; old14 has no221/261 primary data. ')+'Exact shared frozen action normalization and actualtargets/seeds. Decompose correct-condition squared error into shared pair-centroid bias, branch-contrast bias and draw variance. Means/midpoints are diagnostic quantities, never substituted outputs, physical labels or new primary metrics. No threshold or normalization change.',
        automatic_next_action=('Inspect all9TRAIN plots and independent savedreadback, then inspect allfour frozen fullmodel condition encoders on the exact18realTRAIN inputs. ' if replication else 'Inspect all11plots, saved allvariant metric replays, exact decomposition identities and source hashes. ')+'If277 meanbias alone exceeds its unchanged threshold, do not attempt a variance-only or seed-averaging rescue. Use common versus contrast and horizon/channel attribution to choose the next bounded full-model conditioning or denoising diagnostic; no new teacher/weight sweep or training budget from this readback alone.'))
    checks={};rows={};arrays={};hashes={};references={};scale=None
    for name,run in MODELS.items():
        plan=json.loads((run/'PROTOCOL.json').read_text())
        endpoint=json.loads((run/'RESULT.json').read_text())
        saved_metrics=json.loads((run/'frozen_evaluation/SAVED_PHASE_READBACK.json').read_text())
        assert endpoint['execution_completed'] and endpoint['horizon_plots_inspected'] and saved_metrics['saved_metrics_reproduced']
        state=torch.load(plan['normalizer_state'],map_location='cpu')['model']
        current_scale=state['normalizer.params_dict.action.scale'].double().numpy()
        if scale is None:scale=current_scale
        checks[name+'_frozen_scale_exact']=np.array_equal(scale,current_scale)
        for phase,result in endpoint['phases'].items():
            if replication and int(phase) not in plan['data']['train']['phases']:continue
            correct=None;target=None
            for variant,metrics in result['variants'].items():
                path=run/f'frozen_evaluation/phase_{phase}/{variant}.npz'
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path) as a:
                    raw=a['predictions'].copy();raw_target=a['actual_future_command_target'].copy();seeds=a['sample_seeds'].copy()
                key=f'{name}_{phase}_{variant}'
                checks[key+'_shape_seeds_finite']=raw.shape==(32,2,8,36) and seeds.tolist()==plan['frozen_evaluation']['sample_seeds'] and bool(np.isfinite(raw).all())
                if phase not in references:references[phase]=(raw_target,seeds)
                checks[key+'_actual_targets_seeds_exact']=np.array_equal(raw_target,references[phase][0]) and np.array_equal(seeds,references[phase][1])
                prediction=raw.astype(np.float64)*scale;target=raw_target.astype(np.float64)*scale
                error=(prediction-target[None])**2
                own=error.mean((2,3));other=((prediction-target[::-1][None])**2).mean((2,3))
                checks[key+'_original_metrics_replayed']=bool(np.isclose(error.mean(),metrics['normalized_mse'],rtol=1e-5,atol=1e-12) and np.array_equal((own<other).sum(0),metrics['per_branch_correct_preference_draws']))
                if variant=='trained_demo_geometry_correct':correct=prediction
            mean=correct.mean(0);e=mean-target
            common=(e[0]+e[1])/2;contrast=(e[0]-e[1])/2
            variance=((correct-mean[None])**2).mean((0,1))
            total=((correct-target[None])**2).mean((0,1))
            parts=np.stack([common**2,contrast**2,variance])
            checks[f'{name}_{phase}_all288_decomposition_exact']=bool(np.allclose(parts.sum(0),total,rtol=1e-10,atol=1e-12))
            target_delta=(target[0]-target[1])/2;pred_delta=(mean[0]-mean[1])/2
            target_norm=float((target_delta**2).mean());gain=float((pred_delta*target_delta).mean()/target_norm)
            checks[f'{name}_{phase}_original_mean_baseline_replayed']=bool(np.isclose(target_norm,result['two_target_mean_normalized_mse'],rtol=1e-5,atol=1e-12))
            contrast_parallel=(gain-1)**2*target_norm
            contrast_orthogonal=float(((pred_delta-gain*target_delta)**2).mean())
            checks[f'{name}_{phase}_contrast_projection_exact']=bool(np.isclose((contrast**2).mean(),contrast_parallel+contrast_orthogonal,rtol=1e-10,atol=1e-12))
            old=saved_metrics['phases'][phase]['decomposition']['trained_demo_geometry_correct']
            checks[f'{name}_{phase}_saved_mean_variance_exact']=all(np.isclose(value,old[k],rtol=1e-10,atol=1e-12) for k,value in [('total_mse',total.mean()),('draw_mean_mse',parts[:2].sum(0).mean()),('draw_variance',variance.mean())])
            row=dict(total_mse=float(total.mean()),common_bias_mse=float(parts[0].mean()),contrast_bias_mse=float(parts[1].mean()),draw_variance=float(variance.mean()),
                mean_bias_mse=float(parts[:2].sum(0).mean()),contrast_gain=gain,contrast_parallel_mse=contrast_parallel,contrast_orthogonal_mse=contrast_orthogonal,
                baseline_mean_mse=target_norm,mean_threshold=.5*target_norm,
                variance_eliminated_mean_still_fails=bool(parts[:2].sum(0).mean()>.5*target_norm),
                by_channel={g:dict(total_mse=float(total[:,s].mean()),weighted_total_contribution=float(total[:,s].sum()/total.size),
                    common_bias_contribution=float(parts[0,:,s].sum()/total.size),contrast_bias_contribution=float(parts[1,:,s].sum()/total.size),variance_contribution=float(parts[2,:,s].sum()/total.size)) for g,s in {'joint29':slice(0,29),'linear3':slice(29,32),'angular3':slice(32,35),'contact1':slice(35,36)}.items()})
            rows.setdefault(phase,{})[name]=row
            arrays[f'{name}_{phase}_parts']=parts;arrays[f'{name}_{phase}_total']=total
            arrays[f'{name}_{phase}_normalized_mean']=mean;arrays[f'{name}_{phase}_normalized_target']=target
            if replication and name in ('ranked18','self18'):
                original=json.loads((RUN/'frozen_evaluation/centroid_error_readback/RESULT.json').read_text())['phases'][phase][name]
                checks[f'{name}_{phase}_complete_old_row_exact']=row==original
    np.savez_compressed(OUT/'DECOMPOSITION.npz',**arrays)
    with np.load(OUT/'DECOMPOSITION.npz') as a:checks['all_decomposition_arrays_saved_exact']=set(a.files)==set(arrays) and all(np.array_equal(a[k],v) for k,v in arrays.items())
    checks['all_source_npz_files_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3 if replication else 4,figsize=(18,12))
    for phase,ax in zip(sorted(rows,key=int),axes.flat):
        names=list(rows[phase]);x=np.arange(len(names));bottom=np.zeros(len(names))
        for key,color in [('common_bias_mse','C0'),('contrast_bias_mse','C1'),('draw_variance','C2')]:
            y=np.array([rows[phase][name][key] for name in names]);ax.bar(x,y,bottom=bottom,color=color,label=key);bottom+=y
        threshold=rows[phase]['self18']['mean_threshold'];ax.axhline(threshold,color='black',linestyle='--',label='fixed mean criterion')
        labels=['ranked84','self84','ranked400','self400'] if replication else names
        ax.set_xticks(x,labels);ax.set_title(f'Phase{phase}');ax.grid(axis='y',alpha=.2)
    for ax in list(axes.flat)[len(rows):]:ax.axis('off')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('All32saved draws: pair common bias + branch-contrast bias + variance; no new samples')
    fig.tight_layout();fig.savefig(OUT/'CENTROID_ERROR.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,
        source_hashes=hashes,plot_inspected=False,new_model_forwards=0,new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0))
    assert all(checks.values())
    print(json.dumps(dict(checks=len(checks),phase277=rows['277'],phase298=rows['298'])),flush=True)


if __name__=='__main__':main()
