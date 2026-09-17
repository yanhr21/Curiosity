"""Record unchanged official reverse DDPM calls; never a replacement sampler."""
import argparse
import json
import os
import socket
import hashlib
from pathlib import Path

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import (
    BASE, PARENT, ActualBranchGeometryDataset, GeneratorWrapper, CONTEXT_KEY,
    restore_geometry_state, write,
)

SOURCE = BASE / 'matched_generator_branch_paired_rank025512'
SOLVER50 = BASE / 'matched_generator_branch_rank025_solver50'
OUT = SOLVER50 / 'frozen_evaluation/reverse_process_probe'
ORIGINAL_SOURCE = SOURCE
SELF_SOURCE = BASE / 'matched_generator_branch_self_replay01_gaps512'
RANKED_SOURCE = BASE / 'matched_generator_branch_latent_replay01_gaps512'
PAIRED_OUT = SELF_SOURCE / 'frozen_evaluation/paired_own_reverse'


def prepare_pair():
    probe=SELF_SOURCE/'frozen_evaluation/training_replay_probe'
    for filename in ('RESULT.json','SAVED_READBACK.json','DECISION.json'):
        report=json.loads((probe/filename).read_text())
        assert report['checks_passed']
    assert json.loads((probe/'RESULT.json').read_text())['plot_inspected']
    assert json.loads((probe/'DECISION.json').read_text())['all9_correct_epsilon_and_clean_means_lower']
    plans=[json.loads((p/'PROTOCOL.json').read_text()) for p in (RANKED_SOURCE,SELF_SOURCE)]
    assert plans[0]['frozen_evaluation']==plans[1]['frozen_evaluation']
    PAIRED_OUT.mkdir(exist_ok=False)
    protocol=dict(phases=plans[1]['frozen_evaluation']['evaluation_phases'],
        seeds=plans[1]['frozen_evaluation']['sample_seeds'][:4],steps=[16],conditions=['correct','wrong'],
        replayed_sample_count=176,optimizer_updates=0,new_physics_steps=0,
        scope='Diagnostic exact replays of first4 existing primary32 seeds, all11phases, bothbranches/correctwrong. Original full official model and16step sampler unchanged. Reused218/258 remain evaluation only. No new independent paths, training or physics.',
        automatic_next_action='Inspect all22phase panels and independent saved readback, require matched initialnoise/times/targets and exact fullmodel final replays. Then evaluate self18 on frozen ranked18 states to separate direct output changes from state feedback. Do not iterate teachers, sweep coefficients or solvers, extend512, or replace primary32draw metrics.')
    for name,source in [('ranked18',RANKED_SOURCE),('self18',SELF_SOURCE)]:
        out=PAIRED_OUT/name;out.mkdir()
        hashes={}
        for phase in protocol['phases']:
            for condition in protocol['conditions']:
                path=source/f'frozen_evaluation/phase_{phase}/trained_demo_geometry_{condition}.npz'
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        write(out/'PROTOCOL.json',dict(protocol,source_run=str(source),saved_primary_hashes=hashes))
    write(PAIRED_OUT/'PROTOCOL.json',dict(protocol,source_runs=[str(RANKED_SOURCE),str(SELF_SOURCE)],total_exact_replayed_paths=352,total_official_denoiser_forwards=5632))


def run_pair():
    global SOURCE,OUT
    for name,source in [('ranked18',RANKED_SOURCE),('self18',SELF_SOURCE)]:
        SOURCE=source;OUT=PAIRED_OUT/name
        run()
    write(PAIRED_OUT/'RESULT.json',dict(execution_completed=True,checks_passed=True,
        model_results={name:json.loads((PAIRED_OUT/name/'RESULT.json').read_text()) for name in ('ranked18','self18')},
        plots_inspected=False,independent_readback_completed=False,new_optimizer_updates=0,new_physics_steps=0))


def prepare():
    result=json.loads((SOLVER50/'frozen_evaluation/SOLVER_COMPARISON.json').read_text())
    assert result['checks_passed'] and not result['decision']['all_prediction_requirements_passed']
    own_hinge=SOURCE!=ORIGINAL_SOURCE
    if own_hinge:
        transfer=json.loads((SOURCE/'frozen_evaluation/recorded_path_transfer/RESULT.json').read_text())
        assert transfer['checks_passed'] and transfer['plot_inspected']
    plan=json.loads((SOURCE/'PROTOCOL.json').read_text())
    OUT.mkdir(exist_ok=False)
    write(OUT/'PROTOCOL.json',dict(source_run=str(SOURCE),solver50_run=str(SOLVER50),
        phases=plan['frozen_evaluation']['evaluation_phases'],seeds=plan['frozen_evaluation']['sample_seeds'][:4],
        steps=[16] if own_hinge else [16,50],conditions=['correct','wrong'],
        recordings=['model_input_xt','model_epsilon','scheduler_pred_original_sample','scheduler_prev_sample'],
        scope='Wrap the original scheduler.step only to record the complete official model output and scheduler inputs/outputs. No computation, solver, weights, labels or RNG are changed. Exact replay of first4saved draws for each of9phases,2conditions,2branches. '+('Fullhinge own16step paths for comparison with already recorded full025 paths and hinge-on-full025-state predictions. Same seeds/16step grid must give identical initialnoise; actual innovations and direct/state-feedback decomposition require readback. ' if own_hinge else 'Both16and50solvers; cross-solver random innovations are not time-matched. ')+'Actual targets used only after generation for diagnosis. Four replay seeds are diagnostic, not new training replicates or substitute primary32draw metrics.',
        automatic_next_action='Inspect all9phase reverse curves and exact final-sample/full-state checks. Determine whether failed TRAIN futures deviate at high noise, drift at low noise, or retain a clipping-bound residual; report reused-check behavior separately. Predeclare one evidence-driven next method diagnosis, preserving both complete negative solver endpoints. No optimizer updates, controller execution or SMP claims.',
        optimizer_updates=0,new_physics_steps=0,replayed_sample_count=144 if own_hinge else 288,
        own_path_comparison_next_action='For hinge own paths, compare savedfull02516step states and already computed hinge-on-full025-state predictions. Verify matched initialnoise/steps/final replays/fullstates, then decompose model-change versus induced-state feedback and read all9plots before choosing next bounded TRAIN-based hypothesis. No hybrid deployment, optimizer update, custom solver or generated physics.'))


def run():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    assert not (OUT/'PARTIAL_RESULT.json').exists() and not (OUT/'RESULT.json').exists()
    plan=json.loads((SOURCE/'PROTOCOL.json').read_text())
    with (SOURCE/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
        state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
    policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
    restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
    policy.cuda().eval().requires_grad_(False)
    assert sum(p.numel() for p in policy.parameters())==8319216
    assert policy.noise_scheduler.config.clip_sample and policy.noise_scheduler.config.clip_sample_range==1.0
    original_step=policy.noise_scheduler.step
    scheduler_config=dict(policy.noise_scheduler.config)
    scheduler_alpha=policy.noise_scheduler.alphas_cumprod.detach().cpu().clone()
    checks={};rows={}
    for phase in protocol['phases']:
        dataset=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state'])
        samples=[dataset[i] for i in (0,1)]
        obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
        target=policy.normalizer['action'].normalize(torch.stack([s['action'] for s in samples]).cuda()).cpu().numpy().astype(np.float64)
        rows[str(phase)]={}
        for steps in protocol['steps']:
            policy.num_inference_steps=steps
            shape=(4,2,2,steps,8,36)
            arrays={k:np.empty(shape,dtype=np.float32) for k in ('xt','epsilon','pred_original','prev_sample')}
            times=[]
            for ci,condition in enumerate(protocol['conditions']):
                inputs=dict(obs)
                if condition=='wrong':inputs[CONTEXT_KEY]=obs[CONTEXT_KEY].flip(0)
                source=SOURCE if steps==16 else SOLVER50
                with np.load(source/f'frozen_evaluation/phase_{phase}/trained_demo_geometry_{condition}.npz') as a:
                    expected=a['predictions'][:4].copy()
                predictions=[]
                for si,seed in enumerate(protocol['seeds']):
                    pair=[]
                    for branch in (0,1):
                        records=[]
                        def record_step(model_output,t,sample,*args,**kwargs):
                            result=original_step(model_output,t,sample,*args,**kwargs)
                            cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                            records.append((int(t),sample.detach().cpu().clone(),model_output.detach().cpu().clone(),result.pred_original_sample.detach().cpu().clone(),result.prev_sample.detach().cpu().clone()))
                            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
                            return result
                        policy.noise_scheduler.step=record_step
                        try:
                            torch.manual_seed(seed)
                            with torch.inference_mode():
                                prediction=policy.predict_action({k:v[branch:branch+1] for k,v in inputs.items()})
                        finally:
                            policy.noise_scheduler.step=original_step
                        assert len(records)==steps
                        times=[r[0] for r in records]
                        assert times==policy.noise_scheduler.timesteps.tolist()
                        for index,key in enumerate(arrays):
                            arrays[key][si,ci,branch]=torch.cat([r[index+1] for r in records]).numpy()
                        pair.append(prediction.cpu().numpy())
                    predictions.append(np.concatenate(pair))
                checks[f'{phase}_{steps}_{condition}_all_four_saved_samples_exact']=np.array_equal(np.stack(predictions),expected)
            for key,value in arrays.items():checks[f'{phase}_{steps}_{key}_finite']=bool(np.isfinite(value).all())
            np.savez_compressed(OUT/f'phase_{phase}_steps_{steps}.npz',**arrays,times=times,seeds=protocol['seeds'],normalized_actual_target=target)
            with np.load(OUT/f'phase_{phase}_steps_{steps}.npz') as saved:
                checks[f'{phase}_{steps}_all_saved_arrays_exact']=all(np.array_equal(saved[k],v) for k,v in dict(arrays,times=times,seeds=protocol['seeds'],normalized_actual_target=target).items())
            clean=arrays['pred_original'].astype(np.float64)
            own=((clean-target[None,None,:,None])**2).mean((-1,-2))
            other=((clean-target[::-1][None,None,:,None])**2).mean((-1,-2))
            mean=clean.mean(0)
            bias=((mean-target[None,:,None])**2).mean((-1,-2))
            variance=((clean-mean[None])**2).mean((0,-1,-2))
            checks[f'{phase}_{steps}_bias_variance_exact']=bool(np.allclose(own.mean(0),bias+variance,rtol=1e-10,atol=1e-12))
            rows[str(phase)][str(steps)]=dict(times=times,
                axes=['condition','branch','reverse_step'],clean_mse=own.mean(0).tolist(),
                draw_mean_mse=bias.tolist(),draw_variance=variance.tolist(),
                own_preference_draws=(own<other).sum(0).tolist(),
                clipping_floor=float(np.mean((target-target.clip(-1,1))**2)))
            write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,phases=rows))
            if not all(checks.values()):raise RuntimeError('Reverse recorder failed exact replay or metric checks')
            print(json.dumps(dict(phase=phase,steps=steps,all_four_replays_exact=True)),flush=True)
    checks['complete_full_state_unchanged']=all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
    checks['no_parameter_gradients']=all(p.grad is None for p in policy.parameters())
    checks['scheduler_config_and_alpha_unchanged']=scheduler_config==dict(policy.noise_scheduler.config) and torch.equal(scheduler_alpha,policy.noise_scheduler.alphas_cumprod.cpu())
    checks['recording_hook_restored']=policy.noise_scheduler.step==original_step
    for filename,digest in protocol.get('saved_primary_hashes',{}).items():
        checks['primary_unchanged_'+filename]=hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    nrows=(len(protocol['phases'])+2)//3
    fig,axes=plt.subplots(nrows,3,figsize=(18,4.3*nrows))
    for phase,ax in zip(protocol['phases'],axes.flat):
        for ri,steps in enumerate(protocol['steps']):
            row=rows[str(phase)][str(steps)];mse=np.asarray(row['clean_mse'])
            for branch in (0,1):
                for ci,style in enumerate(('-','--')):
                    ax.plot(row['times'],mse[ci,branch],color=f'C{2*ri+branch}',linestyle=style,label=f'{steps}step branch{branch} '+protocol['conditions'][ci])
        ax.set_title(f'Phase{phase}');ax.set_yscale('log');ax.invert_xaxis();ax.set_xlabel('Reverse diffusion time');ax.grid(alpha=.25)
    for ax in list(axes.flat)[len(protocol['phases']):]:ax.set_visible(False)
    axes[0,0].legend(fontsize=6)
    fig.suptitle('Recorded official reverse predictions: same saved4seeds per solver; actual target is readback only; no physics')
    fig.tight_layout();fig.savefig(OUT/'REVERSE_PROCESS.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,
        full_parameter_count=8319216,plot_inspected=False,new_optimizer_updates=0,new_physics_steps=0,
        replayed_sample_count=protocol['replayed_sample_count'],scope=protocol['scope']))
    assert all(checks.values())


def main():
    global SOURCE, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','run'],required=True)
    parser.add_argument('--run',type=Path,default=SOURCE)
    args=parser.parse_args()
    selected=args.run.resolve()
    if selected==SELF_SOURCE:
        prepare_pair() if args.mode=='prepare' else run_pair()
        return
    if selected==ORIGINAL_SOURCE.parent/'matched_generator_branch_paired_hinge025512':
        SOURCE=selected;OUT=SOURCE/'frozen_evaluation/own_reverse_process'
    elif selected!=ORIGINAL_SOURCE:
        raise RuntimeError('Use only the declared full025 or hinge official reverse recorder')
    prepare() if args.mode=='prepare' else run()


if __name__=='__main__':main()
