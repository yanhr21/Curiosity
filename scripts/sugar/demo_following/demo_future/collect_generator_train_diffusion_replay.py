"""Record fresh TRAIN-only latent replay with the unchanged official Generator."""
import json
import os
import socket

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import (
    BASE, PARENT, ActualBranchGeometryDataset, GeneratorWrapper,
    restore_geometry_state, write,
)

SOURCE=BASE/'matched_generator_branch_paired_rank025512'
EVIDENCE=BASE/'matched_generator_branch_paired_hinge025512/frozen_evaluation/own_reverse_process/path_feedback_readback/generated_state_gradient_audit'
OUT=BASE/'generator_train_diffusion_replay8'
SEEDS=list(range(272230,272238))


def main():
    global OUT, SOURCE
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gap-coverage',action='store_true')
    parser.add_argument('--current-gap-model',action='store_true')
    args=parser.parse_args()
    assert not (args.gap_coverage and args.current_gap_model)
    original_phases=[158,178,197,245,277,298,318]
    old_replay=BASE/'generator_train_diffusion_replay8'
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    old=json.loads((EVIDENCE/'RESULT.json').read_text())
    assert old['checks_passed'] and old['plot_inspected']
    if args.current_gap_model:
        decision=json.loads((BASE/'matched_generator_branch_latent_only_gaps512/frozen_evaluation/RANK_ABLATION_COMPARISON.json').read_text())
        assert decision['checks_passed'] and not decision['decision']['all_prediction_requirements_passed']
        assert len(decision['phases'])==11 and all(v['new_mse']>v['old_mse'] for v in decision['phases'].values())
        SOURCE=BASE/'matched_generator_branch_latent_replay01_gaps512'
        OUT=BASE/'generator_train_diffusion_replay8_current_gaps'
    plan=json.loads((SOURCE/'PROTOCOL.json').read_text())
    phases=plan['data']['train']['phases']
    assert phases==([158,178,197,221,245,261,277,298,318] if args.current_gap_model else original_phases)
    data_plan=plan
    if args.gap_coverage:
        corpus=BASE/'generator_actual_branch_coverage_gaps2'
        physical=json.loads((corpus/'RESULT.json').read_text())
        preparation=json.loads((corpus/'MODEL_PREPARATION_READBACK.json').read_text())
        assert physical['all_phase_data_passed'] and physical['physical_plots_inspected'] and physical['all_new_xyz_plots_inspected'] and preparation['checks_passed']
        data_plan=json.loads((corpus/'NEXT_MATCHED_PROTOCOL.json').read_text())
        phases=data_plan['data']['train']['phases']
        assert phases==[158,178,197,221,245,261,277,298,318]
        assert data_plan['generated_state_objective']['teacher_run']==str(SOURCE)
        previous_replay=json.loads((old_replay/'RESULT.json').read_text())
        assert previous_replay['checks_passed'] and previous_replay['plot_inspected']
        OUT=BASE/'generator_train_diffusion_replay8_gaps2'
    assert set(SEEDS).isdisjoint(plan['frozen_evaluation']['sample_seeds'])
    assert len(SEEDS)==8 and len(set(SEEDS))==8
    OUT.mkdir(exist_ok=False)
    protocol=dict(source_run=str(SOURCE),train_phases=phases,inference_seeds=SEEDS,
        conditions=['correct'],inference_steps=16,new_independent_paths=112,
        exact_fresh_replay_calls=112,old_reference_replay_calls=28,
        scope='Full frozen official full025 endpoint. Eight fresh seeds disjoint from all32 primary evaluation seeds, fourteen actualTRAIN cases, original16-step DDPM. These are diffusion latent states for a possible auxiliary objective; original actual futures remain targets. No new physical label, optimizer update, teacher refit or simplified model. Reused check phases218/258 are excluded. Known two source motions only. Fresh sampling compute and teacher training are additional cost, not equal-compute from-scratch evidence.',
        automatic_next_action='Inspect all seven TRAIN curves, saved arrays, full-state and exact replay checks, then execute a bounded full-model gradient audit on all8seeds/all16times before declaring one auxiliary-loss budget. Preserve old negative results and all primary evaluation seeds; no coefficient sweep or physical execution.',
        new_optimizer_updates=0,new_physics_steps=0)
    if args.gap_coverage:
        protocol.update(new_independent_paths=32,exact_fresh_replay_calls=32,reused_existing_paths=112,
            reused_replay_source=str(old_replay),actual_corpus=str(corpus),new_train_phases=[221,261],
            scope='Full frozenoldfull025 official16step teacher and same8seeds. Reuse all112existing paths/14oldTRAIN cases exactly, sample only32newpaths/4newactualTRAIN cases at221/261, and replay each new path once. Reproduce first2original primary draws for all7oldphases (28controlcalls). All18actual future labels stay actual, reused218/258checks excluded, no model update or physics. Same known sources and oldnormalizer, additional teacher/replay cost.',
            automatic_next_action='Inspect all9TRAIN curves, fullstate/scheduler/exactold14combinedarray and fresh32replay checks. Then complete separate18case144row CPU/BF16 preflights and matched512arms under already fixed0.25rank+0.1latent objective. No coefficient selection, new gradient-success claim, endpoint extension or generated physics.')
    if args.current_gap_model:
        protocol.update(new_independent_paths=144,exact_fresh_replay_calls=144,old_reference_replay_calls=36,
            comparison_replay=str(BASE/'generator_train_diffusion_replay8_gaps2'),
            scope='Frozen complete ranked18case Generator endpoint, all18actualTRAIN cases and same8seeds as oldfull025 replay. New current-model paths are paired by seed, not independent of old teacher paths. Original16step official sampler; each fresh final replay exact plus36 original primary boundary replays. No check labels, optimizer updates, changed targets or physics. Additional sampling compute explicitly retained.',
            automatic_next_action='Inspect all9TRAIN plots and exact model/target/seed/primary replay checks. Then audit the complete retained model at old-teacher and current-model xt, including exact own recorded predictions and full gradients against the unchanged base+rank objective. This diagnostic alone does not establish support for a new training budget; existing experiments retain their original replay.')
    write(OUT/'PROTOCOL.json',protocol)
    with (SOURCE/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:
        state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
    policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
    restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
    policy.cuda().eval().requires_grad_(False)
    policy.num_inference_steps=16
    scheduler=policy.noise_scheduler
    alpha=scheduler.alphas_cumprod.clone()
    assert scheduler.config.prediction_type=='epsilon' and scheduler.config.clip_sample
    original_step=scheduler.step
    checks=dict(full_parameter_count=sum(p.numel() for p in policy.parameters())==8319216,
        fresh_seeds_disjoint=set(SEEDS).isdisjoint(plan['frozen_evaluation']['sample_seeds']),
        train_only=not set(phases).intersection([218,258]))
    rows={};all_xt=[];all_target=[];all_initial=[]
    for phase in phases:
        dataset=ActualBranchGeometryDataset(data_plan['phase_corpora'][str(phase)],plan['normalizer_state'])
        samples=[dataset[i] for i in (0,1)]
        obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
        target=policy.normalizer['action'].normalize(torch.stack([s['action'] for s in samples]).cuda()).cpu().numpy().astype(np.float64)
        checks[f'{phase}_causal_inputs_shared']=all(torch.equal(obs[k][0],obs[k][1]) for k in ('obj_pos_b','obj_ori_b','last_action'))
        if phase in original_phases or args.current_gap_model:
            with np.load(SOURCE/f'frozen_evaluation/phase_{phase}/trained_demo_geometry_correct.npz') as a:
                old_predictions=a['predictions'][:2].copy()
            for si,seed in enumerate(plan['frozen_evaluation']['sample_seeds'][:2]):
                pair=[]
                for branch in (0,1):
                    torch.manual_seed(seed)
                    with torch.inference_mode():pred=policy.predict_action({k:v[branch:branch+1] for k,v in obs.items()})
                    pair.append(pred.cpu().numpy())
                checks[f'{phase}_{seed}_original_sample_exact']=np.array_equal(np.concatenate(pair),old_predictions[si])
        if args.gap_coverage and phase in original_phases:
            original_dataset=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state'])
            old_samples=[original_dataset[i] for i in (0,1)]
            checks[f'{phase}_old_actual_inputs_exact']=all(torch.equal(samples[i]['action'],old_samples[i]['action']) and set(samples[i]['obs'])==set(old_samples[i]['obs']) and all(torch.equal(samples[i]['obs'][k],old_samples[i]['obs'][k]) for k in samples[i]['obs']) for i in (0,1))
            source_file=old_replay/f'phase_{phase}_steps_16.npz'
            with np.load(source_file) as a:
                arrays={k:a[k].copy() for k in ('xt','epsilon','pred_original','prev_sample')}
                times=a['times'].tolist()
                checks[f'{phase}_old_targets_exact']=np.array_equal(target,a['normalized_actual_target'])
                checks[f'{phase}_old_seeds_times_exact']=a['seeds'].tolist()==SEEDS and times==list(range(45,-1,-3))
            (OUT/source_file.name).symlink_to(source_file)
            rows[str(phase)]=dict(previous_replay['phases'][str(phase)],reused_existing_paths=True)
            all_xt.append(arrays['xt'][:,0].transpose(0,2,1,3,4));all_target.append(target);all_initial.append(arrays['xt'][:,0,:,0])
            write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,phases=rows))
            assert all(checks.values())
            print(json.dumps(dict(phase=phase,reused_paths=16,all_old_inputs_targets_and_primary_replays_exact=True)),flush=True)
            continue
        arrays={k:np.empty((8,1,2,16,8,36),dtype=np.float32) for k in ('xt','epsilon','pred_original','prev_sample')}
        predictions=[];times=None
        for si,seed in enumerate(SEEDS):
            pair=[]
            for branch in (0,1):
                records=[]
                def record_step(model_output,t,sample,*args,**kwargs):
                    result=original_step(model_output,t,sample,*args,**kwargs)
                    records.append((int(t),sample.detach().cpu().clone(),model_output.detach().cpu().clone(),result.pred_original_sample.detach().cpu().clone(),result.prev_sample.detach().cpu().clone()))
                    return result
                scheduler.step=record_step
                try:
                    torch.manual_seed(seed)
                    with torch.inference_mode():pred=policy.predict_action({k:v[branch:branch+1] for k,v in obs.items()})
                finally:scheduler.step=original_step
                assert len(records)==16
                times=[r[0] for r in records]
                assert times==scheduler.timesteps.tolist()
                for index,key in enumerate(arrays):arrays[key][si,0,branch]=torch.cat([r[index+1] for r in records]).numpy()
                pair.append(pred.cpu().numpy())
                if args.current_gap_model:
                    recorded_rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
                torch.manual_seed(seed)
                with torch.inference_mode():replay=policy.predict_action({k:v[branch:branch+1] for k,v in obs.items()})
                checks[f'{phase}_{seed}_{branch}_fresh_final_exact']=torch.equal(pred,replay)
                if args.current_gap_model:
                    checks[f'{phase}_{seed}_{branch}_recorder_rng_exact']=torch.equal(recorded_rng[0],torch.get_rng_state()) and torch.equal(recorded_rng[1],torch.cuda.get_rng_state())
            predictions.append(np.concatenate(pair))
        predictions=np.stack(predictions)
        checks[f'{phase}_initial_noise_shared']=np.array_equal(arrays['xt'][:,0,0,0],arrays['xt'][:,0,1,0])
        checks[f'{phase}_successive_xt_exact']=np.array_equal(arrays['xt'][:,:,:,1:],arrays['prev_sample'][:,:,:,:-1])
        checks[f'{phase}_final_clean_previous_exact']=np.array_equal(arrays['pred_original'][:,:,:,-1],arrays['prev_sample'][:,:,:,-1])
        for key,value in arrays.items():checks[f'{phase}_{key}_finite']=bool(np.isfinite(value).all())
        filename=OUT/f'phase_{phase}_steps_16.npz'
        np.savez_compressed(filename,**arrays,times=times,seeds=SEEDS,normalized_actual_target=target,predictions=predictions)
        with np.load(filename) as a:
            checks[f'{phase}_saved_arrays_exact']=all(np.array_equal(a[k],v) for k,v in arrays.items()) and np.array_equal(a['predictions'],predictions) and np.array_equal(a['normalized_actual_target'],target)
        mse=((arrays['pred_original'][:,0].astype(np.float64)-target[None,:,None])**2).mean((0,3,4))
        rows[str(phase)]=dict(times=times,branch_clean_mse=mse.tolist(),final_mse=float(mse[:,-1].mean()))
        all_xt.append(arrays['xt'][:,0].transpose(0,2,1,3,4));all_target.append(target);all_initial.append(arrays['xt'][:,0,:,0])
        write(OUT/'PARTIAL_RESULT.json',dict(checks=checks,phases=rows))
        assert all(checks.values())
        print(json.dumps(dict(phase=phase,all_fresh_and_old_replays_exact=True,final_mse=rows[str(phase)]['final_mse'])),flush=True)
    saved=dict(xt=np.concatenate(all_xt,axis=2),normalized_actual_target=np.concatenate(all_target),initial_gaussian=np.concatenate(all_initial,axis=1),times=np.array(times),seeds=np.array(SEEDS),phases=np.repeat(phases,2),branches=np.tile([0,1],len(phases)))
    np.savez_compressed(OUT/'TRAIN_GENERATED_STATE_INPUTS.npz',**saved)
    with np.load(OUT/'TRAIN_GENERATED_STATE_INPUTS.npz') as a:checks['combined_arrays_readback_exact']=all(np.array_equal(a[k],v) for k,v in saved.items())
    checks['full_state_unchanged']=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
    checks['scheduler_values_unchanged']=torch.equal(alpha,scheduler.alphas_cumprod)
    checks['no_gradients']=all(p.grad is None for p in policy.parameters())
    checks['combined_shape']=saved['xt'].shape==(8,16,2*len(phases),8,36)
    if args.gap_coverage:
        indices=[2*phases.index(p)+branch for p in original_phases for branch in (0,1)]
        with np.load(old_replay/'TRAIN_GENERATED_STATE_INPUTS.npz') as old_arrays:
            checks['old14_combined_xt_exact']=np.array_equal(saved['xt'][:,:,indices],old_arrays['xt'])
            checks['old14_targets_exact']=np.array_equal(saved['normalized_actual_target'][indices],old_arrays['normalized_actual_target'])
            checks['old14_initial_gaussian_exact']=np.array_equal(saved['initial_gaussian'][:,indices],old_arrays['initial_gaussian'])
            checks['old14_phase_branch_binding_exact']=all(np.array_equal(saved[k][indices],old_arrays[k]) for k in ('phases','branches'))
            checks['all_saved_seeds_times_exact']=all(np.array_equal(saved[k],old_arrays[k]) for k in ('seeds','times'))
        checks['all_phases_same_initial_seed_noise']=all(np.array_equal(saved['initial_gaussian'][:,i],saved['initial_gaussian'][:,0]) for i in range(2*len(phases)))
    if args.current_gap_model:
        with np.load(BASE/'generator_train_diffusion_replay8_gaps2/TRAIN_GENERATED_STATE_INPUTS.npz') as old_arrays:
            for key in ('normalized_actual_target','initial_gaussian','times','seeds','phases','branches'):
                checks['old_teacher_'+key+'_exact']=np.array_equal(saved[key],old_arrays[key])
        checks['all_phases_same_initial_seed_noise']=all(np.array_equal(saved['initial_gaussian'][:,i],saved['initial_gaussian'][:,0]) for i in range(2*len(phases)))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(15,11)) if args.gap_coverage or args.current_gap_model else plt.subplots(2,4,figsize=(17,8))
    for phase,ax in zip(phases,axes.flat):
        for branch in (0,1):ax.plot(times,rows[str(phase)]['branch_clean_mse'][branch],label=f'branch{branch}')
        if args.current_gap_model:
            teacher=json.loads((BASE/'generator_train_diffusion_replay8_gaps2/RESULT.json').read_text())['phases'][str(phase)]
            for branch in (0,1):ax.plot(times,teacher['branch_clean_mse'][branch],'--',color=f'C{branch}',label=f'old teacher b{branch}')
        ax.invert_xaxis();ax.set_yscale('log');ax.set_title(f'TRAIN{phase}');ax.grid(alpha=.25)
    axes[0,0].legend()
    if not (args.gap_coverage or args.current_gap_model):axes.flat[-1].axis('off')
    fig.suptitle('Current ranked18case own TRAIN paths versus old teacher; same8seeds, no updates or physics' if args.current_gap_model else 'TRAIN latent replay: same full025 official16step and8seeds; only new221/261paths sampled' if args.gap_coverage else 'Fresh TRAIN latent replay: full025 official16step, eight new seeds, no optimizer or physics')
    fig.tight_layout();fig.savefig(OUT/'TRAIN_REPLAY.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,plot_inspected=False,new_independent_paths=protocol['new_independent_paths'],exact_fresh_replays=protocol['exact_fresh_replay_calls'],reused_existing_paths=112 if args.gap_coverage else 0,old_reference_replays=protocol['old_reference_replay_calls'],new_optimizer_updates=0,new_physics_steps=0,scope=protocol['scope']))
    assert all(checks.values())
    if args.gap_coverage or args.current_gap_model:return
    write(OUT/'NEXT_GRADIENT_PROTOCOL.json',dict(source_run=str(SOURCE),train_phases=phases,inference_seeds=SEEDS,diffusion_times=times,gradient_rows=128,input_arrays=str(OUT/'TRAIN_GENERATED_STATE_INPUTS.npz'),checks=checks,
        scope='Same fixed full025 model, all eight fresh TRAIN latent seeds and all16times. Original actual targets, exact recorded point outputs and original q control. Raw FP32/eval directions only, no optimizer or automatic coefficient sweep.'))


if __name__=='__main__':main()
