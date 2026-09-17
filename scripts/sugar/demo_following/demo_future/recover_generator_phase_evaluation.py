"""Recover only unfinished frozen phase evaluation after recorded job cancellation."""
import argparse
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import (
    BASE, PARENT, ARMS, ActualBranchGeometryDataset, GeneratorWrapper,
    CONTEXT_KEY, restore_geometry_state, evaluate_branch, readback_phase_samples, write,
)

RUN = BASE / 'matched_generator_branch_latent_replay01_gaps512'
OUT = RUN / 'frozen_recovery_job295673'
COMPLETE = [158,178,197,218,221]
MISSING = [245,258,261,277,298,318]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def prepare():
    torch.set_num_threads(1)
    assert not OUT.exists() and not (RUN/'RESULT.json').exists()
    accounting=subprocess.run(['sacct','-n','-P','-j','295673','-S','2026-09-14','-o','JobID,State,ExitCode,Start,End'],capture_output=True,text=True,check=True).stdout
    lines=[line.split('|') for line in accounting.splitlines()]
    assert any(row[0]=='295673' and row[1]=='CANCELLED by 2059' for row in lines)
    assert any(row[0]=='295673.0' and row[1]=='CANCELLED by 2059' and row[2]=='0:9' for row in lines)
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    assert sorted(COMPLETE+MISSING)==plan['frozen_evaluation']['evaluation_phases']
    assert all(row['returncode']==0 for row in json.loads((RUN/'CHILDREN.json').read_text()))
    assert json.loads((RUN/'TRAINING_LOSS_READBACK.json').read_text())['checks_passed']
    checks={};manifest={};initial=[]
    immutable=[RUN/'PROTOCOL.json',RUN/'CHILDREN.json',RUN/'TRAINING_LOSS_READBACK.json']
    for arm in ARMS:
        root=RUN/arm
        report=json.loads((root/'RESULT.json').read_text())
        checks[arm+'_completed512']=report['endpoint_checks_passed'] and all(report['checks'].values()) and report['actual_optimizer_updates']==512
        init=torch.load(root/'INITIAL_MODEL.pt',map_location='cpu')['model'];initial.append(init)
        with (root/'checkpoints/endpoint.ckpt').open('rb') as f:payload=torch.load(f,pickle_module=dill,map_location='cpu')
        model=payload['state_dicts']['model'];adam=payload['state_dicts']['optimizer']
        checks[arm+'_full_state_keys_shapes_finite']=model.keys()==init.keys() and all(v.shape==init[k].shape and bool(torch.isfinite(v).all()) for k,v in model.items())
        checks[arm+'_normalizer_exact']=all(torch.equal(v,init[k]) for k,v in model.items() if k.startswith('normalizer.'))
        checks[arm+'_full_adam512_finite']=bool(adam['state']) and all(int(v['step'])==512 and all(bool(torch.isfinite(v[k]).all()) for k in ('exp_avg','exp_avg_sq')) for v in adam['state'].values())
        immutable.extend(root/name for name in ['INITIAL_MODEL.pt','checkpoints/endpoint.ckpt','BATCH_ORDER.json','DATA_AND_BUDGET.json','PAIRED_LOSSES.json','GENERATED_STATE_EXPOSURES.json','RESULT.json','logs.json.txt'])
        del payload,model,adam
    checks['both_full_initial_states_exact']=initial[0].keys()==initial[1].keys() and all(torch.equal(v,initial[1][k]) for k,v in initial[0].items())
    checks['both_actual_batch_orders_exact']=(RUN/ARMS[0]/'BATCH_ORDER.json').read_bytes()==(RUN/ARMS[1]/'BATCH_ORDER.json').read_bytes()
    scale=initial[0]['normalizer.params_dict.action.scale'].numpy()
    offset=initial[0]['normalizer.params_dict.action.offset'].numpy()
    for phase in COMPLETE:
        folder=RUN/f'frozen_evaluation/phase_{phase}'
        report=json.loads((folder/'RESULT.json').read_text())
        checks[f'{phase}_complete_matched_interventions']=report['execution_completed'] and all(report['matched_and_intervention_checks'].values())
        data=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state'])
        target=np.stack([data[i]['action'].numpy() for i in (0,1)])
        last=np.stack([data[i]['obs']['last_action'].numpy() for i in (0,1)]).repeat(8,axis=1)
        predictions={}
        for variant in plan['frozen_evaluation']['variants']:
            with np.load(folder/f'{variant}.npz') as a:
                raw=a['predictions'].copy();predictions[variant]=raw
                checks[f'{phase}_{variant}_actual_binding_exact']=np.array_equal(a['actual_future_command_target'],target) and np.array_equal(a['previous_command'],last) and np.array_equal(a['sample_seeds'],plan['frozen_evaluation']['sample_seeds'])
            # Match the official FP32 normalize-then-subtract order. Algebraic
            # cancellation of offsets changes rounding in tiny horizon entries.
            normalized=raw*scale+offset
            normalized_target=target*scale+offset
            error=((normalized-normalized_target[None])**2).astype(np.float64)
            other=((normalized-normalized_target[::-1][None])**2).astype(np.float64)
            ref=report['variants'][variant]
            checks[f'{phase}_{variant}_full32draw_saved_metrics']=raw.shape==(32,2,8,36) and bool(np.isfinite(raw).all()) and bool(np.isclose(error.mean(),ref['normalized_mse'],rtol=1e-5,atol=1e-12)) and bool(np.allclose(error.mean(0),ref['normalized_mse_by_branch_horizon_dimension'],rtol=1e-5,atol=1e-12)) and np.array_equal((error.mean((2,3))<other.mean((2,3))).sum(0),ref['per_branch_correct_preference_draws'])
        checks[f'{phase}_saved_swapped_context_exact']=np.array_equal(predictions['trained_demo_geometry_correct'][:,::-1],predictions['trained_demo_geometry_wrong'])
        checks[f'{phase}_saved_zero_conditions_exact']=all(np.array_equal(predictions[v][:,0],predictions[v][:,1]) for v in ('initial_shared_goal_zero','trained_zero_context','trained_demo_geometry_zero'))
        immutable.extend(folder.iterdir())
    interrupted=RUN/'frozen_evaluation/phase_245'
    checks['interrupted245_directory_empty']=interrupted.is_dir() and not list(interrupted.iterdir())
    checks['remaining5_phase_directories_absent']=all(not (RUN/f'frozen_evaluation/phase_{phase}').exists() for phase in MISSING[1:])
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    for p in immutable:manifest[str(p.relative_to(RUN))]=dict(sha256=digest(p),bytes=p.stat().st_size)
    OUT.mkdir()
    protocol=dict(old_job_id=295673,accounting=accounting,completed_phases=COMPLETE,remaining_phases=MISSING,
        preserved_full_updates_per_arm=512,new_optimizer_updates=0,new_physics_steps=0,
        completed_primary_trajectories=1600,remaining_primary_trajectories=1920,
        additional_boundary_replay_trajectories=100,
        interrupted_unsaved_245_draw_count='unknown; no complete variant array was saved. Any in-flight unretained computation is additional consumed work, not exact RNG resume.',
        recovery='Full frozen existing endpoints only. All5complete phase arrays/metrics must bind exactly to actual inputs and replay first2seeds for5conditions x2branches on the replacement H200. Verify full model/scheduler state and immutable hashes. Preserve empty interrupted245directory under this incident folder, then use existing evaluate_branch for only6missing phases. Complete original all11phase saved readback,88panels and strict comparison. No retraining, altered solver, normalization, threshold or old file overwrite.',
        automatic_next_action='After all frozen evidence, apply the original scientific criteria and continue the active native/physical/SMP objective from the observed result. No human authorization boundary.')
    write(OUT/'PROTOCOL.json',protocol)
    write(OUT/'PRESERVED_MANIFEST.json',manifest)
    write(OUT/'CPU_READBACK.json',dict(checks_passed=True,checks=checks,completed_phases=COMPLETE,new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0))
    print(json.dumps(dict(cpu_recovery_checks_passed=True,checks=len(checks),immutable_files=len(manifest),completed_phases=COMPLETE,remaining_phases=MISSING)),flush=True)


def verify_manifest():
    manifest=json.loads((OUT/'PRESERVED_MANIFEST.json').read_text())
    return {name:(RUN/name).stat().st_size==row['bytes'] and digest(RUN/name)==row['sha256'] for name,row in manifest.items()}


def pipeline():
    assert os.environ.get('SLURM_STEP_ID') and os.environ.get('SLURM_JOB_ID')!='295673' and not socket.gethostname().startswith(('mgmtserver','login'))
    assert 'H200' in torch.cuda.get_device_name()
    torch.set_num_threads(8)
    assert json.loads((OUT/'CPU_READBACK.json').read_text())['checks_passed']
    assert not (OUT/'REPLAY_CHECKS.json').exists() and not (RUN/'RESULT.json').exists()
    checks={};assert all(verify_manifest().values())
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    replay_calls=0
    for variant in plan['frozen_evaluation']['variants']:
        if variant=='initial_shared_goal_zero':state=torch.load(RUN/'zero_context/INITIAL_MODEL.pt',map_location='cpu')['model']
        else:
            arm='zero_context' if variant=='trained_zero_context' else 'demo_geometry'
            with (RUN/arm/'checkpoints/endpoint.ckpt').open('rb') as f:state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
        policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
        mode='demo_geometry' if variant in ('trained_demo_geometry_correct','trained_demo_geometry_wrong') else 'zero_context'
        restore_geometry_state(policy,state,mode,goal_condition_mode='zero_diagnostic')
        policy.cuda().eval().requires_grad_(False)
        alpha=policy.noise_scheduler.alphas_cumprod.clone()
        checks[variant+'_full8319216_parameters']=sum(p.numel() for p in policy.parameters())==8319216
        for phase in COMPLETE:
            data=ActualBranchGeometryDataset(plan['phase_corpora'][str(phase)],plan['normalizer_state'])
            samples=[data[i] for i in (0,1)]
            obs={k:torch.stack([s['obs'][k] for s in samples]).cuda() for k in samples[0]['obs']}
            if variant=='trained_demo_geometry_wrong':obs[CONTEXT_KEY]=obs[CONTEXT_KEY].flip(0)
            draws=[]
            with torch.inference_mode():
                for seed in plan['frozen_evaluation']['sample_seeds'][:2]:
                    pair=[]
                    for branch in (0,1):
                        torch.manual_seed(seed);pair.append(policy.predict_action({k:v[branch:branch+1] for k,v in obs.items()}));replay_calls+=1
                    draws.append(torch.cat(pair))
            with np.load(RUN/f'frozen_evaluation/phase_{phase}/{variant}.npz') as a:
                checks[f'{phase}_{variant}_first2draws_exact']=np.array_equal(torch.stack(draws).cpu().numpy(),a['predictions'][:2])
            assert all(checks.values()),[k for k,v in checks.items() if not v]
        checks[variant+'_full_frozen_state_exact']=policy.state_dict().keys()==state.keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items())
        checks[variant+'_official16step_scheduler_unchanged']=policy.num_inference_steps==16 and policy.noise_scheduler.timesteps.tolist()==list(range(45,-1,-3)) and torch.equal(alpha,policy.noise_scheduler.alphas_cumprod)
        checks[variant+'_no_gradients']=all(p.grad is None for p in policy.parameters())
        write(OUT/'REPLAY_CHECKS.partial.json',dict(checks=checks,replay_calls=replay_calls))
        assert all(checks.values())
        print(json.dumps(dict(replayed_variant=variant,checks_passed=True,cumulative_replay_calls=replay_calls)),flush=True)
        del policy;torch.cuda.empty_cache()
    checks['exact100_replay_calls']=replay_calls==100
    checks['all_preserved_files_unchanged']=all(verify_manifest().values())
    write(OUT/'REPLAY_CHECKS.json',dict(checks_passed=all(checks.values()),checks=checks,replay_calls=replay_calls,new_optimizer_updates=0,new_physics_steps=0))
    assert all(checks.values())
    interrupted=RUN/'frozen_evaluation/phase_245'
    assert not list(interrupted.iterdir())
    interrupted.rename(OUT/'empty_phase245_at_cancellation')
    results={str(p):json.loads((RUN/f'frozen_evaluation/phase_{p}/RESULT.json').read_text()) for p in COMPLETE}
    for phase in MISSING:
        results[str(phase)]=evaluate_branch(RUN,phase=phase)
        write(OUT/'PARTIAL_RESULT.json',results)
        print(json.dumps(dict(recovered_phase=phase,branch_pass=results[str(phase)]['branch_identifiability_passed'])),flush=True)
    immutable=verify_manifest();assert all(immutable.values())
    checks={str(p):results[str(p)]['branch_identifiability_passed'] for p in plan['data']['heldout_phase']['phases']}
    final=dict(execution_completed=True,phases={str(p):results[str(p)] for p in plan['frozen_evaluation']['evaluation_phases']},heldout_phase_checks=checks,heldout_phase_transfer_passed=all(checks.values()),
        training_phase_checks={str(p):results[str(p)]['branch_identifiability_passed'] for p in plan['data']['train']['phases']},
        real_training_examples=18,real_training_branch_examples=18,real_heldout_phase_examples=4,actual_updates_per_arm=512,native_replay_composition=None,
        new_physics_steps=0,horizon_plots_inspected=False,scope=plan['frozen_evaluation']['scope'],next_action=plan['automatic_next_action'],frozen_evaluation_recovery=str(OUT/'PROTOCOL.json'))
    write(RUN/'frozen_evaluation/RESULT.json',final)
    readback_phase_samples(RUN)
    write(RUN/'RESULT.json',final)
    write(OUT/'RESULT.json',dict(execution_completed=True,all_preserved_file_hashes_exact=all(immutable.values()),preserved_checks=immutable,reused_complete_phases=COMPLETE,newly_completed_phases=MISSING,replay_calls=100,new_primary_trajectories=1920,new_optimizer_updates=0,new_physics_steps=0))
    print(json.dumps(dict(recovery_completed=True,train=final['training_phase_checks'],reused_checks=checks)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument('--prepare',action='store_true');modes.add_argument('--pipeline',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else pipeline()


if __name__=='__main__':main()
