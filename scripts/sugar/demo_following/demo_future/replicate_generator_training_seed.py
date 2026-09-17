"""One fixed-teacher training-RNG replication of two complete official variants."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.profile_generator_existing_checkpoints import OUT as CURVE
from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import BASE,write
from scripts.sugar.demo_following.demo_future.compare_generator_rank_ablation import equal

ROOT=BASE.parents[2]
GROUP=BASE/'generator_fixed_teacher_seed_replication272400'
SOURCES=[BASE/'matched_generator_branch_latent_replay01_gaps512',BASE/'matched_generator_branch_self_replay01_gaps512']
RUNS=[BASE/'matched_generator_branch_ranked18_seed272400_512',BASE/'matched_generator_branch_self18_seed272400_512']
BOOT=ROOT/'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'
PREFIX='scripts.sugar.demo_following.demo_future.'
STABLE=('branch_dataset','batch_size','data','normalizer_state','epochs','actual_optimizer_updates','optimizer','scheduler','frozen_evaluation','paired_objective','generated_state_objective','noise_coupling','goal_condition_mode','phase_corpora','real_branch_examples')


def prepare():
    decision=json.loads((CURVE/'DECISION.json').read_text())
    curve=json.loads((CURVE/'RESULT.json').read_text());readback=json.loads((CURVE/'SAVED_READBACK.json').read_text())
    assert decision['checks_passed'] and decision['selected_next_action']=='independent_matched_training_seed_replication'
    assert curve['checks_passed'] and curve['plot_inspected'] and readback['checks_passed']
    assert not decision['phase277_accuracy_passing_updates'] and len(curve['phases']['277'])==9
    assert not GROUP.exists() and all(not r.exists() for r in RUNS)
    plans=[json.loads((source/'PROTOCOL.json').read_text()) for source in SOURCES]
    assert all(p['seed']==272084 and p['actual_optimizer_updates']==512 and p['batch_size']==144 for p in plans)
    assert all(plans[0][k]==plans[1][k] for k in STABLE if k!='generated_state_objective')
    GROUP.mkdir()
    protocol=dict(new_training_seed=272400,original_training_seed=272084,source_runs=list(map(str,SOURCES)),new_runs=list(map(str,RUNS)),
        fixed_teacher_training_seed_replication=True,new_optimizer_updates_per_arm=512,number_of_variants=2,arms_per_variant=2,total_new_optimizer_updates=2048,
        actual_rows_per_arm=73728,exposures_per_real_case_per_arm=4096,primary_sample_seeds=plans[0]['frozen_evaluation']['sample_seeds'],
        primary_evaluation_phases=plans[0]['frozen_evaluation']['evaluation_phases'],new_physics_steps=0,
        seed_selection='Single fixed seed272400. Before this source was written, rg integer-boundary search across existing experiment JSON and demo_following Python found no occurrence (exit1, no stderr). No outcome-based selection, extension, additional seed or best-seed result.',
        scope='Only student training RNG changes from272084 to272400; both variants and both context arms share the new seed. Complete release initialization, actual18TRAIN/144batch, original normalizer, optimizer/schedule,512budget,.25rank+.1aux, each variant frozen teacher and replay arrays, all11phase5condition32draw evaluation unchanged. Teachers retain their original training seeds: this is conditional on fixed teachers, not an independent end-to-end teacher-training replicate or new motion generalization.',
        required_checks='Actual CPU and H200BF16 full144row initial+learned checks for each variant, exact full release initialization, four-arm batch/budget matching, all512losses and exposures, fullmodel+Adam, all11phase5condition32draws and176horizon panels, saved metrics, exact initial predictions and unchanged targets/seeds, strict comparison of both training seeds.',
        scientific_criteria='For each seed separately, compare self18 to ranked18 using unchanged fullfive-condition criteria and previous passing TRAIN MSE nonregression. Report all11phase paired changes,9TRAIN passcounts,reused2checks,277 accuracy,298 recovery. No averaging two seeds into a success if either fails.',
        automatic_next_action='After all176panels and saved/full-state readbacks, apply both-seed criteria. If either seed fails or old passing TRAIN regresses, retain both negative/inconsistent results and predeclare a TRAIN-only comparison of residual bias and condition representation across all four endpoints before a different bounded hypothesis; no third seed or coefficient/teacher/solver sweep. Only if both seeds satisfy all prediction requirements, reassess broader native prediction and original-position physical compatibility before any generated execution or SMP benefit comparison.')
    hashes={}
    for source,plan in zip(SOURCES,plans):
        for path in [source/'PROTOCOL.json',Path(plan['normalizer_state']),Path(plan['generated_state_objective']['input_arrays'])]:
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for arm in ('zero_context','demo_geometry'):
            for path in [source/arm/'INITIAL_MODEL.pt',source/arm/'checkpoints/endpoint.ckpt']:
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    protocol['source_hashes']=hashes
    write(GROUP/'PROTOCOL.json',protocol)
    admission=dict(checks_passed=True,permits_preparing_matched_objective_experiment=True,weight=.1,fixed_teacher_training_seed_replication=True,
        next_runs=list(map(str,RUNS)),source_curve_decision=str(CURVE/'DECISION.json'),scientific_criteria=protocol['scientific_criteria'],scope=protocol['scope'])
    write(GROUP/'DECISION.json',admission)
    checks={}
    for source,run,old in zip(SOURCES,RUNS,plans):
        plan=copy.deepcopy(old)
        plan.update(run_name=run.name,seed=272400,data_coverage_predecessor=str(source),diagnostic_predecessor=str(source),
            objective_decision=str(GROUP/'DECISION.json'),training_seed_replication=dict(source_run=str(source),original_seed=272084,new_seed=272400,fixed_teacher=True,group_protocol=str(GROUP/'PROTOCOL.json')),
            generator_training_started=False,implementation_status='requires_full_CPU_and_BF16_new_seed_preflights',training=protocol['scope'],optimization=protocol['scope'],comparison_scope=protocol['scope'],automatic_next_action=protocol['automatic_next_action'])
        assert not plan.get('exact_training_replay_source') and not plan.get('rank_ablation_decision')
        checks[run.name+'_all_operational_settings_except_seed_exact']=all(plan[k]==old[k] for k in STABLE)
        run.mkdir();write(run/'PROTOCOL.json',plan)
    assert all(checks.values())
    write(GROUP/'PREPARATION.json',dict(checks_passed=True,checks=checks,new_optimizer_updates=0,new_physics_steps=0))
    print(json.dumps(dict(group=str(GROUP),new_training_updates=2048,checks=len(checks))),flush=True)


def execute(mode):
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    assert json.loads((GROUP/'PREPARATION.json').read_text())['checks_passed']
    assert json.loads((GROUP/'REPLICATION_HYPOTHESIS.json').read_text())['declared_before_new_training']
    protocol=json.loads((GROUP/'PROTOCOL.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in protocol['source_hashes'].items())
    path=GROUP/('CPU_CHILDREN.json' if mode=='cpu' else 'GPU_CHILDREN.json');assert not path.exists()
    commands=[]
    if mode=='cpu':
        commands=[(run.name+'_cpu','preflight_generator_latent_replay',['--run',str(run),'--device','cpu']) for run in RUNS]
    else:
        for run in RUNS:
            assert json.loads((run/'GENERATED_CPU_PREFLIGHT.json').read_text())['passed']
            commands.append((run.name+'_bf16','preflight_generator_latent_replay',['--run',str(run),'--device','cuda']))
        for run in RUNS:
            commands.extend([(run.name+'_pipeline','train_generator_geometry',['--run',str(run),'--pipeline']),
                (run.name+'_loss_readback','readback_generator_latent_training',['--run',str(run)])])
    rows=[]
    for label,module,args in commands:
        cmd=[sys.executable,str(BOOT),PREFIX+module,*args]
        with (GROUP/(label+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(label=label,pid=child.pid,pgid=os.getpgid(child.pid),command=cmd);rows.append(row);write(path,rows)
            row['returncode']=child.wait();write(path,rows)
        assert row['returncode']==0,row
    write(GROUP/('CPU_COMPLETION.json' if mode=='cpu' else 'PIPELINE_COMPLETION.json'),dict(execution_completed=True,children=rows,
        all_horizon_panels_inspected=False,strict_scientific_comparison_completed=False,new_physics_steps=0))


def hypothesis():
    assert not (GROUP/'REPLICATION_HYPOTHESIS.json').exists()
    assert all(not (run/arm).exists() for run in RUNS for arm in ('zero_context','demo_geometry'))
    source_results=[json.loads((run/'RESULT.json').read_text()) for run in SOURCES]
    means=[sum(r['phases'][p]['variants']['trained_demo_geometry_correct']['normalized_mse'] for p in r['training_phase_checks'])/9 for r in source_results]
    counts=[sum(r['training_phase_checks'].values()) for r in source_results]
    assert means[1]<means[0] and counts==[7,8]
    write(GROUP/'REPLICATION_HYPOTHESIS.json',dict(declared_before_new_training=True,old_seed=272084,new_seed=272400,
        original_equal_phase_train_mse=dict(ranked18=means[0],self18=means[1]),original_train_pass_counts=counts,
        prospective_directional_replication='At the fixed new student-training seed, self18 must have strictly lower equal-phase9TRAIN correctprompt MSE AND strictly more fullcriterion TRAIN passes than matched ranked18. Both inequalities are required; all11phase changes and reused checks reported separately. No third seed or outcome-dependent criterion.',
        interpretation='This tests reproducibility of the observed relative improvement, not absolute success. Originalseed overall prediction requirements already failed; that failure cannot be repaired by averaging or selecting the newseed. Strict original fullprediction and oldpassingTRAIN nonregression criteria remain unchanged and reported for eachseed.',
        automatic_next_action='Whether the two-direction improvement repeats or fails, compare frozen TRAIN residual bias and condition representations across allfour endpoints next. Label repeatability versus seed sensitivity explicitly. No further training seed or deployment based only on relative improvement.'))


def compare():
    torch.set_num_threads(8)
    assert not (GROUP/'COMPARISON.json').exists()
    assert json.loads((GROUP/'PIPELINE_COMPLETION.json').read_text())['execution_completed']
    protocol=json.loads((GROUP/'PROTOCOL.json').read_text());checks={};results={};batches=[];budgets=[];initials=[]
    for source,run in zip(SOURCES,RUNS):
        plan=json.loads((run/'PROTOCOL.json').read_text());old=json.loads((source/'PROTOCOL.json').read_text())
        checks[run.name+'_unchanged_variant']=all(plan[k]==old[k] for k in STABLE) and plan['seed']==272400
        result=json.loads((run/'RESULT.json').read_text());results[run.name]=result
        checks[run.name+'_complete_and_all88plots']=result['execution_completed'] and result['horizon_plots_inspected']
        for filename,field in [('TRAINING_LOSS_READBACK.json','checks_passed'),('frozen_evaluation/SAVED_PHASE_READBACK.json','saved_metrics_reproduced')]:
            r=json.loads((run/filename).read_text());checks[run.name+'_'+filename]=r[field] and all(r['checks'].values())
        for arm in ('zero_context','demo_geometry'):
            initial=torch.load(run/arm/'INITIAL_MODEL.pt',map_location='cpu')['model'];initials.append(initial)
            checks[run.name+arm+'_source_initial_exact']=equal(initial,torch.load(source/arm/'INITIAL_MODEL.pt',map_location='cpu')['model'])
            batches.append(json.loads((run/arm/'BATCH_ORDER.json').read_text()));budgets.append(json.loads((run/arm/'DATA_AND_BUDGET.json').read_text()))
            checks[run.name+arm+'_independent_training_order']=batches[-1]!=json.loads((source/arm/'BATCH_ORDER.json').read_text())
            with (run/arm/'checkpoints/endpoint.ckpt').open('rb') as f:payload=torch.load(f,pickle_module=dill,map_location='cpu')
            state=payload['state_dicts']['model'];adam=payload['state_dicts']['optimizer']
            with (source/arm/'checkpoints/endpoint.ckpt').open('rb') as f:reference=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']
            checks[run.name+arm+'_full_model_finite_and_shape']=state.keys()==initial.keys() and all(v.shape==initial[k].shape and bool(torch.isfinite(v).all()) for k,v in state.items())
            checks[run.name+arm+'_full_adam512_finite']=set(int(v['step']) for v in adam['state'].values())=={512} and all(bool(torch.isfinite(v[k]).all()) for v in adam['state'].values() for k in ('exp_avg','exp_avg_sq'))
            checks[run.name+arm+'_full_adam_binding_and_shapes']=equal(adam['param_groups'],reference['optimizer']['param_groups']) and adam['state'].keys()==reference['optimizer']['state'].keys() and all(v[k].shape==reference['optimizer']['state'][index][k].shape for index,v in adam['state'].items() for k in ('exp_avg','exp_avg_sq'))
            checks[run.name+arm+'_normalizer_exact']=all(torch.equal(v,initial[k]) for k,v in state.items() if k.startswith('normalizer.'))
        for phase in protocol['primary_evaluation_phases']:
            image_path=run/f'frozen_evaluation/phase_{phase}/HORIZON_ERRORS.png'
            inspection=json.loads((GROUP/'visual_inspection'/f'{run.name}_phase_{phase}.json').read_text())
            checks[f'{run.name}_{phase}_all8_horizon_panels_viewed']=inspection['all8_panels_viewed'] and inspection['image']==str(image_path) and inspection['sha256']==hashlib.sha256(image_path.read_bytes()).hexdigest()
            for variant in plan['frozen_evaluation']['variants']:
                with np.load(source/f'frozen_evaluation/phase_{phase}/{variant}.npz') as a,np.load(run/f'frozen_evaluation/phase_{phase}/{variant}.npz') as b:
                    checks[f'{run.name}_{phase}_{variant}_target_seeds_exact']=np.array_equal(a['actual_future_command_target'],b['actual_future_command_target']) and np.array_equal(a['sample_seeds'],b['sample_seeds'])
                    if variant=='initial_shared_goal_zero':checks[f'{run.name}_{phase}_initial_samples_exact']=np.array_equal(a['predictions'],b['predictions'])
    checks['all_four_initial_models_exact']=all(equal(initials[0],v) for v in initials[1:])
    checks['all_four_batch_orders_exact']=all(batches[0]==v for v in batches[1:])
    checks['all_four_budgets_exact']=all(budgets[0]==v for v in budgets[1:])
    checks['actual_four512budgets']=all(v['exact_optimizer_updates']==512 and v['seed']==272400 and v['batch_size']==144 for v in budgets)
    for path,digest in protocol['source_hashes'].items():checks['source_hash_'+path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    seed_rows={}
    for seed,runs in [(272084,SOURCES),(272400,RUNS)]:
        a,b=[json.loads((run/'RESULT.json').read_text()) for run in runs];rows={}
        for phase in protocol['primary_evaluation_phases']:
            old,new=[r['phases'][str(phase)] for r in (a,b)]
            om,nm=[r['variants']['trained_demo_geometry_correct']['normalized_mse'] for r in (old,new)]
            rows[str(phase)]=dict(ranked18_mse=om,self18_mse=nm,ratio=nm/om,ranked18_pass=old['branch_identifiability_passed'],self18_pass=new['branch_identifiability_passed'],split=new['evaluation_split'])
        criterion=dict(all9_train=all(b['training_phase_checks'].values()),both_reused_checks=all(b['heldout_phase_checks'].values()),
            all_previous_passing_train_nonregression=all(v['self18_mse']<=v['ranked18_mse'] for v in rows.values() if v['split']=='train' and v['ranked18_pass']))
        criterion['all_prediction_requirements']=all(criterion.values())
        train_rows=[v for v in rows.values() if v['split']=='train']
        means={k:sum(v[k] for v in train_rows)/len(train_rows) for k in ('ranked18_mse','self18_mse')}
        counts=[sum(r['training_phase_checks'].values()) for r in (a,b)]
        direction=means['self18_mse']<means['ranked18_mse'] and counts[1]>counts[0]
        seed_rows[str(seed)]=dict(phases=rows,criteria=criterion,ranked18_train_passes=counts[0],self18_train_passes=counts[1],equal_phase_train_mse=means,relative_improvement_criteria_passed=direction)
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    both=all(v['criteria']['all_prediction_requirements'] for v in seed_rows.values())
    hypothesis=json.loads((GROUP/'REPLICATION_HYPOTHESIS.json').read_text())
    write(GROUP/'COMPARISON.json',dict(execution_completed=True,checks_passed=True,checks=checks,seeds=seed_rows,both_seeds_prediction_requirements_passed=both,
        relative_improvement_replicated=all(v['relative_improvement_criteria_passed'] for v in seed_rows.values()),hypothesis=hypothesis,
        new_optimizer_updates=0,new_model_forwards=0,new_physics_steps=0,scope=protocol['scope'],automatic_next_action=hypothesis['automatic_next_action']))
    print(json.dumps(dict(checks=len(checks),both_seeds_pass=both,seeds=seed_rows)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['prepare','hypothesis','cpu','pipeline','compare'],required=True)
    mode=parser.parse_args().mode
    if mode=='prepare':prepare()
    elif mode=='hypothesis':hypothesis()
    elif mode=='compare':compare()
    else:execute(mode)


if __name__=='__main__':main()
