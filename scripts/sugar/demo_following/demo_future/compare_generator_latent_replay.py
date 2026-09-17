"""Strict old/new full-model comparison with equal auxiliary supervision."""
import json
import argparse
from pathlib import Path

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import BASE, write, readback_phase_samples

OLD=BASE/'matched_generator_branch_paired_rank025512'
RUN=BASE/'matched_generator_branch_latent_replay01512'


def main():
    global OLD, RUN
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=RUN)
    RUN=parser.parse_args().run.resolve();gap_coverage=RUN.name=='matched_generator_branch_latent_replay01_gaps512'
    self_replay=RUN.name=='matched_generator_branch_self_replay01_gaps512'
    actual_state=RUN.name=='matched_generator_actual_state_expert01512'
    measured_robot=RUN.name=='matched_generator_branch_measured_robot32512' or actual_state
    assert RUN.parent==BASE and (measured_robot or RUN.name in ('matched_generator_branch_latent_replay01512','matched_generator_branch_latent_replay01_gaps512','matched_generator_branch_self_replay01_gaps512'))
    if measured_robot:OLD=BASE/('matched_generator_branch_measured_robot32512' if actual_state else 'matched_generator_branch_self_replay01_gaps512')
    if gap_coverage:OLD=BASE/'matched_generator_branch_latent_replay01512'
    if self_replay:OLD=BASE/'matched_generator_branch_latent_replay01_gaps512'
    out=RUN/'frozen_evaluation/LATENT_REPLAY_COMPARISON.json'
    assert not out.exists()
    plans=[json.loads((p/'PROTOCOL.json').read_text()) for p in (OLD,RUN)]
    endpoints=[json.loads((p/'RESULT.json').read_text()) for p in (OLD,RUN)]
    assert all(r['execution_completed'] and r['horizon_plots_inspected'] for r in endpoints)
    stable=('seed','epochs','actual_optimizer_updates','optimizer','scheduler','normalizer_state','paired_objective')
    if not gap_coverage:stable+=('branch_dataset','batch_size','phase_corpora','frozen_evaluation')
    checks={k:plans[0][k]==plans[1][k] for k in stable}
    if measured_robot:
        checks['same_full_replay_teacher_objective_data']=all(plans[0][k]==plans[1][k] for k in ('generated_state_objective','data','noise_coupling','goal_condition_mode'))
        checks['complete_training_loss_readback']=json.loads((RUN/'TRAINING_LOSS_READBACK.json').read_text())['checks_passed']
        preflight_root=Path(plans[1]['robot_preflight_directory'])
        checks['complete_measured_input_preflights']=all(json.loads((preflight_root/name).read_text())['passed'] and json.loads((preflight_root/name).read_text())['parameter_layout']=='partitioned_leaf' for name in ('GENERATED_CPU_PREFLIGHT.json','GENERATED_BF16_PREFLIGHT.json'))
    if self_replay:
        checks['same_actual_data']=plans[0]['data']==plans[1]['data'] and plans[0]['real_branch_examples']==plans[1]['real_branch_examples']==18
        changed={'replay_source','input_arrays','teacher_run','reused_replay_source','reused_actual_cases','new_actual_cases'}
        a,b=[p['generated_state_objective'] for p in plans]
        checks['only_declared_teacher_refresh']=a.keys()==b.keys() and all(a[k]==b[k] for k in a if k not in changed) and b['teacher_run']==str(OLD) and b['reused_replay_source']==a['replay_source'] and b['reused_actual_cases']==18 and b['new_actual_cases']==0
        with np.load(a['input_arrays']) as x,np.load(b['input_arrays']) as y:
            checks['same_replay_labels_initial_noise_and_clocks']=set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files if k!='xt')
            checks['refreshed_diffusion_states']=not np.array_equal(x['xt'],y['xt'])
        checks['complete_training_loss_readback']=json.loads((RUN/'TRAINING_LOSS_READBACK.json').read_text())['checks_passed']
    if gap_coverage:
        checks['declared18case144row_coverage']=plans[1]['real_branch_examples']==18 and plans[1]['batch_size']==144 and plans[1]['data']['train']['phases']==[158,178,197,221,245,261,277,298,318]
        checks['evaluation_seeds_variants_solver_criteria_unchanged']=all(plans[0]['frozen_evaluation'][k]==plans[1]['frozen_evaluation'][k] for k in ('sample_seeds','variants','inference_steps','heldout_phase_criterion'))
        checks['same_replay_objective_and_clocks']=all(plans[0]['generated_state_objective'][k]==plans[1]['generated_state_objective'][k] for k in ('weight','apply_to_both_arms','seeds','times','objective','rng','expected_exposures_per_real_case'))
        checks['same_teacher_and_old_replay']=plans[1]['generated_state_objective']['teacher_run']==str(BASE/'matched_generator_branch_paired_rank025512') and plans[1]['generated_state_objective']['reused_replay_source']==plans[0]['generated_state_objective']['replay_source']
        for split in ('train','heldout_phase'):
            with np.load(plans[0]['data'][split]['arrays']) as a,np.load(plans[1]['data'][split]['arrays']) as b:
                indices=[next(i for i,row in enumerate(plans[1]['data'][split]['samples']) if (row['phase'],row['arm'])==(old['phase'],old['arm'])) for old in plans[0]['data'][split]['samples']]
                checks[split+'_all_old_actual_arrays_exact']=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k][indices]) for k in a.files)
        indices=[2*plans[1]['data']['train']['phases'].index(p)+branch for p in plans[0]['data']['train']['phases'] for branch in (0,1)]
        with np.load(plans[0]['generated_state_objective']['input_arrays']) as a,np.load(plans[1]['generated_state_objective']['input_arrays']) as b:
            checks['all_old14_replay_exact']=np.array_equal(a['xt'],b['xt'][:,:,indices]) and np.array_equal(a['normalized_actual_target'],b['normalized_actual_target'][indices]) and np.array_equal(a['initial_gaussian'],b['initial_gaussian'][:,indices])
        checks['new_both_arm_batches_exact']=(RUN/'zero_context/BATCH_ORDER.json').read_text()==(RUN/'demo_geometry/BATCH_ORDER.json').read_text()
        checks['new_both_arm_budgets_exact']=(RUN/'zero_context/DATA_AND_BUDGET.json').read_text()==(RUN/'demo_geometry/DATA_AND_BUDGET.json').read_text()
    assert plans[1]['generated_state_objective']['apply_to_both_arms']
    losses={}
    for arm in ('zero_context','demo_geometry'):
        if gap_coverage:
            batches=json.loads((RUN/arm/'BATCH_ORDER.json').read_text())
            budgets=[json.loads((root/arm/'DATA_AND_BUDGET.json').read_text()) for root in (OLD,RUN)]
            checks[arm+'_all512_actual144row_batches']=len(batches)==512 and all(sorted(v)==list(range(144)) for v in batches)
            checks[arm+'_declared_budget_change']=all(v['exact_optimizer_updates']==512 and v['epochs']==512 and v['seed']==272084 for v in budgets) and [v['actual_chunks'] for v in budgets]==[112,144] and [v['batch_size'] for v in budgets]==[112,144] and [len(v['timeline_metadata']) for v in budgets]==[14,18]
        else:
            checks[arm+'_actual_batch_order_exact']=(OLD/arm/'BATCH_ORDER.json').read_text()==(RUN/arm/'BATCH_ORDER.json').read_text()
            checks[arm+'_actual_budget_exact']=(OLD/arm/'DATA_AND_BUDGET.json').read_text()==(RUN/arm/'DATA_AND_BUDGET.json').read_text()
        exposure=json.loads((RUN/arm/'GENERATED_STATE_EXPOSURES.json').read_text())
        checks[arm+'_all512_replay_exposures_exact']=exposure['passed'] and all(exposure['checks'].values())
        records=json.loads((RUN/arm/'PAIRED_LOSSES.json').read_text());losses[arm]=records
        checks[arm+'_all512_loss_records_finite']=len(records)==512 and all(np.isfinite(v) for row in records for v in row.values())
        checks[arm+'_loss_arithmetic']=all(np.isclose(row['total'],row['base']+.25*row['rank']+.1*row['generated']+(.1*row['expert'] if actual_state else 0),rtol=2e-6,atol=2e-6) for row in records)
        current=json.loads((RUN/arm/'RESULT.json').read_text())
        checks[arm+'_full_endpoint_and_adam_integrity']=current['endpoint_checks_passed'] and all(current['checks'].values())
        initial=[];models=[];optimizers=[]
        for root in (OLD,RUN):
            initial.append(torch.load(root/arm/'INITIAL_MODEL.pt',map_location='cpu')['model'])
            with (root/arm/'checkpoints/endpoint.ckpt').open('rb') as f:payload=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']
            models.append(payload['model'])
            if measured_robot:optimizers.append(payload['optimizer'])
        if measured_robot:
            from scripts.sugar.demo_following.demo_future.generator_measured_robot import WEIGHT_KEY,EXTRA_KEY,extend_robot_state_dict,original_robot_slice
            expected=initial[0] if actual_state else extend_robot_state_dict(initial[0])
            checks[arm+'_full_initial_state_exact']=expected.keys()==initial[1].keys() and all(torch.equal(v,initial[1][n]) for n,v in expected.items())
            checks[arm+'_only_declared32_input_columns_added']=set(models[1])==set(models[0])|{EXTRA_KEY} and all(v.shape==models[1][n].shape for n,v in models[0].items()) and models[1][EXTRA_KEY].shape==(256,32)
            binding=json.loads((RUN/arm/'MEASURED_ROBOT_ADAM_BINDING.json').read_text())
            checks[arm+'_full8327408_and_new_coefficients_trained']=binding['full_parameter_count']==8327408 and bool(models[1][EXTRA_KEY].count_nonzero())
            old_binding=(json.loads((OLD/arm/'MEASURED_ROBOT_ADAM_BINDING.json').read_text())['parameters'] if actual_state else json.loads((preflight_root/'GENERATED_CPU_PREFLIGHT.json').read_text())['optimizer_bindings'][arm+'_initial']['old'])
            if actual_state:
                checks[arm+'_same_full8327408_architecture']=models[0].keys()==models[1].keys() and all(v.shape==models[1][n].shape for n,v in models[0].items())
                checks.pop(arm+'_only_declared32_input_columns_added')
                expert=json.loads((RUN/arm/'EXPERT_STATE_EXPOSURES.json').read_text())
                checks[arm+'_all355_actual_states73728_exposures']=expert['checks_passed'] and len(expert['actual_counts'])==355 and sum(expert['actual_counts'])==73728
                checks[arm+'_all512_expert_clocks']=all(v['expert_step']==i and v['expert_rows']==144 and v['expert_seed']==272500+i and v['expert_weight']==.1 for i,v in enumerate(records))
            checks[arm+'_same_official_adam_group_settings']=all({k:v for k,v in a.items() if k!='params'}=={k:v for k,v in b.items() if k!='params'} for a,b in zip(optimizers[0]['param_groups'],optimizers[1]['param_groups']))
            checks[arm+'_all_full_adam_slot_shapes_and_clocks']=all(v['parameter_id'] in optimizer['state'] and all(optimizer['state'][v['parameter_id']][k].shape==model[v['name']].shape and bool(torch.isfinite(optimizer['state'][v['parameter_id']][k]).all()) for k in ('exp_avg','exp_avg_sq')) and int(optimizer['state'][v['parameter_id']]['step'])==512 for model,optimizer,bindings in zip(models,optimizers,[old_binding,binding['parameters']]) for v in bindings)
        else:
            checks[arm+'_full_initial_state_exact']=initial[0].keys()==initial[1].keys() and all(torch.equal(v,initial[1][n]) for n,v in initial[0].items())
            checks[arm+'_endpoint_full_architecture_unchanged']=models[0].keys()==models[1].keys() and all(v.shape==models[1][n].shape for n,v in models[0].items())
        checks[arm+'_frozen_normalizer_exact']=all(torch.equal(v,models[1][n]) for n,v in models[0].items() if n.startswith('normalizer.'))
    checks['both_arms_same_auxiliary_clocks_and_rows']=all(all(a[k]==b[k] for k in ('generated_weight','generated_time_index','generated_time','generated_step','generated_rows')) for a,b in zip(losses['zero_context'],losses['demo_geometry']))
    checks['both_arms_auxiliary_nonzero']=all(any(row['generated']>0 for row in losses[arm]) for arm in losses)
    rows={}
    for phase in endpoints[0]['phases']:
        old,new=[r['phases'][phase] for r in endpoints]
        for variant in plans[1]['frozen_evaluation']['variants']:
            with np.load(OLD/f'frozen_evaluation/phase_{phase}/{variant}.npz') as a,np.load(RUN/f'frozen_evaluation/phase_{phase}/{variant}.npz') as b:
                checks[f'{phase}_{variant}_same_target_and_seeds']=np.array_equal(a['actual_future_command_target'],b['actual_future_command_target']) and np.array_equal(a['sample_seeds'],b['sample_seeds'])
                if variant=='initial_shared_goal_zero':checks[phase+'_full_initial_samples_exact']=np.array_equal(a['predictions'],b['predictions'])
        a,b=[r['variants']['trained_demo_geometry_correct'] for r in (old,new)]
        rows[phase]=dict(split=new['evaluation_split'],old_mse=a['normalized_mse'],new_mse=b['normalized_mse'],new_over_old_mse=b['normalized_mse']/a['normalized_mse'],old_correct_draws=a['per_branch_correct_preference_draws'],new_correct_draws=b['per_branch_correct_preference_draws'],old_pass=old['branch_identifiability_passed'],new_pass=new['branch_identifiability_passed'],old_zero_mse=old['variants']['trained_zero_context']['normalized_mse'],new_zero_mse=new['variants']['trained_zero_context']['normalized_mse'])
    if gap_coverage:
        for phase,new in endpoints[1]['phases'].items():
            if phase in rows:continue
            a=new['variants']['trained_demo_geometry_correct'];rows[phase]=dict(split=new['evaluation_split'],old_mse=None,new_mse=a['normalized_mse'],new_over_old_mse=None,old_correct_draws=None,new_correct_draws=a['per_branch_correct_preference_draws'],old_pass=False,new_pass=new['branch_identifiability_passed'],old_zero_mse=None,new_zero_mse=new['variants']['trained_zero_context']['normalized_mse'])
    decision=dict(all_train_pass=all(endpoints[1]['training_phase_checks'].values()),both_reused_checks_pass=all(endpoints[1]['heldout_phase_checks'].values()),previous_passing_train_mse_not_worse={p:v['new_mse']<=v['old_mse'] for p,v in rows.items() if v['split']=='train' and v['old_pass']})
    decision['all_prediction_requirements_passed']=all(checks.values()) and decision['all_train_pass'] and decision['both_reused_checks_pass'] and all(decision['previous_passing_train_mse_not_worse'].values())
    decision['all_nine_train_pass' if gap_coverage or self_replay or measured_robot else 'all_seven_train_pass']=decision['all_train_pass']
    report=dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,decision=decision,
        new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0,
        comparison_scope=plans[1]['comparison_scope'],
        scope=('Botharms use identical18case144row replay supervision. Old14actual/replay and check arrays exact; fullrelease initialization/normalizer/architecture/update budget retained. Batch112to144 and rows57344to73728 are explicitly different, oldbatch RNG and zeroendpoint need not match. All common9phase targets/seeds/initial samples exact; new2TRAIN phases reported without invented oldmodel results. No equalFLOPs, untouched-test or generatedphysics/SMP claim.' if gap_coverage else 'Botharms have the same frozenfull025 TRAIN replay and fixed0.1actualtarget auxiliary. Zero endpoint and samples are intentionally allowed to change. Exact original initial state/batch/budget/normalizer and full current model+Adam integrity required. Teacher and replay cost are additional. SameknownTRAIN sources/reused phase checks, not independent generalization or generatedphysics/SMP benefit.'))
    if self_replay:
        report['scope']='Botharms use identical refreshed18case144row frozen ranked18 teacher replay. Actual labels/initialGaussian/seeds/times/.25rank+.1aux/release initialization/batches/budgets/normalizer unchanged. Complete new model+Adam and exposure checks required. Zero endpoint legitimately changes because its auxiliary inputs change. Teacher and replay compute additional; all11phase original criteria and old passing TRAIN nonregression retained. No independent-motion, generatedphysics or SMP benefit claim.'
    if measured_robot:
        report['scope']='Only existing robot affine input36to68 with8192new coefficients. Allold release weights, frozen normalizer,18TRAIN/144batch,seed272084,q/rank/replay objective and fixedteacher, both512 and all11phase5condition32draws retained. Initial projected fullstate and allsaved initial32draws exact; every fullAdam moment bound. New measured32 fields shared by bothcontextarms; zero learned endpoint need not equal predecessor. Require9of9TRAIN, reused2checks and all8oldpassingTRAIN MSE nonregression. No fullstate sufficiency, new motion, generatedphysics or SMP benefit claim.'
    if actual_state:
        checks['same_measured_robot_architecture_settings']=plans[0]['robot_state_conditioning']==plans[1]['robot_state_conditioning']
        checks['both_actual_state_preflights']=all(json.loads((RUN/name).read_text())['checks_passed'] for name in ('ACTUAL_STATE_CPU_PREFLIGHT.json','ACTUAL_STATE_BF16_PREFLIGHT.json'))
        checks['both_arms_same_expert_exposures']=json.loads((RUN/'zero_context/EXPERT_STATE_EXPOSURES.json').read_text())['actual_counts']==json.loads((RUN/'demo_geometry/EXPERT_STATE_EXPOSURES.json').read_text())['actual_counts']
        report['scope']='Same full8327408 architecture, release initialization, old normalization, old18case144batch, teacher/replay/q/rank and both512. Additional355actual visited-state inputs with desired expert8x36 plans,144extra rows/update,.1q correction BOTHarms. Labels are known expert proposals, not counterfactual physical futures or proven recoveries. All11fixedphase criteria and8oldpassingTRAIN nonregression retained. Actual rollout improvement remains separate; no equalFLOPs, independent generalization or SMP benefit claim.'
        report['checks_passed']=all(checks.values())
        decision['all_prediction_requirements_passed']=all(checks.values()) and decision['all_train_pass'] and decision['both_reused_checks_pass'] and all(decision['previous_passing_train_mse_not_worse'].values())
    write(out,report)
    assert all(checks.values())
    print(json.dumps(dict(checks_passed=True,phases=rows,decision=decision)),flush=True)


if __name__=='__main__':main()
