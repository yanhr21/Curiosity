"""Strict full-model matched ranking presence/absence endpoint comparison."""
import json
import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import BASE,write

OLD=BASE/'matched_generator_branch_latent_replay01_gaps512'
RUN=BASE/'matched_generator_branch_latent_only_gaps512'


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return isinstance(b,(tuple,list)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    out=RUN/'frozen_evaluation/RANK_ABLATION_COMPARISON.json';assert not out.exists()
    plans=[json.loads((p/'PROTOCOL.json').read_text()) for p in (OLD,RUN)]
    ends=[json.loads((p/'RESULT.json').read_text()) for p in (OLD,RUN)]
    assert all(e['execution_completed'] and e['horizon_plots_inspected'] for e in ends)
    readback=json.loads((RUN/'TRAINING_LOSS_READBACK.json').read_text());assert readback['checks_passed']
    stable=('branch_dataset','batch_size','data','phase_corpora','normalizer_state','seed','epochs','actual_optimizer_updates','optimizer','scheduler','frozen_evaluation')
    checks={key:plans[0][key]==plans[1][key] for key in stable}
    paired=dict(plans[1]['paired_objective']);ablation=paired.pop('ablation');paired['weight']=.25
    generated=dict(plans[1]['generated_state_objective']);generated.pop('paired_rank_ablation')
    checks['only_rank_presence_changes']=ablation=='rank_removed_on_real18_replay' and plans[1]['paired_objective']['weight']==0 and paired==plans[0]['paired_objective'] and generated==plans[0]['generated_state_objective']
    checks['all_training_logs_exposures_checked']=readback['checks_passed']
    for arm in ('zero_context','demo_geometry'):
        checks[arm+'_actual_batches_exact']=(OLD/arm/'BATCH_ORDER.json').read_text()==(RUN/arm/'BATCH_ORDER.json').read_text()
        checks[arm+'_actual_budgets_exact']=(OLD/arm/'DATA_AND_BUDGET.json').read_text()==(RUN/arm/'DATA_AND_BUDGET.json').read_text()
        initial=[torch.load(p/arm/'INITIAL_MODEL.pt',map_location='cpu')['model'] for p in (OLD,RUN)]
        checks[arm+'_full_initial_state_exact']=equal(*initial)
        models=[]
        for p in (OLD,RUN):
            with (p/arm/'checkpoints/endpoint.ckpt').open('rb') as stream:models.append(torch.load(stream,pickle_module=dill,map_location='cpu')['state_dicts'])
        a,b=[m['model'] for m in models]
        checks[arm+'_full_architecture_preserved']=a.keys()==b.keys() and all(v.shape==b[k].shape for k,v in a.items())
        checks[arm+'_normalizer_exact']=all(torch.equal(v,b[k]) for k,v in a.items() if k.startswith('normalizer.'))
        if arm=='zero_context':
            checks['zero_full_model_exact']=equal(a,b);checks['zero_full_adam_exact']=equal(models[0]['optimizer'],models[1]['optimizer'])
            losses=[json.loads((p/arm/'PAIRED_LOSSES.json').read_text()) for p in (OLD,RUN)]
            checks['zero_all512_actual_losses_exact']=all(all(x[k]==y[k] for k in ('base','rank','generated','total','generated_time','generated_step')) for x,y in zip(*losses))
    rows={}
    for phase in ends[0]['phases']:
        for variant in plans[0]['frozen_evaluation']['variants']:
            with np.load(OLD/f'frozen_evaluation/phase_{phase}/{variant}.npz') as a,np.load(RUN/f'frozen_evaluation/phase_{phase}/{variant}.npz') as b:
                checks[phase+'_'+variant+'_target_seeds_exact']=np.array_equal(a['actual_future_command_target'],b['actual_future_command_target']) and np.array_equal(a['sample_seeds'],b['sample_seeds'])
                if variant in ('initial_shared_goal_zero','trained_zero_context'):checks[phase+'_'+variant+'_all32_samples_exact']=np.array_equal(a['predictions'],b['predictions'])
        a,b=[e['phases'][phase] for e in ends];va,vb=[e['variants']['trained_demo_geometry_correct'] for e in (a,b)]
        rows[phase]=dict(split=b['evaluation_split'],old_pass=a['branch_identifiability_passed'],new_pass=b['branch_identifiability_passed'],old_mse=va['normalized_mse'],new_mse=vb['normalized_mse'],new_over_old_mse=vb['normalized_mse']/va['normalized_mse'],old_correct_draws=va['per_branch_correct_preference_draws'],new_correct_draws=vb['per_branch_correct_preference_draws'])
    decision=dict(all_nine_train_pass=all(ends[1]['training_phase_checks'].values()),both_reused_checks_pass=all(ends[1]['heldout_phase_checks'].values()),previous_passing_train_mse_not_worse={k:v['new_mse']<=v['old_mse'] for k,v in rows.items() if v['split']=='train' and v['old_pass']})
    decision['all_prediction_requirements_passed']=all(checks.values()) and decision['all_nine_train_pass'] and decision['both_reused_checks_pass'] and all(decision['previous_passing_train_mse_not_worse'].values())
    write(out,dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,phases=rows,decision=decision,new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0,scope=plans[1]['comparison_scope']))
    assert all(checks.values())
    print(json.dumps(dict(checks=len(checks),decision=decision,phases=rows)),flush=True)


if __name__=='__main__':main()
