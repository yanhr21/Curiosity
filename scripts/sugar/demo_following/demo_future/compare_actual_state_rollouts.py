"""Compare actual expert-supervised rollouts with the preceding full models."""
import json

import numpy as np

from scripts.sugar.demo_following.demo_future.run_generator_actual_state_supervision import RUN,BASE,MODEL,write
from scripts.sugar.smp.sugar_g1_box_schema import TRACKED_BODY_NAMES


def compare_condition(old,new):
    plans=[json.loads((r/'PROTOCOL.json').read_text()) for r in (old,new)]
    reports=[json.loads((r/'GENERATED_EXECUTION_READBACK.json').read_text()) for r in (old,new)]
    assert all(x['checks_passed'] and x['physical_plots_inspected'] for x in reports)
    checks={k:plans[0][k]==plans[1][k] for k in ('seed','steps_per_arm','switch_control_frame','reference_pair','physical_condition','alignment','reference_alignment','checkpoint_origin')}
    a,b=[p['generated_rollout'] for p in plans]
    checks['same_context_condition']=a.get('generator_condition_mode','demo_geometry')==b['generator_condition_mode']
    checks['declared_model_pair']=a['generator_run']==str(MODEL) and b['generator_run']==str(RUN)
    checks['same_generation_interface']=all(a[k]==b[k] for k in ('handoff_control_frame','sample_seed_base','sample_seed_rule','known_position_fields','history','geometry','goal'))
    switch=plans[0]['switch_control_frame'];assert switch==158
    rows={}
    for arm in ('original','repeat','alternate'):
        traces=[];starts=[];physical=[]
        for root in (old,new):
            with np.load(root/arm/'TRACE.npz') as f:traces.append({k:f[k].copy() for k in f.files})
            with np.load(root/arm/'STARTUP.npz') as f:starts.append({k:f[k].copy() for k in f.files})
            physical.append(json.loads((root/arm/'PROTOCOL.json').read_text()))
        checks[arm+'_startup_exact']=starts[0].keys()==starts[1].keys() and all(np.array_equal(v,starts[1][k]) for k,v in starts[0].items())
        fields=('robot_body_state_before_w','object_state_before_w','joint_pos_before','joint_vel_before','teacher_observation','requested_action')
        checks[arm+'_all_actual_prefix_states_actions_exact']=all(np.array_equal(traces[0][k][:switch],traces[1][k][:switch]) for k in fields)
        checks[arm+'_same_physics_reference_and_tracker']=all(physical[0][k]==physical[1][k] for k in ('physics','initial_reference_frame','source_lengths','source_ids','checkpoint','position_feedback_source'))
        end=min(len(x['done']) for x in traces);valid=np.arange(end)>=switch
        for x in traces:valid &= ~x['done'][:end].reshape(-1)
        names=starts[0]['body_names'].tolist();body_ids=[names.index(n) for n in TRACKED_BODY_NAMES]
        values=[]
        for index,x in enumerate(traces):
            item=reports[index]['arms'][arm]
            box=x['object_state_before_w'][:end,0,:3]-x['reference_object_pos_w'][:end,0]
            body=x['robot_body_state_before_w'][:end,0][:,body_ids,:3]-x['reference_body_pos_w'][:end,0]
            rms=lambda delta:float(np.sqrt(np.mean(delta[valid]**2))) if valid.any() else None
            values.append(dict(actual_steps=item['actual_steps'],generated_steps=item['generated_steps'],
                full400_pass=item['full_budget_without_reset'],terminal_terms=item['terminal_terms'],
                common_box_coordinate_rmse_m=rms(box),common_tracked_body_coordinate_rmse_m=rms(body),
                own_entire_valid_post_handoff_box_rmse_m=item['post_handoff_box_coordinate_rmse_m']))
        rows[arm]=dict(old=values[0],new=values[1],common_valid_post_handoff_frames=int(valid.sum()),
            generated_step_delta=values[1]['generated_steps']-values[0]['generated_steps'])
    decision=dict(all_three_full400=all(x['new']['full400_pass'] for x in rows.values()),
        both_distinct_branches_survive_longer=all(rows[a]['generated_step_delta']>0 for a in ('original','alternate')),
        neither_distinct_branch_survival_regresses=all(rows[a]['generated_step_delta']>=0 for a in ('original','alternate')))
    return dict(checks_passed=all(checks.values()),checks=checks,arms=rows,decision=decision)


def main():
    out=RUN/'MATCHED_ROLLOUT_COMPARISON.json';assert not out.exists()
    assert json.loads((RUN/'frozen_evaluation/LATENT_REPLAY_COMPARISON.json').read_text())['checks_passed']
    pairs={'demo':('measured32_generated_command_rollout158_r1','actual_state_expert_generated_demo158'),
           'zero':('measured32_generated_zero_rollout158','actual_state_expert_generated_zero158')}
    rows={k:compare_condition(BASE/a,BASE/b) for k,(a,b) in pairs.items()}
    context=json.loads((BASE/'actual_state_expert_generated_demo158/MATCHED_ZERO_COMPARISON.json').read_text())
    report=dict(execution_completed=True,checks_passed=all(x['checks_passed'] for x in rows.values()) and context['checks_passed'],
        conditions=rows,new_context_comparison=context['decision'],new_optimizer_updates=0,new_physics_steps=0,
        scope='Same complete models and paired physical protocol; only added actual-state expert supervision. Both old/new failures retained. Common-window errors exclude terminal frames and are unavailable for empty overlap. Separate whole-episode durations remain primary, repeats are not independent seeds. Known48position assistance and trained-state reuse remain; no independent generalization or SMP benefit claim.')
    write(out,report);print(json.dumps(report),flush=True);assert report['checks_passed']


if __name__=='__main__':main()
