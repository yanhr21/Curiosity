"""Screen the two largest remaining TRAIN phase gaps using saved real states.

This uses the previously validated official termination reconstruction. It
neither simulates recovery nor creates any physical target or model.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scripts.sugar.demo_following.demo_future.audit_switch_admission import BASE, evaluate_switch_state

RUN=BASE/'matched_generator_branch_latent_replay01512'
DENSE=BASE/'generator_actual_branch_coverage_dense'
OUT=RUN/'frozen_evaluation/phase_gap_screen_r1'


def arrays(path):
    with np.load(path) as data:return {k:data[k].copy() for k in data.files}


def main():
    comparison=json.loads((RUN/'frozen_evaluation/LATENT_REPLAY_COMPARISON.json').read_text())
    coverage=json.loads((RUN/'frozen_evaluation/joint_condition_coverage/RESULT.json').read_text())
    denoising=json.loads((RUN/'frozen_evaluation/rank_component_audit/RESULT.json').read_text())
    assert comparison['checks_passed'] and comparison['decision']['all_seven_train_pass'] and not comparison['decision']['both_reused_checks_pass']
    assert coverage['checks_passed'] and coverage['all18_queries_and_54checks_inspected'] and denoising['checks_passed'] and denoising['plot_inspected']
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    train=plan['data']['train']['phases'];check=plan['data']['heldout_phase']['phases']
    gaps=sorted([(b-a,a,b) for a,b in zip(train,train[1:])],key=lambda g:(-g[0],g[1]))[:2]
    assert gaps==[(48,197,245),(32,245,277)]
    OUT.mkdir(exist_ok=False)
    protocol=dict(train_phases=train,reused_check_phases=check,selected_train_gaps=gaps,
        selection_rule='The two largest adjacent TRAIN control-phase gaps, ties lower left endpoint. For each, center=floor((left+right)/2), search integer phases within15controls and strictly within the gap. Exclude previously attempted phases including237/358, exact reused-check future-label source clocks, and static official pre-action termination failures on the saved unswitched real full607state. Choose nearest eligible tocenter, ties lower. No generated or target-error scores used for candidate ranking.',
        centers=[(a+b)//2 for _,a,b in gaps],static_only=True,
        automatic_next_action='Inspect full six-observation term/reference reconstruction, source clocks and all candidates. If one candidate exists for each gap, declare exactly two new actual frozenfull607 groups, original/repeat/alternate400steps each, maximum2400newcontrols,0optimizerupdates. Reuse9admitted groups; preserve237/358 so cumulative attempt denominator13. Complete formatter regression and all actualphysical/reference/XYZ checks before any new Generator data/replay budget. Static compatibility is never physical success.',
        new_optimizer_updates=0,new_physics_steps=0,new_sample_draws=0)
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    names=arrays(DENSE/'switch_237/original/STARTUP.npz')['body_names'].tolist()
    baseline=arrays(DENSE/'switch_318/original/TRACE.npz')
    bank=[]
    for folder in sorted((DENSE/'switch_237/motions').glob('data_*')):
        robot=arrays(folder/'robot_50hz.npz')
        with (folder/'obj_motion_global_50hz.pkl').open('rb') as f:obj=pickle.load(f)
        bank.append((robot,obj))
    old=json.loads((DENSE/'switch_admission_audit/RESULT.json').read_text())
    checks={};observed={}
    for phase in (197,237,277):
        for ai,arm in enumerate(('original','alternate')):
            trace=arrays(DENSE/f'switch_{phase}/{arm}/TRACE.npz')
            value,body,obj=evaluate_switch_state(trace,phase,ai,bank,names)
            key=f'{phase}_{arm}'
            actual=json.loads((DENSE/f'switch_{phase}/{arm}/SWITCH_TERMS_BEFORE_ACTION.json').read_text())
            checks[key+'_official_terms_exact']=value['terms']==actual
            checks[key+'_previous_static_reconstruction_matches']=value['terms']==old['observed'][key]['terms'] and value['all_terms_false']==old['observed'][key]['all_terms_false'] and all(np.allclose(value[field],old['observed'][key][field],rtol=0,atol=1e-12) for field in ('box_position_error_m','ee_position_errors_m','anchor_position_error_m','anchor_projected_gravity_z_error','object_orientation_error_rad'))
            checks[key+'_reference_arrays_match']=bool(np.allclose(body,trace['reference_body_pos_w'][phase,0],rtol=0,atol=2e-6) and np.allclose(obj,trace['reference_object_pos_w'][phase,0],rtol=0,atol=2e-6))
            checks[key+'_saved_unswitched_world_exact']=all(np.array_equal(trace[k][phase],baseline[k][phase]) for k in ('robot_body_state_before_w','object_state_before_w','joint_pos_before','joint_vel_before'))
            observed[key]=value
    provenance={}
    for arm,source in (('original',96),('alternate',90)):
        provenance[source]=arrays(BASE/f'refiner_aligned96_90_feasibility/{arm}/TRACE.npz')
    for split,info in plan['data'].items():
        for sample in info['samples']:
            f=sample['phase']+np.arange(8)*5;raw=provenance[sample['source']]
            checks[f"{split}_{sample['phase']}_{sample['source']}_actual_clocks_exact"]=raw['reference_frame'][f].tolist()==sample['source_frames'] and bool((raw['reference_id'][f]==sample['source']).all())
    used={source:{int(provenance[source]['reference_frame'][f]) for p in check for f in p+np.arange(8)*5} for source in provenance}
    attempted=set(train+check+[237,358]);selections={}
    for width,left,right in gaps:
        center=(left+right)//2;candidates={}
        for phase in range(max(left+1,center-15),min(right-1,center+15)+1):
            clock={source:list(map(int,v['reference_frame'][phase+np.arange(8)*5])) for source,v in provenance.items()}
            disjoint=all(not(set(frames)&used[source]) for source,frames in clock.items())
            terms={arm:evaluate_switch_state(baseline,phase,ai,bank,names)[0] for ai,arm in enumerate(('original','alternate'))}
            eligible=phase not in attempted and disjoint and all(v['all_terms_false'] for v in terms.values())
            candidates[str(phase)]=dict(previously_attempted=phase in attempted,source_clocks=clock,source_clocks_disjoint=disjoint,arms=terms,eligible=eligible)
        eligible=[int(p) for p,row in candidates.items() if row['eligible']]
        selections[str(center)]=dict(train_gap=[left,right],candidates=candidates,selected=min(eligible,key=lambda p:(abs(p-center),p)) if eligible else None)
    choices=[row['selected'] for row in selections.values()]
    passed=all(checks.values()) and all(p is not None for p in choices) and len(set(choices))==2
    report=dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,observed=observed,selections=selections,
        selected_new_phases=choices,permits_preparing_bounded_actual_collection=passed,
        prior_static_audit=str(RUN/'frozen_evaluation/phase_gap_screen/RESULT.json'),
        numerical_incident='Initial cross-host orientation reconstruction differs by at most7.3e-16rad. Preserve failed exact-equality audit; current numeric reconstruction tolerance1e-12, actual boolean terms exact and original2e-6reference-array check unchanged. No physical threshold, candidate rule, world state or model changed.',
        new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0,
        scope='Only static official-formula compatibility on saved real pre-action states and source-label clocks. Candidate physical following, sustained lift, full400steps and official export remain unproven; previous237/358failures are retained. SameknownTRAIN sources96/90. This does not deploy a predictor or use retrieval labels.')
    (OUT/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(checks.values())
    print(json.dumps(dict(checks=len(checks),passed=passed,selected=choices,selected_terms={center:row['candidates'][str(row['selected'])] if row['selected'] is not None else None for center,row in selections.items()})),flush=True)


if __name__=='__main__':main()
