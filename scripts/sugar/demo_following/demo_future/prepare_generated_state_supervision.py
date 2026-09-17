"""Actual generated-state observations with separately identified expert reference plans.

Targets are the known selected expert command plan, not counterfactual physical
states or futures actually executed by the generated controller. No new model.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.sugar.demo_following.demo_future.run_generated_command_rollout import BASE,BASELINE,MODEL,write

OUT=BASE/'actual_generated_state_expert_supervision_v1'
FIELDS=('obj_pos_b','obj_ori_b','joint_pos','project_gravity','target_obj_pos_b','target_obj_ori_b','last_action','original_demo_geometry')


def main():
    assert not OUT.exists()
    comparison=json.loads((BASE/'measured32_generated_command_rollout158_r1/MATCHED_ZERO_COMPARISON.json').read_text())
    assert comparison['checks_passed'] and comparison['decision']['permits_actual_state_expert_supervision_preparation']
    model_plan=json.loads((MODEL/'PROTOCOL.json').read_text())
    excluded={s:set() for s in (96,90)}
    for row in model_plan['data']['heldout_phase']['samples']:excluded[row['source']].update(row['source_frames'])
    groups=[];all_obs={k:[] for k in FIELDS};targets=[];rows=[];checks={};exclusions={};hashes={}
    for mode,folder in [('demo_geometry','measured32_generated_command_rollout158_r1'),('zero_context','measured32_generated_zero_rollout158')]:
        run=BASE/folder
        result=json.loads((run/'GENERATED_EXECUTION_READBACK.json').read_text())
        assert result['checks_passed'] and result['physical_plots_inspected']
        for arm,source in [('original',96),('alternate',90)]:
            group=mode+'_'+arm
            with np.load(run/arm/'TRACE.npz') as f:x={k:f[k].copy() for k in f.files}
            with np.load(BASELINE/arm/'TRACE.npz') as f:baseline={k:f[k].copy() for k in f.files}
            with np.load(BASE/f'refiner_aligned96_90_feasibility/{arm}/TRACE.npz') as f:raw_clock=f['reference_frame'].copy()
            hashes[group]=hashlib.sha256((run/arm/'TRACE.npz').read_bytes()).hexdigest()
            valid=~x['done'].reshape(-1);selected=[];rejected=[]
            for frame in x['generator_control_frame'].tolist():
                idx=frame-158
                if frame==158:
                    rejected.append(dict(frame=frame,reason='already fitted shared-world handoff anchor'));continue
                if not valid[frame]:
                    rejected.append(dict(frame=frame,reason='terminal transition excluded'));continue
                clocks=raw_clock[frame+5*np.arange(8)]
                if set(map(int,clocks)) & excluded[source]:
                    rejected.append(dict(frame=frame,reason='overlaps reused218/258 expert-target source clocks'));continue
                label=x['planned_reference_command'][frame,0]
                assert label.shape==(8,36) and np.array_equal(label,baseline['planned_reference_command'][frame,0])
                assert int(x['reference_frame'][frame])==frame
                obs={k:x['generator_input_'+k][idx,0].copy() for k in FIELDS}
                assert np.array_equal(obs['joint_pos'][0],x['joint_pos_before'][frame,0])
                assert np.array_equal(obs['project_gravity'][0],x['project_gravity'][frame,0])
                assert np.array_equal(obs['last_action'][0],x['issued_command'][frame-5,0])
                for k,v in obs.items():all_obs[k].append(v)
                targets.append(label.copy());selected.append(len(rows))
                rows.append(dict(group=group,rollout_mode=mode,arm=arm,source=source,control_frame=frame,
                    expert_source_frames=clocks.tolist(),trace=str(run/arm/'TRACE.npz'),
                    label_kind='known expert reference command proposal on measured generated-controller state; not executed future'))
            assert selected
            groups.append(dict(name=group,indices=selected,rows=len(selected),source=source,mode=mode))
            exclusions[group]=rejected
            checks[group+'_every_admitted_target_exact_existing_expert_plan']=True
            checks[group+'_measured_state_and_issued_lag5_exact']=True
            checks[group+'_no_reused_check_target_source_clocks']=all(not (set(rows[i]['expert_source_frames']) & excluded[source]) for i in selected)
    arrays={k:np.stack(v) for k,v in all_obs.items()};arrays['expert_future_command_target']=np.stack(targets)
    checks['all_arrays_finite']=all(np.isfinite(v).all() for v in arrays.values())
    checks['all_observation_shapes']=all(v.shape[:2]==(len(rows),1) for k,v in arrays.items() if k!='expert_future_command_target')
    checks['targets_full8x36']=arrays['expert_future_command_target'].shape==(len(rows),8,36)
    checks['four_unique_rollout_groups_no_repeat_reweighting']=len(groups)==4 and len({v['name'] for v in groups})==4
    assert all(checks.values())
    OUT.mkdir();np.savez_compressed(OUT/'EXPERT_PLAN_ON_ACTUAL_STATES.npz',**arrays)
    write(OUT/'PROTOCOL.json',dict(source_generator_run=str(MODEL),groups=groups,rows=rows,excluded_rows=exclusions,
        old_check_source_clocks={str(k):sorted(v) for k,v in excluded.items()},trace_sha256=hashes,
        label_scope='Immutable selected expert8x36 command proposal already present at the actual control call and exactly matching the completed known-plan baseline. It is supervision for desired commands on measured off-policy states. It is not the subsequent action sequence executed by the generated policy or a validated counterfactual physical future.',
        observation_scope='Current actual object pose, joint29/gravity3, real issued36command at t-5, original numeric-demo geometry; existing normalized9Dgoal zero intervention retained. No label in observation.',
        failure_scope='All four unique contributing episodes failed their400step budgets; use their valid preterminal measured states only. Repeats omitted. Original failed endpoints and denominators unchanged.',
        comparison_scope='Existing knownTRAIN96/90 sources; source-target clocks used by repeated218/258 checks excluded. Not untouched motion generalization.',
        next_training_hypothesis='Add a fixed0.1 independent-noise full-model denoising loss for these expert command proposals to BOTH original matched arms while preserving original18case144row q+.25rank+.1frozenreplay. Stratify additional144rows equally over four rollout groups, deterministic cyclic indices, independent preserved RNG. Same fresh release initialization/oldnormalizer/seed272084/both512new budgets. No new modules, no oldendpoint extension, no teacher refresh. FullCPU+actualBF16 preflight required before this separate training experiment.'))
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=True,checks=checks,examples=len(rows),groups=groups,
        new_model_modules=0,new_optimizer_updates=0,new_physics_steps=0,expert_recovery_in_simulator_proven=False,
        scope='Prepared verified desired expert-command supervision on actually visited generated states; not actual future-state labels or proof that the expert would recover.'))
    print(json.dumps(dict(examples=len(rows),groups=groups,checks_passed=True)),flush=True)


if __name__=='__main__':main()
