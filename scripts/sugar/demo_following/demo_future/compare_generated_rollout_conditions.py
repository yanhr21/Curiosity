"""Matched complete frozen-model actual closed-loop comparison, no new physics."""
import json
import argparse
from pathlib import Path
import numpy as np

from scripts.sugar.demo_following.demo_future.run_generated_command_rollout import BASE,write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-supervision',action='store_true')
    parser.add_argument('--demo-run',type=Path,default=BASE/'measured32_generated_command_rollout158_r1')
    parser.add_argument('--zero-run',type=Path,default=BASE/'measured32_generated_zero_rollout158')
    args=parser.parse_args()
    demo=args.demo_run.resolve();zero=args.zero_run.resolve()
    assert demo.parent==zero.parent==BASE
    if args.prepare_supervision:
        assert demo.name=='measured32_generated_command_rollout158_r1' and zero.name=='measured32_generated_zero_rollout158'
    out=demo/'MATCHED_ZERO_COMPARISON.json';assert not out.exists()
    reports=[json.loads((r/'GENERATED_EXECUTION_READBACK.json').read_text()) for r in (demo,zero)]
    assert all(r['checks_passed'] and r['physical_plots_inspected'] for r in reports)
    plans=[json.loads((r/'PROTOCOL.json').read_text()) for r in (demo,zero)]
    checks={k:plans[0][k]==plans[1][k] for k in ('seed','steps_per_arm','switch_control_frame','reference_pair','physical_condition','alignment','reference_alignment','checkpoint_origin')}
    a,b=[p['generated_rollout'] for p in plans]
    checks['same_generator_training_run']=a['generator_run']==b['generator_run']
    checks['only_context_arm_selected']=a.get('generator_condition_mode','demo_geometry')=='demo_geometry' and b['generator_condition_mode']=='zero_context'
    checks['same_seed_handoff_history_position_interface']=all(a[k]==b[k] for k in ('handoff_control_frame','sample_seed_base','sample_seed_rule','known_position_fields','history','geometry','goal'))
    rows={}
    for arm in ('original','repeat','alternate'):
        arrays=[];starts=[];protocols=[]
        for root in (demo,zero):
            with np.load(root/arm/'TRACE.npz') as f:arrays.append({k:f[k].copy() for k in f.files})
            with np.load(root/arm/'STARTUP.npz') as f:starts.append({k:f[k].copy() for k in f.files})
            protocols.append(json.loads((root/arm/'PROTOCOL.json').read_text()))
        checks[arm+'_startup_exact']=starts[0].keys()==starts[1].keys() and all(np.array_equal(v,starts[1][k]) for k,v in starts[0].items())
        fields=('robot_body_state_before_w','object_state_before_w','joint_pos_before','joint_vel_before','teacher_observation','requested_action')
        checks[arm+'_common_158prefix_exact']=all(np.array_equal(arrays[0][k][:158],arrays[1][k][:158]) for k in fields)
        checks[arm+'_same_physics_and_reference']=all(protocols[0][k]==protocols[1][k] for k in ('physics','initial_reference_frame','source_lengths','source_ids','checkpoint','position_feedback_source'))
        end=min(len(v['done']) for v in arrays);mask=np.arange(end)>=158
        for x in arrays:mask &= ~x['done'][:end].reshape(-1)
        values=[]
        for index,x in enumerate(arrays):
            own=x['object_state_before_w'][:end,0,:3]-x['reference_object_pos_w'][:end,0]
            values.append(dict(generated_steps=reports[index]['arms'][arm]['generated_steps'],actual_steps=reports[index]['arms'][arm]['actual_steps'],
                terminal_terms=reports[index]['arms'][arm]['terminal_terms'],full_budget=reports[index]['arms'][arm]['full_budget_without_reset'],
                common_post_handoff_box_coordinate_rmse_m=float(np.sqrt(np.mean(own[mask]**2))) if mask.any() else None))
        rows[arm]=dict(demo=values[0],zero=values[1],common_valid_post_handoff_frames=int(mask.sum()),
                       demo_more_generated_controls=values[0]['generated_steps']>values[1]['generated_steps'])
    decision=dict(both_distinct_demo_branches_survive_longer=all(rows[a]['demo_more_generated_controls'] for a in ('original','alternate')),
                  demo_all400_pass=all(v['demo']['full_budget'] for v in rows.values()),zero_all400_pass=all(v['zero']['full_budget'] for v in rows.values()),
                  permits_actual_state_expert_supervision_preparation=all(checks.values()))
    report=dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,arms=rows,decision=decision,
        new_physics_steps=0,new_optimizer_updates=0,scope='Matched trained zero/demo full8327408 Generators with the same full607 Tracker, positions, nominal worlds and sample schedule. Repeats are deterministic controls, not independent seeds. Duration gains do not reverse failed400step endpoints or remove48knownposition assistance. No SMP intervention or benefit claim.')
    write(out,report);print(json.dumps(report),flush=True);assert all(checks.values())
    if args.prepare_supervision:
        from scripts.sugar.demo_following.demo_future.prepare_generated_state_supervision import main as prepare
        prepare()


if __name__=='__main__':main()
