"""Saved actual generated-command routing and physics evidence, without new physics."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

import numpy as np

from scripts.sugar.demo_following.demo_future.run_generated_command_rollout import RUN,BASELINE,MODEL,ROOT,BOOT,write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--actor-device',choices=['cpu','cuda'],default='cpu')
    options=parser.parse_args();run=options.run
    assert not (run/'GENERATED_EXECUTION_READBACK.json').exists()
    assert json.loads((run/'COLLECTION_COMPLETION.json').read_text())['execution_completed']
    children=[]
    for tag,module,args,artifact in [('pair','readback_refiner_pair',[],'READBACK.json'),
            ('following','readback_reference_following',[],'REFERENCE_FOLLOWING_READBACK.json'),
            ('xyz','readback_reference_following',['--xyz-only'],'REFERENCE_XYZ_READBACK.json')]:
        if (run/artifact).exists():
            assert json.loads((run/artifact).read_text())['execution_completed'];continue
        command=[sys.executable,str(BOOT),'scripts.sugar.demo_following.demo_future.'+module,'--run',str(run),*args]
        with (run/('readback_'+tag+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=tag,pid=child.pid,pgid=os.getpgid(child.pid));children.append(row);write(run/'READBACK_CHILDREN.json',children)
            row['returncode']=child.wait();write(run/'READBACK_CHILDREN.json',children)
        assert row['returncode']==0 and (run/artifact).exists(),row
    plan=json.loads((run/'PROTOCOL.json').read_text());switch=plan['switch_control_frame']
    model_run=Path(plan['generated_rollout']['generator_run'])
    import torch
    from rsl_rl.networks.mlp import MLP
    torch.set_num_threads(1)
    checkpoint=Path(plan['checkpoint_origin'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    actor=MLP(input_dim=846,output_dim=29,hidden_dims=[512,256,128],activation='elu')
    actor.load_state_dict({k.removeprefix('actor.'):v for k,v in payload['model_state_dict'].items() if k.startswith('actor.')},strict=True)
    actor.to(options.actor_device).eval().requires_grad_(False)
    checks={};arms={};traces={};hashes={}
    with np.load(run/'ORIGINAL_DEMO_CONTEXT.npz') as f: contexts=f['geometry'].copy()
    for arm in ('original','repeat','alternate'):
        path=run/arm/'TRACE.npz';hashes[arm]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f: x={k:f[k].copy() for k in f.files}
        traces[arm]=x
        result=json.loads((run/arm/'RESULT.json').read_text())
        live=json.loads((run/arm/'HANDOFF_INPUT_CHECK.json').read_text())
        count=len(x['done']);frames=np.arange(switch,count);g=len(frames)
        with np.load(BASELINE/arm/'TRACE.npz') as f: baseline={k:f[k].copy() for k in f.files}
        checks[arm+'_full_frozen_generator_tracker']=result['teacher_frozen']['passed'] and result['generator_frozen']['passed']
        checks[arm+'_actual_handoff_input_match']=live['passed']
        checks[arm+'_declared_budget_and_generated_clock']=0<count<=400 and result['generated_control_steps']==g and np.array_equal(x['generator_control_frame'],frames)
        checks[arm+'_generated_flag_exact']=np.array_equal(x['generated_control'],np.arange(count)>=switch)
        checks[arm+'_same_seed_sequence']=np.array_equal(x['generator_seed'],272090+np.arange(g))
        checks[arm+'_current_command_generated']=np.array_equal(x['generated_actor_input'][:,:,:36],x['generator_dense_plan'][:,:,0]) and np.array_equal(x['issued_command'][switch:],x['generator_dense_plan'][:,:,0])
        checks[arm+'_future288_generated']=np.array_equal(x['generated_actor_input'][:,:,510:798],x['generator_dense_plan'][:,:,::5].reshape(g,1,288))
        checks[arm+'_actual474_preserved']=np.array_equal(x['generated_actor_input'][:,:,36:510],x['teacher_observation'][switch:,:,36:])
        checks[arm+'_known48_preserved']=np.array_equal(x['generated_actor_input'][:,:,798:],x['actor_reference_position_feedback'][switch:])
        checks[arm+'_exact_causal_issued_lag5']=np.array_equal(x['generator_input_last_action'][:,:,0],x['issued_command'][frames-5])
        checks[arm+'_actual_measured_joint_gravity']=np.array_equal(x['generator_input_joint_pos'][:,:,0],x['joint_pos_before'][switch:]) and np.array_equal(x['generator_input_project_gravity'][:,:,0],x['project_gravity'][switch:])
        branch=int(arm=='alternate')
        checks[arm+'_original_demo_geometry_exact']=np.array_equal(x['generator_input_original_demo_geometry'].reshape(g,8,21),contexts[branch,frames])
        # The prefix is new physics and must reproduce the existing nominal baseline.
        fields=('robot_body_state_before_w','object_state_before_w','joint_pos_before','joint_vel_before','requested_action','teacher_observation')
        prefix={k:float(np.max(np.abs(x[k][:switch]-baseline[k][:switch]))) for k in fields}
        checks[arm+'_old_baseline_common_prefix']=all(v<=1e-5 for v in prefix.values())
        valid=~x['done'].reshape(-1)
        checks[arm+'_requested_actions_executed']=np.array_equal(x['requested_action'][valid],x['executed_action'][valid])
        with torch.inference_mode():
            inputs=torch.as_tensor(x['generated_actor_input'][:,0],device=options.actor_device)
            replay=(torch.cat([actor(row[None]) for row in inputs]) if options.actor_device=='cuda' else actor(inputs)).cpu().numpy()
        actor_error=float(np.max(np.abs(replay-x['requested_action'][switch:,0])))
        checks[arm+'_all_generated_full_actor_outputs_reproduced']=actor_error<=(0. if options.actor_device=='cuda' else 1e-5)
        mode=plan['generated_rollout'].get('generator_condition_mode','demo_geometry')
        variant='trained_demo_geometry_correct' if mode=='demo_geometry' else 'trained_zero_context'
        with np.load(model_run/f'frozen_evaluation/phase_158/{variant}.npz') as f:
            predictions=f['predictions'];seeds=f['sample_seeds']
        seed_index=list(seeds).index(272090)
        # Fixed evaluation stores draws x branches x horizon x channels.
        expected=predictions[seed_index,branch]
        sample_error=float(np.max(np.abs(x['generator_prediction'][0,0]-expected)))
        checks[arm+'_first_draw_matches_fixed_evaluation']=sample_error<=1e-4
        terminal=run/arm/'TERMINAL_TERMS.json'
        terms=json.loads(terminal.read_text()) if terminal.exists() else {}
        mask=valid & (np.arange(count)>=switch)
        box=x['object_state_before_w'][:,0,:3];ref=x['reference_object_pos_w'][:,0]
        arms[arm]=dict(actual_steps=count,generated_steps=g,post_handoff_valid_frames=int(mask.sum()),
            full_budget_without_reset=result['full_budget_without_reset'],terminal_terms=terms,
            first_draw_max_error=sample_error,baseline_prefix_max_errors=prefix,
            full_actor_all_generated_outputs_max_error=actor_error,
            post_handoff_box_coordinate_rmse_m=float(np.sqrt(np.mean((box[mask]-ref[mask])**2))) if mask.any() else None,
            maximum_lift_m=result['maximum_lift_m'],trace_sha256=hashes[arm])
    a,b=traces['original'],traces['repeat']
    checks['repeat_all_recorded_arrays_exact']=a.keys()==b.keys() and all(np.array_equal(v,b[k]) for k,v in a.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(15,10))
    for row,arm in enumerate(('original','repeat','alternate')):
        x=traces[arm];valid=~x['done'].reshape(-1);time=np.arange(len(valid))*.02
        axes[row,0].plot(time[valid],np.linalg.norm(x['object_state_before_w'][valid,0,:3]-x['reference_object_pos_w'][valid,0],axis=-1))
        axes[row,1].plot(time[valid],np.sqrt(np.mean((x['issued_command'][valid,0,:29]-x['reference_command'][valid,0,:29])**2,axis=-1)))
        gvalid=valid[switch:]
        axes[row,2].plot(time[switch:][gvalid],np.sqrt(np.mean((x['requested_action'][switch:,0][gvalid]-x['known_reference_action_same_world'][:,0][gvalid])**2,axis=-1)))
        for column,title in enumerate(('Box/reference distance (m)','Generated/reference joint command RMS (rad)','Generated/known-plan actor action RMS')):
            ax=axes[row,column];ax.set_title(arm+': '+title);ax.axvline(switch*.02,color='gray',ls='--');ax.set_xlabel('Actual control time (s)');ax.grid(alpha=.2)
    fig.suptitle('Actual generated-command diagnostic; known48position assistance; terminal/reset states excluded')
    fig.tight_layout();fig.savefig(run/'GENERATED_COMMAND_DEVIATION.png',dpi=160);plt.close(fig)
    report=dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,arms=arms,
        actual_new_control_steps=sum(v['actual_steps'] for v in arms.values()),
        actual_generated_control_steps=sum(v['generated_steps'] for v in arms.values()),
        all_three_full_budgets=all(v['full_budget_without_reset'] for v in arms.values()),
        original_physical_pair_checks=json.loads((run/'READBACK.json').read_text())['checks'],
        original_reference_following_checks=json.loads((run/'REFERENCE_FOLLOWING_READBACK.json').read_text())['checks'],
        physical_plots_inspected=False,actor_replay_device=options.actor_device,actor_replay_batch='actual batch1' if options.actor_device=='cuda' else 'batched CPU diagnostic',new_optimizer_updates=0,new_physics_steps_in_readback=0,
        scope='Actual current36+future288 generated-command execution, with unchanged known48position assistance, TRAIN96/90 phase158 and one seed. Original400step denominator and terminal failures retained. Not full prediction-floor success, independent deployment, video or SMP benefit.')
    write(run/'GENERATED_EXECUTION_READBACK.json',report)
    print(json.dumps(report),flush=True)
    assert all(checks.values()),{k:v for k,v in checks.items() if not v}


if __name__=='__main__':main()
