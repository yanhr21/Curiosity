"""Bounded actual generated-command diagnostic, with declared known positions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
RUN = BASE / 'measured32_generated_command_rollout158'
MODEL = BASE / 'matched_generator_branch_measured_robot32512'
BASELINE = BASE / 'generator_actual_branch_coverage_gaps2/switch_158'
BOOT = ROOT / 'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def reference_math():
    spec = importlib.util.spec_from_file_location('original_reference_math', ROOT/'SUGAR/scripts/sugar_rl/process_refiner_rollout.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare():
    import numpy as np
    from scripts.sugar.demo_following.demo_future.prepare_generator_demo_context import geometry_context
    assert not RUN.exists()
    plan = json.loads((BASELINE/'PROTOCOL.json').read_text())
    assert plan['switch_control_frame'] == 158 and plan['steps_per_arm'] == 400
    for arm in ('original', 'repeat', 'alternate'):
        result = json.loads((BASELINE/arm/'RESULT.json').read_text())
        assert result['full_budget_without_reset'] and result['teacher_frozen']['passed']
    plan['generated_rollout'] = dict(generator_run=str(MODEL), baseline_run=str(BASELINE),
        handoff_control_frame=158, sample_seed_base=272090,
        sample_seed_rule='272090 + control_frame -158, same sequence for original/repeat/alternate, inside fork_rng',
        arms=['original','repeat','alternate'], maximum_new_controls=1200, maximum_generated_controls=726,
        controller='full frozen8327408 Generator + full frozen601629 Tracker',
        generated_fields='current36 and future288; replan every actual control using original16stepDDPM and official36frame interpolation',
        known_position_fields='unchanged48 measured reference positions, explicitly reference-assisted diagnostic',
        history='last actually issued36command at t-5, five real prefix commands, no29motor-action substitution',
        geometry='Original96/90 numeric box/limb geometry using previously verified raw-source clocks; no future actual states',
        goal='same trained normalized9Dzero intervention; no provided Refiner goal cue reaches encoder',
        priority_amendment='User explicitly prioritizes obtaining actual rollouts on2026-09-14. This bounded reference-assisted diagnostic may reveal failures before broader prediction/deployment criteria pass. It does not relax or replace any original prediction criterion or authorize a success claim.',
        prerequisites='Both512/fullendpoint checks; complete fixed11phase evaluation and loss readback; full CPU routing preflight; retained single GPU idle before PhysX',
        decision='Complete all three bounded attempts with original termination terms. Report actual generated steps, first failure/terminal state, exact repeat/common prefix, body/box following and full frozen states. Negative results remain in denominator.',
        optimizer_updates=0, smp_benefit_claim=False, independent_deployment_claim=False)
    plan['purpose']='Actual generated-command execution feasibility at earliest predeclared TRAIN branch158'
    plan['automatic_next_action']='Finish all3bounded actual attempts and saved physical/routing readback; act on observed failure, without extending completed training budgets.'
    RUN.mkdir(); (RUN/'motions').symlink_to((BASELINE/'motions').resolve(), target_is_directory=True)
    math=reference_math(); contexts=[]; clocks=[]
    for arm, source in [('original',96),('alternate',90)]:
        raw=ROOT/f'SUGAR/data/CarryBox/data_{source:03d}'
        with np.load(raw/'robot_50hz.npz') as f: robot={k:f[k].copy() for k in f.files}
        with (raw/'obj_motion_global_50hz.pkl').open('rb') as f: obj=pickle.load(f)
        with np.load(BASE/f'refiner_aligned96_90_feasibility/{arm}/TRACE.npz') as f: source_frames=f['reference_frame'].copy()
        frames=np.arange(400)[:,None]+5*np.arange(8)[None]
        assert frames.max()<len(source_frames)
        mapped=source_frames[frames]
        contexts.append(np.stack([geometry_context(robot,obj,row,math) for row in mapped]))
        clocks.append(mapped)
    contexts=np.stack(contexts)
    with np.load(BASELINE/'branch_samples/BRANCH_SAMPLES.npz') as f:
        assert np.array_equal(contexts[:,158].reshape(2,1,168),f['original_demo_geometry'])
    np.savez_compressed(RUN/'ORIGINAL_DEMO_CONTEXT.npz',geometry=contexts,source_frames=np.stack(clocks))
    write(RUN/'PROTOCOL.json',plan)
    print(json.dumps(dict(run=str(RUN),maximum_actual_steps=1200,maximum_generated_steps=726,context158_exact=True)),flush=True)


def preflight():
    import dill
    import numpy as np
    import torch
    from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper,GeneratorObs
    from rsl_rl.networks.mlp import MLP
    from scripts.sugar.demo_following.demo_future.generator_demo_geometry import restore_geometry_state,CONTEXT_KEY
    from scripts.sugar.demo_following.demo_future.generated_rollout_controller import prepare_live_input
    from scripts.sugar.demo_following.demo_future.generator_tracker_routing import IssuedCommandHistory,compose_generated_tracker_input
    torch.set_num_threads(1)
    out=RUN/'ROUTING_PREFLIGHT.json'; assert not out.exists()
    rollout=json.loads((RUN/'PROTOCOL.json').read_text())['generated_rollout']
    model_run=Path(rollout['generator_run'])
    mode=rollout.get('generator_condition_mode','demo_geometry')
    assert mode in ('demo_geometry','zero_context')
    wrapper=GeneratorWrapper.load(str(ROOT/'SUGAR/demo_ckpts/CarryBox/generator.ckpt'),device='cpu')
    with (model_run/mode/'checkpoints/endpoint.ckpt').open('rb') as f: state=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']['model']
    audit=restore_geometry_state(wrapper.policy,state,mode,'zero_diagnostic')
    wrapper.policy.eval().requires_grad_(False)
    payload=torch.load(BASE/'matched_train_coverage128/broad_train/training/model_607.pt',map_location='cpu',weights_only=True)
    actor=MLP(input_dim=846,output_dim=29,hidden_dims=[512,256,128],activation='elu')
    actor.load_state_dict({k.removeprefix('actor.'):v for k,v in payload['model_state_dict'].items() if k.startswith('actor.')},strict=True)
    actor.eval().requires_grad_(False)
    actor_initial={k:v.clone() for k,v in actor.state_dict().items()}
    checks={'full_generator_restore':audit['full_state_exact'] and audit['full_parameter_count']==8327408,
            'full_tracker601629':sum(p.numel() for p in actor.parameters())==601629}
    errors={}; math=reference_math()
    with np.load(BASELINE/'branch_samples/BRANCH_SAMPLES.npz') as f: branch={k:f[k].copy() for k in f.files}
    with np.load(RUN/'ORIGINAL_DEMO_CONTEXT.npz') as f: context=f['geometry'].copy()
    for index,arm in enumerate(('original','alternate')):
        with np.load(BASELINE/arm/'TRACE.npz') as f: trace={k:f[k].copy() for k in f.files}
        with np.load(BASELINE/arm/'STARTUP.npz') as f: torso=f['body_names'].tolist().index('torso_link')
        t=158; body=trace['robot_body_state_before_w'][t,0,torso]; obj=trace['object_state_before_w'][t,0]
        pos,quat=math.subtract_frame_transforms_np(body[:3],body[3:7],obj[:3],obj[3:7])
        tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32).reshape(1,1,-1)
        history=IssuedCommandHistory(t,torch.as_tensor(trace['reference_command'][t-5:t,0]).unsqueeze(0))
        raw=GeneratorObs(tensor(pos),tensor(quat),tensor(trace['joint_pos_before'][t]),tensor(trace['project_gravity'][t]),
                         tensor([0,0,0]),tensor([1,0,0,0]),history.previous_10hz_command(t))
        live=prepare_live_input(wrapper,raw,history.previous_10hz_command(t),tensor(context[index,t]))
        expected={k:torch.as_tensor(branch[k][index:index+1]) for k in live}
        compared=('obj_pos_b','obj_ori_b','joint_pos','project_gravity','last_action',CONTEXT_KEY)
        errors[arm]={k:float((live[k]-expected[k]).abs().max()) for k in compared}
        checks[arm+'_all_live_fields_match_saved']=all(v<=1e-5 for v in errors[arm].values())
        # Goal is intentionally ignored after normalization, matching training.
        with torch.inference_mode():
            for draw in range(2):
                torch.manual_seed(272090+draw); before=torch.get_rng_state().clone()
                prediction=wrapper.policy.predict_action(live)
                torch.set_rng_state(before); repeat=wrapper.policy.predict_action(live)
                checks[f'{arm}_draw{draw}_full_official_repeat_exact']=torch.equal(prediction,repeat)
                torch.set_rng_state(before); target_input=wrapper.policy.predict_action(expected)
                error=float((prediction-target_input).abs().max())
                checks[f'{arm}_draw{draw}_live_vs_actual_sample']=error<=1e-4
                errors[arm][f'draw{draw}_max_output_error']=error
                dense=wrapper._parse_action(prediction)
                observation=torch.as_tensor(trace['teacher_observation'][t])
                positions=torch.as_tensor(trace['actor_reference_position_feedback'][t])
                routed=compose_generated_tracker_input(observation,dense,positions,plan_frame=t,control_frame=t)
                checks[f'{arm}_draw{draw}_actual474_and_positions_exact']=torch.equal(routed[:,36:510],observation[:,36:]) and torch.equal(routed[:,798:],positions)
                altered=observation.clone();altered[:,:36]+=99
                checks[f'{arm}_draw{draw}_numeric_command_discarded']=torch.equal(routed,compose_generated_tracker_input(altered,dense,positions,plan_frame=t,control_frame=t))
                checks[f'{arm}_draw{draw}_full_actor_finite']=bool(torch.isfinite(actor(routed)).all())
            for frame in range(t,t+6):
                old=history.previous_10hz_command(frame)
                expected_old=torch.as_tensor(trace['reference_command'][frame-5:frame-4,0]).unsqueeze(0) if frame<t+5 else dense[:,0:1]
                checks[f'{arm}_history_lag5_frame{frame}']=torch.equal(old,expected_old)
                history.record_issued(frame,dense[:,0])
            known=torch.cat([observation,torch.as_tensor(trace['planned_reference_command'][t]).flatten(1),positions],1)
            actor_error=float((actor(known)-torch.as_tensor(trace['requested_action'][t])).abs().max())
            errors[arm]['full_known_actor_readback']=actor_error
            checks[arm+'_full_known_actor_readback']=actor_error<=1e-5
    checks['full_generator_state_unchanged']=all(torch.equal(v,state[k]) for k,v in wrapper.policy.state_dict().items())
    checks['full_tracker_state_unchanged']=all(torch.equal(v,actor_initial[k]) for k,v in actor.state_dict().items())
    report=dict(passed=all(checks.values()),checks=checks,errors=errors,condition_mode=mode,new_optimizer_updates=0,new_physics_steps=0,
                scope='Actual saved158 inputs, full official models, causal five-command lag, full current/future routing. Physical behavior untested.')
    write(out,report);assert report['passed'],report
    print(json.dumps(report),flush=True)


def pipeline():
    assert os.environ.get('SLURM_STEP_ID') == '0'
    assert json.loads((RUN/'ROUTING_PREFLIGHT.json').read_text())['passed']
    model_run=Path(json.loads((RUN/'PROTOCOL.json').read_text())['generated_rollout']['generator_run'])
    assert json.loads((model_run/'RESULT.json').read_text())['execution_completed']
    assert json.loads((model_run/'TRAINING_LOSS_READBACK.json').read_text())['checks_passed']
    assert json.loads((model_run/'frozen_evaluation/LATENT_REPLAY_COMPARISON.json').read_text())['checks_passed']
    assert not (RUN/'CHILDREN.json').exists()
    children=[]
    for arm in ('original','repeat','alternate'):
        command=[sys.executable,str(BOOT),'scripts.sugar.demo_following.demo_future.collect_refiner_pair',
                 '--headless','--controller','tracker','--tracker-checkpoint',str(BASE/'matched_train_coverage128/broad_train/training/model_607.pt'),
                 '--robot-usd',str(BASE/'scene_runtime/converted_g1/g1_29dof_rev_1_0_with_rubber_hand.usd'),'--run',str(RUN),'--arm',arm]
        with (RUN/(arm+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(arm=arm,pid=child.pid,pgid=os.getpgid(child.pid),command=command);children.append(row);write(RUN/'CHILDREN.json',children)
            row['returncode']=child.wait();write(RUN/'CHILDREN.json',children)
        assert row['returncode']==0,row
        # Kit shutdown can return0 after a Python traceback. Completed artifacts
        # and full frozen-state audits, not process code alone, establish success.
        result=json.loads((RUN/arm/'RESULT.json').read_text())
        assert result['execution_completed'] and result['teacher_frozen']['passed'] and result['generator_frozen']['passed']
    write(RUN/'COLLECTION_COMPLETION.json',dict(execution_completed=True,children=children,new_optimizer_updates=0,
          next_action='Complete saved physical and routing readback, inspect actual body/object curves and original termination evidence.'))


def main():
    global RUN
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','preflight','pipeline'])
    parser.add_argument('--run',type=Path,default=RUN)
    args=parser.parse_args();RUN=args.run.resolve();assert RUN.parent==BASE
    globals()[args.stage]()


if __name__=='__main__':main()
