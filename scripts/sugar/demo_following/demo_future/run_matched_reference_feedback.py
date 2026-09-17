"""Matched full official reference-feedback preflights, training and six endpoints."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'experiments/demo_following/demo_future_smp_v1'
RUN=BASE/'matched_reference_feedback96'
BOOT=ROOT/'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'
USD=BASE/'scene_runtime/converted_g1/g1_29dof_rev_1_0_with_rubber_hand.usd'
ARMS=('zero_feedback','reference_feedback')


def main():
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login','mgmtserver')):
        raise RuntimeError('Use recorded retained compute step')
    import numpy as np
    import torch
    from scripts.sugar.demo_following.demo_future.tracker_coverage import equal_state
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    if not json.loads((BASE/'online_tracker_future128/reference_translation_diagnostic/RESULT.json').read_text())['numerical_diagnostic_passed']:
        raise RuntimeError('Reference observability evidence missing')
    record_path=RUN/'CHILDREN.json'
    if record_path.exists():raise RuntimeError('Do not repeat a launched matched pipeline')
    records=[]
    def child(tag,module,arguments,artifact):
        command=[sys.executable,str(BOOT),'scripts.sugar.demo_following.demo_future.'+module,*map(str,arguments)]
        with (RUN/(tag+'.log')).open('x') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(tag=tag,pid=p.pid,pgid=os.getpgid(p.pid),command=command);records.append(row)
            record_path.write_text(json.dumps(records,indent=2)+'\n');row['returncode']=p.wait()
            record_path.write_text(json.dumps(records,indent=2)+'\n')
        if row['returncode'] or not artifact.exists():raise RuntimeError('Inspect child artifact/log: '+tag)
        print(json.dumps(dict(completed=tag,artifact=str(artifact))),flush=True)
    for arm in ARMS:
        out=RUN/arm/'preflight'
        child('preflight_'+arm,'train_tracker_online_future',['--headless','--preflight-only','--feedback-arm',arm,'--output',out,'--robot-usd',USD],out/'RESULT.json')
    initial=[torch.load(RUN/arm/'preflight/model_383.pt',map_location='cpu',weights_only=True) for arm in ARMS]
    checks=dict(whole_model_exact=equal_state(initial[0]['model_state_dict'],initial[1]['model_state_dict']),whole_optimizer_exact=equal_state(initial[0]['optimizer_state_dict'],initial[1]['optimizer_state_dict']))
    for arm in ARMS:
        pre=RUN/arm/'preflight';result=json.loads((pre/'RESULT.json').read_text())
        protocol=json.loads((pre/'PROTOCOL.json').read_text())
        checks[arm+'_actual_preflight']=result['passed'] and result['actual_transitions']==1536 and result['optimizer_updates']==0
        checks[arm+'_whole_parent_resume']=protocol['resume']['passed']
        checks[arm+'_position_readback']=json.loads((pre/'POSITION_PACKET_AUDIT.json').read_text())['passed']
        checks[arm+'_initial_cache']=json.loads((pre/'REFERENCE_CACHE_INITIALIZATION.json').read_text())['passed']
        checks[arm+'_first_step_not_false_terminal']=json.loads((pre/'TERMINATION_READBACK.json').read_text())['step_records'][0]['terminations']==0
    for filename in ('INITIAL_STATE.npz','PREFLIGHT_TRACE.npz'):
        arrays=[dict(np.load(RUN/arm/'preflight'/filename,allow_pickle=False)) for arm in ARMS]
        checks[filename+'_all_fields_exact']=arrays[0].keys()==arrays[1].keys() and all(np.array_equal(v,arrays[1][k]) for k,v in arrays[0].items())
    obs=[dict(np.load(RUN/arm/'preflight/INITIAL_OBSERVATIONS.npz',allow_pickle=False)) for arm in ARMS]
    checks['old_observations_parent_and_adapted_actions_exact']=all(np.array_equal(v,obs[1][k]) for k,v in obs[0].items() if k!='reference_feedback')
    checks['control_feedback_zero']=not bool(obs[0]['reference_feedback'].any())
    checks['treatment_feedback_is_actual_packet']=np.array_equal(obs[1]['reference_feedback'],obs[1]['raw_reference_feedback'])
    passed=all(checks.values())
    (RUN/'MATCHED_PREFLIGHT.json').write_text(json.dumps(dict(passed=passed,checks=checks,scope='Complete initial model/Adam and actual24-step64-env preflight comparison. No physical-following claim.'),indent=2)+'\n')
    if not passed:raise RuntimeError('Matched actual preflight failed')
    plan.update(implemented=True,preflight_passed=True,training_started=True)
    (RUN/'PROTOCOL.json').write_text(json.dumps(plan,indent=2)+'\n')
    for arm in ARMS:
        out=RUN/arm/'training'
        child('training_'+arm,'train_tracker_online_future',['--headless','--feedback-arm',arm,'--output',out,'--robot-usd',USD],out/'RESULT.json')
        result=json.loads((out/'RESULT.json').read_text())
        if result['newly_applied_updates']!=96 or result['bcppo_updates']!=480 or result['actual_transitions']!=147456 or result['optimizer_clocks']!=[9600]:raise RuntimeError('Matched endpoint budget drift')
    # Compare the actual training initial states too, not only the preflights.
    actual=[torch.load(RUN/arm/'training/model_383.pt',map_location='cpu',weights_only=True) for arm in ARMS]
    initialization=dict(whole_models_exact=equal_state(actual[0]['model_state_dict'],actual[1]['model_state_dict']),whole_optimizers_exact=equal_state(actual[0]['optimizer_state_dict'],actual[1]['optimizer_state_dict']),training_matches_preflight=all(equal_state(a['model_state_dict'],b['model_state_dict']) and equal_state(a['optimizer_state_dict'],b['optimizer_state_dict']) for a,b in zip(initial,actual)))
    initialization['passed']=all(initialization.values())
    (RUN/'MATCHED_TRAINING_INITIALIZATION.json').write_text(json.dumps(initialization,indent=2)+'\n')
    if not initialization['passed']:raise RuntimeError('Actual training initialization mismatch')
    results={}
    for arm in ARMS:
        evaluation=RUN/arm/'evaluation';evaluation.mkdir()
        source=BASE/'tracker_refined96_90_feasibility'
        protocol=json.loads((source/'PROTOCOL.json').read_text())
        protocol.update(controller='tracker',paired_input='future_plan',reference_feedback=arm,purpose='Frozen complete official reference-feedback Tracker after96 matched online BC updates',automatic_next_action='Run all three400-step endpoints regardless of preceding failures, then independent physical and reference readback.',checkpoint_origin='Complete official846-input Tracker from online384 parent plus96 updates, zero versus true48-D reference position feedback')
        (evaluation/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
        (evaluation/'motions').symlink_to(source/'motions',target_is_directory=True)
        for selected in ('original','repeat','alternate'):
            child('eval_'+arm+'_'+selected,'collect_refiner_pair',['--headless','--controller','tracker','--tracker-checkpoint',RUN/arm/'training/model_479.pt','--robot-usd',USD,'--run',evaluation,'--arm',selected],evaluation/selected/'RESULT.json')
        child('readback_'+arm,'readback_refiner_pair',['--run',evaluation],evaluation/'READBACK.json')
        child('following_'+arm,'readback_reference_following',['--run',evaluation],evaluation/'REFERENCE_FOLLOWING_READBACK.json')
        pair=json.loads((evaluation/'READBACK.json').read_text());following=json.loads((evaluation/'REFERENCE_FOLLOWING_READBACK.json').read_text())
        results[arm]=dict(physical_pair_passed=pair['data_pair_feasibility_passed'],reference_following_passed=following['exploratory_reference_following_checks_passed'],steps={k:v['actual_steps'] for k,v in pair['arm_results'].items()},maximum_lift_m={k:v['maximum_lift_m'] for k,v in pair['arm_results'].items()})
    result=dict(execution_completed=True,arms=results,matched_initialization_passed=True,updates_per_arm=96,actual_transitions_per_arm=147456,treatment_full_physical_and_reference_pass=all(results['reference_feedback'][k] for k in ('physical_pair_passed','reference_following_passed')),scope='Single TRAIN-pair matched interface experiment. Input requires causal localization; actual camera, held-out tests and broader method evaluation remain necessary. All failed arms retained.',next_action='Inspect all actual curves and exact traces; measure selected-reference feedback benefit and any remaining failure before choosing the next bounded stage.')
    (RUN/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
