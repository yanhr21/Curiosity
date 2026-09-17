"""Serial matched full-Tracker input experiment and actual frozen endpoints."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'experiments/demo_following/demo_future_smp_v1'
RUN=BASE/'matched_tracker_future_plan'
BOOT=ROOT/'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'


def main():
    global RUN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-source',type=Path,required=True)
    parser.add_argument('--evaluate-only',action='store_true',help='Evaluate the already completed matched256-update endpoints without additional training')
    parser.add_argument('--robot-usd',type=Path)
    parser.add_argument('--evaluation-name',default='evaluation')
    parser.add_argument('--coverage',action='store_true',help='Matched128-update recovery-data coverage continuation')
    args=parser.parse_args()
    if args.coverage:RUN=BASE/'matched_tracker_coverage128'
    endpoint=383 if args.coverage else 255
    expected_updates=384 if args.coverage else 256
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login','mgmtserver')):
        raise RuntimeError('Use the recorded retained compute child')
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    records_path=RUN/(f'EVALUATION_CHILDREN_{args.evaluation_name}.json' if args.evaluate_only else 'CHILDREN.json')
    if records_path.exists():raise RuntimeError('Do not repeat a launched matched stage')
    if args.robot_usd:
        replay=json.loads((BASE/'tracker_bcppo_refined_pair_low_exploration/PRECONVERTED_REPLAY_READBACK.json').read_text())
        if not replay['passed']:raise RuntimeError('Converted-USD actual replay equivalence failed')
    records=[]
    def child(tag,module,arguments,result):
        command=[sys.executable,str(BOOT),'scripts.sugar.demo_following.demo_future.'+module,*map(str,arguments)]
        log_tag=args.evaluation_name+'_'+tag if args.evaluate_only else tag
        with (RUN/(log_tag+'.log')).open('x') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(tag=tag,pid=p.pid,pgid=os.getpgid(p.pid),command=command)
            records.append(row);records_path.write_text(json.dumps(records,indent=2)+'\n')
            row['returncode']=p.wait();records_path.write_text(json.dumps(records,indent=2)+'\n')
        if row['returncode'] or not result.exists():raise RuntimeError(f'{tag} failed; inspect exact child log and result')
        print(json.dumps(dict(completed_child=tag,result=str(result))),flush=True)
    # Both arms train before physical endpoints so a control physical failure
    # cannot silently suppress the treatment or consume a different budget.
    for arm in plan['arms']:
        out=RUN/arm
        if not args.evaluate_only:
            training_args=['--output',out,'--paired-input','future_plan' if args.coverage else arm,'--paired-original-source',args.original_source.resolve()]
            if args.coverage:training_args+=['--coverage-arm',arm]
            child('train_'+arm,'train_tracker_replay_bc',
                  training_args,out/'RESULT.json')
        training=json.loads((out/'RESULT.json').read_text())
        if not training['execution_completed'] or training['bcppo_updates']!=expected_updates:raise RuntimeError('Matched requested-update endpoint missing')
    import torch
    initial=[torch.load(RUN/arm/'model_-1.pt',map_location='cpu',weights_only=True) for arm in plan['arms']]
    equal={k:bool(torch.equal(v,initial[1]['model_state_dict'][k])) for k,v in initial[0]['model_state_dict'].items()}
    audit=dict(all_initial_state_equal=all(equal.values()),initial_state_equal=equal,
               both_optimizer_states_empty=all(not x['optimizer_state_dict']['state'] for x in initial))
    if args.coverage:
        from scripts.sugar.demo_following.demo_future.tracker_coverage import equal_state
        audit['optimizer_states_exact']=equal_state(initial[0]['optimizer_state_dict'],initial[1]['optimizer_state_dict'])
        audit['both_restored_parent_state']=all(json.loads((RUN/arm/'RESUME_PREFLIGHT.json').read_text())['passed'] for arm in plan['arms'])
        audit['common_initial_fit_exact']=equal_state(*[json.loads((RUN/arm/'RESULT.json').read_text())['common_initial_fit'] for arm in plan['arms']])
    (RUN/'MATCHED_INITIALIZATION.json').write_text(json.dumps(audit,indent=2)+'\n')
    optimizer_ok=all(audit[k] for k in ('optimizer_states_exact','both_restored_parent_state','common_initial_fit_exact')) if args.coverage else audit['both_optimizer_states_empty']
    if not audit['all_initial_state_equal'] or not optimizer_ok:
        raise RuntimeError('Matched initialization failed')
    evaluations={}
    source=BASE/'tracker_refined96_90_feasibility'
    for arm in plan['arms']:
        evaluation=RUN/arm/args.evaluation_name;evaluation.mkdir()
        eval_plan=json.loads((source/'PROTOCOL.json').read_text())
        eval_plan.update(controller='tracker',paired_input='future_plan' if args.coverage else arm)
        if args.coverage:eval_plan['coverage_arm']=arm
        eval_plan.update(checkpoint_origin=f'Project-adapted full official Tracker, cumulative{expected_updates} fixed-data BC updates',
                         robot_usd=str(args.robot_usd.resolve()) if args.robot_usd else None,
                         automatic_next_action='Complete all three predeclared matched arms and independent readback; report failures, do not add updates.')
        (evaluation/'PROTOCOL.json').write_text(json.dumps(eval_plan,indent=2)+'\n')
        (evaluation/'motions').symlink_to(source/'motions',target_is_directory=True)
        for selected in plan['physical_evaluation']['arms']:
            arguments=['--headless','--controller','tracker','--tracker-checkpoint',RUN/arm/f'model_{endpoint}.pt',
                       '--run',evaluation,'--arm',selected]
            if args.robot_usd:arguments+=['--robot-usd',args.robot_usd.resolve()]
            child('eval_'+arm+'_'+selected,'collect_refiner_pair',arguments,evaluation/selected/'RESULT.json')
        child('readback_'+arm,'readback_refiner_pair',['--run',evaluation],evaluation/'READBACK.json')
        evaluations[arm]=json.loads((evaluation/'READBACK.json').read_text())
    report=dict(execution_completed=True,matched_initialization=audit,
                training={arm:json.loads((RUN/arm/'RESULT.json').read_text()) for arm in plan['arms']},
                physical_pair_passed={arm:r['data_pair_feasibility_passed'] for arm,r in evaluations.items()},
                scope='Exploratory TRAIN input diagnostic. Pair feasibility alone is not independent following, video or generalization success.',
                next_action='Inspect physical curves and branchpoint fit/ablations; when treatment pair passes, run independent reference following, frozen official SMP scores and actual camera evidence before any Generator adaptation.')
    report['evaluation_name']=args.evaluation_name
    report['robot_usd']=str(args.robot_usd.resolve()) if args.robot_usd else None
    (RUN/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
