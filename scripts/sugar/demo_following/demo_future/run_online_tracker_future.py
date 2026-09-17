"""Serial full online Tracker training and complete frozen physical evidence."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'experiments/demo_following/demo_future_smp_v1'
RUN=BASE/'online_tracker_future128'
BOOT=ROOT/'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-usd',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login','mgmtserver')):
        raise RuntimeError('Use retained compute step')
    preflight=json.loads((RUN/'preflight_initialized/RESULT.json').read_text())
    cache=json.loads((RUN/'preflight_initialized/REFERENCE_CACHE_INITIALIZATION.json').read_text())
    terminal=json.loads((RUN/'preflight_initialized/TERMINATION_READBACK.json').read_text())
    if not preflight['passed'] or not cache['passed'] or terminal['step_records'][0]['terminations']:
        raise RuntimeError('Initialized actual online preflight failed')
    records=[];record_path=RUN/'CHILDREN.json'
    if record_path.exists():raise RuntimeError('Do not repeat a launched pipeline')
    def child(tag,module,arguments,artifact):
        command=[sys.executable,str(BOOT),'scripts.sugar.demo_following.demo_future.'+module,*map(str,arguments)]
        with (RUN/(tag+'.log')).open('x') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(tag=tag,pid=p.pid,pgid=os.getpgid(p.pid),command=command);records.append(row)
            record_path.write_text(json.dumps(records,indent=2)+'\n');row['returncode']=p.wait()
            record_path.write_text(json.dumps(records,indent=2)+'\n')
        if row['returncode'] or not artifact.exists():raise RuntimeError('Inspect child result/log: '+tag)
        print(json.dumps(dict(completed=tag,artifact=str(artifact))),flush=True)
    child('training','train_tracker_online_future',['--headless','--output',RUN/'training','--robot-usd',args.robot_usd.resolve()],RUN/'training/RESULT.json')
    training=json.loads((RUN/'training/RESULT.json').read_text())
    if training['newly_applied_updates']!=128 or training['actual_transitions']!=196608:raise RuntimeError('Online budget mismatch')
    evaluation=RUN/'evaluation';evaluation.mkdir()
    source=BASE/'tracker_refined96_90_feasibility'
    plan=json.loads((source/'PROTOCOL.json').read_text())
    plan.update(controller='tracker',paired_input='future_plan',checkpoint_origin='Complete official future-input Tracker after128 actual online BC updates from parent256',
                purpose='Frozen full future-input Tracker after bounded online state-coverage training',
                automatic_next_action='Complete original, repeat and alternate400-step endpoints regardless of preceding failures, then independent physical/reference readback. No additional training follows from execution alone.')
    (evaluation/'PROTOCOL.json').write_text(json.dumps(plan,indent=2)+'\n')
    (evaluation/'motions').symlink_to(source/'motions',target_is_directory=True)
    for arm in ('original','repeat','alternate'):
        child('eval_'+arm,'collect_refiner_pair',['--headless','--controller','tracker','--tracker-checkpoint',RUN/'training/model_383.pt','--robot-usd',args.robot_usd.resolve(),'--run',evaluation,'--arm',arm],evaluation/arm/'RESULT.json')
    child('readback','readback_refiner_pair',['--run',evaluation],evaluation/'READBACK.json')
    child('following','readback_reference_following',['--run',evaluation],evaluation/'REFERENCE_FOLLOWING_READBACK.json')
    pair=json.loads((evaluation/'READBACK.json').read_text());following=json.loads((evaluation/'REFERENCE_FOLLOWING_READBACK.json').read_text())
    result=dict(execution_completed=True,training=training,physical_pair_passed=pair['data_pair_feasibility_passed'],reference_following_passed=following['exploratory_reference_following_checks_passed'],scope='Single TRAIN-pair online-state coverage diagnostic; camera runtime unresolved, no broad following claim.',next_action='Inspect physical curves, exact prefixes and model/optimizer endpoints. If physical checks pass, actual camera evidence and held-out extension remain required before Generator advancement; otherwise diagnose the bounded failure.')
    (RUN/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
