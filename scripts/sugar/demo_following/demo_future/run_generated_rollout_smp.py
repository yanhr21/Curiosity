"""Full official SMP audit of saved generated rollouts and matched known-plan baseline."""
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.sugar.demo_following.demo_future.run_generated_command_rollout import ROOT,BASE,BASELINE,BOOT,write


def main():
    run=BASE/'measured32_generated_command_rollout158_r1'
    out=run/'matched_official_smp';out.mkdir(exist_ok=False)
    assert json.loads((run/'COLLECTION_COMPLETION.json').read_text())['execution_completed']
    baseline=out/'known_plan_baseline';baseline.mkdir()
    plan=json.loads((BASELINE/'PROTOCOL.json').read_text())
    write(baseline/'PROTOCOL.json',plan)
    for arm in plan['arms']:(baseline/arm).symlink_to((BASELINE/arm).resolve(),target_is_directory=True)
    write(out/'PROTOCOL.json',dict(actual_run=str(run),known_plan_baseline=str(BASELINE),
        source_physics_reused_read_only=True,new_physics_steps=0,new_optimizer_updates=0,
        model='Existing complete official TinyMDM shared Carry/Kick prior and original10x216 named actual-state feature functions',
        comparisons='Original and alternate separately, identical common window starts158..lastsharedvalid, same batch size and original score RNG schedule',
        scope='Task-class motion-prior diagnostic only. Not selected-demo distance, physical success or evidence that using SMP improves a trained policy.'))
    children=[]
    stages=[('actual',['--run',str(run)]),('baseline',['--run',str(baseline)])]
    stages.extend((arm,['--run',str(run),'--compare-run',str(baseline),'--compare-arm',arm,'--window-start','158','--common-output',str(out/arm)]) for arm in ('original','alternate'))
    for tag,args in stages:
        command=[sys.executable,str(BOOT),'scripts.sugar.demo_following.demo_future.score_refiner_smp',*args]
        with (out/(tag+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=tag,pid=child.pid,pgid=os.getpgid(child.pid));children.append(row);write(out/'CHILDREN.json',children)
            row['returncode']=child.wait();write(out/'CHILDREN.json',children)
        assert row['returncode']==0,row
    comparisons={arm:json.loads((out/arm/'RESULT.json').read_text()) for arm in ('original','alternate')}
    assert all(v['frozen_unchanged'] and v['identical_batch_shapes_and_score_seed_schedule'] for v in comparisons.values())
    write(out/'RESULT.json',dict(execution_completed=True,comparisons=comparisons,new_physics_steps=0,new_optimizer_updates=0,
        scope='Matched actual post-handoff Carry/Kick task-prior measurements; no policy update or SMP benefit claim.'))
    print(json.dumps(comparisons),flush=True)


if __name__=='__main__':main()
