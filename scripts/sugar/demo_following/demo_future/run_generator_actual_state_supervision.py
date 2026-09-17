"""Matched complete Generator training with actual-state expert-command supervision."""
import argparse
import json
import os
import subprocess
import sys

from scripts.sugar.demo_following.demo_future.run_generated_command_rollout import ROOT,BASE,MODEL,BOOT,write

RUN=BASE/'matched_generator_actual_state_expert01512'
CORPUS=BASE/'actual_generated_state_expert_supervision_v1'


def prepare():
    assert not RUN.exists()
    data=json.loads((CORPUS/'RESULT.json').read_text());assert data['checks_passed'] and data['examples']==355
    plan=json.loads((MODEL/'PROTOCOL.json').read_text())
    for key in ('reuse_preflights_after_preoptimizer_binding_fix','preoptimizer_binding_incident','training_pipeline_directory'):
        plan.pop(key,None)
    plan['comparison_source']=str(MODEL)
    plan['actual_state_supervision']=dict(corpus=str(CORPUS),weight=.1,apply_to_both_arms=True,seed=272500,
        batch_rows=144,rows_per_group=36,examples=355,extra_rows_per_arm=73728,
        label_kind='known expert command proposals on actual generated states, not actual counterfactual physics futures',
        sampling='Four groups equally36rows, deterministic cyclic indices(step*36+offset)%group_size; no repeats of physical episodes or score-based row selection',
        rng='Original q/rank/replay evaluated first; separate seed272500+step for expert q-loss, original RNG restored afterwards',
        loss='Existing IndependentNoiseGenerator full-model normalized denoising q objective, fixed weight0.1; no new module or alternative sampler')
    plan['generator_training_started']=False
    plan['post_prediction_position_floor']=None
    plan['implementation_status']='Actual-state loss/recorder adaptation implemented; full CPU and BF16 qualification pending'
    plan['comparison_scope']='Same full8327408 model, release initialization, old normalization, original18case144rows, seed272084 and both512 budgets. Add144expert rows/update/.1q objective from355actual generated states to BOTH arms. Additional73728denoising rows/arm and collection cost; not equalFLOPs. Existing q/.25rank/.1frozenreplay unchanged. All11fixedphase criteria retained; actual rollout outcomes remain separate primary execution evidence.'
    plan['automatic_next_action']='FullCPU+H200 initial/learned loss/gradient/RNG/data preflight, both512 and full11fixedphase evaluation/loss readback, inspect all88panels and strict comparison, then same bounded reference-assisted generated rollout with unchanged158handoff/400budget/terminations. All failed old endpoints retained; no old512extension or SMP benefit claim.'
    RUN.mkdir();write(RUN/'PROTOCOL.json',plan)
    print(json.dumps(dict(run=str(RUN),full_parameters=8327408,examples=355,extra_rows_per_update=144,updates_each=512)),flush=True)


def pipeline():
    assert os.environ.get('SLURM_STEP_ID')=='0'
    assert json.loads((RUN/'ACTUAL_STATE_CPU_PREFLIGHT.json').read_text())['checks_passed']
    assert not (RUN/'PIPELINE_CHILDREN.json').exists()
    children=[]
    stages=[('bf16','preflight_generator_actual_state',['--device','cuda']),
            ('train_and_evaluate','train_generator_geometry',['--run',str(RUN),'--pipeline']),
            ('loss_readback','readback_generator_latent_training',['--run',str(RUN)])]
    for tag,module,args in stages:
        command=[sys.executable,str(BOOT),'scripts.sugar.demo_following.demo_future.'+module,*args]
        with (RUN/(tag+'.pipeline.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=tag,pid=child.pid,pgid=os.getpgid(child.pid));children.append(row);write(RUN/'PIPELINE_CHILDREN.json',children)
            row['returncode']=child.wait();write(RUN/'PIPELINE_CHILDREN.json',children)
        assert row['returncode']==0,row
    write(RUN/'PIPELINE_COMPLETION.json',dict(execution_completed=True,children=children,all_horizon_panels_inspected=False,
        next_action='Inspect88fixedphase panels and complete strict old/new fullstate/exposure/prediction comparison, then execute predeclared bounded actual generated rollout; no manual transition.'))


def prepare_rollouts():
    assert json.loads((RUN/'PIPELINE_COMPLETION.json').read_text())['execution_completed']
    assert json.loads((RUN/'frozen_evaluation/LATENT_REPLAY_COMPARISON.json').read_text())['checks_passed']
    source=BASE/'measured32_generated_command_rollout158_r1'
    for mode,label in [('demo_geometry','demo'),('zero_context','zero')]:
        out=BASE/('actual_state_expert_generated_'+label+'158');assert not out.exists()
        plan=json.loads((source/'PROTOCOL.json').read_text())
        plan['generated_rollout']['generator_run']=str(RUN)
        plan['generated_rollout']['generator_condition_mode']=mode
        plan['generated_rollout']['comparison_rollout']=str(source if label=='demo' else BASE/'measured32_generated_zero_rollout158')
        plan['purpose']='Actual-state expert supervision: same bounded reference-assisted generated-command execution comparison'
        plan['automatic_next_action']='Complete unchanged three400step attempts and full saved routing/physical readback; compare to both prior trained arms on common actual intervals, retaining all terminal failures.'
        out.mkdir();write(out/'PROTOCOL.json',plan)
        (out/'motions').symlink_to((source/'motions').resolve(),target_is_directory=True)
        (out/'ORIGINAL_DEMO_CONTEXT.npz').symlink_to((source/'ORIGINAL_DEMO_CONTEXT.npz').resolve())
        print(json.dumps(dict(run=str(out),mode=mode,maximum_controls=1200)),flush=True)


def rollout_pipeline():
    assert os.environ.get('SLURM_STEP_ID')=='0'
    assert not (RUN/'ROLLOUT_PIPELINE_CHILDREN.json').exists()
    prepare_rollouts()
    children=[]
    for label in ('demo','zero'):
        out=BASE/('actual_state_expert_generated_'+label+'158')
        for stage,module,args in [
                ('routing','run_generated_command_rollout',['preflight','--run',str(out)]),
                ('physics','run_generated_command_rollout',['pipeline','--run',str(out)]),
                ('readback','readback_generated_command_rollout',['--run',str(out),'--actor-device','cuda'])]:
            command=[sys.executable,str(BOOT),'scripts.sugar.demo_following.demo_future.'+module,*args]
            with (out/(stage+'.pipeline.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                row=dict(condition=label,stage=stage,pid=child.pid,pgid=os.getpgid(child.pid),command=command)
                children.append(row);write(RUN/'ROLLOUT_PIPELINE_CHILDREN.json',children)
                row['returncode']=child.wait();write(RUN/'ROLLOUT_PIPELINE_CHILDREN.json',children)
            assert row['returncode']==0,row
    write(RUN/'ROLLOUT_PIPELINE_COMPLETION.json',dict(execution_completed=True,children=children,
        all_physical_panels_inspected=False,new_optimizer_updates=0,maximum_physics_controls=2400,
        next_action='Inspect both actual rollouts and complete context plus previous-model comparison; retain early terminations and known48position assistance.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','pipeline','prepare_rollouts','rollout_pipeline']);args=parser.parse_args()
    globals()[args.stage]()


if __name__=='__main__':main()
