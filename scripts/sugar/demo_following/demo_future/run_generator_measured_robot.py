"""Bounded matched training and complete frozen evaluation of measured32 input."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from scripts.sugar.demo_following.demo_future.preflight_generator_measured_robot import RUN
from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import ROOT,write


def main():
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    plan=json.loads((RUN/'PROTOCOL.json').read_text())
    preflight_root=Path(plan['robot_preflight_directory'])
    root=Path(plan.get('training_pipeline_directory',str(preflight_root)))
    assert json.loads((preflight_root/'GENERATED_CPU_PREFLIGHT.json').read_text())['passed']
    assert not (root/'MEASURED_PIPELINE_CHILDREN.json').exists()
    bootstrap=ROOT/'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'
    stages=[('bf16','preflight_generator_measured_robot',['--device','cuda']),
            ('train_and_evaluate','train_generator_geometry',['--run',str(RUN),'--pipeline']),
            ('training_loss_readback','readback_generator_latent_training',['--run',str(RUN)])]
    if plan.get('reuse_preflights_after_preoptimizer_binding_fix'):
        incident=json.loads((RUN/'PRE_OPTIMIZER_BINDING_INCIDENT.json').read_text())
        assert incident['no_optimizer_updates'] and incident['workspace_run_not_called'] and json.loads((preflight_root/'GENERATED_BF16_PREFLIGHT.json').read_text())['passed']
        stages=stages[1:]
    rows=[]
    for stage,module,args in stages:
        command=[sys.executable,str(bootstrap),'scripts.sugar.demo_following.demo_future.'+module,*args]
        with (root/(stage+'.pipeline.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=stage,pid=child.pid,pgid=os.getpgid(child.pid),command=command)
            rows.append(row);write(root/'MEASURED_PIPELINE_CHILDREN.json',rows)
            row['returncode']=child.wait();write(root/'MEASURED_PIPELINE_CHILDREN.json',rows)
        assert row['returncode']==0,row
    write(root/'MEASURED_PIPELINE_COMPLETION.json',dict(execution_completed=True,children=rows,
        all_horizon_panels_inspected=False,strict_comparison_completed=False,new_physics_steps=0,
        next_action='Inspect all88fixed horizon panels and saved metric evidence, then execute compare_generator_latent_replay --run for this measured32 run and apply unchanged9TRAIN+2reused+8oldpassingTRAIN nonregression criteria.'))


if __name__=='__main__':main()
