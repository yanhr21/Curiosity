"""Resolve a CPU batched replay mismatch by exact actual-device batch1 replay."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from rsl_rl.networks.mlp import MLP


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();run=args.run
    assert os.environ.get('SLURM_STEP_ID')=='0'
    out=run/'GENERATED_ACTOR_DEVICE_AUDIT.json';assert not out.exists()
    original_path=run/'GENERATED_EXECUTION_READBACK.json';original=json.loads(original_path.read_text())
    failed=[k for k,v in original['checks'].items() if not v]
    assert failed and all(k.endswith('_all_generated_full_actor_outputs_reproduced') for k in failed)
    plan=json.loads((run/'PROTOCOL.json').read_text());switch=plan['switch_control_frame']
    torch.set_num_threads(1)
    payload=torch.load(plan['checkpoint_origin'],map_location='cpu',weights_only=True)
    actor=MLP(input_dim=846,output_dim=29,hidden_dims=[512,256,128],activation='elu')
    state={k.removeprefix('actor.'):v for k,v in payload['model_state_dict'].items() if k.startswith('actor.')}
    actor.load_state_dict(state,strict=True);actor.eval().requires_grad_(False)
    checks={'full601629_parameters':sum(p.numel() for p in actor.parameters())==601629};metrics={};arrays={}
    for arm in ('original','repeat','alternate'):
        with np.load(run/arm/'TRACE.npz') as f:
            inputs=f['generated_actor_input'][:,0].copy();requested=f['requested_action'][switch:,0].copy()
        actor.cpu()
        with torch.inference_mode():cpu=actor(torch.as_tensor(inputs)).numpy()
        cpu_error=float(np.max(np.abs(cpu-requested)))
        checks[arm+'_original_cpu_error_exact']=cpu_error==original['arms'][arm]['full_actor_all_generated_outputs_max_error']
        actor.cuda()
        with torch.inference_mode():
            device_inputs=torch.as_tensor(inputs,device='cuda')
            predictions=torch.cat([actor(row[None]) for row in device_inputs]).cpu().numpy()
        exact=np.array_equal(predictions,requested)
        checks[arm+'_all_actual_gpu_batch1_actions_exact']=exact
        checks[arm+'_full_frozen_state_exact']=all(torch.equal(v.cpu(),state[k]) for k,v in actor.state_dict().items())
        metrics[arm]=dict(rows=len(inputs),cpu_batched_max_error=cpu_error,cuda_batch1_max_error=float(np.max(np.abs(predictions-requested))),cuda_batch1_exact=exact)
        arrays[arm+'_requested']=requested;arrays[arm+'_cuda_replay']=predictions
    report=dict(checks_passed=all(checks.values()),checks=checks,arms=metrics,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        new_physics_steps=0,new_optimizer_updates=0,scope='Complete existing Tracker, all saved actual generated actor inputs, exact CUDA batch1 inference as in execution. CPU batched result and its original threshold failure remain recorded; no tolerance relaxation or physics rerun.')
    np.savez_compressed(run/'GENERATED_ACTOR_DEVICE_AUDIT.npz',**arrays)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    assert report['checks_passed'],checks
    backup=run/'GENERATED_EXECUTION_CPU_READBACK.json';assert not backup.exists();backup.write_text(original_path.read_text())
    for arm,value in metrics.items():
        original['checks'][arm+'_all_generated_full_actor_outputs_reproduced']=value['cuda_batch1_exact']
        original['arms'][arm]['cpu_batched_actor_outputs_max_error']=value['cpu_batched_max_error']
        original['arms'][arm]['full_actor_all_generated_outputs_max_error']=value['cuda_batch1_max_error']
    original['checks_passed']=all(original['checks'].values())
    original.update(actor_replay_device='cuda',actor_replay_batch='actual batch1',original_cpu_batched_readback_passed=False,
        original_cpu_readback=str(backup),exact_device_audit=str(out))
    original_path.write_text(json.dumps(original,indent=2)+'\n')


if __name__=='__main__':main()
