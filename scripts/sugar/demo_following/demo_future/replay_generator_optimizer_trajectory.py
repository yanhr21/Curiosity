"""Exact independent official training replay with passive optimizer-step recording."""
import json
import argparse
import os
import socket

import dill
import torch

from scripts.sugar.demo_following.demo_future.probe_generator_reverse_process import BASE,SELF_SOURCE,PAIRED_OUT,write
from scripts.sugar.demo_following.demo_future.generator_workspace import GeometryGeneratorWorkspace
from scripts.sugar.demo_following.demo_future import train_generator_geometry as training
from scripts.sugar.demo_following.demo_future.compare_generator_rank_ablation import equal

RUN=BASE/'matched_generator_branch_self_replay01_gaps512_optimizer_trace'
EVIDENCE=PAIRED_OUT/'fresh_full_sampler_gradient/stored_adam_direction'


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def main():
    global RUN
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--r1',action='store_true');parser.add_argument('--readback-existing',action='store_true')
    args=parser.parse_args()
    if args.r1:RUN=RUN.with_name(RUN.name+'_r1')
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    if args.readback_existing:
        trace=RUN/'optimizer_trace';partial=json.loads((trace/'PARTIAL_RESULT.json').read_text())
        assert partial['completed_updates']==512 and not (trace/'RESULT.json').exists()
        names=torch.load(trace/'update_497_delta.pt',map_location='cpu')['parameter_names']
        finish_trace(partial['checks'],partial['rows'],names)
        return
    protocol=json.loads((EVIDENCE/('NEXT_EXACT_OPTIMIZER_REPLAY_PROTOCOL_R1.json' if args.r1 else 'NEXT_EXACT_OPTIMIZER_REPLAY_PROTOCOL.json')).read_text())
    for filename in ('RESULT.json','SAVED_READBACK.json'):
        result=json.loads((EVIDENCE/filename).read_text());assert result['checks_passed']
    assert json.loads((EVIDENCE/'RESULT.json').read_text())['plot_inspected']
    assert protocol['source_run']==str(SELF_SOURCE) and protocol['replay_run']==str(RUN)
    plan=json.loads((SELF_SOURCE/'PROTOCOL.json').read_text())
    assert plan['epochs']==512 and plan['batch_size']==144
    RUN.mkdir(exist_ok=False)
    for name in ('GENERATED_CPU_PREFLIGHT.json','GENERATED_BF16_PREFLIGHT.json','PREFLIGHT.json'):
        source=SELF_SOURCE/name
        if source.exists():(RUN/name).write_bytes(source.read_bytes())
    plan['exact_training_replay_source']=str(SELF_SOURCE)
    plan['passive_optimizer_trace']=protocol
    write(RUN/'PROTOCOL.json',plan);write(RUN/'REPLAY_PROTOCOL.json',protocol)
    trace=RUN/'optimizer_trace';trace.mkdir()
    checks={};rows=[];workspace_holder=[]

    class RecordedWorkspace(GeometryGeneratorWorkspace):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.trace_ordinal=0;self.trace_before=None
            self.trace_names,self.trace_params=zip(*[(n,p) for n,p in self.model.named_parameters() if p.requires_grad])
            self.trace_handles=[]
            self.original_get_optimizer=self.model.get_optimizer
            def create_recorded_optimizer(*factory_args,**factory_kwargs):
                optimizer=self.original_get_optimizer(*factory_args,**factory_kwargs)
                self.trace_handles=[optimizer.register_step_pre_hook(self.before_step),optimizer.register_step_post_hook(self.after_step)]
                return optimizer
            self.model.get_optimizer=create_recorded_optimizer
            workspace_holder.append(self)

        def before_step(self,optimizer,args,kwargs):
            ordinal=self.trace_ordinal+1
            if ordinal>=497:
                cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
                self.trace_before=[p.detach().cpu().clone() for p in self.trace_params]
                self.trace_lr=[float(g['lr']) for g in optimizer.param_groups]
                checks[f'{ordinal}_pre_hook_rng_exact']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())

        def after_step(self,optimizer,args,kwargs):
            self.trace_ordinal+=1;ordinal=self.trace_ordinal
            cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
            if ordinal in (496,512):
                torch.save(dict(model=cpu_tree(self.model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),completed_updates=ordinal,
                    scope='Passive post-optimizer/pre-scheduler snapshot, not exact scheduler/sampler resume checkpoint.'),trace/f'after_update_{ordinal}.pt')
            if ordinal>=497:
                delta=[before.double()-p.detach().cpu().double() for before,p in zip(self.trace_before,self.trace_params)]
                clocks=[int(state['step']) for state in optimizer.state.values() if 'step' in state]
                checks[f'{ordinal}_all_adam_clocks']=len(clocks)==len(self.trace_names) and all(c==ordinal for c in clocks)
                checks[f'{ordinal}_finite_delta']=all(bool(torch.isfinite(v).all()) for v in delta)
                torch.save(dict(parameter_names=self.trace_names,descent_delta=delta,applied_learning_rates=self.trace_lr,completed_updates=ordinal),trace/f'update_{ordinal}_delta.pt')
                rows.append(dict(completed_updates=ordinal,applied_learning_rates=self.trace_lr,
                    delta_norm=float(sum(v.square().sum() for v in delta).sqrt())))
                self.trace_before=None
            checks[f'{ordinal}_post_hook_rng_exact']=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
            if ordinal>=496:
                write(trace/'PARTIAL_RESULT.json',dict(checks=checks,rows=rows,completed_updates=ordinal))
                print(json.dumps(dict(passive_optimizer_trace=ordinal)),flush=True)
            assert all(checks.values()),[k for k,v in checks.items() if not v]

    original_workspace=training.GeometryGeneratorWorkspace;original_run=training.RUN
    training.GeometryGeneratorWorkspace=RecordedWorkspace;training.RUN=RUN
    try:training.train_arm('demo_geometry')
    finally:
        training.GeometryGeneratorWorkspace=original_workspace;training.RUN=original_run
        for workspace in workspace_holder:
            for hook in workspace.trace_handles:hook.remove()
            workspace.model.get_optimizer=workspace.original_get_optimizer
    assert len(workspace_holder)==1 and workspace_holder[0].trace_ordinal==512
    finish_trace(checks,rows,workspace_holder[0].trace_names)


def finish_trace(checks,rows,names):
    trace=RUN/'optimizer_trace'
    readback=json.loads((RUN/'demo_geometry/EXACT_TRAINING_REPLAY_READBACK.json').read_text());assert readback['checks_passed']
    for key,value in readback['checks'].items():checks['official_replay_'+key]=value
    original_losses=json.loads((SELF_SOURCE/'demo_geometry/PAIRED_LOSSES.json').read_text())
    replay_losses=json.loads((RUN/'demo_geometry/PAIRED_LOSSES.json').read_text())
    checks['all512_all_loss_fields_exact']=original_losses==replay_losses
    # These are our own recorded snapshots, including original OmegaConf optimizer
    # group metadata; retain that metadata rather than rewriting the artifacts.
    before=torch.load(trace/'after_update_496.pt',map_location='cpu',weights_only=False);after=torch.load(trace/'after_update_512.pt',map_location='cpu',weights_only=False)
    with (SELF_SOURCE/'demo_geometry/checkpoints/endpoint.ckpt').open('rb') as f:reference=torch.load(f,pickle_module=dill,map_location='cpu')['state_dicts']
    checks['recorded512_fullmodel_exact']=equal(after['model'],reference['model'])
    checks['recorded512_allmoments_exact']=equal(after['optimizer']['state'],reference['optimizer']['state'])
    total=[torch.zeros_like(before['model'][name],dtype=torch.float64) for name in names]
    for ordinal in range(497,513):
        saved=torch.load(trace/f'update_{ordinal}_delta.pt',map_location='cpu');assert saved['parameter_names']==names
        for summed,value in zip(total,saved['descent_delta']):summed.add_(value)
    expected=[before['model'][name].double()-after['model'][name].double() for name in names]
    checks['all16_actual_deltas_telescope_exact']=all(torch.equal(a,b) for a,b in zip(total,expected))
    torch.save(dict(parameter_names=names,descent_delta=total),trace/'LAST16_DESCENT_DELTA.pt')
    write(trace/'RESULT.json',dict(execution_completed=True,checks_passed=all(checks.values()),checks=checks,rows=rows,
        extra_executed_training_updates=512,new_unique_trajectory_updates=0,original_endpoint_extended=False,new_model_forwards_for_evaluation=0,new_physics_steps=0,
        scope='Complete exact replay of original512 training updates; passive PyTorch optimizer hooks record actual497..512 parameter displacements and496/512 fullmodel+Adam snapshots. All losses/batches/initial/full endpointmodel+Adam exact. No update513 or new optimization trajectory. Sampling before/after and scientific interpretation remain separate.'))
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    print(json.dumps(dict(checks=len(checks),exact_original_training_replay=True)),flush=True)


if __name__=='__main__':main()
