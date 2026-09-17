"""Stored full Adam moments versus fresh official sampler gradients; no step."""
import argparse
import hashlib
import json

import dill
import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.probe_generator_reverse_process import PAIRED_OUT,SELF_SOURCE,PARENT,GeneratorWrapper,restore_geometry_state,write
from scripts.sugar.demo_following.demo_future.audit_generator_generated_state_gradient import pair_stats
from scripts.sugar.demo_following.demo_future import audit_generator_gap_objective as optimizer_tools

FRESH=PAIRED_OUT/'fresh_full_sampler_gradient'
EVIDENCE=FRESH/'objective_conflict_readback'
OBJECTIVE=PAIRED_OUT/'objective_vs_sampler_gradient'
OUT=FRESH/'stored_adam_direction'


def comparisons(names,direction):
    sampler=torch.load(FRESH/'MEAN_GRADIENTS.pt',map_location='cpu')
    objective=torch.load(OBJECTIVE/'MEAN_GRADIENTS.pt',map_location='cpu')
    assert names==sampler['parameter_names']==objective['parameter_names']
    result={'sampler':{key:pair_stats(names,pair[0],direction) for key,pair in sampler['means'].items()}}
    components=objective['means']['all_noise'];grads=list(components)+[[a+b+c for a,b,c in zip(*components)]]
    result['objective']={key:pair_stats(names,g,direction) for key,g in zip(('base','rank','replay','total'),grads)}
    return result


def readback():
    torch.set_num_threads(8)
    result=json.loads((OUT/'RESULT.json').read_text());assert result['checks_passed'] and result['plot_inspected']
    saved=torch.load(OUT/'STORED_DIRECTION.pt',map_location='cpu')
    current=comparisons(saved['parameter_names'],saved['direction']);checks={}
    for scope,items in current.items():
        for label,groups in items.items():
            for group,metrics in groups.items():
                for key,value in metrics.items():
                    expected=result['comparisons'][scope][label][group][key]
                    checks[f'{scope}_{label}_{group}_{key}']=value==expected if value is None else bool(np.isclose(value,expected,rtol=1e-10,atol=1e-12))
    assert all(checks.values())
    write(OUT/'SAVED_READBACK.json',dict(checks_passed=True,checks=checks,scope='Independent saved fullmoment direction and fullgradient vectors; no optimizer construction, model forwards or updates.'))
    print(json.dumps(dict(checks=len(checks))),flush=True)


def run():
    torch.set_num_threads(8)
    prior=json.loads((EVIDENCE/'RESULT.json').read_text());decision=json.loads((EVIDENCE/'DECISION.json').read_text())
    assert prior['checks_passed'] and prior['plot_inspected'] and decision['next_action']=='stored_adam_direction_vs_fresh_terminal_gradient'
    protocol=json.loads((EVIDENCE/'NEXT_STORED_ADAM_PROTOCOL.json').read_text())
    OUT.mkdir(exist_ok=False);write(OUT/'PROTOCOL.json',protocol)
    path=SELF_SOURCE/'demo_geometry/checkpoints/endpoint.ckpt';checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open('rb') as f:payload=torch.load(f,pickle_module=dill,map_location='cpu')
    state=payload['state_dicts']['model'];saved_optimizer=payload['state_dicts']['optimizer']
    optimizer_copy={pid:{key:value.clone() if torch.is_tensor(value) else value for key,value in record.items()} for pid,record in saved_optimizer['state'].items()}
    policy=GeneratorWrapper.load(str(PARENT),device='cpu').policy
    restore_geometry_state(policy,state,'demo_geometry',goal_condition_mode='zero_diagnostic')
    policy.eval();policy.normalizer.requires_grad_(False)
    names,params=zip(*[(n,p) for n,p in policy.named_parameters() if p.requires_grad])
    optimizer_tools.RUN=SELF_SOURCE
    vectors,binding=optimizer_tools.stored_adam_direction(policy,payload)
    direction=[vectors[n].cpu() for n in names]
    checks=dict(full8319216=sum(p.numel() for p in policy.parameters())==8319216,
        all_optimizer_parameters_bound=set(vectors)==set(names) and len(binding['bindings'])==len(names),
        all512_moment_clocks=all(item['step']==512 for item in binding['bindings']),
        endpoint_learning_rate_zero=all(group['lr']==0 for group in binding['param_groups']),
        finite_direction=all(bool(torch.isfinite(v).all()) for v in direction),
        fullstate_unchanged=state.keys()==policy.state_dict().keys() and all(torch.equal(v.cpu(),state[k]) for k,v in policy.state_dict().items()),
        full_moment_state_unchanged=all(torch.equal(v,saved_optimizer['state'][pid][key]) if torch.is_tensor(v) else v==saved_optimizer['state'][pid][key] for pid,record in optimizer_copy.items() for key,v in record.items()),
        no_parameter_gradients=all(p.grad is None for p in policy.parameters()),
        checkpoint_file_unchanged=hashlib.sha256(path.read_bytes()).hexdigest()==checksum)
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    write(OUT/'PARAMETER_BINDING.json',binding)
    torch.save(dict(parameter_names=names,direction=direction),OUT/'STORED_DIRECTION.pt')
    result=comparisons(names,direction)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(17,12))
    for phase,ax in zip(protocol['train_phases'],axes.flat):
        old=prior['mean_comparisons']['all_noise'][str(phase)]['total']['full_model']['cosine'];moment=result['sampler'][str(phase)]['full_model']['cosine']
        ax.bar(['raw objective','stored moments'],[old,moment],color=['C0','C1']);ax.axhline(0,color='gray');ax.set_ylim(-1.05,1.05);ax.set_title(f'TRAIN{phase}');ax.grid(axis='y',alpha=.25)
    fig.suptitle('Fresh full-sampler gradient cosine; stored Adam per-unit-lr direction only, endpoint lr=0')
    fig.tight_layout();fig.savefig(OUT/'SAMPLER_ADAM.png',dpi=140);plt.close(fig)
    write(OUT/'RESULT.json',dict(execution_completed=True,checks_passed=True,checks=checks,comparisons=result,plot_inspected=False,new_model_forwards=0,new_gradients=0,new_optimizer_updates=0,new_physics_steps=0,
        scope='Exact official optimizer parameter grouping and saved512 moments. Bias-corrected FP64 m/(sqrt(v)+eps)+decay*endpointparameter is a per-unit-learning-rate historical direction. Endpoint learning rate is0, so actual endpoint-state update iszero. No inserted new gradient, optimizer.step, update513, actual step512 reconstruction or training benefit claim.'))
    print(json.dumps(dict(checks=len(checks),bindings=len(binding['bindings']),phase_descent_dots={key:value['full_model']['dot'] for key,value in result['sampler'].items()},objective_descent_dots={key:value['full_model']['dot'] for key,value in result['objective'].items()})),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--readback',action='store_true')
    args=parser.parse_args();readback() if args.readback else run()


if __name__=='__main__':main()
