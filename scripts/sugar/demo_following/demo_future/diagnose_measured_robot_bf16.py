"""Full-model diagnostic of AMP gradient preservation, without optimizer updates."""
import json
import argparse
import os
import socket

import hydra
import torch
from torch import nn

from scripts.sugar.demo_following.demo_future.preflight_generator_measured_robot import RUN,SOURCE
from scripts.sugar.demo_following.demo_future.train_generator_geometry import training_config
from scripts.sugar.demo_following.demo_future.generator_measured_robot import WEIGHT_KEY,EXTRA_KEY,PartitionedRobotAffine,original_robot_slice
from scripts.sugar.demo_following.demo_future.preflight_generator_latent_replay import gradients,same_rng
from scripts.sugar.demo_following.demo_future.generator_paired_objective import rng_state
from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--readback',action='store_true')
    if parser.parse_args().readback:
        torch.set_num_threads(1)
        out=RUN/'bf16_affine_accumulation_diagnostic';assert not (out/'SAVED_READBACK.json').exists()
        result=json.loads((out/'RESULT.json').read_text());assert result['checks_passed']
        a=torch.load(out/'ROBOT_GRADIENTS.pt',map_location='cpu')
        old=a['original'];wide=a['wide68_slices'][:,:36];row=result['rows']['wide68_slices']['different_parameters'][WEIGHT_KEY]
        checks=dict(original_repeat_exact=torch.equal(old,a['original_repeat']),partitioned_exact=torch.equal(old,a['partitioned_leaf_parameters']),
                    wide_difference_count_exact=int((old!=wide).sum())==row['different_elements'],wide_max_error_exact=float((old-wide).abs().max())==row['max_absolute'],
                    added_weight_gradient_finite_nonzero=bool(torch.isfinite(a['partitioned_leaf_parameters_extra']).all()) and bool(a['partitioned_leaf_parameters_extra'].count_nonzero()))
        write(out/'SAVED_READBACK.json',dict(checks_passed=all(checks.values()),checks=checks,
            old_equals_bf16_rounded_wide=torch.equal(old,wide.to(torch.bfloat16).float()),new_model_forwards=0,new_optimizer_updates=0))
        assert all(checks.values());print(json.dumps(dict(checks=len(checks),old_equals_bf16_rounded_wide=torch.equal(old,wide.to(torch.bfloat16).float()))),flush=True)
        return
    assert os.environ.get('SLURM_STEP_ID') and not socket.gethostname().startswith(('login','mgmtserver'))
    torch.set_num_threads(8)
    out=RUN/'bf16_affine_accumulation_diagnostic';out.mkdir(exist_ok=False)
    plan=json.loads((RUN/'PROTOCOL.json').read_text());old=json.loads((SOURCE/'PROTOCOL.json').read_text())
    dataset=hydra.utils.instantiate(training_config(plan,'zero_context').task.dataset)
    torch.manual_seed(plan['seed']);batch=next(iter(torch.utils.data.DataLoader(dataset,batch_size=144,shuffle=True,num_workers=0)))
    batch={k:({n:v.cuda() for n,v in value.items()} if isinstance(value,dict) else value.cuda()) for k,value in batch.items()}
    checks={};rows={};reference=None;robot_gradients={}
    for label,settings in [('original',old),('original_repeat',old),('wide68_slices',plan),('partitioned_leaf_parameters',plan)]:
        torch.manual_seed(272241);policy=hydra.utils.instantiate(training_config(settings,'zero_context').policy)
        policy.set_normalizer(dataset.get_normalizer());policy.normalizer.requires_grad_(False)
        if label=='partitioned_leaf_parameters':
            layer=policy.obs_encoder.robot_state_net[0]
            extra=layer.weight[:,36:].detach().clone();base=layer.weight[:,:36].detach().clone()
            layer.weight=nn.Parameter(base);layer.register_parameter('measured_weight',nn.Parameter(extra));layer.__class__=PartitionedRobotAffine;layer.in_features=68
        before={k:v.cpu().clone() for k,v in policy.state_dict().items()}
        policy.cuda().train();policy.generated_state_step=0;torch.manual_seed(272240)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=policy.compute_loss(batch,True)
        grads=gradients(loss,policy);stream=rng_state(torch.device('cuda'))
        if reference is None:reference=(float(loss.detach()),{k:v.cpu() for k,v in grads.items()},stream)
        differences={}
        for name,value in reference[1].items():
            current=original_robot_slice(name,grads[name]).cpu()
            if not torch.equal(value,current):differences[name]=dict(max_absolute=float((value-current).abs().max()),different_elements=int((value!=current).sum()),old_norm=float(value.double().norm()),new_norm=float(current.double().norm()))
        checks[label+'_same_full_loss_rng']=float(loss.detach())==reference[0] and same_rng(stream,reference[2])
        checks[label+'_fullstate_unchanged']=all(torch.equal(v.cpu(),before[k]) for k,v in policy.state_dict().items())
        checks[label+'_all_gradients_finite']=all(bool(torch.isfinite(v).all()) for v in grads.values())
        rows[label]=dict(loss=float(loss.detach()),different_parameters=differences,full_original_gradients_exact=not differences,parameter_count=sum(p.numel() for p in policy.parameters()))
        robot_gradients[label]=grads[WEIGHT_KEY].cpu()
        if EXTRA_KEY in grads:robot_gradients[label+'_extra']=grads[EXTRA_KEY].cpu()
        print(json.dumps(dict(label=label,**rows[label])),flush=True)
        del policy,grads
    checks['original_repeat_exact']=rows['original_repeat']['full_original_gradients_exact']
    torch.save(robot_gradients,out/'ROBOT_GRADIENTS.pt')
    write(out/'RESULT.json',dict(checks_passed=all(checks.values()),checks=checks,rows=rows,new_optimizer_updates=0,new_physics_steps=0,
        hypothesis='A contiguous non-leaf weight slice gets separate AMP casts per forward; original leaf parameter AMP cache shares casts and BF16 accumulation across q/replay forwards. Partitioning the same affine into original36 and extra32 leaf parameters may restore exact accumulation. Full loss and every named original gradient tested; no prediction benefit claim.'))
    assert all(checks.values())


if __name__=='__main__':main()
