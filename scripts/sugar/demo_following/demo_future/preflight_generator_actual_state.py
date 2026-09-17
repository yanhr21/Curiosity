"""Full original/extended objective qualification on initial and learned actual models."""
import argparse
import copy
import json

import dill
import torch

from scripts.sugar.demo_following.demo_future.run_generator_actual_state_supervision import RUN,MODEL,ROOT,write
from scripts.sugar.demo_following.demo_future.generator_workspace import warm_start_geometry_policy
from scripts.sugar.demo_following.demo_future.generator_latent_replay import ActualBranchLatentReplayDataset
from scripts.sugar.demo_following.demo_future.generator_actual_state_supervision import configure_actual_state_supervision
from scripts.sugar.demo_following.demo_future.generator_paired_objective import rng_state,restore_rng


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(v,w) for v,w in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','cuda'],required=True);args=parser.parse_args()
    device=torch.device(args.device);torch.set_num_threads(1)
    out=RUN/('ACTUAL_STATE_BF16_PREFLIGHT.json' if device.type=='cuda' else 'ACTUAL_STATE_CPU_PREFLIGHT.json');assert not out.exists()
    plan=json.loads((RUN/'PROTOCOL.json').read_text());old_plan=json.loads((MODEL/'PROTOCOL.json').read_text())
    dataset=ActualBranchLatentReplayDataset(replay_source=plan['generated_state_objective']['replay_source'],**plan['branch_dataset'])
    from torch.utils.data._utils.collate import default_collate
    from sugar_il.common.pytorch_util import dict_apply
    batch=dict_apply(default_collate([dataset[i] for i in range(144)]),lambda v:v.to(device))
    checks={k:plan[k]==old_plan[k] for k in ('seed','epochs','batch_size','optimizer','scheduler','branch_dataset','paired_objective','generated_state_objective','normalizer_state','robot_state_conditioning','frozen_evaluation')}
    gradients={};samples={}
    for arm in ('zero_context','demo_geometry'):
        policy=warm_start_geometry_policy(ROOT/'SUGAR/demo_ckpts/CarryBox/generator.ckpt',arm,goal_condition_mode=plan['goal_condition_mode'],
            noise_coupling=plan['noise_coupling'],paired_objective=plan['paired_objective'],generated_state_objective=plan['generated_state_objective'],robot_state_conditioning=plan['robot_state_conditioning'])
        policy.set_normalizer(dataset.get_normalizer());policy.to(device);policy.normalizer.requires_grad_(False);policy.eval()
        before=rng_state(device);new=copy.deepcopy(policy);configure_actual_state_supervision(new,plan['actual_state_supervision'])
        checks[arm+'_configuration_preserves_rng']=same(before,rng_state(device))
        checks[arm+'_full8327408_same_parameter_names_shapes']=sum(p.numel() for p in new.parameters())==8327408 and [(n,tuple(p.shape)) for n,p in new.named_parameters()]==[(n,tuple(p.shape)) for n,p in policy.named_parameters()]
        expected=torch.load(MODEL/arm/'INITIAL_MODEL.pt',map_location=device)['model']
        checks[arm+'_full_release_initial_exact']=same(policy.state_dict(),expected) and same(new.state_dict(),expected)
        for label,state in [('initial',expected),('learned',None)]:
            if state is None:
                with (MODEL/arm/'checkpoints/endpoint.ckpt').open('rb') as f:state=torch.load(f,pickle_module=dill,map_location=device)['state_dicts']['model']
            policy.load_state_dict(state,strict=True);new.load_state_dict(state,strict=True)
            tag=arm+'_'+label
            def evaluate(model,weight):
                model.train();model.zero_grad(set_to_none=True);model.generated_state_step=0
                if hasattr(model,'expert_weight'):
                    model.expert_weight=weight;model.expert_step=0;model.expert_counts.zero_()
                torch.manual_seed(272600)
                if device.type=='cuda':torch.cuda.manual_seed_all(272600)
                captured=[]
                def observe(module,inputs,kwargs):captured.append((inputs[0].detach().clone(),inputs[1].detach().clone()))
                handle=model.model.register_forward_pre_hook(observe,with_kwargs=True)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    loss=model.compute_loss(batch,training=True)
                handle.remove();loss.backward()
                return float(loss.detach()),{n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters()},rng_state(device),dict(model.last_paired_loss),captured
            base,bgrad,brng,brecord,bforwards=evaluate(policy,0)
            disabled,dgrad,drng,drecord,dforwards=evaluate(new,0)
            checks[tag+'_disabled_full_loss_gradient_rng_forwards_exact']=base==disabled and same(bgrad,dgrad) and same(brng,drng) and same(brecord,drecord) and same(bforwards,dforwards)
            active,agrad,arng,arecord,aforwards=evaluate(new,.1)
            checks[tag+'_active_original_components_exact']=all(arecord[k]==v for k,v in brecord.items())
            checks[tag+'_active_original_rng_and_forwards_exact']=same(brng,arng) and same(bforwards,aforwards[:len(bforwards)]) and len(aforwards)==len(bforwards)+1
            checks[tag+'_new_q_full144rows_50time_range']=aforwards[-1][0].shape==(144,8,36) and aforwards[-1][1].shape==(144,) and bool(((aforwards[-1][1]>=0)&(aforwards[-1][1]<50)).all())
            checks[tag+'_finite_correct_loss_arithmetic']=abs(active-(base+.1*arecord['expert']))<=2e-6*max(1,abs(active)) and arecord['expert']>0
            checks[tag+'_all_full_gradients_finite']=all(g is None or torch.isfinite(g).all() for g in agrad.values())
            delta=sum(float((agrad[n]-v).float().square().sum()) for n,v in bgrad.items() if v is not None)
            gradients[tag]=dict(auxiliary_gradient_difference_norm=delta**.5,base_loss=base,expert_loss=arecord['expert'],combined_loss=active)
            checks[tag+'_nonzero_actual_state_correction_gradient']=delta>0
            checks[tag+'_actual_clocks_and144exposures']=new.expert_step==new.generated_state_step==1 and int(new.expert_counts.sum())==144 and arecord['expert_seed']==272500
            checks[tag+'_full_state_normalizer_unchanged']=same(new.state_dict(),state) and same(policy.state_dict(),state)
            # Inference must remain the complete original sampler and have no auxiliary side effects.
            small={k:v[:1] for k,v in batch['obs'].items()}
            policy.eval();new.eval()
            with torch.inference_mode():
                torch.manual_seed(272090);x=policy.predict_action(small)
                torch.manual_seed(272090);y=new.predict_action(small)
            checks[tag+'_original_sampler_exact']=torch.equal(x,y)
            policy.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)
        counts=torch.zeros(355,dtype=torch.long)
        for step in range(512):
            indices=new.expert_indices(step);counts.index_add_(0,indices,torch.ones_like(indices))
            checks[arm+'_cyclic_group_schedule_'+str(step)]=len(indices)==144 and all(torch.isin(indices[g*36:(g+1)*36],group).all() for g,group in enumerate(new.expert_groups))
        checks[arm+'_all355_states_and73728_exposures']=bool((counts>0).all()) and int(counts.sum())==73728
        samples[arm]=dict(counts=counts.tolist(),group_totals=[int(counts[g].sum()) for g in new.expert_groups])
        del policy,new
    checks['both_arms_same355_exposures']=samples['zero_context']==samples['demo_geometry']
    report=dict(checks_passed=all(checks.values()),checks=checks,gradient_metrics=gradients,expert_exposures=samples,
        device=str(device),bf16_autocast=device.type=='cuda',loss_module_training_mode=True,full_parameter_count=8327408,original_rows=144,additional_expert_rows=144,
        new_optimizer_updates=0,new_physics_steps=0,scope='Both full initial and learned models: disabled-aux original loss/all gradients/RNG/forwards exact; active old loss/forwards/RNG exact with finite nonzero expert correction; unchanged sampler/fullstate, all512 group clocks. No optimization benefit or expert physical recovery claimed.')
    write(out,report);print(json.dumps(dict(checks_passed=report['checks_passed'],checks=len(checks),gradients=gradients)),flush=True)
    assert report['checks_passed'],[k for k,v in checks.items() if not v]


if __name__=='__main__':main()
