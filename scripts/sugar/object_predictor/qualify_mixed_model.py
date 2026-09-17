"""Full released model qualification on the actual first balanced training batch."""
import argparse
import gc
import json
import os
from pathlib import Path
import numpy as np
import torch
from .data import ContactDataset,MODES,collate
from .sampling import BalancedAcquisitionBatchSampler
from .model import ObjectPredictor,loss_components,disable_stochastic_regularizers


def main(args):
    if not os.environ.get('SLURM_STEP_ID'): raise RuntimeError('Compute step required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    prepared=json.loads((Path(args.data)/'PROTOCOL.json').read_text())
    if args.normal_policy!=prepared.get('normal_policy','stored'):raise ValueError('Normal policy differs from prospective data protocol')
    if args.normal_policy=='hand_surface':
        from .hand_surface_normals import verify_geometry_signature
        verify_geometry_signature(prepared.get('normal_geometry_signature'))
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=True)['state_dict']
    checks={};details={};identities=None
    for mode in MODES:
        torch.manual_seed(310016);torch.cuda.manual_seed_all(310016)
        dataset=ContactDataset(args.data,mode,'train',32,5,'episode_uniform_recent',150.,args.normal_policy)
        ids=next(iter(BalancedAcquisitionBatchSampler(dataset,4,310018)))
        batch={k:v.cuda() for k,v in collate([dataset[i] for i in ids]).items()}
        actual=list(zip(batch['episode'].tolist(),batch['frame'].tolist()))
        if identities is None: identities=actual
        checks[mode+'_matched_first_batch']=actual==identities
        model=ObjectPredictor(args.checkpoint,32,True).cuda()
        state=model.backbone.state_dict()
        checks[mode+'_all_released_tensors_exact']=all(torch.equal(state[k.replace('embedding.stem.linear.','embedding.stem.linear.original.')].cpu(),v) for k,v in original.items())
        checks[mode+'_full_parameter_counts']=model.original_parameter_count==137253744 and sum(p.numel() for p in model.parameters())==138407503
        extra=list(model.backbone.embedding.stem.linear.extra.parameters())
        other=[p for n,p in model.backbone.named_parameters() if '.extra.' not in n]
        optimizer=torch.optim.AdamW([dict(params=other,lr=1e-5),dict(params=extra,lr=5e-4),dict(params=model.head.parameters(),lr=1e-5)],weight_decay=.01)
        bound=[id(p) for g in optimizer.param_groups for p in g['params']]
        checks[mode+'_all_optimizer_bindings_once']=len(bound)==len(set(bound)) and set(bound)=={id(p) for p in model.parameters()}
        model.train();disable_stochastic_regularizers(model)
        prediction=model(batch);parts=loss_components(prediction,batch['target']);sum(parts.values()).backward()
        missing_gradients=[n for n,p in model.named_parameters() if p.grad is None]
        nonfinite_gradients=[n for n,p in model.named_parameters()
                             if p.grad is not None and not torch.isfinite(p.grad).all().item()]
        checks[mode+'_only_official_unmasked_token_unused']=missing_gradients==['backbone.embedding.mask_token']
        checks[mode+'_finite_all_used_gradients']=not nonfinite_gradients
        checks[mode+'_unused_token_full_and_unchanged']=(model.backbone.embedding.mask_token.numel()==54 and
            torch.equal(model.backbone.embedding.mask_token.detach().cpu(),original['embedding.mask_token']))
        for stage in range(5):
            checks[f'{mode}_stage{stage}_nonzero_gradient']=any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in model.named_parameters() if f'enc.enc{stage}.' in n)
        checks[mode+'_finite_output_and_loss']=bool(torch.isfinite(prediction).all()) and all(bool(torch.isfinite(v)) for v in parts.values())
        model.eval()
        with torch.no_grad(): a=model(batch);b=model(batch)
        checks[mode+'_frozen_repeat_exact']=torch.equal(a,b)
        details[mode]=dict(first_batch=actual,losses={k:float(v.detach()) for k,v in parts.items()},
                           missing_gradients=missing_gradients,nonfinite_gradients=nonfinite_gradients,
                           repeat_max=float((a-b).abs().max()),peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
        del model,optimizer,state,batch,prediction,parts,a,b,extra,other,dataset
        gc.collect();torch.cuda.empty_cache()
    report=dict(checks=checks,passed=all(checks.values()),details=details,new_optimizer_updates=0,
                data=args.data,normal_policy=args.normal_policy,normal_geometry_signature=prepared.get('normal_geometry_signature'))
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--normal-policy',choices=('stored','hand_surface'),default='stored');main(ap.parse_args())
