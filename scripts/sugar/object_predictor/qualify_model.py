"""Actual saved-observation full-model forward/backward qualification; zero updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import OBSERVATION_KEYS, encode_observations, target_at, collate,history_indices
from .model import ObjectPredictor, loss_components, disable_stochastic_regularizers


def main(args):
    torch.manual_seed(310016)
    torch.set_num_threads(4)
    with np.load(args.episode) as source:
        arrays={k:source[k] for k in source.files}
    frame=args.frame if args.frame is not None else int(np.argmax(arrays['normal_load_n'].sum(1)))
    frame=max(frame,args.history-1)
    selected=history_indices(frame,args.history,args.history_policy)
    obs={k:arrays[k][selected].copy() for k in OBSERVATION_KEYS}
    rows=[]
    encoded=[]
    for mode in ('geometry','geometry_contact','geometry_contact_force'):
        row=encode_observations(obs,mode,time_scale_s=args.time_scale_s)
        encoded.append(row)
        row.update(target=target_at(arrays,frame),episode=0,frame=frame,contact=True)
        rows.append(row)
    checks={
        'geometry_no_contact_or_force_features':bool(np.count_nonzero(encoded[0]['feat'][...,9:15])==0),
        'contact_no_force_features':bool(np.count_nonzero(encoded[1]['feat'][...,10:15])==0),
        'force_is_present':bool(np.count_nonzero(encoded[2]['feat'][...,10:15])>0),
        'force_does_not_change_contact_coordinates':bool(np.array_equal(encoded[1]['coord'],encoded[2]['coord'])),
    }
    moved={k:v.copy() for k,v in obs.items()}
    delta=np.array([1.2,-.3,2.1],np.float32)
    moved['hand_pose_w'][...,:3]+=delta
    for key in ('hand_sites_w','contact_position_w'):
        moved[key]+=delta
    translated=encode_observations(moved,'geometry_contact_force',time_scale_s=args.time_scale_s)
    checks['common_world_translation_invariant']=bool(np.allclose(translated['feat'],encoded[2]['feat'],atol=3e-6))
    model=ObjectPredictor(args.checkpoint,history=args.history,deterministic_pooling=args.deterministic_pooling).cuda()
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state=model.backbone.state_dict()
    restored=0
    for name,value in ckpt['state_dict'].items():
        mapped=name.replace('embedding.stem.linear.','embedding.stem.linear.original.')
        if not torch.equal(state[mapped].cpu(),value):
            raise AssertionError('Released parameter mismatch: '+name)
        restored+=value.numel()
    checks['all_released_tensors_exact']=True
    batch={k:v.cuda() for k,v in collate(rows).items()}
    model.train()
    if args.disable_stochastic_regularizers:
        disable_stochastic_regularizers(model)
    pred=model(batch)
    components=loss_components(pred,batch['target'])
    loss=sum(components.values())
    loss.backward()
    gradients={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    checks['finite_outputs_loss_gradients']=bool(torch.isfinite(pred).all() and torch.isfinite(loss)
                                               and all(np.isfinite(v) for v in gradients.values()))
    checks['sensor_affine_receives_gradient']=gradients.get('backbone.embedding.stem.linear.extra.weight',0)>0
    for stage in range(5):
        checks[f'full_encoder_stage_{stage}_receives_gradient']=any(v>0 for k,v in gradients.items() if f'enc.enc{stage}.' in k)
    checks['head_receives_gradient']=gradients.get('head.weight',0)>0
    model.eval()
    with torch.no_grad():
        repeat0=model(batch)
        repeat1=model(batch)
    checks['frozen_repeat']=bool(torch.allclose(repeat0,repeat1,atol=1e-5,rtol=1e-5))
    report=dict(checks=checks,passed=all(checks.values()),checkpoint=args.checkpoint,
                source_episode=args.episode,source_is_physical_diagnostic=True,optimizer_updates=0,
                history=args.history,history_policy=args.history_policy,time_scale_s=args.time_scale_s,selected_frames=selected.tolist(),
                original_parameters=model.original_parameter_count,
                total_parameters=sum(p.numel() for p in model.parameters()),restored_values=restored,
                gradients=gradients,losses={k:float(v.detach()) for k,v in components.items()},
                max_repeat_difference=float((repeat0-repeat1).abs().max()),
                peak_gpu_memory_gb=torch.cuda.max_memory_allocated()/1e9)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='gradients'}),flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--episode',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--deterministic-pooling',action='store_true')
    ap.add_argument('--disable-stochastic-regularizers',action='store_true')
    ap.add_argument('--history-policy',choices=('contiguous','episode_uniform_recent'),default='contiguous')
    ap.add_argument('--time-scale-s',type=float,default=1.)
    ap.add_argument('--frame',type=int)
    ap.add_argument('--history',type=int,default=8)
    main(ap.parse_args())
