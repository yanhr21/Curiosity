"""Matched supervised adaptation of the full released Utonia object encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ContactDataset, MODES, collate
from .model import ObjectPredictor, loss_components, metrics, disable_stochastic_regularizers
from .sampling import BalancedAcquisitionBatchSampler


def device_batch(batch):
    return {k:v.cuda() for k,v in batch.items()}


def supervised_components(prediction,target,mass_label_weight=None):
    parts=loss_components(prediction,target)
    if mass_label_weight is not None:
        error=((prediction[:,12]-target[:,12])/.7).square()
        parts['mass']=(error*mass_label_weight).sum()/mass_label_weight.sum().clamp_min(1.)
    return parts


@torch.no_grad()
def evaluate(model,loader):
    model.eval()
    outputs,targets,episodes,frames,contacts,mass_labels=[],[],[],[],[],[]
    for batch in loader:
        pred=model(device_batch(batch)).cpu()
        outputs.append(pred)
        targets.append(batch['target'])
        episodes.append(batch['episode'])
        frames.append(batch['frame'])
        contacts.append(batch['contact'])
        if 'mass_label_weight' in batch:mass_labels.append(batch['mass_label_weight'])
    pred,target=torch.cat(outputs),torch.cat(targets)
    contact=torch.cat(contacts)
    values=metrics(pred,target)
    episode_ids=torch.cat(episodes)
    report={}
    groups=[('all',torch.ones_like(contact)),('contact',contact),('no_contact',~contact)]
    if mass_labels:groups.append(('airborne_hold',torch.cat(mass_labels)>0))
    for group,mask in groups:
        report[group]={'samples':int(mask.sum())}
        if mass_labels:report[group].update(episodes_covered=int(episode_ids[mask].unique().numel()),total_episodes=int(episode_ids.unique().numel()))
        if mask.any():
            for name,value in values.items():
                report[group][name]=dict(mean=float(value[mask].mean()),median=float(value[mask].median()),
                                         p90=float(torch.quantile(value[mask],.9)))
                per_episode=[value[mask & (episode_ids==e)].mean() for e in episode_ids[mask].unique()]
                report[group][name]['equal_episode_mean']=float(torch.stack(per_episode).mean())
    arrays=dict(prediction=pred.numpy(),target=target.numpy(),episode=torch.cat(episodes).numpy(),
                frame=torch.cat(frames).numpy(),contact=contact.numpy())
    if mass_labels:arrays['mass_label_weight']=torch.cat(mass_labels).numpy()
    return report,arrays


def train(args):
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login','mgmtserver')):
        raise RuntimeError('Training must run in retained compute step')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'model.pt').exists() or (out/'train.jsonl').exists():
        raise FileExistsError('Use an explicit new run; do not overwrite a training endpoint')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    model=ObjectPredictor(args.checkpoint,history=args.history,deterministic_pooling=args.deterministic_pooling).cuda()
    params=sum(p.numel() for p in model.parameters())
    history_options=dict(history_policy=args.history_policy,time_scale_s=args.time_scale_s,normal_policy=args.normal_policy)
    collator=collate
    if args.perception_corpus:
        from .perception_dataset import PerceptionDataset,MatchedGeometrySampler,collate_perception
        if args.train_sampler!='matched_geometry' or args.mode!='geometry_contact_force':
            raise ValueError('Use the declared matched perception experiment')
        factory=lambda split:PerceptionDataset(args.data,args.mode,split,args.history,args.stride,
            representation=args.input_representation,**history_options)
        train_set,val_set,test_set=(factory(s) for s in ('train','val','test'))
        collator=collate_perception
    else:
        if args.mass_airborne_guard or args.input_representation!='centroid':raise ValueError('Surface/guard options require the perception corpus')
        train_set=ContactDataset(args.data,args.mode,'train',args.history,args.stride,**history_options)
        val_set=ContactDataset(args.data,args.mode,'val',args.history,args.stride,**history_options)
        test_set=ContactDataset(args.data,args.mode,'test',args.history,args.stride,**history_options)
    data_protocol_path=Path(args.data)/'PROTOCOL.json'
    if args.train_sampler=='balanced_acquisitions':
        prepared=json.loads(data_protocol_path.read_text())
        required={'steps':prepared['steps_per_arm'],'history':prepared['history'],
                  'history_policy':prepared['history_policy'],'time_scale_s':prepared['time_scale_s'],
                  'normal_policy':prepared.get('normal_policy','stored'),'seed':prepared['training_seed'],
                  'batch':sum(prepared['batch'].values()),'head_lr':prepared['head_lr'],
                  'evaluate_every':prepared['evaluate_every'],'deterministic_pooling':prepared['deterministic_pooling'],
                  'disable_stochastic_regularizers':prepared['disable_stochastic_regularizers']}
        for key,value in required.items():
            if getattr(args,key)!=value:raise ValueError(f'Training differs from prospective mixed protocol: {key}')
        if args.normal_policy=='hand_surface':
            from .hand_surface_normals import verify_geometry_signature
            verify_geometry_signature(prepared.get('normal_geometry_signature'))
    generator=torch.Generator().manual_seed(args.seed+1)
    if args.train_sampler=='matched_geometry':
        if not args.perception_corpus:raise ValueError('Matched geometry sampler requires perception corpus')
        prepared=json.loads(data_protocol_path.read_text())['planned_matched_training']
        required={'steps':prepared['steps_per_arm'],'history':prepared['history'],
                  'history_policy':prepared['history_policy'],'time_scale_s':prepared['time_scale_s'],
                  'stride':prepared['clock_stride'],'batch':prepared['batch'],'seed':prepared['initial_seed'],
                  'head_lr':prepared['head_lr'],'evaluate_every':500,'deterministic_pooling':True,
                  'disable_stochastic_regularizers':True,'normal_policy':'hand_surface'}
        for key,value in required.items():
            if getattr(args,key)!=value:raise ValueError('Perception training protocol differs: '+key)
        if not json.loads((Path(args.data)/'TRAIN_QUALIFICATION.json').read_text())['passed']:
            raise ValueError('TRAIN coverage qualification failed')
        sampler=MatchedGeometrySampler(train_set,prepared['sampler_seed'])
        train_loader=DataLoader(train_set,batch_sampler=sampler,generator=generator,collate_fn=collator)
    elif args.train_sampler=='balanced_acquisitions':
        sampler=BalancedAcquisitionBatchSampler(train_set,args.batch,args.seed+2)
        train_loader=DataLoader(train_set,batch_sampler=sampler,generator=generator,collate_fn=collate)
    else:
        train_loader=DataLoader(train_set,batch_size=args.batch,shuffle=True,generator=generator,collate_fn=collate)
    val_loader=DataLoader(val_set,batch_size=args.batch,shuffle=False,collate_fn=collator)
    test_loader=DataLoader(test_set,batch_size=args.batch,shuffle=False,collate_fn=collator)
    extra=list(model.backbone.embedding.stem.linear.extra.parameters())
    other=[p for n,p in model.backbone.named_parameters() if '.extra.' not in n]
    optimizer=torch.optim.AdamW([dict(params=other,lr=1e-5),dict(params=extra,lr=5e-4),
                                 dict(params=model.head.parameters(),lr=args.head_lr)],weight_decay=.01)
    protocol={**vars(args),'full_backbone_parameters':model.original_parameter_count,
              'total_parameters':params,'optimizer_parameters':sum(p.numel() for g in optimizer.param_groups for p in g['params']),
              'samples':dict(train=len(train_set),val=len(val_set),test=len(test_set)),
              'episode_ids':{s:[m['episode'] for m,_ in d.episodes] for s,d in [('train',train_set),('val',val_set),('test',test_set)]},
              'precision':'float32 with TF32 allowed','backbone_lr':1e-5,'readout_lr':args.head_lr,'sensor_affine_lr':5e-4,
              'source':dict(host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],step=os.environ['SLURM_STEP_ID'])}
    assert protocol['optimizer_parameters']==params
    parameter_names={id(p):n for n,p in model.named_parameters()}
    protocol['optimizer_parameter_bindings']=[
        dict(id=key,name=parameter_names[id(p)],numel=p.numel())
        for live,saved in zip(optimizer.param_groups,optimizer.state_dict()['param_groups'])
        for p,key in zip(live['params'],saved['params'])]
    # Official pretraining token is retained, but absent from the unmasked task path.
    token=model.backbone.embedding.mask_token.detach().cpu().contiguous()
    protocol['unmasked_unused_parameter']={'name':'backbone.embedding.mask_token',
        'numel':token.numel(),'sha256':hashlib.sha256(token.numpy().tobytes()).hexdigest()}
    initial_digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        initial_digest.update(name.encode());initial_digest.update(str((tuple(value.shape),value.dtype)).encode())
        initial_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    protocol['initial_model_sha256']=initial_digest.hexdigest()
    if args.normal_policy=='hand_surface':
        from .hand_surface_normals import geometry_signature
        protocol['normal_geometry_signature']=geometry_signature()
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    baseline_targets=torch.from_numpy(np.stack([train_set.target(i) if args.perception_corpus else train_set[i]['target'] for i in range(len(train_set))]))
    constant=baseline_targets.mean(0)
    initial_report,initial_arrays=evaluate(model,val_loader)
    np.savez_compressed(out/'initial_val.npz',**initial_arrays)
    (out/'initial_val.json').write_text(json.dumps(initial_report,indent=2))
    started=time.monotonic()
    step=0
    loss_file=(out/'train.jsonl').open('w',buffering=1)
    try:
        while step<args.steps:
            for batch in train_loader:
                model.train()
                if args.disable_stochastic_regularizers:
                    disable_stochastic_regularizers(model)
                batch_gpu=device_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                pred=model(batch_gpu)
                components=supervised_components(pred,batch_gpu['target'],batch_gpu['mass_label_weight'] if args.mass_airborne_guard else None)
                loss=sum(components.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step()
                step+=1
                row=dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),
                         **{k:float(v.detach()) for k,v in components.items()},
                         episodes=batch['episode'].tolist(),frames=batch['frame'].tolist(),
                         elapsed=time.monotonic()-started)
                if args.perception_corpus:row['mass_label_weights']=batch['mass_label_weight'].tolist()
                loss_file.write(json.dumps(row)+'\n')
                if step==1 or step%20==0:
                    print(json.dumps(row),flush=True)
                if step%args.evaluate_every==0 or step==args.steps:
                    report,arrays=evaluate(model,val_loader)
                    np.savez_compressed(out/f'val_{step:05d}.npz',**arrays)
                    (out/f'val_{step:05d}.json').write_text(json.dumps(report,indent=2))
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,protocol=protocol),out/'latest.pt')
                if step>=args.steps:
                    break
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,protocol=protocol),out/'model.pt')
        report,arrays=evaluate(model,test_loader)
        np.savez_compressed(out/'test.npz',**arrays)
        target=torch.from_numpy(arrays['target'])
        baseline_metrics=metrics(constant[None].expand(len(target),-1),target)
        report['train_mean_baseline']={k:float(v.mean()) for k,v in baseline_metrics.items()}
        # Read the saved full endpoint back and reproduce one real test batch.
        loaded=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(loaded['model'],strict=True)
        repeated,repeated_arrays=evaluate(model,test_loader)
        delta=float(np.max(np.abs(repeated_arrays['prediction']-arrays['prediction'])))
        report['full_endpoint_reload_max_abs']=delta
        if args.mode=='geometry_contact_force' and args.perception_corpus:
            test_set.force_gain=0.
            corrupted,corrupted_arrays=evaluate(model,test_loader)
            test_set.force_gain=1.
            np.savez_compressed(out/'test_force_zero.npz',**corrupted_arrays)
            report['test_force_zero']=corrupted
        elif args.mode=='geometry_contact_force':
            test_set.shuffle_force=True
            corrupted,corrupted_arrays=evaluate(model,test_loader)
            test_set.shuffle_force=False
            np.savez_compressed(out/'test_hand_force_swap.npz',**corrupted_arrays)
            report['test_hand_force_swap']=corrupted
        report['steps']=step
        report['complete']=True
        if delta>1e-5:
            raise RuntimeError(f'Checkpoint reload prediction mismatch {delta}')
        (out/'RESULT.json').write_text(json.dumps(report,indent=2))
        print('TRAINING_COMPLETE '+json.dumps(report),flush=True)
    except BaseException:
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,protocol=protocol),out/'interrupted.pt')
        raise
    finally:
        loss_file.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--mode',choices=MODES,required=True)
    ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--evaluate-every',type=int,default=100)
    ap.add_argument('--history',type=int,default=8)
    ap.add_argument('--history-policy',choices=('contiguous','episode_uniform_recent'),default='contiguous')
    ap.add_argument('--normal-policy',choices=('stored','hand_surface'),default='stored')
    ap.add_argument('--time-scale-s',type=float,default=1.)
    ap.add_argument('--stride',type=int,default=5)
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--train-sampler',choices=('shuffle','balanced_acquisitions','matched_geometry'),default='shuffle')
    ap.add_argument('--perception-corpus',action='store_true')
    ap.add_argument('--input-representation',choices=('centroid','surface'),default='centroid')
    ap.add_argument('--mass-airborne-guard',action='store_true')
    ap.add_argument('--seed',type=int,default=310016)
    ap.add_argument('--head-lr',type=float,default=1e-3)
    ap.add_argument('--deterministic-pooling',action='store_true')
    ap.add_argument('--disable-stochastic-regularizers',action='store_true')
    train(ap.parse_args())
