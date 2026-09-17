"""Frozen full-model TRAIN/eval diagnostic on fixed real training batches."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import ContactDataset,collate
from .model import ObjectPredictor,loss_components


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Retained compute step required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    payload=torch.load(args.endpoint,map_location='cpu',weights_only=False)
    protocol=payload['protocol']
    model=ObjectPredictor(args.checkpoint,protocol['history'],protocol.get('deterministic_pooling',False)).cuda()
    model.load_state_dict(payload['model'],strict=True)
    dataset=ContactDataset(protocol['data'],protocol['mode'],'train',protocol['history'],protocol['stride'],history_policy=protocol.get('history_policy','contiguous'),time_scale_s=protocol.get('time_scale_s',1.),normal_policy=protocol.get('normal_policy','stored'))
    generator=torch.Generator().manual_seed(310117)
    loader=DataLoader(dataset,batch_size=4,shuffle=True,generator=generator,collate_fn=collate)
    batches=[]
    for index,batch in enumerate(loader):
        if index==8: break
        batches.append({k:v.cuda() for k,v in batch.items()})
    results={};outputs={}
    regularizers={n:dict(type=type(m).__name__,p=getattr(m,'p',getattr(m,'drop_prob',None)))
                  for n,m in model.named_modules() if isinstance(m,torch.nn.Dropout) or type(m).__name__=='DropPath'}
    with torch.inference_mode():
        for mode in ('eval','train','train_without_stochastic_regularizers'):
            model.train(mode!='eval')
            if mode=='train_without_stochastic_regularizers':
                for n,m in model.named_modules():
                    if n in regularizers: m.eval()
            values=[];predictions=[]
            for repeat in range(8 if mode=='train' else 1):
                torch.manual_seed(310118+repeat);torch.cuda.manual_seed_all(310118+repeat)
                for batch in batches:
                    prediction=model(batch)
                    values.append({k:float(v) for k,v in loss_components(prediction,batch['target']).items()})
                    predictions.append(prediction.cpu().numpy())
            results[mode]={k:float(np.mean([v[k] for v in values])) for k in values[0]}
            outputs[mode]=np.stack(predictions)
    exact=all(torch.equal(v.cpu(),payload['model'][n]) for n,v in model.state_dict().items())
    difference=float(np.max(np.abs(outputs['eval']-outputs['train_without_stochastic_regularizers'])))
    result=dict(endpoint=args.endpoint,new_optimizer_updates=0,fixed_real_training_batches=8,
                stochastic_train_repeats=8,regularizers=regularizers,mean_losses=results,
                train_label_means='Raw model.train() including original stochastic regularizers; diagnostic only when protocol disables them.',
                actual_training_mode=('train_without_stochastic_regularizers' if protocol.get('disable_stochastic_regularizers',False) else 'train'),
                full_state_unchanged=exact,eval_vs_train_without_regularizers_max_abs=difference)
    Path(args.output).write_text(json.dumps(result,indent=2))
    np.savez_compressed(Path(args.output).with_suffix('.npz'),**outputs)
    print(json.dumps({k:v for k,v in result.items() if k!='regularizers'}),flush=True)
    if not exact or difference>1e-5: raise RuntimeError('Mode diagnostic integrity failed')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('endpoint','checkpoint','output'): ap.add_argument('--'+name,required=True)
    main(ap.parse_args())
