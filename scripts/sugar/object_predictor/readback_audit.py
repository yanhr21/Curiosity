"""Separate exact checkpoint integrity from GPU reduction repeatability."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import ContactDataset,collate
from .model import ObjectPredictor


def exact(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b,strict=True))
    return a==b


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Compute step required')
    root=Path(args.run)
    source=torch.load(root/'model.pt',map_location='cpu',weights_only=False)
    companion=root/('interrupted.pt' if (root/'interrupted.pt').exists() else 'latest.pt')
    readback=torch.load(companion,map_location='cpu',weights_only=False)
    checks={'full_saved_model_exact':exact(source['model'],readback['model']),
            'full_saved_adam_exact':exact(source['optimizer'],readback['optimizer']),
            'both_saved_clocks_600':source['step']==readback['step']==600}
    del readback
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    protocol=source['protocol']
    model=ObjectPredictor(args.checkpoint,protocol['history']).cuda()
    model.load_state_dict(source['model'],strict=True);model.eval()
    dataset=ContactDataset(protocol['data'],protocol['mode'],'test',protocol['history'],protocol['stride'],history_policy=protocol.get('history_policy','contiguous'),time_scale_s=protocol.get('time_scale_s',1.),normal_policy=protocol.get('normal_policy','stored'))
    loader=DataLoader(dataset,batch_size=protocol['batch'],shuffle=False,collate_fn=collate)
    old_repeat=[];csr_repeat=[];cross=[]
    with torch.inference_mode():
        for index,batch in enumerate(loader):
            if index==8: break
            batch={k:v.cuda() for k,v in batch.items()}
            model.deterministic_pooling=False
            old=[model(batch).cpu() for _ in range(3)]
            model.deterministic_pooling=True
            csr=[model(batch).cpu() for _ in range(3)]
            old_repeat.append(max(float((p-old[0]).abs().max()) for p in old[1:]))
            csr_repeat.append(max(float((p-csr[0]).abs().max()) for p in csr[1:]))
            cross.append(float((old[0]-csr[0]).abs().max()))
    checks['all_model_tensors_unchanged']=all(torch.equal(v.cpu(),source['model'][k]) for k,v in model.state_dict().items())
    checks['csr_repeat_within_original_1e5']=max(csr_repeat)<=1e-5
    report=dict(checks=checks,passed=all(checks.values()),run=str(root),companion=str(companion),
                new_optimizer_updates=0,original_scatter_repeat_max=max(old_repeat),
                csr_repeat_max=max(csr_repeat),scatter_vs_csr_max=max(cross),
                original_failed_prediction_check_retained=True,
                interpretation='CSR changes only nonparametric task-readout reduction; complete official encoder and stored tensors are unchanged. Original scatter replay failure is not reclassified as a pass.')
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True)
    main(ap.parse_args())
