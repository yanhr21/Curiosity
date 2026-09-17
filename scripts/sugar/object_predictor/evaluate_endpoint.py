"""Deployment-batch evaluation of a saved full predictor, including another backend."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ContactDataset,collate
from .model import ObjectPredictor
from .train import evaluate


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('GPU inference requires compute step')
    out=Path(args.output);out.mkdir(exist_ok=False,parents=True)
    payload=torch.load(args.endpoint,map_location='cpu',weights_only=False)
    protocol=payload['protocol']
    if protocol.get('normal_policy','stored')=='hand_surface':
        from .hand_surface_normals import verify_geometry_signature
        verify_geometry_signature(protocol.get('normal_geometry_signature'))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    model=ObjectPredictor(args.checkpoint,protocol['history'],protocol.get('deterministic_pooling',False)).cuda()
    model.load_state_dict(payload['model'],strict=True);model.eval()
    stride=args.stride if args.stride is not None else protocol['stride']
    collator=collate
    if protocol.get('perception_corpus',False):
        from .perception_dataset import PerceptionDataset,collate_perception
        dataset=PerceptionDataset(args.data,protocol['mode'],'test',protocol['history'],stride,
            representation=protocol['input_representation'],history_policy=protocol['history_policy'],
            time_scale_s=protocol['time_scale_s'],normal_policy=protocol['normal_policy'])
        collator=collate_perception
    else:
        dataset=ContactDataset(args.data,protocol['mode'],'test',protocol['history'],stride,history_policy=protocol.get('history_policy','contiguous'),time_scale_s=protocol.get('time_scale_s',1.),normal_policy=protocol.get('normal_policy','stored'))
    # Official non-Flash serialized attention uses the batch minimum cloud
    # size. Evaluate batch1 for the actual streaming deployment contract.
    loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collator)
    report,arrays=evaluate(model,loader)
    np.savez_compressed(out/'predictions.npz',**arrays)
    batch={k:v.cuda() for k,v in next(iter(loader)).items()}
    times=[]
    with torch.inference_mode():
        for _ in range(10):
            torch.cuda.synchronize();started=time.perf_counter()
            model(batch);torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
    report.update(endpoint=args.endpoint,data=args.data,batch=1,mode=protocol['mode'],
                  evaluation_stride=stride,force_zero_requested=not args.skip_force_zero,
                  original_training_steps=payload['step'],new_optimizer_updates=0,
                  model_only_latency_ms_median=float(np.median(times)*1000),
                  warning='Backend/asset/gesture/force calibration shifts require separate interpretation.')
    if Path(args.data).resolve()==Path(protocol['data']).resolve() and stride==protocol['stride']:
        trained=json.loads((Path(args.endpoint).parent/'RESULT.json').read_text())
        report['train_mean_baseline']=trained['train_mean_baseline']
    if protocol['mode']=='geometry_contact_force' and not args.skip_force_zero:
        dataset.force_gain=0.
        zero_report,zero_arrays=evaluate(model,loader)
        dataset.force_gain=1.
        np.savez_compressed(out/'force_zero_predictions.npz',**zero_arrays)
        report['force_zero_keep_contact_geometry_and_area']=zero_report
    (out/'RESULT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--endpoint',required=True)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--stride',type=int)
    ap.add_argument('--skip-force-zero',action='store_true')
    main(ap.parse_args())
