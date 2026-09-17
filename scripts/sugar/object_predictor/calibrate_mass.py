"""One log-mass offset per full model, fitted on VAL only; backbone untouched."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch

MODES=('geometry','geometry_contact','geometry_contact_force')


def fit(args):
    root=Path(args.training_root);models={}
    for mode in MODES:
        with np.load(root/mode/'val_00600.npz') as a:
            p,t,e=a['prediction'],a['target'],a['episode']
        protocol=json.loads((root/mode/'PROTOCOL.json').read_text())
        assert set(e.tolist())==set(protocol['episode_ids']['val'])
        assert not set(e.tolist()) & set(protocol['episode_ids']['test'])
        offset=float(np.mean([np.mean(t[e==episode,12].astype(np.float64)-p[e==episode,12].astype(np.float64)) for episode in np.unique(e)]))
        models[mode]=dict(mass_log_bias=offset,mass_scale_factor=float(np.exp(offset)),
                          val_episodes=np.unique(e).tolist(),val_windows=len(e),
                          source_predictions=str(root/mode/'val_00600.npz'),
                          endpoint=str(root/mode/'model.pt'))
    result=dict(method='One additive log-mass offset per arm; equal-episode mean VAL residual. No slope, seed or coefficient search.',
                fitted_split='val',models=models,new_neural_optimizer_updates=0,
                original_test_failure_retained=True,fresh_physical_test_required=True)
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


def adjusted(report,arrays,offset):
    result=copy.deepcopy(report);out={k:v.copy() for k,v in arrays.items()}
    out['prediction'][:,12]+=np.float32(offset)
    assert np.array_equal(out['prediction'][:,:12],arrays['prediction'][:,:12])
    mass=(torch.from_numpy(out['prediction'][:,12]-out['target'][:,12]).exp()-1).abs()
    episodes=torch.from_numpy(out['episode']);contact=torch.from_numpy(out['contact'])
    for group,mask in [('all',torch.ones_like(contact)),('contact',contact),('no_contact',~contact)]:
        if not mask.any(): continue
        values=mass[mask]
        result[group]['mass_relative']=dict(mean=float(values.mean()),median=float(values.median()),
            p90=float(torch.quantile(values,.9)),equal_episode_mean=float(torch.stack([
                mass[mask & (episodes==e)].mean() for e in episodes[mask].unique()]).mean()))
    return result,out


def apply(args):
    calibration=json.loads(Path(args.calibration).read_text())
    source=Path(args.evaluation_root);root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    for mode in MODES:
        details=calibration['models'][mode];offset=details['mass_log_bias']
        report=json.loads((source/mode/'RESULT.json').read_text())
        assert Path(report['endpoint']).resolve()==Path(details['endpoint']).resolve()
        with np.load(source/mode/'predictions.npz') as a: arrays={k:a[k] for k in a.files}
        assert not set(arrays['episode'].tolist()) & set(details['val_episodes'])
        result,arrays=adjusted(report,arrays,offset)
        out=root/mode;out.mkdir();np.savez_compressed(out/'predictions.npz',**arrays)
        if mode=='geometry_contact_force':
            with np.load(source/mode/'force_zero_predictions.npz') as a: zero={k:a[k] for k in a.files}
            zero_report,zero=adjusted(report['force_zero_keep_contact_geometry_and_area'],zero,offset)
            result['force_zero_keep_contact_geometry_and_area']=zero_report
            np.savez_compressed(out/'force_zero_predictions.npz',**zero)
        if 'train_mean_baseline' in result:
            result['source_train_mean_baseline']=result.pop('train_mean_baseline')
        result.update(mass_calibration=details,calibration_file=args.calibration,
                      source_raw_evaluation=str(source/mode),neural_weights_unchanged=True,
                      non_mass_outputs_bit_exact=True,new_optimizer_updates=0)
        (out/'RESULT.json').write_text(json.dumps(result,indent=2))
    print('CALIBRATION_APPLIED '+str(root),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fit');f.add_argument('--training-root',required=True);f.add_argument('--output',required=True)
    a=sub.add_parser('apply')
    for name in ('calibration','evaluation-root','output'): a.add_argument('--'+name,required=True)
    args=ap.parse_args();(fit if args.command=='fit' else apply)(args)
