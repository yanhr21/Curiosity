"""Observation-only quasi-static support stratification; primary errors retained.

This fixture-specific diagnostic cannot certify squeezing, ground contact, partial
support or an uncalibrated hardware/native sensor. It is not a quality gate.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main(args):
    root=Path(args.evaluation_root)
    with np.load(root/'geometry_contact_force'/'predictions.npz') as a:
        episodes=a['episode'];frames=a['frame'];target=a['target']
    sources={}
    for e in np.unique(episodes):
        with np.load(Path(args.data)/f'episode_{e:04d}.npz') as a:
            sources[int(e)]={k:a[k] for k in ('normal_load_n','hand_pose_w','timestamp_s')}
    masks=[];estimated_mass=[]
    for e,t in zip(episodes,frames,strict=True):
        a=sources[int(e)];sl=slice(int(t)-7,int(t)+1)
        forces=a['normal_load_n'][sl];total=forces.sum(1)
        dt=np.diff(a['timestamp_s'][sl])
        velocity=np.diff(a['hand_pose_w'][sl,:,:3],axis=0)/dt[:,None,None]
        acceleration=np.diff(velocity,axis=0)/((dt[:-1]+dt[1:])/2)[:,None,None]
        both_hands=bool((forces[:,:27].sum(1)>1e-3).all() and (forces[:,27:].sum(1)>1e-3).all())
        masks.append(both_hands and total.std()/max(total.mean(),1e-8)<.1 and np.linalg.norm(acceleration,axis=-1).max()<.981)
        estimated_mass.append(total.mean()/9.81)
    mask=np.asarray(masks);mass=np.exp(target[:,12]);estimate=np.asarray(estimated_mass)
    report=dict(scope=__doc__,mask_uses_only_observations=True,
                definition='All trailing eight frames contact both hands, total-normal-load CV < 0.1, maximum hand translational acceleration < 0.1 g',
                all_windows=len(mask),qualified_windows=int(mask.sum()),
                qualified_episode_counts={str(e):int((mask & (episodes==e)).sum()) for e in np.unique(episodes)},metrics={})
    if mask.any():
        report['direct_normal_sum_mass_relative_mean']=float((np.abs(estimate-mass)/mass)[mask].mean())
        for mode in ('geometry','geometry_contact','geometry_contact_force'):
            with np.load(root/mode/'predictions.npz') as a:
                assert np.array_equal(a['target'],target) and np.array_equal(a['episode'],episodes) and np.array_equal(a['frame'],frames)
                p=a['prediction']
            report['metrics'][mode]=dict(position_cm=float((np.linalg.norm(p[:,:3]-target[:,:3],axis=1)*100)[mask].mean()),
                                        size_relative=float(np.abs(np.exp(p[:,9:12]-target[:,9:12])-1)[mask].mean()),
                                        mass_relative=float(np.abs(np.exp(p[:,12]-target[:,12])-1)[mask].mean()))
    out=Path(args.output)
    np.savez_compressed(out.with_suffix('.npz'),observation_qualified=mask,episode=episodes,frame=frames,
                        direct_normal_sum_mass_kg=estimate,true_mass_kg=mass)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('data','evaluation-root','output'): ap.add_argument('--'+name,required=True)
    main(ap.parse_args())
