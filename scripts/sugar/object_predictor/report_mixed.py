"""Physical-unit readback stratified by predeclared acquisition and clock windows."""
import argparse
import json
from pathlib import Path
import numpy as np


def rotation(value):
    x=value[:,:3];x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    y=value[:,3:]-np.sum(x*value[:,3:],axis=-1,keepdims=True)*x
    y=y/np.maximum(np.linalg.norm(y,axis=-1,keepdims=True),1e-12)
    return np.stack((x,y,np.cross(x,y)),axis=-1)


def errors(pred,target):
    relative=np.swapaxes(rotation(pred[:,3:9]),-1,-2)@rotation(target[:,3:9])
    return dict(position_cm=np.linalg.norm(pred[:,:3]-target[:,:3],axis=1)*100,
                rotation_deg=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))),
                size_relative=np.abs(np.exp(pred[:,9:12]-target[:,9:12])-1).mean(1),
                mass_relative=np.abs(np.exp(pred[:,12]-target[:,12])-1))


def summarize(arrays,metas,clocks):
    episode=arrays['episode'];frame=arrays['frame']
    group=np.array([metas[int(e)]['acquisition_group'] for e in episode])
    times=np.array([clocks[int(e)][int(f)] for e,f in zip(episode,frame)])
    values=errors(arrays['prediction'].astype(np.float64),arrays['target'].astype(np.float64))
    masks={'all':np.ones(len(episode),bool),'probe_all':group=='probe',
           'probe_primary_140_148s':(group=='probe')&(times>=140.)&(times<=148.),'support_all':group=='support'}
    report={}
    for name,mask in masks.items():
        per_episode={str(e):{k:float(v[mask&(episode==e)].mean()) for k,v in values.items()} for e in np.unique(episode[mask])}
        report[name]=dict(samples=int(mask.sum()),episodes=per_episode,
                          equal_episode_mean={k:float(np.mean([r[k] for r in per_episode.values()])) for k in values} if per_episode else {})
    return report


def main(args):
    root=Path(args.data);evaluation=Path(args.evaluation_root)
    metas={};clocks={}
    for path in sorted(root.glob('episode_*.json')):
        meta=json.loads(path.read_text())
        if meta['split']!='test': continue
        metas[meta['episode']]=meta
        with np.load(path.with_suffix('.npz')) as src: clocks[meta['episode']]=src['timestamp_s']
    reports={};checks={};anchor=None
    for mode in ('geometry','geometry_contact','geometry_contact_force'):
        with np.load(evaluation/mode/'predictions.npz') as src: arrays={k:src[k] for k in src.files}
        checks[mode+'_all_test_configurations']=set(arrays['episode'].tolist())==set(metas)
        checks[mode+'_finite_predictions']=bool(np.isfinite(arrays['prediction']).all())
        for e,meta in metas.items():
            checks[f'{mode}_{e}_all_fixed_sample_clocks']=np.array_equal(arrays['frame'][arrays['episode']==e],np.arange(31,len(clocks[e]),meta['sampling_stride']))
        if anchor is None: anchor=arrays
        else:
            for k in ('target','episode','frame','contact'): checks[mode+'_matched_'+k]=np.array_equal(arrays[k],anchor[k])
        reports[mode]=summarize(arrays,metas,clocks)
        checks[mode+'_all_six_primary_probe_episodes']=set(map(int,reports[mode]['probe_primary_140_148s']['episodes']))=={e for e,m in metas.items() if m['acquisition_group']=='probe'}
    with np.load(evaluation/'geometry_contact_force'/'force_zero_predictions.npz') as src: zero={k:src[k] for k in src.files}
    for k in ('target','episode','frame','contact'): checks['force_zero_matched_'+k]=np.array_equal(zero[k],anchor[k])
    reports['force_zero_keep_contact_geometry_and_area']=summarize(zero,metas,clocks)
    force=reports['geometry_contact_force'];geometry=force['probe_primary_140_148s']['equal_episode_mean'];mass=force['support_all']['equal_episode_mean']
    criteria={'center_5cm':geometry.get('position_cm',float('inf'))<=5.,
              'rotation_15deg':geometry.get('rotation_deg',float('inf'))<=15.,
              'size_10pct':geometry.get('size_relative',float('inf'))<=.1,
              'support_mass_10pct':mass.get('mass_relative',float('inf'))<=.1}
    report=dict(checks=checks,readback_passed=all(checks.values()),reports=reports,criteria=criteria,
                all_arms_share_force_feedback_acquisition=True,
                ablation_scope='Explicit predictor sensor channels on the same measured hand trajectories; geometry-only is not touch-free acquisition.',
                full_state_criteria_passed=all(criteria.values()),
                failed_physical_test_episodes=[e for e,m in metas.items() if m['acquisition_group']=='probe' and not m['contact_qualification_passed']],
                scope='Known asset family only. Probe mass is diagnostic. No policy or unseen-shape result.',new_optimizer_updates=0)
    (evaluation/'MIXED_METRICS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['readback_passed']: raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--evaluation-root',required=True);main(ap.parse_args())
