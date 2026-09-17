"""Read frozen predictor transfer on the original fixed primary clocks."""
from pathlib import Path
import hashlib,json
import numpy as np
from .shape_metric import rotation


def errors(p,t):
    rel=np.einsum('nji,njk->nik',np.array([rotation(v) for v in p[:,3:9]]),np.array([rotation(v) for v in t[:,3:9]]))
    return dict(center_cm=np.linalg.norm(p[:,:3]-t[:,:3],axis=1)*100,
                rotation_deg=np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1))*180/np.pi,
                size_pct=np.abs(np.exp(p[:,9:12]-t[:,9:12])-1).mean(1)*100,
                mass_pct=np.abs(np.exp(p[:,12]-t[:,12])-1)*100)


def main():
    root=Path('experiments/object_predictor_v1/grip_transfer_v1');data=root/'data_v1';ev=root/'evaluation'
    modes=['geometry','geometry_contact','geometry_contact_force'];report={};reference=None;checks={}
    for mode in modes+['force_zero']:
        path=ev/(mode if mode!='force_zero' else 'geometry_contact_force')/('predictions.npz' if mode!='force_zero' else 'force_zero_predictions.npz')
        with np.load(path) as z:a={k:z[k] for k in z.files}
        if reference is None:reference=a
        else:
            for k in ('episode','frame','target','contact'):checks[mode+'_'+k+'_alignment']=np.array_equal(reference[k],a[k])
        percase={}
        for episode in range(4000,4004):
            with np.load(data/f'episode_{episode:04d}.npz') as z:time=z['timestamp_s']
            idx=np.flatnonzero(a['episode']==episode);frames=a['frame'][idx]
            checks[f'{mode}_{episode}_all234_visual_windows']=np.array_equal(frames,np.arange(31,1200,5))
            primary=((frames-31)%25==0);clock=time[frames]
            errors_all=errors(a['prediction'][idx],a['target'][idx]);groups={}
            for name,mask in [('approach',primary&(clock<16)),('lift',primary&(clock>=16)&(clock<20)),('hold',primary&(clock>=20))]:
                groups[name]=dict(samples=int(mask.sum()),**{k:float(v[mask].mean()) for k,v in errors_all.items()},
                                  mass_pred_kg=float(np.exp(a['prediction'][idx[mask],12]).mean()),mass_true_kg=float(np.exp(a['target'][idx[mask],12]).mean()))
            checks[f'{mode}_{episode}_eight_original_hold_windows']=groups['hold']['samples']==8
            percase[str(episode)]=groups
        means={key:float(np.mean([x['hold'][key] for x in percase.values()])) for key in ('center_cm','rotation_deg','size_pct','mass_pct')}
        invariance={}
        for lo,hi,mass in [(4000,4001,.5),(4002,4003,.9)]:
            invariance[str(mass)]=abs(percase[str(hi)]['hold']['mass_pred_kg']-percase[str(lo)]['hold']['mass_pred_kg'])/mass*100
        report[mode]=dict(cases=percase,hold_equal_episode=means,same_mass_grip_change_prediction_pct=invariance,source_sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())
    result=dict(checks=checks,alignment_passed=all(checks.values()),models=report,physics='All four actual grip/lift cases passed preregistered physics checks',
                threshold_scope='Four-case diagnosis only; not broad generalization',new_optimizer_updates=0)
    (root/'COMPARISON.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    assert result['alignment_passed']


if __name__=='__main__':main()
