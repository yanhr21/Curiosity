"""Saved32 integration qualification for an explicitly new sensor condition."""
import argparse,json,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .palmar_coverage import continuous_palmar_field


def main():
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Use retained step')
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.output)
    prior=Path('experiments/object_predictor_v1/force_deficit_decomposition_v1')
    p=json.loads((prior/'PROTOCOL.json').read_text());frames=np.asarray(p['frames']);checks=[];rows=[]
    for ep in p['episodes']:
        case=Path(p['source'])/'cases'/f'episode_{ep}'
        with np.load(case/'contact_surface.npz',allow_pickle=False) as f:
            d={k:f[k] for k in ('offset','position_hand_frame_m','area_m2','normal_pressure_pa','shear_traction_hand_frame_pa','pad','hand')}
        with np.load(case/f'episode_{ep}.npz',allow_pickle=False) as f:
            poses=f['hand_pose_w'][frames];oldloads=f['normal_load_n'][frames];oldshear=f['shear_force_w'][frames]
        with np.load(prior/f'episode_{ep}.npz',allow_pickle=False) as f:
            fullfield=f['full_field_shear_support_w'];assert np.array_equal(f['frame'],frames)
        physical_exact=True;assigned_exact=True;raw_exact=True;halfspace=True;load_error=0.;shear_error=0.;old_load_error=0.;old_shear_error=0.;added=0
        for i,frame in enumerate(frames):
            start,end=d['offset'][frame:frame+2]
            f=dict(pos=d['position_hand_frame_m'][start:end],area=d['area_m2'][start:end],
                   pressure=d['normal_pressure_pa'][start:end],traction_vec=d['shear_traction_hand_frame_pa'][start:end],
                   pad=d['pad'][start:end],patch=d['hand'][start:end])
            mapped=continuous_palmar_field(f);raw=f['pad'];pad=mapped['pad'];hand=f['patch']
            original=raw>=0;observed=pad>=0
            physical_exact &= all(np.array_equal(mapped[k],f[k]) for k in ('pos','area','pressure','traction_vec','patch'))
            assigned_exact &= np.array_equal(pad[original],raw[original]);raw_exact &= np.array_equal(mapped['anatomical_pad'],raw)
            eligible=f['pos'][:,1]*np.array([-1.,1.])[hand]>0
            halfspace &= np.array_equal(observed,original|eligible)
            added+=int((observed & ~original).sum())
            weights=f['area'].astype(float)*f['pressure'].astype(float)
            force=f['area'].astype(float)[:,None]*f['traction_vec'].astype(float)
            rot=Rotation.from_quat(poses[i,:,3:]);sum_world=np.zeros(3)
            for h in (0,1):
                ix=observed&(hand==h)
                bins=np.bincount(pad[ix]%27,weights=weights[ix],minlength=27)
                load_error=max(load_error,abs(bins.sum()-weights[eligible&(hand==h)].sum()))
                sum_world-=rot[h].apply(force[ix].sum(0))
                baseline=original&(hand==h)
                old_bins=np.bincount(raw[baseline]%27,weights=weights[baseline],minlength=27)
                old_load_error=max(old_load_error,float(abs(old_bins-oldloads[i,h*27:(h+1)*27]).max()))
                old_shear_error=max(old_shear_error,float(abs(rot[h].apply(force[baseline].sum(0))-oldshear[i,h*27:(h+1)*27].sum(0)).max()))
            # This corpus happens to have all contact in selected halfspaces;
            # the equality is a saved-corpus result, not general full-hand coverage.
            shear_error=max(shear_error,float(abs(sum_world-fullfield[i]).max()))
        c=dict(physical_channels_unchanged=bool(physical_exact),original_assigned_ids_unchanged=bool(assigned_exact),
               original_labels_retained=bool(raw_exact),exact_declared_halfspace=bool(halfspace),
               integrated_new_load_conserved=load_error<1e-10,prior_fullfield_shear_matches=shear_error<1e-10,
               original_load_readback=old_load_error<3e-5,original_shear_readback=old_shear_error<3e-6)
        c = {k: bool(v) for k, v in c.items()}
        checks.append(dict(episode=ep,checks=c));rows.append(dict(episode=ep,added_face_frames=added,new_load_closure_max_n=load_error,
            new_shear_reference_max_n=shear_error,old_load_readback_max_n=old_load_error,old_shear_readback_max_n=old_shear_error))
        print('PALMAR_COVERAGE_CASE',ep,c,flush=True)
    result=dict(passed=all(all(x['checks'].values()) for x in checks),episodes=len(checks),frames=len(frames)*len(checks),
                added_face_frames=sum(x['added_face_frames'] for x in rows),checks=checks,rows=rows,
                new_physics=0,model_forwards=0,optimizer_updates=0,sensor='continuous_palmar_v1',
                scope='Qualification of new ideal sensor routing/integration only, no hardware or learned accuracy claim.')
    (root/'INPUT_QUALIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PALMAR_COVERAGE_COMPLETE',result['passed'],result['added_face_frames'],flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
