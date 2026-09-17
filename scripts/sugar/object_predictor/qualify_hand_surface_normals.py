"""Qualify a full-hand geometry query on saved TRAIN observations; no training."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import OBSERVATION_KEYS
from .hand_surface_normals import query_observation_normals


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);records=[];checks={}
    for source_path in args.episodes:
        path=Path(source_path)
        with np.load(path) as source:obs={k:source[k] for k in OBSERVATION_KEYS}
        # No privileged array is passed into the candidate surface query.
        site,contact,details=query_observation_normals(obs)
        checks[str(path)+'_finite_unit_vectors']=bool(np.isfinite(site).all() and np.isfinite(contact).all() and np.allclose(np.linalg.norm(site,axis=2),1.,atol=1e-6) and np.allclose(np.linalg.norm(contact,axis=2),1.,atol=1e-6))
        inactive=obs['normal_load_n']<=1e-3
        checks[str(path)+'_inactive_identical']=np.array_equal(site[inactive],contact[inactive])
        # Common rigid translation changes no local query; no label dependence.
        moved={k:v[:32].copy() for k,v in obs.items()};delta=np.array([.4,-.7,.2])
        for k in ('hand_sites_w','contact_position_w'):moved[k]=moved[k].astype(np.float64)+delta
        moved['hand_pose_w']=moved['hand_pose_w'].astype(np.float64);moved['hand_pose_w'][...,:3]+=delta
        moved_site,moved_contact,_=query_observation_normals(moved)
        difference=float(max(np.abs(moved_site-site[:32]).max(),np.abs(moved_contact-contact[:32]).max()))
        checks[str(path)+'_translation_invariant']=difference<=1e-5
        name=f'case_{len(records):02d}'
        np.savez_compressed(out/(name+'.npz'),hand_site_surface_normals_w=site,hand_contact_surface_normals_w=contact)
        records.append(dict(source=str(path),output=name+'.npz',details=details,translation_max_difference=difference))
    report=dict(checks=checks,passed=all(checks.values()),episodes=records,training_admitted=False,
                new_optimizer_updates=0,gpu_calls=0,query_inputs=list(OBSERVATION_KEYS),
                scope='Full original known hand CAD only; nearest-surface normals at ideal sensor locations. Not an object reconstruction or measured force vector. Frozen collector and existing network inputs unchanged.')
    (out/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--episodes',nargs='+',required=True);ap.add_argument('--output',required=True);main(ap.parse_args())
