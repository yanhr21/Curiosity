"""Check actual contact-containing causal windows against complete CAD-query outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .data import OBSERVATION_KEYS,history_indices
from .hand_surface_normals import query_observation_normals


def main(args):
    root=Path(args.qualification);source_report=json.loads((root/'RESULT.json').read_text())
    checks={};details=[];transform=Rotation.from_euler('xyz',[.31,-.42,.67]);matrix=transform.as_matrix();translation=np.array([.4,-.7,.2])
    for record in source_report['episodes']:
        with np.load(record['source']) as source: full={k:source[k] for k in OBSERVATION_KEYS}
        selected=history_indices(len(full['timestamp_s'])-1,32,'episode_uniform_recent')
        obs={k:v[selected] for k,v in full.items()}
        with np.load(root/record['output']) as cached:
            expected=[cached[k][selected] for k in ('hand_site_surface_normals_w','hand_contact_surface_normals_w')]
        actual=query_observation_normals(obs)[:2]
        name=record['source']
        checks[name+'_contact_frames_present']=bool(np.any(obs['normal_load_n']>1e-3))
        cache_difference=max(float(np.abs(a-b).max()) for a,b in zip(actual,expected))
        checks[name+'_full_episode_vs_selected_history']=cache_difference<=1e-5
        moved={k:v.astype(np.float64).copy() for k,v in obs.items()}
        for k in ('hand_sites_w','contact_position_w'):
            moved[k]=moved[k]@matrix.T+translation
        for k in ('hand_normals_w','shear_force_w'):moved[k]=moved[k]@matrix.T
        moved['hand_pose_w'][...,:3]=moved['hand_pose_w'][...,:3]@matrix.T+translation
        shape=moved['hand_pose_w'][...,3:].shape
        moved['hand_pose_w'][...,3:]=(transform*Rotation.from_quat(moved['hand_pose_w'][...,3:].reshape(-1,4))).as_quat().reshape(shape)
        changed=query_observation_normals(moved)[:2]
        rigid_difference=max(float(np.abs(a@matrix.T-b).max()) for a,b in zip(actual,changed))
        checks[name+'_common_rigid_transform_equivariant']=rigid_difference<=1e-5
        details.append(dict(source=name,selected_frames=selected.tolist(),contact_sites=int((obs['normal_load_n']>1e-3).sum()),
                            full_vs_history_max=cache_difference,rigid_transform_max=rigid_difference))
    report=dict(checks=checks,passed=all(checks.values()),details=details,tolerance=1e-5,
                new_optimizer_updates=0,gpu_calls=0,training_admitted=False,
                scope='Candidate CAD-normal adapter only. Includes actual contact frames; compares complete-episode cached outputs with causal histories and a common rigid world transform. No trained-model or force-direction claim.')
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--qualification',required=True);ap.add_argument('--output',required=True);main(ap.parse_args())
