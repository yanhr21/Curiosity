"""TRAIN error scales for the project prior/contact fusion, not a new model.

Disjoint from the eight registration qualification clocks. These in-sample
errors are empirical weighting scales, not calibrated posterior uncertainty.
"""
import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import pytorch_volumetric as pv
import torch
from scipy.spatial.transform import Rotation

from .qualify_chsel_backend import read, digest, homogeneous
from .shape_metric import rotation


def main(args):
    assert os.environ.get('SLURM_STEP_ID')
    torch.set_num_threads(4)
    src, out = Path(args.inputs), Path(args.output)
    out.mkdir(exist_ok=False)
    manifest = json.loads((src/'RESULT.json').read_text())
    cases = manifest['cases']
    assert [(c['episode'],c['frame']) for c in cases] == [(e,f) for e in range(5004,5032) for f in (1206,2206)]
    labels = read(src/'evaluation_only.npz')
    mesh_path = Path('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz')
    mesh = read(mesh_path); v = mesh['vertices'].astype(np.float64); dims = np.ptp(v,axis=0)
    v -= (v.max(0)+v.min(0))/2
    errors = []; contacts = [[], []]; rows = []
    for i,c in enumerate(cases):
        assert labels['episode'][i] == c['episode'] and labels['frame'][i] == c['frame']
        path = src/c['input_file']; assert digest(path) == c['input_sha256']
        a = read(path); p = a['prediction']; target = labels['target'][i]
        rotation_error = Rotation.from_matrix(rotation(target[3:9]) @ rotation(p[3:9]).T).as_rotvec()
        errors.append(np.r_[target[:3]-p[:3], rotation_error])
        scaled = v*(np.exp(p[9:12])/dims)
        obj = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(scaled),o3d.utility.Vector3iVector(mesh['faces']))
        factory = pv.MeshObjectFactory(mesh=obj); factory.precompute_sdf()
        h = np.linalg.inv(homogeneous(target))
        q = a['surface_points_current_left_hand_m'] @ h[:3,:3].T+h[:3,3]
        query = factory.object_frame_closest_point(torch.as_tensor(q,dtype=torch.float64))
        distance2 = np.square(q-query.closest.numpy()).sum(-1)
        assert np.allclose(distance2, np.square(query.distance.numpy()), atol=1e-9)
        mse = []
        for hand in (0,1):
            mask = a['surface_hand']==hand
            if mask.any():
                w = a['surface_area_m2'][mask].astype(np.float64)
                assert (w>0).all()
                value = float(np.average(distance2[mask],weights=w)); contacts[hand].append(value)
                mse.append(value)
            else:
                mse.append(None)
        rows.append(dict(episode=c['episode'],frame=c['frame'],points=len(q),
                         contact_mse_m2_per_hand=mse,pose_error_m_rad=errors[-1].tolist()))
        print('CALIBRATION_CASE', c['episode'], c['frame'], mse, flush=True)
    pose_rms = np.sqrt(np.square(errors).mean(0))
    contact_rms = np.sqrt([np.mean(x) for x in contacts])
    # Explicit numerical floors, predeclared before inspecting these errors.
    pose_scales = np.maximum(pose_rms, np.array([.001]*3+[.001]*3))
    contact_scales = np.maximum(contact_rms,.001)
    result = dict(complete=True, cases=rows, pose_rms_m_rad=pose_rms.tolist(),
        pose_scales_m_rad=pose_scales.tolist(), contact_rms_m=contact_rms.tolist(),
        contact_scales_m=contact_scales.tolist(), contact_frame_counts=[len(x) for x in contacts],
        pose_definition='[center_gt-center_pred, Log(R_gt R_pred.T)] in current left-hand frame',
        contact_definition='Per-hand area-weighted squared closest-triangle distance at calibration truth pose but predicted size; frames equal weight',
        floors='1mm center/contact, 0.001rad rotation; numerical floors, not a measurement-noise claim',
        scope='TRAIN in-sample empirical weighting scales; not calibrated confidence or independent likelihood',
        qualification_excluded_episodes=[5000,5001,5002,5003],
        input_manifest_sha256=digest(src/'RESULT.json'), mesh_sha256=digest(mesh_path),
        official_query_source=inspect.getfile(pv.MeshObjectFactory),
        source_sha256=digest(Path(inspect.getfile(pv.MeshObjectFactory))),
        model_forwards=0, optimizer_updates=0, physics_steps=0)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2))
    print('CALIBRATION_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--output',required=True)
    main(ap.parse_args())
