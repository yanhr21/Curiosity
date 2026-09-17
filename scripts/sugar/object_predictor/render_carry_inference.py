"""Frozen causal inference on the existing native carrying trace; no training."""
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .predict import ObjectStateEstimator
from .data import OBSERVATION_KEYS

ROOT = Path('experiments/object_predictor_v1')
OUT = ROOT / 'rendered_carry_v1'


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'isaac_saved_r1/episode_1000.npz'
    endpoint = ROOT / 'training_mixed_geometry_mass_surface_v1/geometry_contact_force/model.pt'
    with np.load(source) as z:
        data = {k: z[k] for k in z.files}
    # Native saved taxel centroids were averaged in float32 world coordinates.
    # Their 1.6 um roundoff exceeds the Newton rigid-site 1 um guard. Restore
    # the known rigid mounting using ONLY the first observed sensor geometry
    # and each measured hand pose; never change contacts, forces or targets.
    site_roundoff = {}
    for side in range(2):
        part = slice(side*27, (side+1)*27)
        pose = data['hand_pose_w'][:, side]
        matrix = Rotation.from_quat(pose[:, 3:]).as_matrix()
        local = (data['hand_sites_w'][0, part]-pose[0, :3]) @ matrix[0]
        rigid = np.einsum('tij,nj->tni', matrix, local)+pose[:, None, :3]
        difference = float(np.max(np.abs(rigid-data['hand_sites_w'][:, part])))
        if difference > 2e-6:
            raise ValueError('Native site discrepancy exceeds verified float32 rounding bound')
        site_roundoff[str(side)] = difference
        if side == 0:
            data['hand_sites_w'] = data['hand_sites_w'].astype(np.float64)
        data['hand_sites_w'][:, part] = rigid
    estimator = ObjectStateEstimator(endpoint, ROOT / 'checkpoints/utonia.pth')
    rows, frames = [], []
    for i in range(len(data['timestamp_s'])):
        result = estimator.update({k: data[k][i] for k in OBSERVATION_KEYS})
        if result is not None:
            rows.append(result)
            frames.append(i)
        if i % 10 == 0:
            print('CAUSAL_FRAME', i, flush=True)
    arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    arrays['frame'] = np.asarray(frames)
    np.savez_compressed(OUT / 'carry_predictions.npz', **arrays)
    idx = arrays['frame']
    true_R = Rotation.from_quat(data['object_pose_w'][idx, 3:])
    center = true_R.apply(data['object_local_center_m'][idx]) + data['object_pose_w'][idx, :3]
    errors = dict(center_cm=np.linalg.norm(center-arrays['center_w_m'], axis=1)*100,
                  rotation_deg=(true_R.inv()*Rotation.from_quat(arrays['object_rotation_w_xyzw'])).magnitude()*180/np.pi,
                  dimensions_pct=(np.abs(arrays['dimensions_m']/data['object_dimensions_m'][idx]-1)).mean(1)*100,
                  mass_pct=np.abs(arrays['mass_kg']/data['object_mass_kg'][idx]-1)*100)
    np.savez_compressed(OUT / 'carry_errors.npz', frame=idx, **errors)
    report = dict(source=str(source), source_sha256=hashlib.file_digest(source.open('rb'), 'sha256').hexdigest(),
                  endpoint=str(endpoint), endpoint_sha256=hashlib.file_digest(endpoint.open('rb'), 'sha256').hexdigest(),
                  frames=80, predictions=len(rows), warmup_frames=31, optimizer_updates=0,
                  inference='Frozen full Utonia force endpoint; strictly causal 32-history, all 80 observations in source order',
                  scope='Existing IsaacLab/TacSL actual official-policy lifting trace; cross-backend/out-of-training-distribution qualitative evaluation; no new physics or policy learning.',
                  rigid_site_world_correction_max_component_m=site_roundoff,
                  adapter='First-observation rigid mounting; at most 2 micrometers; contacts and forces unchanged; prior failed raw input retained in inference.log',
                  errors={k:dict(mean=float(v.mean()), last=float(v[-1])) for k,v in errors.items()})
    (OUT/'INFERENCE.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
