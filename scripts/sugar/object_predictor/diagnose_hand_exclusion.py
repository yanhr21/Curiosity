"""Saved-only geometry diagnosis of potential hand-interior FREE observations.

No registration, candidate reselection, model learning, or physical collection.
Object truth is used solely to measure false exclusions in these TRAIN records.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import pytorch_volumetric as pv
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from sugar_newton.hand.patches import load_hand_mesh
from .shape_metric import rotation


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sdf_for(vertices, faces):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32)))
    return pv.MeshSDF(pv.MeshObjectFactory(mesh=mesh))


def values(sdf, points):
    result, _ = sdf(torch.as_tensor(points, dtype=torch.float32))
    return result.detach().numpy()


def main(args):
    assert os.environ.get('SLURM_STEP_ID')
    np.random.seed(20260916); torch.set_num_threads(4)
    root = Path(args.root); out = root / 'hand_exclusion_diagnosis'; out.mkdir(exist_ok=False)
    report = json.loads((root / 'RESULT.json').read_text())
    protocol = dict(grid_spacing_m=.003, diagnostic_erosion_m=[.005, .010],
        measured_contact_exclusion_radius_m=.010, seed=20260916,
        scope='Two fixed diagnostic bands, not tuned FREE settings. Current full hand CAD only.',
        physics_steps=0, model_updates=0, registration_calls=0,
        ground_truth='Evaluator only, measures false exclusion risk; never used to create interior points')
    (out / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2))
    local = {}; topology = {}
    for side in ('left', 'right'):
        mesh = load_hand_mesh(side)
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
        sdf = sdf_for(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        axes = [np.arange(a+.0015, b, .003) for a, b in mesh.bounds.T]
        points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        depth = values(sdf, points)
        core = points[depth <= -.005]
        assert len(core)
        check = core[np.linspace(0, len(core)-1, min(len(core), 512)).astype(int)]
        inside = mesh.contains(check)
        assert inside.all(), 'Independent ray test disagrees on eroded hand interior'
        local[side] = {d: points[depth <= -d] for d in protocol['diagnostic_erosion_m']}
        topology[side] = dict(vertices=len(mesh.vertices), faces=len(mesh.faces),
            watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
            volume_m3=float(mesh.volume), independent_deep_interior_queries=len(check),
            independent_all_inside=bool(inside.all()), self_intersections='not separately certified')
    prepared = []
    # Derive all putative FREE observations without reading object labels.
    for row in report['cases']:
        name = f'episode_{row["episode"]}_frame_{row["frame"]}.npz'
        saved = read(root / name); hand = saved['hand_pose_w']
        rotations = Rotation.from_quat(hand[:, 3:]).as_matrix()
        sensed = saved['points_current_left_hand_m']; tree = cKDTree(sensed)
        bands = {}
        for depth in protocol['diagnostic_erosion_m']:
            parts = []
            for h, side in enumerate(('left', 'right')):
                world = local[side][depth] @ rotations[h].T + hand[h, :3]
                points = (world-hand[0, :3]) @ rotations[0]
                distance = tree.query(points, workers=4)[0]
                parts.append(points[distance > protocol['measured_contact_exclusion_radius_m']])
            bands[depth] = np.concatenate(parts)
        np.savez_compressed(out / name, **{f'hand_core_{int(d*1000)}mm': p for d,p in bands.items()})
        prepared.append((row, saved, bands))
    # Only now load independent evaluator truth and compare its overlap risk.
    labels = read(root.parent / 'chsel_qualification_inputs/evaluation_only.npz')
    mesh = read(Path('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz'))
    v = mesh['vertices'].astype(np.float64); dims = np.ptp(v, axis=0)
    v -= (v.max(0)+v.min(0))/2
    rows = []
    for i, (row, saved, bands) in enumerate(prepared):
        assert labels['episode'][i] == row['episode'] and labels['frame'][i] == row['frame']
        scores = {}
        for key, state in [('utonia',saved['original_prediction']), ('surface_only_chsel',saved['prediction']), ('truth_evaluator_only',labels['target'][i])]:
            sdf = sdf_for(v*(np.exp(state[9:12])/dims), mesh['faces'])
            per_band = {}
            for depth, points in bands.items():
                if not len(points):
                    per_band[str(depth)] = dict(points=0); continue
                q = (points-state[:3]) @ rotation(state[3:9])
                distance = values(sdf, q); penetration = np.maximum(-distance, 0)
                per_band[str(depth)] = dict(points=len(points), inside_fraction=float((distance<0).mean()),
                    mean_penetration_cm=float(penetration.mean()*100),
                    max_penetration_cm=float(penetration.max()*100),
                    p95_penetration_cm=float(np.quantile(penetration,.95)*100))
            scores[key] = per_band
        result = dict(episode=row['episode'], frame=row['frame'], scores=scores)
        rows.append(result); print('HAND_EXCLUSION',json.dumps(result),flush=True)
    (out / 'RESULT.json').write_text(json.dumps(dict(complete=True, topology=topology,cases=rows,
        scope=protocol['scope'],registration_calls=0,model_updates=0,physics_steps=0,
        warning='Low overlap alone cannot determine pose. Truth overlap measures invalid FREE observations; these bands are not yet accepted as optimizer inputs.'),indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);main(ap.parse_args())
