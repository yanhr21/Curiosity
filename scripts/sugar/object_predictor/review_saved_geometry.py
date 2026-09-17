"""Post-hoc geometry diagnosis on saved predictions, not a new learned model.

Retains raw pose errors and original acceptance. Full-vertex nearest-neighbor
distances are sampling-dependent diagnostics, not exact continuous-surface SDF
or BOP MSSD. No asset-axis symmetry is assumed for the oblique scanned box.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .shape_metric import cloud
from .report_grip_transfer import errors


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use the retained compute step')
    root = Path(args.root); out = Path(args.output); out.mkdir(exist_ok=False)
    data_root = Path(json.loads((root / 'PROTOCOL.json').read_text())['data'])
    mesh_path = Path('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz')
    mesh = load(mesh_path); v = mesh['vertices'].astype(np.float64)
    dimensions = np.ptp(v, axis=0); v -= (v.max(0) + v.min(0)) / 2
    assert len(v) == 15626 and len(mesh['faces']) == 31248
    sources = {str(mesh_path): digest(mesh_path)}; rows = {}; arrays = {}
    cases = {}
    for ep in range(5040, 5048):
        p = data_root / 'cases' / f'episode_{ep}' / f'episode_{ep}.npz'
        cases[ep] = load(p); sources[str(p)] = digest(p)
    reference = None
    for arm in ('centroid_force', 'surface_force', 'surface_force_airborne_mass_guard'):
        p = root / 'batch1' / arm / 'predictions.npz'; a = load(p); sources[str(p)] = digest(p)
        if reference is None:
            reference = a
        else:
            for key in ('episode', 'frame', 'target', 'mass_label_weight'):
                assert np.array_equal(reference[key], a[key]), key
        assert len(a['frame']) == 760
        for ep in range(5040, 5048):
            assert np.array_equal(a['frame'][a['episode'] == ep], np.arange(31, 2400, 25))
        values = errors(a['prediction'], a['target'])
        diagnostic = {k: [] for k in ('vertex_mean_cm', 'vertex_p95_cm',
            'contact_to_prediction_vertex_cm', 'contact_to_truth_vertex_cm',
            'contact_count', 'time_s', 'target_world_center_difference_m')}
        for i, (pred, target) in enumerate(zip(a['prediction'], a['target'], strict=True)):
            ep, frame = int(a['episode'][i]), int(a['frame'][i]); case = cases[ep]
            pc = cloud(v, pred.astype(float), dimensions)
            tc = cloud(v, target.astype(float), dimensions)
            pt, tt = cKDTree(pc), cKDTree(tc)
            distances = np.r_[tt.query(pc, workers=1)[0], pt.query(tc, workers=1)[0]]
            diagnostic['vertex_mean_cm'].append(distances.mean() * 100)
            diagnostic['vertex_p95_cm'].append(np.quantile(distances, .95) * 100)
            hand = case['hand_pose_w'][frame, 0]; rh = Rotation.from_quat(hand[3:]).as_matrix()
            obj = case['object_pose_w'][frame]; ro = Rotation.from_quat(obj[3:]).as_matrix()
            expected_center = obj[:3] + ro @ case['object_local_center_m'][frame]
            diagnostic['target_world_center_difference_m'].append(
                np.linalg.norm(rh @ target[:3] + hand[:3] - expected_center))
            active = case['normal_load_n'][frame] > 1e-3
            contact = (case['contact_position_w'][frame, active] - hand[:3]) @ rh
            diagnostic['contact_count'].append(int(active.sum()))
            diagnostic['contact_to_prediction_vertex_cm'].append(
                float(pt.query(contact, workers=1)[0].mean() * 100) if len(contact) else np.nan)
            diagnostic['contact_to_truth_vertex_cm'].append(
                float(tt.query(contact, workers=1)[0].mean() * 100) if len(contact) else np.nan)
            diagnostic['time_s'].append(float(case['timestamp_s'][frame]))
            if (i + 1) % 95 == 0:
                print(arm, 'complete_episode', ep, flush=True)
        values.update({k: np.asarray(x) for k, x in diagnostic.items()})
        assert np.max(values['target_world_center_difference_m']) < 1e-6
        assert np.isfinite(values['vertex_mean_cm']).all()
        arrays[arm] = dict(**values, episode=a['episode'], frame=a['frame'],
                           mass_label_weight=a['mass_label_weight'])
        np.savez_compressed(out / f'{arm}.npz', **arrays[arm])
        groups = {}
        for group, mask in [('all', np.ones(760, bool)), ('airborne_hold', a['mass_label_weight'] > 0),
                            ('late44s', values['time_s'] >= 44)]:
            per = {}
            for ep in range(5040, 5048):
                selected = mask & (a['episode'] == ep)
                metrics = {}
                for key, value in values.items():
                    finite = selected & np.isfinite(value)
                    metrics[key] = float(value[finite].mean()) if finite.any() else None
                per[str(ep)] = dict(windows=int(selected.sum()), metrics=metrics)
            equal = {}
            for key in values:
                valid = [x['metrics'][key] for x in per.values() if x['metrics'][key] is not None]
                equal[key] = float(np.mean(valid)) if valid else None
            groups[group] = dict(windows=int(mask.sum()), per_episode=per, equal_episode=equal)
        rows[arm] = groups
    result = dict(complete=True, arms=rows, vertices=len(v), faces=len(mesh['faces']),
        scope='Post-hoc full-vertex geometry/contact diagnosis; no new model, no symmetry assumption, no threshold/acceptance change. Nearest-vertex distances are not exact surface distances; contact truth residual is a reference, not zero by construction.',
        optimizer_updates=0, model_forwards=0, physics_steps=0, source_sha256=sources)
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex='col')
    colors = ('tab:blue', 'tab:orange', 'tab:green')
    for col, ep in enumerate((5045, 5046, 5047)):
        for (arm, a), color in zip(arrays.items(), colors):
            ix = a['episode'] == ep; t = a['time_s'][ix]
            for row, key in enumerate(('rotation_deg', 'vertex_mean_cm', 'contact_to_prediction_vertex_cm')):
                axes[row, col].plot(t, a[key][ix], label=arm, color=color)
        base = arrays['centroid_force']; ix = base['episode'] == ep
        axes[2, col].plot(base['time_s'][ix], base['contact_to_truth_vertex_cm'][ix], 'k--', label='Truth contact reference')
        axes[0, col].set_title(f'Original TEST {ep}'); axes[2, col].set_xlabel('Recorded time [s]')
    for row, name in enumerate(('Raw rotation [deg]', 'Mean nearest vertex [cm]', 'Contact to nearest vertex [cm]')):
        axes[row, 0].set_ylabel(name)
    axes[0, 0].legend(fontsize=7); axes[2, 0].legend(fontsize=6)
    fig.suptitle('Saved predictions: raw rotation vs actual geometry/contact discrepancy\nAll original clocks; diagnostic only, original failed acceptance unchanged')
    fig.tight_layout(); fig.savefig(out / 'geometry_contact_diagnosis.png', dpi=140); plt.close(fig)
    print(json.dumps({arm: x['all']['equal_episode'] for arm, x in rows.items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    main(p.parse_args())
