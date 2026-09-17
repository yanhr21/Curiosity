"""Validation-only decomposition of saved tactile support deficits.

Full-surface and solver quantities here are diagnostic references, never model
inputs or a proposed observable force correction. No calibration is fitted.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

TERMS = ('normal', 'field_projection', 'unassigned', 'pad_readout')


def decompose(case, frames):
    with np.load(case / f'{case.name}.npz', allow_pickle=False) as f:
        keys = ('hand_pose_w', 'shear_force_w', 'normal_load_n', 'timestamp_s',
                'object_mass_kg', 'validation_full_mesh_min_z_m',
                'validation_controller_phase', 'validation_hand_normal_vec_w',
                'validation_hand_friction_vec_w', 'validation_resolved_normal_n')
        a = {k: f[k][frames] for k in keys}
    with np.load(case / 'contact_surface.npz', allow_pickle=False) as f:
        dense = {k: f[k] for k in ('offset', 'area_m2', 'normal_pressure_pa',
                 'shear_traction_hand_frame_pa', 'pad', 'hand',
                 'validation_surface_normal_hand_frame')}
    all_shear = []; assigned_shear = []; checks = []
    for i, frame in enumerate(frames):
        start, end = dense['offset'][frame:frame + 2]
        area = dense['area_m2'][start:end].astype(float)
        pressure = dense['normal_pressure_pa'][start:end].astype(float)
        traction = dense['shear_traction_hand_frame_pa'][start:end].astype(float)
        pad = dense['pad'][start:end]; hand = dense['hand'][start:end]
        normals = dense['validation_surface_normal_hand_frame'][start:end].astype(float)
        assigned = (pad >= 0) & (pad < 54)
        assert np.array_equal(hand[assigned], pad[assigned] // 27)
        rotations = Rotation.from_quat(a['hand_pose_w'][i, :, 3:])
        total = np.zeros((2, 3)); covered = np.zeros((2, 3))
        scalar_error = 0.; replay_error = 0.
        for h in (0, 1):
            ix = hand == h
            force = rotations[h].apply(traction[ix] * area[ix, None])
            total[h] = force.sum(0)
            covered[h] = force[assigned[ix]].sum(0)
            resolved = float(a['validation_resolved_normal_n'][i, h])
            weights = pressure[ix] * area[ix]
            scalar_error = max(scalar_error, abs(weights.sum() - resolved))
            # Replay the existing field kernel using validation-only normals.
            ft = rotations[h].inv().apply(a['validation_hand_friction_vec_w'][i, h].astype(float))
            n = normals[ix]
            tangent = ft - (n @ ft)[:, None] * n
            norm = np.linalg.norm(tangent, axis=1)
            unit = tangent / np.maximum(norm[:, None], 1e-12)
            unit[norm <= 1e-12] = 0.
            predicted = unit * (weights * np.linalg.norm(ft) / max(resolved, 1e-9))[:, None]
            if len(predicted):
                replay_error = max(replay_error, float(np.max(abs(predicted - traction[ix] * area[ix, None]))))
        all_shear.append(total.sum(0)); assigned_shear.append(covered.sum(0))
        checks.append(dict(normal_scalar_closure_max_n=scalar_error,
                           per_face_traction_replay_max_n=replay_error))
    total = np.asarray(all_shear); assigned = np.asarray(assigned_shear)
    recorded = a['shear_force_w'].astype(float).sum(1)
    normal = a['validation_hand_normal_vec_w'].astype(float).sum(1)
    friction = a['validation_hand_friction_vec_w'].astype(float).sum(1)
    # All components are force on OBJECT, in world coordinates.
    terms = dict(normal=-normal, field_projection=total-friction,
                 unassigned=assigned-total, pad_readout=recorded-assigned)
    deficit = -normal-friction+recorded
    closure = deficit-sum(terms.values())
    loads = a['normal_load_n'].reshape(-1, 2, 27).sum(2)
    hold = (a['validation_full_mesh_min_z_m'] > .01) & (loads > .01).all(1) & (a['validation_controller_phase'] == 2)
    trace = dict(frame=frames, timestamp_s=a['timestamp_s'], mass_kg=a['object_mass_kg'],
                 airborne_hold=hold, observed_support_w=-recorded,
                 full_solver_support_w=-normal-friction, full_field_shear_support_w=-total,
                 assigned_field_shear_support_w=-assigned, deficit_w=deficit,
                 **{k+'_w': v for k, v in terms.items()})
    checks = dict(normal_scalar_closure_max_n=max(x['normal_scalar_closure_max_n'] for x in checks),
                  per_face_traction_replay_max_n=max(x['per_face_traction_replay_max_n'] for x in checks),
                  pad_readout_max_n=float(np.max(abs(recorded-assigned))),
                  decomposition_closure_max_n=float(np.max(abs(closure))))
    assert all(np.isfinite(v).all() for v in trace.values())
    return trace, checks


def main():
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run on retained compute step')
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); args = ap.parse_args()
    root = Path(args.output); protocol = json.loads((root/'PROTOCOL.json').read_text())
    frames = np.asarray(protocol['frames']); rows = []; checks = []
    for ep in protocol['episodes']:
        case = Path(protocol['source'])/'cases'/f'episode_{ep}'
        trace, check = decompose(case, frames)
        passed = json.loads((case/'RESULT.json').read_text())['passed']
        np.savez_compressed(root/f'episode_{ep}.npz', **trace)
        check['episode'] = ep; checks.append(check)
        for group, mask in dict(all=np.ones(len(frames), bool), airborne_hold=trace['airborne_hold'],
                                late44_airborne_hold=trace['airborne_hold'] & (trace['timestamp_s'] >= 44)).items():
            row = dict(episode=ep, physical_passed=passed, group=group, samples=int(mask.sum()))
            weight = trace['mass_kg'].astype(float)*protocol['gravity_m_s2']
            for key in (*TERMS, 'deficit'):
                value = trace[key+'_w'][:, 2]/weight*100
                row[key+'_signed_pct_weight'] = float(value[mask].mean()) if mask.any() else None
                row[key+'_absolute_pct_weight'] = float(abs(value[mask]).mean()) if mask.any() else None
            for key in ('observed_support', 'full_solver_support', 'full_field_shear_support'):
                value = abs(trace[key+'_w'][:, 2]/weight-1)*100
                row[key+'_mass_error_pct'] = float(value[mask].mean()) if mask.any() else None
            rows.append(row)
        print('FORCE_DEFICIT_CASE_COMPLETE', ep, check, flush=True)
    summary = []
    for group in ('all', 'airborne_hold', 'late44_airborne_hold'):
        for subset in ('all', 'passed', 'failed'):
            part = [r for r in rows if r['group'] == group and (subset == 'all' or r['physical_passed'] == (subset == 'passed'))]
            covered = [r for r in part if r['samples']]
            value = dict(group=group, physical=subset, total_episodes=len(part), episodes_covered=len(covered), samples=sum(r['samples'] for r in part))
            for k in rows[0]:
                if k.endswith(('_pct_weight', '_mass_error_pct')):
                    value[k] = float(np.mean([r[k] for r in covered])) if covered else None
            summary.append(value)
    result = dict(complete=len(checks)==32, rows=rows, summary=summary, checks=checks,
                  validation_only=True, new_physics=0, model_forwards=0, optimizer_updates=0)
    (root/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    part = [r for r in rows if r['group']=='late44_airborne_hold']
    fig, ax = plt.subplots(figsize=(15, 6)); x = np.arange(len(part))
    up = np.zeros(len(part)); down = np.zeros(len(part))
    for term in TERMS:
        y = np.array([r[term+'_signed_pct_weight'] if r['samples'] else np.nan for r in part])
        base = np.where(y >= 0, up, down)
        ax.bar(x, y, bottom=base, label=term)
        up += np.where(y > 0, y, 0); down += np.where(y < 0, y, 0)
    ax.scatter(x, [r['deficit_signed_pct_weight'] if r['samples'] else np.nan for r in part], c='k', marker='_', s=100, label='actual signed deficit')
    for i, r in enumerate(part):
        if not r['samples']: ax.text(i, 0, 'missing', rotation=90, va='bottom', ha='center', fontsize=7, color='gray')
    ax.set_xticks(x, [str(r['episode'])+('*' if r['physical_passed'] else '') for r in part], rotation=90)
    ax.set_ylabel('Signed contribution / true weight (%)'); ax.axhline(0, color='gray', linewidth=.6)
    ax.set_title('Validation-only support deficit decomposition: all32 original cases, late44s airborne hold\n* original physical PASS; missing cases retained; positive = observed shear understates support')
    ax.legend(ncol=5, fontsize=8); ax.grid(axis='y', alpha=.2); fig.tight_layout(); fig.savefig(root/'deficit_components.png', dpi=150); plt.close(fig)
    print('COMPLETE_FORCE_DEFICIT', json.dumps([r for r in summary if r['group']=='late44_airborne_hold']), flush=True)


if __name__ == '__main__':
    main()
