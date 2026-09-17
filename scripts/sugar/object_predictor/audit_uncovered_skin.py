"""Localize saved unassigned contact faces; never expand the sensor implicitly."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sugar_newton.hand.patches import patch_footprints, PATCH_SPECS

PALM_SIGNS = np.array([-1., 1.])  # Actual SupportScene/GripScene convention.


def classify(position, hand, pad):
    """Repeat field.py footprint geometry and measure planar rectangle distance.

    Distance is in the hand XZ plane, not surface geodesic or hardware clearance.
    The unused generic palm_normal_sign helper is not the scene convention.
    """
    footprints = patch_footprints().astype(float)
    dx = position[:, None, 0]-footprints[None, :, 0]
    dz = position[:, None, 2]-footprints[None, :, 1]
    ca = np.cos(footprints[:, 4]); sa = np.sin(footprints[:, 4])
    lx = dx*ca+dz*sa; lz = -dx*sa+dz*ca
    excess = np.stack((abs(lx)-footprints[:, 2], abs(lz)-footprints[:, 3]), axis=-1)
    distances = np.linalg.norm(np.maximum(excess, 0.), axis=-1)
    inside = (excess <= 0).all(-1)
    hemisphere = position[:, 1]*PALM_SIGNS[hand] > 0
    replay = np.where(hemisphere & inside.any(1), inside.argmax(1)+27*hand, -1)
    assigned = (pad >= 0) & (pad < 54)
    categories = np.where(assigned, 0, np.where(hemisphere, 1, 2))
    nearest = distances.argmin(1)
    return categories, replay, distances.min(1), nearest


def weighted_quantile(values, weights, quantile):
    if not len(values) or weights.sum() <= 0: return None
    order = np.argsort(values)
    return float(values[order][np.searchsorted(np.cumsum(weights[order]), quantile*weights.sum())])


def main():
    if not os.environ.get('SLURM_STEP_ID'): raise RuntimeError('Use retained compute step')
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); args = ap.parse_args()
    root = Path(args.output); p = json.loads((root/'PROTOCOL.json').read_text())
    frames = np.asarray(p['frames']); rows = []; checks = []; locations = []
    for ep in p['episodes']:
        case = Path(p['source'])/'cases'/f'episode_{ep}'
        with np.load(case/f'episode_{ep}.npz', allow_pickle=False) as f:
            a = {k:f[k][frames] for k in ('hand_pose_w', 'timestamp_s', 'object_mass_kg',
                 'validation_full_mesh_min_z_m', 'validation_controller_phase', 'normal_load_n')}
        with np.load(case/'contact_surface.npz', allow_pickle=False) as f:
            dense = {k:f[k] for k in ('offset', 'position_hand_frame_m', 'area_m2',
                     'normal_pressure_pa', 'shear_traction_hand_frame_pa', 'pad', 'hand')}
        with np.load(Path(p['decomposition'])/f'episode_{ep}.npz', allow_pickle=False) as f:
            prior_unassigned = f['unassigned_w']; assert np.array_equal(frames, f['frame'])
        passed = json.loads((case/'RESULT.json').read_text())['passed']
        force = np.zeros((len(frames), 3, 3)); counts = np.zeros((len(frames), 3), int)
        load = np.zeros((len(frames), 3)); mismatch = 0; max_partition_error = 0.
        for i, frame in enumerate(frames):
            start, end = dense['offset'][frame:frame+2]
            pos = dense['position_hand_frame_m'][start:end].astype(float)
            hand = dense['hand'][start:end]; pad = dense['pad'][start:end]
            weight = dense['area_m2'][start:end].astype(float)*dense['normal_pressure_pa'][start:end].astype(float)
            if not len(pos): continue
            category, replay, distance, nearest = classify(pos, hand, pad)
            mismatch += int((replay != pad).sum())
            local_force = dense['shear_traction_hand_frame_pa'][start:end].astype(float)*dense['area_m2'][start:end, None]
            world_force = np.empty_like(local_force)
            rot = Rotation.from_quat(a['hand_pose_w'][i, :, 3:])
            for h in (0, 1): world_force[hand == h] = -rot[h].apply(local_force[hand == h])
            for cat in range(3):
                ix = category == cat
                force[i, cat] = world_force[ix].sum(0); load[i, cat] = weight[ix].sum(); counts[i, cat] = ix.sum()
            max_partition_error = max(max_partition_error, float(np.max(abs(force[i, 1:].sum(0)-prior_unassigned[i]))))
            for h in (0, 1):
                ix = (category == 1) & (hand == h)
                if not ix.any(): continue
                locations.append(dict(episode=ep, frame=int(frame), hand=h,
                    timestamp_s=float(a['timestamp_s'][i]), faces=int(ix.sum()), load_n=float(weight[ix].sum()),
                    distance_weighted_p50_mm=weighted_quantile(distance[ix]*1000, weight[ix], .5),
                    distance_weighted_p95_mm=weighted_quantile(distance[ix]*1000, weight[ix], .95),
                    nearest_pads={PATCH_SPECS[k].name:float(weight[ix & (nearest == k)].sum()) for k in np.unique(nearest[ix])}))
            if ep in p['visual_episodes'] and int(frame) == p['visual_frame']:
                np.savez_compressed(root/f'points_{ep}.npz', position_hand_frame_m=pos,
                                    hand=hand, pad=pad, category=category, load_n=weight,
                                    nearest_pad=nearest, distance_m=distance)
        loads = a['normal_load_n'].reshape(-1, 2, 27).sum(2)
        hold = (a['validation_full_mesh_min_z_m']>.01) & (a['validation_controller_phase']==2) & (loads>.01).all(1)
        for group, mask in dict(all=np.ones(len(frames), bool), airborne_hold=hold,
                                late44_airborne_hold=hold & (a['timestamp_s']>=44)).items():
            row = dict(episode=ep, physical_passed=passed, group=group, samples=int(mask.sum()))
            for cat, name in enumerate(('assigned', 'palmar_outside_footprints', 'opposite_halfspace')):
                relative = force[:, cat, 2]/(a['object_mass_kg'].astype(float)*p['gravity_m_s2'])*100
                row[name+'_signed_support_pct_weight'] = float(relative[mask].mean()) if mask.any() else None
                row[name+'_faces'] = int(counts[mask, cat].sum())
                row[name+'_normal_load_n'] = float(load[mask, cat].mean()) if mask.any() else None
            rows.append(row)
        checks.append(dict(episode=ep, footprint_replay_mismatched_faces=mismatch, unassigned_partition_max_n=max_partition_error))
        print('UNCOVERED_SKIN_CASE_COMPLETE', ep, checks[-1], flush=True)
    summary = []
    for group in ('all', 'airborne_hold', 'late44_airborne_hold'):
        for subset in ('all', 'passed', 'failed'):
            part = [x for x in rows if x['group']==group and (subset=='all' or x['physical_passed']==(subset=='passed'))]
            covered = [x for x in part if x['samples']]
            item = dict(group=group, physical=subset, total_episodes=len(part), covered_episodes=len(covered), samples=sum(x['samples'] for x in part))
            for k in rows[0]:
                if k.endswith(('_pct_weight', '_load_n')): item[k] = float(np.mean([x[k] for x in covered])) if covered else None
                if k.endswith('_faces'): item[k] = sum(x[k] for x in part)
            summary.append(item)
    result = dict(complete=len(checks)==32, checks=checks, rows=rows, summary=summary,
                  locations=locations, categories=['assigned', 'palmar_outside_footprints', 'opposite_halfspace'],
                  no_sensor_expansion=True, new_physics=0, new_model_forwards=0, new_optimizer_updates=0)
    (root/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('COMPLETE_UNCOVERED_SKIN', json.dumps([x for x in summary if x['group']=='late44_airborne_hold']), flush=True)


if __name__ == '__main__': main()
