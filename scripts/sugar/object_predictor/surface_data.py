"""Observed tactile surfaces -> the complete official Utonia input interface.

No object state, solver wrench, or true interface normal is consumed. Local plane
normals are fitted before voxelization, with known hand CAD as a fallback. Force
and area are integrated before feature transforms; they are never averaged as
though each contact-surface triangle were an independent full-force sensor.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .data import MODES
from .hand_surface_normals import query_observation_normals
from .inspect_contact_surface import fit_surface


SURFACE_KEYS = ('position_hand_frame_m', 'area_m2', 'normal_pressure_pa',
                'shear_traction_hand_frame_pa', 'pad', 'hand')


def select_surface_frames(arrays, frames):
    """Slice a saved ragged sensor stream without exposing validation arrays."""
    output = []
    for frame in frames:
        begin, end = arrays['offset'][frame:frame + 2]
        output.append({key: arrays[key][begin:end] for key in SURFACE_KEYS})
    return output


def _unit(vector):
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)


def _surface_points(obs, surface, frame, cad_contact):
    pad = np.asarray(surface['pad'])
    hand = np.asarray(surface['hand'])
    area = np.asarray(surface['area_m2'], dtype=np.float64)
    pressure = np.asarray(surface['normal_pressure_pa'], dtype=np.float64)
    assigned = (pad >= 0) & (pad < 54)
    # Occupancy is the same measured binary contact used by the sparse adapter.
    active = np.zeros(len(pad), bool)
    active[assigned] = obs['normal_load_n'][frame, pad[assigned]] > 1e-3
    selected = assigned & active & (area > 0) & (pressure > 0)
    p = pad[selected]; side = hand[selected]
    if not np.array_equal(side, p // 27):
        raise ValueError('Contact surface pad/hand assignment disagrees')
    local = np.asarray(surface['position_hand_frame_m'], dtype=np.float64)[selected]
    area = area[selected]; pressure = pressure[selected]
    traction = np.asarray(surface['shear_traction_hand_frame_pa'], dtype=np.float64)[selected]
    if not all(np.isfinite(v).all() for v in (local, area, pressure, traction)):
        raise ValueError('Nonfinite observed tactile surface')
    positions = np.empty_like(local); normals = np.empty_like(local); shear = np.empty_like(local)
    rotations = Rotation.from_quat(obs['hand_pose_w'][frame, :, 3:])
    resolved = np.zeros(len(local), bool)
    for h in (0, 1):
        ix = side == h
        positions[ix] = rotations[h].apply(local[ix]) + obs['hand_pose_w'][frame, h, :3]
        shear[ix] = rotations[h].apply(traction[ix] * area[ix, None])
    for patch in np.unique(p):
        ix = p == patch; normal = cad_contact[patch]
        if len(np.unique(local[ix], axis=0)) >= 6:
            _, rms, candidate = fit_surface(local[ix], area[ix])
            if rms[1] >= .0002 and rms[0] / max(rms[1], 1e-12) <= .35:
                candidate = rotations[patch // 27].apply(candidate)
                if candidate @ normal < 0:
                    candidate = -candidate
                normal = candidate; resolved[ix] = True
        normals[ix] = normal
    return dict(position=positions, normal=normals, area=area,
                load=area * pressure, shear=shear, hand=side,
                resolved=resolved, pad=p)


def encode_surface_observations(obs, surfaces, mode, *, voxel_m=.002,
                                time_scale_s=150., force_gain=1., prepared_surfaces=None):
    """One causal window; returns variable-size per-frame clouds of 20 features.

54 known hand sites remain in every mode. Contact modes additionally represent
the actual assigned contact surface. Touch can replace an anchor's coordinate
within the same voxel, but never supplies any coordinates to geometry-only mode.
The complete 138M-parameter backbone and its existing 11-column affine adapter
remain unchanged. This function is a sensor/data adapter, not a learned model.
"""
    if mode not in MODES or not np.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError('Invalid mode or fixed voxel size')
    if time_scale_s <= 0 or not np.isfinite(time_scale_s) or force_gain < 0 or not np.isfinite(force_gain):
        raise ValueError('Invalid fixed physical feature scale')
    count = len(obs['timestamp_s'])
    if len(surfaces) != count:
        raise ValueError('Sensor frame count does not match hand history')
    if mode == 'geometry':
        normal_obs = {**obs, 'normal_load_n': np.zeros_like(obs['normal_load_n']),
                      'contact_position_w': obs['hand_sites_w']}
    else:
        normal_obs = obs
    if 'hand_site_surface_normals_w' in obs and 'hand_contact_surface_normals_w' in obs:
        site_normals = obs['hand_site_surface_normals_w']
        contact_normals = obs['hand_contact_surface_normals_w']
    else:
        site_normals, contact_normals, _ = query_observation_normals(normal_obs)
    origin = obs['hand_pose_w'][-1, 0, :3].astype(float)
    current_rotation = Rotation.from_quat(obs['hand_pose_w'][-1, 0, 3:]).as_matrix()
    gravity = np.array([0., 0., -1.]) @ current_rotation
    output = dict(coord=[], feat=[], grid_coord=[], sensor_summary=[])
    for t in range(count):
        anchor = (obs['hand_sites_w'][t].astype(float) - origin) @ current_rotation * 5
        anchor_normal = site_normals[t].astype(float) @ current_rotation
        empty = dict(position=np.empty((0, 3)), normal=np.empty((0, 3)), area=np.empty(0),
                     load=np.empty(0), shear=np.empty((0, 3)), hand=np.empty(0), resolved=np.empty(0, bool))
        sensed = empty if mode == 'geometry' else (prepared_surfaces[t] if prepared_surfaces is not None
                 else _surface_points(obs, surfaces[t], t, contact_normals[t]))
        contact = (sensed['position'] - origin) @ current_rotation * 5
        xyz = np.concatenate((anchor, contact))
        keys = np.floor(xyz / (voxel_m * 5)).astype(np.int32)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        n = len(unique); anchor_idx = inverse[:54]; contact_idx = inverse[54:]
        anchor_count = np.bincount(anchor_idx, minlength=n)
        xyz_sum = np.zeros((n, 3)); normal_sum = np.zeros((n, 3)); side_sum = np.zeros(n)
        np.add.at(xyz_sum, anchor_idx, anchor)
        np.add.at(normal_sum, anchor_idx, anchor_normal)
        np.add.at(side_sum, anchor_idx, np.repeat([0., 1.], 27))
        xyz_sum /= np.maximum(anchor_count[:, None], 1)
        normal_sum /= np.maximum(anchor_count[:, None], 1)
        side_sum /= np.maximum(anchor_count, 1)
        area = np.bincount(contact_idx, weights=sensed['area'], minlength=n)
        load = np.bincount(contact_idx, weights=sensed['load'], minlength=n)
        shear = np.zeros((n, 3)); contact_xyz = np.zeros((n, 3)); contact_normal = np.zeros((n, 3))
        np.add.at(shear, contact_idx, sensed['shear'] @ current_rotation)
        np.add.at(contact_xyz, contact_idx, contact * sensed['area'][:, None])
        np.add.at(contact_normal, contact_idx, (sensed['normal'] @ current_rotation) * sensed['area'][:, None])
        contact_side = np.bincount(contact_idx, weights=sensed['hand'] * sensed['area'], minlength=n)
        occupied = area > 0
        xyz_sum[occupied] = contact_xyz[occupied] / area[occupied, None]
        normal_sum[occupied] = contact_normal[occupied] / area[occupied, None]
        side_sum[occupied] = contact_side[occupied] / area[occupied]
        normal_sum = _unit(normal_sum)
        extras = np.zeros((n, 11))
        extras[:, 0] = occupied
        if mode == 'geometry_contact_force':
            extras[:, 1] = np.log1p(load * force_gain)
            extras[:, 2:5] = np.arcsinh(shear * force_gain / 2)
            extras[:, 5] = area * 1e4
        extras[:, 6] = (obs['timestamp_s'][t] - obs['timestamp_s'][-1]) / time_scale_s
        extras[:, 7] = side_sum
        extras[:, 8:11] = gravity
        feat = np.concatenate((xyz_sum, np.zeros((n, 3)), normal_sum, extras), axis=1)
        output['coord'].append(xyz_sum.astype(np.float32))
        output['feat'].append(feat.astype(np.float32))
        output['grid_coord'].append(unique - unique.min(0))
        output['sensor_summary'].append(dict(input_faces=len(contact), occupied_voxels=int(occupied.sum()),
            fitted_normal_faces=int(sensed['resolved'].sum()), points=n,
            input_load_n=float(sensed['load'].sum()), voxel_load_n=float(load.sum()),
            input_area_m2=float(sensed['area'].sum()), voxel_area_m2=float(area.sum()),
            input_shear_current_hand_n=(sensed['shear'] @ current_rotation).sum(0).tolist(),
            voxel_shear_current_hand_n=shear.sum(0).tolist()))
    return output


def collate_surface(rows):
    """Concatenate already integrated voxels; do not re-average force features."""
    import torch
    coords = []; features = []; grids = []; sizes = []
    for row in rows:
        for coord, feat, grid in zip(row['coord'], row['feat'], row['grid_coord'], strict=True):
            coords.append(coord); features.append(feat); grids.append(grid); sizes.append(len(coord))
    result = dict(coord=torch.from_numpy(np.concatenate(coords)), feat=torch.from_numpy(np.concatenate(features)),
                  grid_coord=torch.from_numpy(np.concatenate(grids)), offset=torch.tensor(np.cumsum(sizes), dtype=torch.long))
    if all('target' in row for row in rows):
        result.update(target=torch.from_numpy(np.stack([row['target'] for row in rows])),
                      episode=torch.tensor([row['episode'] for row in rows]),
                      frame=torch.tensor([row['frame'] for row in rows]))
    return result
