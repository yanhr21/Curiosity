"""Causal observation/target adapters; no simulator dependency."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset

OBSERVATION_KEYS = ('hand_pose_w', 'hand_sites_w', 'hand_normals_w',
                    'contact_position_w', 'normal_load_n', 'shear_force_w',
                    'contact_area_m2', 'timestamp_s')
MODES = ('geometry', 'geometry_contact', 'geometry_contact_force')
HISTORY_POLICIES=('contiguous','episode_uniform_recent')
NORMAL_POLICIES=('stored','hand_surface')


def history_indices(frame,history=8,policy='contiguous'):
    """Clock-only causal sampling; never select using force, labels or phase IDs."""
    if policy not in HISTORY_POLICIES or history<3 or frame<history-1:
        raise ValueError('Invalid history request')
    if policy=='contiguous':
        return np.arange(frame-history+1,frame+1,dtype=np.int64)
    # Six spread-out past frames and the two most recent frames for history8.
    return np.r_[np.linspace(0,frame-2,history-2).astype(np.int64),frame-1,frame]


def encode_observations(obs, mode, shuffle_force=False, force_gain=1.,time_scale_s=1.,normal_policy='stored'):
    """All history ends at current t. Origin/frame is current LEFT HAND, never object.

    Coordinates use one fixed physical scale (5 units/m) for the pretrained model.
    Current Newton hand_normals_w stores rotated local palmar axes, not local
    CAD surface normals or the object's contact normal. 11 extra
    scalar/vector attributes enter only a zero-initialized input affine extension.
    """
    if mode not in MODES:
        raise ValueError(mode)
    origin = obs['hand_pose_w'][-1, 0, :3]
    R = Rotation.from_quat(obs['hand_pose_w'][-1, 0, 3:]).as_matrix()
    positions = obs['hand_sites_w'].copy()
    contact = obs['normal_load_n'] > 1e-3
    if mode != 'geometry':
        positions = np.where(contact[..., None], obs['contact_position_w'], positions)
    coord = ((positions - origin) @ R).astype(np.float32) * 5
    if normal_policy not in NORMAL_POLICIES: raise ValueError(normal_policy)
    normals=obs['hand_normals_w']
    if normal_policy=='hand_surface':
        key='hand_site_surface_normals_w' if mode=='geometry' else 'hand_contact_surface_normals_w'
        if key in obs:
            normals=obs[key]
        else:
            from .hand_surface_normals import query_observation_normals
            site_surface,contact_surface,_=query_observation_normals(obs)
            normals=site_surface if mode=='geometry' else contact_surface
    normal = (normals @ R).astype(np.float32)
    if not np.isfinite(time_scale_s) or time_scale_s<=0:
        raise ValueError('Time scale must be a positive fixed number')
    age = (obs['timestamp_s'] - obs['timestamp_s'][-1])/time_scale_s
    T, N = contact.shape
    extras = np.zeros((T, N, 11), np.float32)
    if mode != 'geometry':
        extras[..., 0] = contact
    if mode == 'geometry_contact_force':
        # Force scale is fixed in physical units; there is no per-cloud whitening.
        load=obs['normal_load_n'] if force_gain==1 else obs['normal_load_n']*force_gain
        shear=obs['shear_force_w'] if force_gain==1 else obs['shear_force_w']*force_gain
        extras[..., 1] = np.log1p(load)
        extras[..., 2:5] = np.arcsinh((shear @ R) / 2)
        extras[..., 5] = obs['contact_area_m2'] * 1e4
        if shuffle_force:
            # Explicit corruption control; never use this in normal training.
            extras[..., 1:6] = np.roll(extras[..., 1:6], 27, axis=1)
    extras[..., 6] = age[:, None]
    extras[:, 27:, 7] = 1
    # Known gravity direction in the current hand frame distinguishes supporting
    # from squeezing. It uses robot orientation, never object state.
    extras[..., 8:11] = np.array([0.,0.,-1.]) @ R
    feat = np.concatenate((coord, np.zeros_like(coord), normal, extras), axis=-1)
    return dict(coord=coord, feat=feat)


def target_at(arrays, frame):
    hand = arrays['hand_pose_w'][frame, 0]
    Rh = Rotation.from_quat(hand[3:]).as_matrix()
    obj = arrays['object_pose_w'][frame]
    Ro = Rotation.from_quat(obj[3:]).as_matrix()
    center_w = obj[:3] + Ro @ arrays['object_local_center_m'][frame]
    center = (center_w-hand[:3]) @ Rh
    relative_R = Rh.T @ Ro
    rot6 = relative_R[:, :2].T.reshape(-1)
    return np.concatenate((center, rot6, np.log(arrays['object_dimensions_m'][frame]),
                           [np.log(arrays['object_mass_kg'][frame])])).astype(np.float32)


class ContactDataset(Dataset):
    def __init__(self, root, mode, split, history=8, stride=5,history_policy='contiguous',time_scale_s=1.,normal_policy='stored'):
        self.mode = mode
        self.history = history
        if history_policy not in HISTORY_POLICIES: raise ValueError(history_policy)
        self.history_policy=history_policy
        self.time_scale_s=time_scale_s
        if normal_policy not in NORMAL_POLICIES: raise ValueError(normal_policy)
        self.normal_policy=normal_policy
        self.shuffle_force = False
        self.force_gain = 1.
        self.episodes = []
        self.index = []
        # Prospective motion split: test motions are absent from TRAIN and VAL.
        split_motions = dict(train={45,90}, val={96}, test={98})
        if split not in split_motions:
            raise ValueError(split)
        for path in sorted(Path(root).glob('episode_*.json')):
            meta = json.loads(path.read_text())
            if 'split' in meta:
                include=meta['split']==split
            else:
                include=meta.get('motion') in split_motions[split]
            if not include:
                continue
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as src:
                arrays = {k:src[k] for k in src.files}
            if arrays['hand_sites_w'].shape[1:] != (54,3):
                raise ValueError('Expected two real anatomical 27-patch hands')
            if normal_policy=='hand_surface':
                from .hand_surface_normals import query_observation_normals
                site_surface,contact_surface,_=query_observation_normals({k:arrays[k] for k in OBSERVATION_KEYS})
                arrays['hand_site_surface_normals_w']=site_surface
                arrays['hand_contact_surface_normals_w']=contact_surface
            e = len(self.episodes)
            self.episodes.append((meta,arrays))
            episode_stride=int(meta.get('sampling_stride',stride))
            if episode_stride<1: raise ValueError('Sampling stride must be positive')
            for t in range(history-1,len(arrays['timestamp_s']),episode_stride):
                self.index.append((e,t))
        if not self.index:
            raise ValueError(f'No complete {split} episodes in {root}')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        episode, frame = self.index[i]
        meta, arrays = self.episodes[episode]
        sl = (slice(frame-self.history+1,frame+1) if self.history_policy=='contiguous'
              else history_indices(frame,self.history,self.history_policy))
        obs = {k:arrays[k][sl] for k in OBSERVATION_KEYS}
        if self.normal_policy=='hand_surface':
            for k in ('hand_site_surface_normals_w','hand_contact_surface_normals_w'):obs[k]=arrays[k][sl]
        result = encode_observations(obs,self.mode,shuffle_force=self.shuffle_force,force_gain=self.force_gain,time_scale_s=self.time_scale_s,normal_policy=self.normal_policy)
        result.update(target=target_at(arrays,frame), episode=meta['episode'], frame=frame,
                      contact=bool(np.any(obs['normal_load_n']>1e-3)))
        return result


def collate(rows):
    # Separate time frames in the sparse backbone. A chronological readout uses
    # all frame features; it does not merge a moving object's history into a mesh.
    coords,features,grid,sizes=[],[],[],[]
    for row in rows:
        for c,f in zip(row['coord'],row['feat'],strict=True):
            g=np.floor(c/.01).astype(np.int32)
            # Sparse convolutions require unique sites. Average within a fixed
            # 2mm physical voxel; no random subsampling or mixing across time.
            unique,inverse=np.unique(g,axis=0,return_inverse=True)
            counts=np.bincount(inverse)
            fc=np.zeros((len(unique),f.shape[1]),np.float32)
            cc=np.zeros((len(unique),3),np.float32)
            np.add.at(fc,inverse,f)
            np.add.at(cc,inverse,c)
            features.append(fc/counts[:,None])
            coords.append(cc/counts[:,None])
            grid.append(unique-unique.min(0))
            sizes.append(len(unique))
    return dict(coord=torch.from_numpy(np.concatenate(coords).astype(np.float32)),
                grid_coord=torch.from_numpy(np.concatenate(grid)),
                feat=torch.from_numpy(np.concatenate(features).astype(np.float32)),
                offset=torch.from_numpy(np.cumsum(sizes)).long(),
                target=torch.from_numpy(np.stack([r['target'] for r in rows])),
                episode=torch.tensor([r['episode'] for r in rows]),
                frame=torch.tensor([r['frame'] for r in rows]),
                contact=torch.tensor([r['contact'] for r in rows]))
