"""Matched perception corpus adapter, shared by centroid and surface experiments."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .data import OBSERVATION_KEYS, encode_observations, history_indices, target_at, collate
from .hand_surface_normals import query_observation_normals
from .surface_data import select_surface_frames, _surface_points, encode_surface_observations, collate_surface


def read(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


class PerceptionDataset(Dataset):
    def __init__(self, root, mode, split, history=32, stride=25, *, representation='surface',
                 history_policy='episode_uniform_recent', time_scale_s=150., normal_policy='hand_surface',
                 allow_incomplete_train=False):
        root = Path(root); status = json.loads((root / 'COLLECTION_RESULT.json').read_text())
        if not status['complete'] and not (allow_incomplete_train and split == 'train'):
            raise ValueError('Formal training/evaluation needs the complete declared corpus')
        if normal_policy != 'hand_surface' or representation not in ('centroid', 'surface'):
            raise ValueError('Unsupported perception representation')
        self.mode = mode; self.representation = representation; self.history = history
        self.history_policy = history_policy; self.time_scale_s = time_scale_s
        self.force_gain = 1.; self.shuffle_force = False
        self.episodes = []; self.index = []; self.surfaces = []; self.prepared = []; self.labels = []
        for record in status['records']:
            if record['split'] != split:
                continue
            path = Path(record['source']); episode = record['episode']
            metadata = json.loads((path / f'episode_{episode:04d}.json').read_text())
            data = read(path / f'episode_{episode:04d}.npz')
            labels = read(path / 'supervision.npz')['mass_label_weight']
            obs = {k: data[k] for k in OBSERVATION_KEYS}
            site, contact, _ = query_observation_normals(obs)
            data['hand_site_surface_normals_w'] = site
            data['hand_contact_surface_normals_w'] = contact
            if representation == 'surface':
                field = read(path / 'contact_surface.npz')
                surfaces = select_surface_frames(field, range(len(labels)))
                prepared = [_surface_points(obs, surface, frame, contact[frame]) for frame, surface in enumerate(surfaces)]
            else:
                surfaces = prepared = None
            e = len(self.episodes)
            self.episodes.append((metadata, data)); self.surfaces.append(surfaces)
            self.prepared.append(prepared); self.labels.append(labels)
            self.index.extend((e, frame) for frame in range(history - 1, len(labels), stride))
        if not self.index:
            raise ValueError('No complete episodes for split ' + split)

    def __len__(self):
        return len(self.index)

    def target(self, index):
        episode, frame = self.index[index]
        return target_at(self.episodes[episode][1], frame)

    def __getitem__(self, index):
        episode, frame = self.index[index]; metadata, data = self.episodes[episode]
        indices = history_indices(frame, self.history, self.history_policy)
        obs = {k: data[k][indices] for k in (*OBSERVATION_KEYS, 'hand_site_surface_normals_w', 'hand_contact_surface_normals_w')}
        if self.representation == 'surface':
            if self.shuffle_force:
                raise ValueError('Use the declared force-zero control for variable surface clouds')
            row = encode_surface_observations(obs, [self.surfaces[episode][i] for i in indices], self.mode,
                prepared_surfaces=[self.prepared[episode][i] for i in indices],
                time_scale_s=self.time_scale_s, force_gain=self.force_gain)
        else:
            row = encode_observations(obs, self.mode, normal_policy='hand_surface',
                time_scale_s=self.time_scale_s, force_gain=self.force_gain, shuffle_force=self.shuffle_force)
        row.update(target=target_at(data, frame), episode=metadata['episode'], frame=frame,
                   contact=bool((obs['normal_load_n'] > 1e-3).any()), mass_label_weight=float(self.labels[episode][frame]))
        return row


def collate_perception(rows):
    result = collate_surface(rows) if 'grid_coord' in rows[0] else collate(rows)
    result['contact'] = torch.tensor([row['contact'] for row in rows], dtype=torch.bool)
    result['mass_label_weight'] = torch.tensor([row['mass_label_weight'] for row in rows], dtype=torch.float32)
    return result


class MatchedGeometrySampler(Sampler):
    """Each draw: one geometry group, all four mass/grip settings, one clock."""
    def __init__(self, dataset, seed):
        self.rng = np.random.default_rng(seed)
        groups = {}
        for e, (meta, _) in enumerate(dataset.episodes):
            groups.setdefault(meta['geometry_group'], []).append(e)
        if any(len(v) != 4 for v in groups.values()):
            raise ValueError('Matched training requires four settings for every included geometry group')
        lookup = {(e, frame): index for index, (e, frame) in enumerate(dataset.index)}
        self.groups = []
        for group in sorted(groups):
            episodes = sorted(groups[group], key=lambda e: dataset.episodes[e][0]['episode'])
            clocks = [frame for e, frame in dataset.index if e == episodes[0]]
            if any([frame for e, frame in dataset.index if e == episode] != clocks for episode in episodes):
                raise ValueError('Case clock grids differ within geometry group')
            self.groups.append([[lookup[e, frame] for e in episodes] for frame in clocks])
        self.batches = math.ceil(len(dataset) / 4)

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            group = self.groups[int(self.rng.integers(len(self.groups)))]
            yield group[int(self.rng.integers(len(group)))].copy()
