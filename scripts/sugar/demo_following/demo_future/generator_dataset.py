"""Actual Tracker data adapter for the full official SUGAR Generator.

Uses the official per-sample formatter and normalizer. This module creates no
model and performs no optimization. Future commands remain labels only.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import dill
import numpy as np
import torch

from sugar_il.common.pytorch_util import dict_apply
from sugar_il.dataset.base_dataset import BaseLowdimDataset
from sugar_il.dataset.generator_dataset import GeneratorDataset
from sugar_il.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

from scripts.sugar.demo_following.demo_future.generator_demo_geometry import CONTEXT_KEY, CONTEXT_WIDTH

FIELDS = ('obj_pos_b', 'obj_ori_b', 'joint_pos', 'project_gravity',
          'target_obj_pos_b', 'target_obj_ori_b', 'action', 'last_action')


def format_geometry_sample(sample, geometry):
    """Keep the official one-observation/eight-command interface exactly."""
    if sample['action'].shape != (8, 36) or geometry.shape != (8, 21):
        raise ValueError('Require full official8x36 labels and original8x21 context')
    layout = SimpleNamespace(n_obs_steps=1, use_last_action=True, use_target=True)
    data = GeneratorDataset._sample_to_data(layout, sample)
    data['obs'][CONTEXT_KEY] = geometry.astype(np.float32).reshape(1, CONTEXT_WIDTH)
    return dict_apply(data, torch.from_numpy)


def released_normalizer_with_geometry(parent_checkpoint, training_geometry):
    """Preserve every released statistic; fit the one additional TRAIN field."""
    with Path(parent_checkpoint).open('rb') as stream:
        payload = torch.load(stream, pickle_module=dill, map_location='cpu')
    prefix = 'normalizer.'
    state = {key[len(prefix):]: value for key, value in payload['state_dicts']['model'].items()
             if key.startswith(prefix)}
    if not state:
        raise RuntimeError('Released complete normalizer is missing')
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(state, strict=True)
    normalizer[CONTEXT_KEY] = SingleFieldLinearNormalizer.create_fit(
        training_geometry.reshape(-1, CONTEXT_WIDTH), mode='limits')
    normalizer.requires_grad_(False)
    if not all(torch.equal(value, normalizer.state_dict()[key]) for key, value in state.items()):
        raise RuntimeError('Released normalization statistics changed')
    return normalizer


class ActualDemoGeometryDataset(BaseLowdimDataset):
    """Read completed native TRAIN records; never admit pending or failed cases.

    Optional old paired TRAIN timelines are explicitly recorded as additional
    timelines of existing sources, not independent motions or replicates.
    Dataset sampling weights and optimization budget belong to the later
    predeclared matched training protocol, not this data adapter.
    """
    def __init__(self, corpus, parent_checkpoint, paired_train=None):
        super().__init__()
        self.corpus = Path(corpus)
        self.parent_checkpoint = Path(parent_checkpoint)
        result = json.loads((self.corpus / 'RESULT.json').read_text())
        plan = json.loads((self.corpus / 'PROTOCOL.json').read_text())
        if not result['execution_completed'] or result['optimizer_updates'] != 0:
            raise RuntimeError('Complete actual frozen TRAIN collection first')
        if set(map(int, result['per_source'])) != set(plan['sources']):
            raise RuntimeError('All predeclared TRAIN attempts must be retained')
        self.episodes, self.indices, self.timeline_metadata = [], [], []
        for source in plan['sources']:
            entry = result['per_source'][str(source)]
            if not entry['usable']:
                continue
            if not entry['context_binding']['source_and_all_frame_indices_exact']:
                raise RuntimeError('Actual labels lack exact source/context binding')
            self._add_episode(
                self.corpus / f'official_il_data/source_{source:03d}_IL.npz',
                self.corpus / f'original_demo_geometry_context/source_{source:03d}.npz',
                source, 'native_train', native=True)
        if paired_train is not None:
            paired = Path(paired_train)
            paired_result = json.loads((paired / 'READBACK.json').read_text())
            if not paired_result['data_pair_feasibility_passed']:
                raise RuntimeError('Old paired TRAIN execution must be complete')
            for source, arm in ((96, 'original'), (90, 'alternate')):
                if source not in plan['sources']:
                    raise RuntimeError('Paired anchor is outside TRAIN')
                self._add_episode(paired / f'official_il_data/{arm}_IL.npz',
                                  paired / f'original_demo_geometry_context/{arm}.npz',
                                  source, 'paired_train_' + arm, native=False)
        if not self.indices:
            raise RuntimeError('No complete usable actual TRAIN episodes')
        self._normalizer = None

    def _add_episode(self, label_path, context_path, source, role, native):
        with np.load(label_path) as labels, np.load(context_path) as context:
            starts = labels['usable_chunk_start'].copy()
            geometry = context['original_demo_geometry'].copy()
            if geometry.shape != (len(starts), 8, 21) or not np.isfinite(geometry).all():
                raise RuntimeError('Original-demo context geometry differs')
            frames = np.stack([labels['control_frame'][i:i + 8] for i in starts])
            if not np.all(np.diff(frames, axis=-1) == 5):
                raise RuntimeError('Padded or discontinuous label chunks are forbidden')
            if not all(np.all(labels['selected_demo'][i:i + 8] == source) for i in starts):
                raise RuntimeError('Chunk source changes across selection')
            if native and not np.array_equal(frames, context['source_frames']):
                raise RuntimeError('Native original-demo source clock differs')
            if not native and not np.array_equal(starts, context['official_il_chunk_start']):
                raise RuntimeError('Paired context uses different official chunks')
            episode = {key: labels[key].copy() for key in FIELDS}
        number = len(self.episodes)
        self.episodes.append((episode, geometry))
        self.indices.extend((number, int(start), i) for i, start in enumerate(starts))
        self.timeline_metadata.append(dict(source=source, role=role, chunks=len(starts),
                                           label_path=str(label_path), context_path=str(context_path)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        episode_index, start, context_index = self.indices[index]
        episode, geometry = self.episodes[episode_index]
        sample = {key: value[start:start + 8] for key, value in episode.items()}
        data = format_geometry_sample(sample, geometry[context_index])
        # Bookkeeping outside obs/action; the unchanged official loss ignores
        # this ID. Record actual matched batch order without model-input changes.
        data['sample_index'] = torch.tensor(index, dtype=torch.int64)
        return data

    def get_normalizer(self, **kwargs):
        if kwargs:
            raise ValueError('Preserve released normalization; fit only TRAIN geometry')
        if self._normalizer is None:
            geometry = np.concatenate([data.reshape(-1, CONTEXT_WIDTH) for _, data in self.episodes])
            self._normalizer = released_normalizer_with_geometry(self.parent_checkpoint, geometry)
        return copy.deepcopy(self._normalizer)

    def get_all_actions(self):
        return torch.stack([self[index]['action'] for index in range(len(self))])

    def get_validation_dataset(self):
        # Frozen evaluation is external and motion-disjoint. Never split nearby
        # chunks of these same TRAIN trajectories into a purported validation set.
        return BaseLowdimDataset()


class ActualBranchGeometryDataset(BaseLowdimDataset):
    """Actual paired futures from audited worlds, repeated for noise draws.

    Repetition supplies balanced batches, not new trajectories or independent
    examples. All statistics are restored from the preceding TRAIN-only fit.
    Original goal fields remain intact here; the explicit diagnostic encoder
    zeroes them after normalization in both training and evaluation.
    """

    def __init__(self, corpus, normalizer_state, repetitions=32, split=None, paired_supervision=False):
        self.corpus = Path(corpus)
        result = json.loads((self.corpus / 'RESULT.json').read_text())
        self.normalizer_state = Path(normalizer_state)
        self.repetitions = int(repetitions)
        if split is None:
            if not result['execution_completed'] or not all(result['checks'].values()):
                raise RuntimeError('Require the complete actual common-world branch audit')
            if self.repetitions != 32:
                raise RuntimeError('Preserve32 repetitions of each of two real branches')
            path = self.corpus / 'BRANCH_SAMPLES.npz'
            samples = [dict(source=result['per_arm'][arm]['source'],
                            phase=result['per_arm'][arm]['actual_control_frames'][0])
                       for arm in ('original', 'alternate')]
        else:
            if split not in ('train', 'heldout_phase') or not result['all_phase_data_passed'] or not result['physical_plots_inspected']:
                raise RuntimeError('Require complete admitted and inspected actual phase data')
            plan = json.loads((self.corpus / 'NEXT_MATCHED_PROTOCOL.json').read_text())
            if self.repetitions != plan['repetitions_per_real_training_example'] or not all(plan['original_source_future_clocks_disjoint'].values()):
                raise RuntimeError('Preserve the declared actual phase split and repetitions')
            info = plan['data'][split]
            path, samples = Path(info['arrays']), info['samples']
        with np.load(path) as archive:
            self.arrays = {key: archive[key].copy() for key in archive.files}
        self.real_case_count = len(samples)
        self.paired_supervision = bool(paired_supervision)
        if self.paired_supervision:
            expected_cases = 14
            if split == 'train' and self.corpus.name == 'generator_actual_branch_coverage_gaps2':
                if plan['run_name'] != 'matched_generator_branch_latent_replay01_gaps512' or plan['data']['train']['phases'] != [158,178,197,221,245,261,277,298,318]:
                    raise RuntimeError('Require the separately declared two-gap paired experiment')
                expected_cases = 18
            if split != 'train' or self.real_case_count != expected_cases:
                raise RuntimeError('Paired objective must use every declared actual TRAIN case')
            for i in range(0, self.real_case_count, 2):
                if samples[i]['phase'] != samples[i+1]['phase'] or any(not np.array_equal(self.arrays[k][i], self.arrays[k][i+1]) for k in ('obj_pos_b','obj_ori_b','last_action')):
                    raise RuntimeError('Paired supervision lacks exact causal common inputs')
        if self.arrays['future_command_target'].shape != (self.real_case_count, 8, 36) or not all(np.isfinite(v).all() for v in self.arrays.values()):
            raise RuntimeError('Require all finite full8x36 actual branch labels')
        if split is not None:
            for i, row in enumerate(samples):
                with np.load(self.corpus / f"switch_{row['phase']}/branch_samples/BRANCH_SAMPLES.npz") as original:
                    branch = 0 if row['arm'] == 'original' else 1
                    if set(original.files) != set(self.arrays) or not all(np.array_equal(self.arrays[k][i], original[k][branch]) for k in original.files):
                        raise RuntimeError('Combined phase data differs from actual original branch arrays')
            clocks = {name: {source: {f for s in info['samples'] if s['source'] == source for f in s['source_frames']}
                             for source in (96, 90)} for name, info in plan['data'].items()}
            if any(clocks['train'][s] & clocks['heldout_phase'][s] for s in (96, 90)):
                raise RuntimeError('TRAIN and heldout future source-frame labels overlap')
        self.timeline_metadata = [dict(source=row['source'], real_branch_examples=1, repetitions=self.repetitions,
                                        shared_world_control_frame=row['phase']) for row in samples]

    def __len__(self):
        return self.real_case_count * self.repetitions

    def __getitem__(self, index):
        branch = int(index) % self.real_case_count
        obs = {key: torch.from_numpy(self.arrays[key][branch].astype(np.float32).copy())
               for key in (*FIELDS[:6], 'last_action', CONTEXT_KEY)}
        sample = dict(obs=obs, action=torch.from_numpy(self.arrays['future_command_target'][branch].copy()),
                    sample_index=torch.tensor(index, dtype=torch.int64))
        if self.paired_supervision:
            other = branch ^ 1
            sample['paired_action'] = torch.from_numpy(self.arrays['future_command_target'][other].copy())
            sample['paired_geometry'] = torch.from_numpy(self.arrays[CONTEXT_KEY][other].astype(np.float32).copy())
        return sample

    def get_validation_dataset(self):
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs):
        if kwargs:
            raise ValueError('Restore existing TRAIN statistics without refitting')
        payload = torch.load(self.normalizer_state, map_location='cpu')['model']
        prefix = 'normalizer.'
        state = {key[len(prefix):]: value for key, value in payload.items() if key.startswith(prefix)}
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(state, strict=True)
        normalizer.requires_grad_(False)
        if not all(torch.equal(value, normalizer.state_dict()[key]) for key, value in state.items()):
            raise RuntimeError('Existing normalizer restoration differs')
        return normalizer


class ActualBranchReplayDataset(ActualBranchGeometryDataset):
    """Full actual native TRAIN coverage plus explicitly repeated fitted branches.

    All ordinary96/90 source chunks are excluded to preserve later branch labels.
    Uses ordinary official shuffled batches; no new sampler or model is created.
    """

    def __init__(self, corpus, normalizer_state, native_corpus, parent_checkpoint, paired_train):
        declared = json.loads((Path(corpus) / 'NEXT_MATCHED_PROTOCOL.json').read_text())
        self.branch = ActualBranchGeometryDataset(corpus, normalizer_state,
            repetitions=declared['repetitions_per_real_training_example'], split='train')
        self.normalizer_state = Path(normalizer_state)
        self.native = ActualDemoGeometryDataset(native_corpus, parent_checkpoint, paired_train)
        self.observation_keys = tuple(self.native[0]['obs'])
        extra = set(self.branch[0]['obs']) - set(self.observation_keys)
        if extra != {'joint_pos', 'project_gravity'} or not set(self.observation_keys).issubset(self.branch[0]['obs']):
            raise RuntimeError('Unexpected native/branch observation schema difference')
        self.native_indices = [i for i, (episode, _, _) in enumerate(self.native.indices)
                               if self.native.timeline_metadata[episode]['source'] not in (96, 90)]
        self.native_count = len(self.native_indices)
        self.branch_repetitions = self.native_count // (3 * self.branch.real_case_count)
        if self.branch_repetitions < 1:
            raise RuntimeError('Require broader actual native coverage')
        used_episodes = {self.native.indices[i][0] for i in self.native_indices}
        self.timeline_metadata = [dict(row) for i, row in enumerate(self.native.timeline_metadata) if i in used_episodes]
        self.timeline_metadata.extend(dict(row, repetitions=self.branch_repetitions,
                                          role='actual_early_branch' if row['shared_world_control_frame'] < 200 else 'actual_late_branch')
                                      for row in self.branch.timeline_metadata)
        self.composition = dict(native_real_chunks=self.native_count, native_sources=sorted({row['source'] for i, row in enumerate(self.native.timeline_metadata) if i in used_episodes}),
            excluded_native_sources=[96, 90], excluded_original_chunks=len(self.native) - self.native_count,
            real_branch_examples=self.branch.real_case_count,
            branch_phases=list(dict.fromkeys(row['shared_world_control_frame'] for row in self.branch.timeline_metadata)),
            branch_repetitions=self.branch_repetitions,
            total_rows_per_epoch=self.native_count + self.branch.real_case_count * self.branch_repetitions,
            branch_row_fraction=self.branch.real_case_count * self.branch_repetitions / (self.native_count + self.branch.real_case_count * self.branch_repetitions))

    def __len__(self):
        return self.native_count + self.branch.real_case_count * self.branch_repetitions

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample = self.native[self.native_indices[index]] if index < self.native_count else self.branch[(index - self.native_count) % self.branch.real_case_count]
        # The released use_last_action=True formatter omits these two unused
        # fields. Branch archives retain them for audit; the full encoder reads
        # last_action instead. Emit exactly the official formatter schema for
        # both kinds of row, independent of which one starts a collated batch.
        sample['obs'] = {key: sample['obs'][key] for key in self.observation_keys}
        sample['sample_index'] = torch.tensor(index, dtype=torch.int64)
        return sample


class FrozenGeometryEvaluationDataset(ActualDemoGeometryDataset):
    """Existing complete actual validation episodes; never fit their statistics."""

    def __init__(self, corpus):
        self.corpus = Path(corpus)
        result = json.loads((self.corpus / 'RESULT.json').read_text())
        if not result['execution_completed'] or result['split'] != 'validation' or result['statistics_fitted']:
            raise RuntimeError('Require frozen validation data without fitted statistics')
        if result['sources'] != [8, 28, 38, 58, 68, 78, 88, 98]:
            raise RuntimeError('Fixed native validation sources changed')
        self.episodes, self.indices, self.timeline_metadata, self.alternate_contexts = [], [], [], []
        for source in result['sources']:
            context = self.corpus / f'original_demo_geometry_context/source_{source:03d}.npz'
            self._add_episode(self.corpus / f'official_il_data/source_{source:03d}_IL.npz',
                              context, source, 'frozen_native_validation', native=True)
            with np.load(context) as arrays:
                other = arrays['alternate_demo_geometry'].copy()
                if other.shape != self.episodes[-1][1].shape or not np.isfinite(other).all():
                    raise RuntimeError('Wrong-demo context dimensions differ')
                self.alternate_contexts.append(other)
        if len(self.indices) != result['total_chunks']:
            raise RuntimeError('Incomplete frozen validation dataset')

    def with_wrong_context(self, index):
        sample = self[index]
        episode, _, context = self.indices[index]
        sample['obs'][CONTEXT_KEY] = torch.from_numpy(
            self.alternate_contexts[episode][context].reshape(1, CONTEXT_WIDTH).copy())
        return sample

    def get_normalizer(self, **kwargs):
        raise RuntimeError('Use the frozen trained model normalizer; validation must never fit statistics')
