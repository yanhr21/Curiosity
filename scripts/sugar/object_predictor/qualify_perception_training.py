"""Verify cached observations, matched sampling and the mass loss on real TRAIN data."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import OBSERVATION_KEYS, history_indices, encode_observations
from .surface_data import encode_surface_observations
from .perception_dataset import PerceptionDataset, MatchedGeometrySampler, collate_perception
from .train import supervised_components
from .model import loss_components


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    out = Path(args.output); out.mkdir(exist_ok=False)
    root = Path(args.data); status = json.loads((root / 'COLLECTION_RESULT.json').read_text())
    # TRAIN-only checks never treat the incomplete corpus as formally trainable.
    all_groups = getattr(args, 'all_train_groups', False)
    records = [r for r in status['records'] if r['split'] == 'train' and
               (all_groups or r['geometry_group'] == 0)]
    if all_groups:
        expected = [c for c in json.loads((root / 'PROTOCOL.json').read_text())['configurations']
                    if c['split'] == 'train']
        if len(records) != 32 or [r['episode'] for r in records] != [c['episode'] for c in expected]:
            raise ValueError('All 32 original TRAIN cases in original order are required')
        if any(r['geometry_group'] != c['geometry_group'] for r, c in zip(records, expected)):
            raise ValueError('Original geometry groups differ')
        if {r['geometry_group'] for r in records} != set(range(8)):
            raise ValueError('Expected all eight original TRAIN groups')
    elif len(records) != 4:
        raise ValueError('The first complete four-setting TRAIN group is required')
    (out / 'COLLECTION_RESULT.json').write_text(json.dumps(dict(complete=False, records=records), indent=2))
    datasets = {representation: PerceptionDataset(out, 'geometry_contact_force', 'train',
        representation=representation, allow_incomplete_train=True) for representation in ('centroid', 'surface')}
    checks = {}; details = []; left = datasets['centroid']; right = datasets['surface']
    checks['identical_original_episode_clock_index'] = left.index == right.index and [m['episode'] for m, _ in left.episodes] == [m['episode'] for m, _ in right.episodes]
    for e in range(len(records)):
        for frame in (31, 756, 2381):
            index = right.index.index((e, frame))
            a, b = left[index], right[index]
            checks[f'{e}/{frame}/labels_identical'] = np.array_equal(a['target'], b['target']) and a['mass_label_weight'] == b['mass_label_weight']
            _, data = right.episodes[e]
            indices = history_indices(frame, 32, 'episode_uniform_recent')
            raw = {k: data[k][indices] for k in OBSERVATION_KEYS}
            surfaces = [right.surfaces[e][i] for i in indices]
            direct = encode_surface_observations(raw, surfaces, 'geometry_contact_force')
            cached_difference = max(float(np.max(abs(x-y))) for x, y in zip(b['feat'], direct['feat']))
            checks[f'{e}/{frame}/cached_surface_equals_raw'] = cached_difference == 0 and all(np.array_equal(x, y) for x, y in zip(b['grid_coord'], direct['grid_coord']))
            original = encode_observations(raw, 'geometry_contact_force', time_scale_s=150., normal_policy='hand_surface')
            checks[f'{e}/{frame}/centroid_is_original_encoder'] = np.array_equal(original['feat'], a['feat'])
            # Force-zero changes only the measured force attributes.
            right.force_gain = 0.; zero = right[index]; right.force_gain = 1.
            checks[f'{e}/{frame}/cached_force_zero_geometry_preserved'] = all(
                np.array_equal(x[:, [*range(10), *range(14, 20)]], y[:, [*range(10), *range(14, 20)]]) and not np.count_nonzero(y[:, 10:14])
                for x, y in zip(b['feat'], zero['feat']))
            details.append(dict(episode=right.episodes[e][0]['episode'], frame=frame, raw_cached_difference=cached_difference,
                                mass_label_weight=b['mass_label_weight'], surface_points=sum(map(len, b['coord']))))
    # Separately advance one sampler for the full declared 2000 clocks.
    sequences = {}
    for key, ds in datasets.items():
        sampler = MatchedGeometrySampler(ds, 310019); sequence = []
        while len(sequence) < 2000:
            sequence.extend(iter(sampler))
        sequences[key] = sequence[:2000]
    checks['all_2000_batch_draws_match'] = sequences['centroid'] == sequences['surface']
    checks['every_batch_four_settings_one_original_clock'] = all(
        len({right.index[i][0] for i in batch}) == 4 and len({right.index[i][1] for i in batch}) == 1
        for batch in sequences['surface'])
    if all_groups:
        groups = [right.episodes[right.index[batch[0]][0]][0]['geometry_group']
                  for batch in sequences['surface']]
        checks['all_eight_groups_sampled'] = set(groups) == set(range(8))
        checks['every_batch_one_geometry_group'] = all(len({
            right.episodes[right.index[i][0]][0]['geometry_group'] for i in batch}) == 1
            for batch in sequences['surface'])
    original_labels = [x.copy() for x in right.labels]
    right.labels = [np.zeros_like(x) for x in right.labels]
    blind = MatchedGeometrySampler(right, 310019); new_sequence = []
    while len(new_sequence) < 2000:
        new_sequence.extend(iter(blind))
    checks['mass_mask_does_not_change_sampler'] = new_sequence[:2000] == sequences['surface']
    right.labels = original_labels
    targets = torch.from_numpy(np.stack([right.target(right.index.index((e, 2381))) for e in range(4)]))
    torch.manual_seed(310016); prediction = torch.randn_like(targets, requires_grad=True)
    original = loss_components(prediction, targets); unmasked = supervised_components(prediction, targets)
    checks['unmasked_loss_is_original'] = all(torch.equal(original[k], unmasked[k]) for k in original)
    mask = torch.tensor([1., 0., 1., 0.]); corrupted = targets.clone(); corrupted[mask == 0, 12] += 5.
    first = supervised_components(prediction, targets, mask)['mass']
    second = supervised_components(prediction, corrupted, mask)['mass']
    grad_first = torch.autograd.grad(first, prediction, retain_graph=True)[0]
    grad_second = torch.autograd.grad(second, prediction, retain_graph=True)[0]
    checks['masked_mass_targets_do_not_change_loss_or_gradient'] = torch.equal(first, second) and torch.equal(grad_first, grad_second) and not bool(torch.count_nonzero(grad_first[mask == 0, 12]))
    all_zero = supervised_components(prediction, targets, torch.zeros(4))
    checks['no_mass_labels_keeps_other_tasks_and_finite_zero_mass'] = bool(all_zero['mass'] == 0) and all(torch.equal(all_zero[k], original[k]) for k in ('position', 'rotation', 'size'))
    # Real four-setting batch for subsequent complete-model training-path check.
    for representation, ds in datasets.items():
        rows = [ds[ds.index.index((e, 2381))] for e in range(4)]
        batch = collate_perception(rows)
        np.savez_compressed(out / f'{representation}_batch.npz', **{k: v.numpy() for k, v in batch.items()})
    report = dict(passed=all(checks.values()), checks=checks, details=details, optimizer_updates=0,
                  scope=('All32 original TRAIN cases/eight groups; cache/loss/sampler implementation check only, no coverage-gate override or model accuracy claim.'
                         if all_groups else 'First complete TRAIN geometry group only; cache/loss/sampler implementation check, not model accuracy or full corpus qualification.'))
    if all_groups:
        report['group_draw_counts'] = {str(g): groups.count(g) for g in range(8)}
        report['batch_clock_sequence'] = [[dict(episode=right.episodes[right.index[i][0]][0]['episode'],
                                                frame=right.index[i][1]) for i in batch]
                                        for batch in sequences['surface']]
    (out / 'TRAINING_PATH_REPORT.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--data', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--all-train-groups', action='store_true', help='Require and verify all32 original TRAIN cases')
    main(parser.parse_args())
