"""Matched full-model endpoints and batch1 perception outcomes, retaining failures."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .report_grip_transfer import errors
from .run_surface_pipeline import ARMS


def read(path):
    with np.load(path) as values:
        return {k: values[k] for k in values.files}


def summarize(values):
    errors_by_frame = errors(values['prediction'], values['target'])
    result = {}
    for group, mask in [('all', np.ones(len(values['frame']), bool)), ('airborne_hold', values['mass_label_weight'] > 0)]:
        episodes = np.unique(values['episode']); per_episode = {}
        for episode in episodes:
            selected = mask & (values['episode'] == episode)
            per_episode[str(episode)] = dict(windows=int(selected.sum()), **{
                key: float(value[selected].mean()) if selected.any() else None for key, value in errors_by_frame.items()})
        means = {key: float(np.mean([row[key] for row in per_episode.values() if row[key] is not None]))
                 if mask.any() else None for key in errors_by_frame}
        result[group] = dict(equal_episode=means, per_episode=per_episode,
                             episodes_covered=sum(row['windows'] > 0 for row in per_episode.values()), total_episodes=len(episodes))
    return result


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use retained compute step')
    root = Path(args.root); protocol = json.loads((root / 'PROTOCOL.json').read_text())
    summaries = {}; checks = {}; curves = {}; reference = None; initial = None; draws = None
    for arm, _, _ in ARMS:
        endpoint = torch.load(root / arm / 'model.pt', map_location='cpu', weights_only=False)
        p = endpoint['protocol']; bindings = p['optimizer_parameter_bindings']; states = endpoint['optimizer']['state']
        checks[arm + '/full2000_model_and_optimizer'] = endpoint['step'] == 2000 and p['total_parameters'] == 138407503 and sum(b['numel'] for b in bindings) == 138407503
        checks[arm + '/optimizer_bindings_unique'] = len({b['id'] for b in bindings}) == len(bindings)
        missing = [b['name'] for b in bindings if b['id'] not in states]
        checks[arm + '/only_unused_official_mask_token_has_no_moments'] = missing == ['backbone.embedding.mask_token']
        checks[arm + '/every_used_moment_finite_and_clock2000'] = all(
            float(states[b['id']]['step']) == 2000 and
            all(states[b['id']][k].shape == endpoint['model'][b['name']].shape and bool(torch.isfinite(states[b['id']][k]).all())
                for k in ('exp_avg', 'exp_avg_sq')) for b in bindings if b['id'] in states)
        checks[arm + '/all_model_values_finite'] = all(bool(torch.isfinite(v).all()) for v in endpoint['model'].values())
        if initial is None:
            initial = p['initial_model_sha256']
        checks[arm + '/same_complete_initial_model'] = p['initial_model_sha256'] == initial
        lines = [json.loads(line) for line in (root / arm / 'train.jsonl').read_text().splitlines()]
        actual = [(r['step'], r['episodes'], r['frames'], r['mass_label_weights']) for r in lines]
        if draws is None:
            draws = actual
        checks[arm + '/all2000_actual_batches_and_labels_match'] = actual == draws and [r['step'] for r in lines] == list(range(1, 2001))
        curves[arm] = lines
        values = read(root / 'batch1' / arm / 'predictions.npz')
        checks[arm + '/finite_test_prediction_and_target'] = bool(np.isfinite(values['prediction']).all() and np.isfinite(values['target']).all())
        if reference is None:
            reference = {k: values[k] for k in ('target', 'episode', 'frame', 'mass_label_weight')}
        checks[arm + '/same_test_targets_clocks_and_masks'] = all(np.array_equal(values[k], v) for k, v in reference.items())
        summaries[arm] = summarize(values)
        zero = read(root / 'batch1' / arm / 'force_zero_predictions.npz')
        checks[arm + '/force_zero_same_test_clocks'] = all(np.array_equal(zero[k], v) for k, v in reference.items())
        summaries[arm]['force_zero'] = summarize(zero)
        del endpoint, states
    target = summaries[ARMS[-1][0]]['airborne_hold']; baseline = summaries[ARMS[0][0]]['airborne_hold']
    thresholds = protocol['evaluation']['state_thresholds']
    acceptance = {'complete_all8_test_coverage': target['total_episodes'] == 8 and all(r['windows'] >= 10 for r in target['per_episode'].values())}
    for key, limit in thresholds.items():
        value = target['equal_episode'][key]
        acceptance[key + '_threshold'] = value is not None and value <= limit
    mass = target['equal_episode']['mass_pct']; base_mass = baseline['equal_episode']['mass_pct']
    acceptance['mass_improves_at_least20pct'] = mass is not None and base_mass is not None and mass <= .8 * base_mass
    for key in ('center_cm', 'rotation_deg', 'size_pct'):
        value = target['equal_episode'][key]; base_value = baseline['equal_episode'][key]
        acceptance[key + '_nonregression'] = value is not None and base_value is not None and value <= base_value
    report = dict(integrity_passed=all(checks.values()), checks=checks, acceptance=acceptance,
                  scientific_acceptance=all(acceptance.values()), summaries=summaries,
                  scope='Known CarryBox-family state prediction in Newton; not arbitrary shape, native sensor transfer or policy benefit.')
    (root / 'COMPARISON.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['Centroid', 'Surface', 'Surface + mass guard']; colors = ['#667b8c', '#008e9b', '#e38d32']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, key in zip(axes, thresholds):
        values = [summaries[a]['airborne_hold']['equal_episode'][key] for a, _, _ in ARMS]
        ax.bar(range(3), [np.nan if x is None else x for x in values], color=colors)
        ax.axhline(thresholds[key], color='firebrick', ls='--', label='Declared threshold')
        ax.set_xticks(range(3), names, rotation=20, ha='right'); ax.set_title(key); ax.legend(fontsize=8)
    fig.suptitle('Batch1 TEST airborne hold: equal episode errors; coverage reported separately')
    fig.tight_layout(); fig.savefig(root / 'primary_comparison.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, key in zip(axes, ('position', 'rotation', 'size', 'mass')):
        for (arm, _, _), name, color in zip(ARMS, names, colors):
            values = np.array([r[key] for r in curves[arm]])
            ax.plot(np.arange(50, 2001), np.convolve(values, np.ones(50) / 50, mode='valid'), label=name, color=color)
        ax.set_title(key); ax.set_xlabel('Optimizer update'); ax.set_yscale('symlog', linthresh=1e-5); ax.legend(fontsize=8)
    fig.suptitle('Training diagnostics (50-step mean); mass supervision differs across arms')
    fig.tight_layout(); fig.savefig(root / 'learning_curves.png', dpi=150); plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != 'summaries'}), flush=True)
    if not report['integrity_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); main(parser.parse_args())
