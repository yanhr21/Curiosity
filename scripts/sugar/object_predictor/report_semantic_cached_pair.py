"""Saved-only comparison of the completed fixed semantic Utonia pair.

No model, physics, checkpoint selection, or threshold changes. All input clocks,
targets, masks and original/new fit roles must match before plotting.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

ARMS = ('no_summary', 'observed_summary')
IDENTITY = ('episode', 'frame', 'timestamp_s', 'target', 'force_target_n',
            'mass_available', 'mass_status', 'contact_present',
            'state_precision_eligible', 'state_contact_history_frames',
            'state_evidence_status', 'original_fit_item', 'semantic_role')


def digest(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def summarize(d, select=None):
    select = np.ones(len(d['episode']), bool) if select is None else select
    candidate = select & (d['state_precision_eligible'] > .5)
    available = select & (d['mass_available'] > .5)
    unavailable = select & ~available
    positive_recall = float((d['availability_probability'][available] >= .5).mean()) if available.any() else None
    negative_recall = float((d['availability_probability'][unavailable] < .5).mean()) if unavailable.any() else None
    def stats(key, mask):
        v = d[key][mask]
        return None if not len(v) else dict(mean=float(v.mean()), p95=float(np.percentile(v, 95)),
                                           maximum=float(v.max()), count=len(v))
    return dict(rows=int(select.sum()), state_candidates=int(candidate.sum()),
                prior_unknown=int((select & ~candidate).sum()),
                center_cm=stats('center_cm', candidate), mesh_nn_cm=stats('mesh_nn_cm', candidate),
                size_relative=stats('size_mean_relative', candidate),
                available_mass_relative=stats('mass_relative', available),
                all_rows_center_cm=stats('center_cm', select), all_rows_mass_relative=stats('mass_relative', select),
                negative_availability_rows=int(unavailable.sum()),
                availability_sensitivity=positive_recall, availability_specificity=negative_recall,
                availability_balanced_accuracy=(positive_recall + negative_recall) / 2
                if positive_recall is not None and negative_recall is not None else None,
                false_confident_fraction=float((d['availability_probability'][unavailable] >= .5).mean())
                if unavailable.any() else None)


def report(root):
    root = Path(root)
    if not (root / 'RESULT.json').is_file():
        raise RuntimeError('Both fixed arms must complete before comparison')
    pair = json.loads((root / 'RESULT.json').read_text())
    assert pair['execution_complete'] is True
    from .semantic_state_cached import verify_artifacts
    output = root / 'saved_pair_analysis'
    if output.exists():
        raise FileExistsError('Preserve previous comparison; no overwrite')
    sources = [digest(root / 'RESULT.json'), digest(root / 'PROTOCOL.json'), digest(Path(__file__))]
    data, reports, training = {}, {}, {}
    for arm in ARMS:
        assert json.loads((root / arm / 'RESULT.json').read_text())['execution_complete'] is True
        verify_artifacts(root / arm)
        sources.extend((digest(root / arm / 'RESULT.json'), digest(root / arm / 'ARTIFACTS.json')))
        training[arm] = [json.loads(line) for line in (root / arm / 'train.jsonl').read_text().splitlines()]
        assert [x['new_optimizer_update'] for x in training[arm]] == list(range(1, 21))
        sources.append(digest(root / arm / 'train.jsonl'))
        for role, before, after in (('fit', 'initial_fit', 'fit_02120'),
                                    ('development', 'initial_same_trajectory_interpolation', 'same_trajectory_interpolation')):
            for stage, stem in (('before', before), ('after', after)):
                path = root / arm / (stem + '.npz')
                with np.load(path, allow_pickle=False) as z:
                    d = {k: z[k].copy() for k in z.files}
                assert len(d['episode']) == (464 if role == 'fit' else 1440)
                assert all(np.isfinite(v).all() for v in d.values())
                # Recompute state arithmetic in its original float32 precision.
                center = np.linalg.norm(d['prediction'][:, :3] - d['target'][:, :3], axis=1) * 100
                mass = np.abs(np.expm1(d['prediction'][:, 12] - d['target'][:, 12]))
                assert np.allclose(center, d['center_cm'], rtol=2e-6, atol=2e-6)
                assert np.allclose(mass, d['mass_relative'], rtol=2e-6, atol=2e-7)
                data[arm, role, stage] = d
                reports[arm, role, stage] = json.loads(path.with_suffix('.json').read_text())
                sources.extend((digest(path), digest(path.with_suffix('.json'))))
    for role in ('fit', 'development'):
        reference = data[ARMS[0], role, 'before']
        for arm in ARMS:
            for stage in ('before', 'after'):
                for key in IDENTITY:
                    assert np.array_equal(reference[key], data[arm, role, stage][key]), (arm, role, stage, key)
    for left, right in zip(training[ARMS[0]], training[ARMS[1]], strict=True):
        assert left['learning_rates'] == right['learning_rates']
        assert left['global_denominators'] == right['global_denominators']
        for a, b in zip(left['microbatches'], right['microbatches'], strict=True):
            for key in ('indices', 'samples', 'state_candidates', 'available_mass', 'availability_positive',
                        'availability_negative', 'current_contact_total'):
                assert a[key] == b[key], key
    result = dict(integrity_passed=True, scope='Same trajectories, development already seen; no generalization claim.',
                  masks_and_targets_exact=True, all_20_training_clocks_and_batches_exact=True,
                  initial_pair_max_abs={}, summaries={}, sources=sources)
    for role in ('fit', 'development'):
        result['initial_pair_max_abs'][role] = {
            key: float(np.max(np.abs(data[ARMS[0], role, 'before'][key] - data[ARMS[1], role, 'before'][key])))
            for key in ('prediction', 'force_prediction_n', 'availability_probability', 'contact_probability')}
        result['summaries'][role] = {}
        for arm in ARMS:
            result['summaries'][role][arm] = {}
            for stage in ('before', 'after'):
                d = data[arm, role, stage]
                entry = summarize(d)
                entry['semantic_acceptance_passed'] = reports[arm, role, stage]['semantic_acceptance_passed']
                entry['per_episode'] = {str(e): summarize(d, d['episode'] == e) for e in np.unique(d['episode'])}
                if role == 'fit':
                    entry['original80'] = summarize(d, d['original_fit_item'] == 1)
                    entry['added384'] = summarize(d, d['original_fit_item'] == 0)
                result['summaries'][role][arm][stage] = entry
    output.mkdir()
    (output / 'COMPARISON.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for arm in ARMS:
        axes[0].plot(range(1, 21), [x['loss'] for x in training[arm]], label=arm)
        axes[1].plot(range(1, 21), [x['grad_norm_before_clip'] for x in training[arm]], label=arm)
    axes[0].set(title='Fixed full464 training objective', xlabel='Full-data Adam update', ylabel='Weighted objective')
    axes[1].set(title='Global gradient before clipping', xlabel='Full-data Adam update', ylabel='L2 norm')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output / 'training.png', dpi=160); plt.close(fig)
    lines = ['# Fixed semantic Utonia pair: saved comparison', '',
             'Both arms keep the original tactile inputs; only the explicit force/height history differs.',
             'Development is previously viewed same-trajectory data. No unseen-object or policy claim.', '',
             '|Role|Arm/stage|State rows|Center cm|Mesh cm|Size %|Mass rows|Mass %|False-confident %|Semantic pass|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for role, arms in result['summaries'].items():
        for arm, stages in arms.items():
            for stage, s in stages.items():
                m = s['available_mass_relative']
                lines.append(f"|{role}|{arm}/{stage}|{s['state_candidates']}|{s['center_cm']['mean']:.3f}|"
                             f"{s['mesh_nn_cm']['mean']:.3f}|{100*s['size_relative']['mean']:.2f}|{m['count']}|"
                             f"{100*m['mean']:.2f}|{100*s['false_confident_fraction']:.2f}|{s['semantic_acceptance_passed']}|")
    lines += ['', 'All-row center/mass errors, per-episode denominators, original80/added384 subsets, source hashes,',
              'and actual initial cross-arm numerical differences are in COMPARISON.json.',
              'All-row mesh/size errors remain in the source NPZ and original evaluation reports.',
              'Mask equality and center/mass arithmetic were independently checked from saved arrays.',
              'Mesh errors are the original full-mesh evaluation values, not recomputed in this report.',
              'Training losses are normalized objective terms, not direct physical error units.',
              'No-contact errors remain recorded even where precision acceptance is not applicable.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(integrity_passed=True, output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    report(parser.parse_args().root)
