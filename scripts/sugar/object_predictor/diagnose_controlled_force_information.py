"""Fixed16 saved-only H32 force readback; no model, new physics or calibration.

Reuses the full Utonia observation encoder and observed_physics_targets exactly.
Signed support-equivalent mass is not object mass outside stable free support.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .overfit_data import (FIXED_EPISODES, fixed_protocol, uniform_history_indices,
    observation_window, mass_supervision, observed_physics_targets)
from .conditional_overfit_data import PROFILE, check_collection_scope
from .hand_surface_normals import query_observation_normals
from .surface_data import SURFACE_KEYS, select_surface_frames, encode_surface_observations

FRAME = 2381
GRAVITY = 9.81


def support_from_encoded(encoded):
    """Observation-only signed force projection; do not clip or calibrate."""
    if len(encoded['feat']) != 32:
        raise ValueError('Require the complete original H32 model input')
    physics = [observed_physics_targets(frame) for frame in encoded['feat']]
    return {
        'encoded_shear_only_kg': np.array([p['shear_support_up_n'][0]/GRAVITY for p in physics]),
        'encoded_normal_plus_shear_kg': np.array([
            p['combined_support_up_approx_n'][0]/GRAVITY for p in physics]),
    }


def run(root, output):
    declared, collection = check_collection_scope(root, PROFILE)
    records = collection['records']
    if tuple(r['episode'] for r in records) != FIXED_EPISODES:
        raise ValueError('Wait for all sixteen declared attempts; never report a partial denominator')
    if declared['overfit_data'] != fixed_protocol():
        raise ValueError('Keep the actual training input/clock/label protocol')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    protocol = fixed_protocol()
    for case in records:
        episode = case['episode']
        row = dict(episode=episode, frame=FRAME, complete=False,
                   original_physical_passed=case.get('controller_passed') is True,
                   original_physical_checks=case.get('controller_checks', {}),
                   collector_late_mass_available=case.get('late_mass_available'))
        try:
            source = Path(case['source'])
            with np.load(source/f'episode_{episode}.npz', allow_pickle=False) as z:
                arrays = {k:z[k] for k in z.files}
            indices = uniform_history_indices(arrays['timestamp_s'], FRAME, 32, .02)
            if not np.array_equal(indices, np.arange(2350, 2382)):
                raise ValueError('Unexpected fixed late H32 indices')
            observations = observation_window(arrays, indices)
            site, contact, _ = query_observation_normals(observations)
            observations.update(hand_site_surface_normals_w=site,
                                hand_contact_surface_normals_w=contact)
            with np.load(source/'contact_surface.npz', allow_pickle=False) as z:
                field = {k:z[k] for k in ('offset', *SURFACE_KEYS)}
            encoded = encode_surface_observations(observations, select_surface_frames(field, indices),
                'geometry_contact_force', time_scale_s=protocol['time_scale_s'])
            signals = support_from_encoded(encoded)
            # All GT/qualification is evaluation-only, after observation calculation.
            mass = np.asarray(arrays['object_mass_kg'])[indices].astype(float)
            if not np.isfinite(mass).all() or np.any(mass <= 0) or np.ptp(mass) > 1e-8:
                raise ValueError('Expected finite constant positive true mass in a case')
            label = mass_supervision(arrays, indices, protocol['mass_criteria'])
            qualified = bool(label['mass_available'])
            values = {}
            for name, signal in signals.items():
                if not np.isfinite(signal).all():
                    raise ValueError('Nonfinite encoded support')
                values[name] = dict(endpoint_kg=float(signal[-1]), mean_h32_kg=float(signal.mean()),
                    endpoint_relative_error_pct=float(abs(signal[-1]/mass[-1]-1)*100),
                    mean_h32_estimate_relative_error_pct=float(abs(signal.mean()/mass[-1]-1)*100),
                    mean_h32_pointwise_relative_error_pct=float(np.mean(abs(signal/mass-1))*100),
                    negative_frames=int((signal < 0).sum()))
            row.update(complete=True, true_mass_kg=float(mass[-1]),
                first_frame=int(indices[0]), history_frames=len(indices),
                start_time_s=float(observations['timestamp_s'][0]),
                end_time_s=float(observations['timestamp_s'][-1]),
                fresh_stable_supported=qualified, mass_status=int(label['mass_status']),
                support_diagnostics=label['evaluation_only'],
                collector_late_label_matches=qualified == case.get('late_mass_available'),
                signals=values)
            np.savez_compressed(output/f'episode_{episode}.npz', frame=indices,
                timestamp_s=observations['timestamp_s'], true_mass_kg=mass, **signals)
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
        print('CONTROLLED_FORCE_INFORMATION', json.dumps(row), flush=True)
    qualified = [r for r in rows if r['complete'] and r['original_physical_passed']
                 and r['fresh_stable_supported'] and r['collector_late_label_matches']]
    aggregate = {name: {key:float(np.mean([r['signals'][name][key] for r in qualified]))
        for key in ('endpoint_relative_error_pct', 'mean_h32_estimate_relative_error_pct',
                    'mean_h32_pointwise_relative_error_pct')}
        for name in ('encoded_shear_only_kg', 'encoded_normal_plus_shear_kg')} if qualified else {}
    result = dict(complete=all(r['complete'] for r in rows), fixed_cases=16, fixed_frame=FRAME,
        history_frames_per_case=32, declared_saved_clocks=512, cases=rows,
        original_collection_qualification_passed=collection.get('qualification_passed'),
        physically_passed_and_stable_cases=len(qualified), conditional_equal_case_errors_pct=aggregate,
        scope='Saved observation-mechanics diagnostic, not a model or an information-theoretic upper bound.',
        limitations='Fup/g is signed support-equivalent mass. CAD/fitted input normals approximate contact directions. Ground support, missed contacts or acceleration invalidate a direct mass interpretation. No GT correction, fitted scale, clipping, error-based selection, model forward or calibration.',
        new_physics_controls=0, model_forwards=0, optimizer_updates=0)
    (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Fixed late H32 encoded force information', '',
        'All sixteen attempts retained. Fixed frame2381, H32 frames2350–2381 (47.02–47.64 s). '
        'Same actual20-feature training encoder. GT only evaluates mass and original support labels; '
        '0 model/physics/updates. No scale calibration or clipping.', '',
        '| Episode | Physical | Stable H32 | True kg | Shear endpoint kg / error% | Normal+shear endpoint kg / error% | H32 mean normal+shear kg / error% |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        if not row['complete']:
            lines.append(f"| {row['episode']} | {row['original_physical_passed']} | ERROR | — | — | — | {row['error']} |")
            continue
        shear = row['signals']['encoded_shear_only_kg']; combined = row['signals']['encoded_normal_plus_shear_kg']
        lines.append(f"| {row['episode']} | {row['original_physical_passed']} | {row['fresh_stable_supported']} | {row['true_mass_kg']:.4f} | {shear['endpoint_kg']:.4f} / {shear['endpoint_relative_error_pct']:.2f} | {combined['endpoint_kg']:.4f} / {combined['endpoint_relative_error_pct']:.2f} | {combined['mean_h32_kg']:.4f} / {combined['mean_h32_estimate_relative_error_pct']:.2f} |")
    lines += ['', f'Original physical + fresh stable support + matching label: {len(qualified)}/16. '
        'Only this explicitly qualified subset permits the conditional mass-error interpretation. '
        'All raw rows remain above and in RESULT.json. Endpoint errors compare to the model frame2381; '
        'the separate H32 mean is a fixed observation statistic, not model training.', '',
        'This diagnostic tests accessible encoded force information; it neither certifies a deployable mass estimator nor bounds every possible predictor.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.root, args.output))
