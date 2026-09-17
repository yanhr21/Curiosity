"""Explicit fixed three-probe counterfactuals; original physical/chart gates.

The default study changes only force gain. The named angular-stiffness study
compares against its corrected report and changes only angular Kp. Neither
study reinterprets failures or authorizes shape training.
"""
from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path

from .fixture_touch_scene import FixtureDrive

PAIRS = (('18704', 0), ('15737', 9), ('13266', 20))
STUDIES = ('fixture_settling_pilot_v1', 'fixture_angular_stiffness_pilot_v1')


def protocol(study='fixture_settling_pilot_v1'):
    if study not in STUDIES:
        raise ValueError('Unknown fixed fixture pilot study: '+str(study))
    result = dict(study='fixture_settling_pilot_v1',
        objects=[p[0] for p in PAIRS], directions=[p[1] for p in PAIRS],
        attempt_order=[dict(object_id=i, direction_index=d) for i, d in PAIRS],
        controls_per_attempt=600, dt=.02, substeps=8, sample_frame=499,
        fixture_drive=asdict(replace(FixtureDrive(), force_gain_m_ns=.003)),
        baseline_root='experiments/object_predictor_v1/overfit_repair_v1/sugar_shape_fixed40',
        scope='Three fixed actual probes: regression, late underload and oscillating-load failure; gain-only counterfactual, no shape learning',
        rationale='15737/d9 is late underload, while 13266/d20 spans 0.452–2.897N and may become less stable with higher gain. Tripling gain tests the underload hypothesis and the oscillation risk with the same .004m/s limit and fixed 10s sample; it is not an established repair. No threshold or observation relabeling.')
    if study == 'fixture_angular_stiffness_pilot_v1':
        baseline = 'experiments/object_predictor_v1/overfit_repair_v1/fixture_settling_pilot_v1'
        result.update(study=study,
            fixture_drive=asdict(replace(FixtureDrive(), force_gain_m_ns=.003, angular_kp_nm_rad=32.)),
            baseline_root=baseline, baseline_report=baseline+'/report_r1/RESULT.json',
            baseline_report_incident=baseline+'/REPORT_ENV_INCIDENT.json',
            scope='Three fixed actual probes; relative to gain .003 baseline, only angular stiffness 8 to 32 Nm/rad. Dynamic hand, original physical/chart gates, no shape learning.',
            rationale='13266/d20 retains a .26-.28s force/pose/contact cycle with gain tripled; observed angular return accompanies contact collapse. Increasing angular stiffness tests rotational compliance while preserving damping .3, effort limit2, all linear parameters, gain .003 and fixed clocks. Approximate known-CAD independent-axis damping remains near critical; not a contact stability guarantee.')
    return result


def validate_protocol(value):
    expected = protocol(value.get('study'))
    for key, required in expected.items():
        if value.get(key) != required:
            raise ValueError('Fixed settling pilot changed: '+key)
    return FixtureDrive(**expected['fixture_drive']), expected['attempt_order']


def read_baseline(declared):
    """Use the declared immediate baseline, including its preserved incident."""
    validate_protocol(declared)
    path = Path(declared.get('baseline_report', str(Path(declared['baseline_root'])/'RESULT.json')))
    data = json.loads(path.read_text())
    provenance = dict(report=str(path), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    incident = declared.get('baseline_report_incident')
    if incident is not None:
        incident_path = Path(incident)
        provenance.update(incident_path=str(incident_path),
            incident_sha256=hashlib.sha256(incident_path.read_bytes()).hexdigest(),
            incident=json.loads(incident_path.read_text()))
    return data, provenance


def report(root, report_output=None):
    from .official_active3d import read_template_obj
    from .shape_fixture_assets import EXPERIMENT
    from .report_fixture_sequences import report_attempt
    from .report_fixture_touch import QUALIFICATION_CRITERIA, write_json
    actual = json.loads((root/'PROTOCOL.json').read_text())
    cfg, attempts = validate_protocol(actual['declared_pilot_protocol'])
    if actual['fixture_drive'] != asdict(cfg) or actual['attempt_order'] != attempts:
        raise ValueError('Collector differs from declared pilot')
    template, faces, _ = read_template_obj(EXPERIMENT/'vendor/Active-3D-Vision-and-Touch/pterotactyl/objects/touch_chart.obj')
    destination = root if report_output is None else report_output
    if destination != root:
        destination.resolve().relative_to(root.resolve())
        destination.mkdir(exist_ok=False)
    charts = destination/'charts'
    charts.mkdir(exist_ok=False)
    declared = actual['declared_pilot_protocol']
    baseline, baseline_provenance = read_baseline(declared)
    rows = []
    for attempt in attempts:
        row, _ = report_attempt(root, attempt, template.numpy(), faces.verts_idx.numpy(), charts)
        old = next(c for c in baseline['cases'] if all(c[k] == v for k,v in attempt.items()))
        row['baseline'] = {k: old[k] for k in ('qualification_passed', 'qualification_checks', 'loads', 'chart')}
        rows.append(row)
    result = dict(complete=all(r['recorded_complete_600'] for r in rows),
        expected_attempts=3, cases=rows, qualification_passed=all(r['qualification_passed'] for r in rows),
        qualification_criteria={**QUALIFICATION_CRITERIA, 'expected_attempts':3},
        scope=declared['scope'], observation_root=str(root), new_model_forwards=0, new_optimizer_updates=0)
    if declared['study'] == 'fixture_angular_stiffness_pilot_v1':
        result.update(study=declared['study'], baseline_provenance=baseline_provenance)
        result['qualification_criteria']['scope'] = 'Only these three fixed fixture attempts; no claim of 48-object coverage or learned shape accuracy'
    write_json(destination/'RESULT.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}), flush=True)
    for r in rows:
        print(r['object_id'],r['direction_index'],r['qualification_passed'],
              r.get('loads',{}).get('pre_snapshot_0p5s'), flush=True)
    return 0 if result['qualification_passed'] else 2


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report-output', type=Path, help='New child directory for a saved-only corrected report; preserve earlier reports')
    args=parser.parse_args()
    raise SystemExit(report(args.root, args.report_output))
