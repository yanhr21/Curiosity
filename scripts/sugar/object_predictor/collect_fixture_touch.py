"""Declared qualification of real dynamic-hand shape sensing.

Uses the first TWO fixed TRAIN assets, official approach indices 0 and 49,
600 controls per attempt. This is not the 48-object training corpus. All failed
attempts remain in the denominator; no adaptive directions or object swaps.
An explicit --protocol enables the separate fixed 4-object/10-direction study;
the original two-object/two-direction default is unchanged.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from .fixture_touch_scene import FixtureDrive, FixtureTouchScene
from .shape_fixture_assets import fixed_selection, load_fixture_asset, official_action_directions


FIELD_KEYS = ('pos', 'area', 'pressure', 'traction_vec', 'patch', 'pad', 'anatomical_pad')
TRACE_KEYS = ('time_s', 'hand_pose_w', 'hand_velocity_w', 'measured_palmar_load_n',
              'validation_full_hand_load_n', 'target_depth_m', 'touched', 'overload_seen')


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Use the retained explicit compute step and pipeline lock')
    root = args.output
    selected = [ident for ident, split in fixed_selection() if split == 'recon_train'][:2]
    if len(selected) != 2:
        raise AssertionError('Expected the fixed first two TRAIN IDs')
    directions, provenance = official_action_directions()
    cfg = FixtureDrive()
    protocol = dict(qualification_only=True, objects=selected, directions=[0, 49],
        controls_per_attempt=600, planned_controls=2400, dt=.02, substeps=8,
        sample_frame=499, sample_time_s=10., fixture_drive=asdict(cfg),
        prepared_root=str(args.prepared_root), directions_source=provenance,
        controller_inputs=['public clock and envelope', 'measured assigned palmar load'],
        sensor='Existing ideal continuous_palmar_v1 from solved Newton contact forces',
        scope='Static known fixture frame shape-sensing qualification. No unknown pose, mass, material, carry or learning result.',
        observation_surface_keys=FIELD_KEYS, model_updates=0,
        attempt_order=[dict(object_id=ident, direction_index=d) for ident in selected for d in (0, 49)])
    declared_path = getattr(args, 'protocol', None)
    if declared_path is not None:
        declared = json.loads(Path(declared_path).read_text())
        if declared.get('study') in ('fixture_settling_pilot_v1', 'fixture_angular_stiffness_pilot_v1'):
            from .fixture_settling_pilot import validate_protocol
            cfg, attempts = validate_protocol(declared)
            if declared['study'] == 'fixture_angular_stiffness_pilot_v1':
                for field in ('prepared_source_sha256', 'baseline_artifact_sha256'):
                    for path, expected in declared.get(field, {}).items():
                        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                            raise ValueError('Prepared fixture binding changed: '+path)
            protocol.update(objects=declared['objects'], directions=declared['directions'],
                fixture_drive=asdict(cfg), attempt_order=attempts, planned_controls=600*len(attempts),
                source_protocol=str(Path(declared_path).resolve()), declared_pilot_protocol=declared,
                study=declared['study'], scope=declared['scope'])
        else:
            from .report_fixture_sequences import validate_sequence_protocol
            attempts = validate_sequence_protocol(declared)
            if 'fixture_drive' in declared and declared['fixture_drive'] != asdict(cfg):
                raise ValueError('Sequence collection must retain the original FixtureDrive parameters')
            protocol.update(objects=declared['objects'], directions=[d for group in declared['direction_groups'] for d in group],
            direction_groups=declared['direction_groups'], attempt_order=attempts,
            planned_controls=600*len(attempts), source_protocol=str(Path(declared_path).resolve()),
            declared_sequence_protocol=declared,
            scope='Four fixed TRAIN objects, two fixed five-touch sequences each, public static fixture. No unknown pose, mass, material, carry or learning result.')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'PROTOCOL.json', protocol)
    rows = []
    for attempt in protocol['attempt_order']:
        ident, direction = attempt['object_id'], attempt['direction_index']
        label = f'object_{ident}_direction_{direction:02d}'
        directory = root/label
        directory.mkdir()
        record = dict(**attempt, directory=label, complete=False, recorded_controls=0,
                      actual_controls=0, actual_physics_substeps=0, partial_control_substeps=0)
        scene = None
        trace = {key: [] for key in TRACE_KEYS}
        fields = {key: [] for key in FIELD_KEYS}
        offsets = [0]
        try:
            asset = load_fixture_asset(args.prepared_root, ident)
            write_json(directory/'ASSET.json', asset.metadata)
            # Full object mesh is a separate physics/render/label artifact,
            # never part of the numeric tactile observation files below.
            np.savez_compressed(directory/'object_mesh.npz', vertices=asset.physics_vertices_m,
                faces=asset.faces, fixture_center_m=asset.center_m,
                fixture_quaternion_xyzw=asset.quaternion_xyzw)
            scene = FixtureTouchScene(asset, directions[direction], cfg)
            write_json(directory/'SCENE.json', dict(
                frames={k: v.tolist() for k, v in scene.frames.items()},
                hand_body=scene.hand_body, hand_shape=scene.hand_shape, object_shape=scene.object_shape,
                body_flags=scene.model.body_flags.numpy().tolist(),
                body_mass=scene.model.body_mass.numpy().tolist(),
                joint_effort_limit=scene.model.joint_effort_limit.numpy().tolist(),
                note='Effort limits bound D6 actuator axes, not impact/contact forces.'))
            for frame in range(600):
                state = scene.step(dt=.02, substeps=8)
                for key in TRACE_KEYS:
                    trace[key].append(state[key])
                for key in FIELD_KEYS:
                    fields[key].append(np.asarray(state['field'][key]).copy())
                offsets.append(offsets[-1]+len(state['field']['pad']))
                record['recorded_controls'] += 1
                if frame in (0, 99, 299, 499, 599):
                    print('FIXTURE_TOUCH_CLOCK', label, frame,
                          state['measured_palmar_load_n'], state['target_depth_m'], flush=True)
            record['complete'] = True
            record['peak_measured_palmar_load_n'] = float(max(trace['measured_palmar_load_n']))
            record['peak_validation_full_hand_load_n'] = float(max(trace['validation_full_hand_load_n']))
            record['sample_measured_palmar_load_n'] = float(trace['measured_palmar_load_n'][499])
            record['field_overflow_steps'] = scene.field.overflow_steps
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            record['traceback'] = traceback.format_exc()
            print('FIXTURE_TOUCH_FAILURE', label, record['error'], flush=True)
        finally:
            if scene is not None:
                record['actual_physics_substeps'] = scene.physics_substeps
                record['actual_controls'], record['partial_control_substeps'] = divmod(scene.physics_substeps, 8)
                record['field_overflow_steps'] = scene.field.overflow_steps
            if record['recorded_controls']:
                np.savez_compressed(directory/'trace.npz', **{k: np.asarray(v) for k, v in trace.items()})
                np.savez_compressed(directory/'observed_surface.npz', offset=np.array(offsets, dtype=np.int64),
                                    **{k: np.concatenate(v, axis=0) for k, v in fields.items()})
            write_json(directory/'ATTEMPT.json', record)
            rows.append(record)
            if scene is not None:
                del scene
            gc.collect()
    expected = len(protocol['attempt_order'])
    result = dict(complete=len(rows)==expected and all(r['complete'] for r in rows), attempted=len(rows), expected_attempts=expected,
                  actual_controls=sum(r['actual_controls'] for r in rows),
                  actual_physics_substeps=sum(r['actual_physics_substeps'] for r in rows),
                  recorded_controls=sum(r['recorded_controls'] for r in rows), planned_controls=protocol['planned_controls'], cases=rows,
                  model_forwards=0, optimizer_updates=0,
                  qualification_passed=None, scope=protocol['scope'])
    write_json(root/'COLLECTION_RESULT.json', result)
    print('FIXTURE_TOUCH_COLLECTION_TERMINAL', json.dumps(result), flush=True)
    # Runtime failures are retained and fail the process; poor physical results
    # with complete traces are assessed separately, never silently excluded.
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, help='Explicit fixed four-object/two-five-touch sequence protocol')
    raise SystemExit(main(parser.parse_args()))
