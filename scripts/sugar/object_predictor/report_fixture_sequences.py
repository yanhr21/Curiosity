"""Saved observation charts for four fixed TRAIN objects x two five-touch groups.

No object mesh or object pose is opened to construct a chart. Each probe is
independent physical acquisition in the same public fixture frame; five probes
are accumulated observations, not one continuous five-contact rollout.
Physical qualification is separate from chart availability: validation-only
loads may fail the whole study but never change a model input or select a probe.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .report_fixture_touch import (
    MAX_EDGE_M, QUALIFICATION_CRITERIA, SNAPSHOT, SURFACE_KEYS, TRACE_KEYS,
    coverage_metrics, empty_chart, qualification_checks, read_npz, snapshot_chart,
    trace_metrics, validate_surface, validate_trace, write_json,
)
from .shape_fixture_assets import EXPERIMENT, FIXTURE_CENTER_M, METERS_PER_CANONICAL_UNIT, fixed_selection


DIRECTION_GROUPS = ((0, 10, 20, 30, 40), (9, 19, 29, 39, 49))


def validate_sequence_protocol(protocol):
    """Validate fixed inputs, without reading any object geometry or observations."""
    selected = [ident for ident, split in fixed_selection() if split == 'recon_train'][:4]
    if protocol.get('objects') != selected:
        raise ValueError('Require the first four frozen TRAIN object IDs in source order')
    if protocol.get('direction_groups') != [list(group) for group in DIRECTION_GROUPS]:
        raise ValueError('Preserve the two fixed five-direction groups and order')
    for key, expected in dict(controls_per_attempt=600, dt=.02, substeps=8, sample_frame=499).items():
        if protocol.get(key) != expected:
            raise ValueError(f'Fixed fixture sequence protocol changed: {key}')
    directions = [d for group in DIRECTION_GROUPS for d in group]
    attempts = [dict(object_id=ident, direction_index=d) for ident in selected for d in directions]
    if 'attempt_order' in protocol and protocol['attempt_order'] != attempts:
        raise ValueError('Attempt order must preserve all forty object/direction pairs')
    if 'directions' in protocol and protocol['directions'] != directions:
        raise ValueError('Direction list differs from sequence groups')
    if 'planned_controls' in protocol and protocol['planned_controls'] != 24000:
        raise ValueError('Require forty independent 600-control attempts')
    from .fixture_touch_scene import FixtureDrive
    if 'fixture_drive' in protocol and protocol['fixture_drive'] != asdict(FixtureDrive()):
        raise ValueError('Do not change the original fixture controller parameters')
    return attempts


def report_attempt(root, expected, template, template_faces, charts):
    """Keep every declared attempt; physical labels never gate numeric charts."""
    label = f'object_{expected["object_id"]}_direction_{expected["direction_index"]:02d}'
    directory = root/label
    row = dict(**expected, directory=label, collection_complete=False,
               trace_available=False, surface_available=False, errors=[])
    arrays, chart_info = empty_chart(template_faces, 'snapshot_unavailable')
    try:
        attempt = json.loads((directory/'ATTEMPT.json').read_text())
        row['attempt_record'] = attempt
        if any(attempt.get(key) != value for key, value in expected.items()):
            raise ValueError('ATTEMPT identity differs from the fixed protocol')
        row['collection_complete'] = bool(attempt['complete'])
        if attempt.get('error'):
            row['errors'].append('collection: '+attempt['error'])
    except Exception as exc:
        row['errors'].append(f'attempt_record: {type(exc).__name__}: {exc}')
    trace = None
    try:
        observation_trace_keys = tuple(key for key in TRACE_KEYS if not key.startswith('validation_'))
        trace = read_npz(directory/'trace.npz', observation_trace_keys)
        # Reuse the original clock/pose/observation validator while keeping its
        # validation-only full-hand channel outside chart availability. Missing
        # or bad GT diagnostics fail qualification below, never sensor inputs.
        n = validate_trace({**trace, 'validation_full_hand_load_n': np.zeros(len(trace['time_s']))})
        row.update(trace_available=True)
        record = row.get('attempt_record', {})
        if (record.get('recorded_controls') != n or n != 600
                or record.get('actual_controls') != 600
                or record.get('actual_physics_substeps') != 4800
                or record.get('partial_control_substeps') != 0):
            row['collection_complete'] = False
            row['errors'].append('Actual/saved controls differ from the complete 600 x 8 protocol')
    except Exception as exc:
        row['collection_complete'] = False
        row['errors'].append(f'trace: {type(exc).__name__}: {exc}')
        trace = None
    qualification_trace = None
    if trace is not None:
        try:
            validation = read_npz(directory/'trace.npz', ('validation_full_hand_load_n',))
            qualification_trace = {**trace, **validation}
            validate_trace(qualification_trace)
            row['loads'] = trace_metrics(qualification_trace)
        except Exception as exc:
            qualification_trace = None
            row['errors'].append(f'qualification_trace: {type(exc).__name__}: {exc}')
    if trace is not None:
        try:
            surface = read_npz(directory/'observed_surface.npz', ('offset',)+SURFACE_KEYS)
            validate_surface(surface, len(trace['time_s']))
            row.update(surface_available=True)
            if len(trace['time_s']) > SNAPSHOT:
                begin, end = surface['offset'][SNAPSHOT:SNAPSHOT+2]
                snapshot = {key: surface[key][begin:end] for key in SURFACE_KEYS}
                arrays, chart_info = snapshot_chart(snapshot, trace['hand_pose_w'][SNAPSHOT], template, template_faces)
                arrays['source_frame_offset'] = np.int64(begin)
            else:
                chart_info['reason'] = 'actual_trace_ended_before_frame499'
        except Exception as exc:
            row['errors'].append(f'surface/chart: {type(exc).__name__}: {exc}')
            arrays, chart_info = empty_chart(template_faces, f'surface/chart: {type(exc).__name__}: {exc}')
    else:
        chart_info['reason'] = 'missing_or_invalid_actual_trace'
    if qualification_trace is not None and row['surface_available']:
        try:
            row['coverage'] = coverage_metrics(surface, qualification_trace)
        except Exception as exc:
            row['errors'].append(f'coverage_diagnostic: {type(exc).__name__}: {exc}')
    # The chart above is already complete. No GT-dependent qualification result
    # below is allowed to modify it, including a full-hand peak-load failure.
    chart_path = charts/(label+'.npz')
    np.savez_compressed(chart_path, **arrays)
    row.update(chart=chart_info, chart_file=str(chart_path.relative_to(root)),
               recorded_complete_600=bool(row['collection_complete'] and row['trace_available'] and row['surface_available']))
    row['qualification_checks'] = qualification_checks(row)
    row['qualification_checks']['record_integrity'] = not row['errors']
    row['qualification_passed'] = all(row['qualification_checks'].values())
    return row, arrays['chart'][0]


def pack_sequences(rows, charts, object_ids):
    """Ordering comes only from the protocol, never a validity/accuracy ranking."""
    expected = [(ident, d) for ident in object_ids for group in DIRECTION_GROUPS for d in group]
    actual = [(row['object_id'], row['direction_index']) for row in rows]
    if actual != expected or len(charts) != 40:
        raise ValueError('All forty declared attempts must be present in exact order')
    values = np.asarray(charts, dtype=np.float32)
    if values.shape != (40, 25, 4):
        raise ValueError('Expected forty original 25-vertex xyz/mask charts')
    from .active3d_charts import validate_chart_masks
    validate_chart_masks(values)
    sequences = []
    for object_index, ident in enumerate(object_ids):
        for group_index, directions in enumerate(DIRECTION_GROUPS):
            start = object_index*10+group_index*5
            selected = rows[start:start+5]
            sequences.append(dict(sequence_index=len(sequences), object_id=ident,
                direction_group=group_index, direction_indices=list(directions),
                attempt_indices=list(range(start, start+5)),
                chart_files=[row['chart_file'] for row in selected],
                observed_chart_count=sum(row['chart']['available'] for row in selected),
                qualification_passed=all(row['qualification_passed'] for row in selected)))
    return values.reshape(8, 5, 25, 4), sequences


def report(root):
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    attempts = validate_sequence_protocol(protocol)
    if 'fixture_drive' not in protocol:
        raise ValueError('Require the actual collector protocol with fixed fixture drive parameters')
    from .official_active3d import read_template_obj
    template, faces, _ = read_template_obj(EXPERIMENT/'vendor/Active-3D-Vision-and-Touch/pterotactyl/objects/touch_chart.obj')
    template, faces = template.numpy(), faces.verts_idx.numpy()
    if template.shape != (25, 3) or faces.shape != (32, 3):
        raise ValueError('Expected the original 25-vertex/32-face touch topology')
    criteria = {**QUALIFICATION_CRITERIA, 'expected_attempts': 40,
                'scope': 'All forty fixed fixture attempts; no unknown shape/pose generalization claim'}
    charts = root/'sequence_charts'
    charts.mkdir(exist_ok=False)
    write_json(charts/'PROTOCOL.json', dict(qualification_criteria=criteria,
        attempt_order=attempts, direction_groups=[list(g) for g in DIRECTION_GROUPS],
        snapshot_frame=SNAPSHOT, max_interpolation_edge_m=MAX_EDGE_M,
        fixture_center_m=list(FIXTURE_CENTER_M), meters_per_canonical_unit=METERS_PER_CANONICAL_UNIT,
        input_selection='Observation chart availability only; physical GT qualification never alters charts',
        failure_policy='Missing/invalid observation chart stays mask0 in its original slot; every physical failure fails the overall gate'))
    rows, chart_arrays = [], []
    for expected in attempts:
        row, chart = report_attempt(root, expected, template, faces, charts)
        rows.append(row)
        chart_arrays.append(chart)
    inputs, sequences = pack_sequences(rows, chart_arrays, protocol['objects'])
    np.savez_compressed(root/'sequence_inputs.npz', charts=inputs)
    result = dict(complete=True, expected_attempts=40, cases=rows, sequences=sequences,
        sequence_count=8, charts_per_sequence=5, input_file='sequence_inputs.npz',
        input_key='charts', input_shape=list(inputs.shape),
        complete_recordings=sum(row['recorded_complete_600'] for row in rows),
        available_snapshot_charts=sum(row['chart']['available'] for row in rows),
        qualification_passed=all(row['qualification_passed'] for row in rows),
        qualification_criteria=criteria, snapshot_frame=SNAPSHOT,
        observation_chart_selection='Largest supported area then lowest pad ID at the fixed snapshot; no GT selection',
        mask_policy='No supported observation -> xyz=0/mask0; supported observation -> mask2 even if physical qualification fails',
        new_physics_controls=0, model_forwards=0, optimizer_updates=0,
        scope='Four TRAIN objects, independent touches in a public fixed frame; observation interpolation, not learned reconstruction',
        limitations=['Five charts are accumulated independent probes, not one continuous rollout.',
                    'Ideal contact geometry is not a pressure-only or optical tactile sensor.',
                    'Physical qualification failures are retained and block downstream training.',
                    'No object mesh, true object pose, or GT surface normal was read to build inputs.'])
    write_json(root/'RESULT.json', result)
    lines = ['# 四物体、八组五次触摸：实际观测资格', '',
        f'完整记录 {result["complete_recordings"]}/40；有效观测 chart {result["available_snapshot_charts"]}/40；整体资格 {result["qualification_passed"]}。',
        '每组五次独立物理探测共用公开固定参考系，不是连续搬运或完整物体预测。所有方向按预先顺序保留。',
        '输入仅来自接触面观测和实测手位姿；验证用全手载荷可使资格失败，但不会删掉 chart 或改变输入 mask。', '',
        '| 物体/方向 | 完整600帧 | 观测chart | 物理/记录资格 | 失败项 |',
        '|---|---|---|---|---|']
    for row in rows:
        failed = ', '.join(k for k, passed in row['qualification_checks'].items() if not passed)
        lines.append(f'| {row["object_id"]}/{row["direction_index"]} | {row["recorded_complete_600"]} | '
                     f'{row["chart"]["available"]} | {row["qualification_passed"]} | {failed} |')
    lines += ['', '输入：sequence_inputs.npz 的 charts，形状 [8,5,25,4]；序列身份与方向仅在 RESULT.json 元数据中。',
              '原600帧、初始25帧载荷≤0.001N、全程峰值≤100N、9.5–10s载荷[1.5,2.5]N占比≥80%、无溢出、499帧有效chart等门槛全部保留。',
              '未运行模型、优化、物理或渲染；这不证明形状重建精度。']
    with (root/'REPORT.md').open('x') as handle:
        handle.write('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    result = report(parser.parse_args().root)
    raise SystemExit(0 if result['qualification_passed'] else 2)
