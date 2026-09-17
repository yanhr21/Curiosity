"""CPU-only, observation-only input adapter for a NEW fixed TRAIN shape diagnostic.

This does not repair or relabel the original 40-attempt controller qualification.
Sources are fixed before any shape-model result: original 37 supported snapshots,
original 11898/d29/frame499 with the approved local chart diagnostic, and the two
13266 search snapshots at frames827/854. Search 18704 is never substituted.
These are five independent probes in a public static fixture, not a rollout.

The full official Utonia accepts variable point counts. This adapter produces its
existing 20 feature columns using one REAL left hand; no right-hand padding and
no model is constructed. Positions are ideal simulated contact-surface readings,
not pressure-only hardware reconstructions. Normals come only from known hand CAD.

ContactField.traction_vec is hand-local Pa, estimated by distributing the solved
patch friction load and projecting its direction onto each contact face. It is
NOT a force in N and is not a separately measured per-face shear channel. Multiply
by face area once, rotate by the measured hand pose, then integrate per probe and
voxel; preserve its sign and do not project it again using a different normal.
Normal pressure * area gives scalar load, not a full vector normal force.

Only coord/feat/grid_coord/offset enter sequence_inputs.npz. Asset IDs, source
paths, force qualification, chart provenance and labels remain separate. Teacher
mesh/pose/mass/latent are never loaded here. No model/GPU/physics/training occurs.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from .shape_fixture_assets import EXPERIMENT, FIXTURE_CENTER_M, METERS_PER_CANONICAL_UNIT

STUDY = EXPERIMENT/'overfit_repair_v1'
OLD_ROOT = STUDY/'sugar_shape_fixed40'
SEARCH_ROOT = STUDY/'fixture_tangent_search_pilot_v1'
OUTPUT = STUDY/'sugar_shape_observed40_v1'
OBJECT_IDS = ('18704', '11898', '15737', '13266')
DIRECTIONS = ((0, 10, 20, 30, 40), (9, 19, 29, 39, 49))
SURFACE_KEYS = ('pos', 'area', 'pressure', 'traction_vec', 'patch', 'pad')
TRACE_KEYS = ('time_s', 'hand_pose_w', 'measured_palmar_load_n')
INPUT_KEYS = ('coord', 'feat', 'grid_coord', 'offset')
CHART_KEYS = ('chart', 'chart_world_m', 'chart_hand_m', 'source_frame_indices',
    'barycentric', 'support_edge_m', 'support_triangles', 'selected_pad',
    'touch_template_faces', 'hand_pose_w', 'source_frame_offset',
    'source_pos_hand_m', 'source_area_m2', 'source_pressure_pa', 'source_pad',
    'source_active', 'snapshot_available')
VOXEL_M = .002
COORD_PER_M = 5.


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def source_plan(old_root=OLD_ROOT, search_root=SEARCH_ROOT):
    """No quality metric or object geometry participates in this fixed mapping."""
    rows = []
    for oid in OBJECT_IDS:
        for group in DIRECTIONS:
            for direction in group:
                kind, root, frame, chart_folder = 'original_supported', Path(old_root), 499, 'sequence_charts'
                if (oid, direction) == ('11898', 29):
                    kind = 'original_fixed_local_fallback'
                elif oid == '13266' and direction in (0, 49):
                    kind, root, frame, chart_folder = 'planned_search_snapshot', Path(search_root), {0: 827, 49: 854}[direction], 'charts'
                name = f'object_{oid}_direction_{direction:02d}'
                rows.append(dict(slot=len(rows), object_id=oid, direction_index=direction,
                    source_kind=kind, source_root=str(root), snapshot_frame=frame,
                    directory=str(root/name), chart_file=str(root/chart_folder/(name+'.npz'))))
    return rows


def read_snapshot(folder, frame):
    """Access only named observation arrays, even if an NPZ contains extra GT."""
    folder = Path(folder)
    with np.load(folder/'trace.npz', allow_pickle=False) as saved:
        trace = {key: saved[key] for key in TRACE_KEYS}
    n = len(trace['time_s'])
    if (not 0 <= frame < n or trace['hand_pose_w'].shape != (n, 7)
            or trace['measured_palmar_load_n'].shape != (n,)
            or any(not np.isfinite(value).all() for value in trace.values())
            or not np.allclose(trace['time_s'], (np.arange(n)+1)*.02, atol=1e-7, rtol=0)):
        raise ValueError('Invalid recorded single-hand observation trace')
    with np.load(folder/'observed_surface.npz', allow_pickle=False) as saved:
        offsets = saved['offset']
        if (offsets.shape != (n+1,) or not np.issubdtype(offsets.dtype, np.integer)
                or offsets[0] != 0 or np.any(np.diff(offsets) < 0)):
            raise ValueError('Invalid ragged observation offsets')
        start, end = map(int, offsets[frame:frame+2])
        surface = {}
        for key in SURFACE_KEYS:
            values = saved[key]
            if len(values) != offsets[-1]:
                raise ValueError('Observed surface length differs from offsets')
            surface[key] = values[start:end].copy()
    return trace['hand_pose_w'][frame].copy(), surface, dict(
        time_s=float(trace['time_s'][frame]), measured_palmar_load_n=float(trace['measured_palmar_load_n'][frame]),
        recorded_controls=n, source_frame_offset=start)


@lru_cache(maxsize=1)
def left_hand_cad():
    from sugar_newton.hand.patches import PATCH_SPECS, load_hand_mesh
    from .hand_surface_normals import nearest_normals
    mesh = load_hand_mesh('left')
    vertices = np.asarray(mesh.vertices)
    anchors = []
    # Original collect_newton.hand_sites rule, explicitly only left / sign -1
    # to match this fixture's actual ContinuousPalmarField and support frame.
    for spec in PATCH_SPECS:
        distance = (vertices[:, 0]-spec.center_x_m)**2+(vertices[:, 2]-spec.center_z_m)**2
        near = np.argsort(distance)[:32]
        anchors.append(vertices[near[np.argmax(-vertices[near, 1])]])
    anchors = np.asarray(anchors)
    normals, _ = nearest_normals(mesh, anchors)
    if anchors.shape != (27, 3):
        raise ValueError('Expected exactly 27 real left-hand CAD anchors')
    return mesh, anchors, normals


def encode_probe(pose, surface):
    """Pure sensor/CAD-to-point-cloud path. No metadata/label argument exists."""
    from .hand_surface_normals import nearest_normals
    if set(surface) != set(SURFACE_KEYS):
        raise ValueError('Only the explicit surface observation allowlist is accepted')
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > 1e-4:
        raise ValueError('Expected one measured left-hand pose')
    count = len(surface['pad'])
    if (surface['pos'].shape != (count, 3) or surface['traction_vec'].shape != (count, 3)
            or any(np.asarray(surface[key]).shape != (count,) for key in ('area', 'pressure', 'patch', 'pad'))
            or any(not np.isfinite(value).all() for value in surface.values())
            or not np.issubdtype(surface['pad'].dtype, np.integer)
            or not np.issubdtype(surface['patch'].dtype, np.integer)
            or np.any(surface['patch'] != 0) or not np.isin(surface['pad'], np.arange(-1, 27)).all()
            or np.any(surface['area'] < 0) or np.any(surface['pressure'] < 0)):
        raise ValueError('Invalid actual single-left-hand tactile field')
    active = (surface['pad'] >= 0) & (surface['area'] > 0) & (surface['pressure'] > 0)
    source = np.flatnonzero(active)
    local = np.asarray(surface['pos'][source], np.float64)
    area = np.asarray(surface['area'][source], np.float64)
    pressure = np.asarray(surface['pressure'][source], np.float64)
    traction = np.asarray(surface['traction_vec'][source], np.float64)
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    mesh, anchors, anchor_normals = left_hand_cad()
    contact_normals = nearest_normals(mesh, local)[0] if len(local) else np.empty((0, 3))
    anchor_world = anchors@rotation.T+pose[:3]
    contact_world = local@rotation.T+pose[:3]
    center = np.asarray(FIXTURE_CENTER_M)
    anchor_coord = (anchor_world-center)*COORD_PER_M
    contact_coord = (contact_world-center)*COORD_PER_M
    # Each invocation is ONE probe. Different probes have separate sparse offsets.
    keys = np.floor(np.concatenate((anchor_coord, contact_coord))/(VOXEL_M*COORD_PER_M)).astype(np.int32)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    ai, ci = inverse[:27], inverse[27:]
    n = len(unique)
    counts = np.bincount(ai, minlength=n)
    coord = np.zeros((n, 3)); normal = np.zeros((n, 3))
    np.add.at(coord, ai, anchor_coord)
    np.add.at(normal, ai, anchor_normals@rotation.T)
    coord /= np.maximum(counts[:, None], 1)
    normal /= np.maximum(counts[:, None], 1)
    voxel_area = np.bincount(ci, weights=area, minlength=n)
    scalar_load = pressure*area
    voxel_load = np.bincount(ci, weights=scalar_load, minlength=n)
    traction_force_world = (traction*area[:, None])@rotation.T
    voxel_traction_force = np.zeros((n, 3)); weighted_coord = np.zeros((n, 3)); weighted_normal = np.zeros((n, 3))
    np.add.at(voxel_traction_force, ci, traction_force_world)
    np.add.at(weighted_coord, ci, contact_coord*area[:, None])
    np.add.at(weighted_normal, ci, (contact_normals@rotation.T)*area[:, None])
    occupied = voxel_area > 0
    coord[occupied] = weighted_coord[occupied]/voxel_area[occupied, None]
    normal[occupied] = weighted_normal[occupied]/voxel_area[occupied, None]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    extra = np.zeros((n, 11))
    extra[:, 0] = occupied
    extra[:, 1] = np.log1p(voxel_load)
    extra[:, 2:5] = np.arcsinh(voxel_traction_force/2.)
    extra[:, 5] = voxel_area*1e4
    # age=0: independent snapshots, not a fictitious continuous timeline.
    # hand=0: the actual left hand, never a virtual right hand.
    extra[:, 8:11] = [0., 0., -1.]
    feat = np.concatenate((coord, np.zeros((n, 3)), normal, extra), axis=1)
    if not np.isfinite(feat).all():
        raise ValueError('Nonfinite encoded observation')
    encoded = dict(coord=coord.astype(np.float32), feat=feat.astype(np.float32),
                   grid_coord=(unique-unique.min(0)).astype(np.int32))
    provenance = dict(source_indices=source, source_to_voxel=ci, anchor_to_voxel=ai,
        source_contact_world_m=contact_world, anchor_world_m=anchor_world,
        source_load_n=scalar_load, source_estimated_traction_force_world_n=traction_force_world,
        voxel_area_m2=voxel_area, voxel_load_n=voxel_load,
        voxel_estimated_traction_force_world_n=voxel_traction_force)
    metrics = dict(actual_hands=1, left_hand_anchors=27, source_faces=count,
        active_assigned_contact_faces=len(source), input_points=n, occupied_voxels=int(occupied.sum()),
        observed_scalar_load_n=float(scalar_load.sum()), voxel_scalar_load_n=float(voxel_load.sum()),
        observed_area_m2=float(area.sum()), voxel_area_m2=float(voxel_area.sum()),
        observed_estimated_traction_force_world_n=traction_force_world.sum(0).tolist(),
        voxel_estimated_traction_force_world_n=voxel_traction_force.sum(0).tolist(),
        encoded_load_roundtrip_error_n=abs(float(np.expm1(encoded['feat'][:, 10].astype(float)).sum())-float(scalar_load.sum())))
    if (abs(metrics['observed_scalar_load_n']-metrics['voxel_scalar_load_n']) > 1e-10
            or not np.allclose(traction_force_world.sum(0), voxel_traction_force.sum(0), atol=1e-10, rtol=0)):
        raise AssertionError('Per-probe voxel integration changed measured quantities')
    return encoded, provenance, metrics


def pack_probes(probes):
    if len(probes) != 40:
        raise ValueError('All fixed 40 probes / eight five-probe inputs are required')
    result = {key: np.concatenate([p[key] for p in probes]) for key in INPUT_KEYS[:-1]}
    result['offset'] = np.cumsum([len(p['coord']) for p in probes], dtype=np.int64)
    return result


def validate_chart(chart, surface, pose, frame_offset):
    """Replay a stored chart against this source; no GT geometry query."""
    for key, raw in (('source_pos_hand_m', 'pos'), ('source_area_m2', 'area'),
                     ('source_pressure_pa', 'pressure'), ('source_pad', 'pad')):
        if not np.array_equal(chart[key], surface[raw]):
            raise ValueError('Chart source differs from actual snapshot: '+key)
    if (chart['chart'].shape != (1, 25, 4) or not np.all(chart['chart'][..., 3] == 2)
            or not np.array_equal(chart['hand_pose_w'], np.asarray(pose, float)[None])
            or int(chart['source_frame_offset']) != frame_offset):
        raise ValueError('Wrong chart mask/pose/source offset')
    indices, weights = chart['source_frame_indices'], chart['barycentric']
    if (indices.shape != (25, 3) or not np.issubdtype(indices.dtype, np.integer)
            or indices.min() < 0 or indices.max() >= len(surface['pos'])
            or weights.shape != (25, 3) or not np.isfinite(weights).all()
            or weights.min() < -1e-9 or not np.allclose(weights.sum(1), 1., atol=1e-9, rtol=0)):
        raise ValueError('Invalid interpolation provenance')
    replay = np.einsum('ni,nij->nj', weights, surface['pos'][indices])
    active = (surface['pad'] >= 0) & (surface['area'] > 0) & (surface['pressure'] > 0)
    if (not np.array_equal(chart['source_active'], active) or not active[indices].all()
            or not np.all(surface['pad'][indices] == int(chart['selected_pad']))):
        raise ValueError('Chart interpolation must use its actual active selected pad')
    world = replay@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
    canonical = (world-np.asarray(FIXTURE_CENTER_M))/METERS_PER_CANONICAL_UNIT
    if (not np.allclose(replay, chart['chart_hand_m'], atol=1e-12, rtol=0)
            or not np.allclose(world, chart['chart_world_m'], atol=1e-12, rtol=0)
            or not np.allclose(canonical, chart['chart'][0, :, :3], atol=2e-8, rtol=0)):
        raise ValueError('Chart hand/world/canonical observation replay failed')
    return float(np.max(abs(replay-chart['chart_hand_m'])))


def build(output=OUTPUT, old_root=OLD_ROOT, search_root=SEARCH_ROOT):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/'probes').mkdir(); (output/'charts').mkdir()
    plans = source_plan(old_root, search_root)
    protocol = dict(study='sugar_shape_observed40_v1', scope='Post-hoc TRAIN-only sparse shape observation diagnostic',
        source_mapping=plans, object_ids=list(OBJECT_IDS), direction_groups=[list(g) for g in DIRECTIONS],
        expected_probes=40, sequence_count=8, probes_per_sequence=5,
        source_policy='37 original snapshots + fixed 11898/d29 local fallback + two preplanned search snapshots; no prediction/GT selection',
        controller_acceptance_reclassified=False, old_40_result_unchanged=True,
        new_observation_criteria=['complete finite actual recording and no overflow',
            'actual assigned contact and observed load integral agrees within 1e-6N',
            'stored or approved local chart replays to raw source and measured hand pose'],
        arbitrary_2n_target_required_for_geometric_observation=False,
        input_allowlist=dict(trace=list(TRACE_KEYS), surface=list(SURFACE_KEYS)),
        input_npz_keys=list(INPUT_KEYS), coordinates='5*(measured_world_xyz-[0,0,.75]); fixed world axes',
        voxel_m=VOXEL_M, actual_hands=1, hand_side='left', palm_sign=-1,
        five_probes='Independent sparse point batches/offsets; never simultaneous force sum',
        teacher_loaded=False, model_forwards=0, optimizer_updates=0, physics_controls=0)
    (output/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2)+'\n')
    old_result = json.loads((Path(old_root)/'RESULT.json').read_text())
    search_result = json.loads((Path(search_root)/'RESULT.json').read_text())
    old_rows = {(r['object_id'], r['direction_index']): r for r in old_result['cases']}
    search_rows = {(r['object_id'], r['direction_index']): r for r in search_result['cases']}
    records, probes = [], []
    for plan in plans:
        row = dict(plan, observation_qualified=False)
        pair = (plan['object_id'], plan['direction_index'])
        old = old_rows[pair]
        selected = search_rows[pair] if plan['source_kind'] == 'planned_search_snapshot' else old
        row.update(old_protocol_qualification_passed=bool(old['qualification_passed']),
            old_protocol_failed_checks=[k for k, v in old['qualification_checks'].items() if not v],
            source_protocol_qualification_passed=bool(selected['qualification_passed']),
            source_protocol_failed_checks=[k for k, v in selected['qualification_checks'].items() if not v],
            source_controller_2n_target_passed=bool(selected['qualification_checks'].get('target_band',
                selected['qualification_checks'].get('pre_snapshot_target_load', False))))
        try:
            folder, frame = Path(plan['directory']), plan['snapshot_frame']
            attempt = json.loads((folder/'ATTEMPT.json').read_text())
            pose, surface, clock = read_snapshot(folder, frame)
            if ((attempt['object_id'], attempt['direction_index']) != pair
                    or not attempt['complete'] or attempt['recorded_controls'] != clock['recorded_controls']
                    or attempt['actual_controls'] != clock['recorded_controls']
                    or attempt['actual_physics_substeps'] != 8*clock['recorded_controls']
                    or attempt['partial_control_substeps'] != 0 or attempt['field_overflow_steps'] != 0):
                raise ValueError('Incomplete/overflowed actual source recording')
            if plan['source_kind'] != 'planned_search_snapshot' and clock['recorded_controls'] != 600:
                raise ValueError('Original source must retain all 600 declared controls')
            if plan['source_kind'] == 'planned_search_snapshot':
                if selected['snapshot_frame'] != frame or abs(selected['snapshot_time_s']-clock['time_s']) > 1e-7:
                    raise ValueError('Search snapshot differs from fixed causal contact+4s clock')
            if plan['source_kind'] == 'original_fixed_local_fallback':
                from .official_active3d import read_template_obj
                from .local_fixture_chart import snapshot_with_local_fallback
                template, faces, _ = read_template_obj(EXPERIMENT/'vendor/Active-3D-Vision-and-Touch/pterotactyl/objects/touch_chart.obj')
                chart, info = snapshot_with_local_fallback(surface, pose, template.numpy(), faces.verts_idx.numpy())
                chart['source_frame_offset'] = np.int64(clock['source_frame_offset'])
                if not info['available'] or old['chart']['available']:
                    raise ValueError('Fixed local fallback must repair only the declared unavailable observation')
                row['local_chart_diagnostic'] = info
            else:
                if not selected['chart']['available']:
                    raise ValueError('The fixed original/search chart is unavailable; never switch source')
                with np.load(plan['chart_file'], allow_pickle=False) as z:
                    chart = {k: z[k] for k in CHART_KEYS}
            replay = validate_chart(chart, surface, pose, clock['source_frame_offset'])
            encoded, provenance, metric = encode_probe(pose, surface)
            if metric['active_assigned_contact_faces'] == 0:
                raise ValueError('Missing measured contact is unknown, not a fabricated shape sample')
            integral_error = abs(metric['observed_scalar_load_n']-clock['measured_palmar_load_n'])
            if integral_error > 1e-6:
                raise ValueError('Observed pressure*area disagrees with actual measured palmar load')
            label = f'{plan["slot"]:02d}'
            np.savez_compressed(output/'probes'/(label+'.npz'), **encoded, **provenance,
                measured_hand_pose_w=pose, **{'raw_'+k: v for k, v in surface.items()})
            np.savez_compressed(output/'charts'/(label+'.npz'), **chart)
            row.update(observation_qualified=True, probe_file='probes/'+label+'.npz', chart_copy='charts/'+label+'.npz',
                clock=clock, metrics=metric, scalar_load_replay_error_n=integral_error,
                chart_barycentric_max_error_m=replay,
                source_sha256={name: digest(folder/name) for name in ('trace.npz', 'observed_surface.npz', 'ATTEMPT.json')})
            probes.append(encoded)
        except Exception as error:
            row.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        records.append(row)
    passed = len(records) == 40 and all(row['observation_qualified'] for row in records)
    if passed:
        packed = pack_probes(probes)
        np.savez_compressed(output/'sequence_inputs.npz', **packed)
    sequences = [dict(sequence_index=i, object_id=OBJECT_IDS[i//2], direction_group=i%2,
        probe_slots=list(range(i*5, (i+1)*5)), observation_qualified=all(r['observation_qualified'] for r in records[i*5:(i+1)*5])) for i in range(8)]
    result = dict(complete=len(records) == 40, expected_probes=40, qualified_observations=sum(r['observation_qualified'] for r in records),
        observation_qualified=passed, input_written=passed, input_file='sequence_inputs.npz' if passed else None,
        input_shapes={k: list(v.shape) for k, v in packed.items()} if passed else None,
        cases=records, sequences=sequences, old_protocol_qualification_passed=bool(old_result['qualification_passed']),
        source_controller_2n_target_pass_count=sum(r['source_controller_2n_target_passed'] for r in records),
        controller_qualification_reclassified=False, shape_prediction_qualified=False,
        model_forwards=0, optimizer_updates=0, physics_controls=0,
        limitation='Input availability/integrity only. Ideal contact geometry, known fixture frame, four TRAIN shapes; no learned reconstruction, full observability, force-control repair, or generalization claim.')
    (output/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'sequences')}, indent=2))
    raise SystemExit(0 if result['observation_qualified'] else 2)
