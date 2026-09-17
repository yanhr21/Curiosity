"""Saved-only four-attempt fixture report/charts and full-mesh still replay.

Run as a module with --mode report (CPU), render (retained EGL step), or all.
No simulation, model forward or training is performed. Report/chart code never
opens object_mesh.npz. Only the separate renderer reads that GT geometry.

The single chart is observation-domain interpolation with the existing fixed
1.7 mm support-edge limit, not an optical touch CNN output or a reconstructed
whole object. The public fixture transform is (world-[0,0,.75])/.93. Every one
of the four fixed attempts remains in the denominator, including runtime errors.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .shape_fixture_assets import (EXPERIMENT, FIXTURE_CENTER_M,
                                   METERS_PER_CANONICAL_UNIT, fixed_selection)


SURFACE_KEYS = ('pos', 'area', 'pressure', 'traction_vec', 'patch', 'pad', 'anatomical_pad')
TRACE_KEYS = ('time_s', 'hand_pose_w', 'hand_velocity_w', 'measured_palmar_load_n',
              'validation_full_hand_load_n', 'target_depth_m', 'touched', 'overload_seen')
SNAPSHOT = 499
RENDER_FRAMES = (0, 499, 599)
MAX_EDGE_M = .0017
TARGET_LOAD_N = 2.
QUALIFICATION_CRITERIA = dict(
    declared_before_actual_collection=True, expected_attempts=4, required_recorded_controls=600,
    allowed_field_overflow_steps=0, initial_frames=25, initial_full_hand_max_load_n=.001,
    palmar_and_full_hand_peak_limit_n=100., pre_snapshot_window_s=[9.5, 10.],
    pre_snapshot_expected_frames=25, palmar_target_band_n=[1.5, 2.5],
    minimum_target_band_fraction=.8, required_snapshot_frame=499, required_valid_snapshot_charts=1,
    scope='Only these four fixed fixture attempts; no claim of 48-object coverage or learned shape accuracy')


def read_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in (saved.files if keys is None else keys)}


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def expected_attempts():
    ids = [ident for ident, split in fixed_selection() if split == 'recon_train'][:2]
    return [dict(object_id=ident, direction_index=direction,
                 directory=f'object_{ident}_direction_{direction:02d}')
            for ident in ids for direction in (0, 49)]


def validate_trace(trace, *, max_controls=600):
    n = len(trace['time_s'])
    if not 1 <= n <= max_controls or any(len(trace[key]) != n for key in TRACE_KEYS):
        raise ValueError('Invalid/unequal trace lengths')
    if any(not np.isfinite(trace[key]).all() for key in TRACE_KEYS):
        raise ValueError('Nonfinite actual trace')
    if trace['hand_pose_w'].shape != (n, 7) or trace['hand_velocity_w'].shape != (n, 6):
        raise ValueError('Expected one actual hand pose/velocity per frame')
    if not np.allclose(np.linalg.norm(trace['hand_pose_w'][:, 3:], axis=1), 1., atol=1e-4, rtol=0):
        raise ValueError('Invalid saved hand quaternion')
    if not np.allclose(trace['time_s'], np.arange(1, n+1)*.02, atol=1e-7, rtol=0):
        raise ValueError('Trace clocks differ from fixed 50 Hz control protocol')
    if (trace['measured_palmar_load_n'] < 0).any() or (trace['validation_full_hand_load_n'] < 0).any():
        raise ValueError('Negative saved hand load')
    return n


def validate_surface(surface, n):
    offset = surface['offset']
    if (offset.shape != (n+1,) or not np.issubdtype(offset.dtype, np.integer)
            or offset[0] != 0 or np.any(np.diff(offset) < 0)):
        raise ValueError('Invalid per-frame surface offsets')
    count = int(offset[-1])
    if any(len(surface[key]) != count for key in SURFACE_KEYS):
        raise ValueError('Surface arrays differ from offsets')
    if surface['pos'].shape != (count, 3) or surface['traction_vec'].shape != (count, 3):
        raise ValueError('Invalid local contact vector shapes')
    if any(not np.isfinite(surface[key]).all() for key in SURFACE_KEYS):
        raise ValueError('Nonfinite observed contact surface')
    if (surface['area'] < 0).any() or (surface['pressure'] < 0).any():
        raise ValueError('Negative observed contact area/pressure')
    if np.any(surface['patch'] != 0) or not np.isin(surface['pad'], np.arange(-1, 27)).all():
        raise ValueError('Expected only the left hand and its fixed pad IDs')


def active_surface(surface):
    return ((surface['patch'] == 0) & (surface['pad'] >= 0) & (surface['pad'] < 27)
            & (surface['area'] > 0) & (surface['pressure'] > 0))


def load_window(trace, start_s, end_s):
    selected = (trace['time_s'] > start_s+1e-7) & (trace['time_s'] <= end_s+1e-7)
    values = trace['measured_palmar_load_n'][selected].astype(float)
    result = dict(start_exclusive_s=start_s, end_inclusive_s=end_s,
                  frames=len(values), expected_frames=25, complete_window=len(values) == 25)
    if len(values):
        error = values-TARGET_LOAD_N
        result.update(mean_load_n=float(values.mean()), min_load_n=float(values.min()), max_load_n=float(values.max()),
                      mean_signed_error_n=float(error.mean()), mean_absolute_error_n=float(abs(error).mean()),
                      rms_error_n=float(np.sqrt(np.mean(error**2))), contact_fraction_ge_0p2n=float((values >= .2).mean()),
                      target_band_fraction=float(((values >= 1.5) & (values <= 2.5)).mean()))
    return result


def trace_metrics(trace):
    return dict(saved_controls=len(trace['time_s']),
                pre_snapshot_0p5s=load_window(trace, 9.5, 10.),
                final_retraction_0p5s=load_window(trace, 11.5, 12.),
                peak_palmar_load_n=float(trace['measured_palmar_load_n'].max()),
                peak_full_hand_load_n=float(trace['validation_full_hand_load_n'].max()),
                initial_full_hand_frames=min(len(trace['time_s']), 25),
                initial_full_hand_peak_n=float(trace['validation_full_hand_load_n'][:25].max()),
                overload_seen=bool(trace['overload_seen'].any()),
                observed_contact_fraction_ge_0p2n=float((trace['measured_palmar_load_n'] >= .2).mean()))


def qualification_checks(row):
    criteria = QUALIFICATION_CRITERIA
    loads = row.get('loads', {})
    window = loads.get('pre_snapshot_0p5s', {})
    return dict(
        complete_600_saved_controls=bool(row['recorded_complete_600']),
        no_field_overflow=row.get('attempt_record', {}).get('field_overflow_steps') == 0,
        initial_full_hand_clearance=(loads.get('initial_full_hand_frames') == criteria['initial_frames']
                                    and loads.get('initial_full_hand_peak_n', float('inf')) <= criteria['initial_full_hand_max_load_n']),
        palmar_peak_bounded=loads.get('peak_palmar_load_n', float('inf')) <= criteria['palmar_and_full_hand_peak_limit_n'],
        full_hand_peak_bounded=loads.get('peak_full_hand_load_n', float('inf')) <= criteria['palmar_and_full_hand_peak_limit_n'],
        pre_snapshot_target_load=(window.get('frames') == criteria['pre_snapshot_expected_frames']
                                 and window.get('target_band_fraction', 0.) >= criteria['minimum_target_band_fraction']),
        valid_snapshot_chart=bool(row['chart']['available']))


def coverage_metrics(surface, trace):
    active = active_surface(surface)
    counts, areas, load = [], [], []
    for begin, end in zip(surface['offset'][:-1], surface['offset'][1:]):
        keep = active[begin:end]
        counts.append(int(keep.sum()))
        areas.append(float(surface['area'][begin:end][keep].sum()))
        load.append(float(np.sum(surface['area'][begin:end][keep]*surface['pressure'][begin:end][keep])))
    result = dict(frames_with_assigned_contact=int(np.count_nonzero(counts)),
                  saved_frame_denominator=len(counts), assigned_centroids_per_frame_mean=float(np.mean(counts)),
                  unique_assigned_pads=np.unique(surface['pad'][active]).tolist(),
                  peak_assigned_area_mm2=max(areas)*1e6,
                  palmar_load_recomputed_max_abs_error_n=float(np.max(abs(np.asarray(load)-trace['measured_palmar_load_n']))),
                  scope='Observed assigned contact area/count, not fraction of the complete object surface')
    if len(counts) > SNAPSHOT:
        full = float(trace['validation_full_hand_load_n'][SNAPSHOT])
        result['snapshot'] = dict(assigned_centroids=counts[SNAPSHOT], assigned_area_mm2=areas[SNAPSHOT]*1e6,
                                 palmar_load_n=load[SNAPSHOT], full_hand_load_n=full,
                                 palmar_to_full_hand_load_ratio=load[SNAPSHOT]/full if full > 1e-8 else None)
    return result


def empty_chart(template_faces, reason):
    arrays = dict(chart=np.zeros((1, 25, 4), np.float32), chart_world_m=np.zeros((25, 3)),
                  chart_hand_m=np.zeros((25, 3)), source_frame_indices=np.full((25, 3), -1, np.int64),
                  barycentric=np.zeros((25, 3)), support_edge_m=np.zeros(25),
                  support_triangles=np.empty((0, 3), np.int64), selected_pad=np.int64(-1),
                  touch_template_faces=np.asarray(template_faces, np.int64),
                  hand_pose_w=np.empty((0, 7)), source_frame_offset=np.int64(-1),
                  source_pos_hand_m=np.empty((0, 3)), source_area_m2=np.empty(0),
                  source_pressure_pa=np.empty(0), source_pad=np.empty(0, np.int64),
                  source_active=np.empty(0, bool), snapshot_available=np.bool_(False))
    return arrays, dict(available=False, reason=reason, selected_pad=None, candidate_count=0, patches=[])


def snapshot_chart(surface, hand_pose, template, template_faces, *, chart_builder=None):
    """Only sensor-local contact data and current measured hand pose enter here."""
    from .active3d_charts import supported_chart, validate_chart_masks
    builder = supported_chart if chart_builder is None else chart_builder

    arrays, info = empty_chart(template_faces, 'no_supported_pad_chart')
    pose = np.asarray(hand_pose, dtype=float)
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    active = active_surface(surface)
    arrays.update(hand_pose_w=pose[None], source_pos_hand_m=surface['pos'], source_area_m2=surface['area'],
                  source_pressure_pa=surface['pressure'], source_pad=surface['pad'], source_active=active,
                  snapshot_available=np.bool_(True))
    candidates = []
    for pad in sorted(np.unique(surface['pad'][active]).tolist()):
        selected = np.flatnonzero(active & (surface['pad'] == pad))
        try:
            chart, patch = builder(surface['pos'][selected], surface['area'][selected], template, MAX_EDGE_M)
            patch.update(pad=int(pad))
            if chart is not None:
                chart['source_indices'] = selected[chart['source_indices']]
                chart['support_triangles'] = selected[chart['support_triangles']]
                candidates.append((chart, patch))
        except Exception as exc:
            patch = dict(pad=int(pad), reason=f'{type(exc).__name__}: {exc}')
        info['patches'].append(patch)
    info['candidate_count'] = len(candidates)
    if not active.any():
        info['reason'] = 'no_active_assigned_contact'
    if candidates:
        chart, patch = sorted(candidates, key=lambda pair: (-pair[1]['chart_area_m2'], pair[1]['pad']))[0]
        world = chart['points'] @ rotation.T + pose[:3]
        canonical = (world-np.asarray(FIXTURE_CENTER_M))/METERS_PER_CANONICAL_UNIT
        replay = np.einsum('ni,nij->nj', chart['barycentric'], surface['pos'][chart['source_indices']])
        if not np.allclose(replay, chart['points'], atol=1e-12, rtol=0):
            raise AssertionError('Barycentric observation replay mismatch')
        if not np.allclose(canonical*METERS_PER_CANONICAL_UNIT+FIXTURE_CENTER_M, world, atol=1e-12, rtol=0):
            raise AssertionError('Known fixture canonical/world roundtrip mismatch')
        arrays['chart'][0, :, :3] = canonical
        arrays['chart'][0, :, 3] = 2
        arrays.update(chart_world_m=world, chart_hand_m=chart['points'],
                      source_frame_indices=chart['source_indices'], barycentric=chart['barycentric'],
                      support_edge_m=chart['support_edge_m'], support_triangles=chart['support_triangles'],
                      selected_pad=np.int64(patch['pad']))
        info.update(available=True, reason='supported_observation_interpolation', selected_pad=patch['pad'],
                    selected_chart_area_m2=patch['chart_area_m2'], selected_width_m=patch['width_m'])
    validate_chart_masks(arrays['chart'])
    return arrays, info


def report(root):
    from .official_active3d import read_template_obj

    attempts = expected_attempts()
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    if (protocol['attempt_order'] != [{k: row[k] for k in ('object_id', 'direction_index')} for row in attempts]
            or protocol['controls_per_attempt'] != 600 or protocol['sample_frame'] != SNAPSHOT
            or protocol['dt'] != .02 or protocol['fixture_drive']['target_load_n'] != TARGET_LOAD_N):
        raise ValueError('Saved protocol differs from the fixed four-attempt qualification')
    template, template_faces, _ = read_template_obj(EXPERIMENT/'vendor/Active-3D-Vision-and-Touch/pterotactyl/objects/touch_chart.obj')
    template, template_faces = template.numpy(), template_faces.verts_idx.numpy()
    if template.shape != (25, 3) or template_faces.shape != (32, 3):
        raise ValueError('Expected the original 25-vertex/32-face touch topology')
    charts = root/'charts'
    charts.mkdir(exist_ok=False)
    write_json(charts/'PROTOCOL.json', dict(qualification_criteria=QUALIFICATION_CRITERIA,
        attempt_order=attempts, snapshot_frame=SNAPSHOT, max_interpolation_edge_m=MAX_EDGE_M,
        selection='Largest supported chart area, then smallest pad ID; one chart or mask0 padding',
        geometry_source='Current measured contact field and hand pose only; no GT mesh read'))
    rows = []
    for expected in attempts:
        directory = root/expected['directory']
        row = dict(**expected, collection_complete=False, trace_available=False, surface_available=False, errors=[])
        arrays, chart_info = empty_chart(template_faces, 'snapshot_unavailable')
        try:
            attempt = json.loads((directory/'ATTEMPT.json').read_text())
            row['attempt_record'] = attempt
            row['collection_complete'] = bool(attempt['complete'])
            if attempt.get('error'):
                row['errors'].append('collection: '+attempt['error'])
        except Exception as exc:
            row['errors'].append(f'attempt_record: {type(exc).__name__}: {exc}')
        trace = None
        try:
            trace = read_npz(directory/'trace.npz', TRACE_KEYS)
            n = validate_trace(trace)
            row.update(trace_available=True, loads=trace_metrics(trace))
            if row.get('attempt_record', {}).get('recorded_controls') != n:
                row['collection_complete'] = False
                row['errors'].append('ATTEMPT recorded_controls differs from saved trace length')
            if row['collection_complete'] and n != 600:
                row['collection_complete'] = False
                row['errors'].append('Collection claims completion but trace has fewer than 600 controls')
        except Exception as exc:
            row['collection_complete'] = False
            row['errors'].append(f'trace: {type(exc).__name__}: {exc}')
            trace = None
        if trace is not None:
            try:
                surface = read_npz(directory/'observed_surface.npz', ('offset',)+SURFACE_KEYS)
                validate_surface(surface, len(trace['time_s']))
                row.update(surface_available=True, coverage=coverage_metrics(surface, trace))
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
        chart_path = charts/(expected['directory']+'.npz')
        np.savez_compressed(chart_path, **arrays)
        row.update(chart=chart_info, chart_file=str(chart_path.relative_to(root)),
                   recorded_complete_600=bool(row['collection_complete'] and row['trace_available'] and row['surface_available']))
        row['qualification_checks'] = qualification_checks(row)
        row['qualification_passed'] = all(row['qualification_checks'].values())
        rows.append(row)
    result = dict(complete=True, expected_attempts=4, cases=rows,
                  complete_recordings=sum(r['recorded_complete_600'] for r in rows),
                  available_snapshot_charts=sum(r['chart']['available'] for r in rows),
                  snapshot_frame=SNAPSHOT, snapshot_time_s=10., target_load_n=TARGET_LOAD_N,
                  chart_selection='Largest supported chart area, then smallest pad ID; one snapshot chart, no history',
                  chart_max_edge_m=MAX_EDGE_M, chart_mask='Whole chart 2, or all-zero xyz and mask0 on failure',
                  chart_coordinates='(hand-local contact @ measured hand rotation.T + measured position - [0,0,.75]) / .93',
                  model_forwards=0, learning_updates=0, new_physics_controls=0,
                  qualification_criteria=QUALIFICATION_CRITERIA,
                  qualification_passed=all(row['qualification_passed'] for row in rows),
                  scope='Fixed-object tactile acquisition/input diagnostic; not object prediction, carry or unseen-shape success',
                  limitations=['Four TRAIN attempts only; every runtime/observation failure retained.',
                               'Chart interpolation does not prove physical connectivity or absence of small holes.',
                               '25 interpolation vertices are not 25 independent measurements.',
                               'Contact coverage describes measured assigned cells, not global object-surface coverage.',
                               'Optical touch-CNN distribution compatibility and downstream learning remain untested.'])
    write_json(root/'RESULT.json', result)
    lines = ['# 固定物体触摸：四次采集资格检查', '',
             '前两个固定 TRAIN 物体 × 官方方向 0/49，各 600 控制步。物体固定在公开夹具坐标中；真实动态左手触摸，目标掌侧载荷 2 N。',
             '本报告只读取保存观测，0 新物理、0 模型前向、0 学习。Chart 是接触观测插值，不是预测出的整个物体。', '',
             f'完整记录 {result["complete_recordings"]}/4；10 秒可用 chart {result["available_snapshot_charts"]}/4；全部预先公开资格门槛通过：{result["qualification_passed"]}。所有失败均保留。', '',
             '资格门槛在实际采集前固定：四例均完整保存 600 帧且无 overflow；各例前 25 帧全手载荷 ≤0.001 N；全程掌侧／全手峰值均 ≤100 N；9.5–10s 的 25 帧中，掌侧载荷处于 [1.5,2.5] N 的比例 ≥0.8；固定 499 帧至少一张有效 chart。',
             '这些门槛仅检验四次夹具采集，不证明 48 个物体覆盖或学习准确率。精确规则保存于 charts/PROTOCOL.json 和 RESULT.json。', '',
             '| 物体 / 方向 | 完整 600 帧 | 9.5–10s 载荷均值 / 2N MAE | 11.5–12s 撤回载荷均值 / 2N MAE | 峰值掌侧 / 全手 N | 10s chart |',
             '|---|---|---|---|---|---|']
    def number(value):
        return '缺失' if value is None else f'{value:.3f}'
    for row in rows:
        loads = row.get('loads', {})
        values = []
        for key in ('pre_snapshot_0p5s', 'final_retraction_0p5s'):
            window = loads.get(key, {})
            values.append(f'{number(window.get("mean_load_n"))} / {number(window.get("mean_absolute_error_n"))} ({window.get("frames",0)}/25 帧)')
        lines.append(f'| {row["object_id"]} / {row["direction_index"]} | {row["recorded_complete_600"]} | {values[0]} | {values[1]} | '
                     f'{number(loads.get("peak_palmar_load_n"))} / {number(loads.get("peak_full_hand_load_n"))} | {row["chart"]["available"]} |')
    lines += ['', '10 秒之后为主动撤回段，因此轨迹末 0.5 秒对 2 N 的误差不代表保持阶段控制质量。',
              '每个 pad 单独使用原 supported_chart，固定 1.7 mm 边限制；按支持 chart 面积降序、pad ID 升序取一个。失败写全零 mask0，原始 25 顶点顺序及 32 面拓扑保留。',
              'Chart 仅使用当前接触点、面积、压力、pad 和当帧实测手位姿；物体 GT 网格未被打开。世界／canonical 变换使用公开夹具 [0,0,0.75] 和 0.93 m。',
              'RESULT.json 保留接触帧数、观测面积、分配区域、载荷覆盖、全部 pad 拒绝原因和运行错误；这不是全物体表面覆盖率。', '', '## 四例记录']
    for row in rows:
        lines += ['', f'### {row["directory"]}', f'Chart：{row["chart"]["reason"]}；pad={row["chart"]["selected_pad"]}。',
                  '资格逐项判定：'+json.dumps(row['qualification_checks'], ensure_ascii=False),
                  '接触覆盖：'+json.dumps(row.get('coverage', {}), ensure_ascii=False),
                  '错误：'+('; '.join(row['errors']) or '无记录错误')]
    lines += ['', '实际完整网格 still 回放由 --mode render 单独生成；固定帧 0/499/599，每张双视角。缺少真实帧时显示文字占位，不合成物理场景。']
    with (root/'REPORT.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    return result


def render(root):
    """Full actual mesh replay; the chart/report functions never call this path."""
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Rendering requires the retained explicit compute step')
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    from PIL import Image, ImageDraw
    import pyrender
    import trimesh
    from sugar_newton.hand.patches import load_hand_mesh
    from .render_device import configure_retained_egl
    from .render_object_state import text, look, material, BG, TEAL, AMBER

    result = json.loads((root/'RESULT.json').read_text())
    if len(result['cases']) != 4:
        raise ValueError('Expected all four saved case reports')
    out = root/'renders'
    out.mkdir(exist_ok=False)
    frame_dir = out/'frames'
    frame_dir.mkdir()
    renderer, render_setup_error = None, None
    hand = None
    try:
        configure_retained_egl()
        renderer = pyrender.OffscreenRenderer(784, 620, point_size=5.)
        hand = load_hand_mesh('left')
    except Exception as exc:
        render_setup_error = f'{type(exc).__name__}: {exc}'
    records = []
    try:
        for row in result['cases']:
            directory = root/row['directory']
            trace, object_data, error = None, None, render_setup_error
            trace_validated = False
            if error is None:
                try:
                    trace = read_npz(directory/'trace.npz', TRACE_KEYS)
                    validate_trace(trace)
                    trace_validated = True
                    object_data = read_npz(directory/'object_mesh.npz')
                    from .shape_fixture_assets import mesh_statistics
                    mesh_statistics(object_data['vertices'], object_data['faces'])
                    if (not np.array_equal(object_data['fixture_center_m'], np.asarray(FIXTURE_CENTER_M))
                            or not np.array_equal(object_data['fixture_quaternion_xyzw'], [0., 0., 0., 1.])):
                        raise ValueError('Unexpected fixed-object rendering transform')
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    if not trace_validated:
                        trace = None
            for frame in RENDER_FRAMES:
                image = Image.new('RGB', (1600, 960), BG)
                draw = ImageDraw.Draw(image)
                text(draw, (22, 14), '固定物体触摸 · 实际完整网格回放', 31)
                text(draw, (22, 61), f'{row["object_id"]} / 官方方向 {row["direction_index"]} | frame {frame} | 目标时刻 {(frame+1)*.02:.2f}s | 0 学习、无物体预测', 20)
                text(draw, (22, 103), '物体世界固定；蓝色为真实左手；青色为完整 ABC 物体', 21)
                frame_error = error
                if frame_error is None and frame >= len(trace['time_s']):
                    frame_error = f'实际记录仅 {len(trace["time_s"])} 帧，缺少 frame {frame}'
                rendered = False
                if frame_error is None:
                    try:
                        scene = pyrender.Scene(bg_color=[*np.asarray(BG)/255, 1], ambient_light=[.45, .45, .45])
                        def add_mesh(mesh, color, pose=None):
                            shader = material(color)
                            shader.doubleSided = True
                            return scene.add(pyrender.Mesh.from_trimesh(mesh, material=shader, smooth=False), pose=pose)
                        body = np.eye(4)
                        body[:3, 3] = object_data['fixture_center_m']
                        add_mesh(trimesh.Trimesh(object_data['vertices'], object_data['faces'], process=False), TEAL, body)
                        pose = trace['hand_pose_w'][frame]
                        hand_transform = np.eye(4)
                        hand_transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
                        hand_transform[:3, 3] = pose[:3]
                        add_mesh(hand, (64, 104, 153), hand_transform)
                        ground = trimesh.creation.box(extents=[3., 3., .01])
                        ground.apply_translation([0., 0., -.005])
                        add_mesh(ground, (220, 227, 232))
                        if frame == SNAPSHOT and row['chart']['available']:
                            chart = read_npz(root/row['chart_file'], ('chart_world_m',))['chart_world_m']
                            scene.add(pyrender.Mesh.from_points(chart, colors=np.tile(np.asarray([*AMBER, 255], np.uint8), (25, 1))))
                        center = np.asarray(FIXTURE_CENTER_M)
                        for eye, intensity in (([2., -3., 4.], 2.5), ([-2., 2., .3], 1.2)):
                            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=look(np.asarray(eye), center))
                        camera = scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(43), znear=.01, zfar=10.))
                        images = []
                        for delta in ([.7, -.9, .5], [-.7, .9, -.32]):
                            scene.set_pose(camera, look(center+delta, center))
                            rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
                            if rgb.std() < 5 or not np.isfinite(depth).all() or np.count_nonzero(depth) < 100:
                                raise RuntimeError('Empty or invalid full-mesh rendered image')
                            images.append(Image.fromarray(rgb))
                        for column, view in enumerate(images):
                            image.paste(view, (8+800*column, 140))
                        rendered = True
                    except Exception as exc:
                        frame_error = f'{type(exc).__name__}: {exc}'
                if not rendered:
                    text(draw, (60, 340), '此帧无可用实际渲染', 34, (180, 45, 45))
                    text(draw, (60, 395), '运行／记录／渲染失败，保留在四次分母；未合成物理画面。', 24)
                    for line, start in enumerate(range(0, len(str(frame_error)), 95)):
                        text(draw, (60, 450+line*30), str(frame_error)[start:start+95], 17)
                if trace is not None and frame < len(trace['time_s']):
                    text(draw, (22, 785), f'实际时刻 {trace["time_s"][frame]:.2f}s | 掌侧 {trace["measured_palmar_load_n"][frame]:.3f} N | 全手 {trace["validation_full_hand_load_n"][frame]:.3f} N | 目标 2.0 N', 23)
                text(draw, (22, 830), ('当帧 chart：可用；橙色为 25 个观测插值点' if row['chart']['available'] else '当帧 chart：不可用，保存全零 padding')
                     if frame == SNAPSHOT else '本帧未构建 chart；协议只在 frame 499（10 秒）取快照。', 23)
                text(draw, (22, 875), '采集完整 600 帧：'+str(row['recorded_complete_600'])+' | 物体固定夹具，不代表未知形状重建或搬运成功。', 21)
                text(draw, (22, 923), '完整原始物体／左手网格；实测手位姿；双视角；缺帧不补造。GT 物体网格仅在此渲染入口读取。', 17)
                path = frame_dir/f'{row["directory"]}_frame_{frame:03d}.png'
                image.save(path)
                records.append(dict(directory=row['directory'], frame=frame, image=str(path.relative_to(root)),
                                    actual_mesh_render=rendered, placeholder=not rendered, error=frame_error))
    finally:
        if renderer is not None:
            renderer.delete()
    summary = dict(complete=True, expected_stills=12, actual_mesh_stills=sum(r['actual_mesh_render'] for r in records),
                   placeholders=sum(r['placeholder'] for r in records), records=records,
                   scope='Saved physical mesh stills, two views each; no prediction and no continuous-video claim',
                   new_physics_controls=0, model_forwards=0, learning_updates=0)
    write_json(out/'RESULT.json', summary)
    html = ['<!doctype html><meta charset="utf-8"><title>固定物体真实触摸</title>',
            '<style>body{max-width:1600px;margin:24px auto;background:#f2f6f9;font:18px/1.7 system-ui}img{width:100%}</style>',
            '<h1>固定物体触摸：全部四次实际采集</h1>',
            '<p>每例固定 0/499/599 帧、双视角。青色物体固定，蓝色左手为真实动态位姿。没有模型预测；失败保留文字占位。</p>',
            '<p><a href="REPORT.md">数值报告</a> · <a href="RESULT.json">全部四例</a></p>']
    for record in records:
        html.append(f'<h2>{record["directory"]} / {record["frame"]}</h2><img loading="lazy" src="{record["image"]}">')
    with (root/'index.html').open('x') as stream:
        stream.write('\n'.join(html)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('report', 'render', 'all'), default='report')
    args = parser.parse_args()
    if args.mode in ('report', 'all'):
        report(args.root)
    if args.mode in ('render', 'all'):
        render(args.root)
