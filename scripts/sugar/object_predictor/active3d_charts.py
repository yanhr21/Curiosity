"""Measured contact centroids -> supported numeric charts for official Active3D.

This is an explicit interpolation diagnostic, not an optical touch-CNN output
or a proof of physical surface connectivity. No object truth is consumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .official_active3d import read_template_obj
from .shape_metric import rotation


def supported_chart(points, areas, template, max_edge_m=.0017):
    """Preserve official chart topology inside a small-triangle support union.

    The 1.7 mm edge cap is a declared resolution assumption for this diagnostic,
    based on the saved eight TRAIN observations, not a sensor guarantee. A chart
    is never extrapolated across the convex hull or a rejected large triangle.
    All 25 interpolated positions retain their three source points and weights.
    """
    points = np.asarray(points, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    if len(points) != len(areas) or not np.isfinite(points).all() or not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError('Invalid measured contact patch')
    unique, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    weight = np.bincount(inverse, weights=areas)
    if len(unique) < 6:
        return None, dict(reason='fewer_than_six_unique_points', points=len(unique))
    center = np.average(unique, axis=0, weights=weight)
    local = unique - center
    values, vectors = np.linalg.eigh((local * weight[:, None]).T @ local / weight.sum())
    rms = np.sqrt(np.maximum(values, 0))
    if rms[1] < .0002 or rms[0] / rms[1] > .35:
        return None, dict(reason='unresolved_or_thick_patch', rms_m=rms.tolist())
    basis = vectors[:, [2, 1]].copy()
    for column in range(2):
        # Stable sign in the measured hand frame; no object normal is needed.
        if basis[np.argmax(abs(basis[:, column])), column] < 0:
            basis[:, column] *= -1
    uv = local @ basis
    try:
        triangulation = Delaunay(uv)
    except QhullError:
        return None, dict(reason='degenerate_2d_triangulation')
    triangles = uv[triangulation.simplices]
    edges = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
    allowed = edges.max(1) <= max_edge_m
    if not allowed.any():
        return None, dict(reason='no_triangles_below_edge_cap')
    polygons = [Polygon(t) for t in triangles[allowed]]
    support = unary_union(polygons)
    # Start at the measured weighted center if supported; otherwise at the
    # centroid of the largest allowed triangle. Both are observation-only.
    chart_center = np.zeros(2)
    if not support.contains(Point(chart_center)):
        largest = max(polygons, key=lambda p: p.area)
        chart_center = np.asarray(largest.centroid.coords[0])
    template_uv = np.asarray(template, dtype=float)[:, 1:3]
    template_uv = (template_uv - template_uv[4]) / np.ptp(template_uv, axis=0).max()
    hull_indices = ConvexHull(template_uv).vertices
    lo, hi = 0., float(np.linalg.norm(np.ptp(uv, axis=0)))
    for _ in range(40):
        width = (lo + hi) / 2
        footprint = Polygon((chart_center + width * template_uv)[hull_indices])
        if support.covers(footprint):
            lo = width
        else:
            hi = width
    width = .95 * lo  # Margin inside supported interpolation, never extrapolation.
    if width < 1e-6:
        return None, dict(reason='no_resolved_supported_chart')
    queries = chart_center + width * template_uv
    if not support.covers(Polygon(queries[hull_indices])):
        raise AssertionError('Full chart footprint extends outside supported union')
    simplices = triangulation.find_simplex(queries)
    if (simplices < 0).any() or not allowed[simplices].all():
        raise AssertionError('Chart vertex lacks an allowed interpolation triangle')
    transform = triangulation.transform[simplices]
    bary = np.einsum('nij,nj->ni', transform[:, :2], queries - transform[:, 2])
    bary = np.column_stack((bary, 1 - bary.sum(1)))
    if (bary < -1e-10).any() or not np.allclose(bary.sum(1), 1):
        raise AssertionError('Invalid barycentric support')
    source = first[triangulation.simplices[simplices]]
    xyz = np.einsum('ni,nij->nj', bary, points[source])
    result = dict(points=xyz, source_indices=source, barycentric=bary,
                  support_edge_m=edges[simplices].max(1), support_triangles=first[triangulation.simplices[allowed]])
    info = dict(reason='accepted', input_points=len(points), unique_points=len(unique),
                rms_m=rms.tolist(), width_m=width, supported_area_m2=float(support.area),
                chart_area_m2=float(Polygon(queries[hull_indices]).area),
                allowed_triangles=int(allowed.sum()), rejected_triangles=int((~allowed).sum()),
                max_interpolation_edge_m=float(result['support_edge_m'].max()),
                full_footprint_supported=True, max_edge_m=max_edge_m)
    return result, info


def validate_chart_masks(charts):
    charts = np.asarray(charts).reshape(-1, 25, 4)
    if not np.isfinite(charts).all():
        raise ValueError('Nonfinite chart')
    if not np.isin(charts[..., 3], (0, 2)).all() or not (charts[..., 3] == charts[:, :1, 3]).all():
        raise ValueError('Each observed chart must have uniform mask2, each missing chart mask0')
    if np.any(charts[charts[..., 3] == 0, :3] != 0):
        raise ValueError('Missing chart coordinates must be zero, matching official padding')


def prepare(root, max_edge_m=.0017):
    source_root = Path('experiments/object_predictor_v1/research_review_20260916/chsel_qualification_inputs')
    collection = json.loads(Path('experiments/object_predictor_v1/response_surface_dataset_v1/COLLECTION_RESULT.json').read_text())
    records = {r['episode']: r for r in collection['records']}
    inputs = json.loads((source_root / 'RESULT.json').read_text())
    expected = [(ep, frame) for ep in range(5000, 5004) for frame in (1206, 2206)]
    if [(r['episode'], r['frame']) for r in inputs['cases']] != expected:
        raise ValueError('Expected the predeclared eight TRAIN clocks')
    template, faces, _ = read_template_obj(Path('experiments/object_predictor_v1/vendor/Active-3D-Vision-and-Touch/pterotactyl/objects/touch_chart.obj'))
    template = template.numpy()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in inputs['cases']:
        ep, frame = case['episode'], case['frame']
        record = records[ep]
        if record['split'] != 'train':
            raise ValueError('Only predeclared TRAIN diagnostics may enter here')
        source = Path(record['source'])
        # Only the allowed observation arrays are even opened from these files.
        with np.load(source / f'episode_{ep}.npz', allow_pickle=False) as z:
            hand_poses = z['hand_pose_w'][frame]
            loads = z['normal_load_n'][frame]
        with np.load(source / 'contact_surface.npz', allow_pickle=False) as z:
            begin, end = z['offset'][frame:frame + 2]
            data = {k: z[k][begin:end] for k in ('position_hand_frame_m', 'area_m2', 'normal_pressure_pa', 'pad', 'hand')}
        with np.load(source_root / case['input_file'], allow_pickle=False) as z:
            prior = z['prediction'].astype(float)
            assert np.array_equal(z['hand_pose_w'], hand_poses)
            prior_points = z['surface_points_current_left_hand_m']
        pad = data['pad']; assigned = (pad >= 0) & (pad < 54)
        active = np.zeros(len(pad), bool); active[assigned] = loads[pad[assigned]] > 1e-3
        active &= (data['area_m2'] > 0) & (data['normal_pressure_pa'] > 0)
        if not np.array_equal(data['hand'][active], pad[active] // 27) or int(active.sum()) != len(prior_points):
            raise ValueError('Saved predictor and chart observation selection differs')
        candidates, patches = [], []
        for p in sorted(set(pad[active].tolist())):
            selected = np.flatnonzero(active & (pad == p))
            chart, info = supported_chart(data['position_hand_frame_m'][selected], data['area_m2'][selected], template, max_edge_m)
            info.update(pad=p, hand=p // 27)
            patches.append(info)
            if chart is not None:
                chart['source_indices'] = selected[chart['source_indices']]
                chart['support_triangles'] = selected[chart['support_triangles']]
                info['support_triangle_frame_indices'] = chart['support_triangles'].tolist()
                chart.update(pad=p, hand=p // 27, info=info)
                candidates.append(chart)
        # Deterministic bilateral area ordering, independent of any truth/error.
        by_hand = [sorted((c for c in candidates if c['hand'] == h),
                          key=lambda c: (-c['info']['chart_area_m2'], c['pad'])) for h in (0, 1)]
        ordered = [part[i] for i in range(max(map(len, by_hand))) for part in by_hand if i < len(part)]
        rh = Rotation.from_quat(hand_poses[:, 3:]).as_matrix()
        selected_world = np.empty((int(active.sum()), 3))
        for h in (0, 1):
            side = data['hand'][active] == h
            selected_world[side] = data['position_hand_frame_m'][active][side] @ rh[h].T + hand_poses[h, :3]
        if not np.allclose((selected_world-hand_poses[0, :3]) @ rh[0], prior_points, atol=2e-7, rtol=0):
            raise ValueError('Contact coordinates disagree with original frozen predictor input')
        center_w = prior[:3] @ rh[0].T + hand_poses[0, :3]
        orientation_w = rh[0] @ rotation(prior[3:9])
        canonical_scale_m = 3.1 * np.exp(prior[9:12]).max()
        arrays = dict(prediction=prior, hand_pose_w=hand_poses, center_w=center_w,
                      orientation_w=orientation_w, canonical_scale_m=canonical_scale_m,
                      touch_template_faces=faces.verts_idx.numpy(),
                      source_position_hand_m=data['position_hand_frame_m'], source_pad=pad,
                      source_area_m2=data['area_m2'], source_active=active,
                      source_frame_offset=np.int64(begin))
        for arm, slots in (('t_p', 5), ('t_g', 20)):
            charts = np.zeros((slots, 25, 4), np.float32)
            support_indices = np.full((slots, 25, 3), -1, np.int64)
            bary = np.zeros((slots, 25, 3), np.float64)
            chart_pads = np.full(slots, -1, np.int64)
            edge = np.zeros((slots, 25))
            for i, chart in enumerate(ordered[:slots]):
                h = chart['hand']
                xyz_w = chart['points'] @ rh[h].T + hand_poses[h, :3]
                canonical = (xyz_w - center_w) @ orientation_w / canonical_scale_m
                charts[i, :, :3] = canonical; charts[i, :, 3] = 2
                support_indices[i] = chart['source_indices']; bary[i] = chart['barycentric']
                chart_pads[i] = chart['pad']; edge[i] = chart['support_edge_m']
                back = canonical * canonical_scale_m @ orientation_w.T + center_w
                if not np.allclose(back, xyz_w, atol=1e-12, rtol=0):
                    raise AssertionError('Canonical/world transform roundtrip mismatch')
            validate_chart_masks(charts)
            arrays.update({arm+'_charts': charts, arm+'_source_indices': support_indices,
                           arm+'_barycentric': bary, arm+'_pads': chart_pads,
                           arm+'_support_edge_m': edge})
        name = f'episode_{ep}_frame_{frame}.npz'
        np.savez_compressed(root / name, **arrays)
        rows.append(dict(episode=ep, frame=frame, input_file=name, source=str(source),
                         observed_centroids=int(active.sum()), accepted_charts=len(ordered), patches=patches,
                         t_p_pads=arrays['t_p_pads'].tolist(), t_g_pads=arrays['t_g_pads'].tolist(),
                         canonical_scale_m=float(canonical_scale_m)))
        print('PREPARED_REAL_CHARTS', ep, frame, len(ordered), flush=True)
    report = dict(complete=True, cases=rows, max_interpolation_edge_m=max_edge_m,
                  coordinate_frame='Frozen Utonia predicted center/orientation and 3.1 * max predicted dimension',
                  observations='Current frame only; same pressure/load/area filter as saved Utonia input',
                  masks='whole-chart 2 for supported interpolation; whole-chart zero xyz/mask0 for unused slots; no mask1',
                  selection='per pad, bilateral round-robin by supported chart area; no history or GT',
                  original_template_order=True, official_touch_template_faces=32,
                  model_forwards=0, model_parameter_updates=0, physics_steps=0,
                  limitations=['Interpolated centroids do not prove physical connectivity or absence of small holes.',
                               'Charts are much smaller than the initial template; the optical CNN output-scale distribution has not been measured.',
                               '25 interpolation vertices are not 25 independent measurements.',
                               'Multiple simultaneous patches replace the original finger/grasp arrangement.',
                               'No prediction accuracy, material inference or demo benefit established.'])
    (root / 'RESULT.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)
