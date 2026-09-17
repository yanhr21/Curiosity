"""Connect frozen Utonia observations to the complete official CHSEL optimizer.

This is a known-mesh registration qualification, not learned shape completion.
Object labels are opened only after every observation-only candidate is saved.
"""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import time

import chsel
import numpy as np
import open3d as o3d
import pytorch_volumetric as pv
import pytorch_kinematics.transforms as pk_transforms
from pytorch_kinematics.transforms.rotation_conversions import axis_angle_to_matrix
import torch
from scipy.spatial import cKDTree

from .shape_metric import rotation, cloud
from .report_grip_transfer import errors


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def homogeneous(state):
    h = np.eye(4)
    h[:3, :3] = rotation(state[3:9])
    h[:3, 3] = state[:3]
    return h


def predict_from_inverse(h, original):
    pose = np.linalg.inv(h)
    pred = original.copy()
    pred[:3] = pose[:3, 3]
    # Project convention is two COLUMNS, unlike pytorch_kinematics' rows.
    pred[3:9] = pose[:3, :2].T.reshape(6)
    return pred


class FirstStartRecorder:
    """Observe the official optimizer without changing its loss or gradients."""
    def __init__(self, cost):
        self.cost = cost
        self.history = []

    def __getattr__(self, name):
        return getattr(self.cost, name)

    def __call__(self, R, T, s, other_info=None):
        value = self.cost(R, T, s, other_info=other_info)
        self.history.append((R[0].detach().cpu().numpy().copy(),
                             T[0].detach().cpu().numpy().copy(),
                             float(value[0].detach())))
        return value


def main(args):
    assert os.environ.get('SLURM_STEP_ID'), 'Use retained compute step'
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    # CHSEL calls this public transform symbol, but released PK wheels omit its
    # re-export. Bind the EXISTING official implementation; no replacement math.
    if not hasattr(pk_transforms, 'axis_angle_to_matrix'):
        pk_transforms.axis_angle_to_matrix = axis_angle_to_matrix
    assert pk_transforms.axis_angle_to_matrix is axis_angle_to_matrix
    logging.basicConfig(level=logging.INFO)
    src = Path(args.inputs); out = Path(args.output); out.mkdir(exist_ok=False)
    manifest = json.loads((src / 'RESULT.json').read_text())
    cases = manifest['cases']
    study = None
    if args.study_protocol:
        study = json.loads(Path(args.study_protocol).read_text())
        assert args.local_only and args.floor and args.fusion_scales == study['fusion_scales']
        assert manifest['study_protocol'] == args.study_protocol
        for name, expected in study['frozen_files'].items():
            assert digest(Path(name)) == expected, name
        expected_cases = [(c['episode'], f) for c in study['configurations'] for f in study['prediction_frames']]
    else:
        expected_cases = [(e, f) for e in range(5000, 5004) for f in (1206, 2206)]
    assert [(r['episode'], r['frame']) for r in cases] == expected_cases
    mesh_path = Path('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz')
    mesh = read(mesh_path); vertices = mesh['vertices'].astype(np.float64)
    faces = mesh['faces'].astype(np.int32)
    dims = np.ptp(vertices, axis=0)
    vertices -= (vertices.max(0) + vertices.min(0)) / 2
    assert vertices.shape == (15626, 3) and faces.shape == (31248, 3)
    protocol = dict(seed=20260916, batch=30, qd_iterations=0 if args.local_only else 100,
        register_iterations=1, qd_measure='official PositionMeasure(3): inverse-pose translation',
        compatibility='Expose official PK rotation_conversions.axis_angle_to_matrix at missing transforms re-export; identical function object',
        sdf='official MeshSDF triangle queries, CPU float32 with seeded ray perturbations; complete mesh and predicted scale',
        semantics=('current-frame SURFACE plus 10mm eroded hand core FREE' if args.free_points else
            'current-frame SURFACE only; no invented free space or static history'),
        free_points_root=args.free_points, free_voxel_resolution_m=.003 if args.free_points else None,
        cost='official VolumetricDoubleDirectCost' if args.free_points else 'official VolumetricDirectSDFCost',
        selection='argmin official observation cost over original prior, returned batch, and QD archive',
        size_mass='frozen original Utonia predictions', mesh_sha256=digest(mesh_path),
        chsel_revision=subprocess.check_output(['git', '-C',
            'experiments/object_predictor_v1/vendor/chsel', 'rev-parse', 'HEAD'], text=True).strip(),
        versions={p: importlib.metadata.version(p) for p in
            ('torch', 'numpy', 'open3d', 'ribs', 'pytorch-volumetric', 'pytorch-kinematics')},
        qualification='8 fixed TRAIN clocks only; no generalization or arbitrary shape claim',
        model_updates=0, model_forwards=0, physics_steps=0)
    protocol['local_only'] = args.local_only
    if args.local_only:
        assert not args.free_points, 'This fixed ablation uses the original surface-only condition'
        protocol.update(selection='argmin official observation cost over original prior and all 30 local endpoints',
            method='Official CHSEL register_single(skip_qd=True); unchanged local SGD defaults and initial perturbations',
            secondary='Original-start endpoint versus untouched prior, selected only by observation cost',
            trace='Every local cost call records first start before the Adam update; final endpoint also saved',
            qualification='8 fixed TRAIN clocks; local-stage ablation, not full CHSEL QD or prior-constrained estimation')
    fusion_scales = None
    assert not args.floor or args.fusion_scales, 'Known-floor factor belongs to the explicit fusion adapter'
    if args.fusion_scales:
        from .fusion_cost import PriorContactCost, qualify_rotation, qualify_cost
        assert args.local_only and not args.free_points
        fusion_scales = json.loads(Path(args.fusion_scales).read_text())
        assert fusion_scales['complete'] and fusion_scales['qualification_excluded_episodes'] == [5000,5001,5002,5003]
        assert {c['episode'] for c in fusion_scales['cases']} == set(range(5004,5032))
        protocol.update(fusion_scales_file=args.fusion_scales, fusion_scales_sha256=digest(Path(args.fusion_scales)),
            cost='Project empirical prior/contact fusion adapter, official closest-triangle queries and TorchLie log',
            selection='Minimum total fusion objective over original prior and 30 official local endpoints',
            method='Project fusion cost with unchanged official CHSEL local optimizer; no QD, no independent Bayesian MAP claim',
            qualification='8 fixed TRAIN clocks, empirical neural/contact fusion qualification, not generalized reconstruction',
            rotation_qualification=qualify_rotation('cuda'))
        if args.floor:
            protocol.update(known_floor=True, floor_world_z_m=0., floor_sigma_m=.001,
                floor_factor='0.5*(relu(-minimum_world_z(full predicted mesh))/.001m)^2; active without object GT masks',
                floor_source='scripts/sugar/object_predictor/support_scene.py: builder.add_ground_plane(height=0.)',
                floor_source_sha256=digest(Path('scripts/sugar/object_predictor/support_scene.py')))
    if study:
        protocol.update(study_protocol=args.study_protocol, data=study['data'],
            qualification='Prospective fresh configurations of the same known mesh; fixed clocks, all attempts retained',
            no_contact='Current filtered surface empty: unchanged current Utonia output; no registration, no invented constraints',
            gradient_checks='Existing fixed fusion-cost qualification retained; no repeated finite differences on evaluation clocks')
    (out / 'executed_adapter.py').write_bytes(Path(__file__).read_bytes())
    (out / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2))
    rows = []; originals = []; predictions = []
    for index, case in enumerate(cases):
        seed = protocol['seed'] + index
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        p = src / case['input_file']; assert digest(p) == case['input_sha256']
        data = read(p); original = data['prediction']
        assert original.shape == (13,) and np.isfinite(original).all()
        scaled = vertices * (np.exp(original[9:12]) / dims)
        if study and len(data['surface_points_current_left_hand_m']) == 0:
            # Hand CAD anchors and past contacts are not current tactile contacts.
            # A missing observation must not be replaced by a fabricated surface.
            from scipy.spatial.transform import Rotation
            inverse = np.linalg.inv(homogeneous(original))
            hand = data['hand_pose_w'][0]
            Rh = Rotation.from_quat(hand[3:]).as_matrix()
            object_in_hand = scaled @ rotation(original[3:9]).T + original[:3]
            min_z = float((object_in_hand @ Rh.T + hand[:3])[:, 2].min())
            np.savez_compressed(out / case['input_file'], original_prediction=original,
                prediction=original.copy(), candidates_hand_to_object=inverse[None],
                candidate_cost=np.empty(0, np.float32), selected_index=0,
                qd_archive=np.empty((0, 9), np.float32),
                points_current_left_hand_m=data['surface_points_current_left_hand_m'],
                surface_hand=data['surface_hand'], hand_pose_w=data['hand_pose_w'],
                timestamp_s=data['timestamp_s'])
            row = dict(episode=case['episode'], frame=case['frame'], seed=seed,
                status='no_current_contact_unchanged_prior', candidate_count=1,
                selected_index=0, original_cost=None, selected_cost=None,
                original_contact_sdf_cm=None, selected_contact_sdf_cm=None,
                seconds_including_sdf_and_checks=0., original_start_or_prior_index=0,
                minimum_world_z_m_prior_then_selected=[min_z, min_z])
            rows.append(row); originals.append(original); predictions.append(original.copy())
            (out / 'OBSERVATION_PROGRESS.json').write_text(json.dumps(rows, indent=2))
            print('REGISTER_NO_CONTACT', case['episode'], case['frame'], flush=True)
            continue
        obj_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(scaled), o3d.utility.Vector3iVector(faces))
        begin = time.monotonic()
        sdf = pv.MeshSDF(pv.MeshObjectFactory(mesh=obj_mesh))
        points = torch.as_tensor(data['surface_points_current_left_hand_m'],
            dtype=torch.float32, device='cuda')
        inv = torch.as_tensor(np.linalg.inv(homogeneous(original)),
            dtype=torch.float32, device='cuda')
        # Check the actual row/column convention before any registration.
        object_points = chsel.apply_similarity_transform(points[None], inv[None, :3, :3], inv[None, :3, 3])
        expected = (points.cpu().numpy() - original[:3]) @ rotation(original[3:9])
        frame_error = float(np.max(abs(object_points[0].cpu().numpy() - expected)))
        assert frame_error < 1e-6
        vertex_sdf, _ = sdf(torch.as_tensor(scaled, device='cuda', dtype=torch.float32))
        vertex_residual = float(vertex_sdf.abs().max())
        assert vertex_residual < 1e-5
        query = object_points.detach().clone().requires_grad_(True)
        vals, sdf_gradient = sdf(query)
        derivative, = torch.autograd.grad(vals.sum(), query)
        assert torch.isfinite(derivative).all()
        assert torch.allclose(derivative, sdf_gradient, atol=1e-6, rtol=1e-5)
        init_contact_cm = float(vals.detach().abs().mean() * 100)
        semantic_points = points
        semantics = torch.zeros(len(points), dtype=torch.long, device=points.device)
        options = {}; free_count = 0
        if args.free_points:
            free = read(Path(args.free_points) / case['input_file'])['hand_core_10mm']
            assert free.shape == (273, 3) and np.isfinite(free).all()
            free = torch.as_tensor(free, device=points.device, dtype=points.dtype)
            free_count = len(free)
            semantic_points = torch.cat([points, free])
            semantics = torch.cat([semantics, torch.full((len(free),), chsel.SemanticsClass.FREE.value,
                dtype=torch.long, device=points.device)])
            options = dict(cost=chsel.VolumetricDoubleDirectCost, free_voxels_resolution=.003)
        solver = chsel.CHSEL(sdf, positions=semantic_points, semantics=semantics,
            qd_measure=chsel.PositionMeasure(3, device=points.device, dtype=points.dtype),
            do_qd=True, qd_iterations=100, savedir=str(out / f'case_{index}'), **options)
        gradient_checks = None
        if fusion_scales is not None:
            solver.volumetric_cost = PriorContactCost(sdf.obj_factory, points,
                data['surface_area_m2'], data['surface_hand'], original[:3],
                rotation(original[3:9]), fusion_scales,
                floor_vertices=scaled if args.floor else None,
                hand_pose_w=data['hand_pose_w'][0] if args.floor else None)
            if not study:
                gradient_checks = qualify_cost(solver.volumetric_cost)
                (out / f'gradient_case_{index}.json').write_text(json.dumps(gradient_checks, indent=2))
        if args.free_points:
            surface_cost = solver.volumetric_cost._cost_sdf(inv[None,:3,:3], inv[None,:3,3], None)
            baseline = json.loads((Path(args.free_points).parent / 'RESULT.json').read_text())['cases'][index]
            assert baseline['episode'] == case['episode'] and baseline['frame'] == case['frame']
            assert abs(float(surface_cost[0])-baseline['original_cost']) < 1e-5
        initial = chsel.reinitialize_transform_estimates(30, inv)
        assert torch.equal(initial[0], inv)
        print('REGISTER_BEGIN', case['episode'], case['frame'], flush=True)
        recorder = None
        if args.local_only:
            recorder = FirstStartRecorder(solver.volumetric_cost)
            solver.volumetric_cost = recorder
            result, archive = solver.register_single(initial_tsf=initial, skip_qd=True)
            solver.volumetric_cost = recorder.cost
        else:
            result, archive = solver.register(iterations=1, initial_tsf=initial)
        returned = chsel.solution_to_world_to_link_matrix(result)
        if args.local_only:
            assert archive is None and returned.shape == (30, 4, 4)
            archive = np.empty((0, 9), dtype=np.float32)
            archive_h = returned[:0]
            trace_R, trace_T, trace_cost = zip(*recorder.history)
            np.savez_compressed(out / ('trace_' + case['input_file']),
                first_start_R=np.stack(trace_R), first_start_T=np.stack(trace_T),
                first_start_cost=np.asarray(trace_cost), initial_hand_to_object=initial.cpu().numpy(),
                local_endpoints_hand_to_object=returned.cpu().numpy())
            assert np.max(abs(trace_R[0] - initial[0,:3,:3].cpu().numpy())) < 1e-6
            assert np.array_equal(trace_T[0], initial[0,:3,3].cpu().numpy())
        else:
            archive_h = chsel.continuous_representation_to_H(archive,
                device=points.device, dtype=points.dtype)
        candidates = torch.cat([inv[None], returned, archive_h]).detach()
        assert torch.isfinite(candidates).all() and (args.local_only or len(archive_h) > 0)
        assert torch.allclose(torch.linalg.det(candidates[:, :3, :3]),
            torch.ones(len(candidates), device=points.device), atol=1e-4, rtol=1e-4)
        # Chunking limits transient tensors, without changing the candidate set.
        costs = torch.cat([solver.evaluate_homogeneous(h).detach()
            for h in candidates.split(30)])
        assert costs.shape == (len(candidates),) and torch.isfinite(costs).all()
        selected = int(costs.argmin())
        candidate_np = candidates.cpu().numpy(); cost_np = costs.cpu().numpy()
        prediction = predict_from_inverse(candidate_np[selected], original)
        assert np.array_equal(prediction[9:], original[9:])
        reconstructed = homogeneous(prediction)
        assert np.max(abs(reconstructed @ candidate_np[selected] - np.eye(4))) < 1e-5
        selected_pts = chsel.apply_similarity_transform(points[None],
            candidates[selected:selected+1, :3, :3], candidates[selected:selected+1, :3, 3])
        final_sdf, _ = sdf(selected_pts)
        final_contact_cm = float(final_sdf.detach().abs().mean() * 100)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - begin
        obj_to_hand = np.linalg.inv(candidate_np)
        # Descriptive spread only; not a calibrated confidence interval.
        eligible = cost_np <= cost_np.min() * 1.05 + 1e-6
        unique_indices = np.unique(candidate_np.round(6).reshape(len(candidates), -1), axis=0, return_index=True)[1]
        unique_eligible = unique_indices[eligible[unique_indices]]
        center_std_cm = obj_to_hand[unique_eligible, :3, 3].std(axis=0) * 100
        np.savez_compressed(out / case['input_file'], original_prediction=original,
            prediction=prediction, candidates_hand_to_object=candidate_np,
            candidate_cost=cost_np, selected_index=selected, qd_archive=archive,
            points_current_left_hand_m=points.cpu().numpy(),
            surface_hand=data['surface_hand'], hand_pose_w=data['hand_pose_w'],
            timestamp_s=data['timestamp_s'])
        row = dict(episode=case['episode'], frame=case['frame'], seed=seed,
            status='registered_current_contact',
            candidate_count=len(candidates), unique_candidates_rounded_1e6=len(unique_indices),
            free_observations=free_count,
            qd_archive_count=len(archive_h),
            selected_index=selected, original_cost=float(cost_np[0]), selected_cost=float(cost_np[selected]),
            original_contact_sdf_cm=init_contact_cm, selected_contact_sdf_cm=final_contact_cm,
            low_cost_candidate_count=int(eligible.sum()), unique_low_cost_candidates=len(unique_eligible),
            center_std_cm=center_std_cm.tolist(),
            seconds_including_sdf_and_checks=elapsed, frame_roundtrip_error_m=frame_error,
            mesh_vertex_sdf_max_m=vertex_residual)
        if args.local_only:
            row.update(local_cost_calls=len(recorder.history),
                original_start_final_cost=float(cost_np[1]),
                original_start_or_prior_index=int(np.argmin(cost_np[:2])),
                original_start_final_center_shift_cm=float(np.linalg.norm(
                    obj_to_hand[1,:3,3]-obj_to_hand[0,:3,3])*100))
        if fusion_scales is not None:
            chosen = candidates[[0, selected]]
            row['fusion_terms_prior_then_selected'] = {k: v.detach().cpu().tolist() for k, v in
                solver.volumetric_cost.terms(chosen[:,:3,:3], chosen[:,:3,3]).items()}
            if gradient_checks is not None:
                row['gradient_max_scaled_difference'] = max(c['max_scaled_difference'] for c in gradient_checks)
            if args.floor:
                row['minimum_world_z_m_prior_then_selected'] = solver.volumetric_cost.minimum_world_z(
                    chosen[:,:3,:3],chosen[:,:3,3]).detach().cpu().tolist()
        rows.append(row); originals.append(original); predictions.append(prediction)
        (out / 'OBSERVATION_PROGRESS.json').write_text(json.dumps(rows, indent=2))
        print('REGISTER_DONE', json.dumps(row), flush=True)

    # Evaluation labels cannot affect fitting, candidate choice, or stopping.
    truth = read(src / 'evaluation_only.npz')
    assert truth['episode'].tolist() == [c['episode'] for c in cases]
    assert truth['frame'].tolist() == [c['frame'] for c in cases]
    targets = truth['target']; originals = np.stack(originals); predictions = np.stack(predictions)
    arms = [('utonia', originals), ('chsel', predictions)]
    if args.local_only:
        first_start = []; first_endpoint = []
        for case, row, original in zip(cases, rows, originals):
            saved = read(out / case['input_file'])
            first_start.append(predict_from_inverse(saved['candidates_hand_to_object'][
                row['original_start_or_prior_index']], original))
            first_endpoint.append(predict_from_inverse(saved['candidates_hand_to_object'][
                min(1, len(saved['candidates_hand_to_object'])-1)], original))
        arms.append(('original_start_or_prior', np.stack(first_start)))
        arms.append(('original_start_endpoint', np.stack(first_endpoint)))
    for name, estimates in arms:
        stats = errors(estimates, targets)
        mesh_errors = []
        for estimate, target in zip(estimates, targets):
            pc = cloud(vertices, estimate, dims); tc = cloud(vertices, target, dims)
            distances = np.r_[cKDTree(tc).query(pc, workers=4)[0], cKDTree(pc).query(tc, workers=4)[0]]
            mesh_errors.append(float(distances.mean() * 100))
        stats['symmetric_nearest_vertex_cm'] = np.array(mesh_errors)
        for i, row in enumerate(rows):
            row[name] = {key: float(value[i]) for key, value in stats.items()}
    mean = {name: {key: float(np.mean([r[name][key] for r in rows]))
        for key in rows[0][name]} for name, _ in arms}
    qualified = bool(mean['chsel']['center_cm'] < mean['utonia']['center_cm'] and
        mean['chsel']['symmetric_nearest_vertex_cm'] < mean['utonia']['symmetric_nearest_vertex_cm'])
    report = dict(complete=True, interface_passed=True, mean=mean, cases=rows,
        geometry_improves_mean_train_qualification=qualified,
        scope=protocol['qualification'], model_updates=0, physics_steps=0,
        selection_uses_ground_truth=False,
        next_action='Inspect all eight actual mesh comparisons; preserve failures. '
            'If mean center and mesh errors do not both improve, diagnose observability '
            'before any wider evaluation; no threshold or candidate selection tuning to labels.')
    if study:
        del report['geometry_improves_mean_train_qualification']
        report.update(mean_center_and_mesh_improve=qualified,
            next_action='Report all16 episodes, four paired geometry groups, continuous world-pose errors and actual renders. These means alone do not establish full scientific acceptance.')
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2))
    print('QUALIFICATION_COMPLETE', json.dumps(mean), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='experiments/object_predictor_v1/research_review_20260916/chsel_qualification_inputs')
    ap.add_argument('--output', required=True)
    ap.add_argument('--free-points', help='Saved observation-only hand-core inputs from fixed diagnosis')
    ap.add_argument('--local-only', action='store_true', help='Fixed official local-stage ablation; no QD')
    ap.add_argument('--fusion-scales', help='Disjoint TRAIN empirical scales; enables explicitly new fusion cost adapter')
    ap.add_argument('--floor', action='store_true', help='Fixed known-environment floor nonpenetration penalty')
    ap.add_argument('--study-protocol', help='Prospective fresh dataset and fixed inference clocks; retain original qualification defaults')
    main(ap.parse_args())
