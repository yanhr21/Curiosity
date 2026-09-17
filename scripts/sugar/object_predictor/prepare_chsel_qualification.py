"""Frozen full-Utonia observations for an official CHSEL adapter qualification.

Eight fixed TRAIN windows, no optimization. Backend input artifacts contain no
object ground truth; independent evaluator targets are stored separately.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .data import OBSERVATION_KEYS, history_indices, target_at
from .hand_surface_normals import query_observation_normals, verify_geometry_signature
from .model import ObjectPredictor
from .surface_data import select_surface_frames, _surface_points, encode_surface_observations, collate_surface


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main(args):
    assert os.environ.get('SLURM_STEP_ID')
    root = Path('experiments/object_predictor_v1')
    data = root / 'response_surface_dataset_v1'
    study = None
    if args.study_protocol:
        study = json.loads(Path(args.study_protocol).read_text())
        data = Path(study['data'])
        for name, expected in study['frozen_files'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    endpoint = root / 'training_response_surface_v1/surface_force/model.pt'
    if study:
        endpoint = Path(study['endpoint'])
    payload = torch.load(endpoint, map_location='cpu', weights_only=False)
    protocol = payload['protocol']; assert payload['step'] == 2000
    verify_geometry_signature(protocol['normal_geometry_signature'])
    assert protocol['input_representation'] == 'surface' and protocol['history'] == 32
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    model = ObjectPredictor(root / 'checkpoints/utonia.pth', 32,
                            protocol['deterministic_pooling']).cuda()
    model.load_state_dict(payload['model'], strict=True); model.eval()
    assert model.original_parameter_count == 137253744
    assert sum(p.numel() for p in model.parameters()) == 138407503
    collection = json.loads((data / 'COLLECTION_RESULT.json').read_text())
    assert collection['complete']
    records = collection['records']
    records = {r['episode']: r for r in records}
    summary = []; truths = []; hashes = {}
    if study:
        episodes = [c['episode'] for c in study['configurations']]
        frames = study['prediction_frames']
        assert sorted(records) == sorted(episodes)
        assert frames == sorted(set(frames)) and all(31 <= f < 2400 for f in frames)
    else:
        assert 5000 <= args.episode_start < args.episode_end <= 5032
        episodes = range(args.episode_start, args.episode_end)
        frames = (1206, 2206)
    for episode in episodes:
        record = records[episode]; assert record['split'] == ('test' if study else 'train')
        source = Path(record['source']); trace_path = source / f'episode_{episode}.npz'
        surface_path = source / 'contact_surface.npz'
        trace = read(trace_path); field = read(surface_path)
        for p in (trace_path, surface_path):
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        for frame in frames:
            indices = history_indices(frame, 32, protocol['history_policy'])
            obs = {key: trace[key][indices] for key in OBSERVATION_KEYS}
            site, contact, _ = query_observation_normals(obs)
            obs.update(hand_site_surface_normals_w=site, hand_contact_surface_normals_w=contact)
            surfaces = select_surface_frames(field, indices)
            prepared = [_surface_points(obs, s, i, contact[i]) for i, s in enumerate(surfaces)]
            row = encode_surface_observations(obs, surfaces, protocol['mode'],
                prepared_surfaces=prepared, time_scale_s=protocol['time_scale_s'])
            batch = {k: v.cuda() for k, v in collate_surface([row]).items()}
            with torch.inference_mode():
                torch.cuda.synchronize()
                start = time.perf_counter()
                prediction = model(batch).cpu().numpy()[0]
                prediction_seconds = time.perf_counter() - start
                repeat = model(batch).cpu().numpy()[0]
            assert np.array_equal(prediction, repeat)
            hand = obs['hand_pose_w'][-1, 0]
            rh = Rotation.from_quat(hand[3:]).as_matrix()
            sensed = prepared[-1]
            points = (sensed['position'] - hand[:3]) @ rh
            assert np.isfinite(points).all() and np.isfinite(prediction).all()
            name = f'episode_{episode}_frame_{frame}.npz'
            np.savez_compressed(out / name, prediction=prediction,
                surface_points_current_left_hand_m=points.astype(np.float32),
                surface_area_m2=sensed['area'].astype(np.float32),
                surface_hand=sensed['hand'], hand_pose_w=obs['hand_pose_w'][-1],
                timestamp_s=obs['timestamp_s'][-1], history_indices=indices)
            # Ground truth enters this separate evaluation artifact only.
            truths.append(target_at(trace, frame))
            summary.append(dict(episode=episode, frame=frame, input_file=name,
                contact_points=len(points), contact_points_each_hand=[int((sensed['hand'] == h).sum()) for h in (0, 1)],
                repeat_max_difference=float(np.max(abs(prediction - repeat))),
                model_forward_and_cpu_transfer_seconds=prediction_seconds,
                input_sha256=hashlib.sha256((out / name).read_bytes()).hexdigest()))
            print('QUALIFICATION_INPUT', summary[-1], flush=True)
    np.savez_compressed(out / 'evaluation_only.npz', target=np.stack(truths),
        episode=np.array([r['episode'] for r in summary]), frame=np.array([r['frame'] for r in summary]))
    report = dict(complete=True, cases=summary, endpoint=str(endpoint),
        full_backbone_parameters=137253744, full_model_parameters=138407503,
        endpoint_sha256=hashlib.sha256(endpoint.read_bytes()).hexdigest(),
        original_training_steps=2000, optimizer_updates=0, model_forwards=2*len(summary), physics_steps=0,
        source_sha256=hashes, scope='Fixed TRAIN adapter inputs only. Backend inputs exclude object ground truth; separate evaluator labels may be used in an explicitly declared TRAIN calibration. No downstream registration results yet.')
    if study:
        report.update(study_protocol=args.study_protocol, data=str(data),
            scope='Prospective frozen known-mesh TEST inputs, all declared clocks and all16 attempts. No calibration or parameter changes.')
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', default='experiments/object_predictor_v1/research_review_20260916/chsel_qualification_inputs')
    ap.add_argument('--episode-start', type=int, default=5000)
    ap.add_argument('--episode-end', type=int, default=5004)
    ap.add_argument('--study-protocol', help='Explicit prospective data and clock manifest; default fixed TRAIN qualification is unchanged')
    main(ap.parse_args())
