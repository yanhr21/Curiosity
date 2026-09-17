"""Frozen full-Utonia transfer to saved carrying fixtures; original sensor codec.

All declared cases/clocks are retained. Controller/ground-truth metadata only
stratifies errors after inference, never enters sampling or network inputs.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import OBSERVATION_KEYS, collate, encode_observations, history_indices, target_at
from .hand_surface_normals import query_observation_normals, verify_geometry_signature
from .model import ObjectPredictor

NORMAL_KEYS = ('hand_site_surface_normals_w', 'hand_contact_surface_normals_w')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(root, protocol):
    cache = root / 'cache'; cache.mkdir(exist_ok=False)
    records = []
    for episode in protocol['episodes']:
        source = Path(protocol['source']) / 'cases' / f'episode_{episode}'
        path = source / f'episode_{episode}.npz'
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if len(arrays['timestamp_s']) != protocol['controls_per_case']:
            raise ValueError(f'Incomplete source episode {episode}')
        obs = {k: arrays[k] for k in OBSERVATION_KEYS}
        site, contact, details = query_observation_normals(obs)
        arrays.update(zip(NORMAL_KEYS, (site, contact)))
        # Verify cached versus the original direct encoder at fixed early,
        # prelift and late clocks, including real contact observations.
        checks = []
        for frame in protocol['cache_check_frames']:
            idx = history_indices(frame, 32, 'episode_uniform_recent')
            raw = {k: arrays[k][idx] for k in OBSERVATION_KEYS}
            cached = {**raw, **{k: arrays[k][idx] for k in NORMAL_KEYS}}
            for mode in protocol['model_modes']:
                direct = encode_observations(raw, mode, time_scale_s=150., normal_policy='hand_surface')
                encoded = encode_observations(cached, mode, time_scale_s=150., normal_policy='hand_surface')
                checks.append(all(np.array_equal(direct[k], encoded[k]) for k in ('coord', 'feat')))
        if not all(checks):
            raise ValueError(f'Cached normal input differs in episode {episode}')
        np.savez_compressed(cache / f'episode_{episode}.npz', **arrays)
        records.append(dict(episode=episode, source=str(path), source_sha256=digest(path),
                            result_sha256=digest(source / 'RESULT.json'),
                            cache_sha256=digest(cache / f'episode_{episode}.npz'),
                            direct_encoder_checks=len(checks), all_exact=True, normal_details=details))
        print('CACHE_COMPLETE', episode, flush=True)
    report = dict(complete=len(records)==32, records=records, new_physics=0, new_optimizer_updates=0)
    (root / 'CACHE.json').write_text(json.dumps(report, indent=2))


def load_models(root, protocol, resume=False):
    models = {}; records = []
    for mode in protocol['model_modes']:
        endpoint = Path(protocol['endpoints']) / mode / 'model.pt'
        payload = torch.load(endpoint, map_location='cpu', weights_only=False)
        p = payload['protocol']
        expected = dict(mode=mode, history=32, history_policy='episode_uniform_recent',
                        time_scale_s=150., normal_policy='hand_surface', deterministic_pooling=True)
        for k, value in expected.items():
            if p[k] != value: raise ValueError(f'Endpoint protocol mismatch {mode}/{k}')
        if payload['step'] != 2000: raise ValueError('Expected declared final2000 endpoint')
        verify_geometry_signature(p['normal_geometry_signature'])
        model = ObjectPredictor(protocol['base_checkpoint'], history=32, deterministic_pooling=True).cuda()
        model.load_state_dict(payload['model'], strict=True); model.eval()
        if model.original_parameter_count != 137253744 or sum(x.numel() for x in model.parameters()) != 138407503:
            raise ValueError('Full released backbone/adapter parameter count differs')
        model.requires_grad_(False)
        models[mode] = model
        records.append(dict(mode=mode, endpoint=str(endpoint), endpoint_sha256=digest(endpoint),
                            original_steps=payload['step'], full_parameters=138407503, strict_load=True))
        del payload; gc.collect()
        print('FULL_MODEL_LOADED', mode, flush=True)
    record = dict(models=records, base_sha256=digest(protocol['base_checkpoint']),
                  optimizer_created=False, new_optimizer_updates=0)
    if resume and record != json.loads((root / 'MODELS.json').read_text()):
        raise ValueError('Reloaded models differ from original frozen endpoints')
    (root / ('MODELS_RESUME.json' if resume else 'MODELS.json')).write_text(json.dumps(record, indent=2))
    return models


def verify_recovery(root, protocol):
    incident = json.loads((root / 'interruptions/job297659/INCIDENT.json').read_text())
    for name, expected in incident['completed_prediction_files_sha256'].items():
        if digest(root / name) != expected: raise ValueError(f'Changed completed result {name}')
    cache = json.loads((root / 'CACHE.json').read_text())
    if not cache['complete'] or len(cache['records']) != 32: raise ValueError('Incomplete cache')
    for record in cache['records']:
        if digest(record['source']) != record['source_sha256']: raise ValueError('Source changed')
        path = root / 'cache' / f"episode_{record['episode']}.npz"
        if digest(path) != record['cache_sha256']: raise ValueError('Cache changed')
    completed = incident['completed_cases']
    for ep in completed:
        case = root / 'evaluation' / f'episode_{ep}'
        context = json.loads((case / 'CONTEXT.json').read_text())
        if [r['frame'] for r in context] != protocol['frames']: raise ValueError('Incomplete context')
        for mode in protocol['evaluation_modes']:
            with np.load(case / f'{mode}.npz') as z:
                if not np.array_equal(z['frame'], protocol['frames']) or z['prediction'].shape != (95, 13):
                    raise ValueError('Incomplete result arrays')
    # Do not silently overwrite partially written cases after a resource kill.
    for ep in protocol['episodes']:
        path = root / 'evaluation' / f'episode_{ep}'
        if ep not in completed and path.exists():
            raise ValueError(f'Unaccounted partial case {ep}; preserve and audit before resuming')
    (root / 'RECOVERY.json').write_text(json.dumps(dict(complete=True, reused_cases=completed,
        unchanged_cache_cases=32, unchanged_prediction_files=len(incident['completed_prediction_files_sha256']),
        discarded_unpersisted_forward_bounds=incident['unpersisted_next_case_forward_count_bounds']), indent=2))
    return completed


@torch.inference_mode()
def replay_saved(root, protocol, models):
    rows = []
    for ep in (5000, 5009):
        with np.load(root / 'cache' / f'episode_{ep}.npz') as z: arrays = dict(z)
        for frame in (31, 1156):
            idx = history_indices(frame, 32, 'episode_uniform_recent')
            obs = {k: arrays[k][idx] for k in OBSERVATION_KEYS + NORMAL_KEYS}
            for mode in protocol['evaluation_modes']:
                key = 'geometry_contact_force' if mode == 'force_zero' else mode
                row = encode_observations(obs, key, force_gain=0. if mode == 'force_zero' else 1.,
                                          time_scale_s=150., normal_policy='hand_surface')
                row.update(target=target_at(arrays, frame), episode=ep, frame=frame, contact=True)
                batch = {k: v.cuda() for k,v in collate([row]).items()}
                pred = models[key](batch)[0].cpu().numpy()
                with np.load(root / 'evaluation' / f'episode_{ep}' / f'{mode}.npz') as z:
                    saved = z['prediction'][np.flatnonzero(z['frame'] == frame)[0]]
                difference = float(np.max(np.abs(pred-saved)))
                rows.append(dict(episode=ep, frame=frame, mode=mode, max_abs_difference=difference))
    report = dict(passed=all(r['max_abs_difference']<=1e-6 for r in rows), rows=rows,
                  replay_forwards=len(rows), threshold=1e-6)
    (root / 'RELOAD_REPLAY.json').write_text(json.dumps(report, indent=2))
    if not report['passed']: raise ValueError('Reloaded frozen inference does not reproduce saved outputs')
    print('RELOAD_REPLAY_PASS', max(r['max_abs_difference'] for r in rows), flush=True)


@torch.inference_mode()
def evaluate(root, protocol, models, completed=()):
    times = {m: [] for m in protocol['evaluation_modes']}
    output = root / 'evaluation'; output.mkdir(exist_ok=bool(completed))
    for episode in protocol['episodes']:
        if episode in completed:
            print('REUSED_COMPLETED_CASE', episode, flush=True)
            continue
        with np.load(root / 'cache' / f'episode_{episode}.npz') as z:
            arrays = {k: z[k] for k in z.files}
        rows = {m: [] for m in protocol['evaluation_modes']}
        targets = []
        frame_info = []
        for frame in protocol['frames']:
            idx = history_indices(frame, 32, 'episode_uniform_recent')
            if idx[-1] != frame or (idx > frame).any(): raise ValueError('Noncausal history')
            obs = {k: arrays[k][idx] for k in OBSERVATION_KEYS + NORMAL_KEYS}
            # Labels are built after encoding and are ignored by the backbone.
            target = target_at(arrays, frame)
            for mode in protocol['model_modes']:
                encoded = encode_observations(obs, mode, time_scale_s=150., normal_policy='hand_surface')
                encoded.update(target=target, episode=episode, frame=frame,
                               contact=bool((obs['normal_load_n'] > 1e-3).any()))
                batch = {k: v.cuda() for k, v in collate([encoded]).items()}
                torch.cuda.synchronize(); start = time.perf_counter()
                pred = models[mode](batch)
                torch.cuda.synchronize(); times[mode].append(time.perf_counter() - start)
                if not torch.isfinite(pred).all(): raise FloatingPointError((episode, frame, mode))
                rows[mode].append(pred[0].cpu().numpy())
                if mode == 'geometry_contact_force':
                    zero = encode_observations(obs, mode, force_gain=0., time_scale_s=150., normal_policy='hand_surface')
                    zero.update(target=target, episode=episode, frame=frame, contact=encoded['contact'])
                    zero_batch = {k: v.cuda() for k, v in collate([zero]).items()}
                    expected = batch['feat'].clone(); expected[:, 10:14] = 0.
                    if not torch.equal(expected, zero_batch['feat']): raise ValueError('Force-zero changes another feature')
                    for k in ('coord', 'grid_coord', 'offset'):
                        if not torch.equal(batch[k], zero_batch[k]): raise ValueError(f'Force-zero changes {k}')
                    torch.cuda.synchronize(); start = time.perf_counter()
                    pred_zero = models[mode](zero_batch)
                    torch.cuda.synchronize(); times['force_zero'].append(time.perf_counter() - start)
                    if not torch.isfinite(pred_zero).all(): raise FloatingPointError('force_zero')
                    rows['force_zero'].append(pred_zero[0].cpu().numpy())
            # Validation-only acquisition state; no subset was filtered above.
            loads = arrays['normal_load_n'][frame].reshape(2, 27).sum(1)
            airborne = bool(arrays['validation_full_mesh_min_z_m'][frame] > .01 and
                            (loads > .01).all() and arrays['validation_controller_phase'][frame] == 2)
            valid = bool(arrays['validation_controller_fit_valid'][frame].all())
            n = arrays['validation_controller_fit_normal_w'][frame].astype(float)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            angle = float(np.degrees(np.arccos(np.clip(-np.dot(*n), -1, 1)))) if valid else None
            targets.append(target)
            frame_info.append(dict(frame=frame, timestamp_s=float(arrays['timestamp_s'][frame]),
                                   airborne_hold=airborne, both_fit_valid=valid, opposition_angle_deg=angle,
                                   current_bilateral_contact=bool((loads > .01).all())))
        case = output / f'episode_{episode}'; case.mkdir()
        for mode in protocol['evaluation_modes']:
            np.savez_compressed(case / f'{mode}.npz', prediction=np.asarray(rows[mode]),
                                target=np.asarray(targets), frame=np.asarray(protocol['frames']),
                                episode=np.full(len(targets), episode))
        (case / 'CONTEXT.json').write_text(json.dumps(frame_info, indent=2, allow_nan=False))
        print('MATCHED_CASE_COMPLETE', episode, 'frames', len(targets), 'all4modes', flush=True)
        (root / 'PROGRESS.json').write_text(json.dumps(dict(last_complete_episode=episode,
             completed_cases=episode-4999, cases=32, new_optimizer_updates=0), indent=2))
    report = dict(complete=True, cases=32, frames_per_case=len(protocol['frames']),
                  saved_prediction_outputs=32*len(protocol['frames'])*4,
                  current_process_evaluation_forwards=sum(map(len,times.values())),
                  reused_completed_cases=list(completed),
                  recovery_replay_forwards=16 if completed else 0,
                  interrupted_unpersisted_forward_bounds=[0,380] if completed else [0,0],
                  latency_scope='Current process completed-case inference only',
                  model_only_latency_ms_median={m:float(np.median(v)*1000) for m,v in times.items()},
                  force_zero_only_features_10_to13=True, new_physics=0, new_optimizer_updates=0)
    (root / 'INFERENCE.json').write_text(json.dumps(report, indent=2))
    print('COMPLETE_FROZEN_INFERENCE', json.dumps(report), flush=True)


def main():
    if not os.environ.get('SLURM_STEP_ID'): raise RuntimeError('Use retained compute step')
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True)
    ap.add_argument('--resume', action='store_true'); args = ap.parse_args()
    root = Path(args.output); protocol = json.loads((root / 'PROTOCOL.json').read_text())
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
    completed = verify_recovery(root, protocol) if args.resume else ()
    if not args.resume: prepare(root, protocol)
    models = load_models(root, protocol, resume=args.resume)
    if args.resume: replay_saved(root, protocol, models)
    evaluate(root, protocol, models, completed=completed)


if __name__ == '__main__': main()
