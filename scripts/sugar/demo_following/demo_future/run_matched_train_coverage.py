"""Matched full official Tracker training on old-pair versus broader TRAIN data."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import torch

from scripts.sugar.demo_following.demo_future.tracker_coverage import equal_state
from scripts.sugar.demo_following.demo_future.run_heldout_reference_validation import write, passed

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
CORPUS = BASE / 'train_reference_coverage80'
PARENT = BASE / 'matched_reference_feedback96'
RUN = BASE / 'matched_train_coverage128'
BOOT = ROOT / 'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'
USD = BASE / 'scene_runtime/converted_g1/g1_29dof_rev_1_0_with_rubber_hand.usd'
ARMS = ('pair_control', 'broad_train')


def prepare():
    from scripts.sugar.demo_following.demo_future.collect_train_reference_corpus import readback
    audit_path = CORPUS / 'READBACK.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else readback()
    if not audit['data_integrity_passed']:
        raise RuntimeError('Complete TRAIN data readback required')
    plan = json.loads((CORPUS / 'NEXT_TRAINING_PROTOCOL.json').read_text())
    corpus = json.loads((CORPUS / 'RESULT.json').read_text())
    if not corpus['execution_completed'] or corpus['all_source_denominator'] != 80:
        raise RuntimeError('Complete all predeclared TRAIN source attempts first')
    accepted = corpus['usable_sources']
    source_plan = json.loads((CORPUS / 'PROTOCOL.json').read_text())
    if len(set(accepted)) != len(accepted) or not set(accepted).issubset(source_plan['sources']):
        raise RuntimeError('Usable source identities differ from predeclared TRAIN split')
    if len(accepted) < plan['minimum_distinct_native_train_sources']:
        raise RuntimeError('Insufficient broader TRAIN coverage; inspect teacher failures before training')
    count = 4 * len(accepted)
    RUN.mkdir(exist_ok=False)
    old_student = sorted((BASE / 'refiner_aligned96_90_feasibility/refined_motion_export/rl_dataset').glob('data_*'))
    old_teacher = sorted((BASE / 'refiner_aligned96_90_feasibility/aligned_original_teacher_bank').glob('data_*'))
    if len(old_student) != 2 or len(old_teacher) != 2:
        raise RuntimeError('Original paired bank identity changed')
    slots = {}
    for arm in ARMS:
        student_bank = RUN / arm / 'student_motion_bank'
        teacher_bank = RUN / arm / 'original_teacher_bank'
        student_bank.mkdir(parents=True)
        teacher_bank.mkdir()
        rows = []
        for index in range(count):
            original = index < count // 2 or arm == 'pair_control'
            if original:
                student, teacher = old_student[index % 2], old_teacher[index % 2]
                source = (96, 90)[index % 2]
            else:
                source = accepted[(index - count // 2) // 2]
                student = CORPUS / f'student_motion_bank/data_{source:03d}'
                teacher = CORPUS / f'original_teacher_bank/data_{source:03d}'
            (student_bank / f'data_{index:03d}').symlink_to(student.resolve(), target_is_directory=True)
            (teacher_bank / f'data_{index:03d}').symlink_to(teacher.resolve(), target_is_directory=True)
            rows.append(dict(slot=index, source=source, kind='original_pair' if original else 'native_train',
                             student=str(student.resolve()), teacher=str(teacher.resolve())))
        slots[arm] = rows
    plan.update(banks_prepared=True, resolved_num_envs=count, usable_native_sources=accepted,
                actual_transitions_per_arm=count * 24 * 128, initial_common_pair_environments=count // 2,
                reference_slots=slots, implementation_completed=True, training_started=False)
    write(RUN / 'PROTOCOL.json', plan)



def physical_readback():
    """Read completed frozen traces, retaining every predeclared source and horizon."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pickle
    from scipy.spatial.transform import Rotation

    result = json.loads((RUN / 'RESULT.json').read_text())
    if not result['execution_completed']:
        raise RuntimeError('Complete both frozen endpoint sets before physical readback')
    validation = PARENT / 'heldout_native_validation'
    sources = [8, 28, 38, 58, 68, 78, 88, 98]
    colors = dict(parent='gray', released='tab:green', pair_control='tab:blue', broad_train='tab:orange')
    rows = []
    fig, axes = plt.subplots(8, 4, figsize=(19, 23), squeeze=False)
    for i, source in enumerate(sources):
        old = validation / f'source_{source:03d}/evaluation'
        paths = dict(parent=old / 'reference_feedback', released=old / 'released_tracker',
                     **{arm: RUN / arm / f'native_validation/source_{source:03d}' for arm in ARMS})
        horizon = json.loads((old / 'PROTOCOL.json').read_text())['steps_per_arm']
        with (old / 'motions/data_000/obj_motion_global_50hz.pkl').open('rb') as f:
            reference_motion = pickle.load(f)
        arrays, row = {}, dict(source=source, horizon=horizon, controllers={})
        for name, directory in paths.items():
            endpoint = json.loads((directory / 'RESULT.json').read_text())
            with np.load(directory / 'TRACE.npz') as trace:
                valid = ~trace['done'].reshape(-1)
                pos = trace['object_state_before_w'][:, 0, :3]
                ref = trace['reference_object_pos_w'][:, 0]
                q = trace['object_state_before_w'][:, 0, 3:7].astype(np.float64)
                frames = trace['reference_frame']
                if not np.all(trace['reference_id'] == source) or not np.array_equal(frames, np.arange(len(frames))):
                    raise RuntimeError('Native source or recorded reference clock differs')
                if not np.allclose(ref, np.asarray(reference_motion['obj_trans'])[frames], atol=1e-6, rtol=0):
                    raise RuntimeError('Raw reference world position differs from recorded phase')
                # target_object_quat_w is the final goal orientation, not the
                # current reference. Read the actual native motion at its clock.
                xyzw = Rotation.from_matrix(np.asarray(reference_motion['obj_rot'])[frames]).as_quat()
                target_q = xyzw[:, [3, 0, 1, 2]]
                norms = np.linalg.norm(q, axis=-1) * np.linalg.norm(target_q, axis=-1)
                if np.any(norms < 1e-10):
                    raise RuntimeError('Invalid actual/reference quaternion')
                angle = 2 * np.arccos(np.clip(np.abs(np.sum(q * target_q, axis=-1)) / norms, 0, 1))
                distance = np.linalg.norm(pos - ref, axis=-1)
                # Same four named hand/foot sensors as the collector. Use only
                # pre-action history here; terminal auto-reset data is excluded.
                hand = np.linalg.norm(trace['contact_force_history_before_w'][:, 0, :2], axis=-1)
                bilateral = (hand.max(axis=-1) > .1).all(axis=-1)
                if len(valid) != endpoint['actual_steps'] or not np.isfinite(distance).all() or not np.isfinite(angle).all():
                    raise RuntimeError('Trace length or physical values differ from endpoint')
                arrays[name] = dict(valid=valid, distance=distance, angle=angle, bilateral=bilateral,
                                    pos=pos.copy(), reference=ref.copy())
                if not np.array_equal(trace['requested_action'][valid], trace['executed_action'][valid]):
                    raise RuntimeError('Requested29-D and actual executed29-D commands differ')
            terminal_path = directory / 'TERMINAL_TERMS.json'
            terminal = [k for k, value in json.loads(terminal_path.read_text()).items() if value] if terminal_path.exists() else []
            if bool(terminal) != endpoint['reset_or_termination']:
                raise RuntimeError('Terminal terms do not match actual endpoint')
            tail = valid & (np.arange(len(valid)) >= max(0, len(valid) - 50))
            row['controllers'][name] = dict(
                steps=endpoint['actual_steps'], passed=passed(endpoint), terminal=terminal,
                maximum_lift_m=endpoint['maximum_lift_m'], sustained_lift=endpoint['at_least_ten_consecutive_lifted_frames'],
                own_valid_frames=int(valid.sum()),
                own_box_coordinate_rmse_m=float(np.sqrt(np.mean(distance[valid] ** 2) / 3)),
                last50_box_distance_mean_m=float(distance[tail].mean()),
                last50_box_orientation_mean_rad=float(angle[tail].mean()),
                last50_bilateral_history_contact_fraction=float(bilateral[tail].mean()))
            t = np.arange(len(valid)) * .02
            style = '--' if name in ('parent', 'released') else '-'
            axes[i, 0].plot(t[valid], pos[valid, 2] - pos[0, 2], color=colors[name], ls=style, label=name)
            axes[i, 1].plot(t[valid], distance[valid], color=colors[name], ls=style)
            axes[i, 2].plot(t[valid], angle[valid], color=colors[name], ls=style)
            axes[i, 3].plot(t[valid], bilateral[valid].astype(float), color=colors[name], ls=style, alpha=.6)
        n = min(len(arrays[arm]['valid']) for arm in ARMS)
        common = arrays[ARMS[0]]['valid'][:n] & arrays[ARMS[1]]['valid'][:n]
        if not np.array_equal(arrays[ARMS[0]]['reference'][:n], arrays[ARMS[1]]['reference'][:n]):
            raise RuntimeError('Matched arms have different reference phase/position')
        row['matched_common_valid_frames'] = int(common.sum())
        row['matched_common_metrics'] = {arm: dict(
            box_coordinate_rmse_m=float(np.sqrt(np.mean(arrays[arm]['distance'][:n][common] ** 2) / 3)),
            box_orientation_mean_rad=float(arrays[arm]['angle'][:n][common].mean()),
            bilateral_history_contact_fraction=float(arrays[arm]['bilateral'][:n][common].mean())) for arm in ARMS}
        row['matching_conditions_passed'] = all(
            result['arms'][arm]['native_sources'][str(source)]['all_startup_fields_exact'] and
            all(result['arms'][arm]['native_sources'][str(source)]['initial_physical_state_exact'].values()) for arm in ARMS)
        ref = np.asarray(reference_motion['obj_trans'])[:horizon]
        axes[i, 0].plot(np.arange(len(ref)) * .02, ref[:, 2] - ref[0, 2], color='black', ls=':', label='reference rise')
        description = ' / '.join(f"{a}:{row['controllers'][a]['steps']} " + (','.join(row['controllers'][a]['terminal']) or 'PASS') for a in ARMS)
        axes[i, 0].set_ylabel(f'Source {source}; budget {horizon}')
        axes[i, 0].set_title(description, fontsize=8)
        for ax in axes[i]:
            ax.set_xlim(0, horizon * .02)
            ax.grid(alpha=.2)
            ax.set_xlabel('Actual control time (s)')
        rows.append(row)
    for j, label in enumerate(('Height rise (m)', 'Box distance to reference (m)', 'Box orientation error (rad)', 'Bilateral hand contact history')):
        axes[0, j].set_title(label + '\n' + axes[0, j].get_title(), fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Frozen coverage comparison: solid matched arms; dashed historical baselines; valid frames only')
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(RUN / 'PHYSICAL_READBACK.png', dpi=140)
    plt.close(fig)
    report = dict(execution_completed=True, rows=rows,
                  native_successes={name: sum(row['controllers'][name]['passed'] for row in rows) for name in colors},
                  matched_initial_conditions_passed=all(row['matching_conditions_passed'] for row in rows),
                  supported_denominator=8, all_source_denominator=10, teacher_failures=[18, 48],
                  broad_execution_floor_passed=result['broad_execution_floor_passed'],
                  metrics='Distances use same-frame pre-action actual/reference world states. Orientations use the original exported motion rotation at each recorded reference_frame, not the final-goal target quaternion. All terminal/reset frames excluded. Hand contact means >0.1N in the pre-action sensor history, not tactile-policy input. Own last50 windows differ in phase; only matched_common_metrics compare identical valid phases. Reference rise is relative to its own initial height, not a physical target-height equality check.',
                  scope='Single matched training seed and fixed previously used validation sources. Parent/released differ in training history; released also differs in input architecture. Curves are physical trace evidence, not actual camera inspection or generative/SMP benefit.')
    write(RUN / 'PHYSICAL_READBACK.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)



def render_native(output=None):
    """Record every fixed native endpoint as its own actual camera rollout."""
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the recorded retained compute step')
    physical = json.loads((RUN / 'PHYSICAL_READBACK.json').read_text())
    if not physical['execution_completed'] or not physical['matched_initial_conditions_passed']:
        raise RuntimeError('Completed physical comparison required')
    out = Path(output) if output is not None else RUN / 'broad_train/native_camera'
    out.mkdir(exist_ok=False)
    sources = [row['source'] for row in physical['rows']]
    checkpoint = RUN / 'broad_train/training/model_607.pt'
    write(out / 'PROTOCOL.json', dict(sources=sources, checkpoint=str(checkpoint), optimizer_updates=0,
          protocol='All eight fixed native validation references, original seeds, horizons and physics, full frozen broad-TRAIN endpoint. Only the actual camera is enabled.',
          scope='Each camera rollout proves its own execution. It is not a frame-exact replay of camera-free statistics. Inspect sampled actual world frames in addition to full video decoding.'))
    records, results = [], {}
    for source in sources:
        original = PARENT / f'heldout_native_validation/source_{source:03d}/evaluation'
        target = out / f'source_{source:03d}'
        cmd = [sys.executable, str(BOOT), 'scripts.sugar.demo_following.demo_future.collect_refiner_pair',
               '--headless', '--video', '--controller', 'tracker', '--tracker-checkpoint', str(checkpoint),
               '--robot-usd', str(USD), '--run', str(original), '--arm', 'original', '--output', str(target)]
        with (out / f'source_{source:03d}.log').open('x') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(source=source, pid=proc.pid, pgid=os.getpgid(proc.pid), command=cmd)
            records.append(row)
            write(out / 'CHILDREN.json', records)
            row['returncode'] = proc.wait()
            write(out / 'CHILDREN.json', records)
        if row['returncode'] or not (target / 'RESULT.json').exists():
            raise RuntimeError('Inspect actual camera child failure: ' + str(source))
        endpoint = json.loads((target / 'RESULT.json').read_text())
        video = target / 'ACTUAL_WORLD.mp4'
        import imageio_ffmpeg
        from PIL import Image, ImageDraw
        with np.load(target / 'TRACE.npz') as trace:
            expected_frames = int((~trace['done'].reshape(-1))[::2].sum())
        sample_indices = sorted(set(np.linspace(0, expected_frames - 1, min(9, expected_frames), dtype=int).tolist()))
        decoder = imageio_ffmpeg.read_frames(str(video))
        probe = next(decoder)
        width, height = probe['size']
        sampled, decoded = {}, 0
        for index, pixels in enumerate(decoder):
            if index in sample_indices:
                sampled[index] = Image.frombytes('RGB', (width, height), pixels)
            decoded += 1
        probe['decoded_frames'] = decoded
        video_pass = (probe['codec'] == 'h264' and probe['source_size'] == (960, 540)
                      and probe['fps'] == 25.0 and decoded == expected_frames)
        sheet = Image.new('RGB', (width * 3, (height + 30) * 3), 'white')
        draw = ImageDraw.Draw(sheet)
        for ordinal, index in enumerate(sample_indices):
            x, y = ordinal % 3 * width, ordinal // 3 * (height + 30)
            sheet.paste(sampled[index], (x, y + 30))
            draw.text((x + 10, y + 8), f'Source {source}; video frame {index}; actual time {(2 * index + 1) * .02:.2f}s', fill='black')
        sheet.save(target / 'CONTACT_SHEET.png')
        with np.load(target / 'STARTUP.npz') as current, np.load(RUN / f'broad_train/native_validation/source_{source:03d}/STARTUP.npz') as old:
            startup_exact = set(current.files) == set(old.files) and all(np.array_equal(current[k], old[k]) for k in current.files)
        results[str(source)] = dict(result=endpoint, physical_passed=passed(endpoint), video=probe,
                                   decoded_all_frames=True, video_contract_passed=video_pass,
                                   expected_video_frames=expected_frames, sampled_video_frames=sample_indices, startup_matches_camera_free=startup_exact,
                                   sampled_visual_inspection_completed=False)
        write(out / 'PARTIAL_RESULT.json', results)
        if not video_pass or not startup_exact:
            raise RuntimeError('Camera video or startup contract failed: ' + str(source))
        print(json.dumps(dict(camera_source=source, steps=endpoint['actual_steps'], physical_passed=passed(endpoint))), flush=True)
    report = dict(execution_completed=True, optimizer_updates=0, per_source=results,
                  physical_successes=sum(r['physical_passed'] for r in results.values()), supported_denominator=8,
                  all_source_denominator=10, teacher_failures=[18, 48],
                  video_contracts_passed=all(r['video_contract_passed'] for r in results.values()),
                  sampled_visual_inspection_completed=False,
                  scope='Actual camera-enabled full-world evidence; frame inspection remains separate from codec and physical checks.')
    write(out / 'RESULT.json', report)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--readback-only', action='store_true')
    mode.add_argument('--render-native', action='store_true')
    parser.add_argument('--camera-output', type=Path, help='Distinct output for an externally interrupted camera attempt; preserve all prior outputs')
    args = parser.parse_args()
    if args.camera_output and not args.render_native:
        parser.error('--camera-output requires --render-native')
    if args.readback_only:
        return physical_readback()
    if args.render_native:
        return render_native(args.camera_output)

    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the recorded retained compute step')
    if not RUN.exists():
        prepare()
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    path = RUN / 'CHILDREN.json'
    if path.exists():
        raise RuntimeError('Existing matched execution; inspect exact children, do not repeat')
    records = []

    def child(tag, module, args, artifact):
        cmd = [sys.executable, str(BOOT), 'scripts.sugar.demo_following.demo_future.' + module, *map(str, args)]
        with (RUN / (tag + '.log')).open('x') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(tag=tag, pid=proc.pid, pgid=os.getpgid(proc.pid), command=cmd)
            records.append(row)
            write(path, records)
            row['returncode'] = proc.wait()
            write(path, records)
        if row['returncode'] or not artifact.exists():
            raise RuntimeError('Inspect actual child failure: ' + tag)
        print(json.dumps(dict(completed=tag, artifact=str(artifact))), flush=True)

    for arm in ARMS:
        out = RUN / arm / 'preflight'
        child('preflight_' + arm, 'train_tracker_online_future',
              ['--headless', '--preflight-only', '--coverage-arm', arm, '--output', out, '--robot-usd', USD], out / 'RESULT.json')
    checkpoints = [torch.load(RUN / arm / 'preflight/model_479.pt', map_location='cpu', weights_only=True) for arm in ARMS]
    checks = dict(whole_models_exact=equal_state(checkpoints[0]['model_state_dict'], checkpoints[1]['model_state_dict']),
                  whole_adam_exact=equal_state(checkpoints[0]['optimizer_state_dict'], checkpoints[1]['optimizer_state_dict']))
    initial = [dict(np.load(RUN / arm / 'preflight/INITIAL_STATE.npz')) for arm in ARMS]
    common = plan['initial_common_pair_environments']
    common_errors = {key: float(np.max(np.abs(initial[0][key][:common] - initial[1][key][:common]))) for key in initial[0]}
    checks['common_old_pair_initial_fields_within1e_5'] = max(common_errors.values()) <= 1e-5
    checks['all_initial_reference_slots_exact'] = np.array_equal(initial[0]['reference_id'], initial[1]['reference_id']) and np.array_equal(initial[0]['reference_id'], np.arange(plan['resolved_num_envs']))
    for arm in ARMS:
        preflight = RUN / arm / 'preflight'
        result = json.loads((preflight / 'RESULT.json').read_text())
        protocol = json.loads((preflight / 'PROTOCOL.json').read_text())
        checks[arm + '_actual_preflight'] = result['passed'] and result['actual_transitions'] == plan['resolved_num_envs'] * 24 and result['optimizer_updates'] == 0
        checks[arm + '_parent_state_restored'] = protocol['resume']['passed']
        checks[arm + '_full_source_slot_coverage'] = sorted(set(protocol['initial_reference_slots'])) == list(range(plan['resolved_num_envs']))
    preflight = dict(passed=all(checks.values()), checks=checks, common_initial_max_errors=common_errors,
                     scope='Both whole models/Adam start equal; common old-pair lanes match. New native lanes intentionally have different physical initial conditions.')
    write(RUN / 'MATCHED_PREFLIGHT.json', preflight)
    if not preflight['passed']:
        raise RuntimeError('Actual matched coverage preflight failed')
    plan['training_started'] = True
    write(RUN / 'PROTOCOL.json', plan)
    for arm in ARMS:
        out = RUN / arm / 'training'
        child('training_' + arm, 'train_tracker_online_future',
              ['--headless', '--coverage-arm', arm, '--output', out, '--robot-usd', USD], out / 'RESULT.json')
        result = json.loads((out / 'RESULT.json').read_text())
        if result['newly_applied_updates'] != 128 or result['bcppo_updates'] != 608 or result['actual_transitions'] != plan['actual_transitions_per_arm']:
            raise RuntimeError('Actual coverage training budget differs')
        start = torch.load(out / 'model_479.pt', map_location='cpu', weights_only=True)
        own = checkpoints[ARMS.index(arm)]
        if not equal_state(start['model_state_dict'], own['model_state_dict']) or not equal_state(start['optimizer_state_dict'], own['optimizer_state_dict']):
            raise RuntimeError('Actual training model/Adam did not match its preflight')
        endpoint = torch.load(out / 'model_607.pt', map_location='cpu', weights_only=True)
        checks = dict(
            cumulative_update_exact=endpoint['infos']['bcppo_update_step'] == 608,
            full_actor_shape=tuple(endpoint['model_state_dict']['actor.0.weight'].shape) == (512, 846),
            state_keys_preserved=endpoint['model_state_dict'].keys() == start['model_state_dict'].keys(),
            all_model_values_finite=all(torch.isfinite(v).all().item() for v in endpoint['model_state_dict'].values()),
            all_adam_values_finite=all(torch.isfinite(v).all().item() for state in endpoint['optimizer_state_dict']['state'].values() for v in state.values() if isinstance(v, torch.Tensor)),
            actual_teacher_unchanged=result['teacher_unchanged'],
            actual_optimizer_clocks=result['optimizer_clocks_by_role'] == {'actor': [12160], 'critic': [2160]},
        )
        write(out / 'CHECKPOINT_READBACK.json', dict(passed=all(checks.values()), checks=checks))
        if not all(checks.values()):
            raise RuntimeError('Full saved coverage checkpoint readback failed')
    results = {}
    validation = PARENT / 'heldout_native_validation'
    native = json.loads((validation / 'RESULT.json').read_text())
    sources = sorted(int(s) for s, value in native['per_source'].items() if value['teacher_passed'])
    if sources != [8, 28, 38, 58, 68, 78, 88, 98]:
        raise RuntimeError('Frozen validation source contract changed')
    for arm in ARMS:
        ckpt = RUN / arm / 'training/model_607.pt'
        evaluation = RUN / arm / 'evaluation'
        evaluation.mkdir()
        source = PARENT / 'reference_feedback/evaluation'
        protocol = json.loads((source / 'PROTOCOL.json').read_text())
        protocol.update(purpose='Frozen full608 endpoint after matched broader TRAIN state coverage', coverage_arm=arm)
        write(evaluation / 'PROTOCOL.json', protocol)
        (evaluation / 'motions').symlink_to(source / 'motions', target_is_directory=True)
        for selected in ('original', 'repeat', 'alternate'):
            child('eval_' + arm + '_' + selected, 'collect_refiner_pair',
                  ['--headless', '--controller', 'tracker', '--tracker-checkpoint', ckpt, '--robot-usd', USD, '--run', evaluation, '--arm', selected], evaluation / selected / 'RESULT.json')
        child('readback_' + arm, 'readback_refiner_pair', ['--run', evaluation], evaluation / 'READBACK.json')
        child('following_' + arm, 'readback_reference_following', ['--run', evaluation], evaluation / 'REFERENCE_FOLLOWING_READBACK.json')
        native_results = {}
        for source_id in sources:
            old = validation / f'source_{source_id:03d}/evaluation'
            out = RUN / arm / f'native_validation/source_{source_id:03d}'
            child(f'native_{arm}_{source_id:03d}', 'collect_refiner_pair',
                  ['--headless', '--controller', 'tracker', '--tracker-checkpoint', ckpt, '--robot-usd', USD, '--run', old, '--arm', 'original', '--output', out], out / 'RESULT.json')
            row = json.loads((out / 'RESULT.json').read_text())
            with np.load(out / 'TRACE.npz') as trace, np.load(old / 'zero_feedback/TRACE.npz') as baseline:
                fields = ('robot_body_state_before_w', 'robot_root_state_before_w', 'object_state_before_w',
                          'joint_pos_before', 'joint_vel_before', 'reference_body_pos_w', 'reference_object_pos_w')
                initial = {key: bool(np.array_equal(trace[key][0], baseline[key][0])) for key in fields}
            with np.load(out / 'STARTUP.npz') as startup, np.load(old / 'zero_feedback/STARTUP.npz') as baseline:
                startup_exact = set(startup.files) == set(baseline.files) and all(
                    np.array_equal(startup[key], baseline[key]) for key in startup.files)
            native_results[str(source_id)] = dict(result=row, passed=passed(row),
                                                  initial_physical_state_exact=initial,
                                                  all_startup_fields_exact=startup_exact)
            write(RUN / arm / 'native_validation/PARTIAL_READBACK.json', native_results)
            if not all(initial.values()) or not startup_exact:
                raise RuntimeError('Frozen native comparison initial conditions changed')
        pair = json.loads((evaluation / 'READBACK.json').read_text())
        following = json.loads((evaluation / 'REFERENCE_FOLLOWING_READBACK.json').read_text())
        results[arm] = dict(train_pair_physical_passed=pair['data_pair_feasibility_passed'],
                            train_pair_following_passed=following['exploratory_reference_following_checks_passed'],
                            native_successes=sum(row['passed'] for row in native_results.values()), native_sources=native_results)
        write(RUN / 'PARTIAL_RESULT.json', results)
    result = dict(execution_completed=True, new_updates_per_arm=128, bcppo_updates=608,
                  actual_transitions_per_arm=plan['actual_transitions_per_arm'], arms=results,
                  teacher_supported_source_denominator=8, all_scheduled_source_denominator=10,
                  teacher_failed_sources=[18, 48],
                  broad_execution_floor_passed=results['broad_train']['train_pair_physical_passed'] and results['broad_train']['train_pair_following_passed'] and results['broad_train']['native_successes'] == 8,
                  scope=plan['scope'], next_action=plan['automatic_next_action'])
    write(RUN / 'RESULT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
