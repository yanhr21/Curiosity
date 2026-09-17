"""Full official frozen Refiner native coverage of all80 Carry TRAIN sources.

Actual PhysX trajectories and the official Refiner formatter only. Original
numeric teacher references are kept separate from measured student motion and
executed29-D actions. No model is trained here.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys

import numpy as np

from scripts.sugar.demo_following.demo_future.run_heldout_reference_validation import (
    export_native, passed, write)

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
RUN = BASE / 'train_reference_coverage80'
BOOT = ROOT / 'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'
USD = BASE / 'scene_runtime/converted_g1/g1_29dof_rev_1_0_with_rubber_hand.usd'


def prepare():
    splits = {}
    for split in ('train', 'validation', 'test'):
        arrays = np.load(BASE / f'dataset_heading_local_v3/{split}/routing.npz')
        splits[split] = sorted(set(arrays['base_source_motion_id'][arrays['base_task'] == 0].tolist()))
    ids = splits['train']
    if len(ids) != 80 or set(ids) & (set(splits['validation']) | set(splits['test'])):
        raise RuntimeError('Original motion-disjoint TRAIN split changed')
    RUN.mkdir(exist_ok=False)
    schedule = []
    for source in ids:
        raw = ROOT / f'SUGAR/data/CarryBox/data_{source:03d}'
        robot = np.load(raw / 'robot_50hz.npz')
        length = len(robot['joint_pos'])
        with (raw / 'obj_motion_global_50hz.pkl').open('rb') as f:
            obj = pickle.load(f)
        if len(obj['obj_trans']) < length or len(np.load(raw / 'contact_labels_50hz.npy')) < length:
            raise RuntimeError('Original reference fields have insufficient real frames')
        steps = min(436, length - 8)
        if steps < 250:
            raise RuntimeError('Unexpected TRAIN clip length')
        unused = 45 if source == 96 else 96
        teacher = RUN / f'source_{source:03d}/teacher'
        motions = teacher / 'motions'
        motions.mkdir(parents=True)
        (motions / 'data_000').symlink_to(raw, target_is_directory=True)
        (motions / 'data_001').symlink_to(ROOT / f'SUGAR/data/CarryBox/data_{unused:03d}', target_is_directory=True)
        protocol = dict(reference_pair=[source, unused], seed=272048,
                        steps_per_arm=steps, switch_control_frame=158, reference_alignment='none',
                        alignment='Native selected TRAIN reference from frame0. No reference switch or rigid transform; second bank entry used only for read-only query.',
                        scope='Frozen full official Refiner native TRAIN prefix collection, not a common-history counterfactual pair or successful learned student policy.')
        write(teacher / 'PROTOCOL.json', protocol)
        schedule.append(dict(source=source, original_frames=length, teacher_steps=steps,
                             downstream_horizon=min(400, steps - 36)))
    plan = dict(sources=ids, schedule=schedule, seed=272048, original_split_counts={k: len(v) for k, v in splits.items()},
                split_rule='All and only original80 Carry TRAIN sources; validation/test remain excluded.',
                teacher_checkpoint=str(ROOT / 'experiments/sugar_reproduction/outputs/final/official_sugar/baseline/ckpts/refiner_model10000.pt'),
                physical_condition='Unmodified complete G1 and frozen890-D official Refiner; nominal friction1/restitution0/mass1, no pushes, zero reset noise. One native rollout per source, serial retained GPU.',
                total_requested_teacher_steps=sum(row['teacher_steps'] for row in schedule),
                automatic_data_checks='Complete predeclared real horizon without reset, all numeric fields finite, weights frozen and ten consecutive frames lifted by0.05m. Preserve every failure in the80-source denominator.',
                conversion='Only full passing trajectories go through the official Refiner formatter without stabilization. Teacher bank is the exact original numeric reference prefix with matching lengths. Actual29-D actions remain in original traces.',
                training_updates=0,
                scope='Broader TRAIN state-coverage data. Native sources start in different worlds and are not80 shared-history conditional futures. Existing true matched switching evidence remains separate.',
                next_action='Inspect full collection coverage and actual motion evidence, then predeclare a bounded full-model TRAIN-only coverage comparison. This collector starts no optimizer or Generator training.')
    write(RUN / 'PROTOCOL.json', plan)
    print(json.dumps(plan), flush=True)


def original_teacher_prefix(source, frames, output):
    raw = ROOT / f'SUGAR/data/CarryBox/data_{source:03d}'
    output.mkdir()
    robot = dict(np.load(raw / 'robot_50hz.npz'))
    selected = {k: v.copy() if k == 'fps' else v[:frames].copy() for k, v in robot.items()}
    np.savez_compressed(output / 'robot_50hz.npz', **selected)
    with (raw / 'obj_motion_global_50hz.pkl').open('rb') as f:
        obj = pickle.load(f)
    cropped = {k: v[:frames].copy() if isinstance(v, np.ndarray) and v.ndim and len(v) >= frames else v for k, v in obj.items()}
    with (output / 'obj_motion_global_50hz.pkl').open('wb') as f:
        pickle.dump(cropped, f)
    contact = np.load(raw / 'contact_labels_50hz.npy')[:frames].copy()
    np.save(output / 'contact_labels_50hz.npy', contact)
    loaded = dict(np.load(output / 'robot_50hz.npz'))
    with (output / 'obj_motion_global_50hz.pkl').open('rb') as f:
        loaded_obj = pickle.load(f)
    checks = dict(robot_exact=all(np.array_equal(v, selected[k]) for k, v in loaded.items()),
                  object_exact=all(np.array_equal(v, cropped[k]) for k, v in loaded_obj.items()),
                  contact_exact=np.array_equal(np.load(output / 'contact_labels_50hz.npy'), contact),
                  real_frame_count=len(loaded['joint_pos']) == frames)
    if not all(checks.values()):
        raise RuntimeError('Original teacher prefix changed values')
    return checks


def collect():
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the recorded retained compute step')
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    # Complete the same fixed released baseline before consuming new corpus work.
    floor = json.loads((BASE / 'matched_reference_feedback96/heldout_native_validation/RELEASED_FLOOR_RESULT.json').read_text())
    if not floor['execution_completed']:
        raise RuntimeError('Released native floor incomplete')
    record = RUN / 'CHILDREN.json'
    if record.exists():
        raise RuntimeError('Existing corpus execution: inspect exact child rather than repeat')
    students, teachers = RUN / 'student_motion_bank', RUN / 'original_teacher_bank'
    students.mkdir()
    teachers.mkdir()
    records, results = [], {}
    for item in plan['schedule']:
        source = item['source']
        case = RUN / f'source_{source:03d}'
        teacher = case / 'teacher'
        output = teacher / 'original'
        cmd = [sys.executable, str(BOOT), 'scripts.sugar.demo_following.demo_future.collect_refiner_pair',
               '--headless', '--controller', 'refiner', '--robot-usd', str(USD),
               '--run', str(teacher), '--arm', 'original', '--output', str(output)]
        with (RUN / f'collect_{source:03d}.log').open('x') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(source=source, pid=proc.pid, pgid=os.getpgid(proc.pid), command=cmd)
            records.append(row)
            write(record, records)
            row['returncode'] = proc.wait()
            write(record, records)
        if row['returncode'] or not (output / 'RESULT.json').exists():
            raise RuntimeError('Corpus runtime failure: ' + str(source))
        result = json.loads((output / 'RESULT.json').read_text())
        entry = dict(result=result, usable=passed(result), requested_steps=item['teacher_steps'],
                     downstream_horizon=item['downstream_horizon'])
        if entry['usable']:
            measured = export_native(case, source)
            (students / f'data_{source:03d}').symlink_to(measured, target_is_directory=True)
            entry['original_teacher_prefix_checks'] = original_teacher_prefix(
                source, result['actual_steps'], teachers / f'data_{source:03d}')
            entry['measured_reference_path'] = str(measured)
        results[str(source)] = entry
        write(RUN / 'PARTIAL_RESULT.json', results)
        print(json.dumps(dict(source=source, steps=result['actual_steps'], usable=entry['usable'], completed_sources=len(results))), flush=True)
    accepted = [int(source) for source, row in results.items() if row['usable']]
    student_ids = [int(path.name.split('_')[1]) for path in sorted(students.glob('data_*'))]
    teacher_ids = [int(path.name.split('_')[1]) for path in sorted(teachers.glob('data_*'))]
    if student_ids != accepted or teacher_ids != accepted:
        raise RuntimeError('Teacher/student source-bank alignment differs')
    result = dict(execution_completed=True, all_source_denominator=80, usable_sources=accepted,
                  usable_count=len(accepted), failed_sources=[int(k) for k, v in results.items() if not v['usable']],
                  actual_control_steps=sum(v['result']['actual_steps'] for v in results.values()),
                  training_updates=0, per_source=results, scope=plan['scope'], next_action=plan['next_action'])
    write(RUN / 'RESULT.json', result)
    print(json.dumps(result), flush=True)


def readback():
    """Inspect all real source endpoints before using the completed TRAIN bank."""
    result = json.loads((RUN / 'RESULT.json').read_text())
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    children = json.loads((RUN / 'CHILDREN.json').read_text())
    sources = plan['sources']
    checks = dict(
        all80_attempted=result['execution_completed'] and len(result['per_source']) == len(sources) == 80
        and set(map(int, result['per_source'])) == set(sources),
        all_children_finished=len(children) == 80 and all(c.get('returncode') == 0 for c in children),
        no_training_updates=result['training_updates'] == 0,
    )
    rows, curves = [], []
    for source in sources:
        entry = result['per_source'][str(source)]
        case = RUN / f'source_{source:03d}'
        with np.load(case / 'teacher/original/TRACE.npz') as trace:
            count = len(trace['done'])
            valid = ~trace['done'].reshape(-1)
            obj = trace['object_state_before_w'][:, 0, :3]
            reference = trace['reference_object_pos_w'][:, 0]
            lifted = obj[:, 2] - obj[0, 2] > .05
            streak = np.convolve(lifted.astype(int), np.ones(10, dtype=int), mode='valid') if count >= 10 else np.array([])
            first = np.flatnonzero(streak == 10)
            action_error = float(np.abs(trace['requested_action'][valid] - trace['executed_action'][valid]).max()) if valid.any() else None
            source_checks = dict(
                actual_frame_count=count == entry['result']['actual_steps'],
                actual_action_shapes=trace['requested_action'].shape == trace['executed_action'].shape == (count, 1, 29),
                actual_action_execution_exact=action_error == 0,
                finite_saved_numbers=all(np.isfinite(trace[k]).all().item() for k in trace.files if np.issubdtype(trace[k].dtype, np.number)),
                terminal_status_matches=bool(trace['done'].any()) == entry['result']['reset_or_termination'],
                source_reference_identity=bool(np.all(trace['reference_id'] == source)),
                original_reference_clock_exact=np.array_equal(trace['reference_frame'], np.arange(count)),
            )
            curves.append((source, np.arange(count)[valid] * .02, obj[valid, 2] - obj[0, 2],
                           reference[valid, 2] - reference[0, 2], entry['usable']))
        student = RUN / f'student_motion_bank/data_{source:03d}'
        teacher = RUN / f'original_teacher_bank/data_{source:03d}'
        if entry['usable']:
            source_checks['official_export_passed'] = json.loads((case / 'official_refined_export/RESULT.json').read_text())['passed']
            source_checks['original_teacher_values_preserved'] = all(entry['original_teacher_prefix_checks'].values())
            source_checks['student_source_binding'] = student.resolve() == Path(entry['measured_reference_path']).resolve()
            with np.load(student / 'robot_50hz.npz') as s, np.load(teacher / 'robot_50hz.npz') as t:
                source_checks['paired_real_lengths'] = len(s['joint_pos']) == len(t['joint_pos']) == count
        else:
            source_checks['failed_source_excluded'] = not student.exists() and not teacher.exists()
        checks[f'source_{source:03d}'] = all(source_checks.values())
        rows.append(dict(source=source, usable=entry['usable'], checks=source_checks,
                         first_ten_lift_start_frame=int(first[0]) if len(first) else None,
                         lift_observed_within_training400=bool(len(first) and first[0] + 10 <= 400),
                         actual_action_max_error=action_error))
    expected = set(result['usable_sources'])
    for bank in ('student_motion_bank', 'original_teacher_bank'):
        checks[bank + '_exact_source_set'] = {int(p.name.split('_')[1]) for p in (RUN / bank).glob('data_*')} == expected
    audit = dict(execution_completed=True, data_integrity_passed=all(checks.values()), checks=checks,
                 teacher_usable_sources=len(expected), all_source_denominator=80, per_source=rows,
                 usable_sources_without_lift_within_training400=[row['source'] for row in rows if row['usable'] and not row['lift_observed_within_training400']],
                 scope='Complete real teacher corpus and original-reference binding. Physical plots exclude terminal auto-reset frames. Teacher data feasibility is not student success, generated futures, camera evidence or independent training replicates.')
    write(RUN / 'READBACK.json', audit)
    if not audit['data_integrity_passed']:
        raise RuntimeError('Completed TRAIN corpus direct readback failed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for group in range(4):
        fig, axes = plt.subplots(5, 4, figsize=(14, 12), sharex=True, sharey=True)
        for ax, (source, time, actual, reference, usable) in zip(axes.flat, curves[group * 20:(group + 1) * 20]):
            ax.plot(time, actual, label='actual', color='#0173b2')
            ax.plot(time, reference, label='reference', color='#de8f05', alpha=.8)
            ax.axhline(.05, color='gray', linestyle=':', linewidth=.7)
            ax.set_title(f'Source{source}: {"usable" if usable else "FAILED"}', color='black' if usable else '#cc3311', fontsize=10)
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.supxlabel('Actual control time (s)'); fig.supylabel('Box height above respective initial height (m)')
        fig.suptitle(f'Full frozen TRAIN corpus, group{group + 1}/4; {len(expected)}/80 usable. Different native initial worlds.')
        fig.tight_layout(); fig.savefig(RUN / f'PHYSICAL_GROUP_{group}.png', dpi=140); plt.close(fig)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('per_source', 'checks')}), flush=True)
    return audit



TRACKER_RUN = BASE / 'tracker_train_generator_corpus76'


def prepare_tracker_corpus():
    """Predeclare real full-Tracker TRAIN rollouts for official Generator data."""
    parent = BASE / 'matched_train_coverage128'
    endpoint = json.loads((parent / 'RESULT.json').read_text())
    corpus = json.loads((RUN / 'RESULT.json').read_text())
    if not endpoint['broad_execution_floor_passed'] or not corpus['execution_completed']:
        raise RuntimeError('Complete passing Tracker execution floor and original TRAIN corpus required')
    source_plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    sources = corpus['usable_sources']
    if len(sources) != 76 or set(sources) & {18, 48} or not set(sources).issubset(source_plan['sources']):
        raise RuntimeError('Preserve all76 teacher-supported TRAIN sources and split')
    TRACKER_RUN.mkdir(exist_ok=False)
    old = BASE / 'refiner_aligned96_90_feasibility'
    old_students = sorted((old / 'refined_motion_export/rl_dataset').glob('data_*'))
    old_teachers = sorted((old / 'aligned_original_teacher_bank').glob('data_*'))
    if len(old_students) != 2 or len(old_teachers) != 2:
        raise RuntimeError('Original unused query references changed')
    schedule = []
    for source in sources:
        entry = corpus['per_source'][str(source)]
        case = TRACKER_RUN / f'source_{source:03d}'
        motions, teachers = case / 'motions', case / 'teacher_motions'
        motions.mkdir(parents=True)
        teachers.mkdir()
        unused_index = 1 if source == 96 else 0
        unused = (96, 90)[unused_index]
        paths = [(RUN / f'student_motion_bank/data_{source:03d}', RUN / f'original_teacher_bank/data_{source:03d}'),
                 (old_students[unused_index], old_teachers[unused_index])]
        counts = []
        for slot, (student, teacher) in enumerate(paths):
            with np.load(student / 'robot_50hz.npz') as a, np.load(teacher / 'robot_50hz.npz') as b:
                length = len(a['joint_pos'])
                if len(b['joint_pos']) != length:
                    raise RuntimeError('Full teacher query must have exact matching real length')
            counts.append(length)
            (motions / f'data_{slot:03d}').symlink_to(student.resolve(), target_is_directory=True)
            (teachers / f'data_{slot:03d}').symlink_to(teacher.resolve(), target_is_directory=True)
        horizon = entry['downstream_horizon']
        if horizon != min(400, counts[0] - 36):
            raise RuntimeError('Original real future-tail horizon differs')
        protocol = dict(reference_pair=[source, unused], seed=272050, steps_per_arm=horizon,
                        switch_control_frame=158, reference_alignment='none',
                        alignment='Native TRAIN source, original recorded clock from frame0. No executed reference switch; second source is a read-only query.',
                        scope='Frozen full model607 actual TRAIN execution for Generator data. Known numeric reference36-D commands are training labels; actual executed29-D actions stay distinct. No generated future or human-video claim.')
        write(case / 'PROTOCOL.json', protocol)
        schedule.append(dict(source=source, horizon=horizon, real_reference_frames=counts[0], unused_query_source=unused))
    plan = dict(sources=sources, schedule=schedule, seed=272050,
                checkpoint=str(parent / 'broad_train/training/model_607.pt'),
                teacher_checkpoint=source_plan['teacher_checkpoint'], optimizer_updates=0,
                supported_train_denominator=76, original_train_denominator=80, failed_teacher_sources=corpus['failed_sources'],
                total_requested_control_steps=sum(row['horizon'] for row in schedule),
                physical_conditions='Unchanged full G1/PhysX native reset and nominal material/mass settings. Each source starts in its own native world; these are not shared-history counterfactual futures.',
                labels='Use the official Tracker-to-Generator formatter on complete physical passes only. Actual state provides Generator observations, selected refined reference provides36-D command targets, original numeric demo provides independent geometry context later. Never substitute29-D executed actions with joint positions.',
                teacher_queries='Full890-D frozen Refiner queried on actual student states using matched original prefixes. Query actions are never executed; label error does not prove teacher recovery.',
                dataset_splits='All and only76 already supported TRAIN sources. No validation or test data enters the dataset; retain all failures in their denominators.',
                automatic_next_action='Complete all76 endpoints, inspect actual TRAIN execution and teacher-query evidence. Convert only full passing episodes, then prepare original-demo geometry and full released Generator matched context experiments. No optimizer is constructed in this stage.')
    write(TRACKER_RUN / 'PROTOCOL.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k != 'schedule'}), flush=True)


def restore_tracker_progress():
    """Preserve completed cases and a verified externally killed initialization."""
    plan = json.loads((TRACKER_RUN / 'PROTOCOL.json').read_text())
    results = json.loads((TRACKER_RUN / 'PARTIAL_RESULT.json').read_text())
    records = json.loads((TRACKER_RUN / 'CHILDREN.json').read_text())
    if list(map(int, results)) != plan['sources'][:len(results)]:
        raise RuntimeError('Completed source prefix differs from the original schedule')
    finished = [row for row in records if not row.get('externally_interrupted') and row['source'] in set(map(int, results))]
    if len(finished) != len(results) or any(row.get('returncode') != 0 for row in finished):
        raise RuntimeError('A completed scientific case lacks a finished exact process')
    for source, entry in results.items():
        case = TRACKER_RUN / f'source_{int(source):03d}/original'
        if json.loads((case / 'RESULT.json').read_text()) != entry['result']:
            raise RuntimeError('Preserved completed endpoint differs')
        if not (case / 'TRACE.npz').exists() or not json.loads((case / 'TEACHER_INTERFACE_AUDIT.json').read_text())['passed']:
            raise RuntimeError('Completed actual trace or full teacher interface is missing')
    for row in records:
        if row['source'] in set(map(int, results)) or row.get('externally_interrupted'):
            continue
        job = row['slurm_job_id']
        account = subprocess.check_output(['sacct', '-j', str(job), '-n', '-P', '-o', 'JobID,State'], text=True)
        job_rows = [line.split('|') for line in account.splitlines() if line.split('|')[0] == str(job)]
        if len(job_rows) != 1 or not job_rows[0][1].startswith('CANCELLED'):
            raise RuntimeError('Do not retry a live or unclassified prior allocation')
        case = TRACKER_RUN / f'source_{row["source"]:03d}'
        target = case / 'original'
        log = TRACKER_RUN / f'source_{row["source"]:03d}.log'
        if any(target.iterdir()) or 'REFINER_PAIR_PHASE=rollout_started' in log.read_text():
            raise RuntimeError('Interrupted actual rollout needs direct trace recovery, not initialization retry')
        preserved = case / f'initialization_interrupted_job{job}'
        preserved_log = log.with_name(f'{log.stem}.interrupted_job{job}.log')
        if preserved.exists() or preserved_log.exists():
            raise RuntimeError('Preserve existing interrupted attempt paths')
        target.rename(preserved); log.rename(preserved_log)
        row.update(externally_interrupted=True, prior_job_state=job_rows[0][1],
                   preserved_output=str(preserved), preserved_log=str(preserved_log),
                   actual_control_steps=0)
    write(TRACKER_RUN / 'CHILDREN.json', records)
    return records, results


def collect_tracker_corpus(resume=False):
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the recorded retained compute step')
    from scripts.sugar.demo_following.demo_future.export_tracker_il import export_episode
    plan = json.loads((TRACKER_RUN / 'PROTOCOL.json').read_text())
    context_result = json.loads((TRACKER_RUN / 'original_demo_geometry_context/RESULT.json').read_text())
    if not context_result['execution_completed'] or context_result['source_count'] != 76:
        raise RuntimeError('Complete original TRAIN context library required')
    record = TRACKER_RUN / 'CHILDREN.json'
    if record.exists() and not resume:
        raise RuntimeError('Existing frozen TRAIN collection: inspect exact children')
    export = TRACKER_RUN / 'official_il_data'
    export.mkdir(exist_ok=resume)
    records, results = restore_tracker_progress() if resume else ([], {})
    for item in plan['schedule']:
        source = item['source']
        if str(source) in results:
            continue
        case = TRACKER_RUN / f'source_{source:03d}'
        target = case / 'original'
        cmd = [sys.executable, str(BOOT), 'scripts.sugar.demo_following.demo_future.collect_refiner_pair',
               '--headless', '--controller', 'tracker', '--tracker-checkpoint', plan['checkpoint'],
               '--robot-usd', str(USD), '--run', str(case), '--arm', 'original', '--output', str(target),
               '--audit-teacher-bank', str(case / 'teacher_motions')]
        with (TRACKER_RUN / f'source_{source:03d}.log').open('x') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(source=source, pid=proc.pid, pgid=os.getpgid(proc.pid), command=cmd,
                       slurm_job_id=os.environ['SLURM_JOB_ID'], slurm_step_id=os.environ['SLURM_STEP_ID'],
                       host=socket.gethostname())
            records.append(row)
            write(record, records)
            row['returncode'] = proc.wait()
            write(record, records)
        if row['returncode'] or not (target / 'RESULT.json').exists():
            raise RuntimeError('Inspect frozen TRAIN runtime failure: ' + str(source))
        endpoint = json.loads((target / 'RESULT.json').read_text())
        query = json.loads((target / 'TEACHER_INTERFACE_AUDIT.json').read_text())
        if not query['passed']:
            raise RuntimeError('Exact full teacher interface failed')
        with np.load(target / 'TRACE.npz') as trace:
            valid = ~trace['done'].reshape(-1)
            if not np.all(trace['reference_id'] == source) or not np.array_equal(trace['reference_frame'], np.arange(len(valid))):
                raise RuntimeError('Native TRAIN source or causal clock changed')
            if trace['frozen_refiner_executed'].any() or not np.array_equal(trace['requested_action'][valid], trace['executed_action'][valid]):
                raise RuntimeError('Actual full Tracker action execution changed')
            mse = float(np.mean((trace['nominal_tracker_action'][valid] - trace['same_world_teacher_action'][valid]) ** 2)) if valid.any() else None
        entry = dict(result=endpoint, usable=passed(endpoint), full_teacher_interface_passed=True,
                     valid_same_state_teacher_action_mse=mse, requested_horizon=item['horizon'])
        if entry['usable']:
            entry['official_export'] = export_episode(target, export / f'source_{source:03d}', 0)
            context_path = TRACKER_RUN / f'original_demo_geometry_context/source_{source:03d}.npz'
            with np.load(context_path) as context, np.load(export / f'source_{source:03d}_IL.npz') as labels:
                starts = labels['usable_chunk_start']
                frames = np.stack([labels['control_frame'][index:index + 8] for index in starts])
                if not np.array_equal(context['source_frames'], frames) or not np.all(labels['selected_demo'] == source):
                    raise RuntimeError('Known original-demo context and actual label source/clock differ')
                if context['original_demo_geometry'].shape != (len(starts), 8, 21):
                    raise RuntimeError('Original-demo context shape differs from actual complete chunks')
            entry['context_binding'] = dict(path=str(context_path), source_and_all_frame_indices_exact=True,
                                            actual_labelled_chunks=len(starts), original_geometry_kept_separate_from_labels=True)

        results[str(source)] = entry
        write(TRACKER_RUN / 'PARTIAL_RESULT.json', results)
        print(json.dumps(dict(source=source, steps=endpoint['actual_steps'], usable=entry['usable'], completed_sources=len(results))), flush=True)
    result = dict(execution_completed=True, optimizer_updates=0, supported_train_denominator=76,
                  original_train_denominator=80, failed_teacher_sources=plan['failed_teacher_sources'],
                  successful_sources=[int(k) for k, v in results.items() if v['usable']],
                  failed_student_sources=[int(k) for k, v in results.items() if not v['usable']],
                  total_actual_control_steps=sum(v['result']['actual_steps'] for v in results.values()), per_source=results,
                  scope='Complete frozen native TRAIN collection for the full official Generator, not common-history switching or generated policy execution.',
                  next_action=plan['automatic_next_action'])
    write(TRACKER_RUN / 'RESULT.json', result)


def readback_tracker_corpus():
    """Read every frozen Tracker outcome and the actual admitted TRAIN dataset."""
    from scripts.sugar.demo_following.demo_future.generator_dataset import ActualDemoGeometryDataset
    result = json.loads((TRACKER_RUN / 'RESULT.json').read_text())
    plan = json.loads((TRACKER_RUN / 'PROTOCOL.json').read_text())
    children = json.loads((TRACKER_RUN / 'CHILDREN.json').read_text())
    finished = [row for row in children if not row.get('externally_interrupted')]
    sources = plan['sources']
    checks = dict(all76_attempted=result['execution_completed']
                  and set(map(int, result['per_source'])) == set(sources) and len(sources) == 76,
                  all_children_finished=len(finished) == 76 and all(c.get('returncode') == 0 for c in finished)
                  and {row['source'] for row in finished} == set(sources),
                  zero_optimizer_updates=result['optimizer_updates'] == 0,
                  original80_denominator=result['original_train_denominator'] == 80
                  and set(result['failed_teacher_sources']) == {40, 46, 60, 66})
    rows, curves = [], []
    for source in sources:
        entry = result['per_source'][str(source)]
        case = TRACKER_RUN / f'source_{source:03d}/original'
        with np.load(case / 'TRACE.npz') as trace:
            n = len(trace['done'])
            valid = ~trace['done'].reshape(-1)
            actual = trace['object_state_before_w'][:, 0, :3]
            reference = trace['reference_object_pos_w'][:, 0]
            item = dict(frame_count=n == entry['result']['actual_steps'],
                        actual_action_shape=trace['executed_action'].shape == (n, 1, 29),
                        action_execution_exact=np.array_equal(trace['requested_action'][valid], trace['executed_action'][valid]),
                        teacher_never_executed=not trace['frozen_refiner_executed'].any(),
                        finite_numbers=all(np.isfinite(trace[k]).all().item() for k in trace.files
                                           if np.issubdtype(trace[k].dtype, np.number)),
                        source_exact=bool(np.all(trace['reference_id'] == source)),
                        clock_exact=np.array_equal(trace['reference_frame'], np.arange(n)),
                        termination_exact=bool(trace['done'].any()) == entry['result']['reset_or_termination'],
                        physical_admission_exact=entry['usable'] == passed(entry['result']),
                        full_teacher_query_passed=json.loads((case / 'TEACHER_INTERFACE_AUDIT.json').read_text())['passed'])
            curves.append((source, np.arange(n)[valid] * .02,
                           actual[valid, 2] - actual[0, 2], reference[valid, 2] - reference[0, 2], entry['usable']))
        prefix = TRACKER_RUN / f'official_il_data/source_{source:03d}'
        exports = [prefix.with_name(prefix.name + suffix) for suffix in ('_RAW.npz', '_IL.npz')]
        item['actual_export_admission_exact'] = all(p.exists() == entry['usable'] for p in exports)
        checks[f'source_{source:03d}'] = all(item.values())
        rows.append(dict(source=source, usable=entry['usable'], actual_steps=n, checks=item))
    dataset = ActualDemoGeometryDataset(TRACKER_RUN, ROOT / 'SUGAR/demo_ckpts/CarryBox/generator.ckpt')
    dataset_sources = {row['source'] for row in dataset.timeline_metadata}
    checks['dataset_sources_exact'] = dataset_sources == set(result['successful_sources'])
    checks['dataset_chunks_exact'] = len(dataset) == sum(
        row['context_binding']['actual_labelled_chunks'] for row in result['per_source'].values() if row['usable'])
    checks['actual_dataset_shapes_finite'] = all(
        sample['action'].shape == (8, 36) and sample['obs']['original_demo_geometry'].shape == (1, 168)
        and all(bool(value.isfinite().all()) for value in (*sample['obs'].values(), sample['action']))
        for sample in (dataset[i] for i in range(len(dataset))))
    # The adapter checks every retained old statistic while fitting only the
    # original-demo field from these actually admitted TRAIN episodes.
    dataset.get_normalizer()
    audit = dict(execution_completed=True, data_integrity_passed=all(checks.values()), checks=checks,
                 successful_count=len(dataset_sources), supported_train_denominator=76,
                 original_train_denominator=80, actual_labelled_chunks=len(dataset),
                 failed_student_sources=result['failed_student_sources'], per_source=rows,
                 externally_interrupted_initializations=sum(bool(row.get('externally_interrupted')) for row in children),
                 optimizer_updates=0, sampled_visual_inspection_completed=False,
                 scope='All real native Tracker outcomes and actual TRAIN dataset binding. Plots exclude terminal auto-reset frames. Known-reference tracking is not generated futures, camera evidence or SMP benefit.')
    write(TRACKER_RUN / 'READBACK.json', audit)
    if not audit['data_integrity_passed']:
        raise RuntimeError('Actual frozen Tracker TRAIN corpus readback failed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for group in range(4):
        fig, axes = plt.subplots(5, 4, figsize=(14, 12), sharex=True, sharey=True)
        current = curves[group * 20:(group + 1) * 20]
        for ax, (source, time, actual, reference, usable) in zip(axes.flat, current):
            ax.plot(time, actual, label='actual', color='#0173b2')
            ax.plot(time, reference, label='reference', color='#de8f05', alpha=.8)
            ax.axhline(.05, color='gray', linestyle=':', linewidth=.7)
            ax.set_title(f'Source{source}: {"usable" if usable else "FAILED"}', color='black' if usable else '#cc3311', fontsize=10)
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(current):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.supxlabel('Actual control time (s)'); fig.supylabel('Box height above respective initial height (m)')
        fig.suptitle(f'Full frozen Tracker TRAIN corpus, group{group + 1}/4; {len(dataset_sources)}/76 supported, original denominator80')
        fig.tight_layout(); fig.savefig(TRACKER_RUN / f'PHYSICAL_GROUP_{group}.png', dpi=140); plt.close(fig)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('checks', 'per_source')}), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--readback-only', action='store_true')
    mode.add_argument('--prepare-tracker-only', action='store_true')
    mode.add_argument('--tracker', action='store_true')
    mode.add_argument('--resume-tracker', action='store_true')
    mode.add_argument('--tracker-readback-only', action='store_true')
    args = parser.parse_args()
    if args.tracker_readback_only:
        readback_tracker_corpus()
    elif args.prepare_tracker_only:
        prepare_tracker_corpus()
    elif args.tracker:
        collect_tracker_corpus()
    elif args.resume_tracker:
        collect_tracker_corpus(resume=True)
    elif args.prepare_only:
        prepare()
    elif args.readback_only:
        readback()
    else:
        collect()


if __name__ == '__main__':
    main()
