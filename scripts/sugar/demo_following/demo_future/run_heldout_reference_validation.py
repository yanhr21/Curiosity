"""Frozen adaptation-held-out native tracking, using official Refiner conversion.

This is a prerequisite coverage diagnostic, not a common-history switch test.
All ten validation sources are fixed before any student result. No optimization.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
PARENT = BASE / 'matched_reference_feedback96'
RUN = PARENT / 'heldout_native_validation'
BOOT = ROOT / 'scripts/sugar/demo_following/paper_zero_wam/run_module_with_import_retry.py'
USD = BASE / 'scene_runtime/converted_g1/g1_29dof_rev_1_0_with_rubber_hand.usd'
ARMS = ('zero_feedback', 'reference_feedback')


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('x') as f:
        f.write(json.dumps(value, indent=2) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def prepare():
    scope = json.loads((PARENT / 'HELDOUT_REFERENCE_SCOPE.json').read_text())
    ids = scope['adaptation_heldout_validation_sources']
    routing = np.load(BASE / 'dataset_heading_local_v3/validation/routing.npz')
    actual_ids = sorted(set(routing['base_source_motion_id'][routing['base_task'] == 0].tolist()))
    if ids != actual_ids or ids != list(range(8, 100, 10)):
        raise RuntimeError('Validation source identity drift')
    for arm in ARMS:
        if not (PARENT / arm / 'training/model_479.pt').is_file():
            raise RuntimeError('Frozen full endpoint missing')
    RUN.mkdir(exist_ok=False)
    schedule = []
    for source in ids:
        raw = ROOT / f'SUGAR/data/CarryBox/data_{source:03d}'
        length = len(np.load(raw / 'robot_50hz.npz')['joint_pos'])
        # Teacher needs its official eight-frame future; student needs all
        # eight stride-five commands plus its next observation (36-frame tail).
        teacher_steps = min(436, length - 8)
        student_steps = min(400, teacher_steps - 36)
        if student_steps < 200:
            raise RuntimeError('Unexpectedly short source')
        case = RUN / f'source_{source:03d}'
        teacher = case / 'teacher'
        motions = teacher / 'motions'
        motions.mkdir(parents=True)
        (motions / 'data_000').symlink_to(raw, target_is_directory=True)
        (motions / 'data_001').symlink_to(ROOT / 'SUGAR/data/CarryBox/data_096', target_is_directory=True)
        protocol = dict(reference_pair=[source, 96], seed=272044,
                        steps_per_arm=teacher_steps, switch_control_frame=158,
                        reference_alignment='none',
                        alignment='Native source from frame0, no reference switch or coordinate transform; unused source96 counter-query only.',
                        scope='Frozen native validation teacher feasibility; separate initial worlds across sources. No student updates or common-history following claim.')
        write(teacher / 'PROTOCOL.json', protocol)
        schedule.append(dict(source=source, raw_frames=length, teacher_steps=teacher_steps,
                             student_steps=student_steps, eligible_fixed400=student_steps == 400))
    protocol = dict(sources=ids, schedule=schedule, arms=list(ARMS),
                    checkpoints={a: str(PARENT / a / 'training/model_479.pt') for a in ARMS},
                    training_updates=0, source_order='Ascending fixed validation IDs; all teachers before any students.',
                    reference_preparation='Only successful complete frozen teacher trajectories use the official Refiner formatter, without stabilization or synthesized frames.',
                    automatic_transition='Run all ten teachers. For each full finite frozen teacher with ten consecutive lifted frames, export its actual trace and evaluate both unchanged students. Preserve teacher failures as unavailable sources in the all10 coverage denominator.',
                    physical_checks=['Full predeclared horizon without reset', 'All recorded numeric fields finite', 'Weights frozen', 'Ten consecutive frames at least0.05m above initial box'],
                    reporting='Report all10 coverage, conditional student success and the fixed400 subset separately. Two shorter sources have explicit shorter horizons; never pad them or claim400-step success.',
                    comparison='Both full model479 endpoints use the same exported reference, seed, native reset, physics and per-source horizon. Compare complete initial physical states and common-valid reference errors.',
                    scope='Adaptation-held-out native tracking prerequisite, not common-history conditional switching or fresh confirmatory generalization. Existing validation split used previously; author Tracker pretraining exposure unknown.',
                    next_action='If native coverage fails, inspect actual failure and teacher/reference compatibility before training. If useful coverage passes, predeclare common-history selected-reference tests and actual visuals. No automatic Generator training.')
    write(RUN / 'PROTOCOL.json', protocol)
    print(json.dumps(protocol), flush=True)


def passed(result):
    return all(result[k] for k in ('execution_completed', 'all_numeric_finite',
               'full_budget_without_reset', 'at_least_ten_consecutive_lifted_frames')) and result['teacher_frozen']['passed']


def export_native(case, source):
    path = ROOT / 'SUGAR/scripts/sugar_rl/process_refiner_rollout.py'
    spec = importlib.util.spec_from_file_location('official_refiner_formatter', path)
    formatter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formatter)
    trace = dict(np.load(case / 'teacher/original/TRACE.npz'))
    if trace['done'].any():
        raise RuntimeError('Cannot export a reset-spliced trace')
    body = trace['robot_body_state_before_w'][:, 0]
    obj = trace['object_state_before_w'][:, 0]
    forces = trace['contact_force_history_before_w'][:, 0]
    max_force = np.linalg.norm(forces, axis=-1).max(axis=-1)
    hands = (max_force[:, 0] > .1) & (max_force[:, 1] > .1)
    raw = dict(joint_pos=trace['joint_pos_before'][:, 0], joint_vel=trace['joint_vel_before'][:, 0],
               body_pos_w=body[..., :3], body_quat_w=body[..., 3:7],
               body_lin_vel_w=body[..., 7:10], body_ang_vel_w=body[..., 10:13],
               obj_pos_w=obj[:, :3], obj_quat_w=obj[:, 3:7], obj_lin_vel_w=obj[:, 7:10],
               obj_ang_vel_w=obj[:, 10:13], hands_contact_label=hands)
    out = case / 'official_refined_export'
    out.mkdir()
    measured = out / f'motion_0_env_0_t0-{len(obj)-1}_idx_0.npz'
    np.savez_compressed(measured, **raw)
    formatter.process_data_to_rl_dataset(str(measured), str(out), 'hands_contact_label',
                                         stabilize_initial_frames_flag=False)
    target = out / 'rl_dataset/data_000_000_t0'
    robot = dict(np.load(target / 'robot_50hz.npz'))
    if not all(np.array_equal(value, raw[k]) for k, value in robot.items()):
        raise RuntimeError('Official formatter changed measured robot fields')
    if not np.array_equal(np.load(target / 'contact_labels_50hz.npy'), hands):
        raise RuntimeError('Official formatter changed contact labels')
    with (target / 'obj_motion_global_50hz.pkl').open('rb') as f:
        exported_object = pickle.load(f)
    expected_object = dict(obj_trans=raw['obj_pos_w'],
                           obj_rot=formatter.matrix_from_quat_np(raw['obj_quat_w']),
                           obj_lin_vel=raw['obj_lin_vel_w'], obj_ang_vel=raw['obj_ang_vel_w'])
    if exported_object.keys() != expected_object.keys() or not all(
            np.array_equal(exported_object[k], v) for k, v in expected_object.items()):
        raise RuntimeError('Official formatter changed measured object fields')
    write(out / 'RESULT.json', dict(passed=True, source=source, frames=len(obj),
          official_formatter=str(path), stabilize_initial_frames=False,
          scope='Actual measured body/joint motion as reference targets; executed29-D actions remain separate and preserved. Reference contact is not live tactile.'))
    return target


def execute():
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the recorded retained compute step')
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    record = RUN / 'CHILDREN.json'
    if record.exists():
        raise RuntimeError('Existing execution: inspect exact children, never repeat blindly')
    records = []

    def collect(tag, case_run, output, controller, checkpoint=None):
        cmd = [sys.executable, str(BOOT), 'scripts.sugar.demo_following.demo_future.collect_refiner_pair',
               '--headless', '--controller', controller, '--robot-usd', str(USD),
               '--run', str(case_run), '--arm', 'original', '--output', str(output)]
        if checkpoint:
            cmd += ['--tracker-checkpoint', checkpoint]
        with (RUN / (tag + '.log')).open('x') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(tag=tag, pid=proc.pid, pgid=os.getpgid(proc.pid), command=cmd)
            records.append(row)
            write(record, records)
            row['returncode'] = proc.wait()
            write(record, records)
        artifact = output / 'RESULT.json'
        if row['returncode'] or not artifact.exists():
            raise RuntimeError('Runtime failure; preserve exact child/log: ' + tag)
        result = json.loads(artifact.read_text())
        print(json.dumps(dict(completed=tag, actual_steps=result['actual_steps'], passed=passed(result))), flush=True)
        return result

    results = {}
    for row in plan['schedule']:
        source = row['source']
        case = RUN / f'source_{source:03d}'
        result = collect(f'teacher_{source:03d}', case / 'teacher', case / 'teacher/original', 'refiner')
        results[str(source)] = dict(teacher=result, teacher_passed=passed(result), students={})
        write(RUN / 'PARTIAL_RESULT.json', results)
    for row in plan['schedule']:
        source = row['source']
        entry = results[str(source)]
        if not entry['teacher_passed']:
            entry['student_status'] = 'unavailable_failed_teacher_reference'
            write(RUN / 'PARTIAL_RESULT.json', results)
            continue
        case = RUN / f'source_{source:03d}'
        target = export_native(case, source)
        evaluation = case / 'evaluation'
        motions = evaluation / 'motions'
        motions.mkdir(parents=True)
        (motions / 'data_000').symlink_to(target, target_is_directory=True)
        old_bank = BASE / 'tracker_refined96_90_feasibility/motions'
        old_motion = sorted(old_bank.glob('data_*'))[0].resolve()
        (motions / 'data_001').symlink_to(old_motion, target_is_directory=True)
        protocol = json.loads((case / 'teacher/PROTOCOL.json').read_text())
        protocol.update(steps_per_arm=row['student_steps'], scope=plan['scope'])
        write(evaluation / 'PROTOCOL.json', protocol)
        traces = {}
        for arm in ARMS:
            out = evaluation / arm
            result = collect(f'{arm}_{source:03d}', evaluation, out, 'tracker', plan['checkpoints'][arm])
            entry['students'][arm] = dict(result=result, passed=passed(result))
            traces[arm] = dict(np.load(out / 'TRACE.npz'))
            write(RUN / 'PARTIAL_RESULT.json', results)
        fields = ('robot_body_state_before_w', 'robot_root_state_before_w', 'joint_pos_before',
                  'joint_vel_before', 'object_state_before_w', 'reference_object_pos_w', 'reference_body_pos_w')
        exact = {k: bool(np.array_equal(traces[ARMS[0]][k][0], traces[ARMS[1]][k][0])) for k in fields}
        entry['matched_initial_physical_state'] = exact
        end = min(len(t['done']) for t in traces.values())
        valid = np.ones(end, dtype=bool)
        for t in traces.values():
            valid &= ~t['done'][:end].reshape(-1)
        entry['common_valid_frames'] = int(valid.sum())
        entry['common_box_coordinate_rmse_m'] = {
            arm: float(np.sqrt(np.mean((t['object_state_before_w'][:end, 0, :3][valid] - t['reference_object_pos_w'][:end, 0][valid]) ** 2))) if valid.any() else None
            for arm, t in traces.items()}
        write(RUN / 'PARTIAL_RESULT.json', results)
        if not all(exact.values()):
            raise RuntimeError('Matched native initial physical state differs')
    counts = {arm: sum(e['students'].get(arm, {}).get('passed', False) for e in results.values()) for arm in ARMS}
    full_ids = [str(r['source']) for r in plan['schedule'] if r['eligible_fixed400']]
    result = dict(execution_completed=True, training_updates=0, all_source_denominator=len(results),
                  teacher_usable_sources=sum(e['teacher_passed'] for e in results.values()),
                  student_success_counts_over_all_sources=counts,
                  fixed400_source_denominator=len(full_ids),
                  fixed400_success_counts={a: sum(results[s]['students'].get(a, {}).get('passed', False) for s in full_ids) for a in ARMS},
                  per_source=results, scope=plan['scope'], next_action=plan['next_action'])
    write(RUN / 'RESULT.json', result)
    print(json.dumps(result), flush=True)


def query_first_source():
    """Same-state full teacher queries on the fixed first validation source."""
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the recorded retained compute step')
    completed = json.loads((RUN / 'RESULT.json').read_text())
    if not completed['execution_completed']:
        raise RuntimeError('Complete the fixed native coverage first')
    case = RUN / 'source_008'
    evaluation = case / 'evaluation'
    teacher_bank = case / 'original_teacher_query_bank'
    bank_audit = json.loads((teacher_bank / 'RESULT.json').read_text())
    if not bank_audit['passed']:
        raise RuntimeError('Original teacher timeline preparation failed')
    protocol = dict(source=8, selection='First source in the predeclared ascending validation schedule, chosen before teacher-query results.',
                    arms=list(ARMS), teacher_bank=str(teacher_bank),
                    teacher_reference='Original source8 frames0:436 without resampling; unused slot1 is the already-audited original96 teacher timeline matching its refined bank. No measured future becomes a teacher target.',
                    training_updates=0, actual_actions='Same unchanged full student endpoints; teacher queries never execute.',
                    checks='All old trace/startup fields exact; both full890-D teacher interfaces agree; retain both complete query outcomes.',
                    scope='Same actual state labels diagnose student discrepancy; teacher recovery from those states is not established.')
    path = RUN / 'SOURCE8_TEACHER_QUERY_PROTOCOL.json'
    if path.exists():
        raise RuntimeError('Query already launched; inspect recorded children')
    write(path, protocol)
    records, results = [], {}
    for arm in ARMS:
        out = evaluation / ('teacher_query_' + arm)
        cmd = [sys.executable, str(BOOT), 'scripts.sugar.demo_following.demo_future.collect_refiner_pair',
               '--headless', '--controller', 'tracker', '--robot-usd', str(USD),
               '--run', str(evaluation), '--arm', 'original', '--output', str(out),
               '--tracker-checkpoint', str(PARENT / arm / 'training/model_479.pt'),
               '--audit-teacher-bank', str(teacher_bank)]
        with (RUN / ('teacher_query_' + arm + '.log')).open('x') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(arm=arm, pid=proc.pid, pgid=os.getpgid(proc.pid), command=cmd)
            records.append(row)
            write(RUN / 'SOURCE8_QUERY_CHILDREN.json', records)
            row['returncode'] = proc.wait()
            write(RUN / 'SOURCE8_QUERY_CHILDREN.json', records)
        if row['returncode'] or not (out / 'RESULT.json').exists():
            raise RuntimeError('Query runtime failure: ' + arm)
        trace = dict(np.load(out / 'TRACE.npz'))
        old = dict(np.load(evaluation / arm / 'TRACE.npz'))
        old_equal = {k: bool(np.array_equal(v, trace[k])) for k, v in old.items()}
        startup = dict(np.load(out / 'STARTUP.npz'))
        old_startup = dict(np.load(evaluation / arm / 'STARTUP.npz'))
        startup_equal = old_startup.keys() == startup.keys() and all(np.array_equal(v, startup[k]) for k, v in old_startup.items())
        valid = ~trace['done'].reshape(-1)
        mse = np.mean((trace['nominal_tracker_action'] - trace['same_world_teacher_action']) ** 2, axis=(1, 2))
        interface = json.loads((out / 'TEACHER_INTERFACE_AUDIT.json').read_text())
        windows = {}
        for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 400)):
            use = valid & (np.arange(len(valid)) >= lo) & (np.arange(len(valid)) < hi)
            windows[f'{lo}:{hi}'] = dict(valid_frames=int(use.sum()), action_mse=float(mse[use].mean()) if use.any() else None)
        results[arm] = dict(actual_steps=len(valid), old_trace_fields_exact=old_equal,
                            all_old_trace_fields_exact=all(old_equal.values()), all_startup_fields_exact=startup_equal,
                            teacher_interfaces=interface, first_frame_action_mse=float(mse[0]),
                            valid_action_mse=float(mse[valid].mean()), windows=windows)
        write(RUN / 'SOURCE8_TEACHER_QUERY_PARTIAL.json', results)
    result = dict(execution_completed=True, optimizer_updates=0, arms=results,
                  readback_passed=all(x['all_old_trace_fields_exact'] and x['all_startup_fields_exact'] and x['teacher_interfaces']['passed'] for x in results.values()),
                  scope=protocol['scope'])
    write(RUN / 'SOURCE8_TEACHER_QUERY_RESULT.json', result)
    print(json.dumps(result), flush=True)


def released_floor():
    """Released510-D Tracker on the same eight teacher-supported references."""
    if not os.environ.get('SLURM_STEP_ID') or socket.gethostname().startswith(('login', 'mgmtserver')):
        raise RuntimeError('Use the recorded retained compute step')
    coverage = json.loads((RUN / 'RESULT.json').read_text())
    query = json.loads((RUN / 'SOURCE8_TEACHER_QUERY_RESULT.json').read_text())
    if not coverage['execution_completed'] or not query['readback_passed']:
        raise RuntimeError('Coverage and exact same-state query must be inspected first')
    ids = sorted(int(s) for s, value in coverage['per_source'].items() if value['teacher_passed'])
    protocol_path = RUN / 'RELEASED_FLOOR_PROTOCOL.json'
    if protocol_path.exists():
        raise RuntimeError('Released floor already launched; inspect exact children')
    protocol = dict(sources=ids, all_source_denominator=10,
                    checkpoint=str(ROOT / 'SUGAR/demo_ckpts/CarryBox/tracker.pt'),
                    controller='Unchanged author-released full510-D Tracker',
                    physical_conditions='Same exported reference, native reset, seed272044, physics and per-source horizon as both full model479 endpoints.',
                    optimizer_updates=0,
                    scope='Diagnostic released-controller floor. Different input interfaces and training histories; not a single-factor estimate of forgetting or selected-demo conditioning. Author pretraining source exposure unknown.',
                    automatic_next_action='Complete all eight regardless of failure. Compare the released floor with the adapted controllers before selecting a broader TRAIN-only adaptation or Generator budget.')
    write(protocol_path, protocol)
    records, results = [], {}
    for source in ids:
        case = RUN / f'source_{source:03d}'
        evaluation = case / 'evaluation'
        out = evaluation / 'released_tracker'
        cmd = [sys.executable, str(BOOT), 'scripts.sugar.demo_following.demo_future.collect_refiner_pair',
               '--headless', '--controller', 'tracker', '--robot-usd', str(USD),
               '--run', str(evaluation), '--arm', 'original', '--output', str(out)]
        with (RUN / f'released_floor_{source:03d}.log').open('x') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row = dict(source=source, pid=proc.pid, pgid=os.getpgid(proc.pid), command=cmd)
            records.append(row)
            write(RUN / 'RELEASED_FLOOR_CHILDREN.json', records)
            row['returncode'] = proc.wait()
            write(RUN / 'RELEASED_FLOOR_CHILDREN.json', records)
        if row['returncode'] or not (out / 'RESULT.json').exists():
            raise RuntimeError('Released floor runtime failure: ' + str(source))
        result = json.loads((out / 'RESULT.json').read_text())
        current = dict(np.load(out / 'TRACE.npz'))
        old = dict(np.load(evaluation / 'zero_feedback/TRACE.npz'))
        fields = ('robot_body_state_before_w', 'robot_root_state_before_w', 'object_state_before_w',
                  'joint_pos_before', 'joint_vel_before', 'reference_body_pos_w', 'reference_object_pos_w')
        initial = {k: bool(np.array_equal(current[k][0], old[k][0])) for k in fields}
        startup = dict(np.load(out / 'STARTUP.npz'))
        old_startup = dict(np.load(evaluation / 'zero_feedback/STARTUP.npz'))
        startup_exact = startup.keys() == old_startup.keys() and all(np.array_equal(v, old_startup[k]) for k, v in startup.items())
        results[str(source)] = dict(result=result, passed=passed(result),
                                    initial_physical_state_exact=initial, all_startup_fields_exact=startup_exact)
        write(RUN / 'RELEASED_FLOOR_PARTIAL.json', results)
        print(json.dumps(dict(source=source, steps=result['actual_steps'], passed=passed(result))), flush=True)
        if not all(initial.values()) or not startup_exact:
            raise RuntimeError('Released comparison physical initialization differs')
    result = dict(execution_completed=True, optimizer_updates=0, evaluated_sources=len(ids),
                  all_source_denominator=10, successes=sum(v['passed'] for v in results.values()),
                  per_source=results, scope=protocol['scope'], next_action=protocol['automatic_next_action'])
    write(RUN / 'RELEASED_FLOOR_RESULT.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--query-first-source', action='store_true')
    parser.add_argument('--released-floor', action='store_true')
    args = parser.parse_args()
    if sum((args.prepare_only, args.query_first_source, args.released_floor)) > 1:
        parser.error('Preparation and actual query are different stages')
    if args.prepare_only:
        prepare()
    elif args.query_first_source:
        query_first_source()
    elif args.released_floor:
        released_floor()
    else:
        execute()


if __name__ == '__main__':
    main()
