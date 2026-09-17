"""Original numeric-demo geometry for the complete official Generator interface.

This prepares project conditioning data only. It creates no model, optimizer or
physical rollout. Reference geometry is not an official SMP latent or human video.
"""
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/demo_following/demo_future_smp_v1'
RUN = BASE / 'matched_reference_feedback96/reference_feedback/evaluation'
SOURCE_RUN = BASE / 'refiner_aligned96_90_feasibility'
sys.path.insert(0, str(ROOT / 'scripts/sugar/smp'))
from sugar_g1_box_schema import SOURCE_BODY_NAMES

CONTEXT_BODY_NAMES = ('torso_link', 'left_rubber_hand', 'right_rubber_hand',
                      'left_ankle_roll_link', 'right_ankle_roll_link')


def geometry_context(robot, obj, frames, math):
    """Known original-demo poses expressed in its current torso frame.

    Deliberately accepts no actual trajectory, actions or future command packet.
    """
    torso_id = SOURCE_BODY_NAMES.index('torso_link')
    limb_ids = [SOURCE_BODY_NAMES.index(n) for n in
                ('left_rubber_hand', 'right_rubber_hand',
                 'left_ankle_roll_link', 'right_ankle_roll_link')]
    position = robot['body_pos_w'][frames]
    origin = position[0, torso_id]
    rotation = math.matrix_from_quat_np(robot['body_quat_w'][frames[0], torso_id])
    inverse = rotation.T
    box = obj['obj_trans'][frames]
    box_position = (inverse @ (box - origin)[..., None])[..., 0]
    box_orientation = (inverse @ obj['obj_rot'][frames])[..., :, :2].reshape(len(frames), 6)
    limb_relative_box = (inverse @ (position[:, limb_ids] - box[:, None])[..., None])[..., 0]
    context = np.concatenate([box_position, box_orientation, limb_relative_box.reshape(len(frames), 12)], axis=-1)
    if context.shape != (8, 21) or not np.isfinite(context).all():
        raise RuntimeError('Original-demo geometry contract failed')
    return context.astype(np.float32)



def prepare_native_train():
    """Prepare only known original-demo context; no actual future labels exist yet."""
    corpus = BASE / 'tracker_train_generator_corpus76'
    plan = json.loads((corpus / 'PROTOCOL.json').read_text())
    source = ROOT / 'SUGAR/scripts/sugar_rl/process_refiner_rollout.py'
    spec = importlib.util.spec_from_file_location('official_reference_math', source)
    math = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    routing = np.load(BASE / 'dataset_heading_local_v3/train/routing.npz')
    train_ids = set(routing['base_source_motion_id'][routing['base_task'] == 0].tolist())
    if not set(plan['sources']) <= train_ids:
        raise RuntimeError('Native geometry must remain inside the original TRAIN split')
    out = corpus / 'original_demo_geometry_context'
    out.mkdir(exist_ok=False)
    summary = {}
    for item in plan['schedule']:
        source_id = item['source']
        folder = ROOT / f'SUGAR/data/CarryBox/data_{source_id:03d}'
        robot = dict(np.load(folder / 'robot_50hz.npz'))
        with (folder / 'obj_motion_global_50hz.pkl').open('rb') as f:
            obj = pickle.load(f)
        starts = np.arange(0, item['horizon'] - 35, 5, dtype=np.int64)
        frames = starts[:, None] + np.arange(0, 36, 5, dtype=np.int64)[None]
        if frames.max() >= min(len(robot['body_pos_w']), len(obj['obj_trans'])):
            raise RuntimeError('Context extends beyond the real original demonstration')
        context = np.stack([geometry_context(robot, obj, selected, math) for selected in frames])
        np.savez_compressed(out / f'source_{source_id:03d}.npz', original_demo_geometry=context,
                            source_frames=frames, control_frame=starts)
        summary[str(source_id)] = dict(planned_chunks=len(starts), context_shape=list(context.shape),
                                      first_frame=int(starts[0]), last_context_frame=int(frames[-1, -1]),
                                      original_robot_frames=len(robot['body_pos_w']), native_clock_without_resampling=True)
    result = dict(execution_completed=True, source_count=len(summary), planned_chunks=sum(r['planned_chunks'] for r in summary.values()),
                  per_source=summary, optimizer_updates=0, actual_rollouts=0,
                  context_only=True, actual_tracker_labels_available=False,
                  context_fields='Each of8 samples: box position3 and first-two rotation columns6, four limb positions relative to box12, all in the fixed original-demo torso frame at chunk start.',
                  phase='Native raw source frame equals recorded control/reference frame. No normalized time remapping, no future extrapolation or best-window search.',
                  excludes=['29-D joint targets', '36-D command targets', 'actual rollout future states', 'actual actions', 'forces', 'reward', 'source ID as an input'],
                  data_binding='After a full successful actual Tracker rollout and official formatter export, require exact source and all chunk frame indices before binding this known context to labels. A prepared context is not an executed future or usable labelled training episode.',
                  scope='Original retargeted numeric TRAIN demonstration geometry for the full official Generator adaptation, not human video or an SMP latent.')
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_source'}), flush=True)


def prepare_native_validation():
    """Freeze existing actual validation trajectories, outside Generator TRAIN."""
    from scripts.sugar.demo_following.demo_future.export_tracker_il import export_episode
    from scripts.sugar.demo_following.demo_future.run_heldout_reference_validation import passed, write
    parent = BASE / 'matched_train_coverage128'
    if not json.loads((parent / 'RESULT.json').read_text())['broad_execution_floor_passed']:
        raise RuntimeError('Require completed full frozen native tracking endpoints')
    sources = [8, 28, 38, 58, 68, 78, 88, 98]
    split_ids = {}
    for split in ('train', 'validation', 'test'):
        with np.load(BASE / f'dataset_heading_local_v3/{split}/routing.npz') as routing:
            split_ids[split] = set(routing['base_source_motion_id'][routing['base_task'] == 0].tolist())
    if not set(sources) <= split_ids['validation'] or set(sources) & (split_ids['train'] | split_ids['test']):
        raise RuntimeError('Original validation split differs')
    out = BASE / 'generator_native_validation8'
    out.mkdir(exist_ok=False)
    labels_out = out / 'official_il_data'; labels_out.mkdir()
    context_out = out / 'original_demo_geometry_context'; context_out.mkdir()
    spec = importlib.util.spec_from_file_location('official_reference_math', ROOT / 'SUGAR/scripts/sugar_rl/process_refiner_rollout.py')
    math = importlib.util.module_from_spec(spec); spec.loader.exec_module(math)
    raw = {}
    for source in sources:
        folder = ROOT / f'SUGAR/data/CarryBox/data_{source:03d}'
        robot = dict(np.load(folder / 'robot_50hz.npz'))
        with (folder / 'obj_motion_global_50hz.pkl').open('rb') as f:
            obj = pickle.load(f)
        raw[source] = (robot, obj)
    summary = {}
    for i, source in enumerate(sources):
        episode = parent / f'broad_train/native_validation/source_{source:03d}'
        endpoint = json.loads((episode / 'RESULT.json').read_text())
        if not passed(endpoint):
            raise RuntimeError('Do not export a failed validation trajectory')
        export = export_episode(episode, labels_out / f'source_{source:03d}', 0)
        robot, obj = raw[source]
        other_source = sources[(i + 1) % len(sources)]
        other_robot, other_obj = raw[other_source]
        with np.load(labels_out / f'source_{source:03d}_IL.npz') as il:
            starts = il['usable_chunk_start']
            frames = np.stack([il['control_frame'][start:start + 8] for start in starts])
            if not np.all(il['selected_demo'] == source) or not np.all(np.diff(frames, axis=-1) == 5):
                raise RuntimeError('Validation source or frame binding differs')
        context = np.stack([geometry_context(robot, obj, row, math) for row in frames])
        # A wrong-demo input intervention, not another observed physical future.
        # Match normalized raw clip phase; do not search for a favorable window.
        other_frames = np.rint(frames / (len(robot['body_pos_w']) - 1)
                               * (len(other_robot['body_pos_w']) - 1)).astype(np.int64)
        other_context = np.stack([geometry_context(other_robot, other_obj, row, math) for row in other_frames])
        np.savez_compressed(context_out / f'source_{source:03d}.npz', original_demo_geometry=context,
                            alternate_demo_geometry=other_context, source_frames=frames,
                            alternate_source_frames=other_frames, control_frame=frames[:, 0])
        summary[str(source)] = dict(actual_steps=endpoint['actual_steps'], chunks=len(starts),
                                    wrong_context_source=other_source, official_export=export,
                                    selected_source_and_clock_exact=True)
    result = dict(execution_completed=True, split='validation', sources=sources,
                  source_count=8, original_validation_denominator=10, failed_teacher_sources=[18, 48],
                  total_chunks=sum(row['chunks'] for row in summary.values()), per_source=summary,
                  optimizer_updates=0, new_physics_rollouts=0, statistics_fitted=False,
                  wrong_demo_rule='Cyclic next source in fixed sorted validation list; same normalized original-clip phase, all existing observation fields and actual labels held fixed.',
                  scope='Frozen observed-future command prediction on the already-used native validation set. Wrong context is an input intervention, not a counterfactual physical label. Not an untouched test set, generated closed-loop success or SMP benefit.')
    write(out / 'RESULT.json', result)
    print(json.dumps(result), flush=True)


def prepare_branch_point(run=RUN, out=None):
    """Recover the actual shared-world TRAIN branch using the official formatter."""
    from scripts.sugar.demo_following.demo_future.run_heldout_reference_validation import write
    run = Path(run)
    plan = json.loads((run / 'PROTOCOL.json').read_text())
    switch = plan['switch_control_frame']
    if switch < 5 or plan['reference_pair'] != [96, 90] or not json.loads((run / 'READBACK.json').read_text())['data_pair_feasibility_passed']:
        raise RuntimeError('Require the complete successful actual96/90 TRAIN branch with causal history')
    out = Path(out) if out is not None else BASE / 'generator_actual_branch_point'
    out.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('official_branch_formatter', ROOT / 'SUGAR/scripts/sugar_rl/process_tracker_rollout.py')
    math = importlib.util.module_from_spec(spec); spec.loader.exec_module(math)
    phase = switch % 5
    frames = switch + np.arange(8) * 5
    inputs, labels, contexts, actual_actions, worlds, rows = [], [], [], [], [], {}
    for arm, source in [('original', 96), ('alternate', 90)]:
        with np.load(run / arm / 'TRACE.npz') as trace:
            if trace['done'].any() or not np.all(trace['reference_id'][frames] == source):
                raise RuntimeError('Branch targets cross a reset or selected-source boundary')
            worlds.append({key: trace[key][switch].copy() for key in (
                'robot_body_state_before_w', 'robot_root_state_before_w', 'object_state_before_w',
                'joint_pos_before', 'joint_vel_before')})
            target = trace['reference_command'][frames, 0].copy()
            previous = trace['reference_command'][switch - 5, 0].copy()
            actual_actions.append(trace['executed_action'][frames, 0].copy())
        with np.load(run / f'official_il_data/{arm}_RAW.npz') as old:
            count = len(old['obj_pos_b'])
            shifted = {key: old[key][phase:].copy() if old[key].ndim and len(old[key]) == count else old[key].copy()
                       for key in old.files}
            shifted['rollout_start'] = np.asarray(phase)
        raw_path = out / f'{arm}_PHASE_RAW.npz'; np.savez_compressed(raw_path, **shifted)
        episode, start_padded, end_padded = math.process_data_to_dict(raw_path)
        # Preserve the official initial duplicate row for phase-zero episodes.
        # Select only real future rows, after that prefix and before any tail padding.
        index = (switch - phase) // 5 + int(start_padded)
        real_rows_end = int(start_padded) + len(shifted['obj_pos_b'][::5])
        if start_padded != (phase == 0) or index + 8 > real_rows_end or not np.array_equal(episode['action'][index:index + 8], target) or not np.array_equal(episode['last_action'][index], previous):
            raise RuntimeError('Official branch formatter changed actual labels or t-5 command history')
        inputs.append({key: episode[key][index:index + 1].copy() for key in episode if key != 'action'})
        labels.append(target)
        raw = ROOT / f'SUGAR/data/CarryBox/data_{source:03d}'
        robot = dict(np.load(raw / 'robot_50hz.npz'))
        with (raw / 'obj_motion_global_50hz.pkl').open('rb') as stream:
            obj = pickle.load(stream)
        with np.load(SOURCE_RUN / arm / 'TRACE.npz') as provenance:
            source_frames = provenance['reference_frame'][frames].copy()
            if not np.all(provenance['reference_id'][frames] == source):
                raise RuntimeError('Original raw-demo context provenance differs')
        contexts.append(geometry_context(robot, obj, source_frames, math))
        rows[arm] = dict(source=source, source_frames=source_frames.tolist(), actual_control_frames=frames.tolist(),
                         official_phase_offset=phase, official_row=index, labels_and_previous_command_exact=True)
    checks = dict(world_exact=all(np.array_equal(value, worlds[1][key]) for key, value in worlds[0].items()),
                  causal_object_and_history_exact=all(np.array_equal(inputs[0][key], inputs[1][key]) for key in ('obj_pos_b', 'obj_ori_b', 'last_action')),
                  futures_differ=not np.array_equal(labels[0], labels[1]),
                  original_demo_geometry_differs=not np.array_equal(contexts[0], contexts[1]),
                  existing_goals_differ=any(not np.array_equal(inputs[0][key], inputs[1][key]) for key in ('target_obj_pos_b', 'target_obj_ori_b')))
    if not all(checks.values()):
        raise RuntimeError('Actual branch-point identifiability prerequisites differ')
    np.savez_compressed(out / 'BRANCH_SAMPLES.npz', **{key: np.stack([value[key] for value in inputs]) for key in inputs[0]},
        original_demo_geometry=np.asarray(contexts)[:, None].reshape(2, 1, 168),
        future_command_target=np.asarray(labels), actual_executed_action_29d=np.asarray(actual_actions))
    report = dict(execution_completed=True, checks=checks, per_arm=rows, optimizer_updates=0, new_physics_steps=0,
        scope=f'Two real TRAIN branch-point examples at control{switch}, full official formatter with unchanged5-frame stride and phase{phase}. Actual common world and causal history, distinct8x36 observed reference-command targets and separate29-D executed actions. Numeric original-demo geometry only; no generated or synthetic trajectory.',
        diagnostic_goal_control='Original goal inputs are preserved in the archive. A later identifiability diagnostic must explicitly zero both normalized goal fields identically to remove their existing source cue; such missing-goal input is diagnostic only, not a validated deployment interface.',
        next_action='Predeclare a bounded full official Generator common-world branch fitting diagnostic with matched zero/demo geometry arms and shared missing-goal input. Verify independent-noise correct/wrong/zero behavior and the two-target unconditional mean baseline. Do not call fitting two TRAIN futures generalization or open generated physical execution.')
    write(out / 'RESULT.json', report)
    print(json.dumps(report), flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--native-train-only', action='store_true')
    mode.add_argument('--native-validation-only', action='store_true')
    mode.add_argument('--branch-point-only', action='store_true')
    args = parser.parse_args()
    if args.native_train_only:
        return prepare_native_train()
    if args.native_validation_only:
        return prepare_native_validation()
    if args.branch_point_only:
        return prepare_branch_point()

    source = ROOT / 'SUGAR/scripts/sugar_rl/process_refiner_rollout.py'
    spec = importlib.util.spec_from_file_location('official_reference_math', source)
    math = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    if not json.loads((RUN / 'READBACK.json').read_text())['data_pair_feasibility_passed']:
        raise RuntimeError('Successful actual training pair required')
    plan = json.loads((SOURCE_RUN / 'PROTOCOL.json').read_text())
    sources = plan['reference_pair']
    if sources != [96, 90]:
        raise RuntimeError('Explicit TRAIN source pair changed')
    routing = np.load(BASE / 'dataset_heading_local_v3/train/routing.npz')
    train_ids = set(routing['base_source_motion_id'][routing['base_task'] == 0].tolist())
    if not set(sources) <= train_ids:
        raise RuntimeError('Context source is not TRAIN')
    source_data = {}
    for source_id in sources:
        folder = ROOT / f'SUGAR/data/CarryBox/data_{source_id:03d}'
        robot = dict(np.load(folder / 'robot_50hz.npz'))
        with (folder / 'obj_motion_global_50hz.pkl').open('rb') as f:
            obj = pickle.load(f)
        source_data[source_id] = (robot, obj)
    primary_length = len(source_data[sources[0]][0]['body_pos_w'])
    out = RUN / 'original_demo_geometry_context'
    out.mkdir(exist_ok=False)
    summary = {}
    for arm_index, arm in enumerate(('original', 'alternate')):
        archive = dict(np.load(RUN / f'official_il_data/{arm}_IL.npz'))
        old = dict(np.load(SOURCE_RUN / arm / 'TRACE.npz'))
        names = np.load(SOURCE_RUN / arm / 'STARTUP.npz')['body_names'].tolist()
        # The importer can reorder unused fixed bodies. Require the exact
        # named indices used here, not equality of unrelated body ordering.
        context_body_indices = {n: names.index(n) for n in CONTEXT_BODY_NAMES}
        if any(context_body_indices[n] != SOURCE_BODY_NAMES.index(n) for n in CONTEXT_BODY_NAMES):
            raise RuntimeError('A context body index differs from the audited source schema')
        selected = sources[arm_index]
        robot, obj = source_data[selected]
        source_length = len(robot['body_pos_w'])
        rows = []
        frames_rows = []
        target_rows = []
        actual_action_rows = []
        start_frames = []
        alternate_rows = []
        for index in archive['usable_chunk_start']:
            timeline = archive['control_frame'][index:index + 8]
            if timeline[0] < plan['switch_control_frame'] or not np.all(np.diff(timeline) == 5):
                raise RuntimeError('Context must start after selection without padding')
            # This is the original predeclared causal normalized clock; do not
            # renormalize the cropped450-frame refined bank to the raw endpoint.
            frames = np.rint(timeline / (primary_length - 1) * (source_length - 1)).astype(np.int64)
            if not np.array_equal(frames, old['reference_frame'][timeline]):
                raise RuntimeError('Original source-frame mapping differs from measured reference provenance')
            if not np.all(old['reference_id'][timeline] == selected):
                raise RuntimeError('Source identity changes inside a selected-demo chunk')
            rows.append(geometry_context(robot, obj, frames, math))
            frames_rows.append(frames)
            other_robot, other_obj = source_data[sources[1 - arm_index]]
            other_frames = np.rint(timeline / (primary_length - 1) * (len(other_robot['body_pos_w']) - 1)).astype(np.int64)
            alternate_rows.append(geometry_context(other_robot, other_obj, other_frames, math))
            target_rows.append(archive['action'][index:index + 8])
            actual_action_rows.append(archive['executed_action_29d'][index:index + 8])
            start_frames.append(timeline[0])
        contexts = np.asarray(rows)
        other_contexts = np.asarray(alternate_rows)
        np.savez_compressed(out / f'{arm}.npz', original_demo_geometry=contexts,
                            alternate_demo_geometry=other_contexts,
                            source_frames=np.asarray(frames_rows),
                            control_frame=np.asarray(start_frames),
                            official_command_target=np.asarray(target_rows),
                            actual_executed_action=np.asarray(actual_action_rows),
                            official_il_chunk_start=archive['usable_chunk_start'])
        summary[arm] = dict(source=selected, chunks=len(rows), context_shape=list(contexts.shape),
                            target_shape=list(np.asarray(target_rows).shape),
                            context_body_indices=context_body_indices,
                            mapped_source_frames_exact=True,
                            selected_vs_other_context_rms=float(np.sqrt(np.mean((contexts - other_contexts) ** 2))),
                            all_chunks_have_distinct_other_context=bool(np.all(np.any(contexts != other_contexts, axis=(1, 2)))))
    result = dict(execution_completed=True, optimizer_updates=0, new_physics_rollouts=0,
                  context_fields_per_frame=['box_position_in_current_demo_torso_frame:3',
                       'box_rotation_first_two_columns_in_same_frame:6',
                       'left_hand/right_hand/left_ankle/right_ankle_relative_box_positions_in_same_frame:12'],
                  offsets_control_frames=[0, 5, 10, 15, 20, 25, 30, 35],
                  coordinate_rule='One original-demo current torso frame per chunk, held fixed throughout all eight known reference samples.',
                  phase_rule='Raw source frame round(refined_timeline_index*(raw_source_length-1)/(raw96_length-1)); verified against the original causal reference trace, no best-window search.',
                  context_excludes=['29-D joint targets', '36-D reference command packet', 'actual rollout future states', 'actual forces', 'executed actions', 'learned reward scores', 'source ID as a model input'],
                  arms=summary,
                  scope='Prepared TRAIN original retargeted numeric-demo geometry, not human video, an SMP latent, a learned world model or future-generation success. Targets and actual actions are separately named arrays; they must never be concatenated into deployment context.',
                  next_action='After frozen coverage results, evaluate a labelled context adaptation of the full released Generator with equal-init/no-context/alternate-context controls. Keep original model, normalizer and diffusion solver. No training is performed by this preparation.')
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
