"""Adapt actual paired Tracker traces through the official Generator data formatter.

No model or kinematic replacement. Recorded reference commands are IL targets;
executed29-D actions and actual futures remain separate evidence.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def export_episode(trace_dir, prefix, minimum_control_frame=0, official=None):
    """Use the full official formatter on one complete actual Tracker rollout."""
    trace_dir, prefix = Path(trace_dir), Path(prefix)
    if any(prefix.with_name(prefix.name + suffix).exists() for suffix in ('_RAW.npz', '_IL.npz')):
        raise RuntimeError('Preserve existing official IL export')
    if official is None:
        source = ROOT / 'SUGAR/scripts/sugar_rl/process_tracker_rollout.py'
        spec = importlib.util.spec_from_file_location('official_tracker_formatter', source)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
    trace = dict(np.load(trace_dir/'TRACE.npz', allow_pickle=False))
    protocol = json.loads((trace_dir/'PROTOCOL.json').read_text())
    names = np.load(trace_dir/'STARTUP.npz')['body_names'].tolist()
    body = trace['robot_body_state_before_w'][:, 0]
    torso = body[:, names.index('torso_link')]
    obj = trace['object_state_before_w'][:, 0]
    cmd = trace['reference_command'][:, 0]
    obs = trace['teacher_observation'][:, 0]
    if protocol['controller'] != 'tracker' or trace['done'].any():
        raise RuntimeError('Exporter requires a complete actual Tracker episode without resets')
    obj_pos, obj_quat = official.subtract_frame_transforms_np(torso[:, :3], torso[:, 3:7], obj[:, :3], obj[:, 3:7])
    goal_pos, goal_quat = official.subtract_frame_transforms_np(torso[:, :3], torso[:, 3:7],
        trace['target_object_pos_w'][:, 0], trace['target_object_quat_w'][:, 0])
    actor_object = np.concatenate([obj_pos, official.matrix_from_quat_np(obj_quat)[:, :, :2].reshape(-1, 6)], axis=-1)
    # Independent cross-check against the exact official live actor input.
    command_error = float(np.max(np.abs(cmd-obs[:, :36])))
    object_error = float(np.max(np.abs(actor_object-obs[:, -9:])))
    if command_error > 1e-6 or object_error > 1e-5:
        raise RuntimeError(f'Live official input mismatch: command={command_error}, object={object_error}')
    count = len(obj)
    raw = dict(body_pos_w=body[..., :3], ref_joint_pos=cmd[:, :29],
               ref_root_lin_vel_b=cmd[:, 29:32], ref_root_ang_vel_b=cmd[:, 32:35],
               ref_contact_label=cmd[:, 35], obj_pos_b=obj_pos, obj_quat_b=obj_quat,
               joint_pos=trace['joint_pos_before'][:, 0], project_gravity=trace['project_gravity'][:, 0],
               target_obj_pos_b=goal_pos, target_obj_quat_b=goal_quat,
               rollout_start=np.array(0), rollout_end=np.array(count-1),
               original_max_timestep=np.array(protocol['source_lengths'][0]))
    raw_path = prefix.with_name(prefix.name + '_RAW.npz')
    np.savez_compressed(raw_path, **raw)
    episode, start_padded, end_padded = official.process_data_to_dict(raw_path)
    frame = np.arange(0, count, 5)
    if start_padded: frame = np.concatenate([frame[:1], frame])
    if end_padded: frame = np.concatenate([frame, np.repeat(frame[-1], 7)])
    # Never train a chunk crossing the unannounced reference switch.
    # All selected-demo chunks start after its fixed causal selection.
    starts = np.array([i for i in range(len(frame)-7)
                       if frame[i] >= minimum_control_frame
                       and np.all(np.diff(frame[i:i+8]) == 5)], dtype=np.int64)
    if not len(starts): raise RuntimeError('No reset-free post-selection full action chunks')
    np.savez_compressed(prefix.with_name(prefix.name + '_IL.npz'), **episode, control_frame=frame,
                        selected_demo=trace['reference_id'][frame], usable_chunk_start=starts,
                        executed_action_29d=trace['executed_action'][frame, 0],
                        actual_body_state_w=body[frame], actual_object_state_w=obj[frame])
    return dict(actual_control_frames=count, official_10hz_rows=len(frame),
                         usable_8step_chunks=len(starts), first_chunk_control_frame=int(frame[starts[0]]),
                         start_padded=start_padded, end_padded=end_padded,
                         recorded_command_vs_live_actor_max_error=command_error,
                         reconstructed_object_vs_live_actor_max_error=object_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    plan = json.loads((run/'PROTOCOL.json').read_text())
    pairing = json.loads((run/'READBACK.json').read_text())
    if not pairing['data_pair_feasibility_passed']:
        raise RuntimeError('Failed actual-pair feasibility: do not create admitted training data')
    source = ROOT/'SUGAR/scripts/sugar_rl/process_tracker_rollout.py'
    spec = importlib.util.spec_from_file_location('official_tracker_formatter', source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    out = run/'official_il_data'
    out.mkdir(exist_ok=False)
    summaries = {}
    for arm in ('original', 'alternate'):
        summaries[arm] = export_episode(run / arm, out / arm, plan['switch_control_frame'], official)
    result = dict(execution_completed=True, formatter=str(source.relative_to(ROOT)), arms=summaries,
                  scope='TRAIN-pair diagnostic data only, no learned Generator result. Official36-D reference-command labels are separate from actual29-D execution and actual future states. Paired refiner-selected96/90 does not constitute held-out generalization.',
                  original_files_modified=False, model_training_started=False)
    (out/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
