"""Direct actual-trace and input-provenance readback of the frozen position test."""
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
from scripts.sugar.demo_following.demo_future.run_heldout_reference_validation import write
from scripts.sugar.smp.sugar_g1_box_schema import SOURCE_BODY_NAMES

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'experiments/demo_following/demo_future_smp_v1/original_position_feedback_floor'


def main():
    result = json.loads((RUN / 'RESULT.json').read_text())
    plan = json.loads((RUN / 'PROTOCOL.json').read_text())
    children = json.loads((RUN / 'CHILDREN.json').read_text())
    if not result['execution_completed'] or len(children) != 16 or any(row['returncode'] for row in children):
        raise RuntimeError('Complete all16 original position-source attempts first')
    source = ROOT / 'SUGAR/scripts/sugar_rl/process_tracker_rollout.py'
    spec = importlib.util.spec_from_file_location('official_formatter', source)
    formatter = importlib.util.module_from_spec(spec); spec.loader.exec_module(formatter)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(8, 3, figsize=(16, 23), squeeze=False)
    colors = {'measured_reference': '#0173b2', 'original_numeric_demo': '#029e73'}
    rows, checks = {}, {}
    total_steps = 0
    for row_index, source in enumerate(plan['sources']):
        case = RUN / f'source_{source:03d}'
        banks = {}
        for kind, folder in [('measured', case / 'motions/data_000'),
                             ('original', ROOT / f'SUGAR/data/CarryBox/data_{source:03d}')]:
            with np.load(folder / 'robot_50hz.npz') as robot, (folder / 'obj_motion_global_50hz.pkl').open('rb') as stream:
                banks[kind] = dict(anchor=robot['body_pos_w'][:, SOURCE_BODY_NAMES.index('torso_link')].copy(),
                                   box=pickle.load(stream)['obj_trans'])
        arms = {}
        for arm in plan['arms']:
            folder = case / arm
            with np.load(folder / 'STARTUP.npz') as startup:
                torso = startup['body_names'].tolist().index('torso_link')
            with np.load(folder / 'TRACE.npz') as trace:
                frame = trace['reference_frame']; valid = ~trace['done'].reshape(-1); count = len(frame)
                total_steps += count
                body = trace['robot_body_state_before_w'][:, 0, torso]
                box = trace['object_state_before_w'][:, 0]
                origin = trace['reference_object_pos_w'][:, 0] - banks['measured']['box'][frame]
                selected = banks['original' if arm == 'original_numeric_demo' else 'measured']
                times = frame[:, None] + np.arange(8)[None]
                if times.max() >= min(len(selected['anchor']), len(selected['box'])):
                    raise RuntimeError('Actual position input exceeded original source data')
                def local(quat, offset):
                    return formatter.quat_apply_np(formatter.quat_inv_np(np.repeat(quat[:, None], 8, axis=1)), offset)
                anchor_packet = local(body[:, 3:7], selected['anchor'][times] + origin[:, None] - body[:, None, :3])
                box_packet = local(box[:, 3:7], selected['box'][times] + origin[:, None] - box[:, None, :3])
                expected = np.concatenate([anchor_packet.reshape(count, 24), box_packet.reshape(count, 24)], axis=-1)
                error = float(np.max(np.abs(expected - trace['actor_reference_position_feedback'][:, 0])))
                item_checks = dict(position_packet_exact=error < 1e-5,
                    source_and_clock_exact=bool(np.all(trace['reference_id'] == source) and np.array_equal(frame, np.arange(count))),
                    actor_packet_matches_raw=np.array_equal(trace['raw_reference_position_feedback'], trace['actor_reference_position_feedback']),
                    executed_actions_exact=np.array_equal(trace['requested_action'][valid], trace['executed_action'][valid]),
                    fixed_world_origin=bool(np.max(np.abs(origin - origin[:1])) < 1e-5),
                    finite_before_states=bool(np.isfinite(body).all() and np.isfinite(box).all()))
                stats = dict(actual_steps=count, physical_passed=result['per_source'][str(source)]['physical_passed'][arm],
                             position_packet_max_error=error, checks=item_checks,
                             root_height_min=float(trace['robot_root_state_before_w'][valid, 0, 2].min()))
                time = frame[valid] * .02
                axes[row_index, 0].plot(time, box[valid, 2] - box[0, 2], color=colors[arm], label=arm)
                for kind, style in [('measured', '-'), ('original', ':')]:
                    anchor_error = np.linalg.norm(body[:, :3] - (banks[kind]['anchor'][frame] + origin), axis=-1)
                    box_error = np.linalg.norm(box[:, :3] - (banks[kind]['box'][frame] + origin), axis=-1)
                    stats[kind + '_anchor_error_mean_m'] = float(anchor_error[valid].mean())
                    stats[kind + '_box_error_mean_m'] = float(box_error[valid].mean())
                    axes[row_index, 1].plot(time, anchor_error[valid], color=colors[arm], linestyle=style,
                                             label=arm + ' vs ' + kind)
                    axes[row_index, 2].plot(time, box_error[valid], color=colors[arm], linestyle=style)
                if not valid.all():
                    stats['terminal_terms'] = json.loads((folder / 'TERMINAL_TERMS.json').read_text())
                    with np.load(folder / 'TERMINAL_STATE.npz') as terminal:
                        anchor = terminal['robot_body_state_w'][0, torso, :3]
                        stats['terminal_before_reset'] = dict(reference_frame=int(frame[-1]),
                            root_height_m=float(terminal['robot_root_state_w'][0, 2]),
                            box_height_m=float(terminal['object_state_w'][0, 2]),
                            anchor_error_to_measured_last_frame_m=float(np.linalg.norm(anchor - banks['measured']['anchor'][frame[-1]] - origin[-1])),
                            anchor_error_to_original_last_frame_m=float(np.linalg.norm(anchor - banks['original']['anchor'][frame[-1]] - origin[-1])))
                arms[arm] = stats
                checks[f'{source}_{arm}'] = all(item_checks.values())
        rows[str(source)] = arms
        for column, title in enumerate(('Box lift (m)', 'Torso position error (m)', 'Box position error (m)')):
            ax = axes[row_index, column]; ax.grid(alpha=.2)
            ax.set_title(f'Source{source} | {title}', color='black' if all(v['physical_passed'] for v in arms.values()) else '#cc3311')
            if row_index == 7: ax.set_xlabel('Actual control time (s)')
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=7)
    fig.suptitle('Frozen original-position comparison: full model607; solid vs measured, dotted vs original reference')
    fig.tight_layout(rect=(0, 0, 1, .985)); fig.savefig(RUN / 'PHYSICAL_READBACK.png', dpi=130); plt.close(fig)
    audit = dict(execution_completed=True, data_integrity_passed=all(checks.values()), checks=checks,
        actual_control_steps=total_steps, optimizer_updates=0, per_source=rows, physical_plot_inspected=False,
        scope='All16 actual frozen rollouts. Reconstructed complete48-D inputs use the declared original or measured source at the native clock. Position errors to both sources are diagnostic; original termination/admission criteria are retained. No generated commands, camera success or SMP benefit.')
    write(RUN / 'PHYSICAL_READBACK.json', audit)
    if not audit['data_integrity_passed']:
        raise RuntimeError('Actual position-source input or execution readback failed')
    print(json.dumps(dict(data_integrity_passed=True, actual_control_steps=total_steps,
                         max_position_packet_error=max(r['position_packet_max_error'] for a in rows.values() for r in a.values()))), flush=True)


if __name__ == '__main__':
    main()
