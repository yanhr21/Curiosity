"""Describe terminal contact loss in saved probes; no controller/physics rerun."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main(args):
    root = Path(args.data)
    records = []
    inputs = {}
    axes = np.array([[[1., 0., 0.], [-1., 0., 0.]],
                     [[0., 1., 0.], [0., -1., 0.]],
                     [[0., 0., -1.], [0., 0., -1.]]])
    for episode in range(3000, args.last_episode + 1):
        case = root / 'cases' / f'episode_{episode}'
        protocol = json.loads((case / 'PROTOCOL.json').read_text())
        assert protocol['stable_servo'] and protocol['gentle_force_servo']
        replay = json.loads((case / 'CONTROLLER_REPLAY_AUDIT.json').read_text())
        assert replay['passed']
        path = case / 'episode_0000.npz'
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as data:
            time = data['timestamp_s']
            load = data['normal_load_n'].reshape(-1, 2, 27).sum(2)
            touched = data['validation_probe_touched']
            pose = data['hand_pose_w'][:, :, :3].astype(np.float64)
        assert len(time) == 7500
        for phase in range(3):
            frames = np.flatnonzero((time >= phase * 50 + 2) &
                                    (time < phase * 50 + 48))
            end = int(frames[-1])
            for hand in range(2):
                contact = load[frames, hand] > .01
                last = np.flatnonzero(contact)
                first_zero = int(frames[last[-1] + 1]) if len(last) and last[-1] < len(frames)-1 else (int(frames[0]) if not len(last) else end + 1)
                count = end - first_zero + 1
                row = dict(episode=episode, phase=phase, hand=hand,
                           terminal_no_contact_frames=count,
                           terminal_no_contact_seconds=count * .02,
                           touched_latched_at_approach_end=bool(touched[end, hand]),
                           final_load_n=float(load[end, hand]))
                if count > 1:
                    start = first_zero
                    elapsed = float(time[end] - time[start])
                    displacement = float((pose[end, hand] - pose[start, hand]) @ axes[phase, hand])
                    # Commands start+1..end consume observations start..end-1.
                    prior = load[start:end, hand].astype(np.float64)
                    flags = touched[start+1:end+1, hand]
                    proposed = np.where(flags, np.clip((1. - prior) * .00005, -.02, .002), .006)
                    proposed_displacement = float(np.sum(proposed * .02))
                    row.update(first_terminal_zero_time_s=float(time[start]),
                               measured_inward_displacement_m=displacement,
                               measured_mean_inward_speed_m_s=displacement / elapsed,
                               source_rule_unclipped_displacement_m=proposed_displacement,
                               displacement_residual_m=displacement-proposed_displacement,
                               flags_latched_through_zero_run=bool(flags.all()))
                records.append(row)
    source = Path(__file__).with_name('probe_scene.py')
    result = dict(episodes=list(range(3000, args.last_episode+1)), complete=args.last_episode==3035,
                  records=records, input_sha256=inputs,
                  controller_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  contact_threshold_n=.01, approach_interval_s=[2.,48.],
                  nominal_unlatched_speed_m_s=.006, nominal_latched_zero_load_speed_m_s=.00005,
                  interpretation='Saved motion and source-rule description only. No proof that a faster recovery would regain contact or improve prediction. No cases or windows removed.',
                  new_optimizer_updates=0, physics_replays=0, model_calls=0)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([r for r in records if r['terminal_no_contact_seconds'] >= 1.]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--last-episode', type=int, required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
