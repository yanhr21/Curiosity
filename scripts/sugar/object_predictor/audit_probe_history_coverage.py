"""Saved physical contact coverage under the fixed causal sampler; no model calls."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import history_indices


def main(args):
    root=Path(args.data);records=[]
    for path in sorted(root.glob('episode_*.json')):
        meta=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as source:
            time=source['timestamp_s'];phases=source['validation_probe_phase']
            loads=source['normal_load_n'].reshape(-1,2,27).sum(2)
        windows=[]
        for frame in range(31,len(time),50):
            if not 140.<=time[frame]<=148.: continue
            selected=history_indices(frame,32,'episode_uniform_recent')
            counts=[];full=[]
            for phase in range(3):
                sampled=loads[selected][phases[selected]==phase]>.01
                prefix=loads[:frame+1][phases[:frame+1]==phase]>.01
                counts.append(dict(phase=phase,per_hand=sampled.sum(0).tolist(),simultaneous=int(sampled.all(1).sum())))
                full.append(dict(phase=phase,per_hand=prefix.sum(0).tolist(),simultaneous=int(prefix.all(1).sum())))
            windows.append(dict(frame=frame,timestamp_s=float(time[frame]),sampled=counts,full_prefix=full,
                                selected_frames=selected.tolist(),all_three_sampled_any_hand=all(sum(x['per_hand'])>0 for x in counts),
                                all_three_sampled_both_hands=all(min(x['per_hand'])>0 for x in counts),
                                all_three_prefix_both_hands=all(min(x['per_hand'])>0 for x in full)))
        assert len(windows)==8
        records.append(dict(episode=meta['episode'],split=meta['split'],physical_qualification_passed=meta['contact_qualification_passed'],
                            primary_windows=windows,
                            three_direction_any_hand_windows=sum(x['all_three_sampled_any_hand'] for x in windows),
                            three_direction_both_hands_windows=sum(x['all_three_sampled_both_hands'] for x in windows)))
    result=dict(episodes=records,complete=len(records)==36,new_optimizer_updates=0,
                selection='Fixed clock-only 32-frame history; validation phase used solely for saved coverage audit, never input or selection.',
                interpretation='Contact-direction coverage is necessary evidence, not proof of identifiable full shape. No window or failed case removed.')
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(complete=result['complete'],episodes=[{k:v for k,v in r.items() if k!='primary_windows'} for r in records])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);main(ap.parse_args())
