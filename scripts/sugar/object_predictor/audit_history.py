"""CPU qualification of causal long-history data/API selection on real physics.

This exercises the actual runtime's observation-selection method without creating
a network or emitting any model prediction. Neural qualification is separate.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np
from .data import OBSERVATION_KEYS,ContactDataset,encode_observations,history_indices,target_at
from .predict import ObjectStateEstimator


def main(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    source=Path(args.episode).resolve()
    (root/'episode_0000.npz').symlink_to(source)
    (root/'episode_0000.json').write_text(json.dumps(dict(episode=0,split='test',diagnostic_only=True,training_admitted=False,source=str(source)),indent=2))
    with np.load(source) as a: arrays={k:a[k] for k in a.files}
    checks={};coverage={}
    def check(name,value): checks[name]=bool(value)
    for history,policy,scale in [(8,'contiguous',1.),(32,'episode_uniform_recent',108.)]:
        for frame in range(history-1,len(arrays['timestamp_s'])):
            selected=history_indices(frame,history,policy)
            if not (len(selected)==history and np.all(np.diff(selected)>0) and selected.min()>=0 and selected[-2:].tolist()==[frame-1,frame]):
                raise AssertionError((history,policy,frame,selected))
        check(policy+'_every_frame_unique_causal_includes_latest_two',True)
        frames=[history-1,100,1000,2000,3000,4000,5100,len(arrays['timestamp_s'])-1]
        datasets={mode:ContactDataset(root,mode,'test',history,1,policy,scale)
                  for mode in ('geometry','geometry_contact','geometry_contact_force')}
        for frame in frames:
            # Use the actual API selection method, with its actual deque layout.
            # __init__ is deliberately skipped: this is data-only qualification.
            runtime=ObjectStateEstimator.__new__(ObjectStateEstimator)
            runtime.history=history;runtime.history_policy=policy
            runtime.frames=deque(maxlen=history if policy=='contiguous' else None)
            for t in range(frame+1): runtime.frames.append({k:arrays[k][t].copy() for k in OBSERVATION_KEYS})
            obs=runtime._history_observations()
            selected=history_indices(frame,history,policy)
            for key in OBSERVATION_KEYS:
                check(f'{policy}_{frame}_{key}_runtime_exact',np.array_equal(obs[key],arrays[key][selected]))
            for mode,dataset in datasets.items():
                actual=dataset[frame-(history-1)]
                streamed=encode_observations(obs,mode,time_scale_s=scale)
                for key in ('coord','feat'):
                    check(f'{policy}_{frame}_{mode}_{key}_runtime_dataset_exact',np.array_equal(actual[key],streamed[key]))
                check(f'{policy}_{frame}_{mode}_target_current_only',np.array_equal(actual['target'],target_at(arrays,frame)))
            check(f'{policy}_{frame}_time_in_range',obs['timestamp_s'][-1]==arrays['timestamp_s'][frame] and np.ptp(obs['timestamp_s'])/scale<=max(1.,108./scale))
        frame=5100;selected=history_indices(frame,history,policy)
        contact=arrays['normal_load_n'][selected].sum(1)>.01
        coverage[policy]=dict(selected_frames=selected.tolist(),selected_timestamps_s=arrays['timestamp_s'][selected].tolist(),
                              contact_phases=np.unique(arrays['validation_probe_phase'][selected][contact]).tolist(),
                              age_span_s=float(np.ptp(arrays['timestamp_s'][selected])))
    check('long_history_preserves_three_real_contact_directions',coverage['episode_uniform_recent']['contact_phases']==[0,1,2])
    result=dict(checks=checks,checks_passed=sum(checks.values()),checks_total=len(checks),passed=all(checks.values()),
                real_contact_coverage=coverage,source=str(source),new_model_parameters_created=0,
                neural_inference_performed=False,new_optimizer_updates=0,
                gpu_full_model_and_original_endpoint_replay_still_required=True)
    (root/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
    if not result['passed']: raise RuntimeError('Causal history qualification failed')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--episode',required=True);ap.add_argument('--output',required=True);main(ap.parse_args())
