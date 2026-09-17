"""Check the actual causal API against saved batch-one endpoint predictions."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .data import OBSERVATION_KEYS
from .predict import ObjectStateEstimator
from .model import rotation_matrix


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('GPU inference requires retained compute step')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    estimator=ObjectStateEstimator(args.endpoint,args.checkpoint,mass_log_bias=args.mass_log_bias)
    with np.load(args.predictions) as src:
        saved={k:src[k] for k in src.files}
    episode=int(saved['episode'][0]) if args.episode is None else args.episode
    if episode not in saved['episode']: raise ValueError('Requested episode absent from saved predictions')
    with np.load(Path(args.data)/f'episode_{episode:04d}.npz') as src:
        observations={k:src[k] for k in OBSERVATION_KEYS}
    expected={int(f):p for f,p in zip(saved['frame'][saved['episode']==episode],
                                    saved['prediction'][saved['episode']==episode],strict=True)}
    differences=[]
    outputs=[]
    for frame in range(len(observations['timestamp_s'])):
        row={k:v[frame] for k,v in observations.items()}
        actual=estimator.update(row)
        if frame<estimator.history-1:
            assert actual is None
            continue
        if frame not in expected:
            continue
        pred=expected[frame]
        Rh=Rotation.from_quat(row['hand_pose_w'][0,3:]).as_matrix()
        expected_R=Rh@rotation_matrix(torch.from_numpy(pred[3:9])).numpy()
        expected_values=np.concatenate((pred[:3],Rh@pred[:3]+row['hand_pose_w'][0,:3],
                                        expected_R.reshape(-1),np.exp(pred[9:13])))
        actual_values=np.concatenate((actual['center_relative_left_hand_m'],actual['center_w_m'],
                                      Rotation.from_quat(actual['object_rotation_w_xyzw']).as_matrix().reshape(-1),
                                      actual['dimensions_m'],[actual['mass_kg']]))
        differences.append(float(np.max(np.abs(expected_values-actual_values))))
        outputs.append(actual_values)
    try:
        estimator.update(row)
    except ValueError:
        duplicate_rejected=True
    else:
        duplicate_rejected=False
    estimator.reset()
    assert estimator.update({k:v[0] for k,v in observations.items()}) is None
    result=dict(episode=episode,matched_saved_frames=len(differences),
                mass_log_bias=args.mass_log_bias,
                all_intermediate_frames_consumed=True,only_sensor_keys=OBSERVATION_KEYS,
                maximum_physical_output_abs_difference=max(differences),
                duplicate_timestamp_rejected=duplicate_rejected,reset_warmup_passed=True,
                tolerance=1e-5,new_optimizer_updates=0)
    result['passed']=max(differences)<=1e-5 and duplicate_rejected and len(differences)==len(expected)
    np.savez_compressed(Path(args.output).with_suffix('.npz'),physical_outputs=outputs,
                        differences=differences,frames=list(expected))
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not result['passed']:
        raise RuntimeError('Streaming API mismatch')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('endpoint','checkpoint','data','predictions','output'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--mass-log-bias',type=float,default=0.)
    ap.add_argument('--episode',type=int)
    main(ap.parse_args())
