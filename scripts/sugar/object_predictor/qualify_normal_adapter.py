"""Real-data integration checks for optional CAD normals; no neural model created."""
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np
from .data import ContactDataset,OBSERVATION_KEYS,MODES,encode_observations,collate
from .predict import ObjectStateEstimator
from .hand_surface_normals import geometry_signature,verify_geometry_signature


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);view=out/'diagnostic_view';view.mkdir()
    original=json.loads((Path(args.qualification)/'RESULT.json').read_text())
    for i,record in enumerate(original['episodes']):
        source=Path(record['source']).resolve()
        with np.load(source) as a:frames=len(a['timestamp_s'])
        (view/f'episode_{i:04d}.npz').symlink_to(source)
        (view/f'episode_{i:04d}.json').write_text(json.dumps(dict(episode=i,split='train',sampling_stride=5 if frames<1000 else 50,
                                                               diagnostic_only=True,training_admitted=False,source=str(source))))
    legacy=ContactDataset(view,'geometry','train',32,5,'episode_uniform_recent',150.)
    datasets={m:ContactDataset(view,m,'train',32,5,'episode_uniform_recent',150.,'hand_surface') for m in MODES}
    checks={};maximum=0.;examined=0
    checks['all_indices_unchanged']=all(d.index==legacy.index for d in datasets.values())
    for episode,(_,arrays) in enumerate(legacy.episodes):
        indices=[i for i,(e,f) in enumerate(legacy.index) if e==episode]
        for index in [indices[0],indices[len(indices)//2],indices[-1]]:
            frame=legacy.index[index][1]
            runtime=ObjectStateEstimator.__new__(ObjectStateEstimator)
            runtime.history=32;runtime.history_policy='episode_uniform_recent'
            runtime.frames=deque({k:arrays[k][t].copy() for k in OBSERVATION_KEYS} for t in range(frame+1))
            obs=runtime._history_observations();rows=[]
            for mode,dataset in datasets.items():
                row=dataset[index];rows.append(row)
                streamed=encode_observations(obs,mode,time_scale_s=150.,normal_policy='hand_surface')
                difference=max(float(np.abs(row[k]-streamed[k]).max()) for k in ('coord','feat'))
                maximum=max(maximum,difference)
                prefix=f'{episode}_{frame}_{mode}'
                checks[prefix+'_cached_runtime_match']=difference<=1e-5
                old=encode_observations(obs,mode,time_scale_s=150.)
                explicit=encode_observations(obs,mode,time_scale_s=150.,normal_policy='stored')
                checks[prefix+'_stored_default_exact']=all(np.array_equal(old[k],explicit[k]) for k in old)
                checks[prefix+'_only_normal_columns_change']=np.array_equal(old['coord'],row['coord']) and np.array_equal(np.delete(old['feat'],[6,7,8],axis=-1),np.delete(row['feat'],[6,7,8],axis=-1))
                checks[prefix+'_finite_complete_32']=np.isfinite(row['feat']).all().item() and row['feat'].shape==(32,54,20)
            checks[f'{episode}_{frame}_all_labels_identical']=all(np.array_equal(rows[0]['target'],r['target']) for r in rows[1:])
            checks[f'{episode}_{frame}_contact_force_same_geometry_normals']=np.array_equal(rows[1]['feat'][...,:10],rows[2]['feat'][...,:10])
            # Geometry-only features must not depend on contact or force values,
            # including when surface normals are computed online.
            changed={k:v.copy() for k,v in obs.items()}
            changed['normal_load_n'][:]=0;changed['contact_position_w'][:]=0
            changed['shear_force_w'][:]=0;changed['contact_area_m2'][:]=0
            geometry=encode_observations(changed,'geometry',time_scale_s=150.,normal_policy='hand_surface')
            checks[f'{episode}_{frame}_geometry_no_tactile_leak']=np.array_equal(geometry['feat'],rows[0]['feat'])
            packed=collate(rows)
            checks[f'{episode}_{frame}_full_sparse_batch_finite']=bool(packed['feat'].isfinite().all()) and len(packed['offset'])==96
            examined+=1
    verify_geometry_signature(geometry_signature());bad=dict(geometry_signature());bad['left']='mismatch'
    try:verify_geometry_signature(bad)
    except RuntimeError:checks['wrong_geometry_signature_rejected']=True
    else:checks['wrong_geometry_signature_rejected']=False
    report=dict(checks=checks,passed=all(checks.values()),windows=examined,maximum_cached_runtime_difference=maximum,
                geometry_signature=geometry_signature(),new_optimizer_updates=0,gpu_calls=0,
                neural_model_created=False,training_admitted=False,
                scope='Actual encoder/collation and runtime history-selection integration on five saved TRAIN traces; full-model GPU qualification remains required.')
    (out/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--qualification',required=True);ap.add_argument('--output',required=True);main(ap.parse_args())
