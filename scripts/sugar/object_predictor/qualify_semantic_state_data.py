"""CPU-only complete464 real encoding/conflict/observation provenance checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from . import semantic_state_data as data


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    p,source=data.source_contract(args.source)
    sourcefiles=[Path(data.__file__),Path(__file__),Path('scripts/sugar/object_predictor/observed_force_summary.py'),
        Path('scripts/sugar/object_predictor/overfit_data.py'),Path('scripts/sugar/object_predictor/surface_data.py'),
        Path('scripts/sugar/object_predictor/failure_aware_overfit.py'),Path('scripts/sugar/object_predictor/conditional_overfit_data.py'),
        Path('scripts/sugar/object_predictor/overfit_supervision_conflicts.py')]
    pins={str(f.resolve()):data.sha(f) for f in sourcefiles}
    save(root/'PROTOCOL.json',dict(study=data.STUDY,status='CPU_DATA_QUALIFICATION_ONLY',source=source,
        data_root=p['source_data'],frames=data.FIT_FRAMES,evaluation_frames=data.EVALUATION_FRAMES,
        summary_fields=data.SUMMARY_FIELDS,summary_shape=[464,32,11],height_scale_m=.2,
        height_provenance='Actual hand_pose_w[history_indices,0,2], not current height replicated.',
        prepared_sources=pins,model_forwards=0,optimizer_updates=0,physics_controls=0))
    print('SEMANTIC_REAL464_ENCODING_BEGIN',flush=True)
    dataset=data.SemanticStateDataset(p['source_data'],p['data'])
    print('SEMANTIC_REAL464_ENCODING_COMPLETE',len(dataset),flush=True)
    report=data.qualification(dataset,p['fit_limits'])
    save(root/'DATA_QUALIFICATION.json',report)
    if not report['passed']:raise ValueError('Actual encoded464 conflict/data qualification failed')
    summaries=[];fingerprints=[];target=[];masks=[];identities=[]
    original={}
    with np.load(args.source/'fit_02100.npz') as old:
        old={k:old[k] for k in old.files}
    oldlookup={(int(e),int(f)):i for i,(e,f) in enumerate(zip(old['episode'],old['frame']))}
    force_names=('normal_load_by_encoded_side_n','shear_on_hand_current_frame_n','normal_on_object_approx_current_frame_n')
    for start in range(0,len(dataset),4):
        rows=dataset.rows[start:start+4]
        observations=data.collate_semantic_observations(rows)
        summary=observations['observation_history'].numpy()
        for j,row in enumerate(rows):
            ft=np.concatenate([row['supervision']['physics'][k] for k in force_names])
            if not np.array_equal(summary[j,-1,:8],ft):raise ValueError('Observation-only current summary differs from original target')
            if not np.array_equal(summary[j,:,10],row['observed_left_hand_world_height_m'][:,0]/.2):
                raise ValueError('Actual historical height mapping changed')
            ident=(row['metadata']['episode'],row['metadata']['frame']);identities.append(ident)
            if ident in oldlookup:
                k=oldlookup[ident]
                exact=(np.array_equal(row['target'],old['target'][k]) and np.array_equal(ft,old['force_target_n'][k]) and
                    row['supervision']['mass_available']==old['mass_available'][k] and
                    row['supervision']['state_precision_eligible']==old['state_precision_eligible'][k] and
                    row['supervision']['state_contact_history_frames']==old['state_contact_history_frames'][k])
                original[str(ident)]=bool(exact)
                if not exact:raise ValueError('Original80 target/mask/observed force changed')
            h=hashlib.sha256()
            for key in ('coord','grid_coord','feat'):
                for v in row['inputs'][key]:h.update(np.ascontiguousarray(v).tobytes())
            h.update(row['observed_left_hand_world_height_m'].tobytes());fingerprints.append(h.hexdigest())
            target.append(row['target']);masks.append([row['supervision']['state_precision_eligible'],row['supervision']['mass_available']])
        summaries.append(summary)
        if (start+4)%64==0:print('SEMANTIC_SUMMARY_ROWS',start+4,flush=True)
    summary=np.concatenate(summaries)
    np.savez_compressed(root/'OBSERVATION_QUALIFICATION.npz',episode=np.array([e for e,f in identities]),
        frame=np.array([f for e,f in identities]),summary=summary,target=np.stack(target),masks=np.asarray(masks),input_sha256=np.asarray(fingerprints))
    if torch.cuda.is_initialized() or 'scripts.sugar.object_predictor.model' in sys.modules or 'utonia' in sys.modules:
        raise RuntimeError('CPU qualification unexpectedly loaded model/GPU')
    if any(data.sha(path)!=digest for path,digest in pins.items()):raise ValueError('Data sources changed during CPU qualification')
    l1=np.abs(summary).sum(axis=(1,2))
    final=dict(passed=True,rows=464,original80_exact=len(original)==80 and all(original.values()),
        denominators=data.task_denominators(dataset),data_qualification=report,
        summary_shape=list(summary.shape),summary_l1=dict(minimum=float(l1.min()),median=float(np.median(l1)),maximum=float(l1.max())),
        summary_per_column_absmax=np.abs(summary).max(axis=(0,1)).tolist(),
        summary_provenance=dataset.summary_input_provenance,
        height_is_actual_per_history_frame=True,
        historical_height_nonconstant_rows=int(np.count_nonzero(np.ptp(summary[:,:,10],axis=1)>0)),
        original80_checks=original,cuda_initialized=False,complete_model_constructed=False,model_forwards=0,optimizer_updates=0,
        source_bindings=source,qualification_npz_sha256=data.sha(root/'OBSERVATION_QUALIFICATION.npz'))
    save(root/'RESULT.json',final)
    print(json.dumps({k:v for k,v in final.items() if k not in ('data_qualification','summary_provenance','original80_checks','source_bindings')}),flush=True)


if __name__=='__main__':main()
