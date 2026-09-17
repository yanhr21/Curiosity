"""CPU-only qualification of deterministic summaries on the fixed real80 inputs.

No complete model/checkpoint is loaded. The old dataset may read GT to preserve
its independent labels; summarize_* receives ONLY the four collated observation
tensors. Exact target reproduction qualifies this adapter, never learning,
object-force truth, mass/shape prediction, or physical acquisition success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from .observed_force_summary import summarize_observed_forces, summarize_support_projections, FORCE_FIELDS, SUPPORT_FIELDS
from .overfit_data import observed_physics_targets, FIXED_EPISODES, FIXED_FRAMES
from .failure_aware_overfit import FailureAwareOverfitDataset
from .surface_data import collate_surface


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(source, output):
    if torch.cuda.is_initialized(): raise RuntimeError('CPU check must not initialize CUDA')
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    p = json.loads((source/'PROTOCOL.json').read_text())
    r = json.loads((source/'RESULT.json').read_text())
    if (p['study'] != 'failure_aware_fullbatch_refinement_v1' or r.get('execution_complete') is not True
            or r.get('total_optimizer_updates') != 2100):
        raise ValueError('Require the completed declared source2100 data contract')
    protocol = dict(scope='deterministic_observed_force_summary_cpu_qualification_only',
        source=str(source), source_optimizer_updates=2100,
        history=32, frame_layout='Batch-major H32; frame k=b*32+t uses [offset[k-1],offset[k]), start0 for k0; current t31.',
        input_keys=['coord','grid_coord','feat','offset'], output_force_shape=[80,32,8],
        output_support_shape=[80,32,2], force_fields=FORCE_FIELDS, support_fields=SUPPORT_FIELDS,
        frame='Current left-hand reference for every frame of each window', units='N',
        arithmetic='CPU NumPy float64 inverse transforms and reductions, then float32, matching original oracle.',
        no_cross_time_force_sum=True, learned_parameters=0, model_forwards=0, physics_controls=0, optimizer_updates=0,
        limitations='All8 are exact functions of encoded observations. Normal3 and combined support are CAD/fitted approximations; encoded side loads can mix hands by area. This does not certify true object wrench, mass, arbitrary shape, learning or acquisition.',
        source_bindings={str(source/name):sha(source/name) for name in ('PROTOCOL.json','RESULT.json','fit_02100.npz')},
        adapter_bindings={str(path.resolve()):sha(path) for path in (
            Path(__file__),Path(__file__).with_name('observed_force_summary.py'),
            Path(__file__).with_name('overfit_data.py'),Path(__file__).with_name('surface_data.py'),
            Path(__file__).with_name('overfit_model.py'),Path('tests/test_observed_force_summary.py'))})
    write(output/'PROTOCOL.json',protocol)
    dataset = FailureAwareOverfitDataset(p['source_data'], p['data'])
    expected = [(e,f) for e in FIXED_EPISODES for f in FIXED_FRAMES]
    identities = [(row['metadata']['episode'],row['metadata']['frame']) for row in dataset.rows]
    if identities != expected or len(dataset) != 80: raise ValueError('Fixed all80 identities/order required')
    print('OBSERVED_FORCE_CPU_DATASET_READY',len(dataset),flush=True)
    with np.load(source/'fit_02100.npz',allow_pickle=False) as saved:
        old_force = saved['force_target_n'].copy()
        if not np.array_equal(saved['episode'],np.array([e for e,f in identities])) or not np.array_equal(saved['frame'],np.array([f for e,f in identities])):
            raise ValueError('Archived source target identities differ')
    all_force, all_support, current_targets, records = [], [], [], []
    for start in range(0,80,8):
        rows = dataset.rows[start:start+8]
        inputs = collate_surface([row['inputs'] for row in rows])
        force = summarize_observed_forces(inputs).numpy()
        support = summarize_support_projections(inputs).numpy()
        for b,row in enumerate(rows):
            one = collate_surface([row['inputs']])
            if not np.array_equal(force[b],summarize_observed_forces(one)[0].numpy()) or not np.array_equal(support[b],summarize_support_projections(one)[0].numpy()):
                raise ValueError('Batch grouping changed deterministic summaries')
            for t,frame in enumerate(row['inputs']['feat']):
                oracle=observed_physics_targets(frame)
                expected_force=np.concatenate([oracle[k] for k in ('normal_load_by_encoded_side_n','shear_on_hand_current_frame_n','normal_on_object_approx_current_frame_n')])
                expected_support=np.concatenate([oracle[k] for k in ('shear_support_up_n','combined_support_up_approx_n')])
                if not np.array_equal(force[b,t],expected_force) or not np.array_equal(support[b,t],expected_support):
                    raise ValueError(f'Original per-frame oracle differs: row{start+b}/time{t}')
            target=np.concatenate([row['supervision']['physics'][k] for k in ('normal_load_by_encoded_side_n','shear_on_hand_current_frame_n','normal_on_object_approx_current_frame_n')])
            if not np.array_equal(force[b,-1],target) or not np.array_equal(force[b,-1],old_force[start+b]):
                raise ValueError('Actual current-frame or saved original80 label differs')
            k=b*32+31
            records.append(dict(episode=identities[start+b][0],frame=identities[start+b][1],
                row_index=start+b,batch_index=b,current_collated_frame=k,
                start=int(inputs['offset'][k-1]),stop=int(inputs['offset'][k]),
                all32_oracle_exact=True,current_original_label_exact=True,current_source2100_label_exact=True))
            current_targets.append(target)
        all_force.append(force);all_support.append(support)
        print('OBSERVED_FORCE_CPU_VERIFIED',start+len(rows),flush=True)
    force=np.concatenate(all_force);support=np.concatenate(all_support)
    if torch.cuda.is_initialized() or 'scripts.sugar.object_predictor.model' in sys.modules or 'utonia' in sys.modules:
        raise RuntimeError('Unexpected model/GPU import during CPU adapter qualification')
    np.savez_compressed(output/'SUMMARIES.npz',force_summary_n=force,support_summary_n=support,
        current_force_target_n=np.stack(current_targets),episode=np.array([e for e,f in identities]),frame=np.array([f for e,f in identities]))
    for path,digest in protocol['adapter_bindings'].items():
        if sha(path)!=digest: raise ValueError('Source changed during qualification: '+path)
    result=dict(passed=True,scope=protocol['scope'],rows=80,history_frames=2560,
        force_oracle_scalars_exact=20480,support_oracle_scalars_exact=5120,current_force_labels_exact=640,
        original_source2100_labels_exact=True,batch1_vs_batch8_exact=True,records=records,
        state_rows=int(sum(row['supervision']['state_precision_eligible'] for row in dataset.rows)),
        mass_rows=int(sum(row['supervision']['mass_available'] for row in dataset.rows)),
        physical_success_unchanged=bool(dataset.collection_result['qualification_passed']),
        learned_parameters=0,model_forwards=0,physics_controls=0,optimizer_updates=0,cuda_initialized=False,
        complete_official_model_constructed=False,learning_success_claim=False,true_object_wrench_claim=False,
        summaries_sha256=sha(output/'SUMMARIES.npz'),protocol_sha256=sha(output/'PROTOCOL.json'))
    write(output/'RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    try: run(args.source,args.output)
    except Exception as error:
        if args.output.is_dir() and not isinstance(error,FileExistsError):
            write(args.output/'FAILURE.json',dict(passed=False,error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
