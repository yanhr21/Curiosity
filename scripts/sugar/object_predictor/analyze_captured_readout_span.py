"""Saved-only CPU span diagnostic of r1 features; no model forward or updates.

The historical strict replay remains FAIL. Conclusions apply only to this
captured matrix, not unsaved historical features or the original model gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import audit_overfit_readout_features as algebra

STUDY = 'failure_aware_captured_span_v1'


def protocol(capture_root):
    capture_root = Path(capture_root).resolve()
    old = json.loads((capture_root/'PROTOCOL.json').read_text())
    result = json.loads((capture_root/'RESULT.json').read_text())
    replay = json.loads((capture_root/'REPLAY.json').read_text())
    source = Path(old['source'])
    if (result.get('model_forwards') != 80 or result.get('optimizer_updates') != 0
            or result.get('physics_controls') != 0 or result.get('strict_replay_passed') is not False
            or result.get('linear_diagnostic_executed') is not False
            or replay.get('passed') is not False):
        raise ValueError('Require the preserved completed eighty-forward strict-replay failure')
    manifest = json.loads((source/'ARTIFACTS.json').read_text())
    paths = {f'capture/{name}':capture_root/name for name in ('FEATURES.npz','REPLAY.json','RESULT.json','PROTOCOL.json')}
    paths.update({f'training/{name}':source/name for name in ('model.pt','fit_02000.npz','ARTIFACTS.json','PROTOCOL.json','RESULT.json')})
    paths.update(adapter=Path(__file__).resolve(), svd_helper=Path(algebra.__file__).resolve())
    bindings = {key:dict(path=str(path),sha256=algebra.digest(path)) for key,path in paths.items()}
    if (bindings['training/model.pt']['sha256'] != manifest['artifacts']['model.pt']['sha256']
            or bindings['training/fit_02000.npz']['sha256'] != manifest['artifacts']['fit_02000.npz']['sha256']
            or bindings['training/ARTIFACTS.json']['sha256'] != old['model_artifact_manifest_sha256']
            or bindings['training/RESULT.json']['sha256'] != old['source_result_sha256']
            or bindings['svd_helper']['sha256'] != old['adapter_sha256']):
        raise ValueError('Actual source/checkpoint/helper differs from preserved capture provenance')
    return dict(study=STUDY,capture_root=str(capture_root),training_source=str(source),bindings=bindings,
        fixed_episodes=old['fixed_episodes'],fixed_frames=old['fixed_frames'],fixed_rows=80,
        feature_dimension=old['feature_dimension'],feature_layout=old['feature_layout'],
        original_strict_replay_passed=False,original_model_acceptance_passed=False,
        model_forwards=0,optimizer_updates=0,physics_controls=0,
        cpu_original_readout='Apply the saved original head and auxiliary matrices to these captured features using NumPy float32 and float64 arithmetic; report differences against captured and historical outputs, with no bitexact or TF32 equivalence claim.',
        masks=old['masks'],solver=old['solver'],
        interpretation='Only this r1 captured matrix: in-sample affine span/conditioning, not a replacement predictor or any acceptance pass. More columns than rows can permit memorization. Similar readout outputs do not imply similar historical features because of the large readout nullspace; historical features were not saved. Microscopic replay differences do not explain the original centimeter/Newton scientific errors.',
        historical_replay_tolerance_changed=False,fitted_weights_saved=False,fitted_weights_deployed=False,
        no_holdout=True,generalization_or_tactile_benefit_claimed=False)


def validate_protocol(capture_root, output):
    declared=json.loads((Path(output)/'PROTOCOL.json').read_text())
    if declared != protocol(capture_root):
        raise ValueError('Saved-only diagnostic bindings or protocol changed')
    return declared


def load_arrays(path):
    with np.load(path,allow_pickle=False) as value:
        return {key:value[key] for key in value.files}


def validate_capture(capture, original, declared, saved_replay):
    clocks=[(e,f) for e in declared['fixed_episodes'] for f in declared['fixed_frames']]
    if len(clocks)!=80 or list(zip(capture['episode'].tolist(),capture['frame'].tolist()))!=clocks:
        raise ValueError('Every original eighty row identity must remain in order')
    fields=('episode','frame','target','mass_available','state_precision_eligible',
            'state_contact_history_frames','force_target_n')
    if any(not np.array_equal(capture[key],original[key]) for key in fields):
        raise ValueError('Captured rows/targets/masks differ from original saved evaluation')
    for key,shape in dict(features=(80,declared['feature_dimension']),target=(80,13),
            prediction=(80,13),force_target_n=(80,8),force_prediction_n=(80,8),
            availability_probability=(80,),contact_probability=(80,)).items():
        if capture[key].shape!=shape or not np.isfinite(capture[key]).all():
            raise ValueError('Malformed/nonfinite captured array: '+key)
    if capture['features'].dtype!=np.float32:
        raise ValueError('Require original float32 captured features')
    for key in ('mass_available','state_precision_eligible'):
        if capture[key].shape!=(80,) or not np.isin(capture[key],(0,1)).all():
            raise ValueError('Malformed captured task mask')
    count=capture['state_contact_history_frames']
    if (count.shape!=(80,) or not np.isin(count,np.arange(33)).all()
            or not np.array_equal(count>0,capture['state_precision_eligible']>0)):
        raise ValueError('State candidate mask disagrees with saved causal contact count')
    replay=algebra.replay_comparison(capture,original)
    if replay!=saved_replay or replay['passed'] is not False:
        raise ValueError('The original strict-replay failure must remain unchanged')
    return dict(rows=80,state_candidate_rows=int(capture['state_precision_eligible'].sum()),
        available_mass_rows=int(capture['mass_available'].sum()),force_rows=80,
        feature_dimension=declared['feature_dimension'],original_strict_replay_passed=False)


def cpu_affine_outputs(features, state_weight, state_bias, auxiliary_weight, auxiliary_bias, dtype):
    """Saved matrix arithmetic only; no model/module is constructed or called."""
    from scipy.special import expit
    x=np.asarray(features,dtype=dtype)
    w,b,a,c=(np.asarray(value,dtype=dtype) for value in
              (state_weight,state_bias,auxiliary_weight,auxiliary_bias))
    if (x.ndim!=2 or w.shape!=(13,x.shape[1]) or b.shape!=(13,)
            or a.shape!=(10,x.shape[1]) or c.shape!=(10,)
            or any(not np.isfinite(v).all() for v in (x,w,b,a,c))):
        raise ValueError('Original saved readout matrices do not match captured features')
    state=x@w.T+b
    auxiliary=x@a.T+c
    return dict(prediction=state,force_prediction_n=auxiliary[:,:8],
        availability_probability=expit(auxiliary[:,8]),contact_probability=expit(auxiliary[:,9]))


def descriptive_differences(actual, reference):
    result={}
    for key in ('prediction','force_prediction_n','availability_probability','contact_probability'):
        delta=np.asarray(actual[key],np.float64)-np.asarray(reference[key],np.float64)
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite saved-matrix readout difference')
        result[key]=dict(max_abs=float(abs(delta).max()),rmse=float(np.sqrt(np.mean(delta**2))))
    result['center_delta_max_cm']=float(np.linalg.norm(actual['prediction'][:,:3]-reference['prediction'][:,:3],axis=1).max()*100)
    result['availability_decision_changes']=int(np.count_nonzero((actual['availability_probability']>=.5)!=(reference['availability_probability']>=.5)))
    return result


def run(capture_root, output, *, check=False):
    import torch
    output=Path(output)
    declared=validate_protocol(capture_root,output)
    if any((output/name).exists() for name in ('RESULT.json','FAILURE.json','LINEAR_SPAN_DIAGNOSTIC.json')):
        raise FileExistsError('No overwrite or automatic retry of a completed/failed saved-only run')
    source=Path(declared['training_source'])
    capture=load_arrays(Path(capture_root)/'FEATURES.npz')
    original=load_arrays(source/'fit_02000.npz')
    replay=json.loads((Path(capture_root)/'REPLAY.json').read_text())
    checked=validate_capture(capture,original,declared,replay)
    endpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=False,mmap=True)
    state=endpoint['model']
    layout=algebra.feature_layout(state,declared['feature_layout']['history'],declared['feature_layout']['enc_channels'])
    if layout!=declared['feature_layout']:
        raise ValueError('Original checkpoint readouts disagree with actual capture layout')
    if torch.cuda.is_initialized():
        raise RuntimeError('Saved-only CPU analysis must not initialize CUDA')
    if check:
        checked.update(cuda_initialized=False,model_forwards=0,svd_executed=False,checkpoint_layout_checked=True)
        algebra.save_json(output/'CPU_INPUT_CHECK.json',checked)
        print(json.dumps(checked),flush=True)
        return 0
    matrices=[state[key].detach().numpy() for key in ('predictor.head.weight','predictor.head.bias',
                                                    'auxiliary.weight','auxiliary.bias')]
    readback={}
    for name,dtype in (('float32',np.float32),('float64',np.float64)):
        actual=cpu_affine_outputs(capture['features'],*matrices,dtype)
        readback[name]=dict(versus_captured_outputs=descriptive_differences(actual,capture),
                            versus_historical_outputs=descriptive_differences(actual,original))
    algebra.save_json(output/'ORIGINAL_READOUT_CPU.json',dict(arithmetic=readback,
        description=declared['cpu_original_readout'],original_strict_replay_passed=False,
        differences_are_not_acceptance_tests=True))
    reports=algebra.analyze_capture(capture,output,declared['feature_dimension'])
    validate_protocol(capture_root,output)
    result=dict(execution_complete=True,study=STUDY,model_forwards=0,optimizer_updates=0,physics_controls=0,
        original_strict_replay_passed=False,original_model_acceptance_passed=False,
        fitted_weights_saved=False,fitted_weights_deployed=False,model_acceptance_claimed=False,
        task_rows={key:value['rows'] for key,value in reports.items()},cuda_initialized=torch.cuda.is_initialized(),
        scope=declared['interpretation'])
    if result['cuda_initialized']:raise RuntimeError('CPU diagnostic initialized CUDA')
    algebra.save_json(output/'RESULT.json',result)
    print(json.dumps(result),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        args.output.mkdir(parents=True,exist_ok=False)
        algebra.save_json(args.output/'PROTOCOL.json',protocol(args.capture_root))
        return
    try:
        code=run(args.capture_root,args.output,check=args.check)
    except Exception as error:
        if args.output.exists() and not isinstance(error,FileExistsError):
            algebra.save_json(args.output/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),
                original_strict_replay_passed=False,model_forwards=0,optimizer_updates=0))
        raise
    raise SystemExit(code)


if __name__=='__main__':main()
