"""Independent saved-array audit of the two complete official reverse recordings."""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/public/home/yanhongru/Curiosity')
OUT = ROOT/'experiments/demo_following/demo_future_smp_v1/matched_generator_branch_self_replay01_gaps512/frozen_evaluation/paired_own_reverse'


def readback_transfer():
    out=OUT/'fixed_path_transfer';result=json.loads((out/'RESULT.json').read_text())
    assert result['execution_completed'] and result['checks_passed']
    protocol=json.loads((out/'PROTOCOL.json').read_text());checks={};hashes={};rows={}
    for phase in protocol['phases']:
        metrics={}
        for name in ('ranked18_on_ranked18','self18_on_self18','self18_on_ranked18'):
            path=out/f'phase_{phase}_{name}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as a:
                checks[f'{phase}_{name}_finite']=all(np.isfinite(a[k]).all() for k in a.files)
                value=((a['clean'].astype(np.float64)-a['target'][None,:,None])**2).mean((-1,-2)).mean(0)
                checks[f'{phase}_{name}_metric_exact']=np.array_equal(value,result['phases'][str(phase)]['clean_mse'][name]);metrics[name]=value
                model,pathname=name.split('_on_')
                with np.load(OUT/pathname/f'phase_{phase}_steps_16.npz') as recorded:
                    for key in ('seeds','times'):checks[f'{phase}_{name}_{key}_exact']=np.array_equal(a[key],recorded[key])
                    checks[f'{phase}_{name}_target_exact']=np.array_equal(a['target'].astype(np.float64),recorded['normalized_actual_target'])
                    if model==pathname:
                        checks[f'{phase}_{name}_epsilon_exact']=np.array_equal(a['epsilon'],recorded['epsilon'][:,0])
                        checks[f'{phase}_{name}_clean_exact']=np.array_equal(a['clean'],recorded['pred_original'][:,0])
        old=metrics['ranked18_on_ranked18'];new=metrics['self18_on_self18'];cross=metrics['self18_on_ranked18']
        for key,value in [('direct_effect',cross-old),('state_feedback_residual',new-cross),('total_change',new-old)]:
            checks[f'{phase}_{key}_exact']=np.array_equal(value,result['phases'][str(phase)][key])
        checks[f'{phase}_initial_feedback_zero']=np.array_equal(new[:,0],cross[:,0])
        rows[str(phase)]=dict(final_direct=float((cross-old)[:,-1].mean()),final_feedback=float((new-cross)[:,-1].mean()),final_total=float((new-old)[:,-1].mean()),time_mean_direct=float((cross-old).mean()),time_mean_feedback=float((new-cross).mean()))
    for filename,digest in protocol['source_hashes'].items():checks['source_hash_'+filename]=hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    with (out/'SAVED_READBACK.json').open('x') as f:json.dump(dict(checks_passed=True,checks=checks,phases=rows,source_hashes=hashes,scope='Independent saved-array readback, original FP64 reduction order; fixed-state algebraic decomposition not unique causal effect; no model forwards.'),f,indent=2)
    print(json.dumps(dict(checks=len(checks),phases=rows)),flush=True)


def readback_reverse():
    protocol=json.loads((OUT/'PROTOCOL.json').read_text());checks={};rows={};hashes={}
    for phase in protocol['phases']:
        arrays={}
        for model in ('ranked18','self18'):
            report=json.loads((OUT/model/'RESULT.json').read_text())
            checks[model+'_execution']=report['execution_completed'] and report['checks_passed']
            path=OUT/model/f'phase_{phase}_steps_16.npz'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as a:arrays[model]={k:a[k].copy() for k in a.files}
            a=arrays[model];ref=report['phases'][str(phase)]['16']
            clean=a['pred_original'].astype(np.float64);target=a['normalized_actual_target']
            own=((clean-target[None,None,:,None])**2).mean((-1,-2))
            other=((clean-target[::-1][None,None,:,None])**2).mean((-1,-2))
            mean=clean.mean(0);bias=((mean-target[None,:,None])**2).mean((-1,-2))
            variance=((clean-mean[None])**2).mean((0,-1,-2))
            for key,value in dict(clean_mse=own.mean(0),draw_mean_mse=bias,draw_variance=variance,own_preference_draws=(own<other).sum(0)).items():
                checks[f'{phase}_{model}_{key}_exact']=np.array_equal(value,ref[key])
            checks[f'{phase}_{model}_finite']=all(np.isfinite(v).all() for v in a.values())
            checks[f'{phase}_{model}_times']=a['times'].tolist()==list(range(45,-1,-3))
            checks[f'{phase}_{model}_seeds']=a['seeds'].tolist()==protocol['seeds']
            checks[f'{phase}_{model}_transition_chain']=np.array_equal(a['xt'][:,:,:,1:],a['prev_sample'][:,:,:,:-1])
            checks[f'{phase}_{model}_final_clean_equals_prev']=np.array_equal(a['pred_original'][:,:,:,-1],a['prev_sample'][:,:,:,-1])
        a,b=arrays['ranked18'],arrays['self18']
        for key in ('times','seeds','normalized_actual_target'):
            checks[f'{phase}_{key}_matched']=np.array_equal(a[key],b[key])
        checks[f'{phase}_initial_gaussian_exact']=np.array_equal(a['xt'][:,:,:,0],b['xt'][:,:,:,0])
        rows[str(phase)]={m:dict(correct_time_mean=float(((v['pred_original'][:,0].astype(np.float64)-v['normalized_actual_target'][None,:,None])**2).mean()),
            correct_final_mean=float(((v['pred_original'][:,0,:,-1].astype(np.float64)-v['normalized_actual_target'][None])**2).mean())) for m,v in arrays.items()}
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    result=dict(checks_passed=True,checks=checks,phases=rows,source_hashes=hashes,
        scope='Four saved primary seeds only; all11phases, bothconditions. No new model forwards, updates or physical controls. Fixed original reduction order; exact chain and initialnoise matches. Not primary32draw replacement.')
    with (OUT/'SAVED_READBACK.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(checks=len(checks),phases=rows)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--transfer',action='store_true')
    args=parser.parse_args();readback_transfer() if args.transfer else readback_reverse()


if __name__=='__main__':main()
