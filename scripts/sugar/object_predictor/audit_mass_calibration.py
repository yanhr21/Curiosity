"""Independent saved-array readback of VAL calibration and fresh physical TEST."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(root):
    root=Path(root)
    calibration=json.loads((root/'MASS_VAL_LOG_BIAS_CALIBRATION.json').read_text())
    fresh=json.loads((root/'mass_calibration_fresh_v1/RESULT.json').read_text())
    old=json.loads((root/'mass_calibration_dataset_v1/RESULT.json').read_text())
    checks={}
    def check(name,value):
        checks[name]=bool(value)
    check('twelve_new_episodes',len(fresh['episodes'])==12)
    check('all_2400_controls_preserved',sum(e['frames'] for e in fresh['episodes'])==2400)
    check('all_test_only',all(e['split']=='test' for e in fresh['episodes']))
    check('all_physical_support_pass',all(e['support_balance_pass'] and e['normal_balance_pass'] for e in fresh['episodes']))
    check('new_episode_ids',not {e['episode'] for e in fresh['episodes']} & {e['episode'] for e in old['episodes']})
    check('new_masses',not {e['mass_kg'] for e in fresh['episodes']} & {e['mass_kg'] for e in old['episodes']})
    errors={}
    for mode,details in calibration['models'].items():
        with np.load(details['source_predictions']) as a:
            residuals=[np.mean(a['target'][a['episode']==e,12].astype(np.float64)-a['prediction'][a['episode']==e,12].astype(np.float64)) for e in np.unique(a['episode'])]
            check(mode+'_fit_val_only',set(a['episode'].tolist())==set(details['val_episodes']))
        check(mode+'_bias_recomputed_exact',float(np.mean(residuals))==details['mass_log_bias'])
        for name in ('predictions','force_zero_predictions') if mode=='geometry_contact_force' else ('predictions',):
            with np.load(root/'evaluation_mass_fresh_raw'/mode/(name+'.npz')) as a:
                raw={k:a[k] for k in a.files}
            with np.load(root/'evaluation_mass_fresh_calibrated'/mode/(name+'.npz')) as a:
                adjusted={k:a[k] for k in a.files}
            prefix=mode+'_'+name
            for key in raw:
                if key!='prediction': check(prefix+'_'+key+'_exact',np.array_equal(raw[key],adjusted[key]))
            check(prefix+'_other_twelve_outputs_exact',np.array_equal(raw['prediction'][:,:12],adjusted['prediction'][:,:12]))
            check(prefix+'_mass_offset_exact',np.array_equal(raw['prediction'][:,12]+np.float32(details['mass_log_bias']),adjusted['prediction'][:,12]))
            check(prefix+'_all_new_windows',len(raw['episode'])==468 and set(raw['episode'].tolist())==set(range(2000,2012)))
            p,t,e=adjusted['prediction'],adjusted['target'],adjusted['episode']
            value=float(np.mean([np.abs(np.exp(p[e==episode,12].astype(np.float64)-t[e==episode,12].astype(np.float64))-1).mean() for episode in np.unique(e)]))
            report=json.loads((root/'evaluation_mass_fresh_calibrated'/mode/'RESULT.json').read_text())
            if name=='force_zero_predictions': report=report['force_zero_keep_contact_geometry_and_area']
            check(prefix+'_mass_metric_fp64_readback',abs(value-report['all']['mass_relative']['equal_episode_mean'])<1e-6)
            errors[prefix]=value
    result=dict(checks=checks,passed=all(checks.values()),checks_passed=sum(checks.values()),checks_total=len(checks),independent_fp64_mass_errors=errors,new_optimizer_updates=0,original_uncalibrated_failure_retained=True)
    (root/'FRESH_MASS_CALIBRATION_READBACK.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not result['passed']: raise RuntimeError('Calibration readback failed')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');main(ap.parse_args().root)
