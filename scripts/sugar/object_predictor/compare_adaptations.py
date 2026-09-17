"""Matched saved-data comparison of runtime stochastic-depth adaptation."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(args):
    base=Path(args.root)
    checks={};reports={};initial_details={}
    for mode in ('geometry','geometry_contact','geometry_contact_force'):
        old=base/'training_headlr1e5'/mode;new=base/'training_no_stochastic'/mode
        op=json.loads((old/'PROTOCOL.json').read_text());np_=json.loads((new/'PROTOCOL.json').read_text())
        for key in ('data','checkpoint','steps','history','stride','batch','seed','head_lr',
                    'deterministic_pooling','episode_ids','samples','total_parameters','optimizer_parameters'):
            checks[f'{mode}_{key}_matched']=op[key]==np_[key]
        checks[f'{mode}_only_new_disables_regularizers']=not op.get('disable_stochastic_regularizers',False) and np_['disable_stochastic_regularizers']
        oldrows=[json.loads(line) for line in (old/'train.jsonl').read_text().splitlines()]
        newrows=[json.loads(line) for line in (new/'train.jsonl').read_text().splitlines()]
        checks[f'{mode}_600_actual_updates']=len(oldrows)==len(newrows)==600
        checks[f'{mode}_every_training_batch_matched']=all(
            all(a[k]==b[k] for k in ('step','episodes','frames'))
            for a,b in zip(oldrows,newrows,strict=True))
        with np.load(old/'initial_val.npz') as a,np.load(new/'initial_val.npz') as b:
            exact=all(np.array_equal(a[k],b[k]) for k in a.files)
            maximum=float(np.max(np.abs(a['prediction']-b['prediction'])))
            labels_exact=all(np.array_equal(a[k],b[k]) for k in a.files if k!='prediction')
            initial_details[mode]=dict(bitwise_all_arrays_equal=exact,prediction_max_abs=maximum,labels_exact=labels_exact)
            if args.numerical_initial_predictions:
                checks[f'{mode}_initial_labels_exact_predictions_within_existing_reload_tolerance']=labels_exact and maximum<=1e-5
            else:
                checks[f'{mode}_same_initial_predictions_and_labels']=exact
        for run,label in ((old,'old'),(new,'new')):
            result=json.loads((run/'RESULT.json').read_text())
            checks[f'{mode}_{label}_complete_reload_passed']=result['complete'] and result['steps']==600 and result['full_endpoint_reload_max_abs']<=1e-5
        for backend in ('newton','isaaclab'):
            aroot=base/'evaluation_headlr1e5_batch1'/backend/mode
            broot=base/'evaluation_no_stochastic_batch1'/backend/mode
            with np.load(aroot/'predictions.npz') as a,np.load(broot/'predictions.npz') as b:
                checks[f'{mode}_{backend}_all_test_conditions_exact']=all(np.array_equal(a[k],b[k]) for k in ('target','episode','frame','contact'))
            old_result=json.loads((aroot/'RESULT.json').read_text());new_result=json.loads((broot/'RESULT.json').read_text())
            reports[f'{backend}/{mode}']={k:dict(old=old_result['all'][k]['equal_episode_mean'],new=new_result['all'][k]['equal_episode_mean'])
                                            for k in ('position_cm','rotation_deg','size_relative','mass_relative')}
    result=dict(passed=all(checks.values()),checks=checks,metrics=reports,new_optimizer_updates=0,
                initial_details=initial_details,numerical_initial_predictions=args.numerical_initial_predictions,
                original_bitwise_comparison_is_not_reclassified=bool(args.numerical_initial_predictions),
                scope='One input-matched seed, one scanned object family and one Newton support fixture; native test is one historical shifted trace.')
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not result['passed']: raise RuntimeError('Matched adaptation comparison failed')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--numerical-initial-predictions',action='store_true')
    main(ap.parse_args())
