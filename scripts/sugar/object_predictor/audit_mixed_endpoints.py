"""Full saved model, Adam, clocks and matched-batch audit; no model forward."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from .readback_audit import exact


def main(args):
    root=Path(args.root);checks={};reference=None;initial_reference=None
    for mode in args.arms:
        run=root/mode
        model=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
        latest=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        protocol=json.loads((run/'PROTOCOL.json').read_text())
        if initial_reference is None:initial_reference=protocol.get('initial_model_sha256')
        checks[mode+'_same_complete_initial_model']=isinstance(initial_reference,str) and len(initial_reference)==64 and protocol.get('initial_model_sha256')==initial_reference
        checks[mode+'_full_model_exact']=exact(model['model'],latest['model'])
        checks[mode+'_full_adam_exact']=exact(model['optimizer'],latest['optimizer'])
        checks[mode+'_full_protocol_exact']=exact(model['protocol'],protocol) and exact(model['protocol'],latest['protocol'])
        checks[mode+'_final_2000_clocks']=model['step']==latest['step']==protocol['steps']==2000
        checks[mode+'_full_model_counts']=protocol['full_backbone_parameters']==137253744 and protocol['total_parameters']==protocol['optimizer_parameters']==138407503
        state=model['optimizer']['state'];groups=model['optimizer']['param_groups']
        bound=[i for g in groups for i in g['params']]
        bindings=protocol['optimizer_parameter_bindings']
        checks[mode+'_full_named_bindings']=([x['id'] for x in bindings]==bound and
            len({x['name'] for x in bindings})==len(bindings) and
            sum(x['numel'] for x in bindings)==138407503 and
            all(model['model'][x['name']].numel()==x['numel'] for x in bindings))
        unused=[x for x in bindings if x['name']=='backbone.embedding.mask_token']
        checks[mode+'_official_unused_binding']=len(unused)==1 and unused[0]['numel']==54
        unused_ids={x['id'] for x in unused}
        checks[mode+'_every_adam_binding_once']=len(bound)==len(set(bound)) and set(state)==set(bound)-unused_ids
        token=model['model']['backbone.embedding.mask_token'].contiguous()
        checks[mode+'_unused_token_unchanged']=hashlib.sha256(token.numpy().tobytes()).hexdigest()==protocol['unmasked_unused_parameter']['sha256']
        checks[mode+'_full_used_first_moments']=sum(v['exp_avg'].numel() for v in state.values())==138407503-54
        checks[mode+'_full_used_second_moments']=sum(v['exp_avg_sq'].numel() for v in state.values())==138407503-54
        checks[mode+'_all_adam_clocks']=all(int(v['step'])==2000 for v in state.values())
        checks[mode+'_all_adam_finite']=all(torch.isfinite(v[key]).all().item() for v in state.values() for key in ('exp_avg','exp_avg_sq'))
        rows=[json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
        batch_rows=[{k:r[k] for k in ('step','episodes','frames')} for r in rows]
        checks[mode+'_all2000_log_clocks']=[r['step'] for r in rows]==list(range(1,2001))
        if reference is None: reference=batch_rows
        checks[mode+'_all_batches_matched']=batch_rows==reference
        result=json.loads((run/'RESULT.json').read_text())
        checks[mode+'_completed_original_reload']=result['complete'] and result['steps']==2000 and result['full_endpoint_reload_max_abs']<=1e-5
        del model,latest,state,groups;gc.collect()
    report=dict(checks=checks,passed=all(checks.values()),new_optimizer_updates=0)
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--arms',nargs='+',default=['geometry','geometry_contact','geometry_contact_force'])
    main(ap.parse_args())
