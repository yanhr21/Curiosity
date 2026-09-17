"""Read actual logged losses and count every saved latent training exposure."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.sugar.demo_following.demo_future.generator_branch_diagnostic import BASE, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--demo-records-source',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    assert run.parent==BASE
    out=run/'TRAINING_LOSS_READBACK.json'
    assert not out.exists()
    plan=json.loads((run/'PROTOCOL.json').read_text())
    cases=plan['real_branch_examples'];batch_size=plan['batch_size'];budget=plan['actual_optimizer_updates']
    replay=plan['generated_state_objective'];seeds=replay['seeds'];times=replay['times']
    assert budget==512 and cases*len(seeds)==batch_size and len(times)==16
    checks={};summary={};all_counts=[];losses={};all_batches=[]
    for arm in ('zero_context','demo_geometry'):
        folder=run/arm
        endpoint=json.loads((folder/'RESULT.json').read_text())
        records=json.loads((folder/'PAIRED_LOSSES.json').read_text());losses[arm]=records
        if arm=='demo_geometry' and args.demo_records_source:
            source=args.demo_records_source.resolve()/arm
            replay_readback=json.loads((source/'EXACT_TRAINING_REPLAY_READBACK.json').read_text())
            assert replay_readback['checks_passed'] and replay_readback['source']==str(folder)
            replacement=json.loads((source/'PAIRED_LOSSES.json').read_text())
            checks['demo_exact_replay_nonbase_fields_preserved']=len(records)==len(replacement)==512 and all(a.keys()==b.keys() and all(a[k]==b[k] for k in a if k!='base') for a,b in zip(records,replacement))
            records=replacement;losses[arm]=records
        batches=json.loads((folder/'BATCH_ORDER.json').read_text());all_batches.append(batches)
        logs=[json.loads(line) for line in (folder/'logs.json.txt').read_text().splitlines() if line.strip()]
        checks[arm+'_full_model_adam_endpoint_checks']=endpoint['endpoint_checks_passed'] and all(endpoint['checks'].values()) and endpoint['actual_optimizer_updates']==budget
        checks[arm+'_all512_saved_records']=len(records)==len(batches)==len(logs)==budget
        checks[arm+'_all_full_shuffled_batches']=all(sorted(v)==list(range(batch_size)) for v in batches)
        checks[arm+'_actual_epoch_update_clock']=all(row['global_step']==row['epoch']==i for i,row in enumerate(logs))
        checks[arm+'_actual_workspace_logged_total_exact']=all(a['train_loss']==b['total'] for a,b in zip(logs,records))
        rank_weight=plan['paired_objective']['weight']
        checks[arm+'_declared_rank_weights_exact']=all(v['weight']==rank_weight for v in records)
        checks[arm+'_loss_components_reconstruct_total']=all(np.isclose(v['total'],v['base']+rank_weight*v['rank']+replay['weight']*v['generated']+(plan['actual_state_supervision']['weight']*v['expert'] if plan.get('actual_state_supervision') else 0),rtol=2e-6,atol=2e-6) for v in records)
        if plan.get('actual_state_supervision'):
            extra=json.loads((folder/'EXPERT_STATE_EXPOSURES.json').read_text())
            checks[arm+'_expert_actual_counts_and_all_rows']=extra['checks_passed'] and sum(extra['actual_counts'])==73728 and min(extra['actual_counts'])>0
            checks[arm+'_expert_loss_clocks']=all(v['expert_step']==i and v['expert_seed']==272500+i and v['expert_weight']==.1 and v['expert_rows']==144 for i,v in enumerate(records))
        if rank_weight==0:checks[arm+'_ablation_rank_records_zero']=all(v['rank']==0 for v in records)
        counts=np.zeros((len(seeds),len(times),cases),dtype=np.int64)
        clock=True
        for i,(indices,row) in enumerate(zip(batches,records)):
            clock=clock and row['generated_step']==i and row['generated_time_index']==i%len(times) and row['generated_time']==times[i%len(times)] and row['generated_rows']==len(indices) and row['generated_weight']==replay['weight']
            indices=np.asarray(indices)
            np.add.at(counts,(indices//cases,np.full(len(indices),i%len(times)),indices%cases),1)
        checks[arm+'_actual_replay_clocks_exact']=clock
        checks[arm+'_every_seed_time_case32exposures']=bool((counts==32).all()) and int(counts.sum())==replay['expected_total_auxiliary_rows']
        checks[arm+'_each_real_case4096exposures']=bool((counts.sum((0,1))==replay['expected_exposures_per_real_case']).all())
        all_counts.append(counts)
        summary[arm]=dict(last=records[-1],total_auxiliary_rows=int(counts.sum()),distinct_seed_time_cases=counts.size,
            by_replay_time={str(t):{k:float(np.mean([row[k] for row in records if row['generated_time']==t])) for k in ('base','rank','generated','total')} for t in times})
    checks['both_arms_exposure_counts_exact']=np.array_equal(*all_counts)
    checks['both_arms_actual_batches_exact']=all_batches[0]==all_batches[1]
    checks['both_arms_auxiliary_clock_records_exact']=all(all(a[k]==b[k] for k in ('generated_step','generated_time','generated_time_index','generated_rows','generated_weight')) for a,b in zip(losses['zero_context'],losses['demo_geometry']))
    if plan.get('actual_state_supervision'):
        checks['both_arms_expert_clock_records_exact']=all(all(a[k]==b[k] for k in ('expert_step','expert_seed','expert_rows','expert_weight')) for a,b in zip(losses['zero_context'],losses['demo_geometry']))
        checks['both_arms_expert_exposure_counts_exact']=json.loads((run/'zero_context/EXPERT_STATE_EXPOSURES.json').read_text())['actual_counts']==json.loads((run/'demo_geometry/EXPERT_STATE_EXPOSURES.json').read_text())['actual_counts']
    report=dict(checks_passed=all(checks.values()),checks=checks,summary=summary,
        demo_record_replay_source=str(args.demo_records_source.resolve()) if args.demo_records_source else None,
        actual_cases=cases,batch_rows=batch_size,source_run=str(run),new_optimizer_updates=0,new_sample_draws=0,new_physics_steps=0,
        scope='Complete saved512TRAIN forwards, original official workspace logs, actual shuffled row indices and deterministic seed/time/case exposure. No new forward, gradient, optimizer or sample. Loss magnitudes and time means are descriptive, not evidence of generation or physical benefit.')
    write(out,report)
    assert report['checks_passed']
    print(json.dumps(dict(checks_passed=True,checks=len(checks),rows_per_arm=replay['expected_total_auxiliary_rows'],last={arm:v['last'] for arm,v in summary.items()})),flush=True)


if __name__=='__main__':main()
