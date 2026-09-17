"""Read back the actual mixed dataset and prospectively replay all training batches."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import ContactDataset, history_indices
from .sampling import BalancedAcquisitionBatchSampler


def schedule(dataset, steps, seed):
    sampler=BalancedAcquisitionBatchSampler(dataset,4,seed+2)
    result=[]
    while len(result)<steps:
        for indices in sampler:
            result.append([(dataset.episodes[dataset.index[i][0]][0]['episode'],dataset.index[i][1]) for i in indices])
            if len(result)==steps: break
    return result


def main(args):
    root=Path(args.data);protocol=json.loads((root/'PROTOCOL.json').read_text())
    manifest=json.loads((root/'MANIFEST.json').read_text());assert manifest['complete']
    datasets={s:ContactDataset(root,'geometry_contact_force',s,32,5,'episode_uniform_recent',150.)
              for s in ('train','val','test')}
    checks={};episode_groups={};counts={}
    for split,dataset in datasets.items():
        counts[split]={}
        for i,(meta,arrays) in enumerate(dataset.episodes):
            episode=meta['episode'];episode_groups[episode]=meta['acquisition_group']
            expected=list(range(31,len(arrays['timestamp_s']),meta['sampling_stride']))
            actual=[f for e,f in dataset.index if e==i]
            checks[f'{episode}_exact_clocks']=actual==expected
            checks[f'{episode}_finite_observations']=all(np.isfinite(v).all() for v in arrays.values() if np.issubdtype(v.dtype,np.number))
            checks[f'{episode}_causal_history']=all(history_indices(f,32,'episode_uniform_recent').max()==f for f in actual)
            counts[split][str(episode)]=len(actual)
        checks[f'{split}_all_episodes']=len(dataset.episodes)==(48 if split=='train' else 12)
    sets=[{m['episode'] for m,_ in d.episodes} for d in datasets.values()]
    checks['disjoint_episode_splits']=not any(sets[i]&sets[j] for i in range(3) for j in range(i))
    batches=schedule(datasets['train'],protocol['steps_per_arm'],protocol['training_seed'])
    checks['same_seed_schedule_exact']=batches==schedule(datasets['train'],protocol['steps_per_arm'],protocol['training_seed'])
    checks['every_batch_two_probe_two_support']=all(sorted(episode_groups[e] for e,_ in batch)==['probe','probe','support','support'] for batch in batches)
    checks['all_training_episodes_exposed']=set(e for b in batches for e,_ in b)==sets[0]
    if args.run:
        losses=[json.loads(line) for line in (Path(args.run)/'train.jsonl').read_text().splitlines()]
        checks['exact_full_optimizer_budget']=len(losses)==len(batches) and [r['step'] for r in losses]==list(range(1,len(batches)+1))
        checks['every_actual_batch_matches']=len(losses)==len(batches) and all(list(zip(r['episodes'],r['frames']))==b for r,b in zip(losses,batches))
    report=dict(checks=checks,passed=all(checks.values()),samples=counts,
                schedule_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
                source_exposures_per_arm={g:sum(episode_groups[e]==g for b in batches for e,_ in b) for g in ('probe','support')},
                new_optimizer_updates=0,training_admitted=False)
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--run');main(ap.parse_args())
