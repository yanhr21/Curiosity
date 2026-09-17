"""CPU-only input/sampler qualification using completed real traces, not a training corpus."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import ContactDataset, MODES
from .audit_mixed_sampling import schedule


def main(args):
    combined=None;checks={};sources=[]
    for group,root,stride in [('probe',args.probe,50),('support',args.support,5)]:
        dataset=ContactDataset(root,'geometry_contact_force','train',32,stride,'episode_uniform_recent',150.)
        for meta,arrays in dataset.episodes:
            # Metadata is copied in memory only. Live source files are untouched.
            meta['acquisition_group']=group
            sources.append(dict(episode=meta['episode'],frames=len(arrays['timestamp_s']),group=group))
        if combined is None: combined=dataset
        else:
            offset=len(combined.episodes);combined.episodes.extend(dataset.episodes)
            combined.index.extend((e+offset,f) for e,f in dataset.index)
    batches=schedule(combined,2000,310016)
    checks['all_2000_batches_repeat_exact']=batches==schedule(combined,2000,310016)
    groups={m['episode']:m['acquisition_group'] for m,_ in combined.episodes}
    checks['all_2000_batches_balanced']=all(sorted(groups[e] for e,f in b)==['probe','probe','support','support'] for b in batches)
    checks['all_available_episodes_exposed']=set(e for b in batches for e,f in b)==set(groups)
    index_lookup={(combined.episodes[e][0]['episode'],f):i for i,(e,f) in enumerate(combined.index)}
    for e,f in batches[0]+batches[-1]:
        i=index_lookup[e,f];rows=[]
        for mode in MODES:
            combined.mode=mode;rows.append(combined[i])
        checks[f'{e}_{f}_same_targets']=all(np.array_equal(rows[0]['target'],r['target']) for r in rows[1:])
        checks[f'{e}_{f}_finite_inputs']=all(np.isfinite(r['feat']).all() and np.isfinite(r['coord']).all() for r in rows)
        checks[f'{e}_{f}_force_keeps_contact_coords']=np.array_equal(rows[1]['coord'],rows[2]['coord'])
        checks[f'{e}_{f}_full32_frames']=all(len(r['coord'])==32 for r in rows)
    report=dict(checks=checks,passed=all(checks.values()),sources=sources,
                new_optimizer_updates=0,gpu_calls=0,
                scope='Input adapter and 2000-batch schedule diagnostic on currently completed TRAIN cases; not the completed 72-case mixed corpus or model qualification.')
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--probe',required=True);ap.add_argument('--support',required=True)
    ap.add_argument('--output',required=True);main(ap.parse_args())
