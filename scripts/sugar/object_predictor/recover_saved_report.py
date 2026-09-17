"""Summarize completed saved training predictions while retaining a failed replay."""
import json
from pathlib import Path
import numpy as np
import torch
from .data import ContactDataset
from .model import metrics


def main(run):
    root=Path(run)
    if (root/'RESULT.json').exists():
        return
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    ledger=[json.loads(s) for s in (root/'train.jsonl').read_text().splitlines()]
    assert len(ledger)==600 and ledger[-1]['step']==600
    with np.load(root/'test.npz') as a:
        pred=torch.from_numpy(a['prediction']);target=torch.from_numpy(a['target'])
        episodes=torch.from_numpy(a['episode']);contact=torch.from_numpy(a['contact'])
    values=metrics(pred,target);report={}
    for group,mask in [('all',torch.ones_like(contact)),('contact',contact),('no_contact',~contact)]:
        report[group]={'samples':int(mask.sum())}
        if mask.any():
            for k,v in values.items():
                report[group][k]=dict(mean=float(v[mask].mean()),median=float(v[mask].median()),
                                     p90=float(torch.quantile(v[mask],.9)),
                                     equal_episode_mean=float(torch.stack([v[mask & (episodes==e)].mean() for e in episodes[mask].unique()]).mean()))
    data=ContactDataset(protocol['data'],protocol['mode'],'train',protocol['history'],protocol['stride'],history_policy=protocol.get('history_policy','contiguous'),time_scale_s=protocol.get('time_scale_s',1.),normal_policy=protocol.get('normal_policy','stored'))
    constant=torch.from_numpy(np.stack([data[i]['target'] for i in range(len(data))])).mean(0)
    report['train_mean_baseline']={k:float(v.mean()) for k,v in metrics(constant[None].expand(len(target),-1),target).items()}
    report.update(steps=600,training_complete=True,complete=False,
                  original_prediction_replay_passed=False,report_recovered_from_saved_arrays=True,
                  warning='Training and saved predictions completed. Original batched prediction-replay failure is retained; no tolerance relaxation or training rerun.')
    (root/'RESULT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
