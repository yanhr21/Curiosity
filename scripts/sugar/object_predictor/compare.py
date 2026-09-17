"""Compare saved matched predictor arms in physical units, without model reruns."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MODES=('geometry','geometry_contact','geometry_contact_force')


def main(root,evaluation_root=None):
    root=Path(root)
    evaluated=Path(evaluation_root) if evaluation_root else root
    runs={m:json.loads((evaluated/m/'RESULT.json').read_text()) for m in MODES}
    protocols={m:json.loads((root/m/'PROTOCOL.json').read_text()) for m in MODES}
    arrays={}
    for mode in MODES:
        with np.load(evaluated/mode/('predictions.npz' if evaluation_root else 'test.npz')) as src:
            arrays[mode]={k:src[k] for k in src.files}
    anchor=arrays[MODES[0]]
    for mode in MODES[1:]:
        for key in ('target','episode','frame','contact'):
            if not np.array_equal(arrays[mode][key],anchor[key]):
                raise ValueError(f'Unmatched test {mode}/{key}')
        for key in ('seed','steps','history','stride','batch','episode_ids','checkpoint','data','head_lr','deterministic_pooling','disable_stochastic_regularizers','history_policy','time_scale_s','train_sampler','normal_policy','normal_geometry_signature','initial_model_sha256'):
            default={'head_lr':1e-3,'deterministic_pooling':False,'disable_stochastic_regularizers':False,'history_policy':'contiguous','time_scale_s':1.,'train_sampler':'shuffle','normal_policy':'stored'}.get(key)
            if protocols[mode].get(key,default)!=protocols[MODES[0]].get(key,default):
                raise ValueError(f'Unmatched protocol {mode}/{key}')
    keys=('position_cm','rotation_deg','size_relative','mass_relative')
    report={'matched_test_samples':len(anchor['target']),
            'source_complete_flags':{m:runs[m].get('complete') for m in MODES},
            'heldout_episodes':np.unique(anchor['episode']).tolist(),
            'metrics':{mode:{k:runs[mode]['all'][k]['equal_episode_mean'] for k in keys} for mode in MODES},
            'baseline':runs[MODES[0]].get('train_mean_baseline'),
            'scope':evaluation_scope(runs,protocols,evaluation_root),
            'force_input_includes_contact_area':True}
    if 'force_zero_keep_contact_geometry_and_area' in runs[MODES[2]]:
        report['force_zero_keep_contact_geometry_and_area']={
            k:runs[MODES[2]]['force_zero_keep_contact_geometry_and_area']['all'][k]['equal_episode_mean'] for k in keys}
    report['force_vs_contact']={k:report['metrics'][MODES[2]][k]-report['metrics'][MODES[1]][k] for k in keys}
    report['contact_vs_geometry']={k:report['metrics'][MODES[1]][k]-report['metrics'][MODES[0]][k] for k in keys}
    if not evaluation_root:
        fig,axes=plt.subplots(1,5,figsize=(18,4))
        for mode in MODES:
            losses=[json.loads(line) for line in (root/mode/'train.jsonl').read_text().splitlines()]
            axes[0].plot([r['step'] for r in losses],[r['loss'] for r in losses],alpha=.7,label=mode)
            validations=sorted((root/mode).glob('val_*.json'))
            clocks=[int(p.stem.split('_')[-1]) for p in validations]
            values=[json.loads(p.read_text()) for p in validations]
            for ax,key in zip(axes[1:],keys,strict=True):
                ax.plot(clocks,[v['all'][key]['equal_episode_mean'] for v in values],label=mode)
                ax.set_title('VAL '+key)
        axes[0].set_yscale('log');axes[0].set_title('Actual full-objective training loss')
        axes[0].legend(fontsize=6)
        for ax in axes: ax.set_xlabel('Optimizer update')
        fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,key in zip(axes,keys,strict=True):
        ax.bar(range(3),[report['metrics'][m][key] for m in MODES])
        if report['baseline']:
            ax.axhline(report['baseline'][key],ls='--',color='black',label='Train mean baseline')
        ax.set_xticks(range(3),['Geometry','+ Contact','+ Force'],rotation=20)
        ax.set_title(key);ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle('Held-out object state: equal episode mean errors (lower is better)')
    fig.tight_layout();fig.savefig(evaluated/'comparison.png',dpi=150);plt.close(fig)
    for episode in report['heldout_episodes']:
        mask=anchor['episode']==episode
        t=anchor['frame'][mask]*.02
        fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True)
        target=anchor['target'][mask]
        for axis in range(3):
            axes[0,axis].plot(t,target[:,axis]*100,'k--',label='Truth')
            axes[1,axis].plot(t,np.exp(target[:,9+axis])*100,'k--',label='Truth')
            for mode in MODES:
                p=arrays[mode]['prediction'][mask]
                axes[0,axis].plot(t,p[:,axis]*100,label=mode)
                axes[1,axis].plot(t,np.exp(p[:,9+axis])*100,label=mode)
            axes[0,axis].set_title(f'Hand-relative center {"XYZ"[axis]} [cm]')
            axes[1,axis].set_title(f'Object size {"XYZ"[axis]} [cm]')
        for axis,mode in enumerate(MODES):
            axes[2,axis].plot(t,np.exp(target[:,12]),'k--',label='Truth')
            axes[2,axis].plot(t,np.exp(arrays[mode]['prediction'][mask,12]),label=mode)
            axes[2,axis].set_title(mode+' mass [kg]');axes[2,axis].set_xlabel('Time [s]')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'Whole held-out episode {episode}; all frames retained')
        fig.tight_layout();fig.savefig(evaluated/f'test_episode_{episode:04d}.png',dpi=140);plt.close(fig)
    (evaluated/'COMPARISON.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def report_same_data(runs,protocols):
    return all(Path(runs[m]['data']).resolve()==Path(protocols[m]['data']).resolve() for m in MODES)


def evaluation_scope(runs,protocols,evaluation_root):
    data=Path(runs[MODES[0]]['data'] if evaluation_root else protocols[MODES[0]]['data'])
    data_protocol=data/'PROTOCOL.json'
    if data_protocol.exists() and json.loads(data_protocol.read_text()).get('train_sampler')=='balanced_acquisitions':
        return 'One scanned asset family: mixed ground-supported multi-face probes and support fixtures. All predictor arms share force-feedback probe trajectories: geometry-only prediction is not touch-free acquisition. Read acquisition-specific geometry and mass metrics separately; no unseen-shape or policy claim.'
    if data_protocol.exists() and json.loads(data_protocol.read_text()).get('mass_only_calibration_diagnostic',False):
        return 'Controlled mass-calibration diagnostic: one fixed scanned geometry, initial pose and support behavior, held-out masses. Not general shape/pose inference.'
    metadata=list(data.glob('episode_*.json'))
    if metadata and json.loads(metadata[0].read_text()).get('backend')=='IsaacLab/TacSL':
        return 'Historical native IsaacLab trace only: different gesture, scale and sensor calibration; no fresh native rollout or native fine-tuning.'
    if evaluation_root and not report_same_data(runs,protocols):
        return 'Different recorded Newton configurations from the training corpus; report this geometry/pose/behavior shift separately.'
    return 'Held-out configurations of one scanned box family and one support behavior; no unseen-shape or policy-benefit claim.'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--evaluation-root')
    args=ap.parse_args();main(args.root,args.evaluation_root)
