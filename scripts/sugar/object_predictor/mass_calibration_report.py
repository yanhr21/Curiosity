"""Saved held-out mass estimates for the controlled physical calibration task."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(args):
    root=Path(args.evaluation_root)
    modes=('geometry','geometry_contact','geometry_contact_force')
    records={};errors={}
    for mode in modes:
        with np.load(root/mode/'predictions.npz') as a:
            pred=np.exp(a['prediction'][:,12]);truth=np.exp(a['target'][:,12]);episodes=a['episode']
        records[mode]=[dict(episode=int(e),true_mass_kg=float(truth[episodes==e].mean()),
                            predicted_mean_kg=float(pred[episodes==e].mean()),
                            mean_relative_error=float((np.abs(pred-truth)/truth)[episodes==e].mean())) for e in np.unique(episodes)]
        errors[mode]=float(np.mean([r['mean_relative_error'] for r in records[mode]]))
    with np.load(root/'geometry_contact_force'/'force_zero_predictions.npz') as a:
        zero=np.exp(a['prediction'][:,12]);assert np.array_equal(np.exp(a['target'][:,12]),truth) and np.array_equal(a['episode'],episodes)
    zero_error=float(np.mean([np.mean((np.abs(zero-truth)/truth)[episodes==e]) for e in np.unique(episodes)]))
    checks=dict(force_mass_error_below_geometry=errors[modes[2]]<errors[modes[0]],
                force_mass_error_below_contact=errors[modes[2]]<errors[modes[1]],
                zero_force_preserve_contact_area_worsens_mass=zero_error>errors[modes[2]],
                force_mass_relative_error_at_most_10pct=errors[modes[2]]<=.1)
    report=dict(scope='Held-out masses under one fixed exact-mesh geometry and support fixture; no pose/shape/material or policy-generalization claim.',
                per_episode=records,equal_episode_mass_relative_errors=errors,
                force_zero_mass_relative_error=zero_error,prospective_checks=checks,
                controlled_mass_objective_passed=all(checks.values()),new_optimizer_updates=0)
    (root/'MASS_CALIBRATION_REPORT.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    lo=float(truth.min())*.8;hi=float(truth.max())*1.2
    axes[0].plot([lo,hi],[lo,hi],'k--',label='Exact mass')
    for mode in modes:
        axes[0].scatter([r['true_mass_kg'] for r in records[mode]],
                        [r['predicted_mean_kg'] for r in records[mode]],label=mode)
    axes[0].scatter([float(truth[episodes==e].mean()) for e in np.unique(episodes)],
                    [float(zero[episodes==e].mean()) for e in np.unique(episodes)],marker='x',label='Force zero, area retained')
    axes[0].set_xlabel('Actual held-out mass [kg]');axes[0].set_ylabel('Mean predicted mass [kg]');axes[0].legend(fontsize=7)
    names=['Geometry','+ Contact','+ Forces / area','Force zero']
    axes[1].bar(names,[errors[m]*100 for m in modes]+[zero_error*100])
    axes[1].axhline(10,ls='--',color='black');axes[1].set_ylabel('Equal-episode relative mass error [%]')
    axes[1].tick_params(axis='x',rotation=15)
    fig.suptitle('Controlled physical mass calibration: all held-out windows retained')
    fig.tight_layout();fig.savefig(root/'mass_calibration.png',dpi=150);plt.close(fig)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('evaluation_root');main(ap.parse_args())
