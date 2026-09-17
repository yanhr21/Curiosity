"""Saved-data physics/coverage plots and an explicit quasi-static mass baseline."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    root=Path(root)
    rows=[]
    fig,axes=plt.subplots(3,1,figsize=(12,10))
    for path in sorted(root.glob('episode_*.json')):
        meta=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as a:
            force=a['normal_load_n'].sum(1)
            observed=force>1e-3
            time=a['timestamp_s']
            mass=a['object_mass_kg']
            # A sensor-only baseline for approximately vertical support. It
            # deliberately retains impacts and failures, where its assumption
            # fails; true force/acceleration/pose are not fed into the estimate.
            estimate=force/9.81
            error=abs(estimate/mass-1)
            rest=observed & (time>=.5) & (time<1.)
            rows.append(dict(episode=meta['episode'],split=meta.get('split','qualification'),
                             frames=len(time),contact_frames=int(observed.sum()),
                             static_observed_frames=int(rest.sum()),
                             sensor_sum_over_gravity_mass_relative_contact=float(error[observed].mean()) if observed.any() else None,
                             sensor_sum_over_gravity_mass_relative_rest=float(error[rest].mean()) if rest.any() else None,
                             support_balance_pass=meta['support_balance_pass'],normal_balance_pass=meta['normal_balance_pass']))
            color=dict(train='tab:blue',val='tab:orange',test='tab:green').get(meta.get('split'),'tab:blue')
            axes[0].bar(meta['episode'],observed.mean(),color=color)
            axes[1].scatter(mass[0],np.median(estimate[rest]) if rest.any() else 0,color=color)
            axes[2].plot(time,a['object_pose_w'][:,2],alpha=.5,color=color)
    axes[0].set_ylabel('Contact fraction');axes[0].set_xlabel('Whole episode ID')
    axes[1].plot([0,1.3],[0,1.3],'k--');axes[1].set_xlabel('True mass [kg]')
    axes[1].set_ylabel('Static sensor sum / g [kg]\nzero = no observed static contact')
    axes[2].set_xlabel('Time [s]');axes[2].set_ylabel('Actual object Z [m]')
    fig.suptitle('All physical episodes retained; blue TRAIN, orange VAL, green TEST')
    fig.tight_layout();fig.savefig(root/'coverage_and_mass.png',dpi=150);plt.close(fig)
    report={'episodes':rows,'total_frames':sum(r['frames'] for r in rows),
            'total_contact_frames':sum(r['contact_frames'] for r in rows),
            'static_mass_baseline_assumption':'Predominantly upward quasi-static support; invalid during impact, squeezing, release or unobserved support.',
            'all_failures_retained':True}
    (root/'DATA_AUDIT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
