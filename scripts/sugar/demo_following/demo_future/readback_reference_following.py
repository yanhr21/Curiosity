"""Independent reference-versus-executed-state comparison for a frozen controller.

An exploratory follow-up to data pairing, not a held-out demonstration test.
References use one fixed causal rigid alignment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/sugar/smp'))
from sugar_g1_box_schema import TRACKED_BODY_NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--xyz-only',action='store_true',help='Read-only XYZ plots and reproduction of existing full3D metrics, without replacing the original result')
    args=parser.parse_args();run=args.run
    plan=json.loads((run/'PROTOCOL.json').read_text())
    pair=json.loads((run/'READBACK.json').read_text())
    arms=('original','alternate')
    trace={k:dict(np.load(run/k/'TRACE.npz')) for k in arms}
    names=np.load(run/'original/STARTUP.npz')['body_names'].tolist()
    body_ids=[names.index(n) for n in TRACKED_BODY_NAMES]
    switch=plan['switch_control_frame']
    end=min(len(t['done']) for t in trace.values())
    valid=np.arange(end)>=switch
    for t in trace.values():valid &= ~t['done'][:end].reshape(-1)
    matrices={field:np.zeros((2,2)) for field in ('box_position_rmse_m','tracked_body_position_rmse_m')}
    if not valid.any():
        result=dict(execution_completed=True,exploratory_reference_following_checks_passed=False,
                    valid_shared_post_switch_frames=0,matrices=None,matching_over_alternate_reference_ratios=None,
                    checks={'jointly_valid_post_switch_frames_exist':False,'primary_data_pairing_passed':pair['data_pair_feasibility_passed']},
                    reason='No jointly valid post-selection frames remain after excluding terminal/reset frames; no conditional-future comparison is available.',
                    scope='Completed negative physical evidence, no following ratio can be computed from absent post-selection frames.')
        name='REFERENCE_XYZ' if args.xyz_only else 'REFERENCE_FOLLOWING'
        if args.xyz_only:
            previous=json.loads((run/'REFERENCE_FOLLOWING_READBACK.json').read_text())
            assert previous['valid_shared_post_switch_frames']==0 and previous['matrices'] is None
        # A failed first generated control can leave no shared valid samples.
        # Preserve a completed negative readback without inventing a curve.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,3));ax.axis('off')
        ax.text(.5,.5,'No jointly valid post-switch frames.\nFollowing error is unavailable; physical failure is retained.',ha='center',va='center',transform=ax.transAxes)
        fig.savefig(run/(name+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)
        (run/(name+'_READBACK.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    for i,actual_arm in enumerate(arms):
        actual=trace[actual_arm]
        for j,reference_arm in enumerate(arms):
            ref=trace[reference_arm]
            box=actual['object_state_before_w'][:end,0,:3][valid]-ref['reference_object_pos_w'][:end,0][valid]
            body=actual['robot_body_state_before_w'][:end,0][:,body_ids,:3][valid]-ref['reference_body_pos_w'][:end,0][valid]
            matrices['box_position_rmse_m'][i,j]=np.sqrt(np.mean(box**2))
            matrices['tracked_body_position_rmse_m'][i,j]=np.sqrt(np.mean(body**2))
    ratios={field:[float(value[i,i]/max(value[i,1-i],1e-12)) for i in range(2)] for field,value in matrices.items()}
    checks={f'{field}_{arm}_matching_reference_5pct_lower':ratios[field][i]<=.95 for field in matrices for i,arm in enumerate(arms)}
    checks['primary_data_pairing_passed']=pair['data_pair_feasibility_passed']
    if args.xyz_only:
        previous=json.loads((run/'REFERENCE_FOLLOWING_READBACK.json').read_text())
        if not all(np.allclose(value,previous['matrices'][key],rtol=1e-7,atol=1e-9) for key,value in matrices.items()):
            raise RuntimeError('Actual full3D matrix differs from the completed readback')
        path=run/'REFERENCE_XYZ.png'
        if path.exists():raise RuntimeError('Do not overwrite the additional XYZ evidence')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for i,arm in enumerate(arms):
            for c,axis in enumerate(('X','Y','Z')):
                ax=axes[i,c];time=np.arange(end)[valid]*.02
                ax.plot(time,trace[arm]['object_state_before_w'][:end,0,c][valid],color='black',label='Actual '+arm)
                for refarm in arms:ax.plot(time,trace[refarm]['reference_object_pos_w'][:end,0,c][valid],ls='--',label='Reference '+refarm)
                ax.set_title(f'{arm}: box {axis}');ax.set_xlabel('Control time(s)');ax.set_ylabel('World position(m)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'Actual frozen Tracker phase{switch}: all XYZ; original full3D metrics reproduced')
        fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
        report=dict(execution_completed=True,existing_full3d_metrics_reproduced=True,valid_frames=int(valid.sum()),
            optimizer_updates=0,new_physics_steps=0,plot=str(path),plot_inspected=False,
            per_coordinate_box_rmse_m={arm:{refarm:np.sqrt(np.mean((trace[arm]['object_state_before_w'][:end,0,:3][valid]-trace[refarm]['reference_object_pos_w'][:end,0][valid])**2,axis=0)).tolist() for refarm in arms} for arm in arms},
            scope='Additional XYZ evidence from the same saved actual traces. Does not change completed feasibility criteria or turn these camera-free curves into a rendered physical video.')
        (run/'REFERENCE_XYZ_READBACK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);return
    result=dict(execution_completed=True,exploratory_reference_following_checks_passed=all(checks.values()),
                checks=checks,valid_shared_post_switch_frames=int(valid.sum()),
                matrix_rows_actual_arms=list(arms),matrix_columns_reference_arms=list(arms),
                matrices={k:v.tolist() for k,v in matrices.items()},matching_over_alternate_reference_ratios=ratios,
                metric_definition='Root mean squared coordinate error over time, selected bodies and XYZ; world frame after fixed reference transform. Same causal phase and time range for every matrix cell. No window minimization or learned reward scorer.',
                scope=f'Exploratory frozen {plan.get("controller", "refiner")} comparison on a TRAIN-selected pair, not established held-out following. Reference metrics were introduced after candidate selection; they are predeclared for the later matched Tracker input experiment, not the original teacher pairing criteria.')
    (run/'REFERENCE_FOLLOWING_READBACK.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    colors=('tab:blue','tab:orange')
    for i,arm in enumerate(arms):
        time=np.arange(end)*.02
        for j,c in enumerate((0,2)):
            axes[i,j].plot(time[valid],trace[arm]['object_state_before_w'][:end,0,c][valid],color='black',label=f'Actual {arm}')
            for k,refarm in enumerate(arms):
                axes[i,j].plot(time[valid],trace[refarm]['reference_object_pos_w'][:end,0,c][valid],ls='--',color=colors[k],label=f'Reference {plan["reference_pair"][k]}')
            axes[i,j].set_title(f'{arm}: box {("X","Z")[j]} (m)');axes[i,j].set_xlabel('Control time(s)');axes[i,j].legend()
    fig.suptitle(f'Frozen {plan.get("controller", "refiner")}: actual trajectories against both fixed-aligned references')
    fig.tight_layout();fig.savefig(run/'REFERENCE_FOLLOWING.png',dpi=160);plt.close(fig)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
