"""Fixed-clock per-episode 3D inspection and saved full-mesh metric grouping."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .shape_metric import cloud
from .geometry import load_sugar_outer_box


def main(args):
    data=Path(args.data);root=Path(args.evaluation_root)
    metas={};times={}
    for path in sorted(data.glob('episode_*.json')):
        meta=json.loads(path.read_text())
        if meta['split']!='test':continue
        metas[meta['episode']]=meta
        with np.load(path.with_suffix('.npz')) as a:times[meta['episode']]=a['timestamp_s']
    vertices,_=load_sugar_outer_box();dimensions=np.ptp(vertices,axis=0)
    vertices=vertices-(vertices.max(0)+vertices.min(0))/2
    modes=('geometry','geometry_contact','geometry_contact_force');arrays={};reports={};checks={};selected={}
    for mode in modes:
        with np.load(root/mode/'predictions.npz') as a:arrays[mode]={k:a[k] for k in a.files}
        with np.load(root/mode/'shape_errors.npz') as a:saved={k:a[k] for k in a.files}
        current=arrays[mode]
        checks[mode+'_all_test_episodes']=set(current['episode'])==set(metas)
        for key in ('episode','frame','contact'):checks[mode+'_metric_alignment_'+key]=np.array_equal(saved[key],current[key])
        by_group={g:{} for g in ('probe_all','probe_primary','support_all')}
        for e,meta in metas.items():
            mask=current['episode']==e;indices=np.flatnonzero(mask)
            clock=times[e][current['frame'][mask]]
            if meta['acquisition_group']=='probe':
                primary=indices[(clock>=140.)&(clock<=148.)]
                if len(primary)!=8:raise ValueError('Expected all eight fixed primary probe samples')
                by_group['probe_primary'][str(e)]=float(saved['symmetric_nearest_vertex_cm'][primary].mean())
                group='probe_all';chosen=int(primary[0])
            else:group='support_all';chosen=int(indices[0])
            by_group[group][str(e)]=float(saved['symmetric_nearest_vertex_cm'][indices].mean())
            if mode==modes[0]:selected[e]=chosen
            else:
                checks[f'{mode}_{e}_same_fixed_plot_frame']=chosen==selected[e] and np.array_equal(current['target'][chosen],arrays[modes[0]]['target'][chosen])
        reports[mode]={k:dict(episodes=v,equal_episode_mean_cm=float(np.mean(list(v.values()))) if v else None) for k,v in by_group.items()}
    if not all(checks.values()):raise ValueError('Saved full-mesh readback alignment failed')
    for e,index in selected.items():
        fig=plt.figure(figsize=(15,5));saved_clouds={};common=[]
        for mode in modes:
            a=arrays[mode];pc=cloud(vertices,a['prediction'][index],dimensions);tc=cloud(vertices,a['target'][index],dimensions)
            saved_clouds[mode]=pc;common.extend((pc,tc))
        saved_clouds['truth']=tc
        both=np.concatenate(common);center=(both.max(0)+both.min(0))/2;radius=np.ptp(both,axis=0).max()/2
        for i,mode in enumerate(modes):
            ax=fig.add_subplot(1,3,i+1,projection='3d')
            ax.scatter(*tc[::15].T,s=1,alpha=.4,label='Truth')
            ax.scatter(*saved_clouds[mode][::15].T,s=1,alpha=.4,label='Predicted pose/size')
            ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_box_aspect((1,1,1));ax.set_title(mode);ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.set_zlabel('Z [m]')
        a=arrays[modes[0]];frame=int(a['frame'][index]);clock=float(times[e][frame]);ax.legend(fontsize=7)
        fig.suptitle(f'Episode {e}, fixed t={clock:.2f}s: known complete mesh, current left-hand frame')
        fig.tight_layout();fig.savefig(root/f'known_mesh_episode_{e:04d}.png',dpi=150,bbox_inches='tight',pad_inches=.15);plt.close(fig)
        np.savez_compressed(root/f'known_mesh_episode_{e:04d}.npz',**saved_clouds,episode=e,frame=frame,timestamp_s=clock)
    result=dict(checks=checks,passed=True,metric='Full-mesh symmetric nearest-vertex cm',vertices=len(vertices),reports=reports,
                plot_selection='First fixed primary probe sample in140..148s; first sampled support frame; never select by prediction error.',
                all_physical_failures_retained=True,new_optimizer_updates=0,
                scope='Known scanned family transformed by predicted pose/size. Not unseen-shape completion; does not replace original raw state criteria.')
    (root/'SHAPE_BY_ACQUISITION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--evaluation-root',required=True);main(ap.parse_args())
