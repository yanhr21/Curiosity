"""Full known-asset 3D reconstruction metric on saved state predictions.

This is a known-scanned-family pose/scale reconstruction, not arbitrary unseen
shape completion. Nearest-surface geometry complements raw canonical-frame angles.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .geometry import load_sugar_outer_box


def rotation(v):
    x=v[:3]/max(np.linalg.norm(v[:3]),1e-12)
    y=v[3:]-np.dot(x,v[3:])*x
    y=y/max(np.linalg.norm(y),1e-12)
    return np.stack([x,y,np.cross(x,y)],axis=1)


def cloud(vertices,state,dimensions):
    scaled=vertices*(np.exp(state[9:12])/dimensions)
    return scaled@rotation(state[3:9]).T+state[:3]


def main(args):
    root=Path(args.evaluation_root)
    vertices,_=load_sugar_outer_box()
    dimensions=np.ptp(vertices,axis=0)
    vertices=vertices-(vertices.max(0)+vertices.min(0))/2
    reports={}
    for mode in ('geometry','geometry_contact','geometry_contact_force'):
        with np.load(root/mode/'predictions.npz') as a:
            pred,target=a['prediction'],a['target']
            episodes,frames,contact=a['episode'],a['frame'],a['contact']
        errors=[]
        for i,(p,t) in enumerate(zip(pred,target,strict=True)):
            pc=cloud(vertices,p,dimensions);tc=cloud(vertices,t,dimensions)
            if not (np.isfinite(pc).all() and np.isfinite(tc).all()):
                raise ValueError('Nonfinite predicted geometry')
            dpt=cKDTree(tc).query(pc,workers=4)[0]
            dtp=cKDTree(pc).query(tc,workers=4)[0]
            errors.append((dpt.mean()+dtp.mean())*.5*100)
            # One actual saved example for inspection, without choosing the
            # lowest-error sample or fitting anything to held-out labels.
            if i==0:
                np.savez_compressed(root/mode/'first_known_asset_reconstruction.npz',
                                    predicted_points_left_hand_m=pc,target_points_left_hand_m=tc,
                                    episode=episodes[i],frame=frames[i])
        errors=np.asarray(errors)
        np.savez_compressed(root/mode/'shape_errors.npz',symmetric_nearest_vertex_cm=errors,
                            episode=episodes,frame=frames,contact=contact)
        report={}
        for group,mask in [('all',np.ones(len(errors),bool)),('contact',contact),('no_contact',~contact)]:
            report[group]={'samples':int(mask.sum())}
            if mask.any():
                report[group].update(mean_cm=float(errors[mask].mean()),p90_cm=float(np.quantile(errors[mask],.9)),
                                    equal_episode_mean_cm=float(np.mean([errors[mask & (episodes==e)].mean() for e in np.unique(episodes[mask])])))
        reports[mode]=report
        print(mode,json.dumps(report),flush=True)
    result=dict(metric='Mean bidirectional nearest-vertex distance in cm, every vertex of complete positive outer mesh',
                vertices=len(vertices),known_asset_family_only=True,does_not_replace_raw_pose_size_metrics=True,arms=reports)
    (root/'KNOWN_ASSET_SHAPE_METRICS.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(15,5))
    for index,mode in enumerate(reports):
        with np.load(root/mode/'first_known_asset_reconstruction.npz') as a:
            pc=a['predicted_points_left_hand_m'];tc=a['target_points_left_hand_m']
        ax=fig.add_subplot(1,3,index+1,projection='3d')
        # Display thinning only; the metric above uses all outer mesh vertices.
        ax.scatter(*tc[::15].T,s=1,alpha=.4,label='Truth')
        ax.scatter(*pc[::15].T,s=1,alpha=.4,label='Predicted pose/scale')
        both=np.concatenate((pc,tc));center=(both.max(0)+both.min(0))/2
        radius=np.ptp(both,axis=0).max()/2
        ax.set_xlim(center[0]-radius,center[0]+radius)
        ax.set_ylim(center[1]-radius,center[1]+radius)
        ax.set_zlim(center[2]-radius,center[2]+radius)
        ax.set_box_aspect((1,1,1));ax.set_title(mode)
        ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.set_zlabel('Z [m]')
    ax.legend(fontsize=7)
    fig.suptitle('First saved test frame: known full mesh transformed by predicted pose and size')
    fig.tight_layout();fig.savefig(root/'known_asset_reconstruction.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('evaluation_root');main(ap.parse_args())
