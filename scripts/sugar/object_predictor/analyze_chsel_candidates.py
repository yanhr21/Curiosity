"""Saved candidate landscape, without new registration or candidate selection.

Observable displacement/cost frontiers are saved before evaluator labels open.
This diagnoses whether the existing candidate pool supports a prior-aware step;
it is not a learned model or a calibrated posterior.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .shape_metric import rotation


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def angle(a,b):
    relative=np.einsum('nji,jk->nik',a,b)
    return np.degrees(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))


def main(args):
    assert os.environ.get('SLURM_STEP_ID')
    root=Path(args.root);out=root/'candidate_landscape';out.mkdir(exist_ok=False)
    arms=['chsel_qualification_v1_r1','chsel_hand_free_v1']
    mesh=read(Path('experiments/object_predictor_v1/rendered_carry_v1/object_mesh.npz'))
    vertices=mesh['vertices'].astype(np.float64);dims=np.ptp(vertices,axis=0)
    vertices-=(vertices.max(0)+vertices.min(0))/2
    plans=[]
    for arm in arms:
        records=json.loads((root/arm/'OBSERVATION_PROGRESS.json').read_text())
        for record in records:
            name=f'episode_{record["episode"]}_frame_{record["frame"]}.npz'
            a=read(root/arm/name);h=a['candidates_hand_to_object'].astype(np.float64)
            indices=np.sort(np.unique(h.round(6).reshape(len(h),-1),axis=0,return_index=True)[1])
            pose=np.linalg.inv(h[indices]);cost=a['candidate_cost'][indices].astype(np.float64)
            original=a['original_prediction'];r0=rotation(original[3:9]);t0=original[:3]
            dc=pose[:,:3,3]-t0;dr=pose[:,:3,:3]-r0
            v=vertices*(np.exp(original[9:12])/dims);mean=v.mean(0);second=v.T@v/len(v)
            # Exact second-moment expression for corresponding full-vertex RMS.
            ms=np.einsum('ni,ni->n',dc,dc)+np.einsum('nij,jk,nik->n',dr,second,dr)+2*np.einsum('ni,nij,j->n',dc,dr,mean)
            shift=np.sqrt(np.maximum(ms,0))*100
            translation=np.linalg.norm(dc,axis=1)*100;rotation_shift=angle(pose[:,:3,:3],r0)
            order=np.lexsort((cost,shift));ordered=cost[order]
            front=order[np.r_[True,ordered[1:]<np.minimum.accumulate(ordered[:-1])]]
            assert indices[0]==0 and shift[0]<1e-4
            lower=cost<cost[0]
            row=dict(arm=arm,episode=record['episode'],frame=record['frame'],unique_candidates=len(indices),
                lower_observation_cost_count=int(lower.sum()),pareto_count=len(front),
                min_full_mesh_shift_cm_for_lower_cost=float(shift[lower].min()) if lower.any() else None,
                min_rotation_shift_deg_for_lower_cost=float(rotation_shift[lower].min()) if lower.any() else None,
                min_translation_shift_cm_for_lower_cost=float(translation[lower].min()) if lower.any() else None)
            np.savez_compressed(out/f'{arm}_{name}',source_indices=indices,full_mesh_prior_shift_rms_cm=shift,
                center_prior_shift_cm=translation,rotation_prior_shift_deg=rotation_shift,
                observation_cost=cost,pareto_indices=front)
            plans.append((row,pose,cost,shift,translation,rotation_shift,front,original))
    (out/'OBSERVABLE_SUMMARY.json').write_text(json.dumps([x[0] for x in plans],indent=2))
    # Pool coverage below is evaluator-only, never used to emit a new predictor.
    labels=read(root/'chsel_qualification_inputs/evaluation_only.npz')
    lookup={(int(e),int(f)):t for e,f,t in zip(labels['episode'],labels['frame'],labels['target'])}
    for arm in arms:
        fig,axs=plt.subplots(4,2,figsize=(13,15),constrained_layout=True)
        for ax,entry in zip(axs.flat,[x for x in plans if x[0]['arm']==arm]):
            row,pose,cost,shift,translation,rotation_shift,front,original=entry
            target=lookup[(row['episode'],row['frame'])]
            center_error=np.linalg.norm(pose[:,:3,3]-target[:3],axis=1)*100
            rotation_error=angle(pose[:,:3,:3],rotation(target[3:9]))
            better=(center_error<center_error[0]) & (rotation_error<rotation_error[0])
            row['evaluator_only']=dict(baseline_center_cm=float(center_error[0]),baseline_rotation_deg=float(rotation_error[0]),
                candidates_improving_both_center_and_rotation=int(better.sum()),
                improving_both_and_lower_observation_cost=int((better&(cost<cost[0])).sum()),
                warning='Candidate-pool coverage only, no oracle selection or new performance claim')
            plot=ax.scatter(shift,cost/cost[0],c=rotation_shift,s=7,cmap='viridis',vmin=0,vmax=180,alpha=.5)
            ax.plot(shift[front],cost[front]/cost[0],color='tomato',lw=1,label='Observation / prior-shift Pareto')
            ax.scatter([shift[0]],[1],marker='*',s=100,color='black',label='Frozen Utonia')
            ax.set(title=f'{row["episode"]} / {row["frame"]}: {len(cost)} unique candidates',
                xlabel='RMS displacement of full predicted mesh from prior (cm)',ylabel='Observation cost / original cost')
            ax.set_yscale('log');ax.grid(alpha=.2)
        fig.colorbar(plot,ax=axs,shrink=.7,label='Rotation change from Utonia (degrees)')
        fig.suptitle(f'{arm}: saved candidates only; red frontier is not a selected estimate')
        fig.savefig(out/f'{arm}.png',dpi=140);plt.close(fig)
    rows=[x[0] for x in plans]
    report=dict(complete=True,cases=rows,model_forwards=0,optimizer_updates=0,registration_calls=0,physics_steps=0,
        observation_metric='Exact corresponding-vertex RMS relative to prior, not symmetric surface accuracy or calibrated uncertainty',
        next_decision='Check whether lower-cost candidates also preserve prior geometry before implementing fusion. Pool absence requires improving observations/search, not just reranking.')
    (out/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);main(ap.parse_args())
