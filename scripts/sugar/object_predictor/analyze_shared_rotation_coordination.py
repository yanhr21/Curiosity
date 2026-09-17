"""Static saved-state suppression candidates, never a rolled-out trajectory."""
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .analyze_shared_cop_kinematics import transported,line_angle
from .hand_floor_feasibility import swept_min_height
from .geometry import palmar_support_frame
from .inspect_contact_surface import fit_surface

R=Path('experiments/object_predictor_v1/overfit_repair_v1')
ROOT=R/'shared_budget_cop_pilot_v1'


def unit(x):return x/np.linalg.norm(x)


def run():
    from sugar_newton.hand.patches import load_hand_mesh
    meshes=[load_hand_mesh(s) for s in ('left','right')]
    vertices=[np.asarray(m.vertices,np.float32).astype(float) for m in meshes]
    supports=np.stack([palmar_support_frame(m,s)[0].inv().apply([0,0,1.]) for m,s in zip(meshes,(-1,1))])
    source=ROOT/'cases/episode_5014/episode_5014.npz'
    surface=source.parent/'contact_surface.npz'
    with np.load(source) as z:a={k:z[k] for k in z.files}
    with np.load(surface) as z:f={k:z[k] for k in z.files}
    v=lambda k:a['validation_controller_'+k]
    base=json.loads((ROOT/'cop_kinematic_decomposition/RESULT.json').read_text())
    rows=[]
    for row in base['all_rows']:
        i=row['frame'];old=a['hand_pose_w'][i-1]
        if v('phase')[i]!=0:raise ValueError('Static simplification applies only prelift: no lift/yaw')
        parent=v('cop_allocation_parent_pose_w')[i]
        target=v('floor_executed_target_pose_w')[i]
        if not np.array_equal(parent[:,3:],target[:,3:]):raise ValueError('Allocation modified rotations')
        ro=Rotation.from_quat(old[:,3:]);rt=Rotation.from_quat(target[:,3:])
        # Read exact previous observed surface; raw controller consumes row i-1.
        sl=slice(f['offset'][i-1],f['offset'][i]);details=[];pivots=[]
        for h in (0,1):
            sel=(f['hand'][sl]==h)&(f['pad'][sl]>=0)&(f['normal_pressure_pa'][sl]>0)&(f['area_m2'][sl]>0)
            points=f['position_hand_frame_m'][sl][sel].astype(float);area=f['area_m2'][sl][sel].astype(float)
            center,rms,n=fit_surface(points,area);pivots.append(center)
            details.append(dict(points=len(points),pads=np.unique(f['pad'][sl][sel]).tolist(),
                area_m2=float(area.sum()),rms_m=rms.tolist(),thickness_ratio=float(rms[0]/rms[1])))
        centers=v('cop_world_m')[i];normals=v('fit_normal_w')[i];n0=unit(normals[0]-normals[1])
        old_alignment=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',ro.apply(supports),normals),-1,1)))
        if not np.allclose(old_alignment,v('alignment_error_deg')[i],rtol=0,atol=1e-8):
            raise ValueError('CAD support normal does not reproduce original alignment angles')
        def prediction(q):
            rot=Rotation.from_quat(q[:,3:]);delta_rot=rot*ro.inv()
            moved=transported(centers,old,q);ns=delta_rot.apply(normals);n=unit(ns[0]-ns[1])
            return line_angle(moved[1]-moved[0],n),line_angle(moved[1]-moved[0],n0),moved,n
        original,oldbasis,_,_=prediction(target)
        candidate_rows=[]
        for suppress in ((True,False),(False,True),(True,True)):
            q=target.copy()
            for h in (0,1):
                if suppress[h]:
                    q[h,:3]+=rt[h].apply(pivots[h])-ro[h].apply(pivots[h])
                    q[h,3:]=old[h,3:]
            predicted,_,_,_=prediction(q)
            align=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',Rotation.from_quat(q[:,3:]).apply(supports),normals),-1,1)))
            floor=np.array([swept_min_height(v,p,t) for v,p,t in zip(vertices,old,q)])
            candidate_rows.append(dict(suppress=list(suppress),predicted_angle_deg=predicted,
                improvement_vs_original_deg=original-predicted,alignment_to_current_fixed_fit_deg=align.tolist(),
                floor_min_m=floor.tolist(),floor_safe=bool((floor>=1e-5).all()),
                current_alignment_cone_safe=bool((align<=5).all())))
        trigger=bool(v('cop_line_angle_deg')[i]>5 and original>oldbasis)
        acceptable=[r for r in candidate_rows if trigger and r['floor_safe'] and r['current_alignment_cone_safe'] and r['improvement_vs_original_deg']>0]
        chosen=min(acceptable,key=lambda r:(sum(r['suppress']),r['predicted_angle_deg'])) if acceptable else None
        rows.append(dict(frame=i,command_s=row['command_s'],borrowed=row['borrowed'],
            current_angle_deg=float(v('cop_line_angle_deg')[i]),predicted_original_angle_deg=original,
            rotation_opposes_transport=trigger,predicted_rotation_penalty_deg=original-oldbasis,
            pca=details,candidates=candidate_rows,selected=chosen))
    groups={}
    for name,select in [('all34_to40',lambda r:True),('borrow69',lambda r:r['borrowed'])]:
        rs=[r for r in rows if select(r)];chosen=[r['selected'] for r in rs if r['selected']]
        groups[name]=dict(clocks=len(rs),opposing_rotation=sum(r['rotation_opposes_transport'] for r in rs),
            safe_improving_candidate=len(chosen),selected_suppression_counts={str(m):sum(tuple(c['suppress'])==m for c in chosen) for m in ((True,False),(False,True),(True,True))},
            static_local_improvement_deg_min_median_max=np.quantile([c['improvement_vs_original_deg'] for c in chosen],[0,.5,1]).tolist() if chosen else None,
            forbidden_cumulative_trajectory_claim=True,
            original_prediction_le5=sum(r['predicted_original_angle_deg']<=5 for r in rs),
            candidate_prediction_le5=sum(c['predicted_angle_deg']<=5 for c in chosen))
    out=ROOT/'rotation_coordination_static_candidates';out.mkdir(exist_ok=False)
    report=dict(complete=True,groups=groups,rows=rows,
        scope='Independent alternatives at each saved state. No integration/re-sampling, no prediction of final admission or success; summing improvements is forbidden.',
        candidate='Only when predicted rigid normal rotation worsens current CoP correction: suppress minimum necessary hand alignment about same measured area pivot, preserving parent closure and executed CoP translation; require original5deg palm/current-fit cone and full sweep floor reserve. Otherwise original candidate.',
        limitations=['Normal transport assumes current measured plane moves rigidly with hand over one command. Actual local refit and force change remain unpredicted.','Current fixed-fit alignment cone and transported-fit CoP model serve different objectives; neither is ground-truth object normal.','Independent safe candidates cannot certify safety or response qualification on an unexecuted trajectory.','Even geometrically improved clocks may violate original load band.'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,surface,Path(__file__))},
        new_physics_controls=0,new_model_forwards=0,uses_object_gt=False)
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(groups,indent=2))


if __name__=='__main__':run()
