"""Saved physical contact coverage against the full known mesh (validation only)."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .geometry import load_sugar_outer_box
from sugar_newton.hand.patches import load_hand_mesh


def main(root):
    root=Path(root)
    with np.load(root/'episode_0000.npz') as a: data={k:a[k] for k in a.files}
    vertices,faces=load_sugar_outer_box()
    vertices=vertices*(data['object_dimensions_m'][0]/np.ptp(vertices,axis=0))
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    tree=cKDTree(vertices);normals=np.asarray(mesh.vertex_normals)
    rows=[];contact_normals=[];distances=[];recorded_phases=[]
    for t in range(len(data['timestamp_s'])):
        contact=data['normal_load_n'][t]>.01
        if not contact.any(): continue
        q=data['object_pose_w'][t];R=Rotation.from_quat(q[3:])
        local=R.inv().apply(data['contact_position_w'][t,contact]-q[:3])
        distance,index=tree.query(local)
        distances.extend(distance);contact_normals.extend(R.apply(normals[index]));recorded_phases.extend([int(data['validation_probe_phase'][t])]*len(distance))
    contact_normals=np.asarray(contact_normals);distances=np.asarray(distances);recorded_phases=np.asarray(recorded_phases)
    fig=plt.figure(figsize=(16,5))
    for phase in range(3):
        frame_mask=data['validation_probe_phase']==phase
        load=data['normal_load_n'].reshape(-1,2,27).sum(2)
        both=frame_mask & (load>.01).all(1)
        idx=np.flatnonzero(both)
        row=dict(phase=phase,both_contact_frames=len(idx),contact_records=int((recorded_phases==phase).sum()))
        if row['contact_records']:
            sel=recorded_phases==phase
            row.update(mean_absolute_true_surface_normal=np.abs(contact_normals[sel]).mean(0).tolist(),
                       median_nearest_full_mesh_vertex_distance_m=float(np.median(distances[sel])),
                       maximum_nearest_full_mesh_vertex_distance_m=float(distances[sel].max()))
        ax=fig.add_subplot(1,3,phase+1,projection='3d')
        if len(idx):
            t=int(idx[0]);q=data['object_pose_w'][t]
            v=Rotation.from_quat(q[3:]).apply(vertices)+q[:3]
            ax.scatter(*v[::8].T,s=1,color='gray',alpha=.18,label='Actual full object (display subset)')
            clouds=[v]
            for side,name,color in [(0,'left','tab:blue'),(1,'right','tab:orange')]:
                hand=load_hand_mesh(name);hq=data['hand_pose_w'][t,side]
                hv=Rotation.from_quat(hq[3:]).apply(hand.vertices)+hq[:3]
                ax.scatter(*hv[::20].T,s=1,color=color,alpha=.2,label=name+' actual mesh')
                clouds.append(hv)
            contact=data['normal_load_n'][t]>.01;p=data['contact_position_w'][t,contact]
            ax.scatter(*p.T,s=25,color='red',label='Measured tactile centroids')
            cloud=np.concatenate(clouds);center=(cloud.max(0)+cloud.min(0))/2;radius=np.ptp(cloud,axis=0).max()/2
            for setter,c in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],center,strict=True): setter(c-radius,c+radius)
            ax.set_box_aspect((1,1,1));row['first_both_contact_frame']=t
            ax.set_title(f'Phase {phase}: actual t={data["timestamp_s"][t]:.2f}s')
        else: ax.set_title(f'Phase {phase}: no simultaneous contact')
        ax.set_xlabel('World X [m]');ax.set_ylabel('World Y [m]');ax.set_zlabel('World Z [m]')
        rows.append(row)
    handles,labels=fig.axes[-1].get_legend_handles_labels()
    if handles: fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7)
    fig.suptitle('Saved physics, first simultaneous touch in each phase; no learned reconstruction')
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(root/'actual_multiface_geometry.png',dpi=150);plt.close(fig)
    report=dict(phases=rows,full_known_mesh_vertices=len(vertices),all_contact_records=len(distances),
                true_normals_and_mesh_query_are_validation_only=True,
                distance_is_to_nearest_vertex_not_continuous_surface=True,
                controller_uses_no_object_state=True,new_optimizer_updates=0)
    (root/'CONTACT_GEOMETRY_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');main(ap.parse_args().root)
