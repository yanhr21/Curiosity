"""Replay new pure selector on prior saved states, never integrate a trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .coordinated_rotation_cop import coordinate_rotation
from .geometry import palmar_support_frame


def run(root):
    from sugar_newton.hand.patches import load_hand_mesh
    baseline=root.parent/'shared_budget_cop_pilot_v1'
    source=baseline/'cases/episode_5014/episode_5014.npz'
    fieldfile=source.parent/'contact_surface.npz'
    expected_path=baseline/'rotation_coordination_static_candidates/RESULT.json'
    expected=json.loads(expected_path.read_text())
    with np.load(source) as z:
        a={k:z[k] for k in z.files if k.startswith('validation_controller_') or k in ('hand_pose_w','timestamp_s')}
    with np.load(fieldfile) as z:f={k:z[k] for k in z.files}
    meshes=[load_hand_mesh(s) for s in ('left','right')]
    support=np.stack([palmar_support_frame(m,sign)[0].inv().apply([0,0,1.]) for m,sign in zip(meshes,(-1,1))])
    vertices=[np.asarray(m.vertices,np.float32).astype(float) for m in meshes]
    rows=[];maximum=0.
    for old in expected['rows']:
        i=old['frame'];record={k.removeprefix('validation_controller_'):v[i] for k,v in a.items() if k.startswith('validation_controller_')}
        sl=slice(f['offset'][i-1],f['offset'][i])
        field=dict(pos=f['position_hand_frame_m'][sl],area=f['area_m2'][sl],pressure=f['normal_pressure_pa'][sl],pad=f['pad'][sl],patch=f['hand'][sl])
        target,diag=coordinate_rotation(a['hand_pose_w'][i-1],record['floor_executed_target_pose_w'],record,
            field,support,vertices,prelift=True,time=float(a['timestamp_s'][i]))
        delta=abs(diag['rotation_coord_original_predicted_angle_deg']-old['predicted_original_angle_deg'])
        maximum=max(maximum,delta)
        for j,c in enumerate(old['candidates']):
            maximum=max(maximum,abs(diag['rotation_coord_candidate_predicted_angle_deg'][j]-c['predicted_angle_deg']),
                float(np.max(abs(diag['rotation_coord_candidate_floor_min_z_m'][j]-c['floor_min_m']))))
        expected_mask=old['selected']['suppress'] if old['selected'] else [False,False]
        if maximum>1e-12 or not np.array_equal(diag['rotation_coord_selected_suppressed'],expected_mask):
            raise ValueError('New executable selector differs from saved static analysis')
        rows.append(dict(frame=i,command_s=old['command_s'],borrowed=old['borrowed'],
            selected_suppressed=diag['rotation_coord_selected_suppressed'].tolist(),
            predicted_candidate_angles_deg=diag['rotation_coord_candidate_predicted_angle_deg'].tolist()))
    result=dict(passed=True,independent_saved_states=len(rows),borrowed_states=sum(r['borrowed'] for r in rows),
        candidate_arrays_max_difference=maximum,rows=rows,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,fieldfile,expected_path,Path(__file__),Path(__file__).with_name('coordinated_rotation_cop.py'))},
        full_controller_calls=0,new_physics_controls=0,new_model_forwards=0,
        scope='Only pure geometry selector replay on original independent states. No evolved contact, force, controller state or physics; no integrated trajectory/admission claim.')
    destination=root/'STATIC_SELECTOR_REPLAY.json'
    with destination.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);run(p.parse_args().root)
