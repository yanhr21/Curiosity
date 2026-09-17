"""Saved-only full-denominator CoP pilot readback. No model or physics calls."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import run_prelift_cop_pilot as pilot


def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def first_time(mask,times):
    rows=np.flatnonzero(mask)
    return float(times[rows[0]]) if len(rows) else None


def original_physical_readback(a,config):
    """Recompute all12 original criteria from complete saved physical arrays."""
    t=a['timestamp_s'];start=float(a['validation_controller_lift_start_s'][-1])
    complete=float(a['validation_controller_lift_complete_s'][-1])
    loads=a['normal_load_n'].reshape(-1,2,27).sum(2)
    hold=t>=start+5. if start>=0 else np.zeros(len(t),bool)
    frames=int(hold.sum())
    if not frames:hold[-1]=True  # Same explicitly invalid original fallback.
    early=(t>=1.)&(t<=2.)
    center=a['object_pose_w'][:,:3]+Rotation.from_quat(a['object_pose_w'][:,3:]).apply(a['object_local_center_m'])
    clearance=a['validation_full_mesh_min_z_m'];c=clearance[hold]
    changes=np.quantile(abs(np.diff(loads[hold],axis=0)),.95,axis=0)/config['load'] if frames>1 else np.full(2,1e9)
    values=dict(bilateral_fraction=float((loads[hold]>.01).all(1).mean()),
        clearance_fraction=float((c>.1).mean()),median_rise_m=float(np.median(center[hold,2])-np.median(center[early,2])),
        peak_hand_load_n=float(loads.max()),hold_mean_hand_load_n=loads[hold].mean(0).tolist(),
        target_band_fraction=(abs(loads[hold]/config['load']-1)<=.25).mean(0).tolist(),
        p95_frame_load_change_over_target=changes.tolist(),hold_clearance_loss_m=float(np.median(c[:50])-np.median(c[-50:])),
        hold_frames=frames,actual_complete_hold_frames=int((t>=complete+1.).sum()) if complete>=0 else 0)
    checks=dict(bilateral=values['bilateral_fraction']>=.8,clearance=values['clearance_fraction']>=.8,
        rise=values['median_rise_m']>=config['lift_height_m']-.04,peak=values['peak_hand_load_n']<=100.,
        load_target=bool(np.max(abs(loads[hold].mean(0)/config['load']-1))<=.25),
        lift_started=start>=0,hold_frames=frames>=100,target_stability=bool((np.asarray(values['target_band_fraction'])>=.9).all()),
        frame_load_stability=bool((changes<=.25).all()),hold_drift=values['hold_clearance_loss_m']<=.05,
        motion_completed=complete>=0,actual_complete_hold_frames=values['actual_complete_hold_frames']>=100)
    return checks,values


def cop_angles(a,folder,config):
    controller=pilot.PreliftCoPAlignmentController(target_load_n=config['load'],response_gain=True,
        force_gain=.000025,controller_revision='sensor_feedback_v2',approach_angle_deg=config['approach_angle_deg'],
        yaw_delta_deg=config['yaw_delta_deg'],lift_height_m=config['lift_height_m'],lateral_xy=config['lateral_xy'])
    s=read(folder/'contact_surface.npz');n=len(a['timestamp_s'])
    valid=np.zeros(n,bool);angles=np.full(n,np.nan);error=np.full(n,np.nan);centers=np.full((n,2,3),np.nan)
    for i in range(1,n):
        sl=slice(s['offset'][i-1],s['offset'][i]);field=dict(pos=s['position_hand_frame_m'][sl],
            area=s['area_m2'][sl],pressure=s['normal_pressure_pa'][sl],pad=s['pad'][sl],patch=s['hand'][sl])
        usable,points,tangent,angle=pilot.observed_alignment(a['hand_pose_w'][i-1],field,controller.support_normals)
        if usable.all():valid[i]=True;angles[i]=angle;error[i]=np.linalg.norm(tangent);centers[i]=points
    return dict(cop_valid=valid,cop_angle_deg=angles,cop_tangent_m=error,cop_centers_w=centers)


def summarize(folder,config):
    a=read(folder/f"episode_{config['episode']}.npz");result=json.loads((folder/'RESULT.json').read_text())
    checks,values=original_physical_readback(a,config)
    if checks!=result['checks'] or all(checks.values())!=result['passed']:
        raise ValueError('Saved original12 decision mismatch: '+str(folder))
    check=pilot.protocol()['late_mass_check']
    indices=pilot.uniform_history_indices(a['timestamp_s'],2381,32,.02)
    mass=pilot.mass_supervision(a,indices,check['criteria'])
    geometry=cop_angles(a,folder,config);t=a['timestamp_s']
    loads=a['normal_load_n'].reshape(-1,2,27).sum(2);lift=a['validation_controller_lift_start_s']
    admitted=np.flatnonzero(lift>=0);first=int(admitted[0]) if len(admitted) else None
    peak=int(np.unravel_index(np.argmax(loads),loads.shape)[0])
    good=geometry['cop_valid']&(geometry['cop_angle_deg']<=5.)
    sustained=np.convolve(good.astype(int),np.ones(50,dtype=int),'full')[:len(good)]>=50
    post=(t>float(lift[first])) if first is not None else np.zeros(len(t),bool)
    support=-(a['validation_hand_normal_vec_w']+a['validation_hand_friction_vec_w']).sum(1)[:,2]
    events=dict(first_bilateral_plane_s=first_time(geometry['cop_valid'],t),
        first_cop_within5deg_s=first_time(good,t),first_cop_within5deg_1s_s=first_time(sustained,t),
        lift_start_s=float(lift[first]) if first is not None else None,
        first_over100N_s=first_time((loads>100).any(1),t),peak_load_s=float(t[peak]),
        first_postlift_hand_contact_loss_s=first_time(post&(loads<.01).any(1),t),
        lift_complete_s=float(a['validation_controller_lift_complete_s'][-1]))
    selected={peak,*[int(np.argmin(abs(t-clock))) for clock in (8,16,24,28,32,40,48)]}
    for time in events.values():
        if time is not None and time>0:selected.add(int(np.argmin(abs(t-time))))
    clocks=[]
    for i in sorted(selected):
        clocks.append(dict(frame=i,post_physics_time_s=float(t[i]),controller_input_time_s=float(t[i-1]) if i else None,
            cop_valid=bool(geometry['cop_valid'][i]),cop_angle_deg=float(geometry['cop_angle_deg'][i]) if geometry['cop_valid'][i] else None,
            cop_tangent_cm=float(geometry['cop_tangent_m'][i]*100) if geometry['cop_valid'][i] else None,
            post_physics_load_n=loads[i].tolist(),post_physics_support_z_n=float(support[i]),
            post_physics_clearance_cm=float(a['validation_full_mesh_min_z_m'][i]*100),
            actual_gain=a['validation_controller_response_gain_m_per_ns'][i].tolist()))
    summary=dict(physical_passed=bool(result['passed']),physical_checks=checks,failed_checks=[k for k,v in checks.items() if not v],
        physical_values=values,late_mass_available=bool(mass['mass_available']),late_mass_status=int(mass['mass_status']),
        late_mass_window=dict(frame=2381,first_frame=int(indices[0]),diagnostics=mass['evaluation_only']),
        events=events,cop_at_admission_deg=float(geometry['cop_angle_deg'][first]) if first is not None and geometry['cop_valid'][first] else None,
        cop_valid_fraction=float(geometry['cop_valid'][1:].mean()),selected_clock_readback=clocks,
        source_sha256={name:pilot.sha(folder/name) for name in ('PROTOCOL.json','RESULT.json',f"episode_{config['episode']}.npz",'contact_surface.npz')})
    if 'validation_controller_cop_travel_m' in a:
        summary['final_added_tangential_travel_m']=a['validation_controller_cop_travel_m'][-1].tolist()
    return summary,geometry


def run(root,output):
    declared=pilot.validate_protocol(root);result=json.loads((root/'RESULT.json').read_text())
    if not result['complete'] or tuple(r['episode'] for r in result['cases'])!=pilot.EPISODES:
        raise ValueError('Require complete fixed three-case pilot; no favorable subset reporting')
    if output.exists():raise FileExistsError('Preserve previous saved-only analysis')
    rows=[];arrays={}
    for config in declared['configurations']:
        episode=config['episode'];row=dict(episode=episode)
        for name,source in [('baseline',pilot.BASELINE),('candidate',root)]:
            summary,geometry=summarize(source/'cases'/f'episode_{episode}',config)
            row[name]=summary
            arrays.update({f'{episode}_{name}_{key}':value for key,value in geometry.items()})
        rows.append(row)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'COP_READBACK.npz',**arrays)
    report=dict(complete=True,cases=rows,planned_cases=3,
        all_original_physical_and_late_passed=all(r['candidate']['physical_passed'] and r['candidate']['late_mass_available'] for r in rows),
        bindings={str(p):pilot.sha(p) for p in (Path(__file__),root/'PROTOCOL.json',root/'RESULT.json')},
        source_timing='CoP/command at row k computed from observed field/hand pose row k-1; load/clearance/support at row k are post-physics. No causal attribution from same-clock correlation alone.',
        limits='Pressure CoP is not exact solver wrench. Passing3 fixed cases does not repair or certify all16 or original blind approach. Prior failures remain.',
        model_forwards=0,physics_controls=0,optimizer_updates=0)
    pilot.write(output/'RESULT.json',report)
    lines=['# Saved-only CoP pilot comparison','',
        '|Case|Physical old→new|Late mass old→new|Peak N old→new|CoP angle at admission old→new|New failed checks|',
        '|---|---|---|---|---|---|']
    def angle(value):return 'N/A' if value is None else f'{value:.2f}°'
    for row in rows:
        b=row['baseline'];c=row['candidate']
        lines.append(f"|{row['episode']}|{b['physical_passed']}→{c['physical_passed']}|{b['late_mass_available']}→{c['late_mass_available']}|{b['physical_values']['peak_hand_load_n']:.2f}→{c['physical_values']['peak_hand_load_n']:.2f}|{angle(b['cop_at_admission_deg'])}→{angle(c['cop_at_admission_deg'])}|{', '.join(c['failed_checks']) or 'none'}|")
    lines+=['',report['source_timing'],'',report['limits'],'',
        'All twelve original physical decisions were independently recomputed and required to equal the saved collector decisions. Fixed frame2381/H32 mass availability is separately recomputed; no new threshold or subset selection. Event and sampled-clock details, including first overload/contact loss, are retained in RESULT.json.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=True,all3_passed=report['all_original_physical_and_late_passed'],output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.root,args.output)
