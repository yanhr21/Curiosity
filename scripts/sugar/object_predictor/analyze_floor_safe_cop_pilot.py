"""Completed fixed-three saved-only readback; no physics/model/render execution."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import run_floor_safe_cop_pilot as pilot


def contiguous_runs(mask):
    index=np.flatnonzero(mask)
    if not len(index):return []
    return np.split(index,np.flatnonzero(np.diff(index)>1)+1)


def controller_summary(a,folder):
    v=lambda key:a['validation_controller_'+key]
    blocked=v('floor_blocked').astype(bool)
    with np.load(folder/'contact_surface.npz',allow_pickle=False) as z:
        s={key:z[key] for key in ('offset','pad','area_m2','normal_pressure_pa')}
    loads=np.zeros((2400,2))
    for i in range(2400):
        lo,hi=s['offset'][i:i+2]
        for side in (0,1):
            selected=s['pad'][lo:hi]//27==side
            loads[i,side]=(s['area_m2'][lo:hi][selected]*s['normal_pressure_pa'][lo:hi][selected]).sum()
    runs=contiguous_runs(blocked)
    diagnostic_indices=set()
    if blocked.any():
        first=int(np.flatnonzero(blocked)[0]);longest=max(runs,key=len)
        diagnostic_indices.update((first,max(0,first-1),min(2399,first+1),int(longest[0]),int(longest[-1])))
    diagnostic_indices.add(int(np.unravel_index(np.argmax(loads),loads.shape)[0]))
    events=[]
    for i in sorted(diagnostic_indices):
        events.append(dict(frame=i,time_s=float(a['timestamp_s'][i]),blocked=bool(blocked[i]),
            previous_observed_loads_n=(loads[i-1] if i else np.zeros(2)).tolist(),
            current_resulting_loads_n=loads[i].tolist(),
            observed_hand_min_z_m=v('floor_observed_min_z_m')[i].tolist(),
            requested_swept_min_z_m=v('floor_requested_swept_min_z_m')[i].tolist(),
            executed_swept_min_z_m=v('floor_executed_swept_min_z_m')[i].tolist(),
            requested_cop_velocity_m_s=v('floor_requested_cop_velocity_m_s')[i].tolist(),
            executed_cop_velocity_m_s=v('cop_servo_velocity_m_s')[i].tolist(),
            requested_distance_m=v('floor_requested_distance_m')[i].tolist(),
            executed_distance_m=v('distance_m')[i].tolist(),
            requested_lift_start_s=float(v('floor_requested_lift_start_s')[i]),
            executed_lift_start_s=float(v('lift_start_s')[i]),
            requested_phase_s=float(v('floor_requested_motion_elapsed_s')[i]),
            executed_phase_s=float(v('motion_elapsed_s')[i]),
            ready_s=float(v('ready_seconds')[i]),cop_valid=v('cop_valid')[i].tolist(),
            cop_line_angle_deg=float(v('cop_line_angle_deg')[i]),
            fit_valid=v('fit_valid')[i].tolist(),alignment_error_deg=v('alignment_error_deg')[i].tolist(),
            actual_gain_m_per_ns=v('response_gain_m_per_ns')[i].tolist(),
            qualified_upper_n_m=v('response_upper_n_m')[i].tolist(),
            qualified_slopes=v('response_samples')[i].tolist()))
    return dict(blocked_commands=int(blocked.sum()),blocked_wall_time_s=float(blocked.sum()*.02),
        first_blocked_s=float(a['timestamp_s'][np.flatnonzero(blocked)[0]]) if blocked.any() else None,
        longest_contiguous_rejection_s=max((len(r)*.02 for r in runs),default=0.),
        block_runs=[dict(start_s=float(a['timestamp_s'][r[0]]),end_s=float(a['timestamp_s'][r[-1]]),frames=len(r)) for r in runs],
        observed_full_palmar_peak_n=float(loads.max()),final_loads_n=loads[-1].tolist(),
        final_travel_m=v('cop_travel_m')[-1].tolist(),final_phase_s=float(v('motion_elapsed_s')[-1]),
        qualified_secants=v('response_qualified_secants_total')[-1].tolist(),events=events)


def run(root,output):
    declared=pilot.validate_protocol(root)
    actual=json.loads((root/'RESULT.json').read_text())
    baseline=json.loads((pilot.BASELINE/'RESULT.json').read_text())
    for report in (actual,baseline):
        if not report['complete'] or tuple(c['episode'] for c in report['cases'])!=pilot.EPISODES:
            raise ValueError('Require all fixed three completed cases before final readback')
    if output.exists():raise FileExistsError('Preserve previous analysis')
    rows=[];arrays={}
    for config,record in zip(declared['configurations'],actual['cases'],strict=True):
        episode=config['episode'];folder=root/'cases'/f'episode_{episode}'
        checked=pilot.validate_case(folder,config,declared)
        for key in ('physical_passed','physical_checks','late_mass_available','hand_floor_passed','passed'):
            if checked[key]!=record[key]:raise ValueError('Recorded decision differs: '+key)
        with np.load(folder/f'episode_{episode}.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        with np.load(folder/'HAND_FLOOR_SUBSTEPS.npz',allow_pickle=False) as z:
            for key in ('timestamp_s','fk_min_z_m','actual_min_z_m'):
                arrays[f'{episode}_substep_{key}']=z[key]
        for key in ('floor_blocked','floor_blocked_count','floor_requested_swept_min_z_m','floor_executed_swept_min_z_m',
                    'response_gain_m_per_ns','response_upper_n_m','response_samples','motion_elapsed_s','ready_seconds'):
            arrays[f'{episode}_{key}']=a['validation_controller_'+key]
        rows.append(dict(episode=episode,revalidated=checked,controller=controller_summary(a,folder),
            baseline=next(c for c in baseline['cases'] if c['episode']==episode),
            source_sha256={str(p):pilot.sha(p) for p in (folder/'PROTOCOL.json',folder/'RESULT.json',
                folder/f'episode_{episode}.npz',folder/'HAND_FLOOR_SUBSTEPS.npz',folder/'contact_surface.npz')}))
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'ACTUAL_FLOOR_TRANSACTION_READBACK.npz',**arrays)
    summary=dict(complete=True,actual_cases=3,controls_per_case=2400,substeps_per_case=19200,
        cases=rows,original12_passes=sum(r['revalidated']['physical_passed'] for r in rows),
        late_mass_available_cases=sum(r['revalidated']['late_mass_available'] for r in rows),
        full_hand_substep_floor_passes=sum(r['revalidated']['hand_floor_passed'] for r in rows),
        joint_passes=sum(r['revalidated']['passed'] for r in rows),
        all_original_decisions_preserved=True,all_command_transactions_and_substeps_recomputed=True,
        source_sha256={str(p):pilot.sha(p) for p in (Path(__file__),root/'PROTOCOL.json',root/'RESULT.json',pilot.BASELINE/'RESULT.json')},
        new_physics_controls=0,new_model_forwards=0,new_optimizer_updates=0,
        visual_inspection='Separate actual rendered-image review required; this entry does not render or claim viewed images.',
        limits='Accepted geometry and estimator replay certify implementation consistency, not successful grasp. All three failures remain; no automatic16, changed force/clock/gates or new interventions.')
    pilot.write(output/'RESULT.json',summary)
    lines=['# Full-hand floor transaction: actual fixed-three readback','',
        '|Case|Original12|Late mass|Actual8substep floor|Joint|Blocked controls|First block s|Longest blocked s|',
        '|---|---|---|---|---|---|---|---|']
    for r in rows:
        c=r['controller'];v=r['revalidated']
        lines.append(f"|{r['episode']}|{v['physical_passed']}|{v['late_mass_available']}|{v['hand_floor_passed']}|{v['passed']}|{c['blocked_commands']}|{c['first_blocked_s']}|{c['longest_contiguous_rejection_s']:.2f}|")
    lines+=['',f"Original12 {summary['original12_passes']}/3; full-hand floor {summary['full_hand_substep_floor_passes']}/3; joint {summary['joint_passes']}/3.",
        '',summary['limits'],'','Readback uses all 2400 control transactions and all 19200 actual FK/solver hand poses per case. Observed load rows follow original previous-observation timing. Video/frame inspection is separate.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('complete','original12_passes','full_hand_substep_floor_passes','joint_passes')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.output)
