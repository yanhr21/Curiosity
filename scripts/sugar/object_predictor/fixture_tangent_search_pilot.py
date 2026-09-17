"""Separate three-attempt contact-acquisition + fixed-settle profile.

The old 600-step/fixed10s failures remain failures. This new explicit timing
profile never searches for a good chart clock or uses object GT to steer.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from .fixture_tangent_search import (DT,RADII,SEARCH_SPEED,HOLD_SECONDS,RETREAT_SECONDS,
    MAX_CONTROLS,SEARCH_LENGTH,TangentSearchApproach,TangentSearchFixtureScene,drive_config)
from .collect_fixture_touch import FIELD_KEYS,TRACE_KEYS,write_json
from .shape_fixture_assets import load_fixture_asset,official_action_directions,EXPERIMENT

PAIRS=(('13266',0),('13266',49),('18704',0))
EXTRA_KEYS=('target_xy_m','search_length_m','contact_time_s','snapshot_time_s',
            'exhausted_time_s','search_phase','finished')


def protocol():
    return dict(study='fixture_tangent_search_pilot_v1',
        attempt_order=[dict(object_id=o,direction_index=d) for o,d in PAIRS],
        fixture_drive=asdict(drive_config()),dt=DT,substeps=8,
        maximum_controls_per_attempt=MAX_CONTROLS,maximum_total_controls=3*MAX_CONTROLS,
        search=dict(frame='Public D6 parent XY, no object-geometry query',radii_m=list(RADII),
            path='Positive-X radial connectors followed by one full CCW circle at each radius',
            maximum_path_length_m=SEARCH_LENGTH,speed_m_s=SEARCH_SPEED,axial_search_depth_m=.28,
            first_contact='First actual measured assigned palmar load >=.2N before search exhaustion; stop XY immediately even during initial axial approach'),
        timing=dict(snapshot='First actual contact observation time + exactly4s; never reset on later load or chart quality',
            contact_to_snapshot_s=HOLD_SECONDS,retreat_after_snapshot_s=RETREAT_SECONDS,
            no_contact='Retain failure/mask0 after finite search and2s axial retreat; no fabricated snapshot',
            overload='Measured load>20N latches immediate retreat; attempt fails'),
        superseded_legacy_clock_fields=['fixture_drive.sample_s','fixture_drive.duration_s'],
        chart_adapter='Original whole-snapshot chart first, exact successful output preserved. Only failed snapshots may use supported_local_chart_v1 on observed per-pad points; original1.7mm support edge and3.4mm local radius.',
        qualification=dict(initial_frames=25,initial_full_hand_max_n=.001,
            peak_palmar_and_full_hand_max_n=100.,no_overload=True,final_pre_snapshot_window_s=.5,
            expected_window_controls=25,target_band_n=[1.5,2.5],minimum_target_band_fraction=.8,
            valid_observed_chart_required=True,no_field_overflow=True),
        scope='New acquisition+fixed-settle dynamic-hand fixture profile; not a pass under the old600-control fixed-ray clock, not arbitrary-shape control or reconstruction.',
        failure_policy='Keep all three attempts, including absent contact and runtime errors. No GT chart construction, direction swap, good-frame selection, automatic40 expansion or training.',
        model_forwards=0,optimizer_updates=0)


def validate_protocol(value):
    for key,expected in protocol().items():
        if value.get(key)!=expected:raise ValueError('Declared search profile changed: '+key)
    for field in ('prepared_source_sha256','baseline_artifact_sha256'):
        for name,digest in value.get(field,{}).items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Prepared binding changed: '+name)


def collect(root,prepared_root,declared_path):
    from .retained_execution import require_active_resource
    resource=require_active_resource()
    declared=json.loads(declared_path.read_text());validate_protocol(declared)
    if any(declared['planned_resource'][k]!=resource[k] for k in ('job_id','step_id','host')):
        raise RuntimeError('Search pilot resource binding changed')
    directions,provenance=official_action_directions()
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'PROTOCOL.json',dict(declared,actual_resource=resource,directions_source=provenance))
    rows=[]
    for ident,direction in PAIRS:
        folder=root/f'object_{ident}_direction_{direction:02d}';folder.mkdir()
        row=dict(object_id=ident,direction_index=direction,complete=False,recorded_controls=0,
                 actual_controls=0,actual_physics_substeps=0,partial_control_substeps=0)
        scene=None;trace={k:[] for k in TRACE_KEYS+EXTRA_KEYS};fields={k:[] for k in FIELD_KEYS};offset=[0]
        try:
            asset=load_fixture_asset(prepared_root,ident)
            write_json(folder/'ASSET.json',asset.metadata)
            np.savez_compressed(folder/'object_mesh.npz',vertices=asset.physics_vertices_m,faces=asset.faces,
                fixture_center_m=asset.center_m,fixture_quaternion_xyzw=asset.quaternion_xyzw)
            scene=TangentSearchFixtureScene(asset,directions[direction],drive_config())
            write_json(folder/'SCENE.json',dict(frames={k:v.tolist() for k,v in scene.frames.items()},
                hand_body=scene.hand_body,hand_shape=scene.hand_shape,object_shape=scene.object_shape,
                body_flags=scene.model.body_flags.numpy().tolist(),body_mass=scene.model.body_mass.numpy().tolist(),
                joint_effort_limit=scene.model.joint_effort_limit.numpy().tolist()))
            for frame in range(MAX_CONTROLS):
                state=scene.step(DT,8)
                for key in trace:trace[key].append(state[key])
                for key in fields:fields[key].append(np.asarray(state['field'][key]).copy())
                offset.append(offset[-1]+len(state['field']['pad']));row['recorded_controls']+=1
                if frame%100==0 or state['finished']:
                    print('SEARCH_FIXTURE_CLOCK',ident,direction,frame,state['time_s'],
                        state['measured_palmar_load_n'],state['contact_time_s'],state['target_xy_m'].tolist(),flush=True)
                if state['finished']:break
            row.update(complete=bool(state['finished']),contact_time_s=scene.controller.contact_time_s,
                snapshot_time_s=scene.controller.snapshot_time_s,overload_seen=scene.controller.overload_seen)
        except Exception as exc:
            row.update(error=repr(exc),traceback=traceback.format_exc())
        finally:
            if scene is not None:
                row['actual_physics_substeps']=scene.physics_substeps
                row['actual_controls'],row['partial_control_substeps']=divmod(scene.physics_substeps,8)
                row['field_overflow_steps']=scene.field.overflow_steps
            if row['recorded_controls']:
                np.savez_compressed(folder/'trace.npz',**{k:np.asarray(v) for k,v in trace.items()})
                np.savez_compressed(folder/'observed_surface.npz',offset=np.asarray(offset,np.int64),
                    **{k:np.concatenate(v,axis=0) for k,v in fields.items()})
            write_json(folder/'ATTEMPT.json',row);rows.append(row)
            if scene is not None:del scene
            gc.collect()
    result=dict(complete=len(rows)==3 and all(r['complete'] for r in rows),cases=rows,
        expected_attempts=3,actual_controls=sum(r['actual_controls'] for r in rows),
        actual_physics_substeps=sum(r['actual_physics_substeps'] for r in rows),
        qualification_passed=None,model_forwards=0,optimizer_updates=0)
    write_json(root/'COLLECTION_RESULT.json',result)
    return 0 if result['complete'] else 1


def replay_trace(trace):
    """Observation-only clock/target readback; no simulator, mesh or GT input."""
    from .report_fixture_touch import validate_trace
    n=validate_trace({**trace,'validation_full_hand_load_n':np.zeros(len(trace['time_s']))},max_controls=MAX_CONTROLS)
    if any(len(trace[k])!=n or not np.isfinite(trace[k]).all() for k in EXTRA_KEYS):
        raise ValueError('Invalid search telemetry')
    c=TangentSearchApproach();previous=0.
    for i,time in enumerate(trace['time_s']):
        c.command(float(time),previous,DT);c.observe(float(time),float(trace['measured_palmar_load_n'][i]))
        expected=dict(target_xy_m=c.xy,target_depth_m=c.depth_m,search_length_m=c.search_length_m,
            contact_time_s=c.contact_time_s,snapshot_time_s=c.snapshot_time_s,exhausted_time_s=c.exhausted_time_s,
            search_phase=c.phase,touched=c.touched,overload_seen=c.overload_seen,
            finished=time>=c.finish_time()-1e-10)
        for key,value in expected.items():
            if not np.allclose(trace[key][i],value,atol=1e-10,rtol=0):raise ValueError('Causal search trace mismatch: '+key)
        if i<n-1 and expected['finished']:raise ValueError('Recorded extra steps after declared finish')
        previous=float(trace['measured_palmar_load_n'][i])
    return c,bool(trace['time_s'][-1]>=c.finish_time()-1e-10)


def report(root):
    from .official_active3d import read_template_obj
    from .report_fixture_touch import read_npz,validate_surface,empty_chart,load_window
    from .local_fixture_chart import snapshot_with_local_fallback
    declared=json.loads((root/'PROTOCOL.json').read_text());validate_protocol(declared)
    template,faces,_=read_template_obj(EXPERIMENT/'vendor/Active-3D-Vision-and-Touch/pterotactyl/objects/touch_chart.obj')
    template=template.numpy();faces=faces.verts_idx.numpy()
    charts=root/'charts';charts.mkdir(exist_ok=False);rows=[]
    for ident,direction in PAIRS:
        label=f'object_{ident}_direction_{direction:02d}';folder=root/label
        row=dict(object_id=ident,direction_index=direction,complete=False,errors=[])
        arrays,info=empty_chart(faces,'no_declared_contact_snapshot');checks={}
        try:
            attempt=json.loads((folder/'ATTEMPT.json').read_text());row['attempt_record']=attempt
            if (attempt['object_id'],attempt['direction_index'])!=(ident,direction):raise ValueError('Attempt identity mismatch')
            trace=read_npz(folder/'trace.npz');controller,finished=replay_trace(trace)
            n=len(trace['time_s'])
            checks['complete_declared_dynamic_clock']=bool(finished and attempt['complete'] and
                attempt['recorded_controls']==n==attempt['actual_controls'] and attempt['actual_physics_substeps']==8*n
                and attempt['partial_control_substeps']==0)
            row['complete']=checks['complete_declared_dynamic_clock']
            checks['contact_acquired']=controller.touched
            checks['no_overload']=not controller.overload_seen
            checks['no_field_overflow']=attempt.get('field_overflow_steps')==0
            surface=read_npz(folder/'observed_surface.npz');validate_surface(surface,n)
            load_replay=[]
            for a,b in zip(surface['offset'][:-1],surface['offset'][1:]):
                keep=(surface['pad'][a:b]>=0)&(surface['area'][a:b]>0)&(surface['pressure'][a:b]>0)
                load_replay.append(float(np.sum(surface['area'][a:b][keep]*surface['pressure'][a:b][keep])))
            checks['observed_load_integrity']=bool(np.allclose(load_replay,trace['measured_palmar_load_n'],rtol=0,atol=1e-6))
            row.update(recorded_controls=n,contact_time_s=controller.contact_time_s,snapshot_time_s=controller.snapshot_time_s)
            if controller.touched:
                matches=np.flatnonzero(abs(trace['time_s']-controller.snapshot_time_s)<1e-7)
                if len(matches)==1:
                    frame=int(matches[0]);a,b=surface['offset'][frame:frame+2]
                    observation={k:surface[k][a:b] for k in FIELD_KEYS}
                    arrays,info=snapshot_with_local_fallback(observation,trace['hand_pose_w'][frame],template,faces)
                    arrays.update(source_frame_offset=np.int64(a),snapshot_frame=np.int64(frame))
                    row['snapshot_frame']=frame
                row['pre_snapshot_window']=load_window(trace,controller.snapshot_time_s-.5,controller.snapshot_time_s)
            window=row.get('pre_snapshot_window',{})
            checks['target_band']=window.get('frames')==25 and window.get('target_band_fraction',0)>=.8
            # Validation force never selects the clock or changes the chart.
            full=np.asarray(trace['validation_full_hand_load_n']);loads=np.asarray(trace['measured_palmar_load_n'])
            valid_full=full.shape==(n,) and np.isfinite(full).all() and (full>=0).all()
            checks['full_hand_record_integrity']=bool(valid_full)
            checks['initial_clearance']=bool(valid_full and n>=25 and full[:25].max()<=.001)
            checks['full_hand_peak']=bool(valid_full and full.max()<=100.)
            checks['palmar_peak']=bool(loads.max()<=100.)
            row['peaks_n']=dict(palmar=float(loads.max()),full_hand=float(full.max()) if valid_full else None)
        except Exception as exc:
            row['errors'].append(f'{type(exc).__name__}: {exc}')
        checks['valid_observed_chart']=bool(info['available'])
        checks['record_integrity']=not row['errors']
        row.update(chart=info,qualification_checks=checks,
            qualification_passed=bool(row['complete'] and checks and all(checks.values())),chart_file='charts/'+label+'.npz')
        np.savez_compressed(charts/(label+'.npz'),**arrays);rows.append(row)
    result=dict(study=declared['study'],complete=all(r['complete'] for r in rows),expected_attempts=3,cases=rows,
        qualification_passed=all(r['qualification_passed'] for r in rows),
        scope=declared['scope'],old_fixed600_failures_reclassified=False,new_model_forwards=0,new_optimizer_updates=0)
    write_json(root/'RESULT.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['qualification_passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('collect','report'),required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--protocol',type=Path);p.add_argument('--prepared-root',type=Path)
    a=p.parse_args()
    raise SystemExit(collect(a.root,a.prepared_root,a.protocol) if a.mode=='collect' else report(a.root))
