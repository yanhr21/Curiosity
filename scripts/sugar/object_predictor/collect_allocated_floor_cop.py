"""Dense ideal tactile contact-surface capture; no object geometry enters observations."""
import argparse,hashlib,json,os,time
from pathlib import Path
from copy import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .controller_interventions import INTERVENTIONS, controller_class, intervention_source_modules


def main(args):
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Retained compute step required')
    if args.sustained_feedback and (not args.surface_feedback or args.purpose != 'diagnostic'):
        raise ValueError('Sustained feedback is currently a surface-feedback diagnostic only')
    alignment_deadband = getattr(args, 'alignment_deadband', False)
    normalized_acquisition = getattr(args, 'normalized_acquisition', False)
    object_sdf_resolution = getattr(args, 'object_sdf_resolution', 128)
    sensor_coverage = getattr(args, 'sensor_coverage', 'anatomical27')
    response_gain = getattr(args, 'response_gain', False)
    controller_revision = getattr(args, 'controller_revision', 'legacy')
    controller_intervention = getattr(args, 'controller_intervention', 'none')
    if controller_revision not in ('legacy', 'sensor_feedback_v2'):
        raise ValueError('Unknown controller revision')
    revised_feedback = controller_revision == 'sensor_feedback_v2'
    if controller_intervention not in ('none', *INTERVENTIONS) or (controller_intervention != 'none' and not revised_feedback):
        raise ValueError('Controller intervention requires explicit sensor_feedback_v2')
    if revised_feedback and (not args.surface_feedback or args.purpose != 'diagnostic' or
            sensor_coverage != 'continuous_palmar_v1' or args.sustained_feedback or alignment_deadband or
            normalized_acquisition or args.force_integral_time != 0. or object_sdf_resolution != 128):
        raise ValueError('sensor_feedback_v2 requires explicit original-physics continuous-palmar diagnostic without legacy variants')
    if response_gain and (not args.surface_feedback or args.purpose != 'diagnostic' or
            sensor_coverage != 'continuous_palmar_v1' or args.sustained_feedback or alignment_deadband or
            normalized_acquisition or args.force_integral_time != 0. or object_sdf_resolution != 128):
        raise ValueError('Response gain is an isolated continuous-palmar diagnostic')
    from .palmar_coverage import COVERAGES
    if sensor_coverage not in COVERAGES:
        raise ValueError('Unknown sensor coverage')
    if sensor_coverage != 'anatomical27' and (not args.surface_feedback or args.purpose != 'diagnostic' or
            args.sustained_feedback or alignment_deadband or normalized_acquisition or
            args.force_integral_time != 0. or object_sdf_resolution != 128):
        raise ValueError('Continuous palmar sensing is currently an isolated original-physics diagnostic')
    if object_sdf_resolution != 128 and args.purpose != 'diagnostic':
        raise ValueError('Object SDF resolution changes are currently diagnostic only')
    if alignment_deadband and (not args.surface_feedback or args.purpose != 'diagnostic'):
        raise ValueError('Alignment deadband is currently a surface-feedback diagnostic only')
    if normalized_acquisition and (not args.surface_feedback or args.purpose != 'diagnostic' or
                                  args.sustained_feedback or alignment_deadband or args.force_integral_time != 0. or
                                  args.load not in (12.,24.) or object_sdf_resolution != 128):
        raise ValueError('Normalized acquisition is an isolated12/24N original-physics diagnostic')
    import warp as wp
    from .grip_motion_scene import GripMotionScene
    from .collect_newton import observation,hand_sites
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(frames=args.frames,dt=.02,mass_kg=args.mass,target_load_n=args.load,seed=args.seed,scale=args.scale,geometry_group=args.geometry_group,approach_angle_deg=args.approach_angle_deg,yaw_delta_deg=args.yaw_delta_deg,lift_height_m=args.lift_height_m,lateral_xy=args.lateral_xy,
                  phases={'approach_and_grip':[0,16],'lift_translate_rotate':[16,20],'hold':[20,24]},
                  controller_inputs=['clock','previous measured palmar normal load of each hand'],
                  original_full_mesh_and_physics=True,model_updates=0,
                  criteria={'final_hold_bilateral_pad_contact_fraction_min':.8,'final_hold_clearance_above10cm_fraction_min':.8,
                            'final_hold_median_rise_min_m':args.lift_height_m-.04,'all_recorded_peak_hand_load_max_n':100.,'final_hold_load_target_relative_error_max':.25},
                  source_sha256={n:hashlib.sha256(Path('scripts/sugar/object_predictor',n).read_bytes()).hexdigest()
                                 for n in ('collect_allocated_floor_cop.py','allocated_floor_cop_scene.py','collect_dense_grip.py','grip_motion_scene.py','grip_scene.py','probe_scene.py','support_scene.py','collect_newton.py','geometry.py')})
    protocol['object_sdf_resolution'] = object_sdf_resolution
    protocol['sensor_coverage'] = sensor_coverage
    if response_gain:
        protocol.update(response_gain=True, response_gain_max_m_per_ns=.00015,
            response_rule='Observed5interval positive secants,5samples/2s recentmax; retain last upper; gain<=.1/(dt*upper), rise<=5%/control, drop immediate; plane invalid forbids gain>base. No stability guarantee.')
        protocol['source_sha256']['response_gain.py'] = hashlib.sha256(
            Path('scripts/sugar/object_predictor/response_gain.py').read_bytes()).hexdigest()
    if sensor_coverage != 'anatomical27':
        protocol.update(sensor_change='Explicit new ideal continuous palmar halfspace skin; original regions preserved, uncovered palmar faces aggregated by nearest XZ footprint; not original27-pad hardware',
                        baseline_channels='Same new trajectory also saves original anatomical pad observations for validation; original trajectories unchanged')
        protocol['source_sha256']['palmar_coverage.py'] = hashlib.sha256(
            Path('scripts/sugar/object_predictor/palmar_coverage.py').read_bytes()).hexdigest()
    if object_sdf_resolution != 128:
        protocol.update(original_full_mesh_and_physics=False, original_full_mesh=True,
                        physics_change='Object SDF resolution128->256 only; handSDF128, material, solver, dt and contact settings unchanged')
    if args.surface_feedback:
        from .allocated_floor_cop_scene import AllocatedFloorCoPScene as SurfaceGripScene
        scene_type=SurfaceGripScene
        protocol.update(controller='tactile surface area-weighted PCA alignment diagnostic',
                        controller_inputs=['clock','hand proprioception','previous measured palmar loads','assigned tactile contact positions and areas'],
                        phases={'align_and_grip_until_ready':[0,40],'lift_duration_s':4,'earliest_lift_s':24},
                        alignment_rate_limit_deg_s=3.,readiness='both planes resolved, alignment <=5 deg, loads within25%, sustained1s',
                        no_forced_lift=True,diagnostic_not_formal_dataset=args.purpose=='diagnostic',purpose=args.purpose,force_gain_m_per_ns=args.force_gain,
                        force_integral_time_s=args.force_integral_time,integral_velocity_limit_m_s=.002,
                        stability_criteria={'per_hand_target_band_fraction_min':.9,'per_hand_p95_frame_load_change_over_target_max':.25,'hold_clearance_loss_max_m':.05})
        if args.sustained_feedback:
            protocol.update(sustained_feedback=True,
                motion_progress='same4s path, advance only after measured readiness sustained1s; continuously align at original3deg/s',
                hold_evaluation='retain original lift_start+5s window including all pauses; additionally require >=100 frames after actual completion+1s',
                no_gain_or_acquisition_change=True)
        if revised_feedback:
            protocol.update(controller_revision=controller_revision,
                plane_admission='Positive-pressure/positive-area observed contact samples and original PCA quality; no 2N load gate',
                motion_progress='same4s path, pause immediately when measured readiness fails; resume after readiness sustained1s; continue original3deg/s alignment and force feedback',
                response_gain_combination='If response_gain is enabled, its estimator is unchanged; geometry-only plane admission controls allow_increase',
                hold_evaluation='retain original lift_start+5s window including pauses; additionally require >=100 frames after actual completion+1s',
                changed_rules=['low_load_plane_admission', 'continued_alignment_after_lift', 'ready_gated_motion_progress'],
                unchanged_rules=['load_target', 'readiness_target_band', 'alignment_rate', 'response_gain_formula', 'all_original_acceptance_thresholds'])
        if controller_intervention != 'none':
            module, _, description, _ = INTERVENTIONS[controller_intervention]
            protocol.update(controller_intervention=controller_intervention,
                intervention=description)
            if controller_intervention in ('continuous_phase_governor_v1', 'legacy_runtime_path_v1',
                                            'freeze_postlift_alignment_v1'):
                protocol['motion_progress'] = description
            if controller_intervention == 'bilateral_common_alignment_v1':
                protocol.update(alignment_progress='Shared observed normal-bisector alignment only when both fits are valid; no unilateral fallback rotation. Original pivot, angular rate, local-plane readiness and closure retained.',
                    intervention_baseline='observed_normal_bisector_alignment_v1',
                    degenerate_common_axis='If both fits are valid but their normal difference is degenerate, retain the existing parent fallback; this intervention changes unilateral-fit behavior only.',
                    alignment_record_semantics='alignment_active is parent requested alignment; suppressed and actual_alignment_active record the executed intervention')
            if controller_intervention == 'regional_bilateral_alignment_v1':
                protocol.update(
                    intervention_baseline='bilateral_common_alignment_v1',
                    geometry_support='Per observed pad, original >=6 points / second RMS>=.0002m / thickness<=.35; greatest eligible positive-contact area, ties smallest pad ID; no hysteresis. Normal and pivot use the same region.',
                    geometry_effects='Support selection propagates to local closure direction, actual-plane readiness and fit-valid admission of the unchanged response-gain estimator. No extra force/control formula changes.',
                    full_observation_preservation='Temporary geometry-only view is restored in finally; all-hand loads and complete sensor field remain original.',
                    regional_record_semantics='contact_area_m2 and regional_total_area_m2 retain whole-hand area; selected area is regional_selected_area_m2. Candidate arrays use global54 pad IDs; zero RMS for <6 points means not fitted. regional_selected_pca_normal_hand has arbitrary PCA sign; actual oriented normal is fit_normal_w. Selection changes include initial admission and valid/invalid transitions.',
                    alignment_record_semantics='actual_alignment_active is executed alignment; unilateral_alignment_suppressed is parent rotation removed. Both-fit degenerate common axis retains the parent fallback.')
            if controller_intervention == 'freeze_postlift_alignment_v1':
                protocol.update(alignment_progress='Retain alignment through the first lift-admission command; later commands undo alignment about the measured contact pivot, while preserving commanded yaw and all translations.',
                    intervention_baseline='legacy_runtime_path_v1',
                    changed_rules=['low_load_plane_admission'],
                    alignment_record_semantics='alignment_active is parent requested alignment; postlift_alignment_suppressed and actual_alignment_active record executed alignment; sustained_ready retains original readiness and motion_rate is actual phase advancement')
            for name in intervention_source_modules(controller_intervention):
                protocol['source_sha256'][name+'.py'] = hashlib.sha256(
                    Path('scripts/sugar/object_predictor', name+'.py').read_bytes()).hexdigest()
        protocol['source_sha256']['surface_grip_scene.py'] = hashlib.sha256(
            Path('scripts/sugar/object_predictor/surface_grip_scene.py').read_bytes()).hexdigest()
        if alignment_deadband:
            protocol.update(alignment_deadband=True, alignment_deadband_deg=5.,
                            alignment_deadband_start_s=24.,
                            alignment_rule='From24s, rotate only outside original5deg readiness cone; earlier acquisition and all force/motion/readiness settings unchanged')
        if normalized_acquisition:
            protocol.update(normalized_acquisition=True,
                            acquisition_rule='Before lift only: gain factor1+clip((.75-load/target)/.25,0,1)*(24/target-1); unchanged approach speed, nearband/postlift gain and all acceptance thresholds')
    else:scene_type=GripMotionScene
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    extra=dict(force_gain=args.force_gain,force_integral_time_s=args.force_integral_time,
               sustained_feedback=args.sustained_feedback, alignment_deadband=alignment_deadband,
               normalized_acquisition=normalized_acquisition, sensor_coverage=sensor_coverage,
               response_gain=response_gain, controller_revision=controller_revision) if args.surface_feedback else {}
    if controller_intervention != 'none':
        extra['controller_type'] = controller_class(controller_intervention)
    wp.init();scene=scene_type(mass=args.mass,scale=args.scale,seed=args.seed,target_load_n=args.load,approach_angle_deg=args.approach_angle_deg,yaw_delta_deg=args.yaw_delta_deg,lift_height_m=args.lift_height_m,lateral_xy=args.lateral_xy,object_sdf_resolution=object_sdf_resolution,**extra)
    mass=float(scene.model.body_mass.numpy()[scene.box_body]);assert np.isclose(mass,args.mass,rtol=1e-6)
    # Newton defines body_com in the rigid body's local frame. This is a
    # supervision/validation label, never a tactile observation or controller
    # input. A mesh AABB center is generally not its center of mass.
    object_local_com = scene.model.body_com.numpy()[scene.box_body].astype(np.float64)
    sites,normals=hand_sites(palm_signs=(-1.,1.));rows=[];dense_rows=[];started=time.monotonic()
    try:
        for frame in range(args.frames):
            scene.step();obs,audit=observation(scene,sites,normals)
            field=scene.field.to_numpy()
            dense_rows.append({k:field[k].copy() for k in ('pos','area','pressure','traction_vec','pad','patch','normal')})
            if sensor_coverage != 'anatomical27':
                dense_rows[-1]['anatomical_pad'] = field['anatomical_pad'].copy()
            if audit['overflow']:raise RuntimeError('Tactile surface overflow')
            pose=scene.state_0.body_q.numpy()[scene.box_body].copy()
            vertex_z=Rotation.from_quat(pose[3:]).apply(scene.box_verts)[:,2]+pose[2]
            normal=scene.tactile.normal_vec.numpy();friction=scene.tactile.friction_vec.numpy()
            row={**obs,'object_pose_w':pose,'object_mass_kg':np.float32(mass),
                 'object_com_w':Rotation.from_quat(pose[3:]).apply(object_local_com)+pose[:3],
                 'validation_object_local_com_m':object_local_com.copy(),
                 'validation_object_velocity_w':scene.state_0.body_qd.numpy()[scene.box_body].copy(),
                 'object_dimensions_m':np.ptp(scene.box_verts,axis=0).astype(np.float32),
                 'object_local_center_m':((scene.box_verts.max(0)+scene.box_verts.min(0))*.5).astype(np.float32),
                 'timestamp_s':np.float64((frame+1)*.02),'validation_full_mesh_min_z_m':np.float64(vertex_z.min()),
                 'validation_hand_normal_vec_w':normal,'validation_hand_friction_vec_w':friction,
                 'validation_resolved_normal_n':scene.tactile.normal_load.numpy(),
                 'validation_unassigned_faces':np.int32(audit['unassigned_faces']),
                 'validation_controller_phase':np.int32(scene.probe_record['phase']),
                 'validation_controller_distance_m':scene.probe_record['distance'].copy()}
            if sensor_coverage != 'anatomical27':
                anatomical_scene = copy(scene)
                anatomical_scene.field = scene.field.raw
                anatomical, _ = observation(anatomical_scene, sites, normals)
                for key in ('normal_load_n', 'shear_force_w', 'contact_position_w', 'contact_area_m2'):
                    row['validation_anatomical_'+key] = anatomical[key]
            if args.surface_feedback:
                for key in ('lift_start_s','ready_seconds','fit_valid','fit_normal_w','alignment_error_deg','integral_velocity_m_s'):
                    row['validation_controller_'+key]=np.asarray(scene.probe_record[key]).copy()
                if args.sustained_feedback or revised_feedback:
                    for key in ('motion_elapsed_s','motion_ready','motion_paused','lift_complete_s'):
                        row['validation_controller_'+key]=np.asarray(scene.probe_record[key]).copy()
                if controller_intervention != 'none':
                    for key in INTERVENTIONS[controller_intervention][3]:
                        row['validation_controller_'+key]=np.asarray(scene.probe_record[key]).copy()
                if alignment_deadband:
                    row['validation_controller_alignment_active']=scene.probe_record['alignment_active'].copy()
                if normalized_acquisition:
                    row['validation_controller_acquisition_gain_multiplier']=scene.probe_record['acquisition_gain_multiplier'].copy()
                if response_gain:
                    for key in ('response_gain_m_per_ns','response_upper_n_m','response_samples'):
                        row['validation_controller_'+key] = scene.probe_record[key].copy()
            if not all(np.isfinite(v).all() for v in row.values()):raise FloatingPointError(frame)
            rows.append(row)
            if frame%50==0:print(json.dumps(dict(frame=frame,time_s=(frame+1)*.02,loads=obs['normal_load_n'].reshape(2,27).sum(1).tolist(),
                clearance_m=float(vertex_z.min()),support_z_n=float(-(normal+friction).sum(0)[2]),elapsed=time.monotonic()-started)),flush=True)
    except BaseException as exc:
        if scene.floor_substep_records:scene.save_floor_receipt(out/'HAND_FLOOR_SUBSTEPS.partial.npz')
        if rows:np.savez_compressed(out/f'episode_{args.episode:04d}.partial.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        (out/'FAILURE.json').write_text(json.dumps(dict(error=repr(exc),frames=len(rows)),indent=2));raise
    scene.save_floor_receipt(out/'HAND_FLOOR_SUBSTEPS.npz')
    a={k:np.stack([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(out/f'episode_{args.episode:04d}.npz',**a)
    offset=np.r_[0,np.cumsum([len(v['pad']) for v in dense_rows])]
    coverage_arrays = {'anatomical_pad': np.concatenate([v['anatomical_pad'] for v in dense_rows])} if sensor_coverage != 'anatomical27' else {}
    np.savez_compressed(out/'contact_surface.npz',offset=offset,position_hand_frame_m=np.concatenate([v['pos'] for v in dense_rows]),area_m2=np.concatenate([v['area'] for v in dense_rows]),normal_pressure_pa=np.concatenate([v['pressure'] for v in dense_rows]),shear_traction_hand_frame_pa=np.concatenate([v['traction_vec'] for v in dense_rows]),pad=np.concatenate([v['pad'] for v in dense_rows]),hand=np.concatenate([v['patch'] for v in dense_rows]),validation_surface_normal_hand_frame=np.concatenate([v['normal'] for v in dense_rows]),**coverage_arrays)
    lift_start=float(scene.probe.lift_start) if args.surface_feedback else 16.
    hold=(a['timestamp_s']>=lift_start+(5. if args.surface_feedback else 4.)) if lift_start>=0 else np.zeros(args.frames,bool)
    hold_frames=int(hold.sum())
    if not hold_frames:hold[-1]=True  # explicitly invalid fallback, never a passing hold
    baseline=(a['timestamp_s']>=1.)&(a['timestamp_s']<=2.)
    loads=a['normal_load_n'].reshape(-1,2,27).sum(2)
    center=a['object_pose_w'][:,:3]+Rotation.from_quat(a['object_pose_w'][:,3:]).apply(a['object_local_center_m'])
    rise=float(np.median(center[hold,2])-np.median(center[baseline,2]))
    values=dict(bilateral_fraction=float((loads[hold]>.01).all(1).mean()),clearance_fraction=float((a['validation_full_mesh_min_z_m'][hold]>.1).mean()),
                median_rise_m=rise,peak_hand_load_n=float(loads.max()),hold_mean_hand_load_n=loads[hold].mean(0).tolist(),
                hold_mean_normal_n=a['validation_resolved_normal_n'][hold].mean(0).tolist(),
                hold_mean_support_z_n=float(-(a['validation_hand_normal_vec_w'][hold]+a['validation_hand_friction_vec_w'][hold]).sum(1)[:,2].mean()))
    checks=dict(bilateral=values['bilateral_fraction']>=.8,clearance=values['clearance_fraction']>=.8,rise=rise>=args.lift_height_m-.04,peak=values['peak_hand_load_n']<=100.,
                load_target=bool(np.max(np.abs(loads[hold].mean(0)/args.load-1))<=.25))
    if args.surface_feedback:
        band=(np.abs(loads[hold]/args.load-1)<=.25).mean(0)
        changes=np.quantile(np.abs(np.diff(loads[hold],axis=0)),.95,axis=0)/args.load if hold_frames>1 else np.full(2,1e9)
        clearance=a['validation_full_mesh_min_z_m'][hold]
        loss=float(np.median(clearance[:50])-np.median(clearance[-50:]))
        values.update(target_band_fraction=band.tolist(),p95_frame_load_change_over_target=changes.tolist(),hold_clearance_loss_m=loss)
        checks.update(lift_started=lift_start>=0,hold_frames=hold_frames>=100,
                      target_stability=bool((band>=.9).all()),frame_load_stability=bool((changes<=.25).all()),hold_drift=loss<=.05)
        if args.sustained_feedback or revised_feedback:
            complete=float(scene.probe.lift_complete)
            actual_hold_frames=int((a['timestamp_s']>=complete+1.).sum()) if complete>=0 else 0
            values.update(lift_complete_s=complete,actual_complete_hold_frames=actual_hold_frames,
                          motion_elapsed_s=float(scene.probe.motion_elapsed),
                          paused_frames=int(a['validation_controller_motion_paused'].sum()))
            checks.update(motion_completed=complete>=0,actual_complete_hold_frames=actual_hold_frames>=100)
    report=dict(episode=args.episode,split=args.split,acquisition_group='grip',sampling_stride=25,frames=args.frames,mass_kg=mass,scale=args.scale,seed=args.seed,geometry_group=args.geometry_group,grip_target_per_hand_n=args.load,approach_angle_deg=args.approach_angle_deg,yaw_delta_deg=args.yaw_delta_deg,lift_height_m=args.lift_height_m,lateral_xy=args.lateral_xy,
                passed=all(checks.values()),checks=checks,values=values,physics_steps=args.frames,optimizer_updates=0,lift_start_s=lift_start,hold_frames=hold_frames,
                sensor_coverage=sensor_coverage,
                scope='Prescribed dual-hand sensing fixture, not a whole-body policy. Validation-only forces/pose excluded from predictor.')
    if revised_feedback:
        report['controller_revision'] = controller_revision
    if controller_intervention != 'none':
        report['controller_intervention'] = controller_intervention
    (out/f'episode_{args.episode:04d}.json').write_text(json.dumps(report,indent=2));(out/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--episode',type=int,required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--surface-feedback',action='store_true');ap.add_argument('--frames',type=int,default=1200)
    ap.add_argument('--force-gain',type=float,default=.00015)
    ap.add_argument('--force-integral-time',type=float,default=0.)
    ap.add_argument('--sustained-feedback',action='store_true')
    ap.add_argument('--alignment-deadband',action='store_true')
    ap.add_argument('--normalized-acquisition',action='store_true')
    ap.add_argument('--sensor-coverage', choices=('anatomical27','continuous_palmar_v1'), default='anatomical27')
    ap.add_argument('--response-gain', action='store_true')
    ap.add_argument('--controller-revision', choices=('legacy','sensor_feedback_v2'), default='legacy')
    ap.add_argument('--controller-intervention', choices=('none', *INTERVENTIONS), default='none')
    ap.add_argument('--object-sdf-resolution',type=int,choices=(128,256),default=128)
    ap.add_argument('--purpose',choices=('diagnostic','perception_dataset'),default='diagnostic')
    selected=ap.parse_args();config=json.loads(Path(selected.protocol).read_text())
    case=next(c for c in config['configurations'] if c['episode']==selected.episode)
    main(argparse.Namespace(**case,output=selected.output,surface_feedback=selected.surface_feedback,frames=selected.frames,force_gain=selected.force_gain,force_integral_time=selected.force_integral_time,purpose=selected.purpose,sustained_feedback=selected.sustained_feedback,alignment_deadband=selected.alignment_deadband,object_sdf_resolution=selected.object_sdf_resolution,normalized_acquisition=selected.normalized_acquisition,sensor_coverage=selected.sensor_coverage,response_gain=selected.response_gain,controller_revision=selected.controller_revision,controller_intervention=selected.controller_intervention))
