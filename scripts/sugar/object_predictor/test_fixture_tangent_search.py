"""CPU-only acquisition state-machine, clock and unchanged-default checks."""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

from .fixture_tangent_search import (TangentSearchApproach,FixtureTouchScene,MAX_CONTROLS,
    SEARCH_LENGTH,SEARCH_SPEED,DT,search_xy,drive_config)
from .fixture_tangent_search_pilot import replay_trace,protocol,validate_protocol
from .collect_fixture_touch import TRACE_KEYS


class SearchFixtureTests(unittest.TestCase):
    def trace(self,contact_frame=None,overload_frame=None):
        c=TangentSearchApproach();previous=0.;rows=[]
        for i in range(MAX_CONTROLS):
            time=(i+1)*DT;c.command(time,previous,DT)
            load=2. if contact_frame is not None and i>=contact_frame else 0.
            if i==overload_frame:load=21.
            c.observe(time,load)
            rows.append(dict(time_s=time,hand_pose_w=np.array([0,0,0,0,0,0,1.]),hand_velocity_w=np.zeros(6),
                measured_palmar_load_n=load,validation_full_hand_load_n=load,target_depth_m=c.depth_m,
                touched=c.touched,overload_seen=c.overload_seen,target_xy_m=c.xy.copy(),search_length_m=c.search_length_m,
                contact_time_s=c.contact_time_s,snapshot_time_s=c.snapshot_time_s,exhausted_time_s=c.exhausted_time_s,
                search_phase=c.phase,finished=time>=c.finish_time()-1e-10))
            previous=load
            if rows[-1]['finished']:break
        return c,{k:np.asarray([r[k] for r in rows]) for k in rows[0]}

    def test_contact_during_axial_approach_stops_without_reaching_center(self):
        c,t=self.trace(contact_frame=99)
        self.assertAlmostEqual(c.contact_time_s,2.)
        self.assertAlmostEqual(c.snapshot_time_s,6.)
        self.assertLess(t['target_depth_m'][99],.28)
        np.testing.assert_array_equal(t['target_xy_m'],np.zeros_like(t['target_xy_m']))
        np.testing.assert_allclose(t['target_depth_m'][99:300],t['target_depth_m'][99],atol=1e-15)
        self.assertAlmostEqual(t['time_s'][-1],8.)

    def test_contact_during_search_locks_xy_and_never_resets_snapshot(self):
        c,t=self.trace(contact_frame=499)
        self.assertGreater(np.linalg.norm(t['target_xy_m'][499]),0.)
        np.testing.assert_array_equal(t['target_xy_m'][499:],np.broadcast_to(t['target_xy_m'][499],t['target_xy_m'][499:].shape))
        self.assertAlmostEqual(c.snapshot_time_s,14.)
        before=c.snapshot_time_s;c.observe(16.,0.);c.observe(16.02,2.)
        self.assertEqual(c.snapshot_time_s,before)
        _,finished=replay_trace(t);self.assertTrue(finished)

    def test_no_contact_is_bounded_exhaustion_not_synthetic_snapshot(self):
        c,t=self.trace()
        self.assertFalse(c.touched);self.assertEqual(c.snapshot_time_s,-1.)
        self.assertLessEqual(len(t['time_s']),MAX_CONTROLS)
        self.assertAlmostEqual(c.search_length_m,SEARCH_LENGTH)
        self.assertLessEqual(np.linalg.norm(t['target_xy_m'],axis=1).max(),.06+1e-12)
        self.assertLessEqual(np.linalg.norm(np.diff(t['target_xy_m'],axis=0),axis=1).max(),SEARCH_SPEED*DT+1e-12)
        self.assertAlmostEqual(t['time_s'][-1]-c.exhausted_time_s,2.)
        replay,finished=replay_trace(t);self.assertTrue(finished);self.assertFalse(replay.touched)

    def test_overload_latches_retreat_and_no_later_contact_restarts_search(self):
        c,t=self.trace(overload_frame=449)
        self.assertTrue(c.overload_seen);self.assertFalse(c.touched)
        self.assertAlmostEqual(t['time_s'][-1],11.)
        self.assertLess(t['target_depth_m'][450],t['target_depth_m'][449])
        np.testing.assert_array_equal(t['target_xy_m'][449:],np.broadcast_to(t['target_xy_m'][449],t['target_xy_m'][449:].shape))
        c.observe(11.1,2.);self.assertFalse(c.touched)

    def test_legacy_target_hook_and_profile_parameters_remain_explicit(self):
        self.assertEqual(FixtureTouchScene.target_translation(None,.123),[0.,0.,.123])
        self.assertEqual(drive_config().angular_kp_nm_rad,32.)
        self.assertEqual(drive_config().angular_kd_nms_rad,.3)
        self.assertEqual(drive_config().angular_effort_limit_nm,2.)
        p=protocol();validate_protocol(p);p['timing']['contact_to_snapshot_s']=2.
        with self.assertRaises(ValueError):validate_protocol(p)

    def test_replay_rejects_reselected_good_snapshot_or_late_xy_motion(self):
        _,t=self.trace(contact_frame=499)
        t['snapshot_time_s'][550]+=DT
        with self.assertRaisesRegex(ValueError,'snapshot_time'):replay_trace(t)
        t['snapshot_time_s'][550]-=DT;t['target_xy_m'][550,0]+=.001
        with self.assertRaisesRegex(ValueError,'target_xy'):replay_trace(t)

    def test_original_600_step_axial_controller_and_default_target_are_unchanged(self):
        from .fixture_touch_scene import FixtureApproach,FixtureDrive
        c=FixtureApproach();cfg=FixtureDrive();depth=0.;touched=False;overload=False;time=0.
        for i in range(600):
            time+=.02;load=0. if i<200 else 1.75
            touched |= load>=.2;overload |= load>20
            speed=(0. if time<=.5+1e-12 else (-.06 if time>10+1e-12 or overload else
                    float(np.clip((2-load)*.001,-.004,.004)) if touched else .04))
            depth=float(np.clip(depth+speed*.02,0.,.32))
            self.assertEqual(c.command(time,load,.02),depth)
            old=np.array([0.,0.,depth,0.,0.,0.])
            new=np.array([*FixtureTouchScene.target_translation(None,depth),0.,0.,0.])
            np.testing.assert_array_equal(old,new)

    def test_saved_successful_snapshot_is_exact_and_never_calls_local_fallback(self):
        from .official_active3d import read_template_obj
        from .shape_fixture_assets import EXPERIMENT
        from .report_fixture_touch import snapshot_chart
        from .local_fixture_chart import snapshot_with_local_fallback
        from .collect_fixture_touch import FIELD_KEYS
        template,faces,_=read_template_obj(EXPERIMENT/'vendor/Active-3D-Vision-and-Touch/pterotactyl/objects/touch_chart.obj')
        folder=Path('experiments/object_predictor_v1/overfit_repair_v1/sugar_shape_fixed40/object_18704_direction_00')
        with np.load(folder/'observed_surface.npz') as f:
            a,b=f['offset'][499:501];surface={k:f[k][a:b] for k in FIELD_KEYS}
        with np.load(folder/'trace.npz') as t:pose=t['hand_pose_w'][499]
        args=(surface,pose,template.numpy(),faces.verts_idx.numpy())
        original,info=snapshot_chart(*args);self.assertTrue(info['available'])
        with patch('scripts.sugar.object_predictor.local_fixture_chart.supported_local_chart',side_effect=AssertionError('must not call')):
            actual,new_info=snapshot_with_local_fallback(*args)
        self.assertEqual(info,new_info)
        for k in original:np.testing.assert_array_equal(original[k],actual[k],err_msg=k)

    def test_saved_failed_snapshot_fallback_replays_measured_source_points(self):
        from .official_active3d import read_template_obj
        from .shape_fixture_assets import EXPERIMENT
        from .local_fixture_chart import snapshot_with_local_fallback
        from .collect_fixture_touch import FIELD_KEYS
        template,faces,_=read_template_obj(EXPERIMENT/'vendor/Active-3D-Vision-and-Touch/pterotactyl/objects/touch_chart.obj')
        folder=Path('experiments/object_predictor_v1/overfit_repair_v1/sugar_shape_fixed40/object_11898_direction_29')
        with np.load(folder/'observed_surface.npz') as f:
            a,b=f['offset'][499:501];surface={k:f[k][a:b] for k in FIELD_KEYS}
        with np.load(folder/'trace.npz') as t:pose=t['hand_pose_w'][499]
        arrays,info=snapshot_with_local_fallback(surface,pose,template.numpy(),faces.verts_idx.numpy())
        self.assertFalse(info['original_snapshot_failure']['available']);self.assertTrue(info['available'])
        replay=np.einsum('ni,nij->nj',arrays['barycentric'],surface['pos'][arrays['source_frame_indices']])
        np.testing.assert_allclose(replay,arrays['chart_hand_m'],atol=1e-12,rtol=0)
        self.assertTrue((arrays['source_frame_indices']>=0).all())
        self.assertLess(arrays['source_frame_indices'].max(),len(surface['pos']))


if __name__=='__main__':unittest.main()
