import copy
import pickle
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.sugar.object_predictor.coordinated_rotation_cop import (
    coordinate_rotation,predict_transport,CoordinatedRotationCoPController)
from scripts.sugar.object_predictor.shared_budget_cop import SharedBudgetCoPController


def observation(tilt_deg=0.):
    poses=np.array([[-.2,0.,.2,0.,0.,0.,1.],[.2,0.,.2,0.,0.,0.,1.]])
    centers=np.array([[0.,0.,.04],[0.,0.,-.04]])
    tilt=Rotation.from_euler('y',-tilt_deg,degrees=True)
    offsets=np.array([[0,y,z] for y in (-.01,0.,.01) for z in (-.01,.01)])
    points=np.concatenate([tilt.apply(offsets)+c for c in centers])
    field=dict(pos=points,area=np.ones(12)*1e-5,pressure=np.ones(12)*100,
        patch=np.repeat([0,1],6),pad=np.repeat([0,27],6))
    return poses,field,centers


def fixture():
    old,field,pivots=observation()
    norms=np.array([[1.,0.,0.],[-1.,0.,0.]])
    turn=Rotation.from_euler('y',-.02,degrees=True)
    target=old.copy();target[:,3:]=turn.as_quat()
    target[:,:3]+=pivots-turn.apply(pivots)
    target[:,:3]+=np.array([[.00001,0.,-.00008],[-.00001,0.,.00008]])
    record=dict(actual_alignment_active=np.ones(2,bool),floor_executed_swept_min_z_m=np.full(2,.19),
        lift_start_s=-1.,floor_blocked=False,cop_valid=np.ones(2,bool),cop_admission_ready=False,
        motion_elapsed_s=0.,phase=0,cop_world_m=old[:,:3]+pivots,fit_normal_w=norms)
    vertices=(np.zeros((1,3)),np.zeros((1,3)))
    return old,target,record,field,norms,vertices


class CoordinatedRotationTests(unittest.TestCase):
    def test_select_minimum_one_side_and_preserve_area_pivot_closure_cop(self):
        old,target,record,field,norms,vertices=fixture()
        q,d=coordinate_rotation(old,target,record,field,norms,vertices,prelift=True,time=36.)
        self.assertTrue(d['rotation_coord_trigger']);self.assertEqual(d['rotation_coord_selected_suppressed'].sum(),1)
        for h in (0,1):
            c=d['rotation_coord_pivot_hand_m'][h]
            np.testing.assert_allclose(Rotation.from_quat(q[h,3:]).apply(c)+q[h,:3],
                Rotation.from_quat(target[h,3:]).apply(c)+target[h,:3],rtol=0,atol=1e-16)
        self.assertLess(predict_transport(old,q,record['cop_world_m'],norms)[0],d['rotation_coord_original_predicted_angle_deg'])

    def test_helpful_rotation_keeps_exact_parent_object(self):
        old,target,r,field,norms,vs=fixture();target[:,3:]=Rotation.from_euler('y',.02,degrees=True).as_quat()
        q,d=coordinate_rotation(old,target,r,field,norms,vs,prelift=True,time=36.)
        self.assertIs(q,target);self.assertFalse(d['rotation_coord_trigger'])

    def test_no_safe_candidate_does_not_replace_safe_parent(self):
        args=fixture()
        with patch('scripts.sugar.object_predictor.coordinated_rotation_cop.swept_min_height',return_value=-1e-4):
            q,d=coordinate_rotation(*args,prelift=True,time=36.)
        self.assertIs(q,args[1]);self.assertFalse(d['rotation_coord_candidate_eligible'].any())

    def test_original_palm_fit_cone_is_not_relaxed(self):
        old,target,r,f,n,v=fixture();supports=Rotation.from_euler('y',10,degrees=True).apply(n)
        q,d=coordinate_rotation(old,target,r,f,supports,v,prelift=True,time=36.)
        self.assertIs(q,target);self.assertTrue((d['rotation_coord_candidate_hand_fit_angle_deg']>5.).any())

    def test_postlift_aligned_deadline_blocked_invalid_keep_exact(self):
        for mode in ('postlift','aligned','late','blocked','invalid','inactive'):
            old,target,r,f,n,v=fixture();pre=True;t=36.
            if mode=='postlift':pre=False;r['lift_start_s']=30.
            if mode=='aligned':r['cop_admission_ready']=True
            if mode=='late':t=40.02
            if mode=='blocked':r['floor_blocked']=True
            if mode=='invalid':r['cop_valid'][0]=False
            if mode=='inactive':r['actual_alignment_active'][:]=False
            q,d=coordinate_rotation(old,target,r,f,n,v,prelift=pre,time=t)
            self.assertIs(q,target);self.assertFalse(d['rotation_coord_trigger'])

    def test_only_active_alignment_can_be_suppressed(self):
        old,target,r,f,n,v=fixture();r['actual_alignment_active'][0]=False
        q,d=coordinate_rotation(old,target,r,f,n,v,prelift=True,time=36.)
        np.testing.assert_array_equal(d['rotation_coord_candidate_evaluated'],[False,True,False])
        np.testing.assert_array_equal(d['rotation_coord_selected_suppressed'],[False,True])

    def pair(self,tilt):
        poses,f,_=observation(tilt)
        old,new=[c(target_load_n=12.,force_gain=.000025,response_gain=True) for c in
                 (SharedBudgetCoPController,CoordinatedRotationCoPController)]
        for c in (old,new):
            c.support_normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
            c.floor_vertices=(np.zeros((1,3)),np.zeros((1,3)))
            c.observe(poses,f)
        return old,new,poses,f

    def test_untriggered_real_parent_outputs_and_state_exact(self):
        old,new,p,f=self.pair(0.)
        for k in range(4):
            a=old.command(24.+k*.02,np.array([9.,10.]),.02);b=new.command(24.+k*.02,np.array([9.,10.]),.02)
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
            for key in a[2]:np.testing.assert_array_equal(a[2][key],b[2][key],err_msg=key)
            for key in old.__dict__:self.assertEqual(pickle.dumps(old.__dict__[key]),pickle.dumps(new.__dict__[key]),key)
            for c in (old,new):c.observe(b[0],f)

    def test_real_suppression_response_once_actual_rotation_and_no_fake_ready(self):
        old,new,p,f=self.pair(2.)
        a=old.command(36.,np.array([9.,10.]),.02);b=new.command(36.,np.array([9.,10.]),.02);r=b[2]
        self.assertTrue(r['rotation_coord_selected_suppressed'].any())
        np.testing.assert_array_equal(old.distance,new.distance);np.testing.assert_array_equal(old.cop_travel,new.cop_travel)
        self.assertEqual(len(new.response.samples[0]),1);self.assertEqual(new.lift_start,-1.)
        self.assertLessEqual(new.ready_seconds,0.);self.assertEqual(new.motion_elapsed,0.)
        np.testing.assert_array_equal(r['response_command_target_quat_xyzw'],b[0][:,3:])
        expected=(Rotation.from_quat(b[0][:,3:])*Rotation.from_quat(p[:,3:]).inv()).magnitude()
        np.testing.assert_array_equal(r['response_command_rotation_rad'],expected)
        np.testing.assert_array_equal(new.response.incoming_pure,r['response_command_pure'])
        self.assertFalse(r['response_command_pure'].any())
        for h in np.flatnonzero(r['rotation_coord_selected_suppressed']):self.assertFalse(r['actual_alignment_active'][h])
        np.testing.assert_array_equal(r['rotation_coord_parent_target_pose_w'],a[0])

    def test_no_gt_inputs_and_field_not_mutated(self):
        args=fixture();before=pickle.dumps(args)
        coordinate_rotation(*args,prelift=True,time=36.)
        self.assertEqual(before,pickle.dumps(args))


if __name__=='__main__':unittest.main()
