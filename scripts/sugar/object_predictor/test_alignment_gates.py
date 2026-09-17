"""CPU geometry checks for the two separately controlled alignment hypotheses."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .bilateral_common_alignment import BilateralCommonAlignmentController
from .coupled_alignment_surface_grip import CoupledAlignmentSurfaceGripController
from .fixed_path_surface_grip import FixedPathSurfaceGripController
from .freeze_postlift_alignment import FreezePostliftAlignmentController
from . import test_coupled_surface_grip as fixture


class AlignmentGateTests(unittest.TestCase):
    def pair(self, postlift=False, **kwargs):
        types = ((FixedPathSurfaceGripController, FreezePostliftAlignmentController) if postlift
                 else (CoupledAlignmentSurfaceGripController, BilateralCommonAlignmentController))
        result = [kind(target_load_n=16., response_gain=True, **kwargs) for kind in types]
        for controller in result:
            controller.support_normals = np.array([[1.,0.,0.],[-1.,0.,0.]])
        return result

    def observation(self, absent=None, **kwargs):
        poses, field = fixture.CoupledSurfaceGripTests().observation(**kwargs)
        if absent is not None:
            field['pressure'][field['patch']==absent] = 0.
        return poses, field

    def assert_original_record_equal(self, original, new):
        for key in original:
            np.testing.assert_array_equal(original[key], new[key], err_msg=key)

    def test_bilateral_common_alignment_is_exact_parent_when_both_valid(self):
        base, candidate = self.pair()
        poses, field = self.observation(angles=(20.,-30.))
        outputs=[]
        for controller in (base,candidate):
            controller.observe(poses,field)
            outputs.append(controller.command(12.,np.array([8.,20.]),.02))
        for i in (0,1): np.testing.assert_array_equal(outputs[0][i],outputs[1][i])
        self.assert_original_record_equal(outputs[0][2],outputs[1][2])
        self.assertFalse(outputs[1][2]['unilateral_alignment_suppressed'].any())
        self.assertTrue(outputs[1][2]['common_alignment_applied'].all())

    def test_unilateral_rotation_is_removed_about_pivot_preserving_closure_and_ready(self):
        base,candidate=self.pair()
        poses,field=self.observation(absent=0,angles=(0.,30.),centers=[[.01,.02,.03],[-.02,.04,.01]])
        outputs=[]
        for controller in (base,candidate):
            controller.observe(poses,field)
            outputs.append(controller.command(12.,np.array([8.,20.]),.02))
        a,va,ra=outputs[0];b,vb,rb=outputs[1]
        self.assert_original_record_equal(ra,rb)
        np.testing.assert_array_equal(rb['unilateral_alignment_suppressed'],[False,True])
        self.assertFalse(rb['actual_alignment_active'].any())
        # With no lift/yaw, the entire remaining displacement is original closure.
        expected=poses[:,:3]+rb['fit_normal_w']*rb['distance'][:,None]
        np.testing.assert_allclose(b[:,:3],expected,atol=1e-15)
        np.testing.assert_allclose(b[:,3:],poses[:,3:],atol=1e-15)
        np.testing.assert_allclose(vb[:,:3],(b[:,:3]-poses[:,:3])/.02,atol=1e-14)
        self.assertGreater(np.linalg.norm(a[1,3:]-b[1,3:]),0.)
        self.assertEqual(rb['ready_seconds'],0.)

    def test_no_geometry_preserves_parent_exactly(self):
        base,candidate=self.pair()
        poses,field=self.observation()
        field['pressure'][:]=0.
        outputs=[]
        for controller in (base,candidate):
            controller.observe(poses,field)
            outputs.append(controller.command(12.,np.zeros(2),.02))
        for i in (0,1):np.testing.assert_array_equal(outputs[0][i],outputs[1][i])
        self.assert_original_record_equal(outputs[0][2],outputs[1][2])
        self.assertFalse(outputs[1][2]['unilateral_alignment_suppressed'].any())

    def test_postlift_freeze_preserves_prestart_and_admission_step(self):
        base,candidate=self.pair(postlift=True)
        poses,field=self.observation(angles=(3.,-4.))
        for controller in (base,candidate):controller.ready_seconds=.98
        outputs=[]
        for controller in (base,candidate):
            controller.observe(poses,field)
            outputs.append(controller.command(24.,np.array([16.,16.]),.02))
        for i in (0,1):np.testing.assert_array_equal(outputs[0][i],outputs[1][i])
        self.assert_original_record_equal(outputs[0][2],outputs[1][2])
        self.assertEqual(candidate.lift_start,24.)
        self.assertFalse(outputs[1][2]['postlift_alignment_frozen'])
        self.assertFalse(outputs[1][2]['postlift_alignment_suppressed'].any())

    def test_postlift_undo_preserves_yaw_lift_closure_and_completion(self):
        base,candidate=self.pair(postlift=True,yaw_delta_deg=35.,lateral_xy=(.03,.05))
        poses,field=self.observation(angles=(20.,-30.),centers=[[.02,.01,.01],[-.01,-.02,-.03]])
        outputs=[]
        for controller in (base,candidate):
            controller.lift_start=24.;controller.motion_elapsed=2.
            controller.previous_lift=.5;controller.previous_yaw=17.5
            controller.observe(poses,field)
            outputs.append(controller.command(27.,np.array([8.,20.]),.02))
        a,va,ra=outputs[0];b,vb,rb=outputs[1]
        self.assert_original_record_equal(ra,rb)
        self.assertTrue(rb['postlift_alignment_suppressed'].all())
        self.assertFalse(rb['actual_alignment_active'].any())
        turn=Rotation.from_euler('z',candidate.previous_yaw-17.5,degrees=True)
        expected=poses[:,:3]+rb['fit_normal_w']*rb['distance'][:,None]
        midpoint=poses[:,:3].mean(0)
        expected=turn.apply(expected-midpoint)+midpoint
        expected+=np.r_[candidate.lateral_xy,candidate.lift_height_m]*(candidate.previous_lift-.5)
        np.testing.assert_allclose(b[:,:3],expected,atol=1e-15)
        np.testing.assert_allclose(b[:,3:],(turn*Rotation.from_quat(poses[:,3:])).as_quat(),atol=1e-15)
        np.testing.assert_allclose(vb[:,:3],(b[:,:3]-poses[:,:3])/.02,atol=1e-14)
        self.assertFalse(rb['sustained_ready']);self.assertGreater(rb['motion_rate'],0.)
        candidate.motion_elapsed=4.;candidate.previous_lift=1.;candidate.previous_yaw=35.
        candidate.observe(poses,field)
        held,_,record=candidate.command(30.,np.array([8.,20.]),.02)
        np.testing.assert_allclose(held[:,3:],poses[:,3:],atol=1e-15)
        self.assertEqual(record['motion_rate'],0.);self.assertFalse(record['motion_paused'])

    def test_unilateral_rollback_is_rigid_frame_equivariant_and_ignores_gt(self):
        _,a=self.pair();_,b=self.pair()
        poses,field=self.observation(absent=0,angles=(0.,30.),centers=[[0.,0.,0.],[-.02,.04,.01]])
        turn=Rotation.from_euler('xyz',[27.,-31.,68.],degrees=True);translation=np.array([.2,-.1,.3])
        world=poses.copy();world[:,:3]=turn.apply(poses[:,:3])+translation
        world[:,3:]=(turn*Rotation.from_quat(poses[:,3:])).as_quat()
        a.observe(poses,field);b.observe(world,dict(field,object_pose_w=np.full(7,np.nan),object_mass_kg=-999))
        pa,va,ra=a.command(12.,np.array([8.,20.]),.02);pb,vb,rb=b.command(12.,np.array([8.,20.]),.02)
        self.assertEqual(set(b.surface),{'pos','area','pressure','pad','patch'})
        np.testing.assert_allclose(pb[:,:3],turn.apply(pa[:,:3])+translation,atol=1e-12)
        np.testing.assert_allclose(vb[:,:3],turn.apply(va[:,:3]),atol=1e-12)
        np.testing.assert_allclose(vb[:,3:],turn.apply(va[:,3:]),atol=1e-12)
        np.testing.assert_array_equal(rb['unilateral_alignment_suppressed'],ra['unilateral_alignment_suppressed'])


if __name__=='__main__':unittest.main()
