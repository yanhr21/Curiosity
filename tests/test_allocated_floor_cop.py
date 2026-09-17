import copy
import pickle
import unittest
from unittest.mock import patch
import numpy as np
from scripts.sugar.object_predictor.allocated_floor_cop import AllocatedFloorCoPController
from scripts.sugar.object_predictor.floor_safe_cop import FloorSafeCoPController
from scripts.sugar.object_predictor.test_coupled_surface_grip import CoupledSurfaceGripTests as Fixture


class AllocationTests(unittest.TestCase):
    def pair(self, misaligned=False, low=False):
        centers=np.array([[0.,0.,.08],[0.,0.,-.08]]) if misaligned else None
        poses,field=Fixture().observation(centers=centers)
        if low:poses[:,2]=.00002
        pair=[c(target_load_n=12.,force_gain=.000025,response_gain=True)
              for c in (FloorSafeCoPController,AllocatedFloorCoPController)]
        for c in pair:
            c.support_normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
            c.floor_vertices=(np.zeros((1,3)),np.zeros((1,3)))
            c.observe(poses,field)
        return *pair,poses,field

    def test_safe_full_command_and_parent_state_exact_across_admission(self):
        for misaligned in (False,True):
            old,new,poses,field=self.pair(misaligned)
            for i in range(80):
                for c in (old,new):c.observe(poses,field)
                a=old.command(24.+.02*i,np.full(2,12.),.02)
                b=new.command(24.+.02*i,np.full(2,12.),.02)
                np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
                for key in a[2]:
                    if key!='floor_executed_fraction':np.testing.assert_array_equal(a[2][key],b[2][key],err_msg=key)
                for key in old.__dict__:
                    self.assertEqual(pickle.dumps(old.__dict__[key]),pickle.dumps(new.__dict__[key]),key)
                poses=b[0]

    def test_downward_left_removed_parent_force_and_right_cop_committed(self):
        old,new,poses,_=self.pair(True,True)
        a=old.command(24.,np.full(2,8.),.02)
        out,velocity,r=new.command(24.,np.full(2,8.),.02)
        self.assertTrue(a[2]['floor_blocked']);self.assertFalse(r['floor_blocked'])
        np.testing.assert_array_equal(r['cop_allocation_selected_mask'],[False,True])
        np.testing.assert_array_equal(out[0],r['cop_allocation_parent_pose_w'][0])
        self.assertTrue((new.distance>0).all());self.assertEqual(new.cop_travel[0],0.)
        self.assertAlmostEqual(new.cop_travel[1],.00008)
        self.assertFalse(r['response_command_pure'].any())
        self.assertEqual(len(new.response.samples[0]),1)
        self.assertEqual(new.response.samples[0][0][2],0.)
        self.assertEqual(r['cop_allocation_changed_count'],1)

    def test_actual_per_hand_budget_not_rejected_travel(self):
        _,new,_,_=self.pair(True,True)
        new.cop_travel[:]=[.09999,.09997]
        _,_,r=new.command(24.,np.full(2,8.),.02)
        # Left requested10um remains safe; make its permitted geometry narrower.
        self.assertLessEqual(new.cop_travel.max(),.1)
        np.testing.assert_allclose(r['cop_allocation_proposed_step_m'][:,2],[-.00001,.00003],atol=1e-16)
        _,new,_,_=self.pair(True,True);new.cop_travel[1]=.09997
        _,_,r=new.command(24.,np.full(2,8.),.02)
        self.assertEqual(new.cop_travel[0],0.);self.assertAlmostEqual(new.cop_travel[1],.1)
        self.assertAlmostEqual(r['cop_servo_velocity_m_s'][1,2],.0015)

    def test_no_safe_combination_keeps_original_hold_and_response_once(self):
        old,new,poses,_=self.pair(True)
        with patch('scripts.sugar.object_predictor.floor_safe_cop.swept_min_height',side_effect=[.1,.1,-.1,.1]):
            a=old.command(24.,np.full(2,8.),.02)
        with patch('scripts.sugar.object_predictor.allocated_floor_cop.swept_min_height',side_effect=[.1,.1]+[-.1,.1]*4):
            b=new.command(24.,np.full(2,8.),.02)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        for k in old.__dict__:self.assertEqual(pickle.dumps(old.__dict__[k]),pickle.dumps(new.__dict__[k]),k)
        self.assertTrue(b[2]['floor_blocked']);self.assertEqual(new.ready_seconds,0.)

    def test_runtime_unsafe_does_not_enumerate_or_advance_phase(self):
        _,new,_,_=self.pair();new.lift_start=24.;new.motion_elapsed=2.;new.previous_lift=.5
        with patch('scripts.sugar.object_predictor.allocated_floor_cop.swept_min_height',side_effect=[.1,.1,-.1,.1]):
            _,_,r=new.command(27.,np.full(2,12.),.02)
        self.assertFalse(r['cop_allocation_attempted']);self.assertTrue(r['floor_blocked'])
        self.assertEqual(new.motion_elapsed,2.)
        np.testing.assert_array_equal(r['cop_allocation_evaluated'],[True,False,False,False])

    def test_safe_parent_only_zero_cop_commits_distance_not_travel(self):
        _,new,_,_=self.pair(True)
        with patch('scripts.sugar.object_predictor.allocated_floor_cop.swept_min_height',side_effect=[.1,.1]+[-.1,.1]*3+[.1,.1]):
            _,_,r=new.command(24.,np.full(2,8.),.02)
        self.assertFalse(r['floor_blocked']);self.assertTrue((new.distance>0).all())
        np.testing.assert_array_equal(new.cop_travel,0.)
        np.testing.assert_array_equal(r['cop_allocation_selected_mask'],[False,False])
        self.assertTrue(r['response_command_pure'].all())

    def test_largest_retained_step_and_fixed_mask_tie_order(self):
        for left_travel,expected in ((.09997,[False,True]),(0.,[True,False])):
            _,new,_,_=self.pair(True)
            new.cop_travel[0]=left_travel
            with patch('scripts.sugar.object_predictor.allocated_floor_cop.swept_min_height',side_effect=[.1,.1,-.1,.1]+[.1,.1]*3):
                _,_,r=new.command(24.,np.full(2,8.),.02)
            np.testing.assert_array_equal(r['cop_allocation_selected_mask'],expected)

    def test_initial_orientation_and_state_exact_no_object_inputs(self):
        a=FloorSafeCoPController(response_gain=True);b=AllocatedFloorCoPController(response_gain=True)
        outa=a.command(0.,np.zeros(2),.02);outb=b.command(0.,np.zeros(2),.02)
        np.testing.assert_array_equal(outa[0],outb[0]);np.testing.assert_array_equal(outa[1],outb[1])
        for k in a.__dict__:self.assertEqual(pickle.dumps(a.__dict__[k]),pickle.dumps(b.__dict__[k]),k)


if __name__=='__main__':unittest.main()
