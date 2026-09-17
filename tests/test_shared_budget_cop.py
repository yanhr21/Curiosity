import pickle
import unittest
from unittest.mock import patch
import numpy as np
from scripts.sugar.object_predictor.shared_budget_cop import SharedBudgetCoPController, budget_steps
from scripts.sugar.object_predictor.allocated_floor_cop import AllocatedFloorCoPController
from scripts.sugar.object_predictor.test_coupled_surface_grip import CoupledSurfaceGripTests as Fixture


class SharedBudgetTests(unittest.TestCase):
    def pair(self,low=False,misaligned=True):
        centers=np.array([[0.,0.,.08],[0.,0.,-.08]]) if misaligned else None
        poses,field=Fixture().observation(centers=centers)
        if low:poses[:,2]=.00002
        result=[c(target_load_n=12.,force_gain=.000025,response_gain=True)
                for c in (AllocatedFloorCoPController,SharedBudgetCoPController)]
        for c in result:
            c.support_normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
            c.floor_vertices=(np.zeros((1,3)),np.zeros((1,3)))
            c.observe(poses,field)
        return *result,poses,field

    def test_no_borrow_complete_parent_state_and_records_exact(self):
        for low,misaligned in ((False,False),(False,True),(True,True)):
            a,b,poses,field=self.pair(low,misaligned)
            for i in range(80):
                for c in (a,b):c.observe(poses,field)
                aa=a.command(24.+i*.02,np.full(2,12.),.02)
                bb=b.command(24.+i*.02,np.full(2,12.),.02)
                np.testing.assert_array_equal(aa[0],bb[0]);np.testing.assert_array_equal(aa[1],bb[1])
                for key in aa[2]:np.testing.assert_array_equal(aa[2][key],bb[2][key],err_msg=key)
                for key in a.__dict__:self.assertEqual(pickle.dumps(a.__dict__[key]),pickle.dumps(b.__dict__[key]),key)
                self.assertFalse(bb[2]['shared_borrow_attempted']);poses=bb[0]

    def test_only_floor_cancelled_donor_allows_quota_limited_recipient(self):
        for low,expect in ((True,True),(False,False)):
            old,new,_,_=self.pair(low);old.cop_travel[:]=new.cop_travel[:]=[.0944,.1]
            a=old.command(36.58,np.full(2,8.),.02)
            b=new.command(36.58,np.full(2,8.),.02);r=b[2]
            self.assertEqual(r['shared_borrow_attempted'],expect)
            np.testing.assert_array_equal(r['shared_borrow_applied'],[False,expect])
            if expect:
                self.assertAlmostEqual(new.cop_travel[1],.10008)
                self.assertEqual(new.cop_travel[0],.0944)
                np.testing.assert_array_equal(new.distance,old.distance)
                self.assertEqual(len(new.response.samples[0]),1)
                np.testing.assert_array_equal(r['cop_allocation_selected_mask'],[False,True])
                self.assertFalse(r['response_command_pure'].any())
            else:np.testing.assert_array_equal(a[0],b[0])

    def test_oblique_nonborrow_force_and_travel_state_exact(self):
        poses,field=Fixture().observation(centers=np.array([[0.,.037,.083],[0.,-.037,-.083]]))
        old,new,_,_=self.pair(False)
        for c in (old,new):c.observe(poses,field)
        for k in range(20):
            aa=old.command(24.+k*.02,np.array([8.31,9.47]),.02)
            bb=new.command(24.+k*.02,np.array([8.31,9.47]),.02)
            np.testing.assert_array_equal(aa[0],bb[0]);np.testing.assert_array_equal(aa[1],bb[1])
            for name in old.__dict__:self.assertEqual(pickle.dumps(old.__dict__[name]),pickle.dumps(new.__dict__[name]),name)
            for c in (old,new):c.observe(bb[0],field)

    def test_unlimited_recipient_no_borrow_and_deadline_unchanged(self):
        for time,travel in ((36.58,[.0944,.09]),(40.02,[.0944,.1])):
            old,new,_,_=self.pair(True);old.cop_travel[:]=new.cop_travel[:]=travel
            aa=old.command(time,np.full(2,8.),.02);bb=new.command(time,np.full(2,8.),.02)
            np.testing.assert_array_equal(aa[0],bb[0]);self.assertFalse(bb[2]['shared_borrow_attempted'])

    def test_total_budget_hard_cap_after_prior_borrow(self):
        _,new,poses,field=self.pair(True);new.cop_travel[:]=[.0944,.10557]
        _,_,r=new.command(36.58,np.full(2,8.),.02)
        self.assertLessEqual(new.cop_travel.sum(),.2)
        self.assertAlmostEqual(new.cop_travel.sum(),.2)
        self.assertAlmostEqual(r['cop_servo_velocity_m_s'][1,2],.0015,places=10)
        self.assertTrue(r['shared_total_limited']);self.assertAlmostEqual(new.cop_travel[0],.0944)
        # Once the recipient used the donated quota the donor cannot later
        # consume its old100mm allowance and exceed the conserved total.
        new.observe(poses,field)
        _,_,r=new.command(36.60,np.full(2,8.),.02)
        self.assertLessEqual(new.cop_travel.sum(),.2)
        np.testing.assert_allclose(r['cop_servo_velocity_m_s'],0.,atol=1e-12,rtol=0)

    def test_roundoff_never_inflates_total200mm(self):
        rng=np.random.default_rng(3)
        for _ in range(200):
            travel=np.array([.0944,.10559999999999])
            steps=rng.normal(size=(2,3));steps*=.00008/np.linalg.norm(steps,axis=1)[:,None]
            actual=budget_steps(steps,np.ones(2,bool),travel)
            self.assertLessEqual(float((travel+np.linalg.norm(actual,axis=1)).sum()),.2)
            self.assertTrue((np.linalg.norm(actual,axis=1)<=.00008).all())

    def test_no_safe_combination_retains_hold_and_single_observation(self):
        old,new,poses,_=self.pair()
        kwargs=dict(side_effect=[.1,.1]+[-.1,.1]*4)
        with patch('scripts.sugar.object_predictor.allocated_floor_cop.swept_min_height',**kwargs):
            aa=old.command(24.,np.full(2,8.),.02)
        with patch('scripts.sugar.object_predictor.shared_budget_cop.swept_min_height',**kwargs):
            bb=new.command(24.,np.full(2,8.),.02)
        np.testing.assert_array_equal(aa[0],bb[0]);self.assertTrue(bb[2]['floor_blocked'])
        for key in old.__dict__:self.assertEqual(pickle.dumps(old.__dict__[key]),pickle.dumps(new.__dict__[key]),key)

    def test_postlift_does_not_borrow_or_enumerate(self):
        _,new,_,_=self.pair(True);new.lift_start=24.;new.motion_elapsed=2.;new.previous_lift=.5
        new.cop_travel[:]=[.0944,.1]
        with patch('scripts.sugar.object_predictor.shared_budget_cop.swept_min_height',side_effect=[.1,.1,-.1,.1]):
            _,_,r=new.command(27.,np.full(2,12.),.02)
        self.assertFalse(r['shared_borrow_attempted']);self.assertTrue(r['floor_blocked'])
        np.testing.assert_array_equal(r['shared_baseline_evaluated'],[True,False,False,False])
        self.assertEqual(new.motion_elapsed,2.)


if __name__=='__main__':unittest.main()
