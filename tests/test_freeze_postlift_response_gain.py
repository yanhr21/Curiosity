import unittest
import numpy as np

from scripts.sugar.object_predictor.freeze_postlift_alignment import FreezePostliftAlignmentController
from scripts.sugar.object_predictor.freeze_postlift_response_gain import FreezePostliftResponseGainController
from scripts.sugar.object_predictor import test_coupled_surface_grip as fixture


class FrozenResponseGainTests(unittest.TestCase):
    def controllers(self):
        result=[kind(target_load_n=16.,response_gain=True,yaw_delta_deg=25.,lateral_xy=(.02,.03))
                for kind in (FreezePostliftAlignmentController,FreezePostliftResponseGainController)]
        for c in result:c.support_normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
        return result

    def observation(self):
        return fixture.CoupledSurfaceGripTests().observation(angles=(0.,0.))

    def admit(self,c):
        pose,field=self.observation();c.ready_seconds=.98;c.observe(pose,field)
        return c.command(24.,np.array([16.,16.]),.02)

    def test_complete_preadmission_and_admission_are_bitexact_parent(self):
        base,new=self.controllers();pose,field=self.observation()
        for index in range(1200):
            time=(index+1)*.02
            loads=np.array([16.+.2*np.sin(index*.1),16.+.1*np.cos(index*.1)])
            results=[]
            for c in (base,new):c.observe(pose,field);results.append(c.command(time,loads,.02))
            for k in (0,1):np.testing.assert_array_equal(results[0][k],results[1][k])
            for k,value in results[0][2].items():np.testing.assert_array_equal(value,results[1][2][k],err_msg=k)
        self.assertEqual(new.lift_start,24.)
        np.testing.assert_array_equal(new.response.frozen,results[0][2]['response_gain_m_per_ns'])
        self.assertFalse(results[1][2]['response_applied_gain_frozen'])

    def test_frozen_values_are_actual_admission_not_hand_tuned_base(self):
        _,c=self.controllers();c.response.original.applied=np.array([.000031,.000067])
        result=self.admit(c)[2]
        np.testing.assert_array_equal(c.response.frozen,result['response_gain_m_per_ns'])
        self.assertFalse(np.array_equal(c.response.frozen,np.full(2,c.force_gain)))
        self.assertEqual(result['response_freeze_time_s'],24.)

    def test_live_force_error_changes_closure_while_candidate_gain_collapses(self):
        _,c=self.controllers();self.admit(c);frozen=c.response.frozen.copy()
        c.response.original.upper[:]=1e12
        old=c.distance.copy();pose,field=self.observation();c.observe(pose,field)
        _,_,record=c.command(24.02,np.array([4.,40.]),.02)
        np.testing.assert_array_equal(record['response_gain_m_per_ns'],frozen)
        np.testing.assert_allclose(c.distance-old,np.clip((16.-np.array([4.,40.]))*frozen,-.008,.004)*.02,atol=1e-16)
        self.assertTrue(np.all(record['response_diagnostic_candidate_gain_m_per_ns']<frozen))
        self.assertTrue(record['response_applied_gain_frozen'])
        self.assertFalse(record['actual_alignment_active'].any())
        self.assertEqual(record['motion_elapsed_s'],.02)
        self.assertEqual(record['ready_seconds'],0.)

    def test_fixed_path_and_freeze_persist_after_contact_loss_and_completion(self):
        _,c=self.controllers();self.admit(c);frozen=c.response.frozen.copy();pose,field=self.observation()
        field['pressure'][:]=0.
        for index in range(1,252):
            c.observe(pose,field);_,_,record=c.command(24.+index*.02,np.zeros(2),.02)
            np.testing.assert_array_equal(record['response_gain_m_per_ns'],frozen)
            self.assertEqual(record['ready_seconds'],0.)
        self.assertEqual(c.motion_elapsed,4.)
        self.assertFalse(record['actual_alignment_active'].any())
        self.assertTrue(record['response_applied_gain_frozen'])

    def test_no_ground_truth_access_and_parent_estimator_is_not_modified(self):
        base,c=self.controllers();self.admit(c);pose,field=self.observation()
        c.observe(pose,dict(field,object_pose_w=np.full(7,np.nan),object_mass_kg=-1.))
        self.assertEqual(set(c.surface),{'pos','area','pressure','pad','patch'})
        from scripts.sugar.object_predictor.response_gain import ObservedForceGain
        self.assertIs(type(base.response),ObservedForceGain)
        self.assertIs(type(c.response.original),ObservedForceGain)
        self.assertNotIn('frozen',vars(base.response))
        c.command(24.02,np.array([12.,20.]),.02)

    def test_response_gain_must_be_explicitly_enabled(self):
        with self.assertRaises(ValueError):FreezePostliftResponseGainController(response_gain=False)


if __name__=='__main__':unittest.main()
