import unittest
import copy
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.sugar.object_predictor.qualified_cop_response import (
    QualifiedObservedForceGain, QualifiedCoPResponseController, command_evidence, ROUND_OFF_TOL)
from scripts.sugar.object_predictor.response_gain import ObservedForceGain
from scripts.sugar.object_predictor.prelift_cop_alignment import PreliftCoPAlignmentController
from scripts.sugar.object_predictor.test_coupled_surface_grip import CoupledSurfaceGripTests as Fixture
from scripts.sugar.object_predictor import run_qualified_cop_response_pilot as pilot


class QualifiedResponseTests(unittest.TestCase):
    def test_pure_windows_are_original_estimator_exact(self):
        old=ObservedForceGain();new=QualifiedObservedForceGain()
        for i in range(150):
            new.incoming_pure[:]=True
            for side in (0,1):
                args=(side,(i+1)*.02,1+i*.1,i*.00001,.02,bool(i%3))
                self.assertEqual(old.update(*args),new.update(*args))
        self.assertGreater(new.accepted_total.min(),0)

    def test_complete_five_intervals_and_causal_recovery(self):
        new=QualifiedObservedForceGain();seen=[]
        for i in range(15):
            new.incoming_pure[:]=i!=7
            new.update(0,(i+1)*.02,1+i*.1,i*.00001,.02,True)
            seen.append(bool(new.accepted[0]))
        self.assertTrue(seen[6]);self.assertEqual(seen[7:12],[False]*5);self.assertTrue(seen[12])

    def test_all_mixed_never_identifies_and_preserves_original_base(self):
        new=QualifiedObservedForceGain()
        for i in range(80):
            result=new.update(0,(i+1)*.02,1+i*.5,i*.000002,.02,True)
            self.assertEqual(result,(.000025,0.,0))
        self.assertEqual(new.accepted_total[0],0);self.assertEqual(new.rejected_total[0],75)

    def test_either_hand_rotation_cop_or_path_excludes_both_sides(self):
        poses=np.zeros((2,7));poses[:,6]=1
        rec=dict(cop_servo_velocity_m_s=np.zeros((2,3)),motion_elapsed_s=0.)
        self.assertTrue(command_evidence(poses,poses,rec,0.,.02)[-1].all())
        moved=poses.copy();moved[0,3:]=Rotation.from_rotvec([1e-10,0,0]).as_quat()
        self.assertFalse(command_evidence(poses,moved,rec,0.,.02)[-1].any())
        rec['cop_servo_velocity_m_s'][1,0]=1e-10
        self.assertFalse(command_evidence(poses,poses,rec,0.,.02)[-1].any())
        rec['cop_servo_velocity_m_s'][:]=0.;rec['motion_elapsed_s']=.02
        self.assertFalse(command_evidence(poses,poses,rec,0.,.02)[-1].any())
        self.assertLess(ROUND_OFF_TOL,1e-12)

    def test_parent_path_and_cop_unchanged_before_first_secant(self):
        poses,field=Fixture().observation(centers=np.array([[0.,0.,-.08],[0.,0.,.08]]))
        old=PreliftCoPAlignmentController(target_load_n=16.,response_gain=True)
        new=QualifiedCoPResponseController(target_load_n=16.,response_gain=True)
        for c in (old,new):
            c.support_normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
            c.observe(poses,field)
        for i in range(5):
            a=old.command(24+i*.02,np.full(2,16.),.02)
            b=new.command(24+i*.02,np.full(2,16.),.02)
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
            for key in a[2]:np.testing.assert_array_equal(a[2][key],b[2][key])
            self.assertFalse(b[2]['response_command_pure'].any())
            self.assertFalse(b[2]['response_window_pure'].any())

    def test_mixed_context_invalidates_upper_and_falls_back_without_gain_jump(self):
        new=QualifiedObservedForceGain()
        for i in range(15):
            new.incoming_pure[:]=True
            new.update(0,(i+1)*.02,1+i*.1,i*.000002,.02,True)
        upper=float(new.upper[0]);self.assertGreater(upper,0)
        prior=float(new.applied[0])
        for i in range(15,150):
            new.incoming_pure[:]=False
            new.update(0,(i+1)*.02,1.,0.,.02,True)
            self.assertEqual(new.upper[0],0.)
            self.assertLessEqual(new.applied[0],prior*1.05+1e-20)
            prior=float(new.applied[0])
        self.assertEqual(len(new.slopes[0]),0);self.assertEqual(new.applied[0],new.base_gain)

    def test_expired_pure_samples_cannot_keep_stale_upper(self):
        new=QualifiedObservedForceGain();new.incoming_pure[:]=True
        for i in range(15):new.update(0,(i+1)*.02,1+i*.1,i*.000002,.02,True)
        self.assertGreater(new.upper[0],0.)
        for i in range(15,150):new.update(0,(i+1)*.02,1.,0.,.02,True)
        self.assertEqual(len(new.slopes[0]),0);self.assertEqual(new.upper[0],0.)

    def test_fixed_negative_baseline_and_original_budgets(self):
        p=pilot.protocol()
        self.assertEqual(p['baseline_intervention'],'prelift_cop_alignment_v1')
        self.assertEqual([c['episode'] for c in p['configurations']],[5014,5012,5000])
        self.assertEqual(p['controls_per_case'],2400);self.assertEqual(len(p['original_physical_checks']),12)
        self.assertEqual(p['cop_rule']['per_hand_accumulated_travel_max_m'],.10)
        self.assertEqual(p['cop_rule']['active_clock_s'],[1.,40.])
        self.assertEqual(p['cop_rule']['admission_angle_max_deg'],5.)
        self.assertFalse(p['automatic_expansion'])

    def test_saved_response_replay_and_forged_window_rejected(self):
        poses,field=Fixture().observation()
        c=QualifiedCoPResponseController(target_load_n=12.,force_gain=.000025,response_gain=True)
        c.support_normals=np.array([[1.,0.,0.],[-1.,0.,0.]])
        loads=np.array([(field['area'][field['patch']==h]*field['pressure'][field['patch']==h]).sum() for h in (0,1)])
        records=[]
        for i in range(30):
            c.observe(poses,field if i else {k:v[:0] for k,v in field.items()})
            records.append(c.command((i+1)*.02,loads if i else np.zeros(2),.02)[2])
        a=dict(timestamp_s=.02*np.arange(1,31),hand_pose_w=np.broadcast_to(poses,(30,2,7)).copy())
        keys=(*pilot.TELEMETRY,'distance_m','motion_elapsed_s','fit_valid','response_gain_m_per_ns','response_upper_n_m','response_samples')
        for key in keys:a['validation_controller_'+key]=np.array([r['distance' if key=='distance_m' else key] for r in records])
        s=dict(offset=np.arange(31)*len(field['pad']))
        for name,key in [('pad','pad'),('area_m2','area'),('normal_pressure_pa','pressure')]:
            s[name]=np.concatenate([field[key]]*30)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);np.savez_compressed(folder/'contact_surface.npz',**s)
            self.assertTrue(pilot.response_readback(a,folder)['passed'])
            forged=copy.deepcopy(a);forged['validation_controller_response_window_pure'][12]=~forged['validation_controller_response_window_pure'][12]
            with self.assertRaises(ValueError):pilot.response_readback(forged,folder)


if __name__=='__main__':unittest.main()
