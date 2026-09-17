"""CPU-only region-support intervention checks; no physical success claim."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from .bilateral_common_alignment import BilateralCommonAlignmentController
from .regional_bilateral_alignment import RegionalBilateralAlignmentController, region_geometry_view
from .test_coupled_surface_grip import CoupledSurfaceGripTests


def append_region(field, points, pad=1, side=0, area=1e-6, pressure=1e6):
    n = len(points)
    extra = dict(pos=np.asarray(points), area=np.full(n, area), pressure=np.full(n, pressure),
                 pad=np.full(n, pad), patch=np.full(n, side))
    return {k: np.concatenate((v, extra[k])) for k, v in field.items()}


class RegionalAlignmentTests(unittest.TestCase):
    def controllers(self):
        pair = [c(target_load_n=16., response_gain=True, yaw_delta_deg=35., lateral_xy=(.03,.05))
                for c in (BilateralCommonAlignmentController, RegionalBilateralAlignmentController)]
        for c in pair:
            c.support_normals = np.array([[1.,0.,0.],[-1.,0.,0.]])
        return pair

    def observation(self, **kwargs):
        return CoupledSurfaceGripTests().observation(**kwargs)

    def test_single_region_is_exact_parent_and_original_thresholds_hold(self):
        poses, field = self.observation(angles=(20.,-30.))
        base, candidate = self.controllers()
        out=[]
        for c in (base,candidate):
            c.observe(poses,field);out.append(c.command(12.,np.array([8.,20.]),.02))
        for i in (0,1):np.testing.assert_array_equal(out[0][i],out[1][i])
        for k,v in out[0][2].items():np.testing.assert_array_equal(v,out[1][2][k],err_msg=k)
        np.testing.assert_array_equal(out[1][2]['regional_selected_pad'],[0,27])
        # Original count, second scale and thickness requirements all still apply.
        for mode in ('five','line','thick'):
            test={k:v.copy() for k,v in field.items()}
            if mode=='five':test['pressure'][5:9]=0.
            if mode=='line':test['pos'][:9,2]=0.
            if mode=='thick':
                test['pos'][:9]=np.array([[x,y,z] for x in (-.002,.002)
                                        for y in (-.002,.002) for z in (-.002,.002)]+[[0.,0.,0.]])
            _, record=region_geometry_view(test)
            self.assertEqual(record['regional_selected_pad'][0],-1,mode)

    def test_isolated_remote_pad_cannot_change_selected_normal_or_pivot(self):
        poses, field=self.observation(angles=(20.,-30.))
        # Huge geometric leverage/area, tiny pressure; one point remains invalid.
        dirty=append_region(field,[[.8,.6,.9]],pad=20,area=1.,pressure=1e-8)
        clean,_=region_geometry_view(field);view,record=region_geometry_view(dirty)
        for k in clean:np.testing.assert_array_equal(clean[k],view[k])
        self.assertEqual(record['regional_candidate_count'][20],1)
        self.assertFalse(record['regional_candidate_valid'][20])
        a,b=self.controllers()
        a.observe(poses,field);b.observe(poses,dirty)
        pa,va,ra=a.command(12.,np.array([8.,20.]),.02)
        pb,vb,rb=b.command(12.,np.array([8.,20.]),.02)
        np.testing.assert_array_equal(pa,pb);np.testing.assert_array_equal(va,vb)
        np.testing.assert_array_equal(ra['fit_normal_w'],rb['fit_normal_w'])
        self.assertGreater(rb['contact_area_m2'][0],ra['contact_area_m2'][0])

    def test_selected_view_controls_both_normal_and_pivot_with_yaw_and_lift(self):
        poses,field=self.observation(angles=(3.,-4.),centers=[[.02,.01,.01],[-.01,-.02,-.03]])
        extra=Rotation.from_euler('z',40,degrees=True).apply(field['pos'][:9])+[.05,.12,.08]
        dirty=append_region(field,extra,pad=1,area=.5e-6)
        base,candidate=self.controllers();outputs=[]
        for c,obs in ((base,field),(candidate,dirty)):
            c.lift_start=24.;c.ready_seconds=1.2;c.motion_elapsed=2.
            c.previous_lift=.5;c.previous_yaw=17.5
            c.observe(poses,obs);outputs.append(c.command(27.,np.array([14.,18.]),.02))
        for i in (0,1):np.testing.assert_array_equal(outputs[0][i],outputs[1][i])
        np.testing.assert_allclose(outputs[1][2]['regional_selected_center_hand_m'][0],[.02,.01,.01],atol=1e-15)
        np.testing.assert_array_equal(outputs[0][2]['fit_normal_w'],outputs[1][2]['fit_normal_w'])
        self.assertTrue(outputs[1][2]['actual_alignment_active'].all())

    def test_full_observation_and_all_hand_loads_survive_success_and_exception(self):
        poses,field=self.observation()
        dirty=append_region(field,[[.2,.3,.4]],pad=20)
        _,candidate=self.controllers();candidate.observe(poses,dict(dirty,object_pose=np.full(7,np.nan)))
        original=candidate.surface;copies={k:v.copy() for k,v in original.items()}
        loads=np.array([4.,24.]);received=[]
        real=BilateralCommonAlignmentController.command
        def inspect(controller,time,actual,dt):
            self.assertIs(actual,loads)
            self.assertEqual(len(controller.surface['pos']),18)
            received.append(True)
            return real(controller,time,actual,dt)
        with patch.object(BilateralCommonAlignmentController,'command',inspect):
            _,_,record=candidate.command(12.,loads,.02)
        self.assertEqual(received,[True]);self.assertIs(candidate.surface,original)
        np.testing.assert_array_equal(record['regional_input_hand_load_n'],loads)
        self.assertEqual(set(original),{'pos','area','pressure','pad','patch'})
        for k in copies:np.testing.assert_array_equal(original[k],copies[k])
        with patch.object(BilateralCommonAlignmentController,'command',side_effect=RuntimeError('sentinel')):
            with self.assertRaisesRegex(RuntimeError,'sentinel'):candidate.command(12.02,loads,.02)
        self.assertIs(candidate.surface,original)
        for k in copies:np.testing.assert_array_equal(original[k],copies[k])

    def test_area_tie_switch_and_valid_to_invalid_are_recorded(self):
        poses,field=self.observation()
        field=append_region(field,field['pos'][:9]+[.03,.1,0],pad=1)
        _,c=self.controllers()
        c.observe(poses,field);_,_,r=c.command(12.,np.array([16.,16.]),.02)
        np.testing.assert_array_equal(r['regional_selected_pad'],[0,27])
        np.testing.assert_array_equal(r['regional_selection_changed'],[True,True])
        c.observe(poses,field);_,_,r=c.command(12.02,np.array([16.,16.]),.02)
        self.assertFalse(r['regional_selection_changed'].any())
        field['area'][field['pad']==1]*=2
        c.observe(poses,field);_,_,r=c.command(12.04,np.array([16.,16.]),.02)
        np.testing.assert_array_equal(r['regional_selected_pad'],[1,27])
        np.testing.assert_array_equal(r['regional_selection_changed'],[True,False])
        field['pressure'][field['patch']==0]=0
        c.observe(poses,field);_,_,r=c.command(12.06,np.array([16.,16.]),.02)
        np.testing.assert_array_equal(r['regional_selected_pad'],[-1,27])
        np.testing.assert_array_equal(r['regional_selection_changed'],[True,False])
        self.assertFalse(r['actual_alignment_active'].any())
        self.assertFalse(r['fit_valid'][0]);self.assertEqual(r['ready_seconds'],0.)

    def test_region_selection_and_commands_are_rigid_world_equivariant(self):
        poses,field=self.observation(angles=(20.,-30.),centers=[[.02,.01,.01],[-.01,-.02,-.03]])
        field=append_region(field,[[1.,2.,3.]],pad=20)
        turn=Rotation.from_euler('xyz',[27.,-31.,68.],degrees=True);shift=np.array([.2,-.1,.3])
        moved=poses.copy();moved[:,:3]=turn.apply(poses[:,:3])+shift
        moved[:,3:]=(turn*Rotation.from_quat(poses[:,3:])).as_quat()
        _,a=self.controllers();_,b=self.controllers()
        a.observe(poses,field);b.observe(moved,field)
        pa,va,ra=a.command(12.,np.array([8.,20.]),.02)
        pb,vb,rb=b.command(12.,np.array([8.,20.]),.02)
        np.testing.assert_array_equal(ra['regional_selected_pad'],rb['regional_selected_pad'])
        np.testing.assert_allclose(pb[:,:3],turn.apply(pa[:,:3])+shift,atol=1e-12)
        np.testing.assert_allclose(vb[:,:3],turn.apply(va[:,:3]),atol=1e-12)
        np.testing.assert_allclose(vb[:,3:],turn.apply(va[:,3:]),atol=1e-12)


if __name__=='__main__':unittest.main()
