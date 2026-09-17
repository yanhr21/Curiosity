import ast
import copy
import pickle
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.sugar.object_predictor import common_up_cop_alignment as adapter
from scripts.sugar.object_predictor import run_common_up_cop_pilot as pilot
from scripts.sugar.object_predictor.qualify_common_up_geometry import initial_poses,qualify
from scripts.sugar.object_predictor.qualified_cop_response import QualifiedCoPResponseController


class CommonUpTests(unittest.TestCase):
    def test_all_fixed_full_mesh_geometry_and_identity(self):
        for c in pilot.protocol()['configurations']:
            result=qualify(c);self.assertTrue(result['passed'])
            for row in result['hands']:
                self.assertGreater(row['min_hand_z_m'],.048)
                self.assertGreater(row['object_separating_gap_m'],.10)
            old,op=initial_poses(c,QualifiedCoPResponseController);new,np_=initial_poses(c)
            np.testing.assert_array_equal(op[1],np_[1])
            delta=Rotation.from_quat(np_[0,3:])*Rotation.from_quat(op[0,3:]).inv()
            self.assertAlmostEqual(delta.magnitude(),np.pi,places=12)
            np.testing.assert_array_equal(old.support_normals,new.support_normals)

    def test_fresh_constructed_state_is_parent_exact_except_frames(self):
        kwargs=dict(target_load_n=12.,response_gain=True,force_gain=.000025,controller_revision='sensor_feedback_v2',approach_angle_deg=90.)
        old=QualifiedCoPResponseController(**kwargs);new=adapter.CommonUpCoPController(**kwargs)
        self.assertEqual(set(old.__dict__),set(new.__dict__))
        for key in old.__dict__:
            if key!='frames':self.assertEqual(pickle.dumps(old.__dict__[key]),pickle.dumps(new.__dict__[key]),key)

    def test_world_yaw_equivariance_and_object_label_independence(self):
        c=pilot.protocol()['configurations'][0];d=copy.deepcopy(c)
        d.update(mass=9.,scale=[2.,.3,4.],seed=1234)
        np.testing.assert_array_equal(initial_poses(c)[1],initial_poses(d)[1])
        d=copy.deepcopy(c);d['approach_angle_deg']+=37.
        a=initial_poses(c)[1];b=initial_poses(d)[1];yaw=Rotation.from_euler('z',37,degrees=True)
        np.testing.assert_allclose(b[:,:3],yaw.apply(a[:,:3]),rtol=0,atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(b[:,3:]).as_matrix(),(yaw*Rotation.from_quat(a[:,3:])).as_matrix(),rtol=0,atol=1e-12)

    def test_same_observation_feedback_is_parent_exact_and_no_first_roll_jump(self):
        cfg=pilot.protocol()['configurations'][0]
        old,_=initial_poses(cfg,QualifiedCoPResponseController);new,poses=initial_poses(cfg)
        field=dict(pos=np.empty((0,3)),area=np.empty(0),pressure=np.empty(0),pad=np.empty(0,int),patch=np.empty(0,int))
        for k in range(80):
            for c in (old,new):c.observe(poses,field)
            a=old.command((k+1)*.02,np.zeros(2),.02);b=new.command((k+1)*.02,np.zeros(2),.02)
            for x,y in zip(a[:2],b[:2]):np.testing.assert_array_equal(x,y)
            for key in a[2]:np.testing.assert_array_equal(a[2][key],b[2][key])
            np.testing.assert_allclose((Rotation.from_quat(b[0][:,3:])*Rotation.from_quat(poses[:,3:]).inv()).magnitude(),0,atol=1e-14)
            poses=b[0]

    def test_scene_initializes_only_hand_joints_before_fk(self):
        config=pilot.protocol()['configurations'][0]
        c=adapter.CommonUpCoPController(response_gain=True,controller_revision='sensor_feedback_v2')
        before=pickle.dumps(c.__dict__)
        class Buffer:
            def __init__(self,value):self.value=value
            def numpy(self):return self.value.copy()
            def assign(self,value):self.value=value.copy()
        q=np.arange(21,dtype=float);qbefore=q.copy();qd=Buffer(np.zeros(18))
        state=SimpleNamespace(joint_q=Buffer(q),joint_qd=qd)
        scene=SimpleNamespace(probe=c,q_starts=[0,7],state_0=state,model=object(),frame=0,time=0.)
        with patch('newton.eval_fk') as fk:
            expected=adapter.initialize_common_up_scene(scene)
        np.testing.assert_array_equal(state.joint_q.value[:14],expected.reshape(-1))
        np.testing.assert_array_equal(state.joint_q.value[14:],qbefore[14:])
        np.testing.assert_array_equal(qd.value,0);fk.assert_called_once()
        self.assertEqual(pickle.dumps(c.__dict__),before)
        self.assertIsNone(c.previous_world);self.assertIsNone(c.last_positions)
        scene.frame=1
        with self.assertRaises(ValueError):adapter.initialize_common_up_scene(scene)

    def test_collector_math_is_original_except_scene_and_initial_receipt(self):
        folder=Path('scripts/sugar/object_predictor')
        expected=(folder/'collect_dense_grip.py').read_text()
        expected=expected.replace('from .surface_grip_scene import SurfaceGripScene','from .common_up_cop_alignment import CommonUpSurfaceGripScene as SurfaceGripScene')
        expected=expected.replace("('collect_dense_grip.py','grip_motion_scene.py'","('collect_common_up_cop.py','collect_dense_grip.py','grip_motion_scene.py'")
        expected=expected.replace('    mass=float(scene.model.body_mass.numpy()[scene.box_body]);',"    (out/'INITIAL_GEOMETRY.json').write_text(json.dumps(scene.initial_geometry_receipt,indent=2))\n    mass=float(scene.model.body_mass.numpy()[scene.box_body]);")
        self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse((folder/'collect_common_up_cop.py').read_text())))

    def test_protocol_fixed_direct_baseline_and_no_gate_change(self):
        p=pilot.protocol()
        self.assertEqual(p['baseline_intervention'],'qualified_cop_response_v1')
        self.assertEqual([c['episode'] for c in p['configurations']],[5014,5012,5000])
        self.assertEqual(p['controls_per_case'],2400);self.assertEqual(len(p['original_physical_checks']),12)
        self.assertEqual(p['cop_rule']['admission_angle_max_deg'],5.)
        self.assertEqual(p['cop_rule']['per_hand_accumulated_travel_max_m'],.1)
        self.assertEqual(p['cop_rule']['active_clock_s'],[1.,40.])
        self.assertEqual(p['late_mass_check']['frame'],2381)
        old=json_read(Path(p['baseline_root'])/'PROTOCOL.json')
        for key in ('configurations','cop_rule','response_gain','sample_rule','late_mass_check','original_physical_checks'):
            self.assertEqual(p[key],old[key])


def json_read(path):
    import json
    return json.loads(path.read_text())
