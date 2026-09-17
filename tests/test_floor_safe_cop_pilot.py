import ast
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts.sugar.object_predictor import run_floor_safe_cop_pilot as pilot


class FloorPilotTests(unittest.TestCase):
    def test_fixed_direct_baseline_and_original_gates(self):
        p=pilot.protocol()
        self.assertEqual(p['baseline_intervention'],'qualified_cop_response_v1')
        self.assertEqual([c['episode'] for c in p['configurations']],[5014,5012,5000])
        self.assertEqual(len(p['original_physical_checks']),12)
        self.assertEqual(p['controls_per_case'],2400);self.assertEqual(p['physics_substeps'],8)
        self.assertEqual(p['cop_rule']['active_clock_s'],[1.,40.])
        self.assertEqual(p['cop_rule']['admission_angle_max_deg'],5.)
        self.assertEqual(p['cop_rule']['per_hand_accumulated_travel_max_m'],.1)
        self.assertEqual(p['late_mass_check']['frame'],2381)
        self.assertFalse(p['automatic_expansion'])
        self.assertGreater(p['known_hand_geometry']['left']['vertices'],1000)

    def test_collector_original_math_except_explicit_scene_and_receipts(self):
        root=Path('scripts/sugar/object_predictor')
        expected=(root/'collect_dense_grip.py').read_text()
        expected=expected.replace('from .surface_grip_scene import SurfaceGripScene','from .floor_safe_cop_scene import FloorSafeCoPScene as SurfaceGripScene')
        expected=expected.replace("('collect_dense_grip.py','grip_motion_scene.py'","('collect_floor_safe_cop.py','floor_safe_cop_scene.py','collect_dense_grip.py','grip_motion_scene.py'")
        expected=expected.replace("    a={k:np.stack([r[k] for r in rows])", "    scene.save_floor_receipt(out/'HAND_FLOOR_SUBSTEPS.npz')\n    a={k:np.stack([r[k] for r in rows])")
        expected=expected.replace("        if rows:np.savez_compressed", "        if scene.floor_substep_records:scene.save_floor_receipt(out/'HAND_FLOOR_SUBSTEPS.partial.npz')\n        if rows:np.savez_compressed")
        self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse((root/'collect_floor_safe_cop.py').read_text())))

    def fixture(self):
        pose=np.array([[0.,0.,1.,0.,0.,0.,1.],[1.,0.,1.,0.,0.,0.,1.]])
        hand=np.broadcast_to(pose,(2400,2,7)).copy()
        sub=dict(timestamp_s=.0025*np.arange(1,19201),control_frame=np.repeat(np.arange(2400),8),
            substep=np.tile(np.arange(8),2400),initial_hand_pose_w=pose,initial_min_z_m=np.ones(2),
            fk_hand_pose_w=np.repeat(hand,8,axis=0),actual_hand_pose_w=np.repeat(hand,8,axis=0),
            fk_min_z_m=np.ones((19200,2)),actual_min_z_m=np.ones((19200,2)))
        a=dict(timestamp_s=.02*np.arange(1,2401),hand_pose_w=hand)
        values=dict(floor_blocked=np.zeros(2400,bool),floor_blocked_count=np.zeros(2400,int),
            floor_executed_fraction=np.ones(2400),floor_requested_target_pose_w=hand.copy(),
            floor_observed_min_z_m=np.ones((2400,2)),floor_requested_swept_min_z_m=np.ones((2400,2)),
            floor_executed_swept_min_z_m=np.ones((2400,2)))
        a.update({'validation_controller_'+k:v for k,v in values.items()})
        return a,sub

    def test_actual_hidden_substep_floor_failure_not_endpoint_pass(self):
        a,sub=self.fixture()
        # Endpoints are safe, but an actual solver substep violates the new gate.
        sub['actual_hand_pose_w'][3,0,2]=-.01;sub['actual_min_z_m'][3,0]=-.01
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(Path(tmp)/'HAND_FLOOR_SUBSTEPS.npz',**sub)
            with patch('sugar_newton.hand.patches.load_hand_mesh',return_value=SimpleNamespace(vertices=np.zeros((1,3)))):
                receipt=pilot.floor_readback(a,Path(tmp))
        self.assertTrue(receipt['passed'])
        self.assertFalse(receipt['actual_substep_hand_floor_passed'])
        self.assertEqual(receipt['actual_solver_min_z_m'][0],-.01)

    def test_forged_floor_minimum_fails_integrity(self):
        a,sub=self.fixture();sub['actual_min_z_m'][3,0]=-.01
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(Path(tmp)/'HAND_FLOOR_SUBSTEPS.npz',**sub)
            with patch('sugar_newton.hand.patches.load_hand_mesh',return_value=SimpleNamespace(vertices=np.zeros((1,3)))):
                with self.assertRaisesRegex(ValueError,'minimum replay'):
                    pilot.floor_readback(a,Path(tmp))


if __name__=='__main__':unittest.main()
