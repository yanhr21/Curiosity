import ast
import copy
import pickle
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scripts.sugar.object_predictor.floor_safe_cop import FloorSafeCoPController
from scripts.sugar.object_predictor.qualified_cop_response import QualifiedCoPResponseController
from scripts.sugar.object_predictor.test_coupled_surface_grip import CoupledSurfaceGripTests


class FloorSafeCoPTests(unittest.TestCase):
    def pair(self, centers=None):
        poses, field = CoupledSurfaceGripTests().observation(centers=centers)
        old = QualifiedCoPResponseController(target_load_n=12., force_gain=.000025, response_gain=True)
        new = FloorSafeCoPController(target_load_n=12., force_gain=.000025, response_gain=True)
        for controller in (old, new):
            controller.support_normals = np.array([[1., 0., 0.], [-1., 0., 0.]])
            controller.observe(poses, field)
        return old, new, poses, field

    def test_safe_full_commands_and_all_parent_state_exact(self):
        old, new, poses, field = self.pair()
        new.floor_vertices = (np.zeros((1, 3)), np.zeros((1, 3)))
        for i in range(80):
            for c in (old, new): c.observe(poses, field)
            a = old.command(24. + .02*i, np.full(2, 12.), .02)
            b = new.command(24. + .02*i, np.full(2, 12.), .02)
            np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
            for key in a[2]: np.testing.assert_array_equal(a[2][key], b[2][key], err_msg=key)
            for key in old.__dict__:
                self.assertEqual(pickle.dumps(old.__dict__[key]), pickle.dumps(new.__dict__[key]), key)
            self.assertFalse(b[2]['floor_blocked'])
            poses = b[0]

    def test_real_downward_cop_rejected_without_distance_or_travel(self):
        _, new, poses, field = self.pair(centers=np.array([[0., 0., .08], [0., 0., -.08]]))
        poses[:, 2] = .00002
        new.floor_vertices = (np.zeros((1, 3)), np.zeros((1, 3)))
        new.observe(poses, field)
        out, velocity, rec = new.command(24., np.full(2, 8.), .02)
        self.assertTrue(rec['floor_blocked'])
        self.assertLess(rec['floor_requested_swept_min_z_m'][0], 0.)
        np.testing.assert_array_equal(out, poses); np.testing.assert_array_equal(velocity, 0.)
        np.testing.assert_array_equal(new.distance, 0.); np.testing.assert_array_equal(new.cop_travel, 0.)
        self.assertTrue((rec['floor_requested_distance_m'] > 0.).all())
        self.assertTrue(rec['response_command_pure'].all())
        self.assertEqual(new.floor_blocked_count, 1)

    def test_rejected_admission_and_motion_state_roll_back(self):
        _, new, poses, field = self.pair()
        new.ready_seconds = .99
        old = {k:copy.deepcopy(getattr(new, k)) for k in (
            'distance','cop_travel','lift_start','lift_complete','motion_elapsed',
            'previous_lift','previous_yaw','last_positions','last_rotations','previous_world')}
        with patch('scripts.sugar.object_predictor.floor_safe_cop.swept_min_height', side_effect=[.1,.1,-.1,.1]):
            out, velocity, rec = new.command(24., np.full(2, 12.), .02)
        self.assertEqual(rec['floor_requested_lift_start_s'], 24.)
        self.assertEqual(new.lift_start, -1.); self.assertEqual(new.ready_seconds, 0.)
        for k,v in old.items(): self.assertEqual(pickle.dumps(getattr(new,k)), pickle.dumps(v), k)
        for k in ('response_command_rotation_rad','response_command_cop_travel_m','response_command_phase_delta_s',
                  'cop_servo_velocity_m_s','actual_alignment_active','actualmotion_rate'):
            np.testing.assert_array_equal(rec[k], 0., err_msg=k)

    def test_response_consumes_real_observation_once_not_candidate_distance(self):
        old, new, poses, field = self.pair()
        new.floor_vertices = (np.zeros((1, 3)), np.zeros((1, 3)))
        expected = copy.deepcopy(new.response)
        for step in range(9):
            time = 24. + step*.02
            loads = np.full(2, 8.+.2*step)
            for side in (0, 1): expected.update(side, time, float(loads[side]), 0., .02, True)
            expected.incoming_pure[:] = True
            with patch('scripts.sugar.object_predictor.floor_safe_cop.swept_min_height', side_effect=[.1,.1,-.1,.1]):
                _, _, rec = new.command(time, loads, .02)
            self.assertEqual(pickle.dumps(new.response.__dict__), pickle.dumps(expected.__dict__))
            self.assertEqual(len(new.response.samples[0]), min(step+1, 6))
            self.assertEqual(new.response.accepted_total[0], 0)

    def test_rejected_runtime_command_preserves_path_clock(self):
        _, new, _, _ = self.pair()
        new.lift_start=24.; new.motion_elapsed=2.; new.previous_lift=.5; new.previous_yaw=10.
        with patch('scripts.sugar.object_predictor.floor_safe_cop.swept_min_height', side_effect=[.1,.1,-.1,.1]):
            _, _, rec = new.command(27., np.full(2, 12.), .02)
        self.assertEqual(new.motion_elapsed, 2.); self.assertEqual(new.previous_lift, .5)
        self.assertEqual(new.previous_yaw, 10.); self.assertTrue(rec['motion_paused'])
        self.assertEqual(rec['floor_requested_motion_elapsed_s'], 2.02)
        self.assertEqual(rec['motion_elapsed_s'], 2.)

    def test_old_initial_orientation_retained_and_no_gt_api(self):
        old=QualifiedCoPResponseController(response_gain=True)
        new=FloorSafeCoPController(response_gain=True)
        a=old.command(0., np.zeros(2), .02); b=new.command(0., np.zeros(2), .02)
        np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
        self.assertGreater(b[2]['floor_executed_swept_min_z_m'].min(), 0.)

    def test_scene_physics_order_original_except_observation_receipts(self):
        folder=Path('scripts/sugar/object_predictor')
        original=ast.parse((folder/'probe_scene.py').read_text())
        revised=ast.parse((folder/'floor_safe_cop_scene.py').read_text())
        def step(tree):
            return next(n for cls in tree.body if isinstance(cls,ast.ClassDef)
                        for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
        a,b=step(original),step(revised)
        b.body=b.body[1:]  # SurfaceGripScene.observe precedes original ProbeScene.step.
        loop=next(n for n in b.body if isinstance(n,ast.For))
        loop.body=[n for n in loop.body if not (
            isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('fk_poses','actual') for t in n.targets)
            or isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='append')]
        self.assertEqual(ast.dump(a),ast.dump(b))


if __name__=='__main__':unittest.main()
