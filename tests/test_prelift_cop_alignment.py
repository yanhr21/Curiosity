import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.sugar.object_predictor.prelift_cop_alignment import (
    PreliftCoPAlignmentController, observed_alignment)
from scripts.sugar.object_predictor.freeze_postlift_alignment import FreezePostliftAlignmentController
from scripts.sugar.object_predictor.test_coupled_surface_grip import CoupledSurfaceGripTests


class CoPAlignmentTests(unittest.TestCase):
    def setup_pair(self, centers=None):
        controllers = [kind(target_load_n=16., response_gain=True)
                       for kind in (FreezePostliftAlignmentController, PreliftCoPAlignmentController)]
        poses, field = CoupledSurfaceGripTests().observation(centers=centers)
        for controller in controllers:
            controller.support_normals = np.array([[1., 0., 0.], [-1., 0., 0.]])
            controller.observe(poses, field)
        return controllers, poses, field

    def test_aligned_admission_preserves_parent_exactly(self):
        controllers, _, _ = self.setup_pair()
        outputs = []
        for c in controllers:
            c.ready_seconds = .98
            outputs.append(c.command(24., np.full(2, 16.), .02))
        for index in (0, 1):
            np.testing.assert_array_equal(outputs[0][index], outputs[1][index])
        for key in outputs[0][2]:
            np.testing.assert_array_equal(outputs[0][2][key], outputs[1][2][key])
        self.assertEqual(controllers[1].lift_start, 24.)

    def test_misaligned_contacts_block_admission_and_move_symmetrically(self):
        controllers, _, _ = self.setup_pair(np.array([[0., 0., -.08], [0., 0., .08]]))
        c = controllers[1]; c.ready_seconds = 100.
        _, _, record = c.command(24., np.full(2, 16.), .02)
        self.assertEqual(c.lift_start, -1.)
        self.assertEqual(c.ready_seconds, 0.)
        np.testing.assert_allclose(record['cop_servo_velocity_m_s'], [[0., 0., .004], [0., 0., -.004]])
        np.testing.assert_array_equal(record['response_gain_m_per_ns'], np.full(2, c.force_gain))

    def test_pressure_weighting_and_world_transform(self):
        controllers, poses, field = self.setup_pair()
        field['pressure'][0] *= 10.
        valid, centers, error, angle = observed_alignment(poses, field, controllers[1].support_normals)
        expected = np.average(field['pos'][:9], axis=0, weights=field['pressure'][:9]*field['area'][:9]) + poses[0, :3]
        np.testing.assert_allclose(centers[0], expected)
        turn = Rotation.from_euler('xyz', [20., 37., 16.], degrees=True)
        moved = poses.copy(); moved[:, :3] = turn.apply(poses[:, :3]) + [.1, .2, .3]
        moved[:, 3:] = (turn*Rotation.from_quat(poses[:, 3:])).as_quat()
        v2, c2, e2, a2 = observed_alignment(moved, field, controllers[1].support_normals)
        np.testing.assert_array_equal(valid, v2)
        np.testing.assert_allclose(c2, turn.apply(centers)+[.1, .2, .3])
        np.testing.assert_allclose(e2, turn.apply(error), atol=1e-16)
        self.assertAlmostEqual(angle, a2)

    def test_missing_contact_no_servo_and_no_ready(self):
        controllers, poses, field = self.setup_pair()
        field['pressure'][9:] = 0.
        c=controllers[1]; c.observe(poses, field); c.ready_seconds=10.
        record=c.command(24., np.full(2,16.), .02)[2]
        self.assertEqual(c.lift_start,-1.)
        self.assertEqual(c.ready_seconds,0.)
        self.assertFalse(record['cop_servo_velocity_m_s'].any())

    def test_budget_and_timeout_remain_failures(self):
        controllers, _, _ = self.setup_pair(np.array([[0., 0., -.08], [0., 0., .08]]))
        c=controllers[1];c.cop_travel[:]=.1-1e-5
        record=c.command(24.,np.full(2,16.),.02)[2]
        np.testing.assert_allclose(c.cop_travel,.1)
        self.assertTrue(record['cop_budget_exhausted'])
        self.assertEqual(c.lift_start,-1.)
        c.cop_travel[:]=0.
        record=c.command(40.02,np.full(2,16.),.02)[2]
        self.assertFalse(record['cop_servo_velocity_m_s'].any())
        self.assertEqual(c.lift_start,-1.)

    def test_postlift_original_path_exact_with_misaligned_contacts(self):
        controllers, _, _ = self.setup_pair(np.array([[0., 0., -.08], [0., 0., .08]]))
        outputs=[]
        for c in controllers:
            c.lift_start=24.;outputs.append(c.command(24.02,np.array([8.,24.]),.02))
        for i in (0,1):np.testing.assert_array_equal(outputs[0][i],outputs[1][i])
        self.assertFalse(outputs[1][2]['cop_servo_velocity_m_s'].any())


if __name__ == '__main__': unittest.main()
