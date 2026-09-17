"""CPU checks of exact official grid coordinates and GT-only label handling."""
import unittest
import numpy as np
import trimesh
from scripts.sugar.object_predictor import prepare_michelangelo_extraction_grid as grid


class ExtractionGridTests(unittest.TestCase):
    def test_official_grid_shape_endpoints_and_flatten_order(self):
        points, shape, length = grid.official_grid()
        self.assertEqual(points.dtype, np.float32)
        np.testing.assert_array_equal(shape, [129, 129, 129])
        np.testing.assert_array_equal(points[0], [-1.25]*3)
        np.testing.assert_array_equal(points[-1], [1.25]*3)
        np.testing.assert_array_equal(points[1], [-1.25, -1.25, -1.25+2.5/128])
        np.testing.assert_array_equal(points[129], [-1.25, -1.25+2.5/128, -1.25])
        np.testing.assert_array_equal(points[129**2], [-1.25+2.5/128, -1.25, -1.25])
        np.testing.assert_array_equal(length, [2.5]*3)

    def test_boundary_outside_and_heldout_nodes_remain_but_ineligible(self):
        # Analytic box checks labels; it is not a shape model or replacement data.
        mesh = trimesh.creation.box(extents=[2., 2., 2.])
        abc = np.array([[0,0,0], [.5,0,0], [1,0,0], [1+5e-7,0,0], [1.1,0,0]])
        points = (abc*grid.ABC_TO_VAE).astype(np.float32)
        arrays = grid.label_grid(mesh, points, points[1:2], chunk=2)
        np.testing.assert_array_equal(arrays['occupancy_labels'][[0,1,4]], [True,True,False])
        np.testing.assert_array_equal(arrays['boundary_ambiguous'], [False,False,True,True,False])
        np.testing.assert_array_equal(arrays['evaluation_overlap'], [False,True,False,False,False])
        np.testing.assert_array_equal(arrays['training_eligible'], [True,False,False,False,True])
        self.assertTrue(arrays['strict_aabb_outside'][3])
        self.assertIn(3, arrays['boundary_candidate_indices'])
        self.assertNotIn(4, arrays['boundary_candidate_indices'])
        self.assertEqual(len(arrays['occupancy_labels']),len(points))
        self.assertTrue(grid.alternate_ray_check(mesh,points,arrays,0)['passed'])

    def test_float32_intersection_including_signed_zero(self):
        points=np.array([[0.,-0.,1.], [1.,2.,3.]],dtype=np.float32)
        other=np.array([[-0.,0.,1.]],dtype=np.float32)
        np.testing.assert_array_equal(grid.overlap_mask(points,other), [True,False])

    def test_open_mesh_is_rejected_not_repaired(self):
        mesh=trimesh.creation.box();mesh.update_faces(np.arange(len(mesh.faces)-1))
        with self.assertRaises(ValueError):
            grid.label_grid(mesh,np.zeros((1,3),np.float32),np.empty((0,3),np.float32))


if __name__ == '__main__':
    unittest.main()
