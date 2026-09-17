"""CPU checks of native-AE qualification inputs/gates; no network execution."""
import unittest
from unittest.mock import patch

import numpy as np

from scripts.sugar.object_predictor.qualify_michelangelo_ae import (
    ABC_TO_VAE, DATASET, DISTANCE, OBJECT_IDS, numerical_gate, read_cases,
    sample_surface, evaluate_mesh,
)


class MichelangeloQualificationTests(unittest.TestCase):
    def test_fixed_four_true_meshes_public_transform_and_source_replay(self):
        cases = read_cases(DATASET)
        self.assertEqual(tuple(case['object_id'] for case in cases), OBJECT_IDS)
        for case in cases:
            vertices, faces = case['vertices'], case['faces']
            points, normals, ids, bary = sample_surface(vertices, faces, 4096, 1729)
            np.testing.assert_allclose(points, np.einsum('ni,nij->nj', bary, vertices[faces[ids]]), atol=1e-15)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-12)
            self.assertGreaterEqual(bary.min(), 0.)
            np.testing.assert_allclose(bary.sum(1), 1., atol=1e-12)
            np.testing.assert_allclose((points*ABC_TO_VAE)/ABC_TO_VAE, points, atol=1e-15)
            self.assertLessEqual(np.abs(vertices*ABC_TO_VAE).max(), .990001)
            for a, b in zip(sample_surface(vertices, faces, 4096, 1729), (points, normals, ids, bary)):
                np.testing.assert_array_equal(a, b)

    def test_gate_boundary_requires_every_metric(self):
        edge = dict(cd_x9000=.45, fscore=.95, area_ratio=1.5,
                    maximum_vertex_distance=DISTANCE,
                    maximum_face_probe_distance=DISTANCE, occupancy_iou=.90)
        self.assertTrue(numerical_gate(edge)['passed'])
        for name in ('cd_x9000', 'area_ratio', 'maximum_vertex_distance', 'maximum_face_probe_distance'):
            self.assertFalse(numerical_gate(dict(edge, **{name: np.nextafter(edge[name], np.inf)}))['passed'])
        for name in ('fscore', 'occupancy_iou'):
            self.assertFalse(numerical_gate(dict(edge, **{name: np.nextafter(edge[name], -np.inf)}))['passed'])
        for name in edge:
            self.assertFalse(numerical_gate(dict(edge, **{name: np.nan}))['passed'])
        self.assertFalse(numerical_gate({})['passed'])
        self.assertTrue(numerical_gate(dict(edge, area_ratio=.7))['passed'])
        self.assertFalse(numerical_gate(dict(edge, area_ratio=np.nextafter(.7, 0)))['passed'])

    def test_true_triangle_metric_detects_duplicate_area_despite_perfect_surface(self):
        case = read_cases(DATASET)[0]
        v, f = case['vertices'], case['faces']
        with patch('scripts.sugar.object_predictor.qualify_michelangelo_ae.EVAL_POINTS', 256):
            exact = evaluate_mesh(v, f, v, f, 1729)
            repeated = evaluate_mesh(v, f, v, np.tile(f, (3, 1)), 1729)
        self.assertLess(exact['cd_x9000'], 1e-20)
        self.assertEqual(exact['fscore'], 1.)
        self.assertAlmostEqual(exact['area_ratio'], 1.)
        self.assertLess(exact['maximum_face_probe_distance'], 1e-12)
        self.assertTrue(numerical_gate(dict(exact, occupancy_iou=1.))['passed'])
        self.assertEqual(repeated['fscore'], 1.)
        self.assertAlmostEqual(repeated['area_ratio'], 3.)
        self.assertFalse(numerical_gate(dict(repeated, occupancy_iou=1.))['passed'])


if __name__ == '__main__':
    unittest.main()
