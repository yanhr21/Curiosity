"""CPU-only observed support/coordinate tests, without GT mesh or any model."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.sugar.object_predictor.active3d_charts import supported_chart
from scripts.sugar.object_predictor.active3d_local_charts import supported_local_chart
from scripts.sugar.object_predictor.official_active3d import read_template_obj


EXPERIMENT = Path(__file__).resolve().parents[1]/'experiments/object_predictor_v1'


class SupportedLocalChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        template, faces, _ = read_template_obj(EXPERIMENT/'vendor/Active-3D-Vision-and-Touch/pterotactyl/objects/touch_chart.obj')
        cls.template, cls.faces = template.numpy(), faces.verts_idx.numpy()

    def assert_replay(self, chart, points):
        self.assertEqual(chart['points'].shape, (25, 3))
        self.assertEqual(chart['source_indices'].shape, (25, 3))
        self.assertTrue(np.all(chart['source_indices'] >= 0))
        self.assertLess(int(chart['source_indices'].max()), len(points))
        np.testing.assert_allclose(chart['barycentric'].sum(1), 1., atol=1e-12, rtol=0)
        self.assertGreaterEqual(chart['barycentric'].min(), -1e-10)
        reconstructed = np.einsum('ni,nij->nj', chart['barycentric'], points[chart['source_indices']])
        np.testing.assert_allclose(reconstructed, chart['points'], atol=1e-12, rtol=0)
        self.assertLessEqual(chart['support_edge_m'].max(), .0017)

    @staticmethod
    def curved_points():
        angle, z = np.meshgrid(np.linspace(-1.2, 1.2, 13), np.linspace(-.001, .001, 5), indexing='ij')
        return np.column_stack((.004*np.sin(angle.ravel()), .004*np.cos(angle.ravel()), z.ravel()))

    def test_successful_original_full_pad_is_bit_exact_and_does_not_enter_local_path(self):
        x, y = np.meshgrid(np.linspace(-.002, .002, 7), np.linspace(-.002, .002, 7))
        points = np.column_stack((x.ravel(), y.ravel(), x.ravel()*0))
        areas = np.ones(len(points))*1e-7
        expected, expected_info = supported_chart(points, areas, self.template)
        result, info = supported_local_chart(points, areas, self.template)
        self.assertEqual(info, expected_info)
        self.assertEqual(result.keys(), expected.keys())
        for key in result:
            np.testing.assert_array_equal(result[key], expected[key])
        self.assertNotIn('adapter', info)

    def test_curved_observation_uses_local_support_without_extrapolation(self):
        points = self.curved_points(); areas = np.ones(len(points))*1e-7
        original, info = supported_chart(points, areas, self.template)
        self.assertIsNone(original)
        self.assertEqual(info['reason'], 'unresolved_or_thick_patch')
        chart, info = supported_local_chart(points, areas, self.template)
        self.assertIsNotNone(chart)
        self.assertEqual(info['local_radius_m'], .0034)
        self.assertTrue(info['full_footprint_supported'])
        indices = chart['local_source_indices']; centre = chart['local_centre_source_index']
        self.assertLessEqual(np.linalg.norm(points[indices]-points[centre], axis=1).max(), .0034)
        self.assertTrue(np.isin(chart['source_indices'], indices).all())
        self.assertTrue(np.isin(chart['support_triangles'], indices).all())
        self.assert_replay(chart, points)
        again, repeat_info = supported_local_chart(points, areas, self.template)
        self.assertEqual(info, repeat_info)
        for key in chart:
            np.testing.assert_array_equal(chart[key], again[key])

    def test_separated_surfaces_do_not_get_a_triangle_across_the_gap(self):
        x, y = np.meshgrid(np.linspace(-.0005, .0005, 3), np.linspace(-.0005, .0005, 3))
        patch = np.column_stack((x.ravel(), y.ravel(), x.ravel()*0))
        centres = np.array([[0., 0., 0.], [.01, 0., .01], [0., .01, .01], [.01, .01, 0.]])
        points = np.concatenate([patch+c for c in centres])
        original, _ = supported_chart(points, np.ones(len(points)), self.template)
        self.assertIsNone(original)  # Whole tetrahedral arrangement is not planar.
        chart, info = supported_local_chart(points, np.ones(len(points)), self.template)
        self.assertIsNotNone(chart)
        self.assertEqual(np.unique(chart['source_indices']//9).size, 1)
        self.assertEqual(np.unique(chart['support_triangles']//9).size, 1)
        self.assertTrue(info['full_footprint_supported'])
        self.assert_replay(chart, points)
        # Three points at each well-separated island cannot be combined to
        # manufacture the six unique supporting points the original gate needs.
        sparse = np.concatenate([patch[[0, 2, 8]]+c for c in centres])
        absent, why = supported_local_chart(sparse, np.ones(len(sparse)), self.template)
        self.assertIsNone(absent)
        self.assertEqual(why['valid_local_candidates'], 0)

    def test_duplicate_sources_still_reference_original_array(self):
        curve = self.curved_points()
        # Interleave duplicates; source indices are not a compact unique-point index.
        points = np.repeat(curve, 2, axis=0)
        chart, info = supported_local_chart(points, np.linspace(1., 2., len(points)), self.template)
        self.assertIsNotNone(chart)
        self.assert_replay(chart, points)
        self.assertTrue(np.all(chart['source_indices'] % 2 == 0))

    def test_invalid_measurements_are_rejected_not_filtered_into_success(self):
        points = self.curved_points(); areas = np.ones(len(points))
        bad = points.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            supported_local_chart(bad, areas, self.template)
        bad_area = areas.copy(); bad_area[0] = -1
        with self.assertRaises(ValueError):
            supported_local_chart(points, bad_area, self.template)
        with self.assertRaises(ValueError):
            supported_local_chart(points, areas, self.template, max_edge_m=0.)

    def test_actual_failed_case_matches_diagnostic_and_replays_world_frame(self):
        root = EXPERIMENT/'overfit_repair_v1'
        case = root/'sugar_shape_fixed40/object_11898_direction_29'
        with np.load(case/'observed_surface.npz', allow_pickle=False) as s:
            begin, end = s['offset'][499:501]
            pos, area = s['pos'][begin:end], s['area'][begin:end]
            indices = np.flatnonzero((s['pad'][begin:end] == 26) & (area > 0)
                                     & (s['pressure'][begin:end] > 0))
        with np.load(case/'trace.npz', allow_pickle=False) as trace:
            pose = trace['hand_pose_w'][499]  # Do not read validation-only channels.
        chart, info = supported_local_chart(pos[indices], area[indices], self.template)
        diagnostic = json.loads((root/'chart_local_support_diagnostic.json').read_text())
        self.assertEqual(info['local_candidate_count'], 27)
        self.assertEqual(info['valid_local_candidates'], 27)
        self.assertEqual(info['local_centre_source_index'], 14)
        self.assertEqual(info['width_m'], max(row['width_m'] for row in diagnostic['valid_regions']))
        self.assertEqual(info['chart_area_m2'], max(row['chart_area_m2'] for row in diagnostic['valid_regions']))
        self.assert_replay(chart, pos[indices])
        frame_indices = indices[chart['source_indices']]
        replay = np.einsum('ni,nij->nj', chart['barycentric'], pos[frame_indices])
        np.testing.assert_allclose(replay, chart['points'], atol=1e-12, rtol=0)
        rotation = Rotation.from_quat(pose[3:]).as_matrix()
        world = chart['points']@rotation.T+pose[:3]
        canonical = (world-[0., 0., .75])/.93
        np.testing.assert_allclose(canonical*.93+[0., 0., .75], world, atol=1e-12, rtol=0)
        # No saved result gets rewritten or requalified by this diagnostic.
        result = json.loads((root/'sugar_shape_fixed40/RESULT.json').read_text())
        self.assertFalse(result['qualification_passed'])
        original = next(row for row in result['cases'] if row['object_id'] == '11898' and row['direction_index'] == 29)
        self.assertFalse(original['chart']['available'])
        self.assertFalse(original['qualification_passed'])


if __name__ == '__main__':
    unittest.main()
