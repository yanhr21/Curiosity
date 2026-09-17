"""Actual saved observation and physical-unit checks; no model/physics/GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.sugar.object_predictor import single_hand_shape_data as data


class SingleHandShapeDataTests(unittest.TestCase):
    def test_fixed_mapping_keeps_all_40_and_only_declared_replacements(self):
        plan = data.source_plan()
        self.assertEqual(len(plan), 40)
        self.assertEqual([p['slot'] for p in plan], list(range(40)))
        self.assertEqual(sum(p['source_kind'] == 'original_supported' for p in plan), 37)
        changed = {(p['object_id'], p['direction_index']): (p['source_kind'], p['snapshot_frame'])
                   for p in plan if p['source_kind'] != 'original_supported'}
        self.assertEqual(changed, {('11898', 29): ('original_fixed_local_fallback', 499),
            ('13266', 0): ('planned_search_snapshot', 827),
            ('13266', 49): ('planned_search_snapshot', 854)})
        self.assertEqual(plan[0]['source_kind'], 'original_supported')
        self.assertEqual([(p['object_id'], p['direction_index']) for p in plan],
            [(oid, d) for oid in data.OBJECT_IDS for group in data.DIRECTIONS for d in group])

    def test_pressure_and_signed_traction_integrate_once_per_probe(self):
        _, anchors, _ = data.left_hand_cad()
        # Same voxel with nonidentical areas and opposite traction components.
        pos = np.array([anchors[0], anchors[0]], dtype=np.float64)
        surface = dict(pos=pos, area=np.array([2e-6, 3e-6]), pressure=np.array([1e5, 2e5]),
            traction_vec=np.array([[3e4, -4e4, 0.], [-1e4, 2e4, 0.]]),
            pad=np.array([0, 0]), patch=np.array([0, 0]))
        rotation = Rotation.from_euler('z', 90, degrees=True)
        pose = np.r_[[.02, -.03, .75], rotation.as_quat()]
        encoded, provenance, metrics = data.encode_probe(pose, surface)
        self.assertAlmostEqual(metrics['observed_scalar_load_n'], .8)
        self.assertAlmostEqual(metrics['voxel_scalar_load_n'], .8)
        expected_local_force = np.array([.03, -.02, 0.])
        expected_world_force = rotation.apply(expected_local_force)
        np.testing.assert_allclose(provenance['voxel_estimated_traction_force_world_n'].sum(0), expected_world_force, atol=1e-14)
        np.testing.assert_allclose((2*np.sinh(encoded['feat'][:, 11:14].astype(float))).sum(0), expected_world_force, atol=1e-8)
        self.assertAlmostEqual(float(np.expm1(encoded['feat'][:, 10].astype(float)).sum()), .8, places=6)
        self.assertEqual(provenance['source_to_voxel'][0], provenance['source_to_voxel'][1])
        np.testing.assert_allclose(provenance['source_contact_world_m'], pos@rotation.as_matrix().T+pose[:3], atol=1e-15)
        self.assertTrue(np.all(encoded['feat'][:, 15:17] == 0))  # age and left identity
        self.assertEqual(len(provenance['anchor_world_m']), 27)

    def test_actual_probe_chart_replay_and_observed_load(self):
        plan = next(p for p in data.source_plan() if (p['object_id'], p['direction_index']) == ('13266', 0))
        pose, surface, clock = data.read_snapshot(plan['directory'], plan['snapshot_frame'])
        with np.load(plan['chart_file'], allow_pickle=False) as z:
            chart = {k: z[k] for k in data.CHART_KEYS}
        self.assertLessEqual(data.validate_chart(chart, surface, pose, clock['source_frame_offset']), 1e-12)
        encoded, provenance, metrics = data.encode_probe(pose, surface)
        self.assertEqual(encoded['feat'].shape[1], 20)
        self.assertLess(abs(metrics['observed_scalar_load_n']-clock['measured_palmar_load_n']), 1e-6)
        self.assertEqual(metrics['actual_hands'], 1)
        np.testing.assert_allclose((provenance['source_contact_world_m']-pose[:3])@Rotation.from_quat(pose[3:]).as_matrix(),
                                   surface['pos'][provenance['source_indices']], atol=1e-15)
        old = json.loads((data.SEARCH_ROOT/'RESULT.json').read_text())['cases'][0]
        self.assertFalse(old['qualification_passed'])
        self.assertFalse(old['qualification_checks']['target_band'])
        self.assertTrue(old['chart']['available'])

    def test_npz_allowlist_ignores_gt_and_controller_arrays(self):
        plan = data.source_plan()[0]
        pose, surface, clock = data.read_snapshot(plan['directory'], 499)
        first, _, _ = data.encode_probe(pose, surface)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            np.savez(root/'trace.npz', time_s=np.array([.02]), hand_pose_w=pose[None],
                measured_palmar_load_n=np.array([clock['measured_palmar_load_n']]),
                object_pose_w=np.full((1, 7), np.nan), object_mass_kg=np.array([999999.]),
                validation_full_hand_load_n=np.array([1e9]), target_depth_m=np.array([np.nan]))
            np.savez(root/'observed_surface.npz', offset=np.array([0, len(surface['pad'])]),
                **surface, object_vertices=np.full((10, 3), np.nan), anatomical_pad=np.full(len(surface['pad']), -1))
            accessed = []
            original_load = np.load
            class AllowedReader:
                def __init__(self, path, **kwargs):
                    self.path = Path(path); self.saved = original_load(path, **kwargs)
                def __enter__(self): return self
                def __exit__(self, *args): self.saved.close()
                def __getitem__(self, key):
                    allowed = data.TRACE_KEYS if self.path.name == 'trace.npz' else ('offset',)+data.SURFACE_KEYS
                    if key not in allowed: raise AssertionError('Forbidden array read: '+key)
                    accessed.append(key); return self.saved[key]
            with patch.object(data.np, 'load', AllowedReader):
                measured, sensed, _ = data.read_snapshot(root, 0)
            second, _, _ = data.encode_probe(measured, sensed)
            for key in first: np.testing.assert_array_equal(first[key], second[key])
            self.assertNotIn('object_pose_w', accessed)
            with self.assertRaises(ValueError):
                data.encode_probe(pose, dict(surface, object_id=np.array([18704])))

    def test_independent_probe_offsets_never_sum_five_force_observations(self):
        p = data.source_plan()[0]
        pose, surface, _ = data.read_snapshot(p['directory'], p['snapshot_frame'])
        encoded, _, metrics = data.encode_probe(pose, surface)
        packed = data.pack_probes([encoded for _ in range(40)])
        self.assertEqual(set(packed), set(data.INPUT_KEYS))
        self.assertEqual(packed['offset'].shape, (40,))
        for lo, hi in zip(np.r_[0, packed['offset'][:-1]], packed['offset']):
            self.assertAlmostEqual(float(np.expm1(packed['feat'][lo:hi, 10].astype(float)).sum()),
                                   metrics['observed_scalar_load_n'], places=6)
        with self.assertRaises(ValueError): data.pack_probes([encoded]*39)

    def test_missing_contact_retains_only_real_left_geometry(self):
        pose = np.array([0., 0., .75, 0., 0., 0., 1.])
        surface = dict(pos=np.empty((0, 3)), area=np.empty(0), pressure=np.empty(0),
            traction_vec=np.empty((0, 3)), pad=np.empty(0, np.int32), patch=np.empty(0, np.int32))
        encoded, provenance, metrics = data.encode_probe(pose, surface)
        self.assertEqual(metrics['active_assigned_contact_faces'], 0)
        self.assertEqual(metrics['actual_hands'], 1)
        self.assertEqual(len(provenance['anchor_to_voxel']), 27)
        self.assertTrue(np.all(encoded['feat'][:, 9:17] == 0))
        self.assertGreater(len(encoded['coord']), 0)


if __name__ == '__main__':
    unittest.main()
