"""Synthetic CPU tests for fixture sequence identity, padding and GT isolation.

No simulation or model is executed. Synthetic charts do not constitute actual
fixture qualification or reconstruction results.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .report_fixture_sequences import DIRECTION_GROUPS, pack_sequences, report_attempt, validate_sequence_protocol
from .shape_fixture_assets import fixed_selection


class FixtureSequenceTests(unittest.TestCase):
    def protocol(self):
        return dict(objects=[i for i, s in fixed_selection() if s == 'recon_train'][:4],
                    direction_groups=[list(g) for g in DIRECTION_GROUPS],
                    controls_per_attempt=600, dt=.02, substeps=8, sample_frame=499)

    def template(self):
        vertices = np.array([[0., y, z] for y in range(5) for z in range(5)])
        faces = []
        for y in range(4):
            for z in range(4):
                i = y*5+z
                faces.extend(((i, i+1, i+5), (i+1, i+6, i+5)))
        return vertices, np.array(faces)

    def create_attempt(self, root, expected, *, gt_peak=2., missing_gt=False):
        label = f'object_{expected["object_id"]}_direction_{expected["direction_index"]:02d}'
        directory = root/label
        directory.mkdir()
        attempt = dict(**expected, directory=label, complete=True, recorded_controls=600,
                       actual_controls=600, actual_physics_substeps=4800,
                       partial_control_substeps=0, field_overflow_steps=0)
        (directory/'ATTEMPT.json').write_text(json.dumps(attempt))
        loads = np.zeros(600)
        loads[25:500] = 2.
        poses = np.zeros((600, 7))
        poses[:, 2] = .75
        poses[:, 6] = 1.
        trace = dict(time_s=np.arange(1, 601)*.02, hand_pose_w=poses,
                     hand_velocity_w=np.zeros((600, 6)), measured_palmar_load_n=loads,
                     target_depth_m=np.zeros(600), touched=loads > 0, overload_seen=np.zeros(600, bool))
        if not missing_gt:
            full = loads.copy()
            full[400] = gt_peak
            trace['validation_full_hand_load_n'] = full
        np.savez_compressed(directory/'trace.npz', **trace)
        points = np.array([[0., y*.0005, z*.0006] for y in range(-2, 3) for z in range(-2, 3)])
        np.savez_compressed(directory/'observed_surface.npz',
            offset=np.arange(601, dtype=np.int64)*25, pos=np.tile(points, (600, 1)),
            area=np.full(600*25, 1e-6), pressure=np.repeat(loads/(25e-6), 25),
            traction_vec=np.zeros((600*25, 3)), patch=np.zeros(600*25, np.int64),
            pad=np.zeros(600*25, np.int64), anatomical_pad=np.zeros(600*25, np.int64))

    def test_exact_fixed_protocol_and_reject_reordering(self):
        protocol = self.protocol()
        self.assertEqual(len(validate_sequence_protocol(protocol)), 40)
        protocol['direction_groups'][0].reverse()
        with self.assertRaises(ValueError):
            validate_sequence_protocol(protocol)

    def test_gt_peak_or_missing_gt_cannot_change_observed_chart(self):
        charts = []
        expected = validate_sequence_protocol(self.protocol())[0]
        template, faces = self.template()
        for peak, missing in ((2., False), (500., False), (2., True)):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                self.create_attempt(root, expected, gt_peak=peak, missing_gt=missing)
                chart_dir = root/'charts'
                chart_dir.mkdir()
                row, chart = report_attempt(root, expected, template, faces, chart_dir)
                self.assertTrue(row['chart']['available'], row['chart'])
                self.assertEqual(row['qualification_passed'], peak == 2. and not missing)
                self.assertFalse((root/expected['object_id']/'object_mesh.npz').exists())
                charts.append(chart)
        for chart in charts[1:]:
            np.testing.assert_array_equal(chart, charts[0])

    def test_missing_attempt_retains_mask_zero(self):
        expected = validate_sequence_protocol(self.protocol())[0]
        template, faces = self.template()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            chart_dir = root/'charts'
            chart_dir.mkdir()
            row, chart = report_attempt(root, expected, template, faces, chart_dir)
            self.assertFalse(row['qualification_passed'])
            self.assertFalse(row['chart']['available'])
            np.testing.assert_array_equal(chart, np.zeros((25, 4)))

    def test_pack_keeps_all_slots_in_protocol_order(self):
        protocol = self.protocol()
        rows = [dict(**attempt, chart_file=str(i), chart={'available': False}, qualification_passed=False)
                for i, attempt in enumerate(validate_sequence_protocol(protocol))]
        inputs, sequences = pack_sequences(rows, np.zeros((40, 25, 4), np.float32), protocol['objects'])
        self.assertEqual(inputs.shape, (8, 5, 25, 4))
        self.assertEqual(sequences[1]['direction_indices'], [9, 19, 29, 39, 49])
        self.assertEqual(sequences[-1]['attempt_indices'], [35, 36, 37, 38, 39])
        self.assertFalse(any(s['qualification_passed'] for s in sequences))
        with self.assertRaises(ValueError):
            pack_sequences(rows[::-1], inputs.reshape(40, 25, 4), protocol['objects'])


if __name__ == '__main__':
    unittest.main()
