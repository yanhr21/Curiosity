"""CPU schema/failure-policy tests using explicit synthetic metadata fixtures.

No synthetic fixture here is presented as a physical acquisition or training
input qualification result. Actual forty-probe success remains untested until
the real collector/report complete.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.sugar.object_predictor.fixture_touch_scene import FixtureDrive
from scripts.sugar.object_predictor.report_fixture_sequences import DIRECTION_GROUPS
from scripts.sugar.object_predictor.report_fixture_touch import QUALIFICATION_CRITERIA, qualification_checks
from scripts.sugar.object_predictor.train_active3d_sugar_overfit import (
    OBJECT_IDS, load_sugar_cases, validate_qualification, verify_observation_chart,
)


def metadata_fixture():
    protocol = dict(objects=list(OBJECT_IDS), direction_groups=[list(g) for g in DIRECTION_GROUPS],
                    controls_per_attempt=600, dt=.02, substeps=8, sample_frame=499,
                    fixture_drive=asdict(FixtureDrive()))
    rows, sequences = [], []
    for obj in OBJECT_IDS:
        for group, directions in enumerate(DIRECTION_GROUPS):
            start = len(rows)
            for direction in directions:
                identity = f"object_{obj}_direction_{direction:02d}"
                actual = dict(object_id=obj, direction_index=direction, complete=True,
                    recorded_controls=600, actual_controls=600, actual_physics_substeps=4800,
                    partial_control_substeps=0, field_overflow_steps=0)
                row = dict(object_id=obj, direction_index=direction, directory=identity,
                    chart_file=f"sequence_charts/{identity}.npz", chart=dict(available=True),
                    recorded_complete_600=True, attempt_record=actual, errors=[],
                    loads=dict(initial_full_hand_frames=25, initial_full_hand_peak_n=0.,
                        peak_palmar_load_n=2., peak_full_hand_load_n=2.,
                        pre_snapshot_0p5s=dict(frames=25, target_band_fraction=1.)))
                row["qualification_checks"] = dict(qualification_checks(row), record_integrity=True)
                row["qualification_passed"] = True
                rows.append(row)
            sequences.append(dict(sequence_index=len(sequences), object_id=obj, direction_group=group,
                direction_indices=list(directions), attempt_indices=list(range(start, start+5)),
                chart_files=[r["chart_file"] for r in rows[start:start+5]], observed_chart_count=5,
                qualification_passed=True))
    result = dict(complete=True, qualification_passed=True, expected_attempts=40,
        complete_recordings=40, available_snapshot_charts=40, sequence_count=8,
        charts_per_sequence=5, snapshot_frame=499, input_file="sequence_inputs.npz",
        input_key="charts", input_shape=[8, 5, 25, 4], cases=rows, sequences=sequences,
        qualification_criteria=dict(QUALIFICATION_CRITERIA, expected_attempts=40))
    return result, protocol


class SugarInputQualificationTests(unittest.TestCase):
    def setUp(self):
        self.result, self.protocol = metadata_fixture()

    def test_complete_metadata_keeps_all_forty_and_eight(self):
        cases, sequences = validate_qualification(self.result, self.protocol)
        self.assertEqual(len(cases), 40)
        self.assertEqual(len(sequences), 8)

    def test_a_physical_failure_cannot_pass_despite_valid_chart_and_summary(self):
        row = self.result["cases"][17]
        row["loads"]["peak_full_hand_load_n"] = 100.1
        self.assertTrue(row["chart"]["available"])
        with self.assertRaises(RuntimeError):
            validate_qualification(self.result, self.protocol)
        self.assertTrue(row["chart"]["available"])  # Never erase physical-failure inputs.

    def test_recorded_controls_cannot_hide_incomplete_physics(self):
        self.result["cases"][0]["attempt_record"]["actual_physics_substeps"] = 4799
        with self.assertRaises(RuntimeError):
            validate_qualification(self.result, self.protocol)

    def test_missing_case_and_changed_sequence_are_rejected(self):
        for mutation in (lambda r: r["cases"].pop(),
                         lambda r: r["sequences"][0]["direction_indices"].reverse(),
                         lambda r: r["sequences"][0].update(observed_chart_count=4)):
            altered = copy.deepcopy(self.result)
            mutation(altered)
            with self.assertRaises(RuntimeError):
                validate_qualification(altered, self.protocol)

    def test_lowered_qualification_threshold_rejected(self):
        self.result["qualification_criteria"]["minimum_target_band_fraction"] = .5
        with self.assertRaises(RuntimeError):
            validate_qualification(self.result, self.protocol)

    def test_foreign_or_reordered_object_cannot_replace_failed_probe(self):
        self.result["cases"][0]["object_id"] = "11898"
        with self.assertRaises(RuntimeError):
            validate_qualification(self.result, self.protocol)

    def test_failure_rejected_before_input_labels_or_torch(self):
        self.result["qualification_passed"] = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"RESULT.json").write_text(json.dumps(self.result))
            (root/"PROTOCOL.json").write_text(json.dumps(self.protocol))
            with self.assertRaisesRegex(RuntimeError, "must qualify"):
                load_sugar_cases(root, root/"nonexistent_GT_labels")
            # Other tests may legitimately import Torch. Verify the actual
            # qualification-before-import guarantee in a fresh interpreter.
            check = subprocess.run([sys.executable, '-c', '''
import sys
from pathlib import Path
from scripts.sugar.object_predictor.train_active3d_sugar_overfit import load_sugar_cases
root = Path(sys.argv[1])
try:
    load_sugar_cases(root, root/'nonexistent_GT_labels')
except RuntimeError as error:
    assert 'must qualify' in str(error)
else:
    raise AssertionError('Failed qualification unexpectedly loaded')
assert 'torch' not in sys.modules
''', str(root)], capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)

    def test_numeric_barycentric_measured_pose_and_public_frame_replay(self):
        # Pure array unit fixture, not a simulated or measured physical probe.
        points = np.array([[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.]])
        xy = np.array([(x, y) for x in np.linspace(.05, .4, 5) for y in np.linspace(.05, .4, 5)])
        bary = np.column_stack((1.-xy.sum(1), xy))
        indices = np.tile(np.arange(3), (25, 1))
        hand = bary @ points
        rot = Rotation.from_euler('z', 25., degrees=True)
        pose = np.r_[[.02, -.01, .75], rot.as_quat()][None]
        world = rot.apply(hand)+pose[0, :3]
        canonical = (world-np.array([0., 0., .75]))/.93
        chart = np.column_stack((canonical, np.full(25, 2))).astype(np.float32)[None]
        arrays = dict(chart=chart, source_pos_hand_m=points, source_frame_indices=indices,
            barycentric=bary, source_active=np.ones(3, bool), source_pad=np.full(3, 7),
            selected_pad=np.int64(7), source_area_m2=np.full(3, 1e-6),
            source_pressure_pa=np.full(3, 1000.), hand_pose_w=pose, chart_hand_m=hand,
            chart_world_m=world, support_edge_m=np.full(25, np.sqrt(2)*.001))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, trace = root/'chart.npz', root/'trace.npz'
            np.savez_compressed(path, **arrays)
            np.savez_compressed(trace, hand_pose_w=np.repeat(pose, 600, axis=0),
                                time_s=np.arange(1, 601)*.02)
            result = verify_observation_chart(path, trace, chart[0])
            self.assertLess(result['canonical_roundtrip_max'], 1e-7)
            bad = dict(arrays, chart_world_m=world+.001)
            np.savez_compressed(path, **bad)
            with self.assertRaisesRegex(RuntimeError, 'canonical transform'):
                verify_observation_chart(path, trace, chart[0])
            np.savez_compressed(path, **arrays)
            np.savez_compressed(trace, hand_pose_w=np.repeat(pose, 600, axis=0)+np.array([.001,0,0,0,0,0,0]),
                                time_s=np.arange(1, 601)*.02)
            with self.assertRaisesRegex(RuntimeError, 'actual measured'):
                verify_observation_chart(path, trace, chart[0])


if __name__ == "__main__":
    unittest.main()
