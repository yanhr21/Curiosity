"""CPU-only protocol/saved-readback checks for the separate four-case fixture."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from .run_canonical_approach_fixture_pilot import (EPISODES,aggregate,collection_command,
    protocol,validate_protocol,validate_saved_case)


BASELINE=Path('experiments/object_predictor_v1/overfit_repair_v1/freeze_postlift_alignment_pilot_v1/cases/episode_5000')


class CanonicalFixtureTests(unittest.TestCase):
    def test_only_approach_changes_and_replacement_status_is_truthful(self):
        p=protocol();validate_protocol(p)
        self.assertEqual(tuple(c['episode'] for c in p['configurations']),EPISODES)
        for original,executed,status in zip(p['original_configurations'],p['configurations'],p['requested_approach_evaluation']):
            self.assertEqual(executed,dict(original,approach_angle_deg=0.))
            self.assertEqual(status['requested_approach_status'],
                'UNCHANGED_EXECUTED' if original['approach_angle_deg']==0 else 'NOT_EXECUTED_REPLACED')
        self.assertEqual(p['late_mass_check']['frame'],2381)

    def test_unrelated_configuration_threshold_and_case_changes_are_rejected(self):
        for key in ('mass','scale','load','seed','yaw_delta_deg','lift_height_m','lateral_xy'):
            p=protocol();p['configurations'][0][key]=None
            with self.assertRaises(ValueError,msg=key):validate_protocol(p)
        for key,value in [('controls_per_case',1200),('physics_substeps',16),('planned_cases',16)]:
            p=protocol();p[key]=value
            with self.assertRaises(ValueError,msg=key):validate_protocol(p)
        p=protocol();p['late_mass_check']['criteria']['clearance_m']=.001
        with self.assertRaises(ValueError):validate_protocol(p)

    def test_collection_command_selects_freeze_and_original_physics_budget(self):
        p=protocol();cmd=collection_command(Path('/synthetic/prepared'),p['configurations'][0])
        for flag,value in [('--frames','2400'),('--controller-intervention','freeze_postlift_alignment_v1'),
                           ('--controller-revision','sensor_feedback_v2'),('--purpose','diagnostic')]:
            self.assertEqual(cmd[cmd.index(flag)+1],value)
        self.assertIn('--response-gain',cmd)

    def test_actual_saved_success_reuses_schema_and_true_com(self):
        p=protocol();row=validate_saved_case(BASELINE,p['configurations'][-1],p)
        self.assertTrue(row['complete']);self.assertTrue(row['physical_passed'])
        self.assertLessEqual(row['actual_com_transform_max_error_m'],1e-12)
        self.assertEqual(row['late_mass_window']['first_frame'],2350)
        self.assertEqual(row['late_mass_window']['frame'],2381)

    def test_corrupt_com_fails_and_grounded_late_mass_cannot_pass(self):
        p=protocol()
        with np.load(BASELINE/'episode_5000.npz') as source:
            arrays={k:source[k].copy() for k in source.files}
        bad=deepcopy(arrays);bad['object_com_w'][100,0]+=1e-5
        with patch('scripts.sugar.object_predictor.run_canonical_approach_fixture_pilot.np.load',return_value=nullcontext(bad)):
            with self.assertRaisesRegex(ValueError,'Saved COM'):validate_saved_case(BASELINE,p['configurations'][-1],p)
        arrays['validation_full_mesh_min_z_m'][2350:2382]=0.
        with patch('scripts.sugar.object_predictor.run_canonical_approach_fixture_pilot.np.load',return_value=nullcontext(arrays)):
            row=validate_saved_case(BASELINE,p['configurations'][-1],p)
        self.assertTrue(row['physical_passed'])
        self.assertFalse(row['late_mass_available']);self.assertFalse(row['passed'])

    def test_missing_and_failed_cases_remain_in_denominator(self):
        p=protocol();ok=dict(complete=True,passed=True)
        self.assertFalse(aggregate([ok]*3,p,{})['qualification_passed'])
        rows=[ok,dict(complete=True,passed=False),ok,ok]
        result=aggregate(rows,p,{})
        self.assertTrue(result['complete']);self.assertFalse(result['qualification_passed'])
        self.assertEqual(len(result['cases']),4)
        self.assertFalse(result['original_requested_approach_regression_repaired'])
        self.assertFalse(result['automatic_expansion_or_training'])


if __name__=='__main__':unittest.main()
