"""CPU-only saved-array and matrix-boundary checks; no model is instantiated."""
from copy import deepcopy
import unittest

import numpy as np

from scripts.sugar.object_predictor.analyze_captured_readout_span import (
    cpu_affine_outputs,descriptive_differences,validate_capture)
from scripts.sugar.object_predictor.audit_overfit_readout_features import replay_comparison


class CapturedSpanTests(unittest.TestCase):
    def fixture(self):
        declared=dict(fixed_episodes=list(range(16)),fixed_frames=list(range(5)),feature_dimension=4)
        original=dict(episode=np.repeat(np.arange(16),5),frame=np.tile(np.arange(5),16),
            target=np.zeros((80,13),np.float32),prediction=np.zeros((80,13),np.float32),
            mass_available=np.zeros(80),state_precision_eligible=np.zeros(80),
            state_contact_history_frames=np.zeros(80,dtype=np.int64),
            force_target_n=np.zeros((80,8),np.float32),force_prediction_n=np.zeros((80,8),np.float32),
            availability_probability=np.full(80,.5),contact_probability=np.full(80,.5))
        captured=deepcopy(original)
        captured['features']=np.ones((80,4),np.float32)
        captured['prediction'][0,0]=1e-6
        return declared,original,captured,replay_comparison(captured,original)

    def test_preserves_all_eighty_and_original_strict_failure(self):
        p,o,c,r=self.fixture()
        result=validate_capture(c,o,p,r)
        self.assertEqual(result['rows'],80)
        self.assertFalse(result['original_strict_replay_passed'])
        with self.assertRaises(ValueError):validate_capture(c,o,p,dict(r,passed=True))
        c['frame'][0]=4
        with self.assertRaises(ValueError):validate_capture(c,o,p,r)

    def test_changed_target_mask_or_feature_shape_rejected(self):
        p,o,c,r=self.fixture()
        for key in ('target','mass_available','state_precision_eligible','state_contact_history_frames'):
            bad=deepcopy(c);bad[key].flat[0]=1
            with self.assertRaises(ValueError):validate_capture(bad,o,p,r)
        bad=deepcopy(c);bad['features']=bad['features'][:,:3]
        with self.assertRaises(ValueError):validate_capture(bad,o,p,r)

    def test_original_matrix_readback_is_descriptive_not_exact_gate(self):
        x=np.arange(12,dtype=np.float32).reshape(3,4)
        w=np.zeros((13,4),np.float32);w[0,0]=2
        a=np.zeros((10,4),np.float32);a[0,1]=3
        output=cpu_affine_outputs(x,w,np.ones(13),a,np.zeros(10),np.float64)
        np.testing.assert_array_equal(output['prediction'][:,0],2*x[:,0]+1)
        np.testing.assert_array_equal(output['force_prediction_n'][:,0],3*x[:,1])
        np.testing.assert_array_equal(output['availability_probability'],.5)
        reference=deepcopy(output);reference['prediction'][0,0]+=.002
        difference=descriptive_differences(output,reference)
        self.assertAlmostEqual(difference['center_delta_max_cm'],.2)
        self.assertNotIn('passed',difference)
        with self.assertRaises(ValueError):cpu_affine_outputs(x,w[:,:3],np.ones(13),a,np.zeros(10),np.float32)


if __name__=='__main__':unittest.main()
