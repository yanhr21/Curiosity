"""CPU algebra/interface qualification only; no official model is executed."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.sugar.object_predictor.audit_overfit_readout_features import (
    feature_layout, forward_observations, linear_span_diagnostic, replay_comparison, residual_statistics)


class ReadoutFeatureDiagnosticTests(unittest.TestCase):
    def test_actual_checkpoint_shape_derives_width_and_rejects_wrong_aux_or_config(self):
        def state(width,aux_width=None):
            return {'predictor.head.weight':SimpleNamespace(shape=(13,width)),
                'predictor.head.bias':SimpleNamespace(shape=(13,)),
                'auxiliary.weight':SimpleNamespace(shape=(10,width if aux_width is None else aux_width)),
                'auxiliary.bias':SimpleNamespace(shape=(10,))}
        channels=[54,108,216,432,576]
        actual=feature_layout(state(88704),32,channels)
        self.assertEqual(actual['feature_dimension'],88704)
        self.assertEqual(actual['frame_feature_dimension'],2772)
        for wrong in (state(88576),state(88704,88576)):
            with self.assertRaises(ValueError):feature_layout(wrong,32,channels)
        with self.assertRaises(ValueError):feature_layout(state(88704),32,[54,108,216,432,574])
        # Derivation is from actual metadata, not a corrected hardcoded constant.
        self.assertEqual(feature_layout(state(640),32,[4,6])['feature_dimension'],640)

    def test_forward_passes_only_original_inputs_never_batch_labels(self):
        inputs={key:object() for key in ('coord','grid_coord','feat','offset')}
        batch=dict(inputs=inputs,target=object(),mass_available=object(),physics=object(),
                   state_precision_eligible=object(),evaluation_only=object(),metadata=object())
        output=object()
        calls=[]
        def original_model_boundary(actual):
            self.assertIs(actual,inputs)
            self.assertEqual(set(actual),{'coord','grid_coord','feat','offset'})
            calls.append(actual)
            return output
        self.assertIs(forward_observations(original_model_boundary,batch),output)
        self.assertEqual(len(calls),1)
        for bad_inputs in (dict(inputs,target=object()),{k:v for k,v in inputs.items() if k!='offset'}):
            with self.assertRaises(ValueError):
                forward_observations(original_model_boundary,dict(batch,inputs=bad_inputs))
        self.assertEqual(len(calls),1)

    def test_strict_replay_rejects_even_one_float32_ulp(self):
        saved=dict(prediction=np.ones((80,13),np.float32),
            force_prediction_n=np.zeros((80,8),np.float32),
            availability_probability=np.full(80,.5),contact_probability=np.full(80,.5))
        actual=deepcopy(saved)
        self.assertTrue(replay_comparison(actual,saved)['passed'])
        actual['prediction'][71,2]=np.nextafter(np.float32(1),np.float32(2))
        report=replay_comparison(actual,saved)
        self.assertFalse(report['passed'])
        self.assertGreater(report['max_abs_difference']['prediction'],0)
        actual=deepcopy(saved);actual['force_prediction_n'][0,0]=np.nan
        with self.assertRaises(ValueError):replay_comparison(actual,saved)

    def test_affine_fit_is_diagnostic_and_does_not_return_coefficients(self):
        x=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]],np.float32)
        y=x@np.array([[2.,3.,1.],[-1.,4.,2.]])+np.array([5.,6.,7.])
        report,r64,r32=linear_span_diagnostic(x,y,np.ones(4,bool),'center')
        self.assertEqual(report['status'],'DIAGNOSTIC_ONLY')
        self.assertEqual(report['rank_float64'],3)
        self.assertIsNone(report['full_row_condition'])
        np.testing.assert_allclose(r64,0,atol=1e-12)
        np.testing.assert_allclose(r32,0,atol=1e-5)
        self.assertNotIn('coefficients',report)

    def test_identical_features_conflicting_targets_cannot_be_fit(self):
        x=np.ones((4,6),np.float32);y=np.array([[0.],[1.],[2.],[3.]])
        report,r64,_=linear_span_diagnostic(x,y,np.ones(4,bool),'logmass')
        self.assertEqual(report['rank_float64'],1)
        self.assertIsNone(report['full_row_condition'])
        self.assertGreater(report['float64_residual']['component_rmse'],1.)
        self.assertAlmostEqual(float(r64.sum()),0.)

    def test_masked_targets_do_not_enter_solver_and_empty_mask_is_na(self):
        x=np.array([[0.],[1.],[2.],[3.]],np.float32);y=np.arange(4.)[:,None]
        mask=np.array([1,1,0,0]);base,r64,_=linear_span_diagnostic(x,y,mask,'force')
        y[2:]=np.nan
        changed,other,_=linear_span_diagnostic(x,y,mask,'force')
        np.testing.assert_array_equal(r64,other)
        self.assertEqual(base,changed)
        empty,a,b=linear_span_diagnostic(x,y,np.zeros(4,bool),'force')
        self.assertEqual(empty,dict(status='NOT_APPLICABLE',rows=0))
        self.assertEqual(a.shape,(0,1));self.assertEqual(b.shape,(0,1))

    def test_ill_conditioning_is_reported_without_pretending_full_rank_confidence(self):
        x=np.array([[0.,0.],[1.,0.],[1.,1e-10]],np.float32)
        y=np.array([[0.],[0.],[1.]])
        report,_,_=linear_span_diagnostic(x,y,np.ones(3,bool),'force')
        self.assertEqual(report['rank_float64'],3)
        self.assertGreater(report['full_row_condition'],1e9)
        self.assertLess(report['rank_float32_roundoff_indicator'],report['rank_float64'])
        self.assertGreater(report['coefficient_max_abs'],1e9)

    def test_metric_units_and_overflow_are_explicit(self):
        r=residual_statistics(np.array([[.03,.04,0.]]),'center')
        self.assertAlmostEqual(r['center_error_cm']['mean'],5.)
        r=residual_statistics(np.array([[np.log(1.05)]]),'logmass')
        self.assertAlmostEqual(r['relative_error']['mean'],.05)
        r=residual_statistics(np.array([[1000.]]),'logsize')
        self.assertFalse(r['relative_error_finite']);self.assertIsNone(r['relative_error'])


if __name__=='__main__':unittest.main()
