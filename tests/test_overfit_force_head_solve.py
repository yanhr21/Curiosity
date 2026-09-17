"""Small algebra/interface fixtures only; never a replacement prediction model."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.sugar.object_predictor import overfit_force_head_solve as solve
from scripts.sugar.object_predictor import train_overfit_force_head_solve as runner


class ForceSolveTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[1,0,0],[0,1,0],[1,0,0],[0,1,0]],dtype=np.float32)
        self.w=np.arange(24,dtype=np.float32).reshape(8,3)/10
        self.b=np.arange(8,dtype=np.float32)/100
        self.y=np.array([[2]*8,[4]*8,[2]*8,[4]*8],dtype=np.float32)

    def test_minimum_increment_fits_and_preserves_original_nullspace(self):
        w,b,r=solve.minimum_increment(self.x,self.y,self.w,self.b)
        np.testing.assert_allclose(self.x@w.T+b,self.y,rtol=0,atol=1e-6)
        self.assertEqual(r['rank'],2)
        np.testing.assert_array_equal(w[:,2],self.w[:,2])
        a=np.c_[self.x,np.ones(4)]
        delta=np.c_[w-self.w,b-self.b]
        self.assertLess(np.linalg.norm(delta@(np.eye(4)-np.linalg.pinv(a)@a)),1e-6)
        self.assertEqual(r['learning_updates'],1);self.assertEqual(r['optimizer_updates'],0)

    def test_duplicate_conflicting_targets_preserved_as_nonzero_residual(self):
        y=self.y.copy();y[2]+=3
        _,_,r=solve.minimum_increment(self.x,y,self.w,self.b)
        self.assertGreater(r['solve_residual_rmse_n'],1.)
        self.assertEqual(r['rank'],2)  # No data deletion or altered tolerance.

    def test_solver_is_not_replacement_minimum_norm_weight(self):
        w,b,_=solve.minimum_increment(self.x,self.y,self.w,self.b)
        a=np.c_[self.x,np.ones(4)]
        old=np.c_[self.w,self.b]
        expected=old+(np.linalg.pinv(a)@(self.y-a@old.T)).T
        np.testing.assert_allclose(np.c_[w,b],expected,rtol=0,atol=3e-7)
        self.assertGreater(abs(w[:,2]).max(),0)

    def test_nonfinite_and_float32_overflow_rejected(self):
        x=self.x.copy();x[0,0]=np.nan
        with self.assertRaises(ValueError):solve.minimum_increment(x,self.y,self.w,self.b)
        with self.assertRaises(ValueError):solve.minimum_increment(self.x.astype(float),self.y,self.w,self.b)
        y=np.full_like(self.y,1e100,dtype=np.float64)
        with self.assertRaises(FloatingPointError):solve.minimum_increment(self.x,y,self.w,self.b)

    def test_only_existing_force_rows_change_and_hash_guard_catches_other_row(self):
        head=torch.nn.Linear(3,10)
        model=SimpleNamespace(auxiliary=head)
        before={k:v.detach().clone() for k,v in head.state_dict().items()}
        w,b,_=solve.minimum_increment(self.x,self.y,self.w,self.b)
        solve.apply_force_rows(model,w,b)
        np.testing.assert_array_equal(head.weight[8:].detach(),before['weight'][8:])
        np.testing.assert_array_equal(head.bias[8:].detach(),before['bias'][8:])
        old={'auxiliary.'+k:v for k,v in before.items()};new={'auxiliary.'+k:v for k,v in head.state_dict().items()}
        self.assertTrue(solve.preserved_state(old,new)['passed'])
        new=copy.deepcopy(new);new['auxiliary.weight'][8,0]+=1
        with self.assertRaises(ValueError):solve.preserved_state(old,new)

    def test_bit_hash_detects_signed_zero_outside_force_rows(self):
        old={'backbone.fixture':torch.tensor([0.])}
        new={'backbone.fixture':torch.tensor([-0.])}
        self.assertTrue(torch.equal(old['backbone.fixture'],new['backbone.fixture']))
        with self.assertRaises(ValueError):solve.preserved_state(old,new)

    def test_same_cached_features_requires_nonforce_exact(self):
        before=dict(state=np.zeros((4,13),np.float32),auxiliary=np.zeros((4,10),np.float32))
        after=copy.deepcopy(before);after['auxiliary'][:,:8]=self.y
        report=runner.same_feature_check(before,after,self.y)
        self.assertTrue(report['force_every_item_gate']);self.assertEqual(report['force_rmse_n'],0)
        after['auxiliary'][0,8]=1e-5
        with self.assertRaises(RuntimeError):runner.same_feature_check(before,after,self.y)

    def test_capture_hook_cleanup_and_original_evaluator_only(self):
        # Test fixture counts unchanged ordinary evaluator calls, not a model.
        head=torch.nn.Linear(3,13)
        model=SimpleNamespace(predictor=SimpleNamespace(head=head))
        dataset=[0,1]
        def evaluation(_model,_dataset,_vertices,_device):
            self.assertIs(_model,model);self.assertIs(_dataset,dataset)
            head(torch.ones(1,3));head(torch.ones(1,3)*2)
            return {'passed':False},{'force_target_n':np.zeros((2,8),np.float32)}
        with patch.object(runner.original,'evaluate',side_effect=evaluation):
            _,_,x=runner.evaluate_capture(model,dataset,None,'cpu')
        self.assertEqual(x.shape,(2,3));self.assertEqual(len(head._forward_pre_hooks),0)
        with patch.object(runner.original,'evaluate',side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):runner.evaluate_capture(model,dataset,None,'cpu')
        self.assertEqual(len(head._forward_pre_hooks),0)

    def test_forward_boundary_never_passes_targets_or_masks(self):
        inputs={k:object() for k in ('coord','grid_coord','feat','offset')}
        observed=[]
        def model(value):observed.append(value);return {}
        runner.capture.forward_observations(model,{'inputs':inputs,'target':'secret','mass_available':'secret'})
        self.assertIs(observed[0],inputs)
        with self.assertRaises(ValueError):runner.capture.forward_observations(model,{'inputs':dict(inputs,target='secret')})

    def test_full_forward_counter_excludes_direct_child_and_cleans_on_error(self):
        # A test-only module fixture; no Utonia construction/forward.
        model=torch.nn.Sequential(torch.nn.Linear(3,2))
        with runner.FullForwardCounter(model) as counter:
            model(torch.ones(1,3));model[0](torch.ones(80,3))
            self.assertEqual(counter.calls,1)
            model(torch.ones(1,3));self.assertEqual(counter.calls,2)
        self.assertEqual(len(model._forward_hooks),0)
        with self.assertRaises(RuntimeError):
            with runner.FullForwardCounter(model):raise RuntimeError('fixture')
        self.assertEqual(len(model._forward_hooks),0)


if __name__=='__main__':unittest.main()
