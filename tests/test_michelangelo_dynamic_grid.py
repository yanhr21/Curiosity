"""CPU sampler/RNG/qualification checks; no substitute model or model forward."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scripts.sugar.object_predictor import michelangelo_dynamic_grid_data as sampling
from scripts.sugar.object_predictor import train_michelangelo_dynamic_grid as trainer


def numeric_case():
    labels=np.arange(1600)%2==0;eligible=np.ones(1600,bool);eligible[:2]=False
    return dict(labels=labels,eligible=eligible,eligible_indices=np.flatnonzero(eligible),
                interior_indices=np.flatnonzero(eligible&labels),exterior_indices=np.flatnonzero(eligible&~labels))


class DynamicGridTests(unittest.TestCase):
    def test_false_sign_pools_exclude_boundary_and_direction_is_correct(self):
        case=numeric_case();logits=np.where(case['labels'],1.,-1.)
        logits[0:8]*=-1
        pools=sampling.error_pools(logits,case)
        np.testing.assert_array_equal(pools['fp'],[3,5,7])
        np.testing.assert_array_equal(pools['fn'],[2,4,6])
        indices,record=sampling.sample_volume(5,case,pools)
        self.assertTrue(record['fp_replacement']);self.assertTrue(record['fn_replacement'])
        self.assertTrue(np.isin(indices[512:768],pools['fp']).all())
        self.assertTrue(np.isin(indices[768:],pools['fn']).all())
        self.assertEqual(len(np.unique(indices[:512])),512)
        self.assertTrue(case['eligible'][indices].all())

    def test_empty_pools_fall_back_to_same_gt_class_and_sampling_is_deterministic(self):
        case=numeric_case();empty={'fp':np.empty(0,np.int64),'fn':np.empty(0,np.int64)}
        indices,record=sampling.sample_volume(9,case,empty)
        self.assertFalse(case['labels'][indices[512:768]].any())
        self.assertTrue(case['labels'][indices[768:]].all())
        self.assertEqual(record['fp_source'],'gt_class_fallback')
        self.assertEqual(record['fn_source'],'gt_class_fallback')
        self.assertFalse(record['fp_replacement']);self.assertFalse(record['fn_replacement'])
        np.testing.assert_array_equal(indices,sampling.sample_volume(9,case,empty)[0])
        bad={'fp':np.array([2]),'fn':np.array([3])}
        with self.assertRaises(ValueError):sampling.sample_volume(9,case,bad)

    def test_all1000_actual_near_batches_and_encoder_inputs_unchanged(self):
        _,queries,cases,_,_=sampling.load_sources()
        # Existing real saved logits exercise nonempty small/large error pools;
        # this CPU test does not infer fresh logits or claim a new model result.
        pools=[]
        root=sampling.original.EXPERIMENT/'overfit_repair_v1/michelangelo_extraction_sign_audit_v1'
        for case in cases:
            with np.load(root/(case['object_id']+'.npz'),allow_pickle=False) as z:
                pools.append(sampling.error_pools(z['grid_logits'],case))
        counts=[0]*4
        for step in range(1,1001):
            idx,item,indices,volume,near,info=sampling.batch_for_step(step,queries,cases,pools)
            oldidx,old=sampling.original.train_indices(step);self.assertEqual(idx,oldidx);counts[idx]+=1
            np.testing.assert_array_equal(near,old[1024:])
            original=cases[idx]['original']
            for key in ('input_surface_xyz_vae','input_normals','eval_queries_vae','eval_labels'):
                self.assertIs(item[key],original[key])
            np.testing.assert_array_equal(item['train_queries_vae'][1024:],original['train_queries_vae'][near])
            np.testing.assert_array_equal(item['train_labels'][1024:],original['train_labels'][near])
            np.testing.assert_array_equal(item['train_queries_vae'][:1024],queries[volume])
            np.testing.assert_array_equal(item['train_labels'][:1024],cases[idx]['labels'][volume])
            np.testing.assert_array_equal(indices,np.arange(2048))
        self.assertEqual(counts,[250]*4)
        self.assertEqual(sampling.REFRESH_STEPS,tuple(range(0,1000,100)))

    def test_refresh_context_restores_cpu_rng_mode_and_exception(self):
        import torch
        # State-only test double: no neural architecture or inference.
        class ModeState:
            training=True
            def eval(self):self.training=False
            def train(self,value=True):self.training=value
        model=ModeState();before=torch.get_rng_state().clone()
        with trainer.refresh_context(model,torch.device('cpu')):
            self.assertFalse(model.training);self.assertFalse(torch.is_grad_enabled())
            torch.rand(17)
        self.assertTrue(model.training);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        with self.assertRaises(RuntimeError):
            with trainer.refresh_context(model,torch.device('cpu')):
                torch.rand(23);raise RuntimeError('test cleanup')
        self.assertTrue(model.training);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_qualification_cannot_be_short_changed_or_contain_updates(self):
        bound={'test_binding':'same'};resource={'test_resource':'same'}
        good=dict(scope=trainer.SCOPE+'_qualification',passed=True,bindings=bound,
                  recipe=sampling.fixed_recipe(),resource=resource,optimizer_updates=0,
                  actual_full_model_backwards=4,cases=[dict(object_id=oid,passed=True,gradients={'passed':True})
                                                     for oid in sampling.OBJECT_IDS])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);file=path/'RESULT.json';file.write_text(json.dumps(good))
            trainer.check_qualification(path,bound,resource)
            for patch in ({'optimizer_updates':1},{'actual_full_model_backwards':3},{'cases':good['cases'][:3]},
                          {'bindings':{'test_binding':'wrong'}}):
                file.write_text(json.dumps(dict(good,**patch)))
                with self.assertRaises(ValueError):trainer.check_qualification(path,bound,resource)


if __name__=='__main__':unittest.main()
