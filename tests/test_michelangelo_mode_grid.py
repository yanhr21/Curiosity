"""CPU-only posterior glue and fixed-rule tests; no substitute neural model."""
import ast
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.sugar.object_predictor import michelangelo_mode_grid_data as sampling
from scripts.sugar.object_predictor import train_michelangelo_mode_grid as trainer
from scripts.sugar.object_predictor import train_michelangelo_dynamic_grid as previous


class ModeGridTests(unittest.TestCase):
    def test_old_sources_and_defaults_unchanged(self):
        expected = {
            'train_michelangelo_overfit.py': 'ff3eeaee86ebad8c18cc9aa1b19b4666a6509bed828c9c3f300e3ae06d3df0e0',
            'train_michelangelo_dynamic_grid.py': '9d4ea5ca9dc344b736b25d1acc36ab6684ec93deb0cdb0d72b0affd5ee047273',
            'michelangelo_dynamic_grid_data.py': '7d4e836391c0255b47c462faf2a3779bf8b3e4aa632a565d9c6464e6f70ffafd',
        }
        for name, sha in expected.items():
            self.assertEqual(sampling.digest(Path(trainer.__file__).with_name(name)), sha)
        source=ast.parse(inspect.getsource(sampling.original.batch_loss))
        flags=[k.value.value for n in ast.walk(source) if isinstance(n,ast.Call)
               for k in n.keywords if k.arg=='sample_posterior']
        self.assertEqual(flags,[True])
        new=ast.parse(inspect.getsource(trainer.batch_loss))
        flags=[k.value.value for n in ast.walk(new) if isinstance(n,ast.Call)
               for k in n.keywords if k.arg=='sample_posterior']
        self.assertEqual(flags,[False])

    def test_formal_loop_and_refresh_are_same_except_declared_flag_and_metadata(self):
        class Normalize(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='batch_loss':
                    return ast.Attribute(value=ast.Name(id='original',ctx=ast.Load()),
                                         attr='batch_loss',ctx=node.ctx)
                return node
            def visit_Constant(self,node):
                if node.value=='MODE_GRID_UPDATE':node.value='DYNAMIC_GRID_UPDATE'
                if node.value=='MODE_GRID_REFRESH':node.value='DYNAMIC_GRID_REFRESH'
                return node
            def visit_Call(self,node):
                self.generic_visit(node)
                node.keywords=[k for k in node.keywords if k.arg not in
                    ('task_adaptation','only_training_change','dynamic_pool_note')]
                for k in node.keywords:
                    if k.arg=='train_posterior':k.value=ast.Constant(value='sample')
                return node
        for name in ('run','refresh','released_model','refresh_context'):
            old=ast.parse(inspect.getsource(getattr(previous,name)))
            new=Normalize().visit(ast.parse(inspect.getsource(getattr(trainer,name))))
            self.assertEqual(ast.dump(new,include_attributes=False),ast.dump(old,include_attributes=False))

    def test_real_fixed_pool_all1000_sampling_and_near_match(self):
        manifest,queries,cases,_,_=sampling.load_sources()
        for name in ('fixed_recipe','error_pools','sample_volume','batch_for_step'):
            self.assertIs(getattr(sampling,name),getattr(sampling.dynamic,name))
        pools=[]
        for case in cases:
            file=sampling.PREVIOUS_RUN/'grid_refresh/step_0900'/(case['object_id']+'.npz')
            with np.load(file,allow_pickle=False) as z:
                pools.append(sampling.error_pools(z['grid_logits'],case))
        counts=[0]*4
        for step in range(1,1001):
            index,item,indices,volume,near,info=sampling.batch_for_step(step,queries,cases,pools)
            oldidx,oldidxs=sampling.original.train_indices(step)
            self.assertEqual(index,oldidx);counts[index]+=1
            np.testing.assert_array_equal(near,oldidxs[1024:])
            oldvol,oldinfo=sampling.dynamic.sample_volume(step,cases[index],pools[index])
            np.testing.assert_array_equal(volume,oldvol);self.assertEqual(info,oldinfo)
            for key in ('input_surface_xyz_vae','input_normals','eval_queries_vae','eval_labels'):
                self.assertIs(item[key],cases[index]['original'][key])
            np.testing.assert_array_equal(item['train_queries_vae'][1024:],cases[index]['original']['train_queries_vae'][near])
            np.testing.assert_array_equal(item['train_labels'][1024:],cases[index]['original']['train_labels'][near])
        self.assertEqual(counts,[250]*4)

    def test_full_forward_glue_keeps_inputs_and_original_criterion(self):
        # A call recorder, not a substitute model or inference result.
        calls=[];sentinel=object();loss_marker=object()
        def record_forward(*args,**kwargs):
            calls.append((args,kwargs));return None,sentinel,sentinel
        def record_loss(posterior,logits,labels):
            self.assertIs(posterior,sentinel);self.assertIs(logits,sentinel)
            np.testing.assert_array_equal(labels.numpy(),[[1,0]])
            return loss_marker,{}
        item=dict(input_surface_xyz_vae=np.arange(12,dtype=np.float32).reshape(4,3),
                  input_normals=np.ones((4,3),np.float32),train_queries_vae=np.zeros((3,3),np.float32),
                  train_labels=np.array([0,1,0],np.float32))
        value,_=trainer.batch_loss(record_forward,record_loss,item,np.array([1,2]),'cpu')
        self.assertIs(value,loss_marker);self.assertEqual(calls[0][1],dict(sample_posterior=False))
        for actual,key in zip(calls[0][0],('input_surface_xyz_vae','input_normals')):
            np.testing.assert_array_equal(actual.numpy()[0],item[key])
        self.assertEqual(tuple(calls[0][0][2].shape),(1,2,3))

    def test_official_posterior_mode_retains_original_kl_logvar_gradient(self):
        import torch
        criterion=sampling.original.original_loss()
        from michelangelo.models.modules.distributions import DiagonalGaussianDistribution
        # Actual official distribution and loss, small numeric tensors only.
        mean=torch.linspace(-.3,.6,24).reshape(1,4,6).requires_grad_()
        logvar=torch.full_like(mean,-.7,requires_grad=True)
        posterior=DiagonalGaussianDistribution([mean,logvar])
        posterior.mean.retain_grad();posterior.logvar.retain_grad()
        before=torch.get_rng_state().clone()
        mode=posterior.mode();self.assertIs(mode,posterior.mean)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        logits=mode.reshape(1,24).repeat(1,86)[:,:2048]
        labels=(torch.arange(2048)%2).reshape(1,2048).float()
        loss,_=criterion(posterior,logits,labels);loss.backward()
        record=trainer.posterior_gradient_check(posterior,criterion.kl_weight)
        self.assertTrue(record['passed']);self.assertGreater(record['logvar_gradient_norm'],0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_qualification_requires_four_full_backwards_and_posterior_gradient(self):
        bound={'binding':'same'};resource={'resource':'same'}
        good=dict(scope=trainer.SCOPE+'_qualification',passed=True,bindings=bound,
                  recipe=sampling.fixed_recipe(),resource=resource,optimizer_updates=0,
                  actual_full_model_backwards=4,cases=[dict(object_id=oid,passed=True,
                  gradients={'passed':True},posterior_gradients={'passed':True}) for oid in sampling.OBJECT_IDS])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);file=path/'RESULT.json';file.write_text(json.dumps(good))
            trainer.check_qualification(path,bound,resource)
            for patch in ({'optimizer_updates':1},{'actual_full_model_backwards':3},
                          {'cases':good['cases'][:3]},{'scope':previous.SCOPE+'_qualification'}):
                file.write_text(json.dumps(dict(good,**patch)))
                with self.assertRaises(ValueError):trainer.check_qualification(path,bound,resource)
            bad=copy.deepcopy(good);bad['cases'][2]['posterior_gradients']['passed']=False
            file.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):trainer.check_qualification(path,bound,resource)


if __name__=='__main__':unittest.main()
