"""CPU numeric/state/counter tests, never a substitute model or GPU experiment."""
import ast
import copy
import inspect
import unittest

import numpy as np

from scripts.sugar.object_predictor import diagnose_michelangelo_adam as diagnostic


class AdamDiagnosticTests(unittest.TestCase):
    def test_actual_fixed_batch_provenance_and_original_near(self):
        fixed,original_items,_,sources=diagnostic.load_sources()
        self.assertEqual([c['object_id'] for c in fixed],list(diagnostic.OBJECT_IDS))
        self.assertIn('michelangelo_mode_grid_overfit_v1',sources['source'])
        for case,item in zip(fixed,original_items):
            arrays=diagnostic.fixed_arrays(case)
            self.assertEqual(arrays['queries_vae'].shape,(2048,3))
            self.assertEqual(arrays['labels'].shape,(2048,))
            _,indices=diagnostic.original.train_indices(case['seed_step'])
            np.testing.assert_array_equal(case['near'],indices[1024:])
            np.testing.assert_array_equal(arrays['queries_vae'][1024:],item['train_queries_vae'][case['near']])
            np.testing.assert_array_equal(arrays['labels'][1024:],item['train_labels'][case['near']])
            self.assertIs(case['item']['input_surface_xyz_vae'],item['input_surface_xyz_vae'])
            self.assertIs(case['item']['input_normals'],item['input_normals'])

    def test_gradient_inner_products_and_actual_displacements(self):
        import torch
        # Numeric vectors, not a neural architecture.
        vectors=np.array([[1,2,-1],[-1,-2,1],[2,0,1],[0,1,3]],np.float32)
        gradients=[{'a':torch.tensor(v[:2]),'b':torch.tensor(v[2:])} for v in vectors]
        dot,cos=diagnostic.gradient_matrix(gradients)
        expected=vectors.astype(np.float64)@vectors.astype(np.float64).T
        np.testing.assert_array_equal(dot,expected)
        np.testing.assert_allclose(cos,expected/np.sqrt(np.outer(np.diag(expected),np.diag(expected))))
        model=torch.nn.ParameterDict({'a':torch.nn.Parameter(torch.tensor([1.,2.])),
                                     'b':torch.nn.Parameter(torch.tensor([3.]))})
        saved={'model':{k:v.detach().clone() for k,v in model.state_dict().items()}}
        with torch.no_grad():model['a'].add_(torch.tensor([.25,-.5]));model['b'].add_(.75)
        actual=diagnostic.displacement(model,saved,gradients)
        delta=np.array([.25,-.5,.75])
        np.testing.assert_array_equal(actual['gradient_dot_actual_displacement'],vectors@delta)
        self.assertAlmostEqual(actual['l2'],float(np.linalg.norm(delta)))
        self.assertEqual(actual['changed_elements'],3)
        self.assertFalse(torch.cuda.is_initialized())

    def test_clone_and_all_eight_cpu_adam_trials_do_not_mutate_cached_source(self):
        import torch
        # Parameter storage only: no forward method/network, and no shape-learning claim.
        model=torch.nn.ParameterDict({'a':torch.nn.Parameter(torch.tensor([1.,2.])),
                                     'b':torch.nn.Parameter(torch.tensor([3.]))})
        optimizer=diagnostic.original.optimizer_for(model)
        for parameter in model.parameters():
            optimizer.state[parameter]={'step':torch.tensor(1000.),
                'exp_avg':torch.full_like(parameter,.01),'exp_avg_sq':torch.full_like(parameter,.02)}
        saved=dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},
                   optimizer=copy.deepcopy(optimizer.state_dict()),rng_state=torch.get_rng_state().clone(),
                   cuda_rng_state=torch.empty(0,dtype=torch.uint8))
        bound=diagnostic.snapshot_digest(saved)
        cloned=diagnostic.clone_optimizer_state(saved['optimizer'],torch.device('cpu'))
        for key,state in cloned['state'].items():
            for field,value in state.items():
                self.assertNotEqual(value.data_ptr(),saved['optimizer']['state'][key][field].data_ptr())
        count=0
        for case in range(4):
            for scale in diagnostic.SCALES:
                self.assertTrue(diagnostic.restore(model,optimizer,saved,torch.device('cpu'))['passed'])
                for parameter in model.parameters():parameter.grad=torch.full_like(parameter,.2+case*.1)
                for group in optimizer.param_groups:group['lr']=1e-4*scale
                optimizer.step();count+=1
                self.assertTrue(all(float(s['step'])==1001 for s in optimizer.state.values()))
                self.assertEqual(diagnostic.snapshot_digest(saved),bound)
                self.assertTrue(all(float(s['step'])==1000 for s in saved['optimizer']['state'].values()))
        self.assertEqual(count,8)
        diagnostic.restore(model,optimizer,saved,torch.device('cpu'))
        self.assertTrue(diagnostic.match_state(model,optimizer,saved)['passed'])
        self.assertEqual(diagnostic.snapshot_digest(saved),bound)
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_budget_counts_repeats_and_rejects_extra_calls(self):
        # Call-tree instrumentation fixture only; no numerical/neural model.
        class CallTree:
            def encode(self,x):return x
            def decode(self,x):return x
            def query_geometry(self,x):return x
            def forward(self,x):return self.query_geometry(self.decode(self.encode(x)))
        model=CallTree();counter=diagnostic.CallCounter();counter.install(model)
        for _ in range(44):self.assertEqual(model.forward('actual delegate'),'actual delegate')
        for _ in range(36):model.decode(model.encode('heldout'))
        for _ in range(288):model.query_geometry('chunk')
        diagnostic.validate_budget(counter,4,8)
        with self.assertRaises(RuntimeError):diagnostic.validate_budget(counter,4,9)
        with self.assertRaises(RuntimeError):model.encode('over budget')

    def test_eval_mode_mean_posterior_original_loss_and_no_checkpoint_writes(self):
        tree=ast.parse(inspect.getsource(diagnostic.forward_loss))
        flags=[k.value.value for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id=='model'
               for k in n.keywords if k.arg=='sample_posterior']
        self.assertEqual(flags,[False])
        self.assertIn('original.tensor',inspect.getsource(diagnostic.forward_loss))
        run=inspect.getsource(diagnostic.run)
        self.assertIn('original.original_loss()',run)
        self.assertIn('original.gradients(model)',run)
        self.assertNotIn('torch.save',run)
        self.assertNotIn('extract_geometry',run)
        self.assertEqual(diagnostic.recipe()['trial_step'],1001)
        self.assertEqual(diagnostic.BUDGET['encode'],80)
        self.assertEqual(diagnostic.BUDGET['query_geometry'],332)


if __name__=='__main__':unittest.main()
