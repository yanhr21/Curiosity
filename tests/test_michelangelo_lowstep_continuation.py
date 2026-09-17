"""CPU data/restore qualification; no official-model forwards or GPU work."""
import ast
import copy
import inspect
import unittest
import numpy as np
from scripts.sugar.object_predictor import michelangelo_lowstep_continuation_data as data
from scripts.sugar.object_predictor import train_michelangelo_lowstep_continuation as trainer

class LowstepContinuationTests(unittest.TestCase):
    def test_only_absolute_guard_changes_original_sampler_algorithms(self):
        self.assertIs(data.train_indices.__code__,data.original.train_indices.__code__)
        self.assertEqual(data.train_indices.__globals__['STEPS'],2000)
        self.assertEqual(data.original.STEPS,1000)
        self.assertEqual(data.train_indices.__defaults__,data.original.train_indices.__defaults__)
        for step in range(1,1001):
            old=data.original.train_indices(step);new=data.train_indices(step)
            self.assertEqual(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
        for step in range(1001,2001):
            index,indices=data.train_indices(step)
            rng=np.random.default_rng(data.original.TRAIN_SEED+step)
            expected=np.concatenate((rng.choice(16384,1024,replace=False),16384+rng.choice(16384,1024,replace=False)))
            self.assertEqual(index,(step-1)%4);np.testing.assert_array_equal(indices,expected)
        for step in [0,2001]:
            with self.assertRaises(ValueError):data.train_indices(step)

    def test_actual_saved_pool_data_and_batch_provenance(self):
        manifest,q,cases,pools,result,sources,previous=data.load_sources()
        self.assertEqual([c['object_id'] for c in cases],list(data.OBJECT_IDS))
        self.assertEqual(result['model_updates'],1000)
        self.assertEqual(len(previous),4)
        self.assertEqual(sources['checkpoint_sha256'],'e4b2b3a92b8b1d569bb47736e33bb4387bf651a82916974aa822aa3f61d7e3db')
        for step in range(1,1001):
            case=cases[(step-1)%4];pool=pools[(step-1)%4]
            old,old_info=data.dynamic.sample_volume(step,case,pool)
            new,new_info=data.sample_volume(step,case,pool)
            np.testing.assert_array_equal(old,new);self.assertEqual(old_info,new_info)
        for step in [1001,1002,1003,1004,1100,1101,1901,2000]:
            index,item,indices,volume,near,info=data.batch_for_step(step,q,cases,pools)
            original=cases[index]['original'];_,whole=data.train_indices(step)
            np.testing.assert_array_equal(near,whole[1024:])
            np.testing.assert_array_equal(item['train_queries_vae'][:1024],q[volume])
            np.testing.assert_array_equal(item['train_labels'][:1024],cases[index]['labels'][volume])
            np.testing.assert_array_equal(item['train_queries_vae'][1024:],original['train_queries_vae'][near])
            np.testing.assert_array_equal(item['train_labels'][1024:],original['train_labels'][near])
            self.assertIs(item['input_surface_xyz_vae'],original['input_surface_xyz_vae'])
            self.assertIs(item['input_normals'],original['input_normals'])
            self.assertIs(item['eval_queries_vae'],original['eval_queries_vae'])
            self.assertIs(item['eval_labels'],original['eval_labels'])
        self.assertEqual(data.REFRESH_STEPS,tuple(range(1100,2000,100)))
        for step in [0,2001]:
            with self.assertRaises(ValueError):data.sample_volume(step,cases[0],pools[0])

    def test_restore_changes_only_lr_and_no_step_scalar_alias(self):
        import torch
        # Bare numeric parameter storage; no neural forward/model substitute.
        model=torch.nn.ParameterDict({'a':torch.nn.Parameter(torch.tensor([1.,2.]))})
        optimizer=data.original.optimizer_for(model)
        for parameter in model.parameters():
            optimizer.state[parameter]=dict(step=torch.tensor(1000.),exp_avg=torch.full_like(parameter,.01),exp_avg_sq=torch.full_like(parameter,.02))
        saved=dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},
            optimizer=copy.deepcopy(optimizer.state_dict()),rng_state=torch.get_rng_state().clone(),cuda_rng_state=torch.empty(0,dtype=torch.uint8))
        before=trainer.state.snapshot_digest(saved)
        receipt,bound=trainer.restore_for_continuation(model,optimizer,saved,torch.device('cpu'))
        self.assertEqual(before,bound);self.assertTrue(receipt['after_only_lr_change']['passed'])
        self.assertEqual(optimizer.param_groups[0]['lr'],1e-5)
        self.assertEqual(saved['optimizer']['param_groups'][0]['lr'],1e-4)
        for parameter in model.parameters():parameter.grad=torch.full_like(parameter,.1)
        optimizer.step()
        self.assertTrue(all(float(s['step'])==1001 for s in optimizer.state.values()))
        self.assertEqual(trainer.state.snapshot_digest(saved),before)
        self.assertFalse(torch.cuda.is_initialized())

    def test_faithful_model_loss_refresh_and_endpoint_audit(self):
        self.assertIs(trainer.batch_loss,trainer.prior.batch_loss)
        self.assertIs(trainer.refresh,trainer.prior.refresh)
        self.assertIs(trainer.audit_endpoint.__code__,data.original.audit_saved_checkpoint.__code__)
        self.assertEqual(trainer.audit_endpoint.__globals__['STEPS'],2000)
        self.assertEqual(data.original.audit_saved_checkpoint.__globals__['STEPS'],1000)
        source=inspect.getsource(trainer.run)
        self.assertIn('range(1001,2001)',source)
        self.assertIn('original.TRAIN_SEED+step',source)
        self.assertIn('restore_for_continuation',source)
        self.assertNotIn('released_model',source)
        self.assertNotIn('load_official_model',source)
        self.assertIn('initial_eval!=expected',source)
        self.assertIn("source_model_updates=1000,total_model_updates=2000",source)
        tree=ast.parse(source)
        result_call=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets))
        fields={k.arg:k.value for k in result_call.keywords}
        self.assertIs(fields['checkpoint_reset'].value,False)
        self.assertEqual(fields['model_updates'].value,1000)
        self.assertEqual(fields['source_model_updates'].value,1000)
        self.assertEqual(fields['total_model_updates'].value,2000)
        self.assertIn('source_checkpoint_sha256',fields)
        self.assertLess(source.index('torch.save('),source.index('endpoint_grid=refresh('))
        self.assertEqual(data.recipe()['new_updates'],1000)
        self.assertEqual(data.recipe()['lr'],1e-5)

if __name__=='__main__':unittest.main()
