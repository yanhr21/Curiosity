"""Synthetic CPU optimizer/objective boundaries; no full model or GPU executes."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.sugar.object_predictor import overfit_fullbatch_refinement as repair
from scripts.sugar.object_predictor.train_overfit_fullbatch_refinement import fixed_epoch_batches


class FullbatchRefinementTests(unittest.TestCase):
    def optimizer(self):
        # Isolated parameters test optimizer state routing, not a substitute model.
        parameters=[torch.nn.Parameter(torch.tensor([1.,2.])) for _ in range(8)]
        groups=[dict(params=parameters[:3],name='full_official_backbone',lr=1e-5),
                dict(params=parameters[3:4],name='sensor_affine',lr=5e-4),
                dict(params=parameters[4:],name='task_readouts',lr=1e-4)]
        optimizer=torch.optim.AdamW(groups,weight_decay=.01)
        for i,p in enumerate(parameters):p.grad=torch.full_like(p,float(i+1))
        optimizer.step()
        for state in optimizer.state.values():state['step'].fill_(2000)
        return parameters,optimizer

    def test_group_split_preserves_all_actual_moment_objects_and_clocks(self):
        p,opt=self.optimizer()
        states={id(v):opt.state[v] for v in p}
        values={id(v):deepcopy(opt.state[v]) for v in p}
        repair.split_readout_group(opt,p[4:6],p[6:])
        self.assertEqual([g['name'] for g in opt.param_groups],list(repair.LR_BASE))
        for parameter in p:
            self.assertIs(opt.state[parameter],states[id(parameter)])
            for key,value in values[id(parameter)].items():
                torch.testing.assert_close(opt.state[parameter][key],value,rtol=0,atol=0)
        model=SimpleNamespace(named_parameters=lambda: [(str(i),v) for i,v in enumerate(p)])
        self.assertEqual(repair.verify_restored_optimizer(model,opt,2000)['active_parameter_tensors'],8)
        opt.step()
        self.assertEqual(repair.verify_restored_optimizer(model,opt,2001)['all_active_adam_steps'],2001)

    def test_reordered_readout_or_wrong_moment_clock_is_rejected(self):
        p,opt=self.optimizer()
        with self.assertRaises(ValueError):repair.split_readout_group(opt,p[5:6]+p[4:5],p[6:])
        model=SimpleNamespace(named_parameters=lambda: [(str(i),v) for i,v in enumerate(p)])
        with self.assertRaises(ValueError):repair.verify_restored_optimizer(model,opt,1999)
        opt.state[p[0]]['exp_avg']=torch.zeros(3)
        with self.assertRaises(ValueError):repair.verify_restored_optimizer(model,opt,2000)

    def test_macroaverage_preserves_mask_denominators_not_uniform_target_mean(self):
        from scripts.sugar.object_predictor.overfit_model import masked_mass_loss
        labels=[([0.,0.,0.,0.],[1,0,0,0]),([1.,1.,1.,1.],[1,1,1,1])]
        labels=labels*10
        p=torch.tensor(.4,requires_grad=True)
        for values,mask in labels:
            state=torch.zeros(4,13)+p
            target=torch.zeros(4,13);target[:,12]=torch.tensor(values)
            mass=masked_mass_loss(state,target,torch.tensor(mask))
            parts={key:p*0 for key in repair.original.LOSS_WEIGHTS};parts['mass']=mass
            repair.scaled_microbatch_loss(parts).backward()
        actual=p.grad.clone()
        q=torch.tensor(.4,requires_grad=True)
        expected=sum((torch.expm1(q-torch.tensor(values))[torch.tensor(mask).bool()].abs().mean()/.05)
                     for values,mask in labels)/20
        expected.backward()
        torch.testing.assert_close(actual,q.grad)
        t=torch.tensor(.4,requires_grad=True)
        uniform=torch.cat([torch.expm1(t-torch.tensor(values))[torch.tensor(mask).bool()].abs()
                           for values,mask in labels]).mean()/.05
        uniform.backward()
        self.assertGreater(float(abs(actual-t.grad)),1.)

    def test_full_epoch_after_source_covers_all_eighty_once(self):
        rows=[dict(metadata=dict(episode=e,frame=f)) for e in repair.original.FIXED_EPISODES
              for f in repair.original.FIXED_FRAMES]
        dataset=SimpleNamespace(rows=rows)
        iterator=repair.original.fixed_batches(dataset,310117)
        for _ in range(2000):next(iterator)
        batches=fixed_epoch_batches(dataset,iterator)
        self.assertEqual(len(batches),20)
        self.assertEqual(sorted(i for b in batches for i in b),list(range(80)))
        with self.assertRaises(RuntimeError):fixed_epoch_batches(dataset,iter([[0,1,2,3]]*20))

    def test_fixed_cosine_endpoints_and_budget(self):
        self.assertEqual(repair.cosine_learning_rates(1),repair.LR_BASE)
        for key,value in repair.LR_BASE.items():
            self.assertAlmostEqual(repair.cosine_learning_rates(100)[key],value*.05)
            values=[repair.cosine_learning_rates(i)[key] for i in range(1,101)]
            self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        for step in (0,101):
            with self.assertRaises(ValueError):repair.cosine_learning_rates(step)
        self.assertEqual(repair.NEW_UPDATES*repair.MICROBATCHES,2000)

    def test_new_manifest_rejects_one_changed_prediction_binding(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'PROTOCOL.json').write_text('{}')
            dummy=root/'source.py';dummy.write_text('source')
            for name in repair.ARTIFACTS:(root/name).write_bytes(name.encode())
            manifest=dict(schema=2,kind=repair.STUDY,complete=True,
                sources={name:dict(path=str(dummy),sha256=repair.original.sha256_file(dummy))
                         for name in (*repair.original.SOURCE_MODULES,*repair.EXTRA_SOURCES)},
                protocol=dict(path='PROTOCOL.json',sha256=repair.original.sha256_file(root/'PROTOCOL.json')),
                artifacts={name:dict(path=name,sha256=repair.original.sha256_file(root/name)) for name in repair.ARTIFACTS})
            repair.verify_artifacts(root,manifest=manifest)
            (root/'fit_02100.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):repair.verify_artifacts(root,manifest=manifest)


if __name__=='__main__':unittest.main()
