"""CPU component/interface qualification, never a replacement task model."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts.sugar.object_predictor import overfit_early_floor_observation as early
from tests.test_overfit_floor_observation import row


class StemFixture(nn.Module):
    def __init__(self, width=54):
        super().__init__()
        self.original = nn.Linear(9, width)
        self.extra = nn.Linear(11, width, bias=False)

    def forward(self, value):
        return self.original(value[:, :9]) + self.extra(value[:, 9:])


def fixture():
    """Only parameter topology and hook dispatch; no official model constructed."""
    model = early.EarlyFloorObservationOverfit.__new__(early.EarlyFloorObservationOverfit)
    nn.Module.__init__(model)
    model.history = 32
    model.predictor = nn.Module()
    model.predictor.backbone = nn.Module()
    model.predictor.backbone.embedding = nn.Module()
    model.predictor.backbone.embedding.stem = nn.Module()
    model.predictor.backbone.embedding.stem.linear = StemFixture()
    model.predictor.head = nn.Linear(54, 13)
    model.auxiliary = nn.Linear(54, 10)
    return model


def original_fixture_forward(model, inputs):
    if set(inputs) != set(early.POINT_KEYS):raise ValueError('Original dispatch leaked extra input')
    value = model.predictor.backbone.embedding.stem.linear(inputs['feat'])
    value = torch.nn.functional.gelu(torch.nn.functional.layer_norm(value, (54,)))
    features = []
    start = 0
    for end in inputs['offset'][31::32]:
        features.append(value[start:int(end)].mean(0))
        start = int(end)
    value = torch.stack(features)
    auxiliary = model.auxiliary(value)
    return dict(state=model.predictor.head(value), force=auxiliary[:, :8],
                availability_logit=auxiliary[:, 8], contact_logit=auxiliary[:, 9])


def inputs(batch=4):
    rows = [row(.1*(i+1)) for i in range(batch)]
    for i, item in enumerate(rows):
        for key, columns in (('coord',3), ('grid_coord',3), ('feat',20)):
            dtype = np.int32 if key == 'grid_coord' else np.float32
            item['inputs'][key] = [np.full((1+(frame+i)%4, columns), .01*(frame+i), dtype=dtype)
                                   for frame in range(32)]
    return early.collate_floor(rows)['inputs']


class EarlyFloorTests(unittest.TestCase):
    def test_batch1_and4_unequal_offsets_map_current_height_to_all32_frames(self):
        for batch in (1,4):
            x=inputs(batch)
            result=early.point_floor_height(x['floor_height_m'],x['offset'],32,len(x['feat']))
            self.assertEqual(result.shape,(len(x['feat']),1))
            start=0
            for frame,end in enumerate(x['offset']):
                self.assertTrue(bool((result[start:int(end)]==x['floor_height_m'][frame//32]).all()))
                start=int(end)
        x=inputs()
        cases=[dict(x,offset=x['offset'].float()),dict(x,offset=x['offset'][:-1]),
               dict(x,floor_height_m=torch.ones(5,1))]
        bad=x['offset'].clone();bad[2]=bad[1];cases.append(dict(x,offset=bad))
        bad=x['offset'].clone();bad[-1]+=1;cases.append(dict(x,offset=bad))
        bad=x['floor_height_m'].clone();bad[0]=float('nan');cases.append(dict(x,floor_height_m=bad))
        for value in cases:
            with self.assertRaises(ValueError):
                early.point_floor_height(value['floor_height_m'],value['offset'],32,len(value['feat']))

    def test_original_names_identities_rng_and_actual_width_unchanged(self):
        model=fixture();old={n:id(p) for n,p in model.named_parameters()}
        state=deepcopy(model.state_dict());rng=torch.random.get_rng_state().clone()
        self.assertEqual(early.derive_stem_width(state),54)
        model.add_floor_branch()
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual(set(model.state_dict()),set(state)|{'floor_embedding.weight'})
        self.assertEqual(model.floor_embedding.weight.shape,(54,1))
        for n,p in model.named_parameters():
            if n in old:self.assertEqual(id(p),old[n]);torch.testing.assert_close(p,state[n],rtol=0,atol=0)
        with self.assertRaises(ValueError):model.add_floor_branch()
        state['predictor.backbone.embedding.stem.linear.extra.weight']=torch.ones(53,11)
        with self.assertRaises(ValueError):early.derive_stem_width(state)

    def test_zero_stem_and_original_forward_dispatch_gradients_and_reload(self):
        model=fixture();x=inputs();expected=original_fixture_forward(model,{k:x[k] for k in early.POINT_KEYS})
        model.add_floor_branch()
        stem=model.predictor.backbone.embedding.stem.linear
        with patch.object(early.original.FullUtoniaOverfit,'forward',original_fixture_forward):
            actual=model(x)
            for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
            self.assertEqual((model.zero_stem_items,model.zero_stem_calls,model.zero_stem_points),(4,1,len(x['feat'])))
            sum(v.square().mean() for v in actual.values()).backward()
            grad=model.floor_embedding.weight.grad
            self.assertTrue(bool(torch.isfinite(grad).all()));self.assertTrue(bool((grad!=0).any()))
            self.assertFalse(stem._forward_hooks);self.assertFalse(model._floor_hook_active)
            with self.assertRaises(ValueError):model(dict(x,mass_available=torch.ones(4)))
            model.zero_stem_enabled=False
            with torch.no_grad():model.floor_embedding.weight[:,0]=torch.linspace(-.01,.01,54)
            before=model(x);saved=deepcopy(model.state_dict())
            with torch.no_grad():model.floor_embedding.weight.fill_(.5)
            model.load_state_dict(saved,strict=True);after=model(x)
            for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(before[k],expected[k]) for k in before))

    def test_finally_cleans_hook_on_error_and_missing_or_duplicate_stem_rejected(self):
        model=fixture();model.add_floor_branch();x=inputs(1)
        stem=model.predictor.backbone.embedding.stem.linear
        def failure(self,values):
            self.predictor.backbone.embedding.stem.linear(values['feat'])
            raise RuntimeError('downstream failure')
        def twice(self,values):
            self.predictor.backbone.embedding.stem.linear(values['feat'])
            return original_fixture_forward(self,values)
        for forward in (failure,twice,lambda self,values:{}):
            with patch.object(early.original.FullUtoniaOverfit,'forward',forward):
                with self.assertRaises(RuntimeError):model(x)
            self.assertFalse(stem._forward_hooks);self.assertFalse(model._floor_hook_active)
        with torch.no_grad():model.floor_embedding.weight.fill_(1.)
        with patch.object(early.original.FullUtoniaOverfit,'forward',original_fixture_forward):
            with self.assertRaises(RuntimeError):model(x)
        self.assertFalse(stem._forward_hooks)

    def test_old_adam_identity_moments_defaults_and_separate_clock(self):
        model=fixture();stem=model.predictor.backbone.embedding.stem.linear
        opt=torch.optim.AdamW([
            dict(params=stem.original.parameters(),name='full_official_backbone',lr=1e-5),
            dict(params=stem.extra.parameters(),name='sensor_affine',lr=5e-4,betas=(.8,.98)),
            dict(params=[*model.predictor.head.parameters(),*model.auxiliary.parameters()],name='task_readouts',lr=1e-4)],weight_decay=.01)
        for p in model.parameters():p.grad=torch.ones_like(p)
        opt.step()
        for state in opt.state.values():state['step'].fill_(2000)
        early.base.split_readout_group(opt,model.predictor.head.parameters(),model.auxiliary.parameters())
        prior={id(p):(opt.state[p],deepcopy(opt.state[p])) for p in model.parameters()}
        model.add_floor_branch();early.add_optimizer_groups(model,opt)
        early.verify_optimizer(model,opt,2000,0)
        self.assertEqual(opt.param_groups[-1]['betas'],(.8,.98))
        for p in model.parameters():
            if id(p) in prior:
                obj,values=prior[id(p)];self.assertIs(obj,opt.state[p])
                for k,v in values.items():torch.testing.assert_close(opt.state[p][k],v,rtol=0,atol=0)
        for update in (1,100):
            rates=early.apply_learning_rates(opt,update);base=early.base.cosine_learning_rates(update)
            self.assertEqual({k:rates[k] for k in base},base)
            self.assertEqual(rates[early.GROUP],base['sensor_affine'])
        for p in model.parameters():p.grad=torch.ones_like(p)
        opt.step();early.verify_optimizer(model,opt,2001,1)
        with self.assertRaises(ValueError):early.verify_optimizer(model,opt,2001,2001)

    def test_original_point_values_and_targets_cannot_change_height_input(self):
        rows=[row(.14),row(.3)];before=early.collate_floor(rows)['inputs']
        for item in rows:
            item['target'][:]=999;item['supervision']['mass_available']=np.float32(1.)
            for value in item['supervision']['physics'].values():value[:]=-999
        after=early.collate_floor(rows)['inputs']
        for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
        raw=early.original.collate_conditional(rows)['inputs']
        for key in early.POINT_KEYS:torch.testing.assert_close(before[key],raw[key],rtol=0,atol=0)

    def test_manifest_rejects_changed_zero_stem_receipt(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'PROTOCOL.json').write_text('{}')
            source=root/'source.py';source.write_text('source')
            for name in early.ARTIFACTS:(root/name).write_bytes(name.encode())
            manifest=dict(schema=4,kind=early.STUDY,complete=True,
                sources={name:dict(path=str(source),sha256=early.original.sha256_file(source))
                    for name in (*early.original.SOURCE_MODULES,*early.EXTRA_SOURCES)},
                protocol=dict(path='PROTOCOL.json',sha256=early.original.sha256_file(root/'PROTOCOL.json')),
                artifacts={name:dict(path=name,sha256=early.original.sha256_file(root/name)) for name in early.ARTIFACTS})
            early.verify_artifacts(root,manifest=manifest)
            (root/'ZERO_STEM_REPLAY.json').write_text('changed')
            with self.assertRaises(ValueError):early.verify_artifacts(root,manifest=manifest)


if __name__=='__main__':unittest.main()
