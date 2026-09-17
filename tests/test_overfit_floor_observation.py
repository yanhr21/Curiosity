"""CPU interface/optimizer diagnostics; no official model or GPU is constructed."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts.sugar.object_predictor import overfit_floor_observation as floor


def row(height=.14):
    state=np.zeros(13,np.float32);state[3]=state[7]=1
    physics={name:np.zeros(size,np.float32) for name,size in floor.original.FORCE_TARGET_FIELDS}
    physics['contact_present']=np.zeros(1,np.float32)
    return dict(inputs=dict(coord=[np.zeros((1,3),np.float32)]*32,
        grid_coord=[np.zeros((1,3),np.int32)]*32,feat=[np.zeros((1,20),np.float32)]*32),
        floor_height_m=np.array([height],np.float32),target=state,
        supervision=dict(mass_available=np.float32(0),mass_uncertain=np.float32(1),mass_status=np.int64(0),
            state_precision_eligible=np.float32(0),state_contact_history_frames=np.int64(0),physics=physics),
        metadata=dict(episode=5000,frame=31,timestamp_s=.64),evaluation_only={})


def interface_fixture():
    """Small parameter fixtures exercise adapter interfaces, not task learning."""
    model=floor.FloorObservationOverfit.__new__(floor.FloorObservationOverfit)
    nn.Module.__init__(model);model.history=32
    model.predictor=nn.Module();model.predictor.backbone=nn.Linear(2,2)
    model.predictor.head=nn.Linear(2,13);model.auxiliary=nn.Linear(2,10)
    return model


class FloorObservationTests(unittest.TestCase):
    def test_current_left_pose_only_and_floor_translation(self):
        poses=np.zeros((40,2,7));poses[:,:,6]=1;poses[:,0,2]=np.arange(40)*.01
        self.assertEqual(float(floor.current_floor_height(poses,31)[0]),float(np.float32(.31)))
        original=floor.current_floor_height(poses,31)
        poses[32:,:,2]=999;poses[:,1,2]=-999
        np.testing.assert_array_equal(floor.current_floor_height(poses,31),original)
        poses[:,:,2]+=.125
        np.testing.assert_allclose(floor.current_floor_height(poses,31)-original,.125,rtol=0,atol=3e-8)
        poses[31,0,2]=np.nan
        with self.assertRaises(ValueError):floor.current_floor_height(poses,31)

    def test_collator_retains_original_four_tensors_and_no_supervision_leak(self):
        rows=[row(.14),row(.32)]
        old=floor.original.collate_conditional(rows)['inputs'];new=floor.collate_floor(rows)['inputs']
        self.assertEqual(set(new),set(floor.POINT_KEYS)|{'floor_height_m'})
        for key in floor.POINT_KEYS:torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
        corrupted=deepcopy(rows)
        for r in corrupted:
            r['target'][:]=999;r['supervision']['mass_available']=np.float32(1)
            for value in r['supervision']['physics'].values():value[:]=1234
        poisoned=floor.collate_floor(corrupted)['inputs']
        for key in new:torch.testing.assert_close(new[key],poisoned[key],rtol=0,atol=0)

    def test_zero_branch_exact_original_dispatch_rng_and_real_branch_gradients(self):
        model=interface_fixture();rng=torch.random.get_rng_state().clone();model.add_floor_branches()
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        expected=dict(state=torch.full((2,13),.123),force=torch.full((2,8),.456),
            availability_logit=torch.tensor([.2,.3]),contact_logit=torch.tensor([.4,.5]))
        received=[]
        def original_forward(self, inputs):
            received.append(set(inputs));return expected
        inputs=floor.collate_floor([row(.14),row(.32)])['inputs']
        with patch.object(floor.original.FullUtoniaOverfit,'forward',original_forward):
            actual=model(inputs)
            for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
            self.assertEqual(received,[set(floor.POINT_KEYS)])
            self.assertEqual(model.zero_replay_items,2)
            sum(value.sum() for value in actual.values()).backward()
            self.assertTrue(bool(torch.isfinite(model.floor_state.weight.grad).all()))
            self.assertTrue(bool((model.floor_state.weight.grad!=0).all()))
            self.assertTrue(bool((model.floor_auxiliary.weight.grad!=0).all()))
            with self.assertRaises(ValueError):model(dict(inputs,mass_available=torch.ones(2)))
            with self.assertRaises(ValueError):model(dict(inputs,floor_height_m=torch.ones(3,1)))
            with torch.no_grad():model.floor_auxiliary.weight[8,0]=.1
            with self.assertRaises(RuntimeError):model(inputs)
            model.zero_replay_enabled=False
            changed=model(inputs)
            torch.testing.assert_close(changed['availability_logit']-expected['availability_logit'],torch.tensor([1.4,3.2]))

    def test_old_adam_preserved_new_clock_and_same_cosine_rates(self):
        model=interface_fixture()
        optimizer=torch.optim.AdamW([
            dict(params=model.predictor.backbone.parameters(),name='full_official_backbone',lr=1e-5),
            dict(params=[],name='sensor_affine',lr=5e-4),
            dict(params=[*model.predictor.head.parameters(),*model.auxiliary.parameters()],name='task_readouts',lr=1e-4)],weight_decay=.01)
        for p in model.parameters():p.grad=torch.ones_like(p)
        optimizer.step()
        for state in optimizer.state.values():state['step'].fill_(2000)
        floor.base.split_readout_group(optimizer,model.predictor.head.parameters(),model.auxiliary.parameters())
        original={id(p):(optimizer.state[p],deepcopy(optimizer.state[p])) for p in model.parameters()}
        model.add_floor_branches();floor.add_optimizer_groups(model,optimizer)
        floor.verify_optimizer(model,optimizer,2000,0)
        for p in model.parameters():
            if id(p) in original:
                obj,values=original[id(p)];self.assertIs(optimizer.state[p],obj)
                for k,v in values.items():torch.testing.assert_close(optimizer.state[p][k],v,rtol=0,atol=0)
        for update in (1,100):
            rates=floor.apply_learning_rates(optimizer,update);base=floor.base.cosine_learning_rates(update)
            self.assertEqual({k:rates[k] for k in base},base)
            for name,parent in floor.FLOOR_GROUPS.items():self.assertEqual(rates[name],base[parent])
        for p in model.parameters():p.grad=torch.ones_like(p)
        optimizer.step();floor.verify_optimizer(model,optimizer,2001,1)
        with self.assertRaises(ValueError):floor.verify_optimizer(model,optimizer,2001,2001)

    def test_manifest_binds_zero_equivalence_and_prediction(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'PROTOCOL.json').write_text('{}')
            source=root/'source.py';source.write_text('source')
            for name in floor.ARTIFACTS:(root/name).write_bytes(name.encode())
            manifest=dict(schema=3,kind=floor.STUDY,complete=True,
                sources={name:dict(path=str(source),sha256=floor.original.sha256_file(source))
                    for name in (*floor.original.SOURCE_MODULES,*floor.EXTRA_SOURCES)},
                protocol=dict(path='PROTOCOL.json',sha256=floor.original.sha256_file(root/'PROTOCOL.json')),
                artifacts={name:dict(path=name,sha256=floor.original.sha256_file(root/name)) for name in floor.ARTIFACTS})
            floor.verify_artifacts(root,manifest=manifest)
            (root/'ZERO_BRANCH_REPLAY.json').write_text('changed')
            with self.assertRaises(ValueError):floor.verify_artifacts(root,manifest=manifest)


if __name__=='__main__':unittest.main()
