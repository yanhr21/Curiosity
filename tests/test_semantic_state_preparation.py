"""CPU adapter/loss contract tests; fixtures are not trained replacement models."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
from scripts.sugar.object_predictor import semantic_state_readout as readout
from scripts.sugar.object_predictor import semantic_state_data as data
from scripts.sugar.object_predictor.semantic_state_loss import semantic_loss_parts,semantic_total_loss


class ReadoutFixture(nn.Module):
    def __init__(self):
        super().__init__();self.predictor=nn.Module()
        self.predictor.backbone=nn.Linear(1,1)
        self.sensor=nn.Linear(1,1)
        self.predictor.head=nn.Linear(64,13)
        self.auxiliary=nn.Linear(64,10)
    def forward(self,points):
        x=points['feat']
        return self.predictor.head(x),self.auxiliary(x)


def row(index=0,available=0,height=.2):
    feat=np.zeros((2,20),np.float32);feat[:,9]=1;feat[:,19]=-1
    return dict(inputs=dict(coord=[feat[:,:3].copy() for _ in range(32)],
        grid_coord=[np.zeros((2,3),np.int32) for _ in range(32)],feat=[feat.copy() for _ in range(32)]),
        observed_left_hand_world_height_m=np.full((32,1),height,np.float32),target=np.zeros(13,np.float32),
        supervision=dict(state_precision_eligible=1.,state_contact_history_frames=32,mass_available=available),
        metadata=dict(episode=5000+index,frame=31))


class SemanticPreparationTests(unittest.TestCase):
    def test_exact_initial_readout_and_frame_interleaving(self):
        torch.manual_seed(17);model=ReadoutFixture();x=torch.randn(3,64)
        before=(model.predictor.head(x),model.auxiliary(x));ids={n:id(p) for n,p in model.named_parameters()}
        receipt=readout.install_semantic_readouts(model)
        self.assertTrue(receipt['original_parameter_names_and_identities_exact'])
        self.assertTrue(all(id(dict(model.named_parameters())[n])==i for n,i in ids.items()))
        history=torch.randn(3,32,11)
        with readout.observation_context(model,history):after=(model.predictor.head(x),model.auxiliary(x))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,after)))
        head=model.predictor.head;expanded=head.expanded_weight().reshape(13,32,13)
        self.assertTrue(torch.equal(expanded[:,:,:2].reshape(13,64),head.weight))
        self.assertEqual(int(torch.count_nonzero(expanded[:,:,2:])),0)
        readout.end_zero_effect_checks(model)
        with torch.no_grad():head.summary_weight[0,7*11+3]=2.
        with readout.observation_context(model,history):actual=head(x)
        self.assertTrue(torch.allclose(actual[:,0]-before[0][:,0],2*history[:,7,3],atol=1e-6,rtol=0))

    def test_cleanup_after_error_and_no_reentrancy(self):
        model=ReadoutFixture();readout.install_semantic_readouts(model);h=torch.zeros(1,32,11)
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with readout.observation_context(model,h):raise RuntimeError('deliberate')
        self.assertIsNone(model.predictor.head._history);self.assertIsNone(model.auxiliary._history)
        with readout.observation_context(model,h):
            with self.assertRaisesRegex(RuntimeError,'Reentrant'):
                with readout.observation_context(model,h):pass
            self.assertIs(model.predictor.head._history,h)
        with self.assertRaises(ValueError):
            readout.forward_semantic_observations(model,dict(target=torch.zeros(1)),h)

    def test_original_adam_objects_and_independent_new_clock(self):
        model=ReadoutFixture()
        groups=[dict(name=n,params=list(m.parameters()),lr=1e-4) for n,m in zip(
            ('full_official_backbone','sensor_affine','state_readout','auxiliary_readout'),
            (model.predictor.backbone,model.sensor,model.predictor.head,model.auxiliary))]
        opt=torch.optim.AdamW(groups,weight_decay=.01)
        for p in model.parameters():
            opt.state[p]=dict(step=torch.tensor(2100.),exp_avg=torch.ones_like(p)*.01,exp_avg_sq=torch.ones_like(p)*.02)
        original={id(p):opt.state[p] for p in model.parameters()}
        readout.install_semantic_readouts(model)
        receipt=readout.add_observation_optimizer_groups(model,opt,state_lr=2e-6,classification_lr=1e-3)
        self.assertTrue(receipt['original_state_objects_exact'])
        self.assertTrue(all(opt.state[p] is original[id(p)] for p in model.parameters() if id(p) in original))
        for p in model.parameters():p.grad=torch.ones_like(p)
        opt.step()
        for p in model.parameters():self.assertEqual(int(opt.state[p]['step']),2101 if id(p) in original else 1)

    def test_legacy_force_has_no_new_observation_columns(self):
        model=ReadoutFixture();readout.install_semantic_readouts(model);readout.end_zero_effect_checks(model)
        x=torch.randn(2,64);original=nn.functional.linear(x,model.auxiliary.weight,model.auxiliary.bias)
        with torch.no_grad():model.auxiliary.summary_weight.fill_(.1)
        with readout.observation_context(model,torch.ones(2,32,11)):actual=model.auxiliary(x)
        self.assertTrue(torch.equal(actual[:,:8],original[:,:8]))
        self.assertFalse(torch.equal(actual[:,8:],original[:,8:]))
        expanded=model.auxiliary.expanded_weight().reshape(10,32,13)
        self.assertEqual(int(torch.count_nonzero(expanded[:8,:,2:])),0)

    def test_rms_train_only_preserves_zero_and_public_height(self):
        h=np.zeros((464,32,11),np.float32);h[:,:,0]=2;h[:,:,10]=1.5
        scale,report=readout.fit_observation_scales(h)
        self.assertEqual(scale[0],2);self.assertEqual(scale[1],1);self.assertEqual(scale[10],1)
        value=readout.scale_observation_history(torch.from_numpy(h),torch.from_numpy(scale))
        self.assertEqual(int(torch.count_nonzero(value[:,:,1:10])),0)
        self.assertTrue(torch.all(value[:,:,10]==1.5));self.assertFalse(report['evaluation_statistics_used'])
        with self.assertRaises(ValueError):readout.fit_observation_scales(h[:80])

    def test_complete_observation_groups_include_equal_targets(self):
        rows=[row(0,0),row(1,1)]
        self.assertTrue(np.array_equal(rows[0]['target'],rows[1]['target']))
        groups=data.full_observation_groups(rows)
        self.assertEqual(groups[0]['row_indices'],[0,1])
        rows[1]['observed_left_hand_world_height_m'][7]=.21
        self.assertEqual(data.full_observation_groups(rows),[])

    def test_availability_conflict_with_identical_continuous_target_rejected(self):
        rows=[row(0,0),row(1,1)]
        class DatasetFixture(SimpleNamespace):
            def __len__(self):return len(self.rows)
        d=DatasetFixture(rows=rows,expected_frames=(31,),semantic_role='fit',qualified=True,conflicts=[],
            failure_source_validation=dict(passed=True),controlled_source_validation=dict(passed=True),
            collection_result=dict(complete=True,qualification_passed=False),
            collection_records={e:dict(controller_passed=e!=5014) for e in data.FIXED_EPISODES})
        q=data.qualification(d,dict(center_cm=1.,size_max_relative=.05))
        self.assertFalse(q['checks']['no_identical_full_observation_availability_conflict'])
        self.assertEqual(len(q['unresolved_availability_conflicts']),1)

    def test_global_denominators_partition_values_and_gradients(self):
        state=torch.zeros(4,13);state[:,3]=state[:,7]=1;state[:,:3]=.01
        state.requires_grad_();avail=torch.tensor([1.,0.,0.,0.]);eligible=torch.tensor([1.,1.,0.,1.])
        logits=torch.tensor([.3,-.5,.2,-.1],requires_grad=True)
        contact=torch.tensor([.2,-.1,.1,-.2],requires_grad=True)
        force=torch.randn(4,8,requires_grad=True)
        target=state.detach().clone();target[:,:3]=0;target[:,12]=-.1
        batch=dict(target=target,mass_available=avail,state_precision_eligible=eligible,
            physics=dict(contact_present=torch.tensor([[1.],[1.],[0.],[1.]])))
        output=dict(state=state,availability_logit=logits,contact_logit=contact,force=force)
        den=dict(state_candidates=3,available_mass=1,availability_positive=1,availability_negative=3,current_contact_total=4)
        vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        allparts=semantic_loss_parts(output,batch,vertices,den);full=semantic_total_loss(allparts)
        grad=torch.autograd.grad(full,(state,logits,contact),retain_graph=True)
        parts={k:[] for k in allparts}
        for ix in (slice(0,1),slice(1,4)):
            b={k:(v[ix] if k!='physics' else {n:w[ix] for n,w in v.items()}) for k,v in batch.items()}
            each=semantic_loss_parts({k:v[ix] for k,v in output.items()},b,vertices,den)
            for k,v in each.items():parts[k].append(v)
        combined={k:sum(v) for k,v in parts.items()}
        for k in allparts:self.assertTrue(torch.allclose(allparts[k],combined[k],atol=1e-6,rtol=1e-6),k)
        splitgrad=torch.autograd.grad(semantic_total_loss(combined),(state,logits,contact),retain_graph=True)
        self.assertTrue(all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(grad,splitgrad)))
        self.assertIsNone(torch.autograd.grad(full,force,allow_unused=True)[0])

    def test_unknown_precision_targets_cannot_produce_gradient(self):
        state=torch.zeros(2,13,requires_grad=True)
        output=dict(state=state,availability_logit=torch.zeros(2,requires_grad=True),contact_logit=torch.zeros(2,requires_grad=True))
        batch=dict(target=torch.full((2,13),1e20),mass_available=torch.zeros(2),state_precision_eligible=torch.zeros(2),
            physics=dict(contact_present=torch.zeros(2,1)))
        den=dict(state_candidates=1,available_mass=1,availability_positive=1,availability_negative=3,current_contact_total=4)
        parts=semantic_loss_parts(output,batch,torch.zeros(3,3),den)
        self.assertTrue(all(float(parts[k])==0 for k in ('center','mesh','size','mass')))
        semantic_total_loss(parts).backward();self.assertEqual(int(torch.count_nonzero(state.grad)),0)


if __name__=='__main__':unittest.main()
