"""CPU index/restore/glue tests. No official model forward or synthetic model."""
import ast
import copy
import inspect
import json
from pathlib import Path
import unittest
import numpy as np
from scripts.sugar.object_predictor import michelangelo_persistent_replay_data as data
from scripts.sugar.object_predictor import train_michelangelo_persistent_replay as trainer


def numeric_fixtures():
    # Pure index-array fixture, never a replacement neural model/physical record.
    n=60000;label=np.arange(n)<30000;eligible=np.ones(n,bool)
    query=np.arange(n*3,dtype=np.float32).reshape(n,3)
    cases=[];pools=[];hist=[];near=[]
    for oid in data.OBJECT_IDS:
        item=dict(input_surface_xyz_vae=np.zeros((4096,3),np.float32),input_normals=np.ones((4096,3),np.float32),
            train_queries_vae=np.arange(32768*3,dtype=np.float32).reshape(32768,3),train_labels=np.arange(32768)%2==0,
            eval_queries_vae=np.zeros((7,3)),eval_labels=np.zeros(7,bool))
        cases.append(dict(object_id=oid,original=item,labels=label,eligible=eligible,eligible_indices=np.arange(n),
                          interior_indices=np.flatnonzero(label),exterior_indices=np.flatnonzero(~label)))
        pools.append(dict(fp=np.arange(30000,30017),fn=np.arange(11)))
        hist.append(dict(fp=np.arange(30000,48725),fn=np.arange(26371)))
        near.append(np.arange(16384,32768))
    return query,cases,pools,hist,near


class PersistentReplayTests(unittest.TestCase):
    def test_queue_append_preserves_unvisited_source_and_repeats_only_after_end(self):
        q=data.PersistentCycle(np.arange(10),0,10)
        first=q.take(6);old=q.order.copy();q.append(np.array([10,11,11]))
        np.testing.assert_array_equal(q.order[:10],old)
        next_=q.take(4);self.assertEqual(set(np.r_[first,next_]),set(range(10)))
        self.assertTrue(q.coverage()['source_complete']);self.assertFalse(q.coverage()['all_complete'])
        np.testing.assert_array_equal(q.take(2),[10,11]);self.assertTrue(q.coverage()['all_complete'])
        self.assertEqual(q.cycle,0);self.assertEqual(len(np.unique(q.take(12))),12);self.assertEqual(q.cycle,1)

    def test_actual_saved_grid_union_matches_all84_sources(self):
        manifest,q,cases,pools,histories,result,sources,previous=data.load_sources()
        self.assertEqual(result['total_model_updates'],2000);self.assertEqual(len(previous),4)
        history=json.loads((data.SOURCE/'HISTORICAL_ERROR_UNION.json').read_text())
        self.assertEqual(len(history['source_bindings']),84)
        for case,pool,h in zip(cases,pools,histories,strict=True):
            parts={k:[] for k in ('fp','fn')}
            for name in history['source_bindings']:
                if Path(name).name!=case['object_id']+'.npz':continue
                with np.load(name,allow_pickle=False) as z:
                    for k in parts:parts[k].append(z[k+'_indices'])
            for k in parts:
                np.testing.assert_array_equal(h[k],np.unique(np.concatenate(parts[k])))
                self.assertTrue(np.isin(pool[k],h[k]).all())
                self.assertLessEqual(len(h[k]),208*128)
            # Original observation, near/eval arrays are not rewritten by this loader.
            self.assertEqual(case['original']['input_surface_xyz_vae'].shape,(4096,3))
            self.assertEqual(case['original']['train_queries_vae'].shape,(32768,3))

    def test_832_steps_guarantee_source_coverage_despite_refresh_appends(self):
        q,cases,pools,hist,near=numeric_fixtures();sampler=data.ReplaySampler(cases,pools,hist,near)
        totals=np.zeros(4,int);history_draws=[[] for _ in range(4)]
        for step in range(2001,2833):
            if step-1 in data.REFRESH_STEPS:
                # Corrected source nodes must remain. New FN IDs extend the queue,
                # but never jump ahead of unvisited original source entries.
                fresh=[dict(fp=np.empty(0,np.int64),fn=np.arange(26371,30000)) for _ in range(4)]
                sampler.refresh(step-1,fresh,[np.empty(0,np.int64) for _ in range(4)])
            i,vol,n,info=sampler.indices(step);totals[i]+=1;history_draws[i].append(vol[896:1024])
            self.assertEqual(len(vol),1024);self.assertEqual(len(n),1024)
            self.assertTrue((~cases[i]['labels'][vol[512:768]]).all())
            self.assertTrue(cases[i]['labels'][vol[768:1024]].all())
            self.assertTrue(((n>=16384)&(n<32768)).all())
        np.testing.assert_array_equal(totals,[208]*4)
        for draws in history_draws:self.assertTrue(np.isin(np.arange(26371),np.concatenate(draws)).all())
        report=sampler.coverage();self.assertTrue(report['source_union_coverage_complete'])
        self.assertFalse(report['all_observed_union_coverage_complete'])
        self.assertEqual(sampler.last_step,2832)
        with self.assertRaises(ValueError):sampler.indices(2833)

    def test_partition_anchor_provenance_empty_fallback_and_no_eval_dependency(self):
        q,cases,_,_,_=numeric_fixtures()
        empty=[dict(fp=np.empty(0,np.int64),fn=np.empty(0,np.int64)) for _ in range(4)]
        near=[np.empty(0,np.int64) for _ in range(4)]
        a=data.ReplaySampler(cases,empty,empty,near);changed=copy.deepcopy(cases)
        for c in changed:c['original']['eval_labels'][:]=True;c['original']['eval_queries_vae'][:]=99
        b=data.ReplaySampler(changed,empty,empty,near)
        for step in range(2001,2005):
            i,item,indices,vol,n,info=a.batch(step,q)
            j,other,_,other_vol,other_near,_=b.batch(step,q)
            self.assertEqual(i,j);np.testing.assert_array_equal(vol,other_vol);np.testing.assert_array_equal(n,other_near)
            rng=np.random.default_rng(data.dynamic.VOLUME_SEED+step)
            np.testing.assert_array_equal(vol[:512],rng.choice(cases[i]['eligible_indices'],512,replace=False))
            np.testing.assert_array_equal(n[:512],data.train_indices(step)[1][1024:1536])
            np.testing.assert_array_equal(item['train_queries_vae'][:1024],q[vol])
            np.testing.assert_array_equal(item['train_queries_vae'][1024:],cases[i]['original']['train_queries_vae'][n])
            np.testing.assert_array_equal(item['train_labels'],np.r_[cases[i]['labels'][vol],cases[i]['original']['train_labels'][n]])
            self.assertEqual(set(item),{'input_surface_xyz_vae','input_normals','train_queries_vae','train_labels'})
            self.assertIs(item['input_normals'],cases[i]['original']['input_normals'])
            self.assertTrue(all(info['fallback'].values()))
            self.assertTrue((~cases[i]['labels'][vol[512:768]]).all());self.assertTrue(cases[i]['labels'][vol[768:]].all())

    def test_refresh_clock_and_training_near_sign_only(self):
        q,cases,pools,hist,near=numeric_fixtures();sampler=data.ReplaySampler(cases,pools,hist,near)
        with self.assertRaises(ValueError):sampler.refresh(2000,pools,near)
        with self.assertRaises(ValueError):sampler.refresh(2100,pools,near)
        for step in range(2001,2101):sampler.indices(step)
        with self.assertRaises(ValueError):sampler.indices(2101)
        sampler.refresh(2100,pools,near);sampler.indices(2101)
        logits=np.where(cases[0]['original']['train_labels'][16384:],1.,-1.)
        self.assertEqual(len(data.near_error_ids(logits,cases[0]['original'])),0)
        logits[10]*=-1;np.testing.assert_array_equal(data.near_error_ids(logits,cases[0]['original']),[16394])
        for bad in [np.full(16384,np.nan),np.zeros(32768)]:
            with self.assertRaises(ValueError):data.near_error_ids(bad,cases[0]['original'])
        with self.assertRaises(ValueError):data.validate_near_ids(np.array([0,32768]))

    def test_full_2000_state_restore_does_not_alias_cpu_adam_clock(self):
        import torch
        # Bare parameter storage for optimizer state equality, no neural forward.
        model=torch.nn.ParameterDict({'a':torch.nn.Parameter(torch.tensor([1.,2.]))})
        optimizer=data.original.optimizer_for(model)
        optimizer.param_groups[0]['lr']=1e-5
        for p in model.parameters():optimizer.state[p]=dict(step=torch.tensor(2000.),exp_avg=torch.full_like(p,.1),exp_avg_sq=torch.full_like(p,.2))
        saved=dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},
            optimizer=copy.deepcopy(optimizer.state_dict()),rng_state=torch.get_rng_state().clone(),cuda_rng_state=torch.empty(0,dtype=torch.uint8))
        bound=trainer.state.snapshot_digest(saved)
        receipt,got=trainer.restore_source(model,optimizer,saved,torch.device('cpu'))
        self.assertEqual(got,bound);self.assertEqual(receipt['every_step'],2000)
        for p in model.parameters():
            self.assertNotEqual(optimizer.state[p]['step'].data_ptr(),saved['optimizer']['state'][0]['step'].data_ptr());p.grad=torch.ones_like(p)
        optimizer.step()
        self.assertEqual(float(optimizer.state[next(model.parameters())]['step']),2001)
        self.assertEqual(trainer.state.snapshot_digest(saved),bound)
        trainer.restore_source(model,optimizer,saved,torch.device('cpu'))
        self.assertTrue(trainer.state.match_state(model,optimizer,saved,expected_step=2000)['passed'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_budget_counts_and_official_loss_identity(self):
        self.assertIs(trainer.batch_loss,trainer.prior.batch_loss)
        self.assertEqual(trainer.audit_endpoint.__globals__['STEPS'],2832)
        self.assertEqual(trainer.original.STEPS,1000)
        self.assertEqual(data.QUALIFICATION_BUDGET['query_geometry'],4*(8+4))
        self.assertEqual(trainer.FORMAL_BUDGET,dict(forward=832,encode=832+4+32+4+32+4+4+8,
            decode=832+4+32+4+32+4+4+4,query_geometry=832+32+8*876+876+8*32+32+32+4*(215+8)))
        source=inspect.getsource(trainer.run)
        self.assertIn('range(2001,2833)',source);self.assertIn('sampler.batch(step,queries)',source)
        self.assertNotIn('near_error_ids(after',source)
        self.assertIn('expected_step=sampling.START',inspect.getsource(trainer.restore_source))

    def test_method_counter_calls_real_method_and_restores_after_failure(self):
        class Calls:
            def forward(self):return 3
            def encode(self):return 5
            def decode(self):return 7
            def query_geometry(self):return 9
        target=Calls();counter=trainer.MethodCounter(dict(forward=1,encode=1,decode=1,query_geometry=1))
        old=target.encode;counter.install(target)
        self.assertEqual([target.forward(),target.encode(),target.decode(),target.query_geometry()],[3,5,7,9]);counter.require_complete()
        with self.assertRaises(RuntimeError):target.encode()
        counter.detach();self.assertEqual(target.encode,old);self.assertEqual(target.encode(),5)
        self.assertEqual(counter.completed['encode'],1)

if __name__=='__main__':unittest.main()
