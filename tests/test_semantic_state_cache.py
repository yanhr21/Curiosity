"""Exact cache plumbing tests; no neural model or simulated data generation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts.sugar.object_predictor import semantic_state_data as data
from scripts.sugar.object_predictor import semantic_state_encoded_cache as cache


def fixture():
    rows=[]
    for e in data.FIXED_EPISODES:
        for f in data.FIT_FRAMES:
            coord=np.array([[.125,-0.,.25]],np.float32)
            feat=np.zeros((1,20),np.float32);feat[:,:3]=coord;feat[:,19]=-1.
            height=np.full((32,1),.2,np.float32);summary=np.zeros((32,11),np.float32);summary[:,10]=1.
            inputs=dict(coord=[coord.copy() for _ in range(32)],grid_coord=[np.zeros((1,3),np.int32) for _ in range(32)],feat=[feat.copy() for _ in range(32)])
            indices=list(range(f-31,f+1));times=.02*(np.asarray(indices)+1)
            sup=dict(mass_available=np.float32(0),mass_uncertain=np.float32(1),mass_status=np.int64(0),
                state_precision_eligible=np.float32(0),state_contact_history_frames=np.int64(0),
                physics=dict(contact_present=np.array([0.],np.float32)))
            rows.append(dict(inputs=inputs,target=np.zeros(13,np.float32),supervision=sup,
                metadata=dict(episode=e,frame=f,history_indices=indices,timestamp_s=float(times[-1])),
                evaluation_only=dict(qualified=True),observed_left_hand_world_height_m=height,
                observed_height_timestamp_s=times,_observed_summary=summary))
    return SimpleNamespace(rows=rows,semantic_role='fit',expected_frames=data.FIT_FRAMES,
        collection_result=dict(complete=True,qualification_passed=False),collection_records={e:dict(controller_passed=e!=5014) for e in data.FIXED_EPISODES},
        qualified=True,failure_source_validation=dict(passed=True),controlled_source_validation=dict(passed=True),summary_input_provenance={},conflicts=[])


class SemanticCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.original=fixture();cache.save_role(self.root,self.original)
    def tearDown(self):self.temp.cleanup()

    def test_full_observation_target_history_signedzero_bytes_preserved_without_encoder(self):
        with patch.object(data,'SemanticStateDataset',side_effect=AssertionError('encoder called')):
            loaded=cache.CachedSemanticDataset(self.root,'fit')
        self.assertEqual(len(loaded),464);self.assertEqual(loaded.cache_readback['encoder_calls'],0)
        for a,b in zip(self.original.rows,loaded.rows):
            self.assertEqual(cache.row_hash(a),cache.row_hash(b))
            for key in ('target','observed_left_hand_world_height_m','observed_height_timestamp_s','_observed_summary'):
                self.assertEqual(a[key].dtype,b[key].dtype);self.assertEqual(a[key].tobytes(),b[key].tobytes())
            self.assertTrue(np.signbit(b['inputs']['coord'][0][0,1]))
        self.assertFalse(loaded.collection_result['qualification_passed'])

    def test_one_ulp_change_rejected_not_tolerated(self):
        with np.load(self.root/'fit.npz') as z:arrays={k:z[k] for k in z.files}
        arrays['coord'][0,0]=np.nextafter(arrays['coord'][0,0],np.float32(1.))
        np.savez(self.root/'fit.npz',**arrays)
        with self.assertRaisesRegex(ValueError,'exact array bytes changed'):
            cache.CachedSemanticDataset(self.root,'fit')

    def test_offset_corruption_with_selfconsistent_array_hash_still_rejected(self):
        with np.load(self.root/'fit.npz') as z:arrays={k:z[k] for k in z.files}
        meta=json.loads((self.root/'fit.json').read_text());arrays['offset'][1]=0
        meta['array_schema']['offset']['sha256']=hashlib.sha256(arrays['offset'].tobytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'offsets malformed'):cache.validate_arrays(arrays,meta)

    def test_label_never_changes_point_hash_or_history_input(self):
        loaded=cache.CachedSemanticDataset(self.root,'fit');row=loaded.rows[0]
        before=cache.row_hash(row);summary=row['_observed_summary'].copy()
        row['target'][:]=100;row['supervision']['mass_available']=1.
        self.assertEqual(cache.row_hash(row),before);np.testing.assert_array_equal(row['_observed_summary'],summary)

    def test_model_observation_boundary_ignores_all_supervision_and_metadata_changes(self):
        import torch
        from scripts.sugar.object_predictor import semantic_state_cached as training
        loaded=cache.CachedSemanticDataset(self.root,'fit');row=loaded.rows[0]
        a,h=training.prepare_batch([row],torch.device('cpu'),torch.ones(11),'observed_summary')
        modified=deepcopy(row);modified['target'][:]=17.
        modified['supervision']['mass_available']=np.float32(1)
        modified['supervision']['physics']['contact_present'][:]=1
        modified['evaluation_only']={'arbitrary_object_gt':100.}
        modified['metadata']['episode']=999;modified['metadata']['geometry_group']=999
        b,hh=training.prepare_batch([modified],torch.device('cpu'),torch.ones(11),'observed_summary')
        self.assertEqual(set(a['inputs']),{'coord','grid_coord','feat','offset'})
        self.assertTrue(all(torch.equal(a['inputs'][k],b['inputs'][k]) for k in a['inputs']))
        self.assertTrue(torch.equal(h,hh));self.assertFalse(torch.equal(a['target'],b['target']))

    def test_fit_development_disjoint_and_declared_order(self):
        self.assertEqual(len(data.FIT_FRAMES),29);self.assertEqual(len(data.EVALUATION_FRAMES),90)
        self.assertFalse(set(data.FIT_FRAMES)&set(data.EVALUATION_FRAMES))
        meta=json.loads((self.root/'fit.json').read_text());meta['metadata'][0]['frame']+=1
        cache.write_json(self.root/'fit.json',meta)
        with self.assertRaisesRegex(ValueError,'metadata/history differs'):cache.CachedSemanticDataset(self.root,'fit')


if __name__=='__main__':unittest.main()
