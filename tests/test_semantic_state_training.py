"""Pure CPU contracts for paired budgets, scope and dynamic semantic gates."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.sugar.object_predictor import semantic_state_data as data
from scripts.sugar.object_predictor import semantic_state_training as training


def arrays(frames,force=0.):
    identities=[(e,f) for e in data.FIXED_EPISODES for f in frames]
    n=len(identities);e=np.asarray([e for e,f in identities]);f=np.asarray([f for e,f in identities])
    available=((e!=5014)&(f>=1181)).astype(float)
    values=dict(episode=e,frame=f,mass_available=available,availability_probability=available.copy(),
        state_precision_eligible=np.ones(n),force_rmse_n=np.full(n,force))
    for key in ('center_cm','mesh_nn_cm','rotation_deg','size_mean_relative','size_max_relative','mass_relative'):
        values[key]=np.zeros(n)
    return values


class SemanticTrainingTests(unittest.TestCase):
    def test_fixed_full464_epoch_and_matched_order(self):
        dataset=SimpleNamespace(rows=[dict(metadata=dict(episode=e,frame=f)) for e in data.FIXED_EPISODES for f in data.FIT_FRAMES])
        a=training.fixed_batches(dataset,3);b=training.fixed_batches(dataset,3)
        for _ in range(20):
            aa,bb=next(a),next(b)
            self.assertEqual(aa,bb);self.assertEqual(len(aa),116)
            self.assertEqual(sorted(i for batch in aa for i in batch),list(range(464)))
            self.assertTrue(all(len(batch)==4 for batch in aa))

    def test_fixed_cosine_budget_and_new_rates(self):
        self.assertEqual(training.learning_rates(1),training.LR_BASE)
        for name,rate in training.LR_BASE.items():self.assertAlmostEqual(training.learning_rates(20)[name],.05*rate)
        for update in (0,21,20.):
            with self.assertRaises(ValueError):training.learning_rates(update)

    def test_semantic_gate_excludes_only_force_keeps_unknown_raw_and_mass_na(self):
        values=arrays(data.FIT_FRAMES,force=10.)
        result=training.acceptance(values,'fit')
        self.assertTrue(result['semantic_acceptance_passed']);self.assertFalse(result['legacy_acceptance_passed'])
        self.assertFalse(result['legacy_force_gate_passed'])
        self.assertIsNone(result['per_case']['5014']['mass_precision_passed'])
        self.assertEqual(result['per_case']['5014']['available_mass_denominator'],0)
        bad={k:v.copy() for k,v in values.items()};bad['center_cm'][0]=1.01
        self.assertFalse(training.acceptance(bad,'fit')['semantic_acceptance_passed'])
        bad['state_precision_eligible'][0]=0
        result=training.acceptance(bad,'fit')
        self.assertTrue(result['passed']);self.assertEqual(result['all_items']['center_cm']['maximum'],1.01)

    def test_dense_singleclass_specificity_and_global_falseconfidence(self):
        values=arrays(data.EVALUATION_FRAMES)
        self.assertTrue(training.acceptance(values,'same_trajectory_interpolation')['passed'])
        values['availability_probability'][values['episode']==5014]=1
        result=training.acceptance(values,'same_trajectory_interpolation')
        self.assertFalse(result['per_case']['5014']['checks']['availability_specificity'])
        self.assertFalse(result['passed'])

    def test_subset_clock_contract_and_missing_clock_rejection(self):
        values=arrays(data.FIXED_FRAMES)
        self.assertTrue(training.acceptance(values,'fit',expected_frames=data.FIXED_FRAMES)['passed'])
        self.assertFalse(training.acceptance(values,'fit')['passed'])

    def test_no_summary_qualification_removes_height_from_observation_key(self):
        row=dict(observed_left_hand_world_height_m=np.ones((32,1),np.float32))
        dataset=SimpleNamespace(rows=[row])
        with patch.object(data,'qualification',side_effect=lambda d,l:float(d.rows[0]['observed_left_hand_world_height_m'].sum())):
            self.assertEqual(training.qualify_condition(dataset,'no_summary',{}),0.)
            self.assertEqual(training.qualify_condition(dataset,'observed_summary',{}),32.)
        self.assertEqual(row['observed_left_hand_world_height_m'].sum(),32.)

    def test_single_artifact_binding_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'PROTOCOL.json').write_text('{}')
            source=root/'source.py';source.write_text('x=1')
            sources={name:dict(path=str(source),sha256=data.sha(source)) for name in set(training.original.SOURCE_MODULES)|set(training.EXTRA_SOURCES)}
            for name in training.ARTIFACTS:(root/name).write_bytes(name.encode())
            manifest=dict(schema=6,kind=training.STUDY,complete=True,sources=sources,
                protocol=dict(path='PROTOCOL.json',sha256=data.sha(root/'PROTOCOL.json')),
                artifacts={name:dict(path=name,sha256=data.sha(root/name)) for name in training.ARTIFACTS})
            training.verify_artifacts(root,manifest=manifest)
            (root/'fit_02120.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                training.verify_artifacts(root,manifest=manifest)


if __name__=='__main__':unittest.main()
