import unittest
import numpy as np
from scripts.sugar.object_predictor import render_semantic_state_cached_pair as render


class SemanticRenderClockTests(unittest.TestCase):
    def test_new_early_clocks_are_causal_and_keep_all_original_clocks(self):
        clocks=np.asarray(render.INFERENCE_FRAMES)
        self.assertEqual(len(clocks),119)
        self.assertTrue(set(range(31,2400,25)).issubset(set(clocks)))
        self.assertTrue(set(range(49,1200,50)).issubset(set(clocks)))
        times=.02*np.arange(1,2401)
        for frame,expected in [(30,None),(31,31),(48,31),(49,49),(55,49),(56,56),(1198,1181),(1199,1199),(1200,1199),(1206,1206),(2399,2381)]:
            result=render.held_evidence(clocks,times,frame,np.zeros(119,dtype=int),np.zeros(119,dtype=int))
            self.assertEqual(result['prediction_frame'],expected)
            if expected is not None:self.assertGreaterEqual(result['age_s'],0.)

    def test_semantic_mask_mismatch_between_matched_arms_rejected(self):
        clocks=np.asarray(render.INFERENCE_FRAMES)
        arm=dict(episode=np.full(119,5000),frame=clocks,state_precision_eligible=np.zeros(119),
            state_contact_history_frames=np.zeros(119,dtype=int),state_evidence_status=np.zeros(119,dtype=int))
        p=dict(supervision_profile=render.SEMANTIC_PROFILE,collection_study='controlled_fixture16_v1',
            state_evidence_status_codes=render.EVIDENCE_CODES)
        other={k:v.copy() for k,v in arm.items()}
        self.assertTrue(render.validate_state_evidence(p,[arm,other]))
        other['state_precision_eligible'][1]=1
        other['state_evidence_status'][1]=1
        other['state_contact_history_frames'][1]=1
        with self.assertRaisesRegex(ValueError,'clocks differ'):render.validate_state_evidence(p,[arm,other])


if __name__=='__main__':unittest.main()
