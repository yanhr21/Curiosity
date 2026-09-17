"""CPU artifact/clock checks only; never instantiate a renderer or a model."""
from copy import deepcopy
import unittest

import numpy as np

from scripts.sugar.object_predictor.render_state_overfit import (
    CONDITIONAL_PROFILE, EVIDENCE_CODES, bind_saved_world_states, evidence_label,
    held_evidence, validate_state_evidence, episode_shards, merge_shard_results,
    worker_environment, WORKER_THREAD_LIMITS, validate_decoded_panel_frame)


def fixture():
    protocol = dict(supervision_profile=CONDITIONAL_PROFILE,
                    collection_study='controlled_fixture16_v1',
                    state_evidence_status_codes=EVIDENCE_CODES.copy())
    arm = dict(episode=np.array([5000, 5000, 5008, 5008]),
               frame=np.array([31, 56, 31, 56]),
               state_precision_eligible=np.array([0., 1., 0., 1.]),
               state_contact_history_frames=np.array([0, 1, 0, 9]),
               state_evidence_status=np.array([0, 1, 0, 1]))
    return protocol, [arm, deepcopy(arm)]


class RenderStateOverfitTests(unittest.TestCase):
    def test_default_does_not_require_or_reinterpret_new_fields(self):
        self.assertFalse(validate_state_evidence({}, [{}, {}]))
        self.assertFalse(validate_state_evidence(dict(supervision_profile='all_state'), [{}, {}]))

    def test_saved_evidence_is_bound_to_clocks_not_row_order_or_predictions(self):
        protocol, arms = fixture()
        self.assertTrue(validate_state_evidence(protocol, arms))
        order = np.array([3, 1, 2, 0])
        arms[1] = {key: value[order] for key, value in arms[1].items()}
        arms[0]['prediction'] = np.zeros((4, 13))
        arms[1]['prediction'] = np.ones((4, 13)) * 100.
        self.assertTrue(validate_state_evidence(protocol, arms))
        arms[1]['state_contact_history_frames'][0] = 8
        with self.assertRaisesRegex(ValueError, 'Before/after'):
            validate_state_evidence(protocol, arms)

    def test_wrong_scope_missing_fields_and_inconsistent_evidence_rejected(self):
        protocol, arms = fixture()
        for edit in (
            lambda p, a: p.update(collection_study='old_blind'),
            lambda p, a: p.update(state_evidence_status_codes={'0': 'confident'}),
            lambda p, a: a[0].pop('state_evidence_status'),
            lambda p, a: a[0].update(state_evidence_status=np.array([0., 1., 0., 1.])),
            lambda p, a: a[0]['state_contact_history_frames'].__setitem__(0, 1),
            lambda p, a: a[0]['state_contact_history_frames'].__setitem__(1, 33),
            lambda p, a: a[0]['state_precision_eligible'].__setitem__(1, np.nan),
            lambda p, a: a[0]['frame'].__setitem__(1, 31),
        ):
            with self.subTest(edit=edit):
                p, a = deepcopy(protocol), deepcopy(arms)
                edit(p, a)
                with self.assertRaises(ValueError): validate_state_evidence(p, a)

    def test_held_evidence_is_causal_and_unknown_until_saved_contact_clock(self):
        clocks = np.array([31, 56, 81]); times = np.arange(100) * .02
        status = np.array([0, 1, 1]); contact = np.array([0, 1, 26])
        call = lambda f: held_evidence(clocks, times, f, status, contact)
        self.assertEqual(call(30)['index'], -1)
        self.assertEqual(evidence_label(call(30))[0], '积累历史')
        before = call(55)
        self.assertEqual(before['prediction_frame'], 31)
        self.assertEqual(before['contact_history_frames'], 0)
        self.assertAlmostEqual(before['age_s'], .48)
        self.assertEqual(evidence_label(before)[0], '先验/未知')
        at_contact = call(56)
        self.assertEqual(at_contact['prediction_frame'], 56)
        self.assertEqual(at_contact['age_s'], 0.)
        self.assertEqual(evidence_label(at_contact)[0], '接触候选')
        self.assertIn('未证明', evidence_label(at_contact)[1])

    def test_malformed_or_future_time_clocks_are_rejected(self):
        times = np.arange(100) * .02
        for clocks in (np.array([31, 31]), np.array([56, 31]), np.array([31., 56.]), np.array([31, 100])):
            with self.subTest(clocks=clocks):
                with self.assertRaises(ValueError):
                    held_evidence(clocks, times, 55, [0, 1], [0, 1])
        times[55] = times[31] - .02
        with self.assertRaisesRegex(ValueError, 'future clock'):
            held_evidence(np.array([31, 56]), times, 55, [0, 1], [0, 1])

    def test_world_state_remains_bound_to_own_hand_clock(self):
        from scipy.spatial.transform import Rotation
        from scripts.sugar.object_predictor.report_prospective_carry import world_state
        clocks = np.array([31, 56]); times = np.arange(100) * .02
        hands = np.zeros((100, 2, 7)); hands[..., 6] = 1.
        hands[31, 0, :3] = [.2, -.1, .4]
        hands[31, 0, 3:] = Rotation.from_euler('z', 35, degrees=True).as_quat()
        hands[55, 0, :3] = [1., 2., 3.]
        predicted = np.zeros((2, 13)); predicted[:, 3] = predicted[:, 7] = 1.
        predicted[:, :3] = [.03, -.02, .01]
        predicted[:, 9:] = np.log([.10, .20, .30, .50])
        states = bind_saved_world_states(predicted, clocks, hands)
        context = held_evidence(clocks, times, 55, [0, 1], [0, 1])
        expected = world_state(predicted[0], hands[31, 0])
        for actual, target in zip(states[context['index']], expected):
            np.testing.assert_array_equal(actual, target)
        # The renderer receives moving current hands separately. Altering that
        # frame cannot change either cached world state or its binding function.
        hands[55, 0, :3] += 10.
        repeated = bind_saved_world_states(predicted, clocks, hands)
        for actual, target in zip(repeated[0], expected):
            np.testing.assert_array_equal(actual, target)
        self.assertGreater(np.linalg.norm(world_state(predicted[0], hands[55, 0])[0] - states[0][0]), 1.)


def shard_receipts():
    episodes=tuple(range(5000,5004))+tuple(range(5008,5016))+tuple(range(5020,5024))
    reports=[]
    for index,shard in enumerate(episode_shards(episodes,4)):
        reports.append(dict(shard_complete=True,render_scope='episode_shard',shard_index=index,
            episode_order=list(shard),frames=960,fps=20,all_frames_decoded=True,
            physical_frame_clocks=list(range(1,2400,10)),
            inference_frame_clocks=list(range(31,2400,25)),
            worker_thread_limits=WORKER_THREAD_LIMITS.copy(),ffmpeg_threads=1,
            supervision_profile=CONDITIONAL_PROFILE,state_evidence_status_codes=EVIDENCE_CODES,
            cases=[dict(episode=episode,frames=240,video_start_s=k*12,
                        controller_passed=episode!=5012,prior_unknown_prediction_clocks=40,
                        contact_candidate_prediction_clocks=55) for k,episode in enumerate(shard)],
            stills=[f'episode_{episode}_frame_{frame:04d}.png'
                    for episode in shard for frame in (31,1181,1581,2381)],actual_mesh_stills=16))
    return episodes,reports


class ParallelRenderInterfaceTests(unittest.TestCase):
    def test_episode_partition_and_thread_budget_are_fixed(self):
        episodes,reports=shard_receipts()
        self.assertEqual(episode_shards(episodes,1),(episodes,))
        self.assertEqual(tuple(e for shard in episode_shards(episodes,4) for e in shard),episodes)
        original={'SLURM_STEP_ID':'0','OMP_NUM_THREADS':'8','LP_NUM_THREADS':'2'}
        result=worker_environment(original)
        self.assertEqual(result['SLURM_STEP_ID'],'0')
        self.assertEqual(original['OMP_NUM_THREADS'],'8')
        for key in WORKER_THREAD_LIMITS:self.assertEqual(result[key],'1')
        with self.assertRaises(ValueError):episode_shards(episodes,8)
        with self.assertRaises(ValueError):episode_shards(episodes[:8],4)

    def test_merge_preserves_all_failures_unknown_counts_and_fixed_timeline(self):
        episodes,reports=shard_receipts()
        result=merge_shard_results(reports,episodes)
        self.assertEqual(result['frames'],3840)
        self.assertEqual(result['actual_mesh_stills'],64)
        self.assertEqual([case['episode'] for case in result['cases']],list(episodes))
        self.assertEqual([case['video_start_s'] for case in result['cases']],[i*12 for i in range(16)])
        self.assertFalse(result['cases'][8]['controller_passed'])
        self.assertEqual(sum(case['prior_unknown_prediction_clocks'] for case in result['cases']),640)
        # Pure metadata merge cannot stand in for the required final decode.
        self.assertNotIn('complete',result)
        self.assertNotIn('all_frames_decoded',result)

    def test_incomplete_misordered_wrong_clock_or_false_whole_run_shards_rejected(self):
        episodes,reports=shard_receipts()
        for edit in (
            lambda r:r.pop(),
            lambda r:r[0].update(complete=True),
            lambda r:r[1].update(frames=959),
            lambda r:r[1].update(all_frames_decoded=False),
            lambda r:r[1]['episode_order'].reverse(),
            lambda r:r[1]['cases'].reverse(),
            lambda r:r[2]['physical_frame_clocks'].__setitem__(10,102),
            lambda r:r[2]['inference_frame_clocks'].__setitem__(0,32),
            lambda r:r[2]['stills'].pop(),
            lambda r:r[3]['cases'][0].update(video_start_s=1.),
            lambda r:r[3].update(supervision_profile='all_state'),
            lambda r:r[3].update(ffmpeg_threads=8),
        ):
            with self.subTest(edit=edit):
                changed=deepcopy(reports);edit(changed)
                with self.assertRaises(ValueError):merge_shard_results(changed,episodes)

    def test_full_decode_panel_rules_still_include_warmup_and_all_later_panels(self):
        frames=list(range(1,2400,10))
        rgb=np.zeros((960,1600,3),np.uint8)
        stripe=np.tile((np.arange(520)%2)*255,(580,1)).astype(np.uint8)
        rgb[144:724,8:528]=stripe[:,:,None]
        validate_decoded_panel_frame(rgb,0,frames)
        with self.assertRaisesRegex(ValueError,'Blank'):
            validate_decoded_panel_frame(rgb,3,frames)  # Frame31 has predictions.
        for panel in (1,2):rgb[144:724,8+532*panel:528+532*panel]=stripe[:,:,None]
        validate_decoded_panel_frame(rgb,3,frames)
        validate_decoded_panel_frame(rgb,3840-1,frames)
        with self.assertRaisesRegex(ValueError,'shape'):
            validate_decoded_panel_frame(rgb[:900],3,frames)


if __name__ == '__main__':
    unittest.main()
