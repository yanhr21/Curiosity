"""CPU observation-array fixtures, no prediction model or simulator."""
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts.sugar.object_predictor.observed_force_summary import (
    summarize_observed_forces, summarize_support_projections)
from scripts.sugar.object_predictor.overfit_data import observed_physics_targets


def collate(frames):
    feat = np.concatenate(frames).astype(np.float32)
    return dict(feat=torch.from_numpy(feat), coord=torch.from_numpy(feat[:, :3].copy()),
                grid_coord=torch.zeros((len(feat), 3), dtype=torch.int32),
                offset=torch.tensor(np.cumsum([len(f) for f in frames]), dtype=torch.int64))


def fixture(batch=2):
    rng = np.random.default_rng(2941)
    frames = []
    for b in range(batch):
        down = Rotation.from_euler('x', b*.3).apply([0., 0., -1.])
        for t in range(32):
            n = 1 + (b+t) % 5
            f = np.zeros((n, 20), np.float32)
            f[:, :3] = rng.normal(size=(n, 3))
            normal = rng.normal(size=(n, 3))
            f[:, 6:9] = normal / np.linalg.norm(normal, axis=1, keepdims=True)
            f[:, 9] = 1
            f[:, 10] = np.log1p(rng.uniform(0, 4, n))
            f[:, 11:14] = np.arcsinh(rng.normal(size=(n, 3))/2)
            f[:, 14] = 1
            f[:, 15] = (t-31)*.02/150
            f[:, 16] = rng.uniform(0, 1, n)
            f[:, 17:20] = down
            frames.append(f)
    return frames


def oracle(frame):
    x = observed_physics_targets(frame)
    return (np.concatenate([x[k] for k in ('normal_load_by_encoded_side_n',
        'shear_on_hand_current_frame_n', 'normal_on_object_approx_current_frame_n')]),
        np.concatenate([x[k] for k in ('shear_support_up_n', 'combined_support_up_approx_n')]))


class ObservedForceSummaryTests(unittest.TestCase):
    def test_original_oracle_exact_every_frame_nonuniform_cross_batch(self):
        frames = fixture()
        inputs = collate(frames)
        force, support = summarize_observed_forces(inputs), summarize_support_projections(inputs)
        self.assertEqual(tuple(force.shape), (2, 32, 8))
        self.assertEqual(tuple(support.shape), (2, 32, 2))
        self.assertEqual(force.dtype, torch.float32)
        for i, frame in enumerate(frames):
            a, b = oracle(frame)
            np.testing.assert_array_equal(force[i//32, i%32], a)
            np.testing.assert_array_equal(support[i//32, i%32], b)

    def test_current_frame_offset_and_no_temporal_accumulation(self):
        frames = fixture()
        inputs = collate(frames)
        output = summarize_observed_forces(inputs)
        for b in range(2):
            k = b*32+31
            start, stop = inputs['offset'][k-1:k+1].tolist()
            np.testing.assert_array_equal(output[b,31], oracle(inputs['feat'][start:stop].numpy())[0])
            self.assertFalse(np.array_equal(output[b,31], output[b].sum(0)))
        one = summarize_observed_forces(collate(frames[:32]))
        np.testing.assert_array_equal(one[0], output[0])

    def test_zero_contact_preserves_anchors_and_zero_force(self):
        frames = fixture(1)
        for f in frames:
            f[:, 9:15] = 0
        x = collate(frames)
        np.testing.assert_array_equal(summarize_observed_forces(x), 0)
        np.testing.assert_array_equal(summarize_support_projections(x), 0)

    def test_mixed_side_is_encoded_area_fraction_not_hidden_true_split(self):
        f = np.zeros((1, 20), np.float32)
        f[:, 6] = 1
        f[:, 9] = 1
        f[:, 10] = np.log1p(4.)
        f[:, 14] = 1
        f[:, 16] = .5
        f[:, 19] = -1
        result = summarize_observed_forces(collate([f]*32))[0,0].numpy()
        np.testing.assert_allclose(result[:2], [2.,2.], atol=1e-6, rtol=0)
        self.assertFalse(np.allclose(result[:2], [1.,3.]))

    def test_vector_coordinate_covariance_and_support_invariance(self):
        frames = fixture(1)
        original = collate(frames)
        q = Rotation.from_euler('xyz', [.3,-.8,1.1]).as_matrix()
        rotated = []
        for f in frames:
            g = f.copy()
            g[:, :3] = f[:, :3] @ q.T
            g[:, 6:9] = f[:, 6:9] @ q.T
            g[:, 11:14] = np.arcsinh((2*np.sinh(f[:,11:14].astype(float)) @ q.T)/2)
            g[:, 17:20] = f[:, 17:20] @ q.T
            rotated.append(g)
        a = summarize_observed_forces(original).numpy()
        b = summarize_observed_forces(collate(rotated)).numpy()
        np.testing.assert_array_equal(a[:,:,:2], b[:,:,:2])
        for part in (slice(2,5), slice(5,8)):
            np.testing.assert_allclose(b[:,:,part], a[:,:,part] @ q.T, atol=3e-6, rtol=0)
        np.testing.assert_allclose(summarize_support_projections(original),
            summarize_support_projections(collate(rotated)), atol=3e-6, rtol=0)

    def test_coordinates_do_not_rescale_newtons_and_inputs_are_immutable(self):
        x = collate(fixture(1)); before = {k:v.clone() for k,v in x.items()}
        force = summarize_observed_forces(x)
        for k in x: self.assertTrue(torch.equal(x[k],before[k]))
        x['coord'] *= 5; x['feat'][:,:3] *= 5
        np.testing.assert_array_equal(force, summarize_observed_forces(x))
        self.assertFalse(force.requires_grad)

    def test_label_mask_metadata_input_and_grad_are_rejected(self):
        x = collate(fixture(1))
        for key in ('target','mass_available','physics','metadata'):
            with self.assertRaises(ValueError): summarize_observed_forces(dict(x, **{key:torch.ones(1)}))
        x['feat'].requires_grad_(True)
        with self.assertRaises(ValueError): summarize_observed_forces(x)

    def test_invalid_boundaries_and_units_fail_closed(self):
        for problem in ('repeated_offset','wrong_end','short_history','nan','negative_load','side','gravity'):
            x = collate(fixture(1))
            if problem == 'repeated_offset': x['offset'][1] = x['offset'][0]
            elif problem == 'wrong_end': x['offset'][-1] -= 1
            elif problem == 'short_history': x['offset'] = x['offset'][:-1]
            elif problem == 'nan': x['feat'][0,0] = float('nan')
            elif problem == 'negative_load': x['feat'][0,10] = -1
            elif problem == 'side': x['feat'][0,16] = 2
            else: x['feat'][:,17:20] *= 2
            with self.subTest(problem=problem), self.assertRaises(ValueError): summarize_observed_forces(x)


if __name__ == '__main__': unittest.main()
