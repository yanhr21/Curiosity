"""CPU tests of the original-loss wiring and fixed data protocol; no model runs."""
import unittest

import numpy as np

from scripts.sugar.object_predictor.train_michelangelo_overfit import (
    BATCH_COUNT, BOUNDARY_ABC, OBJECT_IDS, POOL_COUNT, STEPS,
    occupancy_metrics, original_loss, read_cases, DATASET, train_indices, training_pool,
)
from scripts.sugar.object_predictor.michelangelo_geometry_coordinates import correct_extracted_coordinates


class OriginalAEFinetuneTests(unittest.TestCase):
    def test_exact_fixed_budget_all_four_and_query_halves(self):
        counts = [0]*4
        for step in range(1, STEPS+1):
            case, ids = train_indices(step)
            counts[case] += 1
            self.assertEqual(ids.shape, (2*BATCH_COUNT,))
            self.assertEqual(len(np.unique(ids)), 2*BATCH_COUNT)
            self.assertTrue(np.all(ids[:BATCH_COUNT] < POOL_COUNT))
            self.assertTrue(np.all(ids[BATCH_COUNT:] >= POOL_COUNT))
        self.assertEqual(counts, [250]*4)
        np.testing.assert_array_equal(train_indices(527)[1], train_indices(527)[1])
        for bad in (0, STEPS+1):
            with self.assertRaises(ValueError):
                train_indices(bad)

    def test_original_loss_values_and_backward_no_replacement(self):
        import torch
        criterion = original_loss()
        from michelangelo.models.modules.distributions import DiagonalGaussianDistribution
        self.assertEqual(type(criterion).__module__, 'michelangelo.models.tsal.loss')
        logits = torch.linspace(-2, 2, 2*BATCH_COUNT).reshape(1, -1).requires_grad_()
        labels = (torch.arange(2*BATCH_COUNT) % 3 == 0).float()[None]
        moments = torch.full((1, 256, 128), .2, requires_grad=True)
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        loss, log = criterion(posterior, logits, labels)
        expected = (torch.nn.functional.binary_cross_entropy_with_logits(logits[:, :BATCH_COUNT], labels[:, :BATCH_COUNT])
                    + .1*torch.nn.functional.binary_cross_entropy_with_logits(logits[:, BATCH_COUNT:], labels[:, BATCH_COUNT:])
                    + .001*posterior.kl(dims=(1, 2)).mean())
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(moments.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertGreater(float(moments.grad.abs().sum()), 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_occupancy_sign_exact_zero_and_empty_union(self):
        row = occupancy_metrics(np.array([1, 0, 1, 0]), np.array([0., -1., -2., 3.]))
        self.assertEqual(row['occupancy_iou'], 1/3)
        self.assertEqual(row['false_negative'], 1)
        self.assertEqual(row['false_positive'], 1)
        self.assertEqual(occupancy_metrics(np.zeros(3), -np.ones(3))['occupancy_iou'], 0)
        with self.assertRaises(ValueError):
            occupancy_metrics(np.zeros(2), [np.nan, 1])

    def test_new_small_real_mesh_pool_is_repeatable_binary_and_off_boundary(self):
        import trimesh
        from scripts.sugar.object_predictor.qualify_michelangelo_ae import ABC_TO_VAE
        # Actual complete original ABC asset; this is a data check, not a toy network.
        case = read_cases(DATASET)[0]
        first = training_pool(case, 0, count=64)
        second = training_pool(case, 0, count=64)
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)
        q, labels, distance, _ = first
        self.assertEqual(q.dtype, np.float32)
        self.assertEqual(q.shape, (128, 3))
        self.assertGreater(float(distance.min()), BOUNDARY_ABC)
        truth = trimesh.Trimesh(case['vertices'], case['faces'], process=False)
        np.testing.assert_array_equal(labels, truth.contains(q.astype(np.float64)/ABC_TO_VAE))

    def test_grid_correction_recovers_endpoint_inclusive_coordinate_not_topology(self):
        rng = np.random.default_rng(42)
        indices = rng.uniform(0, 128, (100, 3))
        raw = indices/129*2.5-1.25
        expected = indices/128*2.5-1.25
        corrected = correct_extracted_coordinates(raw)
        np.testing.assert_allclose(corrected, expected, atol=5e-16)
        self.assertEqual(corrected.shape, raw.shape)


if __name__ == '__main__':
    unittest.main()
