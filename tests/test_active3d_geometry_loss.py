"""CPU geometry-operator tests with explicitly synthetic triangles, no model."""
import math
import unittest

import torch

from scripts.sugar.object_predictor.active3d_geometry_loss import (
    GeometryRepairLoss, formal_repair_gate, geometry_case_gate,
)
from scripts.sugar.object_predictor.train_active3d_overfit import training_objective


class GeometryRepairTests(unittest.TestCase):
    def square(self, scale=1.):
        vertices = scale*torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
        x, y = torch.meshgrid(torch.linspace(0, scale, 21), torch.linspace(0, scale, 21), indexing='ij')
        target = torch.stack([x.flatten(), y.flatten(), torch.zeros(x.numel())], dim=1)[None]
        return vertices, faces, target

    def test_disabled_loss_tensor_gradient_and_rng_are_exact(self):
        torch.manual_seed(123)
        vertices = torch.randn(2, 5, 3, requires_grad=True)
        original = (vertices.square()+vertices.sin()).mean()
        rng = torch.get_rng_state().clone()
        total, record = training_objective(original, vertices)
        self.assertIs(total, original)
        self.assertEqual(record, {})
        g0 = torch.autograd.grad(original, vertices, retain_graph=True)[0]
        g1 = torch.autograd.grad(total, vertices)[0]
        self.assertTrue(torch.equal(g0, g1))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_enabled_official_regularizers_have_finite_gradients_without_gt_gradients(self):
        gt, faces, target = self.square()
        gt.requires_grad_(True); target.requires_grad_(True)
        loss = GeometryRepairLoss(target, [gt], [faces], faces)
        prediction = gt.detach().clone()
        prediction[2, 2] = .2
        prediction = prediction[None].requires_grad_(True)
        original = prediction.square().mean()
        total, record = training_objective(original, prediction, loss)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0.)
        self.assertGreater(record['normalized_official_normal'], 0.)
        self.assertGreater(record['normalized_official_uniform_laplacian'], 0.)
        self.assertIsNone(gt.grad)
        self.assertIsNone(target.grad)
        self.assertFalse(torch.cuda.is_initialized())

    def test_correct_face_area_inflation_cannot_dilute_bad_face_distance_terms(self):
        # Increase only one correct triangle's area fourfold. All seven of its
        # probes remain exactly on target grid points; the bad face is unchanged.
        x, y = torch.meshgrid(torch.arange(-15., 16.), torch.arange(-15., 16.), indexing='ij')
        target = torch.stack([x.flatten(), y.flatten(), torch.zeros(x.numel())], dim=1)[None]
        gt = torch.tensor([[-15., -15., 0.], [15., -15., 0.], [15., 15., 0.], [-15., 15., 0.]])
        gt_faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        small = torch.tensor([[0., 0., 0.], [6., 0., 0.], [0., 6., 0.],
                              [0., 0., 2.], [6., 0., 2.], [0., 6., 2.]])[None]
        large = small.clone(); large[:, 1:3] *= 2
        repair = GeometryRepairLoss(target, [gt], [gt_faces], faces)
        a, b = repair.geometry(small), repair.geometry(large)
        self.assertGreater(b['predicted_global_area'].item(), a['predicted_global_area'].item())
        for key in ('uniform_vertex_d2', 'uniform_face_probe_d2', 'worst_face_d2',
                    'maximum_face_probe_label_distance', 'bad_face_fraction'):
            self.assertTrue(torch.equal(a[key], b[key]), key)
        self.assertEqual(a['worst_face_d2'].item(), 4.)
        self.assertEqual(a['bad_face_fraction'].item(), .5)

    def test_regularizer_scale_is_dimensionless_while_distances_scale_squared(self):
        records = []
        for scale in (1., 2.):
            gt, faces, target = self.square(scale)
            pred = gt.clone(); pred[2, 2] = .2*scale
            pred = pred[None].requires_grad_(True)
            repair = GeometryRepairLoss(target, [gt], [faces], faces)
            _, record = repair.augment(pred.sum()*0, pred)
            records.append(record)
        for key in ('normalized_official_edge', 'normalized_official_normal',
                    'normalized_official_uniform_laplacian', 'excess_area'):
            self.assertAlmostEqual(records[0][key], records[1][key], places=5)
        for key in ('uniform_vertex_d2', 'uniform_face_probe_d2', 'worst_face_d2'):
            self.assertAlmostEqual(4*records[0][key], records[1][key], places=6)

    def test_batch_uses_each_gt_area_and_keeps_case_dimensions(self):
        gt, faces, target = self.square()
        repair = GeometryRepairLoss(torch.cat([target, 2*target]), [gt, 2*gt], [faces, faces], faces)
        rows = repair.evaluate(torch.stack([gt, 2*gt]))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]['truth_area'], 1.)
        self.assertAlmostEqual(rows[1]['truth_area'], 4.)
        self.assertAlmostEqual(rows[1]['nominal_edge'], 2*rows[0]['nominal_edge'])
        self.assertAlmostEqual(rows[0]['area_ratio'], 1.)
        self.assertAlmostEqual(rows[1]['area_ratio'], 1.)

    def test_geometry_gate_boundaries_and_each_failure(self):
        valid = dict(area_ratio=1.5, maximum_vertex_label_distance=.01,
                     maximum_face_probe_label_distance=.01, bad_face_fraction=0.,
                     predicted_global_area=1.5, truth_area=1.)
        self.assertTrue(all(geometry_case_gate(valid).values()))
        self.assertTrue(all(geometry_case_gate(dict(valid, area_ratio=.7)).values()))
        for key, value in [('area_ratio', 1.50001), ('area_ratio', .69999),
                           ('maximum_vertex_label_distance', .010001),
                           ('maximum_face_probe_label_distance', .010001),
                           ('bad_face_fraction', 1/2304), ('truth_area', math.nan)]:
            with self.subTest(key=key, value=value):
                self.assertFalse(all(geometry_case_gate(dict(valid, **{key: value})).values()))

    def test_formal_gate_requires_old_numeric_geometry_render_and_bound_visual_review(self):
        names = [f'{ident}_repeat{r}' for ident in ('18704', '11898', '15737', '13266') for r in (0, 1)]
        training = dict(complete=True, fixed_input_cases=8, optimizer_updates=1000,
                        native_numeric_gate_passed=True, geometry_gate_passed=True,
                        checkpoint_reload=dict(passed=True), checkpoint_sha256='synthetic_unit_test')
        rendering = dict(complete=True, actual_mesh_stills=24, frames=768, all_frames_decoded=True,
                         records=[dict(name=name) for name in names], checkpoint_sha256='synthetic_unit_test')
        review = dict(checkpoint_sha256='synthetic_unit_test', cases=[dict(name=name, passed=True,
                      images=[f'{name}_view{v:02d}.png' for v in (0, 32, 64)]) for name in names])
        self.assertFalse(formal_repair_gate(training)['passed'])
        self.assertFalse(formal_repair_gate(training, rendering)['passed'])
        self.assertTrue(formal_repair_gate(training, rendering, review)['passed'])
        for key in ('native_numeric_gate_passed', 'geometry_gate_passed'):
            self.assertFalse(formal_repair_gate(dict(training, **{key: False}), rendering, review)['passed'])
        self.assertFalse(formal_repair_gate(training, rendering, dict(review, cases=review['cases'][:-1]))['passed'])
        self.assertFalse(formal_repair_gate(training, rendering, dict(review, checkpoint_sha256='wrong'))['passed'])
        self.assertFalse(formal_repair_gate(training, dict(rendering, checkpoint_sha256='old_run'), review)['passed'])
        review['cases'][4]['passed'] = False
        self.assertFalse(formal_repair_gate(training, rendering, review)['passed'])


if __name__ == '__main__':
    unittest.main()
