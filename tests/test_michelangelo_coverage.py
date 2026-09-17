"""CPU-only checks of the one-factor occupancy sampling experiment."""
import ast
import inspect
import unittest

import numpy as np

from scripts.sugar.object_predictor import michelangelo_coverage_data as coverage
from scripts.sugar.object_predictor import train_michelangelo_coverage as trainer
from scripts.sugar.object_predictor import train_michelangelo_overfit as previous


class CoverageSamplerTests(unittest.TestCase):
    def test_all_1000_steps_keep_near_indices_case_and_seed_schedule(self):
        counts = [0]*4
        for step in range(1, previous.STEPS+1):
            old_case, old = previous.train_indices(step)
            new_case, new = coverage.train_indices(step)
            self.assertEqual(old_case, new_case); counts[new_case] += 1
            np.testing.assert_array_equal(new[1024:], old[1024:])
            self.assertTrue(np.all((new[:512] >= 0) & (new[:512] < 8192)))
            self.assertTrue(np.all((new[512:1024] >= 8192) & (new[512:1024] < 16384)))
            local = new[512:1024]-8192
            self.assertEqual(int((local % 2 == 0).sum()),256)
            self.assertEqual(int((local % 4 == 1).sum()),128)
            self.assertEqual(int((local % 4 == 3).sum()),128)
            np.testing.assert_array_equal(local[::2]+1,local[1::2])
            self.assertEqual(len(np.unique(new)), 2048)
        self.assertEqual(counts, [250]*4)

    def test_equal_face_probability_ignores_area_and_replays_signed_pairs(self):
        # One triangle has 10,000x the other's area; face IDs still get 1:1
        # probability. This is a sampler unit test, not a substitute model.
        vertices = np.array([[0,0,0], [1,0,0], [0,1,0],
                             [0,0,2], [100,0,2], [0,100,2]], dtype=float)
        faces = np.array([[0,1,2], [3,4,5]])
        triangles, normals = coverage.source_triangles(vertices, faces)
        out = coverage.paired_surface_queries(triangles, normals, np.random.default_rng(178), 10000)
        ids = out['surface_face_indices']; bary = out['surface_barycentric']
        ratio = float((ids == 0).mean())
        self.assertGreater(ratio, .48); self.assertLess(ratio, .52)
        np.testing.assert_array_equal(ids[::2], ids[1::2])
        np.testing.assert_array_equal(bary[::2], bary[1::2])
        np.testing.assert_array_equal(out['surface_offsets_vae'], np.tile([0., -coverage.HALF_CELL_VAE, 0., coverage.HALF_CELL_VAE], 5000))
        replay = (np.einsum('ni,nij->nj', bary, triangles[ids])
                  + out['surface_normals']*out['surface_offsets_vae'][:,None]).astype(np.float32)
        np.testing.assert_array_equal(replay, out['queries'])
        center = np.einsum('ni,nij->nj', bary[::2],triangles[ids[::2]]).astype(np.float32)
        np.testing.assert_array_equal(center,out['queries'][::2])
        self.assertEqual(set(inspect.signature(coverage.paired_surface_queries).parameters),
                         {'triangles','normals','rng','pairs','pair_slots'})

    def test_degenerate_face_is_not_silently_filtered(self):
        with self.assertRaises(ValueError):
            coverage.source_triangles(np.zeros((3,3)), np.array([[0,1,2]]))

    def test_training_body_unchanged_except_protocol_metadata(self):
        old = ast.parse(inspect.getsource(previous.run)).body[0]
        new = ast.parse(inspect.getsource(trainer.run)).body[0]
        updates = [node for node in new.body if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                   and isinstance(node.value.func.value, ast.Name)
                   and node.value.func.value.id == 'protocol' and node.value.func.attr == 'update']
        self.assertEqual(len(updates), 1)
        new.body.remove(updates[0])
        self.assertEqual(ast.dump(new, include_attributes=False), ast.dump(old, include_attributes=False))
        for name in ('load_official_model','original_loss','optimizer_for','batch_loss','gradients',
                     'evaluate_queries','endpoint_meshes','audit_saved_checkpoint'):
            self.assertIs(getattr(trainer,name),getattr(previous,name))
        self.assertIs(trainer.train_indices,coverage.train_indices)
        self.assertIs(trainer.read_data,coverage.read_data)
        self.assertNotEqual(previous.SCOPE,coverage.SCOPE)

    def test_prepared_four_actual_pools_replay_and_exact_reuse(self):
        directory = previous.EXPERIMENT/'overfit_repair_v1/michelangelo_coverage_data_v2'
        manifest, data = coverage.read_data(directory)
        old_manifest, old_data = previous.read_data(coverage.OLD_DATA)
        self.assertEqual(len(data),4)
        for item, old in zip(data,old_data):
            for key in ('input_surface_xyz_vae','input_normals','eval_queries_vae','eval_labels'):
                np.testing.assert_array_equal(item[key],old[key])
            for key in ('train_queries_vae','train_labels','train_surface_distance_abc'):
                np.testing.assert_array_equal(item[key][16384:],old[key][16384:])
            self.assertEqual(item['train_queries_vae'].shape,(32768,3))
            self.assertEqual(item['surface_face_indices'].shape,(8192,))
            self.assertTrue(np.any(np.abs(item['train_queries_vae'][:8192]) > 1.1))


if __name__ == '__main__':
    unittest.main()
