"""CPU arithmetic only: no model construction, training, or GPU benchmark."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .train_single_gpu import accumulate_cpu_gradients


class FlatGradientAccumulationTest(unittest.TestCase):
    def test_reusable_staging_matches_original_and_never_owns_retained_gradients(self):
        original, reusable, staging = {}, {}, {}
        for step in range(8):
            large = torch.arange(64, dtype=torch.float32) + step * 0.125
            small = torch.arange(16, dtype=torch.float32) - step * 0.25
            rows = [("video.weight", SimpleNamespace(grad=large[:16])),
                    ("action.weight", SimpleNamespace(grad=large[32:48])),
                    ("ifp.weight", SimpleNamespace(grad=small[4:12]))]
            accumulate_cpu_gradients(rows, original)
            accumulate_cpu_gradients(rows, reusable, staging=staging)
            if step == 0:
                self.assertFalse(staging)
            else:
                self.assertEqual(staging["buffer"].numel(), 64)
                address = staging["buffer"].untyped_storage().data_ptr()
                self.assertTrue(all(v.untyped_storage().data_ptr() != address for v in reusable.values()))
                staging["buffer"].fill_(float("nan"))
            large.fill_(float("nan"))
            small.fill_(float("nan"))
            for name in original:
                self.assertTrue(torch.equal(original[name], reusable[name]))

    def test_staging_keeps_new_or_missing_parameter_ownership_safe(self):
        original, reusable, staging = {}, {}, {}
        for step in range(8):
            source = torch.arange(32, dtype=torch.float32) + step
            rows = [("video.weight", SimpleNamespace(grad=source[:8]))]
            if step > 1 and step != 4:
                rows.append(("action.weight", SimpleNamespace(grad=source[16:24])))
            accumulate_cpu_gradients(rows, original)
            accumulate_cpu_gradients(rows, reusable, staging=staging)
            if "buffer" in staging:
                staging["buffer"].fill_(float("nan"))
            for name in original:
                self.assertTrue(torch.equal(original[name], reusable[name]))

    def test_optional_timing_is_additive_and_preserves_exact_values(self):
        observed, unobserved = {}, {}
        timing = {"unrelated_phase": 9.0}
        for index in range(8):
            source = torch.arange(32, dtype=torch.float32) + index
            rows = [("video.weight", SimpleNamespace(grad=source[:16])),
                    ("action.weight", SimpleNamespace(grad=source[16:]))]
            # Each storage uses four clock reads: copy start/end, add start/end.
            with patch("scripts.sugar.demo_following.paper_zero_wam.train_single_gpu.time.perf_counter",
                       side_effect=[1.0, 3.0, 4.0, 7.0]):
                self.assertEqual(accumulate_cpu_gradients(rows, observed, timing), 1)
            self.assertEqual(accumulate_cpu_gradients(rows, unobserved), 1)
            for name in observed:
                self.assertTrue(torch.equal(observed[name], unobserved[name]))
        self.assertEqual(timing, {"unrelated_phase": 9.0,
                                 "cpu_gradient_device_to_host_copy": 16.0,
                                 "cpu_gradient_layout_check_and_addition": 24.0})

    def test_eight_sums_are_bitwise_exact_with_one_add_per_storage(self):
        accumulated, expected, calls = {}, {}, []
        original_add = torch.Tensor.add_

        def counted_add(tensor, other):
            calls.append(tuple(tensor.shape))
            return original_add(tensor, other)

        generator = torch.Generator().manual_seed(101)
        for index in range(8):
            source = torch.randn(64, generator=generator)
            rows = [("video.weight", SimpleNamespace(grad=source[3:15].reshape(3, 4))),
                    ("action.weight", SimpleNamespace(grad=source[24:40].reshape(4, 4))),
                    ("ifp.weight", SimpleNamespace(grad=source[48:56]))]
            for name, parameter in rows:
                expected[name] = expected.get(name, torch.zeros_like(parameter.grad)) + parameter.grad
            with patch.object(torch.Tensor, "add_", counted_add):
                self.assertEqual(accumulate_cpu_gradients(rows, accumulated), 1)
            source.fill_(float("nan"))
            for name in expected:
                self.assertTrue(torch.equal(accumulated[name], expected[name]))
        self.assertEqual(calls, [(64,)] * 7)

    def test_missing_gradient_does_not_change_inactive_parameter(self):
        accumulated = {}
        first = torch.arange(16, dtype=torch.float32)
        accumulate_cpu_gradients([
            ("video.weight", SimpleNamespace(grad=first[:8])),
            ("action.weight", SimpleNamespace(grad=first[8:]))], accumulated)
        inactive_before = accumulated["action.weight"].clone()
        second = torch.full((16,), 3.0)
        accumulate_cpu_gradients([
            ("video.weight", SimpleNamespace(grad=second[:8])),
            ("action.weight", SimpleNamespace(grad=None))], accumulated)
        self.assertTrue(torch.equal(accumulated["video.weight"], first[:8] + 3))
        self.assertTrue(torch.equal(accumulated["action.weight"], inactive_before))

    def test_changed_offsets_use_parameterwise_sums(self):
        accumulated = {}
        first = torch.arange(32, dtype=torch.float32)
        second = first + 100
        accumulate_cpu_gradients([
            ("video.weight", SimpleNamespace(grad=first[:8])),
            ("action.weight", SimpleNamespace(grad=first[16:24]))], accumulated)
        accumulate_cpu_gradients([
            ("video.weight", SimpleNamespace(grad=second[8:16])),
            ("action.weight", SimpleNamespace(grad=second[24:32]))], accumulated)
        self.assertTrue(torch.equal(accumulated["video.weight"], first[:8] + second[8:16]))
        self.assertTrue(torch.equal(accumulated["action.weight"], first[16:24] + second[24:32]))

    def test_overlapping_views_retain_original_addition_order(self):
        accumulated = {}
        first = torch.arange(16, dtype=torch.float32)
        second = torch.ones(16)
        for source in (first, second):
            accumulate_cpu_gradients([
                ("video.weight", SimpleNamespace(grad=source[:8])),
                ("action.weight", SimpleNamespace(grad=source[4:12]))], accumulated)
        expected = first.clone()
        expected[:8].add_(1)
        expected[4:12].add_(1)
        self.assertTrue(torch.equal(accumulated["video.weight"], expected[:8]))
        self.assertTrue(torch.equal(accumulated["action.weight"], expected[4:12]))


if __name__ == "__main__":
    unittest.main()
