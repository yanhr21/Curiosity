"""CPU-only acceptance and fixed-denominator tests; never construct a model."""
import math
import unittest

from scripts.sugar.object_predictor.train_active3d_overfit import (
    EXPERIMENT, OBJECT_IDS, all_observed_cases_pass, case_gate, load_cases,
)


class NativeOverfitGateTests(unittest.TestCase):
    def setUp(self):
        self.initial = dict(global_cd_x9000=.45, global_fscore=.95, official_cd_x9000=1.)

    def test_thresholds_are_inclusive(self):
        self.assertTrue(all(case_gate(self.initial, self.initial).values()))

    def test_each_numerical_threshold_can_fail_independently(self):
        for key, value in (("global_cd_x9000", .450001), ("global_fscore", .949999),
                           ("official_cd_x9000", 1.000002)):
            with self.subTest(metric=key):
                self.assertFalse(all(case_gate(dict(self.initial, **{key: value}), self.initial).values()))

    def test_nonfinite_and_impossible_metrics_do_not_pass(self):
        for key in self.initial:
            for value in (math.nan, math.inf, -math.inf, -1.):
                with self.subTest(metric=key, value=value):
                    self.assertFalse(all(case_gate(dict(self.initial, **{key: value}), self.initial).values()))
        self.assertFalse(all(case_gate(dict(self.initial, global_fscore=1.01), self.initial).values()))

    def test_loss_nonregression_is_per_case(self):
        initial = dict(self.initial, official_cd_x9000=.2)
        self.assertFalse(all(case_gate(self.initial, initial).values()))

    @staticmethod
    def records():
        return [dict(name=f"{obj}_repeat{repeat}", condition="observed", passed=True)
                for obj in OBJECT_IDS for repeat in range(2)]

    def test_eight_cases_all_pass(self):
        self.assertTrue(all_observed_cases_pass(self.records()))

    def test_one_failed_case_cannot_be_averaged_away(self):
        rows = self.records()
        rows[4]["passed"] = False  # The predefined free-space-only case.
        self.assertFalse(all_observed_cases_pass(rows))

    def test_missing_duplicate_and_foreign_cases_are_rejected(self):
        rows = self.records()
        foreign = dict(rows[-1], name="replacement_object_repeat1")
        for altered in (rows[:-1], rows[:-1]+[rows[0]], rows[:-1]+[foreign], []):
            with self.subTest(names=[row["name"] for row in altered]):
                with self.assertRaises(RuntimeError):
                    all_observed_cases_pass(altered)

    def test_empty_control_neither_passes_nor_fails_observed_gate(self):
        rows = self.records()
        empty = [dict(row, condition="empty", passed=False) for row in rows]
        self.assertTrue(all_observed_cases_pass(rows+empty))
        rows[0]["passed"] = False
        self.assertFalse(all_observed_cases_pass(rows+[dict(row, passed=True) for row in empty]))

    def test_real_fixed_native_inputs_keep_free_space_only_case(self):
        rows, _ = load_cases(EXPERIMENT / "datasets/active3d_official")
        self.assertEqual([row["object_id"] for row in rows], [obj for obj in OBJECT_IDS for _ in range(2)])
        self.assertEqual([row["mask_vertex_counts"]["2"] for row in rows],
                         [125, 125, 100, 125, 0, 25, 25, 25])
        self.assertEqual(rows[4]["name"], "15737_repeat0")
        self.assertEqual(rows[4]["mask_vertex_counts"]["1"], 125)


if __name__ == "__main__":
    unittest.main()
