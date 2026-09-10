"""CPU score-record tests only; no model, optimizer, or extra evaluation run."""
import copy
import unittest

from .audit_heldout_decision_contract import clean_records
from .config import PaperZeroWAMConfig
from .evaluate_heldout import aggregate, validate_score_prefix, evaluation_scope


class EvaluationPersistenceTest(unittest.TestCase):
    def test_interim_scope_is_exact_and_distinct_from_formal(self):
        self.assertEqual(evaluation_scope(True)[0], 4200)
        self.assertEqual(evaluation_scope(False)[0], 4200)
        step, protocol = evaluation_scope(True, 700)
        self.assertEqual(step, 700)
        self.assertNotEqual(protocol, evaluation_scope(True)[1])
        for single_gpu, step in ((False, 700), (True, 699), (True, 701),
                                 (True, 4200), (True, True), (True, 700.0)):
            with self.assertRaises(ValueError):
                evaluation_scope(single_gpu, step)

    def setUp(self):
        self.records = clean_records()
        self.groups = [dict(group_index=row["group_index"], anchor=row["latent_anchor"],
                            row={key: row[key] for key in ("split", "task", "source_motion_id")})
                       for row in self.records]
        self.config = PaperZeroWAMConfig()

    def test_exact_complete_group_prefix_can_resume_without_changing_cases(self):
        for length in (0, 1, 37, 389, 390):
            self.assertEqual(validate_score_prefix(self.records[:length], self.groups, self.config), length)
        self.assertTrue(aggregate(self.records)["passed"])

    def test_wrong_source_anchor_noise_and_partial_condition_are_rejected(self):
        for field in ("source_motion_id", "latent_anchor", "matched_noise_seed", "group_index"):
            rows = copy.deepcopy(self.records[:3])
            rows[1][field] += 1
            with self.assertRaises(ValueError):
                validate_score_prefix(rows, self.groups, self.config)
        rows = copy.deepcopy(self.records[:3])
        del rows[1]["losses"]["reversed"]
        with self.assertRaises(ValueError):
            validate_score_prefix(rows, self.groups, self.config)

    def test_nonfinite_outcomes_are_preserved_but_cannot_pass(self):
        for value in (float("inf"), float("nan")):
            rows = copy.deepcopy(self.records)
            # An infinite positive margin must not masquerade as prompt evidence.
            rows[0]["losses"]["reversed"]["video_loss"] = value
            self.assertEqual(validate_score_prefix(rows, self.groups, self.config), 390)
            decision = aggregate(rows)
            self.assertFalse(decision["passed"])
            self.assertFalse(decision["checks"]["all_scores_and_connectivity_finite"])


if __name__ == "__main__":
    unittest.main()
