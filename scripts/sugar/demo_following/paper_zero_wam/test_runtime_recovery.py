"""Scheduler failure classification tests; no model, GPU, or optimizer."""
import unittest
from unittest.mock import mock_open, patch
from pathlib import Path
from subprocess import CompletedProcess

from scripts.sugar.demo_following.paper_zero_wam.runtime_recovery import (
    classify_job, inspect_job,
)


class RuntimeRecoveryTest(unittest.TestCase):
    def record(self, state, job_id="279357"):
        return {"job_id": job_id, "state": state, "exit_code": "1:0"}

    def test_only_explicit_interruptions_allow_recovery(self):
        for state in ("TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED"):
            with self.subTest(state=state):
                self.assertTrue(classify_job("279357", [self.record(state)], False)["recovery_allowed"])

    def test_live_or_unclassified_termination_never_retries(self):
        for state in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED",
                      "FAILED", "CANCELLED by 2059", "OUT_OF_MEMORY", "COMPLETED"):
            with self.subTest(state=state):
                self.assertFalse(classify_job("279357", [self.record(state)], False)["recovery_allowed"])

    def test_numerical_failure_overrides_later_timeout(self):
        self.assertFalse(classify_job("279357", [self.record("TIMEOUT")], True)["recovery_allowed"])

    def test_missing_duplicate_and_unrelated_records_fail_closed(self):
        for rows in ([], [self.record("TIMEOUT")] * 2, [self.record("TIMEOUT", "999")]):
            self.assertFalse(classify_job("279357", rows, False)["recovery_allowed"])

    def test_array_element_cannot_inherit_another_elements_timeout(self):
        rows = [self.record("TIMEOUT", "10_0"), self.record("FAILED", "10_1")]
        self.assertTrue(classify_job("10_0", rows, False)["recovery_allowed"])
        self.assertFalse(classify_job("10_1", rows, False)["recovery_allowed"])
        self.assertFalse(classify_job("10_2", rows, False)["recovery_allowed"])

    def test_complete_log_scan_detects_numerical_failure_before_timeout(self):
        output = CompletedProcess([], 0, "279357|TIMEOUT|0:15\n", "")
        log = "[rank2]: FloatingPointError: non-finite loss at step 7\n" + "waiting\n" * 10000
        with patch("subprocess.run", return_value=output), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "open", mock_open(read_data=log)):
            decision = inspect_job("279357", Path("/logs"), "overfit")
        self.assertFalse(decision["recovery_allowed"])
        self.assertEqual(decision["failure_log_evidence"][0]["line"], 1)

    def test_unknown_python_failure_is_not_infrastructure(self):
        output = CompletedProcess([], 0, "279357|FAILED|1:0\n", "")
        with patch("subprocess.run", return_value=output), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "open", mock_open(read_data="ValueError: bad data\n")):
            self.assertFalse(inspect_job("279357", Path("/logs"), "overfit")["recovery_allowed"])


if __name__ == "__main__":
    unittest.main()
