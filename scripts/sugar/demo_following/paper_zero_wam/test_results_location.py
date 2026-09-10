"""CPU-only regression tests for archive-safe result document paths."""

import json
from pathlib import Path
import tempfile
import unittest

from .results import refresh_results_document


class ResultsLocationTest(unittest.TestCase):
    def test_default_page_stays_inside_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            root.mkdir()
            output = refresh_results_document(root)
            self.assertEqual(output, root / "results.md")
            page = output.read_text()
            self.assertIn(
                "[LATENT_RESULT.json](latents/LATENT_RESULT.json)", page
            )
            self.assertNotIn("../../experiments/", page)
            self.assertFalse((Path(temporary) / "PLAN").exists())

    def test_custom_output_preserves_relative_artifact_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            (root / "formal").mkdir(parents=True)
            (root / "formal/FORMAL_PROGRESS.json").write_text(json.dumps({}))
            output = Path(temporary) / "docs/results.md"
            self.assertEqual(refresh_results_document(root, output), output)
            page = output.read_text()
            self.assertIn(
                "[LATENT_RESULT.json](../experiment/latents/LATENT_RESULT.json)",
                page,
            )
            self.assertIn(
                "[FORMAL_PROGRESS.json](../experiment/formal/FORMAL_PROGRESS.json)",
                page,
            )


if __name__ == "__main__":
    unittest.main()
