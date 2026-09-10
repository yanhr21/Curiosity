"""CPU configuration/source checks; numerical model tests use official H200 audit."""
import ast
import json
import tempfile
from dataclasses import replace
from pathlib import Path
import unittest

from .config import PaperZeroWAMConfig, repaired_overfit_config
from .prompt_coverage import prompt_frame_indices, prompt_coverage_record
from .overfit_diagnostic import validate_diagnostic_request, REPAIRED_DIRECTORY, DIAGNOSTIC_DIRECTORY
from .overfit_diagnostic import restored_conditioning_gradient_decision
from .overfit_diagnostic import validate_preupdate_recovery


class RepairedConditioningTests(unittest.TestCase):
    def test_real_full_width_action_output_head_blocks_initial_upstream_gradient(self):
        # Derivative test of the actual 3072-wide project output module only;
        # no optimizer or replacement-model training experiment.
        import torch
        from .model import ActionFlowHead
        head = ActionFlowHead(3072, 29)
        tokens = torch.randn(1, 2, 3072, requires_grad=True)
        output = head(tokens, torch.randn(1, 2, 3072))
        (output - 1).square().mean().backward()
        self.assertEqual(float(tokens.grad.abs().max()), 0.0)
        self.assertGreater(float(head.projection.weight.grad.norm()), 0.0)

    def test_preupdate_recovery_rejects_applied_update_or_completed_boundary(self):
        config = repaired_overfit_config()
        with tempfile.TemporaryDirectory(prefix="pzw_recovery_test_") as directory:
            root = Path(directory)
            (root / "CONFIG.json").write_text(json.dumps({"model_and_schedule_config": config.as_dict()}))
            (root / "TRAIN_TRACE.jsonl").write_text("")
            for name in ("INITIAL_PROMPT_GATE.json", "UNSEEN_NOISE_INITIAL.json"):
                (root / name).write_text("{}")
            validate_preupdate_recovery(root, config)
            (root / "TRAIN_TRACE.jsonl").write_text('{"optimizer_applied": true}\n')
            with self.assertRaises(ValueError):
                validate_preupdate_recovery(root, config)
            (root / "TRAIN_TRACE.jsonl").write_text("")
            (root / "TRAIN_EVAL_FORWARD_EQUIVALENCE.json").write_text("{}")
            with self.assertRaises(ValueError):
                validate_preupdate_recovery(root, config)

    def test_zero_initialized_action_head_only_exempts_final_action_cross_at_update_zero(self):
        values = {f"mot_layers.{i}.{branch}_block": 1.0 for i in range(30) for branch in ("video", "action")}
        values.update({f"ifp_heads.{i}.block": 1.0 for i in range(4)})
        values["mot_layers.29.action_block"] = 0.0
        self.assertTrue(restored_conditioning_gradient_decision(values, 1.0, 0)["passed"])
        self.assertFalse(restored_conditioning_gradient_decision(values, 1.0, 1)["passed"])
        values["mot_layers.29.action_block"] = 1.0
        self.assertTrue(restored_conditioning_gradient_decision(values, 1.0, 1)["passed"])
        values["mot_layers.28.action_block"] = 0.0
        self.assertFalse(restored_conditioning_gradient_decision(values, 1.0, 1)["passed"])
        values["mot_layers.29.action_block"] = 0.0
        self.assertFalse(restored_conditioning_gradient_decision(values, 1.0, 0)["passed"])

    def test_full_span_unique_vae_compatible_selection(self):
        indices = prompt_frame_indices()
        self.assertEqual(len(indices), 61)
        self.assertEqual(len(set(indices)), 61)
        self.assertEqual((indices[0], indices[-1]), (0, 63))
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(1 + 4 * ((len(indices) - 1) // 4), len(indices))
        record = prompt_coverage_record()
        self.assertEqual(record["reversed_source_indices"], indices[::-1])
        self.assertEqual(set(record["reversed_source_indices"]), set(indices))
        self.assertEqual(len(record["omitted_source_indices"]), 3)

    def test_model_not_reduced_and_only_declared_configuration_changes(self):
        old, new = PaperZeroWAMConfig(), repaired_overfit_config()
        old.validate(); new.validate()
        changed = {key for key in old.as_dict() if old.as_dict()[key] != new.as_dict()[key]}
        self.assertEqual(changed, {"expected_parameter_count", "repaired_conditioning", "latent_cache", "neutral_text_cache"})
        self.assertEqual(new.expected_parameter_count - old.expected_parameter_count, 22_026_240)
        self.assertEqual((new.num_layers, new.hidden_dim, new.action_hidden_dim, new.ifp_heads), (30, 3072, 3072, 4))
        with self.assertRaises(ValueError):
            replace(new, expected_parameter_count=old.expected_parameter_count).validate()

    def test_corrected_output_is_isolated_from_both_old_runs(self):
        config = repaired_overfit_config()
        root = config.resolved(config.output_root)
        validate_diagnostic_request(True, "overfit", root / REPAIRED_DIRECTORY, config)
        for directory in (DIAGNOSTIC_DIRECTORY, "overfit", "formal"):
            with self.assertRaises(ValueError):
                validate_diagnostic_request(True, "overfit", root / directory, config)
        with self.assertRaises(ValueError):
            validate_diagnostic_request(True, "formal", root / REPAIRED_DIRECTORY, config)

    def test_cross_attention_precedes_feedforward_and_ifp_reuses_path(self):
        tree = ast.parse(Path(__file__).with_name("model.py").read_text())
        mot = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PaperMoTLayer")
        post = next(n for n in mot.body if isinstance(n, ast.FunctionDef) and n.name == "_post")
        source = ast.unparse(post)
        self.assertLess(source.index("block.cross_attn"), source.index("block.ffn"))
        self.assertIn("block.norm3(tokens)", source)
        ifp = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IFPHead")
        self.assertIn("PaperMoTLayer._post", ast.unparse(ifp))
        self.assertIn("text_context", ast.unparse(ifp))


if __name__ == "__main__":
    unittest.main()
