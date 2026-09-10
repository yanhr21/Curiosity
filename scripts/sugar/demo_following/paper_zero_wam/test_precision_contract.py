"""Source-level precision regression; no Torch import, model, GPU, or forward."""

import ast
from pathlib import Path
import unittest


def overfit_forward_contexts(tree):
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "overfit_prompt_gate"
    )
    contexts = []

    def visit(node, bf16=False):
        if isinstance(node, ast.With):
            for item in node.items:
                expression = item.context_expr
                if (
                    isinstance(expression, ast.Call)
                    and ast.unparse(expression.func) == "torch.autocast"
                    and expression.args
                    and ast.literal_eval(expression.args[0]) == "cuda"
                ):
                    keywords = {
                        keyword.arg: ast.unparse(keyword.value)
                        for keyword in expression.keywords
                    }
                    bf16 = (
                        keywords.get("dtype") == "torch.bfloat16"
                        and keywords.get("enabled", "True") == "True"
                    )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "model":
                contexts.append((ast.unparse(node.args[0]), bf16))
        for child in ast.iter_child_nodes(node):
            visit(child, bf16)

    visit(function)
    return contexts


class PrecisionContractTest(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(
            Path(__file__).with_name("train.py").read_text(encoding="utf-8")
        )

    def test_both_overfit_forwards_use_cuda_bf16_autocast(self):
        self.assertEqual(
            overfit_forward_contexts(self.tree),
            [("batch", True), ("probe_batch", True)],
        )

    def test_detects_original_unguarded_probe_regression(self):
        class RemoveProbeAutocast(ast.NodeTransformer):
            def visit_With(self, node):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Assign):
                    value = node.body[0].value
                    if isinstance(value, ast.Call) and ast.unparse(value) == "model(probe_batch)":
                        return node.body
                return self.generic_visit(node)

        changed = RemoveProbeAutocast().visit(self.tree)
        self.assertEqual(
            overfit_forward_contexts(changed),
            [("batch", True), ("probe_batch", False)],
        )

    def test_training_gates_do_not_create_inference_parameter_views(self):
        for filename, function_name in (("train.py", "overfit_prompt_gate"),
                                        ("train_single_gpu.py", "prompt_gate")):
            tree = ast.parse(Path(__file__).with_name(filename).read_text())
            function = next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == function_name)
            calls = [ast.unparse(node.func) for node in ast.walk(function)
                     if isinstance(node, ast.Call)]
            self.assertNotIn("torch.inference_mode", calls)
            self.assertIn("torch.no_grad", calls)


if __name__ == "__main__":
    unittest.main()
