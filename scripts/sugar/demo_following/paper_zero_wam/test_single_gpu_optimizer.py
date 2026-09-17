"""CPU tensor-only optimizer regression; no model, forward, or GPU experiment."""
import unittest
import copy
import io
from dataclasses import replace
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from .config import PaperZeroWAMConfig, repaired_overfit_config
from .train import decay_parameter_groups
from .train_single_gpu import (
    EXECUTION, accumulate_cpu_gradients, deadline_checkpoint_needed,
    native_adamw_streamed_step, next_checkpoint_directory, single_gpu_checkpoint_step,
    requested_training_end, restore_overfit_adamw, admit_overfit_continuation,
    load_overfit_checkpoint,
    ACTION_INTERFACE_PARAMETERS, FULL_PARAMETER_COUNT,
    action_interface_lr_map, action_interface_intervention,
    cpu_gradient_staging_plan,
)


class NativeOptimizerTest(unittest.TestCase):
    def test_cpu_staging_uses_actual_correct_timings_not_success_flag(self):
        self.assertFalse(cpu_gradient_staging_plan(None, "NVIDIA H200")["enabled"])
        # Benchmark-record fixture only; not GPU performance evidence.
        rows = [dict(bytes_per_copy=size, reuse=reuse, microbatches=8,
                     elapsed_seconds=8.0 if reuse else 10.0,
                     all_accumulated_values_exact=True, retained_storage_independent=True)
                for size in (268435456, 1073741824)
                for reuse in (False, True, True, False)]
        result = dict(execution_completed=True, optimizer_updates=0,
                      device="NVIDIA H200", results=rows, candidate_supports_future_trial=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.log"
            status = path.with_suffix(".status")
            status.write_text("exit_code=0\n")
            def observe():
                path.write_text("runtime message\n" + json.dumps(result) + "\n")
                return cpu_gradient_staging_plan(path, "NVIDIA H200")
            self.assertTrue(observe()["enabled"])
            result["candidate_supports_future_trial"] = True
            for row in rows:
                if row["reuse"]:
                    row["elapsed_seconds"] = 9.0
            self.assertFalse(observe()["enabled"])
            rows[0]["retained_storage_independent"] = False
            with self.assertRaises(ValueError):
                observe()
            rows[0]["retained_storage_independent"] = True
            rows[0]["elapsed_seconds"] = float("nan")
            with self.assertRaises(ValueError):
                observe()
            rows[0]["elapsed_seconds"] = 10.0
            result["device"] = "NVIDIA A100"
            with self.assertRaises(ValueError):
                observe()
            result["device"] = "NVIDIA H200"
            status.write_text("exit_code=1\n")
            with self.assertRaises(ValueError):
                observe()

    def test_interface_selection_is_exact_and_wrapper_independent(self):
        names = ["_fsdp_wrapped_module." + n for n in sorted(ACTION_INTERFACE_PARAMETERS)]
        names += ["_fsdp_wrapped_module.mot_layers.0._fsdp_wrapped_module.action_block.q.weight",
                  "_fsdp_wrapped_module.action_head.modulation"]
        selected = action_interface_lr_map(((n, None) for n in names), 100.0)
        self.assertEqual(set(selected), set(names[:4]))
        self.assertEqual(set(selected.values()), {100.0})
        with self.assertRaises(ValueError):
            action_interface_lr_map(((n, None) for n in names[1:]), 100.0)

    def test_interface_intervention_requires_evidence_and_inherits_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CONFIG.json").write_text("{}")
            self.assertEqual(action_interface_intervention(root, None, 32)[
                "action_interface_lr_multiplier"], 1.0)
            probe = dict(execution_completed=True, checkpoint_step=32, optimizer_updates=0,
                architecture_parameter_count=FULL_PARAMETER_COUNT,
                cases=[dict(slot=i, flow_observations=[dict(action_flow_time=1.0,
                    paired_noise_response=dict(velocity_difference_gain_along_noise_difference=0.02))])
                    for i in range(8)])
            path = root / "ACTION_FLOW_PROBE.json"
            path.write_text(json.dumps(probe))
            result = action_interface_intervention(root, 100.0, 32)
            self.assertEqual(result["changed_at_optimizer_step"], 32)
            self.assertFalse(result["optimizer_moments_reset"])
            probe["cases"][0]["flow_observations"][0]["paired_noise_response"][
                "velocity_difference_gain_along_noise_difference"] = 0.9
            path.write_text(json.dumps(probe))
            with self.assertRaises(ValueError):
                action_interface_intervention(root, 100.0, 32)
            (root / "CONFIG.json").write_text(json.dumps({"optimizer_intervention": result}))
            self.assertEqual(action_interface_intervention(root, None, 64)[
                "action_interface_lr_multiplier"], 100.0)
            with self.assertRaises(ValueError):
                action_interface_intervention(root, 1.0, 64)
            with self.assertRaises(ValueError):
                action_interface_intervention(None, 100.0, 0)

    def test_named_lr_variant_matches_native_groups_with_retained_moments(self):
        # Tensor arithmetic fixture, not a learned model or substitute expert.
        names = ("video.weight", "action_encoder.weight")
        reference = [torch.nn.Parameter(torch.arange(9, dtype=torch.float32) / 9 + i)
                     for i in range(2)]
        serial = [torch.nn.Parameter(p.detach().clone()) for p in reference]
        options = dict(lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, foreach=False)
        ref = torch.optim.AdamW([{"params": [p]} for p in reference], **options)
        opt = torch.optim.AdamW(serial, **options)
        groups = opt.param_groups
        for step in range(5):
            scale = 1.0 if step < 2 else 100.0
            ref.param_groups[1]["lr"] = options["lr"] * scale
            gradients = {}
            for index, (name, p) in enumerate(zip(names, reference)):
                grad = torch.arange(9, dtype=torch.float32) * (step + 1) / 40 + index + 0.1
                p.grad = grad * 0.75
                gradients[name] = grad.clone()
            ref.step()
            _, finite, calls = native_adamw_streamed_step(
                opt, zip(names, serial), gradients, 0.75, {names[1]: scale})
            self.assertTrue(finite)
            self.assertEqual(calls, 2)
            self.assertIs(opt.param_groups, groups)
            self.assertEqual(opt.param_groups[0]["lr"], options["lr"])
            for expected, actual in zip(reference, serial):
                self.assertTrue(torch.equal(expected, actual))
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(ref.state[expected][key], opt.state[actual][key]))

    def test_continuation_requires_complete_isolated_parent_and_diagnostics(self):
        from .overfit_diagnostic import RESAMPLED_DIRECTORY
        # Metadata fixture: placeholder paths are never loaded as checkpoints.
        with tempfile.TemporaryDirectory() as directory, patch(
                "scripts.sugar.demo_following.paper_zero_wam.train_single_gpu.overfit_execution_decision",
                return_value={"passed": True}):
            config = replace(repaired_overfit_config(), output_root=directory)
            root = Path(directory) / RESAMPLED_DIRECTORY
            (root / "training_videos").mkdir(parents=True)
            output = Path(directory) / f"{RESAMPLED_DIRECTORY}_step64"
            records = {
                "OVERFIT_RESULT.json": dict(execution_completed=True, optimizer_steps=32,
                    architecture_parameter_count=config.expected_parameter_count,
                    execution_decision={"passed": True}, passed=False),
                "CONFIG.json": dict(model_and_schedule_config=config.as_dict()),
                "DIAGNOSTIC_RESULT.json": dict(requested_workflow_completed=True,
                    contract=dict(noise_and_flow_time_resampled_every_step_and_slot=True)),
                "training_videos/RENDER_RESULT.json": dict(execution_completed=True,
                    cases=[dict(slot=i, target_split="train", render_frame_count=8) for i in range(8)]),
                "ACTION_RECONSTRUCTION.json": dict(execution_completed=True, cases=[{} for _ in range(8)],
                                                   inverse_dynamics_probe_needed=False),
                "FROZEN_VAE_ALL_FRAMES.json": dict(execution_completed=True, control_frames=64),
            }
            for name, value in records.items():
                (root / name).write_text(json.dumps(value))
            for name in ("diagnostic_model_step32.pt", "diagnostic_optimizer_step32.pt",
                         "INITIAL_PROMPT_GATE.json", "UNSEEN_NOISE_INITIAL.json", "UNSEEN_NOISE_FINAL.json"):
                (root / name).touch()
            self.assertEqual(admit_overfit_continuation(root, output, config)[0], 32)
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                admit_overfit_continuation(root, root, config)
            with self.assertRaises(ValueError):
                admit_overfit_continuation(root, output, replace(config, peak_learning_rate=config.peak_learning_rate * 2))
            actions = records["ACTION_RECONSTRUCTION.json"]
            actions["inverse_dynamics_probe_needed"] = True
            (root / "ACTION_RECONSTRUCTION.json").write_text(json.dumps(actions))
            with self.assertRaises(FileNotFoundError):
                admit_overfit_continuation(root, output, config)
            (root / "INVERSE_DYNAMICS_PROBE.json").write_text(json.dumps(dict(
                execution_completed=True, checkpoint_step=32,
                cases=[dict(saved_action_replay_exact=True) for _ in range(8)])))
            self.assertEqual(admit_overfit_continuation(root, output, config)[0], 32)
            (root / "diagnostic_optimizer_step32.pt").unlink()
            with self.assertRaises(ValueError):
                admit_overfit_continuation(root, output, config)

    @staticmethod
    def retained_optimizer_fixture():
        # Parameter arithmetic only: no network, forward pass, data or GPU.
        config = repaired_overfit_config()
        names = ("video.weight", "action_weight", "ifp.modulation")
        pairs = [(name, torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7))
                 for name in names]
        options = dict(lr=config.peak_learning_rate, betas=(config.adam_beta1, config.adam_beta2),
                       eps=config.adam_epsilon, weight_decay=config.weight_decay, foreach=False)
        optimizer = torch.optim.AdamW(decay_parameter_groups(pairs, config), **options)
        for step in range(4):
            for i, (_, p) in enumerate(pairs):
                # One parameter clock legitimately differs from the global step.
                p.grad = None if step == 2 and i == 1 else torch.full_like(p, (i + 1) * (step + 1) / 10)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        payload = dict(step=4, optimizer=optimizer.state_dict(), parameter_names=list(names),
                       model_and_schedule_config=config.as_dict(), execution=EXECUTION)
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=False)
        restored_pairs = [(name, torch.nn.Parameter(p.detach().clone())) for name, p in pairs]
        restored = torch.optim.AdamW(decay_parameter_groups(restored_pairs, config), **options)
        return config, pairs, optimizer, restored_pairs, restored, saved

    def test_retained_adamw_next_update_equals_uninterrupted_update(self):
        config, pairs, reference, restored_pairs, restored, payload = self.retained_optimizer_fixture()
        evidence = restore_overfit_adamw(restored, restored_pairs, payload, config, 4)
        self.assertTrue(evidence["optimizer_moments_and_clocks_exact"])
        self.assertEqual(evidence["optimizer_updates_added"], 0)
        for i, ((_, p), (_, q)) in enumerate(zip(pairs, restored_pairs)):
            self.assertTrue(torch.equal(p, q))
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(reference.state[p][key], restored.state[q][key]))
            p.grad = torch.full_like(p, (i + 1) * 0.27)
            q.grad = p.grad.clone()
        reference.step()
        restored.step()
        for (_, p), (_, q) in zip(pairs, restored_pairs):
            self.assertTrue(torch.equal(p, q))
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(reference.state[p][key], restored.state[q][key]))

    def test_retained_weight_and_optimizer_files_restore_without_updates(self):
        config, pairs, reference, masters, restored, payload = self.retained_optimizer_fixture()
        # ParameterList is only a serialization container; no model or forward.
        source = torch.nn.ParameterList([p for _, p in pairs])
        destination = torch.nn.ParameterList([
            torch.nn.Parameter(torch.full_like(p, 42.0)) for _, p in pairs])
        master_pairs = [(name, destination[i], master) for i, (name, master) in enumerate(masters)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(dict(step=4, model=source.state_dict()), root / "diagnostic_model_step4.pt")
            torch.save(payload, root / "diagnostic_optimizer_step4.pt")
            evidence = load_overfit_checkpoint(destination, restored, root, config, master_pairs, 4)
            self.assertEqual(evidence["optimizer_updates_added"], 0)
            for (_, original), (_, value, master) in zip(pairs, master_pairs):
                self.assertTrue(torch.equal(original, value))
                self.assertTrue(torch.equal(original, master))
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(reference.state[original][key], restored.state[master][key]))
            torch.save(dict(step=4, model={k: v.bfloat16() for k, v in source.state_dict().items()}),
                       root / "diagnostic_model_step4.pt")
            with self.assertRaises(ValueError):
                load_overfit_checkpoint(destination, restored, root, config, master_pairs, 4)

    def test_retained_adamw_rejects_incompatible_or_incomplete_state_before_loading(self):
        config, _, _, pairs, optimizer, original = self.retained_optimizer_fixture()
        for defect in ("step", "names", "configuration", "missing_moments", "clock", "shape", "nan", "negative"):
            with self.subTest(defect=defect):
                payload = copy.deepcopy(original)
                state = payload["optimizer"]["state"]
                first = next(iter(state))
                if defect == "step":
                    payload["step"] = 3
                elif defect == "names":
                    payload["parameter_names"].reverse()
                elif defect == "configuration":
                    payload["model_and_schedule_config"]["peak_learning_rate"] *= 2
                elif defect == "missing_moments":
                    del state[first]
                elif defect == "clock":
                    state[first]["step"].fill_(5)
                elif defect == "shape":
                    state[first]["exp_avg"] = torch.zeros(1)
                elif defect == "nan":
                    state[first]["exp_avg"].fill_(float("nan"))
                else:
                    state[first]["exp_avg_sq"].fill_(-1)
                with self.assertRaises(ValueError):
                    restore_overfit_adamw(optimizer, pairs, payload, config, 4)
                self.assertFalse(optimizer.state)

    def test_user_stop_does_not_change_original_schedule_budget(self):
        self.assertEqual(requested_training_end("formal", 700), 700)
        self.assertEqual(requested_training_end("formal", None), 4200)
        self.assertEqual(requested_training_end("overfit", None), 32)
        self.assertEqual(PaperZeroWAMConfig().optimizer_steps, 4200)
        for mode, step in (("overfit", 700), ("formal", 699), ("formal", 701),
                           ("formal", True), ("formal", 700.0)):
            with self.assertRaises(ValueError):
                requested_training_end(mode, step)

    def test_partial_checkpoint_is_recovery_not_formal_completion(self):
        config = PaperZeroWAMConfig()
        state = dict(protocol="paper_zero_wam_single_gpu_checkpoint_v2", execution=EXECUTION,
                     model_and_schedule_config=json.loads(json.dumps(config.as_dict())),
                     architecture_parameter_count=config.expected_parameter_count,
                     hash_checks=False, optimizer_boundary_complete=True)
        for step in (1, 35, 36, 280, 4200):
            state.update(completed_optimizer_steps=step, completed_epochs=step // 280,
                         steps_in_current_epoch=step % 280)
            self.assertEqual(single_gpu_checkpoint_step(state, config), step)
        state["optimizer_boundary_complete"] = False
        with self.assertRaises(ValueError):
            single_gpu_checkpoint_step(state, config)
        self.assertTrue(deadline_checkpoint_needed(100, 1000, 60))
        self.assertFalse(deadline_checkpoint_needed(100, 2000, 60))
        self.assertTrue(deadline_checkpoint_needed(100, 2000, 1000))

    def test_checkpoint_write_never_targets_current_latest_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = next_checkpoint_directory(root)
            first.mkdir()
            (root / "latest_checkpoint").symlink_to(first.name)
            self.assertEqual(next_checkpoint_directory(root).name, "checkpoint_slot1")
            (root / "latest_checkpoint").unlink()
            (root / "latest_checkpoint").symlink_to("checkpoint_slot1")
            self.assertEqual(next_checkpoint_directory(root).name, "checkpoint_slot0")

    def test_shared_storage_copy_preserves_offset_stride_and_eight_sums(self):
        accumulated = {}
        expected = {}
        for microbatch in range(8):
            source = torch.arange(64, dtype=torch.float32) + microbatch
            rows = [("a", SimpleNamespace(grad=source[3:15].reshape(3, 4))),
                    ("b", SimpleNamespace(grad=source[20:40:2])),
                    ("inactive", SimpleNamespace(grad=None))]
            for name, value in rows:
                if value.grad is not None:
                    expected[name] = expected.get(name, torch.zeros_like(value.grad)) + value.grad
            self.assertEqual(accumulate_cpu_gradients(rows, accumulated), 1)
            source.fill_(-999)
        for name in expected:
            self.assertTrue(torch.equal(accumulated[name], expected[name]))

    def test_streamed_native_steps_equal_original_adamw_bitwise(self):
        names = ("video.weight", "action_weight", "ifp_head.weight")
        original = [torch.nn.Parameter(torch.arange(7, dtype=torch.float32) / 7 + i)
                    for i in range(3)]
        serial = [torch.nn.Parameter(value.detach().clone()) for value in original]
        options = dict(lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
                       weight_decay=0.01, foreach=False)
        reference = torch.optim.AdamW(original, **options)
        streamed = torch.optim.AdamW(serial, **options)
        initial_groups = streamed.param_groups
        for step in range(4):
            gradients = {}
            for i, (name, parameter) in enumerate(zip(names, original)):
                if step == 2 and i == 1:
                    parameter.grad = None
                    continue
                gradient = torch.full_like(parameter, (step + 1) * (i + 1) / 10)
                gradients[name] = gradient.clone()
                parameter.grad = gradient * 0.75
            before = [parameter.detach().clone() for parameter in serial]
            reference.step()
            updates, finite, calls = native_adamw_streamed_step(
                streamed, zip(names, serial), gradients, 0.75)
            self.assertTrue(finite)
            self.assertEqual(calls, 2 if step == 2 else 3)
            self.assertFalse(gradients)
            self.assertIs(streamed.param_groups, initial_groups)
            for i, (expected, actual) in enumerate(zip(original, serial)):
                self.assertTrue(torch.equal(expected, actual))
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(reference.state[expected][key],
                                                streamed.state[actual][key]))
                branch = ("video", "action", "ifp")[i]
                self.assertAlmostEqual(updates[branch], float((actual.detach() - before[i])
                                                            .double().norm()), places=12)
            reference.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
