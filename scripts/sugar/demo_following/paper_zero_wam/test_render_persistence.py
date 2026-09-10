"""CPU-only render evidence tests; no model inference or generated outcome."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .render_openloop import verify_decodable_render_cases, render_summary, CONDITIONS
from .config import PaperZeroWAMConfig


class RenderPersistenceTest(unittest.TestCase):
    def test_interim_label_preserves_the_same_render_checks(self):
        config = PaperZeroWAMConfig()
        cases = []
        for task_index, task in enumerate(("CarryBox", "KickBox")):
            for condition in CONDITIONS:
                cases.append(dict(
                    task=task, condition=condition, inference_seed=100 + task_index,
                    render_job_id="123", predicted_video_shape=[1, 48, 2, 20, 20],
                    predicted_action_shape=[40, 29], render_frame_count=8,
                    render_frame_names=[f"{i:03d}.png" for i in range(8)],
                    target_split="test", source_motion_id=9, anchor_latent_transition=8,
                    prompt_split="test", prompt_task=("KickBox" if task == "CarryBox" else "CarryBox")
                    if condition == "wrong_task" else task,
                    prompt_source_motion_id=19 if condition == "same_task_alternate" else 9,
                    prompt_reversed=condition == "reversed", video=f"{task}_{condition}.mp4",
                    latent_video_mse=1.0 if condition == "matched" else 2.0,
                    normalized_action_mse=1.0 if condition == "matched" else 2.0))
        with patch("pathlib.Path.is_file", return_value=True), patch(
                "pathlib.Path.stat", return_value=SimpleNamespace(st_size=123)):
            outputs = [render_summary(cases, config,
                        full_parameter_count=config.expected_parameter_count, checkpoint_step=step,
                        execution_world_size=1, checkpoint_binding={"test_only": True})
                       for step in (700, 4200)]
        interim, formal = outputs
        self.assertTrue(interim["execution_completed"])
        self.assertNotEqual(interim["protocol"], formal["protocol"])
        self.assertEqual(interim["execution_checks"], formal["execution_checks"])
        self.assertEqual(interim["directional_checks"], formal["directional_checks"])
        self.assertFalse(interim["formal_execution_complete"])
        self.assertFalse(interim["physical_continuation_allowed"])
        self.assertFalse(interim["automatic_training_continuation"])

    def test_exact_eight_frames_are_required(self):
        frame_bytes = 960 * 352 * 3
        for count in (7, 8, 9):
            with patch("scripts.sugar.demo_following.paper_zero_wam.render_openloop.subprocess.run",
                       return_value=SimpleNamespace(stdout=bytes(count * frame_bytes))) as run:
                if count == 8:
                    verify_decodable_render_cases([{"video": "test-only.mp4"}], "ffmpeg")
                else:
                    with self.assertRaises(ValueError):
                        verify_decodable_render_cases([{"video": "test-only.mp4"}], "ffmpeg")
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("-frames:v") + 1], "9")
                self.assertTrue(run.call_args.kwargs["check"])

    def test_decode_failure_propagates_without_reinference(self):
        import subprocess
        with patch("scripts.sugar.demo_following.paper_zero_wam.render_openloop.subprocess.run",
                   side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            with self.assertRaises(subprocess.CalledProcessError):
                verify_decodable_render_cases([{"video": "test-only.mp4"}], "ffmpeg")


if __name__ == "__main__":
    unittest.main()
