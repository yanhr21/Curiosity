from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load(
    "run_matched_state_predictor",
    ROOT / "scripts/sugar/demo_following/run_matched_state_predictor.py",
)
GATE = load(
    "assess_teacher_floor_overfit",
    ROOT / "scripts/sugar/demo_following/assess_teacher_floor_overfit.py",
)


def test_teacher_floor_protocol_is_matched_and_nonzero() -> None:
    design = RUNNER.DESIGNS["teacher_floor_overfit"]
    output = ROOT / "experiments/unit_test_teacher_floor"
    previous = ROOT / "experiments/source/update_0064/policy.pt"
    payload = RUNNER.protocol_payload(
        arm="correct",
        update=128,
        previous_update=64,
        previous_checkpoint=previous,
        output_root=output,
        seed=161581,
        action_seed=161582,
        num_envs=20,
        design=design,
    )
    shared = payload["shared_runtime"]
    assert payload["protocol"] == "sugar_plan11_teacher_floor_overfit_v1"
    assert shared["teacher_wrapper_mode"] == "wrong_reference_anneal_v1"
    assert shared["teacher_anneal_updates"] == 64
    assert shared["teacher_final_coefficient"] == 0.25
    assert shared["fixed_physics_profile"] == {
        "mass_scale": 1.0,
        "static_friction": 0.6,
        "dynamic_friction": 0.5,
        "com_y_m": 0.0,
        "pulse_delta_velocity_w_mps": [0.0, 0.0, 0.0],
    }

    args = argparse.Namespace(
        design="teacher_floor_overfit",
        num_envs=20,
        seed=161581,
        action_seed=161582,
        device="cuda:0",
    )
    paths = RUNNER.segment_paths(output, "correct", 128, 161581)
    command = RUNNER.runner_command(
        args=args,
        paths=paths,
        protocol=paths["protocol"],
        previous_checkpoint=previous,
        contract=design["arms"]["correct"],
    )
    assert command[command.index("--teacher-final-coefficient") + 1] == "0.25"
    assert command[command.index("--teacher-release-mode") + 1] == "linear"
    assert command[command.index("--resume-checkpoint") + 1] == str(previous)


def test_phase_event_protocol_holds_teacher_fixed_and_changes_selected_demo():
    design = RUNNER.DESIGNS["phase_event_reward_only"]
    assert design["output"].name == (
        "matched_phase_event_reward_reference_aware_v2"
    )
    correct = design["arms"]["correct"]
    unrelated = design["arms"]["unrelated"]
    assert correct["teacher"] == unrelated["teacher"]
    assert correct["event_runtime_config"] == unrelated["event_runtime_config"]
    assert correct["selected_option"] == "correct"
    assert unrelated["selected_option"] == "unrelated"
    assert design["checkpoint_updates"] == [32, 64]

    payload = RUNNER.protocol_payload(
        arm="correct",
        update=64,
        previous_update=0,
        previous_checkpoint=None,
        output_root=ROOT / "experiments/unit_test_phase_event",
        seed=161587,
        action_seed=161588,
        num_envs=20,
        design=design,
    )
    shared = payload["shared_runtime"]
    assert payload["protocol"] == "sugar_phase_event_reward_matched_policy_v1"
    assert shared["checkpoint_updates"] == [32, 64]
    assert shared["demo_event_phase_horizon_steps"] == 650
    for arm in payload["arms"].values():
        assert arm["demo_reward_kind"] == "phase_aware_dense_event"
        assert arm["teacher_motion_folder"] == RUNNER.workspace_relative(
            RUNNER.CORRECT_TEACHER
        )

    args = argparse.Namespace(
        design="phase_event_reward_only",
        arm="correct",
        num_envs=20,
        seed=161587,
        action_seed=161588,
        device="cuda:0",
    )
    paths = RUNNER.segment_paths(
        ROOT / "experiments/unit_test_phase_event", "correct", 64, 161587
    )
    command = RUNNER.runner_command(
        args=args,
        paths=paths,
        protocol=paths["protocol"],
        previous_checkpoint=None,
        contract=correct,
    )
    kit_args = command[command.index("--kit_args") + 1]
    assert command[command.index("--tactile-regime") + 1] == (
        "explicit_zero_control"
    )
    assert "--portable-root /tmp/Curiosity_demo_kit_" in kit_args
    assert "--/renderer/enabled=" in kit_args
    assert "--/app/vulkan=false" in kit_args
    assert "--fast-exit-after-evidence" in command
    assert shared["headless_renderer"] == (
        "disabled_for_policy_training"
    )
    assert shared["headless_graphics_icd"] == (
        "mesa_lavapipe_cpu_no_render"
    )
    assert shared["process_shutdown"] == "fast_exit_after_passing_evidence"
    assert "/renderer/multiGpu/enabled=false" in kit_args
    assert "/renderer/multiGpu/autoEnable=false" in kit_args
    assert "/renderer/multiGpu/maxGpuCount=1" in kit_args
    assert (
        RUNNER.runtime_environment(args, 64)["VK_ICD_FILENAMES"]
        == "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
    )

    inner_source = (
        ROOT / "scripts/sugar/smp/audit_stage_h_smp_icm_policy_integration.py"
    ).read_text(encoding="utf-8")
    config_source = (
        ROOT
        / "SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/"
        "train_refiner/carry_box_smp_icm_goal_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "NoTactileGoalRobotEnvCfg()" in inner_source
    assert '"no_tactile_startup_physics"' in inner_source
    assert '"no_tactile_startup_physics_recorded"' in inner_source
    assert '"selected_demo_changes_ppo_returns"' in inner_source
    assert '"selected_demo_changes_normalized_advantages"' in inner_source
    assert '"selected_demo_changes_actor_surrogate_gradient"' in inner_source
    assert '"fixed_one_teacher_keeps_full_residual_authority"' in inner_source
    assert '"executed_residual_action_reaches_environment"' in inner_source
    assert "runtime_step.teacher_action" in inner_source
    assert "runtime_step.residual_action" in inner_source
    assert "runtime_step.executed_action" in inner_source
    assert "runtime_step.action_manager_raw_action" in inner_source
    assert "applied_policy_unit_roundtrip_max_abs <= 2.0e-6" in inner_source
    assert "_actor_surrogate_gradient_comparison(" in inner_source
    assert "algorithm.storage.rewards.copy_(stored_total_rewards)" in inner_source
    assert "args.probe_result_output.write_text(" in inner_source
    assert "scene: CarryBoxRobotSceneCfg" in config_source

    runner_source = (
        ROOT / "scripts/sugar/demo_following/run_matched_state_predictor.py"
    ).read_text(encoding="utf-8")
    assert "--policy-training-" + "authorized" not in runner_source
    assert "require_passing_probe_result(" in runner_source
    assert '"--probe-result-output", str(temporary_result)' in runner_source
    assert (
        "sugar_phase_event_online_rollout_gradient_authority_smoke_v3"
        in runner_source
    )
    assert 'result.get("policy_updates_executed") != 0' in runner_source
    assert '"--probe-evidence-output"' in runner_source
    assert "json.dumps(probe_result, indent=2, sort_keys=True)" in runner_source

    evaluation_launcher = (
        ROOT / "scripts/sugar/demo_following/evaluate_and_render_matched_endpoint.sh"
    ).read_text(encoding="utf-8")
    assert "teacher_only_gate_no_tactile_v2" in evaluation_launcher
    assert 'p["checks"]["demo_control_has_no_tactile_scene"]' in evaluation_launcher
    assert 'records[0]["values"] == records[1]["values"]' in evaluation_launcher
    assert "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json" in (
        evaluation_launcher
    )
    assert "--fast-exit-after-evidence" in evaluation_launcher
    assert 'RENDER_KIT_ARGS="--/renderer/multiGpu/enabled=false' in (
        evaluation_launcher
    )

    evaluator_source = (
        ROOT / "scripts/sugar/demo_following/evaluate_matched_fixed_teacher.py"
    ).read_text(encoding="utf-8")
    assert '"--fast-exit-after-evidence"' in evaluator_source
    assert "def expand_fixed_one_wrapper_batch_state(" in evaluator_source
    assert '"release_latched": False' in evaluator_source
    assert '"release_progress": 0' in evaluator_source
    assert '"teacher_coefficient": 1.0' in evaluator_source
    assert "evaluation_num_envs=NUM_ENVS" in evaluator_source
    assert '"phase_event_fixed_one_wrapper_batch_restored"' in evaluator_source
    assert "training_shape = (PROFILES_PER_UPDATE, *tensor.shape[1:])" in (
        evaluator_source
    )
    assert '"same_profiles_repeated_exactly_across_updates": True' in (
        evaluator_source
    )
    assert 'if name == "object_coms"' in evaluator_source
    assert "float(torch.finfo(observed[name].dtype).eps)" in evaluator_source
    assert '"readback_max_abs": readback_max_abs' in evaluator_source
    assert "source_action_tolerance = 2.0e-6" in evaluator_source
    assert (
        '"closest_origin_first_teacher_action_matches_source"'
        in evaluator_source
    )
    assert "canonical_environment_index = int(np.argmin(translation_norm))" in (
        evaluator_source
    )
    assert 'frame_lists["goal_policy_core_observation"]' in evaluator_source
    assert 'transition_lists[f"demo_{name}_phase"]' in evaluator_source
    assert 'transition_lists[f"demo_{name}_ready"]' in evaluator_source
    assert 'transition_lists[f"demo_{name}_risk"]' in evaluator_source
    assert "initial_episode_steps=initial_phase_steps" in evaluator_source
    assert 'choices=("reference-aware", "reset-zero-diagnostic")' in (
        evaluator_source
    )
    assert '"initial_phase_matches_restored_reference_frame"' in inner_source
    assert '"reset_reference_frame_plus_causal_control_clock"' in inner_source
    assert inner_source.count(
        "command.last_reset_timestep.fill_(selected_reference_frame)"
    ) == 2
    assert '"phase_event_exact_policy_core_archived"' in evaluator_source
    assert '"phase_event_runtime_signals_archived"' in evaluator_source
    assert "sugar_phase_event_reward_matched_frozen_eval_32_64_v2" in (
        evaluator_source
    )

def test_phase_corrected_pair_reuses_only_complete_arms(tmp_path: Path) -> None:
    launcher = (
        ROOT
        / "scripts/sugar/demo_following/run_reference_aware_phase_event_pair.sh"
    )
    assert "DEMO_POLICY_TRAINING_" + "AUTHORIZED" not in launcher.read_text(
        encoding="utf-8"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_hostname = fake_bin / "hostname"
    fake_hostname.write_text("#!/usr/bin/env bash\necho server-test\n", encoding="utf-8")
    fake_hostname.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "SLURM_JOB_ID": "unit-test-job",
            "SLURM_STEP_ID": "unit-test-step",
            "PYTHON_BIN": sys.executable,
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    def write_endpoint(root: Path, arm: str, phase_source: str) -> None:
        endpoint = root / f"seed161587/{arm}/update_0064"
        endpoint.mkdir(parents=True)
        audit = {
            "phase_source": phase_source,
            "initial_episode_steps_supplied": True,
            "initial_episode_steps_min": 197,
            "initial_episode_steps_max": 197,
        }
        (endpoint / "proof.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "checks": {"demo_event_phase_and_prefix_are_causal": True},
                    "protocol": "sugar_phase_event_reward_matched_policy_v1",
                    "seed": 161587,
                    "action_seed": 161588,
                    "num_envs": 20,
                    "num_updates": 64,
                    "demo_event_reward": {
                        "selected_option": arm,
                        "final_frozen_audit": audit,
                    },
                    "no_tactile_startup_physics": {
                        "passed": True,
                        "values": {"mass": [0.3023375869]},
                    },
                }
            ),
            encoding="utf-8",
        )
        (endpoint / "protocol.json").write_text(
            json.dumps({"protocol": "matched-test"}), encoding="utf-8"
        )
        (endpoint / "policy.pt").write_bytes(b"test-checkpoint")

    def run(root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(launcher)],
            env={**env, "OUTPUT_ROOT": str(root)},
            text=True,
            capture_output=True,
            check=False,
        )

    complete_root = tmp_path / "complete"
    for arm in ("correct", "unrelated"):
        write_endpoint(
            complete_root,
            arm,
            "reset_reference_frame_plus_causal_control_clock",
        )
    completed = run(complete_root)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.index("reusing_complete_arm=correct") < (
        completed.stdout.index("reusing_complete_arm=unrelated")
    )
    assert "endpoint proofs passed" in completed.stdout

    old_phase_root = tmp_path / "old_phase"
    write_endpoint(old_phase_root, "correct", "reset_bounded_causal_control_clock")
    old_phase = run(old_phase_root)
    assert old_phase.returncode != 0
    assert "reusing_complete_arm" not in old_phase.stdout

    incomplete_root = tmp_path / "incomplete"
    (incomplete_root / "seed161587/correct/update_0064").mkdir(parents=True)
    refused = run(incomplete_root)
    assert refused.returncode == 2
    assert "incomplete arm requires inspection; refusing to overwrite: correct" in (
        refused.stderr
    )


def test_phase_event_proof_reuse_requires_reference_aware_clock(
    tmp_path: Path,
) -> None:
    proof = tmp_path / "proof.json"
    payload = {
        "passed": True,
        "protocol": "sugar_phase_event_reward_matched_policy_v1",
        "checks": {"demo_event_phase_and_prefix_are_causal": True},
        "contact_seed": {"selected_reference_frame": 197},
        "demo_event_reward": {
            "final_frozen_audit": {
                "phase_source": "reset_bounded_causal_control_clock",
                "initial_episode_steps_supplied": False,
                "initial_episode_steps_min": 0,
                "initial_episode_steps_max": 0,
            }
        },
    }
    proof.write_text(json.dumps(payload), encoding="utf-8")
    assert RUNNER.proof_passed(proof) is True
    assert (
        RUNNER.proof_passed(proof, require_reference_aware_phase=True) is False
    )
    audit = payload["demo_event_reward"]["final_frozen_audit"]
    audit.update(
        {
            "phase_source": "reset_reference_frame_plus_causal_control_clock",
            "initial_episode_steps_supplied": True,
            "initial_episode_steps_min": 197,
            "initial_episode_steps_max": 197,
        }
    )
    proof.write_text(json.dumps(payload), encoding="utf-8")
    assert RUNNER.proof_passed(proof, require_reference_aware_phase=True) is True


def test_runner_probe_requires_machine_readable_success(tmp_path: Path) -> None:
    result = tmp_path / "probe.json"
    result.write_text(
        json.dumps(
            {
                "protocol": (
                    "sugar_phase_event_online_rollout_gradient_authority_smoke_v3"
                ),
                "passed": True,
                "policy_updates_executed": 0,
            }
        ),
        encoding="utf-8",
    )
    payload = RUNNER.require_passing_probe_result(
        result,
        returncode=0,
        admission_only=False,
    )
    assert payload["passed"] is True

    result.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no valid result"):
        RUNNER.require_passing_probe_result(
            result,
            returncode=0,
            admission_only=False,
        )


def behavior_payload(*, passing: bool) -> dict[str, object]:
    directions = {
        name: {"direction_observed": passing}
        for name in (
            "correct_has_more_lifted_time",
            "correct_has_more_lifted_transport",
            "unrelated_has_more_ground_transport",
            "unrelated_has_more_orbiting",
        )
    }
    return {
        "protocol": "same_teacher_predictor_independent_behavior_audit_v1",
        "evidence_contract": {
            "uses_predictor_output": False,
            "uses_demo_reward": False,
        },
        "required_next_trace_fields": [],
        "actual_arm_summary": {
            "correct": {
                "bilateral_contact_fraction": {"mean": 0.75},
                "lifted_fraction": {"mean": 0.60},
                "lifted_transport_fraction": {"mean": 0.90},
                "any_foot_box_contact_fraction": {"mean": 0.0},
            },
            "unrelated": {
                "bilateral_contact_fraction": {"mean": 0.50},
                "any_foot_box_contact_fraction": {
                    "mean": 0.03 if passing else 0.0
                },
            },
        },
        "predeclared_semantic_directions": directions,
        "semantic_directions_observed": 4 if passing else 0,
        "semantic_directions_total": 4,
    }


def test_teacher_floor_gate_selects_next_branch_without_manual_approval() -> None:
    passed = GATE.assess(behavior_payload(passing=True))
    failed = GATE.assess(behavior_payload(passing=False))
    assert passed["passed"] is True
    assert passed["automatic_next_branch"].startswith("repeat_teacher_floor")
    assert failed["passed"] is False
    assert failed["automatic_next_branch"] == (
        "redesign_internal_reward_contact_event_semantics"
    )


def test_retained_phase_pair_chains_predeclared_evaluation_without_manual_gate() -> None:
    wrapper = (
        ROOT
        / "scripts/sugar/demo_following/run_reference_aware_phase_event_pair_then_hold.sh"
    ).read_text(encoding="utf-8")
    pair_call = wrapper.index("run_reference_aware_phase_event_pair.sh")
    eval_call = wrapper.index("evaluate_and_render_matched_endpoint.sh")
    hold_call = wrapper.index("GPU_HOLD_AFTER_REFERENCE_AWARE_MATCHED_PAIR_READY")
    assert pair_call < eval_call < hold_call
    assert 'if [[ "$pair_rc" == "0" ]]' in wrapper
    assert "REFERENCE_AWARE_MATCHED_EVALUATION_RC" in wrapper
    assert "DEMO_POLICY_TRAINING_" + "AUTHORIZED" not in wrapper


def test_matched_policy_training_disables_unused_vulkan_renderer() -> None:
    source = (
        ROOT / "scripts/sugar/demo_following/run_matched_state_predictor.py"
    ).read_text(encoding="utf-8")
    assert '"--/renderer/enabled= --/app/vulkan=false "' in source
    assert '"headless_renderer": "disabled_for_policy_training"' in source
