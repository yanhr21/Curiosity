from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


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
