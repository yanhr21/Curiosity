from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SUGAR/source/sugar_rl"))

from sugar_rl.utils.demo_event_reward_potential import (  # noqa: E402
    DEFAULT_EVENT_WEIGHTS,
    calibrated_event_risk,
    compatibility_potential,
    event_internal_reward,
)
from sugar_rl.utils.demo_event_reward_runtime import (  # noqa: E402
    FrozenPhaseAwareDemoEventScorer,
    GOAL_POLICY_CORE_TERM_NAMES,
    extract_goal_policy_core,
)


def test_event_weights_are_semantic_and_sum_to_one():
    assert len(DEFAULT_EVENT_WEIGHTS) == 13
    assert abs(sum(DEFAULT_EVENT_WEIGHTS) - 1.0) < 1.0e-9
    assert sum(DEFAULT_EVENT_WEIGHTS[4:]) == 0.75


def test_goal_policy_core_extraction_checks_term_order():
    observation = torch.randn(2, 175)
    core = extract_goal_policy_core(
        observation, list(GOAL_POLICY_CORE_TERM_NAMES) + ["tactile", "strategy"]
    )
    assert core.shape == (2, 121)
    assert torch.equal(core, observation[:, :121])
    wrong = list(GOAL_POLICY_CORE_TERM_NAMES)
    wrong[0], wrong[1] = wrong[1], wrong[0]
    try:
        extract_goal_policy_core(observation, wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("term-order mismatch must be rejected")


def test_calibrated_risk_penalizes_mismatch_and_uncertainty():
    mean = torch.zeros(3, 13)
    log_variance = torch.full((3, 13), -6.0)
    multiplier = torch.ones(13)
    mean[1, 4:8] = 0.5
    log_variance[2] = -1.0
    record = calibrated_event_risk(
        mean,
        log_variance,
        multiplier,
        uncertainty_beta=1.0,
    )
    assert record["risk"][1] > record["risk"][0]
    assert record["risk"][2] > record["risk"][0]
    potential = compatibility_potential(record["risk"])
    assert potential[1] < potential[0]
    assert potential[2] < potential[0]
    assert torch.all((potential > 0) & (potential <= 1))


def test_dense_internal_reward_has_no_warmup_or_benign_terminal_bonus():
    next_value = torch.tensor([0.8, 0.2, 0.8, 0.8, 0.8])
    next_ready = torch.tensor([True, True, False, True, False])
    done = torch.tensor([False, False, False, True, True])
    failure = torch.tensor([False, False, False, False, True])
    reward = event_internal_reward(
        next_value,
        next_ready,
        done,
        failure,
        compatibility_baseline=0.5,
        eta=1.0,
        reward_clip=1.0,
    )
    assert reward[0] > 0
    assert reward[1] < 0
    assert reward[2] == 0
    assert reward[3] == 0
    assert reward[4] == -1.0


class _RecordingRuntime:
    def __init__(self):
        self.phases = []
        self.started = False

    def begin(self, core):
        assert core.shape == (2, 121)
        self.started = True

    def process_step(self, core, phase, done, failure_done):
        assert self.started
        assert core.shape == (2, 121)
        assert not failure_done.any()
        self.phases.append(phase.clone())
        return {"phase": phase, "done": done}

    def audit(self):
        return {"model_training": False, "trainable_parameters": 0}


def test_phase_scorer_uses_only_reset_bounded_causal_clock():
    scorer = FrozenPhaseAwareDemoEventScorer.__new__(
        FrozenPhaseAwareDemoEventScorer
    )
    scorer.num_envs = 2
    scorer.device = torch.device("cpu")
    scorer.cfg = type(
        "Cfg", (), {"phase_horizon_steps": 650, "selected_option": "correct"}
    )()
    scorer.runtime = _RecordingRuntime()
    scorer.episode_steps = torch.zeros(2, dtype=torch.long)
    scorer.initial_episode_steps = torch.zeros(2, dtype=torch.long)
    scorer.initial_episode_steps_supplied = False
    scorer.transitions_scored = 0
    scorer.started = False
    observation = {"policy": torch.randn(2, 175)}
    scorer.begin(observation)
    scorer.process_step(observation, torch.tensor([False, True]))
    assert torch.allclose(
        scorer.runtime.phases[-1], torch.tensor([2.0 / 650.0, 1.0 / 650.0])
    )
    scorer.process_step(observation, torch.tensor([False, False]))
    assert torch.allclose(
        scorer.runtime.phases[-1], torch.tensor([3.0 / 650.0, 2.0 / 650.0])
    )
    assert scorer.transitions_scored == 4


def test_phase_scorer_can_start_from_restored_reference_frame():
    scorer = FrozenPhaseAwareDemoEventScorer.__new__(
        FrozenPhaseAwareDemoEventScorer
    )
    scorer.num_envs = 2
    scorer.device = torch.device("cpu")
    scorer.cfg = type(
        "Cfg", (), {"phase_horizon_steps": 650, "selected_option": "correct"}
    )()
    scorer.runtime = _RecordingRuntime()
    scorer.episode_steps = torch.zeros(2, dtype=torch.long)
    scorer.initial_episode_steps = torch.zeros(2, dtype=torch.long)
    scorer.initial_episode_steps_supplied = False
    scorer.transitions_scored = 0
    scorer.started = False
    observation = {"policy": torch.randn(2, 175)}
    scorer.begin(
        observation,
        initial_episode_steps=torch.tensor([197, 0]),
    )
    scorer.process_step(observation, torch.tensor([False, False]))
    audit = scorer.frozen_model_audit()
    assert audit["initial_episode_steps_supplied"] is True
    assert audit["initial_episode_steps_min"] == 0
    assert audit["initial_episode_steps_max"] == 197
    assert torch.allclose(
        scorer.runtime.phases[-1],
        torch.tensor([199.0 / 650.0, 2.0 / 650.0]),
    )
