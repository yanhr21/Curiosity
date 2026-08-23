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
