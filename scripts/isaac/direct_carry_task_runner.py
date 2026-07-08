#!/usr/bin/env python3
"""Skeleton task runner for direct Isaac carrying.

This file defines the control interface for the next implementation step. It
does not run Isaac by itself; a backend adapter must provide simulation state
and apply actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from direct_carry_task_contract import episode_row, reward_proxy


@dataclass(frozen=True)
class DirectCarryReset:
    box_seed: int
    morphology_config: str = "scaffold_support_feet_v1"
    target_distance_x_m: float = 0.64
    reference_video_id: str | None = None


@dataclass(frozen=True)
class DirectCarryBackendCapabilities:
    """Auditable backend flags for preventing scaffold/robot claim drift."""

    backend_id: str
    backend_family: str
    isaac_backend: bool
    free_dynamic_box: bool
    randomized_box_properties: bool
    active_probe_supported: bool
    trainable_policy_backend: bool
    real_robot_morphology: bool
    support_switching_supported: bool
    video_conditioning_supported: bool
    root_shortcut_audited: bool
    root_shortcut_free_claimed: bool
    hidden_context_isolated: bool
    scaffold_backend: bool
    claim_limit: str
    shortcut_audit_fields: tuple[str, ...] = (
        "root_shortcut_free",
        "body_root_pose_write_count",
        "body_root_velocity_command_count",
        "box_pose_write_count",
        "box_kinematic_pose_write_count",
        "support_root_pose_write_count",
        "anchor_world_joint_retarget_count",
        "foot_pose_write_count",
        "stance_anchor_pose_write_count",
    )
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DirectCarryAction:
    carry_posture: str
    feedback_step_x_gain: float = 0.08
    feedback_step_x_limit_m: float = 0.008
    feedback_step_tilt_gain: float = 0.04
    feedback_step_tilt_limit_m: float = 0.005
    support_foot_double_support_fraction: float = 0.12
    gait_speed_scale: float = 1.0
    hold_height_adjustment_m: float = 0.0
    stance_width_adjustment_m: float = 0.0
    probe_steps: int = 0
    probe_amplitude_x_m: float = 0.0
    probe_amplitude_z_m: float = 0.0


@dataclass
class DirectCarryObservation:
    target_distance_x_m: float
    carry_posture_command: str | None = None
    probe_features: dict[str, Any] = field(default_factory=dict)
    contact_state: dict[str, Any] = field(default_factory=dict)
    support_metrics: dict[str, Any] = field(default_factory=dict)
    video_reference: dict[str, Any] = field(default_factory=dict)


class DirectCarryBackend(Protocol):
    """Backend adapter expected from an Isaac scene or future RL environment."""

    def capabilities(self) -> DirectCarryBackendCapabilities:
        ...

    def reset(self, reset: DirectCarryReset) -> None:
        ...

    def observe(self) -> DirectCarryObservation:
        ...

    def apply_action(self, action: DirectCarryAction) -> None:
        ...

    def step_until_done(self) -> dict[str, Any]:
        ...


class DirectCarryTaskRunner:
    """Small task wrapper around a backend-provided direct carry episode."""

    def __init__(self, backend: DirectCarryBackend) -> None:
        self.backend = backend
        self._reset: DirectCarryReset | None = None
        self._action: DirectCarryAction | None = None

    def reset(self, reset: DirectCarryReset) -> DirectCarryObservation:
        self._reset = reset
        self._action = None
        self.backend.reset(reset)
        return self.observe()

    def observe(self) -> DirectCarryObservation:
        return self.backend.observe()

    def capabilities(self) -> DirectCarryBackendCapabilities:
        return self.backend.capabilities()

    def apply_action(self, action: DirectCarryAction) -> None:
        self._action = action
        self.backend.apply_action(action)

    def run_episode(self, action: DirectCarryAction) -> dict[str, Any]:
        self.apply_action(action)
        summary = self.backend.step_until_done()
        summary.setdefault("backend_capabilities", asdict(self.capabilities()))
        summary.setdefault("carry_posture", action.carry_posture)
        summary.setdefault("feedback_step_x_gain", action.feedback_step_x_gain)
        summary.setdefault("feedback_step_x_limit_m", action.feedback_step_x_limit_m)
        summary.setdefault("feedback_step_tilt_gain", action.feedback_step_tilt_gain)
        summary.setdefault("feedback_step_tilt_limit_m", action.feedback_step_tilt_limit_m)
        summary.setdefault("support_foot_double_support_fraction", action.support_foot_double_support_fraction)
        summary.setdefault("probe_steps_requested", action.probe_steps)
        if self._reset is not None:
            summary.setdefault("box_seed", self._reset.box_seed)
        return summary

    @staticmethod
    def compute_reward(summary: dict[str, Any]) -> float:
        return reward_proxy(summary)

    @staticmethod
    def is_terminated(summary: dict[str, Any]) -> bool:
        return (
            int(summary.get("fall_events") or 0) > 0
            or int(summary.get("box_drop_events") or 0) > 0
            or int(summary.get("nonfinite_state_events") or 0) > 0
            or str(summary.get("status", "")).lower() in {"pass", "fail", "error"}
        )

    def export_episode_row(self, *, source_summary: str, episode_id: str, summary: dict[str, Any]) -> dict[str, Any]:
        parent = {}
        if self._reset is not None:
            parent.update(
                {
                    "box_seed": self._reset.box_seed,
                    "target_distance_x_m": self._reset.target_distance_x_m,
                    "video_reference_id": self._reset.reference_video_id,
                }
            )
        parent["backend_capabilities"] = asdict(self.capabilities())
        if self._action is not None:
            enriched = dict(summary)
            enriched.update({k: v for k, v in asdict(self._action).items() if k not in enriched})
        else:
            enriched = summary
        return episode_row(
            source_summary=source_summary,
            episode_id=episode_id,
            summary=enriched,
            parent_summary=parent,
        )
