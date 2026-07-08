#!/usr/bin/env python3
"""Shell backend adapter for the direct Isaac carry task runner.

This adapter executes the existing Isaac shell launcher from a task-runner
reset/action pair. It is a bridge from audited diagnostics to an executable
task interface; it is not a new controller and does not change success claims.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from direct_carry_task_runner import (
    DirectCarryAction,
    DirectCarryBackendCapabilities,
    DirectCarryObservation,
    DirectCarryReset,
)


def _refuse_login_node() -> None:
    if os.uname().nodename.startswith("mgmtserver"):
        raise RuntimeError("Refusing to run Isaac backend on a login/management node.")


def _env_float(value: float) -> str:
    return f"{float(value):.8g}"


def _env_override(name: str, default: str) -> str:
    value = os.environ.get(name)
    return str(default) if value is None or value == "" else str(value)


class DirectCarryShellBackend:
    """Run one direct carry episode through the existing Isaac shell launcher."""

    def __init__(
        self,
        *,
        root_dir: Path,
        stamp: str,
        support_mode: str = "alternating_anchor_feet",
        steps: int = 3580,
        randomize_payload: bool = True,
        log_dir: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.stamp = stamp
        self.support_mode = support_mode
        self.steps = int(steps)
        self.randomize_payload = bool(randomize_payload)
        self.extra_env = dict(extra_env or {})
        self.output_dir = root_dir / "experiments" / "outputs" / "direct_carry_task_runner" / stamp
        self.summary_path = self.output_dir / "direct_carry_task_physical_backend_summary.json"
        self.log_dir = log_dir or (root_dir / "logs" / "direct_carry_task_runner")
        self.log_path = self.log_dir / f"{stamp}.log"
        self._reset: DirectCarryReset | None = None
        self._action: DirectCarryAction | None = None

    def capabilities(self) -> DirectCarryBackendCapabilities:
        is_directional_placement = self.support_mode == "alternating_placement_feet"
        backend_id = (
            "physical_alternating_placement_feet_cradle_v1"
            if is_directional_placement
            else "physical_alternating_anchor_feet_cradle_v1"
        )
        return DirectCarryBackendCapabilities(
            backend_id=backend_id,
            backend_family="directional_foot_placement_scaffold" if is_directional_placement else "anchored_support_scaffold",
            isaac_backend=True,
            free_dynamic_box=True,
            randomized_box_properties=self.randomize_payload,
            active_probe_supported=True,
            trainable_policy_backend=False,
            real_robot_morphology=False,
            support_switching_supported=is_directional_placement,
            video_conditioning_supported=False,
            root_shortcut_audited=True,
            root_shortcut_free_claimed=True,
            hidden_context_isolated=True,
            scaffold_backend=True,
            claim_limit=(
                "Direct Isaac scaffold backend with audited shortcuts; not a "
                "full walking robot, not RL, and not video-conditioned success."
            ),
            notes=(
                "Uses the current anchored/support-foot cradle backend through a shell launcher.",
                f"support_mode={self.support_mode}",
                "Hidden mass, size, and COM are exported only as evaluation context.",
                "The backend is swappable through the task runner but remains a scaffold.",
            ),
        )

    def reset(self, reset: DirectCarryReset) -> None:
        self._reset = reset

    def observe(self) -> DirectCarryObservation:
        if self._reset is None:
            raise RuntimeError("reset() must be called before observe().")
        return DirectCarryObservation(
            target_distance_x_m=self._reset.target_distance_x_m,
            carry_posture_command=self._action.carry_posture if self._action else None,
            probe_features={
                "probe_steps": self._action.probe_steps if self._action else 0,
                "probe_amplitude_x_m": self._action.probe_amplitude_x_m if self._action else 0.0,
                "probe_amplitude_z_m": self._action.probe_amplitude_z_m if self._action else 0.0,
            },
            contact_state={"support_foot_contact_report_requested": True},
            support_metrics={"morphology_config": self._reset.morphology_config},
            video_reference={"reference_video_id": self._reset.reference_video_id},
        )

    def apply_action(self, action: DirectCarryAction) -> None:
        self._action = action

    def _build_env(self) -> dict[str, str]:
        if self._reset is None:
            raise RuntimeError("reset() must be called before step_until_done().")
        if self._action is None:
            raise RuntimeError("apply_action() must be called before step_until_done().")
        action = self._action
        reset = self._reset
        step_length = max(0.004, 0.016 * float(action.gait_speed_scale))
        controller_mode = (
            "physical_alternating_placement_feet_cradle"
            if self.support_mode == "alternating_placement_feet"
            else "physical_alternating_anchor_feet_cradle"
        )
        physical_support_mode = (
            "alternating_anchor_feet" if self.support_mode == "alternating_placement_feet" else self.support_mode
        )
        env = os.environ.copy()
        env.update(
            {
                "ROOT_DIR": str(self.root_dir),
                "STAMP": self.stamp,
                "OUTPUT_DIR": str(self.output_dir),
                "SUPPORT_MODE": physical_support_mode,
                "CARRY_POSTURE": action.carry_posture,
                "CONTROLLER_MODE": controller_mode,
                "STEPS": str(self.steps),
                "TARGET_X": _env_float(reset.target_distance_x_m),
                "STEP_LENGTH": _env_override("STEP_LENGTH", _env_float(step_length)),
                "STANCE_STEPS": _env_override("STANCE_STEPS", "80"),
                "SETTLE_STEPS": _env_override("SETTLE_STEPS", "10"),
                "STOP_THRESHOLD": _env_override("STOP_THRESHOLD", "0.002"),
                "SUPPORT_FOOT_STANCE_X": _env_override("SUPPORT_FOOT_STANCE_X", "-0.130"),
                "SUPPORT_FOOT_SWING_X": _env_override("SUPPORT_FOOT_SWING_X", "0.130"),
                "SUPPORT_FOOT_STEP_HEIGHT": _env_override("SUPPORT_FOOT_STEP_HEIGHT", "0.100"),
                "SUPPORT_FOOT_CONTACT_Z_THRESHOLD": _env_override("SUPPORT_FOOT_CONTACT_Z_THRESHOLD", "0.055"),
                "ENABLE_SUPPORT_FOOT_CONTACT_REPORT": "1",
                "SUPPORT_FOOT_CONTACT_REPORT_THRESHOLD": _env_override("SUPPORT_FOOT_CONTACT_REPORT_THRESHOLD", "0.0"),
                "SUPPORT_FOOT_EFFORT_CONTACT_THRESHOLD": _env_override("SUPPORT_FOOT_EFFORT_CONTACT_THRESHOLD", "0.001"),
                "SUPPORT_FOOT_DOUBLE_SUPPORT_FRACTION": _env_float(action.support_foot_double_support_fraction),
                "SUPPORT_FOOT_CONTINUITY_GRACE_STEPS": _env_override("SUPPORT_FOOT_CONTINUITY_GRACE_STEPS", "12"),
                "SUPPORT_FOOT_PLACEMENT_MODE": (
                    "alternating_directional_x" if self.support_mode == "alternating_placement_feet" else "alternating_fixed_x"
                ),
                "STANCE_FOOT_WORLD_LOCK": _env_override("STANCE_FOOT_WORLD_LOCK", "0"),
                "FREEZE_LOCKED_STANCE_FOOT_TARGETS": _env_override("FREEZE_LOCKED_STANCE_FOOT_TARGETS", "0"),
                "FREEZE_COMMANDED_STANCE_FOOT_TARGETS": _env_override(
                    "FREEZE_COMMANDED_STANCE_FOOT_TARGETS", "0"
                ),
                "PLANTED_STANCE_RAIL_PROPULSION": _env_override("PLANTED_STANCE_RAIL_PROPULSION", "0"),
                "SUPPORT_FOOT_MASS": _env_override("SUPPORT_FOOT_MASS", "8.0"),
                "SUPPORT_FOOT_X_LOWER": _env_override("SUPPORT_FOOT_X_LOWER", "-0.18"),
                "SUPPORT_FOOT_X_UPPER": _env_override("SUPPORT_FOOT_X_UPPER", "0.18"),
                "SUPPORT_FOOT_Z_LOWER": _env_override("SUPPORT_FOOT_Z_LOWER", "-0.005"),
                "SUPPORT_FOOT_Z_UPPER": _env_override("SUPPORT_FOOT_Z_UPPER", "0.24"),
                "SUPPORT_FOOT_DRIVE_STIFFNESS": _env_override("SUPPORT_FOOT_DRIVE_STIFFNESS", "24000.0"),
                "SUPPORT_FOOT_DRIVE_DAMPING": _env_override("SUPPORT_FOOT_DRIVE_DAMPING", "3400.0"),
                "SUPPORT_FOOT_DRIVE_MAX_FORCE": _env_override("SUPPORT_FOOT_DRIVE_MAX_FORCE", "110000.0"),
                "SUPPORT_FOOT_Z_DRIVE_STIFFNESS": _env_override("SUPPORT_FOOT_Z_DRIVE_STIFFNESS", "36000.0"),
                "SUPPORT_FOOT_Z_DRIVE_DAMPING": _env_override("SUPPORT_FOOT_Z_DRIVE_DAMPING", "3200.0"),
                "SUPPORT_FOOT_Z_DRIVE_MAX_FORCE": _env_override("SUPPORT_FOOT_Z_DRIVE_MAX_FORCE", "130000.0"),
                "FEEDBACK_STEP_CONTROLLER": "1",
                "FEEDBACK_STEP_X_GAIN": _env_float(action.feedback_step_x_gain),
                "FEEDBACK_STEP_X_LIMIT": _env_float(action.feedback_step_x_limit_m),
                "FEEDBACK_STEP_TILT_GAIN": _env_float(action.feedback_step_tilt_gain),
                "FEEDBACK_STEP_TILT_LIMIT": _env_float(action.feedback_step_tilt_limit_m),
                "RANDOMIZE_PAYLOAD": "1" if self.randomize_payload else "0",
                "BOX_SEED": str(reset.box_seed),
                "PAYLOAD_MASS_MIN": "4.0",
                "PAYLOAD_MASS_MAX": "12.0",
                "PAYLOAD_SIZE_JITTER": "0.10",
                "PAYLOAD_COM_OFFSET_RANGE_X": "0.04",
                "PAYLOAD_COM_OFFSET_RANGE_Y": "0.04",
                "PAYLOAD_COM_OFFSET_RANGE_Z": "0.03",
                "RAIL_JOINT_COUNT": "2",
                "RAIL_LOWER": "-0.04",
                "RAIL_UPPER": "0.10",
                "DRIVE_STIFFNESS": "22000.0",
                "DRIVE_DAMPING": "3500.0",
                "DRIVE_MAX_FORCE": "80000.0",
                "STATIC_FRICTION": _env_override("STATIC_FRICTION", "4.5"),
                "DYNAMIC_FRICTION": _env_override("DYNAMIC_FRICTION", "4.0"),
                "PROBE_STEPS": str(int(action.probe_steps)),
                "PROBE_MODE": "horizontal_push_pull",
                "PROBE_X_AMPLITUDE": _env_float(action.probe_amplitude_x_m),
                "PROBE_Z_AMPLITUDE": _env_float(action.probe_amplitude_z_m),
                "DEVICE": "cpu",
            }
        )
        env.update(self.extra_env)
        return env

    def step_until_done(self) -> dict[str, Any]:
        _refuse_login_node()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "bash",
            "scripts/isaac/run_direct_carry_task_physical_backend.sh",
        ]
        with self.log_path.open("w") as log:
            result = subprocess.run(
                cmd,
                cwd=self.root_dir,
                env=self._build_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if not self.summary_path.exists():
            return {
                "status": "error",
                "error": f"backend exited {result.returncode} without summary",
                "backend_returncode": result.returncode,
                "backend_log": str(self.log_path),
                "success_claim": "task_runner_backend_error_not_robot_success",
            }
        summary = json.loads(self.summary_path.read_text())
        summary["task_runner_backend_returncode"] = int(result.returncode)
        summary["task_runner_backend_log"] = str(self.log_path)
        if result.returncode != 0:
            summary.setdefault("status", "error")
            summary["task_runner_backend_error"] = f"backend exited {result.returncode}"
        return summary
