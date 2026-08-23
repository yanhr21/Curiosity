# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Audited tactile-slip and failed-strategy memory for goal-based CarryBox.

The slip path consumes only the official dual-R15 pressure/shear history.  The
strategy descriptor uses declared wrist/box geometry in the box frame and
direct tactile contact.  A failed contact attempt is stored per episode and a
later geometrically similar attempt receives a separate external anti-repeat
cost.  The memory does not assume that all official SUGAR motions share one
"original" pose: the accepted dataset is demonstrably multimodal.

These outputs are external policy observations/objectives.  They are never an
ICM input or target and never define the ICM intrinsic discovery signal.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from sugar_rl.tasks.locomanip.direct_tactile_history import (
    direct_tactile_force_history,
)
from sugar_rl.tasks.locomanip.direct_tactile_slip_spatiotemporal import (
    SpatiotemporalDirectTactileSlipCalibration,
    SpatiotemporalDirectTactileSlipEstimator,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_V16_CALIBRATION = (
    WORKSPACE_ROOT
    / "experiments/sugar_reproduction/outputs/final/smp_tactile"
    / "stage_e_v16_20260723/calibration/v16_development_admission.json"
)
DEFAULT_V16_RESULT = (
    WORKSPACE_ROOT
    / "experiments/sugar_reproduction/outputs/final/smp_tactile"
    / "stage_e_v16_20260723/result/result.json"
)
DEFAULT_ORIGINAL_CLAMP_PROTOTYPE = (
    WORKSPACE_ROOT
    / "experiments/sugar_reproduction/outputs/final/smp_strategy"
    / "original_clamp_prototype_v1/prototype.json"
)
EXPECTED_V16_CALIBRATION_SHA256 = (
    "394619bdb6158be028497ac98bfea5d11a08bc96b2af8be236e40f9f3199d2a6"
)
EXPECTED_V16_RESULT_SHA256 = (
    "b1d6d2ceed26d8bd1175fdd0cfbd2547f3d944390ff7839f5006c9c252116f66"
)
EXPECTED_V16_MODEL_SOURCE_SHA256 = (
    "c4f02f5e65eb341d7b7f95aa5701b039d3dc2b8d5c5db550cbd8f60d4438ab58"
)
EXPECTED_ORIGINAL_CLAMP_PROTOTYPE_SHA256 = (
    "5ac3f85ad93de00e64d771e6df57447dfc00f60112fad6c2e02b2364f2175169"
)

ORIGINAL_CLAMP_DESCRIPTOR_NAMES = (
    "left_position_box_x",
    "left_position_box_y",
    "left_position_box_z",
    "left_tangent_box_x",
    "left_tangent_box_y",
    "left_tangent_box_z",
    "left_normal_box_x",
    "left_normal_box_y",
    "left_normal_box_z",
    "right_position_box_x",
    "right_position_box_y",
    "right_position_box_z",
    "right_tangent_box_x",
    "right_tangent_box_y",
    "right_tangent_box_z",
    "right_normal_box_x",
    "right_normal_box_y",
    "right_normal_box_z",
    "wrist_position_delta_box_x",
    "wrist_position_delta_box_y",
    "wrist_position_delta_box_z",
    "wrist_position_midpoint_box_x",
    "wrist_position_midpoint_box_y",
    "wrist_position_midpoint_box_z",
    "wrist_tangent_dot",
    "wrist_normal_dot",
)
REPEATED_FAILED_SIMILARITY_THRESHOLD = 0.80


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_v16_calibration(
    path: Path,
) -> tuple[SpatiotemporalDirectTactileSlipCalibration, str]:
    resolved = path.expanduser().resolve()
    digest = _sha256(resolved)
    if digest != EXPECTED_V16_CALIBRATION_SHA256:
        raise RuntimeError(
            f"v16 calibration hash drift: expected "
            f"{EXPECTED_V16_CALIBRATION_SHA256}, got {digest}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        payload.get("protocol")
        != "sugar_direct_tacsl_spatiotemporal_v16_observable_development_admission_exact_scope"
        or payload.get("passed") is not True
        or payload.get("stage_e_development_admitted") is not True
    ):
        raise RuntimeError("v16 exact-scope development calibration is not admitted")
    values = dict(payload.get("parameters", {}))
    expected = {field.name for field in fields(SpatiotemporalDirectTactileSlipCalibration)}
    if set(values) != expected:
        raise RuntimeError("v16 calibration parameter schema drift")
    for name in ("feature_mean", "feature_std", "slip_weight", "gross_weight"):
        values[name] = tuple(float(value) for value in values[name])
    return SpatiotemporalDirectTactileSlipCalibration(**values), digest


class OriginalClampPrototype:
    """Frozen official-contact reference used only for robust normalization.

    The historical artifact also contains a global center/threshold.  A
    per-motion audit showed that five accepted official motion IDs have zero
    acceptance under that gate, so active code must not use its score to decide
    whether a strategy is "original".  The center and scale remain an audited,
    fixed coordinate normalization for pairwise failed-attempt distances.
    """

    def __init__(
        self,
        path: str | Path,
        device: torch.device | str,
        expected_sha256: str,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.device = torch.device(device)
        if not expected_sha256:
            raise RuntimeError("original-clamp prototype hash has not been locked")
        self.sha256 = _sha256(self.path)
        if self.sha256 != expected_sha256:
            raise RuntimeError(
                f"original-clamp prototype hash drift: expected "
                f"{expected_sha256}, got {self.sha256}"
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol") != "sugar_official_original_clamp_prototype_v1"
            or payload.get("passed") is not True
        ):
            raise RuntimeError("official-SUGAR original-clamp prototype did not pass")
        if tuple(payload.get("descriptor_names", ())) != ORIGINAL_CLAMP_DESCRIPTOR_NAMES:
            raise RuntimeError("original-clamp descriptor schema drift")
        self.center = torch.tensor(
            payload["descriptor_center"], dtype=torch.float32, device=self.device
        )
        self.scale = torch.tensor(
            payload["descriptor_scale"], dtype=torch.float32, device=self.device
        )
        self.minimum_wrist_separation_m = float(
            payload["minimum_wrist_separation_m"]
        )
        self.threshold = float(payload["original_similarity_threshold"])
        expected_shape = (len(ORIGINAL_CLAMP_DESCRIPTOR_NAMES),)
        if self.center.shape != expected_shape or self.scale.shape != expected_shape:
            raise RuntimeError(
                "original-clamp prototype descriptor dimension drift"
            )
        if (
            not torch.isfinite(self.center).all()
            or not torch.isfinite(self.scale).all()
            or not torch.all(self.scale > 0.0)
            or self.minimum_wrist_separation_m <= 0.0
            or not 0.0 < self.threshold < 1.0
        ):
            raise RuntimeError("invalid original-clamp prototype values")

    def normalized(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim != 2 or descriptor.shape[1] != self.center.numel():
            raise ValueError(
                "contact-geometry descriptor must have shape (env,26)"
            )
        return (descriptor - self.center) / self.scale

    def score(self, descriptor: torch.Tensor) -> torch.Tensor:
        standardized = self.normalized(descriptor)
        pose_score = torch.exp(
            -0.5 * torch.square(standardized).mean(dim=-1)
        )
        wrist_separation = torch.linalg.vector_norm(
            descriptor[:, 18:21], dim=-1
        )
        separation_factor = torch.clamp(
            wrist_separation / self.minimum_wrist_separation_m,
            min=0.0,
            max=1.0,
        )
        return pose_score * separation_factor


@dataclass(frozen=True)
class FailedStrategyAttemptMemoryCfg:
    stable_contact_steps: int = 3
    persistent_slip_steps: int = 2
    micro_lift_window_steps: int = 25
    minimum_micro_lift_m: float = 0.03
    release_regrasp_grace_steps: int = 10
    maximum_support_drop_m: float = 0.04
    failed_memory_size: int = 4

    def __post_init__(self) -> None:
        integer_names = (
            "stable_contact_steps",
            "persistent_slip_steps",
            "micro_lift_window_steps",
            "release_regrasp_grace_steps",
            "failed_memory_size",
        )
        if any(int(getattr(self, name)) < 1 for name in integer_names):
            raise ValueError("attempt-memory step counts and capacity must be positive")
        if self.minimum_micro_lift_m <= 0.0 or self.maximum_support_drop_m <= 0.0:
            raise ValueError("attempt-memory height thresholds must be positive")


class FailedStrategyAttemptMemory:
    """Per-episode failed-attempt memory, independent of ICM novelty.

    Failure is an external task/safety event.  It does not define curiosity.
    The stored descriptor merely prevents the policy optimizer from repeatedly
    choosing the same already-failed contact geometry while the separately
    trained ICM continues to value newly unpredictable controllable
    transitions, including novel failures.
    """

    IDLE = 0
    ACTIVE = 1
    RELEASE_GRACE = 2

    def __init__(
        self,
        num_envs: int,
        descriptor_scale: torch.Tensor,
        device: torch.device | str,
        repeat_similarity_threshold: float = REPEATED_FAILED_SIMILARITY_THRESHOLD,
        cfg: FailedStrategyAttemptMemoryCfg = FailedStrategyAttemptMemoryCfg(),
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self.descriptor_scale = descriptor_scale.detach().clone().to(self.device)
        self.descriptor_dim = int(self.descriptor_scale.numel())
        self.repeat_similarity_threshold = float(repeat_similarity_threshold)
        if not 0.0 < self.repeat_similarity_threshold < 1.0:
            raise ValueError("repeat-similarity threshold must be in (0,1)")
        shape = (self.num_envs,)
        self.phase = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.stable_contact_count = torch.zeros_like(self.phase)
        self.persistent_slip_count = torch.zeros_like(self.phase)
        self.release_count = torch.zeros_like(self.phase)
        self.attempt_steps = torch.zeros_like(self.phase)
        self.attempt_index = torch.zeros_like(self.phase)
        self.failed_attempt_count = torch.zeros_like(self.phase)
        self.initial_strategy_failed = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.blocked_until_release = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.lockout_release_count = torch.zeros_like(self.phase)
        self.start_height = torch.zeros(shape, device=self.device)
        self.max_height = torch.zeros(shape, device=self.device)
        self.descriptor_sum = torch.zeros(
            (self.num_envs, self.descriptor_dim), device=self.device
        )
        self.descriptor_count = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.failed_descriptors = torch.zeros(
            (
                self.num_envs,
                cfg.failed_memory_size,
                self.descriptor_dim,
            ),
            device=self.device,
        )
        self.failed_valid = torch.zeros(
            (self.num_envs, cfg.failed_memory_size),
            dtype=torch.bool,
            device=self.device,
        )
        self.failed_write_index = torch.zeros_like(self.phase)

    def reset(self, reset_mask: torch.Tensor) -> None:
        if reset_mask.shape != (self.num_envs,) or reset_mask.dtype != torch.bool:
            raise ValueError("attempt-memory reset mask must be one-dimensional bool")
        self.phase[reset_mask] = self.IDLE
        self.stable_contact_count[reset_mask] = 0
        self.persistent_slip_count[reset_mask] = 0
        self.release_count[reset_mask] = 0
        self.attempt_steps[reset_mask] = 0
        self.attempt_index[reset_mask] = 0
        self.failed_attempt_count[reset_mask] = 0
        self.initial_strategy_failed[reset_mask] = False
        self.blocked_until_release[reset_mask] = False
        self.lockout_release_count[reset_mask] = 0
        self.start_height[reset_mask] = 0.0
        self.max_height[reset_mask] = 0.0
        self.descriptor_sum[reset_mask] = 0.0
        self.descriptor_count[reset_mask] = 0
        self.failed_descriptors[reset_mask] = 0.0
        self.failed_valid[reset_mask] = False
        self.failed_write_index[reset_mask] = 0

    def _close_failed(self, failure: torch.Tensor) -> None:
        env_ids = torch.nonzero(failure, as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        count = self.descriptor_count[env_ids].clamp_min(1).float().unsqueeze(-1)
        mean_descriptor = self.descriptor_sum[env_ids] / count
        slots = self.failed_write_index[env_ids] % self.cfg.failed_memory_size
        self.failed_descriptors[env_ids, slots] = mean_descriptor
        self.failed_valid[env_ids, slots] = True
        self.failed_write_index[env_ids] += 1
        self.failed_attempt_count[env_ids] += 1
        was_initial_attempt = self.attempt_index[env_ids] == 1
        self.initial_strategy_failed[env_ids] |= was_initial_attempt

    def _clear_active(self, mask: torch.Tensor) -> None:
        self.phase[mask] = self.IDLE
        self.stable_contact_count[mask] = 0
        self.persistent_slip_count[mask] = 0
        self.release_count[mask] = 0
        self.attempt_steps[mask] = 0
        self.descriptor_sum[mask] = 0.0
        self.descriptor_count[mask] = 0

    @torch.no_grad()
    def update(
        self,
        descriptor: torch.Tensor,
        direct_contact: torch.Tensor,
        slip_active: torch.Tensor,
        box_lift_height: torch.Tensor,
        success: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if descriptor.shape != (self.num_envs, self.descriptor_dim):
            raise ValueError("attempt-memory descriptor shape drift")
        expected_vector = (self.num_envs,)
        if (
            box_lift_height.shape != expected_vector
            or success.shape != expected_vector
            or reset_mask.shape != expected_vector
            or direct_contact.shape != (self.num_envs, 2)
            or slip_active.shape != (self.num_envs, 2)
        ):
            raise ValueError("attempt-memory input shape drift")
        if not torch.isfinite(descriptor).all() or not torch.isfinite(
            box_lift_height
        ).all():
            raise ValueError("attempt-memory received non-finite geometry")

        self.reset(reset_mask)
        valid = ~reset_mask
        any_contact = direct_contact.any(dim=-1)
        bilateral_contact = direct_contact.all(dim=-1)
        any_slip = slip_active.any(dim=-1)

        # A persistent-contact failure is one failed attempt, not a new attempt
        # every two slip frames.  Re-arm only after a stable no-contact release.
        self.lockout_release_count = torch.where(
            valid & self.blocked_until_release & ~any_contact,
            self.lockout_release_count + 1,
            torch.zeros_like(self.lockout_release_count),
        )
        release_complete = self.blocked_until_release & (
            self.lockout_release_count >= self.cfg.release_regrasp_grace_steps
        )
        self.blocked_until_release[release_complete] = False
        self.lockout_release_count[release_complete] = 0
        self.stable_contact_count = torch.where(
            valid & any_contact & ~self.blocked_until_release,
            self.stable_contact_count + 1,
            torch.zeros_like(self.stable_contact_count),
        )
        inactive = self.phase == self.IDLE
        start = (
            valid
            & inactive
            & ~self.blocked_until_release
            & any_contact
            & (self.stable_contact_count >= self.cfg.stable_contact_steps)
        )
        self.phase[start] = self.ACTIVE
        self.attempt_index[start] += 1
        self.attempt_steps[start] = 0
        self.start_height[start] = box_lift_height[start]
        self.max_height[start] = box_lift_height[start]
        self.descriptor_sum[start] = 0.0
        self.descriptor_count[start] = 0

        active = valid & (self.phase != self.IDLE)
        self.attempt_steps[active] += 1
        self.max_height[active] = torch.maximum(
            self.max_height[active], box_lift_height[active]
        )
        contact_sample = active & any_contact
        self.descriptor_sum[contact_sample] += descriptor[contact_sample]
        self.descriptor_count[contact_sample] += 1
        self.persistent_slip_count = torch.where(
            active & any_slip,
            self.persistent_slip_count + 1,
            torch.zeros_like(self.persistent_slip_count),
        )
        self.release_count = torch.where(
            active & ~any_contact,
            self.release_count + 1,
            torch.zeros_like(self.release_count),
        )
        self.phase[active & ~any_contact] = self.RELEASE_GRACE
        self.phase[active & any_contact] = self.ACTIVE

        progress = self.max_height - self.start_height
        support_drop = self.max_height - box_lift_height
        persistent_slip_failure = active & (
            self.persistent_slip_count >= self.cfg.persistent_slip_steps
        )
        no_progress_failure = (
            active
            & (self.attempt_steps >= self.cfg.micro_lift_window_steps)
            & (progress < self.cfg.minimum_micro_lift_m)
        )
        release_failure = (
            active
            & (self.release_count > self.cfg.release_regrasp_grace_steps)
            & (
                (support_drop > self.cfg.maximum_support_drop_m)
                | (progress < self.cfg.minimum_micro_lift_m)
            )
        )
        failure = (
            persistent_slip_failure | no_progress_failure | release_failure
        ) & ~success
        success_close = active & success
        self._close_failed(failure)
        self.blocked_until_release |= failure & any_contact
        self.lockout_release_count[failure] = 0
        self._clear_active(failure | success_close)

        standardized = descriptor / self.descriptor_scale
        failed_standardized = self.failed_descriptors / self.descriptor_scale.view(
            1, 1, -1
        )
        distance_squared = torch.square(
            standardized[:, None] - failed_standardized
        ).mean(dim=-1)
        distance_squared = torch.where(
            self.failed_valid,
            distance_squared,
            torch.full_like(distance_squared, torch.inf),
        )
        nearest_distance_squared = distance_squared.min(dim=-1).values
        nearest_failed_similarity = torch.where(
            torch.isfinite(nearest_distance_squared),
            torch.exp(-0.5 * nearest_distance_squared),
            torch.zeros_like(nearest_distance_squared),
        )
        denominator = max(1.0 - self.repeat_similarity_threshold, 1.0e-6)
        repeated_failed_cost = (
            any_contact.float()
            * torch.clamp(
                (
                    nearest_failed_similarity
                    - self.repeat_similarity_threshold
                )
                / denominator,
                min=0.0,
                max=1.0,
            )
        )
        return {
            "phase": self.phase.clone(),
            "attempt_active": self.phase != self.IDLE,
            "attempt_index": self.attempt_index.clone(),
            "attempt_steps": self.attempt_steps.clone(),
            "failed_attempt_count": self.failed_attempt_count.clone(),
            "initial_strategy_failed": self.initial_strategy_failed.clone(),
            "blocked_until_release": self.blocked_until_release.clone(),
            "lockout_release_count": self.lockout_release_count.clone(),
            "contact_descriptor_count": self.descriptor_count.clone(),
            "nearest_failed_similarity": nearest_failed_similarity,
            "repeated_failed_cost": repeated_failed_cost,
            "failure_closed": failure,
            "success_closed": success_close,
            "persistent_slip_failure": persistent_slip_failure,
            "no_progress_failure": no_progress_failure,
            "release_failure": release_failure,
            "bilateral_contact": bilateral_contact,
            "single_hand_contact": direct_contact.sum(dim=-1) == 1,
            "box_lift_height": box_lift_height,
            "attempt_max_lift": self.max_height - self.start_height,
        }


class GoalTactileStrategyRuntime:
    """Once-per-control-step cache shared by observations and external costs."""

    def __init__(
        self,
        env,
        left_sensor_name: str,
        right_sensor_name: str,
        history_steps: int,
        grid_shape: tuple[int, int],
        taxel_area_m2: float,
        stress_scale: float,
        calibration_path: str | Path = DEFAULT_V16_CALIBRATION,
        result_path: str | Path = DEFAULT_V16_RESULT,
        prototype_path: str | Path = DEFAULT_ORIGINAL_CLAMP_PROTOTYPE,
    ) -> None:
        if tuple(grid_shape) != (20, 25) or history_steps < 2:
            raise ValueError("admitted v16 runtime requires at least two 20x25 frames")
        model_source = Path(
            __file__
        ).with_name("direct_tactile_slip_spatiotemporal.py")
        if _sha256(model_source) != EXPECTED_V16_MODEL_SOURCE_SHA256:
            raise RuntimeError("frozen v16 tactile-only model source hash drift")
        result_path = Path(result_path).expanduser().resolve()
        result_digest = _sha256(result_path)
        if result_digest != EXPECTED_V16_RESULT_SHA256:
            raise RuntimeError("immutable v16 fresh result hash drift")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("protocol")
            != "sugar_direct_tacsl_spatiotemporal_v16_fresh_stage_e"
            or result.get("passed") is not True
            or result.get("stage_e_gate_passed") is not True
        ):
            raise RuntimeError("v16 tactile-only Stage E is not admitted")
        calibration, calibration_digest = _load_v16_calibration(
            Path(calibration_path)
        )
        if (
            result.get("artifacts", {}).get("v16_admission_sha256")
            != calibration_digest
        ):
            raise RuntimeError("v16 result does not bind the selected calibration")

        self.env = env
        self.left_sensor_name = left_sensor_name
        self.right_sensor_name = right_sensor_name
        self.history_steps = int(history_steps)
        self.grid_shape = tuple(grid_shape)
        self.taxel_area_m2 = float(taxel_area_m2)
        self.stress_scale = float(stress_scale)
        self.control_dt = float(env.step_dt)
        if abs(self.control_dt - 0.02) > 1.0e-9:
            raise RuntimeError(f"v16 detector expects 50 Hz, got dt={self.control_dt}")
        self.estimator = SpatiotemporalDirectTactileSlipEstimator(
            calibration=calibration,
            num_envs=env.num_envs,
            device=env.device,
            taxel_area_m2=self.taxel_area_m2,
            stress_scale=self.stress_scale,
            control_dt=self.control_dt,
        )
        self.prototype = OriginalClampPrototype(
            prototype_path,
            device=env.device,
            expected_sha256=EXPECTED_ORIGINAL_CLAMP_PROTOTYPE_SHA256,
        )
        robot = env.scene["robot"]
        body_ids, body_names = robot.find_bodies(
            ["left_wrist_yaw_link", "right_wrist_yaw_link"],
            preserve_order=True,
        )
        if list(body_names) != ["left_wrist_yaw_link", "right_wrist_yaw_link"]:
            raise RuntimeError("official SUGAR wrist ordering drift")
        self.robot = robot
        self.obj = env.scene["obj"]
        self.wrist_ids = torch.as_tensor(
            body_ids, dtype=torch.long, device=env.device
        )
        self.memory = FailedStrategyAttemptMemory(
            num_envs=env.num_envs,
            descriptor_scale=self.prototype.scale,
            device=env.device,
        )
        self.last_step = -1
        self.output: dict[str, torch.Tensor] | None = None

    def _box_frame_descriptor(self) -> torch.Tensor:
        wrist_position = self.robot.data.body_pos_w[:, self.wrist_ids]
        wrist_quaternion = self.robot.data.body_quat_w[:, self.wrist_ids]
        parts = []
        positions = []
        tangents = []
        normals = []
        for hand in range(2):
            position_box, quaternion_box = subtract_frame_transforms(
                self.obj.data.root_pos_w,
                self.obj.data.root_quat_w,
                wrist_position[:, hand],
                wrist_quaternion[:, hand],
            )
            rotation_box = matrix_from_quat(quaternion_box)
            tangent = rotation_box[..., :, 0]
            normal = rotation_box[..., :, 2]
            positions.append(position_box)
            tangents.append(tangent)
            normals.append(normal)
            parts.extend(
                (
                    position_box,
                    tangent,
                    normal,
                )
            )
        parts.extend(
            (
                positions[0] - positions[1],
                0.5 * (positions[0] + positions[1]),
                (tangents[0] * tangents[1]).sum(dim=-1, keepdim=True),
                (normals[0] * normals[1]).sum(dim=-1, keepdim=True),
            )
        )
        descriptor = torch.cat(parts, dim=-1)
        expected = (self.env.num_envs, len(ORIGINAL_CLAMP_DESCRIPTOR_NAMES))
        if descriptor.shape != expected:
            raise RuntimeError("runtime original-clamp descriptor shape drift")
        if not torch.isfinite(descriptor).all():
            raise RuntimeError("runtime original-clamp descriptor is non-finite")
        return descriptor

    @torch.no_grad()
    def update(self) -> dict[str, torch.Tensor]:
        step = int(self.env.common_step_counter)
        if self.output is not None and self.last_step == step:
            return self.output
        flattened_history = direct_tactile_force_history(
            self.env,
            left_sensor_name=self.left_sensor_name,
            right_sensor_name=self.right_sensor_name,
            history_steps=self.history_steps,
            grid_shape=self.grid_shape,
            taxel_area_m2=self.taxel_area_m2,
            stress_scale=self.stress_scale,
        )
        tactile_history = flattened_history.reshape(
            self.env.num_envs,
            self.history_steps,
            2,
            3,
            *self.grid_shape,
        )
        reset_mask = self.env.episode_length_buf == 0
        slip = self.estimator.update(tactile_history, reset_mask=reset_mask)
        descriptor = self._box_frame_descriptor()
        command = self.env.command_manager.get_term("motion")
        box_lift_height = command.obj_pos_w[:, 2] - command.initial_obj_height_w
        success = command.goal_stable_counter >= command.cfg.success_stable_steps
        strategy = self.memory.update(
            descriptor=descriptor,
            direct_contact=slip["contact"],
            slip_active=slip["incipient"],
            box_lift_height=box_lift_height,
            success=success,
            reset_mask=reset_mask,
        )
        normalized_descriptor = torch.clamp(
            self.prototype.normalized(descriptor), min=-10.0, max=10.0
        )
        slip_observation = torch.cat(
            (
                slip["contact"].float(),
                slip["slip_probability"],
                slip["gross_probability"],
                slip["state"].float() / 3.0,
                torch.clamp(slip["contact_age"].float() / 16.0, max=1.0),
                slip["slip_event_started"].float(),
                slip["slip_event_ended"].float(),
            ),
            dim=-1,
        )
        strategy_observation = torch.cat(
            (
                normalized_descriptor,
                strategy["nearest_failed_similarity"][:, None],
                strategy["repeated_failed_cost"][:, None],
                strategy["attempt_active"].float()[:, None],
                strategy["phase"].float()[:, None] / 2.0,
                torch.clamp(strategy["attempt_index"].float(), max=8.0)[:, None]
                / 8.0,
                torch.clamp(strategy["failed_attempt_count"].float(), max=8.0)[
                    :, None
                ]
                / 8.0,
                strategy["initial_strategy_failed"].float()[:, None],
                strategy["blocked_until_release"].float()[:, None],
                strategy["bilateral_contact"].float()[:, None],
                strategy["single_hand_contact"].float()[:, None],
                strategy["box_lift_height"][:, None],
                strategy["attempt_max_lift"][:, None],
                torch.clamp(
                    strategy["contact_descriptor_count"].float(),
                    max=float(self.memory.cfg.micro_lift_window_steps),
                )[:, None]
                / float(self.memory.cfg.micro_lift_window_steps),
                strategy["failure_closed"].float()[:, None],
            ),
            dim=-1,
        )
        if slip_observation.shape != (self.env.num_envs, 14):
            raise RuntimeError("v16 slip observation shape drift")
        expected_strategy_dim = len(ORIGINAL_CLAMP_DESCRIPTOR_NAMES) + 14
        if strategy_observation.shape != (
            self.env.num_envs,
            expected_strategy_dim,
        ):
            raise RuntimeError("anti-repeat strategy observation shape drift")
        self.output = {
            **{f"slip/{name}": value for name, value in slip.items()},
            **{f"strategy/{name}": value for name, value in strategy.items()},
            "strategy/descriptor": descriptor,
            "strategy/normalized_descriptor": normalized_descriptor,
            "observation/slip": slip_observation,
            "observation/strategy": strategy_observation,
            "external/slip_cost": (
                slip["incipient"].float() + slip["gross"].float()
            ).mean(dim=-1),
            "external/repeated_failed_cost": strategy["repeated_failed_cost"],
        }
        self.last_step = step
        return self.output


class ExplicitZeroTactileStrategyControlRuntime:
    """Exact-width no-tactile control used only by declared ablations.

    The control intentionally exposes no contact, slip, failed-contact memory,
    or tactile-derived external cost.  It keeps the established observation
    widths so the serious SUGAR-native actor and frozen demo predictor are not
    replaced or resized.  No tactile sensor, proxy contact, or object property
    is read here.
    """

    protocol = "sugar_explicit_zero_tactile_strategy_control_v1"

    def __init__(self, env) -> None:
        self.env = env
        self.update_calls = 0
        self.sensor_read_count = 0
        self.output: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self) -> dict[str, torch.Tensor]:
        self.update_calls += 1
        if self.output is None:
            zeros = torch.zeros(
                self.env.num_envs,
                dtype=torch.float32,
                device=self.env.device,
            )
            false = torch.zeros_like(zeros, dtype=torch.bool)
            self.output = {
                "observation/slip": torch.zeros(
                    (self.env.num_envs, 14),
                    dtype=torch.float32,
                    device=self.env.device,
                ),
                "observation/strategy": torch.zeros(
                    (
                        self.env.num_envs,
                        len(ORIGINAL_CLAMP_DESCRIPTOR_NAMES) + 14,
                    ),
                    dtype=torch.float32,
                    device=self.env.device,
                ),
                "external/slip_cost": zeros,
                "external/repeated_failed_cost": zeros.clone(),
                "strategy/initial_strategy_failed": false,
                "strategy/failure_closed": false.clone(),
            }
        return self.output

    def audit_state(self) -> dict[str, object]:
        output = self.update()
        return {
            "protocol": self.protocol,
            "update_calls": self.update_calls,
            "sensor_read_count": self.sensor_read_count,
            "slip_observation_shape": list(output["observation/slip"].shape),
            "strategy_observation_shape": list(
                output["observation/strategy"].shape
            ),
            "all_outputs_exact_zero": all(
                int(torch.count_nonzero(value)) == 0
                for value in output.values()
            ),
        }


def _explicit_zero_runtime(env) -> ExplicitZeroTactileStrategyControlRuntime:
    key = (ExplicitZeroTactileStrategyControlRuntime.protocol,)
    entry = getattr(env, "_sugar_goal_tactile_strategy_runtime", None)
    if entry is None:
        runtime = ExplicitZeroTactileStrategyControlRuntime(env)
        setattr(env, "_sugar_goal_tactile_strategy_runtime", (key, runtime))
        return runtime
    if (
        not isinstance(entry, tuple)
        or len(entry) != 2
        or entry[0] != key
        or not isinstance(entry[1], ExplicitZeroTactileStrategyControlRuntime)
    ):
        raise RuntimeError(
            "explicit-zero control cannot replace an active tactile strategy runtime"
        )
    return entry[1]


def explicit_zero_tactile_slip_observation(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int,
    grid_shape: tuple[int, int],
    taxel_area_m2: float,
    stress_scale: float,
) -> torch.Tensor:
    del (
        left_sensor_name,
        right_sensor_name,
        history_steps,
        grid_shape,
        taxel_area_m2,
        stress_scale,
    )
    return _explicit_zero_runtime(env).update()["observation/slip"]


def explicit_zero_anti_repeat_strategy_observation(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int,
    grid_shape: tuple[int, int],
    taxel_area_m2: float,
    stress_scale: float,
) -> torch.Tensor:
    del (
        left_sensor_name,
        right_sensor_name,
        history_steps,
        grid_shape,
        taxel_area_m2,
        stress_scale,
    )
    return _explicit_zero_runtime(env).update()["observation/strategy"]


def explicit_zero_tactile_external_cost(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int,
    grid_shape: tuple[int, int],
    taxel_area_m2: float,
    stress_scale: float,
) -> torch.Tensor:
    del (
        left_sensor_name,
        right_sensor_name,
        history_steps,
        grid_shape,
        taxel_area_m2,
        stress_scale,
    )
    return _explicit_zero_runtime(env).update()["external/slip_cost"]


def _runtime(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int,
    grid_shape: tuple[int, int],
    taxel_area_m2: float,
    stress_scale: float,
) -> GoalTactileStrategyRuntime:
    key = (
        left_sensor_name,
        right_sensor_name,
        int(history_steps),
        tuple(grid_shape),
        float(taxel_area_m2),
        float(stress_scale),
    )
    entry: tuple[tuple[Any, ...], GoalTactileStrategyRuntime] | None = getattr(
        env, "_sugar_goal_tactile_strategy_runtime", None
    )
    if entry is None:
        runtime = GoalTactileStrategyRuntime(
            env,
            left_sensor_name=left_sensor_name,
            right_sensor_name=right_sensor_name,
            history_steps=history_steps,
            grid_shape=grid_shape,
            taxel_area_m2=taxel_area_m2,
            stress_scale=stress_scale,
        )
        setattr(env, "_sugar_goal_tactile_strategy_runtime", (key, runtime))
        return runtime
    if entry[0] != key:
        raise RuntimeError("goal tactile/strategy runtime configuration drift")
    return entry[1]


def v16_tactile_slip_observation(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int,
    grid_shape: tuple[int, int],
    taxel_area_m2: float,
    stress_scale: float,
) -> torch.Tensor:
    return _runtime(
        env,
        left_sensor_name,
        right_sensor_name,
        history_steps,
        grid_shape,
        taxel_area_m2,
        stress_scale,
    ).update()["observation/slip"]


def anti_repeat_strategy_observation(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int,
    grid_shape: tuple[int, int],
    taxel_area_m2: float,
    stress_scale: float,
) -> torch.Tensor:
    return _runtime(
        env,
        left_sensor_name,
        right_sensor_name,
        history_steps,
        grid_shape,
        taxel_area_m2,
        stress_scale,
    ).update()["observation/strategy"]


def v16_tactile_slip_cost(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int,
    grid_shape: tuple[int, int],
    taxel_area_m2: float,
    stress_scale: float,
) -> torch.Tensor:
    return _runtime(
        env,
        left_sensor_name,
        right_sensor_name,
        history_steps,
        grid_shape,
        taxel_area_m2,
        stress_scale,
    ).update()["external/slip_cost"]


def repeated_failed_strategy_cost(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int,
    grid_shape: tuple[int, int],
    taxel_area_m2: float,
    stress_scale: float,
) -> torch.Tensor:
    return _runtime(
        env,
        left_sensor_name,
        right_sensor_name,
        history_steps,
        grid_shape,
        taxel_area_m2,
        stress_scale,
    ).update()["external/repeated_failed_cost"]


def _weaker_direct_tacsl_normal_load(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    """Integrate each hand's official taxel-resolved normal-force field."""

    integrated = []
    expected_taxels = int(grid_shape[0]) * int(grid_shape[1])
    for sensor_name in (left_sensor_name, right_sensor_name):
        normal_force = env.scene.sensors[
            sensor_name
        ].data.tactile_normal_force
        if normal_force is None or normal_force.numel() != (
            env.num_envs * expected_taxels
        ):
            shape = None if normal_force is None else tuple(normal_force.shape)
            raise RuntimeError(
                f"unexpected direct TacSL normal-force field for "
                f"{sensor_name!r}: {shape}"
            )
        force = torch.nan_to_num(
            normal_force.reshape(env.num_envs, expected_taxels)
        ).clamp_min(0.0)
        integrated.append(force.sum(dim=-1))
    return torch.stack(integrated, dim=-1).amin(dim=-1)


def pre_failure_bilateral_contact_load_retention(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int,
    grid_shape: tuple[int, int],
    taxel_area_m2: float,
    stress_scale: float,
    target_integrated_normal_n: float,
) -> torch.Tensor:
    """Reward balanced direct-TacSL normal load before the first failed attempt.

    This is a separately logged external curriculum objective, not curiosity.
    It integrates each official sensor's taxel-resolved normal-force field,
    uses the weaker hand, and stops contributing once the attempt-memory
    runtime has closed the first failed strategy.  The actor still receives
    the complete spatial pressure and signed-shear history.
    """

    if target_integrated_normal_n <= 0.0:
        raise ValueError("bilateral contact-load target must be positive")
    runtime = _runtime(
        env,
        left_sensor_name,
        right_sensor_name,
        history_steps,
        grid_shape,
        taxel_area_m2,
        stress_scale,
    ).update()
    weaker_hand_load = _weaker_direct_tacsl_normal_load(
        env, left_sensor_name, right_sensor_name, grid_shape
    )
    pre_failure = ~runtime["strategy/initial_strategy_failed"]
    return (
        torch.clamp(
            weaker_hand_load / float(target_integrated_normal_n),
            min=0.0,
            max=1.0,
        )
        * pre_failure.float()
    )


def bilateral_contact_load_foundation(
    env,
    left_sensor_name: str,
    right_sensor_name: str,
    history_steps: int,
    grid_shape: tuple[int, int],
    taxel_area_m2: float,
    stress_scale: float,
    target_integrated_normal_n: float,
) -> torch.Tensor:
    """Score balanced direct-TacSL load across a nominal foundation horizon.

    This external nominal-clamp curriculum is not ICM. Unlike the rejected
    pre-failure term, it remains active after the attempt-memory runtime closes
    a failure so that later valid contact transitions in the bounded
    foundation segment are not discarded. It must be removed before the
    post-failure alternative-strategy discovery phase.
    """

    if target_integrated_normal_n <= 0.0:
        raise ValueError("bilateral contact-foundation target must be positive")
    _runtime(
        env,
        left_sensor_name,
        right_sensor_name,
        history_steps,
        grid_shape,
        taxel_area_m2,
        stress_scale,
    ).update()
    weaker_hand_load = _weaker_direct_tacsl_normal_load(
        env, left_sensor_name, right_sensor_name, grid_shape
    )
    return torch.clamp(
        weaker_hand_load / float(target_integrated_normal_n),
        min=0.0,
        max=1.0,
    )


# Historical import aliases for already-written v1 audit scripts only.  Active
# task configuration uses FailedStrategyAttemptMemory and
# repeated_failed_strategy_cost.
OriginalClampAttemptMemoryCfg = FailedStrategyAttemptMemoryCfg
OriginalClampAttemptMemory = FailedStrategyAttemptMemory
