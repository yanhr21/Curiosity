# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Serious spatial-TacSL actor for the deployable Tracker-command student."""

import torch

from sugar_rl.utils.reference_only_tactile_actor_critic import (
    ReferenceOnlyTactileActorCritic,
)


class TrackerCommandTactileActorCritic(ReferenceOnlyTactileActorCritic):
    """Full 512/256/128 student with a training-only privileged critic.

    This reuses the existing per-hand SpatialTactileEncoder and SUGAR MLP. The
    504-D deployed actor is initialized from the released 510-D Tracker after
    removing its contact-label and measured-object columns. The released
    Refiner remains the frozen BCPPO teacher.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        first_layer = self.actor[0]
        with torch.no_grad():
            first_layer.weight[:, self.num_actor_base_obs :].mul_(
                self.warm_start_tactile_gain
            )
        self._freeze_zero_preserving_tactile_biases()

    def _freeze_zero_preserving_tactile_biases(self) -> list[str]:
        frozen = []
        for module_name, module in self.actor_tactile_encoder.named_modules():
            bias = getattr(module, "bias", None)
            if bias is not None:
                bias.requires_grad_(False)
                frozen.append(f"{module_name}.bias" if module_name else "bias")
        return sorted(frozen)

    def configure_tactile_actor_finetune(self) -> dict[str, object]:
        """Restore full-student training after loading a matched base checkpoint."""

        for parameter in self.actor.parameters():
            parameter.requires_grad_(True)
        for parameter in self.actor_tactile_encoder.parameters():
            parameter.requires_grad_(True)
        if self.noise_std_type == "scalar":
            self.std.requires_grad_(True)
        else:
            self.log_std.requires_grad_(True)
        frozen_biases = self._freeze_zero_preserving_tactile_biases()
        return {
            "mode": "full_tracker_command_student_with_spatial_tactile",
            "actor_base_columns_trainable": self.num_actor_base_obs,
            "actor_tactile_columns_trainable": (
                self.actor[0].in_features - self.num_actor_base_obs
            ),
            "frozen_zero_preserving_encoder_biases": frozen_biases,
            "critic_training": "unchanged_privileged_critic_optimization",
        }

    def load_sugar_warm_start(
        self, source_state: dict[str, torch.Tensor]
    ) -> dict[str, object]:
        """Initialize the proxy-free actor from the released 510-D Tracker.

        The 35-D command and all five-frame robot, action, and gravity history
        columns transfer directly. The online actor never receives the
        Tracker's contact label or measured object pose. Base linear velocity
        and motion phase are appended with zero initial authority.
        """

        source_first = source_state.get("actor.0.weight")
        if source_first is None or tuple(source_first.shape) != (512, 510):
            raise ValueError(
                "Tracker-command warm start requires the released 510-D "
                f"Tracker actor, got {None if source_first is None else tuple(source_first.shape)}"
            )
        target_first = self.actor[0]
        if tuple(target_first.weight.shape) != (512, 760):
            raise ValueError(
                f"Unexpected reduced tactile actor input {tuple(target_first.weight.shape)}"
            )

        device = target_first.weight.device
        dtype = target_first.weight.dtype

        def source(name: str) -> torch.Tensor:
            if name not in source_state:
                raise KeyError(f"Released Tracker checkpoint is missing {name}")
            return source_state[name].to(device=device, dtype=dtype)

        with torch.no_grad():
            # Preserve the independently initialized low-gain tactile columns.
            target_first.weight[:, : self.num_actor_base_obs].zero_()
            target_first.bias.copy_(source("actor.0.bias"))

            # Column 35 is the excluded contact label. Source columns 36:501
            # are the official five-frame proprioception/action/gravity
            # history and transfer without reshaping or loss.
            target_first.weight[:, 0:35].copy_(source_first[:, 0:35])
            target_first.weight[:, 35:500].copy_(source_first[:, 36:501])

            # Base linear velocity (500:503) and phase (503) are additional
            # deployable inputs with zero initial first-layer authority.
            for name, parameter in self.actor.named_parameters():
                if name in ("0.weight", "0.bias"):
                    continue
                parameter.copy_(source(f"actor.{name}"))
            for name, parameter in self.critic.named_parameters():
                parameter.copy_(source(f"critic.{name}"))
            if self.noise_std_type == "scalar":
                self.std.copy_(source("std"))
            else:
                self.log_std.copy_(source("log_std"))

        return {
            "source_policy": "released_official_carrybox_tracker_510d",
            "target_actor_base_width": int(self.num_actor_base_obs),
            "target_tactile_feature_width": int(
                target_first.weight.shape[1] - self.num_actor_base_obs
            ),
            "direct_tracker_command_columns": 35,
            "direct_robot_action_gravity_history_columns": 465,
            "new_zero_authority_columns": {
                "base_linear_velocity": [500, 503],
                "motion_phase": [503, 504],
            },
            "excluded_source_columns": {
                "contact_label": [35, 36],
                "measured_object_position_orientation": [501, 510],
            },
            "actor_receives_excluded_source_values": False,
            "privileged_critic_transfer": "exact_890d",
            "zero_tactile_feature_at_warm_start": True,
        }
