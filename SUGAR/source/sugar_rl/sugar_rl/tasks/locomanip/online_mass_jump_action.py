# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Action-boundary hook for applying a scheduled mass jump before physics."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class OnlineMassJumpJointPositionAction(JointPositionAction):
    """Preserve the official action mapping and apply only pending mass writes."""

    def process_actions(self, actions: torch.Tensor):
        controller = getattr(self._env, "_online_mass_jump_controller", None)
        if controller is not None:
            controller.apply_pending(control_step=int(self._env.common_step_counter))
            self._env._online_mass_jump_diagnostics = controller.diagnostics()
        super().process_actions(actions)
