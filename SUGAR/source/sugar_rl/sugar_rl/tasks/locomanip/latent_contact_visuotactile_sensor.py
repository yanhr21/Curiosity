# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Auditable per-environment friction adapter for the official TacSL sensor.

This module is intentionally not imported by the active optimizer-clean task.
It is reserved for the separately gated latent-contact-dynamics follow-up.
The adapter does not reimplement the TacSL SDF or penalty-force equations: it
selects the coefficient for the exact environment subset being updated and
then calls the official :class:`VisuoTactileSensor` implementation unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor


class PerEnvironmentFrictionVisuoTactileSensor(VisuoTactileSensor):
    """Official TacSL sensor with an audited per-environment Coulomb cap."""

    def __init__(self, cfg):
        self._friction_coefficient_by_env: torch.Tensor | None = None
        super().__init__(cfg)

    def _initialize_impl(self):
        super()._initialize_impl()
        coefficient = float(self.cfg.friction_coefficient)
        self._friction_coefficient_by_env = torch.full(
            (self._num_envs,), coefficient, dtype=torch.float32, device=self._device
        )

    @property
    def friction_coefficient_by_env(self) -> torch.Tensor:
        """Return a clone so audits cannot mutate the live force-field state."""

        if self._friction_coefficient_by_env is None:
            raise RuntimeError("TacSL sensor is not initialized")
        return self._friction_coefficient_by_env.clone()

    def set_friction_coefficient_by_env(
        self,
        coefficients: torch.Tensor | Sequence[float] | float,
        env_ids: torch.Tensor | Sequence[int] | slice | None = None,
    ) -> None:
        """Set the coefficient used by the official force equation.

        The method accepts scalar broadcast or one value for every selected
        environment.  It deliberately rejects negative/non-finite values.
        """

        if self._friction_coefficient_by_env is None:
            raise RuntimeError("TacSL sensor is not initialized")
        if env_ids is None:
            env_ids = slice(None)
        elif not isinstance(env_ids, slice):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)

        target = self._friction_coefficient_by_env[env_ids]
        values = torch.as_tensor(coefficients, dtype=target.dtype, device=self._device)
        if values.ndim == 0:
            values = values.expand_as(target)
        else:
            values = values.reshape(-1)
            if values.numel() != target.numel():
                raise ValueError(
                    "Expected one TacSL friction coefficient per selected environment: "
                    f"got {values.numel()} for {target.numel()} environments"
                )
        if not torch.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("TacSL friction coefficients must be finite and non-negative")
        self._friction_coefficient_by_env[env_ids] = values

    def _compute_tactile_forces_from_sdf(
        self,
        points_contact_object_local: torch.Tensor,
        sdf_values: torch.Tensor,
        sdf_gradients: torch.Tensor,
        contact_object_pos_w: torch.Tensor,
        contact_object_quat_w: torch.Tensor,
        elastomer_quat_w: torch.Tensor,
        env_ids: Sequence[int] | slice,
    ) -> None:
        """Call the official equation with the coefficient for ``env_ids``.

        The upstream equation multiplies ``cfg.friction_coefficient`` by a
        ``[num_selected_envs, num_taxels]`` normal-force tensor.  A temporary
        ``[num_selected_envs, 1]`` tensor therefore broadcasts over taxels and
        also supports lazy partial-environment sensor refreshes.
        """

        if self._friction_coefficient_by_env is None:
            return super()._compute_tactile_forces_from_sdf(
                points_contact_object_local,
                sdf_values,
                sdf_gradients,
                contact_object_pos_w,
                contact_object_quat_w,
                elastomer_quat_w,
                env_ids,
            )

        selected = self._friction_coefficient_by_env[env_ids].reshape(-1, 1)
        serialized_coefficient = self.cfg.friction_coefficient
        self.cfg.friction_coefficient = selected
        try:
            super()._compute_tactile_forces_from_sdf(
                points_contact_object_local,
                sdf_values,
                sdf_gradients,
                contact_object_pos_w,
                contact_object_quat_w,
                elastomer_quat_w,
                env_ids,
            )
        finally:
            self.cfg.friction_coefficient = serialized_coefficient
