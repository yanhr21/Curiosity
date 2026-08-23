# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Causal in-episode CarryBox mass/inertia changes for Plan 15."""

from __future__ import annotations

from dataclasses import dataclass

import torch


NOMINAL_CARRYBOX_MASS_KG = 0.3023375869
MASS_FACTORS = (1.0, 1.5, 3.0, 6.0, 10.0)


@dataclass(frozen=True)
class MassJumpConfig:
    nominal_mass_kg: float = NOMINAL_CARRYBOX_MASS_KG
    mass_factors: tuple[float, ...] = MASS_FACTORS
    minimum_lift_m: float = 0.05
    stable_lift_frames: int = 10
    delay_frames: tuple[int, int] = (10, 50)
    seed: int = 150814

    def __post_init__(self):
        if self.nominal_mass_kg <= 0.0:
            raise ValueError("nominal mass must be positive")
        if not self.mass_factors or any(value < 1.0 for value in self.mass_factors):
            raise ValueError("mass factors must be non-empty and at least one")
        if self.stable_lift_frames < 1:
            raise ValueError("stable lift frame count must be positive")
        low, high = self.delay_frames
        if low < 0 or high < low:
            raise ValueError("invalid post-qualification delay range")


class OnlineMassJumpController:
    """Stateful per-environment controller; diagnostics are never observations."""

    def __init__(self, env, asset_name: str, config: MassJumpConfig):
        self.env = env
        self.asset = env.scene[asset_name]
        self.asset_name = asset_name
        self.config = config
        self.num_envs = int(env.num_envs)
        self.device = torch.device(env.device)
        self.default_mass = self.asset.data.default_mass.detach().cpu().clone()
        self.default_inertia = self.asset.data.default_inertia.detach().cpu().clone()
        if self.default_mass.shape != (self.num_envs, 1):
            raise RuntimeError(
                "Plan-15 mass jump requires a one-body RigidObject, got default mass "
                f"shape {tuple(self.default_mass.shape)}"
            )
        if self.default_inertia.shape != (self.num_envs, 9):
            raise RuntimeError(
                f"unexpected rigid-object inertia shape {tuple(self.default_inertia.shape)}"
            )
        self.episode_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.initial_height_m = torch.zeros(self.num_envs, device=self.device)
        self.stable_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.qualified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.qualification_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.wait_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_factor = torch.ones(self.num_envs, device=self.device)
        self.pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.jump_applied = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.mass_changed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.mass_readback_kg = torch.full(
            (self.num_envs,), config.nominal_mass_kg, device=self.device
        )
        self.inertia_readback_kg_m2 = torch.zeros(
            (self.num_envs, 9), device=self.device
        )
        self.cumulative_jump_events = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.cumulative_mass_changes = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.cumulative_factor_events = torch.zeros(
            (self.num_envs, len(config.mass_factors)),
            dtype=torch.long,
            device=self.device,
        )

    def _ids(self, env_ids) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)

    def _deterministic_assignment(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low, high = self.config.delay_frames
        span = high - low + 1
        code = (
            int(self.config.seed)
            + 1103515245 * (self.episode_index[ids] + 1)
            + 12345 * (ids + 1)
        ) & 0x7FFFFFFF
        delay = low + torch.remainder(code, span)
        choice = torch.remainder(
            torch.div(code, span, rounding_mode="floor"),
            len(self.config.mass_factors),
        )
        factors = torch.as_tensor(
            self.config.mass_factors, dtype=torch.float32, device=self.device
        )[choice]
        return delay, factors

    def _write_mass(
        self, ids: torch.Tensor, factors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids_cpu = ids.detach().cpu().to(torch.long)
        factors_cpu = factors.detach().cpu().to(self.default_mass.dtype)
        masses = self.asset.root_physx_view.get_masses().clone()
        inertias = self.asset.root_physx_view.get_inertias().clone()
        target_mass = float(self.config.nominal_mass_kg) * factors_cpu
        masses[ids_cpu, 0] = target_mass
        default_mass = self.default_mass[ids_cpu, 0]
        inertia_scale = target_mass / default_mass
        inertias[ids_cpu] = (
            self.default_inertia[ids_cpu] * inertia_scale[:, None]
        )
        self.asset.root_physx_view.set_masses(masses, ids_cpu)
        self.asset.root_physx_view.set_inertias(inertias, ids_cpu)
        mass_readback = self.asset.root_physx_view.get_masses()[ids_cpu, 0]
        inertia_readback = self.asset.root_physx_view.get_inertias()[ids_cpu]
        target_inertia = inertias[ids_cpu]
        if not torch.allclose(mass_readback, target_mass, rtol=1.0e-6, atol=1.0e-7):
            raise RuntimeError(
                f"PhysX mass readback mismatch: target={target_mass.tolist()} "
                f"readback={mass_readback.tolist()}"
            )
        if not torch.allclose(
            inertia_readback, target_inertia, rtol=1.0e-6, atol=1.0e-8
        ):
            raise RuntimeError("PhysX inertia readback mismatch after mass jump")
        return mass_readback.to(self.device), inertia_readback.to(self.device)

    def reset(self, env_ids=None) -> None:
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        self.episode_index[ids] += 1
        self.initialized[ids] = False
        self.stable_count[ids] = 0
        self.qualified[ids] = False
        self.qualification_step[ids] = -1
        self.wait_frames[ids] = 0
        self.pending[ids] = False
        self.pending_step[ids] = -1
        self.jump_applied[ids] = False
        self.mass_changed[ids] = False
        self.jump_step[ids] = -1
        delay, factor = self._deterministic_assignment(ids)
        self.target_delay[ids] = delay
        self.target_factor[ids] = factor
        mass, inertia = self._write_mass(
            ids, torch.ones(ids.numel(), device=self.device)
        )
        self.mass_readback_kg[ids] = mass
        self.inertia_readback_kg_m2[ids] = inertia

    def advance(self, *, control_step: int) -> torch.Tensor:
        """Advance from object lift only; never read tactile to schedule a jump."""

        positions = self.asset.data.root_pos_w[:, 2]
        first = ~self.initialized
        if first.any():
            self.initial_height_m[first] = positions[first]
            self.initialized[first] = True
        handoff = getattr(self.env, "_online_teacher_handoff_controller", None)
        if handoff is None:
            lifted = (
                positions - self.initial_height_m
                >= float(self.config.minimum_lift_m)
            )
            self.stable_count = torch.where(
                lifted,
                self.stable_count + 1,
                torch.zeros_like(self.stable_count),
            )
            newly_qualified = (~self.qualified) & (
                self.stable_count >= int(self.config.stable_lift_frames)
            )
        else:
            newly_qualified = (~self.qualified) & handoff.handoff_active
        self.qualified |= newly_qualified
        self.qualification_step[newly_qualified] = int(control_step)
        waiting = (
            self.qualified
            & (~self.pending)
            & (~self.jump_applied)
        )
        self.wait_frames[waiting] += 1
        due = waiting & (self.wait_frames >= self.target_delay)
        ids = due.nonzero(as_tuple=False).flatten()
        if ids.numel() > 0:
            self.pending[ids] = True
            self.pending_step[ids] = int(control_step)
        return ids

    def apply_pending(self, *, control_step: int) -> torch.Tensor:
        """Apply a due jump after actor inference and before physics substeps."""

        ids = self.pending.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return ids
        changed_ids = ids[self.target_factor[ids] != 1.0]
        if changed_ids.numel() > 0:
            mass, inertia = self._write_mass(
                changed_ids, self.target_factor[changed_ids]
            )
            self.mass_readback_kg[changed_ids] = mass
            self.inertia_readback_kg_m2[changed_ids] = inertia
            self.mass_changed[changed_ids] = True
            self.cumulative_mass_changes[changed_ids] += 1
        factors = torch.as_tensor(
            self.config.mass_factors, dtype=torch.float32, device=self.device
        )
        factor_indices = torch.argmin(
            (self.target_factor[ids, None] - factors[None]).abs(), dim=1
        )
        self.cumulative_jump_events[ids] += 1
        self.cumulative_factor_events[ids, factor_indices] += 1
        self.pending[ids] = False
        self.jump_applied[ids] = True
        self.jump_step[ids] = int(control_step)
        return ids

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return {
            "episode_index": self.episode_index.clone(),
            "target_factor": self.target_factor.clone(),
            "target_delay_frames": self.target_delay.clone(),
            "qualified": self.qualified.clone(),
            "qualification_step": self.qualification_step.clone(),
            "pending": self.pending.clone(),
            "pending_step": self.pending_step.clone(),
            "jump_applied": self.jump_applied.clone(),
            "mass_changed": self.mass_changed.clone(),
            "jump_step": self.jump_step.clone(),
            "mass_readback_kg": self.mass_readback_kg.clone(),
            "inertia_readback_kg_m2": self.inertia_readback_kg_m2.clone(),
            "cumulative_jump_events": self.cumulative_jump_events.clone(),
            "cumulative_mass_changes": self.cumulative_mass_changes.clone(),
            "cumulative_factor_events": self.cumulative_factor_events.clone(),
        }


def _controller(env, asset_name: str, config: MassJumpConfig) -> OnlineMassJumpController:
    controller = getattr(env, "_online_mass_jump_controller", None)
    if controller is None:
        controller = OnlineMassJumpController(env, asset_name, config)
        setattr(env, "_online_mass_jump_controller", controller)
    elif controller.asset_name != asset_name or controller.config != config:
        raise RuntimeError("online mass-jump controller configuration changed in-place")
    return controller


def reset_online_mass_jump(
    env,
    env_ids,
    asset_name: str = "obj",
    nominal_mass_kg: float = NOMINAL_CARRYBOX_MASS_KG,
    mass_factors: tuple[float, ...] = MASS_FACTORS,
    minimum_lift_m: float = 0.05,
    stable_lift_frames: int = 10,
    delay_frames: tuple[int, int] = (10, 50),
    seed: int | None = None,
) -> None:
    resolved_seed = int(env.cfg.seed if seed is None else seed)
    config = MassJumpConfig(
        nominal_mass_kg=nominal_mass_kg,
        mass_factors=tuple(mass_factors),
        minimum_lift_m=minimum_lift_m,
        stable_lift_frames=stable_lift_frames,
        delay_frames=tuple(delay_frames),
        seed=resolved_seed,
    )
    _controller(env, asset_name, config).reset(env_ids)


def step_online_mass_jump(
    env,
    env_ids,
    asset_name: str = "obj",
    nominal_mass_kg: float = NOMINAL_CARRYBOX_MASS_KG,
    mass_factors: tuple[float, ...] = MASS_FACTORS,
    minimum_lift_m: float = 0.05,
    stable_lift_frames: int = 10,
    delay_frames: tuple[int, int] = (10, 50),
    seed: int | None = None,
) -> None:
    """Manager interval event: physics -> jump -> next live observation."""

    del env_ids
    resolved_seed = int(env.cfg.seed if seed is None else seed)
    config = MassJumpConfig(
        nominal_mass_kg=nominal_mass_kg,
        mass_factors=tuple(mass_factors),
        minimum_lift_m=minimum_lift_m,
        stable_lift_frames=stable_lift_frames,
        delay_frames=tuple(delay_frames),
        seed=resolved_seed,
    )
    controller = _controller(env, asset_name, config)
    controller.advance(control_step=int(env.common_step_counter))
    env._online_mass_jump_diagnostics = controller.diagnostics()


def post_handoff_box_lift_reward(
    env,
    asset_name: str = "obj",
    target_lift_m: float = 0.05,
) -> torch.Tensor:
    """Reward maintaining a physically lifted box after student handoff.

    This is an outcome reward, not an actor observation. It reads the same live
    object height used by the mass scheduler and is exactly zero throughout the
    frozen-teacher prefix. The previous Plan-15 configuration had only
    reference-tracking rewards, so a policy could reduce those losses without
    receiving a direct signal for keeping the box above its pickup height.
    """

    if target_lift_m <= 0.0:
        raise ValueError("target_lift_m must be positive")
    controller = getattr(env, "_online_mass_jump_controller", None)
    handoff = getattr(env, "_online_teacher_handoff_controller", None)
    if controller is None or handoff is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if controller.asset_name != asset_name:
        raise RuntimeError("mass controller and lift-reward asset do not match")
    lift_m = controller.asset.data.root_pos_w[:, 2] - controller.initial_height_m
    lift_fraction = torch.clamp(lift_m / float(target_lift_m), min=0.0, max=1.0)
    valid = controller.initialized & handoff.handoff_active
    return torch.where(valid, lift_fraction, torch.zeros_like(lift_fraction))
