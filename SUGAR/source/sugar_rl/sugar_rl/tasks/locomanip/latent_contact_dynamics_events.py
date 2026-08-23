# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Hidden, stratified contact dynamics for the gated tactile follow-up.

These event terms are not imported or registered by the active optimizer-clean
branch.  They preserve official SUGAR assets and manager APIs while making the
latent mass/friction/COM tuple auditable and matching the PhysX interface
friction to the per-environment official TacSL Coulomb coefficient.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from sugar_rl.tasks.locomanip.latent_contact_visuotactile_sensor import (
    PerEnvironmentFrictionVisuoTactileSensor,
)
from sugar_rl.tasks.locomanip.mdp.commands import MotionCommand


_OFFICIAL_R15_COMPLIANT_CONTACT_STIFFNESS = 10.0
_PLAN11_TRAIN_MASSES_KG = tuple(float(value) for value in range(1, 11))


def _stratified_unit_interval(num_envs: int, generator: torch.Generator) -> torch.Tensor:
    centers = (torch.arange(num_envs, dtype=torch.float64) + 0.5) / num_envs
    return centers[torch.randperm(num_envs, generator=generator)]


class apply_stratified_integer_mass_kg(ManagerTermBase):
    """Assign the frozen Plan-11 1--10 kg distribution without exposing mass.

    The term is intentionally object-only.  It changes rigid-body mass and
    scales inertia by the same factor, records exact PhysX readback for audit,
    and never creates an observation or reward field.  Exact equal allocation
    is enforced by requiring a number of environments divisible by ten.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        self.object: RigidObject = env.scene[self.object_cfg.name]
        if not isinstance(self.object, RigidObject):
            raise TypeError(f"Expected RigidObject, got {type(self.object)}")

        masses_kg = tuple(float(value) for value in cfg.params["masses_kg"])
        if masses_kg != _PLAN11_TRAIN_MASSES_KG:
            raise ValueError(
                "Plan-11 training masses are frozen to integer kilograms 1--10"
            )
        if env.num_envs % len(masses_kg) != 0:
            raise ValueError(
                "exact equal 1--10 kg allocation requires num_envs divisible by 10"
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(cfg.params["distribution_seed"]))
        repeats = env.num_envs // len(masses_kg)
        allocation = torch.tensor(masses_kg, dtype=torch.float32).repeat_interleave(repeats)
        self._target_mass_kg_cpu = allocation[
            torch.randperm(env.num_envs, generator=generator)
        ]
        counts = torch.bincount(self._target_mass_kg_cpu.to(torch.int64), minlength=11)[1:]
        if not bool((counts == repeats).all()):
            raise RuntimeError(f"Plan-11 equal-mass allocation failed: {counts.tolist()}")
        self.last_readback: dict[str, torch.Tensor] = {}

    @property
    def target_mass_kg(self) -> torch.Tensor:
        return self._target_mass_kg_cpu.clone()

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | Sequence[int] | None,
        object_cfg: SceneEntityCfg,
        masses_kg: tuple[float, ...],
        distribution_seed: int,
    ) -> None:
        del object_cfg, masses_kg, distribution_seed
        if env_ids is None:
            env_ids_cpu = torch.arange(env.num_envs, dtype=torch.long, device="cpu")
        else:
            env_ids_cpu = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")

        masses = self.object.root_physx_view.get_masses()
        inertias = self.object.root_physx_view.get_inertias()
        default_mass = self.object.data.default_mass.to(device=masses.device)
        default_inertia = self.object.data.default_inertia.to(device=inertias.device)
        default_total_mass = default_mass.sum(dim=-1)
        if not bool((default_total_mass > 0.0).all()):
            raise RuntimeError("CarryBox default total mass must be positive")
        target_total = self._target_mass_kg_cpu[env_ids_cpu].to(device=masses.device)
        scale = target_total / default_total_mass[env_ids_cpu]
        masses[env_ids_cpu] = default_mass[env_ids_cpu] * scale.unsqueeze(-1)
        inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * scale.unsqueeze(-1)
        self.object.root_physx_view.set_masses(masses, env_ids_cpu)
        self.object.root_physx_view.set_inertias(inertias, env_ids_cpu)

        mass_after = self.object.root_physx_view.get_masses()[env_ids_cpu]
        inertia_after = self.object.root_physx_view.get_inertias()[env_ids_cpu]
        actual_total = mass_after.sum(dim=-1)
        expected_inertia = default_inertia[env_ids_cpu] * scale.unsqueeze(-1)
        if not torch.allclose(actual_total, target_total, rtol=2.0e-6, atol=1.0e-6):
            raise RuntimeError(
                "Plan-11 integer-mass PhysX readback mismatch: "
                f"max_abs={float(torch.abs(actual_total - target_total).max())}"
            )
        if not torch.allclose(
            inertia_after, expected_inertia, rtol=2.0e-6, atol=5.0e-8
        ):
            raise RuntimeError(
                "Plan-11 mass-scaled inertia readback mismatch: "
                f"max_abs={float(torch.abs(inertia_after - expected_inertia).max())}"
            )

        self.last_readback = {
            "env_ids": env_ids_cpu.clone(),
            "target_mass_kg": target_total.detach().cpu().clone(),
            "actual_mass_kg": actual_total.detach().cpu().clone(),
            "mass_scale": scale.detach().cpu().clone(),
            "actual_inertia": inertia_after.detach().cpu().clone(),
        }


class apply_stratified_latent_contact_dynamics(ManagerTermBase):
    """Apply one fixed, matched latent dynamics tuple per parallel environment."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        self.robot_cfg: SceneEntityCfg = cfg.params["robot_cfg"]
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.robot: Articulation = env.scene[self.robot_cfg.name]
        if not isinstance(self.object, RigidObject):
            raise TypeError(f"Expected RigidObject, got {type(self.object)}")
        if not isinstance(self.robot, Articulation):
            raise TypeError(f"Expected Articulation, got {type(self.robot)}")

        self.sensors: tuple[PerEnvironmentFrictionVisuoTactileSensor, ...] = tuple(
            env.scene[name] for name in cfg.params["sensor_names"]
        )
        if len(self.sensors) != 2 or not all(
            isinstance(sensor, PerEnvironmentFrictionVisuoTactileSensor) for sensor in self.sensors
        ):
            raise TypeError("Latent dynamics require exactly two per-environment-friction TacSL sensors")

        if self.robot_cfg.body_ids == slice(None):
            raise ValueError("robot_cfg must name only the declared palm-interface bodies")
        (
            self._robot_shape_ids,
            self._robot_expected_material_third_field,
        ) = self._resolve_robot_shape_ids(self.robot_cfg.body_ids)
        self._nominal_object_com = self.object.root_physx_view.get_coms().clone()
        if self._nominal_object_com.ndim != 2 or self._nominal_object_com.shape[-1] != 7:
            raise RuntimeError(
                f"Unexpected RigidObject COM layout: {tuple(self._nominal_object_com.shape)}"
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(cfg.params["distribution_seed"]))
        num_envs = env.num_envs
        mass_u = _stratified_unit_interval(num_envs, generator)
        dynamic_u = _stratified_unit_interval(num_envs, generator)
        static_gap_u = _stratified_unit_interval(num_envs, generator)
        com_u = _stratified_unit_interval(num_envs, generator)
        pulse_magnitude_u = _stratified_unit_interval(num_envs, generator)
        pulse_direction_u = _stratified_unit_interval(num_envs, generator)

        mass_low, mass_high = cfg.params["mass_scale_range"]
        static_low, static_high = cfg.params["static_friction_range"]
        dynamic_low, dynamic_high = cfg.params["dynamic_friction_range"]
        com_low, com_high = cfg.params["com_y_range_m"]
        pulse_low, pulse_high = cfg.params["pulse_magnitude_range_mps"]
        if not 0.0 < mass_low <= mass_high:
            raise ValueError("mass_scale_range must be positive and ordered")
        if not 0.0 <= static_low <= static_high:
            raise ValueError("static_friction_range must be non-negative and ordered")
        if not 0.0 <= dynamic_low <= dynamic_high <= static_high:
            raise ValueError("dynamic friction must fit below the configured static upper bound")
        if not com_low <= com_high:
            raise ValueError("com_y_range_m must be ordered")
        if not 0.0 <= pulse_low <= pulse_high:
            raise ValueError("pulse_magnitude_range_mps must be non-negative and ordered")

        mass_scale = torch.exp(math.log(mass_low) + mass_u * math.log(mass_high / mass_low))
        dynamic_friction = dynamic_low + dynamic_u * (dynamic_high - dynamic_low)
        static_friction = torch.clamp(
            dynamic_friction + static_gap_u * (static_high - dynamic_friction),
            min=static_low,
        )
        com_y_m = com_low + com_u * (com_high - com_low)
        pulse_magnitude = pulse_low + pulse_magnitude_u * (pulse_high - pulse_low)
        pulse_angle = pulse_direction_u * (2.0 * math.pi)
        pulse_delta_velocity_w = torch.stack(
            (
                pulse_magnitude * torch.cos(pulse_angle),
                pulse_magnitude * torch.sin(pulse_angle),
                torch.zeros_like(pulse_magnitude),
            ),
            dim=-1,
        )

        if not torch.all(dynamic_friction <= static_friction):
            raise RuntimeError("Generated dynamic friction exceeds static friction")
        self._tuple_cpu = {
            "mass_scale": mass_scale.float(),
            "static_friction": static_friction.float(),
            "dynamic_friction": dynamic_friction.float(),
            "com_y_m": com_y_m.float(),
            "pulse_delta_velocity_w_mps": pulse_delta_velocity_w.float(),
        }
        self.last_readback: dict[str, torch.Tensor] = {}

    def _resolve_robot_shape_ids(
        self, body_ids: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shapes_per_body: list[int] = []
        for link_path in self.robot.root_physx_view.link_paths[0]:
            view = self.robot._physics_sim_view.create_rigid_body_view(link_path)
            shapes_per_body.append(view.max_shapes)
        if len(shapes_per_body) != self.robot.num_bodies:
            raise RuntimeError(
                "Robot link/material layout mismatch: "
                f"{len(shapes_per_body)} links versus {self.robot.num_bodies} bodies"
            )
        if sum(shapes_per_body) != self.robot.root_physx_view.max_shapes:
            raise RuntimeError("Robot shape counts do not match the root PhysX material view")

        shape_ids: list[int] = []
        expected_material_third_field: list[float] = []
        for body_id in body_ids:
            body_id = int(body_id)
            start = sum(shapes_per_body[:body_id])
            shape_count = shapes_per_body[body_id]
            shape_ids.extend(range(start, start + shape_count))
            body_name = self.robot.body_names[body_id]
            # PhysX TensorAPI exposes compliant-contact stiffness as the
            # negative third material field.  Ordinary rigid materials expose
            # restitution there.  The authored USD restitution remains zero.
            encoded_third_field = (
                -_OFFICIAL_R15_COMPLIANT_CONTACT_STIFFNESS
                if body_name.endswith("_tacsl_r15_elastomer")
                else 0.0
            )
            expected_material_third_field.extend(
                [encoded_third_field] * shape_count
            )
        if not shape_ids:
            raise RuntimeError("The selected palm-interface bodies have no collision shapes")
        return (
            torch.tensor(shape_ids, dtype=torch.long, device="cpu"),
            torch.tensor(
                expected_material_third_field,
                dtype=torch.float32,
                device="cpu",
            ),
        )

    def tuple_for_device(self, device: str | torch.device) -> dict[str, torch.Tensor]:
        """Return cloned tuple tensors for diagnostics or the pulse event."""

        return {name: value.to(device=device).clone() for name, value in self._tuple_cpu.items()}

    @property
    def nominal_object_com(self) -> torch.Tensor:
        """Return the captured pre-randomization RigidObject COM pose."""

        return self._nominal_object_com.clone()

    @property
    def robot_shape_ids(self) -> torch.Tensor:
        """Return the exact articulation-shape indices owned by this event."""

        return self._robot_shape_ids.clone()

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | Sequence[int] | None,
        object_cfg: SceneEntityCfg,
        robot_cfg: SceneEntityCfg,
        sensor_names: tuple[str, str],
        distribution_seed: int,
        mass_scale_range: tuple[float, float],
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        com_y_range_m: tuple[float, float],
        pulse_magnitude_range_mps: tuple[float, float],
    ) -> None:
        del object_cfg, robot_cfg, sensor_names, distribution_seed
        del mass_scale_range, static_friction_range, dynamic_friction_range
        del com_y_range_m, pulse_magnitude_range_mps

        if env_ids is None:
            env_ids_cpu = torch.arange(env.num_envs, dtype=torch.long, device="cpu")
        else:
            env_ids_cpu = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
        values = {name: tensor[env_ids_cpu] for name, tensor in self._tuple_cpu.items()}

        object_materials = self.object.root_physx_view.get_material_properties()
        object_materials[env_ids_cpu, :, 0] = values["static_friction"].unsqueeze(-1)
        object_materials[env_ids_cpu, :, 1] = values["dynamic_friction"].unsqueeze(-1)
        self.object.root_physx_view.set_material_properties(object_materials, env_ids_cpu)

        robot_materials = self.robot.root_physx_view.get_material_properties()
        material_env_ids = env_ids_cpu[:, None]
        robot_materials[material_env_ids, self._robot_shape_ids, 0] = values["static_friction"].unsqueeze(-1)
        robot_materials[material_env_ids, self._robot_shape_ids, 1] = values["dynamic_friction"].unsqueeze(-1)
        self.robot.root_physx_view.set_material_properties(robot_materials, env_ids_cpu)

        for sensor in self.sensors:
            sensor.set_friction_coefficient_by_env(values["dynamic_friction"], env_ids_cpu)

        masses = self.object.root_physx_view.get_masses()
        default_mass = self.object.data.default_mass.to(device=masses.device)
        mass_scale = values["mass_scale"].to(device=masses.device).unsqueeze(-1)
        masses[env_ids_cpu] = default_mass[env_ids_cpu] * mass_scale
        self.object.root_physx_view.set_masses(masses, env_ids_cpu)

        inertias = self.object.root_physx_view.get_inertias()
        default_inertia = self.object.data.default_inertia.to(device=inertias.device)
        inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * mass_scale
        self.object.root_physx_view.set_inertias(inertias, env_ids_cpu)

        coms = self.object.root_physx_view.get_coms()
        coms[env_ids_cpu] = self._nominal_object_com[env_ids_cpu]
        coms[env_ids_cpu, 1] += values["com_y_m"].to(device=coms.device)
        self.object.root_physx_view.set_coms(coms, env_ids_cpu)

        object_after = self.object.root_physx_view.get_material_properties()
        robot_after = self.robot.root_physx_view.get_material_properties()
        mass_after = self.object.root_physx_view.get_masses()
        inertia_after = self.object.root_physx_view.get_inertias()
        com_after = self.object.root_physx_view.get_coms()
        expected_static = values["static_friction"].to(device=object_after.device)
        expected_dynamic = values["dynamic_friction"].to(device=object_after.device)
        if not torch.allclose(object_after[env_ids_cpu, :, 0], expected_static[:, None], rtol=0.0, atol=1.0e-7):
            raise RuntimeError("CarryBox static-friction readback mismatch")
        if not torch.allclose(object_after[env_ids_cpu, :, 1], expected_dynamic[:, None], rtol=0.0, atol=1.0e-7):
            raise RuntimeError("CarryBox dynamic-friction readback mismatch")
        if not torch.allclose(
            object_after[env_ids_cpu, :, 2],
            torch.zeros_like(object_after[env_ids_cpu, :, 2]),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise RuntimeError("CarryBox restitution readback mismatch")
        if not torch.allclose(
            robot_after[material_env_ids, self._robot_shape_ids, 0], expected_static[:, None], rtol=0.0, atol=1.0e-7
        ):
            raise RuntimeError("Palm-interface static-friction readback mismatch")
        if not torch.allclose(
            robot_after[material_env_ids, self._robot_shape_ids, 1], expected_dynamic[:, None], rtol=0.0, atol=1.0e-7
        ):
            raise RuntimeError("Palm-interface dynamic-friction readback mismatch")
        actual_material_third_field = robot_after[
            material_env_ids, self._robot_shape_ids, 2
        ]
        expected_material_third_field = (
            self._robot_expected_material_third_field.to(
                device=actual_material_third_field.device
            )
            .unsqueeze(0)
            .expand_as(actual_material_third_field)
        )
        if not torch.allclose(
            actual_material_third_field,
            expected_material_third_field,
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise RuntimeError(
                "Palm-interface PhysX material third-field readback mismatch: "
                f"shape_ids={self._robot_shape_ids.tolist()}, "
                f"expected={expected_material_third_field.detach().cpu().tolist()}, "
                f"actual={actual_material_third_field.detach().cpu().tolist()}"
            )
        expected_mass = default_mass[env_ids_cpu] * mass_scale
        if not torch.allclose(mass_after[env_ids_cpu], expected_mass, rtol=0.0, atol=1.0e-7):
            raise RuntimeError("CarryBox mass readback mismatch")
        expected_inertia = default_inertia[env_ids_cpu] * mass_scale
        if not torch.allclose(
            inertia_after[env_ids_cpu], expected_inertia, rtol=0.0, atol=1.0e-7
        ):
            raise RuntimeError("CarryBox inertia readback mismatch")
        expected_com_y = self._nominal_object_com[env_ids_cpu, 1] + values["com_y_m"].to(com_after.device)
        if not torch.allclose(com_after[env_ids_cpu, 1], expected_com_y, rtol=0.0, atol=1.0e-7):
            raise RuntimeError("CarryBox COM readback mismatch")
        for sensor in self.sensors:
            sensor_values = sensor.friction_coefficient_by_env[env_ids_cpu.to(sensor.device)]
            if not torch.allclose(
                sensor_values,
                values["dynamic_friction"].to(sensor.device),
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise RuntimeError("TacSL/PhysX dynamic-friction readback mismatch")

        self.last_readback = {
            "env_ids": env_ids_cpu.clone(),
            **{name: value.clone() for name, value in values.items()},
        }


class apply_reference_contact_phase_lateral_pulse(ManagerTermBase):
    """Apply at most one hidden world-XY velocity pulse per episode.

    Eligibility uses the official human-video reference contact schedule,
    ``MotionCommand.contact_label``.  It never uses thresholded measured hand
    forces, and neither the schedule nor sampled pulse is added to actor input.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject = env.scene[self.asset_cfg.name]
        self._pulse_applied = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._pulse_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self._last_delta_velocity_w = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)

    @property
    def pulse_audit_state(self) -> dict[str, torch.Tensor]:
        return {
            "pulse_applied": self._pulse_applied.clone(),
            "pulse_count": self._pulse_count.clone(),
            "last_delta_velocity_w_mps": self._last_delta_velocity_w.clone(),
        }

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        elif not isinstance(env_ids, slice):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._pulse_applied.device)
        self._pulse_applied[env_ids] = False
        self._last_delta_velocity_w[env_ids] = 0.0

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | Sequence[int] | None,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        dynamics_term_name: str,
    ) -> None:
        del asset_cfg
        if env_ids is None:
            env_ids_device = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        else:
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)

        command: MotionCommand = env.command_manager.get_term(command_name)
        eligible = command.contact_label[env_ids_device] & ~self._pulse_applied[env_ids_device]
        if not eligible.any():
            return
        pushed_env_ids = env_ids_device[eligible]

        dynamics_term = env.event_manager.get_term_cfg(dynamics_term_name).func
        if not isinstance(dynamics_term, apply_stratified_latent_contact_dynamics):
            raise TypeError(f"Event {dynamics_term_name!r} is not the latent dynamics term")
        pulse_delta = dynamics_term.tuple_for_device(self.asset.device)["pulse_delta_velocity_w_mps"]

        velocity_w = self.asset.data.root_vel_w[pushed_env_ids].clone()
        velocity_w[:, :3] += pulse_delta[pushed_env_ids]
        self.asset.write_root_velocity_to_sim(velocity_w, env_ids=pushed_env_ids)
        self._pulse_applied[pushed_env_ids] = True
        self._pulse_count[pushed_env_ids] += 1
        self._last_delta_velocity_w[pushed_env_ids] = pulse_delta[pushed_env_ids]
