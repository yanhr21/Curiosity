# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic robustness stresses on direct TacSL pressure/shear fields.

This module never synthesizes tactile contact and never replaces TacSL. It
applies the frozen Stage-E v16 noise, latency, and dead-taxel transformations
to the direct spatial fields at their shared history boundary. Consequently,
the policy, original ICM, tactile slip detector, and failed-attempt memory all
consume the same transformed time series.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


H2_TACTILE_STRESS_ROLES = (
    "nominal",
    "noise",
    "latency_1step",
    "dead_taxels",
    "combined",
)
H2_STRESS_SEED = 20260723
H2_NORMAL_NOISE_STD_N = 1.0e-7
H2_SHEAR_NOISE_STD_N = 2.0e-7
H2_DEAD_TAXEL_FRACTION = 0.05
# Historical H2 unit-audit cardinality. Runtime capacity studies may use any
# positive multiple of the five fixed roles; the transforms and seed do not
# change with batch size.
H2_ENVS_PER_ROLE = 4
H2_NUM_ENVS = len(H2_TACTILE_STRESS_ROLES) * H2_ENVS_PER_ROLE


@dataclass(frozen=True)
class DirectTactileStressCfg:
    """Frozen online transformation contract for Stage-H H2."""

    seed: int = H2_STRESS_SEED
    normal_noise_std_n: float = H2_NORMAL_NOISE_STD_N
    shear_noise_std_n: float = H2_SHEAR_NOISE_STD_N
    dead_taxel_fraction: float = H2_DEAD_TAXEL_FRACTION
    taxel_area_m2: float = 1.18138624e-6
    stress_scale: float = 1.0e-5
    grid_shape: tuple[int, int] = (20, 25)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("tactile stress seed must be non-negative")
        if self.normal_noise_std_n <= 0.0 or self.shear_noise_std_n <= 0.0:
            raise ValueError("tactile stress noise scales must be positive")
        if not 0.0 < self.dead_taxel_fraction < 1.0:
            raise ValueError("dead-taxel fraction must be in (0,1)")
        if self.taxel_area_m2 <= 0.0 or self.stress_scale <= 0.0:
            raise ValueError("taxel area and numerical scale must be positive")
        if tuple(self.grid_shape) != (20, 25):
            raise ValueError("H2 is locked to the official R15 20x25 grid")


class DirectTactileStressRuntime:
    """Apply the five fixed roles once per simulator control step."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: DirectTactileStressCfg,
    ) -> None:
        num_envs = int(num_envs)
        num_roles = len(H2_TACTILE_STRESS_ROLES)
        if num_envs < num_roles or num_envs % num_roles != 0:
            raise ValueError(
                "H2 requires a positive equal-size assignment across its "
                f"{num_roles} roles, got {num_envs} environments"
            )
        self.num_envs = num_envs
        self.envs_per_role = num_envs // num_roles
        self.device = torch.device(device)
        self.cfg = cfg
        self.role_names_by_env = tuple(
            role
            for role in H2_TACTILE_STRESS_ROLES
            for _ in range(self.envs_per_role)
        )
        self.role_ids = torch.arange(
            num_roles,
            device=self.device,
            dtype=torch.long,
        ).repeat_interleave(self.envs_per_role)
        self._role_masks = {
            role: self.role_ids == index
            for index, role in enumerate(H2_TACTILE_STRESS_ROLES)
        }
        self._noise_mask = (
            self._role_masks["noise"] | self._role_masks["combined"]
        )
        self._dead_mask_env = (
            self._role_masks["dead_taxels"]
            | self._role_masks["combined"]
        )
        self._latency_mask = (
            self._role_masks["latency_1step"]
            | self._role_masks["combined"]
        )
        rows, cols = cfg.grid_shape
        taxels = rows * cols
        dead_count = int(round(taxels * cfg.dead_taxel_fraction))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(cfg.seed)
        dead_mask_cpu = torch.zeros((2, rows, cols), dtype=torch.bool)
        self.dead_taxel_indices: dict[str, list[int]] = {}
        for hand, name in enumerate(("left", "right")):
            indices = torch.sort(
                torch.randperm(taxels, generator=generator)[:dead_count]
            ).values
            dead_mask_cpu[hand].flatten()[indices] = True
            self.dead_taxel_indices[name] = [
                int(index) for index in indices.tolist()
            ]
        self.dead_mask = dead_mask_cpu.to(self.device)
        self.normal_noise_std_scaled = (
            cfg.normal_noise_std_n / cfg.taxel_area_m2 * cfg.stress_scale
        )
        self.shear_noise_std_scaled = (
            cfg.shear_noise_std_n / cfg.taxel_area_m2 * cfg.stress_scale
        )

        self.last_step: int | None = None
        self.first_step: int | None = None
        self.cached_output: torch.Tensor | None = None
        self.previous_pre_latency: torch.Tensor | None = None
        self.apply_calls = 0
        self.cache_hits = 0
        self.generated_steps = 0
        self.nominal_raw_output_max_abs = 0.0
        self.dead_output_abs_max = 0.0
        self.latency_reference_max_abs = 0.0
        self.reset_latency_reference_max_abs = 0.0
        self.output_nonzero_by_role = {
            role: 0 for role in H2_TACTILE_STRESS_ROLES
        }
        self.raw_nonzero_values_by_role = {
            role: 0 for role in H2_TACTILE_STRESS_ROLES
        }
        self.raw_nonzero_steps_by_role = {
            role: 0 for role in H2_TACTILE_STRESS_ROLES
        }
        self.initial_raw_nonzero_by_role: dict[str, int] | None = None
        self.noise_normal_sum = 0.0
        self.noise_normal_square_sum = 0.0
        self.noise_normal_count = 0
        self.noise_shear_sum = 0.0
        self.noise_shear_square_sum = 0.0
        self.noise_shear_count = 0
        # A single float32 sample is not a collision-safe freshness witness
        # over a formal run with more than ten thousand generated frames.
        # Preserve a compact eight-value exact signature and the integer RNG
        # seed for every generated frame instead.  This changes audit
        # bookkeeping only; the generated stress tensor is unchanged.
        self.noise_step_signatures: list[tuple[float, ...]] = []
        self.noise_step_seeds: list[int] = []
        self.reset_latency_samples = 0
        self.nonreset_latency_samples = 0

    @property
    def cache_key(self) -> tuple[Any, ...]:
        return (
            "h2r1_five_role_v1",
            self.cfg.seed,
            self.cfg.normal_noise_std_n,
            self.cfg.shear_noise_std_n,
            self.cfg.dead_taxel_fraction,
            self.cfg.taxel_area_m2,
            self.cfg.stress_scale,
            tuple(self.cfg.grid_shape),
            self.role_names_by_env,
        )

    def _record_noise(
        self,
        normal_noise: torch.Tensor,
        shear_noise: torch.Tensor,
    ) -> None:
        selected_normal = normal_noise[self._noise_mask]
        selected_shear = shear_noise[self._noise_mask]
        self.noise_normal_sum += float(selected_normal.double().sum())
        self.noise_normal_square_sum += float(
            torch.square(selected_normal.double()).sum()
        )
        self.noise_normal_count += selected_normal.numel()
        self.noise_shear_sum += float(selected_shear.double().sum())
        self.noise_shear_square_sum += float(
            torch.square(selected_shear.double()).sum()
        )
        self.noise_shear_count += selected_shear.numel()
        signature = torch.cat(
            (
                selected_normal.flatten()[:4],
                selected_shear.flatten()[:4],
            )
        )
        self.noise_step_signatures.append(
            tuple(float(value) for value in signature.detach().cpu().tolist())
        )

    @torch.no_grad()
    def apply(
        self,
        current: torch.Tensor,
        step: int,
        reset_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one deterministic role-transformed direct tactile frame."""

        self.apply_calls += 1
        expected = (
            self.num_envs,
            2,
            3,
            *self.cfg.grid_shape,
        )
        if tuple(current.shape) != expected:
            raise ValueError(
                f"direct tactile stress shape drift: {tuple(current.shape)}"
            )
        if current.device != self.device:
            raise ValueError("direct tactile stress device drift")
        if reset_mask.shape != (self.num_envs,) or reset_mask.dtype != torch.bool:
            raise ValueError("direct tactile stress reset mask shape/dtype drift")
        if not torch.isfinite(current).all():
            raise RuntimeError("direct tactile stress received non-finite TacSL")
        step = int(step)
        if self.last_step == step:
            if self.cached_output is None:
                raise RuntimeError("direct tactile stress cache is empty")
            self.cache_hits += 1
            return self.cached_output
        if self.last_step is not None and step != self.last_step + 1:
            raise RuntimeError(
                "direct tactile stress must advance exactly once per control "
                f"step: previous={self.last_step}, current={step}"
            )
        if self.first_step is None:
            self.first_step = step

        raw = current.detach()
        raw_nonzero_this_step = {
            role: int(torch.count_nonzero(raw[mask]))
            for role, mask in self._role_masks.items()
        }
        if self.generated_steps == 0:
            self.initial_raw_nonzero_by_role = dict(raw_nonzero_this_step)
        for role, count in raw_nonzero_this_step.items():
            self.raw_nonzero_values_by_role[role] += count
            self.raw_nonzero_steps_by_role[role] += int(count > 0)
        pre_latency = raw.clone()
        generator = torch.Generator(device=self.device)
        noise_seed = self.cfg.seed + step
        generator.manual_seed(noise_seed)
        self.noise_step_seeds.append(noise_seed)
        normal_noise = torch.randn(
            (self.num_envs, 2, *self.cfg.grid_shape),
            generator=generator,
            device=self.device,
            dtype=current.dtype,
        ) * self.normal_noise_std_scaled
        shear_noise = torch.randn(
            (self.num_envs, 2, 2, *self.cfg.grid_shape),
            generator=generator,
            device=self.device,
            dtype=current.dtype,
        ) * self.shear_noise_std_scaled
        self._record_noise(normal_noise, shear_noise)
        pre_latency[self._noise_mask, :, 0] = torch.clamp_min(
            pre_latency[self._noise_mask, :, 0]
            + normal_noise[self._noise_mask],
            0.0,
        )
        pre_latency[self._noise_mask, :, 1:3] += shear_noise[
            self._noise_mask
        ]

        live_dead_mask = self.dead_mask[None, :, None]
        dead_values = pre_latency[self._dead_mask_env]
        dead_values = dead_values.masked_fill(live_dead_mask, 0.0)
        pre_latency[self._dead_mask_env] = dead_values

        output = pre_latency.clone()
        if self.previous_pre_latency is not None:
            expected_latency = self.previous_pre_latency[self._latency_mask]
            output[self._latency_mask] = expected_latency
            nonreset_latency = self._latency_mask & ~reset_mask
            if nonreset_latency.any():
                error = torch.abs(
                    output[nonreset_latency]
                    - self.previous_pre_latency[nonreset_latency]
                ).max()
                self.latency_reference_max_abs = max(
                    self.latency_reference_max_abs, float(error)
                )
                self.nonreset_latency_samples += int(
                    nonreset_latency.sum()
                )
        reset_latency = self._latency_mask & reset_mask
        if reset_latency.any():
            output[reset_latency] = pre_latency[reset_latency]
            error = torch.abs(
                output[reset_latency] - pre_latency[reset_latency]
            ).max()
            self.reset_latency_reference_max_abs = max(
                self.reset_latency_reference_max_abs, float(error)
            )
            self.reset_latency_samples += int(reset_latency.sum())

        nominal_error = torch.abs(
            output[self._role_masks["nominal"]]
            - raw[self._role_masks["nominal"]]
        ).max()
        self.nominal_raw_output_max_abs = max(
            self.nominal_raw_output_max_abs, float(nominal_error)
        )
        dead_output = output[self._dead_mask_env].masked_select(
            live_dead_mask.expand_as(output[self._dead_mask_env])
        )
        if dead_output.numel() > 0:
            self.dead_output_abs_max = max(
                self.dead_output_abs_max,
                float(dead_output.abs().max()),
            )
        for role, mask in self._role_masks.items():
            self.output_nonzero_by_role[role] += int(
                torch.count_nonzero(output[mask])
            )
        if not torch.isfinite(output).all():
            raise RuntimeError("direct tactile stress produced non-finite values")

        self.previous_pre_latency = pre_latency.clone()
        self.cached_output = output
        self.last_step = step
        self.generated_steps += 1
        return output

    @torch.no_grad()
    def bootstrap_history(
        self,
        history: torch.Tensor,
        *,
        current_step: int,
    ) -> torch.Tensor:
        """Transform a real causal prehistory before the first live query.

        Mid-trajectory state restoration is not an episode-start reset.  The
        actor, ICM, and slip detector therefore need the recorded TacSL frames
        preceding the restored state, including the one-step latency role.
        This method advances the unchanged frozen stress transform over that
        real history and leaves the runtime cached at ``current_step`` so the
        first live observation reuses the exact transformed latest frame.
        """

        expected_tail = (self.num_envs, 2, 3, *self.cfg.grid_shape)
        if history.ndim != 6 or tuple(history.shape[0:1] + history.shape[2:]) != (
            expected_tail
        ):
            raise ValueError(
                "direct tactile bootstrap history must be "
                "[env,history,hand,channel,row,col], got "
                f"{tuple(history.shape)}"
            )
        history_steps = int(history.shape[1])
        if history_steps < 2:
            raise ValueError(
                "direct tactile bootstrap requires at least two real frames"
            )
        if self.last_step is not None or self.generated_steps != 0:
            raise RuntimeError(
                "direct tactile bootstrap must precede every live stress query"
            )
        if not torch.isfinite(history).all():
            raise RuntimeError(
                "direct tactile bootstrap received non-finite fields"
            )
        if bool(torch.any(history[:, :, :, 0] < 0.0)):
            raise RuntimeError(
                "direct tactile bootstrap received negative normal pressure"
            )
        current_step = int(current_step)
        first_step = current_step - history_steps + 1
        no_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        outputs = []
        for offset in range(history_steps):
            outputs.append(
                self.apply(
                    history[:, offset],
                    step=first_step + offset,
                    reset_mask=no_reset,
                )
            )
        if self.last_step != current_step:
            raise RuntimeError(
                "direct tactile bootstrap did not end at the live step"
            )
        return torch.stack(outputs, dim=1)

    @staticmethod
    def _moments(total: float, square_total: float, count: int) -> dict[str, float]:
        if count <= 0:
            return {"mean": 0.0, "std": 0.0}
        mean = total / count
        variance = max(square_total / count - mean * mean, 0.0)
        return {"mean": mean, "std": math.sqrt(variance)}

    def audit_state(self) -> dict[str, Any]:
        normal = self._moments(
            self.noise_normal_sum,
            self.noise_normal_square_sum,
            self.noise_normal_count,
        )
        shear = self._moments(
            self.noise_shear_sum,
            self.noise_shear_square_sum,
            self.noise_shear_count,
        )
        signature_count = len(set(self.noise_step_signatures))
        seed_count = len(set(self.noise_step_seeds))
        expected_first_step = (
            0 if self.first_step is None else int(self.first_step)
        )
        expected_seeds = [
            self.cfg.seed + expected_first_step + step
            for step in range(self.generated_steps)
        ]
        checks = {
            "five_roles_exact": (
                tuple(self.role_names_by_env)
                == tuple(
                    role
                    for role in H2_TACTILE_STRESS_ROLES
                    for _ in range(self.envs_per_role)
                )
            ),
            "equal_nonempty_envs_per_role": all(
                int(mask.sum()) == self.envs_per_role
                and self.envs_per_role > 0
                for mask in self._role_masks.values()
            ),
            "twenty_five_dead_taxels_per_palm": all(
                len(indices) == 25
                for indices in self.dead_taxel_indices.values()
            ),
            "nominal_is_bitwise_raw": self.nominal_raw_output_max_abs == 0.0,
            "dead_taxels_are_exact_zero": self.dead_output_abs_max == 0.0,
            "one_step_latency_exact": self.latency_reference_max_abs == 0.0,
            "reset_latency_has_no_cross_episode_state": (
                self.reset_latency_reference_max_abs == 0.0
            ),
            "fresh_noise_each_generated_step": (
                self.generated_steps <= 1
                or (
                    signature_count == self.generated_steps
                    and seed_count == self.generated_steps
                    and self.noise_step_seeds == expected_seeds
                )
            ),
            "normal_noise_scale_within_five_percent": (
                self.noise_normal_count > 0
                and abs(normal["std"] / self.normal_noise_std_scaled - 1.0)
                <= 0.05
            ),
            "shear_noise_scale_within_five_percent": (
                self.noise_shear_count > 0
                and abs(shear["std"] / self.shear_noise_std_scaled - 1.0)
                <= 0.05
            ),
            "all_roles_have_finite_nonzero_output": all(
                count > 0 for count in self.output_nonzero_by_role.values()
            ),
            "all_roles_have_raw_tacsl_provenance": all(
                self.raw_nonzero_values_by_role[role] > 0
                and self.raw_nonzero_steps_by_role[role] > 0
                for role in H2_TACTILE_STRESS_ROLES
            ),
            "shared_boundary_cache_is_used": self.cache_hits > 0,
            "cache_accounting_exact": (
                self.apply_calls == self.generated_steps + self.cache_hits
            ),
        }
        return {
            "protocol": "direct_tacsl_h2r1_five_role_stress_runtime_v1",
            "transform_order": ["fresh_noise", "dead_taxels", "latency_1step"],
            "seed_rule": "torch_generator_seed = 20260723 + common_step_counter",
            "seed": self.cfg.seed,
            "num_envs": self.num_envs,
            "envs_per_role": self.envs_per_role,
            "role_names_by_env": list(self.role_names_by_env),
            "role_counts": {
                role: int(mask.sum())
                for role, mask in self._role_masks.items()
            },
            "normal_noise_std_n_per_taxel": self.cfg.normal_noise_std_n,
            "shear_noise_std_n_per_axis_per_taxel": (
                self.cfg.shear_noise_std_n
            ),
            "normal_noise_std_scaled": self.normal_noise_std_scaled,
            "shear_noise_std_scaled": self.shear_noise_std_scaled,
            "dead_taxel_fraction_per_palm": self.cfg.dead_taxel_fraction,
            "dead_taxel_indices": self.dead_taxel_indices,
            "apply_calls": self.apply_calls,
            "cache_hits": self.cache_hits,
            "generated_steps": self.generated_steps,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "nominal_raw_output_max_abs": self.nominal_raw_output_max_abs,
            "dead_output_abs_max": self.dead_output_abs_max,
            "latency_reference_max_abs": self.latency_reference_max_abs,
            "reset_latency_reference_max_abs": (
                self.reset_latency_reference_max_abs
            ),
            "reset_latency_samples": self.reset_latency_samples,
            "nonreset_latency_samples": self.nonreset_latency_samples,
            "noise_normal_generated": {
                **normal,
                "count": self.noise_normal_count,
            },
            "noise_shear_generated": {
                **shear,
                "count": self.noise_shear_count,
            },
            "noise_step_unique_signatures": signature_count,
            "noise_step_signature_width": 8,
            "noise_step_unique_seeds": seed_count,
            "noise_step_seeds_consecutive": (
                self.noise_step_seeds == expected_seeds
            ),
            "output_nonzero_by_role": self.output_nonzero_by_role,
            "initial_raw_nonzero_by_role": (
                self.initial_raw_nonzero_by_role
            ),
            "raw_nonzero_values_by_role": self.raw_nonzero_values_by_role,
            "raw_nonzero_steps_by_role": self.raw_nonzero_steps_by_role,
            "checks": checks,
            "passed": all(checks.values()),
        }


def configure_h2_direct_tactile_stress(
    env,
    taxel_area_m2: float,
    stress_scale: float,
) -> DirectTactileStressRuntime:
    """Configure the locked five-role runtime before the first history query."""

    existing = getattr(env, "_sugar_direct_tactile_stress_runtime", None)
    if existing is not None:
        raise RuntimeError("direct tactile stress runtime is already configured")
    history_cache = getattr(env, "_sugar_direct_tactile_history_cache", {})
    if history_cache:
        raise RuntimeError(
            "H2 tactile stress must be configured before history is observed"
        )
    runtime = DirectTactileStressRuntime(
        num_envs=env.num_envs,
        device=env.device,
        cfg=DirectTactileStressCfg(
            taxel_area_m2=float(taxel_area_m2),
            stress_scale=float(stress_scale),
        ),
    )
    setattr(env, "_sugar_direct_tactile_stress_runtime", runtime)
    return runtime


def direct_tactile_stress_cache_key(env) -> tuple[Any, ...] | None:
    runtime = getattr(env, "_sugar_direct_tactile_stress_runtime", None)
    return runtime.cache_key if runtime is not None else None


def apply_configured_direct_tactile_stress(
    env,
    current: torch.Tensor,
    reset_mask: torch.Tensor,
) -> torch.Tensor:
    runtime = getattr(env, "_sugar_direct_tactile_stress_runtime", None)
    if runtime is None:
        return current
    return runtime.apply(
        current,
        step=int(env.common_step_counter),
        reset_mask=reset_mask,
    )


def direct_tactile_stress_audit(env) -> dict[str, Any] | None:
    runtime = getattr(env, "_sugar_direct_tactile_stress_runtime", None)
    return runtime.audit_state() if runtime is not None else None
