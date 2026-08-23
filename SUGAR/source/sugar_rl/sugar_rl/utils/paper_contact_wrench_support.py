# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Auditable paper-formula reproduction of contact-wrench support.

This is not NVIDIA's unreleased CHORD implementation.  It implements only the
public equations in Zhu et al., *Learning Dexterous Manipulation Using Contact
Wrench Guidance From Human Demonstration* (arXiv:2607.00033v1):

* a unit-edge polyhedral Coulomb friction cone;
* primitive object-frame wrenches ``[f, p x f]``;
* the support function ``max_col(B^T W)``; and
* the two-sided relative-tolerance exponential reward in Equation (3).

The paper does not publish its friction-cone edge count, friction coefficient,
support-direction samples/seed, relative tolerance, or reward variance.  Those
values are therefore mandatory constructor arguments and must be recorded as
reproduction settings rather than represented as official hyperparameters.

Contact normals passed to this module are force directions: the center line of
the compressive force that the contacting body can apply to the object.  They
are not object-outward SDF normals unless the caller has applied the required
sign conversion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class PaperCWSConfig:
    """Explicit settings for the public contact-wrench equations.

    ``support_direction_count`` is locked to 512 because that is the only
    direction count disclosed by the paper.  All other numerical settings
    below are undisclosed by the paper and deliberately have no defaults.
    """

    friction_coefficient: float
    friction_cone_edges: int
    relative_tolerance: float
    reward_variance: float
    support_direction_seed: int
    support_direction_count: int = 512
    support_basis_chunk_size: int = 64
    contact_epsilon: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.friction_coefficient < 0.0:
            raise ValueError("friction_coefficient must be non-negative")
        if self.friction_cone_edges < 3:
            raise ValueError("friction_cone_edges must be at least 3")
        if not 0.0 <= self.relative_tolerance < 1.0:
            raise ValueError("relative_tolerance must lie in [0, 1)")
        if self.reward_variance <= 0.0:
            raise ValueError("reward_variance must be positive")
        if self.support_direction_count != 512:
            raise ValueError(
                "the paper-formula reproduction is locked to the disclosed "
                "512 support directions"
            )
        if not 1 <= self.support_basis_chunk_size <= self.support_direction_count:
            raise ValueError("support_basis_chunk_size is outside [1, 512]")
        if self.contact_epsilon < 0.0:
            raise ValueError("contact_epsilon must be non-negative")

    def to_dict(self) -> dict[str, int | float | str]:
        payload: dict[str, int | float | str] = asdict(self)
        payload["status"] = "paper_formula_reproduction_not_official_chord_code"
        payload["basis_sampling"] = (
            "torch_cpu_float64_gaussian_normalized_then_cast"
        )
        return payload


@dataclass(frozen=True)
class PaperCWSTerms:
    """Equation (3) and the separately exposed contact-mismatch indicators."""

    reward: torch.Tensor
    lower_violation_squared: torch.Tensor
    upper_violation_squared: torch.Tensor
    missed_contact: torch.Tensor
    unintended_contact: torch.Tensor


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def sample_paper_support_directions(
    *,
    count: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return deterministic unit 6-D directions as columns ``(6, count)``.

    The paper discloses neither the original samples nor their sampling seed.
    We use a frozen, hashable Gaussian-on-the-sphere reproduction.  Sampling
    always occurs in CPU float64 before a final cast so the basis identity is
    independent of the execution GPU.
    """

    if count != 512:
        raise ValueError("paper-formula support direction count must be 512")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    directions = torch.randn((count, 6), generator=generator, dtype=torch.float64)
    directions = directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float64).tiny)
    return directions.transpose(0, 1).contiguous().to(device=device, dtype=dtype)


def _validate_contact_inputs(
    positions_object: torch.Tensor,
    force_normals_object: torch.Tensor,
    active_contact: torch.Tensor,
) -> None:
    if positions_object.ndim < 2 or positions_object.shape[-1] != 3:
        raise ValueError(
            "positions_object must have shape (...,contacts,3), got "
            f"{tuple(positions_object.shape)}"
        )
    if force_normals_object.shape != positions_object.shape:
        raise ValueError("force normals and contact positions must have equal shape")
    if active_contact.shape != positions_object.shape[:-1]:
        raise ValueError(
            "active_contact must have shape (...,contacts), got "
            f"{tuple(active_contact.shape)}"
        )
    if active_contact.dtype != torch.bool:
        raise ValueError("active_contact must be boolean")
    _require_finite("positions_object", positions_object)
    _require_finite("force_normals_object", force_normals_object)
    normal_norm = torch.linalg.vector_norm(force_normals_object, dim=-1)
    if torch.any(active_contact & (normal_norm <= 1.0e-12)):
        raise ValueError("an active contact has a zero force normal")


def stable_tangent_basis(
    force_normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize normals and construct a deterministic orthonormal tangent pair."""

    if force_normals.shape[-1] != 3:
        raise ValueError("force_normals must end in dimension 3")
    _require_finite("force_normals", force_normals)
    normal = force_normals / torch.linalg.vector_norm(
        force_normals, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)

    # Choose the Cartesian axis least aligned with each normal.  This avoids
    # the singularity of a fixed reference axis while remaining deterministic.
    reference_index = torch.argmin(torch.abs(normal), dim=-1)
    reference = torch.nn.functional.one_hot(
        reference_index, num_classes=3
    ).to(dtype=normal.dtype, device=normal.device)
    tangent_1 = torch.linalg.cross(normal, reference, dim=-1)
    tangent_1 = tangent_1 / torch.linalg.vector_norm(
        tangent_1, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    tangent_2 = torch.linalg.cross(normal, tangent_1, dim=-1)
    return normal, tangent_1, tangent_2


def unit_friction_cone_edges(
    force_normals: torch.Tensor,
    *,
    friction_coefficient: float,
    edge_count: int,
) -> torch.Tensor:
    """Approximate each Coulomb cone with unit-magnitude edge forces.

    Returns ``(..., contacts, edge_count, 3)``.  Each edge has an axial
    component along the supplied compressive-force normal and a tangential
    component whose magnitude is ``mu`` times the axial component.
    """

    if friction_coefficient < 0.0:
        raise ValueError("friction_coefficient must be non-negative")
    if edge_count < 3:
        raise ValueError("edge_count must be at least 3")
    normal, tangent_1, tangent_2 = stable_tangent_basis(force_normals)
    angles = (
        torch.arange(edge_count, dtype=force_normals.dtype, device=force_normals.device)
        * (2.0 * torch.pi / float(edge_count))
    )
    tangent_ring = (
        torch.cos(angles).view(*((1,) * (normal.ndim - 1)), edge_count, 1)
        * tangent_1.unsqueeze(-2)
        + torch.sin(angles).view(*((1,) * (normal.ndim - 1)), edge_count, 1)
        * tangent_2.unsqueeze(-2)
    )
    raw_edges = normal.unsqueeze(-2) + float(friction_coefficient) * tangent_ring
    return raw_edges / torch.linalg.vector_norm(
        raw_edges, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)


def primitive_contact_wrenches(
    positions_object: torch.Tensor,
    force_normals_object: torch.Tensor,
    active_contact: torch.Tensor,
    *,
    friction_coefficient: float,
    edge_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct Equation (1) primitive wrenches and their active mask."""

    _validate_contact_inputs(
        positions_object, force_normals_object, active_contact
    )
    edge_forces = unit_friction_cone_edges(
        force_normals_object,
        friction_coefficient=friction_coefficient,
        edge_count=edge_count,
    )
    moment_arms = positions_object.unsqueeze(-2).expand_as(edge_forces)
    moments = torch.linalg.cross(moment_arms, edge_forces, dim=-1)
    wrenches = torch.cat((edge_forces, moments), dim=-1)
    wrench_active = active_contact.unsqueeze(-1).expand(
        *active_contact.shape, edge_count
    )
    return wrenches, wrench_active


def contact_wrench_support(
    positions_object: torch.Tensor,
    force_normals_object: torch.Tensor,
    active_contact: torch.Tensor,
    basis_directions: torch.Tensor,
    *,
    friction_coefficient: float,
    edge_count: int,
    basis_chunk_size: int = 64,
) -> torch.Tensor:
    """Compute Equation (2), returning ``(..., basis_count)``.

    Empty contact sets have exactly zero support, which is required for the
    paper's missed/unintended-contact indicators.  Basis chunking bounds the
    temporary tensor size without changing the maximum over wrench columns.
    """

    if basis_directions.ndim != 2 or basis_directions.shape[0] != 6:
        raise ValueError(
            "basis_directions must have shape (6,basis_count), got "
            f"{tuple(basis_directions.shape)}"
        )
    if basis_directions.device != positions_object.device:
        raise ValueError("basis directions and contacts must share a device")
    if basis_directions.dtype != positions_object.dtype:
        raise ValueError("basis directions and contacts must share a dtype")
    if not 1 <= basis_chunk_size <= basis_directions.shape[1]:
        raise ValueError("basis_chunk_size is outside the basis range")
    _require_finite("basis_directions", basis_directions)

    wrenches, wrench_active = primitive_contact_wrenches(
        positions_object,
        force_normals_object,
        active_contact,
        friction_coefficient=friction_coefficient,
        edge_count=edge_count,
    )
    flat_wrenches = wrenches.flatten(start_dim=-3, end_dim=-2)
    flat_active = wrench_active.flatten(start_dim=-2, end_dim=-1)
    has_contact = flat_active.any(dim=-1)

    support_chunks: list[torch.Tensor] = []
    for start in range(0, basis_directions.shape[1], basis_chunk_size):
        basis = basis_directions[:, start : start + basis_chunk_size]
        scores = torch.einsum("...nw,wb->...nb", flat_wrenches, basis)
        scores = scores.masked_fill(~flat_active.unsqueeze(-1), -torch.inf)
        chunk_support = scores.amax(dim=-2)
        chunk_support = torch.where(
            has_contact.unsqueeze(-1),
            chunk_support,
            torch.zeros_like(chunk_support),
        )
        support_chunks.append(chunk_support)
    return torch.cat(support_chunks, dim=-1)


def paper_cws_terms(
    reference_support: torch.Tensor,
    robot_support: torch.Tensor,
    reference_has_contact: torch.Tensor,
    robot_has_contact: torch.Tensor,
    *,
    relative_tolerance: float,
    reward_variance: float,
    contact_epsilon: float = 1.0e-9,
) -> PaperCWSTerms:
    """Compute published Equation (3) and separate mismatch indicators."""

    if reference_support.shape != robot_support.shape:
        raise ValueError("reference and robot support tensors must have equal shape")
    if reference_support.ndim < 1:
        raise ValueError("support tensors require a final basis dimension")
    if reference_has_contact.shape != reference_support.shape[:-1]:
        raise ValueError("reference_has_contact has the wrong shape")
    if robot_has_contact.shape != reference_support.shape[:-1]:
        raise ValueError("robot_has_contact has the wrong shape")
    if reference_has_contact.dtype != torch.bool or robot_has_contact.dtype != torch.bool:
        raise ValueError("contact indicators must be boolean")
    if not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in [0,1)")
    if reward_variance <= 0.0:
        raise ValueError("reward_variance must be positive")
    if contact_epsilon < 0.0:
        raise ValueError("contact_epsilon must be non-negative")
    _require_finite("reference_support", reference_support)
    _require_finite("robot_support", robot_support)

    lower_violation = torch.relu(
        (1.0 - relative_tolerance) * reference_support - robot_support
    )
    upper_violation = torch.relu(
        robot_support - (1.0 + relative_tolerance) * reference_support
    )
    lower_squared = torch.square(lower_violation).sum(dim=-1)
    upper_squared = torch.square(upper_violation).sum(dim=-1)
    reward = torch.exp(-(lower_squared + upper_squared) / reward_variance)

    # Use the explicit contact-set masks as the authoritative paper semantics.
    # The support epsilon is an independent consistency check for callers that
    # reconstruct masks from archived support arrays.
    reference_support_nonzero = torch.any(
        torch.abs(reference_support) > contact_epsilon, dim=-1
    )
    robot_support_nonzero = torch.any(
        torch.abs(robot_support) > contact_epsilon, dim=-1
    )
    if torch.any(reference_support_nonzero != reference_has_contact):
        raise ValueError("reference contact mask disagrees with its support")
    if torch.any(robot_support_nonzero != robot_has_contact):
        raise ValueError("robot contact mask disagrees with its support")
    return PaperCWSTerms(
        reward=reward,
        lower_violation_squared=lower_squared,
        upper_violation_squared=upper_squared,
        missed_contact=reference_has_contact & ~robot_has_contact,
        unintended_contact=~reference_has_contact & robot_has_contact,
    )


def reduced_force_closure_fraction(
    support: torch.Tensor, *, epsilon: float
) -> torch.Tensor:
    """Compute the reduced force-closure objective published in Section 3.2."""

    if support.ndim < 1:
        raise ValueError("support requires a final basis dimension")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    _require_finite("support", support)
    return (support > epsilon).to(dtype=support.dtype).mean(dim=-1)
