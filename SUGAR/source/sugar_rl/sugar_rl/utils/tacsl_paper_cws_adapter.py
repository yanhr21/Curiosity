# Copyright (c) 2026, Curiosity Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact TacSL geometry adapter for the paper contact-wrench equations.

The official TacSL sensor reports SDF object-outward normals and forces acting
on the elastomer.  The paper equations require object-frame positions and the
center direction of the compressive force that the hand can apply *to the
object*.  This adapter performs that sign and frame conversion while retaining
the complete taxel mask.

The measured pressure/shear wrench helper is a separate project diagnostic.
It is not part of the CHORD paper's unit friction-cone support function and is
not an ICM curiosity signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TacSLObjectContacts:
    positions_object: torch.Tensor
    force_normals_object: torch.Tensor
    active_contact: torch.Tensor


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def quat_wxyz_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in dimension 4")
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quat_wxyz_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit quaternions using IsaacLab's wxyz convention."""

    if quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("quaternion/vector must end in 4/3")
    q_vector = quaternion[..., 1:]
    uv = torch.linalg.cross(q_vector, vector, dim=-1)
    uuv = torch.linalg.cross(q_vector, uv, dim=-1)
    return vector + 2.0 * (
        quaternion[..., :1] * uv + uuv
    )


def xyzw_to_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in dimension 4")
    return torch.cat((quaternion[..., 3:], quaternion[..., :3]), dim=-1)


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    _require_finite("quaternion", quaternion)
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm <= 1.0e-12):
        raise ValueError("zero quaternion")
    return quaternion / norm


def world_to_object_vectors(
    vector_world: torch.Tensor, object_quaternion_wxyz: torch.Tensor
) -> torch.Tensor:
    quaternion = _normalize_quaternion(object_quaternion_wxyz)
    while quaternion.ndim < vector_world.ndim:
        quaternion = quaternion.unsqueeze(-2)
    return quat_wxyz_apply(quat_wxyz_conjugate(quaternion), vector_world)


def world_to_object_points(
    point_world: torch.Tensor,
    object_position_world: torch.Tensor,
    object_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    position = object_position_world
    while position.ndim < point_world.ndim:
        position = position.unsqueeze(-2)
    return world_to_object_vectors(
        point_world - position, object_quaternion_wxyz
    )


def tacsl_contacts_to_object_frame(
    *,
    tactile_points_pos_w: torch.Tensor,
    tactile_sdf_outward_normal_w: torch.Tensor,
    object_transform_xyzw: torch.Tensor,
    normal_force: torch.Tensor,
    active_force_epsilon: float = 0.0,
) -> TacSLObjectContacts:
    """Convert exact TacSL SDF geometry into paper contact inputs.

    Input shapes may include arbitrary leading batch/hand/grid dimensions.
    ``object_transform_xyzw`` has the same leading batch/hand dimensions and
    ends in 7; singleton dimensions are inserted before taxel dimensions.
    """

    if tactile_points_pos_w.shape != tactile_sdf_outward_normal_w.shape:
        raise ValueError("TacSL points and SDF normals must have equal shape")
    if tactile_points_pos_w.shape[-1] != 3:
        raise ValueError("TacSL points must end in dimension 3")
    if normal_force.shape != tactile_points_pos_w.shape[:-1]:
        raise ValueError("normal_force shape does not match TacSL points")
    if object_transform_xyzw.shape[-1] != 7:
        raise ValueError("object_transform_xyzw must end in dimension 7")
    if active_force_epsilon < 0.0:
        raise ValueError("active_force_epsilon must be non-negative")
    _require_finite("tactile_points_pos_w", tactile_points_pos_w)
    _require_finite("tactile_sdf_outward_normal_w", tactile_sdf_outward_normal_w)
    _require_finite("object_transform_xyzw", object_transform_xyzw)
    _require_finite("normal_force", normal_force)
    if torch.any(normal_force < -1.0e-9):
        raise ValueError("TacSL normal force contains a negative value")

    object_position = object_transform_xyzw[..., :3]
    object_quaternion = xyzw_to_wxyz(object_transform_xyzw[..., 3:])
    positions_object = world_to_object_points(
        tactile_points_pos_w, object_position, object_quaternion
    )
    # TacSL stores the SDF gradient: object-outward and the direction of force
    # acting on the elastomer. Newton's third law gives the force the hand can
    # apply to the object, which is the opposite direction.
    force_normals_object = world_to_object_vectors(
        -tactile_sdf_outward_normal_w, object_quaternion
    )
    active = normal_force > active_force_epsilon

    normal_norm = torch.linalg.vector_norm(force_normals_object, dim=-1)
    if torch.any(active & (normal_norm <= 1.0e-6)):
        raise ValueError("an active TacSL taxel lacks an SDF normal")
    force_normals_object = torch.where(
        active.unsqueeze(-1),
        force_normals_object
        / normal_norm.unsqueeze(-1).clamp_min(1.0e-12),
        torch.zeros_like(force_normals_object),
    )
    return TacSLObjectContacts(
        positions_object=positions_object,
        force_normals_object=force_normals_object,
        active_contact=active,
    )


def measured_tacsl_object_wrench(
    *,
    contacts: TacSLObjectContacts,
    tactile_sdf_outward_normal_w: torch.Tensor,
    tactile_points_quat_w_wxyz: torch.Tensor,
    object_transform_xyzw: torch.Tensor,
    normal_force: torch.Tensor,
    shear_force: torch.Tensor,
    reduce_dims: tuple[int, ...],
) -> torch.Tensor:
    """Aggregate the actual TacSL pressure/shear force into ``[F, tau]``.

    The result is a project diagnostic in object coordinates.  It must remain
    separate from unit friction-cone capability support and original ICM.
    """

    if tactile_points_quat_w_wxyz.shape[:-1] != normal_force.shape:
        raise ValueError("taxel quaternion shape does not match normal force")
    if tactile_points_quat_w_wxyz.shape[-1] != 4:
        raise ValueError("taxel quaternion must be wxyz")
    if shear_force.shape != (*normal_force.shape, 2):
        raise ValueError("shear force shape does not match normal force")
    if tactile_sdf_outward_normal_w.shape != contacts.positions_object.shape:
        raise ValueError("SDF normal shape does not match converted contacts")
    _require_finite("tactile_points_quat_w_wxyz", tactile_points_quat_w_wxyz)
    _require_finite("normal_force", normal_force)
    _require_finite("shear_force", shear_force)
    if torch.any(normal_force < 0.0):
        raise ValueError("normal_force must be a compression magnitude")

    tactile_quaternion = _normalize_quaternion(tactile_points_quat_w_wxyz)
    # The sensor reports a positive compression magnitude and signed
    # friction traction on the elastomer.  The equal-and-opposite load on the
    # object is therefore +normal along outward taxel Z and -shear in the
    # physical tangent plane.  The SDF normal remains an independent contact
    # geometry/audit channel; it must not replace the archived taxel frame.
    force_on_object_tactile = torch.cat(
        (-shear_force, normal_force.unsqueeze(-1)), dim=-1
    )
    force_on_object_w = quat_wxyz_apply(
        tactile_quaternion, force_on_object_tactile
    )
    object_quaternion = xyzw_to_wxyz(object_transform_xyzw[..., 3:])
    force_on_object = world_to_object_vectors(
        force_on_object_w, object_quaternion
    )
    force_on_object = torch.where(
        contacts.active_contact.unsqueeze(-1),
        force_on_object,
        torch.zeros_like(force_on_object),
    )
    moment_on_object = torch.linalg.cross(
        contacts.positions_object, force_on_object, dim=-1
    )
    wrench = torch.cat((force_on_object, moment_on_object), dim=-1)
    for dim in sorted(reduce_dims, reverse=True):
        wrench = wrench.sum(dim=dim)
    return wrench
