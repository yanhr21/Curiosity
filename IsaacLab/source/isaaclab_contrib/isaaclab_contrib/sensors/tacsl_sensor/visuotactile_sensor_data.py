# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VisuoTactileSensorData:
    """Data container for the visuo-tactile sensor.

    This class contains the tactile sensor data that includes:

    - Camera-based tactile sensing (RGB and depth images)
    - Force field tactile sensing (normal and shear forces)
    - Tactile point positions and contact information

    """

    # Camera-based tactile data
    tactile_depth_image: torch.Tensor | None = None
    """Tactile depth images. Shape is (num_instances, height, width, 1)."""

    tactile_rgb_image: torch.Tensor | None = None
    """Tactile RGB images rendered using the Taxim approach from :cite:t:`si2022taxim`.
    Shape is (num_instances, height, width, 3).
    """

    # Force field tactile data
    tactile_points_pos_w: torch.Tensor | None = None
    """Positions of tactile points in world frame. Shape is (num_instances, num_tactile_points, 3)."""

    tactile_points_quat_w: torch.Tensor | None = None
    """Orientations of tactile points in world frame. Shape is (num_instances, num_tactile_points, 4)."""

    penetration_depth: torch.Tensor | None = None
    """Penetration depth at each tactile point. Shape is (num_instances, num_tactile_points)."""

    tactile_normal_force: torch.Tensor | None = None
    """Signed local-Z force at each tactile point.

    Shape is ``(num_instances, num_tactile_points)``. This preserves the
    released v2.3.2 TacSL projection of the complete SDF penalty force into
    each taxel frame; geometry-bound frames may therefore report compression
    with either local-Z sign.
    """

    tactile_shear_force: torch.Tensor | None = None
    """Signed friction traction on the elastomer in taxel-frame X/Y.

    Shape is ``(num_instances, num_tactile_points, 2)``.  This channel is the
    physical-tangent projection of TacSL's ``F_t`` only; normal pressure is
    never projected into or mixed with signed shear.
    """

    tactile_relative_tangential_velocity_w: torch.Tensor | None = None
    """Simulator-oracle relative tangential velocity at active taxels in world frame.

    Shape is ``(num_instances, num_tactile_points, 3)``. This is the exact
    ``v_t`` already used internally by the official TacSL friction equation;
    exporting it does not change the force computation. It is normally
    reserved for calibration/evaluation labels.  The explicitly authorized
    Plan-11 experiment may expose it as a separately named simulator-oracle
    actor ablation, but it must never be represented as a deployed tactile
    signal or sim-to-real capability.
    """

    tactile_relative_velocity_w: torch.Tensor | None = None
    """Simulator-oracle full relative velocity at active taxels in world frame.

    Shape is ``(num_instances, num_tactile_points, 3)``.  This is the exact
    elastomer-point velocity minus contacted-object closest-point velocity
    computed immediately before TacSL projects out the normal component for
    its released friction equation.  It is exported without finite
    differencing and is zero outside penetrating taxels.  Plan-11 may use it
    only as a separately named simulation ablation; it is not a hardware
    tactile measurement.
    """

    tactile_contact_normal_w: torch.Tensor | None = None
    """Simulator-oracle SDF contact normals at active taxels in world frame.

    Shape is ``(num_instances, num_tactile_points, 3)``. These are the exact
    normalized SDF gradients already used internally to remove normal motion
    from the TacSL friction velocity. They are exported to construct and audit
    controlled calibration probes and, in the authorized Plan-11 simulation
    ablation only, to project the full relative velocity onto its normal.  The
    normal vectors themselves must never be an actor observation or deployed
    tactile-slip feature.
    """
