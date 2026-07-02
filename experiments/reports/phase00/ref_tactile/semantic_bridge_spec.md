# Phase 00 Semantic Bridge Spec

Date: 2026-07-01

This spec makes Gate 00F concrete. It maps the current Newton candidate tactile
channels to official UniVTAC, TaCauchy, and IsaacLab TacSL semantics and states
what evidence is still missing.

Machine-readable spec:
`experiments/configs/phase00/ref_tactile/semantic_bridge_spec_v1.json`

## Candidate Source

Active candidate:
`p00_mjw_marker_v1_20260701_074200`

Current positive evidence:

- base grasp/lift final test passed;
- steel-spec material arrays are present;
- candidate direct `Fn` and `Ft` are nonzero;
- compatible-scene `SensorContact` alignment passed;
- candidate marker/deformation render is nonzero;
- scene/tactile/mechanics video is nonblank.

Current limitation:
these channels are still `candidate.*`, not official tactile semantics.

## Official Targets

### UniVTAC

Relevant official fields:

- `tactile.left_tactile.rgb`
- `tactile.left_tactile.rgb_marker`
- `tactile.left_tactile.depth`
- `tactile.left_tactile.marker`
- `tactile.left_tactile.pose`
- right-side equivalents

Why it matters:
UniVTAC gives the optical tactile image/marker/depth schema that our current
blue marker render must be compared against. Without official sanity, our
marker flow is only a force-derived visualization.

### TaCauchy

Relevant official fields:

- Cauchy stress;
- surface traction;
- normal pressure;
- tangential traction;
- nodal/tributary contact area;
- pressure-normalized contact force.

Why it matters:
TaCauchy gives the physical pressure/traction/contact-area semantics that our
current `Fn`, `Ft`, normal, and area proxy must be compared against.

### Official IsaacLab TacSL

Relevant official fields:

- `VisuoTactileSensorData.tactile_depth_image`
- `VisuoTactileSensorData.tactile_rgb_image`
- `VisuoTactileSensorData.penetration_depth`
- `VisuoTactileSensorData.tactile_normal_force`
- `VisuoTactileSensorData.tactile_shear_force`

Relevant official config:

- `normal_contact_stiffness`
- `friction_coefficient`
- `tangential_stiffness`
- `tactile_array_size`
- `contact_object_prim_path_expr`

Why it matters:
official IsaacLab TacSL gives the closest current official source-level match
to our desired dense tactile RGB/depth/normal/shear/penetration fields. It is
not a passed reference until official TacSL sanity runs in an approved
environment.

## Required Bridge Outcomes

- `candidate.newton_mjw.Fn` must be validated against official pressure/depth
  behavior, including TacSL normal-force and penetration-depth grids if the
  official TacSL sanity path becomes available.
- `candidate.newton_mjw.Ft` must be validated against tangential traction,
  including TacSL shear force, sign, and pad-local frame convention.
- `candidate.newton_mjw.marker_flow` must be validated against official marker
  and `rgb_marker` behavior, not just visual plausibility.
- `candidate.newton_mjw.penetration_or_compression` must be validated against
  official TacSL penetration depth or UniVTAC/TaCauchy depth/deformation output
  before it can be called calibrated tactile deformation.
- `candidate.newton_mjw.area_proxy` must either be calibrated against real
  surface/nodal area or remain explicitly labeled as a proxy.
- `candidate.newton_mjw.contact_normal` must be checked against traction
  decomposition and normal-frame convention.

## Current Gate 00F Status

Blocked. The bridge can be specified from source/document inspection, but it
cannot pass until official UniVTAC/TaCauchy/IsaacLab TacSL sanity runs or
faithful blockers are accepted. `p00_gate_review_v4_20260701_055100` correctly
keeps:

- Gate 00D: `open_reference_semantics_blocked`
- Gate 00E: `open_tactile_validation_blocked`
- Gate 00F: `open_official_semantic_validation_blocked`
- curiosity training: disallowed
