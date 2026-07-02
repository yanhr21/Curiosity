# Phase 00 Tactile Representation Decision

Date: 2026-07-01

Status: active Phase 00 design record. This is environment/base evidence only,
not curiosity training success.

## Decision

The active tactile representation must be dense, pad-resolved, and synchronized
with visual/mechanics state. Scalar contact count, binary contact masks, and
slip-risk labels are legacy low-dimensional proxies only.

For the current Newton Panda hydro base, the immediate schema is:

- `left_pressure_map`, `right_pressure_map`: raw hydro contact-surface
  penetration projected into each finger's local tactile plane.
- `left_fn_map`, `right_fn_map`: Gaussian-spread normal force proxy from Newton
  reducer `contact_area * penetration * effective_hydro_stiffness`.
- `left_stress_map`, `right_stress_map`: Gaussian-spread pressure/stress proxy
  from `Fn / contact_area`.
- `left_deform_proxy_map`, `right_deform_proxy_map`: Gaussian-spread
  penetration/compression proxy.
- `left_shear_vector_y_map`, `left_shear_vector_z_map`,
  `right_shear_vector_y_map`, `right_shear_vector_z_map`: Gaussian-spread
  contact-center-motion shear vectors in each pad's local tactile plane.
- `left_shear_magnitude_map`, `right_shear_magnitude_map`: magnitude of the
  shear vector maps.
- `left_f6_normal_proxy`, `right_f6_normal_proxy`: per-pad
  `[Fx,Fy,Fz,Mx,My,Mz]` normal wrench proxy from reducer normal forces.
- `left_f6_ft_capacity_proxy`, `right_f6_ft_capacity_proxy`: per-pad F6
  wrench proxy from friction-capacity force along the contact-center-motion
  tangent.
- `left_f6_combined_proxy`, `right_f6_combined_proxy`: normal plus
  friction-capacity proxy wrench.
- Global time series: object pose/speed/acceleration, contact area, `Fn`,
  stress, `Ft_capacity`, contact normal summary, force balance, penetration,
  lift/hold/drop/safety metrics, and material parameters.

All fields that are not direct hardware-like tactile measurements remain under
the explicit `hydro_proxy.*` interpretation. Direct solver `Ft` and direct
pad-resolved shear force are still open blockers.

## Source Alignment

HydroShear uses tactile marker/grid fields where contact force is spread across
an elastomer grid, normal contact contributes dilation, and tangential contact
contributes shear displacement. Phase 00 mirrors that idea by spreading Newton
hydro reducer contacts into per-pad grids, while preserving provenance as
Newton `hydro_proxy` fields.

T-Rex is the downstream model reference. Its released schema includes visual
camera observations, per-fingertip 6-axis force/torque histories, tactile raw
images, and tactile deformation videos. The current two-finger Panda schema is
therefore intentionally split into grid maps plus force/time-series fields so
it can later be converted toward:

- per-pad or per-finger `F6` proxy/history windows;
- tactile deformation video-like fields;
- visual+tactile masked training inputs;
- explicit ablations for vision+tactile, tactile-only, vision-only, and noisy
  or mismatched tactile.

## Why This Is The Current Faithful Step

Newton official Panda hydro already provides stable grasp/lift/hold evidence
and hydro contact reducer buffers. Taccel attempts did not yet expose valid
nonzero collision/force/deformation through the instrumented path. The current
best path is therefore to keep the serious official Newton base and enrich its
exported tactile fields without modifying the official repository or replacing
the method with a toy tactile model.

The Gaussian grid fields are not claimed to be a final tactile sensor. They are
a source-derived intermediate representation that makes pressure, deformation,
stress, and shear spatially visible, auditable, and suitable for the next
reference-video-alignment gate.

The F6 proxy arrays are only a bridge shape for later T-Rex-style data
conversion. They are not official T-Rex force observations and must not be used
to claim T-Rex checkpoint compatibility until the full schema, timing, and
normalization gates are passed.

## Open Gaps

- Direct solver tangential force `Ft` is not available yet; the first direct
  probe crashed with CUDA illegal memory access.
- The current shear vector is contact-center-motion proxy, not direct
  tangential shear force.
- The scene panel is still schematic, not a USD/photoreal fused scene.
- The tactile fields are not yet validated against a gel/marker tactile camera
  rendering comparable to the reference video.
- The official single-world Newton benchmark is `67.5 FPS`, still below the
  user's `82 FPS` target.
