# Plan 14: Newton/IsaacLab Universal Native Tactile and Slip

## Objective

Build one reusable tactile contract that is faithful to each simulator's
native contact model, works in both Newton and IsaacLab, and is demonstrated
on CarryBox and a non-box scene. Add causal tactile-only slip detection. Do
not train or fuse a policy in this plan.

Plan 14 supersedes Plan 13 as the only execution queue. Plan 12 remains the
accepted IsaacLab CarryBox representation foundation, and Plan 13 remains a
read-only record of the paused training investigation.

## Scientific boundary

The two engines do not expose identical sensors, so the implementation must
standardize semantics without inventing missing measurements.

- IsaacLab uses the official v2.3.2
  `isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor` and official R15
  configuration. Its raw taxel positions, orientations, penetration, signed
  local-Z force, signed local-XY shear, timestamps, and available GelSight
  RGB/depth are preserved.
- Newton uses native solved contact wrenches and native contact geometry from
  `Contacts`, after `solver.update_contacts()`. Force on the sensing shape is
  transformed into a geometry-fixed patch frame and accumulated on a fixed
  surface grid. It must not substitute `kh * depth`, an aggregate
  `SensorContact` wrench, or a fabricated image for taxel force.
- Newton has no native GelSight optical deformation stream. The common
  contract marks optical data unavailable there; it never fills the field
  with a synthetic zero image and calls that GelSight.
- Raw backend samples remain available beside any common raster. A derived
  raster must conserve the native force vector and preserve sign. It is a
  spatial serialization, not a new physical force model.

## Common frame contract

Every update carries one simulation sequence number, source timestamp and
elapsed time. Each physical patch has a stable name, fixed surface frame and
declared grid geometry. The common force-field channels are:

1. signed force along patch-local Z `[N]`;
2. signed force along patch-local X/Y `[N]`;
3. positive penetration `[m]`;
4. active-sample mask and counterpart identity;
5. sample position and orientation in world and patch coordinates.

The dense layout is `[batch, patch, row, column]` for scalar channels and
`[batch, patch, row, column, 2]` for signed shear. The IsaacLab adapter keeps
the native `20 x 25` R15 ordering: the first tangent/shear axis increases with
row and the second tangent/shear axis increases with column. Patch-size
metadata follows that same row-then-column order. Newton declares the grid when the sensing
surface is constructed and uses conservative bilinear accumulation from the
actual solved contacts. No object pose, reward, contact label, or outcome is
part of the tactile frame.

World orientations use scalar-last `xyzw`, matching Warp/Newton. The thin
IsaacLab adapter reorders the official scalar-first `wxyz` tensor without
changing the represented rotation. Force and optical streams have independent
sequence/timestamp state even when a scene explicitly samples them together.

Optical RGB/depth is an optional, explicitly available modality with its own
clock. Backend-specific diagnostic values may be recorded, but they remain
outside the deployable tactile frame.

## Newton implementation

Add a public `SensorTactile` under `newton.sensors` on the feature branch
`yanhongru/universal-tactile` in the cloned `Newton/` repository.

- Select sensing shapes through Newton's existing label/index mechanism.
- Define each patch from its real shape transform plus a fixed local surface
  origin, orthonormal tangent axes, outward normal and metric bounds.
- Request the native `force` contact attribute before contacts are created.
- At each update, reconstruct the effective world contact point and signed
  separation from Newton's contact buffers, take the solved force on the
  sensing shape, rotate it into the patch frame, and accumulate it without
  clipping or changing sign.
- Expose raw per-contact values, dense normal/shear/penetration fields,
  transforms, sequence and time. Multiple sensing patches, counterparts and
  worlds must use the same class.
- Test force sign, shape0/shape1 symmetry, translation/rotation invariance,
  counterpart filtering, dynamic/kinematic/world-fixed patches, force
  conservation and reset behavior.

The first Newton task case is a box grasp/carry using physical pads. The
second case reuses the same sensor without code changes on a non-box object
such as the existing panda-hydro pencil scene.

## IsaacLab implementation

Add a thin adapter around official `VisuoTactileSensorData`; do not fork the
TacSL force equation into another local sensor.

- Preserve the exact native taxel order and signed channels.
- Support an arbitrary declared set of physical patches and contact-object
  streams. When a scene has several possible SDF counterparts, retain the
  per-counterpart raw stream and sum only the explicitly named derived field.
- Preserve force and optical clocks separately and report whether RGB/depth
  is currently available.
- Demonstrate the adapter on the existing sensorized SUGAR G1 CarryBox scene
  and on a non-box contact scene using the same collector and contract.

## Slip detection

Slip detection is deterministic and causal; it is not ICM, a reward, a policy
input, or a learned classifier in this plan. Its runtime input is only the
common tactile history and timestamps.

The detector reports raw continuous evidence before an ordinal state:

- normal and tangential load;
- friction utilization when the patch friction coefficient is declared;
- metric center-of-pressure velocity in the fixed patch frame;
- pressure-footprint transport/change;
- normal-load loss and contact age.

The ordinal output is `NO_CONTACT`, `STICK`, `INCIPIENT`, or `GROSS`, using a
declared hysteretic configuration. Controlled stick-to-slide sequences in
each backend set and test the configuration. Simulator relative tangential
velocity and object motion are evaluation-only labels and may never enter the
detector.

## Evidence and completion

Completion requires all of the following, not merely import or shape tests:

1. Newton box and non-box runs with native force conservation and synchronized
   world/tactile/slip H.264 evidence.
2. IsaacLab CarryBox and non-box runs with official TacSL fields and the same
   common contract and presentation.
3. Separate controlled stick, incipient-slip and gross-slip intervals in both
   engines, compared against simulator-only relative-motion labels with
   measured detection delay and false positives.
4. One reproduction guide covering environment, construction, update order,
   outputs, videos and numerical checks for both engines.
5. Human inspection of the four scene videos. Until that inspection, the
   result remains high-fidelity simulated tactile and not hardware-validated
   or sim-to-real.

Routine hashes, version ladders and policy training are outside this plan.
