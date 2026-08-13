# Plan 14: Newton/IsaacLab Universal Native Tactile and Slip

## Active full-G1 IsaacLab reset (2026-08-13)

The active deliverable is no longer a detached two-sensor fixture. Every new
object demo runs in IsaacLab/PhysX with the complete SUGAR G1 and exactly 27
physical anatomical TacSL patches on each hand: palm `4 x 3` plus three
segments on each of five fingers. The synchronized video must show the full
G1 actually moving the object and both 27-patch maps. Newton may supply object
geometry but is not an active simulation backend.

Execution is serial: (1) one complete-G1 rigid pickup, (2) a matched physical
failure, (3) another rigid shape using the same collector, and only then (4)
soft-body compatibility. The existing sensorized G1 CarryBox result is the
implementation foundation. The detached cup, flat-block and soft-block R15
fixtures below remain useful sensor diagnostics, but they do not satisfy any
of these four active milestones and must not be reported as complete object
demos. No policy training is authorized.

The object-swappable PickBottle route is implemented. H200 is known to work:
retained job `231928` on `server13` completed the full 660-frame IsaacLab
CarryBox collection and all renderings on 2026-08-11. The later failures reused
two different H200s on `server13` and one on `server53`, all before any SUGAR
scene was built. They therefore establish a current cross-node Kit/Vulkan
runtime failure, not an H200 incompatibility or a PickBottle/TacSL failure.
The Reflex cluster note is for SAPIEN: its empty `DISPLAY` is retained, but
explicit system ICD and default Vulkan-loader selection both reproduced the
same IsaacLab device loss. Do not switch the simulation to Newton as a
workaround. A separate clean Isaac Sim 5.1 runtime with fresh Kit binaries,
extension caches and official base components is now ready. The previous three
allocations were revoked by the scheduler rather than voluntarily released;
retained job `237668` is queued to run the final clean H200 canary before the
formal PickBottle collection.

## Objective

The original cross-engine contract objective below is completed historical
foundation. It no longer authorizes any Newton demo. The active objective is
to reuse its IsaacLab side for complete-G1 rigid pickups and physical failures,
then evaluate an IsaacLab soft-body extension only after the rigid cases pass.
Do not train or fuse a policy in this plan.

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
actual solved contacts. A sample outside the declared planar bounds keeps its
raw position and is accumulated on the nearest grid edge, so the derived grid
still conserves its signed force vector. No object pose, reward, contact label,
or outcome is part of the tactile frame.

World orientations use scalar-last `xyzw`, matching Warp/Newton. The thin
IsaacLab adapter reorders the official scalar-first `wxyz` tensor without
changing the represented rotation. Force and optical streams have independent
sequence/timestamp state even when a scene explicitly samples them together.
Runtime traces persist each stream's sequence, timestamp and elapsed time
directly rather than requiring reconstruction from the control-step index.

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

## Current Newton result (2026-08-12)

The native solved-force sensor, common adapter and causal detector are running
on the exact SUGAR G1/anatomical-54/CarryBox geometry, Newton's rigid Panda
hydroelastic cube scene, and Newton's Franka/deformable-duck scene. The G1
robot follows the recorded official motion kinematically, but the
`0.3023376 kg` box is a free Newton body under gravity. Per-substep root/joint
interpolation fixes the earlier 20-ms replay jump. In the continuous
source-`260...515` result the box lifts `0.922 m` and returns to the ground;
left/right native tactile is present on `255/256` and `256/256` frames, with a
maximum independently reconstructed raw-to-grid difference of `2.74e-4 N`.

The separately playable 128-frame H.264 uses actual Newton VTK state above
both complete 27-patch hand maps. The late set-down is numerically stiff and
must not be presented as a calibrated cross-engine or hardware force. The
rigid Panda result lifts the cube `0.224 m`; the soft Franka result retains
solved particle-rigid force. The common contract, slip tests and MuJoCo
supplied-state contact-localization regression pass. Explicit human inspection
of the retained scene videos is the remaining completion gate. No policy
training is authorized in this plan.

## IsaacLab-only demo expansion (2026-08-12)

All newly requested tactile demonstrations now run in IsaacLab. Newton may
provide a rigid or deformable USD/mesh asset, but it is not the simulation
backend for this expansion. The retained Newton runs above are historical
contract evidence and are not templates for new runtime results.

Execution is serial. First complete a large-contact rigid cup/container pickup
with two official R15 sensors. Using the identical IsaacLab scene and sensor
path, then produce a physical post-lift grip-relaxation slip/drop case. Next
reuse the collector on at least one different rigid shape. Only after those
rigid cases are visibly and numerically valid may the work move to an IsaacLab
soft-body pickup. Each case preserves raw native TacSL taxel force, signed XY
shear, penetration, poses, independent force/optical clocks and available
official R15 RGB/depth. Object pose is diagnostic only. No policy training is
part of this expansion.

## Superseded detached-fixture rigid diagnostics (2026-08-12)

The earlier rigid fixture stage produced four retained IsaacLab/PhysX runs and four
separately playable 400-frame H.264 videos. A Newton cup mesh is used only as
imported geometry; Newton is not a runtime backend. The flat block is created
by IsaacLab itself. Both are dynamic 0.02 kg SDF bodies and use the same two
official R15 sensors and the same native trace/renderer contract.

The cup bottom-support case lifts 0.1667 m and has bilateral tactile on
380/400 frames. Its matched physical failure lifts 0.1753 m before the two
supports withdraw; both native tactile fields then fall to zero and the cup
lands on the IsaacLab ground. The flat-block bottom-support case lifts
0.1667 m with bilateral tactile on 400/400 frames and a median lifted active
area of 126/500 and 128/500 taxels. A side-pinch negative reaches 442/500 and
385/500 active taxels but slips during upward motion and never leaves the
table. This is the useful distinction between a large tactile footprint and a
successful force-closure strategy.

Those force-only diagnostic collections explicitly disabled the RTX cameras.
They retain official native TacSL force/shear/penetration and mark optical
RGB/depth unavailable rather than fabricating it. Their world panels are
trace-driven views of recorded IsaacLab poses, orientations, velocities and
taxel positions; they are not presented as RTX footage.

## Superseded detached-fixture deformable diagnostics (2026-08-12)

The earlier soft fixture stage produced a small, explicitly labeled compatibility
extension. Upstream v2.3.2 TacSL accepts only a rigid SDF contact object, so the
extension replaces that surface-query operation with a signed closest-surface
query on the current native PhysX `SoftBodyView` collision-tetrahedron
boundary. It retains the official R15 `20 x 25` taxel positions and frames,
official `VisuoTactileSensorData` contract, and the official TacSL normal and
Coulomb-limited tangential force equations. Surface velocity is the causal
finite difference of actual collision nodes. No rigid core, rigid contact
wrench, object pose, or hand-painted/generated taxel field enters tactile.
This is a project extension, not upstream out-of-box deformable support.

The stable native soft block lifts `0.1656 m`, remains bilateral on `400/400`
frames, and has median lifted contact areas of `386/500` and `417/500` taxels.
Its maximum extent change is `1.9 mm`, maximum R15 penetration is `0.47 mm`,
and maximum collision stress is `0.392 MPa`. In the matched physical failure,
the same block first lifts `0.1656 m`; the two support faces then move downward
away from the body and laterally clear. Both fields go from active contact to
zero on the next tactile frame, the block falls `0.2734 m`, and the stress peak
occurs at ground impact rather than withdrawal. The earlier direct lateral
withdrawal was rejected because high friction stretched the soft body; it is
archived and is not retained as evidence.

The stable taxel field reconstructs to approximately `0.139 N` upward versus
the object's `0.196 N` weight. This is a measured calibration gap: TacSL's
penalty force is not the PhysX solver wrench. Preserve the native signal and
report the gap rather than tuning stiffness after the fact to manufacture
force balance.
