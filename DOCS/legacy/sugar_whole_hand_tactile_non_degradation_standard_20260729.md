# Frozen Whole-Hand Tactile Non-Degradation Standard V1

Date: 2026-07-29
Status: **authoritative and mandatory**
Protocol ID: `SUGAR_WHOLE_HAND_TACTILE_STANDARD_V1_20260729`

## 1. Authority and claim boundary

This document is the only admissible path for future whole-hand tactile work
in the SUGAR CarryBox mainline. It supersedes every earlier dual-palm,
collision-neutral repeated-R15, shadow replay, one-hull atlas, projected
surface, `4 mm` query-skin, cross-engine replay, and aggregate whole-hand
visualization as a positive sensor path.

The previous native-collision-surface 660-step run remains a valid **nominal
movement baseline only**. Its one-hull spatial field is not an anatomical
whole-hand tactile sensor and must not be used for tactile, slip, policy,
reward, recovery, or strategy evidence.

This contract may be made stricter without changing its identity. It may not be
relaxed, downsampled at the raw-sensor boundary, or replaced after observing a
failure. Changing topology, modalities, thresholds, frame semantics, or video
requirements requires explicit user approval and a new protocol version.

## 2. Official implementation boundary

The primary implementation must use the official IsaacLab v2.3.2
`VisuoTactileSensor` / `VisuoTactileSensorCfg` path:

- query points come from the surface of a real elastomer mesh;
- the CarryBox contact object retains its SDF collision mesh;
- the raw sensor exposes per-taxel penetration, world position, quaternion,
  normal force, and two-axis signed shear;
- local output shapes remain `[environment, taxel]` for normal force and
  `[environment, taxel, xy]` for shear;
- the released total-force projection remains local `Z` normal and local
  `X/Y` signed shear; and
- camera-enabled sensors expose real `tactile_rgb_image` and
  `tactile_depth_image` from the configured camera/gel path.

The normal-force and shear equations remain the official SDF penalty model.
Configurable stiffness, friction, and compliant-contact properties may be
calibrated only through the predeclared bench in Section 7, then frozen before
CarryBox evaluation. No CarryBox trajectory may fit or rescale them.

Tactile Genesis is an optional, separately labeled secondary cross-engine
study only after this IsaacLab path passes and only with new explicit user
authorization. It can never substitute for a failed IsaacLab sensor, provide
missing force values, or be mixed into the primary traces.

## 3. Frozen anatomical topology

The force/shear topology is exactly 27 physical elastomer patches per hand,
54 total:

1. twelve palm patches in fixed row-major `4 x 3` order:
   `palm_r0_c0 ... palm_r3_c2`;
2. thumb proximal, middle, and distal;
3. index proximal, middle, and distal;
4. middle proximal, middle, and distal;
5. ring proximal, middle, and distal; and
6. little proximal, middle, and distal.

The global order is left hand then right hand. Within each hand it is the
twelve palm patches followed by thumb, index, middle, ring, and little, with
proximal-to-distal order inside each digit. This order may not depend on
contact, rollout, seed, object physics, or visualization convenience.

Every patch has a raw `20 x 25` taxel grid. The mandatory force tensors are:

```text
normal:      [T, E, 2, 27, 20, 25]
signed shear:[T, E, 2, 27, 20, 25, 2]
penetration: [T, E, 2, 27, 20, 25]
position_w:  [T, E, 2, 27, 20, 25, 3]
quaternion_w:[T, E, 2, 27, 20, 25, 4]
```

This is 27,000 raw taxels across both hands. Missing or invalid surface
samples are represented by an immutable coverage mask, never filled with zero,
nearest neighbors, interpolation, contact labels, or projected rigid-contact
forces.

The object-facing/contactable surface coverage must be at least 90% separately
for each palm and each of the five digits on both hands. Coverage is
area-weighted on the exact hand mesh and must be reported separately for all
twelve anatomical groups. A global average cannot hide an uncovered thumb,
palm center, or finger.

## 4. Physical sensor construction

Each of the 54 patches must satisfy all of the following:

- it is a physically present compliant elastomer attached to the corresponding
  official SUGAR left/right rubber-hand rigid body;
- its outer taxel surface is the same surface that physically contacts and
  supports the box;
- its collision, material, thickness, mass, inertia contribution, mount
  transform, surface mesh, taxel positions, and local frames are declared and
  hash-bound;
- it does not overlap another sensing/collision patch and owns every contact
  point at most once;
- it is neither collision-neutral nor a query plane floating outside or
  embedded inside a different load-bearing collider; and
- it does not replace the hand by one closed whole-hand convex hull.

The physical contactable palm and finger gaps must remain visible and
reachable. Adding the sensorized asset is an explicit physical robot
modification; it must not be described as the untouched official robot.
Official SUGAR remains the frozen unsensorized control.

Taxel centers must lie on their authored elastomer surface to within
`0.25 * local taxel pitch`, and their outward surface normal must satisfy
`dot(n_taxel, n_surface) >= 0.999`. Local frames must be finite, right-handed,
and orthonormal. Sign errors may not be repaired with `abs`, clamping,
baseline subtraction, or a visualization transform.

## 5. Frozen optical topology

Exactly two palm patches, one per hand, carry the official R15 optical path:

```text
left/palm_r1_c1
right/palm_r1_c1
```

The choice is symmetric and geometry-defined. It may not be changed to the
patch that happens to carry the most load in a rollout.

Both use the official `GELSIGHT_R15_CFG`, an actual camera, the actual
load-bearing R15 elastomer, and camera update period equal to the recorded
control step. They retain:

```text
RGB:   [T, E, 2, 320, 240, 3]
depth: [T, E, 2, 320, 240, 1]
```

The remaining 52 conformal anatomical patches are force/shear sensors, not
fake optical cameras. No RGB/depth may be generated from a heat map,
ElastomerTaxel displacement, a shadow object, a repeated frame, or a
hand-written deformation renderer.

## 6. Forbidden degradations

None of the following counts as tactile progress, sensor admission, or a
temporary substitute:

- `hands_contact_label`, thresholded contact, `net_forces_w`,
  `force_matrix_w`, aggregate body wrench, object pose/state, or force-torque
  summaries;
- one R15 per palm presented as whole-hand coverage;
- one merged palm/finger blob, one whole-hand atlas, or one PhysX convex hull
  rasterized into a tactile image;
- raw PhysX contact points/forces copied, projected, blurred, or distributed
  into taxels;
- collision-neutral, non-load-bearing, floating, embedded, or shadow sensors;
- saved RGB paired with restored/repeated tactile frames;
- cross-engine pose replay presented as same-engine measurement;
- binary arrays, hand-written synthetic taxels, interpolated gaps, absolute
  shear, or channel-first shear used without the required transpose;
- per-frame color autoscaling, hidden zero regions, fabricated palm/thumb
  density, or pressure labels without a taxel-area conversion; and
- a smoke test, controlled press, decoded video, sensor interface, or complete
  lift presented as full sensor acceptance by itself.

Raw per-taxel values must always be retained even if a later policy uses a
learned encoder. Policy compression cannot alter the sensor archive or replace
the full-resolution evaluation path.

## 7. Mandatory calibration and numerical gates

### 7.1 Static and no-contact gate

Before dynamics:

- all 54 patch identities, shapes, transforms, material parameters, coverage
  masks, taxel areas, and hashes must be present;
- no-contact penetration, normal force, and shear must be numerically zero at
  every taxel;
- the two optical streams must be nonblank and their no-contact baselines must
  be temporally stable;
- there must be no preloaded/interpenetrating gel, duplicate collision owner,
  or missing palm/digit group; and
- official G1 joint, action, box, and task identities must be recorded against
  the unsensorized control.

### 7.2 Per-patch controlled response

Every one of the 54 patches receives its own independent probe:

- normal depths `0, 0.25, 0.5, 1.0, 2.0 mm`;
- a hold at each nonzero depth;
- positive and negative local-X tangential motion;
- positive and negative local-Y tangential motion; and
- complete separation and release.

For every patch:

- integrated normal load is strictly monotonic across the four nonzero depths;
- the contact centroid agrees with the commanded indenter location within one
  local taxel pitch;
- at least 95% of active shear taxels have the commanded sign;
- orthogonal shear leakage is at most 10% of commanded-axis magnitude;
- all normal/shear taxels return to numerical zero on physical separation; and
- mirrored left/right loads and active areas agree within 10% under identical
  probes.

### 7.3 Calibration split

Normal stiffness, tangential stiffness, friction, compliant-contact stiffness,
and damping may be selected only from a hash-bound calibration subset of the
controlled probes. A disjoint validation subset must then pass without any
parameter change. There is one frozen parameter set per declared sensor class,
not one fit per rollout, seed, box, frame, hand, or taxel.

Raw PhysX contacts and known indenter loads are audit targets only. They may
score and calibrate the sensor model but may never generate the taxel field.
Before real GelSight calibration, the claim remains **high-fidelity simulated
tactile**, never hardware-validated tactile or sim-to-real.

### 7.4 Spatial and temporal correspondence

On held-out controlled and CarryBox traces:

- for the controlled physical sphere, independently recomputed exact-SDF
  active-taxel intersection-over-union is at least `0.99`, at least 95% of
  union-active taxels reconstruct depth within `10 um`, and maximum depth
  error is at most `0.15 mm`;
- raw PhysX contact buffers are complete, unsaturated, nonempty only on
  physical-contact phases, and owned only by the selected physical patch;
- on at least 95% of eligible controlled frames, force-weighted raw PhysX and
  TacSL contact centroids agree within one local taxel pitch;
- raw PhysX aggregate force and torque remain separate force-balance targets;
  they never create, distribute, or rescale taxels;
- bidirectional raw-manifold/taxel nearest-point fractions remain reported
  diagnostics, not dense-surface admission metrics: PhysX PCM intentionally
  reports a sparse recycled solver manifold, so treating its tuples as a
  dense 500-taxel point cloud is invalid;
- visible palm-center or thumb contact with raw physical load must have a
  same-frame response in that exact anatomical patch;
- every world, normal, shear, RGB, depth, state, action, and raw-audit record
  uses the same source-step index and timestamp;
- sensor lag is zero recorded control steps; and
- stale, repeated, dropped, reordered, or reset-crossing frames are zero.

### 7.5 Optical gate

For both physical R15 palm modules:

- commanded `0.25/0.5/1/2 mm` normal indentation produces monotonic RGB
  difference and depth deformation;
- peak depth error is no larger than
  `max(0.1 mm, 10% of commanded depth)`;
- optical and force contact centroids agree within one force-taxel pitch;
- complete physical release returns to the frozen no-contact baseline; and
- neither the hand shell nor a hidden integration window may occlude or fake
  the contact image.

### 7.6 Object dynamics and gravity gate

Taxel normal/shear vectors are rotated to world coordinates using their
archived quaternions and summed exactly once. During quasi-static holding:

- median vertical support error relative to `m*g` is at most 10%; and
- 95th-percentile vertical support error is at most 20%.

During dynamic lifting, the median normalized vector residual of tactile
contact force plus gravity against `m*a` is at most 20%. The corresponding
object-frame torque residual, normalized by `m*g*box_characteristic_length`,
is also at most 20%.

Mass, gravity, object acceleration, raw PhysX wrench, and required support are
independent audit channels. They are never actor inputs or sources for
constructing tactile fields. A TacSL field that does not pass this gate remains
uncalibrated and cannot be labeled pressure, load, or force-balanced tactile.

## 8. Continuous CarryBox gate

After all controlled gates pass, run the accepted official Refiner from the
beginning of the nominal CarryBox trajectory with the sensorized physical
asset:

- no mid-trajectory reset, state restore, shadow replay, cross-engine bridge,
  hand-written controller, or tactile-driven policy change;
- one world frame and all raw tactile modalities per source step;
- complete pick-up, transport, and normal trajectory termination;
- no sensor-induced unsafe fall or loss of the required nominal task; and
- no recalibration from this rollout.

Then repeat the sensor correspondence and force-balance audit without
recalibration on the predeclared 3x-mass and low-friction conditions. Those
held-out policies need not succeed for sensor admission, but every failure
must retain correct same-frame tactile, physics, and video evidence.

No tactile policy, slip detector, internal reward, SMP adapter, ICM adapter,
CHORD reward, recovery run, or alternative-strategy experiment may use the
new sensor until the controlled gates, nominal continuous CarryBox gate,
held-out sensor audits, independent verifier, and user video review all pass.
All former tactile models and slip checkpoints must be rerun from this source;
they cannot be inherited.

## 9. Frozen video and human-review contract

All scientific tactile evidence is video. PNGs may be navigation aids only.
Every MP4 must be H.264/`avc1`, `yuv420p`, fast-start, at least
`1920 x 1080`, and fully decoded frame-by-frame by an independent audit.
Scientific panels use fixed units and one fixed color scale for the complete
video and all compared physics conditions. Per-frame autoscaling is forbidden.

Controlled probes produce **one separate video per patch**. A 54-patch mosaic
is not a replacement. Each video shows:

1. a large world/hand view with indenter and physical patch geometry;
2. the raw `20 x 25` normal-force map;
3. signed shear-X and shear-Y maps/arrows;
4. the patch name, source frame, exact load, active-taxel count, and contact
   centroid; and
5. raw PhysX contact markers clearly labeled `AUDIT ONLY`.

Each CarryBox rollout produces at least these separate videos:

1. `master_sync`: world-wide RGB occupies the entire top half; below it are
   left/right anatomical overviews, object height, support, `m*g`, and time;
2. `left_detail`: all 27 left patches remain distinct and anatomically placed,
   with fixed-scale normal and signed shear;
3. `right_detail`: the corresponding 27 right patches;
4. `palm_optical`: world view plus the two real R15 RGB and depth streams; and
5. `force_balance`: world view plus per-hand/per-region tactile wrench,
   required object wrench, independent PhysX audit wrench, and residuals.

One rollout/condition gets its own video set. Nominal, 3x mass, low friction,
controlled probes, failures, and alternative seeds may not be concatenated
into one dense comparison video as the sole evidence.

The videos must make zeros visible rather than hiding them. If the world view
shows contact but the corresponding palm center, thumb, or finger patch is
empty, the gate fails. Conversely, a visibly close but unloaded surface may
remain zero if same-frame physical-contact and separation audits prove that it
is not carrying load.

After the independent numerical and codec audits finish, execution stops for
human review. The user must explicitly approve the mount, anatomical coverage,
same-frame correspondence, force balance, and optical response before any
policy or reward integration.

## 10. Acceptance rule

Whole-hand tactile is accepted only if every mandatory gate in this document
passes together:

```text
physical topology
+ 54/54 controlled response
+ two optical R15 gates
+ calibration/validation split
+ spatial and temporal correspondence
+ gravity/dynamics closure
+ continuous nominal CarryBox
+ held-out physics sensor audits
+ independent reconstruction
+ full video decode
+ explicit user approval
```

Any missing or failed term leaves the sensor unresolved. There is no partial,
proxy, interface-only, video-only, or movement-only acceptance.

## Official references

- IsaacLab v2.3.2 Visuo-Tactile Sensor:
  https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/core-concepts/sensors/visuo_tactile_sensor.html
- IsaacLab v2.3.2 sensor API:
  https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab_contrib/isaaclab_contrib.sensors.html
- Tactile Genesis project:
  https://neuroagents-lab.github.io/tactile-genesis/
