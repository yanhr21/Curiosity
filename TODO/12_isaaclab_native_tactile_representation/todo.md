# TODO 12: SUGAR CarryBox Whole-Hand Native Tactile Visualization

This queue is the completed representation foundation. Its only open item is
explicit user visual acceptance or a requested presentation correction.
Policy training and tactile-benefit work now belong exclusively to Plan/TODO
13; do not resume demo-following, Curiosity, mass-OOD, or older tactile queues
from this file.

## A. Whole-hand runtime

- [x] Resolve all 54 physical `VisuoTactileSensor` objects on the sensorized
  SUGAR G1.
- [x] Confirm the fixed left/right 27-patch order: palm `4 x 3`, then thumb,
  index, middle, ring, little, each proximal/middle/distal.
- [x] Confirm every patch supplies real `20 x 25` penetration, signed normal,
  and signed XY shear fields; no proxy or generated patch is allowed.
- [x] Confirm both center-palm R15 RGB/depth streams remain available.

## B. Successful CarryBox grasp

- [x] Establish physical bilateral whole-hand contact on the SUGAR CarryBox.
- [x] Lift and visibly hold the fully dynamic box.
- [x] Archive the synchronized world frame and all 54 raw sensor fields at
  every recorded step.
- [x] Confirm the hold retains native tactile feedback on both hands.

## C. Failed CarryBox grasp

- [x] Establish real bilateral contact before failure.
- [x] Produce a physical slip, release, or failed closure without editing
  tactile arrays.
- [x] Archive the same raw fields and clock as the successful run.
- [x] Confirm the anatomical maps show the corresponding loss/change in
  support.
- [x] Add a distinct left-only closure failure: nine frames of native left
  thumb/index contact, zero right-hand taxels, and no meaningful box lift.

## D. Human-readable videos

- [x] Render `successful_carrybox_whole_hand_tactile.mp4` at `2560 x 1440`.
- [x] Render `failed_carrybox_whole_hand_tactile.mp4` with the identical layout
  and fixed scales.
- [x] Put the continuous SUGAR G1 CarryBox world view across the top half.
- [x] Put both complete hands below it in the same video.
- [x] Show each hand as an anatomical map rather than a generic matrix: four
  upright fingers above the `4 x 3` palm, with the thumb on the outside; mirror
  the right hand and retain readable names and active-taxel counts.
- [x] Keep every patch's `20 x 25` map and signed shear visible; do not replace
  it with one aggregate hand blob.
- [x] Render separate left-detail, right-detail, bilateral R15 optical, and
  simple tactile/PhysX support-force videos for both success and failure.
- [x] Encode H.264/avc1, yuv420p, fast-start and fully decode every frame.
- [x] Bottom-align the same-frame close view so the hands and CarryBox remain
  visible in all three main videos.

Current scientific boundary: the replacement `2560 x 1440` anatomical videos
show all 27 patches on each hand and play through normally. Sustained support
is genuinely concentrated on distal finger patches; the palm and the
middle/proximal finger segments are visible but unloaded. The clock-correct
`5 ms` audit supersedes the old control-sampled force plot: PhysX and
`m(a-g)` have median full-vector residual `8.78e-7` box weights, while the raw
TacSL wrench still has median residual `11.533` box weights. This establishes
correct physical box dynamics and spatial contact correspondence, but not a
calibrated tactile wrench or complete palm loading.

## E. Direct visual review

- [x] Archive taxel world positions/quaternions, contact normals, relative
  tangential velocities, both native R15 RGB/depth streams, actual object
  mass/velocity, and separate per-patch/all-robot PhysX normal and friction
  reference forces.
- [x] Identify the real CarryBox-supporting robot bodies and reconstruct the
  TacSL, patch-only PhysX, all-robot PhysX, and required object world forces.
- [x] Display every requested patch directly from the collected native sensor
  arrays.
- [x] Check that each requested video plays and visibly aligns CarryBox motion
  with the corresponding tactile frame.
- [x] Recompute the full 3-D box force on all four `5 ms` physics substeps per
  control frame; the median lifted-bilateral residual is `8.78e-7` box weights.
- [x] Confirm the 54 sensorized patch bodies account for the all-robot box
  wrench: median difference `1.95e-7 N`, excluding hidden support from
  uninstrumented robot bodies.
- [x] Confirm native TacSL activity and PhysX patch contact agree on `99.797%`
  of lifted-bilateral patch-frame pairs with zero tactile false positives.
- [x] Separate the physical friction result from TacSL shear: median vertical
  PhysX friction support is `2.568 N`; global utilization against patch
  coefficient `0.5` is `8.74%` at the median.
- [x] Check `5 ms` pose continuity: quaternion norm error stays below
  `1.73e-7`, maximum translation increment is `10.25 mm`, and maximum rotation
  increment is `0.00991 rad`. Full rotational torque closure is not claimed
  because contact-point moments and the world inertia tensor were not recorded.
- [x] Render and fully decode the complete synchronized force/kinematics/
  friction video (`430/430` frames), including placement, release, and
  post-release zeros. Mask the robot-only free-body comparison after ground
  contact because the ground then carries the box weight.
- [x] Present the two videos for user review.
- [ ] Record explicit user acceptance or required corrections.

Do not add SHA-256, hash binding, or redundant provenance machinery to this
visualization task.

## F. Workspace governance

- [x] Plans/TODOs 04--11 are under `legacy/`; Plan/TODO 12 is the completed
  representation record and Plan/TODO 13 is the only active training queue.
- [x] Demo following, Curiosity/ICM, and tactile-training history retain five
  declared experiment packages in total, exactly at the maximum of five.
- [x] Move the detached dual-R15 final candidate out of the active whole-hand
  deliverable path after the replacement pair is ready.

## G. Reproduction and reuse boundary

- [x] Make `successful_grasp/whole_hand_trace.npz` the one canonical complete
  660-frame success trace, including all `2640` native physics substeps.
- [x] Move the duplicate no-substep success trace and both withdrawn
  control-sampled `force_balance.mp4` files into the existing single archive.
- [x] Render the main bilateral anatomical video through source frame 660 so
  it includes placement, release, and post-release zeros (`430/430` displayed
  frames).
- [x] Add one executable retained-allocation entry point:
  `scripts/sugar/native_tactile/run_complete_carrybox_visualization.sh`.
- [x] Record the exact output contract, commands, runtime semantics, and object
  generality boundary in `whole_hand_carrybox_v3/REPRODUCE.md`.
- [x] State the current reuse boundary explicitly: online and causal at the
  simulator clock, slower than wall-clock real time, reusable for configured
  SDF objects, but not yet a validated KickBox sensor task. Official KickBox
  uses a non-SDF big-box asset and normally contacts it with an unsensorized
  foot/leg.
