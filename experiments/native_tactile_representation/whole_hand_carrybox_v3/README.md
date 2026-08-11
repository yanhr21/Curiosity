# SUGAR CarryBox bilateral whole-hand tactile videos

This directory contains one successful and two failed SUGAR CarryBox
visualizations. All three use the same physical scene, 27-patch-per-hand
sensor topology, color scale, and anatomical layout.

[`MANIFEST.json`](MANIFEST.json) is the compact machine-readable inventory of
the representation, all retained cases, video roles, frame counts, and the
single reproduction entry point. [`REPRODUCE.md`](REPRODUCE.md) is the
human-readable end-to-end procedure. Neither document uses a hash workflow.

## Videos

- `successful_grasp/successful_carrybox_whole_hand_tactile.mp4`
  shows contact, pickup, a sustained high hold, set-down, release, and the
  final all-zero tactile state. It contains 430 displayed source frames
  (`230:660`). The raw trace contains 192 consecutive frames in which the box
  is lifted at least 0.20 m while both hands retain tactile contact.
- `failed_grasp/failed_carrybox_whole_hand_tactile.mp4`
  uses the identical trajectory through source step 359. At step 360 the
  physical robot action becomes exact zero; bilateral contact is lost at that
  step, all taxels are zero one step later, and the dynamic box drops 0.287 m
  before task termination with vertical velocity `-2.366 m/s`.
- `failed_closure/failed_closure_carrybox_whole_hand_tactile.mp4`
  shows a left-only closure failure at half-speed. The left thumb/index touch
  the box for nine source frames and reach 277 active taxels, the right hand
  remains at zero, and the box rises by only 0.012 m before returning down.

The success and release-failure directories also contain:

- `left_detail.mp4`: all 27 left-hand patches at a readable size;
- `right_detail.mp4`: all 27 right-hand patches at a readable size;
- `palm_optical.mp4`: the same CarryBox frame with both R15 RGB/depth streams;
- `successful_grasp/force_kinematics_friction_complete.mp4`: the corrected
  `5 ms`-substep physical audit, all 27 patches on both hands, friction support,
  and the separate uncalibrated TacSL raw wrench. Its 430 displayed frames
  cover pickup, carry, placement, release, and the final all-zero tactile state.

All five videos in the canonical successful bundle use the same complete
source interval `230:660`; the main, left-detail, right-detail, bilateral R15,
and force videos each independently full-decode `430/430` frames.

The main, hand-detail, and optical H.264 videos are `2560 x 1440`; the force
audit is `1920 x 1080`, and the raw world recordings are `1280 x 720`. The
bilateral main video places the SUGAR CarryBox world view and a same-frame
hand/box close view on top. The bottom shows both complete hands.
Each hand is laid out anatomically rather than as a generic matrix: four
upright fingers are aligned above the `4 x 3` palm, the thumb is attached on
the outside, and the right hand is mirrored. Distal, middle, and proximal
patches remain distinct. Every one of the 27 `20 x 25` taxel maps remains
visible. The center-palm R15 location is outlined separately. A thick border
marks an active patch; the header reports active patches and taxels. Red is
negative raw local-Z, blue is positive local-Z, and arrows are signed local-XY
shear. Both videos use the pooled success/failure active-taxel 95th-percentile
scales: `0.576832 N` for absolute local-Z and `0.514412 N` for shear magnitude.

## What the videos show

Every retained H.264 file in the three outcome directories plays from beginning
to end. No hash or provenance workflow is used for this visual review. The
top-right close view is bottom-aligned so the hands and box remain visible
rather than enlarging empty background.

The result does not establish complete palm loading or calibrated TacSL force.
During the corrected run's 192 lifted-bilateral frames, all physical normal
load is on distal patches. The left thumb is active on `2/192` frames and the
right thumb on `6/192`; the four non-thumb fingertips dominate. All palm and
middle/proximal finger patches are inactive. This is the actual grasp geometry,
not a display omission: TacSL patch activity and PhysX patch contact agree on
`99.797%` of patch-frame pairs, and the sensorized-patch wrench differs from
the all-robot wrench by only `1.95e-7 N` at the median.

At the native physics clock, PhysX force and `m(a-g)` close with median
full-vector residual `8.78e-7` box weights. Median vertical support is
`2.568 N` friction plus `0.187 N` normal, and the summed normal magnitude is
`58.568 N`; the global friction utilization against patch coefficient `0.5`
is `8.74%`. The actor pose is continuous at `5 ms`: the largest recorded
translation and rotation increments are `10.25 mm` and `0.00991 rad`, and the
quaternion norm error stays below `1.73e-7`. The physical grasp is therefore
translationally and kinematically consistent. A full rotational torque balance
is not claimed because the trace does not contain contact-point moments and the
world inertia tensor.

The official TacSL penalty calculation is internally correct and locates the
contact, but its raw taxel vectors are not calibrated to that physical wrench:
the median aggregated magnitude is `32.754 N` and the median residual is
`11.533` box weights. Therefore these are synchronized native simulated tactile
visualizations and a valid success/release comparison, not calibrated force,
hardware tactile, whole-hand loading, or sim-to-real evidence.

Machine-readable sensor details remain beside each run, but the current task
is reviewed through the playable CarryBox/tactile videos. See
`REPRODUCE.md` for the single retained-allocation command that regenerates the
complete raw trace and all presentation videos.
