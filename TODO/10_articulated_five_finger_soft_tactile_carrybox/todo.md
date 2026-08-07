# TODO 10: Articulated Five-Finger Soft-Tactile CarryBox

> 2026-08-03 priority reset: stop this TODO's unfinished execution queue.
> Tactile Genesis direction and four-finger CarryBox work now lives in
> `TODO/11_demo_tg_icm_mass_ood_contact_velocity/todo.md`.

Priority: P0 physical foundation. `[x]` means independently evidenced; `[-]`
means rejected/blocked evidence and never completion.

## Reset and source boundary

- [x] Preserve official SUGAR G1-29 fixed-rubber-hand V83/V87/V88 as the
  frozen control/diagnostic, not as five-finger mechanics.
- [x] Move 128 failed, stopped or superseded anatomical tactile experiment
  directories and 56 old logs/diagnostics, without deletion, to the single
  `/public/home/yanhongru/Curiosity_archive/` root. Retain only six core
  evidence directories and their current logs in the workspace.
- [x] Select Unitree official G1-29+Inspire as the primary articulated
  five-finger source at commit `e30c25b...`; record Apache-2.0 code and asset
  licensing and the source-declared 12 joints per hand.
- [x] Select official Taccel `cb23bc2...` as the primary deformable-gel source;
  preserve its arbitrary-URDF robot, IPC/ABD, tetrahedral gel and tactile
  render paths.
- [x] Execute the unchanged official Taccel soft-teddy path for 80/80 solver
  steps on the retained H200 and independently verify the exact clean source,
  profile coverage and CUDA/IPC runtime. This is engine compatibility only.
- [-] Reject that run's PLY sequence as deformation/tactile evidence: all 80
  scene exports are bitwise identical and it saves no tactile RGB, depth or
  markers. The next sensor gate must call the released `render_tactile()` path
  and preserve its raw outputs.
- [-] The unchanged official TacMan `render_tactile()` example currently exits
  at its first IPC step: the official conjugate-gradient scatter receives 16
  indices for 14 source rows in the pinned compiled runtime. Preserve this as
  an exact compatibility blocker; do not replace it with generated tactile
  arrays or a simplified local renderer.
- [-] Reject Unitree Dex3 for this target because it has only three digits.
- [-] Block `brainco-description` integration until it publishes an actual
  asset license. Keep MIT BrainCo RevoLab only as a secondary official source.
- [-] Keep TacEx and IsaacIPC paper-only until usable official code and license
  are available.

## Official asset acquisition and compatibility

- [x] Acquire the exact official Unitree
  `unitree_sim_isaaclab_usds` package without using its cleanup script; record
  repository revision `394cf244...`, Apache-2.0 metadata, exact
  `1,305,090,539` bytes and SHA-256 `06fbf145...`.
- [x] Extract only after archive integrity passes; independently pass archive
  CRC and hash all seven files in the selected G1-29+Inspire USD/configuration
  tree exactly.
- [x] Independently enumerate runtime bodies, joints, drives, limits,
  collisions, masses and wrist attachments. Confirm palm plus all five digits
  exist independently on both hands. The live unchanged USD has 53 joints,
  65 rigid bodies, 12 finger joints per side and all five digit groups.
- [x] Import the unchanged official asset in IsaacLab v2.3.2/Isaac Sim 5.1 and
  render one held-root bilateral independent-finger-motion video with exact
  runtime topology readback. The 19/19 independent foundation audit and full
  H.264 decode pass; the hold is visualization-only and is not box mechanics.

## No-learning articulated CarryBox mechanics

- [x] Audit the exact official Inspire collision bodies and the unchanged
  official CarryBox PhysX `convexDecomposition`; extract and hash-bind the
  cooked bottom/side envelopes. The old underside targets were 8--40 mm below
  the collision surface and are rejected.
- [-] Reject fixed motion-45 bilateral bottom support as a hand-only lift.
  The clean runs have zero non-hand contact but only `0.0392--0.0440 m`
  apparent lift; V83 supplies median `2.214 N` hand vertical force versus
  `4.935 N` required and has `0.6075` normalized force residual. Its highest
  clean bilateral step loads only left index and right index/middle.
- [-] Reject the fixed-motion COM-support variants. At 0.25 m root retreat,
  moving the right/left palm 40 mm toward COM leaves `30.94/51.50 mm` target
  error. At 0.20 m both targets become sub-millimeter but `head_link` supports
  the box from frame zero with `782.41 N` maximum load. Lowering the setup
  support leaves `24.61/32.67 mm` errors, a `127.72 N` camera contact and only
  `0.0335 m` lift.
- [x] Resolve the v98 non-hand attribution independently. The official
  `head_link` collision vertices extend to body-frame `z=0.5306 m` and overlap
  the restored box state. Raw normal/friction rows reproduce the named
  filtered force-matrix rows within `1.42e-5/2.34e-6 N`; therefore the
  `head_link` load is a real collision, not a ContactSensor index mix-up.
- [-] Reject physical frame-279 replay and the first articulated IK transfer.
  Native-drive V101 does not lift (`-0.00024 m` maximum relative lift), while
  V102--V104 reach at most `0.04941 m` and either hit a camera/wrist, violate
  finger hard limits, or lose the box. The old approximately `0.11 m`
  frame-279 state replay is not physical lift evidence.
- [-] Reject V105--V107 frozen-target side-clamp lift. V105 briefly has all six
  groups on both hands for 23 steps but lifts only `0.01029 m`; V106 reaches
  `0.02835 m` but has zero all-group steps and returns to the floor. V107 is a
  stale-wrapper duplicate of V106, not live-box tracking evidence.
- [x] Implement and source-hash a causal no-learning live-box IK diagnostic
  that reads current PhysX box pose for the next palm target while never
  writing or replaying object state after initialization. V111 and V112 each
  retain bilateral hand contact for 999/1021 frames with zero named non-hand
  load.
- [-] Reject vertical-lead escalation as the physical lift solution. The
  `0.45/3/10 mm` V111/V112/V113 leads reach only
  `0.00188/0.00551/0.01192 m`. V113 crosses the box-weight force threshold but
  contacts `right_ankle_roll_link` from frame 357 with `15.80 N` maximum load;
  all three have zero bilateral all-six-group frames, and the right index and
  little finger never carry load.
- [-] Reject the completed V114--V123 wide-stance side-clamp bracket as a
  physical lift. V115 is topology-only (`664` bilateral all-six-group frames,
  longest `663`) and lifts `0.007224 m`; stronger closure, thumbs, `15/20 mm`
  leads and scheduled trajectories reach at most `0.034556 m` while losing a
  required gate.
- [-] Reject the V124--V152 heuristic side/bottom search. Its best apparent
  peaks are `0.045559/0.052856/0.054853 m` in V145--V147, but no run has one
  bilateral all-six-group frame and the peak states tip into named non-hand
  support. V151/V152 reach only `0.033359/0.038880 m`.
- [x] Add and trace-bind the no-learning cosine live-lead ramp and general
  world-XY grasp anchoring diagnostics without writing object state. The
  producer records both controls, preserves exact official action coupling,
  and passes syntax/source checks.
- [-] Reject V153--V155 as vertical-lift evidence. Unanchored `20/30 mm` ramps
  plateau at `0.048017/0.048832 m` while the box drifts about `0.133 m` and
  rotates about `0.27 rad`; anchoring XY removes all named non-hand load but
  reduces V155 to `0.014124 m`. The earlier rise is sliding/tipping, not a
  stable lift.
- [-] Reject anchored asymmetric V156 as actual bottom-palm support. It reaches
  only `0.010583 m`, has `273/1021` bilateral-contact frames, zero bilateral
  all-six-group frames, then loses the box.
- [x] Export and render V155 and V156 separately from all exact post-PhysX
  robot-body and CarryBox poses. Both H.264 High/yuv420p videos decode at
  `1280x720`, `20 fps`, `256/256` frames; both independent hash/correspondence
  audits pass. Keep them labeled rigid-contact negative diagnostics, not
  tactile or success.
- [x] Close the static contact-seeded palm-reach subproblem. The exact V180
  row-300 source-bound search has 17 strict candidates; selected candidate 8
  reconstructs left/right palm pose errors of `0.158/0.043 mm` and
  `3.49e-5/4.26e-6 rad` at the first V182/V183 dynamic record, with static
  head clearance and zero named non-hand overlap. This is not a grasp or lift.
- [-] Reject the current sequential and simultaneous closure trajectories.
  V182/V183 have `192/90` bilateral-contact frames but zero bilateral
  all-six-group frames, fall below `0.4 m` at frames `368/163`, and never
  lift. The right thumb carries exactly zero force in both controls.
- [x] Export V182 and V183 as separate exact-trace review videos with world
  view, both-hand details, rigid-contact group forces and telemetry. Both are
  H.264 High/yuv420p and pass independent decode/hash/frame-correspondence
  audit. Keep them labeled non-tactile negative mechanics evidence.
- [x] Establish the first real articulated `>0.10 m` CarryBox lift without
  object replay. The 30 mm inset scheduled world-Z run reaches `0.149188 m`,
  has `198` clean bilateral lifted frames, and closes the declared clean raw
  force/torque residuals; however its best topology is only `11/12` for 33
  frames, always missing the left thumb, and it later falls. This is lift-
  authority evidence only, not a stable or tactile pass.
- [-] Reject late-thumb, longer-settle, live-XY and relative-hold repairs as
  the dense stable grasp solution. None produces one bilateral 12/12 frame;
  longer settle/live XY reduce peak lift to `0.089271/0.086013 m`, while the
  relative hold retains an approximately 4--5 cm tracking bias, launches the
  box transiently, then loses contact. Do not extend these parameter sweeps.
- [x] Render the strongest new run as a separate 1280x720 H.264 with world
  view, both-hand details and six rigid-contact groups per hand. Its 256-frame
  exact post-PhysX trace binding and independent decode/hash/correspondence
  audit pass `19/19`. Keep it labeled a negative rigid-contact diagnostic.
- [-] Reject the current strict 20/20 mm inset reproduction. It reaches only
  `0.014055 m`, has zero 12/12 and zero `>0.10 m` lift frames, while retaining
  `499` bilateral-contact frames and zero named non-hand load. Do not use the
  differently hash-bound historical V105 run as current topology evidence.
- [-] Reject the trace-462 left-thumb angle scan. All nine exact-state
  candidates collapse the topology to at most 4/12, produce
  `36.51--104.22 N` local thumb impulses and have `0.280--1.047 rad` tracking
  error. The mesh audit instead places the missing thumb outside the valid
  side silhouette by approximately `4.812 mm` tangent and `18.575 mm` height;
  the corresponding whole-hand shift also fails dynamically.
- [x] Derive one predeclared no-learning right-thumb-bearing closure
  trajectory that preserves bilateral palm support and establishes one stable
  simultaneous palm/thumb/index/middle/ring/little interval on both hands.
  The current-PCA staged closure passes the final `20/20` pre-lift frames with
  12/12 groups, zero named non-hand load and in-limit fingers; the gate stops
  every failed predecessor before lift.
- [-] Reject the gated side clamp as the complete CarryBox lift. It executes a
  real dynamic lift only after the strict gate, reaches `0.053458 m`, loses the
  left thumb/right palm and falls. Slowing to 800 steps reaches only
  `0.027575 m`; 30 mm initial inset has zero 12/12 frames; staged 8 mm
  compression ends at 7/12 despite approximately `21 N` aggregate normal
  load. Do not continue inset, duration, lead or scalar-compression sweeps.
- [x] Export the strict-gate partial lift as a 267-frame, 13.35-second,
  1280x720 H.264 video with world view, both hands, 12 rigid-contact groups and
  telemetry. Its independent full-decode/hash/trace-pose audit passes `19/19`.
  Keep it labeled rejected rigid-contact mechanics, not tactile evidence.
- [ ] Establish a declared no-learning full-body/root/leg repositioning that
  makes a COM-covering bilateral grasp reachable with zero head, camera,
  wrist, arm, torso and support load throughout the admitted lift. Preserve
  official G1-Inspire/CarryBox physics and prohibit object-state replay.
- [ ] Derive the next right-palm underside pose from the official palm
  collision mesh and cooked CarryBox underside normal. Before running any lift,
  require settled real right-palm upward support, left side-brace contact,
  bilateral palm/thumb/index/middle/ring/little load, zero named non-hand load
  and in-limit fingers; stop immediately if this pre-lift gate fails.
- [-] Reject flat bilateral bottom support. Corrected official-state IK reaches
  `2.68 mm / 0.0840 rad` with zero queried non-hand PhysX load only while
  `25,192` head vertices overlap the box OBB. Positive head clearance raises
  the best hand error to `20.58 mm`; a high-torso posture clears the head by at
  least `0.126 m` but leaves `0.197--0.396 m` hand error.
- [-] Reject the exact lower-side/right-bottom branch. The rearward cooked
  side cell produces `21.6 mm` head clearance and `3.06/6.13 mm` hand error,
  but retains `10.14 N` on the left hand-camera base. The predeclared normal-
  roll bracket either retains accessory contact or loses left-arm
  reachability; no candidate passes all gates.
- [-] Reject flat bilateral lower-edge hook and opposite-side approach. The
  former tracks both lower-edge palm targets essentially exactly but retains
  at least `57.14 N` wrist/camera load across the paired-roll bracket. The
  latter exchanges that conflict for torso/arm collision or `0.089--0.370 m`
  reach error. Do not delete official camera collisions or resume flat-box
  pose/roll sweeps.
- [-] Reject both predeclared tilt-to-scoop geometries before dynamics. With
  the left side brace and right bottom support, the head-clear `-35 deg`
  candidate tracks the two palm poses within `0.00012/1.317 mm` but has
  `364.12 N` of left wrist/camera collision. Mirroring the roles gives no
  head-clear bilateral `5 mm` candidate; its head-clear rows retain
  `255.04--699.76 N` non-hand collision and `21.29--53.21 mm` error on the
  unreachable hand. Both scans use exact cooked points, normals and edge
  pivots and contain zero admitted candidates. Do not continue box-tilt,
  palm-roll, root-offset or accessory-deletion sweeps.
- [x] Export the two tilt directions as separate 12-second, 1280x720 H.264
  review videos. Each shows all 12 official G1-Inspire/CarryBox poses, separate
  left/right hand views and the synchronized hand error, head clearance,
  named collision bodies and non-hand load. Both exact-pose exports pass and
  both independent video audits pass all `19/19` checks.
- [ ] Before any further lift-authority test, establish a symmetric bilateral
  palm, thumb, index, middle, ring and little load topology with explicit box
  clearance from both ankles. Do not increase the live-lift lead again to hide
  posture or contact-topology failure.
- [ ] Attach all 27 frozen anatomical physical sensing surfaces per hand to
  their correct articulated palm/digit links; pass ownership, coverage,
  no-contact zero and geometry-frame checks.
- [ ] Establish a real bilateral grasp with physical load on palm, thumb,
  index, middle, ring and little groups on both hands during one stable
  interval. Do not fill unloaded patches.
- [ ] Lift and hold the real box without object-state replay; pass raw solver
  force/torque balance within the frozen dynamic boundary and render the
  synchronized world/contact video.

## Official deformable tactile mechanics

- [ ] Import the exact selected articulated-hand URDF/link topology through
  official Taccel's released robot path. If format conversion cannot preserve
  topology bit-for-bit semantically, record a blocker instead of simplifying.
- [ ] Pass one official tetrahedral-gel link probe with normal press, signed
  shear, static hold, incipient slip, release, spatial pressure/traction,
  marker, depth and RGB evidence.
- [ ] Scale the exact released soft-gel path to 27 surfaces per hand and prove
  bilateral same-run contact/load coverage.
- [ ] Run an actual same-engine articulated box lift/hold and pass integrated
  soft-contact traction versus object-dynamics balance. Cross-engine replay
  cannot close this item.

## Admission and handoff

- [ ] Produce and independently decode `master_sync`, `left_detail`,
  `right_detail`, `palm_optical`, and `force_balance` for nominal, 3x-mass and
  low-friction conditions with fixed scales and parameters.
- [ ] Stop for explicit user video review.
- [ ] Only after approval, update the immutable tactile admission record and
  hand the causal spatial stream back to Plan 06 for slip/policy/SMP/ICM and
  learned-reward experiments.
