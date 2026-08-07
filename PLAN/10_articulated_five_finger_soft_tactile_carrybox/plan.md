# Plan 10: Articulated Five-Finger Soft-Tactile CarryBox

> 2026-08-03 priority reset: stop the unfinished Taccel/TacMan and heuristic
> grasp queues. Preserve the articulated G1 assets and diagnostics as inputs.
> Continue only the official Tactile Genesis direction/four-finger work in
> `PLAN/11_demo_tg_icm_mass_ood_contact_velocity/plan.md`.

Status: active P0 physical-foundation repair from 2026-07-31.

## Outcome

Replace the impossible expectation that a single rigid SUGAR rubber hand will
conform like a human hand with a separately declared official articulated
five-finger G1 research arm and a released deformable-gel contact model. The
first positive result must be a no-learning bilateral CarryBox lift/hold whose
actual loaded anatomy, spatial soft-gel response, world motion, and object
force balance agree in one synchronized run.

This plan repairs the physical foundation only. It does not yet train a
policy, claim slip detection, or count recovery/alternative strategies.

## Frozen control and primary research arm

The control remains official SUGAR G1-29, official CarryBox physics and the
accepted Refiner behavior. Its two rubber hands are each single rigid bodies
with no finger joints. V83/V87/V88 remain the only current diagnostic sensor
evidence; their sparse loaded regions and failed primary tactile force balance
are preserved, not repaired retroactively.

The primary research arm is:

- Unitree official `unitree_sim_isaaclab`, commit
  `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc`, Apache-2.0;
- Unitree official `unitree_sim_isaaclab_usds`, Apache-2.0, whose source config
  points to `g1_29dof_with_inspire_rev_1_0.usd`;
- G1-29 body/arm topology plus the official Inspire five-finger configuration.
  The source declares 12 hand joints per side: four thumb joints and two each
  for index, middle, ring, and little fingers. Runtime USD readback is the
  authority before any experiment.

BrainCo official RevoLab at
`bc2f874a61f386712374888675e6756316d0aadc` is a secondary MIT-licensed
five-finger asset/source control. `brainco-description` is not admissible
because its current README says license information will be provided later.

## Deformable tactile source

Use official Taccel at commit
`cb23bc251b531ba6908a3788c2f91423cd543149` (MIT). Its released implementation
already provides arbitrary-URDF articulated robots, IPC/ABD contact,
tetrahedral soft gel attached to robot links, marker motion, depth and RGB.
Adapters may bind official G1/hand names and file formats; they may not replace
Taccel mechanics or tactile rendering.

Tactile Genesis remains an optional separately named marker-history channel.
It is not the deformable-contact solver. TacEx and IsaacIPC are not executable
mainline choices until usable official code and licenses exist.

## Execution

### 1. Source and asset gate

1. Acquire the exact official Unitree asset package without running its
   destructive convenience script. Record upstream revision, license, byte
   count and SHA-256 for the archive and every selected G1-Inspire entry point.
2. Independently enumerate USD joints, bodies, collisions, masses, limits and
   drives. Confirm five distinct digits on both hands and identify the exact
   wrist attachment and all load-bearing palm/digit links.
3. Audit IsaacLab v2.3.2/Isaac Sim 5.1 compatibility. Any required change must
   be adapter glue around official assets, not a regenerated or simplified
   robot.

### 2. Articulated G1 no-learning gate

1. Import one G1-29+Inspire articulation in the existing retained allocation.
   Save runtime topology readback and a world video showing both complete
   hands opening and closing each of their five digits independently.
2. Mount the frozen 27 anatomical surfaces per hand to the correct moving
   links. First prove geometry, ownership and no-contact zero; do not tune
   tactile force parameters from a CarryBox trajectory.
3. Form a bilateral box grasp with actual joint actuation. During a declared
   stable interval require physical load on palm, thumb, index, middle, ring
   and little groups on both hands. Legitimately unloaded patches stay zero.
4. Lift and hold the real CarryBox without replaying object state. Raw solver
   force and torque must close the measured box dynamics under the existing
   20% dynamic residual boundary.

### 3. Official soft-gel gate

1. Prove one controlled official Taccel gel/link press, shear, static hold,
   incipient slip and release using the selected articulated-hand link.
2. Scale the same released path to all 27 physical sensing surfaces per hand.
   Preserve deformation, normal pressure/traction, signed tangential response,
   marker motion, depth and RGB without aggregating away spatial structure.
3. Run the articulated box grasp and lift inside the admitted soft-contact
   model. Integrated soft-contact tractions and the object's simulated dynamics
   must close in the same engine/run. Cross-engine replay is diagnostic only.

### 4. Human-visible admission

For the first positive candidate, render the required five CarryBox videos:
`master_sync`, `left_detail`, `right_detail`, `palm_optical`, and
`force_balance`. The master must show world motion, both whole hands, the six
loaded anatomical groups per side, box height and force balance at the same
frame. Detail videos must retain all 27 patches without implying that every
patch must be loaded.

Then execute nominal, 3x-mass and low-friction conditions with unchanged
sensor/solver parameters, independent reconstruction, full H.264 decode and
explicit user review. Only after that review may Plan 06 consume the tactile
stream or restart policy/slip/SMP/ICM/learned-reward work.

## Stop rules

- Do not make another fixed-rubber-hand TacSL version to address anatomical
  reachability.
- Do not substitute Unitree Dex3: it has three digits and does not meet the
  five-finger objective.
- Do not use the currently unlicensed BrainCo description repository.
- Do not call PhysX aggregate contacts, a kinematic replay, Tactile Genesis
  displacement, or generated taxels a soft-gel force result.
- If the official G1-Inspire asset cannot load or the official Taccel importer
  cannot preserve its topology, report the exact compatibility blocker before
  selecting another official source.

## Current boundary

The workspace cleanup, official-source selection and official Unitree asset
gate are complete. The `1,305,090,539`-byte archive passes CRC and exact SHA-256
`06fbf145...`; all seven selected G1-Inspire files hash exactly. An unchanged
live IsaacLab/Isaac Sim import enumerates 53 joints, 65 rigid bodies and 12
finger joints on each side, with thumb, index, middle, ring and little groups
and live hand collisions. A held-root 29-second H.264 movie independently moves
all ten digit groups, fully decodes at 1280x1080/20 fps and passes the 19/19
foundation audit. Gravity disabling and root holding are declared solely for
this topology visualization, not as standing or box evidence.

The retained fixed-rubber-hand V87 control confirms the reason for the reset.
At its reviewed middle frame only 6 of 54 patches carry raw load, all on distal
fingers; neither palm center nor thumb is loaded. Its raw PhysX dynamic
force/torque residual medians are `0.1532/0.0296`, but integrated TacSL
force/torque residual medians are `3.8881/0.4802`, and it contains zero
qualifying quasi-static support frames. Therefore the underlying rigid-body
motion is explainable by PhysX contacts, while the alleged tactile force
balance is not admitted.

The unchanged Taccel soft-teddy example passes 80/80 CUDA/IPC solver steps at
one and four environments, but all exported PLY frames are bitwise static and
it saves no RGB/depth/marker stream. The official TacMan path that actually
calls `render_tactile()` fails on its first conjugate-gradient step with an
exact 16-index/14-source scatter mismatch in the pinned compiled runtime.
Therefore no deformable tactile result is admitted yet.

The next executable gate is no-learning bilateral G1-Inspire reachability and
physical CarryBox support: first load palm, thumb and all four finger groups on
both sides, then lift/hold the real object and close raw object force/torque in
the same run. Only after that passes may official Taccel gel compatibility and
spatial rendering proceed. No training is authorized by this plan.

## 2026-08-02 fixed-motion mechanics boundary

The exact official Inspire and CarryBox collision geometry is now audited
before any tactile attachment.  PhysX cooks the unchanged official CarryBox
as `convexDecomposition`; an empirical cooked-bottom envelope shows that the
earlier nominal underside targets were 8--40 mm below the actual collision
surface.  Correcting those targets produces clean bilateral hand contact, but
does not produce a free lift.

With the motion-45 upper body fixed and the robot root retreated 0.25 m, the
best clean bottom-support runs have zero head/camera/wrist contact and reach
only `0.0392--0.0440 m` apparent COM lift.  In the slow V83 run, the clean
bilateral interval's median hand vertical force is `2.214 N` versus
`4.935 N` required, and the normalized force-residual median is `0.6075`,
outside the frozen `0.20` limit.  At its highest clean bilateral step, only
left index plus right index/middle are loaded.  The box rotates/tips while the
temporary support supplies the missing wrench; this is not a hand-only
force-balanced lift and cannot admit tactile.

Moving either bottom palm 40 mm toward the box COM while retaining the 0.25 m
root retreat is not reachable: the right/left target errors are respectively
`30.94/51.50 mm`.  Moving the root forward to 0.20 m makes the split targets
reachable to `0.009/0.495 mm` and preserves bilateral contact for 348 frames,
but the box contacts `head_link` from frame zero with `782.41 N` maximum load.
Lowering the temporary support by 50 mm removes the persistent head contact
but leaves `24.61/32.67 mm` hand-target errors, a two-frame
`right_hand_camera_base_link` contact peak of `127.72 N`, and only
`0.0335 m` lift.

The v98 `head_link` attribution is now independently resolved rather than
assumed from a body-origin distance.  The unchanged hash-locked official USD
places the head collision mesh at body-frame `z=0.3248--0.5306 m`, with a
`0.5306 m` maximum vertex radius from the `head_link` origin.  At the restored
v98 state, that real mesh overlaps the CarryBox envelope.  A separate
four-step no-learning PhysX audit binds the all-robot body-view names to the
raw contact rows and reproduces active `head_link`, left-palm and right-palm
rows.  Per-row raw normal/friction vectors match the corresponding IsaacLab
filtered force-matrix rows within `1.42e-5/2.34e-6 N`.  Thus the large v98
non-hand load is a real official-geometry collision, not a ContactSensor
body-index mix-up.  This audit validates attribution only; it does not make
v98 a valid lift.

Therefore the fixed motion-45 upper-body branch is closed.  Its feasible set
does not simultaneously contain COM-covering support, bilateral reachability,
and zero non-hand contact.  The next no-learning mechanics gate must allow a
declared body/leg/root repositioning before grasp while keeping the official
G1-Inspire articulation, CarryBox asset/material, object-state non-replay, raw
force/torque audit, and `>0.10 m` hand-only lift requirement unchanged.  This
is a strategy/posture mechanics problem, not permission to tune tactile,
friction, training rewards, or sensor displays.

## 2026-08-02 dynamic side-clamp boundary

A subsequent physical re-evaluation removed two misleading shortcuts.  First,
the frame-279 official pose does not lift under native articulation drives:
with gravity on, one initialization write and no object replay, V101 has
`-0.00024 m` maximum lift relative to its declared initial state and then
settles `0.01724 m` lower.  The earlier approximately `0.11 m` frame-279 result
was state replay and remains inadmissible.  Second, zero contact force in one
static frame does not prove geometric clearance when a thin closed box shell
contains a robot collision mesh.  The head collision mesh was inside the
static box envelope in the rejected root/waist pose and acquired real load
once dynamics began.  Every future clearance gate must combine mesh overlap
or containment with named-body dynamic contact, not use a single zero-force
sample alone.

The mechanically closest current branch is the official-geometry side clamp.
V105 briefly loads all six anatomical groups on both hands for 23 steps but
lifts only `0.01029 m`.  A 30 mm palm inset (V106) reaches `0.02835 m` but loses
the all-group interval as the free box rotates away from its frozen open-loop
target.  V107 is bitwise the same stale-wrapper run as V106 and has been
rejected; it is not live tracking evidence.

The producer now has a hash-recorded causal live-box tracking option.  It reads
the current PhysX box pose only to form the next palm IK target and never
writes the box after trial initialization.  V111 preserves bilateral hand
contact for 999/1021 recorded frames with zero named non-hand load, but its
implicit `0.45 mm` vertical lead produces only `0.00188 m` maximum lift.  V112
changes only that lead to `3 mm`; it remains clean and bilateral for 999 frames
but reaches only `0.00551 m`.  V113 changes only the lead to `10 mm`.  Its
median lift-phase hand-on-robot vertical force is `-6.81 N`, so the opposite
force on the box exceeds the `4.91 N` weight, yet the maximum lift is only
`0.01192 m`; from frame 357 onward the box contacts
`right_ankle_roll_link`, peaking at `15.80 N`.  None of V111--V113 ever loads
all six anatomical groups bilaterally; the right hand is dominated by palm
and thumb, with zero maximum index and little-finger load.

This closes blind lift-lead escalation.  The next no-learning mechanics work
must change the declared pre-grasp posture/box-to-body clearance and establish
a symmetric palm-plus-four-finger-plus-thumb load topology before increasing
vertical authority.  The same run must then clear the ankle, head, cameras,
wrists, arms, torso and any support while exceeding `0.10 m` and closing the
raw object wrench.  Tactile attachment, Taccel tuning, policy training and
reward experiments remain blocked until that physical gate passes.

## 2026-08-02 posture, load-ramp and support-geometry boundary

The wide-stance side-clamp branch now has a complete lift-authority bracket.
V115 is the strongest topology-only state: bilateral contact occurs on
`1004/1021` frames and all palm/thumb/index/middle/ring/little groups are
simultaneously loaded for `664` frames, with a longest run of `663`, but the
maximum lift is only `0.007224 m`. Raising the instantaneous lead to `15/20
mm`, strengthening four-finger closure, adding thumb flexion, and replacing
the causal lead with scheduled world-Z motion do not cross `0.034556 m`; the
variants either lose the full topology, hit a non-hand body, or violate a
finger limit.

The asymmetric side/bottom search also fails the physical gate. A corrected
90-degree right-palm orientation can expose every right-hand anatomical group
over a rollout, and the widest temporary setup support raises the apparent
peak to `0.045559 m`. Thumb variants reach `0.052856/0.054853 m`, but never
load all six groups bilaterally in one frame and acquire large non-hand load
after the box tips. Increasing the live lead from `10` to `20 mm` reduces the
peak to `0.033359 m`; an anchored slow scheduled lift reaches only `0.038880
m`. The existing target called "bottom" therefore does not establish a real
upward palm-support state before lift.

A new source-recorded cosine lead ramp removes the instantaneous command jump.
On the wide-stance geometry, `20/30 mm` ramped leads reach
`0.048017/0.048832 m` while retaining bilateral contact for `977/980` frames.
At the unanchored peak the box has drifted approximately `0.133 m` horizontally
and rotated approximately `0.27 rad`. Holding the end-of-grasp world XY anchor
in V155 reduces drift, keeps named non-hand load exactly zero, and makes IK
position/rotation medians pass at `0.01966 m/0.07320 rad`, but the maximum lift
collapses to `0.014124 m`. Thus the earlier approximately 48-mm rise is mainly
physical sliding/tipping, not a stable vertical lift. The corresponding
anchored asymmetric V156 reaches only `0.010583 m`, loses bilateral contact,
and falls; at no point does it have bilateral all-six-group load.

The no-learning gate remains false: no run exceeds `0.10 m`, preserves the
required bilateral topology, excludes all named non-hand support, and closes
the raw force/torque balance together. Tactile surfaces are still deliberately
not attached. The next allowed mechanics experiment must derive the right
palm's upward load-bearing surface and pose from the official collision mesh,
not from another roll-angle label. Before any lift command, the settled state
must independently show real right-palm underside load, a left side brace,
bilateral palm/thumb/four-finger load, zero named non-hand load and in-limit
joints. A failed settled-contact gate stops the run before lift escalation.

Two separate 12.8-second H.264 review videos now preserve the decisive
negative mechanisms. Each contains the world view, both hand details and the
six rigid-contact groups, uses all `256` hash-bound sampled frames from the
exact post-PhysX body/box trace, and passes the generalized independent decode
and correspondence audit. V155 shows the clean but non-lifting side clamp;
V156 shows the alleged bottom-support geometry losing contact and falling.
These videos are rigid-contact diagnostics, explicitly not TacSL/Taccel or a
positive CarryBox result.

## 2026-08-02 static-reach success and dynamic-closure boundary

The fixed-motion upper-body reachability question is now separated from the
grasp-dynamics question.  A source-bound root/waist search against exact V180
row 300 produced 17 strict static candidates with both palm pose errors below
5 mm, head-box OBB clearance and zero named non-hand overlap.  Selected
candidate 8 (`root x/y/z = -0.14/-0.01/-0.0075 m`, waist approximately
`0.52 rad`) reaches the left and right source-relative palm targets with
`0.158/0.043 mm` position error and `3.49e-5/4.26e-6 rad` rotation error.
The exact same errors reconstruct at the first dynamic V182/V183 record.
This closes static target reach for this contact-seeded, fixed-root scene; it
does not establish a standing grasp or lift.

Two no-learning dynamics controls then reject the current closure trajectory.
Sequential V182 records 192 bilateral-contact frames (longest 73) but zero
bilateral all-six-group frames and falls below `0.4 m` at frame 368.
Simultaneous V183 is worse: 90 bilateral-contact frames (longest 35), zero
bilateral all-six-group frames and falls below `0.4 m` at frame 163.  The
right thumb carries exactly zero force in both runs.  V182 additionally leaves
the left middle/ring/little groups unloaded; V183 excites every left digit at
some point but never forms a simultaneous load-bearing topology.  Both runs
begin with zero named non-hand load, do not replay object state after
initialization, never lift, and later acquire non-hand load only after the box
has fallen.

Separate exact-trace H.264 review videos for V182 and V183 preserve the world
view, both hands, rigid-contact group forces and telemetry.  Their independent
decode, frame-count, hash and trace-correspondence audits pass.  They are
negative rigid-contact mechanics evidence, not tactile visualization.

The active mechanical blocker is therefore contact topology and closure, not
basic palm reachability.  The next bounded no-learning experiment must first
derive a right-thumb-bearing closure trajectory that preserves bilateral palm
support and reaches simultaneous palm/thumb/four-finger load before any lift
command.  It must stop if that settled-contact gate fails.  Only after that
may a free-root/leg-ground-support transition attempt the unchanged `>0.10 m`
hand-only lift and force/torque gate.  Tactile attachment and all policy,
reward, SMP and ICM work remain blocked.

## 2026-08-06 first real >0.10 m articulated lift and remaining topology failure

The official G1-Inspire side-clamp geometry now produces a real dynamic lift
without object replay.  With a 30 mm palm inset and a scheduled world-Z target,
the 0.5 kg CarryBox reaches `0.149188 m`; bilateral hand contact is present on
`211` lifted frames and `198` of those frames are free of named non-hand load.
On the clean dynamic interval, direct/raw hand force agrees and the raw
force/torque residual medians are approximately `1.8e-6` and `0.0144` in the
declared normalized units.  This establishes real lift authority, but not the
physical admission gate: the best simultaneous topology is `11/12` groups for
33 frames, always missing the left thumb, and later fall/contact produces hard-
limit and non-hand violations.  Final lift is only `0.056889 m`.

Bounded follow-ups reject three simpler explanations.  Late thumb-yaw commands
can exchange thumb contact for palm/finger contact but never produce one 12/12
frame; extending the pre-lift settle interval reduces the peak to `0.089271 m`;
and removing the world-XY anchor permits `0.17198 m` horizontal drift while the
peak falls to `0.086013 m`.  A causal box-relative terminal hold preserves the
measured 3-D tracking bias, but that bias is approximately
`[+0.0470,-0.0534,+0.0461] m`; retaining it launches the box transiently to
`0.348998 m` and then loses all contact.  The fixed hold is therefore not the
sole failure, and controller-bias tuning is closed.

The strongest run has a separate 1280x720 H.264 human-review video with the
world view, both articulated hands and all six rigid-contact groups per hand.
All 256 sampled frames are bound to the exact post-PhysX trace, and its
independent decode/hash/correspondence audit passes `19/19`.  It remains a
negative rigid-contact mechanics record, not tactile output or a successful
stable CarryBox grasp.  The next work must redesign the pre-grasp load-bearing
posture so both thumbs and both palms remain physical contacts through the
lift; no more thumb-angle, settle-time, XY-anchor or terminal-bias sweeps are
authorized by these results.

The current strict 20/20 mm inset control does not reproduce the historical
V105 all-group contact observation.  Under the current producer and the same
scheduled-Z/anchored-XY contract it reaches only `0.014055 m`, has zero 12/12
frames and zero `>0.10 m` lift frames, despite `499` bilateral-contact frames
and exactly zero named non-hand load.  The older run used a different producer
hash and remains historical diagnostic evidence only.  It cannot supply the
missing current-topology gate or justify returning to inset sweeps.

The next bounded geometry diagnostic is tied to current 30 mm trace frame 462,
where the box is already physically elevated `0.082455 m` and exactly 11/12
groups carry load, with only the left thumb absent.  It restores the recorded
robot and box state, fixes the box for diagnosis, preserves the other official
Inspire commands, and varies only the left thumb's released pitch/yaw pair in
the local bracket motivated by the earlier fixed-posture scan.  This can select
at most one candidate for a subsequent dynamic validation; it is not lift or
tactile evidence by itself.

That frame-462 diagnostic is now closed as a negative.  The corrected scanner
restores the exact source robot and box state and requires all 12 anatomical
groups, rather than accepting thumb contact alone.  All nine pitch/yaw
candidates collapse the restored topology to at most 4/12 groups, produce
`36.51--104.22 N` left-thumb impulses and leave `0.280--1.047 rad` target
tracking errors.  No candidate is admissible.  The older right/left static
scans that selected a contacting thumb despite approximately `0.2 rad` joint
error and absent fingers have been moved out of the workspace and may not be
used as evidence.

The collision-envelope audit also explains why angle-only repair fails.  At
the real elevated frame 462 the absent left thumb lies outside the valid box
side silhouette: the nearest valid envelope sample requires approximately
`+4.812 mm` tangent and `-18.575 mm` height displacement.  Applying that whole-
hand shift before grasp destroys the bilateral topology instead of restoring
the thumb, so it is not a dynamic solution.

## 2026-08-06 first strict pre-lift 12/12 gate and partial-lift boundary

The corrected CarryBox-root PCA and a staged official Inspire closure now
produce the first current-producer pre-lift state that passes every declared
settled gate.  The run
`g1_inspire_currentpca_gate_rightin_extra0p5mm_leftthumb001_lift_20260806`
has 20/20 final pre-lift frames with simultaneous bilateral
palm/thumb/index/middle/ring/little load, zero named non-hand load and finger
hard-limit error below `1e-3 rad`.  The right index and little finger remain on
the released Unitree 12-command mapping; only their declared closure magnitudes
are `0.229572` and `0.226338`, and the left thumb pitch target is `0.001 rad`
so its coupled intermediate joint is not commanded exactly at the hard lower
limit.  The object remains dynamic and is never written after initialization.

This closes the dense pre-grasp topology subproblem but not the lift.  After
the gate passes, the scheduled physical lift reaches `0.053458 m` by the
manifest metric (the review video's trace-first-frame convention displays a
`0.051634 m` peak), then loses the left thumb and right palm and returns to the
floor.  Named non-hand load remains exactly zero.  During the rising interval
near `0.031--0.042 m`, raw hand vertical force reconstructs the required box
force to approximately `7e-6 N`; failure begins when the scheduled hand target
leads the box by roughly `4 cm` and total normal load falls toward the
approximately `9.81 N` frictional support requirement for a 0.5 kg object at
dynamic friction `0.5`.  This is a real partial lift, not a successful hold.

Three bounded controls reject the remaining simple controller explanations.
Doubling the scheduled lift duration from 400 to 800 steps reduces the peak to
`0.027575 m`; beginning with a 30 mm inset fails the pre-lift gate with zero
12/12 frames; and adding a causal post-grasp 8 mm compression raises aggregate
normal load to approximately `21 N` but slides multiple fingers off the box,
ending at 7/12 and stopping before lift.  Therefore speed, deeper initial
penetration and scalar bilateral squeeze are closed for this side-clamp
posture.  The next mechanics work must introduce a genuine load-bearing
support posture rather than another inset, lead, compression or duration
sweep.

The partial-lift run has a separate 1280x720 H.264/yuv420p review video with
world view, both articulated hands, all twelve rigid-contact groups and lift,
force-residual, non-hand and limit telemetry.  It contains 267 exact trace-
bound frames at 20 fps; the independent decode/hash/pose-correspondence audit
passes `19/19`.  It is a rejected rigid-contact mechanics record, not tactile
output or policy evidence.

## 2026-08-07 load-bearing posture feasibility boundary

The next load-bearing study replaced the ambiguous legacy side target with
hash-bound points and normals from the cooked CarryBox collision geometry.  It
first exposed an important geometry error: the former left "side" target at
PCA height `+0.1275 m` lies on the upper-edge/top region at its declared
tangent coordinate, whereas the true lower side face has an outward normal
with magnitude `0.9963` along negative PCA0.  All new bottom, side and lower-
edge targets preserve their exact cooked point, normal and source hash.

Flat bilateral palm support is not feasible for this official G1-Inspire and
unchanged CarryBox posture.  A corrected official-state scan finds bilateral
bottom IK as low as `2.68 mm / 0.0840 rad` with zero queried non-hand PhysX
load, but `25,192` sampled head-collision vertices lie inside the box OBB.
Moving the box outward until head clearance is positive raises the hand error
to `20.58 mm`; placing the head above the box gives `0.126--0.326 m` clearance
but raises hand error to `0.197--0.396 m`.  These are mutually exclusive
reachability/clearance regimes, not a controller-speed failure.

The asymmetric left-side/right-bottom posture is arm-reachable only after the
left side point is moved onto the true lower cooked face.  At its rearward
valid cell, the best clear-body posture has `21.6 mm` head clearance and hand
errors of `3.06/6.13 mm`, but the left hand-camera base still carries
`10.14 N`.  A `+10 deg` palm-normal roll removes that load while increasing
the left-hand error to `33.07 mm / 0.197 rad`.  The predeclared small-roll
bracket contains no joint-reachable, camera-clear candidate, so this branch is
closed before dynamic grasp or lift.

A bilateral lower-edge hook strategy places both palms approximately `22 mm`
above the cooked bottom so the four fingers could in principle wrap under the
edge.  All nine paired palm-roll candidates reproduce both contact poses to
sub-millimeter/sub-microradian accuracy with `20.8 mm` head clearance, but
each has wrist or hand-camera load; the best non-hand load is `57.14 N`.
The official Unitree package contains base-fixed and whole-body Inspire
variants but no alternative official Inspire topology without these wrist
camera links, so collision deletion is forbidden.  A 180-degree opposite-side
approach with swapped physical face assignments also fails: the close IK
solution produces large torso/arm collision, while a clear distant stance has
`0.089--0.370 m` hand error.  Flat-box bottom, asymmetric support, lower-edge
hook and opposite-side approach are therefore closed as static admission
routes.

The next bounded mechanics hypothesis is a real tilt-to-scoop sequence: keep
the official box, robot and collision topology, rotate the dynamic box about
one physical lower edge with hand contact, insert the opposite palm/fingers
under the newly exposed bottom, then return toward level before any lift.
The first gate is static geometry with an explicitly declared edge and tilt;
it must simultaneously pass hand-pose error, head/camera/wrist/arm/torso
clearance and exact cooked-surface correspondence.  Failure closes the tilted
route; it does not authorize deleting accessories, replaying object state
during the trial, or attaching tactile sensors early.

That static tilt-to-scoop gate is now closed in both asymmetric directions.
For the original left-side-brace/right-bottom assignment, rotating the
unchanged box about the declared cooked left lower edge by `-15/-25/-35 deg`
does expose the right bottom target.  The strongest head-clear row at
`-35 deg` reaches the left and right palm poses within
`0.00012/1.317 mm` and `7.92e-7/3.45e-5 rad`, but the official left wrist
pitch/roll and hand-camera geometry intersects the box and carries
`364.12 N` of queried non-hand normal load.  Thus this is a reachable pose,
not an admissible support posture.

The role-mirrored control freezes a cooked left-bottom target, a cooked right
lower-side target and the first sampled right-side cell above the adjoining
lower oblique face as its pivot before behavior.  Positive
`15/25/35 deg` tilts do not remove the conflict.  None of its 12 rows passes;
the three head-clear rows retain `255.04--699.76 N` of right wrist/camera load
and leave the other hand `21.29--53.21 mm` from target.  A closer bilateral
row reaches `0.678/0.00025 mm` but overlaps the head with 12,384 sampled
vertices and still has `202.37 N` non-hand load.  The failure is therefore not
specific to which hand is assigned the side brace.

Two separate 12-second, 1280x720 H.264 records preserve all 12 candidates in
each direction.  Every frame uses the exact reconstructed official body pose;
the maximum reconstruction errors are zero for hand position and below
`6.73e-8 m` for box position.  Each movie shows the world scene, separate
left/right hand views, official camera/wrist geometry, hand-pose error, head
clearance, named active collision bodies and non-hand load.  Both independent
full-decode/hash/metric audits pass `19/19`.  These videos are negative static
mechanics evidence, not dynamics, grasp, lift, tactile or policy output.

Flat bottom support, side/bottom support, bilateral lower-edge hook,
opposite-side approach and both tilted asymmetries now all fail the same
immutable static admission requirement under the licensed official
G1-Inspire and CarryBox topology.  No further inset, root, roll, tilt or
camera-collision deletion sweep is scientifically authorized.  Plan 10 is
blocked at official articulated CarryBox mechanics: the only documented
secondary hand source has unlicensed assets and cannot be integrated, while
sensor attachment remains prohibited until a real bilateral load-bearing
grasp passes.  The retained allocation remains active for review and an
explicitly authorized next mechanical source or topology decision.
