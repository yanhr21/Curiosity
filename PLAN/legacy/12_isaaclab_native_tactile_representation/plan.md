# Plan 12: SUGAR CarryBox Whole-Hand Native Tactile Visualization

## Objective

Build a readable IsaacLab-native tactile representation on the sensorized
SUGAR G1 CarryBox scene. The final evidence must show the complete left and
right hands, not two detached R15 fixtures:

- 27 physical tactile patches per hand;
- twelve palm patches in a fixed `4 x 3` arrangement;
- thumb, index, middle, ring, and little fingers;
- proximal, middle, and distal patches on every finger; and
- a raw `20 x 25` normal-force and signed-XY-shear grid on every patch.

The same implementation must expose genuine feedback during a successful
grasp and during a failed grasp. This phase does not test whether tactile
improves training and does not resume demo following or Curiosity.

## Native sensor boundary

Every displayed patch must be a physical elastomer attached to the SUGAR G1
hand and read through IsaacLab's `VisuoTactileSensor`. Preserve raw
penetration, signed local-Z force, signed local-XY shear, taxel world
positions/quaternions, contact normals, relative tangential velocities,
source step, and time. The two geometry-fixed center-palm patches also
preserve official R15 RGB/depth. The signed local-Z field is displayed without
absolute value or sign clipping: negative is red, positive is blue, and zero
is white.

Rigid contact labels, body wrenches, object state, generated taxels, projected
contact points, repeated R15 images, and zero-filled missing patches cannot be
used as tactile data. Object pose and raw PhysX contacts may appear only as
clearly labelled physical references; they never fill, rescale, or replace a
taxel.

## Paired CarryBox outcomes

Use one SUGAR CarryBox collector and one tensor/layout contract for both runs:

1. `successful_grasp`: the physical G1 hands establish contact, lift the box,
   and retain it during a visible hold;
2. `failed_grasp`: the same hands first establish real contact and then lose
   contact/support through a physical release, slip, or failed closure.

Keep one additional failed-closure example when it adds a visibly different
case: one hand touches the box and reports native taxels while the other hand
misses, so the box is not lifted. This is a visualization example, not a new
sensor gate.

The failure is never produced by editing sensor arrays. Policy learning is not
part of this plan; a direct physical grasp controller is allowed only if it
acts through the simulated robot and leaves the box fully dynamic.

## Readable anatomical visualization

Success and failure each get a separate `2560 x 1440` H.264 video with the
same fixed scale and layout:

- top half: continuous SUGAR G1 CarryBox world camera showing robot, both
  hands, and box;
- bottom left: the complete left hand;
- bottom right: the complete right hand;
- each hand is visibly hand-shaped: four upright fingers above the palm and
  the thumb on the outside, with distal/middle/proximal patches kept separate;
- all twelve palm patches appear below the fingers with four patches across
  and three patches from fingers to wrist;
- every patch remains a distinct `20 x 25` map; zeros remain visible; and
- signed shear arrows and per-patch active-taxel counts stay synchronized with
  the world frame.

The two hands must appear together in the main video. Separate detail or palm
optical videos may supplement it, but cannot replace it.

## Direct review

Before presenting either video:

- check that both videos play normally;
- check that each world frame and tactile frame move together;
- check that both complete hands and all 27 patches per hand remain visible;
- check that the successful run visibly lifts and holds the box while tactile
  feedback remains present; and
- check that the failed run visibly loses support while the native tactile
  response changes or disappears.

Completion still requires the user to verify that visible hand/box contact and
the corresponding anatomical patch response agree.

## Physical-balance interpretation

Judge physical motion and native tactile output separately. Reconstruct the
box force at the native `5 ms` PhysX substep and compare the four-substep
impulse with the COM velocity change. Also compare the sum from the 54
sensorized patch bodies with the all-robot contact wrench so that an
uninstrumented hand, arm, or body collision cannot masquerade as tactile
support. Report normal and friction components separately.

TacSL taxel output must be reconstructed through the official penalty law, but
agreement with that law is not force calibration. Sum the raw taxel vectors in
world coordinates and compare them with the independent PhysX box wrench. If
the spatial contact correspondence passes but the wrench does not, retain the
sensor as a spatial simulated tactile field and report the force calibration as
failed.

## Deliverables

- one reusable whole-hand recorder/representation path;
- successful and failed SUGAR CarryBox raw traces;
- `successful_carrybox_whole_hand_tactile.mp4`;
- `failed_carrybox_whole_hand_tactile.mp4`;
- `failed_closure_carrybox_whole_hand_tactile.mp4`;
- `force_kinematics_friction_complete.mp4`, using the native physics clock and
  covering pickup, carry, placement, release, and post-release zeros;
- left/right detail, bilateral R15, and simple support-force videos for each
  outcome; and
- an experiment README pointing only to these final videos.

Keep one executable reproduction entry point that starts from the official
SUGAR checkpoint/task and writes the raw trace before rendering. The canonical
successful bundle contains one 660-frame raw trace with all four `5 ms`
physics substeps, not separate ordinary and force-audit copies. Document the
online-simulation versus wall-clock-real-time distinction and do not call a
new object supported until its SDF and actual contact-body sensor placement
have been checked.

Do not add routine hashes, provenance manifests, or extra admission machinery
to this visualization task.

The former detached dual-R15 result remains a diagnostic under legacy/archive
and is not a final whole-hand deliverable. Plans/TODOs 04--11 remain legacy.
The historical demo-following, Curiosity, and tactile-training experiment
packages remain capped at five in total.
