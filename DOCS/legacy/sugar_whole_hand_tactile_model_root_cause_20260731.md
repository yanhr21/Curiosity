# Whole-hand tactile installation and model root cause

Date: 2026-07-31
Status: active technical record; it does not supersede the immutable 2026-07-29 standard

## Result first

The current sparse CarryBox display is mainly a real contact-distribution
result, not a plotting omission.  The exact 660-step trace contains 54
physical load-bearing elastomers, exactly 500 SDF taxels per elastomer,
signed two-axis shear, both official R15 camera modules, and same-Isaac-
timeline Tactile Genesis marker history.  During the 300-frame continuous
bilateral-contact interval, TacSL identifies 95.57% of PhysX load-bearing
patch-frames with 100% precision.  At source frame 394, PhysX and TacSL select
exactly the same six patches.  The box is physically carried by those sparse
regions even though the rendered rigid hand mesh appears broadly adjacent to
the box.

This is not a complete sensor pass.  Seventy-five PhysX load-bearing
patch-frames have no TacSL taxel response.  They account for only 0.221% of
the integrated body-level load in the trace, but isolated boundary/release
events reach 7.647 N.  A discrete point-sampled penalty field can miss a
small contact located between taxel centers even when its continuous physical
patch collider is loaded.  The immutable correspondence gate therefore
remains failed.

The force-balance result is also split.  The raw PhysX contact wrench is
consistent with the box dynamics: at the dense-contact frame 356 its smoothed
force residual is 0.822 N, or 8.89% of box weight; at frame 394 it is 0.566 N,
or 6.12%.  The TacSL aggregate does not close the same balance because its
released shear channel is an instantaneous relative-velocity penalty rather
than solver static friction.  This is a sensor-model limitation, not evidence
that PhysX fails to support multiple contacts.

Exact equality with the PhysX constraint wrench is not part of the released
TacSL model and is impossible in a perfectly sticking contact under its
published equation.  TacSL sets

```text
F_n = k_n d
F_t = -v_t / ||v_t|| min(k_t ||v_t||, mu ||F_n||).
```

Therefore `v_t -> 0` implies `F_t -> 0`, while the rigid-body solver can still
apply nonzero static friction to support gravity.  A failed TacSL-versus-`mg`
closure test does not show that a patch is absent; it correctly shows that the
released observation model is not yet a calibrated load-bearing tactile
wrench under the frozen standard.

## Pinned upstream evidence

- Clean IsaacLab v2.3.2 source:
  `/public/home/yanhongru/reference_upstreams/IsaacLab-v2.3.2`, commit
  `37ddf626871758333d6ed89cf64ad702aef127d0`.
- Original official TacSL branch:
  `/public/home/yanhongru/reference_upstreams/IsaacGymEnvs-TacSL`, commit
  `f769e39fe5605324e0c6b3eb04ca7b8a7681f4e0`.
- Tactile Genesis:
  `external/tactile-genesis`, commit
  `de2bcc998dce45aaac93c6817912380c5954ab38`.
- Taccel:
  `/public/home/yanhongru/reference_upstreams/Taccel`, commit
  `cb23bc251b531ba6908a3788c2f91423cd543149`.

Primary public sources:

- TacSL paper and project:
  <https://iakinola23.github.io/tacsl/static/TacSL_paper.pdf> and
  <https://iakinola23.github.io/tacsl/>.
- Official IsaacLab sensor source:
  <https://isaac-sim.github.io/IsaacLab/develop/_modules/isaaclab_contrib/sensors/tacsl_sensor/visuotactile_sensor.html>.
- Original official TacSL code branch:
  <https://github.com/isaac-sim/IsaacGymEnvs/tree/tacsl>.
- Tactile Genesis paper, project, and repository:
  <https://arxiv.org/abs/2606.22332>,
  <https://neuroagents-lab.github.io/tactile-genesis/>, and
  <https://github.com/neuroagents-lab/tactile-genesis>.
- Taccel paper and repository:
  <https://arxiv.org/abs/2504.12908> and
  <https://github.com/Taccel-Simulator/Taccel>.
- TacEx paper: <https://arxiv.org/abs/2411.04776>.
- NVIDIA PhysX persistent-contact-manifold and contact-report documentation:
  <https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html>.
- Isaac Sim contact-sensor documentation, including per-contact position,
  normal and impulse records:
  <https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_physics_contact.html>.
- NVIDIA forum discussion distinguishing contact forces from a summed
  six-axis wrench:
  <https://forums.developer.nvidia.com/t/relationship-between-contact-sensor-and-6af-sensor/327474>.
- Official Unitree teleoperation repository showing G1-29 plus separately
  selected dexterous end effectors:
  <https://github.com/unitreerobotics/xr_teleoperate>.
- Official BrainCo assets and G1 integration for an articulated five-finger
  alternative:
  <https://github.com/braincotech/brainco-description> and
  <https://github.com/BrainCoTech/unitree-g1-brainco-hand>.

PhysX documentation is explicit that PCM generates and recycles a contact
manifold and may generate fewer tuples than legacy collision detection.  A
solver manifold is therefore not a dense pressure raster.  It nevertheless
supports multiple simultaneous contact pairs and points.  The current trace
demonstrates that capability directly: frame 356 contains ten independently
loaded anatomical patches and 559 active TacSL taxels.

## What actually went wrong during installation

The investigation found separate faults that had previously been conflated.

1. The original unsensed hand collision shell initially remained an active
   contact owner.  The box could physically touch that shell while bypassing
   the new elastomer patches.  The current mount disables the original shell
   and makes the 27 patches per hand the only exterior hand-contact owners.
2. Early optical mounts drifted or used stale transforms.  The current center
   R15 modules use the official v2.3.2 Fabric/XformPrimView path and passed the
   independent moving mount and camera-freshness tests.
3. An early patch-frame D6 fixture rotated or translated with the selected
   elastomer, invalidating commanded shear.  The accepted controlled fixture
   is world-anchored and advances only through native PhysX drives.
4. Early local tangent axes were inverted.  The original TacSL repository
   explicitly uses a sensor-specific normal and two signed tangent axes.  The
   local IsaacLab adaptation now binds each taxel frame to its sampled physical
   triangle and preserves the released signed X/Y convention.
5. The last remaining force-balance failure compared different physical
   quantities: PhysX contact-constraint impulses versus the released TacSL
   SDF/velocity penalty estimate.  This is the present root cause, not another
   missing sensor mount.

## CarryBox sparsity and whole-hand reachability

The official SUGAR asset is a 29-DoF G1.  Its action list ends at the two
wrist-yaw joints; `left_rubber_hand` and `right_rubber_hand` are single rigid
bodies with no finger joints.  The 27 elastomers on a hand therefore move as
one rigid open-hand shape.  Rendering proximity cannot make its palm, thumb
and four fingers independently conform to a box face.

The actual trajectory provides the following upper bound:

- maximum simultaneous response: ten patches and 559 taxels at source frame
  356;
- left hand at that frame: two palm patches plus index, middle, ring and
  little distal patches;
- right hand: index, middle, ring and little distal patches;
- maximum anatomical groups simultaneously present on both hands anywhere in
  the trace: four of six, namely the four non-thumb digits;
- no frame contains palm, thumb, index, middle, ring and little contact on
  both hands.

Exact signed-distance evaluation against the immutable CarryBox exterior at
frame 356 places the closest left/right thumb taxels 5.312/4.152 mm outside
the box and the closest right-palm taxel 5.203 mm outside.  The translations
needed to close these gaps point in opposite directions on the two hands, so
one box translation cannot repair them.  An optimistic plane-orientation
search that ignores arm limits and non-taxel hand collisions still requires
at least 5.475 mm left and 5.612 mm right depth spread to place all six
anatomical groups against one planar face.  That exceeds both the 0--2 mm
frozen validation range and the 4.9-mm authored gel stand-off.

Consequently, broad palm plus four-finger contact is already physically
possible and observed, but bilateral palm plus all five digits is not a
credible shallow-contact target for the current rigid rubber hands.  The
non-degraded ways to pursue it are either a separately declared articulated
five-finger G1 hand, or a real soft-gel/soft-skin model able to conform and
spread contact.  The first changes the official SUGAR robot and action space;
the second changes the contact model.  Neither may be silently described as
the unchanged SUGAR-29 control.

## Exact model boundaries

### Released TacSL

TacSL produces spatial SDF penetration and an instantaneous velocity-based
signed shear estimate.  It is useful for contact localization, loading,
motion, and slip cues.  It has no per-contact tangential displacement state,
so its shear does not persist during ideal zero-slip static hold.  The TacSL
paper also distinguishes its tactile rendering/field from the simulator's
rigid contact solver.

The original TacSL code additionally demonstrates that the tactile normal is
asset-frame-specific rather than universally local Z.  IsaacLab v2.3.2 uses a
constant local frame suitable for the released planar asset.  Whole-hand and
curved R15 geometry therefore require a geometry-bound frame adapter; changing
the geometry frame does not change the released force equation.

### Tactile Genesis

The official `KinematicTaxel` is also a spring/damper estimate and its shear is
velocity-based, so replacing TacSL with it does not create quasi-static shear
force closure.  `ElastomerTaxel` instead keeps material-point entry anchors
until release and yields persistent signed marker displacement.  The current
same-Isaac adapter ports that history equation around the exact 54 physical
meshes.  It must always be named marker displacement in metres, never Newtons.

### Taccel and TacEx

Taccel is a serious high-fidelity soft-gel simulator, but the released system
is a standalone Warp-IPC/ABD dynamics stack rather than an IsaacLab sensor
plugin.  Substituting it would change the SUGAR physics baseline; replaying
Isaac poses through it would be cross-engine replay.  Its public interface
primarily exposes gel deformation/marker/depth/RGB rather than a drop-in
Isaac per-taxel static-force channel.

TacEx combines GIPC soft-body simulation and tactile rendering inside Isaac
Sim, but no usable official repository was found with the paper.  It cannot be
faithfully integrated by inventing a local replacement.

## Correct non-degraded channel contract

The currently implementable, source-faithful sensor package is:

1. released TacSL per-taxel normal and signed XY instantaneous shear fields;
2. Tactile Genesis per-taxel signed XY history-bearing marker displacement;
3. official geometry-fixed R15 RGB and depth on both palms;
4. raw PhysX force/torque and object `mg` as an audit-only solver channel.

These channels must remain separate.  Raw PhysX contacts may not be spatially
distributed to fabricate taxels.  TG displacement may not be fitted with one
scalar and relabeled as Newtons.  The offline V88 diagnostic already showed
that such a scalar fit has large relative error and is not a valid force
calibration.

## Current evidence and stop boundary

The all-patch V89 probe was stopped and is not an active result.  The retained
evidence for the present question is the unmodified 660-step V87 CarryBox
trace plus the read-only body/taxel correspondence analysis at
`standard_sim/sparse_contact_root_cause_v1.json`.  The two short synchronized
review clips cover source frames 340--380, including the densest loaded frame:

```text
videos_current_actual_carrybox_clean/
  actual_dense_support_frames_340_380_world_taxels.mp4
  actual_dense_support_frames_340_380_force_balance.mp4
```

Policy, reward and slip use remain blocked.  The current path fails the frozen
spatial-correspondence and TacSL force-balance gates, and the current rigid
hand cannot justify a bilateral palm-plus-five-digit load claim.  A future
experiment must declare either a sensor-model correction or an articulated
hand research arm before execution; repeatedly pressing all 54 installed
patches does not resolve either root cause.
