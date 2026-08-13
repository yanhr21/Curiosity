# Anatomical-27 TacSL installation root cause

Date: 2026-07-31

## Decision

The official IsaacLab v2.3.2 TacSL installation is operational. The rejected
whole-hand results were caused by project integration and calibration-fixture
errors, not by an unavailable TacSL runtime.

The current repair keeps 27 physical compliant patches per hand, 20 x 25
normal plus signed-XY-shear taxels per patch, and one official R15 optical
module on the center palm of each hand. No contact label, aggregate wrench,
whole-hand hull, generated taxel field, interpolation, or shadow sensor is
admitted.

## Direct evidence

1. A clean IsaacLab v2.3.2 checkout at commit
   `37ddf626871758333d6ed89cf64ad702aef127d0` passes the released
   `test_sensor_rgb_forcefield` on the retained H200 allocation. The test
   returns nonblank official R15 RGB/depth and nonzero penetration, normal
   force, and signed two-axis shear.
2. In the exact released single-R15 scene, the clean official sensor and the
   isolated workspace sensor produce elementwise-identical penetration,
   taxel positions, RGB, and depth. This rules out asset loading, SDF object
   creation, camera rendering, and taxel sampling as the whole-hand failure.
3. The former whole-hand G1 composition retained the imported
   `left/right_rubber_hand/collisions` subtrees while adding 54 new patch
   colliders. The box could therefore be supported by an unsensed parallel
   collision shell. This exactly explains dense-looking hand/box contact with
   sparse or zero spatial TacSL activation.
4. The former optical mount attached the R15 elastomer and camera tip to the
   hand with two independent constraints. Their measured relative drift was
   `1.0799 mm / 0.172 deg`, so RGB/depth and force taxels were not guaranteed
   to describe one rigid sensor at the same frame.
5. The V23--V26 controlled fixture used a D6 drive while the articulated hand
   continued to move under joint drives. The selected patch translated by
   roughly `30--44 mm`; the D6 tracking error reached `0.3497 mm` and the
   rotation error `2.97 deg`. Its shear leakage and sign failures therefore
   mix sensor behavior with a moving-fixture error.
6. The workspace per-triangle surface-frame adapter originally chose the
   positive mesh-grid tangent, reversing both signed tangent directions
   relative to the released negative grid-axis convention. The tangent sign
   is corrected. The physical surface-normal adapter remains intentional
   because the released R15 constant local-Z frame is approximately 90
   degrees from the actual gel surface normal and cannot satisfy the frozen
   anatomical surface-frame gate.
7. The first surface-frame repair still projected the total
   `F_n + F_t` vector into the curved taxel frame. A zero-tangential-velocity
   normal press then reported up to `0.10796 N` of integrated "shear" even
   though the exact TacSL `v_t` and `F_t` were both zero. The same projection
   also reported compression with a negative local-Z sign. This was a channel
   decomposition error, not real friction.

## Implemented repair

- Deactivate only each imported hand's original `collisions` subtree in the
  sensorized robot variant. Preserve the official hand rigid body, mass,
  inertia, joints, and visible anatomy. The 54 physical elastomers are now the
  only exterior contact owners.
- Fix each optical R15 tip/camera directly to its physical center-palm
  elastomer using the hash-bound official relative transform. Filter only
  patch-tip and patch-patch self-collision.
- Use physical surface-normal taxel Z with the released negative tangent-grid
  convention. Report the model's documented non-negative
  `F_n = k_n * depth` compression magnitude directly. Project only TacSL's
  signed friction traction `F_t` into taxel X/Y; normal pressure may never
  leak into shear.
- Expose separate raw PhysX patch/probe and original-hand/probe views. These
  are independent audit signals and never create or fill a TacSL taxel.
- Retire D6 calibration. The successor calibration path directly commands an
  external SDF probe while explicitly clamping the robot state. State writes
  are calibration-only and remain forbidden in continuous CarryBox.
- Require both a pre-contact zero phase and a post-contact physical separation
  phase. A baseline recorded only before loading is not release evidence.

## Current direct diagnostic boundary

The first direct-pose center-palm run after correcting only the normal channel
produced strictly increasing `0.14424/0.25292/0.50001/0.99999 N` integrated
loads for commanded `0.25/0.5/1/2 mm`, exact
`normal_force = stiffness * penetration`, `390/472/500/500` active taxels, and
monotonic official R15 RGB/depth deformation. It also exposed the zero-motion
shear contamination described above. Therefore that archive is retained only
as the diagnostic that found the second channel bug; it is not a controlled
probe admission. Fresh static and dynamic evidence must bind the corrected
normal-plus-friction-only source.

The fresh corrected-source evidence now closes the static and bilateral
center-R15 gates:

- `static_admission_job209442_v28_compression_scalar` passes all `11/11`
  independent checks over 54 physical patches and binds sensor SHA-256
  `b52d7fcf20d183e27ea34115eafd3f9f400893f749df9f266ff6160bfa37e786`.
- `direct_pose_center_r15_v30_mirrored_camera_job209442` passes all `27/27`
  independent checks separately on the left and right center patches.
  Zero-tangential-velocity normal holds have exactly zero reported shear;
  signed U/V friction reverses with the commanded motion; the Coulomb bound,
  spatial footprint, monotonic load/area, physical release, stationary mount,
  and R15 RGB/depth response-and-return checks all pass.
- The synchronized bilateral H.264 video contains `228` exact source rows,
  decodes at `1920 x 1080 / 20 fps`, and passes its independent source,
  admission, codec, frame-count, and bilateral-row-alignment audit. Human
  frame inspection found and corrected a mirrored right close-up camera-axis
  error before retaining V30.

This is a real controlled sensor calibration, but it is still only the two
center patches. It does not by itself close all-54 controlled response,
continuous CarryBox, force balance, held-out physics, or final human review.

## Required proof before a positive video

The repair is not accepted from source inspection alone. It must pass:

1. [passed] a fresh 54-patch independent static audit, including empty
   original-hand collision inventories and exact optical patch-tip exclusion;
2. [passed] bilateral center-palm direct-pose normal, signed-X/Y shear,
   release, R15 response, and tip-to-elastomer stability checks;
3. raw contact ownership, all-54 controlled response, and held-out validation;
4. continuous nominal, 3x-mass, and low-friction CarryBox with synchronized
   world video, bilateral 27-patch maps, optical RGB/depth, and force balance;
5. independent decode and explicit human review.

Until these gates pass together, all prior whole-hand videos remain rejected
diagnostics and no tactile policy, slip, reward, SMP, ICM, CHORD, recovery, or
strategy claim is reopened.
