# SUGAR conformal whole-hand TacSL skin protocol — 2026-07-29

## Decision boundary

This protocol replaces the rejected whole-hand convex-decomposition collider.
It does not change the official SUGAR policy, action space, robot joints,
rigid bodies, masses, inertias, CarryBox asset, PhysX engine, or TacSL SDF
penalty equations. It is evaluation-only until every gate below passes.

The only admitted runtime is the existing SUGAR CarryBox environment in the
matching IsaacLab/Isaac Sim/PhysX stack. Genesis, MuJoCo, shadow sensors,
offline optical replay, per-frame physics restoration, binary contact labels,
aggregate hand wrenches, and hand-written tactile values are forbidden.

The current H0/H1 convex-decomposition paths remain frozen negative evidence:

- H0 has `4,426/4,427` exact-source taxels, but their distance to the exposed
  physical hull union reaches `1.294/1.568 mm`; the exposed hull union itself
  strays as far as `11.734/10.814 mm` from the exact hand source.
- In the continuous motion-45 run, significant raw PhysX contact occurs on
  frames where same-frame direct TacSL is zero. The integrated TacSL load also
  fails the CarryBox weight/acceleration check.
- H1 fails the all-taxel static gate at `2.830/2.445 mm`.

None of those assets, runs, or videos can be reused as a positive result.

## Frozen official R15 reference contract

The local official R15 asset and exact released spawner are the structural
reference:

- the source USD has one non-colliding visual sampling mesh and one separately
  extruded collision mesh;
- the released spawner binds compliant-contact stiffness `10.0`, damping
  `1.0`, static friction `0.5`, dynamic friction `0.5`, restitution `0.0`,
  and does not author contact/rest offsets;
- the released `5 x 10` taxel grid is at most `0.136817 mm` from its physical
  collision surface; a denser `20 x 25` grid is at most `0.149765 mm` away.

The reference artifacts are:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260729_cleanup/wholehand_tacsl_physical_20260728/official_r15_contract/
  official_r15_physical_contract_v2.json
  official_r15_spawned_contract_v1.json
  official_r15_surface_coincidence_v2.json
```

This reference does not imply that a whole-hand adapter is an official R15
hardware product. The admitted claim remains “high-fidelity simulated
whole-hand tactile using the official TacSL force core.”

## One frozen repair: P1 finite-area conformal taxel prisms

P1 is the only candidate admitted by this protocol. No convex-decomposition
capacity sweep, offset sweep, taxel deletion, atlas shift, force rescaling, or
parameter search is allowed.

For each hand:

1. Reproduce the existing deterministic `96 x 80` one-sided projected atlas
   from the exact official `left/right_rubber_hand.STL`.
2. Require the unchanged float32 taxel-center hashes:

   - left:
     `bf100b8ab45b24af585d957401ba8f4609bc4a4f38c7b3aa6e4ddfb71c332633`
   - right:
     `13480daad631eb9b7e17e70d0b31acbc75bf86e85513ca465ab8ad5fe49fcece`

3. Preserve all `4,426/4,427` taxels. The unsupported/excluded index list must
   be empty.
4. Give every taxel exactly one regular-hexagonal convex prism on the existing
   rubber-hand rigid body:

   - prism outer-face center: the exact taxel position;
   - outer tangent axes: the taxel frame’s signed X/Y axes;
   - outer normal: opposite the taxel frame’s `+Z`; taxel `+Z` continues to
     point inward, in the direction of the force on the gel;
   - outer hexagon area: `1.18138624e-6 m^2`, the already declared represented
     area per spatial TacSL taxel;
   - derived circumradius:
     `0.0006743261642464126 m`;
   - inward prism thickness: exactly `0.001 m`;
   - one explicit `convexHull` collision mesh per taxel, with no new rigid
     body, joint, mass, or inertia.

5. Disable the converted whole-hand collision and do not compose either
   rejected H0/H1 hull asset.
6. Bind the official R15 spawned material values exactly: stiffness `10.0`,
   damping `1.0`, static/dynamic friction `0.5/0.5`, restitution `0.0`,
   average combine modes. Do not author contact/rest offsets.
7. Keep the non-colliding exact hand mesh as the source from which the
   projected TacSL centers and signed per-taxel frames are regenerated.

Each physical shape therefore belongs to exactly one taxel. A raw contact on
the replacement hand collider has a declared spatial taxel owner; there is no
convex hull spanning a finger gap or unrelated hand region.

## Gate A — independent static geometry

The asset is rejected unless an independent reader verifies all of the
following:

- exact source STL hashes, atlas shape, taxel counts, center hashes, face
  hashes, and empty exclusion lists;
- exactly `4,426/4,427` enabled convex-prism shapes;
- each shape is finite, convex, non-degenerate, has the frozen area,
  circumradius, thickness, and one unique taxel owner;
- every taxel center lies on its owned outer collision face to at most
  `1e-7 m`;
- every outer hex vertex is within `0.8 mm` of the exact hand source surface;
- no shape or ancestor adds a rigid body, joint, mass, inertia, articulation,
  or drive;
- source-to-taxel anatomical coverage remains nonzero on palm, thumb, index,
  middle, ring, and little finger for both hands;
- the asset and manifest are hash-bound and the auditor reconstructs every
  prism from the manifest rather than trusting producer summary fields.

A static pass is geometry evidence only, not tactile response.

## Gate B — composed SUGAR stage

One environment must initialize and reset on retained Slurm GPU allocation.
The independent stage audit must verify:

- official SUGAR robot/CarryBox/runtime asset hashes;
- unchanged rigid-body, mass, inertia, joint, drive, and action contracts;
- old converted and H0/H1 collision paths are absent or disabled;
- every P1 prism is enabled and bound to the exact official-reference
  compliant material values;
- no P1 collision has authored contact/rest offsets;
- the CarryBox collision remains the SDF queried by TacSL;
- direct outputs remain float32 taxel-resolved penetration, normal force, and
  signed two-axis shear; no binary or aggregate field is read as tactile.

This stage pass is wiring evidence only.

## Gate C — controlled per-zone response

With no policy or learning update, run a same-engine controlled probe on each
of the twelve `(side, anatomical zone)` targets.

For each target:

- archive baseline, `0.25`, `0.5`, `1`, `2`, and `4 mm` normal presses;
- archive matched positive and negative tangential motion while normal depth
  is held;
- preserve raw PhysX contact point/force, exact taxel centers/frames, direct
  TacSL penetration/normal/signed shear, world RGB, and the spatial pressure
  and shear maps on the same source step.

Required checks:

- baseline is zero before contact;
- normal depth and integrated normal force are positive and monotonic with
  press depth;
- the active patch contains the intended taxel and zone;
- positive/negative tangential motion reverses the corresponding signed shear
  component without changing channel order;
- every significant raw contact on a P1 prism has a same-frame spatial TacSL
  response owned by that prism or a geometrically overlapping neighboring
  taxel;
- release returns the direct field to zero;
- no reset, replay, threshold, absolute-value repair, or post-hoc frame shift
  is used.

Any zone failure rejects P1.

## Gate D — uninterrupted official CarryBox correspondence

Only after A–C pass, collect one fresh uninterrupted motion-45/seed-4263
official Refiner rollout for at least 600 source steps. It must start from the
normal SUGAR reset and apply one live official action per step. No state
restoration or shadow replay is allowed.

The independent audit must verify:

- one world frame and one direct tactile frame per source step;
- reconstructed same-frame hand/taxel poses and official SDF force arithmetic;
- no significant raw contact (`>= 1 N` per hand/frame) on the P1 collider with
  zero same-frame direct TacSL;
- raw-contact-to-owned-taxel distance is bounded by the frozen taxel
  circumradius plus `0.15 mm`;
- palm and thumb are evaluated explicitly rather than hidden in a whole-hand
  aggregate;
- integrated direct normal/shear, raw PhysX contact, object mass, gravity,
  vertical acceleration, and object motion are reported together. The result
  must close the box weight/acceleration relation within a predeclared
  measurement tolerance; values may not be rescaled afterward.

If the uncalibrated official TacSL parameters fail the load relation, report
that exact negative. A later calibration would require a separate protocol;
this protocol does not authorize one.

## Human-review stop

After Gate D, create separate H.264/yuv420p videos:

- world-wide official CarryBox;
- left pressure;
- right pressure;
- left signed shear X/Y;
- right signed shear X/Y;
- one combined comparison with world-wide RGB across the full top row and
  large bilateral tactile panels below.

The videos must show the full uninterrupted timing and labeled contact-focus
clips. They are audit evidence, not policy inputs. Stop after publishing the
videos and numeric report for human inspection. Do not train from P1 until the
user accepts this sensor audit.

## Failure policy

P1 either passes all four gates or remains rejected. Do not silently:

- delete sparse/problematic taxels;
- merge prisms across hand gaps;
- offset the atlas or contact surface;
- tune stiffness, friction, SDF, or thresholds;
- substitute binary contact, a hand wrench, a synthetic force map, Genesis,
  or a replayed optical sensor;
- call a static, stage-only, controlled-only, or negative CarryBox result a
  correct installed tactile sensor.

## Executed result — P1 rejected at Gate C

Gate A and Gate B pass only as geometry and wiring checks. The authoritative
post-capacity Gate-B audit is
`p1_composed_stage_job206425_v1/composed_stage_audit_v3.json`, SHA-256
`8632159b5074b9eb6903dc941d2cdc083d08f245662805465bfe0a0f3457fd15`.

The first mandatory Gate-C trial, `left/palm`, is complete for `350` source
steps and independently fails:

- the intended left-hand normal, signed shear, and penetration fields remain
  zero at every `0.25/0.5/1/2/4 mm` target hold;
- the non-target right hand instead responds on `350/350` frames, including
  baseline;
- the one positive raw left contact is `4.976162 mm` from the nearest left
  taxel, beyond the frozen `0.824326 mm` limit, with no same-frame direct
  response;
- the official normal and channel-last signed-shear calculations reconstruct
  from the raw archived quantities with maximum error `0.0 N`, so the defect
  is not a visualization or channel-processing artifact;
- the fixture saturates force on `349/350` frames and torque on `49/350`.

The independent report is
`p1_zone_response_job206425_v3/
independent_partial_negative_audit_v1.json`, SHA-256
`3e5fbf43ee0c9060d519a1c3b5a7fd0ca2544c0e4d70fc45fcf2336aaba858da`,
with `passed=false`. The reset error that occurred before trial 1 cannot erase
the already archived failure of a mandatory target. Under the frozen
any-zone-failure rule, P1 is rejected; the remaining zones are not run as an
acceptance path.

Gate D and training are prohibited. Browser-compatible synchronized negative
videos and the exact interpretation are recorded in
`DOCS/sugar_conformal_whole_hand_tacsl_p1_gate_c_negative_result_20260729.md`.
The tactile installation remains unresolved.
