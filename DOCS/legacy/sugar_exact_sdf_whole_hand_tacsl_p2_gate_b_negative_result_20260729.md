# SUGAR exact-SDF whole-hand TacSL P2 Gate-B negative result — 2026-07-29

## Decision

P2 passes exact-stage Gate A and fails the frozen cooked-SDF/taxel Gate B.
It must not proceed to controlled contact, CarryBox behavior, optical output,
training, slip detection, reward use, or a policy input.

This is a same-process IsaacLab/SUGAR/PhysX result. It contains no Genesis,
shadow sensor, contact proxy, replay, state restoration, training, or
parameter sweep.

## What passed

The live official CarryBox environment contains one continuous dynamic SDF
collision mesh under each unchanged rubber-hand rigid body:

- left: `22,876` vertices, `45,748` triangles, area
  `0.030959132843152364 m²`;
- right: `21,928` vertices, `43,852` triangles, area
  `0.03095905181228843 m²`.

Each collision mesh is point-for-point and face-for-face identical to the
official hand visual after lossless duplicate welding. Both meshes are
watertight and consistently wound. The original converted one-hull collider
is disabled; no P1 prism or rejected H0/H1 collider is present.

The frozen SDF cooking values are `256 / 6 / 0.01 / 0.01`. The material is
the official R15 reference `10/1`, friction `0.5/0.5`, restitution zero, with
no authored contact or rest offset. The official robot and CarryBox source
hashes are unchanged. All live body, mass, inertia, joint, actuator, limit,
and 29-D action-transform fingerprints exactly match the separately executed
untouched SUGAR baseline.

The complete float32 direct fields initialize at
`(1,4426) / (1,4426,2) / (1,4426)` on the left and
`(1,4427) / (1,4427,2) / (1,4427)` on the right. No taxel is deleted or
represented as a zero placeholder. All taxel centers lie within
`0.168209 mm` of the cooked physical SDF surface, below the frozen
`0.25 mm` limit.

## Why Gate B failed

P2 used the discrete source triangle normal as each taxel's local surface
normal. The frozen Gate B required every taxel to have correct SDF signs and
strict monotonicity from `-2` through `+2 mm`, and required the cooked SDF
gradient to agree with that triangle normal by cosine at least `0.95`.

The authoritative result is:

| Check | Left | Right |
|---|---:|---:|
| center tolerance failures | 0 / 4,426 | 0 / 4,427 |
| source-normal sign failures | 10 | 10 |
| source-normal strict-monotonic failures | 46 | 48 |
| source-normal/gradient cosine failures | 7 | 7 |
| minimum center cosine | 0.630314 | 0.634090 |

The failures are bilateral anatomical mirrors, not random numerical noise.
They concentrate along a palm/finger-root fold plus the first atlas row at
the thin wrist edge. At the wrist edge, moving `2 mm` inward reaches the
solid's medial region or another boundary, so distance to the nearest
surface need not remain monotonic. At several folds, a discrete triangle
normal points toward a neighboring part of the same non-convex hand.

These facts do not permit P2 to pass: its all-taxel frozen rules were declared
before execution. They do show that its `2 mm` two-sided rule conflates a
local surface-coordinate test with global thin-solid thickness and
self-proximity.

## Non-admitting diagnostic

The same fixed SDF was queried along its own normalized center gradient
without changing the P2 verdict. Every center gradient has norm one to
float32 precision. Along the physical-gradient direction:

- every left and right taxel has correct sign and strict monotonicity across
  the local `-0.5 / center / +0.5 mm` interval;
- one taxel per hand still fails the full `±2 mm` sign rule because the
  outward ray approaches another surface of the same non-convex hand;
- full-range strict monotonic failures reduce to `36/38`, dominated by the
  thin wrist edge and nearby surfaces.

This diagnostic motivates a new physical-frame protocol; it cannot be used
to rewrite P2 as a pass.

## Authoritative artifacts

The corrected audit is:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260729_cleanup/wholehand_tacsl_physical_20260728/
  p2_exact_sdf_gate_ab_job206425_v3/
    gate_ab_audit_v3.json
    gate_b_full_diagnostic_v3.npz
```

- JSON SHA-256:
  `79f8e774a458aad19efe6398ed8865928746d919cd478f5f4311fda7d1f7826e`;
- NPZ SHA-256:
  `7be2497d66fdb4e4f96c0e2ab0db3291475f9f3ee30fdeede676d138789e1c21`.

The v2 NPZ is not valid for comparing the two query directions because the
PhysX tensor view reused its return buffer. The v2 failure counts were
computed before that reuse and agree with v3, but v3 clones both query
responses immediately and is the only admitted full-array diagnostic.

P2 Gate C and Gate D were not run. No physical response, CarryBox load,
GelSight RGB/depth, behavior, or learning claim follows from this result.
