# Physical Whole-Hand TacSL Corrective Audit: Negative Result

Date: 2026-07-29
Scope: official SUGAR CarryBox, official Refiner `model_10000.pt`, motion 45,
seed 4263, one continuous IsaacLab/PhysX process, no training.

## Result

The physical whole-hand tactile installation is **not accepted**. Two
independent gates fail:

1. the single frozen `1024`-hull physical-collider candidate does not support
   every curved-hand taxel within `2 mm`;
2. the already-run `2048` diagnostic has real PhysX hand-box load on frames
   where the corresponding direct TacSL field is completely empty, and its
   uncalibrated integrated TacSL force does not close the box dynamics.

The implementation does preserve float taxel normal force, signed two-axis
shear, official SDF-penalty algebra, and same-frame spatial poses. Those
interface facts do not overcome the physical-contact failure.

## Frozen `1024`-hull gate

Artifact:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260729_cleanup/wholehand_tacsl_physical_20260728/
  h1_capacity1024_offline_physx_bake_alloc206425_v1/
```

- exact bake settings: `1,000,000` voxels, `1%` error,
  `maxConvexHulls=1024`, `64` vertices per hull, `0.25 mm` minimum thickness,
  shrink wrapping;
- generated explicit PhysX hulls: `876` left, `851` right;
- asset SHA256:
  `40c26dd4c87c81292e01273c9e7e8cd8e4935eb7be739cbc2378f4c46e46f4a1`;
- manifest SHA256:
  `ea550a4f32102ef01c3334482837837baac53ba278b90350a9eec9fe755b80c6`.

The old `baked_hulls_independent_audit_v6_supported_runtime.json` returned
`passed=true` only after excluding one taxel from each hand. That interpretation
violates the frozen requirement that all `4426/4427` taxels pass.

The corrected report is:

```text
baked_hulls_independent_audit_v7_frozen_all_taxels.json
```

Its SHA256 is
`90d2378a1fba589176883cf7b1fddf28a74e933e61500789e2bddfd87ab477e6`
and `passed=false`:

| hand | required taxels | taxels over 2 mm | offending index | maximum |
| --- | ---: | ---: | ---: | ---: |
| left | 4426 | 1 | 3316 | 2.82965 mm |
| right | 4427 | 1 | 3631 | 2.44532 mm |

No taxel may be deleted, converted to a binary absence, or zero-filled to
convert this into a pass. Per the predeclared plan, this closes the collider
capacity branch.

## Retrospective `2048` diagnostic

The `2048` bake and rollout were executed before restoring the frozen
no-`2048` boundary. They are retained only as negative diagnostic evidence:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260729_cleanup/wholehand_tacsl_physical_20260728/
  motion45_seed4263_continuous_physical_h0_v1_job206425/
```

Raw archive SHA256:

```text
dfcf2832802dec021f254b45867938b23554e766f104ea12625ee09ec922ac62
```

The rollout contains 600 consecutive control frames with no reset or
termination. The independent runtime audit passes:

- official Refiner 890-D observation and 29-D action boundary;
- exact curved official rubber-hand source meshes;
- `4426/4427` left/right float taxels over palm, thumb, and all four fingers;
- native normal, signed shear X/Y, SDF penetration, world position, quaternion,
  source face, and atlas index;
- exact official normal-force reconstruction, with maximum error below
  `1.4e-9 N`;
- same-frame world taxel reconstruction from archived skin poses, with maximum
  position error below `1e-7 m`.

This rules out a one-frame delay, per-frame state restoration, shadow replay,
or video/tactile indexing error.

The physical correspondence report
`audits/correspondence_audit_v2_framewise_hard_gate.json` returns
`passed=false`. Whenever raw aggregate normal load is at least `1 N`, the
corresponding hand must have at least one positive-depth direct TacSL taxel.
That condition fails:

| hand | significant-load frames | zero-taxel failures | failed frame indices | largest missed load |
| --- | ---: | ---: | --- | ---: |
| left | 277 | 5 | 244, 245, 248, 260, 290 | 38.8631 N |
| right | 273 | 12 | 246, 401, 491, 492, 511–518 | 46.7138 N |

At frame 401, for example, the right-hand raw PhysX normal load is
`46.7138 N` while all right direct TacSL taxels are zero. The strongest raw
contact point is only about `0.197 mm` from the exact official hand STL and
`0.413 mm` from an installed taxel, but its PhysX separation is positive:
contact-offset response occurs before the queried box SDF penetrates the
TacSL sampling surface.

Spatial-force gates also fail:

- force-weighted raw contact to exact source within `2/4 mm`:
  left `82.44%/97.68%`, right `78.10%/98.40%`;
- force-weighted raw contact to an active taxel within `4 mm`:
  left `82.42%`, right `82.74%`;
- active taxel to raw contact within `2 mm`:
  left `76.47%`, right `77.62%`, below the frozen `80%` gate.

The object mass is `0.379198 kg`, so weight is `3.719934 N`. On the lifted
bilateral comparison frames, required vertical support has median `3.48474 N`
and raw PhysX gives `3.26130 N`, with median absolute error `0.31728 N`.
Uncalibrated TacSL gives only `-0.000819 N` median vertical support and fails
force closure.

Anatomical activity confirms the visible sparsity:

- left palm: active on `3/600` frames; left thumb: `0/600`;
- right palm: `0/600`; right thumb: `4/600`;
- most activity is confined to sparse finger taxels.

## Interpretation and boundary

The failure is not that the archive contains only old binary hand contact.
The recorded tactile tensors are detailed official TacSL float fields. The
failure is that the current physical collider/SDF/taxel geometry does not make
those fields correspond reliably to the contacts that actually support the
box.

Two mechanisms are observed:

1. PhysX contact offsets produce load at positive separation before the box
   SDF has negative penetration at a TacSL taxel;
2. the approximately `1.3 mm` projected taxel spacing can miss narrow contact
   between samples even when the exact mesh is penetrating.

Changing contact/rest offsets, adding an SDF margin, increasing atlas density,
or trying a different hull capacity would alter the frozen contract. No such
experiment is admitted without a new explicit protocol. Controlled-response
calibration, slip-detector integration, policy training, and tactile-success
claims remain blocked.

The task-registration entry point and the smoke, controlled-response, and
continuous CarryBox wrappers now fail closed with this result record before
starting another physical-whole-hand run. Archived NPZ audits and H.264 review
remain usable.

## Human-review videos

All videos below are browser-compatible H.264/yuv420p, 50 fps, 12 seconds, and
fully decode `600/600` frames. They are negative inspection evidence.

- `videos/world_official_sugar_carrybox.mp4`
- `videos/left_whole_hand_normal_pressure.mp4`
- `videos/right_whole_hand_normal_pressure.mp4`
- `videos/left_whole_hand_signed_shear_xy.mp4`
- `videos/right_whole_hand_signed_shear_xy.mp4`
- `videos/world_top_bilateral_whole_hand_normal_pressure.mp4`
- `videos/world_top_bilateral_whole_hand_signed_shear_xy.mp4`

The two combined videos are `1920 x 1620` and keep world RGB on top. The four
separate hand videos are `1280 x 800` so each normal or signed-shear field can
be inspected without mixing hands.

## Claim boundary

This result proves a real defect and prevents false tactile claims. It does
not prove a correct whole-hand sensor, calibrated high-fidelity tactile,
GelSight RGB/depth, slip detection, learned policy behavior, recovery, or
alternative carrying strategy.
