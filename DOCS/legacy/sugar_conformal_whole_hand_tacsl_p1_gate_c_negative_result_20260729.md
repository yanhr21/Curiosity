# SUGAR conformal whole-hand TacSL P1 Gate-C negative result — 2026-07-29

## Decision

P1 is rejected. It is not a correct installed whole-hand tactile sensor and
must not be used for Gate D, training, slip detection, reward learning, or a
policy input.

Gate A and Gate B establish only that the hash-locked finite-area taxel
prisms exist and that the official SUGAR CarryBox stage preserves the robot,
box, action, material, and float output contracts. The first mandatory Gate-C
target, `left/palm`, fails the frozen physical-response and spatial-
correspondence requirements. Under the protocol rule that any zone failure
rejects P1, the remaining eleven zones are not a route to acceptance.

## Exact execution boundary

The authoritative controlled-response execution used:

- task:
  `Sugar-G129dof-CarryBox-Official-Refiner-Physical-WholeHand-TacSL-Audit`;
- host/job: `server28`, retained Slurm allocation `206425`;
- the exact official SUGAR CarryBox environment and IsaacLab/Isaac Sim/PhysX
  process;
- one initial box pose and velocity write for the trial, followed only by
  logged external force and torque;
- zero 29-D actions, no policy update, no learning, no replay, no per-step
  state restoration, no binary contact, and no force rescaling;
- complete float32 direct normal, signed two-axis shear, penetration, taxel
  pose/frame, raw PhysX audit contact, and world RGB for all `350` source
  steps.

The trial archive is
`/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260729_cleanup/wholehand_tacsl_physical_20260728/
p1_zone_response_job206425_v3/trial_00_left_palm.npz`, `545,784,662` bytes,
SHA-256
`fed89c0b7ed3ab5512ca957bd847acccf21663f73586c17a41f02e21ea82afb3`.
The producer manifest SHA-256 is
`486799542b4b5f67cc99180b42ae8b30e0b1642b948601eb9f5bbe707b93019f`.

The producer encountered a PyTorch inference-tensor reset error before trial
1, after trial 0 had been atomically archived. This does not make the
left-palm result indeterminate: left/palm is itself one of the twelve
mandatory targets, all its `350` rows are present, and it independently fails
multiple frozen gates. The incomplete execution can never be reported as a
Gate-C pass.

Two earlier attempts are infrastructure failures only:

- `p1_zone_response_job206425_v1`: stopped before contact collection because
  the producer expected the old four-item Gym API;
- `p1_zone_response_job206425_v2`: stopped before a trial was archived
  because the raw audit-contact count exceeded the reporting buffer.

They are retained but provide no tactile evidence. Raising only the raw
reporting capacity to `65,536` did not change collision, physics, TacSL
equations, material, or policy state. Gate B was independently rerun after
that capacity-only change:
`p1_composed_stage_job206425_v1/composed_stage_audit_v3.json`, SHA-256
`8632159b5074b9eb6903dc941d2cdc083d08f245662805465bfe0a0f3457fd15`.

## Independent findings

The authoritative report is
`p1_zone_response_job206425_v3/
independent_partial_negative_audit_v1.json`, SHA-256
`3e5fbf43ee0c9060d519a1c3b5a7fd0ca2544c0e4d70fc45fcf2336aaba858da`.
It reports `passed=false`.

- Left-hand baseline and release direct fields are exactly zero.
- The intended left-palm target stays exactly zero at every frozen
  `0.25/0.5/1/2/4 mm` hold. Its active fraction is `0` at all five depths.
- Left normal, signed shear X/Y, and penetration remain zero for all `350`
  frames, so monotonic response and signed reversal both fail.
- The non-target right hand responds on `350/350` frames, including the
  baseline. Its maximum direct normal is `0.005758085 N`, maximum absolute
  signed shear is `0.006424560 N`, and maximum penetration is
  `4.666227 mm`.
- The one positive raw left-hand pair has no active same-frame taxel. The
  original audit compared its raw contact point directly to the nearest
  taxel and reported `4.976162 mm`; the correction below withdraws that
  separation-unaware number as an ownership failure. The missing same-frame
  direct response remains true.
- The requested object servo saturates force on `349/350` frames and torque
  on `49/350`, so the fixture itself also fails the frozen no-saturation
  requirement.
- World RGB contains `350` distinct, decodable source frames.
- Recomputing official TacSL normal and channel-last signed shear from the
  archived penetration, contact normals, relative velocity, and taxel frames
  gives maximum error `0.0 N`.

The last point is important: the zero-left/nonzero-right result is not a
heat-map channel swap or visualization artifact. The official force algebra
and archive processing reconstruct exactly. The trial still fails the
controlled physical-response fixture and same-frame response gates, but it
does not by itself localize the failure to a collider/taxel translation.

Additional read-only trajectory inspection shows that the intended target
position error grows to roughly `0.4 m`, the object moves about `0.97 m` in X
and `0.48 m` in Z, and zero actions allow the robot to move away from its
reset state while the box continues contacting the right side. This makes
the P1 external box-servo fixture invalid independently of the direct field.

## Post-result separation-aware correction

The raw PhysX contact record is a complete
`point / normal / separation / force` tuple. The installed PhysX 107.3 tensor
API documents the returned separation, and PhysX may generate a contact pair
while the surfaces still have positive separation. For the sole positive
left-hand pair:

```text
raw point to nearest taxel                     4.976162 mm
reported separation                            4.610698 mm
(raw point + normal * separation) to taxel     0.623510 mm
frozen taxel radius + tolerance                0.824326 mm
```

The separation-aware point is inside the frozen geometric limit. Therefore
the earlier direct `4.976162 mm` comparison is withdrawn as proof of an
installation translation or ownership error. This correction does not make
P1 pass: the intended left field is zero for all 350 frames, the opposite
hand is active from baseline, the target drifts by roughly 0.4 m, and the
external force/torque saturates on `349/49` frames. P1 remains rejected
because its mandatory controlled-response and fixture gates fail.

## Human-review videos

All videos are browser-compatible H.264/yuv420p at `20 fps`, contain exactly
`350/350` decoded frames, and passed full OpenCV and FFmpeg decoding. A video
validation pass means only that the negative evidence is readable; it is not
a sensor pass.

The combined `1920 x 1620` video puts world RGB on the full top row and the
bilateral normal/signed-X/signed-Y maps below:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260729_cleanup/wholehand_tacsl_physical_20260728/
  p1_zone_response_job206425_v3/videos_negative_v1/
    combined_world_pressure_shear_gate_c_negative.mp4
```

SHA-256:
`a555d84787c12e8a71d4537072c3c56ccf480a5d6e6988be40ac6b74510063c8`.

Separate videos:

| Video | Resolution | SHA-256 |
|---|---:|---|
| `world_left_palm_gate_c_negative.mp4` | `1280 x 720` | `4ef6282989c1597a178606efb7cc6452a332260ae311da7b3ab2abe1fe959a68` |
| `left_pressure_left_palm_gate_c_negative.mp4` | `960 x 720` | `75072ed5c86581db6ecc29e23f24a74b04554665345d56386347f18549eb9304` |
| `left_signed_shear_xy_left_palm_gate_c_negative.mp4` | `1280 x 720` | `bdc106b7780942373911767ab542775e07bf4a64818d4ef375acbdfabc089b6d` |
| `right_pressure_left_palm_gate_c_negative.mp4` | `960 x 720` | `50267828775766ae6cec5dad793557daaa875a241a504852792821d7552bd427` |
| `right_signed_shear_xy_left_palm_gate_c_negative.mp4` | `1280 x 720` | `ef992f6b48bc82381d5425d3a06ed76a00267dd8ecff72158c0ad89690842788` |

Fixed scales are used for the complete videos: white is zero, red is
positive, blue is negative, missing atlas cells are black, and the intended
left-palm taxel is marked in yellow. No per-frame normalization hides the
zero-left/nonzero-right failure.

The video manifest is
`videos_negative_v1/manifest.json`, SHA-256
`6652f10e0988205c79fb7b984758e9057d555d12851aea19ce43b54b599cbda8`.

## Consequence

Gate D is blocked by the frozen ordering and has not been run. There is no
600-step official-policy correspondence/load result for P1, no valid
whole-hand GelSight RGB/depth result, and no training result. The tactile
problem remains unresolved.

Do not repair this negative result with a force scale, threshold, axis swap,
frame shift, binary contact, repeated-R15 shadow, another engine, or a
hand-written controller. A successor physical design and a new frozen
protocol require explicit user review and authorization.
