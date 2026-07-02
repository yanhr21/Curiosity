# Dense Tactile Infant Curiosity

This file is the active research idea. It must stay short, current, and free of
legacy experiment narratives that could mislead future training decisions.
Historical or wrong-path evidence belongs in `IDEA/legacy/` or
`experiments/reports/`, not in this file.

## Active Goal

Build a reference-video-aligned dense tactile infant:

1. A base controller or model can complete basic grasp, lift, and hold.
2. The same rollout exports dense visual and tactile mechanics comparable to
   the user reference video `0780e5ec3fdb26b63ae63de0f49f07c4.mp4`.
3. Only after dense tactile/base evidence exists, restart curiosity as true
   closed-loop active probing over dense visuo-tactile prediction.
4. Final success requires harder held-out tasks beating the strongest baseline
   without safety regression.

The project is not trying to prove that a script runs, a checkpoint exists, or
a video is nonblank. The scientific claim is that curiosity improves a basic
grasping infant on harder manipulation settings.

## Required Dense Tactile Evidence

Before curiosity training can be claimed, the environment/base evidence must
include synchronized:

- visual scene;
- left/right tactile pad maps;
- pressure and compression heatmaps;
- normal force `Fn`;
- tangential/shear force `Ft`;
- shear direction;
- contact area;
- center of pressure;
- penetration/compression;
- material, friction, and stiffness statistics;
- grip, shear, contact, and safety time-series.

The target representation is pad-resolved:

```text
left_pad.pressure:      [T, H, W]
left_pad.compression:   [T, H, W]
left_pad.shear_u/v:     [T, H, W]
left_pad.contact_mask:  [T, H, W]
left_pad.Fn/Ft:         [T] or [T, H, W]
right_pad.*:            same contract
```

Candidate Newton/MJWarp outputs must preserve provenance:

```text
candidate.newton_mjw.Fn
candidate.newton_mjw.Ft
candidate.newton_mjw.area_proxy
candidate.newton_mjw.marker_flow
candidate.newton_mjw.contact_normal
```

Do not promote proxy fields into official tactile semantics:

```text
area_proxy != real contact area
marker_flow render != photometric GelSight marker output
contact_count != tactile map
candidate Fn/Ft != validated official tactile force field
```

## Closed-Loop Curiosity Requirement

Future curiosity must be closed-loop, active, and tactile-rich:

- the forward/world model predicts tactile/contact/mechanics, not just object
  height or contact count;
- intrinsic reward affects policy optimization and therefore changes future
  rollout data;
- the policy can choose meaningful exploration actions: probing, regrasping,
  grip-force adjustment, pressure balancing, and shear-minimizing behavior;
- exploration is safety constrained, with penalties or hard guards for drop,
  slip, contact loss, excessive acceleration, and excessive force;
- sample reweighting alone is not closed-loop curiosity.

The training and evaluation must include tactile-mask evidence:

- vision+tactile;
- tactile-only with vision masked after contact;
- vision-only;
- noisy, delayed, shuffled, or mismatched tactile.

The policy must not collapse to pure vision or pure tactile. Multimodal
vision+touch should outperform ablations, and corrupted tactile should hurt if
touch is genuinely online and causal.

## Baselines And Metrics

Do not let an easy or strong base controller hide the research question. If
base grasp/lift/hold already succeeds, move to harder tasks or finer metrics.

Required comparisons:

- no-adaptation base;
- scripted feedback;
- no-curiosity residual/adaptation baseline;
- curiosity ablations;
- serious official/reference methods when available, or documented blockers.

Required metrics:

- lift;
- hold duration;
- slip;
- drop;
- contact loss;
- object acceleration;
- force/contact cost;
- safety regression;
- held-out improvement over the strongest baseline.

Success claim condition:

```text
harder held-out tasks beat strongest baseline without safety regression
```

## Official Methods And Checkpoints

Use serious source code and official checkpoints when claiming an official
method. Do not hand-roll toy substitutes and present them as T-Rex, VQ-VAE,
Transformer, tactile encoder, or world-model progress.

T-Rex remains a future model/reference path. Use official repository,
released checkpoints, embedded tactile VQ-VAE path, and faithful adapters only.
If an official checkpoint, config, schema, or runtime is missing or
incompatible, record it as a blocker or comparison gap.

Newton/Taccel are the main simulator path. UniVTAC, TaCauchy, HydroShear, and
IsaacLab TacSL/IsaacLabTactile are final semantic/reference comparison paths.
Gate 00F is low-priority final semantic validation and must not block current
dense tactile infant/base-evidence work.

## Active Records

Active plan:

```text
PLAN/00_dense_tactile_infant/plan.md
```

Active TODO:

```text
TODO/00_dense_tactile_infant/todo.md
```

Legacy pre-reset records:

```text
IDEA/legacy/
PLAN/legacy_*/
TODO/legacy_*/
```

Current execution rule:

```text
current target = reference-video-aligned dense tactile environment + base grasp/lift/hold
curiosity restart condition = dense tactile/base evidence and closed-loop training contract are ready
success claim condition = harder held-out tasks beat strongest baseline without safety regression
```
