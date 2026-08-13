# Curiosity Workspace Cleanup Record

Date: 2026-07-29

## Current state after the 2026-08-10 tactile reset

The July whitelist below is historical. The current `experiments/` root has
four active top-level directories:

```text
native_tactile_representation
sugar_demo_reward
sugar_reproduction
sugar_smp_exploration
```

For the user-specified combined cap on Curiosity, demo following, and
"does tactile help training" experiments, only three main experiment packages
remain:

1. `sugar_demo_reward`: the retained demo-following package;
2. the matched original-ICM policy-credit pair inside
   `sugar_smp_exploration`; and
3. `official_tactile_genesis_pinned_de2bcc9`, the retained tactile-training
   comparison inside `sugar_smp_exploration`.

Supporting `audits/` and `priors/` directories are not separate experiments.
The combined retained count is therefore three, below the maximum of five.
Plan/TODO 04--11 are under their respective `legacy/` directories; Plan/TODO
12 is the only active execution queue.

The completed native-tactile launch logs and the two superseded
`native_audit.json` files were moved into the existing single archive root at
`current_workspace_experiments/20260810_native_tactile_reset/`. The active
native-tactile directory now contains only its README, three raw CarryBox
traces, three world videos, three final main videos, and the useful success /
release-failure detail videos.

## Outcome

The cleanup was move-only. No experiment or sibling workspace was deleted.

At `/public/home/yanhongru/`, the only remaining Curiosity-prefixed
directories are:

```text
Curiosity
Curiosity_archive
```

Seventeen former archive, failed, invalid, legacy, and superseded sibling
directories retain their original names under:

```text
/public/home/yanhongru/Curiosity_archive/workspace_siblings/
```

## Current experiment whitelist

The active `experiments/` root now contains exactly five directories:

```text
reports
sugar_chord_paper_reproduction
sugar_demo_reward
sugar_reproduction
sugar_smp_exploration
```

Six obsolete or rejected top-level experiment trees were moved under:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260729_cleanup/
```

These include the old Refiner human-review, July-27 redo, from-scratch TacSL,
Tactile Genesis probe, physical whole-hand TacSL, and whole-hand TacSL redo
trees. Historical documentation now points to their archive locations.

## CHORD boundary

CHORD was not classified as a failed method. The current workspace retains one
23-file, independently audited public-paper-formula package:

- exact formula audit;
- nominal, 3x-mass, and low-friction TacSL sources;
- exact offline contact-wrench-support arrays;
- independent reconstruction; and
- final synchronized MP4 plus figures.

Old CHORD wiring, duplicated visualizations, and policy-effect runs were moved
to `sugar_chord_paper_reproduction_pruned/` in the cleanup archive. Their
tactile installation/input was later rejected, so those runs cannot establish
either positive or negative CHORD policy effectiveness.

## GPU retention

Slurm allocations `206401` and `206425` were not released. Their CUDA
retention workloads were restarted within the same allocations with outputs
below the single archive root:

```text
/public/home/yanhongru/Curiosity_archive/runtime_gpu_retention/
```

Both allocations remained `RUNNING`, and both restarted workloads reported
`passed_so_far=True` after the move.

## Completion boundary

The mandatory mainline ledger audit still reports
`completion_passed=false`. This cleanup changes filesystem organization and
provenance paths only; it is not a project-completion, tactile-sensor, policy,
recovery, or alternative-strategy result.

## 2026-08-02 Plan-10 posture-search cleanup

After closing the wide-stance, heuristic side/bottom, lead-ramp and XY-anchor
branches, `76` V114--V154 intermediate run directories/logs were moved without
deletion to:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/
  20260802_plan10_posture_load_ramp_transients_v114_v154/
```

The active experiment root retains only the scientifically distinct records
from that range: V115 (best stable bilateral topology) and V147 (best
unconstrained asymmetric apparent-rise control), plus the later V155/V156
exact-trace video/audit records. No artifact was copied into Git and no second
archive root was created.

## 2026-08-03 Plan-11 visualization cleanup

After the final readable Tactile Genesis controlled-direction video and ICM
explanation-video package passed their independent audits, superseded render
iterations and extracted review frames were moved without deletion to:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/
  20260803_plan11_intermediate_visualizations/
```

The active experiment roots retain only `videos_h264_v5/` for the ICM semantic
result; `controlled_raw_link_state_direction_h264_v5.mp4` and its V5 audit for
the controlled KinematicTaxel direction result; the final V2 accepted-window
render; the final controlled ElastomerTaxel render; and the V3 continuous
no-box-reset CarryBox render. Raw arrays, deliberately preserved producer
negatives and authoritative audits remain in place. Older V2--V4 direction
renders, the V1 accepted-window render, extracted Elastomer review frames, and
the V1--V2 continuous CarryBox renders were moved below the single archive
root. No second archive root was created.

The same archive package now also contains the V1--V6 intermediate
contact-velocity callable audits. The active audit directory retains only
`AUDIT_v7.json`, which adds the frozen anatomical patch order and symmetric
optical-module positions to the exact normal/signed-XY callable, four-frame
force+velocity history and non-degraded bilateral 27-patch topology checks.
These moved files are interface-test iterations, not discarded scientific
outcomes.

## 2026-08-07 Plan-10 load-bearing posture cleanup

After the two exact tilt-to-scoop directions both produced zero admissible
static candidates and their separate H.264 review packages passed independent
audit, two invalid wrapper runs and nine superseded intermediate scans were
moved without deletion to:

```text
/public/home/yanhongru/Curiosity_archive/
  plan10_withdrawn_diagnostics_20260807/
```

The invalid records are the first bilateral lower-edge source, whose right
target was overwritten by a legacy bottom default, and the first
opposite-approach source, whose left outward normal was overwritten.  Their
corrected V2 sources remain active.  The other moved records are intermediate
clearance/root/roll scans replaced by the retained final bracket for the same
branch.  All geometry contracts, final branch scans, exact source traces, the
two 12-candidate tilt scans, their official-pose exports, H.264 movies and
independent audits remain in the active experiment tree.  No artifact was
deleted, no second archive root was created, and retained allocation `224695`
was not released.
