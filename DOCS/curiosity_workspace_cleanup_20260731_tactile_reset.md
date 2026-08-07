# Curiosity tactile experiment cleanup record

Date: 2026-07-31

## Outcome

The cleanup was move-only. No experiment artifact was deleted.

The former 20-GB directory
`experiments/sugar_reproduction/anatomical27_whole_hand_tacsl/` contained 134
top-level experiment directories. It now contains six core evidence
directories and 13 associated current logs:

1. `static_admission_job209917_v80_source_current_calibrated`
2. `direct_pose_center_r15_v30_mirrored_camera_job209442`
3. `continuous_carrybox_v83_solid_outer_job209917`
4. `continuous_carrybox_v87_tg_megadense_history_job209917`
5. `controlled_tg_history_v88_job209917`
6. `official_reference_comparison_20260731`

The remaining 128 failed, stopped, superseded or intermediate directories and
56 old logs/diagnostic files retain their original names under:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260731_fixed_hand_tactile_reset/anatomical27_whole_hand_tacsl/
```

The retained workspace subtree is approximately 2.4 GB; the archived subtree
is approximately 18 GB. V83 is retained because the independent V87 audit
explicitly uses it as its bitwise control. V89 and V84--V86 were archived
because V89 was stopped and the others were superseded by V87.

The first Plan-10 Taccel launch created only an `args.json` before failing to
resolve the released runtime path. That incomplete precursor was also moved,
without deletion, below the same cleanup root at
`plan10_precursor_failures/taccel_official_soft_teddy_h200_v1/`. The corrected
official-source run remains in the workspace, but its independent audit
admits only CUDA/IPC solver execution: all 80 exported PLY files are bitwise
identical and no RGB/depth/marker artifacts were saved, so it is explicitly
not soft-gel deformation or tactile evidence.

The subsequent unchanged official TacMan probe did call that released path,
but failed on its first solver step with a conjugate-gradient scatter-shape
mismatch before producing RGB/depth/marker evidence. Its args-only workspace
directory was moved to
`plan10_precursor_failures/taccel_official_tacman_render_v1/`, and its
compressed raw log remains under `plan10_raw_logs/` in the same single archive
root.

After the exact Unitree archive passed CRC, byte-count, SHA-256 and selected-
tree extraction audits, its segmented-download pieces, an incomplete `wget`
file and incomplete Hugging Face caches were moved to
`plan10_acquisition_precursors/` below the same archive root. The reference
source directory now retains only the exact `assets.zip` and its verified
`extracted/` tree.

Two superseded Plan-10 topology movies were also moved without deletion to
`plan10_precursor_failures/`. The first let the uncontrolled default robot fall
and showed mostly floor/partial hands; the second stabilized both hands but
cropped the feet in the world panel. They are retained as
`unitree_g1_inspire_finger_motion_bad_camera_20260801/` and
`unitree_g1_inspire_finger_motion_world_feet_cropped_20260801/`. Neither may be
presented. The workspace retains only the corrected full-body/bilateral-hand
movie and its exact trace/manifest.

## Scientific boundary

This cleanup does not turn V83/V87/V88 into tactile admission. It freezes them
as the minimum evidence needed to reproduce the rigid-hand sparsity and
force-model diagnosis. New execution moves to Plan 10's official articulated
five-finger plus deformable-gel arm. Project completion remains governed by the
mainline evidence ledger.

## 2026-08-02 Plan-10 fixed-motion mechanics sweep

The fixed motion-45 upper-body mechanics branch was also cleaned move-only
after its terminal force/reachability audit. Twelve superseded or rejected
experiment directories and thirteen logs (including the startup-only V88 log)
retain their original names under:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260802_plan10_fixed_motion_negative_sweep/
```

The workspace retains only the exact cooked-bottom envelope, the clean
slow-lift V83 force-balance negative, the cooked V89 clean negative, and the
V98/V99 terminal reachability/non-hand-collision boundary records. No moved
run is completion, tactile, or hand-only lift evidence.
