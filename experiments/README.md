# Curated Experiment Index

`experiments/` is local-only. It now contains the active native-tactile work,
five retained historical experiment packages, and the minimum official SUGAR
support needed by those packages.

For the current CarryBox tactile result, start at
[`native_tactile_representation/whole_hand_carrybox_v3/README.md`](native_tactile_representation/whole_hand_carrybox_v3/README.md).
The complete raw-data-to-video command and exact output inventory are in the
adjacent
[`REPRODUCE.md`](native_tactile_representation/whole_hand_carrybox_v3/REPRODUCE.md).
The compact machine-readable file list and validated capability boundary are
in
[`MANIFEST.json`](native_tactile_representation/whole_hand_carrybox_v3/MANIFEST.json).
The active code map is
[`scripts/sugar/native_tactile/README.md`](../scripts/sugar/native_tactile/README.md).
Do not follow older numbered tactile directories in archived records as active
execution routes.

The canonical successful bundle has five synchronized review videos:

- [world plus both anatomical hands](native_tactile_representation/whole_hand_carrybox_v3/successful_grasp/successful_carrybox_whole_hand_tactile.mp4)
- [all 27 left-hand patches](native_tactile_representation/whole_hand_carrybox_v3/successful_grasp/left_detail.mp4)
- [all 27 right-hand patches](native_tactile_representation/whole_hand_carrybox_v3/successful_grasp/right_detail.mp4)
- [world plus bilateral R15 RGB/depth](native_tactile_representation/whole_hand_carrybox_v3/successful_grasp/palm_optical.mp4)
- [force, friction, kinematics, and calibration](native_tactile_representation/whole_hand_carrybox_v3/successful_grasp/force_kinematics_friction_complete.mp4)

All five show the same source interval `230:660` and fully decode `430/430`
frames. The underlying trace contains all `660` control frames and `2640`
physics substeps.

## Active native tactile work

- `native_tactile_representation/`: current IsaacLab-native R15
  representation, paired successful/failed grasps, raw tensors, independent
  audits, and synchronized H.264 videos. This active category is not counted
  against the five-package historical quota, but redundant versions must still
  be archived.
- `native_tactile_training/`: the Plan-13 matched serious SUGAR training route.
  Only fresh continuous-frame-zero tactile/zero runs belong here. The earlier
  random mid-trajectory reset pair is archived because inserting physical skin
  directly into contact produced a nonphysical reset-time tactile burst. The
  completed action-residual update-63 tactile/zero pair is indexed in the
  directory's own `README.md`; its common-horizon outcome is negative at one
  seed. The direct synchronized video is
  `action_residual_64u_policy_visualization_20260811/tactile_trained_vs_zero_trained_side_by_side.mp4`.
  Its exact training and video reproduction route is in
  [`REPRODUCE.md`](native_tactile_training/REPRODUCE.md), and the compact file
  inventory is [`MANIFEST.json`](native_tactile_training/MANIFEST.json).

## Runtime and reuse boundary

The force/shear fields are online and causal at the simulator clock, including
when an external camera cannot see the contact. The complete optical scene is
currently slower than wall-clock real time. The implementation is geometry-
reusable, not automatically task-general: the current admitted object is the
SDF CarryBox and the installed patches cover the hands. Official KickBox uses a
different non-SDF big-box asset and normally contacts it with the foot/leg, so
this CarryBox configuration does not yet provide a valid KickBox tactile result.

## Five retained historical packages

1. Demo following, correct versus wrong official demo V3:
   `sugar_demo_reward/policy_training/plan11_demo_conflict_authority_rework_seed130581_v3/`
   with its matching evaluation and visualization.
2. Demo following, fixed unrelated KickBox teacher through update 1216:
   `sugar_demo_reward/policy_training/plan11_fixed_teacher_demo_identity_seed131581_v2/`
   with only the update-1216 evaluation and visualization retained.
3. Original ICM concrete semantics and cross-seed controls:
   `sugar_smp_exploration/audits/plan11_original_icm_concrete_seed110381_v1/`
   plus its declared cross-seed/fresh-noise companions.
4. Original ICM policy-credit comparison:
   `sugar_smp_exploration/plan11_original_icm_policy_credit_zero_tactile_seed110381_v1/`
   with its frozen evaluation.
5. Official Tactile Genesis tactile-training-effect package:
   `sugar_smp_exploration/official_tactile_genesis_pinned_de2bcc9/`, pruned to
   the corrected Stage-2 training/evaluation/import evidence and the distinct
   CarryBox Question-9 native sensor presentation.

## Support

- `sugar_reproduction/outputs/final/`: curated official SUGAR baseline and
  checkpoints referenced by retained experiments. Its only child is now
  `official_sugar/`; old demo, RGB, ICM, tactile, physics, and report output is
  archived.
- `sugar_reproduction/assets/` and `sugar_reproduction/render_runtime/`:
  retained rendering/runtime support.
- `sugar_smp_exploration/priors/sugar_g1_box_tinymdm_v1/`: frozen official-
  architecture prior referenced by the retained policy-credit proof.

Everything else moved, without deletion, to:

```text
/public/home/yanhongru/Curiosity_archive/current_workspace_experiments/20260810_native_tactile_reset/
```

Do not create any additional archive root.

Intermediate endpoint checkpoints are intentionally absent from the active
tree. Correct/wrong V3 retains final policies only; unrelated KickBox retains
only update 1216; the serious predictor retains only `validation_best.pt`; ICM
policy credit retains final policies; and Tactile Genesis Stage 2 retains only
`model_5999.pt`. Their predecessor chains remain recoverable from the single
archive tree above.
