# Phase 06 T-Rex Bridge Source Reassessment V1

## Scope

This reassesses the T-Rex bridge after Newton-native adaptation showed useful
behavior on the current cup benchmark. It uses existing reports, manifests, and
strict inventories only. No training, model creation, dataset conversion, or
new T-Rex field generation was performed.

## Inputs

- Source audit config:
  `experiments/configs/trex_bridge_source_reassessment_v1.json`.
- Reference checkpoint sanity:
  `experiments/reports/2026-06-27_phase06_trex_checkpoint_current_sanity.md`.
- Newton learned residual held-out evaluation:
  `experiments/reports/2026-06-27_phase04_residual_adapter_heldout_eval_v1.md`.
- Extra ordinary evaluation:
  `experiments/reports/2026-06-27_phase04_residual_adapter_extra_ordinary_eval_v1.md`.
- Failure-mode comparison:
  `experiments/reports/2026-06-27_phase04_residual_adapter_failure_mode_comparison_v1.md`.
- Newton dense strict inventories:
  `data/processed/newton_contact_aware_frontier_source_manifest_v2_cube_dense_0_239_20260626/strict_inventory.json`
  and
  `data/processed/newton_contact_aware_frontier_source_manifest_v2_pen_dense_0_239_20260626/strict_inventory.json`.
- Newton lift-hold contact source manifest:
  `data/processed/newton_lift_hold_contact_source_manifest_v1_20260627/manifest.json`.

## Useful-Behavior Gate

The Newton-native learned residual adapter now has useful current-benchmark
behavior:

- held-out `full_low` and `empty_high` pass visual and metric gates;
- extra ordinary `half_high` and `full_medium` pass visual and metric gates;
- failure-mode comparison shows learned residual passes where no-adaptation and
  scripted-feedback baselines fail only on `object_accel_above_threshold`.

This unlocks the reassessment question, but it does not imply T-Rex source
compatibility.

## Checkpoint Sanity

The official T-Rex checkpoint sanity passed as reference-only evidence:

- checkpoint integrity sanity: pass;
- official midtrain model-load sanity: pass;
- embedded VQ-VAE, deform encoder/proj, tactile code embedder, and TacF6 stats
  are present.

This means T-Rex assets can remain useful as reference assets. It does not
make the current Newton lift-hold data acceptable for a strict T-Rex bridge.

## Source Contract Audit

The strict T-Rex bridge remains blocked for the current Newton mainline.

Required group status:

- Bimanual 62D `observation.state`, `action`, `action_abs`: missing in current
  Newton strict inventories.
- Accepted synchronized image keys
  `observation.images.head`, `observation.images.wrist_right`,
  `observation.images.wrist_left`: missing as T-Rex keys in current Newton
  strict inventories. Newton has real `newton.camera.*` evidence and manual
  visual inspection, but it is not promoted to accepted T-Rex keys.
- Calibrated nonzero `observation.tactile_f6` with shape `[10,6]`: missing for
  current Newton mainline. Newton contact proxy exists, but it must not be
  renamed into F6.
- Ten dense tactile deformation streams
  `observation.tactile_deform.l0-l4` and `r0-r4`: missing.

The dense Newton cube and pen strict inventories both report:

- status: `blocked`;
- missing field count: `17`;
- incompatible field count: `0`;
- missing fields:
  `observation.state`, `action`, `action_abs`, `observation.tactile_f6`,
  three accepted image keys, and ten dense tactile deformation streams.

The public compatibility slice exposes state/action/action_abs/images/F6
features, but it is reference-only for this decision: it is not the current
Newton lift-hold source and it still has no tactile deformation streams.

## Decision

Go/no-go: **no-go for strict T-Rex bridge promotion now**.

Allowed route:

- keep T-Rex as reference-only;
- continue Newton-native adaptation and contact-proxy curiosity work;
- revisit a strict bridge only after real synchronized source groups exist.

Forbidden routes:

- pad missing 62D state/action/action_abs fields;
- rename `newton.camera.*` into accepted T-Rex image keys without a faithful
  source contract;
- promote Newton contact proxy into `observation.tactile_f6`;
- fabricate ten tactile deformation streams;
- claim official T-Rex bridge compatibility from the current cup benchmark.

`generated_trex_fields=[]` and `schema_promotion=blocked` remain unchanged.
