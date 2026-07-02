# Phase 00 Core Asset Generation Note

This note records the first core asset design for the next curiosity-training
scene. It is a design and catalog step only. It does not claim simulation,
training, or policy improvement.

## Why This Scene Shape

The immediate training scene should be a constrained tabletop arena, not a
large open exploration world. The goal is to test whether curiosity over
object-motion and contact prediction improves grasp, lift, hold, slip
recovery, and stabilization. A large scene would make failures hard to
diagnose because navigation, clutter, sparse contact, and perception errors
would all be mixed into the same signal.

The scene is still hard enough because it varies physical properties that
vision cannot solve alone:

- visually similar cups can have different mass and friction;
- fill cues may be truthful, hidden, or misleading;
- cylinders can roll or slip after contact;
- boxes can have offset center of mass;
- contact-only masks force the tactile/contact stream to stay online.

## Generated Artifacts

- Plan: `PLAN/00_core_asset_generation/plan.md`
- TODO: `TODO/00_core_asset_generation/todo.md`
- Asset catalog: `experiments/configs/phase00_core_tabletop_asset_catalog_v1.json`
- Arena visualization: `experiments/visuals/phase00_core_asset_design/core_asset_arena_map.svg`
- Split visualization: `experiments/visuals/phase00_core_asset_design/core_asset_split_matrix.svg`
- Visual browser: `experiments/visuals/phase00_core_asset_design/frame_browser.html`

## Core Asset Families

1. `cup_like_official`

Uses the existing official Newton cup asset path already recorded in the
project. It is the anchor family because the current project has prior cup
lift/hold evidence and a basic grasping prior.

2. `box_procedural`

Uses a Newton primitive or official primitive path later during compute-node
validation. It tests edge/corner contact and offset center-of-mass response.

3. `cylinder_procedural`

Uses a Newton primitive or official primitive path later during compute-node
validation. It tests rolling and slip under visually simple geometry.

## Split Summary

- Train: 8 cells.
- Validation: 3 cells.
- Held-out: 4 cells.

Held-out cells are locked. They must not be used for training, label
construction, threshold tuning, hyperparameter selection, or controller repair.

## Tactile/Contact Strategy

The current concrete source is Newton contact/contact-proxy evidence. Real
Taccel marker evidence can be added only when it is nonzero, visually
inspected, and kept under `taccel.marker.*`.

The key mask modes are:

- `vision_contact`: normal multimodal training.
- `contact_only_masked_vision`: vision is masked so tactile/contact must
  carry the manipulation signal.
- `vision_only_masked_contact`: later ablation to prove contact is useful.
- `alternating_mask`: scheduled masking so neither modality dominates.

## Required Next Validation

The next step must run on a compute node:

1. Fresh official Newton Panda hydro sanity.
2. Render every catalog cell under the active long-horizon H200 profile:
   at least 1800 simulated steps, 60 warmup steps, 12 second final hold,
   8 second minimum hold, and dense rollout video evidence.
3. Generate frame browser, contact sheet, full rollout video, and metric
   traces for each cell.
4. Manually inspect that objects, masks, contact, and motion match the catalog.
5. Stop if any asset cell fails visual or contact validation.

The earlier 450-step H200 run is useful pipeline evidence only. It is too short
to satisfy Phase 00 full asset-generation completion.

Until that validation exists, this phase is an asset design/catalog generation
step, not a completed dataset or training result.
