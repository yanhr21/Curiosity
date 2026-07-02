# Phase 00 Core Asset Generation Verification

Date: 2026-06-29

## Status

Status: lightweight asset design generated and verified.

This is not simulation evidence, not dataset collection, not training, and not
a curiosity success claim. No Newton rendering, validation builder, model
loading, training, or evaluation was run on the login node.

## Verified Files

- `PLAN/20260629/00_core_asset_generation/plan.md`
- `TODO/20260629/00_core_asset_generation/todo.md`
- `experiments/configs/phase00_core_tabletop_asset_catalog_v1.json`
- `docs/phase00_core_asset_generation.md`
- `experiments/visuals/phase00_core_asset_design/core_asset_arena_map.svg`
- `experiments/visuals/phase00_core_asset_design/core_asset_split_matrix.svg`
- `experiments/visuals/phase00_core_asset_design/frame_browser.html`

## Verification Commands

```bash
jq empty experiments/configs/phase00_core_tabletop_asset_catalog_v1.json
```

Result: pass.

```bash
jq -r '"train=" + ((.split_cells.train | length)|tostring) + " validation=" + ((.split_cells.validation | length)|tostring) + " held_out=" + ((.split_cells.held_out | length)|tostring) + " families=" + ((.object_families | length)|tostring)' experiments/configs/phase00_core_tabletop_asset_catalog_v1.json
```

Result:

```text
train=8 validation=3 held_out=4 families=3
```

## Directory Organization Check

`PLAN/` now contains the dated active planning tree under `PLAN/20260629/`.
`TODO/` now contains the dated active task tree under `TODO/20260629/`.

## Asset Scope

The generated core asset catalog declares:

- one constrained tabletop arena;
- the official Newton Panda hydro scripted grasp/lift prior as base behavior;
- 3 object families: cup-like official asset, procedural box, procedural
  cylinder;
- 8 train cells;
- 3 validation cells;
- 4 held-out cells;
- explicit no-leakage rules for held-out cells;
- mask modes for vision/contact balancing, including contact-only masked
  vision.

## Next Required Step

After user approval, run compute-node validation:

1. Fresh official Newton Panda hydro sanity.
2. Render every catalog cell.
3. Produce frame browsers, contact sheets, rollout videos, contact traces, and
   manual visual inspection notes.
4. Stop if an asset cell fails visual/physics/contact validation.

The project must not train on this catalog until the compute-node visual and
contact validation evidence exists.
