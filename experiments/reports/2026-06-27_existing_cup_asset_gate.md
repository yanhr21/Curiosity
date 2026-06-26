# Existing Cup Asset Gate

Run tag:

```text
lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105
```

Execution context:

```text
Slurm allocation: 154023
tmux session: curiosity_next_source_alloc_20260626_232937
tmux window: cup_asset_gate_0105
node: server56
scene: cube
tracked_object: existing_cup_asset
```

The gate reused the held allocation and did not submit a new GPU job. The
compute script reread `AGENTS.md`, reran fresh official Newton
`sensor_contact` sanity, then exported 9 SensorTiledCamera frames from the
official Panda hydro scene with a cup-object retarget adapter.

Key evidence:

```text
logs/newton/lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105.log
experiments/outputs/lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105_fresh_newton_sensor_contact_sanity.json
experiments/outputs/lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105_summary.json
experiments/outputs/lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105_visual_validation.json
experiments/outputs/lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105_manual_visual_inspection.json
experiments/outputs/lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105_downstream_gate_cleared.json
experiments/visuals/lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105/contact_sheet.png
experiments/visuals/lift_hold_variable_mass_cup_v1_existing_cup_asset_gate_20260627_0105/frame_browser.html
```

Result:

- Official Newton `sensor_contact` sanity: pass.
- SensorTiledCamera visual validation: pass.
- Manual visual inspection: pass with task limitations.
- `generated_trex_fields=[]`.
- `schema_promotion=blocked`.
- `no_model_or_training=true`.

The summary metadata confirms:

```text
tracked_object=existing_cup_asset
adapter=retarget_existing_official_cup_asset_as_object
cup_body_local=15
body_label=cup
max_lift=0.15901388227939606
```

Manual inspection of the contact sheet and frames `0000`, `0120`, and `0239`
confirmed that the pink cup asset is visible and retargeted as the tracked
object. The final inspected frame shows the cup tilted/fallen, so this gate
does not clear stable cup grasp success. It clears only the cup-asset retarget
and visual evidence needed for the next Phase 01 iteration.

Next concrete step:

Adjust cup grasp initialization and physical task parameters while preserving
the same official Newton asset path and Newton-native source namespaces.
