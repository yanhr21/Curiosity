# Existing Cup Metric Gate

Run tag:

```text
lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210
```

Execution context:

```text
Slurm allocation: 154023
tmux session: curiosity_next_source_alloc_20260626_232937
tmux window: cup_metric_gate_0210
node: server56
scene: cube
tracked_object: existing_cup_asset
final_hold_duration: 999.0
num_steps: 420
```

Purpose:

This gate upgrades the previous visual-only extended hold gate to an explicit
metric gate. It uses the official Newton Panda hydro scene, retargets the
already-loaded official cup asset as the tracked object, keeps the final hold
segment closed, and evaluates lift/hold/drop metrics from Newton-native object
pose traces.

Evidence:

```text
logs/newton/lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210.log
experiments/outputs/lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210_fresh_newton_sensor_contact_sanity.json
experiments/outputs/lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210_summary.json
experiments/outputs/lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210_visual_validation.json
experiments/outputs/lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210_manual_visual_inspection.json
experiments/outputs/lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210_downstream_gate_cleared.json
experiments/visuals/lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210/contact_sheet.png
experiments/visuals/lift_hold_variable_mass_cup_v1_existing_cup_metric_gate_20260627_0210/frame_browser.html
```

Result:

- Official Newton `sensor_contact` sanity: pass.
- SensorTiledCamera visual validation: pass.
- Manual visual inspection: pass.
- Explicit task metric gate: pass.
- `generated_trex_fields=[]`.
- `schema_promotion=blocked`.
- `no_model_or_training=true`.

Metric summary:

```text
success_all_worlds=true
initial_z=0.14850126206874847
max_z=0.3085133135318756
final_z=0.3085133135318756
max_lift=0.16001205146312714
drop_from_max=0.0
longest_hold_frames=246
longest_hold_s=4.1
failure_reasons=[]
```

Manual inspection of the contact sheet and frames `0240`, `0360`, and `0419`
confirmed that the cup remains elevated near the gripper through the final
sampled frame. This clears the Phase 01 cup lift-and-hold metric gate for the
scripted controller setup.

This is still not a learned curiosity result, policy adaptation result, T-Rex
schema sample, calibrated tactile F6 result, or training result.
