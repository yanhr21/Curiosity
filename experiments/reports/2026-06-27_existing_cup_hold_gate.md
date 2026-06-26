# Existing Cup Hold Gate

Run tag:

```text
lift_hold_variable_mass_cup_v1_existing_cup_hold_gate_20260627_0145
```

Execution context:

```text
Slurm allocation: 154023
tmux session: curiosity_next_source_alloc_20260626_232937
tmux window: cup_hold_gate_0145
node: server56
scene: cube
tracked_object: existing_cup_asset
final_hold_duration: 999.0
```

Purpose:

The previous cup-asset gate showed the cup retargeted and visible, but the
final frame showed the cup tilted/fallen. Inspection of the official controller
revealed that the waypoint loop naturally transitions from lift-with-grasp
back to rest/no-grasp after the short cycle. This gate keeps the final hold
segment effectively closed during the 240-frame diagnostic window.

Result:

- Official Newton `sensor_contact` sanity: pass.
- SensorTiledCamera visual validation: pass.
- Manual visual inspection: pass for hold-control visual gate.
- Full two-second hold metric: pending, because the 240-frame window covers
  only about one second after the cup reaches the high hold pose.
- `generated_trex_fields=[]`.
- `schema_promotion=blocked`.
- `no_model_or_training=true`.

Numeric summary:

```text
initial_object_z=0.14850126206874847
final_object_z=0.30836987495422363
max_object_z=0.30836987495422363
max_lift=0.15986861288547516
```

Manual inspection of the contact sheet and frames `0120` and `0239` confirmed
that the cup remains elevated near the gripper at the final inspected frame.
This fixes the short-cycle release artifact for the visual gate. It does not
yet clear the configured full 2s hold success metric.

Next concrete step:

Run a longer-window metric gate, for example 360 frames or more, with the same
extended-hold controller and explicit success/failure metric extraction.
